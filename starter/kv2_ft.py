"""Fine-tune Kim Vocal 2 (UVR MDX-Net, 16.7M, the fast CPU separator) toward MelBand Kim's vocals on our own songs.

Adapted from ~/Code/lyric-asr/train/kv2_student.py (its onnx2torch conversion, BatchNorm handling and ONNX
write-back), with this repo's label layout (train/label.py: <labels>/<id>/mixture.flac + vocals.flac), remix
augmentation (teacher vocals of one excerpt over another's accompaniment, p=REMIX) and MSST's loss (waveform L1 +
multi-resolution STFT L1) as LOSS=msst (default). The exported student keeps Kim Vocal 2's graph and I/O, so it runs
in bench/models.py as a variant: data/models/kimv2_<tag>/core.onnx.
    python train/kv2_ft.py check <kv2.onnx>
    python train/kv2_ft.py train <kv2.onnx> <labels dir> <out dir>     env: LR STEPS BATCH EMA REMIX LOSS
    python train/kv2_ft.py export <kv2.onnx> <ckpt.pt> <out.onnx>
"""
import glob, hashlib, json, os, random, sys, time
import numpy as np, soundfile as sf, torch, onnx
from onnx import numpy_helper

N_FFT, HOP, DIM_F, DIM_T = 7680, 1024, 3072, 256
CHUNK = HOP * (DIM_T - 1)              # 261120 samples = 5.92 s, audio-separator's chunk_size for this model
dev = "cuda" if torch.cuda.is_available() else "cpu"
WIN = torch.hann_window(N_FFT, periodic=True).to(dev)


def stft(x):                           # [B, 2, N] -> [B, 4, DIM_F, T], audio-separator's uvr_lib_v5 STFT layout
    B = x.shape[0]
    s = torch.stft(x.reshape(-1, x.shape[-1]), N_FFT, HOP, window=WIN, center=True, return_complex=True)
    s = torch.view_as_real(s).permute(0, 3, 1, 2).reshape(B, 4, -1, s.shape[-1])
    return s[:, :, :DIM_F]


def istft(s):                          # [B, 4, DIM_F, T] -> [B, 2, N] (bins above DIM_F zero, as audio-separator)
    B, _, F, T = s.shape
    s = torch.cat([s, s.new_zeros(B, 4, N_FFT // 2 + 1 - F, T)], 2).reshape(B * 2, 2, -1, T).permute(0, 2, 3, 1)
    return torch.istft(torch.view_as_complex(s.contiguous()), N_FFT, HOP, window=WIN, center=True).reshape(B, 2, -1)


def model_in(mix):                     # what MDXSeparator.run_model feeds the net
    s = stft(mix)
    s[:, :, :3] = 0
    return s


def load_student(path):
    from onnx2torch import convert
    net = convert(path).to(dev).eval()      # eval: BatchNorm keeps its running stats while fine-tuning
    for mod in net.modules():              # onnx2torch keeps MatMul weights (the TDF linears, 6.3M) as buffers
        for n in [n for n, b in mod._buffers.items() if b is not None and b.is_floating_point()
                  and n not in ("running_mean", "running_var")]:
            mod._parameters[n] = torch.nn.Parameter(mod._buffers.pop(n))
    return net


def init_map(path, net):               # torch param/buffer name -> ONNX initializer name, by exact value at load time
    inits = {i.name: numpy_helper.to_array(i) for i in onnx.load(path).graph.initializer}
    by_key = {}
    for n, a in inits.items():
        by_key.setdefault((a.shape, a.dtype.str, hashlib.md5(a.tobytes()).hexdigest()), []).append(n)
    m, missing = {}, []                # identical tensors (e.g. zero biases): 1:1 in graph order
    for n, t in list(net.named_parameters()) + list(net.named_buffers()):
        a = t.detach().cpu().numpy()
        cands = by_key.get((a.shape, a.dtype.str, hashlib.md5(a.tobytes()).hexdigest()), [])
        if cands:
            m[n] = [cands.pop(0)]
        elif isinstance(t, torch.nn.Parameter):
            missing.append(n)          # shared initializer or a tensor onnx2torch made up: can't write back
    return m, missing


def write_back(path, net, mp, out):    # trained torch weights -> copy of the ONNX graph
    model = onnx.load(path)
    sd = dict(net.named_parameters()); sd.update(dict(net.named_buffers()))
    new = {}
    for n, names in mp.items():
        for iname in names:
            new[iname] = sd[n].detach().cpu().numpy()
    for i in model.graph.initializer:
        if i.name in new:
            i.CopyFrom(numpy_helper.from_array(new[i.name].astype(numpy_helper.to_array(i).dtype), i.name))
    onnx.save(model, out)


def ort_run(path, x):
    import onnxruntime as ort
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return s.run(None, {s.get_inputs()[0].name: x})[0]


def check(path):
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    m = onnx.load(path)
    io = lambda v: (v.name, [d.dim_value or d.dim_param for d in v.type.tensor_type.shape.dim])
    print("inputs", [io(v) for v in m.graph.input if v.name not in {i.name for i in m.graph.initializer}])
    print("outputs", [io(v) for v in m.graph.output])
    print("ops", sorted({n.op_type for n in m.graph.node}))
    print("initializers", len(m.graph.initializer), "params", sum(numpy_helper.to_array(i).size for i in m.graph.initializer))
    net = load_student(path)
    print("torch params", sum(p.numel() for p in net.parameters()), "buffers", sum(b.numel() for b in net.buffers()))
    torch.manual_seed(0)
    src = sorted(glob.glob(sys.argv[3] + "/*.mix.flac"))[:1] if len(sys.argv) > 3 else []
    if src:
        a, _ = sf.read(src[0], dtype="float32", always_2d=True)
        mix = torch.tensor(a[:CHUNK].T[None]).to(dev)
    else:
        mix = 0.1 * torch.randn(1, 2, CHUNK, device=dev)
    x = model_in(mix)
    with torch.no_grad():
        yt = net(x).cpu().numpy()
    yo = ort_run(path, x.cpu().numpy())
    rel = np.abs(yt - yo).max() / np.abs(yo).max()
    print(f"torch vs ORT: max abs {np.abs(yt - yo).max():.3g}, rel to peak {rel:.3g}, out shape {yo.shape}")
    mp, missing = init_map(path, net)
    print(f"mapped {len(mp)} tensors, unmapped params: {missing[:5]} ({len(missing)})")
    with torch.no_grad():                  # perturb, write back, ORT must follow torch
        for p in net.parameters():
            p.add_(0.01 * p.abs().mean() * torch.randn_like(p))
        yt2 = net(x).cpu().numpy()
    write_back(path, net, mp, "/tmp/kv2_rt.onnx")
    yo2 = ort_run("/tmp/kv2_rt.onnx", x.cpu().numpy())
    print(f"write-back round trip: max abs {np.abs(yt2 - yo2).max():.3g} (perturbed vs original moved {np.abs(yo2 - yo).max():.3g})")


LR = float(os.environ.get("LR", 3e-5))
STEPS = int(os.environ.get("STEPS", 6000))
BATCH = int(os.environ.get("BATCH", 8))
EMA = float(os.environ.get("EMA", 0.999))
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", 500))
VAL_EVERY = int(os.environ.get("VAL_EVERY", 250))
F8K = int(8000 / (44100 / N_FFT))      # bin of 8 kHz: all the 16 kHz ASR ever hears


def is_val(sid):                       # fixed ~2% of excerpts, never trained on
    return int(hashlib.md5(sid.encode()).hexdigest(), 16) % 50 == 0


def excerpts(d):
    return sorted(s for s in os.listdir(d) if os.path.exists(f"{d}/{s}/vocals.flac"))


def read(d, sid, which, st, n):
    a, _ = sf.read(f"{d}/{sid}/{'mixture' if which == 'mix' else 'vocals'}.flac", start=st, frames=n, dtype="float32",
                   always_2d=True)
    return a


FREE_LIC = ("Attribution", "CC0")              # CC BY (any version) and CC0; never NonCommercial


def free_license(lic):
    lic = lic or ""
    return lic.startswith(FREE_LIC) and "Noncommercial" not in lic.replace(" ", "") and "NonCommercial" not in lic


def acap_takes_free():
    """CC BY / CC0 a cappellas only (weights trained on them may ship): kept takes of data/acap/manifest.json and of
    the growing data/acap_cc/manifest.jsonl (train/acap_fetch.py --lic by,zero), minus cc30 take ids and uploaders,
    minus the held-out 1/12 (htd_kd.acap_takes' split) -> [(path, id, user, license)], deduplicated by take id."""
    import htd_kd
    cc = json.load(open(htd_kd.CC30))
    bad_ids = {str(t["id"]) for t in cc["takes"] + cc.get("rejected", []) if isinstance(t, dict) and "id" in t}
    bad_users = {t["user"].lower() for t in cc["takes"]}
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    cand = []
    try:
        cand += [(os.path.join(root, "acap", f"{t['id']}.flac"), t) for t in json.load(open(f"{root}/acap/manifest.json"))["takes"]]
    except OSError:
        pass
    try:
        for line in open(f"{root}/acap_cc/manifest.jsonl"):
            try:
                t = json.loads(line)
            except ValueError:                       # a line being written
                continue
            cand.append((os.path.join(root, "acap_cc", f"{t['id']}.flac"), t))
    except OSError:
        pass
    out, seen = [], set()
    for p, t in cand:
        tid = str(t["id"])
        if (tid in seen or not t.get("kept") or not free_license(t.get("license")) or not os.path.exists(p)
                or tid in bad_ids or t.get("user", "").lower() in bad_users
                or int(hashlib.md5(tid.encode()).hexdigest(), 16) % 12 == 0):
            continue
        seen.add(tid)
        out.append((p, tid, t.get("user", ""), t.get("license", "")))
    return out


class Crops(torch.utils.data.IterableDataset):   # random CHUNK crops of random excerpts; re-lists the growing dir
    """P_ACAP=<p> (with data/acap, train/acap_fetch.py; cc30 takes and uploaders excluded by htd_kd.acap_takes, its
    held-out 1/12 too): with probability p a true a cappella crop (gain -6..+3 dB) over an excerpt's teacher
    accompaniment (mixture - label vocals), target = the a cappella; then each item carries a third tensor, 1.0 for
    such crops (qat_mdx.py TEACHER=acap uses it to pick the target per sample).
    ACAP_LIC=free: only CC BY / CC0 takes (acap_takes_free, re-read every 512 crops as data/acap_cc grows; each take
    id first used is appended to ACAP_LOG, for attribution). d may be several label dirs joined by ':' (excerpts
    sampled uniformly over their union)."""
    def __init__(self, d):
        self.ds = d.split(":")
        self.d = self.ds[0]
        self.p_acap = float(os.environ.get("P_ACAP", 0))
        self.free = os.environ.get("ACAP_LIC") == "free"
        self.acap = []
        if self.p_acap:
            if self.free:
                self.acap = [p for p, *_ in acap_takes_free()]
                assert self.acap, "no CC BY / CC0 a cappellas found"
            else:
                import htd_kd
                self.acap = [p for p, val in htd_kd.acap_takes() if not val]

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = random.Random((info.id if info else 0) * 7919 + int(time.time()))
        ids, logged = [], set()
        for k in range(10**12):
            if k % 64 == 0:
                ids = [(dd, s) for dd in self.ds for s in excerpts(dd) if not is_val(s)]
            if self.free and self.p_acap and k % 512 == 0 and k:
                self.acap = [p for p, *_ in acap_takes_free()] or self.acap
            dd, sid = rng.choice(ids)
            n = sf.info(f"{dd}/{sid}/vocals.flac").frames
            st = rng.randrange(max(1, n - CHUNK))
            mix, voc = read(dd, sid, "mix", st, CHUNK), read(dd, sid, "voc", st, CHUNK)
            if len(mix) < CHUNK or len(voc) < CHUNK:
                continue
            if rng.random() < REMIX:                 # these vocals over another excerpt's accompaniment
                dd2, sid2 = rng.choice(ids)
                st2 = rng.randrange(max(1, sf.info(f"{dd2}/{sid2}/vocals.flac").frames - CHUNK))
                m2, v2 = read(dd2, sid2, "mix", st2, CHUNK), read(dd2, sid2, "voc", st2, CHUNK)
                if len(m2) < CHUNK or len(v2) < CHUNK:
                    continue
                voc = voc * 10 ** (rng.uniform(-6, 3) / 20)
                mix = voc + (m2 - v2) * 10 ** (rng.uniform(-3, 3) / 20)
                peak = np.abs(mix).max()
                if peak > 1:
                    mix, voc = mix / peak, voc / peak
            flag = 0.0
            if self.acap and rng.random() < self.p_acap:   # a true a cappella over this excerpt's accompaniment
                a = rng.choice(self.acap)
                na = sf.info(a).frames
                if na < CHUNK:
                    continue
                sa = rng.randrange(na - CHUNK + 1)
                ac, _ = sf.read(a, start=sa, frames=CHUNK, dtype="float32", always_2d=True)
                if ac.shape[1] == 1:
                    ac = np.repeat(ac, 2, 1)
                if np.sqrt((ac ** 2).mean()) < 1e-3:       # silent stretch of the take
                    continue
                voc = ac[:, :2] * 10 ** (rng.uniform(-6, 3) / 20)
                if self.free and a not in logged and os.environ.get("ACAP_LOG"):
                    logged.add(a)
                    with open(os.environ["ACAP_LOG"], "a") as f:
                        f.write(os.path.basename(a)[:-5] + "\n")
                acc = read(dd, sid, "mix", st, CHUNK) - read(dd, sid, "voc", st, CHUNK)   # (before any remix)
                mix = voc + acc * 10 ** (rng.uniform(-3, 3) / 20)
                peak = np.abs(mix).max()
                if peak > 1:
                    mix, voc = mix / peak, voc / peak
                flag = 1.0
            g = 10 ** (rng.uniform(-10, 0) / 20)    # songs are peak-normalised whole, so most chunks sit lower
            if rng.random() < 0.5:                   # L/R swap
                mix, voc = mix[:, ::-1], voc[:, ::-1]
            if self.acap:
                yield torch.tensor((mix * g).T.copy()), torch.tensor((voc * g).T.copy()), torch.tensor(flag)
            else:
                yield torch.tensor((mix * g).T.copy()), torch.tensor((voc * g).T.copy())


def spec_loss(y, t):                   # [B, 4, F, T]; bins 0-2 are zeroed on input and never used
    y, t = y[:, :, 3:], t[:, :, 3:]
    mag = lambda s: torch.sqrt(s[:, 0::2] ** 2 + s[:, 1::2] ** 2 + 1e-8)
    return (y - t).abs().mean() + (mag(y) - mag(t)).abs().mean()


REMIX = float(os.environ.get("REMIX", 0.5))
LOSS = os.environ.get("LOSS", "msst")   # msst: waveform L1 + multi-res STFT; spec: L1 per bin (all 3072 bins weigh alike, so mostly the quiet highs);
                                        # wave: MDX-Net's own time-domain L1, dominated by the loud low-mids (SDR-like)


def msst_loss(y, t):                   # MSST: waveform L1 + multi-resolution STFT L1 (windows 4096..256, hop 147)
    yw, tw = istft(y).reshape(-1, CHUNK), istft(t).reshape(-1, CHUNK)
    loss = (yw - tw).abs().mean()
    for w in (4096, 2048, 1024, 512, 256):
        kw = dict(n_fft=max(w, 2048), hop_length=147, win_length=w, window=torch.hann_window(w, device=dev), return_complex=True)
        loss = loss + (torch.view_as_real(torch.stft(yw, **kw)) - torch.view_as_real(torch.stft(tw, **kw))).abs().mean()
    return loss


def train_loss(y, t):
    if LOSS == "spec":
        return spec_loss(y, t)
    if LOSS == "msst":
        return msst_loss(y, t)
    return (istft(y) - istft(t)).abs().mean() * 100


def val_set(d, n=32):
    out = []
    for sid in [s for s in excerpts(d) if is_val(s)][:n]:
        mix, voc = read(d, sid, "mix", 0, -1), read(d, sid, "voc", 0, -1)
        for st in (len(mix) // 4, len(mix) // 2):
            out.append((torch.tensor(mix[st:st + CHUNK].T.copy()), torch.tensor(voc[st:st + CHUNK].T.copy())))
    return out


@torch.no_grad()
def validate(net, vs):
    l, l8, sdr = [], [], []
    for i in range(0, len(vs), 4):
        mix = torch.stack([m for m, _ in vs[i:i + 4]]).to(dev)
        voc = torch.stack([v for _, v in vs[i:i + 4]]).to(dev)
        t = stft(voc)
        y = net(model_in(mix)).float()
        l.append(spec_loss(y, t).item())
        l8.append((y[:, :, 3:F8K] - t[:, :, 3:F8K]).abs().mean().item())
        w, r = istft(y)[..., :CHUNK], istft(t)[..., :CHUNK]      # SDR vs the teacher within the student's band
        sdr += (10 * torch.log10(r.pow(2).sum((1, 2)) / ((r - w).pow(2).sum((1, 2)) + 1e-9) + 1e-9)).tolist()
    return np.mean(l), np.mean(l8), float(np.median(sdr))


def train(path, d, out):
    os.makedirs(out, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = True
    net = load_student(path)
    ema = load_student(path)
    for p in ema.parameters():
        p.requires_grad_(False)
    orig = load_student(path)              # the val set grows with the teacher data: always pair with the original
    log = open(f"{out}/log.txt", "a")
    def say(s):
        print(s, flush=True); log.write(s + "\n"); log.flush()
    def val(step):
        vs = val_set(d)
        say(f"step {step} val ({len(vs) // 2} excerpts) ema / original: loss %.5f / %.5f  L1<8k %.5f / %.5f  "
            "SDR vs teacher %.2f / %.2f dB" % tuple(x for p in zip(validate(ema, vs), validate(orig, vs)) for x in p))
    say(f"{len(excerpts(d))} excerpts, loss {LOSS} lr {LR} steps {STEPS} batch {BATCH} ema {EMA}")
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 200) * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * min(s, STEPS) / STEPS))))
    dl = torch.utils.data.DataLoader(Crops(d), batch_size=BATCH, num_workers=8, prefetch_factor=4, persistent_workers=True)
    t0, run = time.time(), []
    for step, (mix, voc) in enumerate(dl, 1):
        mix, voc = mix.to(dev, non_blocking=True), voc.to(dev, non_blocking=True)
        with torch.no_grad():
            x, t = model_in(mix), stft(voc)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = net(x)
        loss = train_loss(y.float(), t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), net.parameters()):
                pe.lerp_(p, 1 - EMA)
        run.append(loss.item())
        if step % 50 == 0:
            say(f"step {step} loss {np.mean(run):.5f} gn {gn:.3f} lr {sched.get_last_lr()[0]:.2e} {(time.time() - t0) / step:.2f}s/step n={len(excerpts(d))}")
            run = []
        if step % VAL_EVERY == 0:
            val(step)
        if step % SAVE_EVERY == 0 or step == STEPS:
            torch.save({"ema": ema.state_dict(), "step": step}, f"{out}/ema{step}.pt")
        if step >= STEPS:
            break


def export(path, ckpt, out):           # -> e.g. data/models/kimv2_ft/core.onnx (bench/models.py variant)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    net = load_student(path)
    mp, missing = init_map(path, net)
    assert not missing, missing
    net.load_state_dict(torch.load(ckpt, map_location=dev)["ema"])
    write_back(path, net, mp, out)
    x = model_in(0.1 * torch.randn(1, 2, CHUNK, device=dev))
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    with torch.no_grad():
        yt = net(x).cpu().numpy()
    yo = ort_run(out, x.cpu().numpy())
    print(f"export {out}: torch vs ORT max rel {np.abs(yt - yo).max() / np.abs(yo).max():.2g}")


if __name__ == "__main__":
    a = sys.argv
    {"check": lambda: check(a[2]), "train": lambda: train(a[2], a[3], a[4]),
     "export": lambda: export(a[2], a[3], a[4])}[a[1]]()
