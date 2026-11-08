"""Self-distil sped-up HTDemucs shapes (train/arch_htd.py probes) back toward composer's own HTDemucs:
FINDINGS.md "Sped-up HTDemucs, self-distilled (agent)".

Student = arch_htd.Probe(build(SPEC)) initialised from composer's checkpoint with the cut applied (e.g.
"tpoolp=2,notime" or "crop=1024,tpoolp=2,notime"), run on SEGS-sample segments (343980 = 7.8 s, 171990 = s2).
Teacher = the stock demucs htdemucs (= composer's htdemucs_fp32.onnx to 84 dB, train/htd_kd.py check), frozen, on
the full 7.8 s window; a shorter student segment is a random inner window of it, and its targets are the teacher's
outputs on that window (so the teacher keeps its full context).

    python train/htd_sd.py check <spec> [segs]           # untrained student vs teacher, ONNX export round trip
    python train/htd_sd.py train <spec> <out dir>        # env below
    python train/htd_sd.py export <spec> <ckpt.pt> <name> [segs]   # -> data/models/htdemucs_<name>/core.onnx

Examples (each 7.8 s for the teacher):
  label  an excerpt of our songs (data/labels.npy mixture)                        target: teacher's vocals  P_LABEL
  musdb  a MUSDB18-HQ train mixture                                               target: teacher's vocals  P_MUSDB
  acap   a true ccMixter a cappella (data/acap train takes, cc30 excluded:       target: the a cappella    P_ACAP
         htd_kd.acap_takes) over a teacher accompaniment (label mix - MelBand Kim vocals), vocal/backing -9..+3 dB;
         ACAP_LICENSES=by: CC BY / CC0 takes only (+ data/acap_cc, re-read every 500 examples) for shippable weights
Loss: waveform L1 of the vocals + ANCHOR x L1 of the other three stems vs the teacher. Adam, warmup + cosine, bf16
autocast, EMA (warm-up form); INIT=<ema_s<N>.pt> continues from a run's EMA weights, LR_FLOOR sets the cosine's end. torch.cuda memory capped at MEMFRAC of the card (shared GPU). Every EVAL_EVERY steps:
EMA -> <out>/ema_s<step>.pt, and SDR of the EMA student on held-out label crops (vs the teacher's vocals: "val vs
parent") and on held-out a cappella mixes (vs the true a cappella; the teacher's own score logged once).
"""
import copy
import json
import math
import os
import random
import sys
import time

import numpy as np
import soundfile as sf
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
import arch_htd as A  # noqa: E402
import htd_kd as K    # noqa: E402

TSEG, VOC = 343980, 3
dev = "cuda" if torch.cuda.is_available() else "cpu"
E = K.E


class Student(torch.nn.Module):
    """mix [B, 2, L] -> all stems [B, 4, 2, L] through the probe, with demucs' own _spec / _mask / _ispec."""
    def __init__(self, spec):
        super().__init__()
        self.o = A.parse(spec)
        self.m = A.build(self.o)
        self.p = A.Probe(self.m, self.o)

    def forward(self, mix):
        m = self.m
        z = m._spec(mix)
        x = m._magnitude(z)
        fo, to = self.p(mix, x)
        w = m._ispec(m._mask(z, fo.float()), mix.shape[-1])
        return to.to(w.device).float() + w


def teacher():
    return K.load_net().eval().requires_grad_(False)


def window(segs, rnd):
    if segs >= TSEG:
        return 0
    lo = (TSEG - segs) // 4
    return rnd.randrange(lo, TSEG - segs - lo + 1)


class Data(torch.utils.data.IterableDataset):
    def __init__(self):
        self.p = dict(label=E("P_LABEL", 0.45), musdb=E("P_MUSDB", 0.25), acap=E("P_ACAP", 0.3))

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rnd = random.Random((wi.id if wi else 0) * 7919 + int(time.time()))
        lab = np.load(K.LABELS, mmap_mode="r")
        L = lab.shape[-1]
        ids = [i for i in range(len(lab)) if not K.is_val(i)]
        mus = sorted(os.path.join(K.MUSDB, d) for d in os.listdir(K.MUSDB)
                     if os.path.exists(os.path.join(K.MUSDB, d, "mixture.flac")))
        acap, k = [], 0
        while True:
            if k % 500 == 0:               # re-read: data/acap_cc grows during training (ACAP_LICENSES=by)
                acap = [p for p, v in K.acap_takes() if not v] or acap
            k += 1
            kind = rnd.choices(list(self.p), list(self.p.values()))[0]
            v, flag = np.zeros((2, TSEG), np.float32), 0.0
            if kind == "label":
                s = rnd.randrange(L - TSEG)
                m = lab[rnd.choice(ids), 0, :, s:s + TSEG].astype(np.float32) / 32768
            elif kind == "musdb":
                d = os.path.join(rnd.choice(mus), "mixture.flac")
                s = rnd.randrange(sf.info(d).frames - TSEG)
                m = sf.read(d, start=s, frames=TSEG, dtype="float32", always_2d=True)[0].T
                if m.shape[1] < TSEG:
                    continue
            else:
                v = K.read_crop(rnd.choice(acap), rnd, TSEG)
                i, s = rnd.choice(ids), rnd.randrange(L - TSEG)
                acc = (lab[i, 0, :, s:s + TSEG].astype(np.float32) - lab[i, 1, :, s:s + TSEG]) / 32768
                if K.rms(v) < 1e-4:
                    continue
                v = (v * (K.rms(acc) / K.rms(v)) * 10 ** (rnd.uniform(-9, 3) / 20)).astype(np.float32)
                m, flag = v + acc, 1.0
            if rnd.random() < 0.5:
                m, v = m[::-1], v[::-1]
            g = 10 ** (rnd.uniform(-6, 3) / 20)
            peak = float(np.abs(m).max()) * g
            if peak > 1:
                g /= peak
            yield (torch.from_numpy(np.ascontiguousarray(m * g, np.float32)),
                   torch.from_numpy(np.ascontiguousarray(v * g, np.float32)), torch.tensor(flag))


@torch.no_grad()
def run_batched(net, X, n=4):
    out = []
    for i in range(0, len(X), n):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            out.append(net(X[i:i + n].to(dev))[:, VOC].float().cpu())
    return torch.cat(out)


def val_data(segs):
    T, Av = K.val_sets()               # label crops (teacher vocals unused) and a cappella mixes, 7.8 s
    o = window(segs, random.Random(2))
    tx = torch.from_numpy(np.stack([m for m, _ in T]))
    ax = torch.from_numpy(np.stack([m for m, _ in Av])) if Av else None
    ay = torch.from_numpy(np.stack([v for _, v in Av]))[..., o:o + segs] if Av else None
    return tx, ax, ay, o


def train(spec, out):
    os.makedirs(out, exist_ok=True)
    LR, STEPS, BS, SEGS = E("LR", 3e-5), E("STEPS", 8000), E("BS", 4), E("SEGS", TSEG)
    EMA, ANCHOR, EVAL_EVERY, WARM = E("EMA", 0.999), E("ANCHOR", 0.2), E("EVAL_EVERY", 1000), E("WARMUP", 300)
    FLOOR = E("LR_FLOOR", 0.1)          # cosine from LR down to FLOOR x LR
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(E("MEMFRAC", 0.45))
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    torch.set_num_threads(E("THREADS", 4))    # cap to the taskset CPU count
    net = Student(spec).to(dev).train()
    if os.environ.get("INIT"):         # continue from a run's EMA weights (fresh Adam state; use a short WARMUP)
        net.m.load_state_dict(torch.load(os.environ["INIT"], map_location=dev)["ema"])
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    tea = teacher().to(dev)
    logf = open(os.path.join(out, "log.jsonl"), "a")

    def log(rec):
        print(json.dumps(rec), flush=True)
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
    log({"start": time.strftime("%F %T"), "spec": spec, "segs": SEGS, "lr": LR, "lr_floor": FLOOR, "init": os.environ.get("INIT"),
         "steps": STEPS, "bs": BS, "ema": EMA,
         "anchor": ANCHOR, "p": Data().p})
    tx, ax, ay, vo = val_data(SEGS)
    tv = run_batched(tea, tx)[..., vo:vo + SEGS]

    def val(model, step, **kw):
        sx = tx[..., vo:vo + SEGS]
        rec = {"step": step, "val_parent": round(K.pooled_sdr(tv, run_batched(model, sx)), 3), **kw}
        if ax is not None:
            rec["val_acap"] = round(K.pooled_sdr(ay, run_batched(model, ax[..., vo:vo + SEGS])), 3)
        log(rec)
    teacher_acap = round(K.pooled_sdr(ay, run_batched(tea, ax)[..., vo:vo + SEGS]), 3) if ax is not None else None
    val(net.eval(), 0, teacher_val_acap=teacher_acap)
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = lambda s: min(1, (s + 1) / WARM) * (FLOOR + (1 - FLOOR) * 0.5 * (1 + math.cos(math.pi * min(1, s / STEPS))))
    dl = torch.utils.data.DataLoader(Data(), batch_size=BS, num_workers=E("WORKERS", 6), prefetch_factor=4,
                                     persistent_workers=True)
    rnd = random.Random(0)
    t0, lv, la = time.time(), [], []
    for step, (mix, voc, flag) in enumerate(dl):
        if step >= STEPS:
            break
        for g in opt.param_groups:
            g["lr"] = LR * sched(step)
        mix, voc, flag = mix.to(dev, non_blocking=True), voc.to(dev, non_blocking=True), flag.to(dev)
        o = window(SEGS, rnd)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            ref = tea(mix).float()[..., o:o + SEGS]
        tgt = torch.where(flag[:, None, None] > 0, voc[..., o:o + SEGS], ref[:, VOC])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            y = net(mix[..., o:o + SEGS].contiguous())
        lvoc = (y[:, VOC] - tgt).abs().mean()
        lanc = (y[:, :VOC] - ref[:, :VOC]).abs().mean()
        loss = lvoc + ANCHOR * lanc
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        with torch.no_grad():
            d = min(EMA, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), net.parameters()):
                pe.lerp_(pm, 1 - d)
        lv.append(lvoc.item())
        la.append(lanc.item())
        s = step + 1
        if s % 100 == 0:
            log({"step": s, "l_voc": round(float(np.mean(lv)), 6), "l_anchor": round(float(np.mean(la)), 6),
                 "lr": opt.param_groups[0]["lr"], "s_per_step": round((time.time() - t0) / s, 3),
                 "mem_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 1) if dev == "cuda" else None})
            lv, la = [], []
        if s % EVAL_EVERY == 0 or s == STEPS:
            torch.save({"ema": ema.m.state_dict(), "step": s, "spec": spec, "segs": SEGS}, os.path.join(out, f"ema_s{s}.pt"))
            val(ema, s, h=round((time.time() - t0) / 3600, 3))


def export(spec, ckpt, name, segs=None):
    """arch_htd.main's export with trained weights: data/models/htdemucs_<name>/core.onnx (bench HtDemucs layout)."""
    import onnxruntime as ort
    torch.backends.mha.set_fastpath_enabled(False)
    o = A.parse(spec)
    m = A.build(o)
    if ckpt != "-":
        c = torch.load(ckpt, map_location="cpu")
        m.load_state_dict(c["ema"])
        segs = segs or c.get("segs")
    seg = int(segs or TSEG)
    frames = -(-seg // 1024)
    p = A.Probe(m, o).eval()
    torch.manual_seed(0)
    mix, x = torch.randn(1, 2, seg) * 0.1, torch.randn(1, 4, 2048, frames) * 0.1
    d = os.path.join(ROOT, "data", "models", f"htdemucs_{name}")
    os.makedirs(d, exist_ok=True)
    out = os.path.join(d, "core.onnx")
    with torch.no_grad():
        y, yt = p(mix, x)
        torch.onnx.export(p, (mix, x), out, input_names=["input", "x"], output_names=["output", "add_67"],
                          opset_version=17, dynamo=False, do_constant_folding=True)
    s = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    a, b = s.run(["output", "add_67"], {"input": mix.numpy(), "x": x.numpy()})
    err = float(max(np.abs(a - y.numpy()).max() / np.abs(y.numpy()).max(), np.abs(b - yt.numpy()).max() / max(np.abs(yt.numpy()).max(), 1e-9)))
    print(json.dumps({"export": out, "mb": round(os.path.getsize(out) / 2 ** 20, 1), "seg": seg, "ort_vs_torch_rel": err}))


def check(spec, segs=None):
    """Untrained student vs the teacher on 4 held-out label crops (vocals SDR), and Student vs exported ONNX through
    bench/models.py HtDemucs on a real chunk."""
    seg = int(segs or TSEG)
    tx, _, _, o = val_data(seg)
    tx = tx[:4]
    tea, st = teacher().to(dev), Student(spec).to(dev).eval()
    tv = run_batched(tea, tx)[..., o:o + seg]
    print(json.dumps({"spec": spec, "seg": seg, "untrained_vs_parent_db": round(K.pooled_sdr(tv, run_batched(st, tx[..., o:o + seg])), 2)}))
    export(spec, "-", "sdcheck", seg)
    sys.path.insert(0, os.path.join(ROOT, "bench"))
    from models import MD, HtDemucs
    core = HtDemucs(f"{MD}/htdemucs_sdcheck/core.onnx", 4)
    x = K.test_clip()[:, :seg]
    mono = x.astype(np.float64).mean(0)
    mean, std = float(mono.mean()), float(mono.std(ddof=1))
    with torch.no_grad():
        yt = st.float()(torch.tensor(((x - mean) / std).astype(np.float32))[None].to(dev))[0, VOC].cpu().numpy() * std + mean
    print(json.dumps({"student_torch_vs_onnx_db": round(float(K.snr(core(x), yt)), 1)}))


if __name__ == "__main__":
    a = sys.argv
    {"check": lambda: check(a[2], *a[3:4]), "train": lambda: train(a[2], a[3]),
     "export": lambda: export(a[2], a[3], a[4], *a[5:6])}[a[1]]()
