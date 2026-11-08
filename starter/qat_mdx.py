"""Quantization-aware int8 for UVR MDX-Net cores (Voc_FT, Kim Vocal 2): fake-quant the onnx2torch conversion, train it
toward the fp32 model's own output (self-distillation, waveform L1), write the weights back into the original ONNX
graph and emit QDQ nodes with exactly the scales the fake quantizers used. The QDQ file runs in bench/models.py as
<base>_<tag> (ORT: correct but 0.7x fp32 speed) and <base>_<tag>_ov (OpenVINO: real int8 kernels, ~2x).
FINDINGS.md "Quantization-aware int8 MDX-Net (agent)".

Scheme (what ORT static QDQ / OpenVINO's low-precision transformations expect):
  weights      int8 symmetric per output channel, absmax / 127 (Conv [O,..] axis 0, ConvTranspose [I,O,..] axis 1,
               MatMul [K,N] axis 1), computed from the live weights (so export recomputes them identically)
  activations  uint8 asymmetric per tensor on the input of every quantized op (OUTQ=1: also on its output); EMA
               min/max observers (ACT=ema), or learned scales started from the current ranges (ACT=lsq)
  fp32         the 1x1 input conv (raw STFT in) and 1x1 output conv (raw STFT out), ~1% of the MACs, plus FP32=...;
               BatchNorm frozen (eval mode, running stats), biases fp32
    python train/qat_mdx.py ptq <mdx.onnx> <labels dir>                     fake-quant SNR with calibrated ranges only
    python train/qat_mdx.py train <mdx.onnx> <labels dir> <out dir>
        env: LR STEPS BATCH ACCUM ACT SLR OUTQ NORM GMAX FP32 FREEZE CALIB TEACHER INIT VAL_EVERY
    python train/qat_mdx.py export <mdx.onnx> <ckpt.pt> <out.onnx>          QDQ graph with the trained weights + ranges
    python train/qat_mdx.py snr <a.onnx> <b.onnx> <labels dir> [ov]          core-output SNR of b vs a on val chunks
    python train/qat_mdx.py export_fp32 <mdx.onnx> <ckpt.pt> <out.onnx>     NOQ=1 checkpoint -> plain fp32 graph

vocft_qat8mix (the best one) was four chained runs on data/labels, batch 2 x accum 2, ~6500 steps, ~2.4 h on one A5000:
  ACT=ema LR=2e-5                       -> qat2500.pt  (= vocft_qat8ema)
  INIT=<that> ACT=lsq SLR=1e-3 LR=1e-5  -> qat1000.pt
  INIT=<that> NORM=6.7 ACT=lsq LR=2e-5  -> qat1000.pt  (= vocft_qat8norm)
  INIT=<that> NORM=6.7 FP32=MatMul_152,Conv_163,Conv_5,Conv_7,Conv_3 ACT=lsq LR=2e-5 STEPS=2000 -> qat2000.pt
"""
import math, os, sys, time
import numpy as np, torch, onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kv2_ft as K  # noqa: E402  (STFT layout, loader, write-back, label crops; its env defaults are not used here)

dev = K.dev
SKIP = {"Conv_0", "Conv_177"}          # fp32: first/last 1x1 conv (the raw-STFT input and output)
# FP32=<node,...>: more ops left fp32 (mixed precision), e.g. the few full-resolution layers that carry most of the
# int8 noise (qat_agent sensitivity: MatMul_152, Conv_163, Conv_5, Conv_7, Conv_3 = 88% of it for Voc_FT)
SKIP |= set(filter(None, os.environ.get("FP32", "").split(",")))
ACT = os.environ.get("ACT", "ema")
# NORM=<spectrum RMS>: scale each chunk's input spectrum to this RMS inside the graph and undo it on the output, so
# every chunk uses the same activation ranges (without it a chunk 6 dB louder than the training level clips, 20 dB
# quieter loses its resolution). Spectrum RMS = 42.4 x waveform RMS for this STFT; 6.7 = -16 dBFS. Gain capped at GMAX.
NORM = float(os.environ.get("NORM", 0)) or None
GMAX = float(os.environ.get("GMAX", 10))
OUTQ = os.environ.get("OUTQ", "0") == "1"   # also fake-quant each quantized op's output (ORT's QDQ layout: OpenVINO
                                            # then keeps the BN/ReLU/Add paths between int8 ops in int8, ~1.2x faster)


class _LsqFQ(torch.autograd.Function):
    """uint8 fake quant with a learned scale (LSQ gradients), saving only x for backward."""
    @staticmethod
    def forward(ctx, x, s, zp):
        ctx.save_for_backward(x, s); ctx.zp = zp
        return (torch.clamp(torch.round(x / s + zp), 0, 255) - zp) * s

    @staticmethod
    def backward(ctx, g):
        x, s = ctx.saved_tensors
        v = x / s + ctx.zp
        q = torch.clamp(torch.round(v), 0, 255)
        inside = (v >= -0.5) & (v <= 255.5)
        gs = (g * torch.where(inside, q - v, q - ctx.zp)).sum().reshape(s.shape)
        return g * inside, gs, None


class FQAct(torch.nn.Module):
    """uint8 per-tensor fake quant of an activation. observe: update the EMA min/max; quant: apply fake quant."""
    def __init__(self, mom=0.01):
        super().__init__()
        self.register_buffer("lo", torch.tensor(float("inf")))
        self.register_buffer("hi", torch.tensor(float("-inf")))
        self.log_s = None                  # ACT=lsq: learned log-scale (zero point stays from the EMA range)
        self.mom, self.observe, self.quant, self.n = mom, True, False, 0

    def qparams(self):                     # (scale, zero point) exactly as the export writes them
        lo, hi = min(float(self.lo), 0.0), max(float(self.hi), 0.0)
        s = max(hi - lo, 1e-8) / 255
        return s, int(min(255, max(0, round(-lo / s))))

    def learn(self):                       # switch to a learned scale, initialised from the observed range
        s, self.zp = self.qparams()
        self.log_s = torch.nn.Parameter(torch.tensor(math.log(s), device=self.lo.device))
        self.observe = False

    def forward(self, x):
        if self.observe:
            with torch.no_grad():
                lo, hi = x.detach().min().float(), x.detach().max().float()
                if self.n == 0:
                    self.lo.copy_(lo); self.hi.copy_(hi)
                else:
                    self.lo.lerp_(lo, self.mom); self.hi.lerp_(hi, self.mom)
                self.n += 1
        if not self.quant:
            return x
        if self.log_s is not None:
            return _LsqFQ.apply(x, self.log_s.exp(), self.zp)   # straight-through round, clipped grads 0
        s, zp = self.qparams()
        return torch.fake_quantize_per_tensor_affine(x, s, zp, 0, 255)


def fq_weight(w, axis):                # int8 symmetric per channel, absmax / 127, straight-through
    dims = [d for d in range(w.dim()) if d != axis]
    s = (w.detach().abs().amax(dims) / 127).clamp_min(1e-12).float()
    return torch.fake_quantize_per_channel_affine(w, s, torch.zeros_like(s, dtype=torch.int32), axis, -127, 127)


class QConv(torch.nn.Module):
    def __init__(self, conv):
        super().__init__()
        self.conv, self.act = conv, FQAct()
        self.oact = FQAct() if OUTQ else None
        self.axis = 1 if isinstance(conv, torch.nn.ConvTranspose2d) else 0
        self.wq = False

    def forward(self, x):
        x = self.act(x)
        c = self.conv
        if not self.wq:
            y = c(x)
        elif isinstance(c, torch.nn.ConvTranspose2d):
            y = torch.nn.functional.conv_transpose2d(x, fq_weight(c.weight, 1), c.bias, c.stride, c.padding,
                                                     c.output_padding, c.groups, c.dilation)
        else:
            y = c._conv_forward(x, fq_weight(c.weight, 0), c.bias)
        return self.oact(y) if self.oact is not None else y


class QMatMul(torch.nn.Module):        # onnx2torch OnnxMatMul(x, w): the TDF linears over frequency, w [K, N]
    def __init__(self):
        super().__init__()
        self.act, self.wq = FQAct(), False
        self.oact = FQAct() if OUTQ else None

    def forward(self, x, w):
        y = torch.matmul(self.act(x), fq_weight(w, 1) if self.wq else w)
        return self.oact(y) if self.oact is not None else y


def acts(q):                           # (key, quantizer): "<node>" for an op's input, "<node>:out" for its output
    for n, w in q.items():
        yield n, w.act
        if w.oact is not None:
            yield n + ":out", w.oact


def qp_dict(q):
    return {k: a.qparams() if a.log_s is None else (float(a.log_s.exp()), a.zp) for k, a in acts(q)}


def wrap(net):
    """Replace the quantized modules in place; returns {onnx node name: wrapper}. onnx2torch names modules after the
    ONNX nodes, so the names are what export() needs."""
    q = {}
    for name, m in list(net.named_children()):
        if name in SKIP:
            continue
        if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            q[name] = QConv(m)
        elif type(m).__name__ == "OnnxMatMul":
            q[name] = QMatMul()
        else:
            continue
        setattr(net, name, q[name].to(dev))
    return q


def set_mode(q, observe=None, quant=None):
    for _, a in acts(q):
        if observe is not None:
            a.observe = observe and a.log_s is None
        if quant is not None:
            a.quant = quant
    if quant is not None:
        for w in q.values():
            w.wq = quant


def add_norm(net, target, gmax):        # forward hooks, so the state dict keeps the plain net's names
    def pre(m, args):
        x = args[0]
        m._g = (target / x.pow(2).mean((1, 2, 3), keepdim=True).sqrt().clamp_min(1e-8)).clamp(max=gmax)
        return (x * m._g,)
    net.register_forward_pre_hook(pre)
    net.register_forward_hook(lambda m, args, out: out / m._g)


def qat_net(path):
    net = K.load_student(path)
    for p in net.parameters():
        p.data = p.data.contiguous()
    if NORM:
        add_norm(net, NORM, GMAX)
    return net, wrap(net)


def batches(d, n, bs):                 # n batches of training crops (remix augmentation as kv2_ft, REMIX env)
    K.REMIX = float(os.environ.get("REMIX", 0.5))
    it = iter(torch.utils.data.DataLoader(K.Crops(d), batch_size=bs, num_workers=int(os.environ.get("WORKERS", 4))))
    for _ in range(n):
        yield next(it)[0].to(dev)


@torch.no_grad()
def calibrate(net, q, d, n, bs):
    set_mode(q, observe=True, quant=False)
    for i, mix in enumerate(batches(d, n, bs)):
        for _, a in acts(q):
            a.mom = 1 / (i + 1)        # EMA with 1/(i+1) = mean of the batch min/max
        net(K.model_in(mix))
    for _, a in acts(q):
        a.mom = 0.01


def val_chunks(d, n):                 # VAL_DIR: val crops from this label dir (default: the first training dir)
    return torch.stack([m for m, _ in K.val_set(os.environ.get("VAL_DIR", d.split(":")[0]))[:n]])


def snr(ref, est):                     # dB, over a batch of waveforms or spectra
    return float(10 * torch.log10(ref.pow(2).sum() / (ref - est).pow(2).sum().clamp_min(1e-20)))


@torch.no_grad()
def compare(net, ref_net, q, chunks, bs=2):
    """Spectrum SNR and waveform SNR (= SDR vs the fp32 model) of the fake-quant net on fixed chunks."""
    sp, wv = [], []
    for i in range(0, len(chunks), bs):
        x = K.model_in(chunks[i:i + bs].to(dev))
        set_mode(q, observe=False, quant=not NOQ)
        tr = [m.training for m in net.modules()]
        net.eval()
        y = net(x)
        for m, t_ in zip(net.modules(), tr):
            m.train(t_)
        t = ref_net(x)
        sp.append(snr(t, y)); wv.append(snr(K.istft(t), K.istft(y)))
    return float(np.mean(sp)), float(np.median(wv))


def ptq(path, d):
    torch.set_num_threads(int(os.environ.get("THREADS", 12)))
    net, q = qat_net(path)
    ref = K.load_student(path)
    calibrate(net, q, d, int(os.environ.get("CALIB", 8)), 2)
    if os.environ.get("PTQ_OUT"):      # checkpoint of the calibrated, untrained net (export it like a QAT one)
        torch.save({"state": net.state_dict(), "qp": qp_dict(q), "outq": OUTQ, "norm": NORM and (NORM, GMAX), "step": 0},
                   os.environ["PTQ_OUT"])
    ch = val_chunks(d, int(os.environ.get("NVAL", 8)))
    print("all quantized: spectrum SNR %.2f dB, waveform SNR median %.2f dB" % compare(net, ref, q, ch), flush=True)
    for label in os.environ.get("PTQ_VARIANTS", "weights only,activations only").split(","):
        if not label:
            continue
        set_mode(q, observe=False, quant=True)
        saved = [(a.lo.clone(), a.hi.clone()) for _, a in acts(q)]
        for w in q.values():
            if label == "activations only":
                w.wq = False
        for _, a in acts(q):
            if label == "weights only":
                a.quant = False
            elif label.startswith("clip"):  # all ranges scaled by this factor (clipping vs resolution)
                a.lo.mul_(float(label[4:])); a.hi.mul_(float(label[4:]))
        sp, wv = [], []
        with torch.no_grad():
            for i in range(0, len(ch), 2):
                x = K.model_in(ch[i:i + 2].to(dev))
                y, t = net(x), ref(x)
                sp.append(snr(t, y)); wv.append(snr(K.istft(t), K.istft(y)))
        for (_, a), (lo, hi) in zip(acts(q), saved):
            a.lo.copy_(lo); a.hi.copy_(hi)
        print(f"{label}: spectrum SNR {np.mean(sp):.2f} dB, waveform SNR median {np.median(wv):.2f} dB", flush=True)


LR = float(os.environ.get("LR", 2e-5))
STEPS = int(os.environ.get("STEPS", 4000))
BATCH = int(os.environ.get("BATCH", 2))
ACCUM = int(os.environ.get("ACCUM", 2))
FREEZE = float(os.environ.get("FREEZE", 0.5))     # fraction of steps after which EMA ranges stop moving
TEACHER = os.environ.get("TEACHER", "self")        # self: fp32 model's output; labels: MelBand Kim vocals; mix: both;
                                                   # acap: teacher output, but the true a cappella on P_ACAP crops
# TEACHER_PATH=<onnx>: the fp32 teacher for TEACHER=self/mix is this model instead of <mdx.onnx> itself (a pruned
# student distilled from the full model: train/arch_mdx.py prune); NOQ=1: no fake quant at all (plain fp32
# self-distillation; `export` then also leaves <out>.fp32.onnx, the fp32 graph with the trained weights);
# BNTRAIN=1: BatchNorm in train mode (batch statistics, running stats re-learned), for nets whose stats are stale
NOQ = os.environ.get("NOQ", "0") == "1"
BNTRAIN = os.environ.get("BNTRAIN", "0") == "1"
# EMA=<decay> (NOQ=1 only): validate and checkpoint an exponential moving average of the weights; AMP=bf16: the
# student's forward/backward under bf16 autocast (the teacher stays fp32/TF32); SAVE_EVERY: checkpoint interval
# (default VAL_EVERY)
EMA = float(os.environ.get("EMA", 0))
AMP = os.environ.get("AMP", "")


def train(path, d, out):
    os.makedirs(out, exist_ok=True)
    if dev == "cuda" and os.environ.get("GPU_MEM_FRAC"):   # share a card: cap this process's CUDA memory
        torch.cuda.set_per_process_memory_fraction(float(os.environ["GPU_MEM_FRAC"]))
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = True
    net, q = qat_net(path)
    ref = K.load_student(os.environ.get("TEACHER_PATH", path))
    for p in ref.parameters():
        p.requires_grad_(False)
    bns = [m for m in net.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    for m in bns:
        m.train(BNTRAIN)
    log = open(f"{out}/log.txt", "a")
    def say(s):
        print(s, flush=True); log.write(s + "\n"); log.flush()
    say(f"qat {path} act {ACT} outq {OUTQ} lr {LR} steps {STEPS} batch {BATCH}x{ACCUM} freeze {FREEZE} teacher {TEACHER}"
        f" {os.environ.get('TEACHER_PATH', '')} noq {NOQ} bntrain {BNTRAIN}")
    if os.environ.get("INIT"):         # continue from a QAT checkpoint (its weights and ranges), e.g. EMA -> lsq
        load_ckpt(q, net, torch.load(os.environ["INIT"], map_location=dev))
        say(f"init from {os.environ['INIT']}")
    elif not NOQ:
        calibrate(net, q, d, int(os.environ.get("CALIB", 32)), BATCH)
    if int(os.environ.get("RECAL_N", 0)):  # e.g. continuing a BNTRAIN=1 run with frozen, re-estimated BN stats
        recal_bn(net, d, int(os.environ["RECAL_N"]), BATCH)
        for m in bns:
            m.train(BNTRAIN)
        say(f"BatchNorm stats re-estimated on {os.environ['RECAL_N']} crops")
    ch = val_chunks(d, int(os.environ.get("NVAL", 16)))
    say("step 0 (calibrated PTQ / init): spectrum SNR %.2f dB, waveform SNR vs fp32 %.2f dB" % compare(net, ref, q, ch))
    ema = None
    if EMA and NOQ:
        import copy
        ema = copy.deepcopy(net)
        for p in ema.parameters():
            p.requires_grad_(False)
        say(f"EMA {EMA}")
    if ACT == "lsq":
        for _, a in acts(q):
            a.learn()
    set_mode(q, observe=ACT == "ema" and not NOQ, quant=not NOQ)
    scales = [a.log_s for _, a in acts(q) if a.log_s is not None]
    groups = [{"params": [p for p in net.parameters() if all(p is not s for s in scales)], "lr": LR}]
    if scales:
        groups.append({"params": scales, "lr": float(os.environ.get("SLR", 1e-3))})
    opt = torch.optim.AdamW(groups, weight_decay=0)
    # DEADLINE=<unix time>: the cosine runs over wall time until then instead of over STEPS (STEPS = upper bound),
    # and training stops (with a checkpoint) when it is reached: two runs sharing a GPU end together whatever s/step is
    deadline = float(os.environ.get("DEADLINE", 0))
    t_start = time.time()
    prog = (lambda s: min(1.0, (time.time() - t_start) / max(1.0, deadline - t_start))) if deadline else (lambda s: min(s, STEPS) / STEPS)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * prog(s)))))
    if deadline:
        say(f"deadline {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(deadline))} UTC ({(deadline - t_start) / 3600:.2f} h)")
    K.REMIX = float(os.environ.get("REMIX", 0.5))
    dl = torch.utils.data.DataLoader(K.Crops(d), batch_size=BATCH, num_workers=int(os.environ.get("WORKERS", 6)),
                                     prefetch_factor=4, persistent_workers=True)
    t0, run = time.time(), []
    it = iter(dl)
    for step in range(1, STEPS + 1):
        last = step == STEPS or (deadline and time.time() >= deadline)
        if step == int(FREEZE * STEPS) and not NOQ:
            set_mode(q, observe=False)
            say(f"step {step}: activation ranges frozen")
        opt.zero_grad(set_to_none=True)
        for _ in range(ACCUM):             # BATCH x ACCUM crops per step (batch 4 of fp32 fake quant needs >24 GB)
            item = next(it)
            mix, voc = item[0], item[1]
            isac = item[2].to(dev).view(-1, 1, 1) if len(item) > 2 else None
            mix = mix.to(dev, non_blocking=True)
            with torch.no_grad():
                x = K.model_in(mix)
                tgt = ref(x) if TEACHER != "labels" else None
                if TEACHER != "self" and (TEACHER != "acap" or isac is not None):
                    lab = K.stft(voc.to(dev, non_blocking=True))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP == "bf16"):
                ys = net(x)
            yw = K.istft(ys.float())
            loss = 0
            if TEACHER == "acap":          # per sample: the true a cappella for P_ACAP crops, else the teacher's output
                tw = K.istft(tgt)
                tw = torch.where(isac > 0, K.istft(lab), tw) if isac is not None else tw
                loss = (yw - tw).abs().mean() * 100
            else:
                if tgt is not None:
                    loss = loss + (yw - K.istft(tgt)).abs().mean() * 100
                if TEACHER != "self":
                    loss = loss + (yw - K.istft(lab)).abs().mean() * 100
            (loss / ACCUM).backward()
        gn = torch.nn.utils.clip_grad_norm_(groups[0]["params"], 1.0)
        opt.step(); sched.step()
        if ema is not None:
            with torch.no_grad():
                for pe, p in zip(ema.parameters(), net.parameters()):
                    pe.lerp_(p, 1 - EMA)
        run.append(loss.item())
        if step % 50 == 0:
            say(f"step {step} loss {np.mean(run):.5f} gn {gn:.3f} lr {sched.get_last_lr()[0]:.2e} {(time.time() - t0) / step:.2f}s/step")
            run = []
        ve = int(os.environ.get("VAL_EVERY", 500))
        if step % ve == 0 or last:
            obs = [a.observe for _, a in acts(q)]
            say(f"step {step} val: spectrum SNR %.2f dB, waveform SNR vs fp32 %.2f dB" % compare(net, ref, q, ch)
                + (" | EMA %.2f / %.2f" % compare(ema, ref, {}, ch) if ema is not None else ""))
            for (_, a), o in zip(acts(q), obs):
                a.observe = o
        if step % int(os.environ.get("SAVE_EVERY", ve)) == 0 or last:
            torch.save({"state": (ema if ema is not None else net).state_dict(), "qp": qp_dict(q), "outq": OUTQ,
                        "norm": NORM and (NORM, GMAX), "skip": sorted(SKIP), "step": step}, f"{out}/qat{step}.pt")
        if last:
            torch.save({"state": (ema if ema is not None else net).state_dict(), "qp": qp_dict(q), "outq": OUTQ,
                        "norm": NORM and (NORM, GMAX), "skip": sorted(SKIP), "step": step}, f"{out}/final.pt")
            say(f"step {step}: done ({(time.time() - t0) / step:.2f} s/step)")
            break


def load_ckpt(q, net, c):              # QAT checkpoint -> wrapped net: weights, and ranges from the exported qparams
    assert c.get("outq", False) == OUTQ, "checkpoint and OUTQ disagree"
    sd = {}
    for k, v in c["state"].items():    # ops wrapped there but fp32 here: Conv_3.conv.weight -> Conv_3.weight
        mod = k.split(".")[0]
        if mod in q or not mod.startswith(("Conv", "MatMul")):
            sd[k] = v
        elif ".act." not in k and ".oact." not in k:
            sd[k.replace(".conv.", ".")] = v
    bad = net.load_state_dict(sd, strict=False)       # (learned log-scales are not modules here: qp carries them)
    assert not bad.missing_keys, bad.missing_keys[:5]
    for k, a in acts(q):
        s, zp = c["qp"][k]
        a.lo.fill_(-zp * s); a.hi.fill_((255 - zp) * s)
        a.n = 1


def _plain_state(sd):                  # wrapped names (Conv_3.conv.weight) -> the plain net's (Conv_3.weight)
    return {k.replace(".conv.", "."): v for k, v in sd.items() if ".act." not in k and ".oact." not in k}


@torch.no_grad()
def recal_bn(net, d, n, bs=4):
    """BatchNorm running stats re-estimated as the plain average over n training crops (train-mode BN, momentum
    None), then BN back to eval. For BNTRAIN=1 nets: their momentum-0.1 running stats made a few val chunks explode in
    eval mode (vocft_axp50 self-distillation, step 1000: -150 dB on 3 of 8 chunks, 2-12 dB in train mode)."""
    bns = [m for m in net.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    for m in bns:
        m.reset_running_stats(); m.momentum = None; m.train()
    for mix in batches(d, max(1, n // bs), bs):
        net(K.model_in(mix))
    for m in bns:
        m.momentum = 0.1; m.eval()


def export_fp32(path, ckpt, out):
    """Weights of a NOQ=1 (plain fp32) checkpoint -> copy of <mdx.onnx>'s graph (no QDQ); RECAL_N=<crops>
    re-estimates the BatchNorm stats first (recal_bn)."""
    c = torch.load(ckpt, map_location=dev)
    net = K.load_student(path)
    mp, missing = K.init_map(path, net)
    assert not missing, missing
    net.load_state_dict(_plain_state(c["state"]))
    if int(os.environ.get("RECAL_N", 0)):
        recal_bn(net, os.environ.get("LABELS", "data/labels"), int(os.environ["RECAL_N"]))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    K.write_back(path, net, mp, out)
    print(out)


def export(path, ckpt, out):
    """QAT weights -> copy of the ONNX graph (kv2_ft.write_back), then QDQ: int8 weight initializers with
    DequantizeLinear, QuantizeLinear/DequantizeLinear pairs on each quantized op's activation input (and output
    when trained with OUTQ=1)."""
    global OUTQ
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    c = torch.load(ckpt, map_location=dev)
    global NORM, GMAX
    OUTQ = c.get("outq", False)
    if c.get("norm"):
        NORM, GMAX = c["norm"]
    SKIP.update(c.get("skip", []))
    net = K.load_student(path)
    mp, missing = K.init_map(path, net)
    assert not missing, missing
    net.load_state_dict(_plain_state(c["state"]))
    tmp = out + ".fp32.onnx"
    K.write_back(path, net, mp, tmp)
    m = onnx.load(tmp)
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    new_nodes, new_inits, drop = [], [], set()
    excl = set(filter(None, os.environ.get("EXCLUDE", "").split(",")))   # op types / node names left fp32
    def qdq(src, dst, key, pre):       # QuantizeLinear/DequantizeLinear src -> dst with the trained (scale, zp)
        s, zp = c["qp"][key]
        new_inits.extend([numpy_helper.from_array(np.array(s, np.float32), pre + "_s"),
                          numpy_helper.from_array(np.array(zp, np.uint8), pre + "_zp")])
        return [helper.make_node("QuantizeLinear", [src, pre + "_s", pre + "_zp"], [pre + "_q"], name=pre + "_Q"),
                helper.make_node("DequantizeLinear", [pre + "_q", pre + "_s", pre + "_zp"], [dst], name=pre + "_DQ")]
    for n in g.node:
        after = []
        if n.name in c["qp"] and n.name not in excl and n.op_type not in excl:
            new_nodes += qdq(n.input[0], f"{n.name}_qin_dq", n.name, f"{n.name}_qin")
            n.input[0] = f"{n.name}_qin_dq"
            if n.name + ":out" in c["qp"]:
                o = n.output[0]
                n.output[0] = f"{n.name}_qout_raw"
                after = qdq(n.output[0], o, n.name + ":out", f"{n.name}_qout")
            wn = n.input[1]
            w = numpy_helper.to_array(inits[wn]).astype(np.float32)
            axis = {"Conv": 0, "ConvTranspose": 1, "MatMul": 1}[n.op_type]
            dims = tuple(d for d in range(w.ndim) if d != axis)
            ws = np.maximum(np.abs(w).max(dims) / 127, 1e-12).astype(np.float32)
            shape = [1] * w.ndim; shape[axis] = -1
            wq = np.clip(np.round(w / ws.reshape(shape)), -127, 127).astype(np.int8)   # == torch's fake quant (half-even)
            pw = f"{n.name}_w"
            new_inits += [numpy_helper.from_array(wq, pw + "_q"), numpy_helper.from_array(ws, pw + "_s"),
                          numpy_helper.from_array(np.zeros(len(ws), np.int8), pw + "_zp")]
            new_nodes.append(helper.make_node("DequantizeLinear", [pw + "_q", pw + "_s", pw + "_zp"], [pw + "_dq"],
                                              name=pw + "_DQ", axis=axis))
            n.input[1] = pw + "_dq"
            drop.add(wn)
        new_nodes.append(n)
        new_nodes += after
    still = {i for n in new_nodes for i in n.input}
    keep = [i for i in g.initializer if i.name not in drop or i.name in still]
    del g.initializer[:]
    if NORM:                           # g = min(NORM / rms(input), GMAX); net(input * g) / g  (batch 1: rms over all)
        inp, outn = g.input[0].name, g.output[0].name
        for n in new_nodes:
            n.input[:] = [f"{inp}_n" if i == inp else i for i in n.input]
            n.output[:] = [f"{outn}_raw" if o == outn else o for o in n.output]
        new_inits += [numpy_helper.from_array(np.array(NORM, np.float32), "norm_target"),
                      numpy_helper.from_array(np.array(1e-8, np.float32), "norm_eps"),
                      numpy_helper.from_array(np.array(GMAX, np.float32), "norm_gmax")]
        mk = helper.make_node
        new_nodes = [mk("Mul", [inp, inp], ["norm_sq"]), mk("ReduceMean", ["norm_sq"], ["norm_ms"], keepdims=1),
                     mk("Sqrt", ["norm_ms"], ["norm_rms"]), mk("Max", ["norm_rms", "norm_eps"], ["norm_rmsc"]),
                     mk("Div", ["norm_target", "norm_rmsc"], ["norm_g0"]), mk("Min", ["norm_g0", "norm_gmax"], ["norm_g"]),
                     mk("Mul", [inp, "norm_g"], [f"{inp}_n"])] + new_nodes + [mk("Div", [f"{outn}_raw", "norm_g"], [outn])]
    g.initializer.extend(keep + new_inits)
    del g.node[:]
    g.node.extend(new_nodes)
    for o in m.opset_import:
        if o.domain in ("", "ai.onnx") and o.version < 13:
            o.version = 13                  # per-channel DequantizeLinear (axis) needs opset 13
    onnx.checker.check_model(m)
    onnx.save(m, out)
    os.remove(tmp)
    # the fake-quant torch net with the same weights and ranges must match ORT running the QDQ graph
    qnet, q = qat_net(path)
    load_ckpt(q, qnet, c)
    set_mode(q, observe=False, quant=True)
    for n, w in q.items():
        if n in excl or n.split("_")[0] in excl:   # node names are <OpType>_<index>
            w.wq = False
            for a in (w.act, w.oact):
                if a is not None:
                    a.quant = False
    lab = os.environ.get("LABELS", "data/labels")   # a real val chunk: on noise the int8 net is meaningless
    x = K.model_in(K.val_set(lab, 1)[0][0][None].to(dev) if os.path.isdir(lab) else 0.1 * torch.randn(1, 2, K.CHUNK, device=dev))
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    with torch.no_grad():
        yt = qnet(x)
    yo = torch.tensor(K.ort_run(out, x.cpu().numpy()))
    print(f"export {out} ({os.path.getsize(out) / 2**20:.1f} MB): ORT QDQ vs torch fake-quant SNR {snr(yt.cpu(), yo):.1f} dB")


def ov_run(path, x):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))
    from models import OvSession
    s = OvSession(path, int(os.environ.get("THREADS", 4)))
    return np.concatenate([s.run(None, {s.get_inputs()[0].name: x[i:i + 1]})[0] for i in range(len(x))])


def snr_cmd(a, b, d, rt="ort"):
    """Core-output SNR of graph b vs graph a (spectrum, and waveform after iSTFT) on NVAL val chunks; rt=ov runs b
    on OpenVINO."""
    ch = val_chunks(d, int(os.environ.get("NVAL", 8)))
    xs = K.model_in(ch.to("cpu") if dev == "cpu" else ch.to(dev)).cpu().numpy()
    ya = np.concatenate([K.ort_run(a, xs[i:i + 1]) for i in range(len(xs))])
    yb = ov_run(b, xs) if rt == "ov" else np.concatenate([K.ort_run(b, xs[i:i + 1]) for i in range(len(xs))])
    ta, tb = torch.tensor(ya).to(K.WIN.device), torch.tensor(yb).to(K.WIN.device)
    wv = [snr(K.istft(ta[i:i + 1]), K.istft(tb[i:i + 1])) for i in range(len(ta))]
    print(f"{b} ({rt}) vs {a}: spectrum SNR {snr(ta, tb):.2f} dB, waveform SNR median {np.median(wv):.2f} dB")


if __name__ == "__main__":
    a = sys.argv
    {"ptq": lambda: ptq(a[2], a[3]), "train": lambda: train(a[2], a[3], a[4]),
     "export": lambda: export(a[2], a[3], a[4]), "export_fp32": lambda: export_fp32(a[2], a[3], a[4]), "snr": lambda: snr_cmd(*a[2:])}[a[1]]()
