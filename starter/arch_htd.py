"""Training-free architecture probes of HTDemucs (composer's checkpoint 955717e8): export modified cores to ONNX in the
bench/models.py HtDemucs layout (same inputs "input" [1, 2, seg] and "x" [1, 4, 2048, frames], outputs "output"
and "add_67"), so `htdemucs_<name>` auto-registers and `htdemucs_<name>_ov` runs on OpenVINO.
FINDINGS.md "Architecture optimizations: prior art and quick tests (agent)".

    python train/arch_htd.py <name> [spec]     # -> data/models/htdemucs_<name>/core.onnx

spec = comma-separated options (none = a plain re-export, checked against composer's file):
  crop=K        spectral branch sees only the lowest K of 2048 bins (Slice in, zero Pad out); 1024 = 11 kHz, 768 = 8.3 kHz;
                a K that is not a multiple of 256 (e.g. 896) gets its freq axis zero-padded to a multiple of 4 before
                each encoder layer and trimmed back to the skip's size before each decoder layer
  notime        time-branch decoder skipped, time output zeros (vocals = spectral branch only)
  tpool=k       cross-transformer runs on k-times time-pooled tokens of both branches, residual form
                x + up(F(down x) - down x) (k=1 is exact), positions = pooled full-length sin embeddings
  tpoolp=k      same, plain form up(F(down x))
  layers=a.b.c  keep only these cross-transformer layers (of 0-4; even = self, odd = cross attention)
  nodconv       drop every DConv residual branch (LayerScale-gated, init 1e-3)
  seg=N         segment of N samples instead of 343980 (frames = ceil(N / 1024))
  ch=C          random weights, base channels C (speed screens only); hid=S transformer hidden scale (random)
CKPT=<state dict .pt> exports trained weights of the demucs HTDemucs instead of checkpoint 955717e8 (the cuts above
change no parameter shapes except nodconv / ch / hid).
"""
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")


def parse(spec):
    o = {}
    for p in filter(None, (spec or "").split(",")):
        k, _, v = p.partition("=")
        o[k] = v or "1"
    return o


def pooled_ct(ct, k, residual):
    """CrossTransformerEncoder.forward on k-times time-pooled tokens."""
    from demucs.transformer import create_2d_sin_embedding, create_sin_embedding

    def down(t):                      # [..., T] -> [..., T / k]
        return t.unflatten(-1, (-1, k)).mean(-1)

    def up(t, n):
        return t.unsqueeze(-1).expand(*t.shape, k).flatten(-2)[..., :n]

    def fwd(x, xt):
        B, C, Fr, T1 = x.shape
        T2 = xt.shape[-1]
        xd, xtd = down(x), down(xt)
        pe2 = down(create_2d_sin_embedding(C, Fr, T1, x.device, ct.max_period))
        pe2 = rearrange(pe2, "b c fr t1 -> b (t1 fr) c")
        y = rearrange(xd, "b c fr t1 -> b (t1 fr) c")
        y = ct.norm_in(y) + ct.weight_pos_embed * pe2
        pe = create_sin_embedding(T2, C, device=x.device, max_period=ct.max_period)    # [T2, 1, C]
        pe = down(rearrange(pe, "t b c -> b c t"))
        pe = rearrange(pe, "b c t -> b t c")
        yt = ct.norm_in_t(rearrange(xtd, "b c t -> b t c")) + ct.weight_pos_embed * pe
        for idx in range(ct.num_layers):
            if idx in ct.keep_layers:
                if idx % 2 == ct.classic_parity:
                    y, yt = ct.layers[idx](y), ct.layers_t[idx](yt)
                else:
                    old = y
                    y, yt = ct.layers[idx](y, yt), ct.layers_t[idx](yt, old)
        y = rearrange(y, "b (t1 fr) c -> b c fr t1", t1=T1 // k)
        yt = rearrange(yt, "b t c -> b c t")
        if residual:
            return x + up(y - xd, T1), xt + up(yt - xtd, T2)
        return up(y, T1), up(yt, T2)
    return fwd


def build(o):
    from demucs.pretrained import get_model
    m = get_model("htdemucs").models[0]
    if "ch" in o or "hid" in o:
        from demucs.htdemucs import HTDemucs
        torch.manual_seed(0)
        args, kw = m._init_args_kwargs
        kw = dict(kw, channels=int(o.get("ch", kw.get("channels", 48))),
                  t_hidden_scale=float(o.get("hid", kw.get("t_hidden_scale", 4.0))))
        m = HTDemucs(*args, **kw)
    if os.environ.get("CKPT"):          # trained weights of the same (uncut) HTDemucs, e.g. a distilled probe
        sd = torch.load(os.environ["CKPT"], map_location="cpu")
        m.load_state_dict(sd.get("state_dict", sd.get("model", sd)) if isinstance(sd, dict) else sd)
    m.eval()
    ct = m.crosstransformer
    ct.keep_layers = [int(i) for i in o["layers"].split(".")] if "layers" in o else list(range(ct.num_layers))
    k = int(o.get("tpool", o.get("tpoolp", 1)))
    if k > 1 or "layers" in o:
        ct.forward = pooled_ct(ct, k, "tpoolp" not in o)
    if "nodconv" in o:
        for layer in list(m.encoder) + list(m.tencoder):
            layer.dconv = None
    return m


class Probe(torch.nn.Module):
    """bench/litert_export.py HTCore (demucs HTDemucs.forward between _magnitude and _mask/_ispec, as sevagh's
    demucs.onnx cuts it) plus the probes: crop (spectrum stats still over all 2048 bins), notime."""
    def __init__(self, m, o):
        super().__init__()
        self.m = m
        self.crop = int(o.get("crop", 0))
        self.notime = "notime" in o

    def forward(self, mix, x):
        m = self.m
        B, C, Fq, T = x.shape
        length = mix.shape[-1]
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)
        if self.crop:
            x = x[:, :, :self.crop]
        xt = mix
        meant = xt.mean(dim=(1, 2), keepdim=True)
        stdt = xt.std(dim=(1, 2), keepdim=True)
        xt = (xt - meant) / (1e-5 + stdt)
        saved, saved_t, lengths, lengths_t = [], [], [], []
        for idx, encode in enumerate(m.encoder):
            lengths.append(x.shape[-1])
            inject = None
            if idx < len(m.tencoder):
                lengths_t.append(xt.shape[-1])
                tenc = m.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    saved_t.append(xt)
                else:
                    inject = xt
            if x.shape[2] % 4:          # crop not a multiple of 256: pad freq so the k8/s4 conv tiles exactly
                x = F.pad(x, (0, 0, 0, 4 - x.shape[2] % 4))
            x = encode(x, inject)
            if idx == 0 and m.freq_emb is not None:
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = m.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + m.freq_emb_scale * emb
            saved.append(x)
        if m.bottom_channels:
            f = x.shape[2]
            x = rearrange(x, "b c f t-> b c (f t)")
            x = m.channel_upsampler(x)
            x = rearrange(x, "b c (f t)-> b c f t", f=f)
            xt = m.channel_upsampler_t(xt)
        x, xt = m.crosstransformer(x, xt)
        if m.bottom_channels:
            x = rearrange(x, "b c f t-> b c (f t)")
            x = m.channel_downsampler(x)
            x = rearrange(x, "b c (f t)-> b c f t", f=f)
            xt = m.channel_downsampler_t(xt)
        for idx, decode in enumerate(m.decoder):
            skip = saved.pop(-1)
            if x.shape[2] > skip.shape[2]:   # padded level (see crop): drop the padded bins
                x = x[:, :, :skip.shape[2]]
            x, pre = decode(x, skip, lengths.pop(-1))
            offset = m.depth - len(m.tdecoder)
            if idx >= offset and not self.notime:
                tdec = m.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    pre = pre[:, :, 0]
                    xt, _ = tdec(pre, None, length_t)
                else:
                    skip = saved_t.pop(-1)
                    xt, _ = tdec(xt, skip, length_t)
        S = len(m.sources)
        x = x.view(B, S, -1, x.shape[-2], T)
        x = x * std[:, None] + mean[:, None]
        if self.crop:
            x = F.pad(x, (0, 0, 0, Fq - self.crop))
        if self.notime:
            return x, torch.zeros(B, S, 2, length, device=mix.device) + meant[:, None]
        xt = xt.view(B, S, -1, length)
        xt = xt * stdt[:, None] + meant[:, None]
        return x, xt


def main(name, spec=""):
    torch.backends.mha.set_fastpath_enabled(False)   # the fast path exports as an op ONNX lacks
    o = parse(spec)
    m = build(o)
    seg = int(o.get("seg", 343980))
    frames = -(-seg // 1024)
    mix = torch.randn(1, 2, seg) * 0.1
    x = torch.randn(1, 4, 2048, frames) * 0.1
    p = Probe(m, o).eval()
    d = os.path.join(ROOT, "data", "models", f"htdemucs_{name}")
    os.makedirs(d, exist_ok=True)
    out = os.path.join(d, "core.onnx")
    with torch.no_grad():
        y, yt = p(mix, x)
        torch.onnx.export(p, (mix, x), out, input_names=["input", "x"], output_names=["output", "add_67"],
                          opset_version=17, dynamo=False, do_constant_folding=True)
    import onnxruntime as ort
    s = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    a, b = s.run(["output", "add_67"], {"input": mix.numpy(), "x": x.numpy()})
    print({"onnx": out, "mb": round(os.path.getsize(out) / 2 ** 20, 1), "shapes": [list(a.shape), list(b.shape)],
           "ort_vs_torch": float(max(np.abs(a - y.numpy()).max(), np.abs(b - yt.numpy()).max())),
           "params_M": round(sum(q.numel() for q in m.parameters()) / 1e6, 2)})


if __name__ == "__main__":
    main(*sys.argv[1:])
