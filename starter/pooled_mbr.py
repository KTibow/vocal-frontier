"""Pooled MelBand Kim students: musetric's PooledMelBandRoformer (train/musetric_roformer, vendored from musetric/lab,
MIT) built in MelBand Kim's shape and initialised from Kim's own weights, as musetric's duality student was made
from the (Kim-shaped) duality teacher. FINDINGS.md "Pooled MelBand Kim student (agent)".

    python train/pooled_mbr.py config <out config.yaml> [key=<json> ...]  # Kim's config + PooledMelBandRoformer keys
    python train/pooled_mbr.py init <config.yaml> <out.ckpt>              # Kim's weights into the student
    python train/pooled_mbr.py check <config.yaml> [ckpt]                 # pool 1 = Kim? student vs Kim on real audio
    python train/pooled_mbr.py export <config.yaml> <ckpt|-> <name>       # data/models/<name>/{core.onnx,config.yaml}
    python train/pooled_mbr.py convert <model.safetensors> <out.ckpt>      # e.g. musetric's duality student
    python train/pooled_mbr.py derive <base cfg> <out cfg> [key=<json> ...]  # any base config, model keys / chunk_size

e.g. `config data/models/pk_p/config.yaml 'pool=[1,1,4,4,2,2]'` is the pooling-only student: every other weight
of the teacher keeps its shape, so `init` loads Kim's state dict whole. Keys whose shape changes (fewer time heads,
narrower FFN, low-rank mask middle) are cut from Kim's weights by `init` (heads / hidden units by weight norm,
the mask middle by SVD). The export is models.MaskCore's layout (bench/export_msst.py): real STFT [1, (f s), t, 2]
in, per-bin complex masks out, mel-band averaging and zero-DC baked in. Train with
train/distill.py --arch pooled --init <ckpt>.
"""
import json
import os
import sys

import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
KIM = os.path.join(ROOT, "data", "models", "melkim")
POOLED_KEYS = ("pool", "merged_bands_from", "time_window", "time_heads", "ff_mult", "band_map", "detail",
               "mask_rank", "mask_pool")


def load_cfg(path):
    return yaml.safe_load(open(path).read().replace("!!python/tuple", ""))


def build(cfg):
    sys.path.insert(0, HERE)
    from musetric_roformer.pooled_mel_band_roformer import PooledMelBandRoformer
    cfg = load_cfg(cfg) if isinstance(cfg, str) else cfg
    return PooledMelBandRoformer(**cfg["model"])


def kim_state():
    sd = torch.load(os.path.join(KIM, "MelBandRoformer.ckpt"), map_location="cpu", weights_only=False)
    return sd.get("state_dict", sd)


def cmd_config(out, *sets):
    """Kim's config.yaml text with PooledMelBandRoformer keys appended under model: (keeps its !!python/tuple)."""
    txt = open(os.path.join(KIM, "config.yaml")).read()
    extra = {}
    for s in sets:
        k, v = s.split("=", 1)
        assert k in POOLED_KEYS or k == "chunk_size", k
        extra[k] = json.loads(v)
    chunk = extra.pop("chunk_size", None)
    lines = "".join(f"  {k}: {json.dumps(v)}\n" for k, v in extra.items())
    head, rest = txt.split("\nmodel:\n", 1)
    if chunk:
        head = head.replace("chunk_size: 352800", f"chunk_size: {chunk}")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    open(out, "w").write(head + "\nmodel:\n" + lines + rest)
    print(out, extra)


def _cut(new, old, name):
    """Kim weight `old` cut to the student's shape for the reshaped parts the pooled class allows."""
    if new.shape == old.shape:
        return old
    raise ValueError(f"{name}: no rule for {tuple(old.shape)} -> {tuple(new.shape)}")


def surgery(student, sd):
    """Fill the student's reshaped modules from Kim's (fewer heads, narrower FFN, low-rank mask middle)."""
    out = dict(sd)
    st = student.state_dict()
    # narrower FFN: keep the hidden units with the largest |w_in| * |w_out|
    for k in [k for k in st if k.endswith("net.1.weight") and st[k].shape != sd[k].shape]:
        p = k[: -len("net.1.weight")]
        w1, b1, w4 = sd[p + "net.1.weight"], sd[p + "net.1.bias"], sd[p + "net.4.weight"]
        keep = (w1.norm(dim=1) * w4.norm(dim=0)).topk(st[k].shape[0]).indices.sort().values
        out[p + "net.1.weight"], out[p + "net.1.bias"], out[p + "net.4.weight"] = w1[keep], b1[keep], w4[:, keep]
    # fewer time heads: keep the heads with the largest output-projection norm; to_out gets a zero bias
    for k in [k for k in st if k.endswith("to_gates.weight") and st[k].shape != sd[k].shape]:
        p = k[: -len("to_gates.weight")]
        h, H = st[k].shape[0], sd[k].shape[0]
        qkv, wo = sd[p + "to_qkv.weight"], sd[p + "to_out.0.weight"]
        d = wo.shape[1] // H
        keep = wo.view(wo.shape[0], H, d).norm(dim=(0, 2)).topk(h).indices.sort().values
        rows = torch.cat([(keep[:, None] * d + torch.arange(d)).flatten() + j * H * d for j in range(3)])
        out[p + "to_qkv.weight"] = qkv[rows]
        out[p + "to_gates.weight"], out[p + "to_gates.bias"] = sd[k][keep], sd[p + "to_gates.bias"][keep]
        out[p + "to_out.0.weight"] = wo.view(-1, H, d)[:, keep].reshape(wo.shape[0], h * d)
        out[p + "to_out.0.bias"] = torch.zeros(wo.shape[0])
    # low-rank mask middle: Kim's Linear(hidden, hidden) -> rank-r SVD, bias through the left factor's pseudo-inverse
    for k in [k for k in st if k.endswith(".0.2.0.weight") and k.startswith("mask_estimators")]:
        p = k[: -len("2.0.weight")]                          # mask_estimators.0.to_freqs.<b>.0.
        r = st[k].shape[0]
        W, b = sd[p + "2.weight"].double(), sd[p + "2.bias"].double()
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        A, B = (S[:r, None].sqrt() * Vh[:r]), (U[:, :r] * S[:r].sqrt())
        out[p + "2.0.weight"], out[p + "2.1.weight"] = A.float(), B.float()
        out[p + "2.0.bias"] = (torch.linalg.pinv(B) @ b).float()
        out[p + "4.weight"], out[p + "4.bias"] = sd[p + "4.weight"], sd[p + "4.bias"]
        for j in ("2.weight", "2.bias"):
            out.pop(p + j)
    for k in [k for k in out if k in st and out[k].shape != st[k].shape]:
        out[k] = _cut(st[k], out[k], k)
    return out


def cmd_init(cfg, out):
    student = build(cfg)
    sd = surgery(student, kim_state())
    missing, unexpected = student.load_state_dict(sd, strict=False)
    print(f"missing {len(missing)} {missing[:6]}\nunexpected {len(unexpected)} {unexpected[:6]}")
    torch.save(student.state_dict(), out)
    print(f"{out}: {sum(p.numel() for p in student.parameters()) / 1e6:.1f}M params")


def real_audio(seconds=8.0, offset=10.0, track=3):
    import soundfile as sf
    d = os.path.join(ROOT, "data", "sets", "musdb30")
    tr = sorted(os.listdir(d))[track]
    x, sr = sf.read(os.path.join(d, tr, "mixture.flac"), dtype="float32", start=int(offset * 44100))
    n = int(seconds * 44100) // 441 * 441
    return torch.from_numpy(x[:n].T.copy())[None]


def sdr(ref, est):
    return float(10 * torch.log10(ref.pow(2).sum() / (ref - est).pow(2).sum()))


def cmd_check(cfg, ckpt=None):
    sys.path.insert(0, os.path.join(ROOT, "data", "ext", "MSST"))
    from utils.settings import get_model_from_config
    kim, _ = get_model_from_config("mel_band_roformer", os.path.join(KIM, "config.yaml"))
    sd = kim_state()
    kim.load_state_dict(sd)
    kim.eval()
    x = torch.cat([real_audio(track=t) for t in (3, 11, 23, 37)])   # 4 musdb30 clips, 8 s each
    torch.set_grad_enabled(False)
    ref = kim(x)
    same = build(os.path.join(KIM, "config.yaml"))           # PooledMelBandRoformer, all pools 1
    same.load_state_dict(sd)
    y = same.eval()(x)
    print(json.dumps({"pool1_vs_kim_maxabs": float((y - ref).abs().max()), "ref_maxabs": float(ref.abs().max()),
                      "pool1_vs_kim_sdr": sdr(ref, y)}))
    st = build(cfg)
    st.load_state_dict(torch.load(ckpt, map_location="cpu") if ckpt else surgery(st, sd))
    z = st.eval()(x)
    print(json.dumps({"student_vs_kim_sdr": sdr(ref, z), "cfg": cfg, "ckpt": ckpt}))


class Core(torch.nn.Module):
    """net_forward + the host's mel-band mask averaging and zero-DC (bench/export_msst.py Core)."""
    def __init__(self, m):
        super().__init__()
        self.m = m
        F2 = int(m.num_bands_per_freq.numel()) * 2
        keep = torch.ones(F2)
        if m.zero_dc:
            keep[:2] = 0
        self.register_buffer("scale", (keep / m.num_bands_per_freq.repeat_interleave(2).float().clamp(min=1e-8)).view(1, F2, 1, 1))
        self.F2 = F2

    def forward(self, stft):
        mask = self.m.net_forward(stft)[:, 0]                 # [1, f', t, 2]
        idx = self.m.freq_indices.view(1, -1, 1, 1).expand(1, -1, mask.shape[2], 2)
        return torch.zeros(1, self.F2, mask.shape[2], 2, dtype=mask.dtype).scatter_add(1, idx, mask) * self.scale


def _rotate_half(x):
    """rotary_embedding_torch.rotate_half without unbind: torch 2.12's TorchScript exporter wrote its
    unbind(dim=-1) as Split(axis=1) here, which ORT rejects. Same values."""
    x = x.unflatten(-1, (-1, 2))
    return torch.stack((-x[..., 1], x[..., 0]), -1).flatten(-2)


def cmd_export(cfg, ckpt, name):
    import rotary_embedding_torch.rotary_embedding_torch as ret
    ret.rotate_half = _rotate_half
    m = build(cfg)
    m.load_state_dict(torch.load(ckpt, map_location="cpu") if ckpt != "-" else surgery(m, kim_state()))
    m.eval().requires_grad_(False)
    c = load_cfg(cfg)
    hop = m.stft_kwargs["hop_length"]
    frames = int(c["audio"]["chunk_size"]) // hop + 1
    T = hop * (frames - 1)
    audio = real_audio(T / 44100)[..., :T]
    core = Core(m).eval()
    win = m.stft_window_fn(device="cpu")
    z = torch.stft(audio.reshape(2, T), **m.stft_kwargs, window=win, return_complex=True)
    stft = torch.view_as_real(z).permute(1, 0, 2, 3).reshape(1, -1, z.shape[-1], 2).contiguous()
    with torch.inference_mode():
        ref = m(audio)[0]
        mk = core(stft)
        y = torch.view_as_complex(stft.contiguous()) * torch.view_as_complex(mk.contiguous())
        y = torch.istft(y.reshape(-1, 2, y.shape[-1]).transpose(0, 1), **m.stft_kwargs, window=win, length=T)
        diff = float((y - ref).abs().max())
    d = os.path.join(ROOT, "data", "models", name)
    os.makedirs(d, exist_ok=True)
    out = os.path.join(d, "core.onnx")
    with torch.no_grad():
        torch.onnx.export(core, (stft,), out, input_names=["stft_repr"], output_names=["masks"], opset_version=17,
                          dynamo=False, do_constant_folding=True)
    if os.path.abspath(cfg) != os.path.abspath(os.path.join(d, "config.yaml")):
        open(os.path.join(d, "config.yaml"), "w").write(open(cfg).read())
    import onnxruntime as ort
    o = ort.InferenceSession(out, providers=["CPUExecutionProvider"]).run(None, {"stft_repr": stft.numpy()})[0]
    print(json.dumps({"onnx": out, "mb": round(os.path.getsize(out) / 2 ** 20, 1), "frames": frames,
                      "core_host_vs_model_maxdiff": diff, "ort_vs_torch_maxdiff": float(np.abs(mk.numpy() - o).max())}))


def cmd_convert(src, out):
    """A .safetensors file (e.g. musetric's duality student, Apache-2.0) -> torch state dict, without the safetensors
    package: 8-byte little-endian header size, JSON header {name: {dtype, shape, data_offsets}}, raw tensor bytes."""
    import struct
    raw = open(src, "rb").read()
    n = struct.unpack("<Q", raw[:8])[0]
    head = json.loads(raw[8:8 + n])
    dt = {"F32": np.float32, "F16": np.float16, "I64": np.int64, "I32": np.int32}
    sd = {}
    for k, v in head.items():
        if k == "__metadata__":
            continue
        a, b = v["data_offsets"]
        sd[k] = torch.from_numpy(np.frombuffer(raw[8 + n + a:8 + n + b], dt[v["dtype"]]).reshape(v["shape"]).copy())
    torch.save(sd, out)
    print(f"{out}: {len(sd)} tensors, {sum(t.numel() for t in sd.values()) / 1e6:.1f}M")


def cmd_derive(base, out, *sets):
    """A config from any base config (e.g. the duality student's): model keys and audio chunk_size replaced."""
    c = load_cfg(base)
    for s in sets:
        k, v = s.split("=", 1)
        (c["audio"] if k == "chunk_size" else c["model"])[k] = json.loads(v)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    yaml.safe_dump(c, open(out, "w"), sort_keys=False)
    print(out, sets)


if __name__ == "__main__":
    {"config": cmd_config, "init": cmd_init, "check": cmd_check, "export": cmd_export, "convert": cmd_convert,
     "derive": cmd_derive}[sys.argv[1]](*sys.argv[2:])
