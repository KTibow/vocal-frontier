"""Export a CpuNet core to data/models/<name>/core.onnx (+ config.yaml), runnable by bench/models.MaskCore.

    python train/export_cpunet.py <config.yaml> <name> [ckpt]

config.yaml: {model: CpuNet kwargs, audio: {chunk_size: samples}}. Without a ckpt the weights are random
(for speed screening).
"""
import json
import os
import sys

import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
from cpunet import CpuNet  # noqa: E402


def build(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    return CpuNet(**cfg["model"]), cfg


def main(cfg_path, name, ckpt=None):
    m, cfg = build(cfg_path)
    if ckpt:
        m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    m.eval()
    chunk = int(cfg["audio"]["chunk_size"])
    frames = chunk // m.hop + 1
    audio = torch.randn(1, 2, m.hop * (frames - 1)) * 0.1
    rep = m.stft(audio)
    od = os.path.join(ROOT, "data", "models", name)
    os.makedirs(od, exist_ok=True)
    out = os.path.join(od, "core.onnx")
    with torch.no_grad():
        ref = m.core(rep).numpy()
        torch.onnx.export(m, (rep,), out, input_names=["stft_repr"], output_names=["masks"], opset_version=17,
                          dynamo=False, do_constant_folding=True, **{})
    c = dict(cfg)
    c["model"] = dict(cfg["model"], stft_hop_length=m.hop)
    yaml.safe_dump(c, open(os.path.join(od, "config.yaml"), "w"))
    import onnxruntime as ort
    o = ort.InferenceSession(out, providers=["CPUExecutionProvider"]).run(None, {"stft_repr": rep.numpy()})[0]
    print(json.dumps({"name": name, "params_m": round(sum(p.numel() for p in m.parameters()) / 1e6, 2),
                      "mb": round(os.path.getsize(out) / 2 ** 20, 1), "frames": frames,
                      "ort_vs_torch_maxdiff": float(np.abs(ref - o).max())}))


if __name__ == "__main__":
    # torch.onnx.export traces forward(); point it at the core
    CpuNet.forward_raw = CpuNet.forward
    CpuNet.forward = lambda self, rep: self.core(rep)
    main(*sys.argv[1:])
