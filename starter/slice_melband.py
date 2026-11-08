"""Cuts a MelBand Kim checkpoint into one ONNX graph per stage, in one bundle file for a browser engine (see
baselines/melband-kim-wasm/engine/worker.js). Run with the bench's torch environment. Expect to edit paths.

    python starter/slice_melband.py <config.yaml> <ckpt> <out.bin> [--w16] [--merged-from N]

Stages: head (band split), one graph per time layer and per band layer, tail (mask estimators and mask averaging).
The engine feeds each transformer a few bands or frames at a time, so ONNX Runtime never holds the full 60x8x801x801
attention scores in its 4 GB WASM heap, and no two layers' tensors fragment it. The layers are exact: fused attention
as in the bench's export_fast.py, which matches torch to 6e-7. --merged-from averages neighbouring bands from N up inside each
layer, as PooledMelBandRoformer does: nothing is retrained. --w16 stores fp32 weights in fp16 with a Cast back, so fp32 compute.
"""
import json
import os
import struct
import sys
import tempfile

import numpy as np
import onnx
import torch
from einops import rearrange
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, os.path.expanduser("~/Code/vocal-separation-bench/bench"))
import export_fast as ef                                          # noqa: E402
from export_msst import load                                      # noqa: E402


class Head(torch.nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core

    def forward(self, stft):
        x = stft[:, self.core.m.freq_indices]
        return self.core.band_split(rearrange(x, "b f t c -> b t (f c)"))


class Tail(torch.nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core

    def forward(self, x):
        c = self.core
        mask = rearrange(c.m.mask_estimators[0](x), "b t (f c) -> b f t c", c=2)
        return sum(mask[:, getattr(c, f"src{k}")] * getattr(c, f"w{k}") for k in range(c.K))


def w16(m):
    g, cast = m.graph, []
    for t in list(g.initializer):
        if t.data_type == TensorProto.FLOAT and np.prod(t.dims) >= 1024 and t.name not in ("cos", "sin"):
            a = numpy_helper.to_array(t).astype(np.float16)
            stored = t.name + "_w16"
            cast.append(helper.make_node("Cast", [stored], [t.name], to=TensorProto.FLOAT, name=stored + "_cast"))
            t.CopyFrom(numpy_helper.from_array(a, stored))
    nodes = cast + list(g.node)
    del g.node[:]
    g.node.extend(nodes)


def to_bundle_part(path, name, use_w16, blobs, offset):
    m = onnx.load(path)
    if use_w16:
        w16(m)
    data = []
    size = 0
    for t in m.graph.initializer:
        if t.HasField("raw_data") and len(t.raw_data) >= 1024:
            b = t.raw_data
            onnx.external_data_helper.set_external_data(t, location="w", offset=size, length=len(b))
            t.ClearField("raw_data")
            t.data_location = TensorProto.EXTERNAL
            data.append(b)
            size += len(b)
    graph = m.SerializeToString(deterministic=True)
    blobs.append(graph)
    blobs.extend(data)
    e = {"name": name, "graph": [offset, len(graph)], "data": [offset + len(graph), size]}
    return e, offset + len(graph) + size


def main(cfg, ckpt, out, use_w16=False, first=None):
    torch.set_grad_enabled(False)
    m, c = load("mel_band_roformer", cfg, ckpt)
    hop = m.stft_kwargs["hop_length"]
    frames = int(c.inference.get("chunk_size", c.audio.chunk_size)) // hop + 1
    core = ef.FastCore(m, frames).eval()
    bands = len(m.band_split.to_features)
    tokens = bands
    if first is not None:
        tokens = first + (bands - first) // 2
        pos = torch.tensor(list(range(first)) + [first + 2 * j + .5 for j in range((bands - first) // 2)], dtype=torch.float32)
        for tr, blk in zip(core.ft, m.layers):
            freqs = blk[-1].layers[0][0].rotary_embed.freqs.detach().float()
            ang = pos[:, None] * freqs[None]
            tr.cos, tr.sin = ang.cos().contiguous(), ang.sin().contiguous()
    dim = m.band_split.to_features[0][1].weight.shape[0]
    tmp = tempfile.mkdtemp()      # these are big: point TMPDIR at a disk if /tmp is RAM
    stages = [("head", Head(core), torch.zeros(1, 2050, frames, 2), ["x"], None)]
    for i in range(len(core.tt)):
        stages.append((f"tt{i}", core.tt[i], torch.zeros(10, frames, dim), ["y"], {"x": {0: "b"}, "y": {0: "b"}}))
        stages.append((f"ft{i}", core.ft[i], torch.zeros(10, tokens, dim), ["y"], {"x": {0: "b"}, "y": {0: "b"}}))
    stages.append(("tail", Tail(core), torch.zeros(1, frames, bands, dim), ["masks"], None))
    blobs, offset, table = [], 0, []
    for name, mod, ex, outs, axes in stages:
        p = os.path.join(tmp, name + ".onnx")
        torch.onnx.export(mod.eval(), (ex,), p, input_names=["stft_repr" if name == "head" else "x"], output_names=outs, opset_version=17,
                          dynamo=False, do_constant_folding=True, custom_opsets={"com.microsoft": 1}, dynamic_axes=axes)
        e, offset = to_bundle_part(p, name, use_w16, blobs, offset)
        table.append(e)
        os.remove(p)
        print(name, e["graph"][1], e["data"][1], flush=True)
    head = json.dumps({"frames": frames, "bands": bands, "tokens": tokens, "dim": dim, "layers": len(core.tt), "merged_from": first, "stages": table}).encode()
    with open(out, "wb") as f:
        f.write(struct.pack("<I", len(head)) + head)
        for b in blobs:
            f.write(b)
    os.rmdir(tmp)
    print(out, os.path.getsize(out) >> 20, "MB")


if __name__ == "__main__":
    a = sys.argv[1:]
    w = "--w16" in a
    first = int(a[a.index("--merged-from") + 1]) if "--merged-from" in a else None
    main(a[0], a[1], a[2], w, first)
