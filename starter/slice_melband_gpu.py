"""Builds the WebGPU stage chain for the MelBand Kim baseline, the way Better Lyrics Composer's PR #461
describes its setup: a plain trace (no ONNX Runtime fusions) with time attention in 6 band groups, fp16 with the RMSNorms kept
in fp32, one graph per stage. Settings only: no code or graphs of theirs. Bundle format as in slice_melband.py. Run with the bench's torch environment.

    python starter/slice_melband_gpu.py <config.yaml> <ckpt> <out.bin>
"""
import json
import os
import struct
import sys
import tempfile

import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from einops import rearrange

B = os.path.expanduser("~/Code/vocal-separation-bench/bench")
sys.path.insert(0, B)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import web_f16                                                    # noqa: E402  (before export_msst, which takes bench/ off sys.path)
import export_fast as ef                                          # noqa: E402
from export_msst import load                                      # noqa: E402
from slice_melband import Head, Tail, to_bundle_part              # noqa: E402

sys.path = [p for p in sys.path if os.path.abspath(p or ".") != B]   # bench/models.py would shadow MSST's models/
GROUP = 10


class GroupedAttend(torch.nn.Module):
    def forward(self, q, k, v):
        return torch.cat([F.scaled_dot_product_attention(q[i:i + GROUP], k[i:i + GROUP], v[i:i + GROUP]) for i in range(0, q.shape[0], GROUP)], 0)


class Stage(torch.nn.Module):
    def __init__(self, tr, time):
        super().__init__()
        self.tr, self.time = tr, time

    def forward(self, x):                 # [1, T, F, D]
        T, Fq, D = x.shape[1:]
        if self.time:
            y = self.tr(x[0].transpose(0, 1))      # [F, T, D]
            return y.transpose(0, 1)[None]
        return self.tr(x[0])[None]        # [T, F, D]


def split_trees(m, G=8):
    """Split nodes with more than 8 outputs as trees, for the 16 storage buffers WebGPU allows per shader."""
    import numpy as np
    from onnx import helper, numpy_helper
    g = m.graph
    const = {i.name: numpy_helper.to_array(i) for i in g.initializer if i.data_type == 7}
    new, extra = [], []
    for n in g.node:
        if n.op_type == "Split" and len(n.output) > G and len(n.input) == 2 and n.input[1] in const:
            sizes = [int(s) for s in const[n.input[1]]]
            axis = next((a.i for a in n.attribute if a.name == "axis"), 0)
            groups = [list(range(i, min(i + G, len(sizes)))) for i in range(0, len(sizes), G)]
            outs = [f"{n.name}_g{j}" for j in range(len(groups))]
            top = numpy_helper.from_array(np.array([sum(sizes[j] for j in ix) for ix in groups], dtype=np.int64), n.name + "_top_c")
            extra.append(top)
            new.append(helper.make_node("Split", [n.input[0], top.name], outs, axis=axis, name=n.name + "_top"))
            for j, ix in enumerate(groups):
                c = numpy_helper.from_array(np.array([sizes[k] for k in ix], dtype=np.int64), f"{n.name}_g{j}_c")
                extra.append(c)
                new.append(helper.make_node("Split", [outs[j], c.name], [n.output[k] for k in ix], axis=axis, name=f"{n.name}_sub{j}"))
        else:
            new.append(n)
    g.initializer.extend(extra)
    del g.node[:]
    g.node.extend(new)
    return m


def main(cfg, ckpt, out):
    import rotary_embedding_torch.rotary_embedding_torch as ret

    def rotate_half(x):
        x = x.unflatten(-1, (-1, 2))
        return torch.stack((-x[..., 1], x[..., 0]), -1).flatten(-2)
    ret.rotate_half = rotate_half
    torch.set_grad_enabled(False)
    m, c = load("mel_band_roformer", cfg, ckpt)
    frames = int(c.inference.get("chunk_size", c.audio.chunk_size)) // m.stft_kwargs["hop_length"] + 1
    core = ef.FastCore(m, frames).eval()
    bands = len(m.band_split.to_features)
    dim = m.band_split.to_features[0][1].weight.shape[0]
    x = torch.zeros(1, frames, bands, dim)
    stages = [("head", Head(core), torch.zeros(1, 2050, frames, 2), "stft_repr", "x")]
    for i, blk in enumerate(m.layers):
        for layer in blk[-2].layers:
            layer[0].attend = GroupedAttend()
        stages.append((f"tt{i}", Stage(blk[-2], True), x, "x", "y"))
        stages.append((f"ft{i}", Stage(blk[-1], False), x, "x", "y"))
    stages.append(("tail", Tail(core), x, "x", "masks"))
    tmp = tempfile.mkdtemp()      # these are big: point TMPDIR at a disk if /tmp is RAM
    blobs, offset, table = [], 0, []
    for name, mod, ex, inp, outp in stages:
        raw, fold, p = (os.path.join(tmp, f"{name}{s}.onnx") for s in ("_raw", "_fold", ""))
        torch.onnx.export(mod.eval(), (ex,), raw, input_names=[inp], output_names=[outp], opset_version=17, dynamo=False, do_constant_folding=True)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        so.optimized_model_filepath = fold
        so.log_severity_level = 3
        ort.InferenceSession(raw, so, providers=["CPUExecutionProvider"])
        g = onnx.load(fold)
        g = web_f16.split_concat(g, 8)
        g = split_trees(g)
        norm = {n.name for n in g.graph.node if n.op_type in ("ReduceL2", "Clip", "Expand", "Div", "Pow", "ReduceMean", "Sqrt") and "/net.2/" not in n.name}
        if name != "head":        # the band split stays fp32: the sum of squares of raw STFT features overflows fp16, and its 1e-24 epsilon flushes to zero, so a silent band divides 0 by 0
            g = web_f16.convert(g, set(), norm)
        onnx.save(g, p)
        e, offset = to_bundle_part(p, name, False, blobs, offset)
        table.append(e)
        for f in (raw, fold, p):
            os.remove(f)
        print(name, e["graph"][1], e["data"][1], flush=True)
    head = json.dumps({"frames": frames, "bands": bands, "dim": dim, "stages": table}).encode()
    with open(out, "wb") as f:
        f.write(struct.pack("<I", len(head)) + head)
        for b in blobs:
            f.write(b)
    os.rmdir(tmp)
    print(out, os.path.getsize(out) >> 20, "MB")


if __name__ == "__main__":
    main(*sys.argv[1:])
