"""Training-free architecture probes of UVR MDX-Net cores (Voc_FT / Kim Vocal 2, TFC-TDF U-Net, ONNX graph surgery
only). FINDINGS.md "Architecture optimizations: prior art and quick tests (agent)".

    python train/arch_mdx.py crop  <src.onnx> <K> <out.onnx>               # dim_f 3072 -> K bins (TDF weights sliced)
    python train/arch_mdx.py prune <src.onnx> <keep> <out.onnx> [levels]   # channel pruning, see below
    python train/arch_mdx.py prune_ls <src.onnx> <keep> <out.onnx> [chunks] [rows/chunk]   # + least-squares refit
    python train/arch_mdx.py macs <src.onnx> [keep spec]               # MACs per level, and of a keep spec vs parent
    python train/arch_mdx.py shape <src.onnx> <keep spec> <out.onnx>   # data-free slice (first channels): speed only

keep spec (prune_ls, macs): a fraction for every group ("0.75"), or comma-separated key=value with value a fraction
  (<= 1) or a channel count (> 1); keys: L<n> (both encoder and decoder groups of U-Net level n, 0 = full
  resolution, 5 = bottleneck), E<n> / D<n> (encoder / decoder side only), G:<conv name> (one group, e.g. G:Conv_0 =
  the 1x1 input conv's output, G:Conv_167 = what the 1x1 output conv reads), * (default; 1.0 when absent). Later,
  more specific keys win (G > E/D > L > *). E.g. "L0=1,L1=0.75,L2=0.6,L3=0.5,L4=0.5,L5=0.5".

crop: the net sees only the lowest K STFT bins (K divisible by 32; 2048 = 11.8 kHz at n_fft 7680); each TDF layer's
  dense freq->freq/8 / freq/8->freq pair keeps the rows / columns of the bins that remain. Run it with
  models.MdxNet(..., dim_f=K).
prune: structured channel pruning. Channels that must stay aligned (a level's residual stream, its skip and the
  decoder's multiplicative skip) are grouped by union-find over the graph; each group keeps the `keep` fraction of
  channels with the largest mean |BN scale| (or the producing Conv's output-channel norm), at the U-Net levels
  listed (default all). Removed channels' mean activations (8 MUSDB18-HQ *train* chunks) are folded into the
  biases of the Convs that consumed them, then every BatchNorm's running stats are re-estimated on the same chunks
  (two passes). Input and output channels are never pruned.
"""
import collections
import os
import sys

import numpy as np
import onnx
from onnx import numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "bench"))


def arr(init, n):
    return numpy_helper.to_array(init[n])


def put(init, n, a):
    init[n].CopyFrom(numpy_helper.from_array(np.ascontiguousarray(a, np.float32), n))


def crop(src, K, out):
    K = int(K)
    m = onnx.load(src)
    g = m.graph
    init = {i.name: i for i in g.initializer}
    F0 = g.input[0].type.tensor_type.shape.dim[2].dim_value
    r = K / F0
    for n in g.node:
        if n.op_type == "MatMul" and n.input[1] in init:
            w = arr(init, n.input[1])
            a, b = w.shape
            put(init, n.input[1], w[:round(a * r)] if a > b else w[:, :round(b * r)])
    g.input[0].type.tensor_type.shape.dim[2].dim_value = K
    for o in g.output:
        o.type.tensor_type.shape.dim[2].dim_value = K
    for v in list(g.value_info):
        g.value_info.remove(v)
    onnx.save(m, out)
    print(out, os.path.getsize(out) >> 20, "MB")


def calib_inputs(n=8, dim_f=3072):
    """MdxNet core inputs from MUSDB18-HQ train mixtures (not an eval set)."""
    from audio import load
    from models import MdxNet
    d = os.environ.get("CALIB_DIR") or os.path.join(ROOT, "data", "musdb_train")   # CALIB_DIR: any <dir>/<t>/mixture.flac set
    tracks = sorted(os.listdir(d))[::12][:n]
    xs = []

    class Grab(MdxNet):
        def __init__(self):
            self.n_fft, self.hop, self.dim_f, self.dim_t, self.comp = 7680, 1024, dim_f, 256, 1.0
            self.chunk = 1024 * 255

        def run(self, spec):
            xs.append(spec)
            return np.zeros_like(spec)
    gr = Grab()
    for t in tracks:
        x = load(os.path.join(d, t, "mixture.flac"))
        s = x.shape[1] // 2
        gr(np.ascontiguousarray(x[:, s:s + gr.chunk]))
    return xs


def run_with(m, names, xs):
    """ORT run of model m with extra graph outputs `names`; per-channel (axis 1) mean and var over the chunks."""
    import onnxruntime as ort
    m2 = onnx.ModelProto()
    m2.CopyFrom(m)
    for v in list(m2.graph.value_info):
        m2.graph.value_info.remove(v)
    have = {o.name for o in m2.graph.output}
    for nm in names:
        if nm not in have:
            m2.graph.output.append(onnx.helper.make_empty_tensor_value_info(nm))
    acc = collections.defaultdict(lambda: [0.0, 0.0, 0])
    for gi in range(0, len(names), 12):        # 12 tensors per run, no ORT arena: bounded RAM (~2 GB, not ~10)
        part = names[gi:gi + 12]
        s = _subgraph_session(m, part)
        for x in xs:
            outs = s.run(part, {s.get_inputs()[0].name: x})
            for nm, o in zip(part, outs):
                ax = tuple(i for i in range(o.ndim) if i != 1)
                a = acc[nm]
                a[0] += o.mean(ax, dtype=np.float64)
                a[1] += np.square(o, dtype=np.float64).mean(ax)
                a[2] += 1
            del outs
        del s
    return {k: (v[0] / v[2], v[1] / v[2] - (v[0] / v[2]) ** 2) for k, v in acc.items()}


def bn_train_stats(model, xs):
    """Per BatchNorm node (graph order): mean / var of its input over the chunks with every BN in train mode."""
    import torch
    from onnx2torch import convert
    m2 = onnx.ModelProto()
    m2.CopyFrom(model)
    for v in list(m2.graph.value_info):
        m2.graph.value_info.remove(v)
    net = convert(m2)
    mods = [mod for mod in net.modules() if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm)]
    init = {i.name: i for i in model.graph.initializer}
    bnm = []                                   # graph order, matched by the running-mean values (module order may differ)
    for n in [n for n in model.graph.node if n.op_type == "BatchNormalization"]:
        rm = torch.from_numpy(arr(init, n.input[3]).copy())
        hit = [mod for mod in mods if mod.running_mean.shape == rm.shape and torch.equal(mod.running_mean.cpu(), rm)
               and all(mod is not b for b in bnm)]
        assert hit, n.name
        bnm.append(hit[0])
    for mod in bnm:
        mod.train()
        mod.momentum = None
        mod.reset_running_stats()
    torch.set_num_threads(4)
    with torch.no_grad():
        for x in xs:
            net(torch.from_numpy(x))
    return [(mod.running_mean.double().numpy(), mod.running_var.double().numpy()) for mod in bnm]


def prune(src, keep, out, levels=None):
    keep = float(keep)
    m = onnx.load(src)
    g = m.graph
    init = {i.name: i for i in g.initializer}
    par = {}

    def find(a):
        while par.setdefault(a, a) != a:
            a = par[a]
        return a

    def union(a, b):
        par[find(a)] = find(b)
    grp = {g.input[0].name: "IN"}          # tensor -> group seed
    for n in g.node:
        ins = [i for i in n.input if i not in init]
        if n.op_type in ("Conv", "ConvTranspose"):
            grp[n.output[0]] = "G:" + n.name
        elif n.op_type in ("BatchNormalization", "Relu", "MatMul", "Transpose"):
            grp[n.output[0]] = grp[ins[0]]
        elif n.op_type in ("Add", "Mul"):
            for i in ins[1:]:
                union(grp[ins[0]], grp[i])
            grp[n.output[0]] = grp[ins[0]]
        else:
            raise ValueError(n.op_type)
    out_name = g.output[0].name
    fixed = {find("IN"), find(grp[out_name])}
    G = lambda t: find(grp[t])
    # level of a group = log2(3072 / its freq size), from the Conv producing it; channel count from its weight
    size, imp, level = {}, collections.defaultdict(float), {}
    conv_level, lvl = {}, 0
    for n in g.node:
        if n.op_type == "Conv" and arr(init, n.input[1]).shape[-1] == 2:
            lvl += 1
        if n.op_type == "ConvTranspose":
            lvl -= 1
        if n.op_type in ("Conv", "ConvTranspose"):
            w = arr(init, n.input[1])
            o = w.shape[1] if n.op_type == "ConvTranspose" else w.shape[0]
            k = G(n.output[0])
            size[k] = o
            level.setdefault(k, lvl)
            nrm = np.abs(w).sum(axis=tuple(i for i in range(w.ndim) if i != (1 if n.op_type == "ConvTranspose" else 0)))
            imp[k] = imp[k] + nrm / nrm.mean() if isinstance(imp[k], np.ndarray) else nrm / nrm.mean()
    for n in g.node:
        if n.op_type == "BatchNormalization":
            k = G(n.input[0])
            gam = np.abs(arr(init, n.input[1])) / np.sqrt(arr(init, n.input[4]) + 1e-5)
            imp[k] = imp[k] + 4 * gam / gam.mean()          # BN-scaled weight dominates where there is one
    levels = None if levels is None else {int(v) for v in str(levels).split(",")}
    kept = {}
    for k, c in size.items():
        if k in fixed or (levels is not None and level[k] not in levels):
            continue
        kept[k] = np.sort(np.argsort(-imp[k])[:max(1, int(round(c * keep)))])
    print({"groups": len(size), "pruned": len(kept), "channels": {k: f"{size[k]}->{len(v)} L{level[k]}" for k, v in kept.items()}})
    if os.environ.get("ZERO") == "1":            # sanity probe: zero the removed channels instead of slicing them
        for n in g.node:
            if n.op_type in ("Conv", "ConvTranspose") and G(n.output[0]) in kept:
                ct = n.op_type == "ConvTranspose"
                w = arr(init, n.input[1]).copy()
                gone = np.setdiff1d(np.arange(w.shape[1 if ct else 0]), kept[G(n.output[0])])
                if ct:
                    w[:, gone] = 0
                else:
                    w[gone] = 0
                put(init, n.input[1], w)
                if len(n.input) > 2:
                    b = arr(init, n.input[2]).copy()
                    b[gone] = 0
                    put(init, n.input[2], b)
            elif n.op_type == "BatchNormalization" and G(n.input[0]) in kept:
                gone = np.setdiff1d(np.arange(arr(init, n.input[1]).shape[0]), kept[G(n.input[0])])
                for i in n.input[1:3]:
                    a = arr(init, i).copy()
                    a[gone] = 0
                    put(init, i, a)
        onnx.save(m, out)
        return print(out, "zeroed")
    # mean activations of every tensor a Conv/ConvTranspose consumes, before any change
    xs = calib_inputs()
    cons = [n for n in g.node if n.op_type in ("Conv", "ConvTranspose")]
    stats = run_with(m, sorted({n.input[0] for n in cons if n.input[0] != g.input[0].name}), xs)
    bns = [n for n in g.node if n.op_type == "BatchNormalization"]
    run0 = {n.name: (arr(init, n.input[3]), arr(init, n.input[4])) for n in bns}
    orig_tr = bn_train_stats(m, xs)
    for n in g.node:
        if n.op_type in ("Conv", "ConvTranspose"):
            w = arr(init, n.input[1])
            ct = n.op_type == "ConvTranspose"
            ko, ki = kept.get(G(n.output[0])), kept.get(G(n.input[0])) if n.input[0] in grp else None
            b = arr(init, n.input[2]) if len(n.input) > 2 else np.zeros(w.shape[1 if ct else 0], np.float32)
            if ki is not None:                               # fold removed inputs' means into the bias
                gone = np.setdiff1d(np.arange(w.shape[0 if ct else 1]), ki)
                mu = stats[n.input[0]][0][gone]
                wg = w[gone] if ct else w[:, gone]
                b = b + (np.einsum("iokl,i->o", wg, mu) / (wg.shape[-1] * wg.shape[-2] if ct else 1) if ct
                         else np.einsum("oikl,i->o", wg, mu))
                w = w[ki] if ct else w[:, ki]
            if ko is not None:
                w = w[:, ko] if ct else w[ko]
                b = b[ko]
            put(init, n.input[1], w)
            if len(n.input) > 2:
                put(init, n.input[2], b)
        elif n.op_type == "BatchNormalization":
            ko = kept.get(G(n.input[0]))
            if ko is not None:
                for i in n.input[1:]:
                    put(init, i, arr(init, i)[ko])
    for v in list(g.value_info):
        g.value_info.remove(v)
    # BatchNorm re-estimation, relative to the unpruned net: both nets run with every BatchNorm in train mode (batch
    # statistics, so each layer sees already-corrected inputs in one pass) on the same chunks, and each pruned BN gets
    # its trained running stats shifted / scaled by how its input's statistics changed
    if os.environ.get("RECAL", "1") == "1":
        st = bn_train_stats(m, xs)
        for i, n in enumerate(bns):
            ko = kept.get(G(n.input[0]))
            sl = (lambda a: a[ko]) if ko is not None else (lambda a: a)
            mu, var = st[i]
            mu0, var0 = sl(orig_tr[i][0]), sl(orig_tr[i][1])
            rm, rv = sl(run0[n.name][0]), sl(run0[n.name][1])
            put(init, n.input[3], rm + (mu - mu0))
            put(init, n.input[4], np.maximum(rv * var / np.maximum(var0, 1e-12), 1e-8))
    onnx.save(m, out)
    print(out, os.path.getsize(out) >> 20, "MB")


def _groups(m):
    """Channel groups of an MDX-Net graph (see prune): tensor -> group, group sizes, U-Net levels, fixed groups."""
    g = m.graph
    init = {i.name: i for i in g.initializer}
    par = {}

    def find(a):
        while par.setdefault(a, a) != a:
            a = par[a]
        return a
    grp = {g.input[0].name: "IN"}
    for n in g.node:
        ins = [i for i in n.input if i not in init]
        if n.op_type in ("Conv", "ConvTranspose"):
            grp[n.output[0]] = "G:" + n.name
        elif n.op_type in ("BatchNormalization", "Relu", "MatMul", "Transpose"):
            grp[n.output[0]] = grp[ins[0]]
        elif n.op_type in ("Add", "Mul"):
            for i in ins[1:]:
                par[find(grp[i])] = find(grp[ins[0]])
            grp[n.output[0]] = grp[ins[0]]
    G = lambda t: find(grp[t])
    size, level, lvl, dec, up = {}, {}, 0, {}, False
    for n in g.node:
        if n.op_type == "Conv" and arr(init, n.input[1]).shape[-1] == 2:
            lvl += 1
        if n.op_type == "ConvTranspose":
            lvl -= 1
            up = True
        if n.op_type in ("Conv", "ConvTranspose"):
            w = arr(init, n.input[1])
            size[G(n.output[0])] = w.shape[1] if n.op_type == "ConvTranspose" else w.shape[0]
            level.setdefault(G(n.output[0]), lvl)
            dec.setdefault(G(n.output[0]), up)
    fixed = {G(g.input[0].name), G(g.output[0].name)}
    _groups.dec = dec
    return G, size, level, fixed, init


def keep_counts(spec, G, size, level, fixed):
    """keep spec (see the module doc) -> {group: number of channels kept}, only for groups that lose channels."""
    spec = str(spec)
    dec = _groups.dec
    if "=" not in spec:
        spec = f"*={spec}"
    kv = dict(x.split("=") for x in spec.split(",") if x)
    out = {}
    for k, c in size.items():
        if k in fixed:
            continue
        side = ("D" if dec[k] else "E") + str(level[k])
        v = kv.get(k, kv.get(side, kv.get(f"L{level[k]}", kv.get("*", "1"))))
        v = float(v)
        n = int(v) if v > 1 else max(1, int(round(c * v)))
        if n < c:
            out[k] = min(n, c)
    return out


def mac_table(m, counts=None):
    """MACs of every Conv / ConvTranspose / MatMul at batch 1 (shape inference), scaled by the kept fraction of the
    channels it reads and writes (counts: {group: kept}, None = parent). Returns {(level, op type): MACs}."""
    from onnx import shape_inference
    m2 = onnx.ModelProto(); m2.CopyFrom(m)
    m2.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1
    for o in m2.graph.output:
        o.type.tensor_type.shape.dim[0].dim_value = 1
    for v in list(m2.graph.value_info):
        m2.graph.value_info.remove(v)
    m2 = shape_inference.infer_shapes(m2)
    shp = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim] for v in
           list(m2.graph.value_info) + list(m2.graph.input) + list(m2.graph.output)}
    G, size, level, fixed, init = _groups(m)
    counts = counts or {}
    fr = lambda t: counts.get(G(t), size.get(G(t), 1)) / size.get(G(t), 1) if t in shp and t != m.graph.input[0].name else 1.0
    tab = collections.defaultdict(float)
    for n in m.graph.node:
        if n.op_type not in ("Conv", "ConvTranspose", "MatMul"):
            continue
        w = arr(init, n.input[1]).shape
        if n.op_type == "Conv":
            mac = np.prod(shp[n.output[0]]) * w[1] * w[2] * w[3]
            f = fr(n.input[0]) * fr(n.output[0])
        elif n.op_type == "ConvTranspose":
            mac = np.prod(shp[n.input[0]]) * w[1] * w[2] * w[3]
            f = fr(n.input[0]) * fr(n.output[0])
        else:
            mac = np.prod(shp[n.output[0]]) * w[0]
            f = fr(n.input[0])
        tab[(level.get(G(n.output[0]), 0), n.op_type)] += mac * f
    return tab


def macs(src, spec=None):
    m = onnx.load(src)
    G, size, level, fixed, init = _groups(m)
    t0 = mac_table(m)
    tot0 = sum(t0.values())
    lv = sorted({k[0] for k in t0})
    print("parent GMACs per 5.9 s chunk: %.1f" % (tot0 / 1e9))
    for l in lv:
        print(f"  L{l}: " + ", ".join(f"{op} {v / 1e9:.1f}" for (ll, op), v in sorted(t0.items()) if ll == l)
              + f"  = {sum(v for (ll, _), v in t0.items() if ll == l) / tot0:.1%}")
    if spec:
        c = keep_counts(spec, G, size, level, fixed)
        t1 = mac_table(m, c)
        print(f"spec {spec}: {sum(t1.values()) / tot0:.1%} of the parent's MACs; per level " +
              ", ".join(f"L{l} {sum(v for (ll, _), v in t1.items() if ll == l) / max(1, sum(v for (ll, _), v in t0.items() if ll == l)):.0%}" for l in lv))
        print({k: f"{size[k]}->{v}" for k, v in c.items()})
    return t0


def _subgraph_session(m, names, threads=4):
    """ORT session that computes only `names` (graph outputs replaced, nodes not upstream of them dropped)."""
    import onnxruntime as ort
    m2 = onnx.ModelProto(); m2.CopyFrom(m)
    g = m2.graph
    for v in list(g.value_info):
        g.value_info.remove(v)
    prod = {o: n for n in g.node for o in n.output}
    need, todo = set(), list(names)
    while todo:
        t = todo.pop()
        n = prod.get(t)
        if n is None or id(n) in need:
            continue
        need.add(id(n))
        todo += list(n.input)
    nodes = [n for n in g.node if id(n) in need]
    used = {i for n in nodes for i in n.input}
    del g.node[:]
    g.node.extend(nodes)
    keep_i = [i for i in g.initializer if i.name in used]
    del g.initializer[:]
    g.initializer.extend(keep_i)
    del g.output[:]
    for nm in names:
        g.output.append(onnx.helper.make_empty_tensor_value_info(nm))
    so = ort.SessionOptions(); so.intra_op_num_threads = threads
    so.enable_cpu_mem_arena = False
    return ort.InferenceSession(m2.SerializeToString(), so, providers=["CPUExecutionProvider"])


def _attrs(n):
    a = {x.name: onnx.helper.get_attribute_value(x) for x in n.attribute}
    k = a.get("kernel_shape", [1, 1]); st = a.get("strides", [1, 1]); pd = a.get("pads", [0, 0, 0, 0])
    assert a.get("group", 1) == 1
    return k, st, pd


def _pos(shape_x, shape_y, n, ct, rs):
    """n random sample positions: input grid (i, j) for a ConvTranspose, output grid for a Conv."""
    H, W = (shape_x if ct else shape_y)[1:]
    return rs.randint(0, H, n), rs.randint(0, W, n)


def _feat(x, pos, k, st, pd, ct, cin):
    i, j = pos
    if ct:
        return x[cin][:, i, j].T
    xp = np.pad(x[cin], ((0, 0), (pd[0], pd[2]), (pd[1], pd[3])))
    di, dj = np.meshgrid(np.arange(k[0]), np.arange(k[1]), indexing="ij")
    rows = xp[:, (i * st[0])[:, None, None] + di[None], (j * st[1])[:, None, None] + dj[None]]   # [C', n, kh, kw]
    return rows.transpose(1, 0, 2, 3).reshape(len(i), -1)


def _targ(y, pos, k, st, ct, cout):
    i, j = pos
    if ct:                                   # out[2i+di, 2j+dj] = x[:, i, j] . W[:, :, di, dj]
        return {(di, dj): y[cout][:, st[0] * i + di, st[1] * j + dj].T for di in range(k[0]) for dj in range(k[1])}
    return {None: y[cout][:, i, j].T}


def _session(m, names):
    import onnxruntime as ort
    m2 = onnx.ModelProto(); m2.CopyFrom(m)
    for v in list(m2.graph.value_info):
        m2.graph.value_info.remove(v)
    have = {o.name for o in m2.graph.output}
    for nm in names:
        if nm not in have:
            m2.graph.output.append(onnx.helper.make_empty_tensor_value_info(nm))
    so = ort.SessionOptions(); so.intra_op_num_threads = 4
    return ort.InferenceSession(m2.SerializeToString(), so, providers=["CPUExecutionProvider"])


def _fit(n, w, F_, T_):
    """ridge LS (bias column) of one Conv / ConvTranspose: returns weight [O', I', kh, kw] or [I', O', kh, kw], bias."""
    ct = n.op_type == "ConvTranspose"
    new_w, new_b, res = None, None, []
    for tap in T_:
        X = np.concatenate(F_).astype(np.float64); Y = np.concatenate(T_[tap]).astype(np.float64)
        X1 = np.concatenate([X, np.ones((len(X), 1))], 1)
        A = X1.T @ X1
        A += 1e-4 * np.mean(np.diag(A)) * np.eye(len(A))
        Wb = np.linalg.solve(A, X1.T @ Y)
        res.append(float(((Y - X1 @ Wb) ** 2).sum() / max((Y ** 2).sum(), 1e-12)))
        if ct:
            if new_w is None:
                new_w, new_b = np.zeros((X.shape[1], Y.shape[1]) + tuple(w.shape[2:])), np.zeros(Y.shape[1])
            new_w[:, :, tap[0], tap[1]] = Wb[:-1]; new_b += Wb[-1] / (w.shape[2] * w.shape[3])
        else:
            new_w, new_b = Wb[:-1].T.reshape(Y.shape[1], -1, w.shape[2], w.shape[3]), Wb[-1]
    return new_w, new_b, max(res)


def prune_ls(src, keep, out, nchunks=4, per=6000):
    """Layer-wise reconstruction pruning (He et al. 2017 / ThiNet style): per channel group keep the channels with the
    largest data-driven contribution E[x_c^2] * ||W[:, c]||^2 summed over the Convs that read them (MUSDB train
    chunks), then refit every Conv / ConvTranspose that lost inputs by ridge least squares (with a bias column) to the
    parent's own pre-BatchNorm outputs on the parent's activations. BatchNorm stats stay valid (same targets)."""
    nchunks, per = int(nchunks), int(per)
    seq_only = os.environ.get("SEQ") == "1" and int(os.environ.get("REFINE", 0)) == 0
    m = onnx.load(src)
    g = m.graph
    G, size, level, fixed, init = _groups(m)
    convs = [n for n in g.node if n.op_type in ("Conv", "ConvTranspose")]
    xs = calib_inputs(max(8, nchunks))
    st = run_with(m, sorted({n.input[0] for n in convs if n.input[0] != g.input[0].name}), xs[:4])
    imp = collections.defaultdict(float)
    for n in convs:
        if n.input[0] == g.input[0].name:
            continue
        w = arr(init, n.input[1]).astype(np.float64)
        ct = n.op_type == "ConvTranspose"
        wn = (w ** 2).sum(axis=(1, 2, 3)) if ct else (w ** 2).sum(axis=(0, 2, 3))
        ex2 = st[n.input[0]][1] + st[n.input[0]][0] ** 2
        imp[G(n.input[0])] = imp[G(n.input[0])] + ex2 * wn
    counts = keep_counts(keep, G, size, level, fixed)
    kept = {k: np.sort(np.argsort(-imp[k])[:n]) for k, n in counts.items()}
    t0, t1 = mac_table(m), mac_table(m, counts)
    print(f"MACs {sum(t1.values()) / sum(t0.values()):.1%} of the parent", flush=True)
    print({"pruned_groups": len(kept), "channels": {k: f"{size[k]}->{len(v)}" for k, v in kept.items()}}, flush=True)
    allc = lambda k, c: kept.get(k, np.arange(c))
    inp = g.input[0].name
    outs_names = sorted({n.output[0] for n in convs})
    in_names = sorted({n.input[0] for n in convs} - {inp})
    orig = onnx.ModelProto(); orig.CopyFrom(m)
    # parent activations fetched 6 convs (in graph order) at a time from upstream-only subgraphs without an ORT arena
    cgroups = [convs[i:i + 6] for i in range(0, len(convs), 6)]
    cnames = [sorted({t for n in cg for t in (n.input[0], n.output[0])} - {inp}) for cg in cgroups]
    s_os = [_subgraph_session(orig, nm) for nm in cnames]
    # targets (parent's pre-BN outputs at kept output channels) and positions, fixed for all passes
    pos, T = collections.defaultdict(list), collections.defaultdict(lambda: collections.defaultdict(list))
    F = collections.defaultdict(list)
    rs = np.random.RandomState(0)
    for x in xs[:nchunks]:
      for cg, nm, s_o in zip(cgroups, cnames, s_os):
        o = dict(zip(nm, s_o.run(nm, {inp: x})))
        for n in cg:
            ct = n.op_type == "ConvTranspose"
            k, st_, pd = _attrs(n)
            xi = x[0] if n.input[0] == inp else o[n.input[0]][0]
            yo = o[n.output[0]][0]
            p_ = _pos(xi.shape, yo.shape, per, ct, rs)
            pos[n.name].append(p_)
            for tap, t in _targ(yo, p_, k, st_, ct, allc(G(n.output[0]), yo.shape[0])).items():
                T[n.name][tap].append(t.astype(np.float32))
            if not seq_only or n.input[0] == inp:
                F[n.name].append(_feat(xi, p_, k, st_, pd, ct, allc(G(n.input[0]), xi.shape[0])).astype(np.float32))
        del o
    del s_os
    if seq_only:      # every layer but the input conv is refit below on the pruned net's own inputs: slice for shape
        for n in convs:
            if n.input[0] == inp:
                nw, nb, res = _fit(n, arr(onnx_init(orig), n.input[1]), F[n.name], T[n.name])
            else:
                ct = n.op_type == "ConvTranspose"
                w = arr(init, n.input[1])
                ki, ko = allc(G(n.input[0]), w.shape[0 if ct else 1]), allc(G(n.output[0]), w.shape[1 if ct else 0])
                nw = w[ki][:, ko] if ct else w[ko][:, ki]
                nb = arr(init, n.input[2])[ko] if len(n.input) > 2 else None
            put(init, n.input[1], nw)
            if len(n.input) > 2:
                put(init, n.input[2], nb)
        for nb_ in g.node:
            if nb_.op_type == "BatchNormalization" and G(nb_.input[0]) in kept:
                for i in nb_.input[1:]:
                    put(init, i, arr(init, i)[kept[G(nb_.input[0])]])
        for v in list(g.value_info):
            g.value_info.remove(v)
    passes = int(os.environ.get("REFINE", 0))
    for it in ([] if seq_only else range(passes + 1)):
        if it:                               # features from the pruned net's own activations (sequential error)
            s_p = _session(m, in_names)
            F = collections.defaultdict(list)
            for ci, x in enumerate(xs[:nchunks]):
                o = dict(zip(in_names, s_p.run(in_names, {inp: x})))
                for n in convs:
                    ct = n.op_type == "ConvTranspose"
                    k, st_, pd = _attrs(n)
                    xi = x[0] if n.input[0] == inp else o[n.input[0]][0]
                    F[n.name].append(_feat(xi, pos[n.name][ci], k, st_, pd, ct, np.arange(xi.shape[0])).astype(np.float32))
                del o
            del s_p
        worst = 0
        for n in convs:
            w0 = arr(onnx_init(orig), n.input[1])
            nw, nb, res = _fit(n, w0, F[n.name], T[n.name])
            worst = max(worst, res)
            put(init, n.input[1], nw)
            if len(n.input) > 2:
                put(init, n.input[2], nb)
            if it == 0:
                ct = n.op_type == "ConvTranspose"
                print(f"{n.name}: in {w0.shape[0 if ct else 1]}->{nw.shape[0 if ct else 1]} out {w0.shape[1 if ct else 0]}->{nw.shape[1 if ct else 0]} rel. residual {res:.4f}", flush=True)
        if it == 0:
            for nb_ in g.node:
                if nb_.op_type == "BatchNormalization" and G(nb_.input[0]) in kept:
                    for i in nb_.input[1:]:
                        put(init, i, arr(init, i)[kept[G(nb_.input[0])]])
            for v in list(g.value_info):
                g.value_info.remove(v)
        print(f"pass {it}: worst layer residual {worst:.4f}", flush=True)
    if os.environ.get("SEQ") == "1":         # sequential: each layer refit on the pruned net's own inputs, in graph order
        for li, n in enumerate(convs):
            if n.input[0] == inp:
                continue
            ct = n.op_type == "ConvTranspose"
            k, st_, pd = _attrs(n)
            s_p = _subgraph_session(m, [n.input[0]], int(os.environ.get("THREADS", 4)))
            Fs = []
            for ci, x in enumerate(xs[:nchunks]):
                xi = s_p.run([n.input[0]], {inp: x})[0][0]
                Fs.append(_feat(xi, pos[n.name][ci], k, st_, pd, ct, np.arange(xi.shape[0])).astype(np.float32))
            del s_p
            nw, nb, res = _fit(n, arr(onnx_init(orig), n.input[1]), Fs, T[n.name])
            put(init, n.input[1], nw)
            if len(n.input) > 2:
                put(init, n.input[2], nb)
            print(f"seq {li} {n.name}: rel. residual {res:.4f}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    onnx.save(m, out)
    print(out, os.path.getsize(out) >> 20, "MB")


def shape(src, spec, out):
    """The pruned graph's shape with the parent's first channels kept: no data, no refit (speed screens only)."""
    m = onnx.load(src)
    G, size, level, fixed, init = _groups(m)
    counts = keep_counts(spec, G, size, level, fixed)
    allc = lambda k, c: np.arange(counts.get(k, c))
    for n in m.graph.node:
        if n.op_type in ("Conv", "ConvTranspose"):
            ct = n.op_type == "ConvTranspose"
            w = arr(init, n.input[1])
            ki, ko = allc(G(n.input[0]), w.shape[0 if ct else 1]), allc(G(n.output[0]), w.shape[1 if ct else 0])
            put(init, n.input[1], w[ki][:, ko] if ct else w[ko][:, ki])
            if len(n.input) > 2:
                put(init, n.input[2], arr(init, n.input[2])[ko])
        elif n.op_type == "BatchNormalization" and G(n.input[0]) in counts:
            for i in n.input[1:]:
                put(init, i, arr(init, i)[:counts[G(n.input[0])]])
    for v in list(m.graph.value_info):
        m.graph.value_info.remove(v)
    p0 = onnx.load(src)
    print(f"{out}: MACs {sum(mac_table(p0, counts).values()) / sum(mac_table(p0).values()):.1%} of the parent")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    onnx.save(m, out)


def onnx_init(model):
    return {i.name: i for i in model.graph.initializer}


if __name__ == "__main__":
    {"crop": crop, "prune": prune, "prune_ls": prune_ls, "macs": macs, "shape": shape}[sys.argv[1]](*sys.argv[2:])
