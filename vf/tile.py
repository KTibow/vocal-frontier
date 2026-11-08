"""A GPU graph cut into time pieces, for GPUs that allow less in one buffer than the graph needs. `vf build` calls this.

WebGPU only promises 128 MiB in one storage binding, and Adreno gives exactly that. An MDX-Net with dim_t 512 in fp16, or dim_t 256
computing in fp32, holds [1, 48, T, 3072] tensors of 144 MiB at full resolution, and ONNX Runtime Web fails there with "Binding size
... is larger than the maximum storage buffer binding size". Those layers only look a few frames apart in time (3x3 convolutions, and
frequency layers inside one frame), so the copy runs them on overlapping time pieces and joins the pieces where the graph halves its
resolution, and at the output. Same weights, same arithmetic per value, so the output matches to rounding, and the page uses the
copy only on a GPU whose limit the original doesn't fit. It refuses any graph with an operation it doesn't know to be local in time.
"""
import os
import tempfile

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference

LOCAL_UNARY = {"Relu", "Cast", "BatchNormalization", "QuantizeLinear", "DequantizeLinear", "Sigmoid", "Tanh", "Identity", "Erf", "Sqrt"}
LOCAL_BINARY = {"Add", "Mul", "Sub", "Div", "Max", "Min", "Pow"}


def shapes(model):
    m = onnx.ModelProto()
    m.CopyFrom(model)
    for i in m.graph.input:
        for d in i.type.tensor_type.shape.dim:
            if not d.dim_value:
                d.dim_value = 1
    g = shape_inference.infer_shapes(m, data_prop=True).graph
    out = {v.name: [x.dim_value for x in v.type.tensor_type.shape.dim] for v in list(g.value_info) + list(g.output) + list(g.input)}
    for t in g.initializer:
        out[t.name] = list(t.dims)
    return out


def tile(model, pieces=2, halo=8):
    g = model.graph
    sh = shapes(model)
    inits = {t.name for t in g.initializer}
    x = g.input[0]
    T = sh[x.name][3]
    F = sh[x.name][2]
    assert T % (2 * pieces) == 0 and F != T, (T, F)
    bounds = [(T * k // pieces, T * (k + 1) // pieces) for k in range(pieces)]
    new, extra = [], []
    uid = [0]

    def name(base):
        uid[0] += 1
        return f"{base}__t{uid[0]}"

    def const(vals):
        n = name("c")
        extra.append(numpy_helper.from_array(np.array(vals, dtype=np.int64), n))
        return n

    def slice_(src, axis, a, b):
        o = name(src)
        new.append(helper.make_node("Slice", [src, const([a]), const([b]), const([axis])], [o]))
        return o

    split = {}          # tensor -> list of (name, start, end, time axis)

    def crop(p, a, b):
        n, s, e, ax = p
        assert s <= a and b <= e, (p, a, b)
        return p if (s, e) == (a, b) else (slice_(n, ax, a - s, b - s), a, b, ax)

    def join(pcs, out):
        ax = pcs[0][3]
        parts = [crop(p, lo, hi)[0] for p, (lo, hi) in zip(pcs, bounds)]
        new.append(helper.make_node("Concat", parts, [out], axis=ax))

    def time_axis(t):
        s = sh.get(t, [])
        return 3 if len(s) == 4 and s[3] == T else 2 if len(s) == 4 and s[2] == T else None

    for n in g.node:
        ins = [i for i in n.input if i]
        ax0 = time_axis(n.input[0]) if n.input else None
        if n.op_type == "Conv" and n.input[0] not in split and ax0 is not None:
            # the first full-resolution convolution: cut its input into overlapping pieces. Anything before it (a loudness
            # normalization over the whole clip, say) stays whole; it's on the 4-channel input, which is small
            split[n.input[0]] = [(slice_(n.input[0], ax0, max(0, lo - halo), min(T, hi + halo)), max(0, lo - halo), min(T, hi + halo), ax0) for lo, hi in bounds]
        sp = [i for i in ins if i in split]
        if not sp:
            ax = time_axis(n.output[0])
            if n.op_type == "ConvTranspose" and ax is not None:      # back up to full resolution: start pieces again
                src = n.input[0]
                sax = ax
                outs = []
                for lo, hi in bounds:
                    a, b = max(0, lo - halo), min(T, hi + halo)
                    a, b = a - a % 2, b + b % 2
                    s_in = slice_(src, sax, a // 2, b // 2)
                    o = name(n.output[0])
                    nn = onnx.NodeProto()
                    nn.CopyFrom(n)
                    nn.input[0] = s_in
                    nn.output[0] = o
                    new.append(nn)
                    outs.append((o, a, b, ax))
                split[n.output[0]] = outs
                continue
            new.append(n)
            continue
        op = n.op_type
        attrs = {a.name: helper.get_attribute_value(a) for a in n.attribute}
        if op == "Conv":
            pcs = split[n.input[0]]
            ax = pcs[0][3]
            k = attrs.get("kernel_shape")
            strides = attrs.get("strides", [1, 1])
            d = ax - 2                                  # 0 = H, 1 = W
            if strides[d] != 1:                         # halves the resolution: join, then go on whole
                assert strides[d] == 2 and k[d] == 2 and not any(attrs.get("pads", [0, 0, 0, 0])), attrs
                parts = []
                for p, (lo, hi) in zip(pcs, bounds):       # each piece's exact half, so pairs of frames don't straddle a cut
                    nn = onnx.NodeProto()
                    nn.CopyFrom(n)
                    nn.input[0] = crop(p, lo, hi)[0]
                    nn.output[0] = name(n.output[0])
                    new.append(nn)
                    parts.append(nn.output[0])
                new.append(helper.make_node("Concat", parts, [n.output[0]], axis=ax))
                continue
            pads = list(attrs.get("pads", [0, 0, 0, 0]))
            lead, trail = pads[d], pads[d + 2]
            assert lead + trail == k[d] - 1 and attrs.get("dilations", [1, 1])[d] == 1, attrs
            outs = []
            for (pn, s, e, _) in pcs:
                p2 = list(pads)
                p2[d] = lead if s == 0 else 0
                p2[d + 2] = trail if e == T else 0
                o = name(n.output[0])
                nn = onnx.NodeProto()
                nn.CopyFrom(n)
                nn.input[0] = pn
                nn.output[0] = o
                for a in nn.attribute:
                    if a.name == "pads":
                        a.ClearField("ints")
                        a.ints.extend(p2)
                if not any(a.name == "pads" for a in nn.attribute):
                    nn.attribute.append(helper.make_attribute("pads", p2))
                new.append(nn)
                outs.append((o, s + (lead if s else 0), e - (trail if e < T else 0), ax))
            split[n.output[0]] = outs
            continue
        if op == "Transpose":
            perm = attrs["perm"]
            assert len(sp) == 1
            pcs = split[n.input[0]]
            outs = []
            for (pn, s, e, ax) in pcs:
                o = name(n.output[0])
                nn = onnx.NodeProto()
                nn.CopyFrom(n)
                nn.input[0] = pn
                nn.output[0] = o
                new.append(nn)
                outs.append((o, s, e, perm.index(ax)))
            split[n.output[0]] = outs
            continue
        if op == "MatMul":
            assert n.input[0] in split and n.input[1] not in split
            assert split[n.input[0]][0][3] != 3, "MatMul over the time axis"
        elif op in LOCAL_UNARY or op in LOCAL_BINARY or op == "MatMul":
            pass
        else:
            raise SystemExit(f"{n.name or n.output[0]}: {op} reads across time; not supported")
        if op in LOCAL_UNARY or op == "MatMul":
            for i in ins[1:]:
                assert i not in split, (op, i)
        for i in ins:          # a whole operand must not vary along time
            if i not in split:
                s = sh.get(i)
                assert s is not None and (len(s) < 4 or s[2] in (1, F) and s[3] in (1, F) and T not in s[2:]) or i in inits, (op, i, s)
        pcs = [split[i] for i in sp]
        outs = []
        for k in range(pieces):
            lo = max(p[k][1] for p in pcs)
            hi = min(p[k][2] for p in pcs)
            ax = pcs[0][k][3]
            assert all(p[k][3] == ax for p in pcs)
            nn = onnx.NodeProto()
            nn.CopyFrom(n)
            for j, i in enumerate(n.input):
                if i in split:
                    nn.input[j] = crop(split[i][k], lo, hi)[0]
            o = name(n.output[0])
            del nn.output[:]
            nn.output.append(o)
            assert len(n.output) == 1, n
            new.append(nn)
            outs.append((o, lo, hi, ax))
        split[n.output[0]] = outs

    for out in g.output:
        if out.name in split:
            pcs = split.pop(out.name)
            join(pcs, out.name)
    for k, nd in enumerate(new):
        nd.name = f"n{k}"
    del g.node[:]
    g.node.extend(new)
    g.initializer.extend(extra)
    return model


def biggest(model):
    sh = shapes(model)
    size = {1: 4, 10: 2, 11: 8, 7: 8, 6: 4, 2: 1, 3: 1, 9: 1}
    m = onnx.ModelProto()
    m.CopyFrom(model)
    for i in m.graph.input:
        for d in i.type.tensor_type.shape.dim:
            if not d.dim_value:
                d.dim_value = 1
    g = shape_inference.infer_shapes(m, data_prop=True).graph
    best = (0, "")
    for v in list(g.value_info) + list(g.output):
        d = [x.dim_value for x in v.type.tensor_type.shape.dim]
        if d and all(d):
            best = max(best, (int(np.prod(d)) * size.get(v.type.tensor_type.elem_type, 4), v.name))
    return best


LIMIT = 128 * 2**20            # what WebGPU promises every device
HEADROOM = 1.5                 # ONNX Runtime asked for a 160 MiB binding around a 144 MiB tensor


def for_gpu(path):
    """(tiled model, largest tensor in the original, largest in the copy), or None when the original fits or the graph can't be cut."""
    m = onnx.load(path)
    before = biggest(m)[0]
    if before * HEADROOM <= LIMIT:
        return None
    try:
        t = tile(m)
        onnx.checker.check_model(t)
    except (AssertionError, SystemExit, KeyError, IndexError, ValueError) as e:
        print(f"{path}: largest tensor {before / 2**20:.0f} MiB, near WebGPU's 128 MiB binding limit, and it can't be cut into time pieces ({str(e)[:80]}). It goes as it is")
        return None
    after = biggest(t)[0]
    if after * HEADROOM > LIMIT or not same_output(path, t):
        print(f"{path}: the time-piece copy didn't fit or didn't match, so it isn't used")
        return None
    return t, before, after


def same_output(path, tiled):
    """Both graphs on the CPU without fusions, on the same random input: the copy has to match to fp16 rounding."""
    import onnxruntime as ort
    src = onnx.load(path)
    inp = src.graph.input[0]
    dims = [d if d else 1 for d in shapes(src)[inp.name]]
    del src
    x = np.random.default_rng(0).standard_normal(dims).astype(np.float32)
    o = ort.SessionOptions()
    o.intra_op_num_threads = 4
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    r0 = ort.InferenceSession(path, o, providers=["CPUExecutionProvider"]).run(None, {inp.name: x})[0]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.onnx")
        onnx.save(tiled, p)
        r1 = ort.InferenceSession(p, o, providers=["CPUExecutionProvider"]).run(None, {inp.name: x})[0]
    return float(np.abs(r0.astype(np.float64) - r1).max()) <= 1e-3 * max(1.0, float(np.abs(r0).max()))
