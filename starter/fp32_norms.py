"""Moves a fp16 RoFormer graph's transformer RMSNorms back to fp32 compute. Weights aren't touched, so only the graph is rewritten.

    python starter/fp32_norms.py <in core.onnx> <out core.onnx>

In fp16, x^2 is zero for |x| under about 2e-4 and the 1e-24 epsilon is zero too, so quiet passages lose their norm. Nothing
goes NaN and it doesn't crash. The output is just quieter and worse, by nearly 8 dB on one of our test clips, so check any
fp16 file against its fp32 parent with `vf transfer`.
"""
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

F16, F32 = TensorProto.FLOAT16, TensorProto.FLOAT


def main(src, dst):
    m = onnx.load(src, load_external_data=False)
    g = m.graph
    prod = {o: n for n in g.node for o in n.output}
    cons = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)

    def uncast(x):           # the tensor behind a Cast to fp16, or None if x isn't one
        n = prod.get(x)
        return n.input[0] if n is not None and n.op_type == "Cast" and n.attribute[0].i == F16 else None

    def constant32(x):       # the fp32 value behind a constant, as Pow's exponent or the epsilon
        n = prod[uncast(x) or x]
        return numpy_helper.to_array(n.attribute[0].t).astype(np.float32)

    out, after, done = [], {}, 0
    for n in g.node:
        out.append(n)
        if n.op_type != "Pow" or not cons.get(n.output[0]) or cons[n.output[0]][0].op_type != "ReduceMean":
            continue
        mean = cons[n.output[0]][0]
        add = cons[mean.output[0]][0]
        sqrt = cons[add.output[0]][0]
        div = cons[sqrt.output[0]][0]
        assert (add.op_type, sqrt.op_type, div.op_type) == ("Add", "Sqrt", "Div") and div.input[0] == n.input[0], n.name
        tag = f"norm32_{done}"
        x32, exp32, eps32, back = f"{tag}_x", f"{tag}_exp", f"{tag}_eps", f"{tag}_div"
        eps_i = 1 if add.input[0] == mean.output[0] else 0
        g.initializer.extend([numpy_helper.from_array(constant32(n.input[1]), exp32), numpy_helper.from_array(constant32(add.input[eps_i]), eps32)])
        out.insert(len(out) - 1, helper.make_node("Cast", [n.input[0]], [x32], to=F32, name=f"{tag}_cast"))
        n.input[0], n.input[1] = x32, exp32
        add.input[eps_i] = eps32
        div.input[0] = x32
        result = div.output[0]
        div.output[0] = back
        after[div.name] = helper.make_node("Cast", [back], [result], to=F16, name=f"{tag}_back")
        done += 1
    nodes = []
    for n in out:
        nodes.append(n)
        if n.name in after:
            nodes.append(after[n.name])
    del g.node[:]
    g.node.extend(nodes)
    del g.value_info[:]
    onnx.save(m, dst)
    print(f"{done} RMSNorms now compute in fp32: {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
