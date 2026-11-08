"""vf/tile.py on a small graph shaped like an MDX-Net: same output in pieces, and it refuses what reads across time."""
import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from vf import tile

T, F, C = 16, 24, 6


def mdx_like(across_time=False):
    r = np.random.default_rng(1)
    w = lambda n, *s: numpy_helper.from_array((r.standard_normal(s) * 0.3).astype(np.float32), n)
    inits = [w("w_in", C, 4, 1, 1), w("w_a", C, C, 3, 3), w("w_b", C, C, 3, 3), w("tdf", F, F), w("w_dn", C, C, 2, 2), w("w_mid", C, C, 3, 3),
             w("w_up", C, C, 2, 2), w("w_c", C, C, 3, 3), w("w_out", 4, C, 1, 1)]
    p1 = dict(pads=[1, 1, 1, 1], kernel_shape=[3, 3])
    nodes = [
        helper.make_node("Conv", ["x", "w_in"], ["a0"], kernel_shape=[1, 1]),
        helper.make_node("Transpose", ["a0"], ["a1"], perm=[0, 1, 3, 2]),                  # [1, C, T, F]
        helper.make_node("Conv", ["a1", "w_a"], ["a2"], **p1),
        helper.make_node("Relu", ["a2"], ["a3"]),
        helper.make_node("ReduceMean", ["a3"], ["a4"], axes=[2], keepdims=1) if across_time else
        helper.make_node("MatMul", ["a3", "tdf"], ["a4"]),                                 # across frequency, inside a frame
        helper.make_node("Add", ["a3", "a4"], ["skip"]),
        helper.make_node("Conv", ["skip", "w_dn"], ["d0"], kernel_shape=[2, 2], strides=[2, 2]),
        helper.make_node("Conv", ["d0", "w_mid"], ["d1"], **p1),
        helper.make_node("ConvTranspose", ["d1", "w_up"], ["u0"], kernel_shape=[2, 2], strides=[2, 2]),
        helper.make_node("Mul", ["u0", "skip"], ["u1"]),
        helper.make_node("Conv", ["u1", "w_b"], ["u2"], **p1),
        helper.make_node("Conv", ["u2", "w_c"], ["u3"], **p1),
        helper.make_node("Transpose", ["u3"], ["u4"], perm=[0, 1, 3, 2]),
        helper.make_node("Conv", ["u4", "w_out"], ["y"], kernel_shape=[1, 1]),
    ]
    g = helper.make_graph(nodes, "mdx", [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch_size", 4, F, T])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch_size", 4, F, T])], inits)
    return helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)


def run(m, x):
    o = ort.SessionOptions()
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(m.SerializeToString(), o, providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]


def test_pieces_match_the_whole():
    x = np.random.default_rng(2).standard_normal((1, 4, F, T)).astype(np.float32)
    whole = run(mdx_like(), x)
    t = tile.tile(mdx_like(), halo=4)
    onnx.checker.check_model(t)
    assert sum(n.op_type == "Slice" for n in t.graph.node) > 0
    np.testing.assert_allclose(run(t, x), whole, rtol=1e-5, atol=1e-5)


def test_refuses_what_reads_across_time():
    with pytest.raises(SystemExit, match="reads across time"):
        tile.tile(mdx_like(across_time=True), halo=4)
