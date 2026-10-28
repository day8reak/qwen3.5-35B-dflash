"""The projection rewrite preserves operands and a single FP16 result boundary.

CPU Conv2D tests cover the algebra/layout, not selected-DDK compiler support or
device arithmetic. Arbitrary reduction implementations need numerical checks.
"""
from dataclasses import replace

import numpy as np
import pytest
import torch
import torch.nn.functional as functional

from qwen35_moe_dflash.ge_cpp import _Emitter
from qwen35_moe_dflash.ge_dense_projection import draft_fc_conv_plan, emit_draft_fc_conv2d
from qwen35_moe_dflash.ge_ir import Graph, Value


def _projection(rows=2, graph_name="draft"):
    graph = Graph(graph_name)
    left = graph.input("target_features", (1, rows, 16384))
    right = graph.weight("draft.fc.weight", (2048, 16384))
    output = graph.op("matmul", left, right, shape=(1, rows, 2048),
                      attrs={"transpose_b": True}, name="v20")
    graph.outputs = [output]
    return graph, left, right, output


def _emitter(graph, left, right):
    emitter = _Emitter(graph)
    for value, kind in ((left, "Data"), (right, "FileConstant")):
        var = emitter.op(value.name, kind, [], [("y", value)])
        emitter.refs[value.name] = (var, "y")
    return emitter


def test_complete_full_projection_becomes_one_nchw_conv_with_only_views():
    graph, left, right, output = _projection(64)
    emitter = _emitter(graph, left, right)
    plan = emit_draft_fc_conv2d(emitter, output, left, right, {"transpose_b": True})
    conv, = [row for row in emitter.records if row["type"] == "Conv2D"]
    arithmetic = [row["type"] for row in emitter.records if row["type"] not in {"Data", "FileConstant", "Const", "Reshape"}]
    assert arithmetic == ["Conv2D"]
    assert [row["port"] for row in conv["inputs"]] == ["x", "filter"]
    assert [row["shape"] for row in conv["inputs"]] == [[64, 1024, 4, 4], [2048, 1024, 4, 4]]
    assert conv["outputs"] == [{"port": "y", "shape": [64, 2048, 1, 1], "dtype": "float16", "format": "NCHW"}]
    assert all(port["dtype"] == "float16" and port["format"] == "NCHW" for port in conv["inputs"])
    assert conv["attrs"] == {
        "strides": {"type": "ints", "value": [1, 1, 1, 1]},
        "pads": {"type": "ints", "value": [0, 0, 0, 0]},
        "dilations": {"type": "ints", "value": [1, 1, 1, 1]},
        "groups": {"type": "int", "value": 1},
        "data_format": {"type": "string", "value": "NCHW"},
        "offset_x": {"type": "int", "value": 0},
    }
    result = emitter.records[-1]
    assert result["type"] == "Reshape" and result["name"] == "v20"
    assert result["outputs"] == [{"port": "y", "shape": [1, 64, 2048], "dtype": "float16"}]
    assert "ge::FORMAT_NCHW" in "\n".join(emitter.lines)
    assert plan["K"] == 16384 and plan["N"] == 2048
    assert plan["operand_storage"] == "UNCHANGED_FP16_BYTES"
    assert plan["target_validation"] == "NOT_RUN" and plan["bitwise_equivalence"] == "NOT_ESTABLISHED"
    assert graph.nodes[0].kind == "matmul"  # No source/reference graph mutation.


@pytest.mark.parametrize("change", [
    "tiny", "other_weight", "other_input", "fp32_left", "fp32_output", "fp32_weight",
    "other_k", "other_n", "wrong_output", "left_transpose", "no_right_transpose",
    "extra_attribute", "batch2", "rank2", "nchw_left",
])
def test_only_exact_full_initial_projection_is_eligible(change):
    _, left, right, output = _projection()
    attrs = {"transpose_b": True}
    if change == "tiny":
        left, right, output = replace(left, shape=(1, 8, 32)), replace(right, shape=(16, 32)), replace(output, shape=(1, 8, 16))
    elif change == "other_weight": right = replace(right, name="draft.layers.0.self_attn.q_proj.weight")
    elif change == "other_input": left = replace(left, name="unrelated_features")
    elif change == "fp32_left": left = replace(left, dtype="float32")
    elif change == "fp32_weight": right = replace(right, dtype="float32")
    elif change == "fp32_output": output = replace(output, dtype="float32")
    elif change == "other_k": left, right = replace(left, shape=(1, 2, 8192)), replace(right, shape=(2048, 8192))
    elif change == "other_n": right, output = replace(right, shape=(1024, 16384)), replace(output, shape=(1, 2, 1024))
    elif change == "wrong_output": output = replace(output, shape=(1, 3, 2048))
    elif change == "left_transpose": attrs["transpose_a"] = True
    elif change == "no_right_transpose": attrs["transpose_b"] = False
    elif change == "extra_attribute": attrs["unknown_semantics"] = 1
    elif change == "batch2": left, output = replace(left, shape=(2, 2, 16384)), replace(output, shape=(2, 2, 2048))
    elif change == "rank2": left, output = replace(left, shape=(2, 16384)), replace(output, shape=(2, 2048))
    elif change == "nchw_left": left = replace(left, format="NCHW")
    assert draft_fc_conv_plan(output, left, right, attrs) is None


def test_other_graph_or_unowned_weights_emit_nothing():
    for name in ("target", "another_graph"):
        graph, left, right, output = _projection(graph_name=name)
        emitter = _emitter(graph, left, right)
        before = len(emitter.records)
        assert emit_draft_fc_conv2d(emitter, output, left, right, {"transpose_b": True}) is None
        assert len(emitter.records) == before
    graph, left, right, output = _projection()
    emitter = _emitter(graph, left, right)
    assert emit_draft_fc_conv2d(emitter, output, left, replace(right), {"transpose_b": True}) is None


def test_reshape_preserves_weight_channel_order_and_all_fp16_payload_bits():
    _, left, right, output = _projection()
    plan = draft_fc_conv_plan(output, left, right, {"transpose_b": True})
    # A row contains every interesting FP16 bit pattern, including signed zero,
    # infinities, subnormals and NaN payloads. No arithmetic touches these views.
    markers = np.array([0, 0x8000, 1, 0x8001, 0x7c00, 0xfc00, 0x7e31, 0x3c00], dtype=np.uint16)
    raw = np.resize(markers, 2 * 16384).reshape(1, 2, 16384)
    view = raw.view(np.float16).reshape(plan["input_view"])
    assert np.shares_memory(raw, view)
    np.testing.assert_array_equal(view.view(np.uint16).reshape(raw.shape), raw)
    # Check independent channel/height/width indexing across the complete K.
    row = np.arange(16384, dtype=np.uint16)
    weights = np.broadcast_to(row, right.shape).view(np.float16)
    filters = weights.reshape(plan["filter_view"])
    for channel, height, width in ((0, 0, 0), (0, 1, 0), (1, 0, 0), (1023, 3, 3)):
        assert filters.view(np.uint16)[2047, channel, height, width] == channel * 16 + height * 4 + width
    assert np.shares_memory(weights, filters)


@pytest.mark.parametrize("dyadic", [True, False], ids=["exact_dyadic", "random_numeric"])
def test_full_k_and_n_cpu_convolution_matches_independent_dot_product(dyadic):
    _, left, right, output = _projection()
    plan = draft_fc_conv_plan(output, left, right, {"transpose_b": True})
    rng = np.random.default_rng(174)
    if dyadic:
        a = (rng.integers(-2, 3, left.shape).astype(np.float16) / np.float16(32))
        w = (rng.integers(-2, 3, right.shape, dtype=np.int8).astype(np.float16) / np.float16(32))
    else:
        a = rng.normal(0, .0625, left.shape).astype(np.float16)
        w = rng.normal(0, .03125, right.shape).astype(np.float16)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.inference_mode():
            # Explicit FP32 CPU accumulation followed by a single FP16 output
            # is a host algebra check, not a claim about the device kernel.
            ta, tw = torch.from_numpy(a).float(), torch.from_numpy(w).float()
            actual = functional.conv2d(ta.reshape(plan["input_view"]), tw.reshape(plan["filter_view"]),
                                       stride=1, padding=0, dilation=1, groups=1).reshape(output.shape).half()
            expected = (ta.reshape(2, 16384) @ tw.T).reshape(output.shape).half()
            if dyadic:
                # All products and sums in this bounded dyadic case are exactly
                # representable in FP32, independent of reduction grouping.
                np.testing.assert_array_equal(actual.numpy().view(np.uint16), expected.numpy().view(np.uint16))
            else:
                np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=0.002, atol=0.000125)
                # A separate float64 dot checks arbitrary output channels.
                rows = np.array([0, 1, 17, 1023, 2047])
                oracle = (a.reshape(2, 16384).astype(np.float64) @ w[rows].astype(np.float64).T).astype(np.float16)
                np.testing.assert_allclose(actual.numpy().reshape(2, 2048)[:, rows], oracle, rtol=0.002, atol=0.000125)
            assert torch.isfinite(actual).all() and actual.shape == output.shape
            # Also compare the actual FP16 PyTorch primitive boundaries used
            # by the preserved standalone reference, independently of the
            # explicit FP32 algebra check above.
            half_a, half_w = torch.from_numpy(a), torch.from_numpy(w)
            half_conv = functional.conv2d(half_a.reshape(plan["input_view"]),
                                          half_w.reshape(plan["filter_view"])).reshape(output.shape)
            half_linear = functional.linear(half_a, half_w)
            if dyadic:
                np.testing.assert_array_equal(half_conv.numpy().view(np.uint16), half_linear.numpy().view(np.uint16))
            else:
                np.testing.assert_allclose(half_conv.numpy(), half_linear.numpy(), rtol=0.002, atol=0.000125)
    finally:
        torch.set_num_threads(previous)
