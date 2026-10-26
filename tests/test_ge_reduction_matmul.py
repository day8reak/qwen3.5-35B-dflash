"""Host checks for isolated candidates; no device or bitwise parity claim."""
import numpy as np
import pytest

from qwen35_moe_dflash.ge_eval import evaluate
from qwen35_moe_dflash.ge_ir import Graph
from qwen35_moe_dflash.ge_reduction_matmul import reduce_via_matmul


def _graph(shape, kind="reduce_sum", axis=-1, keepdims=True):
    graph = Graph("candidate")
    source = graph.input("x", shape)
    graph.outputs = [reduce_via_matmul(graph, source, kind=kind, axis=axis, keepdims=keepdims)]
    return graph


def _double_reference(data, kind, axis, keepdims):
    operation = np.mean if kind == "reduce_mean" else np.sum
    with np.errstate(over="ignore", invalid="ignore"):
        return operation(data.astype(np.float64), axis=axis, keepdims=keepdims).astype(np.float16)


@pytest.mark.parametrize("shape,axis", [((8,), 0), ((2, 3), 0), ((2, 3), -1),
                                       ((2, 3, 4), 1), ((1, 2, 3, 4), 0),
                                       ((1, 2, 3, 4), 2), ((2, 1, 7, 7), 2)])
@pytest.mark.parametrize("keepdims", [False, True])
def test_single_axis_layout_matches_independent_double_reference(shape, axis, keepdims):
    graph = _graph(shape, axis=axis, keepdims=keepdims)
    data = (((np.arange(np.prod(shape)) * 7) % 37 - 18) / 16).astype(np.float16).reshape(shape)
    actual, = evaluate(graph, {"x": data})
    expected = _double_reference(data, "reduce_sum", axis, keepdims)
    np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))
    assert actual.dtype == np.float16 and actual.shape == expected.shape
    assert sum(node.kind == "matmul" for node in graph.nodes) == 1
    assert not {"reduce_sum", "reduce_mean", "cast", "mul", "scalar_mul"}.intersection(n.kind for n in graph.nodes)


@pytest.mark.parametrize("shape,axis", [((1, 8, 16), -1), ((1, 8, 2, 8), -1),
                                       ((2, 8, 4), 1), ((1, 8, 2048), -1)])
@pytest.mark.parametrize("keepdims", [False, True])
def test_mean_uses_exact_reciprocal_and_one_fp16_arithmetic_boundary(shape, axis, keepdims):
    graph = _graph(shape, "reduce_mean", axis, keepdims)
    data = (((np.arange(np.prod(shape)) * 11) % 61 - 30) / 32).astype(np.float16).reshape(shape)
    actual, = evaluate(graph, {"x": data})
    np.testing.assert_array_equal(actual, _double_reference(data, "reduce_mean", axis, keepdims))
    matmul, = [node for node in graph.nodes if node.kind == "matmul"]
    weights = graph.constants[matmul.inputs[1].name]
    assert weights.shape == (1, shape[axis]) and weights.dtype == np.float16
    assert np.all(weights.astype(np.float64) == 1 / shape[axis])
    assert matmul.attrs == {"transpose_a": False, "transpose_b": True}


def test_mean_does_not_materialize_overflowed_fp16_sum():
    data = np.full((1, 16), 4096, np.float16)
    actual, = evaluate(_graph(data.shape, "reduce_mean"), {"x": data})
    np.testing.assert_array_equal(actual, [[4096]])
    np.testing.assert_array_equal(actual, _double_reference(data, "reduce_mean", -1, True))
    with np.errstate(over="ignore"):
        wrong = np.sum(data.astype(np.float32), axis=-1, keepdims=True).astype(np.float16)
    assert np.isinf(wrong).all()  # Sum-then-scale is not an equivalent lowering.


@pytest.mark.parametrize("kind", ["reduce_sum", "reduce_mean"])
def test_subnormal_sparse_and_cancellation_inputs_keep_existing_tolerance(kind):
    tiny = np.nextafter(np.float16(0), np.float16(1), dtype=np.float16)
    data = np.zeros((4, 64), dtype=np.float16)
    data[0, :3] = tiny
    data[1, ::2] = np.float16(1)
    data[1, 1::2] = np.float16(-1)
    data[2, 0], data[2, -1] = np.float16(65504), np.float16(-65504)
    data[3, 31] = np.float16(-4)
    actual, = evaluate(_graph(data.shape, kind), {"x": data})
    expected = _double_reference(data, kind, -1, True)
    np.testing.assert_allclose(actual, expected, rtol=0.01, atol=0.002)
    assert np.isfinite(actual).all()
    # Tiny values are checked explicitly, rather than hidden by absolute tol.
    np.testing.assert_array_equal(actual[0].view(np.uint16), expected[0].view(np.uint16))


@pytest.mark.parametrize("kind", ["reduce_sum", "reduce_mean"])
def test_nonfinite_values_are_not_multiplied_by_zero_masks(kind):
    data = np.array([[np.inf, 1, 2, 3], [-np.inf, 0, 1, 2],
                     [np.inf, -np.inf, 1, 2], [np.nan, 1, 2, 3]], np.float16)
    actual, = evaluate(_graph(data.shape, kind), {"x": data})
    expected = _double_reference(data, kind, -1, True)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))


def test_existing_fp16_product_boundary_is_retained_before_correction_sum():
    graph = Graph("correction")
    left = graph.input("left", (2, 1, 7, 1))
    right = graph.input("right", (2, 1, 7, 7))
    product = graph.op("mul", left, right, shape=right.shape)
    graph.outputs = [reduce_via_matmul(graph, product, kind="reduce_sum", axis=2, keepdims=False)]
    a = np.linspace(-0.9, 1.1, 14, dtype=np.float16).reshape(left.shape)
    b = np.linspace(-1.1, 0.9, 98, dtype=np.float16).reshape(right.shape)
    rounded = (a.astype(np.float32) * b.astype(np.float32)).astype(np.float16)
    actual, = evaluate(graph, {"left": a, "right": b})
    np.testing.assert_allclose(actual, _double_reference(rounded, "reduce_sum", 2, False), rtol=0.01, atol=0.002)
    assert graph.nodes[0].kind == "mul" and graph.nodes[0].output is product
    assert graph.nodes[1].kind == "transpose" and graph.nodes[1].inputs == (product,)


def test_k1_still_emits_a_real_matmul_with_ones():
    graph = _graph((2, 1, 1, 1), axis=2, keepdims=False)
    matmul, = [node for node in graph.nodes if node.kind == "matmul"]
    assert matmul.inputs[0].shape == (2, 1)
    np.testing.assert_array_equal(graph.constants[matmul.inputs[1].name], np.ones((1, 1), np.float16))
    data = np.array([-1, 2], np.float16).reshape(2, 1, 1, 1)
    actual, = evaluate(graph, {"x": data})
    np.testing.assert_array_equal(actual, data.reshape(2, 1, 1))


@pytest.mark.parametrize("dtype", ["int32", "int64", "bool"])
def test_integer_or_boolean_reduction_never_casts_to_fp16(dtype):
    graph = Graph("invalid")
    source = graph.input("x", (2, 8), dtype)
    with pytest.raises(ValueError, match="FP16"):
        reduce_via_matmul(graph, source, kind="reduce_sum", axis=1, keepdims=True)
    assert not graph.nodes and not graph.constants


@pytest.mark.parametrize("shape,kind,axis,keepdims", [
    ((2, 3), "reduce_mean", 1, True), ((2, 5), "reduce_mean", 1, True),
    ((1, 1 << 25), "reduce_mean", 1, True),
    ((2, 8), "reduce_max", 1, True), ((2, 8), "reduce_sum", True, True),
    ((2, 8), "reduce_sum", 0.0, True), ((2, 8), "reduce_sum", [1], True),
    ((2, 8), "reduce_sum", 2, True), ((2, 8), "reduce_sum", -3, True),
    ((2, 8), "reduce_sum", 1, 1), ((2, 0), "reduce_sum", 1, True),
    ((), "reduce_sum", 0, True),
])
def test_invalid_contract_is_rejected_before_graph_changes(shape, kind, axis, keepdims):
    graph = Graph("invalid")
    source = graph.input("x", shape)
    with pytest.raises(ValueError):
        reduce_via_matmul(graph, source, kind=kind, axis=axis, keepdims=keepdims)
    assert not graph.nodes and not graph.constants


def test_source_from_another_graph_is_rejected():
    graph = Graph("receiver")
    source = Graph("other").input("x", (2, 8))
    with pytest.raises(ValueError, match="belong"):
        reduce_via_matmul(graph, source, kind="reduce_sum", axis=1, keepdims=True)
    assert not graph.nodes and not graph.constants
