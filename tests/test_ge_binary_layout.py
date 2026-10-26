"""Probe-only layout contracts: host bit patterns, then emitted recipe math."""
from collections import Counter
import math
from pathlib import Path
import pickle

import numpy as np
import pytest

from qwen35_moe_dflash.ge_binary_layout import outer_sub_compact, row_scale_mul
from qwen35_moe_dflash.ge_cpp import emit_cpp
from qwen35_moe_dflash.ge_eval import evaluate
from qwen35_moe_dflash.ge_ir import DTYPES, Graph, Value
from test_ge_select_lowering import RecipeReader


def _graph(helper, left_shape, right_shape):
    graph = Graph("binary_layout")
    left = graph.input("left", left_shape)
    right = graph.input("right", right_shape)
    graph.outputs = [helper(graph, left, right)]
    return graph


def _reference(left, right, kind):
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        operation = np.multiply if kind == "mul" else np.subtract
        return operation(left.astype(np.float32), right.astype(np.float32)).astype(np.float16)


def _assert_bits_and_nonfinite(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype == np.float16
    for classification in (np.isnan, np.isposinf, np.isneginf):
        np.testing.assert_array_equal(classification(actual), classification(expected))
    finite = np.isfinite(expected)
    np.testing.assert_array_equal(actual[finite].view(np.uint16), expected[finite].view(np.uint16))


def _values(shape, offset):
    # Each row differs: this catches accidentally swapping token/head order.
    return (((np.arange(math.prod(shape)).reshape(shape) * 7 + offset) % 127 - 63) / 32).astype(np.float16)


@pytest.mark.parametrize("shape", [(3, 7), (2, 3, 5), (1, 8, 2, 8), (1, 2, 64, 8),
                                    (2, 1, 64, 8), (2, 3, 1, 5, 7), (1, 1)])
def test_row_scale_keeps_row_order_and_fp16_bits(shape):
    rhs_shape = (*shape[:-1], 1)
    graph = _graph(row_scale_mul, shape, rhs_shape)
    left, right = _values(shape, 5), _values(rhs_shape, 23)
    actual, = evaluate(graph, {"left": left, "right": right})
    _assert_bits_and_nonfinite(actual, _reference(left, right, "mul"))
    assert Counter(n.kind for n in graph.nodes) == {"reshape": 3, "mul": 1}
    operation, = [n for n in graph.nodes if n.kind == "mul"]
    assert [v.shape for v in operation.inputs] == [(math.prod(shape[:-1]), shape[-1]), (math.prod(shape[:-1]), 1)]
    assert graph.outputs[0].shape == shape and graph.outputs[0].name == "result"


@pytest.mark.parametrize("batch,rows,width", [(2, 64, 64), (2, 7, 5), (1, 1, 8), (3, 5, 1), (1, 1, 1)])
def test_outer_sub_keeps_operand_order_and_fp16_bits(batch, rows, width):
    a_shape, b_shape = (batch, 1, rows, 1), (batch, 1, 1, width)
    graph = _graph(outer_sub_compact, a_shape, b_shape)
    left, right = _values(a_shape, 5), _values(b_shape, 23)
    actual, = evaluate(graph, {"left": left, "right": right})
    _assert_bits_and_nonfinite(actual, _reference(left, right, "sub"))
    assert Counter(n.kind for n in graph.nodes) == {"reshape": 3, "sub": 1}
    operation, = [n for n in graph.nodes if n.kind == "sub"]
    assert [v.shape for v in operation.inputs] == [(batch, rows, 1), (batch, 1, width)]
    assert graph.outputs[0].shape == (batch, 1, rows, width)


@pytest.mark.parametrize("helper,kind", [(row_scale_mul, "mul"), (outer_sub_compact, "sub")])
def test_signed_zero_subnormal_overflow_and_nonfinite_positions(helper, kind):
    # Cartesian products include +/-0, both subnormal extremes, finite overflow,
    # +/-Inf, qNaN and sNaN. NaN payloads are not a promised device contract.
    bits = np.array([0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x83ff, 0x0400, 0x8400,
                     0x3c00, 0xbc00, 0x4000, 0xc000, 0x7bff, 0xfbff, 0x7c00, 0xfc00,
                     0x7e01, 0x7d01, 0xfe01, 0xfd01], dtype=np.uint16)
    values = bits.view(np.float16)
    if kind == "mul":
        left = np.broadcast_to(values, (1, len(values), len(values))).copy()
        right = values.reshape(1, len(values), 1).copy()
    else:
        left = values.reshape(1, 1, len(values), 1).copy()
        right = values.reshape(1, 1, 1, len(values)).copy()
    left_bits, right_bits = left.view(np.uint16).copy(), right.view(np.uint16).copy()
    graph = _graph(helper, left.shape, right.shape)
    actual, = evaluate(graph, {"left": left, "right": right})
    expected = _reference(left, right, kind)
    _assert_bits_and_nonfinite(actual, expected)
    assert np.signbit(actual[actual == 0]).any() and (~np.signbit(actual[actual == 0])).any()
    assert np.any((np.abs(actual) < np.finfo(np.float16).tiny) & (actual != 0))
    np.testing.assert_array_equal(left.view(np.uint16), left_bits)
    np.testing.assert_array_equal(right.view(np.uint16), right_bits)


def _execute_recipe(records, outputs, feeds):
    values = {}
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        for index, record in enumerate(records):
            output, = record["outputs"]
            args = [values[(port["src"], port["src_port"])] for port in record["inputs"]]
            if record["type"] == "Data":
                value = feeds[record["name"]]
            elif record["type"] == "Const":
                path, offset, size = record["constant"]
                with Path(path).open("rb") as stream:
                    stream.seek(offset)
                    raw = stream.read(size)
                assert len(raw) == size
                value = np.frombuffer(raw, DTYPES[output["dtype"]]).reshape(output["shape"])
            elif record["type"] == "Reshape":
                assert args[1].dtype == np.int32
                value = args[0].reshape(tuple(args[1]))
            elif record["type"] in {"Mul", "Sub"}:
                value = _reference(args[0], args[1], "mul" if record["type"] == "Mul" else "sub")
            else:
                raise AssertionError(f"unexpected layout operator {record['type']}")
            assert value.shape == output["shape"] and value.dtype == DTYPES[output["dtype"]]
            values[(index, output["port"])] = value
    return [values[(index, records[index]["outputs"][port]["port"])] for index, port in outputs]


@pytest.mark.parametrize("helper,kind,left_shape,right_shape,compact_shape", [
    (row_scale_mul, "Mul", (1, 8, 2, 8), (1, 8, 2, 1), (16, 8)),
    (row_scale_mul, "Mul", (1, 2, 64, 8), (1, 2, 64, 1), (128, 8)),
    (outer_sub_compact, "Sub", (2, 1, 64, 1), (2, 1, 1, 64), (2, 64, 64)),
])
def test_actual_recipe_has_one_fp16_operation_and_original_boundary(tmp_path, helper, kind, left_shape, right_shape, compact_shape):
    graph = _graph(helper, left_shape, right_shape)
    before = pickle.dumps(graph)
    emit_cpp(graph, recipe_dir=tmp_path / "recipe")
    assert pickle.dumps(graph) == before
    records, inputs, outputs = RecipeReader(tmp_path / "recipe/graph.recipe").read()
    counts = Counter(r["type"] for r in records)
    assert counts == {"Data": 2, "Const": 3, "Reshape": 3, kind: 1}
    operation, = [r for r in records if r["type"] == kind]
    assert operation["outputs"][0]["shape"] == compact_shape
    assert all(p["dtype"] == "float16" for p in operation["inputs"] + operation["outputs"])
    assert [records[i]["name"] for i in inputs] == ["left", "right"]
    assert [records[i]["outputs"][0]["shape"] for i in inputs] == [left_shape, right_shape]
    out, = outputs
    assert records[out[0]]["name"] == "result"
    assert records[out[0]]["outputs"][out[1]]["shape"] == graph.outputs[0].shape
    left, right = _values(left_shape, 17), _values(right_shape, 39)
    # The emitted graph must preserve exceptional values as well as finite rows.
    left.flat[:4] = [0., -0., np.inf, np.nan]
    right.flat[:4] = [np.inf, -1., -np.inf, 0.]
    actual, = _execute_recipe(records, outputs, {"left": left, "right": right})
    _assert_bits_and_nonfinite(actual, _reference(left, right, kind.lower()))


@pytest.mark.parametrize("helper", [row_scale_mul, outer_sub_compact])
@pytest.mark.parametrize("dtype", ["int32", "int64", "bool"])
def test_non_fp16_inputs_are_rejected_without_casts_or_partial_nodes(helper, dtype):
    graph = Graph("invalid_dtype")
    left = graph.input("left", (2, 1, 3, 1))
    right = graph.input("right", (2, 1, 1, 4), dtype)
    before = pickle.dumps(graph)
    with pytest.raises(ValueError, match="FP16"):
        helper(graph, left, right)
    assert pickle.dumps(graph) == before


@pytest.mark.parametrize("helper", [row_scale_mul, outer_sub_compact])
@pytest.mark.parametrize("foreign", ["left", "right", "equivalent_value", "wrong_graph", "not_value"])
def test_graph_ownership_is_checked_before_mutation(helper, foreign):
    graph = Graph("owner")
    left = graph.input("left", (2, 1, 3, 1))
    right = graph.input("right", (2, 1, 1, 4))
    other = Graph("other")
    if foreign == "left":
        left = other.input("left", left.shape)
    elif foreign == "right":
        right = other.input("right", right.shape)
    elif foreign == "equivalent_value":
        left = Value(left.name, left.shape, left.dtype)
    elif foreign == "not_value":
        right = None
    before = pickle.dumps(graph)
    with pytest.raises(ValueError, match="belong"):
        helper(other if foreign == "wrong_graph" else graph, left, right)
    assert pickle.dumps(graph) == before


@pytest.mark.parametrize("left_shape,right_shape", [
    ((8,), (1,)), ((1, 2, 3, 8), (2, 3, 1)), ((1, 2, 3, 8), (1, 1, 3, 1)),
    ((1, 2, 3, 8), (1, 3, 2, 1)), ((1, 2, 3, 8), (1, 2, 3, 8)),
    ((0, 8), (0, 1)), ((2, 0), (2, 1)), ((1 << 31, 8), (1 << 31, 1)),
    ((65536, 65536, 8), (65536, 65536, 1)),
])
def test_row_scale_rejects_other_broadcasts_and_invalid_reshape_dimensions(left_shape, right_shape):
    graph = Graph("invalid_row")
    left, right = graph.input("left", left_shape), graph.input("right", right_shape)
    before = pickle.dumps(graph)
    with pytest.raises(ValueError):
        row_scale_mul(graph, left, right)
    assert pickle.dumps(graph) == before


@pytest.mark.parametrize("left_shape,right_shape", [
    ((2, 3, 1), (2, 1, 4)), ((2, 2, 3, 1), (2, 1, 1, 4)),
    ((2, 1, 3, 2), (2, 1, 1, 4)), ((2, 1, 3, 1), (1, 1, 1, 4)),
    ((2, 1, 3, 1), (2, 1, 3, 4)), ((2, 1, 3, 1), (2, 2, 1, 4)),
    ((2, 1, 0, 1), (2, 1, 1, 4)), ((2, 1, 3, 1), (2, 1, 1, 0)),
    ((2, 1, 1 << 31, 1), (2, 1, 1, 4)),
])
def test_outer_sub_rejects_changed_batch_order_or_broadcast_contract(left_shape, right_shape):
    graph = Graph("invalid_outer")
    left, right = graph.input("left", left_shape), graph.input("right", right_shape)
    before = pickle.dumps(graph)
    with pytest.raises(ValueError):
        outer_sub_compact(graph, left, right)
    assert pickle.dumps(graph) == before


@pytest.mark.parametrize("helper,left_shape,right_shape", [
    (row_scale_mul, (2, 4), (2, 1)), (outer_sub_compact, (2, 1, 3, 1), (2, 1, 1, 4))])
@pytest.mark.parametrize("name", ["left", "bad name", None, "reserved"])
def test_output_names_fail_before_partial_graph_changes(helper, left_shape, right_shape, name):
    graph = Graph("names")
    left, right = graph.input("left", left_shape), graph.input("right", right_shape)
    graph.input("__binary_layout/reserved/arithmetic", (1,))
    before = pickle.dumps(graph)
    with pytest.raises(ValueError, match="name"):
        helper(graph, left, right, name=name)
    assert pickle.dumps(graph) == before


def test_explicit_result_name_cannot_collide_with_automatic_intermediate_names():
    graph = Graph("names")
    left, right = graph.input("left", (2, 4)), graph.input("right", (2, 1))
    result = row_scale_mul(graph, left, right, name="v0")
    assert result.name == "v0" and len({node.output.name for node in graph.nodes}) == 4
