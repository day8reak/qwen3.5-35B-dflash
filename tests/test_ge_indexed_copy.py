"""Independent copy/subtraction references and actual emitted Gather recipes."""
from collections import Counter
import math
from pathlib import Path
import pickle

import numpy as np
import pytest

from qwen35_moe_dflash.ge_cpp import emit_cpp
from qwen35_moe_dflash.ge_eval import evaluate
from qwen35_moe_dflash.ge_indexed_copy import MAX_OUTPUT_ELEMENTS, outer_sub_gather, triangular_mask_gather
from qwen35_moe_dflash.ge_ir import DTYPES, Graph, Value
from test_ge_select_lowering import RecipeReader


def _mask_graph(shape, strict=False):
    graph = Graph("indexed_triangle")
    x = graph.input("x", shape)
    graph.outputs = [triangular_mask_gather(graph, x, strict=strict)]
    return graph


def _sub_graph(batch, rows, columns):
    graph = Graph("indexed_outer_sub")
    left = graph.input("left", (batch, 1, rows, 1))
    right = graph.input("right", (batch, 1, 1, columns))
    graph.outputs = [outer_sub_gather(graph, left, right)]
    return graph


def _bits(shape):
    markers = np.array([0, 0x8000, 0x7c00, 0xfc00, 0x7e31, 0x7d01, 0xfe07,
                        0xfd02, 1, 0x8001, 0x3c00, 0xbc00, 0x7bff], np.uint16)
    array = np.resize(markers, shape)
    # Distinguish batches/channels even when a matrix is a multiple of 13 cells.
    for index, matrix in enumerate(array.reshape(-1, shape[-2], shape[-1])):
        matrix[-1, 0] = 0x2000 + index
    return array


@pytest.mark.parametrize("shape", [(1, 1, 1, 1), (2, 1, 7, 7), (2, 3, 4, 4), (2, 1, 64, 64)])
@pytest.mark.parametrize("strict", [False, True])
def test_triangle_copies_exact_bits_and_positive_zero_without_arithmetic(shape, strict):
    graph = _mask_graph(shape, strict)
    bits = _bits(shape)
    before = bits.tobytes()
    actual, = evaluate(graph, {"x": bits.view(np.float16)})
    expected = np.tril(bits, k=-1 if strict else 0)
    np.testing.assert_array_equal(actual.view(np.uint16), expected)
    assert bits.tobytes() == before
    assert Counter(node.kind for node in graph.nodes) == {"reshape": 2, "concat": 1, "gather": 1}
    assert graph.outputs[0].shape == shape and graph.outputs[0].dtype == "float16"
    zero, = [array for array in graph.constants.values() if array.dtype == np.float16]
    assert zero.shape == (1,) and zero.view(np.uint16).tolist() == [0]
    indices, = [array for array in graph.constants.values() if array.dtype == np.int32]
    assert indices.shape == (math.prod(shape),)
    assert indices.min() >= 0 and indices.max() <= math.prod(shape)


def _reference_sub(left, right):
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        return (left.astype(np.float32) - right.astype(np.float32)).astype(np.float16)


def _assert_sub(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype == np.float16
    for classify in (np.isnan, np.isposinf, np.isneginf):
        np.testing.assert_array_equal(classify(actual), classify(expected))
    finite = np.isfinite(expected)
    np.testing.assert_array_equal(actual[finite].view(np.uint16), expected[finite].view(np.uint16))


@pytest.mark.parametrize("batch,rows,columns", [(1, 1, 1), (3, 5, 7), (2, 64, 64), (2, 1, 64), (2, 64, 1)])
def test_outer_sub_preserves_batch_row_column_order_and_one_fp16_boundary(batch, rows, columns):
    graph = _sub_graph(batch, rows, columns)
    left = ((np.arange(batch * rows).reshape(batch, 1, rows, 1) * 3 - 17) / 8).astype(np.float16)
    right = ((np.arange(batch * columns).reshape(batch, 1, 1, columns) * 7 - 11) / 16).astype(np.float16)
    actual, = evaluate(graph, {"left": left, "right": right})
    _assert_sub(actual, _reference_sub(left, right))
    assert Counter(node.kind for node in graph.nodes) == {"reshape": 4, "gather": 2, "sub": 1}
    sub, = [node for node in graph.nodes if node.kind == "sub"]
    assert [value.shape for value in sub.inputs] == [(batch, 1, rows, columns)] * 2
    assert all(value.dtype == "float16" for value in (*sub.inputs, sub.output))
    assert all(array.dtype == np.int32 for array in graph.constants.values())


def test_outer_sub_nonfinite_locations_signed_zero_subnormals_and_overflow():
    markers = np.array([0, 0x8000, 1, 0x8001, 0x03ff, 0x83ff, 0x3c00, 0xbc00,
                        0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7e31, 0x7d01], np.uint16)
    left = np.stack([markers, markers[::-1]]).view(np.float16).reshape(2, 1, len(markers), 1)
    right = np.stack([np.roll(markers, 3), np.roll(markers, 7)]).view(np.float16).reshape(2, 1, 1, len(markers))
    before = (left.tobytes(), right.tobytes())
    actual, = evaluate(_sub_graph(2, len(markers), len(markers)), {"left": left, "right": right})
    _assert_sub(actual, _reference_sub(left, right))
    assert (left.tobytes(), right.tobytes()) == before


def _execute_recipe(records, outputs, feeds):
    values = {}
    for index, record in enumerate(records):
        output, = record["outputs"]
        args = [values[(port["src"], port["src_port"])] for port in record["inputs"]]
        if record["type"] == "Data":
            value = feeds[record["name"]]
        elif record["type"] == "Const":
            filename, offset, size = record["constant"]
            with Path(filename).open("rb") as stream:
                stream.seek(offset)
                payload = stream.read(size)
            assert len(payload) == size
            value = np.frombuffer(payload, DTYPES[output["dtype"]]).reshape(output["shape"])
        elif record["type"] == "Reshape":
            assert args[1].dtype == np.int32
            value = args[0].reshape(tuple(args[1]))
        elif record["type"] == "ConcatD":
            assert record["attrs"]["N"] == len(args) == 2
            assert record["attrs"]["concat_dim"] == 0
            value = np.concatenate(args, axis=0)
        elif record["type"] == "GatherV2":
            assert args[1].dtype == args[2].dtype == np.int32 and args[2].shape == ()
            assert int(args[2]) == 0 and args[1].ndim == 1
            assert args[1].min() >= 0 and args[1].max() < args[0].size
            assert record["attrs"]["batch_dims"] == 0
            value = np.take(args[0], args[1], axis=0)
        elif record["type"] == "Sub":
            assert args[0].shape == args[1].shape == output["shape"]
            value = _reference_sub(*args)
        else:
            raise AssertionError(f"unexpected indexed-copy recipe operator {record['type']}")
        assert value.shape == output["shape"] and value.dtype == DTYPES[output["dtype"]]
        values[(index, output["port"])] = value
    return [values[(index, records[index]["outputs"][port]["port"])] for index, port in outputs]


@pytest.mark.parametrize("strict", [False, True])
def test_real_triangle_recipe_has_int32_indices_axis_zero_and_no_arithmetic_mask(tmp_path, strict):
    graph = _mask_graph((2, 1, 64, 64), strict)
    before = pickle.dumps(graph)
    emit_cpp(graph, recipe_dir=tmp_path / "recipe")
    assert pickle.dumps(graph) == before
    records, inputs, outputs = RecipeReader(tmp_path / "recipe/graph.recipe").read()
    assert Counter(record["type"] for record in records) == {
        "Data": 1, "Const": 5, "Reshape": 2, "ConcatD": 1, "GatherV2": 1}
    assert [records[i]["name"] for i in inputs] == ["x"]
    bits = _bits((2, 1, 64, 64))
    actual, = _execute_recipe(records, outputs, {"x": bits.view(np.float16)})
    np.testing.assert_array_equal(actual.view(np.uint16), np.tril(bits, k=-1 if strict else 0))


def test_real_outer_recipe_copies_two_inputs_then_one_same_shape_sub(tmp_path):
    graph = _sub_graph(2, 64, 64)
    before = pickle.dumps(graph)
    emit_cpp(graph, recipe_dir=tmp_path / "recipe")
    assert pickle.dumps(graph) == before
    records, inputs, outputs = RecipeReader(tmp_path / "recipe/graph.recipe").read()
    assert Counter(record["type"] for record in records) == {
        "Data": 2, "Const": 8, "Reshape": 4, "GatherV2": 2, "Sub": 1}
    assert [records[i]["name"] for i in inputs] == ["left", "right"]
    left = _bits((2, 1, 64, 1)).view(np.float16)
    right = np.roll(_bits((2, 1, 1, 64)), 5).view(np.float16)
    actual, = _execute_recipe(records, outputs, {"left": left, "right": right})
    _assert_sub(actual, _reference_sub(left, right))


@pytest.mark.parametrize("helper", ["mask", "sub"])
@pytest.mark.parametrize("dtype", ["int32", "int64", "bool"])
def test_non_fp16_input_rejected_before_mutation(helper, dtype):
    graph = Graph("bad_dtype")
    x = graph.input("x", (2, 1, 4, 4) if helper == "mask" else (2, 1, 4, 1), dtype)
    y = graph.input("y", (2, 1, 1, 4))
    before = pickle.dumps(graph)
    with pytest.raises(ValueError, match="FP16"):
        triangular_mask_gather(graph, x) if helper == "mask" else outer_sub_gather(graph, x, y)
    assert pickle.dumps(graph) == before


@pytest.mark.parametrize("helper", ["mask", "sub"])
@pytest.mark.parametrize("fault", ["foreign", "equivalent", "not_value", "wrong_graph"])
def test_ownership_is_checked_without_partial_constants(helper, fault):
    graph = Graph("ownership")
    x = graph.input("x", (2, 1, 4, 4) if helper == "mask" else (2, 1, 4, 1))
    y = graph.input("y", (2, 1, 1, 4))
    if fault == "foreign":
        x = Graph("other").input("x", x.shape)
    elif fault == "equivalent":
        x = Value(x.name, x.shape, x.dtype)
    elif fault == "not_value":
        x = None
    before = pickle.dumps(graph)
    with pytest.raises(ValueError, match="belong"):
        owner = None if fault == "wrong_graph" else graph
        triangular_mask_gather(owner, x) if helper == "mask" else outer_sub_gather(owner, x, y)
    assert pickle.dumps(graph) == before


@pytest.mark.parametrize("shape", [(), (2, 64, 64), (2, 1, 3, 4), (2, 1, 0, 0),
                                   (1 << 31, 1, 1, 1), (1, 1, 1025, 1025)])
def test_mask_shape_and_size_rejection_precedes_graph_changes(shape):
    graph = Graph("invalid_mask")
    x = graph.input("x", shape)
    before = pickle.dumps(graph)
    with pytest.raises(ValueError):
        triangular_mask_gather(graph, x)
    assert pickle.dumps(graph) == before


@pytest.mark.parametrize("left,right", [
    ((2, 4, 1), (2, 1, 4)), ((2, 2, 4, 1), (2, 1, 1, 4)),
    ((2, 1, 4, 2), (2, 1, 1, 4)), ((2, 1, 4, 1), (1, 1, 1, 4)),
    ((2, 1, 4, 1), (2, 1, 4, 4)), ((2, 1, 0, 1), (2, 1, 1, 4)),
    ((1 << 31, 1, 1, 1), (1 << 31, 1, 1, 1)), ((1, 1, 1025, 1), (1, 1, 1, 1025)),
])
def test_sub_shape_and_size_rejection_precedes_graph_changes(left, right):
    graph = Graph("invalid_sub")
    x, y = graph.input("x", left), graph.input("y", right)
    before = pickle.dumps(graph)
    with pytest.raises(ValueError):
        outer_sub_gather(graph, x, y)
    assert pickle.dumps(graph) == before


@pytest.mark.parametrize("strict", [0, 1, None, "true", []])
def test_strict_flag_must_be_boolean(strict):
    graph = Graph("strict")
    x = graph.input("x", (2, 1, 4, 4))
    before = pickle.dumps(graph)
    with pytest.raises(ValueError, match="boolean"):
        triangular_mask_gather(graph, x, strict=strict)
    assert pickle.dumps(graph) == before


@pytest.mark.parametrize("helper", ["mask", "sub"])
@pytest.mark.parametrize("name", [None, "bad name", "x", "reserved"])
def test_all_names_checked_before_any_partial_graph_change(helper, name):
    graph = Graph("names")
    x = graph.input("x", (2, 1, 4, 4) if helper == "mask" else (2, 1, 4, 1))
    y = graph.input("y", (2, 1, 1, 4))
    suffix = "gather" if helper == "mask" else "right_full"
    graph.input(f"__indexed_copy/reserved/{suffix}", (1,))
    before = pickle.dumps(graph)
    with pytest.raises(ValueError, match="name"):
        triangular_mask_gather(graph, x, name=name) if helper == "mask" else outer_sub_gather(graph, x, y, name=name)
    assert pickle.dumps(graph) == before


def test_exact_output_element_limit_is_allowed_for_both_candidates():
    assert MAX_OUTPUT_ELEMENTS == 1 << 20
    mask = _mask_graph((1, 1, 1024, 1024))
    sub = _sub_graph(1, 1024, 1024)
    assert math.prod(mask.outputs[0].shape) == math.prod(sub.outputs[0].shape) == MAX_OUTPUT_ELEMENTS
    assert all(array.dtype in (np.float16, np.int32) for graph in (mask, sub) for array in graph.constants.values())
