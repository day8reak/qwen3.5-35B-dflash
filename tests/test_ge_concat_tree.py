"""Host bit-copy and emitted-recipe checks; no device placement assertions."""
from collections import Counter
import math
import pickle

import numpy as np
import pytest

from qwen35_moe_dflash.ge_concat_tree import concat_tree
from qwen35_moe_dflash.ge_cpp import emit_cpp
from qwen35_moe_dflash.ge_eval import evaluate
from qwen35_moe_dflash.ge_ir import DTYPES, Graph, Value
from test_ge_select_lowering import RecipeReader


def _graph(count, fan_in, shape=(2, 1, 1, 64), axis=2, widths=None, dtype="float16"):
    graph = Graph("concat_tree")
    values = []
    for index in range(count):
        current = list(shape)
        if widths is not None:
            current[axis] = widths[index]
        values.append(graph.input(f"row_{index}", current, dtype))
    graph.outputs = [concat_tree(graph, values, axis, fan_in)]
    return graph


def _feeds(graph):
    # Payloads include both zero signs, infinities, quiet/signalling NaNs,
    # subnormals and ordinary values. The per-input roll exposes row reordering.
    markers = np.array([0x0000, 0x8000, 0x7c00, 0xfc00, 0x7e31, 0x7d01,
                        0xfe07, 0xfd02, 0x0001, 0x8001, 0x3c00, 0xbc00], np.uint16)
    feeds = {}
    for index, value in enumerate(graph.inputs):
        bits = np.resize(np.roll(markers, index), math.prod(value.shape)).reshape(value.shape)
        # A distinct finite marker catches permutations separated by 12 inputs.
        bits.flat[-1] = 0x2000 + index
        feeds[value.name] = bits.view(np.float16)
    return feeds


def _leaf_order(graph, output):
    producers = {node.output.name: node for node in graph.nodes}

    def visit(value):
        if value.name not in producers:
            return [value.name]
        return [leaf for item in producers[value.name].inputs for leaf in visit(item)]

    return visit(output)


@pytest.mark.parametrize("count,fan_in,expected_nodes", [(64, 8, 9), (64, 2, 63),
                                                       (17, 8, 3), (9, 8, 2),
                                                       (65, 8, 10), (7, 2, 6)])
def test_adjacent_tree_preserves_input_order_and_all_fp16_bits(count, fan_in, expected_nodes):
    graph = _graph(count, fan_in)
    feeds = _feeds(graph)
    before = {key: value.tobytes() for key, value in feeds.items()}
    expected = np.concatenate([feeds[value.name] for value in graph.inputs], axis=2)
    actual, = evaluate(graph, feeds)
    np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))
    assert {key: value.tobytes() for key, value in feeds.items()} == before
    assert graph.outputs[0].shape == (2, 1, count, 64)
    assert graph.outputs[0].name == "result"
    assert len(graph.nodes) == expected_nodes
    assert {node.kind for node in graph.nodes} == {"concat"}
    assert all(2 <= len(node.inputs) <= fan_in for node in graph.nodes)
    assert _leaf_order(graph, graph.outputs[0]) == [value.name for value in graph.inputs]
    if count == 17:
        assert [value.name for value in graph.nodes[-1].inputs][-1] == "row_16"
        assert [len(node.inputs) for node in graph.nodes] == [8, 8, 3]


@pytest.mark.parametrize("axis,shape,widths", [(0, (1, 3), [2, 1, 4, 3, 1]),
                                              (-1, (2, 1), [1, 3, 2, 4, 5]),
                                              (-2, (2, 1, 1, 64), [1, 2, 1, 3, 2])])
def test_negative_axis_and_nonuniform_concat_lengths(axis, shape, widths):
    graph = _graph(len(widths), 2, shape, axis, widths)
    feeds = _feeds(graph)
    actual, = evaluate(graph, feeds)
    expected = np.concatenate(list(feeds.values()), axis=axis)
    np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))
    assert all(node.attrs["axis"] == axis % len(shape) for node in graph.nodes)


def _execute_recipe(records, outputs, feeds, fan_in):
    values = {}
    for index, record in enumerate(records):
        output, = record["outputs"]
        if record["type"] == "Data":
            value = feeds[record["name"]]
        else:
            assert record["type"] == "ConcatD"
            args = [values[(port["src"], port["src_port"])] for port in record["inputs"]]
            assert record["attrs"]["N"] == len(args)
            assert 2 <= len(args) <= fan_in
            for port, array in zip(record["inputs"], args):
                assert array.shape == port["shape"] and array.dtype == DTYPES[port["dtype"]]
            value = np.concatenate(args, axis=record["attrs"]["concat_dim"])
        assert value.shape == output["shape"] and value.dtype == DTYPES[output["dtype"]]
        values[(index, output["port"])] = value
    return [values[(index, records[index]["outputs"][port]["port"])] for index, port in outputs]


@pytest.mark.parametrize("count,fan_in,node_count", [(64, 8, 9), (64, 2, 63), (17, 8, 3)])
def test_real_recipe_has_bounded_concatd_fan_in_and_exact_shape_and_payload(tmp_path, count, fan_in, node_count):
    graph = _graph(count, fan_in, axis=-2)
    before = pickle.dumps(graph)
    emit_cpp(graph, recipe_dir=tmp_path / "recipe")
    assert pickle.dumps(graph) == before
    records, inputs, outputs = RecipeReader(tmp_path / "recipe/graph.recipe").read()
    assert Counter(record["type"] for record in records) == {"Data": count, "ConcatD": node_count}
    assert [records[index]["name"] for index in inputs] == [value.name for value in graph.inputs]
    assert all(record["attrs"]["concat_dim"] == 2 for record in records if record["type"] == "ConcatD")
    feeds = _feeds(graph)
    actual, = _execute_recipe(records, outputs, feeds, fan_in)
    expected = np.concatenate(list(feeds.values()), axis=2)
    np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))
    assert actual.shape == (2, 1, count, 64)


@pytest.mark.parametrize("dtype", ["float16", "int32", "int64", "bool"])
def test_single_input_is_returned_without_singleton_concat(dtype):
    graph = Graph("single")
    value = graph.input("source", (2, 3), dtype)
    before = pickle.dumps(graph)
    assert concat_tree(graph, [value], -1) is value
    assert pickle.dumps(graph) == before


@pytest.mark.parametrize("dtype", ["int32", "int64", "bool"])
def test_same_dtype_integer_and_boolean_inputs_are_copied_without_cast(dtype):
    graph = _graph(5, 2, shape=(2, 3), axis=1, dtype=dtype)
    feeds = {value.name: (np.arange(6).reshape(2, 3) + index).astype(DTYPES[dtype])
             for index, value in enumerate(graph.inputs)}
    actual, = evaluate(graph, feeds)
    expected = np.concatenate(list(feeds.values()), axis=1)
    assert actual.dtype == expected.dtype == DTYPES[dtype]
    assert actual.tobytes() == expected.tobytes()
    assert all(node.output.dtype == dtype for node in graph.nodes)


def test_repeated_input_remains_repeated_and_explicit_name_is_safe():
    graph = Graph("repeated")
    value = graph.input("source", (2, 1))
    result = concat_tree(graph, [value] * 9, 1, name="v0")
    assert result.name == "v0" and result.shape == (2, 9)
    assert _leaf_order(graph, result) == ["source"] * 9


@pytest.mark.parametrize("fault", ["empty", "foreign", "equivalent_value", "not_value", "wrong_graph",
                                  "dtype", "rank", "nonaxis_shape", "zero", "scalar"])
def test_invalid_inputs_are_rejected_before_graph_mutation(fault):
    graph = Graph("bad_inputs")
    left = graph.input("left", (2, 1, 1, 64))
    shape = {"rank": (2, 1, 64), "nonaxis_shape": (3, 1, 1, 64),
             "zero": (2, 1, 0, 64), "scalar": ()}.get(fault, left.shape)
    right = graph.input("right", shape, "int32" if fault == "dtype" else "float16")
    values = [left, right]
    if fault == "empty":
        values = []
    elif fault == "foreign":
        values[-1] = Graph("other").input("right", left.shape)
    elif fault == "equivalent_value":
        values[-1] = Value(right.name, right.shape, right.dtype)
    elif fault == "not_value":
        values[-1] = None
    elif fault == "scalar":
        values = [right]
    before = pickle.dumps(graph)
    with pytest.raises(ValueError):
        concat_tree(None if fault == "wrong_graph" else graph, values, 2)
    assert pickle.dumps(graph) == before


@pytest.mark.parametrize("axis", [True, False, np.bool_(True), 1.0, "2", None, 4, -5])
def test_axis_requires_a_nonboolean_in_range_integer(axis):
    graph = Graph("axis")
    values = [graph.input(f"x{i}", (2, 1, 1, 64)) for i in range(9)]
    before = pickle.dumps(graph)
    with pytest.raises(ValueError, match="axis"):
        concat_tree(graph, values, axis)
    assert pickle.dumps(graph) == before


@pytest.mark.parametrize("fan_in", [True, False, np.bool_(True), 2.0, "8", None, 0, 1, -2])
def test_fan_in_requires_an_integer_of_at_least_two(fan_in):
    graph = Graph("fan_in")
    values = [graph.input(f"x{i}", (2, 1, 1, 64)) for i in range(9)]
    before = pickle.dumps(graph)
    with pytest.raises(ValueError, match="max_fan_in"):
        concat_tree(graph, values, 2, fan_in)
    assert pickle.dumps(graph) == before


@pytest.mark.parametrize("name", ["x0", "bad name", None, "reserved"])
def test_names_are_validated_before_even_the_first_tree_node(name):
    graph = Graph("names")
    values = [graph.input(f"x{i}", (2, 1, 1, 64)) for i in range(65)]
    graph.input("__concat/reserved/l1_g0", (1,))
    before = pickle.dumps(graph)
    with pytest.raises(ValueError, match="name"):
        concat_tree(graph, values, 2, name=name)
    assert pickle.dumps(graph) == before


def test_numpy_integer_parameters_are_accepted():
    graph = _graph(9, np.int64(8), axis=np.int32(-2))
    assert graph.outputs[0].shape == (2, 1, 9, 64)
