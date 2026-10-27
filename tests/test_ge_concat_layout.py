"""Bit-exact host/recipe contracts for the observed FP16 rank3 Concat fault."""
import math
import pickle

import numpy as np
import pytest

from qwen35_moe_dflash.ge_cpp import emit_cpp
from qwen35_moe_dflash.ge_ir import Graph
from test_ge_reduction_lowering import _execute
from test_ge_select_lowering import RecipeReader


def _graph(shapes, axis, dtype="float16", constant=None):
    graph = Graph("concat_layout")
    values = [graph.const(np.zeros(shape, dtype=dtype), dtype=dtype, name=f"x{i}") if i == constant
              else graph.input(f"x{i}", shape, dtype) for i, shape in enumerate(shapes)]
    shape = list(shapes[0]); shape[axis] = sum(item[axis] for item in shapes)
    graph.outputs = [graph.op("concat", *values, shape=shape, dtype=dtype, attrs={"axis": axis}, name="result")]
    return graph


def _recipe(graph, folder, **kwargs):
    emit_cpp(graph, recipe_dir=folder, **kwargs)
    records, _, outputs = RecipeReader(folder / "graph.recipe").read()
    return records, outputs


@pytest.mark.parametrize("shapes,axis,constant", [
    ([(1, 48, 3), (1, 48, 8)], 2, 0),
    ([(1, 2, 8), (1, 2, 56)], 2, 1),
    ([(1, 2, 8), (1, 2, 56)], -1, 1),
])
def test_observed_padding_shapes_keep_each_row_and_head_in_order(tmp_path, shapes, axis, constant):
    graph = _graph(shapes, axis, constant=constant)
    before = pickle.dumps(graph)
    feeds = {value.name: (np.arange(math.prod(value.shape), dtype=np.float16) + 1).reshape(value.shape)
             for value in graph.inputs}
    inputs = [graph.constants.get(f"x{i}", feeds.get(f"x{i}")) for i in range(len(shapes))]
    records, outputs = _recipe(graph, tmp_path / "recipe")
    actual, = _execute(records, outputs, feeds)
    expected = np.concatenate(inputs, axis=axis)
    np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))
    assert pickle.dumps(graph) == before
    assert np.count_nonzero(actual) == sum(np.count_nonzero(value) for value in inputs)
    concat, = [record for record in records if record["type"] == "ConcatD"]
    assert concat["attrs"] == {"concat_dim": axis % 3, "N": 2}
    assert [port["shape"] for port in concat["inputs"]] == [tuple(shape) + (1,) for shape in shapes]
    assert concat["outputs"][0]["shape"] == expected.shape + (1,)
    assert records[outputs[0][0]]["type"] == "Reshape"
    assert records[outputs[0][0]]["name"] == "result"
    assert records[outputs[0][0]]["outputs"][0]["shape"] == expected.shape
    assert not {"Cast", "Mul", "Add", "Transpose"} & {record["type"] for record in records}


@pytest.mark.parametrize("axis", [0, 1, 2, -1, -2, -3])
def test_all_rank3_axes_preserve_every_half_bit_and_nonuniform_lengths(tmp_path, axis):
    shapes = []
    for length in (1, 3, 2):
        shape = [2, 3, 4]; shape[axis] = length; shapes.append(shape)
    graph = _graph(shapes, axis)
    markers = np.array([0, 0x8000, 1, 0x8001, 0x7c00, 0xfc00, 0x7e31, 0x7d01, 0xfe07, 0x3c00], np.uint16)
    feeds = {value.name: np.resize(np.roll(markers, index), math.prod(value.shape)).reshape(value.shape).view(np.float16)
             for index, value in enumerate(graph.inputs)}
    original = {name: value.tobytes() for name, value in feeds.items()}
    records, outputs = _recipe(graph, tmp_path / "recipe")
    actual, = _execute(records, outputs, feeds)
    expected = np.concatenate(list(feeds.values()), axis=axis)
    np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))
    assert original == {name: value.tobytes() for name, value in feeds.items()}
    assert all(len(port["shape"]) == 4 for record in records if record["type"] == "ConcatD"
               for port in record["inputs"] + record["outputs"])


@pytest.mark.parametrize("mode", ["native", "matmul"])
def test_generated_scan_prefix_concat_uses_same_explicit_rank4_path(tmp_path, mode):
    graph = Graph("prefix_concat")
    source = graph.input("x", (2, 1, 64))
    graph.outputs = [graph.op("cumsum", source, shape=source.shape, name="scan",
                              attrs={"axis": -1, "scope": "gdn_nonpositive"})]
    value = np.zeros(source.shape, np.float16)
    value[0, 0, :8] = -.125
    value[1, 0, :8] = -.25
    report = {}
    records, outputs = _recipe(graph, tmp_path / mode, fp16_reductions=mode, reduction_report=report)
    actual, = _execute(records, outputs, {"x": value})
    expected = value.astype(np.float32).cumsum(axis=-1).astype(np.float16)
    np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))
    assert np.count_nonzero(actual[0]) == np.count_nonzero(actual[1]) == 64
    concat, = [record for record in records if record["type"] == "ConcatD"]
    assert concat["attrs"] == {"concat_dim": 2, "N": 64}
    assert all(port["shape"] == (2, 1, 1, 1) for port in concat["inputs"])
    assert concat["outputs"][0]["shape"] == (2, 1, 64, 1)
    assert report["counts"]["scan_prefix_replaced"] == (64 if mode == "matmul" else 0)


@pytest.mark.parametrize("dtype,shape", [("int32", (1, 2, 8)), ("int64", (1, 2, 8)),
                                         ("bool", (1, 2, 8)), ("float16", (2, 8)),
                                         ("float16", (1, 2, 8, 8))])
def test_other_dtype_and_rank_paths_keep_native_concat(tmp_path, dtype, shape):
    graph = _graph([shape, shape], -1, dtype)
    records, _ = _recipe(graph, tmp_path / "recipe")
    concat, = [record for record in records if record["type"] == "ConcatD"]
    assert concat["inputs"][0]["shape"] == shape
    assert concat["attrs"] == {"concat_dim": -1, "N": 2}
    assert not any("/concat_rank4/" in record["name"] for record in records)


def test_inline_and_recipe_serialization_keep_same_rank4_descriptors(tmp_path):
    graph = _graph([(1, 2, 8), (1, 2, 56)], 2, constant=1)
    text = emit_cpp(graph)
    records, _ = _recipe(graph, tmp_path / "recipe")
    for record in records:
        assert f'Op("{record["name"]}", "{record["type"]}"' in text
    assert 'Desc(std::vector<int64_t>{1,2,8,1}, ge::DT_FLOAT16)' in text
    assert 'Desc(std::vector<int64_t>{1,2,64,1}, ge::DT_FLOAT16)' in text
