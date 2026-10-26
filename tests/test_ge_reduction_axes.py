"""Reduce axes use validated INT32 constants; arithmetic and order are unchanged."""
from pathlib import Path

import numpy as np
import pytest

from qwen35_moe_dflash.ge_cpp import emit_cpp
from qwen35_moe_dflash.ge_eval import evaluate
from qwen35_moe_dflash.ge_ir import Graph
from test_ge_select_lowering import RecipeReader


def _graph(kind, axes, *, dtype="float16", shape=(2, 3, 4), output_shape=(1, 3, 1)):
    graph = Graph("reduction_axes")
    source = graph.input("x", shape, dtype)
    graph.outputs = [graph.op(kind, source, shape=output_shape, dtype=dtype, name="result",
                              attrs={"axes": axes, "keepdims": True})]
    return graph


@pytest.mark.parametrize("kind", ["reduce_sum", "reduce_mean"])
@pytest.mark.parametrize("axes", [[2, 0], [-1, 0], (-3, -1), np.array([-1, 0], dtype=np.int64)])
def test_cpp_and_recipe_keep_axes_values_order_and_fp16_data(kind, axes, tmp_path):
    graph = _graph(kind, axes)
    expected = np.asarray(axes, dtype="<i4")
    cpp = emit_cpp(graph)
    descriptor = f"Desc(std::vector<int64_t>{{{len(axes)}}}, ge::DT_INT32)"
    literal = "std::vector<uint8_t>{" + ",".join(map(str, expected.tobytes())) + "}"
    assert '"__ge/result/axes","Const"' in cpp
    assert f'{descriptor}, {literal});' in cpp  # SetConst payload and dtype.
    assert f'UpdateInputDesc("axes", {descriptor})' in cpp
    emit_cpp(graph, recipe_dir=tmp_path / "recipe")
    records, _, _ = RecipeReader(tmp_path / "recipe/graph.recipe").read()
    reduction, = [record for record in records if record["type"] in ("ReduceSum", "ReduceMean")]
    x, axis_port = reduction["inputs"]
    assert x["dtype"] == reduction["outputs"][0]["dtype"] == "float16"
    assert axis_port["dtype"] == "int32" and axis_port["shape"] == (len(axes),)
    constant = records[axis_port["src"]]
    assert constant["type"] == "Const" and constant["outputs"][0]["dtype"] == "int32"
    path, offset, size = constant["constant"]
    with Path(path).open("rb") as stream:
        stream.seek(offset)
        raw = stream.read(size)
    assert size == len(axes) * 4 and raw == expected.tobytes()
    assert reduction["attrs"]["keep_dims"]
    if kind == "reduce_mean":
        assert reduction["attrs"]["noop_with_empty_axes"]
    values = np.arange(24, dtype=np.float16).reshape(2, 3, 4) / np.float16(7)
    operation = np.sum if kind == "reduce_sum" else np.mean
    serialized_result = operation(values.astype(np.float32), axis=tuple(np.frombuffer(raw, dtype="<i4")),
                                  keepdims=True).astype(np.float16)
    actual, = evaluate(graph, {"x": values})
    np.testing.assert_array_equal(serialized_result.view(np.uint16), actual.view(np.uint16))


def test_int32_sum_data_stays_int32(tmp_path):
    graph = _graph("reduce_sum", [1], dtype="int32", shape=(1, 8), output_shape=(1, 1))
    emit_cpp(graph, recipe_dir=tmp_path / "recipe")
    records, _, _ = RecipeReader(tmp_path / "recipe/graph.recipe").read()
    reduction, = [record for record in records if record["type"] == "ReduceSum"]
    assert [port["dtype"] for port in reduction["inputs"] + reduction["outputs"]] == ["int32"] * 3
    assert not any(record["type"] == "Cast" for record in records)


@pytest.mark.parametrize("kind", ["reduce_sum", "reduce_mean"])
@pytest.mark.parametrize("axes,match", [
    ([1.5], "integers"), ([1.0], "integers"), ([float("nan")], "integers"),
    ([float("inf")], "integers"), ([True], "integers"), ([np.bool_(False)], "integers"),
    (["1"], "integers"), ([None], "integers"), ([[1]], "integers"),
    ([1 << 31], "INT32"), ([-(1 << 31) - 1], "INT32"), ([np.uint64(1 << 63)], "INT32"),
    ([3], "outside input rank"), ([-4], "outside input rank"),
    ([1, 1], "duplicate"), ([-1, 2], "duplicate"),
    (None, "one-dimensional"), (1, "one-dimensional"), ("1", "one-dimensional"),
    (np.array(1, dtype=np.int32), "one-dimensional"),
    (np.array([[1]], dtype=np.int32), "one-dimensional"),
])
def test_invalid_axes_rejected_before_constant_serialization(kind, axes, match, tmp_path):
    graph = _graph(kind, axes)
    with pytest.raises(ValueError, match=match):
        emit_cpp(graph)
    with pytest.raises(ValueError, match=match):
        emit_cpp(graph, recipe_dir=tmp_path / "recipe")
    assert not (tmp_path / "recipe").exists()


@pytest.mark.parametrize("kind", ["reduce_sum", "reduce_mean"])
def test_empty_axes_remain_empty_without_changing_operator_attributes(kind, tmp_path):
    graph = _graph(kind, [], output_shape=(2, 3, 4))
    emit_cpp(graph, recipe_dir=tmp_path / "recipe")
    records, _, _ = RecipeReader(tmp_path / "recipe/graph.recipe").read()
    reduction, = [record for record in records if record["type"] in ("ReduceSum", "ReduceMean")]
    axes = reduction["inputs"][1]
    assert axes["dtype"] == "int32" and axes["shape"] == (0,)
    assert records[axes["src"]]["constant"][2] == 0
    assert reduction["attrs"] == ({"keep_dims": True, "noop_with_empty_axes": True}
                                   if kind == "reduce_mean" else {"keep_dims": True})
