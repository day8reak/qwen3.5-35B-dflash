"""Reject inconsistent Cast/selection boundaries before constructing AIR."""
import numpy as np
import pytest

from qwen35_moe_dflash.ge_cpp import emit_cpp
from qwen35_moe_dflash.ge_ir import Graph
from qwen35_moe_dflash.ge_recipe_read import read_recipe


def _emit(graph, tmp_path, mode, **kwargs):
    return emit_cpp(graph, recipe_dir=tmp_path / "recipe" if mode == "recipe" else None,
                    **kwargs)


@pytest.mark.parametrize("mode", ["inline", "recipe"])
@pytest.mark.parametrize("kind,dtype,shape,attrs,error", [
    ("cast", "int32", (8,), {}, "Cast must preserve"),
    ("argmax", "float16", (2,), {"axis": 1}, "output dtype"),
    ("argmax", "bool", (2,), {"axis": 1}, "output dtype"),
    ("argmax", "int32", (2,), {"axis": 1, "keepdims": True}, "output shape"),
    ("argmax", "int32", (2, 1), {"axis": 1}, "output shape"),
    ("argmax", "int32", (2,), {"axis": 3}, "axis"),
    ("argmax", "int32", (2,), {"axis": -3}, "axis"),
    ("argmax", "int32", (2,), {"axis": True}, "axis"),
    ("argmax", "int32", (2,), {"axis": 1, "keepdims": "false"}, "keepdims"),
    ("topk_values", "int32", (2, 2), {"k": 2}, "output dtype"),
    ("topk_indices", "int64", (2, 2), {"k": 2}, "output dtype"),
    ("topk_values", "float16", (2, 3), {"k": 2}, "output shape"),
    ("topk_indices", "int32", (2, 3), {"k": 2}, "output shape"),
    ("topk_values", "float16", (2, 2), {"k": -1}, "TopKV2 k"),
    ("topk_values", "float16", (2, 2), {"k": 5}, "TopKV2 k"),
    ("topk_values", "float16", (2, 2), {"k": 2.0}, "TopKV2 k"),
    ("topk_values", "float16", (2, 1), {"k": True}, "TopKV2 k"),
    ("topk_values", "float16", (2, 2), {"k": 2, "axis": 3}, "axis"),
])
def test_invalid_source_descriptors_fail_in_both_serializers(
        tmp_path, mode, kind, dtype, shape, attrs, error):
    graph = Graph("invalid_descriptor")
    source = graph.input("source", (2, 4))
    graph.outputs = [graph.op(kind, source, shape=shape, dtype=dtype, attrs=attrs, name="result")]
    with pytest.raises(ValueError, match=error):
        _emit(graph, tmp_path, mode)
    assert not (tmp_path / "recipe").exists()


@pytest.mark.parametrize("mode", ["inline", "recipe"])
def test_topk_cache_cannot_change_declared_graph_output_shape(tmp_path, mode):
    graph = Graph("shared_topk_descriptors")
    source = graph.input("source", (2, 4))
    values = graph.op("topk_values", source, shape=(2, 2), attrs={"k": 2}, name="values")
    indices = graph.op("topk_indices", source, shape=(2, 3), dtype="int32",
                       attrs={"k": 2}, name="indices")
    graph.outputs = [values, indices]
    with pytest.raises(ValueError, match="indices: topk_indices output shape"):
        _emit(graph, tmp_path, mode)


def test_invalid_constant_selection_rejected_before_folding(tmp_path, monkeypatch):
    graph = Graph("invalid_constant_selection")
    graph.input("unused_input", (1,))
    source = graph.const([[1, 2, 3, 4]], name="scores")
    graph.outputs = [graph.op("topk_values", source, shape=(1, 2), dtype="int32",
                              attrs={"k": 2}, name="result")]

    def unexpected_fold(*args, **kwargs):
        pytest.fail("invalid source descriptor reached constant folding")

    monkeypatch.setattr("qwen35_moe_dflash.ge_constants.fold_constants", unexpected_fold)
    with pytest.raises(ValueError, match="output dtype"):
        _emit(graph, tmp_path, "recipe", fold_constant_arithmetic=True)
    assert not (tmp_path / "recipe").exists()


@pytest.mark.parametrize("dtype", ["int32", "int64"])
@pytest.mark.parametrize("keepdims", [False, True])
def test_valid_argmax_attributes_and_output_abi(tmp_path, dtype, keepdims):
    graph = Graph("valid_argmax")
    source = graph.input("source", (2, 3, 4))
    shape = (2, 1, 4) if keepdims else (2, 4)
    graph.outputs = [graph.op("argmax", source, shape=shape, dtype=dtype,
                              attrs={"axis": -2, "keepdims": keepdims}, name="result")]
    _emit(graph, tmp_path, "recipe")
    recipe = read_recipe(tmp_path / "recipe/graph.recipe")
    argmax, = [record for record in recipe["records"] if record["type"] == "ArgMaxV2"]
    assert argmax["attrs"]["dtype"] == {"type": "dtype", "value": dtype}
    assert argmax["outputs"][0]["dtype"] == dtype
    assert argmax["outputs"][0]["shape"] == [2, 4]
    index, port = recipe["outputs"][0]
    output = recipe["records"][index]["outputs"][port]
    assert (output["shape"], output["dtype"]) == (list(shape), dtype)


@pytest.mark.parametrize("first", ["values", "indices"])
def test_valid_topk_sharing_preserves_both_output_contracts(tmp_path, first):
    graph = Graph("valid_topk")
    source = graph.input("source", (2, 4))
    outputs = {}
    for name in (first, "indices" if first == "values" else "values"):
        outputs[name] = graph.op("topk_" + name, source, shape=(2, 2),
                                 dtype="int32" if name == "indices" else "float16",
                                 attrs={"axis": np.int32(-1), "k": np.int32(2)}, name=name)
    graph.outputs = [outputs["values"], outputs["indices"]]
    _emit(graph, tmp_path, "recipe")
    recipe = read_recipe(tmp_path / "recipe/graph.recipe")
    topks = [(index, record) for index, record in enumerate(recipe["records"])
             if record["type"] == "TopKV2"]
    assert len(topks) == 1
    index, record = topks[0]
    assert recipe["outputs"] == [[index, 0], [index, 1]]
    assert [(port["port"], port["shape"], port["dtype"]) for port in record["outputs"]] == [
        ("values", [2, 2], "float16"), ("indices", [2, 2], "int32")]


@pytest.mark.parametrize("kind", ["reduce_sum", "reduce_mean"])
@pytest.mark.parametrize("policy", ["native", "matmul"])
@pytest.mark.parametrize("fold", [False, True])
@pytest.mark.parametrize("fault", ["dtype", "shape"])
def test_reduction_descriptor_checks_do_not_depend_on_policy_or_folding(
        tmp_path, monkeypatch, kind, policy, fold, fault):
    graph = Graph("invalid_reduction")
    graph.input("unused_input", (1,))
    source = graph.const(np.ones((2, 4), np.float16), name="source")
    graph.outputs = [graph.op(kind, source, shape=(2, 4) if fault == "shape" else (2,),
                              dtype="int32" if fault == "dtype" else "float16",
                              attrs={"axes": [1], "keepdims": False}, name="result")]

    def unexpected_fold(*args, **kwargs):
        pytest.fail("invalid reduction descriptor reached constant folding")

    monkeypatch.setattr("qwen35_moe_dflash.ge_constants.fold_constants", unexpected_fold)
    with pytest.raises(ValueError, match="dtype must match" if fault == "dtype" else "output shape"):
        _emit(graph, tmp_path, "recipe", fp16_reductions=policy, fold_constant_arithmetic=fold)
    assert not (tmp_path / "recipe").exists()


@pytest.mark.parametrize("kind", ["reduce_sum", "reduce_mean"])
@pytest.mark.parametrize("policy", ["native", "matmul"])
@pytest.mark.parametrize("shape,axes,keepdims,expected", [
    ((2, 4), [], False, (2, 4)),
    ((2, 4), [], True, (2, 4)),
    ((), [], False, ()),
    ((2, 0, 4), [-2], False, (2, 4)),
    ((2, 0, 4), [-2], True, (2, 1, 4)),
    ((0, 4), [-1], False, (0,)),
])
def test_reduction_empty_axes_and_zero_dimensions_preserve_native_contract(
        tmp_path, kind, policy, shape, axes, keepdims, expected):
    graph = Graph("native_reduction_edges")
    source = graph.input("source", shape)
    graph.outputs = [graph.op(kind, source, shape=expected,
                              attrs={"axes": axes, "keepdims": keepdims}, name="result")]
    _emit(graph, tmp_path, "recipe", fp16_reductions=policy, fold_constant_arithmetic=True)
    recipe = read_recipe(tmp_path / "recipe/graph.recipe")
    reduction, = [record for record in recipe["records"]
                  if record["type"] in {"ReduceSum", "ReduceMean"}]
    assert reduction["outputs"][0]["shape"] == list(expected)
    assert reduction["outputs"][0]["dtype"] == "float16"
    assert reduction["attrs"]["keep_dims"] == {"type": "bool", "value": keepdims}
    if kind == "reduce_mean":
        assert reduction["attrs"]["noop_with_empty_axes"] == {"type": "bool", "value": True}
    assert not any(record["type"] == "MatMul" for record in recipe["records"])
