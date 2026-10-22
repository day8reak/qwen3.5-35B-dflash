"""INT32 Reshape controls avoid the observed INT64 low-word sentinel rewrite.

These verify host serializers and exact byte layouts, not the private OMG binary.
"""
import json
import re

import numpy as np
import pytest

from qwen35_moe_dflash.ge_cpp import emit_cpp
from qwen35_moe_dflash.ge_export import build_pair, configurations
from qwen35_moe_dflash.ge_ir import Graph
from test_ge_select_lowering import RecipeReader
from test_ge_static_broadcast import _payload, _serialize_both


def _graph(source_shape, target_shape, *, dtype="float16", name="view"):
    graph = Graph("static_reshape")
    source = graph.input("source", source_shape, dtype)
    graph.outputs = [graph.op("reshape", source, shape=target_shape, dtype=dtype,
                              attrs={"shape": target_shape}, name=name)]
    return graph


def _check_reshape(records, record):
    assert record["type"] == "Reshape"
    assert record["attrs"] == {"axis": 0, "num_axes": -1}
    data, shape = record["inputs"]
    result, = record["outputs"]
    assert data["dtype"] == result["dtype"]
    assert shape["dtype"] == "int32"
    literal = records[shape["src"]]
    assert literal["type"] == "Const"
    assert literal["outputs"][0]["dtype"] == "int32"
    assert np.frombuffer(_payload(literal), dtype="<i4").tolist() == list(result["shape"])
    return literal


@pytest.mark.parametrize("source,target,name", [
    ((1, 2, 1, 1, 55), (1, 2, 1, 55), "v974"),
    ((1, 2, 1, 55), (1, 2, 1, 55, 1), "v975"),
    ((1, 2, 3), (6,), "merged"),
    ((1, 2, 3), (3, 2), "reordered"),
    ((1, 1), (), "scalar"),
])
def test_reshape_shape_in_both_serializers_is_checked_int32(tmp_path, source, target, name):
    records, _, outputs = _serialize_both(_graph(source, target, name=name), tmp_path)
    record = records[outputs[0][0]]
    literal = _check_reshape(records, record)
    assert literal["name"] == f"__ge/{name}/shape"
    assert len(_payload(literal)) == 4 * len(target)
    assert {item["type"] for item in records} == {"Data", "Const", "Reshape"}


@pytest.mark.parametrize("dimensions", [(1, 2, 1, 55), (1, 2, 1, 55, 1)])
def test_observed_32bit_minus_one_rewrite_has_correct_sign_with_int32(dimensions):
    # User ConvertReshape prints ...55 0... -> ...-1 0... on an INT64
    # Const. Reproduce that byte write independently of the GE emitter.
    dimension = dimensions.index(55)
    wide = np.array(dimensions, dtype="<i8")
    wide.view("<i4")[2 * dimension] = -1
    assert wide[dimension] == 4294967295  # not the required signed sentinel
    narrow = np.array(dimensions, dtype="<i4")
    narrow[dimension] = -1
    assert narrow[dimension] == -1
    assert narrow.tobytes()[4 * dimension:4 * (dimension + 1)] == b"\xff" * 4


@pytest.mark.parametrize("dtype", ["float16", "int32", "int64", "bool"])
def test_reshape_data_dtype_unchanged_and_no_casts_inserted(tmp_path, dtype):
    records, _, outputs = _serialize_both(_graph((1, 2, 1), (2,), dtype=dtype), tmp_path)
    record = records[outputs[0][0]]
    _check_reshape(records, record)
    assert record["inputs"][0]["dtype"] == record["outputs"][0]["dtype"] == dtype
    assert not any(item["type"] in {"Cast", "Squeeze", "ExpandDims"} for item in records)


def test_reshape_preserves_fp16_special_value_payloads(tmp_path):
    graph = Graph("bit_exact_view")
    graph.input("unused", (1,))
    bits = np.array([0x0000, 0x8000, 0x7c00, 0xfc00, 0x7e31, 0x7d01], dtype="<u2")
    source = graph.const(bits.view("<f2").reshape(1, 6, 1))
    graph.outputs = [graph.op("reshape", source, shape=(6,), attrs={"shape": (6,)})]
    # Normal export folds this static view byte-for-byte. Both serializers
    # must preserve every special value without executing FP16 arithmetic.
    records, _, outputs = _serialize_both(graph, tmp_path)
    folded = records[outputs[0][0]]
    assert folded["type"] == "Const"
    assert folded["outputs"][0]["shape"] == (6,)
    assert _payload(folded) == bits.tobytes()
    assert not any(record["type"] == "Reshape" for record in records)

    # Explicit descriptor-only mode still exercises the INT32 shape path.
    emit_cpp(graph, fold_constant_arithmetic=False, recipe_dir=tmp_path / "unfolded")
    records, _, outputs = RecipeReader(tmp_path / "unfolded/graph.recipe").read()
    record = records[outputs[0][0]]
    shape = _check_reshape(records, record)
    data = records[record["inputs"][0]["src"]]
    actual = np.frombuffer(_payload(data), dtype="<f2").reshape(
        tuple(np.frombuffer(_payload(shape), dtype="<i4")))
    assert actual.shape == (6,)
    assert actual.tobytes() == bits.tobytes()


@pytest.mark.parametrize("dimension", [1 << 31, 1 << 40])
def test_out_of_int32_range_rejected_before_numpy_narrowing(tmp_path, dimension):
    graph = _graph((dimension,), (dimension,))
    with pytest.raises(ValueError, match="Reshape dimensions must fit nonnegative INT32"):
        emit_cpp(graph, recipe_dir=tmp_path / "recipe")
    assert not (tmp_path / "recipe").exists()


def test_int32_upper_bound_dimension_remains_exact(tmp_path):
    dimension = (1 << 31) - 1
    records, _, outputs = _serialize_both(_graph((dimension,), (dimension,)), tmp_path)
    _check_reshape(records, records[outputs[0][0]])


def test_shape_parameter_must_match_output_descriptor(tmp_path):
    graph = _graph((2, 3), (6,))
    graph.nodes[-1].attrs["shape"] = [3, 2]
    with pytest.raises(ValueError, match=("Reshape shape differs from output descriptor"
                                          "|static reshape shape/element count mismatch")):
        emit_cpp(graph, recipe_dir=tmp_path / "recipe")


def test_dense_flatten_dimension_uses_the_same_overflow_guard(tmp_path):
    graph = Graph("flatten_overflow")
    source = graph.input("source", (2, 1 << 30, 1))
    weight = graph.input("weight", (1, 2))
    graph.outputs = [graph.op("matmul", source, weight, shape=(2, 1 << 30, 2),
                             attrs={"transpose_a": False, "transpose_b": False})]
    with pytest.raises(ValueError, match="flat_input: Reshape dimensions must fit nonnegative INT32"):
        emit_cpp(graph, recipe_dir=tmp_path / "recipe")


@pytest.mark.parametrize("kind", ["matmul", "argmax"])
def test_synthetic_dense_and_argmax_helper_reshapes_are_int32(tmp_path, kind):
    graph = Graph("helper_views")
    source = graph.input("source", (1, 2, 3))
    if kind == "matmul":
        weight = graph.input("weight", (3, 4))
        output = graph.op(kind, source, weight, shape=(1, 2, 4),
                          attrs={"transpose_a": False, "transpose_b": False})
    else:
        output = graph.op(kind, source, shape=(1, 2, 1), dtype="int32",
                          attrs={"axis": -1, "keepdims": True})
    graph.outputs = [output]
    records, _, _ = _serialize_both(graph, tmp_path)
    reshapes = [item for item in records if item["type"] == "Reshape"]
    assert len(reshapes) == (2 if kind == "matmul" else 1)
    for record in reshapes:
        _check_reshape(records, record)


@pytest.mark.parametrize("graph_index", [0, 1], ids=["target", "draft"])
def test_every_tiny_model_reshape_uses_int32_and_other_controls_keep_width(tmp_path, graph_index):
    graph = build_pair(*configurations(tiny=True), 8)[graph_index]
    paths = {key: tmp_path / f"unopened-{index}.bin" for index, key in enumerate(graph.weights)}
    options = dict(external_weights=True, fold_constant_arithmetic=False)
    emit_cpp(graph, paths, recipe_dir=tmp_path / "recipe", **options)
    records, _, _ = RecipeReader(tmp_path / "recipe/graph.recipe").read()
    cpp = emit_cpp(graph, paths, **options)
    declarations = [(json.loads(name), json.loads(kind)) for name, kind in re.findall(
        r'ops\.push_back\(Op\(("[^"\n]+"), ("[^"\n]+"),', cpp)]
    assert declarations == [(record["name"], record["type"]) for record in records]
    reshapes = [item for item in records if item["type"] == "Reshape"]
    normalized_batches = sum(node.kind == "matmul" and min(len(x.shape) for x in node.inputs) > 2
                             and max(len(x.shape) for x in node.inputs) > 3 for node in graph.nodes)
    assert len(reshapes) == [739, 64][graph_index] + 3 * normalized_batches
    for record in reshapes:
        _check_reshape(records, record)
    if graph_index == 0:
        by_name = {record["name"]: record for record in records}
        assert by_name["v974"]["outputs"][0]["shape"] == (1, 2, 1, 55)
        assert by_name["v975"]["outputs"][0]["shape"] == (1, 2, 1, 55, 1)
    controls = {"Slice": {"offsets", "size"}, "ReduceSum": {"axes"},
                "ReduceMean": {"axes"}, "Transpose": {"perm"}, "BroadcastTo": {"shape"}}
    for record in records:
        for port in record["inputs"]:
            if port["port"] in controls.get(record["type"], set()):
                assert port["dtype"] == "int64", (record["name"], port)


def test_full_model_reshape_dimensions_fit_int32_without_weight_loading():
    for graph in build_pair(*configurations(tiny=False), 8):
        for node in graph.nodes:
            if node.kind == "reshape":
                assert all(0 <= value <= (1 << 31) - 1 for value in node.attrs["shape"])
            elif node.kind == "matmul" and len(node.inputs[0].shape) > 2 and len(node.inputs[1].shape) == 2:
                assert np.prod(node.inputs[0].shape[:-1], dtype=np.int64) <= (1 << 31) - 1
