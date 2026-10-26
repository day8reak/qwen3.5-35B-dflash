"""Contradictory dtype metadata must never reach the AIR constructor."""
import pytest

from qwen35_moe_dflash.ge_cpp import emit_cpp
from qwen35_moe_dflash.ge_ir import Graph
from qwen35_moe_dflash.ge_recipe import emit_recipe_cpp


@pytest.mark.parametrize("recipe", [False, True])
@pytest.mark.parametrize("kind,input_dtype,output_dtype,attrs", [
    ("neg", "float16", "int32", {}),
    ("add", "float16", "int32", {}),
    ("eq", "int32", "int32", {}),
    ("logical_and", "int32", "bool", {}),
    ("reshape", "float16", "int32", {"shape": [2, 2]}),
    ("transpose", "float16", "int32", {"perm": [1, 0]}),
    ("slice", "float16", "int32", {"starts": [0, 0], "sizes": [2, 2]}),
    ("concat", "float16", "int32", {"axis": 0}),
    ("reduce_sum", "float16", "int32", {"axes": [1], "keepdims": True}),
])
def test_invalid_model_dtype_relationship_fails_for_both_exporters(tmp_path, recipe, kind, input_dtype, output_dtype, attrs):
    graph = Graph("wrong_dtype")
    x = graph.input("x", (2, 2), input_dtype)
    ins = [x, x] if kind in {"add", "eq", "logical_and", "concat"} else [x]
    shape = (4, 2) if kind == "concat" else (2, 1) if kind == "reduce_sum" else (2, 2)
    graph.outputs = [graph.op(kind, *ins, shape=shape, dtype=output_dtype, attrs=attrs)]
    output = tmp_path / "recipe"
    with pytest.raises(ValueError, match="dtype|bool"):
        emit_cpp(graph, **({"recipe_dir": output} if recipe else {}))
    assert not output.exists()


def _records(typ, input_dtype, output_dtype, attr):
    inputs = [{"port": "x", "shape": [2, 2], "dtype": input_dtype, "src": 0, "src_port": "y"}]
    records = [{"name": "data", "type": "Data", "inputs": [],
                "outputs": [{"port": "y", "shape": [2, 2], "dtype": input_dtype}],
                "attrs": {"index": {"type": "int", "value": 0}}}]
    if typ == "ArgMaxV2":
        records.append({"name": "axis", "type": "Const", "inputs": [],
                        "outputs": [{"port": "y", "shape": [], "dtype": "int32"}],
                        "constant": {"data": b"\x01\x00\x00\x00"}})
        inputs.append({"port": "dimension", "shape": [], "dtype": "int32", "src": 1, "src_port": "y"})
    records.append({"name": "convert", "type": typ, "inputs": inputs,
                    "outputs": [{"port": "y", "shape": [2] if typ == "ArgMaxV2" else [2, 2], "dtype": output_dtype}],
                    "attrs": attr})
    return records


@pytest.mark.parametrize("typ,attrs", [
    ("Cast", {}),
    ("Cast", {"dst_type": {"type": "int", "value": 1}}),
    # Q35GE001 INT32 wire tag is 2; public GE DT_INT32 is 3.
    ("Cast", {"dst_type": {"type": "int", "value": 2}}),
    ("Cast", {"dst_type": {"type": "dtype", "value": "int32"}}),
    ("ArgMaxV2", {"dtype": {"type": "dtype", "value": "float16"}}),
    ("ArgMaxV2", {"dtype": {"type": "int", "value": 3}}),
])
def test_dtype_attribute_must_match_output_descriptor_before_writing(tmp_path, typ, attrs):
    records = _records(typ, "float16", "int32", attrs)
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="dst_type|ArgMax dtype"):
        emit_recipe_cpp(records, "invalid", [0], [(len(records) - 1, 0)], output)
    assert not output.exists()


@pytest.mark.parametrize("dtype,enum", [("float16", 1), ("int32", 3), ("int64", 9), ("bool", 12)])
def test_all_supported_cast_targets_keep_integer_attribute_encoding(tmp_path, dtype, enum):
    records = _records("Cast", "int32", dtype, {"dst_type": {"type": "int", "value": enum}})
    emit_recipe_cpp(records, "cast", [0], [(1, 0)], tmp_path / "valid")
    assert (tmp_path / "valid/graph.recipe").is_file()


@pytest.mark.parametrize("recipe", [False, True])
@pytest.mark.parametrize("kind", ["argmax", "topk_values", "topk_indices"])
def test_selection_rejects_bool_source_with_otherwise_correct_output(tmp_path, recipe, kind):
    graph = Graph("boolean_selection")
    source = graph.input("mask", (2, 2), "bool")
    graph.outputs = [graph.op(kind, source, shape=(2,) if kind == "argmax" else (2, 1),
                              dtype="bool" if kind == "topk_values" else "int32",
                              attrs={"axis": -1, "k": 1})]
    with pytest.raises(ValueError, match="does not accept bool data"):
        emit_cpp(graph, recipe_dir=tmp_path / "invalid" if recipe else None)
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("policy", ["native", "matmul"])
def test_reduction_keepdims_does_not_treat_false_string_as_true(tmp_path, policy):
    graph = Graph("reduction_flag")
    source = graph.input("x", (2, 2))
    graph.outputs = [graph.op("reduce_sum", source, shape=(2, 1),
                              attrs={"axes": [1], "keepdims": "false"})]
    with pytest.raises(ValueError, match="keepdims must be boolean"):
        emit_cpp(graph, fp16_reductions=policy, recipe_dir=tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
