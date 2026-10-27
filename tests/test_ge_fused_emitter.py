"""Stateful GE structural/serialization checks, without Torch, GE, or a board."""
import copy
import shutil
import subprocess

import numpy as np
import pytest

from qwen35_moe_dflash.ge_fused.emitter import emit_cpp
from qwen35_moe_dflash.ge_fused.ir import Graph
from qwen35_moe_dflash.ge_fused.recipe import read_recipe
from qwen35_moe_dflash.ge_ir import Graph as LegacyGraph, DTYPES as LEGACY_DTYPES
from qwen35_moe_dflash.ge_recipe_read import read_recipe as read_legacy_recipe


def state_graph(*, state_only=False):
    graph = Graph("stateful")
    x = graph.input("query", (1, 3, 2, 8))
    state = graph.input("state", (1, 2, 8, 8), "float32")
    length = graph.const([3], dtype="int16", name="valid_length")
    outputs = graph.custom("BoundDeltaRule", inputs={"query": x, "state": state, "length": length},
                           outputs={"core": (x.shape, "float16"), "next_state": (state.shape, "float32")},
                           attrs={"chunk_size": {"type": "int", "value": 64},
                                  "state_type": {"type": "dtype", "value": "float32"},
                                  "enabled": {"type": "bool", "value": True},
                                  "scale": {"type": "float", "value": 0.125},
                                  "axes": {"type": "ints", "value": [0, 1]},
                                  "layout": {"type": "string", "value": "BSHD"}},
                           name="layer0_delta")
    state_copy = graph.op("reshape", outputs["next_state"], shape=state.shape, dtype="float32",
                          attrs={"shape": list(state.shape)}, name="state_output")
    graph.outputs = [state_copy] if state_only else [outputs["core"], outputs["next_state"], state_copy]
    graph.state_abi = {"recurrent": "explicit", "input": "state", "output": state_copy.name}
    return graph


def test_custom_core_and_state_ports_remain_connected(tmp_path):
    graph = state_graph()
    emit_cpp(graph, recipe_dir=tmp_path / "recipe", fp16_row_mul="compact")
    recipe = read_recipe(tmp_path / "recipe/graph.recipe")
    records = recipe["records"]
    index = next(i for i, record in enumerate(records) if record["name"] == "layer0_delta")
    custom = records[index]
    assert [(p["port"], p["dtype"]) for p in custom["outputs"]] == [
        ("core", "float16"), ("next_state", "float32")]
    consumer = next(record for record in records if record["name"] == "state_output")
    assert consumer["inputs"][0]["src"] == index
    assert consumer["inputs"][0]["src_port"] == "next_state"
    assert recipe["outputs"][:2] == [[index, 0], [index, 1]]
    assert custom["attrs"]["state_type"] == {"type": "dtype", "value": "float32"}
    assert graph.manifest()["state_abi"] == graph.state_abi
    assert graph.manifest()["custom_ops"][0]["type"] == "BoundDeltaRule"


def test_secondary_state_only_keeps_custom_and_all_input_dependencies(tmp_path):
    graph = state_graph(state_only=True)
    emit_cpp(graph, recipe_dir=tmp_path / "recipe")
    records = read_recipe(tmp_path / "recipe/graph.recipe")["records"]
    names = {record["name"] for record in records}
    assert {"layer0_delta", "query", "state", "valid_length", "state_output"} <= names
    assert next(r for r in records if r["name"] == "valid_length")["outputs"][0]["dtype"] == "int16"
    assert sum(r["type"] == "BoundDeltaRule" for r in records) == 1


def test_opaque_constant_custom_is_never_host_folded(tmp_path):
    graph = Graph("opaque")
    graph.input("unused_boundary", (1,))
    x = graph.const([2.0], name="constant_input")
    value = graph.op("scalar_mul", x, shape=(1,), attrs={"value": 0.5}, name="foldable")
    out = graph.custom("Opaque", {"x": value}, {"y": ((1,), "float16"), "state": ((1,), "float32")},
                       name="opaque")
    graph.outputs = [out["state"]]
    emit_cpp(graph, recipe_dir=tmp_path / "recipe")
    records = read_recipe(tmp_path / "recipe/graph.recipe")["records"]
    assert next(r for r in records if r["name"] == "foldable")["type"] == "Const"
    assert next(r for r in records if r["name"] == "opaque")["type"] == "Opaque"
    assert read_recipe(tmp_path / "recipe/graph.recipe")["outputs"][-1][1] == 1


def test_local_fp32_int16_constants_casts_and_inline_descriptors(tmp_path):
    graph = Graph("types")
    state = graph.input("state", (1, 2), "float32")
    bias = graph.const([[0.25, 0.5]], dtype="float32", name="state_bias")
    added = graph.op("add", state, bias, shape=state.shape, dtype="float32")
    half = graph.op("cast", added, shape=state.shape, dtype="float16")
    full = graph.op("cast", half, shape=state.shape, dtype="float32")
    length = graph.op("cast", full, shape=state.shape, dtype="int16")
    graph.outputs = [full, length]
    code = emit_cpp(graph)
    assert "ge::DT_FLOAT" in code and "ge::DT_INT16" in code
    emit_cpp(graph, recipe_dir=tmp_path / "recipe")
    records = read_recipe(tmp_path / "recipe/graph.recipe")["records"]
    constant = next(r for r in records if r["name"] == "state_bias")
    assert constant["constant"]["size"] == 8
    blob = (tmp_path / "recipe/constants.bin").read_bytes()
    assert blob == np.asarray([[0.25, 0.5]], dtype="<f4").tobytes()
    casts = [r["attrs"]["dst_type"]["value"] for r in records if r["type"] == "Cast"]
    assert casts == [1, 0, 6]


@pytest.mark.parametrize("fault", ["foreign", "untyped_attr", "nan_attr", "bad_dtype", "duplicate_name"])
def test_custom_graph_rejects_invalid_bindings_without_partial_nodes(fault):
    graph = Graph("contract")
    x = graph.input("input", (1,))
    inputs, outputs, attrs, name = {"x": x}, {"y": ((1,), "float16")}, {}, "custom_op"
    if fault == "foreign":
        inputs["x"] = Graph("foreign").input("input", (1,))
    elif fault == "untyped_attr":
        attrs["chunk_size"] = 64
    elif fault == "nan_attr":
        attrs["epsilon"] = {"type": "float", "value": float("nan")}
    elif fault == "bad_dtype":
        outputs["later_port"] = ((1,), "float64")
    else:
        name = "input"
    before = copy.copy(graph._values)
    with pytest.raises((ValueError, TypeError)):
        graph.custom("Opaque", inputs, outputs, attrs, name)
    assert graph._values == before and not graph.nodes


def test_legacy_dtype_registry_and_wire_format_stay_unchanged(tmp_path):
    assert "float32" not in LEGACY_DTYPES and "int16" not in LEGACY_DTYPES
    with pytest.raises(ValueError):
        LegacyGraph("old").input("state", (1,), "float32")
    emit_cpp(state_graph(), recipe_dir=tmp_path / "recipe")
    payload = tmp_path / "recipe/graph.recipe"
    assert payload.read_bytes()[:8] == b"Q35GF001"
    with pytest.raises(ValueError, match="unknown recipe"):
        read_legacy_recipe(payload)


def test_cpp_parser_reads_extended_descriptors_attributes_and_constants(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    folder = tmp_path / "recipe"
    source = tmp_path / "reader.cpp"
    source.write_text(emit_cpp(state_graph(), recipe_dir=folder))
    binary = tmp_path / "reader"
    subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                    "-DQ35_RECIPE_PARSE_ONLY", str(source), "-o", str(binary)],
                   check=True, capture_output=True, text=True)
    result = subprocess.run([str(binary), str(folder / "graph.recipe")],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("RECIPE_VALID ") and "AIR_EXPORTED" not in result.stdout
    invalid = tmp_path / "old_magic.recipe"
    invalid.write_bytes(b"Q35GE001" + (folder / "graph.recipe").read_bytes()[8:])
    result = subprocess.run([str(binary), str(invalid)], capture_output=True, text=True, timeout=15)
    assert result.returncode != 0 and "unknown recipe format" in result.stderr
