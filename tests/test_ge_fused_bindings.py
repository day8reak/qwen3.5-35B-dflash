"""Vendor binding contracts: explicit ports, types, order and no silent fallback."""
import hashlib
import json

import pytest

from qwen35_moe_dflash.ge_fused.bindings import OperatorBindings
from qwen35_moe_dflash.ge_fused.ir import Graph
from qwen35_moe_dflash.ge_fused.emitter import emit_cpp
from qwen35_moe_dflash.ge_fused.recipe import read_recipe


def document():
    # Deliberately synthetic ports, reverse output order and INT flag encoding.
    # This fixture is never a claim about the yellow-zone operator schema.
    return {"schema_version": 1, "target": "test-only", "ddk_id": "synthetic",
            "source": {"path": "test.py", "sha256": "0" * 64}, "operators": {
        "gdn": {"op_type": "TestDelta", "layout": "ND",
                "inputs": {logical: {"port": "in_" + logical, "dtype": dtype}
                           for logical, dtype in [("query", "float16"), ("key", "float16"),
                                                  ("value", "float16"), ("g", "float16"),
                                                  ("beta", "float16"), ("initial_state", "float32"),
                                                  ("effective_length", "int16")]},
                "outputs": {"final_state": {"port": "state_out", "dtype": "float32"},
                            "core_attn": {"port": "core_out", "dtype": "float16"}},
                "attrs": {logical: {"name": "attr_" + logical, "type": "int"}
                          for logical in ("chunk_size", "output_final_state", "use_qk_l2norm_in_kernel")}}}}


def arguments(graph):
    shapes = {"query": (1, 2, 2, 8), "key": (1, 2, 2, 8), "value": (1, 2, 2, 8),
              "g": (1, 2, 2), "beta": (1, 2, 2), "initial_state": (1, 2, 8, 8),
              "effective_length": (1,)}
    inputs = {key: graph.input(key, shape, "float32" if key == "initial_state" else
                              "int16" if key == "effective_length" else "float16")
              for key, shape in shapes.items()}
    return {"inputs": inputs, "outputs": {"core_attn": ((1, 2, 2, 8), "float16"),
                                           "final_state": ((1, 2, 8, 8), "float32")},
            "attrs": {"chunk_size": 64, "output_final_state": True,
                      "use_qk_l2norm_in_kernel": True}}


def test_custom_output_ordinal_follows_binding_not_logical_call_order(tmp_path):
    bindings = OperatorBindings(document())
    graph = Graph("test_bound_ports")
    out = bindings.call(graph, "gdn", **arguments(graph))
    graph.outputs = [out["core_attn"], out["final_state"]]
    node = graph.nodes[-1]
    assert node.output_ports == ("state_out", "core_out")
    assert node.attrs["attr_output_final_state"] == {"type": "int", "value": 1}
    emit_cpp(graph, {}, recipe_dir=tmp_path / "recipe")
    recipe = read_recipe(tmp_path / "recipe" / "graph.recipe")
    assert [port for _, port in recipe["outputs"]] == [1, 0]


def test_file_identity_and_defensive_copy(tmp_path):
    source = document()
    path = tmp_path / "binding.json"
    path.write_text(json.dumps(source))
    bound = OperatorBindings.from_file(path)
    assert bound.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    source["operators"]["gdn"]["op_type"] = "Changed"
    view = bound.manifest()
    view["operators"].clear()
    assert bound.manifest()["operators"]["gdn"]["op_type"] == "TestDelta"


@pytest.mark.parametrize("corruption", ["missing_port", "duplicate_port", "layout", "attr_type",
                                         "static_override", "unknown_role", "source_hash", "shape"])
def test_invalid_schema_rejected(corruption):
    doc = document()
    op = doc["operators"]["gdn"]
    if corruption == "missing_port":
        del op["inputs"]["initial_state"]
    elif corruption == "duplicate_port":
        op["inputs"]["key"]["port"] = op["inputs"]["query"]["port"]
    elif corruption == "layout":
        op["layout"] = "NZ"
    elif corruption == "attr_type":
        op["attrs"]["chunk_size"]["type"] = "string"
    elif corruption == "static_override":
        op["static_attrs"] = {"attr_chunk_size": {"type": "int", "value": 1}}
    elif corruption == "unknown_role":
        doc["operators"]["attention"] = op
    elif corruption == "source_hash":
        doc["source"]["sha256"] = "unverified"
    elif corruption == "shape":
        op["inputs"]["query"]["shape"] = [True, 2, 2, 8]
    with pytest.raises(ValueError):
        OperatorBindings(doc)


def test_duplicate_json_field_rejected(tmp_path):
    path = tmp_path / "binding.json"
    path.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ValueError, match="duplicate"):
        OperatorBindings.from_file(path)


def test_no_missing_role_fallback_or_implicit_dtype_cast():
    bindings = OperatorBindings(document())
    with pytest.raises(ValueError, match="missing enabled"):
        bindings.require(["gdn", "rms"])
    graph = Graph("bad_dtype")
    args = arguments(graph)
    args["outputs"]["final_state"] = ((1, 2, 8, 8), "float16")
    with pytest.raises(ValueError, match="no implicit cast"):
        bindings.call(graph, "gdn", **args)
    assert not graph.nodes


def test_shape_constraints_reject_unsupported_head_shape():
    doc = document()
    doc["operators"]["gdn"]["inputs"]["query"]["shape"] = [1, None, 32, 128]
    graph = Graph("bad_shape")
    with pytest.raises(ValueError, match="shape"):
        OperatorBindings(doc).call(graph, "gdn", **arguments(graph))
    assert not graph.nodes


@pytest.mark.parametrize("width,size", [("kernel", 4), ("kernel_minus_one", 3)])
@pytest.mark.parametrize("policy", ["last_kernel_minus_one", "padded_tail"])
def test_conv_state_requires_explicit_history_convention(width, size, policy):
    doc = document()
    doc["operators"] = {"conv_state": {
        "op_type": "TestConvState", "layout": "ND", "attrs": {},
        "inputs": {n: {"port": n, "dtype": "int32" if n == "update_control" else "float16"}
                   for n in ("projected", "history", "update_control")},
        "outputs": {n: {"port": n, "dtype": "float16"} for n in ("history", "padded")},
        "history_output_width": width, "history_keep": policy}}
    bound = OperatorBindings(doc)
    assert bound.conv_history_size(4) == size
    assert bound.conv_history_slice == policy
    del doc["operators"]["conv_state"]["history_keep"]
    with pytest.raises(ValueError, match="explicit"):
        OperatorBindings(doc)
