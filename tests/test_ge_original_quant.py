"""Host-only original-W4 graph tests; synthetic schemas are not DDK evidence."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from qwen35_moe_dflash.config import Qwen35DFlashConfig
from qwen35_moe_dflash.ge_export import configurations
from qwen35_moe_dflash.ge_eval import evaluate
from qwen35_moe_dflash.ge_fused.ir import Graph, CustomNode, DTYPES
from qwen35_moe_dflash.ge_fused.quant_bindings import QuantOperatorBindings
from qwen35_moe_dflash.ge_fused.quant_target import (
    _QuantBuilder, build_quantized_target, emit_quantized_cpp,
)
from qwen35_moe_dflash.ge_fused.recipe import read_recipe
from qwen35_moe_dflash.ge_fused.target import build_target
from qwen35_moe_dflash.original_quant import canonicalkeys


def contract_document():
    def port(name, dtype):
        return {"port": name, "dtype": dtype}

    def schema(name, kind="required", **extra):
        return {"name": name, "kind": kind, **extra}

    operators = {
        "dynamic_quant": {
            "op_type": "TestDynamicQuant", "descriptor_format": "ND",
            "inputs": {"x": port("features", "float16")},
            "outputs": {"quantized": port("quantized", "int8"), "scale": port("scale", "float32")},
            "input_schema": [schema("features"), schema("smoothing", "optional")],
            "output_schema": [schema("quantized"), schema("scale")], "attrs": {},
        },
        "quant_matmul": {
            "op_type": "TestQuantMatmul", "descriptor_format": "ND",
            "inputs": {"x": port("activation", "int8"), "weight": port("packed", "int32"),
                       "scale": port("groups", "float32"), "token_scale": port("tokens", "float32")},
            "outputs": {"y": port("result", "float16")},
            "input_schema": [schema("activation"), schema("packed"), schema("groups"),
                             schema("offset", "optional"), schema("tokens"), schema("bias", "optional")],
            "output_schema": [schema("result")], "attrs": {},
            "static_attrs": {"output_type": {"type": "dtype", "value": "float16"}},
            "weight_encoding": "I4ZN_INT32", "group_size": 256, "scale_encoding": "FP32_GROUP_SCALE",
        },
        "grouped_matmul": {
            "op_type": "TestGroupedMatmul", "descriptor_format": "ND",
            "inputs": {"x": port("x0", "int8"), "weight": port("w0", "int32"),
                       "scale": port("qparam0", "int64"), "token_scale": port("token_scale0", "float32"),
                       "group_list": port("counts", "int64")},
            "outputs": {"y": port("y0", "float16")},
            "input_schema": [schema("x", "dynamic", count=1), schema("w", "dynamic", count=1),
                             schema("bias", "dynamic", count=0), schema("qparam", "dynamic", count=1),
                             schema("offset", "optional"), schema("token_scale", "dynamic", count=1), schema("counts")],
            "output_schema": [schema("y", "dynamic", count=1)],
            "attrs": {name: {"name": "test_" + name, "type": "int"}
                      for name in ("group_list_type", "split_item", "group_type", "act_type")},
            "weight_encoding": "I4ZN_INT32", "group_size": 256, "scale_encoding": "VENDOR_QPARAM_INT64",
        },
    }
    return {"schema_version": 1, "target": "kirinx90", "ddk_id": "SYNTHETIC_NOT_INSTALLED",
            "source": {"path": "synthetic_test_schema", "sha256": "0" * 64}, "operators": operators}


def descriptor(key, shape, pending=False):
    expert = len(shape) == 3
    n, k = shape[-2:]
    return {"key": key, "logical_shape": list(shape), "kind": "expert" if expert else "linear",
            "packed_shape": [*shape[:-2], k, n // 8], "packed_dtype": "int32",
            "scale_shape": [*shape[:-2], k // 256, n], "scale_dtype": "int64" if expert and not pending else "float32",
            "group_size": 256, "scale_encoding": ("QPARAM_REQUIRED" if pending else "QPARAM_VENDOR_CONVERTED")
            if expert else "FP32_GROUP_SCALE"}


@pytest.fixture(scope="module")
def small_config():
    config, draft = configurations(tiny=True)
    config.update(hidden_size=256, vocab_size=256, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=1, head_dim=64,
                  linear_num_key_heads=16, linear_num_value_heads=32,
                  linear_key_head_dim=8, linear_value_head_dim=8,
                  num_experts=16, num_experts_per_tok=2,
                  moe_intermediate_size=256, shared_expert_intermediate_size=256,
                  layer_types=["linear_attention", "full_attention"])
    draft.update(hidden_size=256, vocab_size=256, num_target_layers=2)
    draft["dflash_config"]["target_layer_ids"] = [0, 1]
    return config, Qwen35DFlashConfig.from_dict(draft)


def descriptors_for(config, draft, pending=False):
    graph = build_target(config, draft, 1, past_length=1, expert_dispatch="selected")
    result = {}
    for key, value in graph.weights.items():
        try:
            canonicalkeys(key)
        except ValueError:
            continue
        result[key] = descriptor(key, value.shape, pending)
    return result


def test_external_embedding_keeps_original_int8_fp32_arithmetic(monkeypatch):
    # The legacy oracle intentionally exposes only legacy dtypes. Extend its
    # local test table without changing the production legacy vocabulary.
    monkeypatch.setattr("qwen35_moe_dflash.ge_eval.DTYPES", DTYPES)
    graph = Graph("embedding")
    builder = _QuantBuilder(graph, (), None, quant_bindings=QuantOperatorBindings(contract_document()), quant_descriptors={})
    graph.outputs = [builder.input_embedding(1, {"hidden_size": 4})]
    values = np.asarray([[[-128, -7, 3, 127]]], dtype=np.int8)
    scale = np.asarray([[[0.0200123]]], dtype=np.float32)
    actual = evaluate(graph, {"inputs_embeds": values, "embedding_scale": scale})[0]
    np.testing.assert_array_equal(actual, values.astype(np.float32) * scale)
    assert actual.dtype == np.float32
    assert [value.dtype for value in graph.inputs] == ["int8", "float32"]
    assert not graph.weights


@pytest.mark.parametrize("sequence,past", [(1, 64), (64, 0)])
def test_selected_quantized_model_shape_and_precision_contract(small_config, sequence, past):
    config, draft = small_config
    metadata = descriptors_for(config, draft)
    graph = build_quantized_target(config, draft, sequence, past_length=past,
        quant_bindings=QuantOperatorBindings(contract_document()), quant_descriptors=metadata)
    assert [value.name for value in graph.inputs[:2]] == ["inputs_embeds", "embedding_scale"]
    assert "target.embed_tokens.weight" not in graph.weights
    assert graph.state_abi["sequence_length"] == sequence
    assert [value.name for value in graph.outputs[:2]] == ["target_top1", "target_features"]
    custom = [node for node in graph.nodes if isinstance(node, CustomNode)]
    grouped = [node for node in custom if node.op_type == "TestGroupedMatmul"]
    assert len(grouped) == config["num_hidden_layers"] * sequence * 2
    assert all(node.inputs[0].shape[0] == config["num_experts_per_tok"] for node in grouped)
    assert all(node.inputs[1].dtype == "int32" and node.inputs[2].dtype == "int64" for node in grouped)
    assert all(node.inputs[1].shape[0] == config["num_experts_per_tok"] for node in grouped)
    assert all(node.inputs[1].shape[0] < config["num_experts"] for node in grouped)
    assert all(node.attrs["test_group_list_type"] == {"type": "int", "value": 1} for node in grouped)
    assert all(np.all(graph.constants[node.inputs[-1].name] == 1) for node in grouped)
    for key, weight in graph.weights.items():
        if key.endswith(".quant_scale"):
            assert weight.dtype == ("int64" if ".experts." in key else "float32")
        elif key in metadata:
            assert weight.dtype == "int32"
        else:
            assert weight.dtype == "float16"
    assert graph.weights["target.layers.0.linear_attn.conv1d.weight"].shape[-1] == 4
    assert graph.weights["target.layers.0.mlp.shared_expert_gate.weight"].dtype == "float16"
    assert not {"AdnRmsNorm", "LinearAttentionConvstateUpdate"} & {node.op_type for node in custom}
    assert graph.manifest()["quantization_contract"]["pending_expert_qparam"] == []


def test_recipe_preserves_int8_secondary_outputs_and_noncompact_ports(small_config, tmp_path):
    config, draft = small_config
    graph = build_quantized_target(config, draft, 1, past_length=64,
        quant_bindings=QuantOperatorBindings(contract_document()), quant_descriptors=descriptors_for(config, draft))
    paths = {key: tmp_path / "unbound" / (key + ".bin") for key in graph.weights}
    code = emit_quantized_cpp(graph, paths, recipe_dir=tmp_path / "recipe")
    recipe = read_recipe(tmp_path / "recipe/graph.recipe")
    assert 'DynamicInputRegister("bias", 0)' in code
    assert 'DynamicOutputRegister("y", 1)' in code
    assert 'OptionalInputRegister("offset")' in code
    assert '"activation","packed","groups","offset","tokens","bias"' in code
    assert "case 7: return ge::DT_INT8;" in code
    records = recipe["records"]
    dq_indices = {i for i, row in enumerate(records) if row["type"] == "TestDynamicQuant"}
    assert dq_indices
    for index in dq_indices:
        assert [out["dtype"] for out in records[index]["outputs"]] == ["int8", "float32"]
        consumers = [port for row in records for port in row["inputs"] if port["src"] == index]
        assert {port["src_port"] for port in consumers} == {"quantized", "scale"}
    for row in records:
        if row["type"] == "TestQuantMatmul":
            assert [port["port"] for port in row["inputs"]] == ["activation", "packed", "groups", "tokens"]
        if row["type"] == "FileConstant":
            assert row["attrs"]["dtype"]["value"] == row["outputs"][0]["dtype"]


def test_pending_vendor_qparam_cannot_be_exported(small_config, tmp_path):
    config, draft = small_config
    graph = build_quantized_target(config, draft, 1,
        quant_bindings=QuantOperatorBindings(contract_document()), quant_descriptors=descriptors_for(config, draft, True))
    paths = {key: tmp_path / (key + ".bin") for key in graph.weights}
    with pytest.raises(ValueError, match="vendor converter"):
        emit_quantized_cpp(graph, paths, recipe_dir=tmp_path / "recipe")
    assert not (tmp_path / "recipe").exists()


@pytest.mark.parametrize("mutation,match", [
    (lambda doc: doc["operators"]["quant_matmul"].pop("input_schema"), "complete ordered"),
    (lambda doc: doc["operators"]["quant_matmul"]["inputs"]["scale"].update(dtype="int64"), "float32"),
    (lambda doc: doc["operators"]["grouped_matmul"].update(scale_encoding="FP32_GROUP_SCALE"), "QPARAM"),
    (lambda doc: doc["operators"]["grouped_matmul"]["attrs"].pop("group_list_type"), "attribute mappings"),
    (lambda doc: doc["operators"]["quant_matmul"]["inputs"]["token_scale"].update(port="tokens_missing"), "complete schema"),
])
def test_incomplete_or_guessed_bindings_rejected(mutation, match):
    doc = contract_document()
    mutation(doc)
    with pytest.raises(ValueError, match=match):
        QuantOperatorBindings(doc)


def test_missing_quant_weight_has_no_float_or_requantization_fallback(small_config):
    config, draft = small_config
    metadata = descriptors_for(config, draft)
    del metadata["target.layers.0.linear_attn.in_proj_qkv.weight"]
    with pytest.raises(ValueError, match="metadata is missing"):
        build_quantized_target(config, draft, 1, quant_bindings=QuantOperatorBindings(contract_document()), quant_descriptors=metadata)


@pytest.mark.parametrize("sequence,past", [(1, 64), (64, 0)])
def test_complete_locked_40_layer_graph_uses_only_selected_experts(sequence, past):
    config, draft = configurations(tiny=False)
    draft = Qwen35DFlashConfig.from_dict(draft)
    metadata = descriptors_for(config, draft)
    graph = build_quantized_target(config, draft, sequence, past_length=past,
        quant_bindings=QuantOperatorBindings(contract_document()), quant_descriptors=metadata)
    grouped = [node for node in graph.nodes if isinstance(node, CustomNode) and node.op_type == "TestGroupedMatmul"]
    assert len(grouped) == 80 * sequence and len(graph.state_abi["states"]) == 80
    assert all(node.inputs[0].shape[0] == 8 for node in grouped)
    assert graph.quantization_contract["device_execution"] == "NOT_RUN"


def test_yaml_cache_capacity_bounds_static_graph_before_construction(small_config):
    config, draft = small_config
    limited = dict(config, kv_cache_max_len=64, max_position_embeddings=262144)
    metadata = descriptors_for(config, draft)
    bindings = QuantOperatorBindings(contract_document())
    with pytest.raises(ValueError, match="kv_cache_max_len"):
        build_quantized_target(limited, draft, 1, past_length=64,
            quant_bindings=bindings, quant_descriptors=metadata)
    graph = build_quantized_target(limited, draft, 1, past_length=63,
        quant_bindings=bindings, quant_descriptors=metadata)
    assert graph.state_abi["next_past_length"] == 64
    assert graph.quantization_contract["kv_cache_max_len"] == 64


def _precision_builder(graph):
    return _QuantBuilder(graph, (), None,
                         quant_bindings=QuantOperatorBindings(contract_document()), quant_descriptors={})


def test_decay_uses_source_fp32_equation_before_declared_boundary(monkeypatch, tmp_path):
    from qwen35_moe_dflash.ge_fused.emitter import emit_cpp
    monkeypatch.setattr("qwen35_moe_dflash.ge_eval.DTYPES", DTYPES)
    graph = Graph("source_decay")
    builder = _precision_builder(graph)
    activation = graph.input("a", (1, 8, 32))
    graph.outputs = [builder.gdn_decay(activation, "linear", 32)]
    rng = np.random.default_rng(42)
    a = rng.normal(0, 2, activation.shape).astype(np.float16)
    bias = rng.normal(0, 1, 32).astype(np.float16)
    log_rate = rng.normal(0, 1, 32).astype(np.float16)
    actual = evaluate(graph, {"a": a}, {"linear.dt_bias": bias, "linear.A_log": log_rate})[0]
    expected = (-np.exp(log_rate.astype(np.float32))
                * np.logaddexp(np.float32(0), a.astype(np.float32) + bias.astype(np.float32))).astype(np.float16)
    np.testing.assert_array_equal(actual, expected)
    old = (-np.exp(log_rate.astype(np.float32)).astype(np.float16)
           * np.logaddexp(np.float32(0), (a + bias).astype(np.float16).astype(np.float32)).astype(np.float16)).astype(np.float16)
    assert np.any(old != actual), "fixture must expose premature FP16 rounding"
    assert builder.precision_casts[-1]["from_dtype"] == "float32"
    emit_cpp(graph, {key: tmp_path / (key + ".bin") for key in graph.weights}, recipe_dir=tmp_path / "recipe")
    softplus = next(row for row in read_recipe(tmp_path / "recipe/graph.recipe")["records"] if row["type"] == "Softplus")
    assert softplus["outputs"][0]["dtype"] == "float32"


def test_gated_norm_rounds_only_after_fp32_silu_product(monkeypatch):
    monkeypatch.setattr("qwen35_moe_dflash.ge_eval.DTYPES", DTYPES)
    graph = Graph("source_gated_norm")
    builder = _precision_builder(graph)
    core = graph.input("core", (1, 2, 2, 32))
    gate = graph.input("gate", core.shape)
    graph.outputs = [builder.gated_norm(core, gate, "gated.weight", 1e-6)]
    rng = np.random.default_rng(71)
    x, z = [rng.normal(0, 1, core.shape).astype(np.float16) for _ in range(2)]
    gamma = rng.uniform(.3, 1.4, 32).astype(np.float16)
    actual = evaluate(graph, {"core": x, "gate": z}, {"gated.weight": gamma})[0]
    xf, zf = x.astype(np.float32), z.astype(np.float32)
    normalized = xf * (1 / np.sqrt(np.mean(xf * xf, axis=-1, keepdims=True) + np.float32(1e-6)))
    expected = (normalized * gamma.astype(np.float32) * (zf / (1 + np.exp(-zf)))).astype(np.float16)
    np.testing.assert_array_equal(actual, expected)
    assert next(node for node in graph.nodes if node.kind == "silu").output.dtype == "float32"


def test_rotary_uses_exact_positions_and_fp32_angles_before_half_cast(monkeypatch):
    monkeypatch.setattr("qwen35_moe_dflash.ge_eval.DTYPES", DTYPES)
    graph = Graph("source_rope")
    graph.input("unused", (1,))
    builder = _precision_builder(graph)
    graph.outputs = list(builder.rotary_tables({"rope_theta": 1000000., "rotary_dim": 8}, 4, 2049))
    actual = evaluate(graph, {"unused": np.zeros(1, dtype=np.float16)})
    inv = 1 / (np.float32(1000000.) ** (np.arange(0, 8, 2, dtype=np.float32) / 8))
    angles = np.arange(2049, 2053, dtype=np.float32)[:, None] * inv[None, :]
    angles = np.concatenate([angles, angles], axis=-1).reshape(1, 1, 4, 8)
    for output, function in zip(actual, (np.cos, np.sin)):
        np.testing.assert_array_equal(output, function(angles).astype(np.float16))
    assert actual[0][0, 0, 0, 0] != actual[0][0, 0, 1, 0]


def test_quantized_head_selects_rows_and_appends_full_logits(small_config):
    config, draft = small_config
    graph = build_quantized_target(config, draft, 64,
        quant_bindings=QuantOperatorBindings(contract_document()), quant_descriptors=descriptors_for(config, draft),
        output_positions=(-1, 0, 20), return_logits=True)
    assert graph.outputs[0].shape == (1, 3)
    assert graph.outputs[1].shape == (1, 64, 512) and graph.outputs[1].dtype == "float16"
    assert graph.outputs[-1].name == "target_logits" and graph.outputs[-1].shape == (1, 3, 256)
    assert graph.outputs[-1].dtype == "float32"
    assert [v.name for v in graph.outputs[2:-1]] == [row["output"] for row in graph.state_abi["states"]]
    assert graph.manifest()["output_contract"]["head_positions"] == [63, 0, 20]
    head = [node for node in graph.nodes if isinstance(node, CustomNode)
            and node.op_type == "TestQuantMatmul" and node.inputs[1].name == "target.lm_head.weight"]
    assert len(head) == 1 and head[0].inputs[0].shape[0] == 3
    assert graph.quantization_contract["residual_dtype"] == "float32"
    assert graph.quantization_contract["precision_boundary_casts"]


def test_explicit_fp32_dynamic_quant_contract_is_respected(small_config, tmp_path):
    config, draft = small_config
    doc = contract_document()
    doc["operators"]["dynamic_quant"]["inputs"]["x"]["dtype"] = "float32"
    graph = build_quantized_target(config, draft, 1, past_length=64,
        quant_bindings=QuantOperatorBindings(doc), quant_descriptors=descriptors_for(config, draft),
        output_positions=(-1,), return_logits=True)
    quantizers = [node for node in graph.nodes if isinstance(node, CustomNode) and node.op_type == "TestDynamicQuant"]
    assert quantizers and all(node.inputs[0].dtype == "float32" for node in quantizers)
    casts = graph.quantization_contract["precision_boundary_casts"]
    assert not any(row["to_dtype"] == "float16" and row["reason"].startswith("DynamicQuant") for row in casts)
    code = emit_quantized_cpp(graph, {key: tmp_path / (key + ".bin") for key in graph.weights}, recipe_dir=tmp_path / "recipe")
    records = read_recipe(tmp_path / "recipe/graph.recipe")["records"]
    assert "TestDynamicQuant" in code
    for row in records:
        if row["type"] == "TestDynamicQuant":
            producer = records[row["inputs"][0]["src"]]
            assert producer["outputs"][0]["dtype"] == "float32"


@pytest.mark.parametrize("positions", [(), (64,), (-65,), (True,), [0]])
def test_invalid_output_positions_rejected(small_config, positions):
    config, draft = small_config
    with pytest.raises(ValueError, match="output_positions"):
        build_quantized_target(config, draft, 64,
            quant_bindings=QuantOperatorBindings(contract_document()), quant_descriptors=descriptors_for(config, draft),
            output_positions=positions)


def test_row_selection_preserves_numerical_logits_features_and_all_states(monkeypatch):
    from qwen35_moe_dflash.ge_weights import tiny_weights
    monkeypatch.setattr("qwen35_moe_dflash.ge_eval.DTYPES", DTYPES)
    config, draft = configurations(tiny=True)
    draft = Qwen35DFlashConfig.from_dict(draft)
    all_rows = build_target(config, draft, 4, return_logits=True, head_chunk_rows=2)
    selected = build_target(config, draft, 4, output_positions=(-1, 1, -1), return_logits=True, head_chunk_rows=2)
    weights = tiny_weights({key: value.shape for key, value in all_rows.weights.items()})
    values = {"input_ids": np.asarray([[1, 3, 7, 9]], dtype=np.int32)}
    expected, actual = evaluate(all_rows, values, weights), evaluate(selected, values, weights)
    np.testing.assert_array_equal(actual[0], expected[0][:, [3, 1, 3]])
    np.testing.assert_array_equal(actual[-1], expected[-1][:, [3, 1, 3]])
    for observed, reference in zip(actual[1:-1], expected[1:-1]):
        np.testing.assert_array_equal(observed, reference)
