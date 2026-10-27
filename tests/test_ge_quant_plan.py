"""Full target policy coverage and portable-cache boundaries, without weights."""
import hashlib
from pathlib import Path

import pytest

from qwen35_moe_dflash.ge_quant_plan import target_quantization_plan, target_tensor_quantization
from qwen35_moe_dflash.ge_weights import SafeTensorFiles


@pytest.fixture(scope="module")
def plan():
    return target_quantization_plan()


def test_complete_target_policy_and_exact_portable_sizes(plan):
    summary = plan["summary"]
    assert summary["tensor_count"] == 693
    assert summary["parameter_count"] == 34_660_610_688
    assert summary["counts"] == {
        "linear_int4": 351, "expert_int4": 80, "embedding_int8": 1, "float16": 261}
    assert summary["weight_bytes"] == 17_586_447_616
    assert summary["scale_bytes"] == 534_599_680
    assert summary["payload_bytes"] == 18_121_047_296
    keys = [item["key"] for item in plan["tensors"]]
    assert keys == sorted(set(keys))
    assert {int(key.split(".")[2]) for key in keys if key.startswith("target.layers.")} == set(range(40))
    assert all(item["source_expected_dtype"] == "BF16" for item in plan["tensors"])
    assert all(item["source_tensor"].startswith("model.language_model.")
               for item in plan["tensors"] if item["key"] != "target.lm_head.weight")
    assert next(item for item in plan["tensors"] if item["key"] == "target.lm_head.weight")["source_tensor"] == "lm_head.weight"


def test_policy_keeps_small_projections_quantized_and_norms_unfolded(plan):
    by_key = {item["key"]: item for item in plan["tensors"]}
    for index in range(40):
        prefix = f"target.layers.{index}."
        gate = by_key[prefix + "mlp.shared_expert_gate.weight"]
        assert gate["kind"] == "float16" and gate["shape"] == [1, 2048]
        assert by_key[prefix + "mlp.gate.weight"]["kind"] == "linear_int4"
        for name in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            norm = by_key[prefix + name]
            assert norm["checkpoint_weight_addend"] == 0
            assert norm["graph_norm_weight_addend"] == 1
            assert norm["weight_transform"] == "CAST_ONLY_NO_OFFSET"
        if index % 4 != 3:
            for name in ("in_proj_a.weight", "in_proj_b.weight"):
                assert by_key[prefix + "linear_attn." + name]["kind"] == "linear_int4"
            gated = by_key[prefix + "linear_attn.norm.weight"]
            assert gated["checkpoint_weight_addend"] == gated["graph_norm_weight_addend"] == 0
        else:
            for name in ("q_norm.weight", "k_norm.weight"):
                norm = by_key[prefix + "self_attn." + name]
                assert norm["checkpoint_weight_addend"] == 0 and norm["graph_norm_weight_addend"] == 1


def test_cache_layout_is_checkpoint_k_order_not_hardware_packing(plan):
    by_key = {item["key"]: item for item in plan["tensors"]}
    expert = by_key["target.layers.0.mlp.experts.gate_up_proj"]
    assert expert["weight_shape"] == [256, 1024, 1024]
    assert expert["scale_shape"] == [256, 1024, 8]
    assert expert["weight_dtype"] == "uint8" and expert["scale_dtype"] == "float32"
    embedding = by_key["target.embed_tokens.weight"]
    assert embedding["weight_dtype"] == "int8" and embedding["scale_shape"] == [248320]
    assert plan["status"] == "PORTABLE_QUANTIZATION_PLAN"
    assert plan["deployment_ready"] is False and plan["hardware_format"] == "UNRESOLVED"
    assert plan["hardware_mapping"]["QPARAM"] == plan["hardware_mapping"]["I4ZN"] == "NOT_IMPLEMENTED"
    assert plan["model_compilation"] == plan["device_execution"] == "NOT_RUN"
    assert plan["graph_inventory"] == {"sequence_length": 1, "past_length": 1,
        "expert_dispatch": "selected", "fused_ops": [], "num_hidden_layers": 40}


def test_sources_are_hash_bound_and_checkpoint_never_opened(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("a plan must not open checkpoint weights")
    monkeypatch.setattr(SafeTensorFiles, "__init__", forbidden)
    plan = target_quantization_plan()
    root = Path(__file__).resolve().parents[1]
    for name in ("target", "draft"):
        source = plan["sources"][name]
        assert len(source["revision"]) == 40
        assert hashlib.sha256((root / source["config"]["path"]).read_bytes()).hexdigest() == source["config"]["sha256"]
    document = plan["sources"]["document"]
    assert hashlib.sha256((root / document["path"]).read_bytes()).hexdigest() == document["sha256"]
    assert plan["weight_access"] == "NOT_RUN"


@pytest.mark.parametrize("key,shape", [
    ("target.layers.0.unknown.weight", (256, 256)),
    ("target.layers.0.self_attn.q_proj.weight", (256, 257)),
    ("target.layers.0.mlp.experts.down_proj", (256, 256)),
    ("target.embed_tokens.weight", (256,)),
    ("target.norm.weight", (True,)),
])
def test_unrecognized_or_incompatible_weight_is_rejected(key, shape):
    with pytest.raises(ValueError):
        target_tensor_quantization(key, shape)
