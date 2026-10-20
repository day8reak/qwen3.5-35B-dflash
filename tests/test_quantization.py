import copy
import json
import os
from pathlib import Path
import pytest
import torch
from reference.tiny import create
from qwen35_moe_dflash.quantization import (
    expose_experts, make_plan, meta_graphs, memory_budget, dopt_config,
    calibrate, restore_for_reference,
    load_dopt, stabilize_input_quantizers,
)


def test_exposed_experts_preserve_values_and_dynamic_routes():
    torch.set_num_threads(1)
    target, draft = create(quantization_aligned=True)
    oracle = copy.deepcopy(target).eval()
    parameters = sum(p.numel() for p in target.parameters())
    expose_experts(target)
    assert sum(p.numel() for p in target.parameters()) == parameters
    for tokens in [[1, 2], [9, 5, 13, 2], [14]]:
        ids = torch.tensor([tokens], dtype=torch.int32)
        with torch.inference_mode():
            a, b = target(ids, torch.ones_like(ids)), oracle(ids, torch.ones_like(ids))
        torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
        torch.testing.assert_close(a[1], b[1], rtol=1e-4, atol=1e-5)


def test_full_meta_plan_counts_all_experts_and_rejects_18g_budget():
    target, draft = meta_graphs()
    plan = make_plan(target, draft)
    assert plan["expert_count"] == 40 * 256
    assert plan["expert_matrix_count"] == 40 * 256 * 2
    assert plan["parameter_count"]["target"] == 34660610688
    assert plan["weight_budget"]["total_bytes"] == 22751793408
    assert memory_budget(plan, available_gib=18)["status"] == "INSUFFICIENT_FOR_ESTIMATE"
    assert memory_budget(plan, available_gib=32)["status"] == "UNVERIFIED_HEADROOM"
    assert not memory_budget(plan, available_gib=32)["fit_proven"]
    config = dopt_config(plan)["layer_strategy"]
    assert len(config) == len(plan["matrices"])
    assert all(x["weight"]["bit"] == 4 and x["input"]["bit"] == 16 for x in config.values())
    assert not any(n == "lm_head" or n.endswith("in_proj_a") for n in config)


@pytest.mark.skipif(not os.environ.get("QWEN35_TEST_DOPT_ROOT"), reason="optional real blue DOPT integration")
def test_real_dopt_calibration_and_reference_bridge(tmp_path, monkeypatch):
    """Runs actual vendor Cython quantizers. No AIR/device claim."""
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    torch.set_num_threads(1)
    vendor = os.environ["QWEN35_TEST_DOPT_ROOT"]
    target, draft = create(quantization_aligned=True)
    identity = {"kind": "tiny-random", "seed": 17}
    result = calibrate(target, draft, [[1, 2], [7, 8, 3]], tmp_path/"q",
                       sequence_length=8, dopt_root=vendor, source_identity=identity)
    assert result["status"] == "CALIBRATED"
    assert result["plan"]["expert_matrix_count"] == 32
    target, draft = create(quantization_aligned=True)
    target, draft, quant = restore_for_reference(target, draft, tmp_path/"q", vendor, source_identity=identity)
    ids = torch.tensor([[1, 2, 0, 0, 0, 0, 0, 0]], dtype=torch.int32)
    # The bridge must be capturable despite the vendor's Cython forward.
    from qwen35_moe_dflash.precision import prepare_fp16_pair
    from qwen35_moe_dflash.runtime import TorchExecutor, generate
    target, draft, audit = prepare_fp16_pair(target, draft, 8, quantization=True)
    assert audit["target"]["status"] == "PASS"
    executor = TorchExecutor(target, draft, 8, 0)
    a = generate(executor, [1, 2], ordinary=True, max_new_tokens=3, eos_token_ids=())
    b = generate(executor, [1, 2], max_new_tokens=3, eos_token_ids=())
    assert a.token_ids == b.token_ids
    # Tampering a calibration payload must fail before any restore.
    (tmp_path/"q/dopt_config.json").write_text("{}")
    t, d = create(quantization_aligned=True)
    with pytest.raises(ValueError, match="SHA256"):
        restore_for_reference(t, d, tmp_path/"q", vendor, source_identity=identity)


@pytest.mark.skipif(not os.environ.get("QWEN35_TEST_DOPT_ROOT"), reason="optional real blue DOPT integration")
def test_real_dopt_a16_scaling_overflow_is_avoided(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    dopt, _ = load_dopt(os.environ["QWEN35_TEST_DOPT_ROOT"])
    model = torch.nn.Sequential(torch.nn.Linear(128, 128, bias=False)).half().eval()
    config = {"layer_strategy": {"0": {"type": "<class 'torch.nn.modules.linear.Linear'>",
              "quant_strategy": "Quant_act_weight_eco",
              "weight": {"bit": 4, "group_size": 128, "weight_algo": "group_min_max"},
              "input": {"bit": 16, "input_algo": "ema"}}}}
    path = tmp_path/"config.json"; path.write_text(json.dumps(config))
    model = dopt.optimize_model(model, str(path))
    dopt.set_quant_state(model, True, True); dopt.set_calibrate_state(model, True)
    stabilize_input_quantizers(model, ["0"])
    with torch.no_grad():
        model(torch.linspace(-1, 1, 128).half().view(1, 128))
    dopt.set_calibrate_state(model, False)
    quantizer = model[0].quant_op.input_quantizer
    x = torch.tensor([-100, -3, -1, 0, 1, 3, 100], dtype=torch.float16)
    with torch.no_grad():
        actual = quantizer(x)
    assert torch.isfinite(actual).all() and actual.dtype == torch.float16
    lower, upper = quantizer.get_clip_range()
    # Independent high-precision quantize/dequantize oracle for saturated inputs.
    s, z = quantizer.s.double(), quantizer.offset.double()
    expected = ((torch.clamp(torch.round(x.double()/s)+z, lower, upper)-z)*s).half()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
