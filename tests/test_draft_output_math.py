"""Retained-value diagnostics distinguish residual hypotheses without passing one."""
from copy import deepcopy
import json

import numpy as np
import pytest

from qwen35_moe_dflash.config import Qwen35DFlashConfig
from qwen35_moe_dflash.draft_output_math import diagnose_output_relations
from qwen35_moe_dflash.ge_export import configurations


@pytest.fixture
def case():
    raw = deepcopy(configurations(tiny=True)[1])
    raw.update(hidden_size=4, num_hidden_layers=2, layer_types=["full_attention"] * 2)
    raw["dflash_config"]["block_size"] = 16
    cfg = Qwen35DFlashConfig.from_dict(raw)
    block = lambda row: np.broadcast_to(np.array(row, np.float16), (1, 16, 4)).copy()
    values = {"layer_0_input": block([1, 0, 2, -1]), "layer_0_attention": block([0, 2, -1, 1]),
              "layer_0_mlp_down": block([2, -3, 0, 1])}
    # Scalar multiples of one direct scale simplify the independently defined
    # normalization: input+attention+down == [3,-1,1,1], mean-square == 3.
    gamma = np.array([1, 2, .5, 1], np.float16)
    return values, cfg, gamma, block


def _torch_norm(x, gamma, eps):
    import torch
    from qwen35_moe_dflash.ops import TorchDFlashOps
    return TorchDFlashOps().rms_norm(torch.from_numpy(x), torch.from_numpy(gamma), eps).numpy()


@pytest.mark.parametrize("candidate", ["full_residual_sum", "mlp_down_only", "post_attention_only"])
def test_independent_norm_distinguishes_each_residual_hypothesis_without_promoting_it(case, candidate):
    values, cfg, gamma, _ = case
    post = values["layer_0_input"] + values["layer_0_attention"]
    full = post + values["layer_0_mlp_down"]
    source = {"full_residual_sum": full, "mlp_down_only": values["layer_0_mlp_down"], "post_attention_only": post}[candidate]
    values["layer_1_input_norm"] = _torch_norm(source, gamma, cfg.rms_norm_eps)
    before = {key: value.tobytes() for key, value in values.items()}
    report = diagnose_output_relations(values, cfg, gamma)
    rows = report["layer_1_input_norm_candidates"]
    assert rows[candidate]["exact_match"]
    assert all(not row["exact_match"] for name, row in rows.items() if name != candidate)
    assert report["status"] == "DIAGNOSTIC_COMPLETE"
    assert report["root_cause_attribution"] == "NOT_ESTABLISHED"
    assert report["arithmetic_acceptance_threshold"] == "NOT_DEFINED"
    assert before == {key: value.tobytes() for key, value in values.items()}
    json.dumps(report, allow_nan=False)


def test_zero_exports_do_not_erase_reconstructed_residual_candidates(case):
    values, cfg, gamma, block = case
    full = values["layer_0_input"] + values["layer_0_attention"] + values["layer_0_mlp_down"]
    values.update(layer_0_post_attention=block([0] * 4), layer_0_output=block([0] * 4), layer_1_input=block([0] * 4),
                  layer_1_input_norm=_torch_norm(full, gamma, cfg.rms_norm_eps))
    report = diagnose_output_relations(values, cfg, gamma)
    assert all(not row["exact_match"] for row in report["residual_reconstruction"].values())
    assert report["layer_1_input_norm_candidates"]["full_residual_sum"]["exact_match"]
    assert report["root_cause_attribution"] == "NOT_ESTABLISHED"


def test_missing_inputs_or_gamma_stay_unavailable(case):
    values, cfg, gamma, _ = case
    values.pop("layer_0_input")
    report = diagnose_output_relations(values, cfg, None)
    full = report["layer_1_input_norm_candidates"]["full_residual_sum"]
    assert full["status"] == "UNAVAILABLE"
    assert set(full["missing"]) == {"layer_0_input", "layers.1.input_layernorm.weight", "layer_1_input_norm"}
    assert report["masked_score"]["status"] == "UNAVAILABLE"
    assert "layer_0_visible" in report["masked_score"]["missing"]


@pytest.mark.parametrize("all_zero", [True, False])
def test_probability_rows_have_explicit_zero_sum_diagnostic(case, all_zero):
    values, cfg, gamma, _ = case
    shape = (1, cfg.num_attention_heads, cfg.block_size, 8)
    values["layer_0_probability_fp32"] = np.full(shape, 0 if all_zero else .125, np.float32)
    report = diagnose_output_relations(values, cfg, gamma)
    row = report["probability_row_sums"]
    assert row["rows"] == cfg.num_attention_heads * cfg.block_size
    assert row["zero_rows"] == (row["rows"] if all_zero else 0)
    assert row["max_abs_error_from_one"] == (1 if all_zero else 0)
    assert report["masked_score"]["status"] == "UNAVAILABLE"


def test_masked_score_is_checked_only_with_explicit_mask(case):
    values, cfg, gamma, _ = case
    shape = (1, cfg.num_attention_heads, cfg.block_size, 8)
    qk = np.zeros(shape, np.float16)
    mask = np.ones((1, 1, cfg.block_size, 8), bool)
    mask[..., 0] = False
    scores = np.zeros(shape, np.float32)
    scores[..., 0] = -np.inf
    values.update(layer_0_qk_fp16=qk, layer_0_visible=mask, layer_0_scores_fp32=scores)
    row = diagnose_output_relations(values, cfg, gamma)["masked_score"]
    assert row["exact_match"] and row["nonfinite_mismatch_count"] == 0
    values["layer_0_scores_fp32"][..., 1] = 1
    row = diagnose_output_relations(values, cfg, gamma)["masked_score"]
    assert not row["exact_match"] and row["relative_l2"] is None


@pytest.mark.parametrize("bad", ["shape", "dtype", "nonfinite", "gamma_dtype", "gamma_shape", "gamma_nonfinite", "score_shape", "mask_values"])
def test_rejects_invalid_descriptors_and_nonfinite_source_values(case, bad):
    values, cfg, gamma, _ = case
    if bad == "shape":
        values["layer_0_input"] = values["layer_0_input"][:, :1]
    elif bad == "dtype":
        values["layer_0_input"] = values["layer_0_input"].astype(np.int32)
    elif bad == "nonfinite":
        values["layer_0_input"].flat[0] = np.nan
    elif bad == "gamma_dtype":
        gamma = gamma.astype(np.float32)
    elif bad == "gamma_shape":
        gamma = gamma[None, :]
    elif bad == "gamma_nonfinite":
        gamma[0] = np.inf
    elif bad == "score_shape":
        values["layer_0_probability_fp32"] = np.zeros((1, 1, 16, 8), np.float32)
    else:
        shape = (1, cfg.num_attention_heads, cfg.block_size, 8)
        values.update(layer_0_qk_fp16=np.zeros(shape, np.float16), layer_0_scores_fp32=np.zeros(shape, np.float32),
                      layer_0_visible=np.full((1, 1, 16, 8), 2, np.int32))
    with pytest.raises(ValueError):
        diagnose_output_relations(values, cfg, gamma)


def test_fp16_overflow_is_unavailable_not_a_matching_candidate(case):
    values, cfg, gamma, block = case
    values.update(layer_0_input=block([65000] * 4), layer_0_attention=block([65000] * 4),
                  layer_1_input_norm=block([1] * 4))
    report = diagnose_output_relations(values, cfg, gamma)
    assert report["layer_1_input_norm_candidates"]["full_residual_sum"]["status"] == "UNAVAILABLE"
    assert report["layer_1_input_norm_candidates"]["full_residual_sum"]["reason"] == "RECONSTRUCTED_FP16_VALUE_NONFINITE"
    json.dumps(report, allow_nan=False)
