"""Bounded algebraic diagnostics on already retrieved original draft values.

Candidate comparisons describe observations, not an accuracy gate or a proof
of the compiled graph's internal values. No files or model objects are changed.
"""
from __future__ import annotations

import math

import numpy as np

from .config import Qwen35DFlashConfig
from .draft_trace_integrity import _same_values


def diagnose_output_relations(values, config, norm_gamma_l1):
    """Compare layer-one norm against full and missing-residual hypotheses.

    ``norm_gamma_l1`` is the source FP16 ``layers.1.input_layernorm.weight``
    vector, with no offset added. Missing observations remain UNAVAILABLE.
    NumPy's FP32 reduction ordering is diagnostic, not a target kernel model.
    """
    cfg = config if isinstance(config, Qwen35DFlashConfig) else Qwen35DFlashConfig.from_dict(config)
    if not math.isfinite(cfg.rms_norm_eps) or cfg.rms_norm_eps <= 0:
        raise ValueError("output diagnostics require positive finite RMS epsilon")
    if (not isinstance(values, dict) or len(values) > 256
            or any(not isinstance(name, str) or not isinstance(value, np.ndarray)
                   or value.dtype not in (np.dtype("float16"), np.dtype("float32"), np.dtype("int32"), np.dtype("bool"))
                   for name, value in values.items())
            or sum(value.nbytes for value in values.values()) > 64 << 20):
        raise ValueError("output diagnostics require bounded typed NumPy observations")
    residual_names = ("layer_0_input", "layer_0_attention", "layer_0_post_attention",
                      "layer_0_mlp_down", "layer_0_output", "layer_1_input", "layer_1_input_norm")
    residual_shape = (1, cfg.block_size, cfg.hidden_size)
    for name in residual_names:
        if name in values and (values[name].shape != residual_shape
                or values[name].dtype not in (np.dtype("float16"), np.dtype("float32"))
                or not np.all(np.isfinite(values[name]))):
            raise ValueError("invalid finite residual/norm observation: " + name)
    if norm_gamma_l1 is not None and (not isinstance(norm_gamma_l1, np.ndarray)
            or norm_gamma_l1.dtype != np.dtype("float16")
            or norm_gamma_l1.shape != (cfg.hidden_size,)
            or not np.all(np.isfinite(norm_gamma_l1))):
        raise ValueError("layer-one direct norm gamma must be finite FP16 [hidden_size]")

    def add_fp16(left, right):
        with np.errstate(all="ignore"):
            return (left.astype(np.float16) + right.astype(np.float16)).astype(np.float16)

    reconstructed_post = (add_fp16(values["layer_0_input"], values["layer_0_attention"])
                          if all(name in values for name in ("layer_0_input", "layer_0_attention")) else None)
    reconstructed_output = (add_fp16(reconstructed_post, values["layer_0_mlp_down"])
                            if reconstructed_post is not None and "layer_0_mlp_down" in values else None)

    def comparison(expected, observed, missing):
        if expected is None or observed not in values:
            return {"status": "UNAVAILABLE", "missing": missing + ([] if observed in values else [observed])}
        if not np.all(np.isfinite(expected)):
            return {"status": "UNAVAILABLE", "reason": "RECONSTRUCTED_FP16_VALUE_NONFINITE"}
        return {"status": "DIAGNOSTIC_COMPLETE", **_same_values(expected, values[observed])}

    residuals = {}
    for name, expected, required in (
            ("layer_0_post_attention", reconstructed_post, ("layer_0_input", "layer_0_attention")),
            ("layer_0_output", reconstructed_output, ("layer_0_input", "layer_0_attention", "layer_0_mlp_down")),
            ("layer_1_input", reconstructed_output, ("layer_0_input", "layer_0_attention", "layer_0_mlp_down"))):
        residuals[name] = comparison(expected, name, [item for item in required if item not in values])

    candidates = {}
    for name, source, required, formula in (
            ("full_residual_sum", reconstructed_output, ("layer_0_input", "layer_0_attention", "layer_0_mlp_down"),
             "RMS(FP16(FP16(layer_0_input+layer_0_attention)+layer_0_mlp_down))"),
            ("mlp_down_only", values.get("layer_0_mlp_down"), ("layer_0_mlp_down",), "RMS(layer_0_mlp_down)"),
            ("post_attention_only", reconstructed_post, ("layer_0_input", "layer_0_attention"),
             "RMS(FP16(layer_0_input+layer_0_attention))")):
        missing = [item for item in required if item not in values]
        if norm_gamma_l1 is None:
            missing.append("layers.1.input_layernorm.weight")
        expected = None
        if source is not None and norm_gamma_l1 is not None and np.all(np.isfinite(source)):
            x = source.astype(np.float16).astype(np.float32)
            with np.errstate(all="ignore"):
                inverse_rms = np.reciprocal(np.sqrt(np.mean(x * x, axis=-1, keepdims=True, dtype=np.float32)
                                                   + np.float32(cfg.rms_norm_eps)))
                expected = ((x * inverse_rms).astype(np.float16) * norm_gamma_l1).astype(np.float16)
        candidates[name] = {"formula": formula, **comparison(expected, "layer_1_input_norm", missing)}
        if source is not None and not np.all(np.isfinite(source)):
            candidates[name]["reason"] = "RECONSTRUCTED_FP16_VALUE_NONFINITE"

    probability = values.get("layer_0_probability_fp32")
    score_names = ("layer_0_qk_fp16", "layer_0_scores_fp32", "layer_0_probability_fp32")
    present_scores = [values[name] for name in score_names if name in values]
    for name in score_names:
        if name not in values:
            continue
        array = values[name]
        if (array.dtype not in (np.dtype("float16"), np.dtype("float32")) or array.ndim != 4
                or array.shape[:3] != (1, cfg.num_attention_heads, cfg.block_size)
                or not array.shape[-1] or (present_scores and array.shape != present_scores[0].shape)
                or np.any(np.isnan(array)) or np.any(np.isposinf(array))
                or (name != "layer_0_scores_fp32" and not np.all(np.isfinite(array)))):
            raise ValueError("invalid score/probability observation: " + name)
    if probability is None:
        probability_summary = {"status": "UNAVAILABLE", "missing": ["layer_0_probability_fp32"]}
    else:
        sums = np.sum(probability.astype(np.float64), axis=-1)
        probability_summary = {"status": "DIAGNOSTIC_COMPLETE", "rows": int(sums.size),
            "minimum": float(np.min(sums)), "maximum": float(np.max(sums)),
            "zero_rows": int(np.count_nonzero(sums == 0)), "max_abs_error_from_one": float(np.max(np.abs(sums - 1))),
            "negative_elements": int(np.count_nonzero(probability < 0)),
            "greater_than_one_elements": int(np.count_nonzero(probability > 1))}

    required_score = ("layer_0_qk_fp16", "layer_0_visible", "layer_0_scores_fp32")
    missing_score = [name for name in required_score if name not in values]
    masked_score = {"status": "UNAVAILABLE", "missing": missing_score}
    if not missing_score:
        qk, visible, scores = (values[name] for name in required_score)
        if visible.dtype not in (np.dtype("bool"), np.dtype("int32")) or np.any((visible != 0) & (visible != 1)):
            raise ValueError("visible mask must contain BOOL/INT32 zero or one")
        try:
            mask = np.broadcast_to(visible.astype(bool), qk.shape)
        except ValueError as error:
            raise ValueError("visible mask shape does not broadcast to QK") from error
        expected = np.where(mask, qk.astype(np.float16).astype(np.float32) * np.float32(cfg.head_dim ** -.5), np.float32(-np.inf))
        masked_score = {"status": "DIAGNOSTIC_COMPLETE", **_same_values(expected, scores)}
    return {"schema_version": 1, "artifact_kind": "qwen35-draft-output-math-diagnostic", "status": "DIAGNOSTIC_COMPLETE",
        "residual_reconstruction": residuals, "layer_1_input_norm_candidates": candidates,
        "probability_row_sums": probability_summary, "masked_score": masked_score,
        "arithmetic_acceptance_threshold": "NOT_DEFINED", "root_cause_attribution": "NOT_ESTABLISHED",
        "scope": "Algebraic hypotheses from retained observations; unreliable source observations can invalidate every reconstruction. No model or candidate pass."}
