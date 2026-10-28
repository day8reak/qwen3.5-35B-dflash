"""Bounded host inspection of retained standalone draft numerical failures.

This does not regenerate a reference, invoke Torch, compile, or contact a board.
All original execution and comparison evidence remains unchanged.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from .artifacts import output_path
from .draft_board import _file
from .draft_review import _json, _reference_audit, review_draft_result
from .hashing import sha256_file


KIND = "qwen35-retained-draft-numerical-diagnostic"
MAX_HIDDEN_BYTES = 1 << 20
MAX_ROWS = 256
SAMPLE_VALUES = 16


def _number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _summary(array):
    values = array.astype(np.float64, copy=False)
    finite = np.isfinite(values)
    selected = values[finite]
    all_finite = bool(finite.all())
    return {"elements": int(values.size), "nonfinite_count": int((~finite).sum()),
            "zero_count": int(np.count_nonzero(values == 0)),
            "finite_min": _number(selected.min()) if selected.size else None,
            "finite_max": _number(selected.max()) if selected.size else None,
            "finite_abs_max": _number(np.abs(selected).max()) if selected.size else None,
            "l2_norm": _number(np.linalg.norm(values)) if all_finite else None,
            "rms": _number(np.sqrt(np.mean(values * values))) if all_finite and values.size else None,
            "fp16_max_finite_magnitude_count": int(np.count_nonzero(np.abs(selected) == 65504)),
            "all_finite": all_finite}


def _fingerprint(array):
    flat = np.asarray(array, dtype="<f2").reshape(-1)
    bits, counts = np.unique(flat.view("<u2"), return_counts=True)
    order = np.lexsort((bits, -counts))[:SAMPLE_VALUES]
    common = []
    for index in order:
        pattern = int(bits[index])
        value = np.asarray([pattern], dtype="<u2").view("<f2")[0]
        common.append({"bits_hex": f"0x{pattern:04x}", "value": _number(value),
                       "classification": "finite" if np.isfinite(value) else
                           "nan" if np.isnan(value) else "positive_infinity" if value > 0 else "negative_infinity",
                       "count": int(counts[index])})
    return {"unique_fp16_bit_patterns": int(bits.size), "common_fp16_values": common,
            "first_values": [_number(v) for v in flat[:SAMPLE_VALUES]],
            "first_bits_hex": [f"0x{int(v):04x}" for v in flat.view("<u2")[:SAMPLE_VALUES]],
            "sample_limit": SAMPLE_VALUES}


def _output_interpretation(array, native):
    """Expose byte-pattern hypotheses without changing the declared FP16 ABI."""
    half = np.ascontiguousarray(array, dtype="<f2").reshape(-1)
    words = half.view("<u2")
    lanes = []
    for index in (0, 1):
        values, patterns = half[index::2], words[index::2]
        finite = np.isfinite(values)
        selected = values[finite].astype(np.float64)
        lanes.append({
            "half_word_index_mod_2": index,
            "fp32_hypothesis_word": "low16" if index == 0 else "high16",
            "elements": int(values.size), "nonfinite_count": int((~finite).sum()),
            "rms": _number(np.sqrt(np.mean(selected * selected)))
                if selected.size and finite.all() else None,
            "finite_abs_gt_32_count": int(np.count_nonzero(np.abs(selected) > 32)),
            "positive_257_75_word_count": int(np.count_nonzero(patterns == 0x5C07)),
            "negative_257_75_word_count": int(np.count_nonzero(patterns == 0xDC07)),
        })
    allocation = []
    for item in native.get("io_diagnostic", {}).get("outputs", []):
        if isinstance(item, dict) and item.get("index") in (0, 1):
            allocation.append({key: item[key] for key in (
                "index", "logical_bytes", "allocated_bytes_before", "allocated_bytes_after",
                "object_same", "buffer_same") if key in item})
    # This view consumes exactly the retained bytes; it neither requests nor
    # invents a second half of a possible FP32 output buffer.
    if half.nbytes % 4:
        reinterpretation = {"available": False, "reason": "retained byte count is not divisible by four"}
    else:
        floated = half.view("<f4")
        finite = np.isfinite(floated)
        selected = floated[finite].astype(np.float64)
        reinterpretation = {
            "available": True, "elements": int(floated.size),
            "finite_count": int(finite.sum()), "nonfinite_count": int((~finite).sum()),
            "finite_min": _number(selected.min()) if selected.size else None,
            "finite_max": _number(selected.max()) if selected.size else None,
            "rms": _number(np.sqrt(np.mean(selected * selected)))
                if selected.size and finite.all() else None,
        }
    return {
        "declared_output_dtype": "float16", "comparison_interpretation": "UNCHANGED",
        "retained_bytes": int(half.nbytes), "native_io_outputs": allocation,
        "fp16_half_word_lanes": lanes,
        "same_bytes_as_little_endian_fp32": reinterpretation,
        "hypothesis_note": (
            "A simple misread of normal FP32 normalized values would have small high16 half-words; "
            "large high16 half-word values do not fit that simple signature. "
            "These byte views establish neither the runtime dtype nor a root cause, "
            "and the FP32 view covers only the retained bytes."),
    }


def _row_metrics(actual, expected):
    rows = []
    width = actual.shape[-1]
    for index, row in enumerate(actual.reshape(-1, width)):
        item = {"row": index, "actual": _summary(row)}
        if expected is not None:
            reference = expected.reshape(-1, width)[index]
            item["expected"] = _summary(reference)
            comparable = item["actual"]["all_finite"] and item["expected"]["all_finite"]
            item["comparison_status"] = "FINITE" if comparable else "NONFINITE_INPUT"
            metrics = {"cosine_similarity": None, "max_abs_error": None, "rmse": None,
                       "relative_l2_error": None, "l2_norm_ratio": None}
            if comparable:
                left, right = row.astype(np.float64), reference.astype(np.float64)
                difference = left - right
                actual_norm, expected_norm = np.linalg.norm(left), np.linalg.norm(right)
                metrics.update(max_abs_error=_number(np.abs(difference).max()),
                               rmse=_number(np.sqrt(np.mean(difference * difference))))
                if expected_norm > 0:
                    metrics.update(relative_l2_error=_number(np.linalg.norm(difference) / expected_norm),
                                   l2_norm_ratio=_number(actual_norm / expected_norm))
                else:
                    item["zero_reference_norm"] = True
                if actual_norm > 0 and expected_norm > 0:
                    metrics["cosine_similarity"] = _number(np.clip(
                        np.dot(left / actual_norm, right / expected_norm), -1.0, 1.0))
            item.update(metrics)
        rows.append(item)
    return rows


def _unit_scale_constraint(reference_audit, width, rows, epsilon):
    scales = [item for item in reference_audit.get("final_norms", [])
              if item.get("name") == "draft.norm.weight"]
    result = {"status": "NOT_ESTABLISHED", "scope": "Structural fault isolation only; not an accuracy acceptance threshold."}
    if (isinstance(epsilon, bool) or not isinstance(epsilon, (int, float))
            or not math.isfinite(epsilon) or epsilon < 0):
        result["reason"] = "No verified finite, nonnegative normalization epsilon in the executed bundle."
        return result
    result["rms_norm_epsilon"] = epsilon
    if (reference_audit.get("status") != "INSPECTED"
            or reference_audit.get("manifest_binding") != "MATCHED" or len(scales) != 1):
        result["reason"] = "No bound final normalization scale audit."
        return result
    scale = scales[0]
    result["scale"] = scale
    if (scale.get("elements") != width or scale.get("nonfinite_count") != 0
            or scale.get("nonzero_count") != width
            or scale.get("finite_min") != 1.0 or scale.get("finite_max") != 1.0):
        result["reason"] = "Final normalization scale is not verified as exactly one at every hidden channel."
        return result
    # x/sqrt(mean(x^2)+epsilon) has RMS <= 1 for epsilon >= 0. A factor
    # of two is deliberately a coarse diagnostic guard, well outside ordinary
    # FP32 normalization / FP16 rounding. It never changes comparator gates.
    gross = [row["row"] for row in rows
             if row["actual"]["rms"] is not None and row["actual"]["rms"] > 2.0]
    result.update(status="GROSSLY_VIOLATED" if gross else "NO_GROSS_VIOLATION",
                  unit_scale_verified=True, ideal_row_rms_upper_bound=1.0,
                  gross_fault_isolation_bound=2.0, actual_rms_gt_2_rows=gross,
                  formula="RMS(x / sqrt(mean(x*x) + epsilon)) <= 1, epsilon >= 0; scale = 1",
                  qualification="The ideal bound precedes floating-point rounding. Values above two are a gross structural symptom; no particular kernel or cause is inferred.")
    return result


def _report_path(job):
    raw = job / "numerical-diagnostic"
    for path in (raw, *raw.parents):
        if path.is_symlink():
            raise ValueError("numerical diagnostic output must not traverse symlinks")
    folder = output_path(raw)
    if folder.exists():
        if not folder.is_dir() or {p.name for p in folder.iterdir()} - {"report.json"}:
            raise ValueError("numerical diagnostic directory contains unowned files")
        report = folder / "report.json"
        if report.exists() or report.is_symlink():
            previous = _json(report)
            if previous.get("artifact_kind") != KIND or previous.get("job") != str(job):
                raise ValueError("existing numerical diagnostic report is not owned by this job")
    return folder / "report.json"


def _read_hidden(root, record):
    if (not isinstance(record, dict) or type(record.get("bytes")) is not int
            or not 0 < record["bytes"] <= MAX_HIDDEN_BYTES):
        raise ValueError("hidden diagnostic exceeds the bounded payload size")
    shape = record.get("shape")
    if (not isinstance(shape, list) or len(shape) != 3
            or any(type(d) is not int or d <= 0 for d in shape)
            or math.prod(shape[:-1]) > MAX_ROWS or record.get("dtype") != "float16"
            or math.prod(shape) * 2 != record["bytes"]):
        raise ValueError("invalid bounded hidden diagnostic descriptor")
    return np.fromfile(_file(root, record), dtype="<f2").reshape(shape)


def diagnose_draft_numerics(job_dir, schema_library=None):
    """Inspect hash-bound retained data and atomically save one small report.

    The caller owns the fixed-workspace lock. Repeated calls replace only this
    diagnostic's own report; original reports and binary outputs are untouched.
    """
    raw_job = Path(job_dir).expanduser()
    if any(path.is_symlink() for path in (raw_job, *raw_job.parents)):
        raise ValueError("retained job must not traverse symlinks")
    job_root = raw_job.resolve()
    report_path = _report_path(job_root)
    review = review_draft_result(job_root)
    job = _json(job_root / "result.json")
    component = job["components"]["bundle"]
    bundle = Path(component["directory"]).resolve()
    manifest = _json(_file(bundle, component["manifest"]))
    reference_policy = {}
    fixture_component = job.get("components", {}).get("fixture")
    if fixture_component:
        fixture_root = Path(fixture_component["directory"]).resolve()
        fixture = _json(_file(fixture_root, fixture_component["manifest"]))
        reference_policy = {key: fixture.get(key) for key in (
            "reference", "reference_contract", "rms_normalization", "attention_precision", "rotary_precision", "precision")}
    native = _json(job_root / "board/results/result.json")
    actual_record = native["outputs"][1]
    actual = _read_hidden(job_root / "board/results", actual_record)
    expected_record = manifest["expected"][1]
    expected = _read_hidden(bundle, expected_record) if expected_record is not None else None
    if expected is not None and actual.shape != expected.shape:
        raise ValueError("actual/reference hidden shapes differ")
    rows = _row_metrics(actual, expected)
    from .draft_precision_inspect import inspect_draft_precision
    precision = inspect_draft_precision(job_root, schema_library=schema_library)
    report = {"schema_version": 1, "artifact_kind": KIND, "status": "INSPECTED",
              "job": str(job_root), "output": str(report_path),
              "operation": "HOST_RETAINED_NUMERICAL_DIAGNOSTIC",
              "new_device_execution": "NOT_RUN", "new_compilation": "NOT_RUN",
              "new_reference_execution": "NOT_RUN", "threshold_changes": "NONE",
              "device_execution": review["device_execution"],
              "numerical_validation": review["numerical_validation"],
              "runtime_placement": "UNVERIFIED",
              "identities": {name: review[name] for name in (
                  "job_result_sha256", "board_result_sha256", "bundle_manifest_sha256")},
              "retrieved_files": review["retrieved_files"],
              "native_process_id": review["native_process_id"],
              "comparison": review["comparison"], "reference_audit": review["reference_audit"],
              "reference_policy": reference_policy,
              "output_interpretation": _output_interpretation(actual, native),
              "hidden": {"shape": list(actual.shape), "actual": {**_summary(actual), **_fingerprint(actual)},
                         "expected": {**_summary(expected), **_fingerprint(expected)} if expected is not None else None,
                         "per_row": rows},
              "unit_final_scale_constraint": _unit_scale_constraint(
                  review["reference_audit"], actual.shape[-1], rows, manifest.get("draft_config", {}).get("rms_norm_eps")),
              "compiled_precision": precision,
              "scope": "Retained synthetic-fixture diagnostics only. The current FP16 deployment reference is not an original BF16/FA4 oracle; no particular chip or kernel cause is established."}
    constraint = report["unit_final_scale_constraint"]
    report["summary"] = {"status": report["status"], "device_execution": report["device_execution"],
                         "numerical_validation": report["numerical_validation"],
                         "native_process_id": report["native_process_id"],
                         "token_mismatches": review["comparison"].get("token_mismatches"),
                         "reference_kind": review["reference_audit"].get("reference_kind"),
                         "reference_policy": {key: value for key, value in reference_policy.items() if value is not None},
                         "unit_final_scale_constraint": constraint,
                         "reason": "Final hidden RMS grossly exceeds the unit-scale normalization constraint." if constraint["status"] == "GROSSLY_VIOLATED" else "Inspect retained per-row errors and compiled precision; no cause inferred.",
                         "actual_hidden": _summary(actual),
                         "expected_hidden": _summary(expected) if expected is not None else None,
                         "per_row": [{"row": row["row"], "actual_rms": row["actual"]["rms"],
                                      "expected_rms": row.get("expected", {}).get("rms"),
                                      "cosine": row.get("cosine_similarity"),
                                      "max_abs_error": row.get("max_abs_error"),
                                      "relative_l2": row.get("relative_l2_error"),
                                      "norm_ratio": row.get("l2_norm_ratio")} for row in rows],
                         "common_actual_fp16_values": report["hidden"]["actual"]["common_fp16_values"],
                         "output_interpretation": report["output_interpretation"],
                         "compiled_precision": precision.get("summary", {"status": precision.get("status"), "reason": precision.get("reason")}),
                         "new_device_execution": "NOT_RUN", "new_compilation": "NOT_RUN",
                         "threshold_changes": "NONE", "report": str(report_path)}
    # Avoid publishing a report whose primary receipts changed while static
    # inspection ran. The files themselves were checked by the strict review.
    for name, path in (("job_result_sha256", job_root / "result.json"),
                       ("board_result_sha256", job_root / "board/result.json"),
                       ("bundle_manifest_sha256", bundle / "manifest.json")):
        if sha256_file(path) != report["identities"][name]:
            raise ValueError("retained identity changed during numerical inspection: " + name)
    for record in review["retrieved_files"]:
        _file(job_root / "board", record)
    if expected_record is not None:
        _file(bundle, expected_record)
    if _reference_audit(job, manifest) != review["reference_audit"]:
        raise ValueError("prepared normalization audit changed during numerical inspection")
    payload = (json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    if len(payload) > 4 << 20:
        raise ValueError("numerical diagnostic report exceeds its byte bound")
    report_path.parent.mkdir(exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".report-", suffix=".tmp", dir=report_path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, report_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return report
