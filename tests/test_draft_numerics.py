"""Host-only diagnostics over fabricated device receipts; no device claims."""
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from qwen35_moe_dflash import draft_numerics as subject
from qwen35_moe_dflash.hashing import sha256_file
from test_draft_review import (bundle, retained, _rebind_output, _snapshot,
                               _store_execution, record, write_json)


@pytest.fixture
def inspection(monkeypatch):
    module = types.ModuleType("qwen35_moe_dflash.draft_precision_inspect")
    calls = []
    def inspect(job, schema_library=None):
        calls.append((job, schema_library))
        return {"status": "UNAVAILABLE", "reason": "fabricated fixture has no compiled graph", "summary": {"status": "UNAVAILABLE"}}
    module.inspect_draft_precision = inspect
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return calls, module


def _bind_norm(case, *, scale=1.0):
    job_root = case["job"]
    prepared, fixture = job_root / "prepared", job_root / "fixture"
    prepared.mkdir(); fixture.mkdir()
    parameter = prepared / "final_norm.bin"
    np.full(2048, scale, dtype="<f2").tofile(parameter)
    write_json(prepared / "prepared.json", {"weights": {
        "draft.norm.weight": record(parameter, prepared, dtype="float16", shape=[2048]),
        "draft.fc.weight": {"path": "must-not-load-large-weight.bin"}}})
    write_json(fixture / "report.json", {
        "prepared_manifest_sha256": sha256_file(prepared / "prepared.json"),
        "reference": "PYTORCH_FP16_AIR4B_PRECISION",
        "reference_contract": "pytorch-fp16-air4b-precision-v3",
        "precision": "float16"})
    case["manifest"]["fixture_report_sha256"] = sha256_file(fixture / "report.json")
    write_json(case["bundle"] / "manifest.json", case["manifest"])
    case["execution"]["bundle_manifest_sha256"] = sha256_file(case["bundle"] / "manifest.json")
    _store_execution(case)
    job = json.loads((job_root / "result.json").read_text())
    for name, folder, filename in (("prepared", prepared, "prepared.json"),
                                    ("fixture", fixture, "report.json"),
                                    ("bundle", case["bundle"], "manifest.json")):
        job["components"][name] = {"directory": str(folder), "manifest": record(folder / filename, folder)}
    write_json(job_root / "result.json", job)


def _actual(case, array):
    array.astype("<f2").tofile(case["board"] / "results/output_0001.bin")
    _rebind_output(case, 1)


def test_retained_severe_scale_failure_is_inspected_without_changing_acceptance_or_originals(retained, inspection):
    calls, _ = inspection
    _bind_norm(retained)
    actual = np.full((1, 15, 2048), 258.75, dtype="<f2")
    actual[:, 1] *= -1
    actual[0, 0, 0] = 0
    _actual(retained, actual)
    np.zeros((1, 15), dtype="<i4").tofile(retained["board"] / "results/output_0000.bin")
    _rebind_output(retained, 0)
    before = _snapshot(retained["job"])
    schema = Path("/selected/schema.so")
    for _ in range(2):
        report = subject.diagnose_draft_numerics(retained["job"], schema_library=schema)
        assert report["status"] == "INSPECTED"
        assert report["device_execution"] == "PASSED" and report["numerical_validation"] == "FAILED"
        assert report["comparison"]["token_mismatches"] == 15
        assert report["threshold_changes"] == "NONE"
        assert report["new_device_execution"] == report["new_compilation"] == report["new_reference_execution"] == "NOT_RUN"
        assert report["reference_policy"]["reference_contract"] == "pytorch-fp16-air4b-precision-v3"
        constraint = report["unit_final_scale_constraint"]
        assert constraint["status"] == "GROSSLY_VIOLATED"
        assert constraint["ideal_row_rms_upper_bound"] == 1.0
        assert constraint["actual_rms_gt_2_rows"] == list(range(15))
        rows = report["hidden"]["per_row"]
        assert rows[1]["actual"]["rms"] == 258.75
        assert rows[1]["expected"]["rms"] == .5
        assert rows[1]["l2_norm_ratio"] == pytest.approx(517.5)
        assert rows[1]["cosine_similarity"] == pytest.approx(-1)
        assert rows[1]["max_abs_error"] == 259.25
        assert len(rows) == 15
        fingerprint = report["hidden"]["actual"]
        assert fingerprint["zero_count"] == 1
        assert fingerprint["common_fp16_values"][0]["value"] == 258.75
        assert len(fingerprint["first_values"]) == 16
        assert len(fingerprint["common_fp16_values"]) <= 16
        assert report["summary"]["unit_final_scale_constraint"]["status"] == "GROSSLY_VIOLATED"
        assert report["summary"]["compiled_precision"] == {"status": "UNAVAILABLE"}
        assert set(report["summary"]["per_row"][0]) == {
            "row", "actual_rms", "expected_rms", "cosine", "max_abs_error", "relative_l2", "norm_ratio"}
        assert all(value is not None for value in report["summary"]["reference_policy"].values())
        interpretation = report["output_interpretation"]
        assert report["summary"]["output_interpretation"] == interpretation
        assert interpretation["declared_output_dtype"] == "float16"
        assert interpretation["comparison_interpretation"] == "UNCHANGED"
        assert interpretation["retained_bytes"] == 61440
        assert interpretation["fp16_half_word_lanes"][1]["finite_abs_gt_32_count"] == 15360
        path = retained["job"] / "numerical-diagnostic/report.json"
        assert json.loads(path.read_text()) == report
        json.dumps(report, allow_nan=False)
        assert all(sha256_file(retained["job"] / name) == digest for name, digest in before.items())
    assert calls == [(retained["job"], schema)] * 2
    assert set(_snapshot(retained["job"])) - set(before) == {"numerical-diagnostic/report.json"}


@pytest.mark.parametrize("scale", [0, 2, np.inf])
def test_unit_rms_constraint_requires_actual_bound_unit_scale(retained, inspection, scale):
    _bind_norm(retained, scale=scale)
    _actual(retained, np.full((1, 15, 2048), 258.75, dtype="<f2"))
    report = subject.diagnose_draft_numerics(retained["job"])
    assert report["unit_final_scale_constraint"]["status"] == "NOT_ESTABLISHED"
    assert "not verified as exactly one" in report["unit_final_scale_constraint"]["reason"]


def test_missing_norm_receipt_never_infers_unit_scale(retained, inspection):
    report = subject.diagnose_draft_numerics(retained["job"])
    assert report["unit_final_scale_constraint"]["status"] == "NOT_ESTABLISHED"
    assert report["numerical_validation"] == "TOP1_MATCHED"


def test_nonfinite_and_zero_norm_metrics_remain_json_safe():
    actual = np.asarray([[[np.nan, np.inf, -np.inf, 0], [0, 0, 0, 0]]], dtype="<f2")
    expected = np.zeros_like(actual)
    rows = subject._row_metrics(actual, expected)
    assert rows[0]["comparison_status"] == "NONFINITE_INPUT"
    assert rows[0]["relative_l2_error"] is rows[0]["actual"]["rms"] is None
    assert rows[1]["zero_reference_norm"] is True
    assert rows[1]["cosine_similarity"] is rows[1]["relative_l2_error"] is None
    assert rows[1]["rmse"] == 0
    json.dumps({"rows": rows, "fingerprint": subject._fingerprint(actual)}, allow_nan=False)


def test_tampered_output_is_rejected_before_any_report_is_written(retained, inspection):
    path = retained["board"] / "results/output_0001.bin"
    path.write_bytes(b"altered")
    with pytest.raises(ValueError, match="SHA256/size mismatch"):
        subject.diagnose_draft_numerics(retained["job"])
    assert not (retained["job"] / "numerical-diagnostic").exists()


def test_tampering_during_static_inspection_is_rejected(retained, inspection):
    _, module = inspection
    def inspect(*args, **kwargs):
        (retained["board"] / "results/output_0001.bin").write_bytes(b"changed")
        return {"status": "UNAVAILABLE"}
    module.inspect_draft_precision = inspect
    with pytest.raises(ValueError, match="SHA256/size mismatch"):
        subject.diagnose_draft_numerics(retained["job"])
    assert not (retained["job"] / "numerical-diagnostic").exists()


@pytest.mark.parametrize("fault", ["symlink-directory", "symlink-report", "unowned-file", "unowned-report"])
def test_diagnostic_never_overwrites_unowned_paths(retained, inspection, tmp_path, fault):
    folder = retained["job"] / "numerical-diagnostic"
    outside = tmp_path / "unowned"
    outside.mkdir()
    (outside / "report.json").write_text("keep")
    if fault == "symlink-directory":
        folder.symlink_to(outside, target_is_directory=True)
    else:
        folder.mkdir()
        if fault == "symlink-report": (folder / "report.json").symlink_to(outside / "report.json")
        elif fault == "unowned-file": (folder / "other.txt").write_text("keep")
        else: write_json(folder / "report.json", {"artifact_kind": "another-tool", "job": str(retained["job"])})
    with pytest.raises(ValueError):
        subject.diagnose_draft_numerics(retained["job"])
    assert (outside / "report.json").read_text() == "keep"


def test_hidden_reader_rejects_oversized_payload_before_access(tmp_path):
    with pytest.raises(ValueError, match="bounded payload size"):
        subject._read_hidden(tmp_path, {"bytes": subject.MAX_HIDDEN_BYTES + 1})


def test_output_interpretation_reports_fp32_misread_signature_without_changing_bytes():
    original = np.asarray([1., 2., 3., -.5], dtype="<f4")
    half = original.view("<f2")
    before = half.tobytes()
    native = {"io_diagnostic": {"outputs": [
        {"index": 0, "logical_bytes": 60, "allocated_bytes_before": 64,
         "allocated_bytes_after": 64, "object_same": True, "buffer_same": True},
        {"index": 1, "logical_bytes": 16, "allocated_bytes_before": 32,
         "allocated_bytes_after": 32, "object_same": True, "buffer_same": True}]}}
    report = subject._output_interpretation(half, native)
    assert report["native_io_outputs"] == native["io_diagnostic"]["outputs"]
    low, high = report["fp16_half_word_lanes"]
    assert low["rms"] == 0
    assert high["rms"] == pytest.approx(np.sqrt(np.mean([1.875**2, 2.**2, 2.125**2, 1.75**2])))
    assert high["fp32_hypothesis_word"] == "high16"
    assert low["finite_abs_gt_32_count"] == high["finite_abs_gt_32_count"] == 0
    viewed = report["same_bytes_as_little_endian_fp32"]
    assert viewed["finite_count"] == viewed["elements"] == 4
    assert viewed["nonfinite_count"] == 0
    assert viewed["finite_min"] == -.5 and viewed["finite_max"] == 3.
    assert viewed["rms"] == pytest.approx(np.sqrt(np.mean(original.astype(np.float64)**2)))
    assert "runtime dtype nor a root cause" in report["hypothesis_note"]
    assert "status" not in report
    assert half.tobytes() == before


def test_output_interpretation_counts_large_half_words_in_both_lanes_without_inference():
    half = np.asarray([257.75, -257.75, -257.75, 257.75, 1., 0.], dtype="<f2")
    report = subject._output_interpretation(half, {})
    for lane in report["fp16_half_word_lanes"]:
        assert lane["positive_257_75_word_count"] == 1
        assert lane["negative_257_75_word_count"] == 1
        assert lane["finite_abs_gt_32_count"] == 2
    assert report["native_io_outputs"] == []
    assert report["declared_output_dtype"] == "float16"
    assert report["comparison_interpretation"] == "UNCHANGED"
    assert report["same_bytes_as_little_endian_fp32"]["elements"] == 3
    json.dumps(report, allow_nan=False)


def test_output_interpretation_keeps_nonfinite_and_partial_words_json_safe():
    half = np.asarray([np.inf, 0, np.nan], dtype="<f2")
    report = subject._output_interpretation(half, {})
    assert report["fp16_half_word_lanes"][0]["nonfinite_count"] == 2
    assert report["fp16_half_word_lanes"][0]["rms"] is None
    assert not report["same_bytes_as_little_endian_fp32"]["available"]
    paired = np.asarray([0x0000, 0x7F80, 0x0001, 0x7FC0], dtype="<u2").view("<f2")
    viewed = subject._output_interpretation(paired, {})["same_bytes_as_little_endian_fp32"]
    assert viewed["nonfinite_count"] == 2 and viewed["finite_count"] == 0
    assert viewed["rms"] is viewed["finite_min"] is viewed["finite_max"] is None
    json.dumps({"partial": report, "paired": viewed}, allow_nan=False)
