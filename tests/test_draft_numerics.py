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
