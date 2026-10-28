"""Recovery uses only retained evidence, never a compiler or board process."""
import json
from pathlib import Path
import subprocess

import pytest

from qwen35_moe_dflash import retained_air as module
from qwen35_moe_dflash.existing_air import _generated_inventory, _snapshot, _stat
from qwen35_moe_dflash.hashing import sha256_file


OPTIONS = {"limit_partition_size": "false", "save_weights_as_external_data": "true",
           "use_origin_format": "true", "weight_data_type": "FP16"}


def _write(path, value):
    path.write_text(json.dumps(value))


def _file(path, root):
    return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


@pytest.fixture
def retained(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("recovery must not run tools"))
    exported, old, new = tmp_path / "exported", tmp_path / "old", tmp_path / "recovered"
    (exported / "weights").mkdir(parents=True)
    (exported / "draft.air").write_bytes(b"MOCK AIR")
    (exported / "weights/fc.bin").write_bytes(b"MOCK FP16 WEIGHT")
    files = [_file(exported / relative, exported) for relative in ("draft.air", "weights/fc.bin")]
    _write(exported / "manifest.json", {"artifact_kind": "qwen35-cpro-draft-air", "status": "EXPORTED",
        "graphs": [{"name": "draft", "air": "draft.air", "files": files}]})
    snapshots = []
    for row in files:
        destination = old / "compile-work/generated" / row["path"]
        record = _snapshot(exported / row["path"], destination)
        record.update(path=destination.relative_to(old).as_posix(), source_relative=row["path"],
                      role="air" if row["path"].endswith(".air") else "neighbor_data_candidate")
        snapshots.append(record)
    (old / "compile-work/generated/model").mkdir()
    (old / "compile-work/generated/model/model.omc").write_bytes(b"MOCK COMPILED OMC")
    (old / "compile-work/generated/model/SubGraph_0.weight").write_bytes(b"MOCK EXTERNAL WEIGHT")
    (old / "compile-work/help").mkdir()
    (old / "compile-work/help/context.bin").write_bytes(b"MOCK HELP CONTEXT")
    generated, context, _ = _generated_inventory(old, snapshots)
    # The old collector mislabeled nested model.omc as external data.
    for row in generated:
        row["role"] = "external_data_candidate"
    (old / "compile.log").write_text('I/OMG_TOOL main.cpp main(22)::"OMG generate offline model success."\n')
    command = ["/mock/ddk/omg", "--framework=1", f"--model={old}/compile-work/generated/draft.air",
               "--output=model", "--target=omc", "--platform=kirinx90"]
    command += [f"--{key}={value}" for key, value in OPTIONS.items()]
    _write(old / "compile-command.json", {"argv": command, "cwd": str(old / "compile-work/generated")})
    report = {"artifact_kind": "qwen35-existing-air-compilation", "schema_version": 1,
        "status": "FAILED", "stage": "output-check", "error": module.LEGACY_OUTPUT_ERROR,
        "model_compilation": "FAILED", "platform": "kirinx90", "ddk_id": "mock-ddk",
        "ddk_root": "/mock/ddk", "omg_path": "/mock/ddk/omg", "requested_options": OPTIONS,
        "air_source": str(exported / "draft.air"), "air": {**files[0], "path": str(exported / "draft.air")},
        "input_snapshots": snapshots, "generated_artifacts": generated, "compiler_work_artifacts": context,
        "compile_log_sha256": sha256_file(old / "compile.log"), "device_execution": "NOT_RUN",
        "numerical_validation": "NOT_RUN", "device_ready": False,
        "source_preservation": {"status": "UNCHANGED_BY_STAT", "changed": []}}
    _write(old / "report.json", report)
    return exported, old, new, report


def _recover(retained, **kwargs):
    exported, old, new, _ = retained
    return module.recover_retained_air(old, new, export_dir=exported,
                                      expected_options=OPTIONS, ddk_id="mock-ddk", **kwargs)


@pytest.mark.parametrize("observed", [False, True])
def test_recovers_nested_omc_sidecar_and_bound_inputs_without_compiling_or_overwriting(retained, observed):
    exported, old, new, report = retained
    if observed:
        report["returncode"] = 0
        _write(old / "report.json", report)
    before = {path: (path.read_bytes(), _stat(path)) for root in (old, exported)
              for path in root.rglob("*") if path.is_file()}
    result = _recover(retained, ddk_root="/mock/ddk")
    assert result["status"] == "COMPILED" and result["model_compilation"] == "PASSED"
    assert result["recovery"]["compiler_execution"] == "NOT_RUN"
    assert result["returncode_evidence"] == ("OBSERVED" if observed else
                                            "INFERRED_FROM_LEGACY_POSTPROCESS_STAGE_AND_SUCCESS_LOG")
    assert result["device_execution"] == result["numerical_validation"] == "NOT_RUN"
    assert result["device_ready"] is False
    assert result["omc"]["path"] == "compile-work/generated/model/model.omc"
    assert result["primary_omc_layout"] == "model/model.omc"
    assert {row["path"] for row in result["external_data_candidates"]} == {
        "compile-work/generated/model/SubGraph_0.weight"}
    for group in ("input_snapshots", "generated_artifacts", "compiler_work_artifacts"):
        for row in result[group]:
            assert (new / row["path"]).read_bytes() == (old / row["path"]).read_bytes()
            assert sha256_file(new / row["path"]) == row["sha256"]
            assert _stat(new / row["path"])[:2] != _stat(old / row["path"])[:2]
            if group == "input_snapshots":
                assert row["snapshot_source"] == str(old / row["path"])
                assert row["snapshot_method"] in {"COPY", "REFLINK"}
    assert json.loads((new / "recovery/original-report.json").read_text()) == report
    assert before == {path: (path.read_bytes(), _stat(path)) for path in before}
    assert json.loads((old / "report.json").read_text())["status"] == "FAILED"


@pytest.mark.parametrize("fault", ["returncode", "bool_returncode", "stage", "error", "options", "ddk",
                                  "platform", "preservation", "audit_error", "inventory_error",
                                  "omc_tamper", "sidecar_tamper", "extra_artifact", "missing_sidecar",
                                  "duplicate_omc", "symlink", "source_changed", "snapshot_changed",
                                  "export_changed", "log_changed", "no_terminal_success", "command_changed",
                                  "duplicate_snapshot", "tbe_rejected"])
def test_recovery_rejects_unrelated_failures_drift_and_incomplete_identity(retained, fault):
    exported, old, new, report = retained
    if fault == "returncode": report["returncode"] = 1
    elif fault == "bool_returncode": report["returncode"] = False
    elif fault == "stage": report["stage"] = "omg"
    elif fault == "error": report["error"] = "CalledProcessError: OMG failed"
    elif fault == "options": report["requested_options"] = {**OPTIONS, "weight_data_type": "FP32"}
    elif fault == "ddk": report["ddk_id"] = "other-ddk"
    elif fault == "platform": report["platform"] = "other-chip"
    elif fault == "preservation": report["source_preservation"]["status"] = "CHANGED"
    elif fault in {"audit_error", "inventory_error"}:
        report["audit_error" if fault == "audit_error" else "artifact_inventory_error"] = "failure"
    elif fault in {"omc_tamper", "sidecar_tamper", "missing_sidecar"}:
        path = old / "compile-work/generated/model" / ("model.omc" if fault == "omc_tamper" else "SubGraph_0.weight")
        path.unlink() if fault == "missing_sidecar" else path.write_bytes(b"CORRUPT")
    elif fault == "extra_artifact": (old / "compile-work/generated/model/extra.weight").write_bytes(b"UNBOUND")
    elif fault == "duplicate_omc": (old / "compile-work/generated/model.omc").write_bytes(b"OTHER OMC")
    elif fault == "symlink":
        path = old / "compile-work/generated/model/SubGraph_0.weight"
        path.unlink()
        path.symlink_to(exported / "weights/fc.bin")
    elif fault == "source_changed": (exported / "weights/fc.bin").write_bytes(b"CHANGED")
    elif fault == "snapshot_changed":
        path = old / "compile-work/generated/weights/fc.bin"
        path.chmod(0o644)
        path.write_bytes(b"CHANGED")
    elif fault == "export_changed":
        manifest = json.loads((exported / "manifest.json").read_text())
        manifest["graphs"][0]["files"][1]["sha256"] = "0" * 64
        _write(exported / "manifest.json", manifest)
    elif fault in {"log_changed", "no_terminal_success"}:
        (old / "compile.log").write_text("OMG generate offline model failed\n")
        if fault == "no_terminal_success": report["compile_log_sha256"] = sha256_file(old / "compile.log")
    elif fault == "command_changed":
        command = json.loads((old / "compile-command.json").read_text())
        command["argv"].append("--unapproved=true")
        _write(old / "compile-command.json", command)
    elif fault == "duplicate_snapshot": report["input_snapshots"].append(report["input_snapshots"][0])
    elif fault == "tbe_rejected":
        report["tbe_compatibility"] = {"compiler_invocation": "ATTEMPTED", "audit": {"status": "REJECTED"}}
    _write(old / "report.json", report)
    with pytest.raises((ValueError, FileNotFoundError)):
        _recover(retained)
    assert not new.exists(), "reject invalid evidence before copying gigabytes"


def test_tbe_audit_is_revalidated_without_invoking_tbe(retained):
    _, old, _, report = retained
    metadata = {"audit_log": str(old / "tbe-compat/worker-audit.jsonl")}
    report["tbe_compatibility"] = {"compiler_invocation": "ATTEMPTED", "compile": "FAILED",
                                   "metadata": metadata, "audit": module.verify_tbe_compatibility(metadata)}
    _write(old / "report.json", report)
    result = _recover(retained)
    assert result["tbe_compatibility"]["audit"]["status"] == "NOT_OBSERVED"
    assert result["tbe_compatibility"]["compile"] == "PASSED"


def test_recovery_does_not_publish_success_after_copy_corruption(retained, monkeypatch):
    _, _, new, _ = retained
    snapshot = module._snapshot
    def corrupt(source, destination):
        record = snapshot(source, destination)
        if source.name == "SubGraph_0.weight": record["sha256"] = "0" * 64
        return record
    monkeypatch.setattr(module, "_snapshot", corrupt)
    with pytest.raises(ValueError, match="changed during recovery"):
        _recover(retained)
    assert not (new / "report.json").exists()
