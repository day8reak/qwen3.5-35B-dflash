"""Draft job sequencing/resume checks; mocked tools are not device evidence."""
import json
from pathlib import Path
import runpy

import pytest

from qwen35_moe_dflash.artifacts import save_json
from qwen35_moe_dflash.hashing import sha256_file


@pytest.fixture
def job(tmp_path, monkeypatch):
    from qwen35_moe_dflash import draft_standalone, draft_reference, draft_board, existing_air

    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    for name in ("CPRO_DDK_ROOT", "CPRO_DDK_ID", "CANN_ROOT", "DDK_PATH"):
        monkeypatch.delenv(name, raising=False)
    target, draft = tmp_path / "target", tmp_path / "draft"
    target.mkdir(); draft.mkdir()
    calls, failures = [], set()
    config = {"num_hidden_layers": 6}

    def prepare(folder, length, **kwargs):
        calls.append("prepared")
        assert kwargs["target_path"] == target and kwargs["draft_path"] == draft
        folder.mkdir()
        save_json(folder / "prepared.json", {"status": "PREPARED", "tiny": False,
                  "draft_config": config, "sequence_length": length})

    def reference(prepared, folder, **kwargs):
        calls.append("fixture")
        assert kwargs["valid_length"] == 6
        folder.mkdir()
        save_json(folder / "report.json", {"status": "REFERENCE_COMPLETE",
                  "expected_top1": {"path": "top1.bin"}, "expected_hidden": {"path": "hidden.bin"}})

    def file_record(path, root):
        return {"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                "sha256": sha256_file(path)}

    def export(prepared, folder, ge, **kwargs):
        calls.append("air")
        folder.mkdir()
        air = folder / "draft.air"
        air.write_bytes(b"fixture-draft-air")
        save_json(folder / "manifest.json", {"status": "EXPORTED",
            "export_contract": draft_standalone.CURRENT_DRAFT_EXPORT_CONTRACT, "graphs": [
            {"name": "draft", "air": "draft.air", "files": [file_record(air, folder)]}]})

    def compile_air(source, folder, ddk, **kwargs):
        calls.append("compilation")
        assert source.name == "draft.air" and kwargs["input_layout"] == "air-directory"
        assert kwargs["save_weights_as_external_data"] is True
        assert kwargs["use_origin_format"] is True and kwargs["weight_data_type"] == "FP16"
        assert "input_shape" not in kwargs  # never the original 40-layer profile
        folder.mkdir()
        if "compilation" in failures:
            (folder / "compile.log").write_text("ERROR: fixture compile failure\n")
            return {"status": "FAILED", "error": "fixture compile failure"}
        model = folder / "model.omc"
        model.write_bytes(b"fixture-compiled-draft")
        result = {"status": "COMPILED", "ddk_id": "test-ddk", "omc": file_record(model, folder),
                  "requested_options": {"use_origin_format": "true", "weight_data_type": "FP16",
                                        "limit_partition_size": "false", "save_weights_as_external_data": "true"}}
        save_json(folder / "report.json", result)
        return result

    def bundle(compiled, fixture, folder, **kwargs):
        calls.append("bundle")
        assert kwargs["config"] == config and kwargs["export_dir"].name == "air"
        folder.mkdir()
        save_json(folder / "manifest.json", {"status": "PREPARED"})

    def board(bundle, folder, **kwargs):
        calls.append("board")
        folder.mkdir()
        return ({"status": "FAILED", "error": "fixture HDC disconnected", "device_execution": "NOT_RUN"}
                if "board" in failures else {"status": "EXECUTION_COMPLETE", "device_execution": "PASSED",
                                             "numerical_validation": "TOP1_MATCHED"})

    monkeypatch.setattr(draft_standalone, "prepare_draft", prepare)
    monkeypatch.setattr(draft_standalone, "export_draft", export)
    monkeypatch.setattr(draft_reference, "prepare_draft_reference", reference)
    monkeypatch.setattr(existing_air, "compile_existing_air", compile_air)
    monkeypatch.setattr(draft_board, "prepare_draft_bundle", bundle)
    monkeypatch.setattr(draft_board, "execute_draft_bundle", board)
    main = runpy.run_path(str(Path(__file__).resolve().parents[1] / "targets/cpro/scripts/run_full_draft.py"))["main"]
    common = ["--ge-root", str(tmp_path / "ge"), "--ddk-root", str(tmp_path / "ddk"), "--ddk-id", "test-ddk"]
    fresh = ["--target-weights", str(target), "--draft-weights", str(draft), *common]
    return main, fresh, common, calls, failures


def _make_legacy_air(previous, *, contract=None):
    """Emulate receipts genuinely produced by an older exporter, not corruption."""
    manifest_path = previous / "air/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if contract is None:
        manifest.pop("export_contract")
    else:
        manifest["export_contract"] = contract
    manifest_path.write_text(json.dumps(manifest))
    receipt_path = previous / "completed-air.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["manifest"].update(bytes=manifest_path.stat().st_size, sha256=sha256_file(manifest_path))
    receipt_path.write_text(json.dumps(receipt))
    result_path = previous / "result.json"
    result = json.loads(result_path.read_text())
    result["components"]["air"] = receipt
    result_path.write_text(json.dumps(result))
    return result


def test_complete_job_only_runs_draft_pipeline(job, tmp_path):
    main, fresh, _, calls, _ = job
    root = tmp_path / "job"
    assert main([*fresh, "--output", str(root)]) == 0
    assert calls == ["prepared", "fixture", "air", "compilation", "bundle", "board"]
    result = json.loads((root / "result.json").read_text())
    assert result["status"] == "EXECUTION_COMPLETE" and result["target_execution"] == "NOT_RUN"
    assert result["runtime_placement"] == "UNVERIFIED"


def test_hdc_failure_resume_reuses_compilation_and_fixtures(job, tmp_path):
    main, fresh, common, calls, failures = job
    previous = tmp_path / "failed"
    failures.add("board")
    assert main([*fresh, "--output", str(previous)]) == 1
    failures.clear(); calls.clear()
    assert main([*common, "--resume", str(previous), "--output", str(tmp_path / "resumed")]) == 0
    assert calls == ["board"]


@pytest.mark.parametrize("native_reason", [True, False])
def test_board_failure_reports_native_cause_or_evidence_error_without_retry(job, tmp_path, monkeypatch, capsys, native_reason):
    from qwen35_moe_dflash import draft_board
    main, fresh, _, calls, _ = job
    def failed_board(bundle, folder, **kwargs):
        calls.append("board")
        folder.mkdir()
        (folder / "draft.stderr").write_text("ERROR: mock HIAI load failed with code 17\n")
        report = {"status": "FAILED", "stage": "execute", "device_execution": "FAILED" if native_reason else "UNVERIFIED",
                  "numerical_validation": "NOT_EVALUATED", "remote_exitcode": 1,
                  "retrieval_errors": []}
        if native_reason:
            report.update(error="mock HIAI load failed with code 17", failure_stage="LOAD_OR_SETUP", native_status="RUNTIME_FAILED")
            report["hilog"] = {"status": "CAPTURED", "summary_lines": ["L17: HIAI fixture diagnostic candidate"]}
        else:
            report["comparison_error"] = "ValueError: native runtime identity mismatch"
        return report
    monkeypatch.setattr(draft_board, "execute_draft_bundle", failed_board)
    root = tmp_path / "board-failed"
    assert main([*fresh, "--output", str(root)]) == 1
    assert calls == ["prepared", "fixture", "air", "compilation", "bundle", "board"]
    report = json.loads((root / "result.json").read_text())
    assert report["stage"] == "board" and report["board_stage"] == "execute"
    assert report["remote_exitcode"] == 1
    assert report["error"] == ("mock HIAI load failed with code 17" if native_reason else "ValueError: native runtime identity mismatch")
    if native_reason:
        assert report["failure_stage"] == "LOAD_OR_SETUP"
    text = capsys.readouterr().out
    assert "draft.stderr" in text and "ERROR: mock HIAI load failed with code 17" in text
    if native_reason:
        assert "draft-hilog.txt" in text and "L17: HIAI fixture diagnostic candidate" in text


def test_interruption_receipts_resume_without_final_result(job, tmp_path):
    main, fresh, common, calls, _ = job
    previous = tmp_path / "interrupted"
    assert main([*fresh, "--compile-only", "--output", str(previous)]) == 0
    (previous / "result.json").unlink()  # emulate interruption before final parent report
    calls.clear()
    assert main([*common, "--resume", str(previous), "--output", str(tmp_path / "resumed")]) == 0
    assert calls == ["bundle", "board"]


def test_compile_failure_never_reaches_board_and_retry_reuses_air(job, tmp_path):
    main, fresh, common, calls, failures = job
    previous = tmp_path / "compile-failed"
    failures.add("compilation")
    assert main([*fresh, "--output", str(previous)]) == 1
    assert calls == ["prepared", "fixture", "air", "compilation"]
    failures.clear(); calls.clear()
    assert main([*common, "--resume", str(previous), "--output", str(tmp_path / "retry")]) == 0
    assert calls == ["compilation", "bundle", "board"]


@pytest.fixture
def retained_job(job, tmp_path, monkeypatch):
    """Model the legacy post-compile error; file verification has separate tests."""
    from qwen35_moe_dflash import retained_air
    main, fresh, _, calls, failures = job
    previous = tmp_path / "retained"
    failures.add("compilation")
    assert main([*fresh, "--output", str(previous)]) == 1
    options = {"use_origin_format": "true", "weight_data_type": "FP16",
               "limit_partition_size": "false", "save_weights_as_external_data": "true"}
    save_json(previous / "compilation/report.json", {
        "artifact_kind": "qwen35-existing-air-compilation", "status": "FAILED",
        "stage": "output-check", "error": retained_air.LEGACY_OUTPUT_ERROR,
        "requested_options": options})
    failures.clear(); calls.clear()

    def recover(source, destination, *, export_dir, expected_options, **kwargs):
        calls.append("recovery")
        assert source == previous / "compilation" and export_dir == previous / "air"
        assert expected_options == options
        if "recovery" in failures:
            raise ValueError("retained external weight SHA256 mismatch")
        model = destination / "compile-work/generated/model/model.omc"
        model.parent.mkdir(parents=True)
        model.write_bytes(b"retained-model")
        report = {"status": "COMPILED", "ddk_id": "test-ddk", "requested_options": options,
                  "recovery": {"source": str(source), "compiler_execution": "NOT_RUN"},
                  "omc": {"path": str(model.relative_to(destination)),
                          "bytes": model.stat().st_size, "sha256": sha256_file(model)}}
        save_json(destination / "report.json", report)
        return report
    monkeypatch.setattr(retained_air, "recover_retained_air", recover)
    return previous


@pytest.mark.parametrize("compile_only", [False, True])
def test_resume_recovers_retained_omc_without_ge_or_compiler(job, retained_job, tmp_path, compile_only):
    main, _, _, calls, _ = job
    before = {p: p.read_bytes() for p in retained_job.rglob("*") if p.is_file()}
    output = tmp_path / "recovered"
    args = ["--resume", str(retained_job), "--output", str(output)]
    assert main(args + (["--compile-only"] if compile_only else [])) == 0
    assert calls == ["recovery"] + ([] if compile_only else ["bundle", "board"])
    result = json.loads((output / "result.json").read_text())
    assert result["compilation_recovery"]["compiler_execution"] == "NOT_RUN"
    assert result["status"] == ("COMPILED" if compile_only else "EXECUTION_COMPLETE")
    assert all(p.read_bytes() == data for p, data in before.items())
    calls.clear()
    assert main(["--resume", str(output), "--compile-only", "--output", str(tmp_path / "again")]) == 0
    assert calls == []


def test_bad_retained_weight_never_recompiles_or_runs_board(job, retained_job, tmp_path):
    main, _, common, calls, failures = job
    failures.add("recovery")
    output = tmp_path / "refused"
    assert main([*common, "--resume", str(retained_job), "--output", str(output)]) == 1
    assert calls == ["recovery"]
    result = json.loads((output / "result.json").read_text())
    assert "SHA256 mismatch" in result["error"] and result["device_execution"] == "NOT_RUN"


def test_prepare_only_does_not_copy_retained_outputs(job, retained_job, tmp_path):
    main, _, _, calls, _ = job
    assert main(["--resume", str(retained_job), "--prepare-only", "--output", str(tmp_path / "prepared-only")]) == 0
    assert calls == []


def test_obsolete_air_never_recovers_incompatible_compiler_outputs(job, retained_job, tmp_path):
    main, _, common, calls, _ = job
    _make_legacy_air(retained_job)
    assert main([*common, "--resume", str(retained_job), "--output", str(tmp_path / "reexport")]) == 0
    assert calls == ["air", "compilation", "bundle", "board"]


@pytest.mark.parametrize("interrupted", [False, True])
def test_old_compile_options_reuse_air_and_rebuild_only_compilation_bundle(job, tmp_path, interrupted):
    main, fresh, common, calls, _ = job
    previous = tmp_path / "old-profile"
    assert main([*fresh, "--output", str(previous)]) == 0
    report_path = previous / "compilation/report.json"
    report = json.loads(report_path.read_text())
    report["requested_options"].pop("use_origin_format")
    report["requested_options"].pop("weight_data_type")
    report_path.write_text(json.dumps(report))
    receipt_path = previous / "completed-compilation.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["manifest"].update(bytes=report_path.stat().st_size, sha256=sha256_file(report_path))
    receipt_path.write_text(json.dumps(receipt))
    old = json.loads((previous / "result.json").read_text())
    old["components"]["compilation"] = receipt
    (previous / "result.json").write_text(json.dumps(old))
    if interrupted:
        (previous / "result.json").unlink()
    before = {p: p.read_bytes() for p in previous.rglob("*") if p.is_file()}
    calls.clear()
    resumed = tmp_path / "new-profile"
    # Successful AIR means host GE is not needed for this retry.
    without_ge = common[2:]
    assert main([*without_ge, "--resume", str(previous), "--output", str(resumed)]) == 0
    assert calls == ["compilation", "bundle", "board"]
    result = json.loads((resumed / "result.json").read_text())
    assert all(result["components"][name] == old["components"][name] for name in ("prepared", "fixture", "air"))
    invalidation = result["resume_invalidations"][0]
    assert invalidation["reason"] == "DRAFT_COMPILE_OPTIONS_CHANGED"
    assert set(invalidation["components"]) == {"compilation", "bundle"}
    assert invalidation["preserved"] == ["prepared", "fixture", "air"]
    assert all(p.read_bytes() == data for p, data in before.items())


@pytest.mark.parametrize("contract", [None, "legacy-absolute-fileconstant", "cpro-fileconstant-relative-int-v1",
                                    "cpro-fileconstant-relative-int-fc-conv2d-v2"])
def test_old_air_after_compile_failure_reexports_without_loading_weights_or_reference(job, tmp_path, contract):
    main, fresh, common, calls, failures = job
    previous = tmp_path / "old-compile-failed"
    failures.add("compilation")
    assert main([*fresh, "--output", str(previous)]) == 1
    old = _make_legacy_air(previous, contract=contract)
    previous_bytes = {path: path.read_bytes() for path in previous.rglob("*") if path.is_file()}
    failures.clear(); calls.clear()
    resumed = tmp_path / "resumed"
    assert main([*common, "--resume", str(previous), "--output", str(resumed)]) == 0
    assert calls == ["air", "compilation", "bundle", "board"]
    result = json.loads((resumed / "result.json").read_text())
    assert result["components"]["prepared"] == old["components"]["prepared"]
    assert result["components"]["fixture"] == old["components"]["fixture"]
    invalidation = result["resume_invalidations"][0]
    assert invalidation["reason"] == "OBSOLETE_DRAFT_EXPORT_CONTRACT"
    assert invalidation["components"] == {"air": old["components"]["air"]}
    assert invalidation["preserved"] == ["prepared", "fixture"]
    assert invalidation["previous_export_contract"] == contract
    assert all(path.read_bytes() == payload for path, payload in previous_bytes.items())


def test_old_air_receipts_invalidate_dependent_compilation_and_bundle_after_interruption(job, tmp_path):
    main, fresh, common, calls, _ = job
    previous = tmp_path / "old-complete"
    assert main([*fresh, "--output", str(previous)]) == 0
    old = _make_legacy_air(previous)
    (previous / "result.json").unlink()
    calls.clear()
    resumed = tmp_path / "resumed"
    assert main([*common, "--resume", str(previous), "--output", str(resumed)]) == 0
    assert calls == ["air", "compilation", "bundle", "board"]
    result = json.loads((resumed / "result.json").read_text())
    assert result["resume_invalidations"][0]["components"] == {
        name: old["components"][name] for name in ("air", "compilation", "bundle")}


def test_int64_broadcast_air_rebuilds_after_board_failure_preserving_reference(job, tmp_path):
    main, fresh, common, calls, failures = job
    previous = tmp_path / "compiled-old-broadcast"
    failures.add("board")
    assert main([*fresh, "--output", str(previous)]) == 1
    old = _make_legacy_air(previous, contract="cpro-fileconstant-relative-int-fc-conv2d-v2")
    before = {path: path.read_bytes() for path in previous.rglob("*") if path.is_file()}
    failures.clear(); calls.clear()
    resumed = tmp_path / "int32-broadcast"
    assert main([*common, "--resume", str(previous), "--output", str(resumed)]) == 0
    assert calls == ["air", "compilation", "bundle", "board"]
    result = json.loads((resumed / "result.json").read_text())
    assert all(result["components"][name] == old["components"][name] for name in ("prepared", "fixture"))
    invalidation, = result["resume_invalidations"]
    assert invalidation["reason"] == "OBSOLETE_DRAFT_EXPORT_CONTRACT"
    assert set(invalidation["components"]) == {"air", "compilation", "bundle"}
    assert invalidation["required_export_contract"].endswith("broadcast-i32-v3")
    assert all(path.read_bytes() == content for path, content in before.items())


def test_obsolete_air_still_requires_verified_old_manifest_before_rebuild(job, tmp_path):
    main, fresh, common, calls, _ = job
    previous = tmp_path / "old-air-corrupted"
    assert main([*fresh, "--compile-only", "--output", str(previous)]) == 0
    _make_legacy_air(previous)
    with (previous / "air/manifest.json").open("a") as stream:
        stream.write(" ")
    calls.clear()
    assert main([*common, "--resume", str(previous), "--output", str(tmp_path / "resumed")]) == 1
    assert calls == []


def test_snapshot_console_reports_bounded_progress(job, tmp_path, monkeypatch, capsys):
    main, fresh, _, calls, _ = job
    from qwen35_moe_dflash import existing_air
    original = existing_air.compile_existing_air
    def observed_compile(*args, **kwargs):
        for index in range(23):
            kwargs["progress"]({"stage": "input-snapshot", "message": "Snapshotting input",
                "completed": index, "total": 23, "source_relative": f"weight_{index}.bin", "bytes": 4096})
        return original(*args, **kwargs)
    monkeypatch.setattr(existing_air, "compile_existing_air", observed_compile)
    assert main([*fresh, "--compile-only", "--output", str(tmp_path / "progress")]) == 0
    rows = [line for line in capsys.readouterr().out.splitlines() if line.startswith("full-draft: Snapshotting input")]
    assert len(rows) == 4
    assert "1/23: weight_0.bin (4096 bytes)" in rows[0]
    assert "23/23: weight_22.bin (4096 bytes)" in rows[-1]


def test_resume_rejects_changed_air_before_board(job, tmp_path):
    main, fresh, common, calls, _ = job
    previous = tmp_path / "compiled"
    assert main([*fresh, "--compile-only", "--output", str(previous)]) == 0
    (previous / "air/draft.air").write_bytes(b"mutated-air")
    calls.clear()
    assert main([*common, "--resume", str(previous), "--output", str(tmp_path / "retry")]) == 1
    assert calls == []


def test_prepare_only_needs_neither_ddk_nor_board(job, tmp_path):
    main, fresh, _, calls, _ = job
    assert main([*fresh[:4], "--prepare-only", "--output", str(tmp_path / "prepared")]) == 0
    assert calls == ["prepared", "fixture"]


def test_inspect_only_dispatches_without_any_model_or_tool_execution(job, tmp_path, monkeypatch):
    main, _, _, calls, _ = job
    import sys
    import types
    previous = tmp_path / "retained-job"
    output = tmp_path / "inspection"
    library = tmp_path / "selected-schema.so"
    inspected = []
    def inspect(source, destination, *, schema_library):
        inspected.append((source, destination, schema_library))
        destination.mkdir()
        (destination / "report.txt").write_text("Existing compiler evidence; no execution.\n")
        return {"status": "INSPECTED"}
    monkeypatch.setitem(sys.modules, "qwen35_moe_dflash.draft_compile_inspect",
                        types.SimpleNamespace(inspect_draft_compile=inspect))
    assert main(["--inspect-only", "--resume", str(previous), "--output", str(output),
                 "--schema-library", str(library)]) == 0
    assert calls == [] and inspected == [(str(previous), output, str(library))]


@pytest.mark.parametrize("extra", [["--inspect-only"], ["--schema-library", "schema.so"],
    ["--inspect-only", "--resume", "retained", "--sequence-length", "64"]])
def test_invalid_inspection_options_do_not_run_pipeline(job, tmp_path, extra):
    main, _, _, calls, _ = job
    assert main([*extra, "--output", str(tmp_path / "unused")]) == 1
    assert calls == [] and not (tmp_path / "unused").exists()


@pytest.mark.parametrize("extra", [["--sequence-length", "0"], ["--valid-length", "65"], ["--timeout-seconds", "0"]])
def test_invalid_job_settings_fail_before_execution(job, tmp_path, extra):
    main, fresh, _, calls, _ = job
    assert main([*fresh, *extra, "--output", str(tmp_path / "invalid")]) == 1
    assert calls == []


def test_resume_refuses_changed_inputs_and_preserves_prior_run(job, tmp_path):
    main, fresh, common, calls, _ = job
    previous = tmp_path / "compiled"
    assert main([*fresh, "--compile-only", "--output", str(previous)]) == 0
    original = (previous / "result.json").read_bytes()
    calls.clear()
    assert main([*common, "--resume", str(previous), "--sequence-length", "128"]) == 1
    assert calls == [] and (previous / "result.json").read_bytes() == original
