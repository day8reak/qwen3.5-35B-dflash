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
        save_json(folder / "prepared.json", {"artifact_kind": "qwen35-draft-ge-prepared", "status": "PREPARED", "tiny": False,
                  "draft_config": config, "sequence_length": length})

    def reference(prepared, folder, **kwargs):
        calls.append("fixture")
        assert kwargs["valid_length"] == 6
        folder.mkdir()
        save_json(folder / "report.json", {"artifact_kind": "qwen35-draft-reference", "status": "REFERENCE_COMPLETE",
                  "expected_top1": {"path": "top1.bin"}, "expected_hidden": {"path": "hidden.bin"}})

    def file_record(path, root):
        return {"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                "sha256": sha256_file(path)}

    def export(prepared, folder, ge, **kwargs):
        calls.append("air")
        folder.mkdir()
        air = folder / "draft.air"
        air.write_bytes(b"fixture-draft-air")
        save_json(folder / "manifest.json", {"artifact_kind": "qwen35-cpro-draft-air", "status": "EXPORTED",
            "export_contract": draft_standalone.CURRENT_DRAFT_EXPORT_CONTRACT, "graphs": [
            {"name": "draft", "air": "draft.air", "files": [file_record(air, folder)]}]})

    def compile_air(source, folder, ddk, **kwargs):
        calls.append("compilation")
        assert source.name == "draft.air" and kwargs["input_layout"] == "air-directory"
        assert type(kwargs["save_weights_as_external_data"]) is bool
        assert kwargs["use_origin_format"] is True and kwargs["weight_data_type"] == "FP16"
        assert "input_shape" not in kwargs  # never the original 40-layer profile
        folder.mkdir()
        if "compilation" in failures:
            (folder / "compile.log").write_text("ERROR: fixture compile failure\n")
            return {"status": "FAILED", "error": "fixture compile failure"}
        model = folder / "model.omc"
        model.write_bytes(b"fixture-compiled-draft")
        result = {"artifact_kind": "qwen35-existing-air-compilation", "status": "COMPILED", "ddk_id": "test-ddk", "omc": file_record(model, folder),
                  "requested_options": {"use_origin_format": "true", "weight_data_type": "FP16",
                                        "limit_partition_size": "false", "save_weights_as_external_data":
                                        str(kwargs["save_weights_as_external_data"]).lower()},
                  "external_data_candidates": []}
        save_json(folder / "report.json", result)
        return result

    def bundle(compiled, fixture, folder, **kwargs):
        calls.append("bundle")
        assert kwargs["config"] == config and kwargs["export_dir"].name == "air"
        folder.mkdir()
        save_json(folder / "manifest.json", {"artifact_kind": "qwen35-standalone-draft-bundle", "status": "PREPARED"})

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


@pytest.mark.parametrize("storage,explicit,expected", [
    ("external", None, "file-v2"), ("embedded", None, "legacy-buffer"),
    ("external", "legacy-buffer", "legacy-buffer"), ("embedded", "file-v2", "file-v2"),
])
def test_board_loader_selection_does_not_change_compiler_options(
        job, tmp_path, monkeypatch, storage, explicit, expected):
    from qwen35_moe_dflash import draft_board
    main, fresh, _, calls, _ = job
    original = draft_board.execute_draft_bundle
    received = []
    def execute(bundle, folder, **kwargs):
        received.append(kwargs["model_load_api"])
        return original(bundle, folder, **kwargs)
    monkeypatch.setattr(draft_board, "execute_draft_bundle", execute)
    root = tmp_path / "selected-loader"
    args = [*fresh, "--weight-storage", storage, "--output", str(root)]
    if explicit is not None:
        args += ["--model-load-api", explicit]
    assert main(args) == 0
    result = json.loads((root / "result.json").read_text())
    assert result["model_load_api"] == expected and received == [expected]
    assert result["compile_options"]["save_weights_as_external_data"] == str(storage == "external").lower()


def test_changing_loader_reuses_all_compiled_payloads_and_replaces_only_board(job, tmp_path, monkeypatch):
    from qwen35_moe_dflash import draft_board
    main, fresh, _, calls, failures = job
    root = tmp_path / "full-draft"
    failures.add("board")
    assert main([*fresh, "--model-load-api", "legacy-buffer"]) == 1
    old = json.loads((root / "result.json").read_text())
    preserved = _stage_payloads(old)
    (root / "board/stale-legacy-result.json").write_text("previous load failed")
    original = draft_board.execute_draft_bundle
    def execute(bundle, folder, **kwargs):
        assert kwargs["model_load_api"] == "file-v2"
        return original(bundle, folder, **kwargs)
    monkeypatch.setattr(draft_board, "execute_draft_bundle", execute)
    failures.clear(); calls.clear()
    assert main(["--model-load-api", "file-v2"]) == 0
    assert calls == ["board"]
    assert not (root / "board/stale-legacy-result.json").exists()
    _assert_payloads_unchanged(preserved)
    result = json.loads((root / "result.json").read_text())
    assert result["model_load_api"] == "file-v2"
    assert result["components"] == old["components"]
    assert [p.name for p in tmp_path.glob("full-draft*")] == ["full-draft"]


def test_automatic_loader_uses_inherited_compilation_storage(job, tmp_path, monkeypatch):
    from qwen35_moe_dflash import draft_board
    main, fresh, _, calls, _ = job
    root = tmp_path / "full-draft"
    assert main([*fresh, "--weight-storage", "embedded", "--compile-only"]) == 0
    original = draft_board.execute_draft_bundle
    def execute(bundle, folder, **kwargs):
        assert kwargs["model_load_api"] == "legacy-buffer"
        return original(bundle, folder, **kwargs)
    monkeypatch.setattr(draft_board, "execute_draft_bundle", execute)
    calls.clear()
    assert main([]) == 0
    assert calls == ["bundle", "board"]
    assert json.loads((root / "result.json").read_text())["model_load_api"] == "legacy-buffer"


def test_external_board_failure_can_switch_to_embedded_without_reexport_or_reference(job, tmp_path):
    main, fresh, common, calls, failures = job
    previous = tmp_path / "external-load-failed"
    failures.add("board")
    assert main([*fresh, "--output", str(previous)]) == 1
    old = json.loads((previous / "result.json").read_text())
    before = {path: path.read_bytes() for path in previous.rglob("*") if path.is_file()}
    failures.clear(); calls.clear()
    embedded = tmp_path / "embedded"
    # Reuse the current AIR: neither GE nor the full source weights are needed.
    assert main([*common[2:], "--resume", str(previous), "--weight-storage", "embedded",
                 "--output", str(embedded)]) == 0
    assert calls == ["compilation", "bundle", "board"]
    result = json.loads((embedded / "result.json").read_text())
    assert result["compile_options"]["save_weights_as_external_data"] == "false"
    assert all(result["components"][name] == old["components"][name]
               for name in ("prepared", "fixture", "air"))
    invalidation, = result["resume_invalidations"]
    assert invalidation["reason"] == "DRAFT_COMPILE_OPTIONS_CHANGED"
    assert set(invalidation["components"]) == {"compilation", "bundle"}
    assert invalidation["required_options"]["save_weights_as_external_data"] == "false"
    assert invalidation["preserved"] == ["prepared", "fixture", "air"]
    assert all(path.read_bytes() == payload for path, payload in before.items())

    calls.clear()
    unchanged = tmp_path / "embedded-resumed"
    assert main(["--resume", str(embedded), "--output", str(unchanged)]) == 0
    assert calls == ["board"]
    resumed = json.loads((unchanged / "result.json").read_text())
    assert resumed["compile_options"]["save_weights_as_external_data"] == "false"
    assert resumed["components"] == result["components"]

    calls.clear()
    external = tmp_path / "explicit-external"
    assert main([*common[2:], "--resume", str(unchanged), "--weight-storage", "external",
                 "--output", str(external)]) == 0
    assert calls == ["compilation", "bundle", "board"]
    restored = json.loads((external / "result.json").read_text())
    assert restored["compile_options"]["save_weights_as_external_data"] == "true"
    assert all(restored["components"][name] == result["components"][name]
               for name in ("prepared", "fixture", "air"))
    assert all(path.read_bytes() == payload for path, payload in before.items())


@pytest.mark.parametrize("storage", ["external", "embedded"])
def test_fresh_storage_option_is_passed_to_compiler_and_recorded(job, tmp_path, storage):
    main, fresh, _, calls, _ = job
    root = tmp_path / storage
    assert main([*fresh, "--weight-storage", storage, "--compile-only", "--output", str(root)]) == 0
    assert calls == ["prepared", "fixture", "air", "compilation"]
    expected = "false" if storage == "embedded" else "true"
    for filename in ("job.json", "result.json"):
        report = json.loads((root / filename).read_text())
        assert report["compile_options"]["save_weights_as_external_data"] == expected
    compiled = json.loads((root / "compilation/report.json").read_text())
    assert compiled["requested_options"]["save_weights_as_external_data"] == expected


@pytest.mark.parametrize("interrupted", [False, True])
def test_embedded_resume_inherits_verified_receipt_instead_of_stale_parent_defaults(job, tmp_path, interrupted):
    main, fresh, _, calls, _ = job
    previous = tmp_path / "embedded-compiled"
    assert main([*fresh, "--weight-storage", "embedded", "--compile-only", "--output", str(previous)]) == 0
    result_path = previous / "result.json"
    if interrupted:
        result_path.unlink()
        parent_path = previous / "job.json"
    else:
        parent_path = result_path
    parent = json.loads(parent_path.read_text())
    parent["compile_options"]["save_weights_as_external_data"] = "true"
    parent_path.write_text(json.dumps(parent))
    calls.clear()
    resumed = tmp_path / "inherited"
    assert main(["--resume", str(previous), "--compile-only", "--output", str(resumed)]) == 0
    assert calls == []
    result = json.loads((resumed / "result.json").read_text())
    assert result["compile_options"]["save_weights_as_external_data"] == "false"


def test_failed_embedded_compilation_keeps_storage_choice_without_completed_receipt(job, tmp_path):
    main, fresh, common, calls, failures = job
    previous = tmp_path / "embedded-compile-failed"
    failures.add("compilation")
    assert main([*fresh, "--weight-storage", "embedded", "--output", str(previous)]) == 1
    assert not (previous / "completed-compilation.json").exists()
    failures.clear(); calls.clear()
    resumed = tmp_path / "embedded-retry"
    assert main([*common[2:], "--resume", str(previous), "--output", str(resumed)]) == 0
    assert calls == ["compilation", "bundle", "board"]
    result = json.loads((resumed / "result.json").read_text())
    assert result["compile_options"]["save_weights_as_external_data"] == "false"


@pytest.mark.parametrize("filename", ["SubGraph_1.weight", "model.weights", "model.omc.data",
                                     "model.weights/part_0.bin"])
@pytest.mark.parametrize("compile_only", [False, True])
def test_embedded_mode_rejects_observed_nonempty_weight_sidecars_before_bundle(
        job, tmp_path, monkeypatch, filename, compile_only):
    from qwen35_moe_dflash import existing_air
    main, fresh, _, calls, _ = job
    original = existing_air.compile_existing_air
    def compile_with_sidecar(source, folder, ddk, **kwargs):
        report = original(source, folder, ddk, **kwargs)
        path = folder / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"unexpected-external-weight")
        report["external_data_candidates"] = [{"path": filename, "bytes": path.stat().st_size,
                                                "sha256": sha256_file(path)}]
        (folder / "report.json").write_text(json.dumps(report))
        return report
    monkeypatch.setattr(existing_air, "compile_existing_air", compile_with_sidecar)
    root = tmp_path / "unexpected-sidecar"
    args = [*fresh, "--weight-storage", "embedded", "--output", str(root)]
    assert main(args + (["--compile-only"] if compile_only else [])) == 1
    assert calls == ["prepared", "fixture", "air", "compilation"]
    report = json.loads((root / "result.json").read_text())
    assert report["device_execution"] == "NOT_RUN"
    assert "embedded" in report["error"].lower() and "weight" in report["error"].lower()


@pytest.mark.parametrize("filename", ["dump_graph/SubGraph_1.weight", "dump_graph/model.omc.data",
                                     "diagnostic.bin"])
def test_embedded_mode_does_not_mistake_generated_diagnostics_for_weight_sidecars(
        job, tmp_path, monkeypatch, filename):
    from qwen35_moe_dflash import existing_air
    main, fresh, _, calls, _ = job
    original = existing_air.compile_existing_air
    def compile_with_diagnostic(source, folder, ddk, **kwargs):
        report = original(source, folder, ddk, **kwargs)
        path = folder / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"compiler-diagnostic")
        report["external_data_candidates"] = [{"path": filename, "bytes": path.stat().st_size,
                                                "sha256": sha256_file(path)}]
        (folder / "report.json").write_text(json.dumps(report))
        return report
    monkeypatch.setattr(existing_air, "compile_existing_air", compile_with_diagnostic)
    assert main([*fresh, "--weight-storage", "embedded", "--output", str(tmp_path / "diagnostics")]) == 0
    assert calls == ["prepared", "fixture", "air", "compilation", "bundle", "board"]


@pytest.mark.parametrize("compile_only", [False, True])
def test_embedded_omc_over_uint32_limit_is_rejected_before_hash_or_board(
        job, tmp_path, monkeypatch, compile_only):
    from qwen35_moe_dflash import artifacts, existing_air
    main, fresh, _, calls, _ = job
    original = existing_air.compile_existing_air
    def oversized_compile(source, folder, ddk, **kwargs):
        report = original(source, folder, ddk, **kwargs)
        # The declared size alone must reject this result before any payload
        # read. No multi-gigabyte allocation or fake successful SHA check.
        report["omc"]["bytes"] = 1 << 32
        (folder / "report.json").write_text(json.dumps(report))
        return report
    original_check = artifacts.checked_file
    def no_model_check(root, record):
        if record["path"].endswith(".omc"):
            pytest.fail("oversized OMC must fail before checked_file hashes its payload")
        return original_check(root, record)
    monkeypatch.setattr(existing_air, "compile_existing_air", oversized_compile)
    monkeypatch.setattr(artifacts, "checked_file", no_model_check)
    root = tmp_path / "oversized-embedded"
    args = [*fresh, "--weight-storage", "embedded", "--output", str(root)]
    assert main(args + (["--compile-only"] if compile_only else [])) == 1
    assert calls == ["prepared", "fixture", "air", "compilation"]
    report = json.loads((root / "result.json").read_text())
    assert "UINT32" in report["error"] and report["device_execution"] == "NOT_RUN"


def test_storage_option_rejects_invalid_enum_before_execution(job, tmp_path):
    main, fresh, _, calls, _ = job
    with pytest.raises(SystemExit) as error:
        main([*fresh, "--weight-storage", "both", "--output", str(tmp_path / "invalid")])
    assert error.value.code == 2 and calls == [] and not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("storage", ["external", "embedded"])
def test_inspect_only_rejects_weight_storage_selection(job, tmp_path, storage):
    main, _, _, calls, _ = job
    output = tmp_path / "invalid-inspection"
    assert main(["--inspect-only", "--resume", "retained", "--weight-storage", storage,
                 "--output", str(output)]) == 1
    assert calls == [] and not output.exists()


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


@pytest.mark.parametrize("summary", [
    {"status": "NOT_SELECTED", "reason": "snapshot exceeds diagnostic selection byte limit",
     "snapshot_bytes": 17 * 1024 * 1024, "max_bytes": 16 * 1024 * 1024},
    {"status": "NO_MATCHING_RECORDS", "recorded_pid": 29314,
     "scope": "No records match the recorded PID; no other process was substituted."},
])
def test_captured_hilog_with_empty_summary_explains_selection_without_execution(job, tmp_path, capsys, summary):
    main, _, _, calls, _ = job
    result = {"status": "FAILED", "board": str(tmp_path), "failure_stage": "LOAD_OR_SETUP",
              "error": "HIAI Load failed: 1"}
    original = dict(result)
    diagnostic = {"status": "CAPTURED", "summary_lines": [], "summary": summary,
                  "process_identity": {"recorded_pid": 29314},
                  "scope": "Shared system-log snapshot; may include other processes and runs."}
    main.__globals__["_print_board_failure"](result, diagnostic)
    output = capsys.readouterr().out
    assert "status=CAPTURED" in output and f"summary: status={summary['status']}" in output
    assert "recorded_pid=29314" in output and "HIAI Load failed: 1" in output
    assert summary.get("scope", diagnostic["scope"]) in output
    if summary["status"] == "NOT_SELECTED":
        assert summary["reason"] in output
        assert f"snapshot_bytes: {17 * 1024 * 1024}" in output
        assert f"max_bytes: {16 * 1024 * 1024}" in output
    assert result == original and calls == []


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
        report = {"artifact_kind": "qwen35-existing-air-compilation", "status": "COMPILED", "ddk_id": "test-ddk", "requested_options": options,
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
                                    "cpro-fileconstant-relative-int-fc-conv2d-v2",
                                    "cpro-fileconstant-relative-int-fc-conv2d-broadcast-i32-v3"])
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


@pytest.mark.parametrize("contract", ["cpro-fileconstant-relative-int-fc-conv2d-v2",
                                    "cpro-fileconstant-relative-int-fc-conv2d-broadcast-i32-v3"])
def test_native_broadcast_air_rebuilds_after_board_failure_preserving_reference(job, tmp_path, contract):
    main, fresh, common, calls, failures = job
    previous = tmp_path / "compiled-old-broadcast"
    failures.add("board")
    assert main([*fresh, "--output", str(previous)]) == 1
    old = _make_legacy_air(previous, contract=contract)
    before = {path: path.read_bytes() for path in previous.rglob("*") if path.is_file()}
    failures.clear(); calls.clear()
    resumed = tmp_path / "gather-mask"
    assert main([*common, "--resume", str(previous), "--output", str(resumed)]) == 0
    assert calls == ["air", "compilation", "bundle", "board"]
    result = json.loads((resumed / "result.json").read_text())
    assert all(result["components"][name] == old["components"][name] for name in ("prepared", "fixture"))
    invalidation, = result["resume_invalidations"]
    assert invalidation["reason"] == "OBSOLETE_DRAFT_EXPORT_CONTRACT"
    assert set(invalidation["components"]) == {"air", "compilation", "bundle"}
    assert invalidation["required_export_contract"].endswith("mask-gather-v4")
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


def _stage_payloads(result, stages=None):
    """Capture real bytes and inode identity, independently of job receipts."""
    selected = result["components"] if stages is None else stages
    return {path: (path.read_bytes(), path.stat().st_ino)
            for name in selected
            for path in Path(result["components"][name]["directory"]).rglob("*") if path.is_file()}


def _assert_payloads_unchanged(payloads):
    assert all(path.read_bytes() == data and path.stat().st_ino == inode
               for path, (data, inode) in payloads.items())


def test_fixed_default_directory_retries_original_command_without_rebuilding_completed_stages(job, tmp_path):
    main, fresh, _, calls, failures = job
    root = tmp_path / "full-draft"
    failures.add("board")
    assert main(fresh) == 1
    old = json.loads((root / "result.json").read_text())
    assert old["output"] == str(root)
    preserved = _stage_payloads(old)
    (root / "board/stale-device.log").write_text("previous board attempt")
    failures.clear(); calls.clear()
    assert main(fresh) == 0
    assert calls == ["board"]
    assert not (root / "board/stale-device.log").exists()
    _assert_payloads_unchanged(preserved)
    assert json.loads((root / "result.json").read_text())["status"] == "EXECUTION_COMPLETE"
    assert [path.name for path in tmp_path.glob("full-draft*")] == ["full-draft"]


def test_explicit_same_output_and_resume_retries_only_board(job, tmp_path):
    main, fresh, _, calls, failures = job
    root = tmp_path / "named-workspace"
    failures.add("board")
    assert main([*fresh, "--output", str(root)]) == 1
    preserved = _stage_payloads(json.loads((root / "result.json").read_text()))
    failures.clear(); calls.clear()
    assert main(["--resume", str(root), "--output", str(root)]) == 0
    assert calls == ["board"]
    _assert_payloads_unchanged(preserved)


def test_fixed_retry_removes_partial_compilation_and_downstream_but_preserves_air(job, tmp_path):
    main, fresh, common, calls, failures = job
    root = tmp_path / "full-draft"
    failures.add("compilation")
    assert main(fresh) == 1
    old = json.loads((root / "result.json").read_text())
    preserved = _stage_payloads(old)
    (root / "compilation/partial-output.bin").write_bytes(b"incomplete compiler output")
    (root / "bundle").mkdir()
    (root / "bundle/partial-upload.bin").write_bytes(b"incomplete bundle")
    failures.clear(); calls.clear()
    assert main(common[2:]) == 0
    assert calls == ["compilation", "bundle", "board"]
    assert not (root / "compilation/partial-output.bin").exists()
    assert not (root / "bundle/partial-upload.bin").exists()
    _assert_payloads_unchanged(preserved)


def test_fixed_changed_compile_profile_cannot_reuse_old_receipts_after_failed_recompile(job, tmp_path):
    main, fresh, common, calls, failures = job
    root = tmp_path / "full-draft"
    assert main(fresh) == 0
    old = json.loads((root / "result.json").read_text())
    preserved = _stage_payloads(old, ("prepared", "fixture", "air"))
    (root / "compilation/old-external-weight.weight").write_bytes(b"old compiler payload")
    (root / "board/old.log").write_text("old board")
    calls.clear(); failures.add("compilation")
    assert main([*common[2:], "--weight-storage", "embedded"]) == 1
    assert calls == ["compilation"]
    for name in ("compilation", "bundle"):
        assert not (root / f"completed-{name}.json").exists()
    assert not (root / "bundle").exists() and not (root / "board").exists()
    assert not (root / "compilation/old-external-weight.weight").exists()
    _assert_payloads_unchanged(preserved)
    failures.clear(); calls.clear()
    assert main(common[2:]) == 0
    assert calls == ["compilation", "bundle", "board"]
    assert json.loads((root / "result.json").read_text())["weight_storage"] == "embedded"
    _assert_payloads_unchanged(preserved)


def test_fixed_obsolete_air_removes_all_dependent_receipts_even_if_export_fails(job, tmp_path, monkeypatch):
    from qwen35_moe_dflash import draft_standalone
    main, fresh, common, calls, _ = job
    root = tmp_path / "full-draft"
    assert main(fresh) == 0
    old = _make_legacy_air(root, contract="cpro-fileconstant-relative-int-fc-conv2d-broadcast-i32-v3")
    preserved = _stage_payloads(old, ("prepared", "fixture"))
    original = draft_standalone.export_draft
    def failed_export(prepared, folder, ge, **kwargs):
        calls.append("air")
        folder.mkdir()
        (folder / "partial.air").write_bytes(b"interrupted exporter")
        raise RuntimeError("fixture AIR export failure")
    monkeypatch.setattr(draft_standalone, "export_draft", failed_export)
    calls.clear()
    assert main(common) == 1
    assert calls == ["air"]
    for name in ("air", "compilation", "bundle"):
        assert not (root / f"completed-{name}.json").exists()
    for name in ("compilation", "bundle", "board"):
        assert not (root / name).exists()
    _assert_payloads_unchanged(preserved)
    monkeypatch.setattr(draft_standalone, "export_draft", original)
    calls.clear()
    assert main(common) == 0
    assert calls == ["air", "compilation", "bundle", "board"]
    assert not (root / "air/partial.air").exists()
    _assert_payloads_unchanged(preserved)


@pytest.mark.parametrize("changed", ["weights", "inputs", "manifest"])
def test_fixed_invalid_retry_does_not_clean_or_overwrite_the_previous_attempt(job, tmp_path, changed):
    main, fresh, _, calls, failures = job
    root = tmp_path / "full-draft"
    failures.add("board")
    assert main(fresh) == 1
    (root / "board/keep-failure.log").write_text("retain until retry preflight succeeds")
    args = []
    if changed == "weights":
        alternate = tmp_path / "different-target"
        alternate.mkdir()
        args = ["--target-weights", str(alternate)]
    elif changed == "inputs":
        args = ["--sequence-length", "128"]
    else:
        with (root / "air/manifest.json").open("a") as stream:
            stream.write(" ")
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    calls.clear()
    assert main(args) == 1
    assert calls == []
    assert all(path.read_bytes() == payload for path, payload in before.items())
    assert (root / "board/keep-failure.log").exists()


def test_fixed_workspace_lock_rejects_concurrent_cli_without_rewriting_active_evidence(job, tmp_path):
    from qwen35_moe_dflash.draft_workspace import DraftWorkspace
    main, fresh, _, calls, _ = job
    root = tmp_path / "full-draft"
    assert main([*fresh, "--compile-only"]) == 0
    calls.clear()
    with DraftWorkspace(root):
        before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
        assert main(["--compile-only"]) == 1
        assert calls == []
        assert {path: path.read_bytes() for path in root.rglob("*") if path.is_file()} == before


def test_fixed_default_refuses_unrecognized_nonempty_directory_without_deleting_files(job, tmp_path):
    main, fresh, _, calls, _ = job
    root = tmp_path / "full-draft"
    root.mkdir()
    protected = root / "unrelated-user-file.bin"
    protected.write_bytes(b"not a recognized draft job")
    assert main(fresh) == 1
    assert calls == [] and protected.read_bytes() == b"not a recognized draft job"
    assert not (root / "job.json").exists() and not (root / "result.json").exists()


def test_legacy_resume_adopts_cross_job_stages_into_self_contained_fixed_directory(job, tmp_path):
    main, fresh, common, calls, _ = job
    origin, later = tmp_path / "legacy-first", tmp_path / "legacy-second"
    assert main([*fresh, "--compile-only", "--output", str(origin)]) == 0
    assert main([*common, "--resume", str(origin), "--output", str(later)]) == 0
    old = json.loads((later / "result.json").read_text())
    before = {name: {path.relative_to(record["directory"]): (path.read_bytes(), path.stat().st_ino)
                     for path in Path(record["directory"]).rglob("*") if path.is_file()}
              for name, record in old["components"].items()}
    assert Path(old["components"]["prepared"]["directory"]).parent == origin
    assert Path(old["components"]["bundle"]["directory"]).parent == later
    calls.clear()
    assert main(["--resume", str(later)]) == 0
    assert calls == ["board"]
    root = tmp_path / "full-draft"
    adopted = json.loads((root / "result.json").read_text())
    for name, record in adopted["components"].items():
        folder = Path(record["directory"])
        assert folder == root / name and not Path(old["components"][name]["directory"]).exists()
        assert {path.relative_to(folder): (path.read_bytes(), path.stat().st_ino)
                for path in folder.rglob("*") if path.is_file()} == before[name]
        receipt = json.loads((root / f"completed-{name}.json").read_text())
        assert receipt == record
    origin.rename(tmp_path / "old-origin-unavailable")
    later.rename(tmp_path / "old-later-unavailable")
    calls.clear()
    assert main(["--compile-only"]) == 0
    assert calls == []


def test_cli_resumes_partial_adoption_from_journal_after_move_before_checkpoint(job, tmp_path, monkeypatch):
    from qwen35_moe_dflash import draft_workspace
    main, fresh, _, calls, _ = job
    source = tmp_path / "legacy-to-adopt"
    assert main([*fresh, "--output", str(source)]) == 0
    old = json.loads((source / "result.json").read_text())
    before = {name: {path.relative_to(record["directory"]): (path.read_bytes(), path.stat().st_ino)
                     for path in Path(record["directory"]).rglob("*") if path.is_file()}
              for name, record in old["components"].items()}
    original_rename = draft_workspace.os.rename
    interrupted = False
    def rename_then_interrupt(src, dst, *args, **kwargs):
        nonlocal interrupted
        result = original_rename(src, dst, *args, **kwargs)
        if not interrupted and Path(src) == source / "prepared":
            interrupted = True
            raise OSError("fixture interruption after stage rename")
        return result
    monkeypatch.setattr(draft_workspace.os, "rename", rename_then_interrupt)
    calls.clear()
    assert main(["--resume", str(source)]) == 1
    assert interrupted and calls == []
    root = tmp_path / "full-draft"
    assert (root / "prepared").is_dir() and not (source / "prepared").exists()
    assert (root / ".draft-migration.json").is_file()
    # The first failure happens before CLI commits its new job root. The
    # journal, not a possibly missing/stale result.json, must drive retry.
    monkeypatch.setattr(draft_workspace.os, "rename", original_rename)
    assert main([]) == 0
    assert calls == ["board"]
    adopted = json.loads((root / "result.json").read_text())
    assert not (root / ".draft-migration.json").exists()
    for name, record in adopted["components"].items():
        folder = Path(record["directory"])
        assert folder == root / name
        assert {path.relative_to(folder): (path.read_bytes(), path.stat().st_ino)
                for path in folder.rglob("*") if path.is_file()} == before[name]
    source.rename(tmp_path / "legacy-source-unavailable")
    calls.clear()
    assert main(["--compile-only"]) == 0 and calls == []


def test_cli_refuses_adoption_from_a_locked_source_workspace_without_moving_payloads(job, tmp_path):
    from qwen35_moe_dflash.draft_workspace import DraftWorkspace
    main, fresh, _, calls, _ = job
    source = tmp_path / "active-source"
    assert main([*fresh, "--output", str(source)]) == 0
    root = tmp_path / "full-draft"
    calls.clear()
    with DraftWorkspace(source):
        before = {path: (path.read_bytes(), path.stat().st_ino)
                  for path in source.rglob("*") if path.is_file()}
        assert main(["--resume", str(source)]) == 1
        assert calls == []
        assert {path: (path.read_bytes(), path.stat().st_ino)
                for path in source.rglob("*") if path.is_file()} == before
        assert not any((root / stage).exists() for stage in ("prepared", "fixture", "air", "compilation", "bundle"))


def test_cli_rejects_symlink_output_before_resolving_or_touching_the_other_workspace(job, tmp_path):
    main, fresh, _, calls, _ = job
    source = tmp_path / "real-job"
    assert main([*fresh, "--compile-only", "--output", str(source)]) == 0
    alias = tmp_path / "output-alias"
    alias.symlink_to(source, target_is_directory=True)
    before = {path: (path.read_bytes(), path.stat().st_ino)
              for path in source.rglob("*") if path.is_file()}
    calls.clear()
    assert main(["--output", str(alias), "--compile-only"]) == 1
    assert calls == [] and alias.is_symlink()
    assert {path: (path.read_bytes(), path.stat().st_ino)
            for path in source.rglob("*") if path.is_file()} == before
