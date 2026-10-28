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
        save_json(folder / "manifest.json", {"status": "EXPORTED", "graphs": [
            {"name": "draft", "air": "draft.air", "files": [file_record(air, folder)]}]})

    def compile_air(source, folder, ddk, **kwargs):
        calls.append("compilation")
        assert source.name == "draft.air" and kwargs["input_layout"] == "air-directory"
        assert kwargs["save_weights_as_external_data"] is True
        assert "input_shape" not in kwargs  # never the original 40-layer profile
        folder.mkdir()
        if "compilation" in failures:
            (folder / "compile.log").write_text("ERROR: fixture compile failure\n")
            return {"status": "FAILED", "error": "fixture compile failure"}
        model = folder / "model.omc"
        model.write_bytes(b"fixture-compiled-draft")
        result = {"status": "COMPILED", "ddk_id": "test-ddk", "omc": file_record(model, folder)}
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
