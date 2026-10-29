"""Morning summaries read small evidence only and never run a model or HDC."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import runpy
import subprocess

import pytest

from qwen35_moe_dflash import target_morning_status as morning


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("read-only morning summary must not execute any process"))
    return {key: tmp_path / name for key, name in morning._DEFAULTS.items()}


def io(sequence):
    return {"artifact_kind": "qwen35-target-model-io-probe", "status": "IO_METADATA_OBSERVED", "stage": "complete",
        "profile": "cpro-40block-prefill-64" if sequence == 64 else "cpro-40block-decode-1", "sequence_length": sequence,
        "input_count": 89, "output_count": 82, "manager_init": "NOT_RUN", "descriptor_stage": "restore-only",
        "compiled_boundary_count_check": "MATCHED_89_INPUT_82_OUTPUT_COUNTS_ONLY", "dtype_enum": "hiai::DataType",
        "lock_state": "NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT", "retrieval_errors": [],
        "source_preservation": "UNCHANGED_METADATA_AND_SMALL_PRIMARY_SHA256; WEIGHTS_NOT_OPENED",
        "dtype_enum_values": {"FLOAT16": 1, "INT32": 3}, "format_enum_values": {"ND": 2}}


def ready_probes(case):
    save(case["prefill_probe"] / "result.json", io(64))
    save(case["decode_probe"] / "result.json", io(1))


def plan():
    return {"artifact_kind": "qwen35-target-prefill-smoke-job", "mode": "PLAN_ONLY", "status": "PREFILL_SMOKE_PLANNED",
            "input_count": 89, "output_count": 82, "device_execution": "NOT_RUN"}


def smoke():
    return {"artifact_kind": "qwen35-target-prefill-smoke-job", "mode": "APPLY", "status": "EXECUTED_UNVALIDATED",
        "device_execution": "EXECUTED_OUTPUTS_RETRIEVED", "remote_exitcode": 0, "retrieval_errors": [],
        "retrieved": {"outputs/vendor_tensor.bin": {"bytes": 256, "sha256": "a" * 64},
                      "tool.stdout": {"bytes": 128, "sha256": "b" * 64}}, "tool_process_id": 123,
        "numerical_validation": "NOT_RUN_NO_REFERENCE", "state_continuity": "NOT_VALIDATED_SINGLE_TOOL_INVOCATION",
        "lock_state": "NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT", "source_preservation": "UNCHANGED_PREFILL_PLAN_METADATA_AND_ORIGINAL_INPUT_SHA256"}


def test_missing_reports_are_waiting_not_failure_and_create_no_files(case):
    report = morning.inspect_target_morning_status()
    assert report["next"] == "WAIT_FOR_RUNTIME_PROBES" and report["failure_stages"] == []
    assert all(row["metadata_state"] == "MISSING" for row in report["stages"].values())
    assert not any(path.exists() for path in case.values())
    assert report["actions_executed"] == "NONE"


def test_both_correct_profiles_prepare_advisory_plan_but_no_execution(case):
    ready_probes(case)
    report = morning.inspect_target_morning_status()
    assert report["next"] == "READY_FOR_PREFILL_PLAN" and report["both_runtime_metadata_reports_ready"]
    assert report["model_completion"] == "NOT_ESTABLISHED" and report["performance_speedup"] == "NOT_MEASURED"


@pytest.mark.parametrize("fault", ["count", "output_count", "sequence", "wrong_profile", "manager_init", "stage", "false_status", "count_flag"])
def test_incorrect_io_claim_never_marks_plan_ready(case, fault):
    ready_probes(case)
    value = io(1)
    if fault == "count": value["input_count"] = 88
    elif fault == "output_count": value["output_count"] = 81
    elif fault == "sequence": value["sequence_length"] = True
    elif fault == "wrong_profile": value["profile"] = "cpro-40block-prefill-64"
    elif fault == "manager_init": value["manager_init"] = "PASSED"
    elif fault == "stage": value["descriptor_stage"] = "manager-init"
    elif fault == "false_status": value["status"] = "INSPECTED"
    else: value["compiled_boundary_count_check"] = "DIFFERENT_COUNTS_RAW_METADATA_RETAINED"
    save(case["decode_probe"] / "result.json", value)
    save(case["prefill_plan"] / "result.json", plan())
    report = morning.inspect_target_morning_status()
    assert report["next"] == "INSPECT_RUNTIME_METADATA" and not report["both_runtime_metadata_reports_ready"]


def test_saved_plan_cannot_skip_missing_runtime_reports(case):
    save(case["prefill_plan"] / "result.json", plan())
    report = morning.inspect_target_morning_status()
    assert report["next"] == "WAIT_FOR_RUNTIME_PROBES" and not report["first_prefill_outputs_observed"]


@pytest.mark.parametrize("fault", ["held_lock", "lock_error", "source_error", "native_error", "source_preservation", "retrieval_error"])
def test_unclean_probe_cannot_advertise_readiness_even_with_correct_raw_counts(case, fault):
    ready_probes(case)
    value = io(64)
    if fault == "held_lock": value["lock_state"] = "RETAINED_OR_UNCERTAIN"
    elif fault == "lock_error": value["lock_release_error"] = "release unconfirmed"
    elif fault == "source_error": value["source_error"] = "source changed"
    elif fault == "native_error": value["native_evidence_error"] = "native result unbound"
    elif fault == "source_preservation": value["source_preservation"] = "FAILED"
    else: value["retrieval_errors"] = [{"path": "model-io.json"}]
    save(case["prefill_probe"] / "result.json", value)
    save(case["prefill_plan"] / "result.json", plan())
    report = morning.inspect_target_morning_status()
    assert not report["both_runtime_metadata_reports_ready"]
    assert report["next"] not in {"READY_FOR_PREFILL_PLAN", "READY_FOR_PREFILL_EXECUTION"}


def test_valid_plan_after_both_probes_is_ready_for_explicit_execution_only(case):
    ready_probes(case)
    save(case["prefill_plan"] / "result.json", plan())
    report = morning.inspect_target_morning_status()
    assert report["next"] == "READY_FOR_PREFILL_EXECUTION" and report["actions_executed"] == "NONE"
    assert not case["prefill_smoke"].exists()


def test_retrieved_first_prefill_is_not_target_draft_or_state_pass(case):
    ready_probes(case)
    save(case["prefill_plan"] / "result.json", plan())
    save(case["prefill_smoke"] / "result.json", smoke())
    board = smoke(); board["artifact_kind"] = "qwen35-original-target-tool-execution"
    save(case["prefill_smoke"] / "board/result.json", board)
    report = morning.inspect_target_morning_status()
    assert report["first_prefill_outputs_observed"]
    assert report["next"] == "PREFILL_OUTPUTS_OBSERVED_DECODE_STATE_AND_FEATURE_MAPPING_PENDING"
    assert report["state_continuity"] == "UNVERIFIED" and report["numerical_validation"] == "NOT_ESTABLISHED"


@pytest.mark.parametrize("fault", ["nonzero_exit", "boolean_exit", "zero_outputs", "zero_bytes", "bad_hash", "retrieval_error", "device_status", "traversal"])
def test_false_first_prefill_success_is_never_promoted(case, fault):
    ready_probes(case)
    save(case["prefill_plan"] / "result.json", plan())
    value = smoke()
    if fault == "nonzero_exit": value["remote_exitcode"] = 1
    elif fault == "boolean_exit": value["remote_exitcode"] = False
    elif fault == "zero_outputs": value["retrieved"] = {"tool.stdout": value["retrieved"]["tool.stdout"]}
    elif fault == "zero_bytes": value["retrieved"]["outputs/vendor_tensor.bin"]["bytes"] = 0
    elif fault == "bad_hash": value["retrieved"]["outputs/vendor_tensor.bin"]["sha256"] = "invalid"
    elif fault == "retrieval_error": value["retrieval_errors"] = [{"path": "missing.bin"}]
    elif fault == "device_status": value["device_execution"] = "ATTEMPTED"
    else: value["retrieved"]["outputs/../escape"] = value["retrieved"].pop("outputs/vendor_tensor.bin")
    save(case["prefill_smoke"] / "result.json", value)
    report = morning.inspect_target_morning_status()
    assert report["next"] == "INSPECT_RETAINED_FAILURE" and not report["first_prefill_outputs_observed"]


def test_nonzero_board_receipt_cannot_be_hidden_by_outer_success(case):
    ready_probes(case)
    save(case["prefill_plan"] / "result.json", plan())
    save(case["prefill_smoke"] / "result.json", smoke())
    board = smoke(); board.update(artifact_kind="qwen35-original-target-tool-execution", remote_exitcode=1)
    save(case["prefill_smoke"] / "board/result.json", board)
    report = morning.inspect_target_morning_status()
    assert report["next"] == "INSPECT_RETAINED_FAILURE" and report["stages"]["prefill_smoke"]["metadata_state"] == "INVALID"


@pytest.mark.parametrize("fault", ["malformed", "duplicate", "nonfinite", "overflow", "list", "oversized", "symlink", "directory", "wrong_kind", "enum_table"])
def test_bad_metadata_is_bounded_and_fail_closed_without_actions(case, fault):
    ready_probes(case)
    path = case["prefill_probe"] / "result.json"
    if fault == "malformed": path.write_text("{")
    elif fault == "duplicate": path.write_text('{"status":"A","status":"B"}')
    elif fault == "nonfinite": path.write_text('{"value":NaN}')
    elif fault == "overflow": path.write_text('{"value":1e999}')
    elif fault == "list": path.write_text("[]")
    elif fault == "oversized":
        with path.open("wb") as stream: stream.truncate(morning._MAX_JSON + 1)
    elif fault == "symlink":
        other = path.with_name("actual.json"); path.rename(other); path.symlink_to(other)
    elif fault == "directory": path.unlink(); path.mkdir()
    elif fault == "wrong_kind":
        value = io(64); value["artifact_kind"] = "unrelated"; save(path, value)
    else:
        value = io(64); value["dtype_enum_values"] = {"FLOAT16": {"wrong": 1}}; save(path, value)
    report = morning.inspect_target_morning_status()
    assert report["next"] == "INSPECT_RETAINED_FAILURE"
    assert report["stages"]["prefill_probe"]["metadata_state"] == "INVALID"
    assert report["actions_executed"] == "NONE"


def test_log_tails_read_at_most_four_kib_and_never_read_output_payload(case):
    ready_probes(case)
    root = case["prefill_smoke"]
    save(root / "result.json", {"artifact_kind": "qwen35-target-prefill-smoke-job", "status": "FAILED", "error": "fixture"})
    stderr = root / "board/tool.stderr"
    stderr.parent.mkdir()
    stderr.write_text("X" * 20000 + "last failure")
    output = root / "board/outputs/vendor.bin"
    output.parent.mkdir(); output.write_bytes(b"must not be opened")
    before = hashlib.sha256(output.read_bytes()).hexdigest()
    report = morning.inspect_target_morning_status()
    tail = report["stages"]["prefill_smoke"]["log_tails"][0]
    assert tail["tail_read_bytes"] == 4096 and tail["text"].endswith("last failure") and tail["truncated"]
    assert hashlib.sha256(output.read_bytes()).hexdigest() == before


def test_explicit_paths_work_without_active_environment_and_cli_only_prints(case, monkeypatch, capsys):
    ready_probes(case)
    monkeypatch.delenv("AI_RUN_DIR")
    script = Path(__file__).resolve().parents[1] / "targets/cpro/scripts/inspect_target_morning_status.py"
    main = runpy.run_path(str(script))["main"]
    args = [value for role, path in case.items() for value in ("--" + role.replace("_", "-"), str(path))]
    assert main(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["next"] == "READY_FOR_PREFILL_PLAN" and report["actions_executed"] == "NONE"


def test_missing_default_root_is_configuration_error_only(case, monkeypatch):
    monkeypatch.delenv("AI_RUN_DIR")
    with pytest.raises(ValueError, match="AI_RUN_DIR"):
        morning.inspect_target_morning_status()
