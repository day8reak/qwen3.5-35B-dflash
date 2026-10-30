"""A completed tool and failed retrieval remain separate observations."""
from copy import deepcopy
import json

import pytest

from qwen35_moe_dflash import target_decode_continuation_run as runner
from test_target_decode_continuation_board import case as board_case, execute, DecodeFakeHdc
from test_target_decode_continuation_run import case as chain_case, emit


@pytest.fixture
def transport(tmp_path, monkeypatch):
    return board_case.__wrapped__(tmp_path, monkeypatch)


@pytest.mark.parametrize("missing", ["output", "exitcode"])
def test_zero_exit_with_missing_receipt_is_transport_failure_not_a_model_failure(transport, monkeypatch, missing):
    receive = DecodeFakeHdc.receive
    def disconnected(self, remote, local):
        if (missing == "output" and "/outputs/" in str(remote)) or (missing == "exitcode" and remote.name == "tool.exitcode"):
            raise RuntimeError("HDC process exited0 but Device not founded or connected")
        return receive(self, remote, local)
    monkeypatch.setattr(DecodeFakeHdc, "receive", disconnected)
    result = execute(transport)
    assert result["status"] == "FAILED"
    assert result["remote_exitcode"] == 0 and result["model_process_status"] == "EXITED_ZERO"
    assert result["device_execution"] == "TOOL_EXITED_ZERO_OUTPUTS_UNVERIFIED"
    assert result["output_retrieval_status"] == "INCOMPLETE"
    inventory = json.loads((transport["out"] / "output-inventory.json").read_text())
    assert len(inventory["outputs"]) == 82 and inventory["board_attempt"] == result["board_attempt"]
    assert sum(label == "execute-original-target" for _, label, _ in DecodeFakeHdc.instances[0].calls) == 1


def test_nonzero_tool_exit_keeps_real_execution_failure(transport, monkeypatch):
    monkeypatch.setattr(DecodeFakeHdc, "status", 7)
    result = execute(transport)
    assert result["status"] == result["device_execution"] == "FAILED"
    assert result["remote_exitcode"] == 7 and result["model_process_status"] == "EXITED_NONZERO"
    assert result["output_retrieval_status"] != "COMPLETE_HASH_VERIFIED"


@pytest.fixture
def retained(tmp_path, monkeypatch):
    case = chain_case.__wrapped__(tmp_path, monkeypatch)
    root, observed, _ = emit(case)
    observed.update(lock_state="OWNED_COMPLETED_LOCKS_RETAINED_FOR_RECOVERY",
        recovery_only="NO_MODEL_RERUN_NO_INPUT_OR_WEIGHT_UPLOAD", board_lock_token="a" * 32)
    observed["recovery_completion"] = {"exitcode": 0, "pid": observed["tool_process_id"], "pid_absent": True,
        "original_owner_token": observed["board_lock_token"], "global_owner_verified": True, "cache_owner_verified": True,
        "compilation_report_sha256": case["plan"]["compilation_report_sha256"], "board_attempt": observed["board_attempt"]}
    return case, root, observed


def test_local_byte_chain_can_be_checked_before_releasing_proven_completed_original_locks(retained):
    case, root, observed = retained
    with pytest.raises(ValueError, match="locked V2"):
        runner.inspect_continuation_outputs(case["plan"], root, observed)
    chain = runner.inspect_continuation_outputs(case["plan"], root, observed, allow_retained_completed_lock=True)
    assert chain["feedback_input_count"] == 80 and chain["bound_child_output_count"] == 82
    assert chain["lock_inspection_scope"] == "READ_ONLY_UNDER_VERIFIED_ORIGINAL_COMPLETED_LOCKS"
    assert observed["lock_state"] == "OWNED_COMPLETED_LOCKS_RETAINED_FOR_RECOVERY"


@pytest.mark.parametrize("field,value", [("exitcode", False), ("exitcode", 7), ("pid_absent", False),
    ("global_owner_verified", False), ("cache_owner_verified", False), ("original_owner_token", "b" * 32),
    ("compilation_report_sha256", "0" * 64), ("board_attempt", "/data/local/tmp/other-attempt")])
def test_incomplete_or_foreign_completion_proof_cannot_enable_retained_lock_inspection(retained, field, value):
    case, root, observed = retained
    observed["recovery_completion"][field] = value
    with pytest.raises(ValueError, match="locked V2"):
        runner.inspect_continuation_outputs(case["plan"], root, observed, allow_retained_completed_lock=True)
