"""The new serial adapter gets an explicit board contract, with old paths intact."""
import sys
import types

import pytest

from qwen35_moe_dflash import original_target_board as board
from qwen35_moe_dflash import target_decode_continuation_run as runner
from qwen35_moe_dflash.existing_air_profile import get_existing_air_profile
from test_target_decode_continuation_board import (
    case as continuation_case, cached_payloads,
)
from test_target_decode_continuation_run import case as chain_case, emit


PRESERVATION = "UNCHANGED_SMOKE_BACKEND_PROOF_AND_ALL89_PREPARED_INPUT_SHA256"


@pytest.fixture
def backend(tmp_path, monkeypatch):
    case = continuation_case.__wrapped__(tmp_path, monkeypatch)
    old = sys.modules["qwen35_moe_dflash.target_decode_continuation"]
    module = types.ModuleType("qwen35_moe_dflash.original_target_smoke_backend")
    module.validate_smoke_backend_step = old.validate_decode_continuation
    monkeypatch.setitem(sys.modules, module.__name__, module)
    case["plan"]["execution_scope"] = "COMMITTED_TOKEN_DECODE1_EXPERIMENTAL_CONTROLS"
    return case


def execute(case, **changes):
    options = dict(profile=get_existing_air_profile("cpro-40block-decode-1"), gear=1,
        host_input_dir=case["continuation_inputs"], target_features=True, lock_wait_seconds=0,
        runtime_smoke_backend_prepared=case["prepared"])
    options.update(changes)
    return board.execute_target_tool(case["compilation"], case["job"] / "target-contract/report.json",
        case["out"], **options)


def test_backend_path_has_its_own_89_input_proof_and_source_receipt(backend):
    result = execute(backend)
    assert result["status"] == "EXECUTED_UNVALIDATED"
    assert result["source_preservation"] == PRESERVATION
    assert result["runtime_smoke_backend_prepared"] == str(backend["prepared"])
    assert result["execution_scope"] == backend["plan"]["execution_scope"]
    assert (backend["out"] / "runtime-backend-proof.json").is_file()
    assert len(result["inputs"]) == 89
    assert backend["validation"]["calls"] >= 3


def test_backend_remote_cache_hit_does_not_read_stale_local_weight(backend):
    cache = cached_payloads(backend)
    # A payload hit is decided by the original declared bytes on the phone.
    # The stale local source is not needed and must not trigger a 19GB rehash.
    backend["weight"].write_bytes(b"stale local payload")
    result = execute(backend)
    assert result["status"] == "EXECUTED_UNVALIDATED"
    weight = next(row for row in result["model_payloads"]
                  if row["path"] == str(cache / "payload/model/SubGraph_0.weight"))
    assert weight["upload"] != "SEND"


def test_backend_proof_is_mutually_exclusive_with_old_decode_proof(backend):
    result = execute(backend, runtime_decode_probe=backend["probe"])
    assert result["status"] == "FAILED"
    assert "mutually exclusive" in result["error"]
    assert result["device_execution"] == "NOT_RUN"


def test_backend_active_proof_change_cannot_be_accepted(backend):
    backend["validation"]["change_at"] = 3
    result = execute(backend)
    assert result["status"] == "FAILED"
    assert result.get("source_preservation") == "FAILED"


@pytest.fixture
def chain(tmp_path, monkeypatch):
    return chain_case.__wrapped__(tmp_path, monkeypatch)


def test_new_source_contract_must_be_explicit_for_reused_output_inspector(chain):
    root, observed, _ = emit(chain)
    observed["source_preservation"] = PRESERVATION
    with pytest.raises(ValueError, match="locked V2"):
        runner.inspect_continuation_outputs(chain["plan"], root, observed)
    result = runner.inspect_continuation_outputs(chain["plan"], root, observed,
        expected_source_preservation=PRESERVATION)
    assert result["feedback_input_count"] == 80


def test_arbitrary_source_preservation_cannot_bypass_output_inspector(chain):
    root, observed, _ = emit(chain)
    with pytest.raises(ValueError, match="unrecognized"):
        runner.inspect_continuation_outputs(chain["plan"], root, observed,
            expected_source_preservation="trust me")
