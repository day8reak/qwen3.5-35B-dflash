"""Entry-point wiring tests; these doubles never establish device execution."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen35_moe_dflash import serial_dflash_smoke as job
from qwen35_moe_dflash.incremental_smoke import SmokeContract
from test_incremental_smoke import Target, Draft


class TargetDouble(Target):
    instances = []
    def __init__(self, decode_job, runtime_probe, lineage, lookup, output, **kwargs):
        super().__init__(SmokeContract())
        self.runtime_version, self.firmware_id = "test runtime", "test firmware"
        self.prompt_length = 0
        self.instances.append(self)
    def preflight(self, prompt):
        self.prompt_length = len(prompt)
        return {"prompt_token_ids": list(prompt), "scope": "host-double"}
    def execution_summary(self):
        return {"native_calls": self.prompt_length + len(self.advanced),
                "validated_native_calls": True, "receipt_paths": [], "scope": "host-double"}


class DraftDouble(Draft):
    instances = []
    def __init__(self, bundle, output, **kwargs):
        super().__init__(SmokeContract(), mismatch_at=0)
        self.instances.append(self)
    def preflight(self):
        return {"sequence_length": 64, "scope": "host-double"}
    def execution_summary(self):
        return {"native_calls": self.calls, "validated_native_calls": True,
                "receipt_paths": [], "scope": "host-double"}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    prompt = tmp_path / "prompt.bin"
    prompt.write_bytes(b"".join(value.to_bytes(4, "little", signed=True) for value in range(1, 65)))
    monkeypatch.setattr(job, "OriginalEmbeddingLookup", lambda *args: SimpleNamespace(initialization={"scope": "host-double"}))
    monkeypatch.setattr(job, "_backends", lambda: (TargetDouble, DraftDouble))
    monkeypatch.setattr(job.shutil, "disk_usage", lambda path: SimpleNamespace(free=8 << 30))
    TargetDouble.instances.clear(); DraftDouble.instances.clear()
    return dict(decode_job=tmp_path / "job", runtime_probe=tmp_path / "probe", state_lineage=tmp_path / "lineage",
        source_torchair_pb2=tmp_path / "schema.py", draft_bundle=tmp_path / "bundle", weight_table=tmp_path / "weights",
        scale_table=tmp_path / "scale", prompt_input_ids=prompt, output=tmp_path / "execution")


def test_plan_does_not_invoke_target_or_draft_and_keeps_explicit_short_prompt(setup):
    result = job.run_serial_dflash_smoke(**setup)
    assert result["status"] == "DFLASH_SERIAL_SMOKE_PLANNED"
    assert result["device_execution"] == result["model_compilation"] == "NOT_RUN"
    assert TargetDouble.instances[0].calls == DraftDouble.instances[0].calls == 0
    plan = json.loads((setup["output"] / "plan.json").read_text())
    assert plan["prompt_token_ids"] == list(range(1, 9))
    assert plan["eos_policy"] == "DISABLED_FOR_BOUNDED_EXECUTION_MILESTONE"
    assert plan["prompt_source"]["policy"].startswith("NEW_SHORT_PROMPT")
    assert plan["max_new_tokens"] == 8 and plan["max_proposals_per_round"] == 2


def test_host_protocol_run_is_never_promoted_to_device_success(setup):
    result = job.run_serial_dflash_smoke(**setup, apply=True)
    assert result["status"] == "FAILED"
    assert "native execution receipts" in result["error"]
    assert result["device_execution"] != "EXECUTED_OUTPUTS_RETRIEVED"
    loop = json.loads((setup["output"] / "loop.json").read_text())
    assert loop["evidence_kinds"] == ["host-fake-hiai"]
    assert loop["generated_token_ids"] == list(range(9, 17))
    assert len(loop["rounds"]) == 7
    assert TargetDouble.instances[0].advanced == list(range(9, 17))


@pytest.mark.parametrize("changes", [dict(max_new_tokens=0), dict(max_proposals_per_round=16),
    dict(prompt_length=64), dict(prompt_length=60, max_new_tokens=8), dict(lock_wait_seconds=-1)])
def test_invalid_capacity_and_options_stop_before_constructing_adapters(setup, changes):
    result = job.run_serial_dflash_smoke(**(setup | changes), apply=True)
    assert result["status"] == "FAILED" and result["device_execution"] == "NOT_RUN"
    assert TargetDouble.instances == DraftDouble.instances == []


def test_wrong_prompt_extent_is_not_silently_padded(setup):
    setup["prompt_input_ids"].write_bytes(b"\x00" * 7)
    result = job.run_serial_dflash_smoke(**setup)
    assert result["status"] == "FAILED" and "INT32" in result["error"]
    assert TargetDouble.instances == []


def test_insufficient_retained_step_space_stops_before_model_binding(setup, monkeypatch):
    monkeypatch.setattr(job.shutil, "disk_usage", lambda path: SimpleNamespace(free=1 << 20))
    result = job.run_serial_dflash_smoke(**setup)
    assert result["status"] == "FAILED" and "host bytes" in result["error"]
    assert TargetDouble.instances == DraftDouble.instances == []


def test_failed_target_step_keeps_one_attempt_and_never_calls_draft(setup, monkeypatch):
    def disconnected(self, prompt):
        raise RuntimeError("HDC unavailable; retained attempt /test/run-1")
    monkeypatch.setattr(TargetDouble, "prefill", disconnected)
    result = job.run_serial_dflash_smoke(**setup, apply=True)
    assert result["status"] == "FAILED" and "retained attempt" in result["error"]
    assert len(TargetDouble.instances) == 1 and DraftDouble.instances[0].calls == 0
    assert result["automatic_retry"].startswith("NONE")
    assert (setup["output"] / "result.json").is_file()


def test_failed_draft_host_preflight_still_reports_no_device_execution(setup, monkeypatch):
    class BrokenDraft:
        def __init__(self, *args, **kwargs):
            raise ValueError("source model changed during host pin")
    monkeypatch.setattr(job, "_backends", lambda: (TargetDouble, BrokenDraft))
    result = job.run_serial_dflash_smoke(**setup, apply=True)
    assert result["status"] == "FAILED" and result["stage"] == "host-preflight"
    assert result["device_execution"] == "NOT_RUN" and TargetDouble.instances[0].calls == 0
