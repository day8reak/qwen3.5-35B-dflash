"""Independent decode transport fixtures are not inference or feedback evidence."""
import json
from pathlib import PurePosixPath

import pytest

from qwen35_moe_dflash import original_target_board as board
from qwen35_moe_dflash import target_decode_smoke_plan as planner
from qwen35_moe_dflash.existing_air_profile import get_existing_air_profile
from qwen35_moe_dflash.hashing import sha256_file
from test_target_decode_smoke_plan import case as planner_case, rewrite_native
from test_target_prefill_smoke_execution import FakeHdc as PrefillFakeHdc


class DecodeFakeHdc(PrefillFakeHdc):
    """Reuse the existing exact selected-tool protocol simulator."""
    def shell(self, command, label, **options):
        if label == "decode-board-disk":
            label = "prefill-board-disk"
        return super().shell(command, label, **options)


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("SN", "fixture-device")
    monkeypatch.setenv("BIP", "fixture-server:8710")
    value = planner_case.__wrapped__(tmp_path)
    monkeypatch.setattr(board, "Hdc", DecodeFakeHdc)
    DecodeFakeHdc.remote, DecodeFakeHdc.instances, DecodeFakeHdc.sends = {}, [], []
    for key, setting in (("status", 0), ("busy", False), ("timeout_label", None), ("available_kb", 1000000)):
        monkeypatch.setattr(DecodeFakeHdc, key, setting)
    from qwen35_moe_dflash import draft_hilog
    monkeypatch.setattr(draft_hilog, "capture_draft_hilog", lambda *a, **k: {"status": "HOST_FIXTURE_ONLY"})
    return dict(value, out=tmp_path / "decode-smoke")


def execute(case, **changes):
    options = {"profile": get_existing_air_profile("cpro-40block-decode-1"), "gear": 1,
        "host_input_dir": case["source"], "target_features": True,
        "runtime_decode_probe": case["probe"], "lock_wait_seconds": 0}
    options.update(changes)
    return board.execute_target_tool(case["compilation"], case["job"] / "target-contract/report.json",
                                     case["out"], **options)


def test_named_decode_inputs_execute_once_without_prefill_feedback_claim(case):
    original = {path.name: path.read_bytes() for path in case["source"].iterdir()}
    result = execute(case)
    assert result["status"] == "EXECUTED_UNVALIDATED", result
    assert result["execution_scope"] == "SINGLE_DECODE1_USER_SUPPLIED_INPUTS_NO_PREFILL_FEEDBACK"
    assert result["source_semantics"] == "UNVERIFIED_USER_INPUTS"
    assert result["prefill_state_feedback"] == "NOT_RUN"
    assert result["state_continuity"] == "NOT_VALIDATED_SINGLE_TOOL_INVOCATION"
    assert result["state_input_layer_identity"] == result["state_output_layer_identity"] == "UNVERIFIED"
    assert result["gear"] == 1 and len(result["inputs"]) == 89
    assert len([name for name in result["retrieved"] if name.startswith("outputs/")]) == 82
    assert result["numerical_validation"] == "NOT_RUN_NO_REFERENCE"
    assert result["tool_process_id"] == 1337 and result["remote_exitcode"] == 0
    assert result["lock_state"] == "NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT"
    assert result["source_preservation"] == "UNCHANGED_DECODE_PLAN_METADATA_AND_USER_INPUT_SHA256"
    assert all(path.read_bytes() == original[path.name] for path in case["source"].iterdir())
    assert (case["out"] / "runtime-decode-proof.json").is_file()
    assert not (case["out"] / "runtime-prefill-proof.json").exists()
    argv = json.loads((case["out"] / "command.json").read_text())["argv"]
    assert all(flag in argv for flag in ("--times=1", "--modelmanager_itf=V2", "--direct_from_file=true", "--enable_item=1"))
    assert not any(arg.startswith(("--input_shape", "--output_shape", "--input_format", "--output_format")) for arg in argv)
    assert any(arg.startswith("--weight_dir=") for arg in argv)
    for option, count in (("input_type", 89), ("output_type", 82)):
        value = next(arg.split("=", 1)[1] for arg in argv if arg.startswith("--" + option + "="))
        assert len(value.split(",")) == count
    calls = DecodeFakeHdc.instances[0].calls
    assert sum(label == "execute-original-target" for _, label, _ in calls) == 1


@pytest.mark.parametrize("change", ["both", "wrong-gear", "boolean-gear", "wrong-profile", "missing-features"])
def test_decode_has_explicit_separate_profile_and_probe_guard(case, monkeypatch, change):
    options = {}
    if change == "both": options["runtime_prefill_probe"] = case["probe"]
    elif change == "wrong-gear": options["gear"] = 64
    elif change == "boolean-gear": options["gear"] = True
    elif change == "wrong-profile": options["profile"] = get_existing_air_profile("cpro-40block-prefill-64")
    else: options["target_features"] = False
    monkeypatch.setattr(board, "_recorded_file", lambda *a: pytest.fail("guard must precede model payload reads"))
    result = execute(case, **options)
    assert result["status"] == "FAILED" and result["device_execution"] == "NOT_RUN"
    assert not DecodeFakeHdc.instances and not DecodeFakeHdc.sends


def test_missing_or_changed_runtime_names_stop_before_large_payload_access(case, monkeypatch):
    case["native"]["inputs"][0].pop("tensor_name", None)
    rewrite_native(case)
    monkeypatch.setattr(board, "_recorded_file", lambda *a: pytest.fail("unbound names cannot read model weights"))
    result = execute(case)
    assert result["status"] == "FAILED" and not DecodeFakeHdc.instances


def test_input_mutation_after_planning_is_rejected_without_upload(case, monkeypatch):
    validate = planner.validate_decode_smoke_plan
    def mutate(plan):
        path = case["source"] / "effective_length.bin"
        path.write_bytes(b"changed")
        return validate(plan)
    monkeypatch.setattr(planner, "validate_decode_smoke_plan", mutate)
    result = execute(case)
    assert result["status"] == "FAILED" and not DecodeFakeHdc.instances


def test_disk_shortfall_or_global_contention_cannot_start_decode(case, monkeypatch):
    monkeypatch.setattr(DecodeFakeHdc, "available_kb", 1)
    result = execute(case)
    assert result["status"] == "FAILED" and "decode board disk budget insufficient" in result["error"]
    assert not any(local.endswith((".omc", ".weight", ".bin")) for local, _ in DecodeFakeHdc.sends)
    assert not any(label == "execute-original-target" for _, label, _ in DecodeFakeHdc.instances[0].calls)


def test_existing_matching_decode_payload_is_reused_without_model_upload(case):
    digest = sha256_file(case["compilation"] / "report.json")
    cache = PurePosixPath("/data/local/tmp/qwen3_5_moe") / ("codex-target-" + digest[:20])
    for row in case["compiled"]["generated_artifacts"]:
        relative = PurePosixPath(row["path"]).relative_to("compile-work/generated")
        DecodeFakeHdc.remote[str(cache / "payload" / relative)] = (case["compilation"] / row["path"]).read_bytes()
    result = execute(case)
    assert result["status"] == "EXECUTED_UNVALIDATED"
    assert result["board_disk_budget"]["missing_model_payload_bytes"] == 0
    assert not any(local.endswith((".omc", ".weight")) for local, _ in DecodeFakeHdc.sends)


@pytest.mark.parametrize("fault", ["native", "timeout", "lock-timeout"])
def test_failed_decode_retains_pid_evidence_or_uncertain_locks_without_retry(case, monkeypatch, fault):
    if fault == "native": monkeypatch.setattr(DecodeFakeHdc, "status", 7)
    elif fault == "timeout": monkeypatch.setattr(DecodeFakeHdc, "timeout_label", "execute-original-target")
    else: monkeypatch.setattr(DecodeFakeHdc, "timeout_label", "acquire-global-model-io-lock")
    result = execute(case)
    assert result["status"] == "FAILED" and result["numerical_validation"] == "NOT_RUN_NO_REFERENCE"
    labels = [label for _, label, _ in DecodeFakeHdc.instances[0].calls]
    assert labels.count("execute-original-target") == (0 if fault == "lock-timeout" else 1)
    if fault == "native":
        assert result["tool_process_id"] == 1337 and result["remote_exitcode"] == 7
        assert result["lock_state"] == "NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT"
        assert result["hilog"]["artifact_kind"] == "qwen35-target-decode-hilog"
    else:
        assert result["lock_state"] == "RETAINED_OR_UNCERTAIN"
        assert "release-global-model-io-lock" not in labels


def test_decode_lock_release_failure_does_not_announce_a_completed_attempt(case, monkeypatch):
    monkeypatch.setattr(DecodeFakeHdc, "timeout_label", "release-global-model-io-lock")
    result = execute(case)
    assert result["status"] == "FAILED" and result["lock_state"] == "RETAINED_OR_UNCERTAIN"
    assert result["device_execution"] == "EXECUTED_OUTPUTS_RETRIEVED"
    assert result["lock_release_error"]
