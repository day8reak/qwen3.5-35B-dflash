"""One-call transport fixtures do not execute or validate a target model."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import runpy
import shlex
import subprocess

import pytest

from qwen35_moe_dflash import original_target_board as board
from qwen35_moe_dflash import target_prefill_smoke_plan as planner
from qwen35_moe_dflash.existing_air_profile import get_existing_air_profile
from test_target_prefill_smoke_plan import case as planner_case, prepare, rewrite_native


class FakeHdc:
    remote = {}
    instances = []
    status = 0
    busy = False
    timeout_label = None
    available_kb = 1000000
    sends = []

    def __init__(self, executable, sn, bip, out, timeout):
        self.out, self.timeout, self.count, self.calls = out, timeout, 0, []
        self.instances.append(self)

    def send(self, local, remote):
        self.count += 1
        self.sends.append((str(local), str(remote)))
        self.remote[str(remote)] = Path(local).read_bytes()

    def receive(self, remote, local):
        self.count += 1
        Path(local).write_bytes(self.remote[str(remote)])

    def shell(self, command, label, **options):
        self.count += 1
        self.calls.append((command, label, options))
        if label == self.timeout_label:
            raise subprocess.TimeoutExpired(["fixture-hdc"], 1)
        def hashed(path):
            return hashlib.sha256(self.remote[path]).hexdigest() + "  " + path
        if label == "device-preflight":
            return 0, "sha256sum\nfind\n" + "a" * 64 + "  /data/local/tmp/tools_sysdbg/model_run_tool_internal\nfixture-firmware"
        if label == "acquire-global-model-io-lock":
            return 0, "BUSY" if self.busy else "ACQUIRED"
        if label == "inspect-model-cache":
            manifest = next(value.decode() for key, value in self.remote.items() if key.endswith("model-files.tsv"))
            return 0, "\n".join("SEND " + line.split(" ", 1)[1] for line in manifest.splitlines())
        if label == "prefill-board-disk":
            return 0, f"Filesystem 1024-blocks Used Available Capacity Mounted on\nfixture 9999999 1 {self.available_kb} 1% /data"
        if label == "verify-upload" or label.startswith("hash-tool-"):
            return 0, hashed(shlex.split(command)[1])
        if label == "prepare-inputs":
            script = self.remote[shlex.split(command)[1]].decode()
            result = []
            for line in script.splitlines():
                if line.startswith("cp "):
                    _, source, destination = shlex.split(line)
                    self.remote[destination] = self.remote[source]
                elif line.startswith("sha256sum "):
                    result.append(hashed(shlex.split(line)[1]))
            return 0, "\n".join(result)
        if label == "execute-original-target":
            path = next(key for key in self.remote if key.endswith("prepare-inputs.sh"))
            attempt = str(PurePosixPath(path).parent)
            self.attempt = attempt
            self.remote[attempt + "/tool.stdout"] = b"fixture one tool invocation; no target inference"
            self.remote[attempt + "/tool.stderr"] = b"ERROR: fixture external weight initialization failed" if self.status else b""
            self.remote[attempt + "/tool.exitcode"] = str(self.status).encode() + b"\n"
            self.remote[attempt + "/tool.pid"] = b"1337\n"
            for index in range(82):
                self.remote[attempt + f"/outputs/vendor_{index:03d}.bin"] = b"opaque fixture output"
            return self.status, ""
        if label == "output-inventory":
            return 0, "\n".join(hashed(path) for path in self.remote if path.startswith(self.attempt + "/outputs/"))
        return 0, ""


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("SN", "fixture-device")
    monkeypatch.setenv("BIP", "fixture-server:8710")
    value = planner_case.__wrapped__(tmp_path)
    monkeypatch.setattr(board, "Hdc", FakeHdc)
    FakeHdc.remote, FakeHdc.instances, FakeHdc.sends = {}, [], []
    for key, setting in (("status", 0), ("busy", False), ("timeout_label", None), ("available_kb", 1000000)):
        monkeypatch.setattr(FakeHdc, key, setting)
    # Auxiliary diagnostic capture is independent of execution acceptance.
    from qwen35_moe_dflash import draft_hilog
    monkeypatch.setattr(draft_hilog, "capture_draft_hilog", lambda *a, **k: {"status": "FIXTURE_ONLY"})
    return dict(value, out=tmp_path / "prefill-smoke")


def execute(case, **kwargs):
    return board.execute_target_tool(case["compilation"], case["job"] / "target-contract/report.json", case["out"],
        profile=get_existing_air_profile("cpro-40block-prefill-64"), gear=64, host_input_dir=case["source"],
        target_features=True, runtime_prefill_probe=case["probe"], lock_wait_seconds=0, **kwargs)


def test_default_guard_still_rejects_actual_lowered_boundaries_and_contradictory_indices(case):
    with pytest.raises(ValueError, match="matching actual compiled"):
        board.tool_plan(case["contract"], get_existing_air_profile("cpro-40block-prefill-64"), 64, "/board/inputs", target_features=True)
    case["contract"]["boundaries"]["declared_profile_check"]["status"] = "MATCHED_VISIBLE_BOUNDARIES"
    with pytest.raises(ValueError, match="complete compiled input indices contradict"):
        board.tool_plan(case["contract"], get_existing_air_profile("cpro-40block-prefill-64"), 64, "/board/inputs", target_features=True)
    assert not FakeHdc.instances


def test_one_prefill_executes_only_from_runtime_value_class_proof_and_preserves_unmapped_outputs(case):
    original = {path.name: path.read_bytes() for path in case["source"].iterdir()}
    result = execute(case)
    assert result["status"] == "EXECUTED_UNVALIDATED", result
    assert result["execution_scope"] == "SINGLE_PREFILL_ONLY_IDENTICAL_STATE_GROUPS"
    assert len(result["inputs"]) == 89 and len([name for name in result["retrieved"] if name.startswith("outputs/")]) == 82
    assert result["inputs"][0]["name"] == "arg18_1"  # Actual runtime order is reversed from compiled source order.
    assert result["state_input_layer_identity"] == result["state_output_layer_identity"] == "UNVERIFIED"
    assert result["output_mapping"] == "UNVERIFIED_TOOL_FILENAMES" and result["numerical_validation"] == "NOT_RUN_NO_REFERENCE"
    assert result["feature_output_evidence"]["runtime_output_index"] == 0
    assert result["tool_process_id"] == 1337 and result["remote_exitcode"] == 0
    assert result["lock_state"] == "NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT"
    assert all(path.read_bytes() == original[path.name] for path in case["source"].iterdir())
    argv = json.loads((case["out"] / "command.json").read_text())["argv"]
    assert "--direct_from_file=true" in argv and "--modelmanager_itf=V2" in argv and "--times=1" in argv
    assert "--enable_item=1" in argv and "--enable_item=2" not in argv
    for option, count in (("input_type", 89), ("output_type", 82), ("input_format", 89), ("output_format", 82)):
        values = next(arg.split("=", 1)[1] for arg in argv if arg.startswith("--" + option + "="))
        assert len(values.split(",")) == count
        if option.endswith("format"): assert set(values.split(",")) == {"ND"}
    assert not any(arg.startswith(("--input_shape", "--output_shape")) for arg in argv)
    command = next(command for command, label, _ in FakeHdc.instances[0].calls if label == "execute-original-target")
    assert "tool_pid=$!" in command and 'wait "$tool_pid"' in command


@pytest.mark.parametrize("fault", ["kv", "input_changed", "ordinary_ambiguous"])
def test_unproven_or_changed_inputs_prevent_hdc_and_large_payload_reads(case, monkeypatch, fault):
    if fault == "kv":
        path = case["source"] / "past_v.bin"
        with path.open("r+b") as stream: stream.write(b"x")
    elif fault == "input_changed":
        original = planner.validate_prefill_smoke_plan
        def mutate(plan):
            (case["source"] / "effective_length.bin").write_bytes(b"bad")
            return original(plan)
        monkeypatch.setattr(planner, "validate_prefill_smoke_plan", mutate)
    else:
        row = case["native"]["inputs"][0]
        row["shape"], row["raw_dtype"] = [1], case["native"]["dtype_enum_values"]["INT32"]
        rewrite_native(case)
    monkeypatch.setattr(board, "_recorded_file", lambda *a: pytest.fail("blocked proof cannot read compiler payloads"))
    result = execute(case)
    assert result["status"] == "FAILED" and not FakeHdc.instances and FakeHdc.sends == []


def test_disk_budget_stops_before_model_or_source_input_upload(case, monkeypatch):
    monkeypatch.setattr(FakeHdc, "available_kb", 1)
    result = execute(case)
    assert result["status"] == "FAILED" and "disk budget insufficient" in result["error"]
    assert not any(local.endswith((".omc", ".weight", ".bin")) for local, _ in FakeHdc.sends)
    assert result["lock_state"] == "NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT"


def test_phone_global_lock_stops_overlapping_prefill_before_model_or_inputs(case, monkeypatch):
    monkeypatch.setattr(FakeHdc, "busy", True)
    result = execute(case)
    assert result["status"] == "FAILED" and "BOARD_MODEL_IO_BUSY" in result["error"]
    assert FakeHdc.sends == []
    assert result["board_global_lock"] == "/data/local/tmp/.qwen35-target-model-io-lock"


@pytest.mark.parametrize("fault", ["native", "timeout"])
def test_failure_keeps_pid_logs_and_never_retries_or_claims_state_continuity(case, monkeypatch, fault):
    if fault == "native": monkeypatch.setattr(FakeHdc, "status", 7)
    else: monkeypatch.setattr(FakeHdc, "timeout_label", "execute-original-target")
    result = execute(case)
    assert result["status"] == "FAILED" and result["numerical_validation"] == "NOT_RUN_NO_REFERENCE"
    labels = [label for _, label, _ in FakeHdc.instances[0].calls]
    assert labels.count("execute-original-target") == 1
    if fault == "native":
        assert result["tool_process_id"] == 1337 and result["remote_exitcode"] == 7
        assert all((case["out"] / name).is_file() for name in ("tool.stdout", "tool.stderr", "tool.exitcode", "tool.pid"))
        assert result["hilog"]["status"] == "FIXTURE_ONLY"
        assert result["lock_state"] == "NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT"
    else:
        assert result["lock_state"] == "RETAINED_OR_UNCERTAIN" and "release-global-model-io-lock" not in labels


def _main():
    script = Path(__file__).resolve().parents[1] / "targets/cpro/scripts/run_target_prefill_smoke.py"
    return runpy.run_path(str(script))["main"]


def test_cli_defaults_to_plan_only_without_hdc_or_model_upload(case, capsys):
    assert _main()(["--job", str(case["job"]), "--runtime-probe", str(case["probe"]), "--input-dir", str(case["source"]), "--output", str(case["out"])]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["status"] == "PREFILL_SMOKE_PLANNED" and shown["device_execution"] == "NOT_RUN"
    assert not FakeHdc.instances and not (case["out"] / "board").exists()


def test_cli_apply_is_explicit_and_preserves_failed_tool_diagnostics(case, monkeypatch, capsys):
    monkeypatch.setattr(FakeHdc, "status", 7)
    assert _main()(["--job", str(case["job"]), "--runtime-probe", str(case["probe"]), "--input-dir", str(case["source"]), "--output", str(case["out"]), "--apply", "--lock-wait-seconds", "0"]) == 1
    text = capsys.readouterr().out
    assert "fixture external weight initialization failed" in text
    saved = json.loads((case["out"] / "result.json").read_text())
    assert saved["status"] == "FAILED" and saved["tool_process_id"] == 1337


def test_cli_existing_output_is_never_overwritten(case):
    case["out"].mkdir()
    existing = case["out"] / "result.json"
    existing.write_text("retained previous attempt")
    assert _main()(["--job", str(case["job"]), "--runtime-probe", str(case["probe"]), "--input-dir", str(case["source"]), "--output", str(case["out"])]) == 1
    assert existing.read_text() == "retained previous attempt" and not FakeHdc.instances
