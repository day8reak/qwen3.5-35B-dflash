"""Host-only transport fixtures do not establish any actual device ABI."""
import json
from pathlib import Path
import re
import runpy

import pytest

from qwen35_moe_dflash import target_board_tool_inspect as probe


HASH = "a" * 64
HELP = "Usage: model_run_tool_internal --model FILE --input FILE\nversion fixture-only"


class FakeHdc:
    instances = []
    help_status = 0
    preflight_status = 0
    after_hash = HASH
    fail_label = None
    marker_missing = False

    def __init__(self, executable, sn, bip, output, timeout):
        self.configuration = executable, sn, bip, output, timeout
        self.calls = []
        self.instances.append(self)

    def shell(self, command, label, *, timeout):
        self.calls.append((command, label, timeout))
        if label == self.fail_label:
            raise RuntimeError("fixture disconnected; no remote marker")
        marker = re.search(r"__QWEN35_TOOL_INSPECT_[a-f0-9]+__", command).group()
        result = {
            "target-tool-before": (self.preflight_status, HASH + "  " + probe.DEFAULT_TOOL),
            "target-tool-firmware": (0, "fixture-only HarmonyOS"),
            "target-tool-help": (self.help_status, HELP),
            "target-tool-after": (0, self.after_hash + "  " + probe.DEFAULT_TOOL),
        }[label]
        status, text = result
        return 0, text + ("" if self.marker_missing else "\n" + marker + "=" + str(status))


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("HDC", "/selected/hdc")
    monkeypatch.setenv("SN", "fixture-device")
    monkeypatch.setenv("BIP", "fixture-server")
    monkeypatch.setattr(probe, "Hdc", FakeHdc)
    FakeHdc.instances = []
    for key, value in (("help_status", 0), ("preflight_status", 0), ("after_hash", HASH),
                       ("fail_label", None), ("marker_missing", False)):
        monkeypatch.setattr(FakeHdc, key, value)
    return tmp_path / "tool-inspect"


def test_only_hash_firmware_and_help_commands_never_prove_slots(case):
    report = probe.inspect_target_board_tool(case)
    assert report["status"] == "INSPECTED"
    assert report["tool_sha256_before"] == report["tool_sha256_after"] == HASH
    assert report["runtime_input_order"] == report["runtime_tensor_names"] == "NOT_QUERIED"
    assert report["hiai_runtime_version"] == "NOT_QUERIED"
    assert all(report[key] == "NOT_RUN" for key in ("model_upload", "model_load", "model_inference"))
    assert report["help"]["text"] == HELP
    hdc = FakeHdc.instances[0]
    assert hdc.configuration[:3] == ("/selected/hdc", "fixture-device", "fixture-server")
    assert len(hdc.calls) == 4
    commands = [row[0] for row in hdc.calls]
    assert "command -v timeout" in commands[0]
    assert commands[2].startswith("timeout -k 1 8 " + probe.DEFAULT_TOOL + " --help 2>&1")
    assert all(row[2] == 10 for row in hdc.calls)
    assert not any(word in "\n".join(commands) for word in (".omc", "file send", "file recv", "--input ", "--model ", "mkdir", "rm "))
    assert json.loads((case / "report.json").read_text()) == report


@pytest.mark.parametrize("help_status", [1, 2])
def test_nonzero_help_exit_is_retained_without_guessing_support(case, monkeypatch, help_status):
    monkeypatch.setattr(FakeHdc, "help_status", help_status)
    report = probe.inspect_target_board_tool(case)
    assert report["status"] == "INSPECTED" and report["help"]["returncode"] == help_status
    assert report["runtime_input_order"] == "NOT_QUERIED"


@pytest.mark.parametrize("help_status", [124, 137])
def test_remote_help_timeout_is_blocked_after_final_hash(case, monkeypatch, help_status):
    monkeypatch.setattr(FakeHdc, "help_status", help_status)
    report = probe.inspect_target_board_tool(case)
    assert report["status"] == "BLOCKED" and "timeout" in report["error"]
    assert report["tool_sha256_after"] == HASH
    assert len(FakeHdc.instances[0].calls) == 4


def test_changed_executable_cannot_be_bound_to_help(case, monkeypatch):
    monkeypatch.setattr(FakeHdc, "after_hash", "b" * 64)
    report = probe.inspect_target_board_tool(case)
    assert report["status"] == "BLOCKED" and "changed" in report["error"]
    assert "tool_identity" not in report


def test_preflight_failure_prevents_help(case, monkeypatch):
    monkeypatch.setattr(FakeHdc, "preflight_status", 1)
    report = probe.inspect_target_board_tool(case)
    assert report["status"] == "BLOCKED" and len(FakeHdc.instances[0].calls) == 1
    assert "help" not in report


@pytest.mark.parametrize("missing", ["SN", "BIP"])
def test_missing_selected_connection_cannot_probe_another_device(case, monkeypatch, missing):
    monkeypatch.delenv(missing)
    report = probe.inspect_target_board_tool(case)
    assert report["status"] == "BLOCKED" and missing in report["error"]
    assert not FakeHdc.instances and report["device_execution"] == "NOT_RUN"


@pytest.mark.parametrize("value", ["relative/tool", "/data/../tool", "/data/tool;echo", "/data/tool\n"])
def test_unsafe_tool_paths_prevent_connection(case, value):
    report = probe.inspect_target_board_tool(case, tool=value)
    assert report["status"] == "BLOCKED" and not FakeHdc.instances


@pytest.mark.parametrize("value", [0, 2, 11, True, 10.0])
def test_timeout_outside_bounded_integer_contract_prevents_connection(case, value):
    report = probe.inspect_target_board_tool(case, timeout_seconds=value)
    assert report["status"] == "BLOCKED" and not FakeHdc.instances


def test_transport_failure_is_not_classified_as_device_absence(case, monkeypatch):
    monkeypatch.setattr(FakeHdc, "fail_label", "target-tool-help")
    report = probe.inspect_target_board_tool(case)
    assert report["status"] == "BLOCKED" and report["stage"] == "help"
    assert "disconnected" in report["error"] and "help" not in report


def test_missing_remote_command_marker_cannot_pass(case, monkeypatch):
    monkeypatch.setattr(FakeHdc, "marker_missing", True)
    report = probe.inspect_target_board_tool(case)
    assert report["status"] == "BLOCKED" and "unique completion marker" in report["error"]


def test_never_overwrite_report_or_write_outside_active_run(case):
    probe.inspect_target_board_tool(case)
    original = (case / "report.json").read_bytes()
    with pytest.raises(FileExistsError):
        probe.inspect_target_board_tool(case)
    with pytest.raises(ValueError, match="below AI_RUN_DIR"):
        probe.inspect_target_board_tool(case.parent.parent / "not-owned")
    assert (case / "report.json").read_bytes() == original


def test_cli_reports_probe_and_preserves_no_model_claim(case, capsys):
    script = Path(__file__).resolve().parents[1] / "targets/cpro/scripts/inspect_target_board_tool.py"
    main = runpy.run_path(str(script))["main"]
    assert main(["--output", str(case)]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["status"] == "INSPECTED" and shown["model_load"] == "NOT_RUN"
    assert shown["runtime_input_order"] == "NOT_QUERIED"
