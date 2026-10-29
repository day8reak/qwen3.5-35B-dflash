"""Host-only transport fixtures do not establish any actual device ABI."""
import hashlib
import json
from pathlib import Path
import re
import runpy

import pytest

from qwen35_moe_dflash import target_board_tool_inspect as probe


HASH = "a" * 64
HELP = "Usage: model_run_tool_internal --model FILE --input FILE\nversion fixture-only"
OUTPUT_ERROR = ("[WARNING] [debug]: from img skip realpath\n"
    "ERROR: --output_dir must be set on the commandline (default value fails validation)\n"
    "[ERROR] invalid value for --output_dir: ./. directory is not writable: Read-only file system.")


class FakeHdc:
    instances = []
    help_status = 0
    help_text = HELP
    preflight_status = 0
    before_hash = HASH
    after_hash = HASH
    workdir_status = 0
    fail_label = None
    marker_missing = False
    marker_missing_label = None
    mutate_label = None
    mutation = None

    def __init__(self, executable, sn, bip, output, timeout):
        self.configuration = executable, sn, bip, output, timeout
        self.calls = []
        self.instances.append(self)

    def shell(self, command, label, *, timeout):
        self.calls.append((command, label, timeout))
        if label == self.fail_label:
            raise RuntimeError("fixture disconnected; no remote marker")
        if label == self.mutate_label and type(self).mutation is not None:
            type(self).mutation()
        marker = re.search(r"__QWEN35_TOOL_INSPECT_[a-f0-9]+__", command).group()
        result = {
            "target-tool-before": (self.preflight_status, self.before_hash + "  " + probe.DEFAULT_TOOL),
            "target-tool-firmware": (0, "fixture-only HarmonyOS"),
            "target-tool-workdir": (self.workdir_status, "fixture exclusive directory creation"),
            "target-tool-help": (self.help_status, self.help_text),
            "target-tool-after": (0, self.after_hash + "  " + probe.DEFAULT_TOOL),
        }[label]
        status, text = result
        missing = self.marker_missing or label == self.marker_missing_label
        return 0, text + ("" if missing else "\n" + marker + "=" + str(status))


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("HDC", "/selected/hdc")
    monkeypatch.setenv("SN", "fixture-device")
    monkeypatch.setenv("BIP", "fixture-server")
    monkeypatch.setattr(probe, "Hdc", FakeHdc)
    FakeHdc.instances = []
    for key, value in (("help_status", 0), ("help_text", HELP), ("preflight_status", 0),
                       ("before_hash", HASH), ("after_hash", HASH), ("workdir_status", 0),
                       ("fail_label", None), ("marker_missing", False), ("marker_missing_label", None),
                       ("mutate_label", None), ("mutation", None)):
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
    assert report["help"]["observation"] == "HELP_TEXT_OBSERVED"
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


def test_help_flag_description_is_not_an_argument_validation_error(case, monkeypatch):
    monkeypatch.setattr(FakeHdc, "help_text", "Usage: tool [options]\n--output_dir: writable directory; must be set before model execution")
    report = probe.inspect_target_board_tool(case)
    assert report["status"] == "INSPECTED" and report["help"]["observation"] == "HELP_TEXT_OBSERVED"


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


def _prior(case, monkeypatch, *, legacy_status=False):
    """An actual error-only first report is preserved for a narrow new attempt."""
    monkeypatch.setattr(FakeHdc, "help_status", 1)
    monkeypatch.setattr(FakeHdc, "help_text", OUTPUT_ERROR)
    prior_dir = case.parent / "first-probe"
    report = probe.inspect_target_board_tool(prior_dir)
    if legacy_status:
        # This is the old probe's actual observed status; its text still does
        # not establish that help was emitted.
        report["status"] = "INSPECTED"
        (prior_dir / "report.json").write_text(json.dumps(report))
    monkeypatch.setattr(FakeHdc, "help_status", 0)
    monkeypatch.setattr(FakeHdc, "help_text", HELP)
    FakeHdc.instances = []
    return prior_dir, prior_dir / "report.json"


def test_actual_user_error_text_is_not_promoted_to_observed_help(case, monkeypatch):
    monkeypatch.setattr(FakeHdc, "help_status", 1)
    monkeypatch.setattr(FakeHdc, "help_text", OUTPUT_ERROR)
    report = probe.inspect_target_board_tool(case)
    assert report["status"] == "BLOCKED"
    assert report["help"]["observation"] == "OUTPUT_DIRECTORY_VALIDATION_BLOCKED"
    assert report["help"]["text"] == OUTPUT_ERROR
    assert report["tool_identity"] == "SHA256_MATCHED_BEFORE_AND_AFTER"
    assert all(label != "target-tool-workdir" for _, label, _ in FakeHdc.instances[0].calls)


@pytest.mark.parametrize("text,observation", [
    ("ERROR: --model must be set (default value fails validation)", "ARGUMENT_VALIDATION_BLOCKED"),
    ("", "HELP_NOT_OBSERVED"),
    ("[WARNING] loader text alone", "HELP_NOT_OBSERVED"),
])
def test_error_only_or_empty_text_cannot_claim_help_even_with_zero_status(case, monkeypatch, text, observation):
    monkeypatch.setattr(FakeHdc, "help_text", text)
    report = probe.inspect_target_board_tool(case)
    assert report["status"] == "BLOCKED" and report["help"]["observation"] == observation
    assert report["tool_identity"] == "SHA256_MATCHED_BEFORE_AND_AFTER"


@pytest.mark.parametrize("prior_argument", ["directory", "report"])
@pytest.mark.parametrize("legacy_status", [False, True])
def test_bound_writable_help_retry_uses_only_its_exclusive_scratch(case, monkeypatch, prior_argument, legacy_status):
    directory, prior = _prior(case, monkeypatch, legacy_status=legacy_status)
    original = prior.read_bytes()
    argument = directory if prior_argument == "directory" else prior
    report = probe.inspect_target_board_tool(case, writable_help_from=argument)
    assert report["status"] == "INSPECTED" and report["help"]["observation"] == "HELP_TEXT_OBSERVED"
    assert prior.read_bytes() == original
    assert report["previous_inspection"]["path"] == str(prior)
    assert report["previous_inspection"]["sha256"] == hashlib.sha256(original).hexdigest()
    assert report["runtime_input_order"] == report["runtime_tensor_names"] == "NOT_QUERIED"
    scratch = report["writable_help_directory"]
    assert re.fullmatch(r"/data/local/tmp/qwen35-tool-help-[0-9a-f]+", scratch["path"])
    assert scratch["output_dir"] == scratch["path"] + "/outputs" and scratch["status"]
    hdc = FakeHdc.instances[0]
    assert len(hdc.calls) == 5 and all(timeout == 10 for _, _, timeout in hdc.calls)
    creation = next(command for command, label, _ in hdc.calls if label == "target-tool-workdir")
    assert "umask 077" in creation and "mkdir " + scratch["path"] in creation
    assert "mkdir -p" not in creation and "chmod" not in creation and "rm " not in creation
    command = next(command for command, label, _ in hdc.calls if label == "target-tool-help")
    assert "cd " + scratch["path"] in command
    assert "timeout -k 1 8 " + probe.DEFAULT_TOOL in command
    assert "--output_dir=" + scratch["output_dir"] in command and "--help" in command
    assert "read -r probe_owner < .probe-owner" in command
    assert 'test "$probe_owner" = ' + scratch["path"].rsplit("-", 1)[1] in command
    assert report["device_execution"] == "TOOL_HELP_WITH_OWNED_SCRATCH_ONLY"
    assert not any(word in command for word in ("--model", "--input", "--weight_dir", "--enable_item", ".omc"))
    assert all(report[key] == "NOT_RUN" for key in ("model_upload", "model_load", "model_inference"))


@pytest.mark.parametrize("fault", ["artifact", "status", "tool_identity", "tool", "hash", "model_load", "help", "help_sha", "help_bytes", "help_truncated"])
def test_invalid_previous_report_prevents_remote_writes_and_help(case, monkeypatch, fault):
    _, prior = _prior(case, monkeypatch)
    report = json.loads(prior.read_text())
    if fault == "artifact": report["artifact_kind"] = "unrelated-report"
    elif fault == "status": report["status"] = "PREPARING"
    elif fault == "tool_identity": report["tool_identity"] = "UNVERIFIED"
    elif fault == "tool": report["tool"] = "/data/local/tmp/other-tool"
    elif fault == "hash": report["tool_sha256_after"] = "b" * 64
    elif fault == "model_load": report["model_load"] = "ATTEMPTED"
    elif fault == "help_sha": report["help"]["utf8_sha256"] = "b" * 64
    elif fault == "help_bytes": report["help"]["utf8_bytes"] += 1
    elif fault == "help_truncated": report["help"]["truncated"] = True
    else: report["help"]["text"] = HELP
    prior.write_text(json.dumps(report))
    result = probe.inspect_target_board_tool(case, writable_help_from=prior)
    assert result["status"] == "BLOCKED" and not FakeHdc.instances


def test_changed_remote_tool_rejects_bound_retry_before_scratch_creation(case, monkeypatch):
    _, prior = _prior(case, monkeypatch)
    monkeypatch.setattr(FakeHdc, "before_hash", "b" * 64)
    report = probe.inspect_target_board_tool(case, writable_help_from=prior)
    assert report["status"] == "BLOCKED" and "help" not in report
    assert [label for _, label, _ in FakeHdc.instances[0].calls] == ["target-tool-before"]


@pytest.mark.parametrize("failure", ["exit", "transport", "marker"])
def test_scratch_creation_failure_never_invokes_tool_help(case, monkeypatch, failure):
    _, prior = _prior(case, monkeypatch)
    if failure == "exit": monkeypatch.setattr(FakeHdc, "workdir_status", 1)
    elif failure == "transport": monkeypatch.setattr(FakeHdc, "fail_label", "target-tool-workdir")
    else: monkeypatch.setattr(FakeHdc, "marker_missing_label", "target-tool-workdir")
    report = probe.inspect_target_board_tool(case, writable_help_from=prior)
    assert report["status"] == "BLOCKED" and "help" not in report
    assert all(label != "target-tool-help" for _, label, _ in FakeHdc.instances[0].calls)


def test_previous_report_mutation_during_help_is_detected_and_retained(case, monkeypatch):
    _, prior = _prior(case, monkeypatch)
    monkeypatch.setattr(FakeHdc, "mutate_label", "target-tool-help")
    monkeypatch.setattr(FakeHdc, "mutation", lambda: prior.write_text('{}'))
    report = probe.inspect_target_board_tool(case, writable_help_from=prior)
    assert report["status"] == "BLOCKED"
    assert prior.read_text() == '{}' and report["tool_sha256_after"] == HASH


@pytest.mark.parametrize("status,text,observation", [
    (124, HELP, "TIMED_OUT"), (137, HELP, "TIMED_OUT"),
    (1, OUTPUT_ERROR, "OUTPUT_DIRECTORY_VALIDATION_BLOCKED"),
    (1, "ERROR: --model must be set (default value fails validation)", "ARGUMENT_VALIDATION_BLOCKED"),
])
def test_writable_retry_does_not_escalate_after_timeout_or_another_validator(case, monkeypatch, status, text, observation):
    _, prior = _prior(case, monkeypatch)
    monkeypatch.setattr(FakeHdc, "help_status", status)
    monkeypatch.setattr(FakeHdc, "help_text", text)
    report = probe.inspect_target_board_tool(case, writable_help_from=prior)
    assert report["status"] == "BLOCKED" and report["help"]["observation"] == observation
    assert report["tool_sha256_after"] == HASH
    assert len([label for _, label, _ in FakeHdc.instances[0].calls if label == "target-tool-help"]) == 1


def test_cli_passes_bound_prior_directory_and_reports_scratch_mode(case, monkeypatch, capsys):
    directory, prior = _prior(case, monkeypatch, legacy_status=True)
    script = Path(__file__).resolve().parents[1] / "targets/cpro/scripts/inspect_target_board_tool.py"
    main = runpy.run_path(str(script))["main"]
    assert main(["--writable-help-from", str(directory), "--output", str(case)]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["status"] == "INSPECTED"
    assert shown["previous_inspection"]["path"] == str(prior)
    assert shown["board_mutation"] == "UNIQUE_PROBE_DIRECTORY_ONLY"
    assert shown["model_arguments"] == "NONE_SUPPLIED" and shown["model_load"] == "NOT_RUN"
