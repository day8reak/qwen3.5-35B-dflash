"""Observe the existing phone tool's identity and help without a model argument."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import uuid

from .artifacts import output_path, save_json
from .hdc_run import Hdc


DEFAULT_TOOL = "/data/local/tmp/tools_sysdbg/model_run_tool_internal"
_PATH = re.compile(r"/[A-Za-z0-9_./-]+\Z")


def _tool_path(value):
    if (not isinstance(value, str) or not _PATH.fullmatch(value)
            or ".." in PurePosixPath(value).parts):
        raise ValueError("board tool must be an absolute path with simple components")
    return value


def _shell_result(hdc, command, label, timeout):
    # HDC versions differ in how they propagate remote nonzero exit codes. The
    # outer shell succeeds only after recording this specific command's status;
    # Hdc.shell independently requires its own unique transport completion marker.
    marker = "__QWEN35_TOOL_INSPECT_" + uuid.uuid4().hex + "__"
    payload = (command + " 2>&1\ncommand_status=$?\nprintf '\\n" + marker
               + "=%s\\n' \"$command_status\"\nexit 0")
    _, text = hdc.shell(payload, label, timeout=timeout)
    matches = list(re.finditer(r"^" + re.escape(marker) + r"=([0-9]+)\r?$", text, re.MULTILINE))
    if len(matches) != 1:
        raise ValueError("tool inspection command did not return its unique completion marker")
    return int(matches[0].group(1)), text[:matches[0].start()].rstrip("\r\n")


def _text_record(text):
    raw = text.encode("utf-8")
    limit = 65536
    return {"utf8_bytes": len(raw), "utf8_sha256": hashlib.sha256(raw).hexdigest(),
            "text": raw[:limit].decode("utf-8", errors="replace"),
            "truncated": len(raw) > limit,
            "scope": "Decoded HDC text; complete response remains in calls/*.json"}


def _tool_hash(text, tool):
    hashes = [match.group(1) for line in text.splitlines()
              if (match := re.fullmatch(r"([0-9a-fA-F]{64}) [ *]" + re.escape(tool), line))]
    if len(hashes) != 1:
        raise ValueError("selected board tool has no unique sha256sum record")
    return hashes[0].lower()


def inspect_target_board_tool(output_dir, *, tool=DEFAULT_TOOL, timeout_seconds=10):
    """Query four bounded shell commands; never send, load, or run a model."""
    out = output_path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "artifact_kind": "qwen35-target-board-tool-inspection",
              "status": "BLOCKED", "stage": "configuration", "tool": tool,
              "model_upload": "NOT_RUN", "model_load": "NOT_RUN", "model_inference": "NOT_RUN",
              "device_execution": "NOT_RUN",
              "runtime_input_order": "NOT_QUERIED", "runtime_tensor_names": "NOT_QUERIED",
              "hiai_runtime_version": "NOT_QUERIED", "numerical_validation": "NOT_RUN",
              "scope": "Existing tool SHA256, firmware text and --help only; no model, native tensor descriptors or input-slot proof"}
    try:
        tool = _tool_path(tool)
        if type(timeout_seconds) is not int or not 3 <= timeout_seconds <= 10:
            raise ValueError("per-call timeout must be an integer between 3 and 10 seconds")
        missing = [key for key in ("SN", "BIP") if not os.environ.get(key)]
        if missing:
            raise ValueError("selected HDC environment is missing " + ", ".join(missing))
        hdc = Hdc(os.environ.get("HDC", "hdc"), os.environ["SN"], os.environ["BIP"], out, timeout_seconds)
        report["host_call_timeout_seconds"] = timeout_seconds
        report["remote_help_timeout_seconds"] = timeout_seconds - 2
        report["remote_help_kill_after_seconds"] = 1
        report["stage"] = "preflight"
        report["device_execution"] = "READ_ONLY_TOOL_PREFLIGHT_ONLY"
        preflight = ("command -v sha256sum >/dev/null && command -v timeout >/dev/null && "
                     "test -f " + shlex.quote(tool) + " && test -x " + shlex.quote(tool)
                     + " && sha256sum " + shlex.quote(tool))
        status, text = _shell_result(hdc, preflight, "target-tool-before", timeout_seconds)
        report["preflight"] = {"returncode": status, **_text_record(text)}
        if status:
            raise ValueError("selected board tool, sha256sum or remote timeout is unavailable; inspect preflight text")
        before = _tool_hash(text, tool)
        report["tool_sha256_before"] = before
        report["stage"] = "firmware"
        status, text = _shell_result(hdc, "param get const.ohos.fullname", "target-tool-firmware", timeout_seconds)
        report["firmware"] = {"returncode": status, **_text_record(text)}
        report["stage"] = "help"
        status, text = _shell_result(hdc, "timeout -k 1 " + str(timeout_seconds - 2) + " "
                                    + shlex.quote(tool) + " --help", "target-tool-help", timeout_seconds)
        report["help"] = {"returncode": status, "status": "TEXT_RETAINED" if text else "NO_TEXT",
                          **_text_record(text)}
        report["tool_version"] = "HELP_TEXT_RETAINED_NO_SEPARATE_VERSION_INVOCATION"
        report["stage"] = "identity-after"
        status, text = _shell_result(hdc, "sha256sum " + shlex.quote(tool), "target-tool-after", timeout_seconds)
        report["identity_after"] = {"returncode": status, **_text_record(text)}
        if status:
            raise ValueError("could not hash selected board tool after help")
        report["tool_sha256_after"] = _tool_hash(text, tool)
        if report["tool_sha256_after"] != before:
            raise ValueError("selected board tool changed during inspection")
        if report["help"]["returncode"] in {124, 137}:
            raise ValueError("remote tool help exceeded its timeout; no model was supplied")
        report.update(status="INSPECTED", stage="complete", tool_identity="SHA256_MATCHED_BEFORE_AND_AFTER")
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    report["report"] = str(out / "report.json")
    save_json(out / "report.json", report)
    return report
