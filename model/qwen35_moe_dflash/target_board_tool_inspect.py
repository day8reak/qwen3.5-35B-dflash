"""Observe the existing phone tool's identity and help without a model argument."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import uuid

from .artifacts import output_path, save_json
from .hdc_run import Hdc
from .target_export_reuse import _absolute, _read_small, _unique_object, _reject_constant


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


def _help_observation(text, status):
    if status in {124, 137}:
        return "TIMED_OUT"
    if re.search(r"(?i)ERROR:\s*--output_dir|invalid value for --output_dir", text):
        return "OUTPUT_DIRECTORY_VALIDATION_BLOCKED"
    if re.search(r"(?i)invalid value for --|ERROR:\s*--\S+.*must be set", text):
        return "ARGUMENT_VALIDATION_BLOCKED"
    if status >= 128:
        return "HELP_PROCESS_FAILED"
    if (re.search(r"(?i)\b(?:usage|flags|options|parameters)\b", text)
            and re.search(r"(?<![A-Za-z0-9_])--[a-z][a-z0-9_-]*", text)):
        return "HELP_TEXT_OBSERVED"
    return "HELP_NOT_OBSERVED"


def _previous_inspection(value, tool):
    path = _absolute(value)
    if path.is_dir():
        path = path / "report.json"
    raw, identity = _read_small(path, 1 << 20)
    prior = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    if (not isinstance(prior, dict)
            or prior.get("artifact_kind") != "qwen35-target-board-tool-inspection"
            or prior.get("status") not in {"INSPECTED", "BLOCKED"}
            or prior.get("tool") != tool
            or prior.get("tool_identity") != "SHA256_MATCHED_BEFORE_AND_AFTER"
            or not re.fullmatch(r"[0-9a-f]{64}", str(prior.get("tool_sha256_before", "")))
            or prior.get("tool_sha256_before") != prior.get("tool_sha256_after")
            or any(prior.get(key) != "NOT_RUN" for key in ("model_upload", "model_load", "model_inference"))):
        raise ValueError("writable help retry requires an identity-matched prior inspection without model actions")
    help_row = prior.get("help", {})
    if (not isinstance(help_row, dict) or not isinstance(help_row.get("text"), str)
            or type(help_row.get("returncode")) is not int
            or _help_observation(help_row["text"], help_row["returncode"]) != "OUTPUT_DIRECTORY_VALIDATION_BLOCKED"):
        raise ValueError("writable help retry requires the observed output-directory validation failure")
    raw_help = help_row["text"].encode("utf-8")
    if (help_row.get("truncated") is not False or help_row.get("utf8_bytes") != len(raw_help)
            or help_row.get("utf8_sha256") != hashlib.sha256(raw_help).hexdigest()):
        raise ValueError("prior inspection must retain complete hash-bound validation text")
    return prior, identity


def inspect_target_board_tool(output_dir, *, tool=DEFAULT_TOOL, timeout_seconds=10,
                              writable_help_from=None):
    """Inspect tool help; a bound retry may create one unique scratch folder.

    Never upload a model or supply model/input/weight arguments. Help output
    does not establish runtime tensor order or the tool's internal behavior.
    """
    out = output_path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "artifact_kind": "qwen35-target-board-tool-inspection",
              "status": "BLOCKED", "stage": "configuration", "tool": tool,
              "model_upload": "NOT_RUN", "model_load": "NOT_RUN", "model_inference": "NOT_RUN",
              "device_execution": "NOT_RUN",
              "runtime_input_order": "NOT_QUERIED", "runtime_tensor_names": "NOT_QUERIED",
              "hiai_runtime_version": "NOT_QUERIED", "numerical_validation": "NOT_RUN",
              "board_mutation": "NONE", "model_arguments": "NONE_SUPPLIED",
              "scope": "Existing tool SHA256, firmware text and --help only; no model, native tensor descriptors or input-slot proof"}
    try:
        tool = _tool_path(tool)
        if type(timeout_seconds) is not int or not 3 <= timeout_seconds <= 10:
            raise ValueError("per-call timeout must be an integer between 3 and 10 seconds")
        previous = previous_id = None
        if writable_help_from is not None:
            previous, previous_id = _previous_inspection(writable_help_from, tool)
            report["previous_inspection"] = previous_id
        missing = [key for key in ("SN", "BIP") if not os.environ.get(key)]
        if missing:
            raise ValueError("selected HDC environment is missing " + ", ".join(missing))
        hdc = Hdc(os.environ.get("HDC", "hdc"), os.environ["SN"], os.environ["BIP"], out, timeout_seconds)
        report["host_call_timeout_seconds"] = timeout_seconds
        report["remote_help_timeout_seconds"] = timeout_seconds - 2
        report["remote_help_kill_after_seconds"] = 1
        report["stage"] = "preflight"
        report["device_execution"] = "READ_ONLY_TOOL_PREFLIGHT_ONLY"
        if previous is not None:
            report["device_execution"] = "TOOL_HELP_WITH_OWNED_SCRATCH_ONLY"
        preflight = ("command -v sha256sum >/dev/null && command -v timeout >/dev/null && "
                     "test -f " + shlex.quote(tool) + " && test -x " + shlex.quote(tool)
                     + " && sha256sum " + shlex.quote(tool))
        status, text = _shell_result(hdc, preflight, "target-tool-before", timeout_seconds)
        report["preflight"] = {"returncode": status, **_text_record(text)}
        if status:
            raise ValueError("selected board tool, sha256sum or remote timeout is unavailable; inspect preflight text")
        before = _tool_hash(text, tool)
        report["tool_sha256_before"] = before
        if previous is not None and before != previous["tool_sha256_before"]:
            raise ValueError("selected board tool differs from the prior inspection; no writable help directory created")
        report["stage"] = "firmware"
        status, text = _shell_result(hdc, "param get const.ohos.fullname", "target-tool-firmware", timeout_seconds)
        report["firmware"] = {"returncode": status, **_text_record(text)}
        command = "timeout -k 1 " + str(timeout_seconds - 2) + " " + shlex.quote(tool)
        if previous is not None:
            report["stage"] = "writable-help-directory"
            token = uuid.uuid4().hex
            scratch = "/data/local/tmp/qwen35-tool-help-" + token
            outputs = scratch + "/outputs"
            report["writable_help_directory"] = {"path": scratch, "output_dir": outputs,
                "owner_marker": scratch + "/.probe-owner", "status": "CREATION_REQUESTED",
                "retention": "Preserved for inspection; no automatic deletion"}
            report["board_mutation"] = "UNIQUE_PROBE_DIRECTORY_ONLY"
            setup = ("umask 077 && mkdir " + shlex.quote(scratch) + " && cd " + shlex.quote(scratch)
                     + " && printf '%s\\n' " + shlex.quote(token) + " > .probe-owner"
                     + " && mkdir outputs && test -w . && test -w outputs && pwd -P")
            status, text = _shell_result(hdc, setup, "target-tool-workdir", timeout_seconds)
            report["writable_help_directory"].update(returncode=status, **_text_record(text))
            if status:
                raise ValueError("exclusive writable help-directory preparation failed; no help invocation")
            report["writable_help_directory"]["status"] = "CREATED_EXCLUSIVELY"
            command = ("cd " + shlex.quote(scratch) + " && test -f .probe-owner && "
                       + "read -r probe_owner < .probe-owner && test \"$probe_owner\" = "
                       + shlex.quote(token) + " && " + command + " --output_dir=" + shlex.quote(outputs))
        report["stage"] = "help"
        status, text = _shell_result(hdc, command + " --help", "target-tool-help", timeout_seconds)
        report["help"] = {"returncode": status, "status": "TEXT_RETAINED" if text else "NO_TEXT",
                          "observation": _help_observation(text, status),
                          **_text_record(text)}
        report["tool_version"] = ("HELP_TEXT_RETAINED_NO_SEPARATE_VERSION_INVOCATION"
                                  if report["help"]["observation"] == "HELP_TEXT_OBSERVED" else "NOT_OBSERVED")
        report["stage"] = "identity-after"
        status, text = _shell_result(hdc, "sha256sum " + shlex.quote(tool), "target-tool-after", timeout_seconds)
        report["identity_after"] = {"returncode": status, **_text_record(text)}
        if status:
            raise ValueError("could not hash selected board tool after help")
        report["tool_sha256_after"] = _tool_hash(text, tool)
        if report["tool_sha256_after"] != before:
            raise ValueError("selected board tool changed during inspection")
        report["tool_identity"] = "SHA256_MATCHED_BEFORE_AND_AFTER"
        if previous_id is not None and _read_small(Path(previous_id["path"]), 1 << 20)[1] != previous_id:
            raise ValueError("prior inspection changed during writable help retry; all evidence retained")
        if report["help"]["returncode"] in {124, 137}:
            raise ValueError("remote tool help exceeded its timeout; no model was supplied")
        if report["help"]["observation"] != "HELP_TEXT_OBSERVED":
            report["next"] = "Inspect retained validation text; no model arguments or automatic further tool invocation"
            raise ValueError("tool identity observed but actual option help was not obtained: " + report["help"]["observation"])
        report.update(status="INSPECTED", stage="complete")
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    report["report"] = str(out / "report.json")
    save_json(out / "report.json", report)
    return report
