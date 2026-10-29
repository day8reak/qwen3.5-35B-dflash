"""Print-only summary of four retained target stages; never execute anything."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat


_MAX_JSON = 8 << 20
_TAIL_BYTES = 4096
_DEFAULTS = {"prefill_probe": "target-prefill64-runtime-io", "decode_probe": "target-decode1-runtime-io",
             "prefill_plan": "target-prefill64-smoke-plan", "prefill_smoke": "target-prefill64-smoke"}
_KINDS = {"prefill_probe": "qwen35-target-model-io-probe", "decode_probe": "qwen35-target-model-io-probe",
          "prefill_plan": "qwen35-target-prefill-smoke-job", "prefill_smoke": "qwen35-target-prefill-smoke-job"}
_DISPLAY = ("status", "stage", "mode", "error", "evidence_error", "native_evidence_error", "source_error",
            "lock_state", "lock_release_error", "board_global_lock", "board_attempt", "profile", "sequence_length",
            "input_count", "output_count", "compiled_boundary_count_check", "manager_init", "descriptor_stage",
            "model_restore", "device_execution", "native_status", "runtime_stage", "remote_exitcode", "tool_process_id",
            "dtype_enum", "format_enum", "dtype_enum_values", "format_enum_values", "source_preservation")


def _path(value):
    result = Path(os.path.abspath(Path(value).expanduser()))
    cursor = Path(result.anchor)
    for part in result.parts[1:]:
        cursor /= part
        try:
            info = cursor.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("summary path contains a symlink: " + str(cursor))
    return result


def _stamp(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _read(path, *, tail=False):
    path = _path(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("summary evidence is not a regular file: " + str(path))
    if not tail and not 0 < before.st_size <= _MAX_JSON:
        raise ValueError("summary metadata must be nonempty and at most8MiB: " + str(path))
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        if _stamp(os.fstat(stream.fileno())) != _stamp(before):
            raise ValueError("summary evidence changed before read")
        if tail:
            stream.seek(max(0, before.st_size - _TAIL_BYTES))
        raw = stream.read(_TAIL_BYTES if tail else _MAX_JSON + 1)
        if (_stamp(os.fstat(stream.fileno())) != _stamp(before) or _stamp(path.lstat()) != _stamp(before)):
            raise ValueError("summary evidence changed during read")
    if tail:
        return {"path": str(path), "source_bytes": before.st_size, "tail_read_bytes": len(raw),
                "truncated": before.st_size > len(raw), "text": raw.decode("utf-8", errors="replace")}
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate summary metadata key: " + key)
            result[key] = value
        return result
    def finite(value):
        raise ValueError("nonfinite summary metadata: " + value)
    def number(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("nonfinite summary metadata number")
        return result
    value = json.loads(raw, object_pairs_hook=unique, parse_constant=finite, parse_float=number)
    if not isinstance(value, dict):
        raise ValueError("summary metadata root must be an object")
    return value


def _outputs(value):
    retrieved = value.get("retrieved", {})
    if not isinstance(retrieved, dict):
        return [], "retrieved output metadata is not an object"
    rows = []
    for name, row in retrieved.items():
        if not isinstance(name, str) or not name.startswith("outputs/"):
            continue
        if (len(PurePosixPath(name).parts) < 2 or any(part in {"", ".", ".."} for part in name.split("/"))
                or not isinstance(row, dict) or type(row.get("bytes")) is not int or row["bytes"] <= 0
                or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256", "")))):
            return [], "retrieved output file has no positive byte count and SHA256 declaration"
        rows.append(name)
    return rows, None


def _display(value):
    result = {}
    for key in _DISPLAY:
        if key not in value:
            continue
        item = value[key]
        if key in {"dtype_enum_values", "format_enum_values"}:
            if (not isinstance(item, dict) or len(item) > 32
                    or any(not isinstance(name, str) or len(name) > 64 or type(number) is not int for name, number in item.items())):
                raise ValueError("malformed or oversized runtime enum value table")
            result[key] = item
        elif isinstance(item, str):
            result[key] = item[:4096]
        elif item is None or isinstance(item, (bool, int, float)):
            result[key] = item
        else:
            result[key] = json.dumps(item, ensure_ascii=False, allow_nan=False)[:4096]
    return result


def _stage(directory, role):
    row, value, board = {"directory": str(directory), "metadata_state": "MISSING"}, None, None
    try:
        root = _path(directory)
        if root.exists() and not root.is_dir():
            raise ValueError("stage root is not a directory")
        try:
            value = _read(root / "result.json")
        except FileNotFoundError:
            value = None
        if value is not None:
            if value.get("artifact_kind") != _KINDS[role]:
                raise ValueError("stage report has an unexpected artifact_kind")
            row.update(metadata_state="READ", report=str(root / "result.json"))
            row.update(_display(value))
        if role == "prefill_smoke":
            try:
                board = _read(root / "board/result.json")
            except FileNotFoundError:
                pass
            if board is not None:
                if board.get("artifact_kind") != "qwen35-original-target-tool-execution":
                    raise ValueError("unexpected optional board result kind")
                row["board"] = _display(board)
                if value is not None and any(value.get(key) != board.get(key) for key in ("status", "remote_exitcode", "device_execution", "lock_state", "source_preservation", "retrieved", "retrieval_errors")):
                    raise ValueError("outer and board receipts disagree on execution")
                if value is None and board.get("status") in {"FAILED", "BLOCKED", "RUNTIME_FAILED"}:
                    row["status"] = board["status"]
                    row["board_only_evidence"] = True
        tails = []
        for name in (("board/tool.stderr", "board/tool.stdout") if role == "prefill_smoke" else ("model-io.stderr", "model-io.stdout")):
            try:
                tails.append(_read(root / name, tail=True))
            except FileNotFoundError:
                pass
        if tails:
            row["log_tails"] = tails
    except (OSError, ValueError, TypeError, RecursionError) as error:
        row.update(metadata_state="INVALID", inspection_error=f"{type(error).__name__}: {error}")
        value = None
    return row, value, board


def _probe_ready(value, sequence):
    if not isinstance(value, dict):
        return False
    profile = "cpro-40block-prefill-64" if sequence == 64 else "cpro-40block-decode-1"
    return (value.get("status") == "IO_METADATA_OBSERVED" and value.get("profile") == profile
        and type(value.get("sequence_length")) is int and value["sequence_length"] == sequence
        and type(value.get("input_count")) is int and value["input_count"] == 89
        and type(value.get("output_count")) is int and value["output_count"] == 82
        and value.get("compiled_boundary_count_check") == "MATCHED_89_INPUT_82_OUTPUT_COUNTS_ONLY"
        and value.get("manager_init") == "NOT_RUN" and value.get("descriptor_stage") == "restore-only"
        and value.get("lock_state") == "NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT"
        and value.get("source_preservation") == "UNCHANGED_METADATA_AND_SMALL_PRIMARY_SHA256; WEIGHTS_NOT_OPENED"
        and value.get("retrieval_errors") == []
        and not any(value.get(key) for key in ("lock_release_error", "native_evidence_error", "source_error", "error")))


def inspect_target_morning_status(*, prefill_probe=None, decode_probe=None, prefill_plan=None, prefill_smoke=None):
    """Read small saved evidence and return advisory next steps without actions."""
    arguments = {"prefill_probe": prefill_probe, "decode_probe": decode_probe,
                 "prefill_plan": prefill_plan, "prefill_smoke": prefill_smoke}
    if any(value is None for value in arguments.values()) and not os.environ.get("AI_RUN_DIR"):
        raise ValueError("AI_RUN_DIR is required for omitted stage-directory arguments")
    base = Path(os.environ.get("AI_RUN_DIR", "."))
    stages, values = {}, {}
    for role, value in arguments.items():
        directory = Path(value) if value is not None else base / _DEFAULTS[role]
        stages[role], values[role], _ = _stage(directory, role)
    failures = [key for key, row in stages.items() if row["metadata_state"] == "INVALID"
                or row.get("status") in {"FAILED", "BLOCKED", "RUNTIME_FAILED", "MODEL_IO_FAILED"}
                or row.get("lock_state") == "RETAINED_OR_UNCERTAIN"
                or any(row.get(name) for name in ("lock_release_error", "native_evidence_error", "source_error"))]
    probes_ready = _probe_ready(values["prefill_probe"], 64) and _probe_ready(values["decode_probe"], 1)
    plan = values["prefill_plan"]
    plan_ready = (isinstance(plan, dict) and plan.get("status") == "PREFILL_SMOKE_PLANNED"
                  and plan.get("mode") == "PLAN_ONLY" and plan.get("input_count") == 89 and plan.get("output_count") == 82)
    smoke = values["prefill_smoke"]
    outputs, output_error = _outputs(smoke) if isinstance(smoke, dict) else ([], None)
    smoke_observed = (isinstance(smoke, dict) and smoke.get("status") == "EXECUTED_UNVALIDATED"
        and smoke.get("mode") == "APPLY" and type(smoke.get("remote_exitcode")) is int and smoke["remote_exitcode"] == 0
        and smoke.get("device_execution") == "EXECUTED_OUTPUTS_RETRIEVED"
        and smoke.get("retrieval_errors") == [] and bool(outputs) and output_error is None
        and smoke.get("lock_state") == "NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT"
        and smoke.get("source_preservation") == "UNCHANGED_PREFILL_PLAN_METADATA_AND_ORIGINAL_INPUT_SHA256"
        and not any(smoke.get(key) for key in ("lock_release_error", "native_evidence_error", "source_error", "error", "evidence_error")))
    if isinstance(smoke, dict):
        stages["prefill_smoke"].update(retrieved_output_file_count=len(outputs), first_prefill_observation=smoke_observed)
        if smoke.get("status") == "EXECUTED_UNVALIDATED" and not smoke_observed:
            failures.append("prefill_smoke")
            stages["prefill_smoke"]["observation_error"] = output_error or "exit0 and nonempty hash-declared output files were not established"
    if failures:
        next_step = "INSPECT_RETAINED_FAILURE"
    elif not probes_ready:
        next_step = "WAIT_FOR_RUNTIME_PROBES" if any(values[key] is None for key in ("prefill_probe", "decode_probe")) else "INSPECT_RUNTIME_METADATA"
    elif smoke_observed and plan_ready:
        next_step = "PREFILL_OUTPUTS_OBSERVED_DECODE_STATE_AND_FEATURE_MAPPING_PENDING"
    elif plan_ready:
        next_step = "READY_FOR_PREFILL_EXECUTION" if smoke is None else "WAIT_FOR_PREFILL_EXECUTION_RESULT"
    else:
        next_step = "READY_FOR_PREFILL_PLAN"
    return {"schema_version": 1, "artifact_kind": "qwen35-target-morning-status", "status": "READ_ONLY_SUMMARY",
        "next": next_step, "failure_stages": sorted(set(failures)), "stages": stages,
        "both_runtime_metadata_reports_ready": probes_ready, "prefill_plan_report_ready": plan_ready,
        "first_prefill_outputs_observed": smoke_observed and probes_ready and plan_ready,
        "actions_executed": "NONE", "model_completion": "NOT_ESTABLISHED", "state_continuity": "UNVERIFIED",
        "numerical_validation": "NOT_ESTABLISHED", "performance_speedup": "NOT_MEASURED",
        "scope": "Small saved metadata and up to4096bytes per log tail only. No HDC, model/weight/output-payload reads, writes, process control or automatic next command. First-prefill output retrieval alone proves no slot mapping, state feedback, accuracy or full DFlash completion."}
