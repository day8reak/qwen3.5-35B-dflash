"""Read-only evidence gate before retrying a stopped static target compilation.

Probe success establishes the source path predicate and Python IPC, never an
embedded OMG, native CCE, complete-model or device pass.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re

from .compile import _OMG_IPC_MULTIPROCESSING
from .tbe_tmpdir_probe import _CHILD, _MAX_SOURCE, _SOURCE, _predicate
from .target_export_reuse import _absolute, _directory, _read_small, _reject_constant, _unique_object


_MAX_JSON = 16 << 20
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _same(left, right):
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def _read(path, bindings, limit=_MAX_JSON, *, decode=True):
    raw, identity = _read_small(path, limit)
    bindings.append(identity)
    if not decode:
        return raw, identity
    value = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError("retry preflight metadata must be an object")
    return value, identity


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _quiescent(row, proc_root):
    pid, start = row.get("pid"), row.get("start_ticks")
    _require(type(pid) is int and pid > 1 and type(start) is int and start >= 0,
             "stop receipt has an invalid process identity")
    try:
        fd = os.open(proc_root / str(pid) / "stat", os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as stream:
            raw = stream.read(65537)
    except (FileNotFoundError, ProcessLookupError):
        return {"pid": pid, "recorded_start_ticks": start, "status": "NOT_PRESENT"}
    _require(len(raw) <= 65536, "process stat exceeds the preflight read limit")
    fields = raw.decode().rsplit(")", 1)[1].split()
    _require(len(fields) >= 20, "process stat is incomplete")
    current = int(fields[19])
    _require(current != start or fields[0] == "Z",
             f"recorded compiler/driver process {pid} is still present; wait for final audit and worker exit")
    return {"pid": pid, "recorded_start_ticks": start, "current_start_ticks": current,
            "status": "EXITED_ZOMBIE" if current == start else "PID_REUSED_OTHER_GENERATION"}


def validate_static_retry_gate(failed_job, *, tmpdir_probe, stop_receipt, proc_root=Path("/proc")):
    """Verify complete small receipts and quiescence; never access model payloads."""
    bindings = []
    job, probe, stopped = map(_absolute, (failed_job, tmpdir_probe, stop_receipt))
    run_value = os.environ.get("AI_RUN_DIR")
    _require(bool(run_value), "AI_RUN_DIR is required for retry preflight")
    run = _absolute(run_value)
    for path in (job, probe, stopped):
        _require(run in path.parents, "retry source/probe/stop receipts must be below AI_RUN_DIR")
        with _directory(path):
            pass
    _require(len({job, probe, stopped}) == 3 and job not in probe.parents and job not in stopped.parents,
             "probe and stop receipts must be separate from the retained job")
    comp = job / "compilation"
    outer, _ = _read(job / "result.json", bindings)
    compiled, compiled_id = _read(comp / "report.json", bindings)
    audit, _ = _read(comp / "retained-input-reuse.json", bindings)
    preservation = compiled.get("source_preservation")
    _require(isinstance(preservation, dict), "final compilation source preservation is missing")
    _require(outer.get("artifact_kind") == "qwen35-original-target-job" and outer.get("status") == "FAILED"
             and compiled.get("status") == "FAILED" and compiled.get("model_compilation") == "FAILED"
             and outer.get("compilation_report_sha256") == compiled_id["sha256"],
             "retained static job has no bound final failed compilation report")
    _require(audit.get("status") == "PASSED" and _same(compiled.get("retained_input_reuse"), audit)
             and not compiled.get("retained_input_audit_error") and not compiled.get("audit_error")
             and preservation.get("status") == "UNCHANGED_BY_STAT" and _same(preservation.get("changed"), []),
             "retained static job has not completed its input/compatibility audit")
    command, command_id = _read(comp / "compile-command.json", bindings, 1 << 20)
    _, environment_id = _read(comp / "omg-environment.json", bindings, 1 << 20)
    source, source_id = _read(comp / _SOURCE, bindings, _MAX_SOURCE, decode=False)
    _require(command.get("cwd") == str(comp / "compile-work/generated"), "failed command cwd differs from its job")

    report, _ = _read(probe / "report.json", bindings)
    child, _ = _read(probe / "child-result.json", bindings, 1 << 20)
    probe_command, _ = _read(probe / "command.json", bindings, 1 << 20)
    validator, validator_id = _read(probe / "validator-source.py", bindings, _MAX_SOURCE, decode=False)
    script, _ = _read(probe / "probe-child.py", bindings, 1 << 20, decode=False)
    _require(report.get("artifact_kind") == "qwen35-tbe-tmpdir-probe" and report.get("status") == "PROBE_PASSED"
             and type(report.get("returncode")) is int and report["returncode"] == 0
             and report.get("job") == str(job) and _same(report.get("child"), child)
             and all(report.get(key) == value for key, value in {"weight_payload_access": "NONE", "model_compilation": "NOT_RUN",
                    "device_execution": "NOT_RUN", "embedded_omg_startup": "NOT_TESTED", "input_mutation": "NONE"}.items()),
             "completed source-directory probe does not bind this failed job")
    _require(_same(report.get("source"), source_id) and _same(report.get("input_command"), command_id)
             and _same(report.get("input_environment"), environment_id)
             and validator == _predicate(source).encode() and report.get("source_predicate_sha256") == validator_id["sha256"]
             and script == _CHILD.encode(), "probe source, inputs or executable body changed")
    ipc = report.get("probe_environment", {})
    _require(isinstance(ipc, dict), "probe environment must be an object")
    physical, short = ipc.get("physical_path"), ipc.get("path")
    _require(isinstance(physical, str) and Path(physical).parent == probe and Path(physical).name.startswith(".ipc-")
             and isinstance(short, str) and re.fullmatch(r"/proc/[0-9]+/fd/[0-9]+/\.ipc-[A-Za-z0-9_-]+", short)
             and Path(short).name == Path(physical).name and ipc.get("compiler_tmpdir") == physical
             and ipc.get("multiprocessing_tmpdir") == short, "probe temporary/IPC paths are not bound")
    overlay = child.get("multiprocessing_overlay", {})
    planned_overlay = ipc.get("multiprocessing_overlay", {})
    _require(isinstance(overlay, dict) and isinstance(planned_overlay, dict), "probe overlay metadata must be objects")
    overlay_sha = hashlib.sha256(_OMG_IPC_MULTIPROCESSING.replace("__QWEN35_IPC_PATHS__", repr((physical, short))).encode()).hexdigest()
    _require(planned_overlay.get("path") == str(Path(physical) / "python-ipc/multiprocessing/__init__.py")
             and planned_overlay.get("sha256") == overlay_sha and overlay.get("sha256") == overlay_sha
             and overlay.get("path") == planned_overlay["path"] and overlay.get("physical_tmpdir") == physical
             and overlay.get("multiprocessing_tmpdir") == short, "probe IPC overlay identity differs from the current fix")
    module, socket = child.get("multiprocessing_module", {}), child.get("forkserver_socket", {})
    _require(isinstance(module, dict) and isinstance(socket, dict), "probe module/socket metadata must be objects")
    _require(child.get("status") == "PROBE_PASSED" and child.get("old_alias_control") == "REJECTED_AS_EXPECTED"
             and child.get("physical_tmpdir_predicate") == "PASSED" and socket.get("status") == "PASSED"
             and child.get("physical_tmpdir") == physical and child.get("multiprocessing_tmpdir") == short
             and child.get("ipc_configuration") == "BOUND_SHORT_ALIAS_OBSERVED_IN_ACTUAL_PROCESS_CONFIG"
             and all(child.get(key) == value for key, value in {"vendor_package_import": "NOT_PERFORMED", "model_compilation": "NOT_RUN",
                    "device_execution": "NOT_RUN", "embedded_omg_startup": "NOT_TESTED", "native_cce_compilation": "NOT_RUN"}.items())
             and _same(child.get("controlled_build_flags"), {"debug": False, "dump": False, "random_cce_file_location": True})
             and module.get("path") == module.get("spec_origin") == overlay.get("stdlib_path")
             and module.get("sha256") == overlay.get("stdlib_sha256") and _SHA.fullmatch(str(module.get("sha256", "")))
             and isinstance(socket.get("path"), str) and socket["path"].startswith(short + "/")
             and type(socket.get("bytes")) is int and socket["bytes"] == len(os.fsencode(socket["path"])) < 108,
             "probe did not retain the expected negative/positive predicate and forkserver proof")
    argv = probe_command.get("argv")
    _require(isinstance(argv, list) and len(argv) == 5 and isinstance(argv[0], str) and Path(argv[0]).is_absolute()
             and argv[1:] == [str(probe / "probe-child.py"), str(probe / "child-result.json"), str(probe / "validator-source.py"), short]
             and probe_command.get("cwd") == str(probe), "probe invocation differs from its retained executable body")

    stop, _ = _read(stopped / "report.json", bindings)
    plan, _ = _read(stopped / "plan.json", bindings)
    journal, _ = _read(stopped / "events.jsonl", bindings, 1 << 20, decode=False)
    events = [json.loads(line, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
              for line in journal.splitlines() if line.strip()]
    selected, driver = stop.get("selected_processes"), stop.get("bound_driver")
    _require(stop.get("artifact_kind") == "qwen35-target-compile-stop" and stop.get("status") == "STOP_REQUESTED"
             and stop.get("job") == str(job) and _same(stop.get("command"), command_id)
             and plan.get("status") == "APPLY_READY" and _same(plan.get("events"), [])
             and all(_same(plan.get(key), stop.get(key)) for key in ("job", "command", "selected_processes", "bound_driver", "mode", "signal_backend"))
             and stop.get("mode") == "APPLY" and isinstance(selected, list) and selected and isinstance(driver, dict),
             "stop receipt is not a complete exact-job request")
    results = stop.get("events")
    _require(isinstance(results, list) and len(events) == 2 * len(results) and results, "stop journal is incomplete")
    for i, result in enumerate(results):
        intent = events[2 * i]
        _require(isinstance(intent, dict) and isinstance(result, dict) and intent.get("event") == "SIGNAL_INTENT"
                 and _same(events[2 * i + 1], result)
                 and all(_same(intent.get(key), result.get(key)) for key in ("pid", "start_ticks", "signal")),
                 "stop journal intent/result binding changed")
    identities = selected + [driver]
    by_pid = {row.get("pid"): row for row in identities if isinstance(row, dict)}
    _require(len(by_pid) == len(identities) and stop.get("omg_pid") in by_pid and driver.get("pid") == stop.get("driver_pid"),
             "stop receipt process identities are missing or duplicated")
    for result in results:
        row = by_pid.get(result.get("pid"))
        _require(row is not None and _same(row.get("start_ticks"), result.get("start_ticks"))
                 and result.get("signal") in {"SIGTERM", "SIGCONT"}
                 and (row is not driver or result.get("signal") == "SIGCONT")
                 and result.get("event") in {"SIGTERM_SENT", "SIGCONT_SENT", "ALREADY_EXITED", "NO_LONGER_STOPPED", "IDENTITY_CHANGED_NOT_SIGNALED"},
                 "stop receipt contains an unrelated process or signal")
    _require(all(any(row.get("pid") == result.get("pid") and result.get("signal") == "SIGTERM" for result in results)
                 for row in selected), "stop receipt omits a selected compiler process")
    processes = [_quiescent(row, proc_root) for row in identities]
    for identity in bindings:
        _require(_read_small(Path(identity["path"]), max(1, identity["bytes"]))[1] == identity,
                 "retry preflight metadata changed while checking")
    return {"schema_version": 1, "artifact_kind": "qwen35-static-retry-preflight", "status": "PREFLIGHT_PASSED",
            "failed_job": str(job), "tmpdir_probe": str(probe), "stop_receipt": str(stopped),
            "metadata_bindings": bindings, "recorded_processes": processes, "weight_payload_access": "NONE",
            "model_compilation": "NOT_RUN", "device_execution": "NOT_RUN",
            "scope": "Final small receipts, source predicate/IPC probe and recorded-process quiescence only; full snapshot reuse audit and actual embedded OMG compilation remain separate gates"}
