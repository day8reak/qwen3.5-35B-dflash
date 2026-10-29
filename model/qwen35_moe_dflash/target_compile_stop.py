"""Stop one exact OMG process tree while preserving its Python driver.

This is an explicit, journaled SIGTERM request, not proof of successful compiler
cleanup or final input auditing. No process group or caller process is signaled.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal

from .target_export_reuse import _absolute, _directory, _read_small, _reject_constant, _unique_object


_MAX_PROCESSES = 65536
_MAX_PROC_BYTES = 65536


def _proc_bytes(path):
    with path.open("rb") as stream:
        raw = stream.read(_MAX_PROC_BYTES + 1)
    if len(raw) > _MAX_PROC_BYTES:
        raise ValueError("process metadata exceeds the bounded read limit")
    return raw


def _process(pid, proc_root):
    raw = _proc_bytes(proc_root / str(pid) / "stat").decode()
    fields = raw.rsplit(")", 1)[1].split()
    if len(fields) < 20:
        raise ValueError("incomplete process stat")
    return {"pid": pid, "ppid": int(fields[1]), "start_ticks": int(fields[19]), "state": fields[0]}


def _argv(pid, proc_root):
    raw = _proc_bytes(proc_root / str(pid) / "cmdline")
    return [os.fsdecode(value) for value in raw.rstrip(b"\0").split(b"\0")] if raw else []


def _cwd(pid, proc_root):
    return os.readlink(proc_root / str(pid) / "cwd")


def _processes(proc_root):
    values = {}
    for path in proc_root.iterdir():
        if not path.name.isdecimal():
            continue
        if len(values) >= _MAX_PROCESSES:
            raise ValueError("process scan exceeds the bounded process count")
        pid = int(path.name)
        try:
            values[pid] = _process(pid, proc_root)
        except (FileNotFoundError, ProcessLookupError):
            continue
    return values


def _same_identity(before, after):
    return all(before[key] == after[key] for key in ("pid", "ppid", "start_ticks"))


def _write_json(folder, name, value):
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=folder)
    with os.fdopen(fd, "w") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.fsync(folder)


def _event(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def stop_target_compile(job, *, apply=False, output=None, proc_root=Path("/proc")):
    """Match a single recorded OMG command; apply only with stable Linux pidfds."""
    report = {"schema_version": 1, "artifact_kind": "qwen35-target-compile-stop",
        "status": "BLOCKED", "mode": "APPLY" if apply else "PLAN_ONLY", "signal": "SIGTERM",
        "process_control": "EXACT_OMG_AND_OBSERVED_DESCENDANTS_ONLY",
        "driver_process": "NOT_SIGNALED", "final_compiler_report": "NOT_VERIFIED",
        "input_mutation": "NONE", "events": [],
        "scope": "Stop request only; outer driver must finish input audit and report. No SIGKILL, process groups, input changes, compiler restart or automatic escalation."}
    handles = {}
    folder = journal = None
    try:
        if type(apply) is not bool:
            raise ValueError("apply must be an explicit boolean")
        job = _absolute(job)
        run_value = os.environ.get("AI_RUN_DIR")
        if not run_value:
            raise ValueError("AI_RUN_DIR is required")
        run = _absolute(run_value)
        if run not in job.parents:
            raise ValueError("job must be below AI_RUN_DIR")
        with _directory(job):
            pass
        command_path = job / "compilation/compile-command.json"
        raw, identity = _read_small(command_path, 1 << 20)
        command = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        argv = command.get("argv") if isinstance(command, dict) else None
        expected_cwd = str(job / "compilation/compile-work/generated")
        model_arg = "--model=" + str(job / "compilation/compile-work/generated/export.air")
        if (not isinstance(argv, list) or not argv or any(not isinstance(arg, str) or not arg or "\0" in arg for arg in argv)
                or not Path(argv[0]).is_absolute() or Path(argv[0]).name != "omg"
                or command.get("cwd") != expected_cwd or argv.count(model_arg) != 1
                or sum(arg.startswith("--model=") for arg in argv) != 1
                or not {"--framework=1", "--target=omc", "--platform=kirinx90"}.issubset(argv)):
            raise ValueError("recorded command does not bind this exact kirinx90 OMG job")
        processes = _processes(proc_root)
        roots = []
        for pid, row in processes.items():
            if row["state"] == "Z":
                continue
            try:
                if _argv(pid, proc_root) == argv and _cwd(pid, proc_root) == expected_cwd:
                    roots.append(pid)
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
        report.update(job=str(job), command=identity, matching_omg_pids=roots)
        if not roots:
            report["status"] = "NOT_RUNNING"
            return report
        if len(roots) != 1:
            raise ValueError("multiple exact OMG matches; no process was signaled")
        root = roots[0]
        depths = {root: 0}
        while True:
            added = {pid: depths[row["ppid"]] + 1 for pid, row in processes.items()
                     if pid not in depths and row["ppid"] in depths}
            if not added:
                break
            depths.update(added)
        if any(pid in {0, 1, os.getpid(), os.getppid()} for pid in depths):
            raise ValueError("refusing a tree containing this utility or its parent")
        selected = [dict(processes[pid], depth=depths[pid]) for pid in sorted(depths, key=lambda pid: (-depths[pid], pid))]
        report.update(omg_pid=root, driver_pid=processes[root]["ppid"], selected_processes=selected)
        if not apply:
            if output is not None:
                raise ValueError("--output is only used with --apply; plan mode writes nothing")
            report["status"] = "PLAN_READY"
            return report
        if output is None:
            raise ValueError("apply requires a new stop-receipt directory")
        target = _absolute(output)
        if run not in target.parents or target == job or job in target.parents:
            raise ValueError("stop receipt must be below AI_RUN_DIR and outside the compilation job")
        if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
            raise ValueError("Linux pidfd support is required; no numeric-PID signal fallback")
        # Acquire every process handle before the first signal. PID reuse after
        # this point cannot redirect pidfd_send_signal to an unrelated process.
        for row in selected:
            handles[row["pid"]] = os.pidfd_open(row["pid"], 0)
            if not _same_identity(row, _process(row["pid"], proc_root)):
                raise ValueError("process identity changed before pinning")
        if _argv(root, proc_root) != argv or _cwd(root, proc_root) != expected_cwd:
            raise ValueError("OMG command/cwd changed before pinning")
        if _read_small(command_path, 1 << 20)[1] != identity:
            raise ValueError("recorded compile command changed")
        with _directory(target.parent) as parent:
            os.mkdir(target.name, dir_fd=parent)
            folder = os.open(target.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        report.update(receipt_directory=str(target), status="APPLY_READY")
        _write_json(folder, "plan.json", report)
        fd = os.open("events.jsonl", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=folder)
        journal = os.fdopen(fd, "w")
        os.fsync(folder)
        # The driver is not in this list. Its normal subprocess nonzero-return
        # path can therefore retain audit evidence and write the failed report.
        for row in selected:
            event = {"event": "SIGNAL_INTENT", "pid": row["pid"], "start_ticks": row["start_ticks"], "signal": "SIGTERM"}
            _event(journal, event)
            try:
                current = _process(row["pid"], proc_root)
                if current["start_ticks"] != row["start_ticks"]:
                    event = dict(event, event="IDENTITY_CHANGED_NOT_SIGNALED")
                elif current["state"] == "Z":
                    event = dict(event, event="ALREADY_EXITED")
                else:
                    signal.pidfd_send_signal(handles[row["pid"]], signal.SIGTERM, None, 0)
                    event = dict(event, event="SIGTERM_SENT")
            except (FileNotFoundError, ProcessLookupError):
                event = dict(event, event="ALREADY_EXITED")
            report["events"].append(event)
            _event(journal, event)
        report["status"] = "STOP_REQUESTED"
        report["next"] = "Wait for the outer driver final audit/report; then check for remaining recorded workers before any retry"
    except (OSError, ValueError, TypeError, KeyError, IndexError, RecursionError) as error:
        report.update(status="BLOCKED", error=f"{type(error).__name__}: {error}")
        if journal is not None:
            _event(journal, {"event": "STOP_ERROR", "error": report["error"]})
    finally:
        if journal is not None:
            journal.close()
        if folder is not None:
            try:
                _write_json(folder, "report.json", report)
            finally:
                os.close(folder)
        for handle in handles.values():
            os.close(handle)
    return report
