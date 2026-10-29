"""Stop one exact OMG process tree while preserving its Python driver.

This is an explicit, journaled SIGTERM request, not proof of successful compiler
cleanup or final input auditing. Bound paused targets may receive SIGCONT; no
process group or caller process is signaled. Explicit legacy mode is not pidfd-equivalent.
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


def _process_fd(pid, descriptor):
    fd = os.open("stat", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=descriptor)
    with os.fdopen(fd, "rb") as stream:
        raw = stream.read(_MAX_PROC_BYTES + 1)
    if len(raw) > _MAX_PROC_BYTES:
        raise ValueError("process metadata exceeds the bounded read limit")
    fields = raw.decode().rsplit(")", 1)[1].split()
    if len(fields) < 20:
        raise ValueError("incomplete process stat")
    return {"pid": pid, "ppid": int(fields[1]), "start_ticks": int(fields[19]), "state": fields[0]}


def _driver_binding(pid, job, proc_root):
    row, argv = _process(pid, proc_root), _argv(pid, proc_root)
    scripts = [i for i, value in enumerate(argv) if Path(value).is_absolute()
        and Path(value).parts[-4:] == ("targets", "cpro", "scripts", "compile_full_target.py")]
    if (not argv or not Path(argv[0]).name.startswith("python") or len(scripts) != 1
            or scripts[0] not in {1, 2} or (scripts[0] == 2 and argv[1] not in {"-B", "-u"})):
        raise ValueError("parent driver is not the bound compile_full_target.py invocation")
    outputs = []
    for index, arg in enumerate(argv):
        if arg == "--output":
            outputs.append(argv[index + 1] if index + 1 < len(argv) else None)
        elif arg.startswith("--output="):
            outputs.append(arg.split("=", 1)[1])
    if outputs != [str(job)] or row["state"] not in {"S", "T", "t"}:
        raise ValueError("parent driver output/state does not bind this paused compilation")
    if pid in {0, 1, os.getpid(), os.getppid()}:
        raise ValueError("refusing to continue this utility or its caller")
    return dict(row, argv=argv)


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


def stop_target_compile(job, *, apply=False, output=None, legacy_stopped_tree=False,
                        continue_driver=False, proc_root=Path("/proc")):
    """Bind exact compiler ownership; numeric signals require explicit stopped-tree mode."""
    report = {"schema_version": 1, "artifact_kind": "qwen35-target-compile-stop",
        "status": "BLOCKED", "mode": "APPLY" if apply else "PLAN_ONLY", "signal": "SIGTERM",
        "process_control": "EXACT_OMG_AND_OBSERVED_DESCENDANTS_ONLY",
        "driver_process": "NOT_SIGNALED", "final_compiler_report": "NOT_VERIFIED",
        "input_mutation": "NONE", "events": [],
        "scope": "Stop request only; outer driver must finish input audit and report. No SIGKILL, process groups, input changes, compiler restart or automatic escalation."}
    handles = {}
    folder = journal = None
    try:
        if any(type(value) is not bool for value in (apply, legacy_stopped_tree, continue_driver)):
            raise ValueError("apply/legacy/continue flags must be explicit booleans")
        if legacy_stopped_tree:
            report.update(signal_backend="EXPLICIT_LEGACY_STOPPED_TREE", driver_process="SIGCONT_ONLY_IF_BOUND_AND_STOPPED",
                numeric_signal_limitation="Held proc directory plus immediate identity checks reduce but cannot eliminate a concurrent external PID replacement race; not pidfd-equivalent")
        else:
            report["signal_backend"] = "PIDFD"
        driver = None
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
        paused = [row for row in selected if row["state"] in {"T", "t"}]
        if legacy_stopped_tree and len(paused) != len(selected):
            raise ValueError("legacy mode requires every selected compiler process already stopped (T/t)")
        if legacy_stopped_tree or continue_driver:
            driver = _driver_binding(processes[root]["ppid"], job, proc_root)
            report.update(bound_driver=driver, driver_process="SIGCONT_ONLY_IF_BOUND_AND_STOPPED")
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
        if not legacy_stopped_tree and (not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal")):
            raise ValueError("Linux pidfd support is required; use explicit legacy stopped-tree mode only for an already paused tree")
        # Pin all identities before signaling. Legacy proc descriptors are not
        # pidfds: the remaining check-to-numeric-signal race is explicit.
        for row in selected + ([driver] if driver is not None else []):
            if legacy_stopped_tree:
                handles[row["pid"]] = os.open(proc_root / str(row["pid"]), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                held = _process_fd(row["pid"], handles[row["pid"]])
                if not _same_identity(row, held):
                    raise ValueError("process identity changed before pinning")
            else:
                handles[row["pid"]] = os.pidfd_open(row["pid"], 0)
            current = _process(row["pid"], proc_root)
            if not _same_identity(row, current):
                raise ValueError("process identity changed before pinning")
            if legacy_stopped_tree and row in selected and current["state"] not in {"T", "t"}:
                raise ValueError("legacy compiler tree resumed before pinning")
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
        def send(row, requested):
            event = {"event": "SIGNAL_INTENT", "pid": row["pid"], "start_ticks": row["start_ticks"], "signal": requested.name}
            _event(journal, event)
            try:
                current = _process(row["pid"], proc_root)
                held = _process_fd(row["pid"], handles[row["pid"]]) if legacy_stopped_tree else current
                if not _same_identity(row, current) or not _same_identity(row, held):
                    event = dict(event, event="IDENTITY_CHANGED_NOT_SIGNALED")
                elif current["state"] == "Z":
                    event = dict(event, event="ALREADY_EXITED")
                elif requested == signal.SIGCONT and current["state"] not in {"T", "t"}:
                    event = dict(event, event="NO_LONGER_STOPPED")
                elif legacy_stopped_tree and requested == signal.SIGTERM and current["state"] not in {"T", "t"}:
                    raise ValueError("legacy compiler process resumed before signal")
                else:
                    if row is driver and _argv(row["pid"], proc_root) != driver["argv"]:
                        raise ValueError("bound driver command changed before continue")
                    if legacy_stopped_tree:
                        os.kill(row["pid"], requested)
                    else:
                        signal.pidfd_send_signal(handles[row["pid"]], requested, None, 0)
                    event = dict(event, event=requested.name + "_SENT")
            except (FileNotFoundError, ProcessLookupError):
                event = dict(event, event="ALREADY_EXITED")
            report["events"].append(event)
            _event(journal, event)
            return event["event"] == requested.name + "_SENT"

        terminated = set()
        # Queue TERM for the whole paused tree before continuing any member.
        # The outer Python driver never receives a termination signal.
        for row in selected:
            if send(row, signal.SIGTERM):
                terminated.add(row["pid"])
        for row in paused:
            if row["pid"] in terminated:
                send(row, signal.SIGCONT)
        if driver is not None and driver["state"] in {"T", "t"}:
            send(driver, signal.SIGCONT)
        report["status"] = "STOP_REQUESTED"
        report["next"] = "Let the bound outer driver finish final audit/report; then check recorded workers before any retry. STOP_REQUESTED is not an exit or cleanup proof"
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
