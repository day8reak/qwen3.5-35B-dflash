"""Request termination of one stopped compiler whose owned job was deleted.

This separate route binds live process metadata, not a fabricated compiler
receipt. It never deletes files, requests a compiler restart, signals process
groups, or claims that a signal request proves exit or reclaimed disk space.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import signal
import stat

from .target_compile_stop import (
    _argv, _cwd, _event, _process, _process_fd, _processes, _proc_bytes,
    _same_identity, _write_json,
)
from .target_export_reuse import _absolute, _directory


_SCRIPT = ("targets", "cpro", "scripts", "compile_full_target.py")
_STOPPED = {"T", "t"}
_MAX_TREE = 256
_MAX_FDS = 4096
_MAX_TOTAL_FDS = 32768


def _option(argv, name):
    values = []
    for index, value in enumerate(argv):
        if value == name:
            values.append(argv[index + 1] if index + 1 < len(argv) else None)
        elif value.startswith(name + "="):
            values.append(value[len(name) + 1:])
    return values


def _missing(job):
    if job.exists() or job.is_symlink():
        raise ValueError("this route requires the exact job directory already absent")


def _parent(pid, job, proc_root):
    row, argv = _process(pid, proc_root), _argv(pid, proc_root)
    scripts = [i for i, value in enumerate(argv)
               if Path(value).is_absolute() and Path(value).parts[-4:] == _SCRIPT]
    if (not argv or not Path(argv[0]).name.startswith("python") or len(scripts) != 1
            or scripts[0] not in {1, 2} or (scripts[0] == 2 and argv[1] not in {"-B", "-u"})
            or row["state"] not in _STOPPED):
        raise ValueError("parent must be the stopped canonical compile_full_target.py driver")
    if any(_option(argv, flag) for flag in ("--resume", "--retry-static-prefill-from", "--borrow-decode-inputs-from")):
        raise ValueError("alternate/resumed driver modes are outside deleted-job cleanup")
    airs = _option(argv, "--air")
    if len(airs) != 1 or not isinstance(airs[0], str) or not Path(airs[0]).is_absolute():
        raise ValueError("deleted-job driver requires one explicit absolute source AIR")
    outputs = _option(argv, "--output")
    if outputs == [str(job)]:
        binding = "EXACT_EXPLICIT_DRIVER_OUTPUT"
    elif not outputs and re.fullmatch(r"full-target-compile-[0-9a-f]{12}", job.name):
        # Only this value is retained; never log the process's complete env.
        values = [value.split(b"=", 1)[1] for value in _proc_bytes(proc_root / str(pid) / "environ").split(b"\0")
                  if value.startswith(b"AI_RUN_DIR=")]
        if values != [os.fsencode(str(job.parent))]:
            raise ValueError("implicit generated job is not bound to the driver's AI_RUN_DIR")
        binding = "GENERATED_JOB_NAME_AND_DRIVER_RUN_ENV_AND_EXACT_CHILD"
    else:
        raise ValueError("driver output does not bind this deleted job")
    return dict(row, argv=argv, job_binding=binding)


def _compiler(pid, job, proc_root):
    row, argv = _process(pid, proc_root), _argv(pid, proc_root)
    expected_cwd = str(job / "compilation/compile-work/generated") + " (deleted)"
    expected_model = str(job / "compilation/compile-work/generated/export.air")
    if (not argv or not Path(argv[0]).is_absolute() or Path(argv[0]).name != "omg"
            or _option(argv, "--model") != [expected_model]
            or not {"--framework=1", "--target=omc", "--platform=kirinx90"}.issubset(argv)
            or _cwd(pid, proc_root) != expected_cwd or row["state"] not in _STOPPED):
        raise ValueError("OMG must be stopped with exact model argument and deleted job cwd")
    return dict(row, argv=argv, cwd=expected_cwd)


def _held_deleted_files(rows, proc_root):
    files, errors, count = {}, [], 0
    for row in rows:
        folder = proc_root / str(row["pid"]) / "fd"
        try:
            descriptors = list(folder.iterdir())
            if len(descriptors) > _MAX_FDS:
                raise ValueError("process descriptor count exceeds the observation bound")
            count += len(descriptors)
            if count > _MAX_TOTAL_FDS:
                raise ValueError("whole-tree descriptor count exceeds the observation bound")
            for entry in descriptors:
                if not entry.name.isdecimal():
                    continue
                try:
                    target = os.readlink(entry)
                    if not target.endswith(" (deleted)"):
                        continue
                    info = entry.stat()
                    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 0:
                        continue
                    key = info.st_dev, info.st_ino
                    item = files.setdefault(key, {"device": info.st_dev, "inode": info.st_ino,
                        "bytes": info.st_size, "allocated_bytes": info.st_blocks * 512, "holders": []})
                    item["holders"].append({"pid": row["pid"], "fd": int(entry.name), "target": target})
                except OSError as error:
                    errors.append({"pid": row["pid"], "fd": entry.name, "error": str(error)})
        except OSError as error:
            errors.append({"pid": row["pid"], "error": str(error)})
    return {"files": list(files.values()), "unique_regular_inodes": len(files),
            "logical_bytes": sum(row["bytes"] for row in files.values()),
            "allocated_bytes": sum(row["allocated_bytes"] for row in files.values()), "errors": errors,
            "scope": "Deleted regular open descriptors only, deduplicated by device/inode. Mappings without open fds and other processes' holders are not counted; allocation is not guaranteed reclaim."}


def stop_deleted_target_compile(pid, job, *, apply=False, output=None,
                                legacy_stopped_tree=False, proc_root=Path("/proc")):
    """Plan by default; explicit apply queues TERM then CONT for this exact tree."""
    report = {"schema_version": 1, "artifact_kind": "qwen35-deleted-target-compile-stop",
        "status": "BLOCKED", "mode": "APPLY" if apply else "PLAN_ONLY", "events": [],
        "process_control": "EXACT_STOPPED_OMG_DESCENDANTS_AND_BOUND_DELETED_JOB_DRIVER_ONLY",
        "input_mutation": "NONE", "final_compiler_report": "UNAVAILABLE_JOB_ALREADY_DELETED",
        "exit_validation": "NOT_PERFORMED", "reclaimed_bytes": "NOT_INFERRED",
        "scope": "Signal request only; no file deletion, SIGKILL, process groups, compiler restart or automatic escalation."}
    handles, folder, journal = {}, None, None
    try:
        if type(pid) is not int or pid <= 1 or any(type(flag) is not bool for flag in (apply, legacy_stopped_tree)):
            raise ValueError("positive explicit PID and boolean action flags required")
        run = _absolute(os.environ.get("AI_RUN_DIR", "")) if os.environ.get("AI_RUN_DIR") else None
        job = _absolute(job)
        if run is None or job.parent != run:
            raise ValueError("deleted job must be a direct child of AI_RUN_DIR")
        with _directory(run):
            pass
        _missing(job)
        report.update(job=str(job), omg_pid=pid, signal_backend="EXPLICIT_LEGACY_STOPPED_TREE" if legacy_stopped_tree else "PIDFD")
        if legacy_stopped_tree:
            report["numeric_signal_limitation"] = "Held proc directory and immediate identity checks reduce but cannot eliminate a concurrent external PID replacement race; not pidfd-equivalent"
        try:
            root = _compiler(pid, job, proc_root)
        except (FileNotFoundError, ProcessLookupError):
            report["status"] = "NOT_RUNNING"
            return report
        driver = _parent(root["ppid"], job, proc_root)
        processes = _processes(proc_root)
        if (root["pid"] not in processes or not _same_identity(root, processes[root["pid"]])
                or driver["pid"] not in processes or not _same_identity(driver, processes[driver["pid"]])):
            raise ValueError("root/driver identity changed during tree scan")
        if any(row["ppid"] == driver["pid"] and row["pid"] != pid for row in processes.values()):
            raise ValueError("driver has another child outside the exact OMG tree")
        depths = {pid: 0}
        while True:
            added = {n: depths[row["ppid"]] + 1 for n, row in processes.items()
                     if n not in depths and row["ppid"] in depths}
            if not added:
                break
            depths.update(added)
            if len(depths) > _MAX_TREE:
                raise ValueError("compiler process tree exceeds the cleanup bound")
        selected = [dict(processes[n], depth=depths[n]) for n in sorted(depths, key=lambda n: (-depths[n], n))]
        selected.append(dict(driver, depth=-1, role="deleted_job_driver"))
        if any(row["pid"] in {0, 1, os.getpid(), os.getppid()} for row in selected):
            raise ValueError("refusing a tree containing this utility or its caller")
        if any(row["state"] not in _STOPPED for row in selected):
            raise ValueError("every selected compiler/worker/driver must already be stopped")
        report.update(driver_pid=driver["pid"], root_command=root["argv"], root_cwd=root["cwd"],
            driver_binding=driver, selected_processes=selected,
            deleted_open_files=_held_deleted_files(selected, proc_root))
        # Reread exact bindings after the read-only observations.
        if _compiler(pid, job, proc_root) != root or _parent(driver["pid"], job, proc_root) != driver:
            raise ValueError("compiler/driver binding changed during inspection")
        if not apply:
            if output is not None:
                raise ValueError("plan mode writes nothing; --output requires --apply")
            report["status"] = "PLAN_READY"
            return report
        if output is None:
            raise ValueError("apply requires a new receipt output directory")
        target = _absolute(output)
        if run not in target.parents or target == job or job in target.parents:
            raise ValueError("receipt must be below AI_RUN_DIR and outside the deleted job")
        if not legacy_stopped_tree and (not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal")):
            raise ValueError("pidfd support unavailable; explicit --legacy-stopped-tree is required for this already stopped tree")
        for row in selected:
            if legacy_stopped_tree:
                handle = os.open(proc_root / str(row["pid"]), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            else:
                handle = os.pidfd_open(row["pid"], 0)
            handles[row["pid"]] = handle
            current = _process(row["pid"], proc_root)
            held = _process_fd(row["pid"], handle) if legacy_stopped_tree else current
            if not _same_identity(row, current) or not _same_identity(row, held) or current["state"] not in _STOPPED:
                raise ValueError("process identity/state changed before pinning")
        _missing(job)
        if _compiler(pid, job, proc_root) != root or _parent(driver["pid"], job, proc_root) != driver:
            raise ValueError("compiler/driver binding changed before receipt")
        with _directory(target.parent) as parent:
            os.mkdir(target.name, dir_fd=parent)
            folder = os.open(target.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        report.update(status="APPLY_READY", receipt_directory=str(target))
        _write_json(folder, "plan.json", report)
        descriptor = os.open("events.jsonl", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=folder)
        journal = os.fdopen(descriptor, "w")

        def send(row, requested):
            event = {"event": "SIGNAL_INTENT", "pid": row["pid"], "start_ticks": row["start_ticks"], "signal": requested.name}
            _event(journal, event)
            try:
                _missing(job)
                current = _process(row["pid"], proc_root)
                held = _process_fd(row["pid"], handles[row["pid"]]) if legacy_stopped_tree else current
                if not _same_identity(row, current) or not _same_identity(row, held):
                    event["event"] = "IDENTITY_CHANGED_NOT_SIGNALED"
                elif current["state"] == "Z":
                    event["event"] = "ALREADY_EXITED"
                elif current["state"] not in _STOPPED:
                    raise ValueError("bound process resumed before signal; no further cleanup")
                else:
                    if row["pid"] == pid and (_argv(pid, proc_root) != root["argv"] or _cwd(pid, proc_root) != root["cwd"]):
                        raise ValueError("OMG command/cwd changed before signal")
                    if row["pid"] == driver["pid"] and _argv(row["pid"], proc_root) != driver["argv"]:
                        raise ValueError("driver command changed before signal")
                    if legacy_stopped_tree:
                        os.kill(row["pid"], requested)
                    else:
                        signal.pidfd_send_signal(handles[row["pid"]], requested, None, 0)
                    event["event"] = requested.name + "_SENT"
            except (FileNotFoundError, ProcessLookupError):
                event["event"] = "ALREADY_EXITED"
            report["events"].append(event)
            _event(journal, event)
            return event["event"] == requested.name + "_SENT"

        # Queue TERM for every stopped member, including the obsolete driver,
        # before continuing any of them. CONT is requested only after TERM;
        # signal handling and actual process exit remain separate observations.
        terminated = {row["pid"] for row in selected if send(row, signal.SIGTERM)}
        for row in selected:
            if row["pid"] in terminated:
                send(row, signal.SIGCONT)
        report.update(status="STOP_REQUESTED", next="Check recorded process identities for exit and filesystem free space; this receipt is not proof of exit or cleanup savings")
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
