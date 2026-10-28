"""One bounded, diagnostic-only HiLog snapshot after a draft execution failure."""
from __future__ import annotations

import math
from collections import deque
from pathlib import Path, PurePosixPath
import re
import shlex

from .hashing import sha256_file
from .hdc_run import _hilog_summary_details


_NAMES = ("draft-hilog.txt", "draft-hilog.stderr", "draft-hilog.exitcode")
_SCOPE = ("Recent shared system-log snapshot; may include other processes and runs. "
          "A matching PID can be reused. Diagnostic candidates are not proof of this run's cause.")

_SERVICE_RECORD = re.compile(
    r"^(?P<date>\d{2}-\d{2})\s+(?P<time>\d{2}:\d{2}:\d{2}\.\d+)\s+"
    r"(?P<pid>\d+)\s+(?P<tid>\d+)\s+[DIWEF]\s+[^\s:]*\/hiaiserver\/[^\s:]+:(?P<message>.*)$")


def _service_failure_context(snapshot, recorded_pid, candidate):
    """Stream same-thread context around service failures naming the runner PID.

    A server PID differs from the runner PID. Never select an unrelated server
    error solely by keyword or proximity to the end of a shared snapshot.
    """
    if not recorded_pid or not candidate.get("candidate_date"):
        return [], {}
    pid_pattern = re.compile(r"\bPid\s*[:=]\s*" + str(recorded_pid) + r"\b", re.IGNORECASE)
    failures = re.compile(r"\b(?:fail\w*|error\w*|invalid)\b", re.IGNORECASE)
    before = deque(maxlen=40)
    selected = {}
    anchor = None
    anchor_line = None
    with snapshot.open(encoding="utf-8", errors="replace") as stream:
        for number, line in enumerate(stream, 1):
            if number < candidate["candidate_start_line"]:
                continue
            if number > candidate["candidate_end_line"]:
                break
            match = _SERVICE_RECORD.match(line)
            if not match or match["date"] != candidate["candidate_date"]:
                continue
            hours, minutes, seconds = map(float, match["time"].split(":"))
            moment = hours * 3600 + minutes * 60 + seconds
            item = (number, f"L{number}: {line.rstrip()}"[:1000])
            thread = (match["pid"], match["tid"])
            if pid_pattern.search(match["message"]) and failures.search(match["message"]):
                anchor = (thread, moment)
                anchor_line = number
                selected = dict((old_item for old_thread, old_time, old_item in before
                                 if old_thread == thread and 0 <= moment - old_time <= 2))
                selected[number] = item[1]
            elif anchor and thread == anchor[0] and 0 <= moment - anchor[1] <= 2 and len(selected) < 40:
                selected[number] = item[1]
            before.append((thread, moment, item))
    if not selected:
        return [], {"status": "NO_EXPLICIT_PID_FAILURE"}
    return [selected[n] for n in sorted(selected)], {
        "status": "SELECTED", "recorded_pid": recorded_pid, "anchor_line": anchor_line,
        "server_pid": int(anchor[0][0]), "server_tid": int(anchor[0][1]),
        "scope": "Same service thread within two seconds of a failure explicitly naming the runner PID; diagnostic context, not exclusive attribution."}


def capture_draft_hilog(hdc, board, out, *, receive, recorded_pid=None):
    """Return auxiliary evidence without changing an execution result.

    ``receive(hdc, board, out, name, timeout_seconds=...)`` must return the
    verified SHA256 or None for a missing file. Each HDC call has a timeout of
    at most ten seconds; no shared timeout is changed. The caller supplies a
    process ID only after validating the native result's identity. All ordinary
    capture, retrieval and selection failures are returned as diagnostics.
    """
    diagnostic = {"artifact_kind": "qwen35-draft-hilog", "status": "FAILED",
                  "command": ["hilog", "-x"], "scope": _SCOPE,
                  "returncode": None, "retrieved_files": [], "retrieved_sha256": {},
                  "retrieval_errors": [], "summary_lines": [],
                  "process_identity": {"recorded_pid": recorded_pid,
                      "scope": "Caller-supplied native PID; PID reuse remains possible."}}
    try:
        if recorded_pid is not None and (type(recorded_pid) is not int or recorded_pid <= 0):
            raise ValueError("recorded_pid must be a positive integer")
        if (isinstance(hdc.timeout, bool) or not isinstance(hdc.timeout, (int, float))
                or not math.isfinite(hdc.timeout) or hdc.timeout <= 0):
            raise ValueError("HDC timeout must be finite and positive")
        timeout = min(10, hdc.timeout)
        diagnostic["timeout_seconds_per_call"] = timeout
        board, out = PurePosixPath(board), Path(out)
        if not board.is_absolute() or str(board) == "/" or ".." in board.parts:
            raise ValueError("board root must be an absolute non-root path without traversal")
        if not out.is_dir():
            raise ValueError("diagnostic output directory must already exist")
        candidate_path = out / "draft-hilog-candidate.txt"
        if any((out / name).exists() or (out / name).is_symlink() for name in _NAMES) or candidate_path.exists():
            raise FileExistsError("draft HiLog evidence already exists")
        diagnostic["paths"] = {name: {"local": str(out / name), "remote": str(board / name)} for name in _NAMES}
        command = ("hilog -x > " + shlex.quote(str(board / _NAMES[0]))
                   + " 2> " + shlex.quote(str(board / _NAMES[1]))
                   + "\nrc=$?; printf '%s\\n' \"$rc\" > " + shlex.quote(str(board / _NAMES[2]))
                   + '\nexit "$rc"')
        try:
            status, _ = hdc.shell(command, "capture-draft-hilog", allow_failure=True, timeout=timeout)
            if type(status) is not int or not 0 <= status <= 255:
                raise ValueError("invalid hilog completion marker exitcode")
            diagnostic["returncode"] = status
        except Exception as error:
            diagnostic["error"] = f"{type(error).__name__}: {error}"
        # Recover even after a transport error; hash-verified partial evidence
        # remains available, but cannot become a successfully captured snapshot.
        for name in _NAMES:
            try:
                digest = receive(hdc, board, out, name, timeout_seconds=timeout)
                if digest is None:
                    raise FileNotFoundError("remote diagnostic file is missing")
                if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                    raise ValueError("invalid diagnostic SHA256 record")
                if sha256_file(out / name) != digest:
                    raise ValueError("diagnostic file differs from verified SHA256")
                diagnostic["retrieved_files"].append(name)
                diagnostic["retrieved_sha256"][name] = digest
            except Exception as error:
                diagnostic["retrieval_errors"].append({"path": name, "error": f"{type(error).__name__}: {error}"})
        if diagnostic["retrieval_errors"]:
            raise RuntimeError("could not retrieve and hash-verify all HiLog outputs")
        exit_path = out / _NAMES[2]
        with exit_path.open("rb") as stream:
            saved = stream.read(64)
        if not re.fullmatch(rb"(?:0|[1-9][0-9]{0,2})\r?\n?", saved) or int(saved) > 255:
            raise ValueError("invalid draft-hilog.exitcode")
        diagnostic["recorded_exitcode"] = int(saved)
        if diagnostic["returncode"] != int(saved):
            raise RuntimeError("hilog completion marker and exitcode file do not agree")
        diagnostic["status"] = "CAPTURED" if int(saved) == 0 else "UNAVAILABLE"
        if int(saved) != 0:
            diagnostic["reason"] = "hilog -x returned a nonzero exitcode"
            return diagnostic
        # Some installed versions put this argument error on stdout and return 0.
        for name in _NAMES[:2]:
            with (out / name).open("rb") as stream:
                header = stream.read(8192).decode("utf-8", errors="replace")
            if any(re.fullmatch(r"(?:Mutlti|Multi) commands can't be used in combination\s*\[CODE:\s*-31\]",
                                line.strip()) for line in header.splitlines()):
                diagnostic.update(status="UNAVAILABLE", reason="hilog rejected the command combination (CODE -31)")
                return diagnostic
        snapshot = out / _NAMES[0]
        snapshot_bytes = snapshot.stat().st_size
        try:
            lines, matches, metadata = _hilog_summary_details(
                snapshot, recorded_pid=recorded_pid, candidate_output=candidate_path)
            # Both selectors stream the snapshot and retain bounded display
            # windows. A 20MB shared HiLog must not hide a useful load failure.
            service_lines, service_metadata = _service_failure_context(snapshot, recorded_pid, metadata)
            merged = {int(line.split(":", 1)[0][1:]): line for line in lines + service_lines}
            lines = [merged[number] for number in sorted(merged)]
            if sha256_file(snapshot) != diagnostic["retrieved_sha256"][_NAMES[0]]:
                raise ValueError("HiLog snapshot changed during diagnostic selection")
            diagnostic["summary"] = {"matching_lines": matches, "shown_lines": len(lines),
                                     "snapshot_bytes": snapshot_bytes, "selection_io": "streamed",
                                     "service_context": service_metadata, **metadata}
            diagnostic["summary_lines"] = lines
            if candidate_path.is_file():
                diagnostic["candidate_log"] = {"path": str(candidate_path), "bytes": candidate_path.stat().st_size,
                    "sha256": sha256_file(candidate_path), "source_path": str(snapshot),
                    "source_sha256": diagnostic["retrieved_sha256"][_NAMES[0]], "selection": metadata,
                    "scope": "Selected candidate records with original snapshot line numbers; PID reuse remains possible."}
        except Exception as error:
            diagnostic.update(status="FAILED", summary_lines=[], summary_error=f"{type(error).__name__}: {error}")
    except Exception as error:
        diagnostic["status"] = "FAILED"
        diagnostic.setdefault("error", f"{type(error).__name__}: {error}")
    return diagnostic
