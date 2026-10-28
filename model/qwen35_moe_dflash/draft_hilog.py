"""One bounded, diagnostic-only HiLog snapshot after a draft execution failure."""
from __future__ import annotations

import math
from pathlib import Path, PurePosixPath
import re
import shlex

from .hashing import sha256_file
from .hdc_run import _hilog_summary_details


_NAMES = ("draft-hilog.txt", "draft-hilog.stderr", "draft-hilog.exitcode")
_MAX_SUMMARY_BYTES = 16 * 1024 * 1024
_SCOPE = ("Recent shared system-log snapshot; may include other processes and runs. "
          "A matching PID can be reused. Diagnostic candidates are not proof of this run's cause.")


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
        if snapshot_bytes > _MAX_SUMMARY_BYTES:
            diagnostic["summary"] = {"status": "NOT_SELECTED", "reason": "snapshot exceeds diagnostic selection byte limit",
                                     "snapshot_bytes": snapshot_bytes, "max_bytes": _MAX_SUMMARY_BYTES}
            return diagnostic
        try:
            lines, matches, metadata = _hilog_summary_details(
                snapshot, recorded_pid=recorded_pid, candidate_output=candidate_path)
            if sha256_file(snapshot) != diagnostic["retrieved_sha256"][_NAMES[0]]:
                raise ValueError("HiLog snapshot changed during diagnostic selection")
            diagnostic["summary"] = {"matching_lines": matches, "shown_lines": len(lines), **metadata}
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
