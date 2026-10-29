"""Bounded read-only observations from one failed target compilation.

External weight payloads and partial OMC/weight outputs are never opened.
Recorded hashes plus current file stats are not fresh content verification.
"""
from __future__ import annotations

from collections import Counter, deque
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat


MAX_METADATA_BYTES = 32 << 20
MAX_AIR_BYTES = 64 << 20
MAX_LOG_READ_BYTES = 64 << 20
MAX_LOG_OUTPUT_BYTES = 12 << 10
_MAX_ROWS = 16384
_MAX_GRAPH_ENTRIES = 1024


def _stamp(info):
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def _safe(path, *, directory=False):
    path = Path(os.path.abspath(Path(path).expanduser()))
    current = Path(path.anchor)
    for index, part in enumerate(path.parts[1:], 1):
        current /= part
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("symlink is unsupported: " + str(current))
        last = index == len(path.parts) - 1
        if not (stat.S_ISDIR(info.st_mode) if not last or directory else stat.S_ISREG(info.st_mode)):
            raise ValueError("expected regular file/directory: " + str(current))
    return path, _stamp(path.stat())


def _relative(root, text):
    if not isinstance(text, str) or "\\" in text or "\x00" in text:
        raise ValueError("artifact path must be explicit relative POSIX text")
    parts = PurePosixPath(text)
    if parts.is_absolute() or not text or any(piece in {"", ".", ".."} for piece in text.split("/")):
        raise ValueError("artifact path escapes the failed compilation")
    return root.joinpath(*parts.parts)


def _read_small(path, maximum):
    path, before = _safe(path)
    if not 0 < before[2] <= maximum:
        raise ValueError("small evidence file is empty or exceeds its byte budget")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    digest, chunks, remaining = hashlib.sha256(), [], before[2]
    with os.fdopen(fd, "rb") as stream:
        if _stamp(os.fstat(stream.fileno())) != before:
            raise ValueError("evidence identity changed before read")
        while remaining:
            block = stream.read(min(1 << 20, remaining))
            if not block:
                raise ValueError("evidence truncated during read")
            remaining -= len(block); digest.update(block); chunks.append(block)
        if stream.read(1) or _stamp(os.fstat(stream.fileno())) != before or _safe(path)[1] != before:
            raise ValueError("evidence identity changed during read")
    return b"".join(chunks), {"path": str(path), "bytes": before[2], "sha256": digest.hexdigest(), "stat": before}


def _pairs(rows):
    result = {}
    for key, value in rows:
        if key in result:
            raise ValueError("duplicate metadata JSON key: " + key)
        result[key] = value
    return result


def _constant(value):
    raise ValueError("nonfinite metadata constant: " + value)


def _metadata(path, evidence, label, *, required=False):
    try:
        raw, identity = _read_small(path, MAX_METADATA_BYTES)
        value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant)
        if not isinstance(value, dict):
            raise ValueError("metadata root must be an object")
        evidence[label] = {"status": "READ_SHA256_BOUND", **identity}
        return value
    except (OSError, ValueError, UnicodeError, RecursionError) as error:
        evidence[label] = {"status": "UNAVAILABLE", "path": str(path), "error": str(error)[:300]}
        if required:
            raise ValueError(label + ": " + str(error)) from error
        return {}


def _rows(value, field):
    rows = value.get(field, [])
    if not isinstance(rows, list) or len(rows) > _MAX_ROWS or any(not isinstance(row, dict) for row in rows):
        raise ValueError("invalid/excessive " + field)
    return rows


def _snapshot_summary(compilation, rows):
    methods, states, digests, mismatches = Counter(), Counter(), Counter(), []
    total, air_rows, seen = 0, [], set()
    for row in rows:
        relative = row.get("path")
        size = row.get("bytes")
        if type(size) is not int or size < 0 or relative in seen:
            raise ValueError("invalid/duplicate snapshot size/path")
        seen.add(relative); total += size
        methods[str(row.get("snapshot_method", "UNKNOWN"))] += 1
        digests["VALID_HEX" if isinstance(row.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
                else "MISSING_OR_INVALID"] += 1
        try:
            path, current = _safe(_relative(compilation, relative))
            status = ("MATCHED" if current == row.get("snapshot_stat") and current[2] == size
                      else "DIFFERENT_OR_UNRECORDED")
        except (OSError, ValueError) as error:
            path, current, status = None, None, "UNAVAILABLE"
        states[status] += 1
        if status != "MATCHED" and len(mismatches) < 8:
            mismatches.append({"path": relative, "status": status})
        if row.get("role") == "air":
            air_rows.append((row, path, current))
    summary = {"count": len(rows), "logical_bytes": total, "snapshot_methods": dict(methods),
               "current_stat_counts": dict(states), "mismatches": mismatches,
               "recorded_sha256_counts": dict(digests),
               "hash_validation": "RECORDED_HASH_WITH_CURRENT_STAT",
               "weight_payload_access": "NONE", "complete_snapshot_content_verification": "NOT_PERFORMED"}
    air_path = None
    if len(air_rows) == 1:
        row, path, _ = air_rows[0]
        if path is not None and path.suffix.lower() == ".air":
            air_path = str(path)
            try:
                _, identity = _read_small(path, MAX_AIR_BYTES)
                summary["air"] = {**identity, "recorded_sha256": row.get("sha256"),
                    "hash_validation": "FRESH_SHA256_MATCHED" if identity["sha256"] == row.get("sha256")
                                       else "FRESH_SHA256_DIFFERENT"}
            except (OSError, ValueError) as error:
                summary["air"] = {"status": "UNAVAILABLE", "error": str(error)[:300]}
        else:
            summary["air"] = {"status": "UNAVAILABLE", "reason": "recorded AIR path is unavailable or not an explicit .air file"}
    else:
        summary["air"] = {"status": "UNAVAILABLE", "reason": "snapshot requires exactly one recorded AIR row",
                          "recorded_air_rows": len(air_rows)}
    return summary, air_path, seen


def _partial_summary(compilation, rows, snapshot_paths):
    total, counts, examples, stat_counts = 0, Counter(), [], Counter()
    seen = set()
    for row in rows:
        name, size = row.get("path"), row.get("bytes")
        if name in seen or name in snapshot_paths or type(size) is not int or size < 0:
            raise ValueError("invalid/duplicate generated artifact or input overlap")
        seen.add(name); total += size
        counts[str(row.get("role", "UNKNOWN"))] += 1
        try:
            _, current = _safe(_relative(compilation, name))
            state = ("MATCHED" if current[2] == size and current == row.get("stat") else
                     "SIZE_MATCHED_STAT_NOT_RECORDED" if current[2] == size and "stat" not in row else "DIFFERENT")
        except (OSError, ValueError):
            state = "UNAVAILABLE"
        stat_counts[state] += 1
        examples.append({"path": name, "recorded_bytes": size, "current_stat": state})
    examples.sort(key=lambda row: (-row["recorded_bytes"], row["path"]))
    return {"count": len(rows), "recorded_bytes": total, "roles": dict(counts),
            "current_stat_counts": dict(stat_counts), "largest_examples": examples[:8],
            "hash_validation": "RECORDED_HASH_WITH_CURRENT_STAT", "payload_access": "NONE"}


def _graph_candidates(compilation, report, diagnostics):
    candidates, errors = {}, []
    def add(path, source, recorded=None):
        if path.suffix.lower() != ".irpb":
            return
        try:
            path, stamp = _safe(path)
            if compilation not in path.parents:
                raise ValueError("graph path escapes compilation")
            if not 0 < stamp[2] <= MAX_AIR_BYTES:
                raise ValueError("compiler graph is empty or exceeds the 64 MiB metadata parser budget: " + path.name)
            candidates.setdefault(str(path), {"path": str(path), "bytes": stamp[2], "source": source,
                "recorded_sha256": (recorded or {}).get("sha256"), "content_read": "NOT_PERFORMED"})
        except (OSError, ValueError) as error:
            if len(errors) < 4:
                errors.append(str(error)[:180])
    for field in ("generated_artifacts", "compiler_work_artifacts"):
        for row in _rows(report, field):
            try:
                add(_relative(compilation, row.get("path")), field, row)
            except ValueError as error:
                if len(errors) < 4: errors.append(str(error))
    for row in _rows(diagnostics, "files"):
        if row.get("role") == "graph-dump" and row.get("status") == "RETAINED":
            name = row.get("name")
            if isinstance(name, str) and Path(name).name == name:
                add(compilation / "diagnostics" / name, "retained_diagnostics", row)
    # Only these fixed direct dump directories are inspected. Never walk a
    # parent run directory or any model/weight subtree recursively.
    for relative in ("diagnostics", "compile-work/generated/dump_graph", "compile-work/dump_graph"):
        folder = compilation / relative
        try:
            folder, _ = _safe(folder, directory=True)
            with os.scandir(folder) as entries:
                for index, entry in enumerate(entries):
                    if index >= _MAX_GRAPH_ENTRIES:
                        errors.append("fixed dump directory entry budget exceeded: " + relative); break
                    if entry.name.endswith(".irpb"):
                        add(folder / entry.name, "fixed_direct_dump_directory")
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as error:
            if len(errors) < 4: errors.append(str(error)[:180])
    def priority(row):
        name = Path(row["path"]).name.lower()
        return (0 if "fail" in name else 1 if "after_partition" in name else 2,
                0 if row["source"] == "retained_diagnostics" else 1, row["path"])
    selected = sorted(candidates.values(), key=priority)[:4]
    return selected, errors[:4]


def _log_observation(path):
    path, before = _safe(path)
    recent, matches = deque(maxlen=3), Counter()
    windows = {"reshape_error_candidates": deque(maxlen=4),
               "branch_error_candidates": deque(maxlen=2), "branch_candidates": deque(maxlen=2)}
    active = []
    digest, scanned, line_number = hashlib.sha256(), 0, 0
    line_prefix, search_tail, current_match = bytearray(), b"", set()
    def finish_line():
        nonlocal line_number, active, line_prefix, search_tail, current_match
        line_number += 1
        text = bytes(line_prefix).decode("utf-8", errors="replace").rstrip("\r\n")
        if len(line_prefix) >= 2048: text += " [line shortened]"
        row = (line_number, text)
        for window in active:
            window["rows"].append(row)
            window["remaining"] -= 1
        for kind in current_match:
            matches[kind] += 1
            window = {"anchor": line_number, "rows": [*recent, row], "remaining": 4}
            windows[kind].append(window)
            active.append(window)
        retained = {id(window) for rows in windows.values() for window in rows}
        active = [window for window in active if window["remaining"] and id(window) in retained]
        recent.append(row)
        line_prefix, search_tail, current_match = bytearray(), b"", set()
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        if _stamp(os.fstat(stream.fileno())) != before:
            raise ValueError("compiler log identity changed before read")
        while scanned < min(before[2], MAX_LOG_READ_BYTES):
            block = stream.readline(min(64 << 10, MAX_LOG_READ_BYTES - scanned))
            if not block: break
            digest.update(block); scanned += len(block)
            if len(line_prefix) < 2048: line_prefix.extend(block[:2048 - len(line_prefix)])
            search = (search_tail + block).lower()
            explicit_failure = any(token in search for token in
                    (b"error", b"fail", b"mismatch", b"invalid", b"not equal", b"not match"))
            reshape_false = (b"reshapecomputeoutputdim" in search and b"false" in search)
            if b"reshape" in search and (explicit_failure or reshape_false):
                current_match.add("reshape_error_candidates")
            if re.search(rb"branch(?:es)?[_/]?[01](?=$|[^0-9])", search):
                current_match.add("branch_candidates")
                if explicit_failure: current_match.add("branch_error_candidates")
            search_tail = search[-512:]
            if block.endswith(b"\n"): finish_line()
        if line_prefix: finish_line()
    stable = _safe(path)[1] == before
    # Keep the last few actual failures rather than exhausting the selection
    # on normal E-level shape traces. Prioritize every retained failure anchor
    # before its surrounding lines, then favor newer nearby context.
    selected, priorities = {}, {}
    for kind_rank, kind in enumerate(("reshape_error_candidates", "branch_error_candidates", "branch_candidates")):
        for window in windows[kind]:
            anchor = window["anchor"]
            for number, text in window["rows"]:
                selected[number] = text
                priority = (0 if number == anchor else 1, kind_rank, -anchor, abs(number - anchor), number)
                priorities[number] = min(priorities.get(number, priority), priority)
    chosen, used, omitted = {}, 0, 0
    for number in sorted(selected, key=priorities.get):
        text = selected[number]
        raw = f"L{number}: {text}\n".encode("utf-8")
        if used + len(raw) > MAX_LOG_OUTPUT_BYTES:
            omitted += 1; continue
        chosen[number] = raw; used += len(raw)
    chunks = [raw for _, raw in sorted(chosen.items())]
    complete = scanned == before[2]
    return {"path": str(path), "bytes": before[2], "scanned_bytes": scanned, "stable_stat": stable,
            "sha256": digest.hexdigest() if complete and stable else None,
            "hash_validation": "FRESH_LOG_SHA256" if complete and stable else "PARTIAL_OR_CHANGED_NOT_VERIFIED",
            "match_counts": dict(matches), "selected_text": b"".join(chunks).decode("utf-8"),
            "omitted_selected_lines": omitted, "selection_limit_bytes": MAX_LOG_OUTPUT_BYTES,
            "scope": "Candidate error/branch context only; branch-to-gear mapping and root cause are not inferred"}


def inspect_target_compile_failure(job):
    """Inspect one failed job without writing, compiling or hashing weights."""
    result = {"schema_version": 1, "artifact_kind": "qwen35-target-compile-failure-inspection",
        "status": "BLOCKED", "job": str(job), "metadata_evidence": {}, "snapshot_air_path": None,
        "compiler_graph_paths": [], "model_execution": "NOT_RUN", "compiler_invocation": "NOT_RUN",
        "weight_payload_access": "NONE", "limitations": [
            "Snapshot/partial payload hashes are recorded values plus current stat only; fresh weight SHA validation was not performed",
            "A 64-only compile, branch labels or retained graph dumps cannot prove decode1 or end-to-end DFlash"]}
    try:
        job, _ = _safe(job, directory=True)
        result["job"] = str(job)
        root, _ = _safe(job / "compilation", directory=True)
        evidence = result["metadata_evidence"]
        outer = _metadata(job / "result.json", evidence, "job_result", required=True)
        report = _metadata(root / "report.json", evidence, "compile_report", required=True)
        if (outer.get("artifact_kind") != "qwen35-original-target-job" or outer.get("status") != "FAILED"
                or report.get("artifact_kind") != "qwen35-existing-air-compilation" or report.get("status") != "FAILED"):
            raise ValueError("requires an original-target FAILED job and FAILED compilation report")
        profile = _metadata(job / "compile-profile.json", evidence, "compile_profile")
        command = _metadata(root / "compile-command.json", evidence, "compile_command")
        _metadata(root / "input-inventory.json", evidence, "input_inventory")
        environment = _metadata(root / "omg-environment.json", evidence, "omg_environment")
        compatibility = _metadata(root / "tbe-compatibility-result.json", evidence, "tbe_compatibility")
        diagnostics = _metadata(root / "diagnostics/manifest.json", evidence, "diagnostics_manifest")
        expected = outer.get("compilation_report_sha256")
        result["failure"] = {key: report.get(key) for key in ("status", "stage", "model_compilation", "returncode", "error",
            "timeout_seconds", "source_preservation", "audit_error", "artifact_inventory_error")}
        result["failure"]["compilation_report_binding"] = ("NOT_RECORDED" if expected is None else
            "MATCHED" if expected == evidence["compile_report"]["sha256"] else "DIFFERENT")
        result["options"] = report.get("requested_options", {})
        result["profile"] = {"name": profile.get("name"), "dynamic_dims": profile.get("dynamic_dims"),
            "input_shape_matches_options": profile.get("input_shape") == result["options"].get("input_shape"),
            "command_argv": command.get("argv"), "compiler_cwd": command.get("cwd"),
            "input_layout": report.get("input_layout")}
        result["ddk"] = {key: report.get(key) for key in ("ddk_id", "ddk_root", "omg_path", "omg_sha256", "platform")}
        result["ddk"].update(omg_hash_validation="RECORDED_NOT_REHASHED", environment_status=evidence["omg_environment"]["status"],
            tbe_audit=compatibility.get("audit"), tbe_compiler_invocation=compatibility.get("compiler_invocation"))
        try:
            _, tool_stat = _safe(report["omg_path"])
            result["ddk"]["current_omg_stat"] = tool_stat
        except (OSError, ValueError, TypeError, KeyError) as error:
            result["ddk"]["current_omg_stat"] = {"status": "UNAVAILABLE", "error": str(error)[:200]}
        result["snapshots"], result["snapshot_air_path"], paths = _snapshot_summary(root, _rows(report, "input_snapshots"))
        result["partial_artifacts"] = _partial_summary(root, _rows(report, "generated_artifacts"), paths)
        candidates, errors = _graph_candidates(root, report, diagnostics)
        result["compiler_graph_candidates"] = candidates
        result["compiler_graph_paths"] = [row["path"] for row in candidates]
        if errors: result["compiler_graph_discovery_errors"] = errors
        try:
            result["log_evidence"] = _log_observation(root / "compile.log")
        except (OSError, ValueError) as error:
            result["log_evidence"] = {"status": "UNAVAILABLE", "error": str(error)[:300]}
        space = os.statvfs(job)
        result["free_bytes"] = space.f_bavail * space.f_frsize
        result["free_space_scope"] = "Current shared-filesystem observation, not a retry/compilation budget guarantee"
        result["status"] = "INSPECTED"
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as error:
        result["error"] = f"{type(error).__name__}: {str(error)[:500]}"
    return result
