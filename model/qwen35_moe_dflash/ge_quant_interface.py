"""Read-only, bounded source evidence for quantized GE interfaces and HIAI I/O.

Text presence is not proof of registration, an installed kernel, target support,
device packing, or a usable ABI. No DDK module is imported or executed here.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re


MAX_ENTRIES = 16_000
MAX_DEPTH = 6
MAX_FILE_BYTES = 4 << 20
MAX_TOTAL_BYTES = 64 << 20
MAX_COPIES = 48
TEXT_SUFFIXES = {".py", ".h", ".hpp", ".hh", ".hxx", ".c", ".cc", ".cpp", ".cxx",
                 ".json", ".ini", ".cfg", ".yaml", ".yml", ".txt"}
HEADERS = {".h", ".hpp", ".hh", ".hxx"}
NAME_PATTERN = re.compile(r"quant|grouped[_-]?mat|matmul|i4zn|qparam", re.I)
QUANT_PATTERN = re.compile(
    r"Dynamic_?Quant|Quant_?Batch_?Mat_?mul|Quant_?Mat_?mul|Grouped_?Mat_?mul|"
    r"I4ZN|QPARAM|trans_quant_param|nd_to_i4zn", re.I)
PORT_PATTERN = re.compile(r"REG_OP|REGISTER|\bINPUT\b|\bOUTPUT\b|\bATTR\b|\bdef\b|extern|class|struct")
RUNTIME_PATTERN = re.compile(r"cache|flush|invalidat|sync|process|GetBuffer|GetSize", re.I)
SCOPE = ("Text evidence only; no installation, registration, kernel, target support, "
         "packing ABI, numerical accuracy or device execution is established.")


def _excerpt(text, pattern, *, include_ports=False):
    lines = text.splitlines()
    hits = [i for i, line in enumerate(lines) if pattern.search(line)]
    if not hits and include_ports:
        hits = [i for i, line in enumerate(lines) if PORT_PATTERN.search(line)]
    if not hits:
        hits = list(range(min(8, len(lines))))
    anchors = sorted(set(hits))[:12]
    selected = set(anchors)
    if anchors:
        # Always preserve the first actual match and its next three lines for
        # the short report, even when many distant matches share the budget.
        selected.update(range(anchors[0] + 1, min(len(lines), anchors[0] + 4)))
    for distance in range(1, 9):
        for anchor in anchors:
            for index in (anchor + distance, anchor - distance):
                if 0 <= index < len(lines) and len(selected) < 32:
                    selected.add(index)
    return [{"line": index + 1, "text": lines[index][:500],
             "matched": bool(pattern.search(lines[index]))} for index in sorted(selected)]


def _summary_excerpt(excerpt):
    anchor = next((row for row in excerpt if row["matched"]), None)
    if anchor is None:
        anchor = next((row for row in excerpt if PORT_PATTERN.search(row["text"])),
                      excerpt[0] if excerpt else None)
    if anchor is None:
        return []
    return [row for row in excerpt if anchor["line"] <= row["line"] <= anchor["line"] + 3]


def inspect_quant_interfaces(ddk_root, output_dir, *, ge_root=None, hiai_include=None):
    """Copy bounded matching text from only the selected roots into a new folder."""
    ddk_root = Path(ddk_root).resolve()
    output = Path(output_dir).resolve()
    protected = [ddk_root] + [Path(p).resolve() for p in (ge_root, hiai_include) if p is not None]
    if any(output == root or root in output.parents for root in protected):
        raise ValueError("inspection output must be outside the selected read-only toolchains")
    root_specs = [("selected_ddk", ddk_root / relative) for relative in (
        "ddk/ascendc/ops", "ddk/tbe/ops", "tools/tools_ascendc/include", "ddk/ai_ddk_lib/include")]
    if ge_root is not None:
        root_specs.append(("ge", Path(ge_root).resolve() / "include"))
    if hiai_include is not None:
        root_specs.append(("hiai", Path(hiai_include).resolve()))
    def allowed_root(origin, path):
        base = ddk_root if origin == "selected_ddk" else Path(
            ge_root if origin == "ge" else hiai_include).resolve()
        resolved = path.resolve()
        return not path.is_symlink() and (resolved == base or base in resolved.parents)

    roots = [{"origin": origin, "path": str(path), "exists": path.is_dir(),
              "symlink": path.is_symlink(), "allowed": allowed_root(origin, path)}
             for origin, path in root_specs]
    output.mkdir(parents=True, exist_ok=False)
    copies = output / "sources"
    copies.mkdir()
    report = {"status": "TEXT_EVIDENCE_ONLY", "scope": SCOPE,
              "ddk_root": str(ddk_root), "roots": roots, "files": [], "runtime_headers": [],
              "limits": {"max_entries": MAX_ENTRIES, "max_depth": MAX_DEPTH,
                         "max_file_bytes": MAX_FILE_BYTES, "max_total_bytes": MAX_TOTAL_BYTES,
                         "max_copies": MAX_COPIES},
              "scan": {"entries": 0, "files_read": 0, "bytes_read": 0,
                       "copied_files": 0, "limited": False, "stop_reasons": []},
              "skipped": []}
    seen = {}

    def skipped(path, reason, limited=False):
        report["skipped"].append({"path": str(path), "reason": reason})
        if limited:
            report["scan"]["limited"] = True
            if reason not in report["scan"]["stop_reasons"]:
                report["scan"]["stop_reasons"].append(reason)

    def budget_available():
        return (report["scan"]["entries"] < MAX_ENTRIES and
                report["scan"]["bytes_read"] < MAX_TOTAL_BYTES and
                report["scan"]["copied_files"] < MAX_COPIES)

    def inspect_file(path, origin, root, *, runtime=False):
        name_match = bool(NAME_PATTERN.search(path.name))
        if not runtime and (path.suffix.lower() not in TEXT_SUFFIXES or
                            not (name_match or path.suffix.lower() in HEADERS)):
            return "NOT_CANDIDATE"
        if path.is_symlink() or root.resolve() not in path.resolve().parents:
            skipped(path, "SYMLINK_OR_OUTSIDE_ROOT")
            return "SKIPPED"
        canonical = str(path.resolve())
        if canonical in seen:
            record = seen[canonical]
            if record is not None and origin not in record["origins"]:
                record["origins"].append(origin)
            return "ALREADY_INSPECTED"
        seen[canonical] = None
        try:
            size = path.stat().st_size
            if size > MAX_FILE_BYTES:
                skipped(path, "MAX_FILE_BYTES", True)
                return "SKIPPED"
            if report["scan"]["bytes_read"] + size > MAX_TOTAL_BYTES:
                skipped(path, "MAX_TOTAL_BYTES", True)
                return "SKIPPED"
            with path.open("rb") as stream:
                data = stream.read(size)
            report["scan"]["files_read"] += 1
            report["scan"]["bytes_read"] += len(data)
            if len(data) != size or path.stat().st_size != size:
                skipped(path, "FILE_CHANGED_DURING_READ")
                return "SKIPPED"
            if b"\0" in data:
                skipped(path, "BINARY_CONTENT")
                return "SKIPPED"
            text = data.decode("utf-8", errors="replace")
        except OSError as exc:
            skipped(path, f"READ_ERROR: {exc}")
            return "SKIPPED"
        matches = sorted(set(match.group(0) for match in QUANT_PATTERN.finditer(text)))
        if not (runtime or name_match or matches):
            return "NO_TEXT_MATCH"
        if report["scan"]["copied_files"] >= MAX_COPIES:
            skipped(path, "MAX_COPIES", True)
            return "SKIPPED"
        destination = copies / f"{len(report['files']):02d}-{path.name}"
        with destination.open("xb") as stream:
            stream.write(data)
        record = {"path": str(path), "relative_path": str(path.relative_to(root)),
                  "origins": [origin], "root": str(root), "bytes": size,
                  "sha256": hashlib.sha256(data).hexdigest(),
                  "copy": str(destination.relative_to(output)),
                  "evidence_type": "HIAI_RUNTIME_HEADER" if runtime else "QUANT_INTERFACE_TEXT",
                  "filename_match": name_match, "text_matches": matches,
                  "excerpt": _excerpt(text, RUNTIME_PATTERN if runtime else QUANT_PATTERN,
                                      include_ports=not runtime)}
        report["files"].append(record)
        report["scan"]["copied_files"] += 1
        seen[canonical] = record
        return "COPIED"

    # Preserve the two exact runtime headers first, before generic quant matches.
    hiai_roots = [("selected_ddk", ddk_root / "ddk/ai_ddk_lib/include")]
    if hiai_include is not None:
        hiai_roots.append(("hiai", Path(hiai_include).resolve()))
    for origin, root in hiai_roots:
        for relative in ("HiAiModelManagerService.h", "compatible/AiTensor.h"):
            path = root / relative
            status = "MISSING"
            if not allowed_root(origin, root):
                status = "SYMLINK_OR_OUTSIDE_ROOT"
            elif path.is_file():
                if budget_available():
                    report["scan"]["entries"] += 1
                    status = inspect_file(path, origin, root, runtime=True)
                else:
                    status = "BUDGET_EXHAUSTED"
            report["runtime_headers"].append({"origin": origin, "path": str(path), "status": status})

    for origin, root in root_specs:
        if not root.is_dir():
            continue
        if not allowed_root(origin, root):
            skipped(root, "SYMLINK_OR_OUTSIDE_ROOT")
            continue
        stack = [(root, 0)]
        while stack and budget_available():
            directory, depth = stack.pop()
            try:
                # Do not materialize an unbounded directory listing or follow links.
                with os.scandir(directory) as entries:
                    for entry in entries:
                        if not budget_available():
                            break
                        report["scan"]["entries"] += 1
                        path = Path(entry.path)
                        if entry.is_symlink():
                            skipped(path, "SYMLINK_ENTRY")
                        elif entry.is_dir(follow_symlinks=False):
                            if depth < MAX_DEPTH:
                                stack.append((path, depth + 1))
                            else:
                                skipped(path, "MAX_DEPTH", True)
                        elif entry.is_file(follow_symlinks=False):
                            inspect_file(path, origin, root)
            except OSError as exc:
                skipped(directory, f"DIRECTORY_READ_ERROR: {exc}")
    for field, maximum, reason in (("entries", MAX_ENTRIES, "MAX_ENTRIES"),
                                   ("bytes_read", MAX_TOTAL_BYTES, "MAX_TOTAL_BYTES"),
                                   ("copied_files", MAX_COPIES, "MAX_COPIES")):
        if report["scan"][field] >= maximum:
            report["scan"]["limited"] = True
            if reason not in report["scan"]["stop_reasons"]:
                report["scan"]["stop_reasons"].append(reason)
    report["output"] = str(output)
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    lines = ["Quantization interface text evidence (read-only; no compilation or device execution)",
             SCOPE, f"status={report['status']}", f"scan={json.dumps(report['scan'])}"]
    for root in roots:
        lines.append(f"Root [{root['origin']}]: {root['path']} exists={root['exists']}")
    for record in report["files"][:12]:
        lines.append(f"[{','.join(record['origins'])}] {record['path']} -> {record['copy']}")
        lines.append(f"  bytes={record['bytes']} sha256={record['sha256']} matches={record['text_matches']}")
        for line in _summary_excerpt(record["excerpt"]):
            lines.append(f"  L{line['line']}: {line['text']}")
    if len(report["files"]) > 12:
        lines.append(f"Additional matching files in report.json: {len(report['files']) - 12}")
    lines += [f"Runtime header [{row['origin']}]: {row['path']} {row['status']}"
              for row in report["runtime_headers"]]
    lines.append("Full excerpts, copy hashes and limits: report.json. Missing text is not proof of no kernel.")
    (output / "report.txt").write_text("\n".join(lines) + "\n")
    return report
