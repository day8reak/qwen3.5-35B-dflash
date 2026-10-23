"""Report bounded probe evidence without equating support checks with execution."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import tarfile

import numpy as np


_PARTITIONS = re.compile(r"partition\s+type\s+NPU\s*:\s*(\d+)\s*,\s*CPU\s*:\s*(\d+)"
                         r"\s*,\s*GPU\s*:\s*(\d+)\s*,\s*ISP\s*:\s*(\d+)", re.I)
_HILOG = re.compile(r"^\s*(?:\d+[-:])?(?:\d{4}-)?\d{2}-\d{2}\s+\S+\s+(\d+)\s+\d+\s+[VDIWEF]\s")
_DTYPES = {"float16": np.dtype("<f2"), "float32": np.dtype("<f4"), "int32": np.dtype("<i4")}
_LIMIT = 32 * 1024 * 1024
_LOG_LIMIT = 256 * 1024


def _diagnostic_priority(line):
    """Rank concrete rejection details above noisy per-store capability checks."""
    low = line.lower()
    if "return hiai::success" in low or "infershape success" in low:
        return None
    if re.search(r"kernelid\s+not\s+registered|castt\s+selectkernel\s+failed", low):
        return -1, "missing_kernel"
    if re.search(r"modulenotfounderror|importerror|attributeerror|cannot open shared|no such file|tbeinitialize failed|failed to initialize tbe", low):
        return 0, "environment"
    if re.search(r"does(?:n't| not) exist.*tbe.*ops.*store|(?:not found|missing).*tbe.*ops.*store", low):
        return 0, "missing_kernel"
    if (re.search(r"dtype|data\s*type|rank|dim(?:num|cnt|ension)|realdimcnt", low)
            and re.search(r"wrong|must|should|support|>|<|exceed|limit|failed", low)):
        return 1, "candidate_check" if re.search(r"checksupport|check support", low) else "compiler_diagnostic"
    if re.search(r"infershape.*failed|verify.*failed|check ir model compatibility failed|beforecompile.*fail|failed to generator|omg generate.*failed|\[fatal\]", low):
        return 2, "compiler_diagnostic"
    if re.search(r"don.t support|not support|unsupported", low):
        return (4, "candidate_check") if re.search(r"checksupport|check support|not supported in .*store", low) else (3, "compiler_diagnostic")
    return None


def compile_evidence(logtext: str) -> dict:
    """Retain placement events, including their known or unknown graph scope.

    CheckSupported and opDeviceSupport are candidate capability queries. Even a
    final compiler partition is a compile decision, not evidence of board NPU
    execution. With no explicit graph/phase context, scope stays unknown.
    """
    partitions, subgraphs = [], []
    diagnostics, diagnostic_seen, diagnostic_total = [], {}, 0
    graph, scope, candidates = None, "unknown", 0
    for number, line in enumerate(logtext.splitlines(), 1):
        low = line.lower()
        graph_match = re.search(r"(?:start\s+)?infershape\s+for\s+graph\s*\[([^]]+)\]", line, re.I)
        if graph_match:
            graph = graph_match.group(1)
            scope = ("constant_folding" if "constnodegraph" in graph.lower() else
                     "subgraph" if "subgraph" in graph.lower() else "main")
        if re.search(r"constantfolding\s+begin", low):
            scope, graph = "constant_folding", None
        elif re.search(r"constantfolding\s+(?:end|success|finish)", low):
            scope, graph = "unknown", None
        if any(word in low for word in ("opdevicesupport", "checksupported", "checksupport", "check support")):
            candidates += 1
        diagnostic = _diagnostic_priority(line)
        if diagnostic is not None:
            diagnostic_total += 1
            text = line[:1600]
            # Remove volatile process IDs from the deduplication key.
            key = re.sub(r"\(\d+\)", "(PID)", text)
            if key in diagnostic_seen:
                diagnostic_seen[key]["occurrences"] += 1
            else:
                priority, category = diagnostic
                item = {"line": number, "text": text, "occurrences": 1,
                        "category": category, "priority": priority}
                diagnostic_seen[key] = item
        match = _PARTITIONS.search(line)
        if match:
            partitions.append({"line": number, "scope": scope, "graph": graph,
                               "counts": dict(zip(("NPU", "CPU", "GPU", "ISP"), map(int, match.groups()))),
                               "text": line})
        # Only explicit assignment/report records, never arbitrary messages from
        # a CPUCL/NPUCL logger or a candidate kernel's CheckSupport function.
        if (re.search(r"run[_\s-]*by[_\s-]*cl\b", line, re.I)
                or ("subgraph" in low and re.search(r"(?:cpucl|npucl).*subgraph.*\.irpb", low))):
            subgraphs.append({"line": number, "scope": scope, "graph": graph,
                              "text": line, "kind": "run_by_cl" if "irpb" not in low else "partition_dump"})
    diagnostics = sorted(diagnostic_seen.values(), key=lambda item: (item["priority"], item["line"]))[:32]
    return {"partition_events": partitions, "subgraph_events": subgraphs,
            "main_partition_events": [e for e in partitions if e["scope"] == "main"],
            "constant_folding_partition_events": [e for e in partitions if e["scope"] == "constant_folding"],
            "support_candidate_lines": candidates, "runtime_placement": "UNVERIFIED",
            "diagnostics": diagnostics, "diagnostic_matching_lines": diagnostic_total,
            "diagnostic_omitted_distinct": max(0, len(diagnostic_seen) - len(diagnostics)),
            "note": "Compiler partitions describe compilation only. Candidate support checks do not prove placement or execution."}


def _inside(directory: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("tensor file path must be relative")
    root = directory.resolve()
    path = (root / name).resolve()
    if path == root or root not in path.parents:
        raise ValueError("tensor file path escapes its case directory")
    return path


def _sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def compare_outputs(case_record: dict, output_dir: Path, received_files=None) -> dict:
    """Check exact file sizes and finite values before dtype-aware comparison."""
    records = []
    outputs = case_record.get("outputs", [])
    if not outputs:
        return {"status": "FAILED", "outputs": [], "error": "no output specifications"}
    for index, spec in enumerate(outputs):
        row = {"index": index, "path": spec.get("path") if isinstance(spec, dict) else None, "status": "FAILED"}
        records.append(row)
        try:
            dtype = _DTYPES[spec["dtype"]]
            shape = spec["shape"]
            if not isinstance(shape, (list, tuple)) or any(type(n) is not int or n <= 0 for n in shape):
                raise ValueError("tensor shape must contain positive integer dimensions")
            size = math.prod(shape) * dtype.itemsize
            if size <= 0 or size > _LIMIT:
                raise ValueError("probe tensor exceeds 32 MiB bound")
            rtol, atol = float(spec.get("rtol", 0.01)), float(spec.get("atol", 0.002))
            if not all(math.isfinite(v) and v >= 0 for v in (rtol, atol)):
                raise ValueError("tolerances must be finite and nonnegative")
            if dtype.kind in "iu":
                rtol = atol = 0.0
            expected_path = _inside(Path(case_record["directory"]), spec["expected"])
            actual_path = _inside(Path(output_dir), spec["path"])
            row.update({"dtype": spec["dtype"], "shape": list(shape), "expected_bytes": size,
                        "rtol": rtol, "atol": atol})
            for label, path in (("reference", expected_path), ("actual", actual_path)):
                if not path.is_file():
                    row.update(reason="MISSING_REFERENCE" if label == "reference" else "MISSING_OUTPUT",
                               error=f"{label} tensor file is missing")
                    break
                length = path.stat().st_size
                if length != size:
                    row.update(reason="WRONG_REFERENCE_SIZE" if label == "reference" else "WRONG_OUTPUT_SIZE",
                               actual_bytes=length, error=f"{label} tensor has {length} bytes; expected {size}")
                    break
            else:
                actual_records = [item for item in (received_files or [])
                                  if isinstance(item, dict) and item.get("path")
                                  and Path(item["path"]).resolve() == actual_path]
                hash_records = (("reference", expected_path, spec.get("reference", {})),
                                ("actual", actual_path, actual_records[0] if actual_records else {}))
                hash_failure = False
                for label, path, metadata in hash_records:
                    digest = _sha256(path)
                    row[label + "_sha256"] = digest
                    declared = metadata.get("sha256")
                    if declared is not None:
                        if not isinstance(declared, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", declared):
                            raise ValueError(f"invalid {label} SHA256 metadata")
                        if digest != declared.lower():
                            row.update(reason="REFERENCE_HASH_MISMATCH" if label == "reference" else "OUTPUT_HASH_MISMATCH",
                                       error=f"{label} file differs from its recorded SHA256")
                            hash_failure = True
                            break
                if hash_failure:
                    continue
                expected = np.fromfile(expected_path, dtype=dtype).reshape(shape)
                actual = np.fromfile(actual_path, dtype=dtype).reshape(shape)
                if not np.isfinite(expected).all() or not np.isfinite(actual).all():
                    row.update(reason="NONFINITE", reference_nonfinite=int((~np.isfinite(expected)).sum()),
                               actual_nonfinite=int((~np.isfinite(actual)).sum()),
                               error="probe outputs and references must be finite")
                    continue
                a, b = actual.astype(np.float64), expected.astype(np.float64)
                difference = np.abs(a - b)
                matched = actual == expected if dtype.kind in "iu" else difference <= atol + rtol * np.abs(b)
                mismatch = np.flatnonzero(~matched.ravel())
                nonzero = np.abs(b) > 0
                relative = difference[nonzero] / np.abs(b[nonzero])
                row.update(status="PASSED" if mismatch.size == 0 else "FAILED", reason=None if mismatch.size == 0 else "MISMATCH",
                           elements=int(actual.size), mismatches=int(mismatch.size),
                           max_abs_error=float(difference.max()),
                           max_relative_error_on_nonzero_reference=float(relative.max()) if relative.size else 0.0,
                           examples=[{"flat_index": int(i), "actual": actual.ravel()[i].item(),
                                      "expected": expected.ravel()[i].item()} for i in mismatch[:8]])
        except (KeyError, TypeError, ValueError, OSError, OverflowError) as error:
            row.update(reason="INVALID_SPEC_OR_IO", error=str(error))
    return {"status": "PASSED" if all(r["status"] == "PASSED" for r in records) else "FAILED", "outputs": records}


def _read(path):
    try:
        return Path(path).read_text(errors="replace")
    except (OSError, UnicodeError):
        return ""


def _native_record(text):
    result = None
    for line in text.splitlines():
        try:
            item = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(item, dict) and item.get("status") in {"EXECUTED", "FAILED"}:
            result = item
    return result


def _integer_file(path):
    text = _read(path).strip()
    return int(text) if re.fullmatch(r"-?\d+", text) else None


def _runtime_status(directory):
    native = _native_record(_read(directory / "stdout"))
    rc = _integer_file(directory / "exitcode")
    result = {"load": "UNVERIFIED", "run": "UNVERIFIED", "exitcode": rc,
              "native": native, "placement": "UNVERIFIED"}
    if native is None:
        if not (directory / "stdout").is_file():
            result.update(load="NOT_RUN", run="NOT_RUN")
        elif rc is not None and rc != 0:
            result["error"] = f"native process exited {rc} without a stage JSON record; see stderr"
        return result
    result["runtime_version"] = native.get("runtime_version")
    if native["status"] == "EXECUTED":
        if rc == 0:
            result.update(load="PASSED", run="PASSED")
        else:
            result["error"] = "native success JSON is not corroborated by a zero remote exit code"
        return result
    stage = native.get("stage", "unknown")
    result.update(error=native.get("error"), failed_stage=stage)
    if stage in {"arguments", "io-spec", "model-buffer", "init"}:
        result.update(load="NOT_RUN", run="NOT_RUN")
    elif stage == "load":
        result.update(load="FAILED", run="NOT_RUN")
    elif stage == "io":
        result.update(load="PASSED", run="NOT_RUN")
    elif stage == "process":
        result.update(load="PASSED", run="FAILED")
    elif stage in {"output", "unload"}:
        result.update(load="PASSED", run="PASSED")
    return result


def _runtime_diagnostics(runtime, pid, lines, source):
    """Interpret only affirmative, PID-bound evidence of a failed model load."""
    native = runtime.get("native") or {}
    if (not isinstance(pid, int) or pid <= 0 or runtime.get("load") != "FAILED"
            or native.get("status") != "FAILED" or native.get("stage") != "load"
            or runtime.get("exitcode") in (None, 0)):
        return []
    missing, failures, context = [], [], []
    for number, line in enumerate(lines, 1):
        match = _HILOG.match(line)
        if not match or int(match.group(1)) != pid:
            continue
        low = line.lower()
        item = {"line": number, "text": line.rstrip()[:1600]}
        if re.search(r"\bmodel\s+exist\s+graph\s+op\s+with\s+npucl,\s+but\s+this\s+cl\s+not\s+existed\b", low):
            missing.append(item)
        elif "checkclsupported" in low and re.search(r"fail|false", low):
            failures.append(item)
        elif (re.search(r"lib\S+\.so", low)
              and re.search(r"dlopen|load|shared library", low)
              and re.search(r"failed|no such file|cannot open", low)):
            context.append(item)
    if not missing:
        return []
    # A missing library may explain registration failure, but cannot establish
    # that copying a single library supplies the complete matching backend.
    evidence = missing[:2] + failures[:2]
    return [{"code": "NPUCL_NOT_REGISTERED", "pid": pid, "source": source,
             "summary": "The model requires NPUCL, but the loaded runtime reports that this compute library is absent.",
             "guidance": "Check the complete board NPU runtime backend and its registration against the compiler/DDK and board firmware. This evidence does not establish an unsupported dtype.",
             "evidence": evidence, "context": context[:4],
             "evidence_omitted": len(missing) + len(failures) - len(evidence),
             "context_omitted": max(0, len(context) - 4)}]


def _case_records(manifest):
    records = manifest.get("cases", [])
    return list(records.values()) if isinstance(records, dict) else list(records)


def _json_text(value):
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False,
                      default=lambda obj: str(obj) if isinstance(obj, Path) else obj.item()) + "\n"


def _exclusive_text(path, text):
    with Path(path).open("x", encoding="utf-8") as output:
        output.write(text)


def _persisted_board_result(root):
    for name in ("failure.json", "result.json"):
        path = root / "board" / name
        if path.is_file():
            try:
                result = json.loads(path.read_text())
                if not isinstance(result, dict):
                    raise ValueError("board evidence must be a JSON object")
                return result
            except (OSError, ValueError) as error:
                return {"status": "INVALID_EVIDENCE", "error": f"{path}: {error}"}
    return {}


def _bounded_log(source, target):
    """Preserve a bounded head/tail of ordinary logs without modifying sources."""
    size = source.stat().st_size
    with source.open("rb") as stream:
        if size <= _LOG_LIMIT:
            value = stream.read()
        else:
            value = stream.read(_LOG_LIMIT // 2)
            stream.seek(-_LOG_LIMIT // 2, 2)
            value += f"\n[report archive omitted {size - _LOG_LIMIT} middle bytes]\n".encode() + stream.read()
    with target.open("xb") as output:
        output.write(value)
    return size > _LOG_LIMIT


def write_report(root: Path, manifest, boardresult=None):
    """Write reports and a small log-only archive; never package model binaries.

    The archive uses an explicit allowlist. HiLog is filtered by exact process
    ID even if its input file contains unrelated device logs. Shared/raw HiLog,
    model/weight/input/output binaries and build trees are excluded.
    """
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path, report_path, text_path = root / "report-manifest.json", root / "report.json", root / "report.txt"
    archive_path, evidence_root = root / "results.tar.gz", root / "report-evidence"
    for path in (manifest_path, report_path, text_path, archive_path, evidence_root):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite probe report evidence: {path}")
    if boardresult is None:
        boardresult = _persisted_board_result(root)
    evidence_root.mkdir()
    rows, logs, archive_notes = [], [], []
    board_cases = (boardresult or {}).get("cases", {})
    for case in _case_records(manifest):
        name = case["name"]
        if not isinstance(name, str) or name in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError("unsafe probe case name")
        info = board_cases.get(name, {}) if isinstance(board_cases, dict) else {}
        directory = Path(info.get("directory", root / "board" / name)).resolve()
        # Transport may retain an existing run elsewhere; explicit directory is
        # allowed as a source, but tar names remain confined and generated here.
        runtime = _runtime_status(directory)
        verified_success = (runtime.get("native") is not None
                            and runtime["native"].get("status") == "EXECUTED" and runtime["exitcode"] == 0)
        comparison = compare_outputs(case, directory, info.get("output_files", info.get("files"))) if verified_success else {"status": "NOT_RUN", "outputs": []}
        compilation = case.get("compile", {"status": "NOT_RUN"})
        row = {"name": name, "description": case.get("description", ""),
               "compile": compilation, "load": runtime["load"], "run": runtime["run"],
               "numerical": comparison, "placement": "UNVERIFIED", "runtime": runtime,
               "equivalence_group": case.get("equivalence_group"),
               "source_sha256": case.get("source_sha256"),
               "input_tensors": case.get("input_tensors", []), "output_tensors": case.get("outputs", [])}
        rows.append(row)
        evidence = evidence_root / name
        evidence.mkdir()
        for basename in ("stdout", "stderr", "exitcode", "pid"):
            file = directory / basename
            if file.is_file() and not file.is_symlink():
                target = evidence / basename
                if _bounded_log(file, target):
                    archive_notes.append(f"{name}: {basename} archived as a bounded head/tail excerpt")
                logs.append((target, f"board/{name}/{basename}"))
        hilog = directory / "hilog.txt"
        pid = _integer_file(directory / "pid")
        runtime["diagnostics"] = []
        if hilog.is_file() and pid is not None and pid > 0:
            filtered, filtered_bytes, omitted = [], 0, 0
            with hilog.open(errors="replace") as source:
                for line in source:
                    match = _HILOG.match(line)
                    if not match or int(match.group(1)) != pid:
                        continue
                    length = len(line.encode("utf-8"))
                    if filtered_bytes + length <= _LOG_LIMIT:
                        filtered.append(line)
                        filtered_bytes += length
                    else:
                        omitted += 1
            if omitted:
                archive_notes.append(f"{name}: archived PID-filtered HiLog excerpt; omitted {omitted} matching lines")
            target = evidence / "hilog.pid.txt"
            _exclusive_text(target, "".join(filtered))
            relative = f"board/{name}/hilog.pid.txt"
            logs.append((target, relative))
            runtime["diagnostics"] = _runtime_diagnostics(runtime, pid, filtered, relative)
        elif hilog.is_file():
            archive_notes.append(f"{name}: omitted HiLog because no valid PID was recorded")
    report = {"schema_version": 1, "cases": rows,
              "summary": {"cases": len(rows),
                          "compiled": sum(r["compile"].get("status") == "COMPILED" for r in rows),
                          "load_passed": sum(r["load"] == "PASSED" for r in rows),
                          "run_passed": sum(r["run"] == "PASSED" for r in rows),
                          "numerical_passed": sum(r["numerical"]["status"] == "PASSED" for r in rows)},
              "board_transport_status": (boardresult or {}).get("status", "NOT_RUN"),
              "board_transport": {key: boardresult[key] for key in ("status", "stage", "error", "serial", "hdc_server", "firmware", "board_root", "board_cleanup") if key in boardresult},
              "manifest_status": manifest.get("status"), "manifest_error": manifest.get("error"),
              "limitations": ["Each result applies only to the tested shape, dtype, attributes and runtime.",
                              "Compile, load, execution, numerical agreement and NPU placement are separate checks.",
                              "Compiler support candidates and successful Process calls do not prove NPU execution.",
                              "The final HiLog snapshot may miss earlier cases after ring-buffer rotation; PID reuse is possible.",
                              "Runtime placement remains UNVERIFIED without independent execution evidence."],
              "archive_notes": archive_notes}
    _exclusive_text(manifest_path, _json_text(manifest))
    _exclusive_text(report_path, _json_text(report))
    lines = ["Operator probe results", "", "Board transport: " + report["board_transport_status"]]
    if report["board_transport"].get("error"):
        lines.append("Transport error: " + str(report["board_transport"]["error"]))
    if report["manifest_error"]:
        lines.append("Workflow error: " + str(report["manifest_error"]))
    lines.extend(["", "case | compile | load | run | numerical | placement"])
    for row in rows:
        lines.append(" | ".join((row["name"], row["compile"].get("status", "NOT_RUN"), row["load"], row["run"],
                                 row["numerical"]["status"], row["placement"])))
        if row["runtime"].get("error"):
            lines.append("  runtime: " + str(row["runtime"]["error"]))
        for diagnostic in row["runtime"].get("diagnostics", []):
            lines.append(f"  runtime diagnosis [{diagnostic['code']}]: {diagnostic['summary']}")
            lines.append("  next: " + diagnostic["guidance"])
            for kind in ("evidence", "context"):
                for item in diagnostic[kind]:
                    lines.append(f"    {kind} ({diagnostic['source']}:{item['line']}): {item['text']}")
        if row["compile"].get("error"):
            lines.append("  compile: " + str(row["compile"]["error"]))
        compile_info = row["compile"].get("evidence", {})
        for diagnostic in compile_info.get("diagnostics", [])[:6]:
            lines.append(f"  compiler diagnostic ({diagnostic['category']}, log line {diagnostic['line']}): {diagnostic['text']}")
        events = compile_info.get("main_partition_events") or [event for event in compile_info.get("partition_events", []) if event.get("scope") == "unknown"]
        if events:
            event = events[-1]
            counts = ", ".join(f"{key}={value}" for key, value in event["counts"].items())
            lines.append(f"  compiler partition ({event['scope']}, log line {event['line']}): {counts}; runtime placement unverified")
    lines.extend(["", *report["limitations"], *archive_notes])
    _exclusive_text(text_path, "\n".join(lines) + "\n")
    with tarfile.open(archive_path, "x:gz") as archive:
        for path in (manifest_path, report_path, text_path):
            archive.add(path, arcname=path.name, recursive=False)
        for path, relative in logs:
            archive.add(path, arcname=relative, recursive=False)
    return report
