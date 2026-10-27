"""Report bounded probe evidence without equating support checks with execution."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import tarfile

import numpy as np

from .compiler_evidence import compile_evidence


_HILOG = re.compile(r"^\s*(?:\d+[-:])?(?:\d{4}-)?\d{2}-\d{2}\s+\S+\s+(\d+)\s+\d+\s+[VDIWEF]\s")
_DTYPES = {"float16": np.dtype("<f2"), "float32": np.dtype("<f4"), "int32": np.dtype("<i4")}
_LIMIT = 32 * 1024 * 1024
_LOG_LIMIT = 256 * 1024


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


def _signal_stats(values):
    """Describe finite tensors without changing their acceptance threshold."""
    return {"min": float(values.min()), "max": float(values.max()),
            "max_abs": float(np.abs(values).max()),
            "nonzero": int(np.count_nonzero(values)),
            "zeros": int(values.size - np.count_nonzero(values)),
            "rms": float(np.sqrt(np.mean(values * values))),
            "l2": float(np.linalg.norm(values.ravel()))}


def _nonzero_by_axis(values):
    """Bounded marginals reveal head/row loss without interpreting device layout."""
    return [{"axis": axis, "counts": np.count_nonzero(values,
                axis=tuple(other for other in range(values.ndim) if other != axis)).tolist()}
            for axis, size in enumerate(values.shape) if size <= 64]


def _count_runs(counts):
    """Compact repeated marginal counts as inclusive start..end=count ranges."""
    runs = []
    for index, count in enumerate(counts):
        if runs and runs[-1][2] == count:
            runs[-1][1] = index
        else:
            runs.append([index, index, count])
    return ",".join(f"{lo}..{hi}={count}" if lo != hi else f"{lo}={count}"
                    for lo, hi, count in runs)


def _comprehensive_summary(report):
    lines = ["GDN comprehensive diagnostic summary", "Board transport: " + report["board_transport_status"],
             "Numerical gates unchanged. Detailed evidence: report.json / report.txt.",
             "Axis count ranges are inclusive logical indices, not physical strides."]
    for row in report["cases"]:
        if row.get("probe_suite") != "gdn-comprehensive":
            continue
        lines.append(f"\n{row['name']} | {row['compile'].get('status', 'NOT_RUN')} | "
                     f"load={row['load']} run={row['run']} numerical={row['numerical']['status']}")
        native = row["runtime"].get("native") or {}
        tensors = native.get("io_tensors")
        if not isinstance(tensors, list) or not tensors:
            lines.append("  IO metadata: NOT_RECORDED")
        else:
            for item in tensors:
                if not isinstance(item, dict):
                    continue
                lines.append(f"  IO {item.get('direction')}[{item.get('index')}] "
                    f"{item.get('logical_dtype')} {item.get('logical_shape')} -> {item.get('runtime_shape')} "
                    f"format={item.get('runtime_format')} bytes={item.get('logical_bytes')}/{item.get('allocation_bytes')}")
        for output in row["numerical"]["outputs"]:
            prefix = f"  {output.get('label', output.get('name'))}: {output['status']}"
            signal = output.get("signal")
            if signal is None:
                lines.append(prefix + " " + str(output.get("reason", output.get("error", ""))))
                continue
            actual, reference = signal["actual"], signal["reference"]
            lines.append(prefix + f" maxerr={output['max_abs_error']:.6g} "
                f"relL2={signal['relative_l2_to_reference']} "
                f"nonzero(actual/ref)={actual['nonzero']}/{reference['nonzero']} "
                f"maxabs(actual/ref)={actual['max_abs']:.6g}/{reference['max_abs']:.6g}")
            by_axis = signal.get("nonzero_by_axis", {})
            for actual_axis, ref_axis in zip(by_axis.get("actual", []), by_axis.get("reference", [])):
                if actual_axis["counts"] != ref_axis["counts"]:
                    lines.append(f"    axis{actual_axis['axis']} nonzero actual[{_count_runs(actual_axis['counts'])}] "
                                 f"ref[{_count_runs(ref_axis['counts'])}]")
        if row["runtime"].get("error"):
            lines.append("  runtime: " + str(row["runtime"]["error"]))
        if row["compile"].get("error"):
            lines.append("  compile failed: see report.txt and case omg.log")
    lines.extend(["", *report["limitations"]])
    return "\n".join(lines) + "\n"


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
            row.update({"name": spec.get("name"), "label": spec.get("label", spec.get("name")),
                        "dtype": spec["dtype"], "shape": list(shape), "expected_bytes": size,
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
                reference_signal, actual_signal = _signal_stats(b), _signal_stats(a)
                error_l2 = float(np.linalg.norm(difference.ravel()))
                row.update(status="PASSED" if mismatch.size == 0 else "FAILED", reason=None if mismatch.size == 0 else "MISMATCH",
                           elements=int(actual.size), mismatches=int(mismatch.size),
                           max_abs_error=float(difference.max()),
                           max_relative_error_on_nonzero_reference=float(relative.max()) if relative.size else 0.0,
                           signal={"reference": reference_signal, "actual": actual_signal,
                                   "all_zero_with_nonzero_reference": actual_signal["nonzero"] == 0
                                       and reference_signal["nonzero"] > 0,
                                   "exact_elements": int(np.count_nonzero(actual == expected)),
                                   "error_rms": float(np.sqrt(np.mean(difference * difference))),
                                   "relative_l2_to_reference": error_l2 / reference_signal["l2"]
                                       if reference_signal["l2"] > 0 else None,
                                   "relative_l2_defined": reference_signal["l2"] > 0},
                           examples=[{"flat_index": int(i), "actual": actual.ravel()[i].item(),
                                      "expected": expected.ravel()[i].item()} for i in mismatch[:8]])
                if case_record.get("probe_suite") == "gdn-comprehensive":
                    row["signal"]["nonzero_by_axis"] = {
                        "reference": _nonzero_by_axis(b), "actual": _nonzero_by_axis(a)}
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
    comprehensive = any(case.get("probe_suite") == "gdn-comprehensive" for case in _case_records(manifest))
    summary_path = root / "gdn-summary.txt" if comprehensive else None
    for path in (manifest_path, report_path, text_path, archive_path, evidence_root, *([summary_path] if comprehensive else [])):
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
        if case.get("probe_suite"):
            row.update(probe_suite=case["probe_suite"], probe_family=case.get("probe_family"))
        if case.get("input_domain"):
            row["input_domain"] = case["input_domain"]
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
              "board_transport": {key: boardresult[key] for key in ("status", "stage", "error", "serial", "hdc_server", "firmware", "board_root", "board_cleanup", "library_search_paths") if key in boardresult},
              "board_diagnostics": {"hilog": {key: boardresult["hilog"][key]
                  for key in ("status", "command", "error", "scope", "deliverable_policy")
                  if key in boardresult["hilog"]}} if "hilog" in boardresult else {},
              "manifest_status": manifest.get("status"), "manifest_error": manifest.get("error"),
              "limitations": ["Each result applies only to the tested shape, dtype, attributes and runtime.",
                              "Compile, load, execution, numerical agreement and NPU placement are separate checks.",
                              "Compiler support candidates and successful Process calls do not prove NPU execution.",
                              "The final HiLog snapshot may miss earlier cases after ring-buffer rotation; PID reuse is possible.",
                              "Runtime placement remains UNVERIFIED without independent execution evidence."],
              "archive_notes": archive_notes}
    report["selection"] = manifest.get("selection")
    if any(row.get("probe_suite") in {"gdn-accuracy", "gdn-core-accuracy", "gdn-comprehensive"} for row in rows):
        report["limitations"].append(
            "GDN signal statistics are diagnostic only; unchanged absolute tolerance may accept a tiny output that is entirely zero. "
            "New probe graphs may optimize differently from the original OMC; compare final-only and staged block outputs with its captured features.")
    if any(row.get("probe_suite") == "gdn-comprehensive" for row in rows):
        report["limitations"].append(
            "Nonzero counts by axis are marginals, not a physical-stride measurement. "
            "A buffer larger than logical tensor bytes does not by itself establish row padding; packing is unchanged.")
    families = sorted({r["probe_family"] for r in rows if r.get("probe_family")})
    report["probe_families"] = {family: sum(r.get("probe_family") == family for r in rows)
                                for family in families}
    _exclusive_text(manifest_path, _json_text(manifest))
    _exclusive_text(report_path, _json_text(report))
    lines = ["Operator probe results", "", "Board transport: " + report["board_transport_status"]]
    if report["probe_families"]:
        lines.append("Probe families: " + ", ".join(f"{name}={count}" for name, count in report["probe_families"].items()))
    if report["board_transport"].get("error"):
        lines.append("Transport error: " + str(report["board_transport"]["error"]))
    hilog = report["board_diagnostics"].get("hilog", {})
    if hilog.get("status") == "INCOMPLETE":
        lines.append("Diagnostic warning: HiLog INCOMPLETE; " + str(hilog.get("error", "")))
        lines.append("Remote evidence retained: " + str(report["board_transport"].get("board_root", "unknown")))
    if report["manifest_error"]:
        lines.append("Workflow error: " + str(report["manifest_error"]))
    lines.extend(["", "case | compile | load | run | numerical | placement"])
    for row in rows:
        lines.append(" | ".join((row["name"], row["compile"].get("status", "NOT_RUN"), row["load"], row["run"],
                                 row["numerical"]["status"], row["placement"])))
        if row.get("probe_suite") == "gdn-comprehensive":
            native = row["runtime"].get("native") or {}
            io_tensors = native.get("io_tensors")
            if not isinstance(io_tensors, list) or not io_tensors:
                lines.append("  runtime IO metadata: NOT_RECORDED (use --refresh-runner for the new batch)")
            else:
                for tensor in io_tensors:
                    lines.append("  runtime IO: " + json.dumps(tensor, allow_nan=False))
        if row.get("probe_suite") in {"gdn-accuracy", "gdn-core-accuracy", "gdn-comprehensive"}:
            for output in row["numerical"]["outputs"]:
                details = {key: output[key] for key in
                    ("status", "reason", "error", "shape", "elements", "max_abs_error", "signal")
                    if key in output}
                if row["probe_suite"] in {"gdn-core-accuracy", "gdn-comprehensive"} and "signal" in details:
                    # Keep the larger stage trace readable; report.json retains
                    # the complete statistics for every tensor.
                    signal = details.pop("signal")
                    details["signal"] = {key: signal[key] for key in
                        ("all_zero_with_nonzero_reference", "relative_l2_to_reference")}
                    for side in ("reference", "actual"):
                        details["signal"][side] = {key: signal[side][key]
                            for key in ("nonzero", "max_abs", "rms")}
                    if "nonzero_by_axis" in signal:
                        details["signal"]["nonzero_by_axis"] = signal["nonzero_by_axis"]
                lines.append(f"  output {output['index']} ({output.get('label') or output.get('name')}): "
                             + json.dumps(details, allow_nan=False))
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
        rollbacks = [event for event in compile_info.get("rollback_events", [])
                     if event.get("scope") in ("main", "unknown")]
        rollback = rollbacks[-1] if rollbacks else None
        if rollback is not None:
            lines.append(f"  compiler rollback ({rollback['scope']}, log line {rollback['line']}): "
                         f"{rollback['device']} partition count exceeds limit {rollback['limit']}; "
                         f"compiler will rollback to {rollback['target_device']}; runtime placement unverified")
        for diagnostic in compile_info.get("diagnostics", [])[:6]:
            lines.append(f"  compiler diagnostic ({diagnostic['category']}, log line {diagnostic['line']}): {diagnostic['text']}")
        main = [event for event in compile_info.get("main_partition_events", []) if event.get("scope") == "main"]
        events = main or [event for event in compile_info.get("partition_events", [])
                          if event.get("scope") == "main"]
        unknown = [event for event in compile_info.get("partition_events", []) if event.get("scope") == "unknown"]
        if rollback is not None:
            boundary = rollback.get("end_line", rollback["line"])
            events = [event for event in events if event["line"] > boundary]
            unknown = [event for event in unknown if event["line"] > boundary]
        events = events or unknown
        if events:
            event = events[-1]
            counts = ", ".join(f"{key}={value}" for key, value in event["counts"].items())
            lines.append(f"  compiler partition ({event['scope']}, log line {event['line']}): {counts}; runtime placement unverified")
        else:
            qualifier = "post-rollback " if rollback is not None else ""
            lines.append(f"  compiler partition: no {qualifier}main/unknown summary recorded; runtime placement unverified")
    lines.extend(["", *report["limitations"], *archive_notes])
    _exclusive_text(text_path, "\n".join(lines) + "\n")
    reports = [manifest_path, report_path, text_path]
    if comprehensive:
        _exclusive_text(summary_path, _comprehensive_summary(report))
        reports.append(summary_path)
    with tarfile.open(archive_path, "x:gz") as archive:
        for path in reports:
            archive.add(path, arcname=path.name, recursive=False)
        for path, relative in logs:
            archive.add(path, arcname=relative, recursive=False)
    return report
