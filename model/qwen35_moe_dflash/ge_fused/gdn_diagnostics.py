"""Bounded evidence for GDN buffer diagnostics; no kernel or ABI changes."""
from pathlib import Path
import re

import numpy as np

from ..artifacts import output_path, save_json
from ..compile import _stateful_json
from ..hashing import sha256_file
from .export import _step_file
from .ir import DTYPES


GDN_CASES = ("gdn_prefill", "gdn_decode")
FILLS = (0, 165, 90)


def collect_sources(ddk_root, output):
    """Copy matching GDN source files, never import or execute DDK code."""
    output = output_path(output)
    if ddk_root and (output == Path(ddk_root).resolve() or Path(ddk_root).resolve() in output.parents):
        raise ValueError("source evidence must be written outside the immutable DDK")
    output.mkdir(parents=True, exist_ok=False)
    result = {"status": "UNAVAILABLE", "files": [], "entries_examined": 0,
              "limited": False, "scope": "Installed source only; not proof of compiled kernel identity."}
    if not ddk_root:
        result["reason"] = "CPRO_DDK_ROOT / DDK_PATH is unset"
    else:
        ddk = Path(ddk_root).resolve()
        root = ddk / "ddk/ascendc/ops"
        result["root"] = str(root)
        pattern = re.compile(r"Process|Init\(|effective|chunkSize|chunk_size|seqLen|seq_len|"
                             r"blockDim|block_dim|SetBlockDim|Tiling|RECUR|CHUNK|DataCopy|"
                             r"SetGlobalBuffer|initial_state|last_recurrent|headDim|head_dim|"
                             r"numHead|num_head|outputFinal|L2Norm|L2norm|l2norm|GetBlockIdx")
        total = 0
        if root.is_dir() and not root.is_symlink():
            for path in root.rglob("*"):
                result["entries_examined"] += 1
                if result["entries_examined"] > 20000 or len(result["files"]) >= 24:
                    result["limited"] = True
                    break
                normalized = re.sub(r"[^a-z0-9]", "", path.stem.lower())
                if "chunkgateddeltarule" not in normalized or path.suffix not in {".cpp", ".h", ".hpp", ".py", ".json", ".ini"}:
                    continue
                record = {"source": str(path), "status": "UNREADABLE"}
                result["files"].append(record)
                try:
                    resolved = path.resolve()
                    if not resolved.is_file() or ddk not in resolved.parents:
                        raise ValueError("source resolves outside selected DDK or is not a file")
                    size = resolved.stat().st_size
                    if size > 32 << 20 or total + size > 64 << 20:
                        result["limited"] = True
                        record.update(status="SIZE_LIMIT", bytes=size)
                        continue
                    payload = resolved.read_bytes()
                    total += len(payload)
                    relative = path.relative_to(root)
                    destination = output / "sources" / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(payload)
                    lines = payload.decode("utf-8", errors="replace").splitlines()
                    hits = [i for i, line in enumerate(lines) if pattern.search(line)]
                    chosen = hits if len(hits) <= 120 else hits[:60] + hits[-60:]
                    indices = sorted({j for i in chosen for j in range(max(0, i - 2), min(len(lines), i + 8))})
                    record.update(status="COPIED", path=str(destination.relative_to(output)), bytes=len(payload),
                                  sha256=sha256_file(destination), total_lines=len(lines), matched_lines=len(hits),
                                  excerpts=[{"line": i + 1, "text": lines[i]} for i in indices])
                except (OSError, ValueError) as error:
                    record["error"] = str(error)
            result["status"] = "COLLECTED" if result["files"] else "NO_MATCH"
        else:
            result["reason"] = "selected DDK ascendc/ops directory unavailable"
    save_json(output / "report.json", result)
    lines = ["GDN installed source evidence (read only)", result["scope"],
             f"status={result['status']} limited={result['limited']} entries_examined={result['entries_examined']}",
             result.get("reason", ""), ""]
    for record in result["files"]:
        lines.append(f"{record['source']} [{record['status']}]")
        lines.extend(f"L{row['line']}: {row['text']}" for row in record.get("excerpts", []))
    lines.append("Full bounded copies and SHA256 inventory: report.json, sources/.")
    (output / "report.txt").write_text("\n".join(lines) + "\n")
    return result


def _size(value, minimum, maximum=512 << 20):
    return type(value) is int and minimum <= value <= maximum


def collect_compile_evidence(root, names, output):
    """Copy two cases' retained text/JSON only, with strict entry/size bounds."""
    root, output = Path(root).resolve(), output_path(output)
    if root == output or root in output.parents or not set(names) <= set(GDN_CASES):
        raise ValueError("GDN compiler evidence requires a separate destination and known cases")
    output.mkdir(parents=True, exist_ok=False)
    records, total, examined, limited = [], 0, 0, False
    for name in names:
        folder = root / "cases" / name
        candidates = [folder / item for item in ("compile.log", "compile-command.json", "omg-environment.json",
                                                 "tbe-compatibility-result.json")]
        for subdir in ("compile-work", "omc"):
            scan = folder / subdir
            if scan.is_dir() and not scan.is_symlink():
                for path in scan.rglob("*"):
                    examined += 1
                    if examined > 4000:
                        limited = True
                        break
                    if path.suffix in {".json", ".log", ".txt", ".ini"}:
                        candidates.append(path)
        for path in candidates:
            if len(records) >= 48:
                limited = True
                break
            if not path.exists():
                continue
            record = {"source": str(path), "status": "UNREADABLE"}
            records.append(record)
            try:
                resolved = path.resolve()
                if root not in resolved.parents or not resolved.is_file():
                    raise ValueError("compiler evidence escapes the original probe run")
                size = resolved.stat().st_size
                if size > 8 << 20 or total + size > 32 << 20:
                    record.update(status="SIZE_LIMIT", bytes=size)
                    limited = True
                    continue
                relative = path.relative_to(root)
                dest = output / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                payload = resolved.read_bytes()
                dest.write_bytes(payload)
                total += len(payload)
                record.update(status="COPIED", path=str(relative), bytes=len(payload), sha256=sha256_file(dest))
            except (OSError, ValueError) as error:
                record["error"] = str(error)
    result = {"files": records, "entries_examined": examined, "limited": limited,
              "scope": "Existing GDN compiler text/JSON only; no compiler or DDK imports executed."}
    save_json(output / "report.json", result)
    return result


def validate_io(native, step, result_root, fill):
    """Validate actual host-buffer evidence, not merely the input file inventory."""
    diagnostic = native.get("io_diagnostic", {})
    if (diagnostic.get("enabled") is not True or type(diagnostic.get("output_fill_byte")) is not int
            or diagnostic["output_fill_byte"] != fill or diagnostic.get("process_called") is not True
            or diagnostic.get("input_readback_point") != "AFTER_ALL_INPUT_COPIES_AND_OUTPUT_FILL_BEFORE_PROCESS"):
        raise ValueError("runner lacks the requested output prefill diagnostic")
    inputs, outputs = diagnostic.get("inputs", []), diagnostic.get("outputs", [])
    if (len(inputs) != len(step["inputs"]) or len(outputs) != len(step["outputs"])
            or len(native.get("outputs", [])) != len(step["outputs"])):
        raise ValueError("buffer diagnostic does not cover all ordered GDN ports")
    for i, (item, desc) in enumerate(zip(inputs, step["inputs"])):
        if (item.get("index") != i or type(item.get("index")) is not int
                or item.get("logical_bytes") != desc["bytes"]
                or not _size(item.get("allocated_bytes"), desc["bytes"])
                or item.get("readback_matches_source") is not True
                or item.get("readback_sha256") != desc["sha256"]):
            raise ValueError(f"GDN input {i} host buffer readback differs from bound input")
    payloads = []
    for i, (item, desc) in enumerate(zip(outputs, native["outputs"])):
        if (item.get("index") != i or type(item.get("index")) is not int
                or item.get("logical_bytes") != desc["bytes"]
                or not _size(item.get("allocated_bytes_before"), desc["bytes"])
                or not _size(item.get("allocated_bytes_after"), desc["bytes"])
                or item.get("checked_after_process") is not True or item.get("fill_count_available") is not True
                or item.get("object_same") is not True or item.get("buffer_same") is not True):
            raise ValueError(f"GDN output {i} buffer identity/size changed")
        payload = _step_file(Path(result_root), desc).read_bytes()
        if (type(item.get("fill_byte_count")) is not int
                or item["fill_byte_count"] != payload.count(bytes([fill]))):
            raise ValueError(f"GDN output {i} retained-fill count differs from returned bytes")
        payloads.append(payload)
    return payloads


def compare_fills(payloads, outputs):
    """Report prefill-independent zero output, variability and matching fill patterns."""
    if set(payloads) != set(FILLS):
        raise ValueError("all three output-prefill runs are required")
    rows = []
    for index, desc in enumerate(outputs):
        raw = [np.frombuffer(payloads[fill][index], np.uint8) for fill in FILLS]
        if any(value.size != raw[0].size for value in raw):
            raise ValueError("output sizes differ between prefill runs")
        if desc["dtype"] not in {"float16", "float32"}:
            raise ValueError("GDN output requires FP16 or FP32")
        dtype = DTYPES[desc["dtype"]]
        expected_bytes = int(np.prod(desc["shape"])) * dtype.itemsize
        if raw[0].size != expected_bytes:
            raise ValueError("GDN output bytes differ from declared shape/dtype")
        raw_identical = all(np.array_equal(raw[0], value) for value in raw[1:])
        compared_shape = list(desc["shape"])
        if "valid_prefix_length" in desc:
            length = desc["valid_prefix_length"]
            if len(compared_shape) != 4 or type(length) is not int or not 1 <= length <= compared_shape[1]:
                raise ValueError("GDN core output requires a valid sequence prefix")
            raw = [value.reshape(*compared_shape, dtype.itemsize)[:, :length].reshape(-1) for value in raw]
            compared_shape[1] = length
        follows = (raw[0] == FILLS[0]) & (raw[1] == FILLS[1]) & (raw[2] == FILLS[2])
        varying = (raw[0] != raw[1]) | (raw[0] != raw[2])
        values = [value.view(dtype) for value in raw]
        all_zero = all(np.all(value == 0) for value in values)
        signal = ("PREFILL_PATTERN_OBSERVED" if np.any(follows) else
                  "ALL_ZERO_IN_ALL_RUNS" if all_zero else
                  "OUTPUT_VARIES_BETWEEN_RUNS" if np.any(varying) else "IDENTICAL_OUTPUTS")
        rows.append({"index": index, "name": desc["name"], "signal": signal,
                     "bytes": expected_bytes, "compared_bytes": raw[0].size, "compared_shape": compared_shape,
                     "padding_bytes_unchecked": expected_bytes - raw[0].size,
                     "bytes_following_prefill": int(np.count_nonzero(follows)),
                     "bytes_varying": int(np.count_nonzero(varying)),
                     "identical_across_runs": not bool(np.any(varying)),
                     "raw_identical_across_runs": raw_identical,
                     "nonzero_elements": [int(np.count_nonzero(value)) for value in values],
                     "nonfinite_elements": [int(np.count_nonzero(~np.isfinite(value))) for value in values]})
    return rows


def _retained_json(path, digest=None):
    path = Path(path)
    if (not path.is_absolute() or path.is_symlink() or not path.is_file()
            or path.stat().st_size > 8 << 20):
        raise ValueError("retained JSON requires an absolute regular file below 8 MiB")
    actual = sha256_file(path)
    if digest is not None and actual != digest:
        raise ValueError("retained JSON SHA256 mismatch: " + str(path))
    value = _stateful_json(path)
    if not isinstance(value, dict):
        raise ValueError("retained JSON must contain an object")
    return value, actual


def _sentinel_rows(mask, name):
    counts = np.count_nonzero(mask, axis=-1)
    groups = []
    for batch in range(counts.shape[0]):
        for index in range(counts.shape[1]):
            row = counts[batch, index]
            groups.append({"batch": batch, "token" if name == "core_attn" else "head": index,
                           "row_element_counts": row.tolist(),
                           "rows_with_matches": int(np.count_nonzero(row)),
                           "fully_matching_rows": int(np.count_nonzero(row == mask.shape[-1]))})
    return {"row_axis": "head" if name == "core_attn" else "key_row",
            "row_width": mask.shape[-1], "groups": groups}


def _retained_sentinels(payloads, descriptors):
    byte_summary = compare_fills(payloads, descriptors)
    result = []
    for index, (desc, previous) in enumerate(zip(descriptors, byte_summary)):
        shape = desc["shape"]
        if (desc["name"] not in {"core_attn", "last_recurrent_state"} or len(shape) != 4
                or any(type(size) is not int or size < 1 for size in shape)
                or shape[0] != 1 or shape[-1] > 1024 or np.prod(shape[:-1]) > 65536):
            raise ValueError("retained GDN rows require bounded rank-four output descriptors")
        dtype = DTYPES[desc["dtype"]]
        integer = np.dtype("<u2" if dtype.itemsize == 2 else "<u4")
        masks, trials = [], []
        for fill in FILLS:
            words = np.frombuffer(payloads[fill][index], integer).reshape(shape)
            sentinel = int.from_bytes(bytes([fill]) * dtype.itemsize, "little")
            complete = words == sentinel
            valid = complete[:, :previous["compared_shape"][1]] if desc["name"] == "core_attn" else complete
            masks.append(valid)
            matched = int(np.count_nonzero(valid))
            trials.append({"fill": fill, "sentinel_word": f"0x{sentinel:0{dtype.itemsize * 2}x}",
                           "matching_elements": int(np.count_nonzero(complete)),
                           "valid_matching_elements": matched,
                           "padding_matching_elements": int(np.count_nonzero(complete)) - matched,
                           "valid_rows": _sentinel_rows(valid, desc["name"])})
        follows = np.logical_and.reduce(masks)
        result.append({**previous, "dtype": desc["dtype"], "element_bytes": dtype.itemsize,
                       "fills": trials, "complete_elements_following_prefill": int(np.count_nonzero(follows)),
                       "following_prefill_rows": _sentinel_rows(follows, desc["name"])})
    return result


def _retained_source_context(installed):
    inventory, digest = _retained_json(installed / "report.json")
    files = inventory.get("files")
    if not isinstance(files, list) or len(files) > 24:
        raise ValueError("retained installed-source inventory exceeds its capture bound")
    critical_pattern = re.compile(r"InitOutput|SetAtomic\w*|\batomic\w*|Duplicate|"
                                  r"\b(?:memset|memset_s|SetValue|Fill)\b", re.I)
    pattern = re.compile(critical_pattern.pattern + r"|SetGlobalBuffer|DataCopy\w*|CopyOut|"
                         r"\bStore\w*|last_recurrent|core_attn", re.I)

    def first_last(indices, limit):
        if not limit:
            return []
        if len(indices) <= limit:
            return indices
        return indices[:(limit + 1) // 2] + (indices[-(limit // 2):] if limit // 2 else [])

    rows, total = [], 0
    for record in files:
        if not isinstance(record, dict):
            raise ValueError("invalid retained source file record")
        if record.get("status") != "COPIED":
            rows.append({"source": record.get("source"), "status": record.get("status")})
            continue
        if not _size(record.get("bytes"), 0, 32 << 20) or total + record["bytes"] > 64 << 20:
            raise ValueError("retained source payload exceeds its capture bound")
        path = _step_file(installed, record)
        total += record["bytes"]
        lines = path.read_text(errors="replace").splitlines()
        hits = [i for i, line in enumerate(lines) if pattern.search(line)]
        critical = [i for i in hits if critical_pattern.search(lines[i])]
        critical_set = set(critical)
        preferred = first_last(critical, 24)
        chosen = sorted(preferred + first_last([i for i in hits if i not in critical_set], 24 - len(preferred)))
        contexts = [{"match_line": i + 1, "match_text": lines[i], "critical": i in critical_set,
                     "lines": [{"line": j + 1, "text": lines[j]}
                               for j in range(max(0, i - 3), min(len(lines), i + 6))]} for i in chosen]
        rows.append({"source": record.get("source"), "captured_path": str(path), "status": "VERIFIED_COPY",
                     "sha256": record["sha256"], "matched_lines": len(hits), "contexts_limited": len(chosen) < len(hits),
                     "critical_matched_lines": len(critical), "critical_contexts_limited": len(preferred) < len(critical),
                     "contexts": contexts})
    return {"inventory": str(installed / "report.json"), "inventory_sha256": digest,
            "capture_status": inventory.get("status"), "capture_limited": inventory.get("limited"),
            "files": rows, "scope": "Previously captured copies only. Matches are clues, not proof of output initialization "
            "or complete writes. Uncaptured helper files and compiled-kernel identity remain unverified."}


def summarize_retained_diagnosis(diagnosis_run, output_dir):
    """Summarize a completed three-fill run without DDK scans or device calls."""
    from .probes import _validate_probe_manifest

    root, output = Path(diagnosis_run).expanduser().resolve(), output_path(output_dir)
    report, report_hash = _retained_json(root / "report.json")
    if (type(report.get("schema_version")) is not int or report["schema_version"] != 1
            or report.get("status") != "DIAGNOSIS_COMPLETE" or report.get("fills") != list(FILLS)
            or report.get("model_compilation") != "NOT_RUN"):
        raise ValueError("a completed three-fill GDN diagnostic report is required")
    protected = [root]
    manifests = report.get("probe_manifests")
    if not isinstance(manifests, dict) or not 1 <= len(manifests) <= 2:
        raise ValueError("retained diagnosis requires one or two explicit probe manifests")
    contracts = {}
    for folder, digest in manifests.items():
        probe = Path(folder)
        manifest, _ = _retained_json(probe / "manifest.json", digest)
        _validate_probe_manifest(manifest)
        protected.append(probe.resolve())
        for case in manifest["cases"]:
            if case["name"] not in GDN_CASES or case["status"] != "COMPILED":
                continue
            if case["name"] in contracts:
                raise ValueError("duplicate retained GDN source case")
            step_path = _step_file(probe, case["bundle_manifest"])
            step, _ = _retained_json(step_path)
            if (step.get("artifact_kind") != "qwen35-fused-op-step" or step.get("status") != "PREPARED"
                    or step.get("operator_type") != "ChunkGatedDeltaRule" or len(step.get("outputs", [])) != 2):
                raise ValueError("retained source is not a prepared GDN step")
            _step_file(probe, case["model"])
            _step_file(step_path.parent, step["model"])
            _step_file(step_path.parent, step["operator_abi"])
            _step_file(step_path.parent, step["step_plan"])
            if any(step["model"][key] != case["model"][key] for key in ("sha256", "bytes")):
                raise ValueError("retained bundled model differs from compiled GDN source")
            if len(step["inputs"]) != len(case["inputs"]):
                raise ValueError("retained GDN input count differs from source")
            for actual, expected in zip(step["inputs"], case["inputs"]):
                if any(actual.get(key) != expected[key] for key in ("index", "name", "shape", "dtype", "bytes", "sha256")):
                    raise ValueError("retained GDN input differs from source")
                _step_file(probe, expected)
                _step_file(step_path.parent, actual)
            for actual, expected in zip(step["outputs"], case["outputs"]):
                if any(actual.get(key) != expected[key] for key in ("index", "name", "shape", "dtype")):
                    raise ValueError("retained GDN output differs from source")
            contracts[case["name"]] = (case, step, manifest["ddk_id"])
    cases = report.get("cases")
    if (not isinstance(cases, list) or len(cases) != 2 or any(not isinstance(case, dict) for case in cases)
            or {case.get("name") for case in cases} != set(GDN_CASES) or set(contracts) != set(GDN_CASES)):
        raise ValueError("retained diagnosis must contain both GDN cases exactly once")
    summaries, identity = [], None
    for case in sorted(cases, key=lambda row: GDN_CASES.index(row["name"])):
        original, step, ddk = contracts[case["name"]]
        runs = case.get("runs")
        if (case.get("host_input_readback") != "MATCHED" or not isinstance(runs, list) or len(runs) != 3
                or any(not isinstance(row, dict) or type(row.get("fill")) is not int for row in runs)
                or {row["fill"] for row in runs} != set(FILLS)):
            raise ValueError("each retained GDN case requires three unique verified fills")
        payloads, trial_records = {}, []
        for trial in sorted(runs, key=lambda row: FILLS.index(row["fill"])):
            native_path = Path(trial["result"])
            native, _ = _retained_json(native_path, trial.get("result_sha256", ""))
            protected.append(native_path.parent.resolve())
            if (native.get("status") != "EXECUTED" or native.get("scope") != "fused-operator-probe"
                    or native.get("operator_type") != "ChunkGatedDeltaRule" or native.get("ddk_id") != ddk
                    or native.get("model_sha256") != step["model"]["sha256"]
                    or native.get("operator_abi_sha256") != step["operator_abi"]["sha256"]
                    or native.get("step_plan_sha256") != step["step_plan"]["sha256"]):
                raise ValueError("retained native result differs from its source GDN step")
            current_identity = tuple(native.get(key) for key in ("runtime_version", "firmware_id", "ddk_id", "evidence_kind"))
            if any(not isinstance(value, str) or not value for value in current_identity):
                raise ValueError("retained native result lacks runtime identity")
            if identity is not None and identity != current_identity:
                raise ValueError("retained runtime/firmware/DDK/evidence identity differs across trials")
            identity = current_identity
            for key, fields in (("inputs", ("index", "name", "shape", "dtype", "bytes", "sha256")),
                                ("outputs", ("index", "name", "shape", "dtype"))):
                if len(native.get(key, [])) != len(step[key]):
                    raise ValueError("retained native port count differs from GDN step")
                for actual, expected in zip(native[key], step[key]):
                    if any(actual.get(field) != expected[field] for field in fields):
                        raise ValueError("retained native port differs from GDN step")
            payloads[trial["fill"]] = validate_io(native, step, native_path.parent, trial["fill"])
            trial_records.append({"fill": trial["fill"], "result": str(native_path), "result_sha256": trial["result_sha256"]})
        summaries.append({"name": case["name"], "host_input_readback": "MATCHED",
                          "runs": trial_records, "outputs": _retained_sentinels(payloads, original["outputs"])})
    sources = _retained_source_context(root / "installed")
    if any(output == path or path in output.parents or output in path.parents for path in protected):
        raise ValueError("summary output must be separate from every retained input directory")
    if sha256_file(root / "report.json") != report_hash:
        raise ValueError("retained diagnostic report changed while reading")
    result = {"schema_version": 1, "status": "SUMMARIZED", "output": str(output), "diagnosis_run": str(root),
              "diagnosis_report_sha256": report_hash, "model_compilation": "NOT_RUN", "device_execution": "NOT_RUN",
              "numerical_acceptance": "NOT_DEFINED", "placement": "UNVERIFIED", "cases": summaries,
              "installed_source": sources,
              "scope": "Read-only retained-file integrity and sentinel statistics. A single-fill match can be coincidental; "
              "cross-fill matching still does not distinguish accumulation, incomplete writes or cache visibility."}
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "report.json", result)
    lines = ["Retained GDN diagnosis summary: SUMMARIZED", "No compilation, DDK scan or device execution.",
             "Source: " + str(root), "Report SHA256: " + report_hash, ""]
    for case in summaries:
        lines.append(case["name"] + ": host input readback MATCHED; output/source SHA256 checked")
        for row in case["outputs"]:
            counts = ", ".join(f"{trial['fill']:02x}={trial['valid_matching_elements']}" for trial in row["fills"])
            lines.append(f"  {row['name']} ({row['dtype']}): complete elements following all fills="
                         f"{row['complete_elements_following_prefill']}; valid matches {counts}; "
                         f"padding excluded={row['padding_bytes_unchecked']} bytes")
    verified = [row for row in sources["files"] if row["status"] == "VERIFIED_COPY"]
    lines += ["", f"Installed source copies verified={len(verified)}; capture_limited={sources['capture_limited']}."]
    for row in verified:
        lines.append(f"  {row['source']}: output/atomic/write clue lines={row['matched_lines']}")
        critical = [context for context in row["contexts"] if context["critical"]]
        if critical:
            lines.extend(f"    L{context['match_line']}: {context['match_text']}" for context in critical[:4])
        else:
            lines.append("    No explicit initialization/atomic match in this captured source.")
    lines += [sources["scope"],
              "Per-token/head and state-row counts, source SHA256 and numbered contexts: report.json.", result["scope"]]
    (output / "report.txt").write_text("\n".join(lines) + "\n")
    return result
