"""Bounded evidence for GDN buffer diagnostics; no kernel or ABI changes."""
from pathlib import Path
import re

import numpy as np

from ..artifacts import output_path, save_json
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
