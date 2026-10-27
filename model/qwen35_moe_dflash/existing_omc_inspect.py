"""Inspect a compiled original AIR's OMC without assuming this runner's ABI."""
from collections import Counter
import hashlib
from itertools import islice
import json
import math
import os
from pathlib import Path
import re

from .artifacts import output_path, save_json
from .compile import omg_environment
from .hashing import sha256_file
from .omc_inspect import _call, _supports_conversion


MAX_JSON_BYTES = 64 << 20
MAX_ARTIFACTS = 4096
MAX_IO_NODES = 1024
MAX_SELECTED_NODES = 200
MAX_VISITED_OBJECTS = 2_000_000
_BLOBS = {"data", "raw_data", "rawdata", "tensor_content", "tensorcontent", "weights", "weight_data"}
_CUSTOM = {"CacheUpdate", "AdnRmsNorm", "AdnFusedInferAttention", "ChunkGatedDeltaRule",
           "LinearAttentionConvstateUpdate"}
_QUANT = re.compile(r"quant|grouped_?mat_?mul", re.I)
_LIMITATIONS = [
    "Existing OMC conversion only; no graph reconstruction, compilation or board execution.",
    "Only visible GraphDef.op objects are counted; opaque compiled payload coverage is unknown.",
    "Raw descriptor/attribute values and references are retained without dtype-enum decoding or runtime input-order inference.",
    "Data/RefData/NetOutput objects include all visible nested graphs; this is not a validated runtime ABI.",
    "Artifact hashes are recorded compilation identities; this inspection checks size/stat and does not rehash large weight files.",
    "Constant tensor payloads are omitted from summaries. Conversion does not establish numerical correctness or NPU placement.",
]


def _stat(path):
    value = path.stat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def _nonfinite_json(token):
    raise ValueError(f"nonfinite JSON token: {token}")


def _preview(value, *, items=32, budget=None, depth=0):
    budget = [256] if budget is None else budget
    budget[0] -= 1
    if budget[0] < 0:
        return {"preview_truncated": True}
    if isinstance(value, str):
        return value if len(value) <= 512 else {"truncated_string": value[:128], "length": len(value)}
    if isinstance(value, (dict, list)) and depth >= 10:
        return {"truncated_container": type(value).__name__, "length": len(value)}
    if isinstance(value, dict):
        result = {}
        for key, child in islice(value.items(), 80):
            if key.lower() in _BLOBS:
                result[key] = {"tensor_payload_omitted": True}
            else:
                result[key] = _preview(child, items=items, budget=budget, depth=depth + 1)
        if len(value) > 80:
            result["__truncated_keys__"] = len(value) - 80
        return result
    if isinstance(value, list):
        result = [_preview(child, items=items, budget=budget, depth=depth + 1) for child in value[:items]]
        if len(value) > items:
            result.append({"__truncated_items__": len(value) - items})
        return result
    return value


def summarize_json(document):
    """Count visible op arrays and retain bounded raw I/O and selected nodes."""
    stack = [(document, "$", None, False)]
    counts, io_counts, selected_counts = Counter(), Counter(), Counter()
    io, selected, visited = [], [], 0
    while stack and visited < MAX_VISITED_OBJECTS:
        value, path, scope, is_op = stack.pop()
        visited += 1
        if isinstance(value, dict):
            kind = value.get("type")
            if is_op and isinstance(kind, str):
                counts[kind] += 1
                if kind in {"Data", "RefData", "NetOutput"}:
                    io_counts[kind] += 1
                    if len(io) < MAX_IO_NODES:
                        io.append({"json_path": path, "op_array_path": scope, "type": kind,
                                   "raw_node": _preview(value, items=128, budget=[2048])})
                elif kind in _CUSTOM or _QUANT.search(kind) or "custom" in kind.lower():
                    selected_counts[kind] += 1
                    if len(selected) < MAX_SELECTED_NODES:
                        selected.append({"json_path": path, "op_array_path": scope, "type": kind,
                                         "raw_node": _preview(value)})
                if kind in {"Const", "Constant"}:
                    continue
            for key, child in value.items():
                if key.lower() in _BLOBS:
                    continue
                child_path = f"{path}/{key.replace('~', '~0').replace('/', '~1')}"
                if key == "op" and isinstance(child, list):
                    stack.extend((node, f"{child_path}/{i}", child_path, True)
                                 for i, node in reversed(list(enumerate(child))) if isinstance(node, dict))
                elif isinstance(child, (dict, list)):
                    stack.append((child, child_path, None, False))
        elif isinstance(value, list):
            stack.extend((child, f"{path}/{i}", None, False)
                         for i, child in reversed(list(enumerate(value))) if isinstance(child, (dict, list)))
    return {"schema_status": "OP_ARRAYS_OBSERVED" if counts else "NO_RECOGNIZED_OP_OBJECTS",
            "scan_complete": not stack, "op_count": sum(counts.values()),
            "op_type_counts": dict(sorted(counts.items())), "io_type_counts": dict(sorted(io_counts.items())),
            "io_nodes": io, "io_nodes_omitted": sum(io_counts.values()) - len(io),
            "selected_type_counts": dict(sorted(selected_counts.items())),
            "selected_nodes": selected, "selected_nodes_omitted": sum(selected_counts.values()) - len(selected),
            "limits": {"io_nodes": MAX_IO_NODES, "selected_nodes": MAX_SELECTED_NODES,
                       "visited_objects": MAX_VISITED_OBJECTS},
            "input_order": "NOT_INFERRED", "dtype_enums": "NOT_DECODED",
            "runtime_abi": "UNVERIFIED", "constant_payloads": "OMITTED_FROM_SUMMARY"}


def _artifact(root, record):
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise ValueError("invalid generated artifact record")
    named = root / record["path"]
    path = named.resolve()
    if named.is_symlink() or root not in path.parents or not path.is_file():
        raise ValueError("generated artifact missing, symlinked or outside compilation directory")
    if (type(record.get("bytes")) is not int or path.stat().st_size != record["bytes"]
            or not re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256", "")))):
        raise ValueError("generated artifact size/recorded hash is invalid")
    before = _stat(path)
    recorded_stat = record.get("stat", record.get("source_stat"))
    if recorded_stat is not None and list(recorded_stat) != before:
        raise ValueError("generated artifact stat differs from compilation record")
    return path, before


def inspect_existing_omc(compilation_dir, output_dir, timeout_seconds=300):
    """Return INSPECTED or UNAVAILABLE; never change the compilation result."""
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError("timeout_seconds must be finite and positive")
    root = Path(compilation_dir).resolve()
    source_report = root / "report.json"
    if source_report.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("compilation report exceeds JSON read budget")
    with source_report.open("rb") as stream:
        raw = stream.read(MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError("compilation report exceeds JSON read budget")
    compilation = json.loads(raw)
    if (not isinstance(compilation, dict) or compilation.get("status") != "COMPILED"
            or compilation.get("artifact_kind") != "qwen35-existing-air-compilation"):
        raise ValueError("inspection requires a COMPILED existing AIR report")
    ddk = Path(compilation["ddk_root"]).resolve()
    output = output_path(output_dir)
    if output == root or root in output.parents or output == ddk or ddk in output.parents:
        raise ValueError("inspection output must be outside compilation and DDK directories")
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "artifact_kind": "qwen35-existing-omc-inspection",
              "status": "UNAVAILABLE", "output": str(output), "comparison": str(output / "report.txt"),
              "compilation_status": "COMPILED", "source_report": str(source_report),
              "source_report_sha256": hashlib.sha256(raw).hexdigest(),
              "ddk_root": str(ddk), "ddk_id": compilation.get("ddk_id"),
              "device_execution": "NOT_RUN", "runtime_abi": "UNVERIFIED", "device_ready": False,
              "runtime_placement": "UNVERIFIED", "limits": {"json_bytes": MAX_JSON_BYTES,
              "timeout_seconds": min(timeout_seconds, 120), "generated_artifacts": MAX_ARTIFACTS},
              "artifacts": [], "limitations": _LIMITATIONS}
    observed, links = [], []
    stage = "compiler-identity"
    try:
        default_omg, env = omg_environment(ddk)
        omg_record = compilation.get("omg", {})
        if not isinstance(omg_record, dict):
            omg_record = {}
        omg = Path(omg_record.get("path", compilation.get("omg_path", default_omg))).resolve()
        expected_hash = omg_record.get("sha256", compilation.get("omg_sha256"))
        if ddk not in omg.parents or not os.access(omg, os.X_OK):
            raise ValueError("recorded OMG is not executable within the selected DDK")
        with omg.open("rb") as stream:
            if stream.read(4) != b"\x7fELF":
                raise ValueError("recorded OMG must be an ELF executable, not an environment wrapper")
        if sha256_file(omg) != expected_hash:
            raise ValueError("recorded OMG executable hash changed")
        report["omg"] = {"path": str(omg), "sha256": expected_hash}
        observed.append((omg, _stat(omg)))
        stage = "artifact-identity"
        primary_record = compilation["omc"]
        primary, stamp = _artifact(root, primary_record)
        if not primary_record["bytes"]:
            raise ValueError("compiled OMC is empty")
        observed.extend([(source_report, _stat(source_report)), (primary, stamp)])
        stage = "conversion-help"
        help_result = _call([str(omg), "--help"], output, env, ddk, "omg-help", min(timeout_seconds, 30))
        report["help"] = help_result
        with (output / "omg-help.log").open("rb") as stream:
            help_text = stream.read(1 << 20).decode("utf-8", errors="replace")
        if help_result["returncode"] != 0 or not _supports_conversion(help_text):
            raise RuntimeError("selected OMG help does not confirm --mode=1 model-to-JSON conversion")
        report["conversion_capability"] = "ADVERTISED_BY_SELECTED_OMG"
        stage = "stage-artifact-links"
        records = compilation.get("generated_artifacts", [primary_record])
        if not isinstance(records, list) or not 0 < len(records) <= MAX_ARTIFACTS:
            raise ValueError("invalid or excessive generated artifact inventory")
        report["sidecar_inventory"] = "RECORDED" if "generated_artifacts" in compilation else "UNAVAILABLE_PRIMARY_ONLY"
        work = output / "work"
        work.mkdir()
        by_path = {row["path"]: row for row in records}
        if len(by_path) != len(records):
            raise ValueError("duplicate generated artifact paths")
        if primary_record["path"] in by_path and any(
                by_path[primary_record["path"]].get(key) != primary_record.get(key)
                for key in ("bytes", "sha256")):
            raise ValueError("primary OMC identity conflicts with generated artifact inventory")
        by_path.setdefault(primary_record["path"], primary_record)
        for record in by_path.values():
            source, before = _artifact(root, record)
            if source != primary and primary.parent not in source.parents:
                raise ValueError("generated sidecar is outside the OMC directory tree")
            relative = source.relative_to(primary.parent)
            if relative.as_posix() in {"input.omc", "model.json", "convert.log", "convert-command.json"}:
                raise ValueError(f"generated artifact conflicts with conversion work name: {relative}")
            destination = work / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(source)
            links.append(destination)
            observed.append((source, before))
            report["artifacts"].append(dict(record, staged_path=str(destination.relative_to(output)),
                                            observed_stat=before, content_hash_recomputed=False))
        alias = work / "input.omc"
        alias.symlink_to(primary.name)
        links.append(alias)
        stage = "convert"
        conversion = _call([str(omg), "--mode=1", "--om=input.omc", "--json=model.json"],
                           work, env, ddk, "convert", min(timeout_seconds, 120))
        report["conversion"] = conversion
        converted = work / "model.json"
        if conversion["returncode"] != 0:
            raise RuntimeError(conversion.get("error", f"OMG conversion returned {conversion['returncode']}"))
        if converted.is_symlink() or not converted.is_file():
            raise RuntimeError("OMG did not produce a regular model.json")
        report["json"] = {"path": str(converted.relative_to(output)), "bytes": converted.stat().st_size}
        if converted.stat().st_size > MAX_JSON_BYTES:
            raise RuntimeError("converted JSON exceeds 64 MiB parse budget; retained without parsing or hashing")
        stage = "json-summary"
        with converted.open("rb") as stream:
            payload = stream.read(MAX_JSON_BYTES + 1)
        if len(payload) > MAX_JSON_BYTES:
            raise RuntimeError("converted JSON grew beyond parse budget")
        report["json"]["sha256"] = hashlib.sha256(payload).hexdigest()
        report["inspection"] = summarize_json(json.loads(payload, parse_constant=_nonfinite_json))
        if not report["inspection"]["op_count"] or not report["inspection"]["scan_complete"]:
            raise RuntimeError("converted JSON has no recognized op arrays or exceeds traversal budget")
        report["status"] = "INSPECTED"
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, RecursionError) as exc:
        report.update(stage=stage, error=f"{type(exc).__name__}: {exc}")
    finally:
        for link in reversed(links):
            if link.is_symlink():
                link.unlink()
        changed = []
        for path, before in observed:
            try:
                if _stat(path) != before:
                    changed.append(str(path))
            except OSError:
                changed.append(str(path))
        report["source_preservation"] = {"status": "CHANGED" if changed else (
                                          "UNCHANGED_BY_STAT" if observed else "NOT_CHECKED"),
                                          "changed": sorted(set(changed))}
        if changed:
            report.update(status="UNAVAILABLE", stage="source-recheck",
                          error="Input artifact/report/compiler metadata changed during inspection")
    save_json(output / "report.json", report)
    lines = ["Existing OMC raw graph inspection", f"status={report['status']}",
             "Compilation remains COMPILED; runtime ABI and execution are unverified."]
    if "error" in report:
        lines.append(report["error"])
    if "inspection" in report:
        data = report["inspection"]
        lines.extend(["Op type counts: " + json.dumps(data["op_type_counts"]),
                      "I/O node counts: " + json.dumps(data["io_type_counts"]),
                      "Quant/custom node counts: " + json.dumps(data["selected_type_counts"]),
                      "Raw descriptors, attrs and JSON paths: report.json; ordering/enums are not inferred."])
    (output / "report.txt").write_text("\n".join(lines + _LIMITATIONS) + "\n")
    return report
