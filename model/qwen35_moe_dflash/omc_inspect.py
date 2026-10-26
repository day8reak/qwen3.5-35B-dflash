"""Inspect existing OMCs with their recorded OMG; never compile or execute them."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from .artifacts import checked_file, output_path, save_json
from .compile import _environment_record, _omg_ipc_environment, omg_environment
from .hashing import sha256_file


_LIMITATIONS = [
    "Existing OMC inspection only: no compilation or board execution.",
    "JSON may omit opaque compiled subgraphs; missing names do not prove Load-time insertion.",
    "Byte matches are strings, not decoded nodes or proof of a particular graph edge.",
    "Raw attributes and descriptors are not interpreted using a different DDK's dtype enum.",
    "Compiler partitions and successful conversion do not prove runtime NPU placement.",
]
_INSERTED = re.compile(rb"insert_cast_[0-9]{1,20}(?![0-9])")


def _scan_names(path):
    """Stream names without treating a compiled payload as a graph schema."""
    counts, offsets = Counter(), {}
    tail, position = b"", 0
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            data = tail + chunk
            base = position - len(tail)
            # Delay the overlap to avoid counting an incomplete suffix twice.
            end = max(0, len(data) - 128)
            for match in _INSERTED.finditer(data):
                if match.start() >= end:
                    break
                name = match[0].decode("ascii")
                if len(counts) < 1000 or name in counts:
                    counts[name] += 1
                    offsets.setdefault(name, [])
                    if len(offsets[name]) < 3:
                        offsets[name].append(base + match.start())
            tail, position = data[end:], position + len(chunk)
        for match in _INSERTED.finditer(tail):
            name = match[0].decode("ascii")
            if len(counts) < 1000 or name in counts:
                counts[name] += 1
                offsets.setdefault(name, [])
                if len(offsets[name]) < 3:
                    offsets[name].append(position - len(tail) + match.start())
    return {"counts": dict(counts), "example_byte_offsets": offsets,
            "distinct_name_limit": 1000, "scope": "Raw byte strings, not decoded graph nodes"}


def _preview(value, depth=0, budget=None):
    """Bound report size while retaining raw scalar types and attribute keys."""
    budget = [256] if budget is None else budget
    budget[0] -= 1
    if budget[0] < 0:
        return {"preview_truncated": True}
    if isinstance(value, str):
        return value if len(value) <= 512 else {"truncated_string": value[:128], "length": len(value)}
    if isinstance(value, (dict, list)) and depth >= 10:
        return {"truncated_container": type(value).__name__, "length": len(value)}
    if isinstance(value, dict):
        result = {key: _preview(item, depth + 1, budget) for key, item in list(value.items())[:80]}
        if len(value) > 80:
            result["__omc_inspect_truncated_keys__"] = len(value) - 80
        return result
    if isinstance(value, list):
        result = [_preview(item, depth + 1, budget) for item in value[:32]]
        if len(value) > 32:
            result.append({"__omc_inspect_truncated_items__": len(value) - 32})
        return result
    return value


def inspect_json(document):
    """Find raw Cast objects; resolve edges only within the same op array.

    No prefix stripping, dtype decoding, or association with an AIR recipe.
    JSON paths identify exactly which converted object is being displayed.
    """
    stack = [(document, "$", None, None, False)]
    matches, count, nodes = [], 0, 0
    while stack:
        value, path, scope, siblings, is_op = stack.pop()
        if isinstance(value, dict):
            name, kind = value.get("name"), value.get("type")
            if is_op and isinstance(name, str) and isinstance(kind, str):
                nodes += 1
                if kind in {"Cast", "CastT"} or re.fullmatch(r"insert_cast_\d+", name):
                    count += 1
                    if len(matches) < 200 or name == "insert_cast_4":
                        if len(matches) >= 200:
                            # Keep the reported failing node even in cast-heavy graphs.
                            discard = next((i for i in reversed(range(len(matches)))
                                            if matches[i]["name"] != "insert_cast_4"), None)
                            if discard is None:
                                continue
                            matches.pop(discard)
                        row = {"json_path": path, "op_array_path": scope,
                               "name": name, "type": kind, "raw_node": _preview(value)}
                        parents, consumers, unresolved = [], [], []
                        parent_count = consumer_count = unresolved_count = 0
                        if siblings is not None:
                            by_name = {}
                            for other in siblings:
                                if isinstance(other, dict) and isinstance(other.get("name"), str):
                                    by_name.setdefault(other["name"], []).append(other)
                            inputs = value.get("src_name", value.get("srcName", []))
                            row["edge_source"] = "explicit src_name/srcName fields only"
                            if isinstance(inputs, list):
                                for source in inputs:
                                    if not isinstance(source, str):
                                        unresolved_count += 1
                                        if len(unresolved) < 8:
                                            unresolved.append(_preview(source))
                                        continue
                                    found = by_name.get(source, [])
                                    if len(found) == 1:
                                        parent_count += 1
                                        if len(parents) < 8:
                                            parents.append({"source_name": source, "raw_node": _preview(found[0])})
                                    else:
                                        unresolved_count += 1
                                        if len(unresolved) < 8:
                                            unresolved.append(_preview(source))
                            for other in siblings:
                                if not isinstance(other, dict):
                                    continue
                                sources = other.get("src_name", other.get("srcName", []))
                                if isinstance(sources, list) and name in sources:
                                    consumer_count += 1
                                    if len(consumers) < 8:
                                        consumers.append(_preview(other))
                            row["raw_input_references"] = _preview(value.get("input", []))
                            row["raw_input_encoding"] = "NOT_INTERPRETED"
                        row.update(producers=parents, consumers=consumers,
                                   producer_count=parent_count, consumer_count=consumer_count,
                                   unresolved_inputs=unresolved, unresolved_input_count=unresolved_count,
                                   neighbor_limit=8)
                        matches.append(row)
            for key, child in reversed(list(value.items())):
                child_path = f"{path}/{key.replace('~', '~0').replace('/', '~1')}"
                if key == "op" and isinstance(child, list):
                    stack.extend((node, f"{child_path}/{i}", child_path, child, True)
                                 for i, node in reversed(list(enumerate(child))))
                elif isinstance(child, (dict, list)):
                    stack.append((child, child_path, None, None, False))
        elif isinstance(value, list):
            stack.extend((value[i], f"{path}/{i}", None, None, False)
                         for i in reversed(range(len(value))) if isinstance(value[i], (dict, list)))
    return {"schema_status": "OP_ARRAYS_OBSERVED" if nodes else "NO_RECOGNIZED_OP_OBJECTS",
            "named_typed_object_count": nodes, "cast_object_count": count,
            "reported_cast_count": len(matches), "cast_limit": 200, "casts": matches,
            "coverage": "Only objects visible in OMG JSON; opaque payload coverage is unknown"}


def _call(argv, folder, env, ddk_root, stem, timeout):
    with _omg_ipc_environment(folder, env) as (child, ipc):
        save_json(folder / f"{stem}-command.json", {"argv": argv, "cwd": str(folder),
                  "environment": _environment_record(ddk_root, child), "ipc": ipc})
        log = folder / f"{stem}.log"
        with log.open("xb") as stream:
            try:
                completed = subprocess.run(argv, cwd=folder, env=child, stdout=stream,
                                           stderr=subprocess.STDOUT, timeout=timeout, check=False)
                result = {"returncode": completed.returncode}
            except (OSError, subprocess.TimeoutExpired) as error:
                result = {"returncode": None, "error": f"{type(error).__name__}: {error}"}
        result.update(log=str(log), log_sha256=sha256_file(log))
        return result


def _supports_conversion(help_text):
    # Only the documented OM conversion branch. --framework selects a different
    # conversion route, so never reuse compilation arguments here.
    return (all(re.search(r"--" + flag + r"\b", help_text) for flag in ("mode", "om", "json"))
            and bool(re.search(r"--mode\b[^\n]*\b1\s*:\s*model\s+to\s+json", help_text, re.I)))


def _render(report):
    lines = ["Existing OMC Cast inspection", f"Status: {report['status']}",
             f"OMG conversion help: {report['conversion_capability']}", ""]
    for graph in report["graphs"]:
        lines += [f"GRAPH: {graph['name']}",
                  f"Compiler main partitions: {json.dumps(graph['main_partition_events'])}",
                  f"Compiler rollback: {json.dumps(graph['rollback_events'])}",
                  f"OMC inserted-Cast byte strings: {json.dumps(graph['byte_scan']['counts'])}",
                  f"Conversion: {graph['conversion']['status']}"]
        if "error" in graph["conversion"]:
            lines.append(f"  {graph['conversion']['error']}")
        if "inspection" in graph:
            data = graph["inspection"]
            lines.append(f"JSON Cast objects: {data['cast_object_count']} ({data['reported_cast_count']} retained in report.json)")
            selected = sorted(data["casts"], key=lambda row: row["name"] != "insert_cast_4")
            for index, row in enumerate(selected[:12]):
                lines.append(f"  {row['json_path']} name={row['name']} type={row['type']}")
                if index >= 2:
                    continue
                text = json.dumps(row["raw_node"], ensure_ascii=False)
                lines.append("    node=" + (text if len(text) <= 4000 else text[:4000] + "... [see report.json]"))
                for parent in row["producers"][:2]:
                    text = json.dumps(parent, ensure_ascii=False)
                    lines.append("    producer=" + (text if len(text) <= 4000 else text[:4000] + "... [see report.json]"))
        lines.append("")
    return "\n".join(lines + _LIMITATIONS) + "\n"


def inspect_omc(bundle, output_dir, ddk_root, *, ddk_id, timeout_seconds=300,
                max_json_bytes=64 << 20):
    if timeout_seconds <= 0 or max_json_bytes <= 0:
        raise ValueError("timeout and JSON byte budget must be positive")
    root = Path(bundle).resolve()
    manifest_path = root / "manifest.json"
    raw = manifest_path.read_bytes()
    manifest_hash = hashlib.sha256(raw).hexdigest()
    manifest = json.loads(raw)
    if manifest.get("artifact_kind") != "qwen35-cpro-omc-pair" or manifest.get("status") != "COMPILED":
        raise ValueError("inspect-omc requires a successful existing OMC pair")
    graphs = manifest.get("graphs", [])
    if [row.get("name") for row in graphs] != ["target", "draft"]:
        raise ValueError("expected exactly the target and draft OMC records, in order")
    omg, env = omg_environment(ddk_root)
    omg_hash = sha256_file(omg)
    if not ddk_id or manifest.get("ddk_id") != ddk_id or manifest.get("omg_sha256") != omg_hash:
        raise ValueError("inspection DDK identity/OMG hash differs from the compiler recorded in the OMC manifest")
    if not os.access(omg, os.X_OK):
        raise PermissionError(f"OMG is not executable: {omg}")
    sources = []
    for graph in graphs:
        record = {"path": graph["omc"], "bytes": graph["bytes"], "sha256": graph["sha256"]}
        sources.append((checked_file(root, record), record))
    output = output_path(output_dir)
    if output.is_relative_to(root):
        raise ValueError("inspection output must be outside the existing OMC bundle")
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "status": "DIAGNOSTIC_PARTIAL",
              "artifact_kind": "qwen35-omc-cast-inspection", "source_manifest": str(manifest_path),
              "source_manifest_sha256": manifest_hash, "ddk_id": ddk_id, "omg_sha256": omg_hash,
              "device_execution": "NOT_RUN", "runtime_placement": "UNVERIFIED",
              "graphs": [], "limitations": _LIMITATIONS}
    help_result = _call([str(omg), "--help"], output, env, ddk_root, "omg-help", min(timeout_seconds, 30))
    help_log = output / "omg-help.log"
    with help_log.open("rb") as stream:
        help_text = stream.read(1 << 20).decode("utf-8", errors="replace")
    supported = help_result["returncode"] == 0 and _supports_conversion(help_text)
    report.update(help=help_result, conversion_capability="SUPPORTED" if supported else "UNCONFIRMED")
    for graph, (source, record) in zip(graphs, sources):
        folder = output / graph["name"]
        folder.mkdir()
        evidence = graph.get("compiler_evidence", {})
        row = {"name": graph["name"], "source": str(source), "source_sha256": record["sha256"],
               "main_partition_events": evidence.get("main_partition_events", []),
               "rollback_events": evidence.get("rollback_events", []), "byte_scan": _scan_names(source),
               "conversion": {"status": "SKIPPED_HELP_UNCONFIRMED"}}
        report["graphs"].append(row)
        if not supported:
            continue
        link = folder / "input.omc"
        link.symlink_to(source)
        try:
            run = _call([str(omg), "--mode=1", "--om=input.omc", "--json=model.json"],
                        folder, env, ddk_root, "convert", timeout_seconds)
        finally:
            link.unlink()
        row["conversion"] = dict(run, status="FAILED")
        converted = folder / "model.json"
        if run["returncode"] != 0:
            row["conversion"]["error"] = run.get("error", f"OMG returned {run['returncode']}; see {folder / 'convert.log'}")
        elif converted.is_symlink() or not converted.is_file():
            row["conversion"]["error"] = "OMG returned zero without a regular model.json"
        else:
            row["conversion"].update(json_bytes=converted.stat().st_size, json_sha256=sha256_file(converted))
            if converted.stat().st_size > max_json_bytes:
                row["conversion"].update(status="JSON_RETAINED_NOT_PARSED", error=f"JSON exceeds parse budget {max_json_bytes}")
            else:
                try:
                    row["inspection"] = inspect_json(json.loads(converted.read_bytes()))
                    row["conversion"]["status"] = ("CONVERTED" if row["inspection"]["named_typed_object_count"]
                                                   else "JSON_SCHEMA_UNRECOGNIZED")
                    if row["conversion"]["status"] == "JSON_SCHEMA_UNRECOGNIZED":
                        row["conversion"]["error"] = "JSON retained but contains no recognized GraphDef.op objects; node coverage is unknown"
                except (ValueError, RecursionError) as error:
                    row["conversion"]["error"] = f"JSON inspection failed: {error}"
    if sha256_file(manifest_path) != manifest_hash or sha256_file(omg) != omg_hash:
        raise ValueError("inspection input changed: OMC manifest or OMG executable")
    for _, record in sources:
        checked_file(root, record)
    report["status"] = ("DIAGNOSTIC_UNSUPPORTED" if not supported else
                        "DIAGNOSTIC_COMPLETE" if all(row["conversion"]["status"] == "CONVERTED"
                                                     for row in report["graphs"]) else "DIAGNOSTIC_PARTIAL")
    save_json(output / "report.json", report)
    (output / "report.txt").write_text(_render(report), encoding="utf-8")
    return report
