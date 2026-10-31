"""Read CPU GraphDef slices from existing OMC, bound to prior OMG JSON.

No vendor executable is invoked. The reference container layout is accepted
only after its main partition agrees with the same-artifact OMG conversion.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path
import re

from qwen35_moe_dflash.common.artifacts import checked_file, output_path, save_json
from qwen35_moe_dflash.toolchain.compile import omg_environment
from qwen35_moe_dflash.common.hashing import sha256_file
from qwen35_moe_dflash.toolchain.omc_container import read_container, read_partition
from qwen35_moe_dflash.toolchain.omc_inspect import _preview, inspect_json
from qwen35_moe_dflash.toolchain.omc_wire import load_schema


_LIMITATIONS = [
    "Offline CPU subgraph inspection only; no OMG invocation, compilation or board execution.",
    "CPU slices use explicit GraphOp offsets/sizes after main-partition agreement with the retained OMG JSON.",
    "Descriptor dtype names come from the selected DDK protobuf schema; Cast attribute integers use a separate enum.",
    "A matching serialized node does not bind the older HiLog PID to this run or prove a complete runtime root cause.",
    "Unsupported layouts, schema differences and ambiguous graph ownership remain explicit; no byte-pattern graph recovery.",
]


def _one(items, name, label):
    matches = [row for row in items if isinstance(row, dict) and row.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {label}: {name}")
    return matches[0]


def _attr(node, name):
    attrs = node.get("attr", [])
    if not isinstance(attrs, list):
        raise ValueError("unsupported attribute representation; expected key/value entries")
    matches = [entry.get("value") for entry in attrs if isinstance(entry, dict) and entry.get("key") == name]
    if len(matches) != 1 or not isinstance(matches[0], dict):
        raise ValueError(f"missing or ambiguous {node.get('name')}.{name}")
    return matches[0]


def _uint(value):
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < (1 << 64):
        raise ValueError("expected a nonnegative integer, not a float/bool or negative offset")
    return value


def _dim(value):
    if isinstance(value, str) and re.fullmatch(r"-?(?:0|[1-9][0-9]*)", value, re.ASCII):
        value = int(value)
    if type(value) is not int or not -(1 << 63) <= value < (1 << 63):
        raise ValueError("shape dimensions must be exact int64 values")
    return value


def _object(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _list(value, label):
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _descriptors(value):
    rows = []
    for desc in _list(value, "descriptors"):
        _object(desc, "descriptor")
        shape = _object(desc.get("shape", {}), "descriptor shape")
        dims = _list(shape.get("dim", []), "shape dimensions")
        layout = desc.get("layout", "")
        if not isinstance(layout, str):
            raise ValueError("descriptor layout must be a string")
        rows.append({"dtype": _uint(desc.get("dtype", 0)),
                     "shape": [_dim(dim) for dim in dims], "layout": layout})
    return rows


def _task(node):
    cl = _attr(node, "cl_name").get("s")
    if not isinstance(cl, str):
        raise ValueError("GraphOp cl_name is not decoded text")
    return {"cl_name": cl,
            "offset": _uint(_attr(node, "graphop_task_offset").get("i")),
            "bytes": _uint(_attr(node, "graphop_task_size").get("i"))}


def _structure(document):
    """Compare only fields retained by mode1; it deliberately omits payloads.

    Graph names, ordered operations, input references, tensor dtypes/shapes and
    explicit task ownership must agree. No dtype translation is performed.
    """
    result = []
    _object(document, "main ModelDef")
    graphs = document.get("graph")
    if not isinstance(graphs, list) or not graphs:
        raise ValueError("main ModelDef has no graph list")
    for graph in graphs:
        _object(graph, "main graph")
        if not isinstance(graph.get("name", ""), str):
            raise ValueError("main graph name must be a string")
        ops = graph.get("op")
        if not isinstance(ops, list) or not ops:
            raise ValueError("main graph has no op list")
        rows = []
        names = set()
        for node in ops:
            _object(node, "main op")
            name, kind = node.get("name"), node.get("type")
            if not isinstance(name, str) or not name or name in names or not isinstance(kind, str) or not kind:
                raise ValueError("main graph operation names/types are missing or ambiguous")
            names.add(name)
            row = {"name": name, "type": kind, "input": node.get("input", [])}
            if any(not isinstance(edge, str) for edge in _list(row["input"], "input references")):
                raise ValueError("input references must be strings")
            for key in ("input_desc", "output_desc"):
                row[key] = _descriptors(node.get(key, []))
            if kind == "GraphOp":
                row["task"] = _task(node)
            rows.append(row)
        result.append({"name": graph.get("name", ""), "op": rows})
    return result


def _port_candidates(graph, inspection):
    """Preserve exact node:port matches as candidates, without name stripping.

    The raw reference remains in the report. Duplicate names, absent descriptor
    ports and malformed references never produce an association.
    """
    ops = graph.get("op", [])
    ports = {}
    names = Counter(node.get("name") for node in ops)
    for node in ops:
        name = node.get("name")
        if not isinstance(name, str) or names[name] != 1:
            continue
        for port, descriptor in enumerate(node.get("output_desc", [])):
            ports.setdefault(f"{name}:{port}", []).append((node, port, descriptor))
    for cast in inspection["casts"]:
        # This helper resolves only this GraphDef's direct operations. Nested
        # graphs retain their own path and are not joined to the parent's ops.
        if cast["op_array_path"] != "$/op":
            continue
        nodes = [node for node in ops if node.get("name") == cast["name"]]
        if len(nodes) != 1:
            continue
        candidates = []
        for edge in nodes[0].get("input", []):
            found = ports.get(edge, []) if isinstance(edge, str) else []
            if len(found) == 1 and len(candidates) < 8:
                node, port, descriptor = found[0]
                candidates.append({"raw_reference": edge, "output_port": port,
                                   "output_descriptor": _preview(descriptor),
                                   "raw_node": _preview(node)})
        cast["same_graph_port_candidates"] = candidates


def _parse_graph(source, main_json, folder, schema, max_bytes):
    container = read_container(source)
    main = next(row for row in container["partitions"] if row["type"] == 0)
    tasks = next(row for row in container["partitions"] if row["type"] == 2)
    model = schema.decode(read_partition(source, main, max_bytes=max_bytes), "ModelDef")
    structure = _structure(model)
    if structure != _structure(main_json):
        raise ValueError("decoded main partition differs from the retained OMG JSON; CPU slices not interpreted")
    structure_hash = hashlib.sha256(json.dumps(structure, sort_keys=True).encode()).hexdigest()
    result = {"container": container, "main_structure_sha256": structure_hash,
              "main_json_agreement": "MATCHED", "cpu_subgraphs": [], "other_subgraphs": []}
    spans = []
    for graph_index, graph in enumerate(model["graph"]):
        for node in graph["op"]:
            if node["type"] != "GraphOp":
                continue
            task = _task(node)
            if task["offset"] > tasks["bytes"] or task["bytes"] == 0 or task["bytes"] > tasks["bytes"] - task["offset"]:
                raise ValueError(f"GraphOp task slice outside compiled-target partition: {node['name']}")
            # Distinct task owners must not overlap. Such a container remains
            # unsupported even if one byte range happens to parse as GraphDef.
            end = task["offset"] + task["bytes"]
            if any(task["offset"] < other_end and start < end for start, other_end in spans):
                raise ValueError(f"overlapping GraphOp task slices: {node['name']}")
            spans.append((task["offset"], end))
            record = {"owner_graph_index": graph_index, "owner_graph_name": graph.get("name", ""),
                      "graphop_name": node["name"], "task": task,
                      "file_offset": tasks["offset"] + task["offset"]}
            if task["cl_name"] != "CPUCL":
                result["other_subgraphs"].append(record)
                continue
            result["cpu_subgraphs"].append(record)
    # Resolve all ownership bounds before decoding or retaining any CPU graph.
    for index, record in enumerate(result["cpu_subgraphs"]):
        try:
            raw = read_partition(source, {"offset": record["file_offset"], "bytes": record["task"]["bytes"]},
                                 max_bytes=max_bytes)
            record["slice_sha256"] = hashlib.sha256(raw).hexdigest()
            graph = schema.decode(raw, "GraphDef")
            record["serialized_graph_name"] = graph.get("name", "")
            if graph.get("name") != record["graphop_name"]:
                raise ValueError(f"GraphDef name {graph.get('name')!r} differs from GraphOp {record['graphop_name']!r}")
            names = [node.get("name") for node in graph.get("op", [])]
            if not names or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
                raise ValueError("CPU graph has missing/ambiguous operation names")
            path = folder / f"cpu-{index:03d}.json"
            save_json(path, graph)
            record["decoded_json"] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            record["inspection"] = inspect_json(graph)
            record["inspection"]["coverage"] = "This decoded CPU GraphDef slice only; other task kinds and opaque byte fields remain uninspected"
            _port_candidates(graph, record["inspection"])
            record["status"] = "DECODED"
        except (OSError, ValueError, RecursionError) as error:
            record.update(status="UNSUPPORTED_CPU_SLICE", error=f"{type(error).__name__}: {error}")
    result["status"] = ("CPU_GRAPHS_DECODED" if result["cpu_subgraphs"] and
                        all(row["status"] == "DECODED" for row in result["cpu_subgraphs"]) else "CPU_GRAPH_COVERAGE_PARTIAL")
    return result


def _render(report):
    lines = ["Existing OMC CPU subgraph inspection", f"Status: {report['status']}"]
    if "schema_error" in report:
        lines.append("Schema: " + report["schema_error"])
    details = 0
    for graph in report["graphs"]:
        lines.extend(["", f"GRAPH: {graph['name']}", f"CPU inspection: {graph['status']}"])
        if "error" in graph:
            lines.append(graph["error"])
        for subgraph in graph.get("cpu_subgraphs", []):
            lines.append(f"  {subgraph['graphop_name']}: {subgraph['status']}; offset={subgraph['file_offset']}, bytes={subgraph['task']['bytes']}")
            if "error" in subgraph:
                lines.append("    " + subgraph["error"])
            inspection = subgraph.get("inspection", {})
            lines.append(f"    Cast nodes: {inspection.get('cast_object_count', 'unknown')}")
            for cast in sorted(inspection.get("casts", []), key=lambda row: row["name"] != "insert_cast_4")[:12]:
                lines.append(f"    {cast['name']} ({cast['type']}) @ {cast['json_path']}")
                if cast["name"] != "insert_cast_4" or details >= 4:
                    continue
                details += 1
                for label, value in [("node", cast["raw_node"]),
                                     *[("input port candidate", row) for row in cast.get("same_graph_port_candidates", [])[:2]]]:
                    text = json.dumps(value, ensure_ascii=False)
                    lines.append("      " + label + "=" + (text if len(text) < 6000 else text[:6000] + "... [see report.json]"))
    return "\n".join(lines + [""] + _LIMITATIONS) + "\n"


def inspect_omc_cpu(bundle, prior_inspection, output_dir, ddk_root, *, ddk_id, max_partition_bytes=64 << 20):
    if max_partition_bytes <= 0:
        raise ValueError("partition byte budget must be positive")
    root, prior = Path(bundle).resolve(), Path(prior_inspection).resolve()
    manifest_file, prior_file = root / "manifest.json", prior / "report.json"
    manifest_raw, prior_raw = manifest_file.read_bytes(), prior_file.read_bytes()
    manifest, previous = json.loads(manifest_raw), json.loads(prior_raw)
    manifest_sha, prior_sha = hashlib.sha256(manifest_raw).hexdigest(), hashlib.sha256(prior_raw).hexdigest()
    if manifest.get("artifact_kind") != "qwen35-cpro-omc-pair" or manifest.get("status") != "COMPILED":
        raise ValueError("expected successful existing OMC pair")
    if previous.get("artifact_kind") != "qwen35-omc-cast-inspection" or previous.get("source_manifest_sha256") != manifest_sha:
        raise ValueError("prior inspection is not bound to this OMC manifest")
    omg, _ = omg_environment(ddk_root)
    omg_sha = sha256_file(omg)
    if not ddk_id or any(doc.get("ddk_id") != ddk_id or doc.get("omg_sha256") != omg_sha for doc in (manifest, previous)):
        raise ValueError("DDK identity/OMG hash differs from recorded compiler or inspection")
    graphs = manifest.get("graphs", [])
    if [row.get("name") for row in graphs] != ["target", "draft"]:
        raise ValueError("expected target and draft records, in order")
    inputs = []
    for row in graphs:
        old = _one(previous.get("graphs", []), row["name"], "prior inspected graph")
        if old.get("source_sha256") != row["sha256"] or old.get("conversion", {}).get("status") != "CONVERTED":
            raise ValueError("prior conversion is missing or belongs to different OMC bytes")
        source_record = {"path": row["omc"], "bytes": row["bytes"], "sha256": row["sha256"]}
        source = checked_file(root, source_record)
        converted_record = {"path": f"{row['name']}/model.json", "bytes": old["conversion"]["json_bytes"],
                            "sha256": old["conversion"]["json_sha256"]}
        converted = checked_file(prior, converted_record)
        if converted_record["bytes"] > max_partition_bytes:
            raise ValueError("retained main JSON exceeds inspection byte budget")
        inputs.append((row, source, source_record, converted, converted_record))
    output = output_path(output_dir)
    if output.is_relative_to(root) or output.is_relative_to(prior):
        raise ValueError("new inspection must be outside the original OMC and prior inspection directories")
    output.mkdir(parents=True, exist_ok=False)
    result = {"schema_version": 1, "artifact_kind": "qwen35-omc-cpu-inspection",
              "status": "DIAGNOSTIC_PARTIAL", "source_manifest": str(manifest_file),
              "source_manifest_sha256": manifest_sha, "prior_report": str(prior_file),
              "prior_report_sha256": prior_sha, "ddk_id": ddk_id, "omg_sha256": omg_sha,
              "device_execution": "NOT_RUN", "runtime_placement": "UNVERIFIED",
              "graphs": [], "limitations": _LIMITATIONS}
    library = Path(ddk_root).resolve() / "tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so"
    schema = None
    try:
        schema = load_schema(library)
        save_json(output / "schema-evidence.json", {"source": schema.evidence, "protobuf_enums": schema.enums})
        result["schema"] = schema.evidence
    except (OSError, ValueError) as error:
        result["schema_error"] = f"{type(error).__name__}: {error}"
    for row, source, _, converted, _ in inputs:
        folder = output / row["name"]
        folder.mkdir()
        graph = {"name": row["name"], "source": str(source), "source_sha256": row["sha256"],
                 "status": "SCHEMA_UNAVAILABLE"}
        result["graphs"].append(graph)
        if schema is None:
            continue
        try:
            graph.update(_parse_graph(source, json.loads(converted.read_bytes()), folder, schema, max_partition_bytes))
        except (OSError, ValueError, RecursionError) as error:
            graph.update(status="UNSUPPORTED_LAYOUT_OR_SCHEMA", error=f"{type(error).__name__}: {error}")
    if sha256_file(manifest_file) != manifest_sha or sha256_file(prior_file) != prior_sha or sha256_file(omg) != omg_sha:
        raise ValueError("inspection inputs changed: manifest, prior report or OMG")
    if schema is not None and sha256_file(library) != schema.evidence["library_sha256"]:
        raise ValueError("inspection schema library changed")
    for _, _, source_record, _, converted_record in inputs:
        checked_file(root, source_record)
        checked_file(prior, converted_record)
    if all(graph["status"] == "CPU_GRAPHS_DECODED" for graph in result["graphs"]):
        result["status"] = "DIAGNOSTIC_COMPLETE"
    save_json(output / "report.json", result)
    (output / "report.txt").write_text(_render(result), encoding="utf-8")
    return result
