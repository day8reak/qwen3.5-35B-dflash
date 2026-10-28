"""Read selected compiler IR descriptors without loading vendor code or weights.

Only the selected DDK's embedded HIAI ModelDef schema is tried. A dump that uses
another container/schema remains unsupported; no AIR or GraphDef guessing is
performed. Descriptor enum names do not explain integer operator attributes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from .omc_inspect import _preview
from .omc_wire import load_schema


_DUMP_DIRECTORY = Path("compile-work/generated/dump_graph")
_MAX_FILES = 16
_MAX_DIRECTORY_ENTRIES = 256
_MAX_FILE_BYTES = 64 << 20
_MAX_TOTAL_BYTES = 128 << 20
_MAX_REPORT_BYTES = 16 << 20
_MAX_GRAPHS = 64
_MAX_MATCHES = 16
_MAX_OUTPUT_BYTES = 256 << 10
_ATTRIBUTES = {"dtype", "shape", "file_path", "file_id", "transpose_x1", "transpose_x2",
               "transpose_a", "transpose_b", "adj_x1", "adj_x2"}


def _read(path, limit):
    if path.is_symlink() or not path.is_file():
        raise ValueError("inspection input must be a regular non-symlink file")
    before = path.stat()
    if before.st_size > limit:
        raise ValueError(f"inspection input exceeds {limit} byte budget")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    after = path.stat()
    if len(data) > limit:
        raise ValueError(f"inspection input exceeds {limit} byte budget")
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("inspection input changed while reading")
    return data


def _enum_fields(schema):
    """Obtain enum domains from the selected schema's descriptor field types."""
    fields = {}
    for field in schema._messages.get("hiai.proto.TensorDescriptor", {}).values():
        if field.kind == 14:
            values = schema.enums[field.type_name]
            if len(values) > 256:
                raise ValueError("TensorDescriptor enum exceeds report budget")
            fields[field.name] = {"enum_type": field.type_name, "values": values}
    if len(json.dumps(fields).encode()) > 32 << 10:
        raise ValueError("TensorDescriptor enum tables exceed report byte budget")
    return fields


def _descriptors(node, enum_fields):
    result = {}
    for key in ("input_desc", "output_desc"):
        raw = node.get(key, [])
        if not isinstance(raw, list):
            raise ValueError(f"{key} must be a descriptor list")
        rows = []
        for desc in raw[:16]:
            if not isinstance(desc, dict):
                raise ValueError("tensor descriptor must be an object")
            row = {"raw": _preview(desc, budget=[128]), "schema_enum_names": {}}
            for field, domain in enum_fields.items():
                if field in desc:
                    value = desc[field]
                    row["schema_enum_names"][field] = {
                        "value": value, "enum_type": domain["enum_type"],
                        "names": [name for name, number in domain["values"].items() if number == value]}
            rows.append(row)
        result[key] = rows
        if len(raw) > 16:
            result[key + "_omitted"] = len(raw) - 16
    return result


def _node(node, enum_fields):
    # Exclude tensor value attributes even though the decoder has already
    # replaced byte payloads by hashes. Only small conversion-relevant attrs.
    attrs = node.get("attr", [])
    if not isinstance(attrs, list):
        raise ValueError("operator attributes must be key/value entries")
    selected = [entry for entry in attrs if isinstance(entry, dict) and entry.get("key") in _ATTRIBUTES]
    return {"name": _preview(node.get("name")), "type": _preview(node.get("type")),
            "input": _preview(node.get("input", [])),
            "src_name": _preview(node.get("src_name", [])),
            "src_index": _preview(node.get("src_index", [])),
            "attributes": _preview(selected, budget=[256]),
            "other_attribute_count": len(attrs) - len(selected),
            **_descriptors(node, enum_fields)}


def _graphs(model):
    graphs = model.get("graph")
    if not isinstance(graphs, list) or not graphs:
        raise ValueError("decoded ModelDef has no graph list")
    pending = [(graph, f"$/graph/{i}") for i, graph in enumerate(graphs)]
    count = 0
    while pending:
        graph, path = pending.pop(0)
        count += 1
        if count > _MAX_GRAPHS:
            raise ValueError("graph count exceeds inspection budget")
        if not isinstance(graph, dict) or not isinstance(graph.get("op"), list):
            raise ValueError("decoded graph has no op list")
        yield graph, path
        for index, node in enumerate(graph["op"]):
            if not isinstance(node, dict):
                raise ValueError("decoded operation must be an object")
            for attr_index, entry in enumerate(node.get("attr", [])):
                value = entry.get("value", {}) if isinstance(entry, dict) else {}
                if isinstance(value, dict) and isinstance(value.get("g"), dict):
                    pending.append((value["g"], f"{path}/op/{index}/attr/{attr_index}/value/g"))


def _inputs(node):
    """Resolve only explicit exact same-graph references; retain their domain."""
    rows = []
    names, indices = node.get("src_name", []), node.get("src_index", [])
    if isinstance(names, list):
        for i, name in enumerate(names[:16]):
            rows.append({"field": "src_name", "input_index": i, "name": name,
                         "output_port": indices[i] if isinstance(indices, list) and i < len(indices) else None})
    edges = node.get("input", [])
    if isinstance(edges, list):
        for i, edge in enumerate(edges[:16]):
            name, port = None, None
            if isinstance(edge, str) and ":" in edge:
                prefix, suffix = edge.rsplit(":", 1)
                if re.fullmatch(r"-?\d+", suffix):
                    name, port = prefix, int(suffix)
            rows.append({"field": "input", "input_index": i, "raw_reference": _preview(edge),
                         "name": name, "output_port": port})
    return rows


def _select(model, node_name, enum_fields):
    matches = []
    for graph, graph_path in _graphs(model):
        by_name = {}
        for node in graph["op"]:
            if isinstance(node, dict) and isinstance(node.get("name"), str):
                by_name.setdefault(node["name"], []).append(node)
        targets = by_name.get(node_name, [])
        for target in targets:
            if len(matches) >= _MAX_MATCHES:
                raise ValueError("target match count exceeds inspection budget")
            row = {"graph_path": graph_path, "graph_name": _preview(graph.get("name")),
                   "same_graph_target_count": len(targets), "node": _node(target, enum_fields),
                   "input_producers": [], "named_fc_weights": []}
            for edge in _inputs(target):
                name = edge["name"]
                producers = by_name.get(name, []) if isinstance(name, str) else []
                edge["name"] = _preview(name)
                edge["status"] = "RESOLVED" if len(producers) == 1 else "AMBIGUOUS" if producers else "UNRESOLVED"
                if len(producers) == 1:
                    edge["producer"] = _node(producers[0], enum_fields)
                row["input_producers"].append(edge)
            # Exact weight name only; do not infer this is a parent when edges
            # are absent, and do not join names between separate graphs.
            weights = by_name.get("draft.fc.weight", [])
            row["named_fc_weight_count"] = len(weights)
            row["named_fc_weights"] = [_node(node, enum_fields) for node in weights[:2]]
            matches.append(row)
    return matches


def _inventory(root):
    path = root / "report.json"
    if not path.exists():
        return {}, {"status": "NO_RECORDED_INVENTORY"}
    raw = _read(path, _MAX_REPORT_BYTES)
    report = json.loads(raw)
    records = {}
    for key in ("generated_artifacts", "compiler_work_artifacts"):
        for row in report.get(key, []):
            if isinstance(row, dict) and isinstance(row.get("path"), str):
                if row["path"] in records:
                    raise ValueError("duplicate compiler artifact inventory path")
                records[row["path"]] = row
    return records, {"status": "RECORDED", "path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def inspect_draft_ir(compile_dir, schema_library, node_name="__ge/v20/flat_output"):
    """Return bounded diagnostics; never invoke the compiler, library or board."""
    if not isinstance(node_name, str) or not node_name or len(node_name) > 512:
        raise ValueError("node_name must contain 1..512 characters")
    root = Path(compile_dir).resolve()
    result = {"schema_version": 1, "artifact_kind": "qwen35-draft-ir-inspection",
              "status": "PARTIAL", "compilation_dir": str(root), "node_name": node_name,
              "dump_directory": str(root / _DUMP_DIRECTORY), "files": [],
              "limits": {"files": _MAX_FILES, "file_bytes": _MAX_FILE_BYTES,
                         "total_bytes": _MAX_TOTAL_BYTES, "output_bytes": _MAX_OUTPUT_BYTES},
              "limitations": ["Only selected-DDK HIAI ModelDef is decoded; other dump schemas remain unsupported.",
                              "Only exact target names and explicit same-graph producer edges are associated.",
                              "Tensor descriptor enum names do not decode FileConstant/MatMul integer attributes.",
                              "Serialized descriptors do not establish device support or the compilation root cause."]}
    try:
        schema = load_schema(schema_library)
        result["schema"] = schema.evidence
        enum_fields = _enum_fields(schema)
        result["tensor_descriptor_enum_fields"] = enum_fields
    except (OSError, ValueError, TypeError) as error:
        result.update(status="SCHEMA_UNAVAILABLE", schema_error=f"{type(error).__name__}: {error}")
        return result
    try:
        inventory, result["inventory"] = _inventory(root)
    except (OSError, ValueError, TypeError) as error:
        result.update(status="INVENTORY_UNAVAILABLE", inventory_error=f"{type(error).__name__}: {error}")
        return result
    folder = root / _DUMP_DIRECTORY
    if not folder.is_dir():
        result["status"] = "NO_DUMPS"
        return result
    if any(path.is_symlink() for path in (root / "compile-work", root / "compile-work/generated", folder)):
        result.update(status="UNSUPPORTED_DIRECTORY", directory_error="dump ancestors must not be symlinks")
        return result
    candidates = []
    try:
        for i, path in enumerate(folder.iterdir()):
            if i >= _MAX_DIRECTORY_ENTRIES:
                result["directory_scan_limited"] = True
                break
            if path.suffix == ".irpb":
                candidates.append(path)
    except OSError as error:
        result.update(status="UNSUPPORTED_DIRECTORY", directory_error=str(error))
        return result
    candidates.sort(key=lambda path: path.name)
    result["candidate_count"] = len(candidates)
    result["files_omitted"] = max(0, len(candidates) - _MAX_FILES)
    total = 0
    for path in candidates[:_MAX_FILES]:
        relative = path.relative_to(root).as_posix()
        row = {"path": relative, "status": "UNSUPPORTED_DUMP"}
        result["files"].append(row)
        try:
            raw = _read(path, min(_MAX_FILE_BYTES, _MAX_TOTAL_BYTES - total))
            total += len(raw)
            row.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            recorded = inventory.get(relative)
            row["inventory_binding"] = "UNRECORDED" if recorded is None else "MATCHED"
            if recorded is not None and (recorded.get("bytes") != len(raw) or recorded.get("sha256") != row["sha256"]):
                row.update(status="INTEGRITY_MISMATCH", inventory_binding="MISMATCHED")
                continue
            model = schema.decode(raw, "ModelDef")
            row["matches"] = _select(model, node_name, enum_fields)
            row["status"] = "TARGET_FOUND" if row["matches"] else "TARGET_NOT_FOUND"
            if len(json.dumps(result, ensure_ascii=True).encode()) > _MAX_OUTPUT_BYTES:
                row.pop("matches", None)
                row.update(status="OUTPUT_LIMIT", error="selected descriptor report exceeds output byte budget")
                result["output_limited"] = True
                break
        except (OSError, ValueError, TypeError, RecursionError) as error:
            row["error"] = f"{type(error).__name__}: {str(error)[:512]}"
    result["read_bytes"] = total
    if not candidates:
        result["status"] = "NO_DUMPS"
    elif any(row["status"] == "INTEGRITY_MISMATCH" for row in result["files"]):
        result["status"] = "INTEGRITY_MISMATCH"
    elif (not result.get("directory_scan_limited") and not result["files_omitted"] and
          all(row["status"] in {"TARGET_FOUND", "TARGET_NOT_FOUND"} for row in result["files"])):
        result["status"] = "INSPECTED"
    return result
