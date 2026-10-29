"""Read bounded AIR graph metadata without loading weights or vendor code.

An explicit TorchAir descriptor or selected HIAI schema is a candidate for AIR. A decoded protobuf is
recognized only after its actual root boundaries pass the original 89/81
checks. This module never serializes a graph or infers decoder tap producers.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import stat

from .omc_container import read_container
from .omc_wire import load_schema
from .air_schema import load_torchair_schema


MAX_AIR_BYTES = 64 << 20
MAX_OUTPUT_BYTES = 16 << 20
_MAX_OPS = 32768
_MAX_TEXT = 1024
_EXAMPLES = 6
_LAYER_IDS = (1, 6, 11, 16, 22, 27, 32, 37)


def _stamp(path):
    value = path.stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _text(value):
    if not isinstance(value, str):
        raise ValueError("graph name/reference is not text")
    return value if len(value) <= _MAX_TEXT else value[:_MAX_TEXT] + " [truncated]"


def _attrs(node):
    entries = node.get("attr", [])
    if not isinstance(entries, list) or len(entries) > 4096:
        raise ValueError("invalid or excessive attribute entries")
    result = {}
    for row in entries:
        key = row.get("key") if isinstance(row, dict) else None
        if not isinstance(key, str) or key in result:
            raise ValueError("missing or duplicate attribute key")
        result[key] = row.get("value", {})
    return result


def _desc(value, enum):
    if not isinstance(value, dict):
        raise ValueError("tensor descriptor is missing")
    raw_shape = value.get("shape")
    shape = raw_shape.get("dim", []) if isinstance(raw_shape, dict) else None
    if (not isinstance(shape, list) or len(shape) > 16
            or any(type(d) is not int or d < -1 for d in shape)):
        raise ValueError("tensor shape is missing or unsupported")
    raw = value.get("dtype")
    names = [k for k, v in enum.items() if type(raw) is int and v == raw]
    return {"shape": shape, "raw_dtype": raw, "schema_dtype_names": names,
            "layout": value.get("layout")}


def _metadata_desc(value, enum):
    """Incomplete internal metadata cannot weaken strict host boundaries."""
    try:
        return _desc(value, enum)
    except (ValueError, TypeError) as error:
        return {"status": "UNAVAILABLE", "error": str(error)[:256]}


def _edge(edge, by_name, enum):
    if not isinstance(edge, str) or ":" not in edge:
        raise ValueError("output has no explicit producer port")
    name, port = edge.rsplit(":", 1)
    if not re.fullmatch(r"0|[1-9][0-9]*", port):
        raise ValueError("output producer port is unsupported")
    node, port = by_name.get(name), int(port)
    if node is None or port >= len(node.get("output_desc", [])):
        raise ValueError("output producer/descriptor does not exist")
    return _desc(node["output_desc"][port], enum)


def _file_constant(node, enum):
    attrs = _attrs(node)
    path = attrs.get("file_path", {}).get("s")
    row = {"name": _text(node["name"]), "path_status": "TEXT" if isinstance(path, str) else "UNRESOLVED",
           "file_path": _text(path) if isinstance(path, str) else None,
           "outputs": [_metadata_desc(v, enum) for v in node.get("output_desc", [])[:4]],
           "attribute_names": [_text(k) for k in attrs], "payload": "NOT_OPENED_NOT_HASHED_NOT_CHECKED"}
    # Preserve only declared integer extent metadata. Do not resolve a path,
    # stat a sidecar, or infer physical storage from the tensor's logical size.
    row["declared_extent"] = {key: attrs[key]["i"] for key in ("offset", "length", "size", "file_offset", "file_size")
                              if isinstance(attrs.get(key), dict) and type(attrs[key].get("i")) is int}
    row["extent_status"] = "DECLARED_UNVERIFIED" if row["declared_extent"] else "NOT_DECLARED"
    for key in ("dtype", "shape"):
        value = attrs.get(key, {})
        if isinstance(value, dict) and (type(value.get("i")) is int or isinstance(value.get("list"), dict)):
            # Integer/shape metadata only; never include tensor or byte blobs.
            row[key + "_attribute"] = ({"i": value["i"]} if "i" in value else
                                       {"list_i": value["list"].get("i", [])[:16]})
    return row


def _boundaries(graph, enum):
    ops = graph.get("op")
    if not isinstance(ops, list) or not 1 <= len(ops) <= _MAX_OPS:
        raise ValueError("root graph has no supported operation list")
    by_name = {}
    for node in ops:
        name = node.get("name") if isinstance(node, dict) else None
        if not isinstance(name, str) or not name or len(name) > _MAX_TEXT or name in by_name:
            raise ValueError("root operation name is missing, excessive or duplicated")
        by_name[name] = node
    inputs = []
    for node in ops:
        if node.get("type") not in {"Data", "RefData"}:
            continue
        descs = node.get("output_desc", [])
        if len(descs) != 1:
            raise ValueError("root input must have exactly one descriptor")
        inputs.append({"name": node["name"], "type": node["type"],
                       "serialized_index": _attrs(node).get("index", {}).get("i"), **_desc(descs[0], enum)})
    netoutputs = [node for node in ops if node.get("type") == "NetOutput"]
    outputs = []
    if len(netoutputs) == 1:
        boundary_kind = "NETOUTPUT_INPUT_REFERENCES"
        edges = netoutputs[0].get("input", [])
        descs = netoutputs[0].get("input_desc", [])
        if not isinstance(edges, list) or not isinstance(descs, list):
            raise ValueError("NetOutput references/descriptors are not lists")
        edges = [edge for edge in edges if not (isinstance(edge, str) and edge.endswith(":-1"))]
        if len(edges) != len(descs) or len(edges) > 512:
            raise ValueError("NetOutput references and descriptors differ")
        for index, (edge, raw) in enumerate(zip(edges, descs)):
            desc = _desc(raw, enum)
            if desc != _edge(edge, by_name, enum):
                raise ValueError("NetOutput differs from its explicit producer descriptor")
            outputs.append({"output_index": index, "producer_reference": _text(edge), **desc})
    elif not netoutputs:
        boundary_kind = "GRAPHDEF_OUTPUT_REFERENCES"
        edges = graph.get("output", [])
        if not isinstance(edges, list) or len(edges) > 512:
            raise ValueError("GraphDef output references are unsupported")
        for index, edge in enumerate(edges):
            outputs.append({"output_index": index, "producer_reference": _text(edge), **_edge(edge, by_name, enum)})
    else:
        raise ValueError("multiple root NetOutput nodes are ambiguous")
    summary = {"graph_name": _text(graph.get("name", "")), "root_op_count": len(ops),
               "root_op_counts": dict(sorted(Counter(node.get("type") for node in ops).items())),
               "input_type_counts": dict(Counter(row["type"] for row in inputs)),
               "input_count": len(inputs), "output_count": len(outputs), "boundary_source": boundary_kind}
    errors = []
    if len(inputs) != 89 or summary["input_type_counts"] != {"Data": 9, "RefData": 80}:
        errors.append("REQUIRES_9_DATA_AND_80_REFDATA_INPUTS")
    actual_refs = {row["name"]: row for row in inputs if row["type"] == "RefData"}
    if set(actual_refs) != {f"RefData_{index}" for index in range(80)}:
        errors.append("EXACT_REFDATA_NAMES_DO_NOT_MATCH")
    for index in range(80):
        state = actual_refs.get(f"RefData_{index}")
        expected = ([64, 32, 32, 16] if index // 2 % 4 == 3 else
                    [1, 8192, 4] if index % 2 == 0 else [1, 32, 128, 128])
        if state and (state["shape"] != expected or state["schema_dtype_names"] != ["DT_FLOAT16"]):
            errors.append(f"REFDATA_{index}_SHAPE_OR_DTYPE_DIFFERS")
    if any(not row["schema_dtype_names"] for row in inputs):
        errors.append("INPUT_DTYPE_UNRESOLVED")
    if len(outputs) != 81:
        errors.append("REQUIRES_ORIGINAL_81_OUTPUTS")
    else:
        if (outputs[0]["shape"] != [1, 1, 248320] or not outputs[0]["schema_dtype_names"]
                or not set(outputs[0]["schema_dtype_names"]) <= {"DT_FLOAT", "DT_FLOAT32", "DT_FLOAT16"}):
            errors.append("LOGITS_SHAPE_OR_DTYPE_DIFFERS")
        for index, out in enumerate(outputs[1:]):
            state = actual_refs.get(f"RefData_{index}")
            if state and any(out[key] != state[key] for key in ("shape", "raw_dtype", "layout")):
                errors.append(f"OUTPUT_{index + 1}_REFDATA_DESCRIPTOR_DIFFERS")
    summary["original_boundary_check"] = {"status": "MATCHED_COUNTS_SHAPES_DTYPES_ONLY" if not errors else "DIFFERENT",
                                           "errors": errors[:16], "error_count": len(errors)}
    summary["runtime_input_order"] = "UNVERIFIED_SERIALIZED_ORDER_ONLY"
    summary["model_completeness"] = "NOT_ESTABLISHED_BY_BOUNDARY_CHECK"
    return summary, inputs, outputs


def _candidate(document, kind, enum, full):
    if not isinstance(document, dict):
        raise ValueError("protobuf did not decode to an object")
    if kind == "GraphDef":
        graph = document
    else:
        graphs = document.get("graph")
        if not isinstance(graphs, list) or len(graphs) != 1:
            raise ValueError("ModelDef does not contain exactly one root graph")
        graph = graphs[0]
    summary, inputs, outputs = _boundaries(graph, enum)
    constants = [node for node in graph["op"] if node.get("type") == "FileConstant"]
    selected = constants if full else constants[:_EXAMPLES]
    summary["file_constants"] = {"count": len(constants), "shown": len(selected),
                                 "examples": [_file_constant(node, enum) for node in selected],
                                 "payloads": "NOT_OPENED_NOT_HASHED_NOT_CHECKED"}
    # Keep all ordinary input descriptors visible. Do not guess meanings from
    # arg numbers or a matching shape. State descriptors are already checked.
    summary["ordinary_inputs"] = [row for row in inputs if row["type"] == "Data"]
    summary["output_examples"] = outputs[:2] + (outputs[-1:] if len(outputs) > 2 else [])
    if full:
        summary["inputs"] = inputs
        summary["outputs"] = outputs
        summary["operations"] = [{"name": node["name"], "type": node.get("type"),
            "input": [_text(edge) for edge in node.get("input", [])],
            "input_desc": [_metadata_desc(desc, enum) for desc in node.get("input_desc", [])],
            "output_desc": [_metadata_desc(desc, enum) for desc in node.get("output_desc", [])],
            "attribute_names": [_text(key) for key in _attrs(node)]} for node in graph["op"]]
    return summary


def inspect_original_air(air, ddk_root=None, *, full=False, torchair_pb2=None):
    """Return metadata only. No external path in the AIR is opened or tested."""
    result = {"schema_version": 1, "artifact_kind": "qwen35-original-air-inspection", "status": "OPAQUE",
              "read_only": True, "external_weights": "NOT_OPENED_NOT_HASHED_NOT_CHECKED",
              "limits": {"air_bytes": MAX_AIR_BYTES, "output_bytes": MAX_OUTPUT_BYTES, "file_constant_examples": _EXAMPLES},
              "tap_resolution": {"status": "UNRESOLVED", "requested_post_decoder_layer_indices": list(_LAYER_IDS),
                  "reason": "No verified mapping from original decoder statements to actual AIR producer ports; variable/name similarity is not evidence"},
              "candidates": [], "device_execution": "NOT_RUN", "model_compilation": "NOT_RUN",
              "scope": "Graph metadata only; no graph changes, external payload existence, feature semantics, full-model accuracy or state-continuity claim"}
    path, before, library_before = Path(air).expanduser(), None, None
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("AIR must be a regular non-symlink file")
        result["air"] = {"path": str(path.resolve()), "bytes": info.st_size}
        if not 0 < info.st_size <= MAX_AIR_BYTES:
            raise ValueError("AIR is empty or exceeds 64 MiB; payload read refused")
        before = _stamp(path)
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as stream:
            opened = os.fstat(stream.fileno())
            if (not stat.S_ISREG(opened.st_mode) or
                    (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns) != before):
                raise ValueError("AIR changed before bounded read")
            raw = stream.read(MAX_AIR_BYTES + 1)
        if len(raw) != info.st_size or _stamp(path) != before:
            raise ValueError("AIR changed during bounded read")
        head = raw[:64]
        result["air"].update(sha256=hashlib.sha256(raw).hexdigest(),
                             header={"bytes": len(head), "sha256": hashlib.sha256(head).hexdigest(),
                                     "first_16_bytes_hex": head[:16].hex()})
        payloads = [("RAW_GRAPHDEF", raw, "GraphDef"), ("RAW_MODELDEF", raw, "ModelDef")]
        if raw[:4] == b"IMOD":
            try:
                container = read_container(path)
                result["container"] = {"status": "OBSERVED_REFERENCE_IMOD_LAYOUT", **container,
                                       "scope": "Container observation does not identify an editable raw AIR graph"}
                part = next(row for row in container["partitions"] if row["type"] == 0)
                payloads.append(("IMOD_MODEL_PARTITION", memoryview(raw)[part["offset"]:part["offset"] + part["bytes"]], "ModelDef"))
            except (OSError, ValueError) as error:
                result["container"] = {"status": "IMOD_MAGIC_UNSUPPORTED_LAYOUT", "error": str(error)[:256]}
        else:
            result["container"] = {"status": "NO_IMOD_MAGIC", "scope": "Other containers remain possible"}
        if not torchair_pb2 and not ddk_root:
            result["schema"] = {"status": "UNAVAILABLE", "reason": "Declare --torchair-pb2 or --ddk-root/CPRO_DDK_ROOT/DDK_PATH"}
            return result
        library = (Path(torchair_pb2).expanduser() if torchair_pb2 else
                   Path(ddk_root).expanduser() / "tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so")
        try:
            library_before = _stamp(library)
            schema = load_torchair_schema(library) if torchair_pb2 else load_schema(library)
            result["schema"] = {"status": "CANDIDATE_ONLY", **schema.evidence,
                                "selection": "EXPLICIT_TORCHAIR_PB2" if torchair_pb2 else "SELECTED_DDK_HIAI_CANDIDATE"}
        except (OSError, ValueError, TypeError) as error:
            result["schema"] = {"status": "UNAVAILABLE", "error": str(error)[:256]}
            return result
        enum = schema.enums.get(schema.evidence["schema_package"] + ".DataType", {})
        for label, payload, kind in payloads:
            row = {"format": label, "message_type": kind, "status": "UNSUPPORTED"}
            result["candidates"].append(row)
            try:
                summary = _candidate(schema.decode(payload, kind), kind, enum, full)
                row.update(status="STRUCTURE_OBSERVED" if summary["original_boundary_check"]["status"] == "MATCHED_COUNTS_SHAPES_DTYPES_ONLY"
                           else "DECODED_STRUCTURALLY_DIFFERENT", graph=summary)
            except (ValueError, TypeError, KeyError, RecursionError) as error:
                row["error"] = f"{type(error).__name__}: {str(error)[:256]}"
        recognized = [row for row in result["candidates"] if row["status"] == "STRUCTURE_OBSERVED"]
        if len(recognized) == 1:
            result.update(status="STRUCTURE_OBSERVED", observed_format=recognized[0]["format"],
                          editability="NOT_ESTABLISHED_REQUIRES_LOSSLESS_FORMAT_AND_TAP_PROVENANCE")
        elif recognized:
            result["ambiguity"] = "MULTIPLE_CANDIDATE_STRUCTURES; NO_FORMAT_SELECTED"
        if _stamp(library) != library_before:
            raise ValueError("selected schema source changed during inspection")
        if len(json.dumps(result, ensure_ascii=True).encode()) > MAX_OUTPUT_BYTES:
            result["candidates"] = [{"format": row["format"], "status": row["status"],
                "metadata": "OMITTED_OUTPUT_BUDGET"} for row in result["candidates"]]
            result.update(status="OPAQUE", output_limit="Full metadata exceeded 16 MiB; no graph proof retained")
            result.pop("observed_format", None)
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as error:
        result.update(status="ERROR", error=f"{type(error).__name__}: {str(error)[:256]}")
        result["candidates"] = []
        result.pop("observed_format", None)
    finally:
        if before is not None:
            try:
                unchanged = _stamp(path) == before
            except OSError:
                unchanged = False
            result["source_preservation"] = "UNCHANGED_BY_STAT" if unchanged else "CHANGED_DURING_INSPECTION"
            if not unchanged:
                result.update(status="ERROR", error="AIR changed during inspection; decoded metadata discarded", candidates=[])
                result.pop("observed_format", None)
    return result
