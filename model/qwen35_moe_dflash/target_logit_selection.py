"""Read-only source AIR evidence for the original target's selected logit row.

Only a saved successful prefill control and its exact graph dependency are
compared.  This never changes output_pos, runs a model, reads weights, or
attributes the observed numerical output to an out-of-range access.
"""
from __future__ import annotations

from collections import deque
import hashlib
import json
import math
from pathlib import Path
import struct

from .artifacts import output_path, save_json
from .original_air_inspect import _attrs
from .target_compile_failure import _read_small, _safe
from .target_prefill_output_inventory import _bound_execution, _json
from .target_prefill_smoke_plan import _source
from .target_reshape_metadata import _children
from .target_state_lineage import load_bound_source_air, _graph, _inputs, graph_edge_policy


_PROFILE = "cpro-40block-prefill-64"
_CONTROL = "arg1238_1"
_MAX_NODES, _MAX_EDGES = 32768, 262144
_PRESERVE = {"Identity", "Squeeze", "Unsqueeze", "Cast"}


def _edge(text, by_name):
    if not isinstance(text, str) or len(text) > 1152 or ":" not in text:
        raise ValueError("expected an explicit producer port")
    name, port = text.rsplit(":", 1)
    if not port.isdecimal() or len(port) > 4:
        raise ValueError("invalid data port")
    node, port = by_name.get(name), int(port)
    if node is None or port >= len(node.get("output_desc", [])):
        raise ValueError("producer port is missing")
    return node, port


def _integer_kind(desc, enum):
    raw = desc.get("dtype") if isinstance(desc, dict) else None
    names = {name for name, value in enum.items() if type(raw) is int and value == raw}
    if names and names <= {"DT_INT32", "INT32"}:
        return "i"
    if names and names <= {"DT_INT64", "INT64"}:
        return "q"
    raise ValueError("selected-schema signed INT32/INT64 descriptor is required")


def _dimensions(desc):
    value = desc.get("shape", {}).get("dim") if isinstance(desc, dict) else None
    if (not isinstance(value, list) or not 1 <= len(value) <= 16
            or any(type(dim) is not int or dim <= 0 for dim in value)):
        return None
    return value


def _gather_ports(node):
    attrs = _attrs(node)
    keys = attrs.get("_input_name_key", {}).get("list", {}).get("s")
    values = attrs.get("_input_name_value", {}).get("list", {}).get("i")
    if (not isinstance(keys, list) or not isinstance(values, list) or len(keys) != len(values)
            or len(keys) != 3 or len(set(keys)) != 3 or set(values) != {0, 1, 2}
            or any(not isinstance(key, str) or type(value) is not int for key, value in zip(keys, values))):
        raise ValueError("GatherV2 requires an explicit unique serialized x/indices/axis port-name map")
    mapping = dict(zip(keys, values))
    params = "x" if "x" in mapping else "params"
    if set(mapping) != {params, "indices", "axis"}:
        raise ValueError("GatherV2 serialized input names do not identify params,indices,axis")
    return {"params": mapping[params], "indices": mapping["indices"], "axis": mapping["axis"]}


def _compact(node):
    def descriptor(raw):
        attrs = _attrs(raw) if isinstance(raw, dict) else {}
        meta = attrs.get("_meta", {}).get("s")
        fx = attrs.get("_fx_tensor_name", {}).get("s")
        return {"shape": raw.get("shape"), "dtype": raw.get("dtype"), "layout": raw.get("layout"),
                "declared_meta": meta[:512] if isinstance(meta, str) else None,
                "declared_meta_truncated": isinstance(meta, str) and len(meta) > 512,
                "fx_tensor_name": fx[:512] if isinstance(fx, str) else None,
                "fx_tensor_name_truncated": isinstance(fx, str) and len(fx) > 512}
    return {"name": node["name"], "type": node["type"], "input": _inputs(node)[:16],
        "input_desc": [descriptor(row) for row in node.get("input_desc", [])[:3]],
        "output_desc": [descriptor(row) for row in node.get("output_desc", [])[:2]]}


def _constant(schema, node, raw_node, enum):
    """Decode only one embedded signed integer; never follow a FileConstant."""
    if node.get("type") != "Const" or len(node.get("output_desc", [])) != 1:
        raise ValueError("axis must be one embedded Const output0; external payloads are not opened")
    tensor = _attrs(node).get("value", {}).get("t", {})
    desc = tensor.get("desc", {})
    encoding = _integer_kind(desc, enum)
    output = node["output_desc"][0]
    if _integer_kind(output, enum) != encoding:
        raise ValueError("axis Const tensor and output dtypes differ")
    for candidate in (desc, output):
        dimensions = candidate.get("shape", {}).get("dim")
        if dimensions is not None and (not isinstance(dimensions, list) or len(dimensions) > 4
                or any(type(dim) is not int or dim <= 0 for dim in dimensions)
                or math.prod(dimensions) != 1):
            raise ValueError("axis Const must contain exactly one integer")
    attr_field, entries = _children(schema, raw_node, schema.package + ".OpDef", "attr")
    found = []
    for entry in entries:
        decoded = schema.decode(entry, attr_field.type_name)
        if decoded.get("key") != "value":
            continue
        value_field, values = _children(schema, entry, attr_field.type_name, "value")
        if len(values) != 1:
            raise ValueError("axis Const value is ambiguous")
        tensor_field, tensors = _children(schema, values[0], value_field.type_name, "t")
        if len(tensors) != 1:
            raise ValueError("axis Const tensor is ambiguous")
        _, payloads = _children(schema, tensors[0], tensor_field.type_name, "data")
        found.extend(payloads)
    if len(found) != 1 or len(found[0]) != struct.calcsize("<" + encoding):
        raise ValueError("axis Const requires exactly4 or8 signed integer bytes")
    raw = bytes(found[0])
    receipt = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    if tensor.get("data") != receipt:
        raise ValueError("embedded axis bytes differ from strict schema decode")
    return {"value": struct.unpack("<" + encoding, raw)[0], **receipt,
        "encoding": "SIGNED_INT32_LE" if encoding == "i" else "SIGNED_INT64_LE",
        "scalar_basis": "EXACT_SINGLE_INTEGER_EMBEDDED_BYTES_AND_DESCRIPTOR_AGREEMENT"}


def _index_path(reference, by_name, enum, value):
    path, seen = [], set()
    for _ in range(32):
        node, port = _edge(reference, by_name)
        key = (node["name"], port)
        if key in seen:
            raise ValueError("cycle in index dependency")
        seen.add(key)
        if port != 0 or len(node.get("output_desc", [])) != 1:
            raise ValueError("index passthrough must use a unique output0")
        encoding = _integer_kind(node["output_desc"][0], enum)
        width = 32 if encoding == "i" else 64
        if not -(1 << (width - 1)) <= value < 1 << (width - 1):
            raise ValueError("observed control is not preserved by integer conversion")
        path.append({"node": node["name"], "type": node["type"], "port": port, "integer_bits": width})
        if node["name"] == _CONTROL:
            if node["type"] != "Data" or encoding != "i" or _dimensions(node["output_desc"][0]) != [1]:
                raise ValueError("source control must be exact INT32 Data arg1238_1[1]")
            return path
        if node["type"] not in _PRESERVE:
            raise ValueError("index path contains an unproved value transform: " + node["type"])
        inputs = _inputs(node)
        if len(inputs) != 1:
            raise ValueError("index passthrough must have exactly one data operand")
        parent, parent_port = _edge(inputs[0], by_name)
        parent_kind = _integer_kind(parent["output_desc"][parent_port], enum)
        if node["type"] != "Cast" and parent_kind != encoding:
            raise ValueError("non-Cast passthrough changes integer dtype")
        reference = inputs[0]
    raise ValueError("index dependency exceeds32-node bound")


def analyze_logit_selection(graph, raw_graph, schema, selected_index):
    """Trace a unique GatherV2 on the serialized logits0 dependency cone."""
    if type(selected_index) is not int or not -(1 << 31) <= selected_index < 1 << 31:
        raise ValueError("observed control must be one signed INT32 value")
    _, by_name, net = _graph(graph)
    edges = _inputs(net)
    if not edges:
        raise ValueError("NetOutput has no data outputs")
    _, raw_ops = _children(schema, raw_graph, schema.package + ".GraphDef", "op")
    if len(raw_ops) != len(graph["op"]):
        raise ValueError("raw and strict-decoded graph operation counts differ")
    raw_nodes = {node["name"]: raw for node, raw in zip(graph["op"], raw_ops)}
    dtype_field = next((field for field in schema._messages[schema.package + ".TensorDescriptor"].values()
                        if field.name == "dtype"), None)
    if dtype_field is None:
        raise ValueError("selected schema lacks TensorDescriptor.dtype")
    enum = schema.enums.get(dtype_field.type_name, {})
    pending, visited, gathers, parents, edge_count = deque([(edges[0], None)]), set(), [], {}, 0
    while pending:
        edge, parent = pending.popleft()
        node, port = _edge(edge, by_name)
        key = (node["name"], port)
        if key in visited:
            continue
        visited.add(key)
        parents[key] = parent
        if len(visited) > _MAX_NODES:
            raise ValueError("logit dependency node budget exceeded")
        inputs = _inputs(node)
        edge_count += len(inputs)
        if edge_count > _MAX_EDGES:
            raise ValueError("logit dependency edge budget exceeded")
        if node["type"] == "GatherV2":
            gathers.append((node, port))
        # Empty optional operands preserve named-port positions but carry no
        # dependency. Keep their slots in _inputs; skip only this traversal.
        pending.extend((reference, key) for reference in inputs if reference)
    def dependency_path(key):
        path = []
        while key is not None and len(path) < 32:
            name, port = key
            path.append({"output_port": port, "node": _compact(by_name[name])})
            key = parents[key]
        return {"direction": "LOGITS_PRODUCER_TOWARD_GATHER", "nodes": list(reversed(path)),
                "truncated": key is not None, "selection": "ONE_EXPLICIT_SHORTEST_DATA_DEPENDENCY_PATH",
                "scope": "Literal serialized edges and descriptors only; no layout conversion or operation semantics are inferred"}
    candidates, nearby = [], []
    for gather, gather_port in gathers:
        inputs = _inputs(gather)
        row = {"node": _compact(gather), "status": "UNVERIFIED"}
        try:
            if len(inputs) != 3:
                raise ValueError("GatherV2 must have exactly params,indices,axis operands")
            ports = _gather_ports(gather)
            row["serialized_input_ports"] = ports
            row["index_path"] = _index_path(inputs[ports["indices"]], by_name, enum, selected_index)
            row["index_source"] = _CONTROL
            row["logits_dependency_path"] = dependency_path((gather["name"], gather_port))
            candidates.append(row)
            axis_node, axis_port = _edge(inputs[ports["axis"]], by_name)
            if axis_port != 0:
                raise ValueError("axis must name Const output0")
            row["axis"] = _constant(schema, axis_node, raw_nodes[axis_node["name"]], enum)
            param, port = _edge(inputs[ports["params"]], by_name)
            row["params_producer"] = _compact(param)
            row["params_output_port"] = port
            row["params_consumer_descriptor_index"] = ports["params"]
            consumer = gather.get("input_desc", [])
            consumer_shape = _dimensions(consumer[ports["params"]]) if len(consumer) > ports["params"] else None
            producer_shape = _dimensions(param["output_desc"][port])
            if consumer_shape and producer_shape and consumer_shape != producer_shape:
                raise ValueError("GatherV2 params consumer/producer dimensions disagree")
            shape = consumer_shape or producer_shape
            if shape is None:
                raise ValueError("GatherV2 params dimensions are absent/dynamic; profile dimensions are not substituted")
            axis = row["axis"]["value"]
            if not -len(shape) <= axis < len(shape):
                raise ValueError("GatherV2 axis is outside the serialized params rank")
            axis %= len(shape)
            extent = shape[axis]
            row.update(status="SOURCE_CONTROL_GATHER_DEPENDENCY_OBSERVED", params_shape=shape,
                params_shape_evidence=("BOTH_SERIALIZED_CONSUMER_AND_PRODUCER" if consumer_shape and producer_shape else
                    "SERIALIZED_CONSUMER_ONLY" if consumer_shape else "SERIALIZED_PRODUCER_ONLY"),
                normalized_axis=axis, serialized_axis_extent=extent, selected_index=selected_index,
                nonnegative_valid_index_range=[0, extent - 1],
                index_range=("OUTSIDE_SERIALIZED_AXIS_EXTENT" if selected_index >= extent else
                             "WITHIN_NONNEGATIVE_SERIALIZED_AXIS_EXTENT" if selected_index >= 0 else
                             "NEGATIVE_INDEX_SEMANTICS_UNVERIFIED"))
        except (ValueError, KeyError, TypeError, struct.error) as error:
            row["reason"] = str(error)[:512]
        nearby.append(row)
    result = {"status": "LOGIT_SELECTION_UNVERIFIED", "logits_output_index": 0,
        "graph_edge_policy": graph_edge_policy(),
        "logits_reference": edges[0], "observed_control_value": selected_index,
        "logit_ancestor_count": len(visited), "ancestor_gather_count": len(gathers),
        "control_gather_candidate_count": len(candidates), "candidates": candidates[:8],
        "other_gathers": [row for row in nearby if row not in candidates][:8],
        "candidate_rows_truncated": len(candidates) > 8 or len(nearby) - len(candidates) > 8,
        "scope": "Serialized source graph dependency and integer bounds only; no runtime out-of-range behavior, numerical root cause or correction acceptance is inferred."}
    if len(candidates) == 1 and candidates[0]["status"] == "SOURCE_CONTROL_GATHER_DEPENDENCY_OBSERVED":
        bounds = candidates[0]["index_range"]
        result["status"] = ("LOGIT_INDEX_OUT_OF_RANGE_OBSERVED" if bounds == "OUTSIDE_SERIALIZED_AXIS_EXTENT" else
            "LOGIT_INDEX_WITHIN_SERIALIZED_EXTENT" if bounds == "WITHIN_NONNEGATIVE_SERIALIZED_AXIS_EXTENT" else
            "LOGIT_SELECTION_OBSERVED_RANGE_UNVERIFIED")
    elif len(candidates) == 1:
        result["status"] = "LOGIT_SELECTION_OBSERVED_RANGE_UNVERIFIED"
    return result


def _observed_control(execution, job, bindings):
    execution, _ = _safe(execution, directory=True)
    _, _, plan = _bound_execution(execution, bindings)
    if (plan.get("compilation_report_sha256") != job["compilation_report_sha256"]
            or plan.get("model_sha256") != job["model_sha256"]):
        raise ValueError("successful prefill execution differs from this bound job")
    proof = _json(execution / "plan.json", bindings)
    rows = [row for row in proof.get("inputs", []) if _CONTROL in row.get("compiled_candidates", [])]
    if (len(rows) != 1 or rows[0].get("compiled_candidates") != [_CONTROL]
            or rows[0].get("dtype") != "INT32" or rows[0].get("bytes") != 4
            or rows[0].get("source_filename") != "output_pos.bin"
            or rows[0].get("semantic_slot_identity") != "UNIQUE_TYPED_ND_VIEW"):
        raise ValueError("saved prefill plan does not uniquely bind arg1238_1 to original output_pos.bin")
    row = rows[0]
    matches = [record for record in proof.get("source_inputs", []) if record.get("path") == row.get("source")]
    if len(matches) != 1 or _source(row["source"], 4) != matches[0]:
        raise ValueError("original output_pos bytes/identity differ from successful prefill source binding")
    raw, identity = _read_small(row["source"], 4)
    original_identity = matches[0].get("identity", {})
    expected_stat = [original_identity.get(key) for key in ("device", "inode", "bytes", "mtime_ns", "ctime_ns")]
    if identity["sha256"] != matches[0]["sha256"] or identity["stat"] != expected_stat:
        raise ValueError("original output_pos changed after source hash validation")
    bindings.append(identity)
    return {**identity, "value": struct.unpack("<i", raw)[0], "dtype": "SIGNED_INT32_LE",
        "source_input_name": _CONTROL, "semantic_source": "SAVED_SUCCESSFUL_PREFILL_PLAN_UNIQUE_TYPED_CONTROL_FILE"}


def inspect_target_logit_selection(job_dir, execution_dir, output_dir, *,
                                   source_torchair_pb2=None, source_export_receipt=None):
    out = output_path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    bindings = []
    result = {"schema_version": 1, "artifact_kind": "qwen35-target-logit-selection-audit",
        "status": "FAILED", "output": str(out), "device_execution": "NOT_RUN", "model_compilation": "NOT_RUN",
        "input_mutation": "NONE", "weight_payload_access": "NONE", "numerical_validation": "NOT_RUN",
        "source_bindings": bindings}
    result["graph_edge_policy"] = graph_edge_policy()
    result["source_schema_request"] = {"source_torchair_pb2": str(source_torchair_pb2) if source_torchair_pb2 is not None else None,
        "source_export_receipt": str(source_export_receipt) if source_export_receipt is not None else None,
        "pb2_execution": "NOT_EXECUTED", "schema_fallback": "NOT_PERMITTED"}
    try:
        result["stage"] = "read-bound-source-air"
        options = {}
        if source_torchair_pb2 is not None:
            options["source_torchair_pb2"] = source_torchair_pb2
        if source_export_receipt is not None:
            options["source_export_receipt"] = source_export_receipt
        job, schema, raw, document, kind = load_bound_source_air(job_dir, _PROFILE, bindings, **options)
        result["job"] = job
        result["stage"] = "read-observed-control"
        control = _observed_control(execution_dir, job, bindings)
        result["control"] = control
        if kind == "ModelDef":
            _, graph_rows = _children(schema, raw, schema.package + ".ModelDef", "graph")
            if len(graph_rows) != 1 or len(document.get("graph", [])) != 1:
                raise ValueError("source AIR requires one explicit root graph")
            graph, graph_raw = document["graph"][0], graph_rows[0]
        elif kind == "GraphDef":
            graph, graph_raw = document, raw
        else:
            raise ValueError("unsupported bound source AIR format")
        result["stage"] = "analyze-source-gather"
        result["selection"] = analyze_logit_selection(graph, graph_raw, schema, control["value"])
        result["status"] = result["selection"]["status"]
        for identity in bindings:
            if _safe(identity["path"])[1] != identity["stat"]:
                raise ValueError("source evidence identity changed during logit audit")
        result["source_preservation"] = "UNCHANGED_BOUND_METADATA_AIR_PRIMARY_OMC_SCHEMA_AND_ORIGINAL_CONTROL"
        result["stage"] = "complete"
        result["next"] = ("Review the exact graph/control mismatch before preparing a separate corrected-input experiment; no source control was changed"
            if result["status"] == "LOGIT_INDEX_OUT_OF_RANGE_OBSERVED" else
            "Review exact index path, axis constant and serialized params dimensions; unresolved metadata is not a numerical or semantic pass")
    except (OSError, ValueError, TypeError, KeyError, StopIteration, RecursionError, struct.error) as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}"[:1024])
    if len(json.dumps(result, ensure_ascii=False).encode()) > 2 << 20:
        raise ValueError("logit selection report exceeds2MiB budget")
    save_json(out / "report.json", result)
    return result
