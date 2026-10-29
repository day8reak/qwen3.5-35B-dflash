"""Focused, read-only exact-name Reshape metadata and tiny shape constants.

Strict parsing is not proof of dynamic-gear semantics. No GE library executes,
no graph is changed, and FileConstant payload paths are never followed.
"""
from __future__ import annotations

from collections import deque
import hashlib
import math
from pathlib import Path
import re
import struct

from .air_schema import load_torchair_schema
from .omc_wire import _fields, load_schema
from .original_air_inspect import _attrs, _descriptor_metadata, _metadata_desc
from .original_air_roundtrip import _read_air, _stamp


MAX_AIR_BYTES = 64 << 20
_MAX_GRAPHS, _MAX_NODES, _MAX_MATCHES = 256, 32768, 8
_EDGE = re.compile(r"(.+):(0|[1-9][0-9]*|-1)\Z")


def _field(schema, message, name):
    rows = [field for field in schema._messages[message].values() if field.name == name]
    if len(rows) != 1:
        raise ValueError("selected schema lacks unique field " + message + "." + name)
    return rows[0]


def _children(schema, raw, message, name):
    field = _field(schema, message, name)
    return field, [value for number, wire, value in _fields(raw, [2_000_000]) if number == field.number]


def _graphs(schema, raw, decoded, message):
    """Traverse typed message fields, including nested/list graph attributes."""
    budget, count = [2_000_000], [0]
    def walk(data, value, typename, path, depth=0):
        if depth > 64:
            raise ValueError("typed graph traversal depth exceeded")
        if typename == schema.package + ".GraphDef":
            count[0] += 1
            if count[0] > _MAX_GRAPHS:
                raise ValueError("graph count exceeds inspection bound")
            yield value, data, path
        occurrences = {}
        for number, wire, payload in _fields(data, budget):
            field = schema._messages[typename][number]
            index = occurrences.get(number, 0)
            occurrences[number] = index + 1
            if field.kind != 11:
                continue
            child = value[field.name][index] if field.repeated else value[field.name]
            suffix = "/" + field.name + ("/" + str(index) if field.repeated else "")
            yield from walk(payload, child, field.type_name, path + suffix, depth + 1)
    yield from walk(memoryview(raw), decoded, schema.package + "." + message, "$")


def _desc(raw, enum):
    result = _metadata_desc(raw, enum)
    result["metadata"] = _descriptor_metadata(raw)
    shape = result.get("shape")
    result["shape_semantics"] = ("EMPTY_OR_ABSENT_DIMENSIONS_UNRESOLVED; NOT_ASSUMED_SCALAR"
                                 if not shape else "SERIALIZED_DIMENSIONS_ONLY")
    return result


def _tiny(value):
    if not isinstance(value, dict):
        return None
    result = {}
    for key in ("i", "b", "s"):
        item = value.get(key)
        if type(item) in (int, bool):
            result[key] = item
        elif isinstance(item, str) and len(item) <= 256:
            result[key] = item
    listed = value.get("list", {})
    if isinstance(listed, dict):
        for key in ("i", "b", "s"):
            items = listed.get(key)
            if (isinstance(items, list) and len(items) <= 8
                    and all(type(item) in (int, bool) or isinstance(item, str) and len(item) <= 128 for item in items)):
                result["list_" + key] = items
    return result or None


def _node(node, enum):
    attrs = _attrs(node)
    if any(len(key) > 256 for key in attrs):
        raise ValueError("attribute key exceeds focused report text bound")
    tiny = {key: selected for key, value in list(attrs.items())[:64] if (selected := _tiny(value)) is not None}
    references = _refs(node)
    result = {"name": node["name"], "type": node.get("type"), "input": [row[1] for row in references[:16]],
              "input_representation": "input" if node.get("input") else "src_name/src_index",
              "input_omitted": max(0, len(references) - 16), "attributes": tiny,
              "input_desc": [_desc(value, enum) for value in node.get("input_desc", [])[:8]],
              "output_desc": [_desc(value, enum) for value in node.get("output_desc", [])[:8]]}
    if node.get("type") == "FileConstant":
        path = attrs.get("file_path", {}).get("s")
        result.update(file_path=path[:512] if isinstance(path, str) else None,
                      payload_access="NOT_OPENED_NOT_STAT_ED_NOT_HASHED")
    return result


def _refs(node):
    direct = node.get("input", [])
    if direct:
        return [(index, edge, "input") for index, edge in enumerate(direct)]
    names, indices = node.get("src_name", []), node.get("src_index", [])
    if isinstance(names, list) and isinstance(indices, list) and len(names) == len(indices):
        return [(index, name + ":" + str(port), "src_name/src_index") for index, (name, port) in enumerate(zip(names, indices))
                if isinstance(name, str) and type(port) is int]
    return []


def _edge(reference, by_name):
    match = _EDGE.fullmatch(reference) if isinstance(reference, str) else None
    if match is None or match[2] == "-1":
        return None
    name, port = match[1], int(match[2])
    node = by_name.get(name)
    return (node, port) if node is not None and port < len(node.get("output_desc", [])) else None


def _edge_report(node, index, by_name, enum):
    selected = [row for row in _refs(node) if row[0] == index]
    if len(selected) != 1:
        return {"status": "UNRESOLVED", "reason": "no unique serialized input reference", "input_index": index}
    _, reference, origin = selected[0]
    result = {"reference": reference, "input_index": index, "representation": origin}
    descs = node.get("input_desc", [])
    result["consumer_descriptor"] = _desc(descs[index], enum) if index < len(descs) else {"status": "UNAVAILABLE"}
    resolved = _edge(reference, by_name)
    if resolved is None:
        result.update(status="UNRESOLVED", reason="exact same-graph producer port unavailable")
    else:
        producer, port = resolved
        result.update(status="RESOLVED_EXACT_SAME_GRAPH_PORT", producer=_node(producer, enum), producer_port=port,
                      producer_descriptor=_desc(producer["output_desc"][port], enum))
    names, ports = node.get("src_name", []), node.get("src_index", [])
    if origin == "input" and index < len(names) and index < len(ports):
        alternative = str(names[index]) + ":" + str(ports[index])
        result["alternate_reference"] = alternative
        result["reference_representations_agree"] = alternative == reference
    return result


def _shape_constant(schema, shape_edge, by_name, raw_nodes, enum):
    result = {"status": "UNRESOLVED", "scope": "Tiny declared signed integer tensor bytes only; no dynamic-shape proof"}
    resolved = _edge(shape_edge.get("reference"), by_name)
    if resolved is None:
        result["reason"] = "shape producer unresolved"
        return result
    node, port = resolved
    if node.get("type") != "Const" or port != 0 or len(node.get("output_desc", [])) != 1:
        result.update(reason="shape producer is not a unique embedded Const output0", producer_type=node.get("type"))
        return result
    if shape_edge.get("reference_representations_agree") is False:
        result["reason"] = "input and src_name/src_index disagree"
        return result
    try:
        attrs = _attrs(node)
        tensor = attrs.get("value", {}).get("t", {})
        descriptor = tensor.get("desc", {})
        desc = _desc(descriptor, enum)
        result["descriptor"] = desc
        shape = descriptor.get("shape", {}).get("dim")
        names = set(desc.get("schema_dtype_names", []))
        encoding = ("i" if names and names <= {"DT_INT32", "INT32"} else
                    "q" if names and names <= {"DT_INT64", "INT64"} else None)
        if encoding is None or type(descriptor.get("dtype")) is not int:
            raise ValueError("shape Const must explicitly declare selected-schema INT32 or INT64")
        if (not isinstance(shape, list) or not 1 <= len(shape) <= 4
                or any(type(dim) is not int or dim <= 0 for dim in shape) or not 1 <= math.prod(shape) <= 8):
            raise ValueError("shape Const requires explicit nonempty dimensions and 1..8 elements; shape[] is unresolved")
        output = node["output_desc"][0]
        if "dtype" in output and output["dtype"] != descriptor["dtype"]:
            raise ValueError("shape Const output/tensor dtype disagree")
        output_dims = output.get("shape", {}).get("dim")
        if output_dims and output_dims != shape:
            raise ValueError("shape Const output/tensor dimensions disagree")
        message = schema.package + ".OpDef"
        attr_field, entries = _children(schema, raw_nodes[node["name"]], message, "attr")
        selected = []
        for entry in entries:
            decoded = schema.decode(entry, attr_field.type_name)
            if decoded.get("key") == "value":
                value_field, values = _children(schema, entry, attr_field.type_name, "value")
                if len(values) != 1:
                    raise ValueError("Const value attribute is ambiguous")
                tensor_field, tensors = _children(schema, values[0], value_field.type_name, "t")
                if len(tensors) != 1:
                    raise ValueError("Const tensor attribute is ambiguous")
                _, selected = _children(schema, tensors[0], tensor_field.type_name, "data")
        if len(selected) != 1 or not 0 < len(selected[0]) <= 64:
            raise ValueError("shape Const embedded data missing or above 64-byte bound")
        data = selected[0]
        count = math.prod(shape)
        if len(data) != count * struct.calcsize("<" + encoding):
            raise ValueError("shape Const byte count disagrees with declared dtype/dimensions")
        receipt = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        if tensor.get("data") != receipt:
            raise ValueError("raw tiny-data identity differs from strict-decoded tensor hash")
        result.update(status="DECODED_TINY_INTEGER_TENSOR", decoded_values=list(struct.unpack("<" + encoding * count, data)),
                      data=receipt, encoding="SIGNED_INT32_LE" if encoding == "i" else "SIGNED_INT64_LE",
                      output_dimensions_status="EXPLICIT_AGREEMENT" if output_dims else "UNRESOLVED; NOT_FILLED")
    except (ValueError, KeyError, TypeError, struct.error) as error:
        result["reason"] = str(error)[:256]
    return result


def _match(node, graph, graph_path, graph_raw, schema, enum):
    nodes = graph["op"]
    by_name = {row["name"]: row for row in nodes}
    op_field, raw_ops = _children(schema, graph_raw, schema.package + ".GraphDef", "op")
    if len(raw_ops) != len(nodes):
        raise ValueError("raw/decoded operation count differs")
    raw_nodes = {row["name"]: raw for row, raw in zip(nodes, raw_ops)}
    outgoing, edge_count = {}, 0
    for consumer in nodes:
        for index, reference, origin in _refs(consumer):
            resolved = _edge(reference, by_name)
            if resolved:
                outgoing.setdefault(resolved[0]["name"], []).append((consumer["name"], index, reference))
                edge_count += 1
                if edge_count > 262144:
                    raise ValueError("same-graph data-edge count exceeds bound")
    seen, queue, relations, truncated = {node["name"]}, deque([(node["name"], 0)]), [], False
    while queue:
        name, depth = queue.popleft()
        if depth == 2:
            continue
        neighbors = []
        for index, reference, origin in _refs(by_name[name]):
            resolved = _edge(reference, by_name)
            if resolved:
                neighbors.append(resolved[0]["name"])
                relations.append({"producer": reference, "consumer": name, "input_index": index})
        for consumer, index, reference in outgoing.get(name, []):
            neighbors.append(consumer)
            relations.append({"producer": reference, "consumer": consumer, "input_index": index})
        for neighbor in neighbors:
            if neighbor not in seen:
                if len(seen) >= 32:
                    truncated = True
                    continue
                seen.add(neighbor)
                queue.append((neighbor, depth + 1))
    shape_edge = _edge_report(node, 1, by_name, enum)
    weights = []
    for name in sorted(seen):
        for index, reference, origin in _refs(by_name[name]):
            resolved = _edge(reference, by_name)
            if resolved and resolved[0].get("type") == "FileConstant":
                weights.append({"consumer": name, "input_index": index, "reference": reference,
                    **_node(resolved[0], enum), "relation": "FILECONSTANT_OPERAND_OF_TWO_HOP_NEIGHBOR"})
    constant = (_shape_constant(schema, shape_edge, by_name, raw_nodes, enum) if node.get("type") == "Reshape" else
                {"status": "UNRESOLVED", "reason": "exact node name matched, but actual operator type is not Reshape"})
    return {"graph_path": graph_path, "graph_name": graph.get("name"), "node_name": node["name"], "node_type": node.get("type"),
        "node": _node(node, enum), "data_edge": _edge_report(node, 0, by_name, enum), "shape_edge": shape_edge,
        "shape_constant": constant, "file_constant_operands": weights[:16], "file_constant_operands_omitted": max(0, len(weights) - 16),
        "neighborhood": {"nodes": [_node(row, enum) for row in nodes if row["name"] in seen],
            "edges": list({(row["producer"], row["consumer"], row["input_index"]): row for row in relations}.values())[:96],
            "truncated": truncated or len(relations) > 96, "max_hops": 2, "max_nodes": 32}}


def inspect_target_reshape_metadata(air, *, torchair_pb2=None, ddk_root=None, node_name="Reshape"):
    result = {"schema_version": 1, "artifact_kind": "qwen35-target-reshape-metadata", "status": "UNKNOWN",
        "protobuf_decode": "NOT_RUN", "matches": [], "attempts": [], "model_compilation": "NOT_RUN",
        "device_execution": "NOT_RUN", "dynamic_gear_semantics": "UNVERIFIED", "graph_changes": "NONE",
        "external_weights": "NOT_OPENED_NOT_STAT_ED_NOT_HASHED"}
    try:
        if not isinstance(node_name, str) or not node_name or len(node_name) > 256:
            raise ValueError("exact node_name must be nonempty text at most 256 characters")
        raw, identity = _read_air(air)
        result["input"] = {key: identity[key] for key in ("path", "bytes", "sha256")}
        if torchair_pb2:
            schema = load_torchair_schema(torchair_pb2)
            selection = "EXPLICIT_TORCHAIR_PB2"
        elif ddk_root:
            schema = load_schema(Path(ddk_root).expanduser() / "tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so")
            selection = "EXPLICIT_DDK_HIAI_SCHEMA"
        else:
            raise ValueError("explicit torchair_pb2 or ddk_root is required")
        result["schema"] = {"selection": selection, **schema.evidence}
        candidates = []
        for kind in ("ModelDef", "GraphDef"):
            attempt = {"format": kind, "protobuf_decode": "NOT_RUN", "graph_metadata": "NOT_RUN"}
            try:
                decoded = schema.decode(raw, kind)
                attempt["protobuf_decode"] = "PASSED"
                graphs = list(_graphs(schema, raw, decoded, kind))
                if not graphs:
                    raise ValueError("no typed GraphDef present")
                for graph, _, path in graphs:
                    ops = graph.get("op")
                    if not isinstance(ops, list) or not 1 <= len(ops) <= _MAX_NODES:
                        raise ValueError("graph lacks a bounded nonempty operation list")
                    names = [row.get("name") for row in ops]
                    if any(not isinstance(name, str) or not name or len(name) > 512 for name in names) or len(set(names)) != len(names):
                        raise ValueError("graph operation names are missing or duplicated")
                    for node in ops:
                        if (not isinstance(node.get("type"), str) or not 0 < len(node["type"]) <= 256
                                or any(not isinstance(edge, str) or len(edge) > 1024 for edge in node.get("input", []))
                                or any(not isinstance(name, str) or len(name) > 512 for name in node.get("src_name", []))):
                            raise ValueError("operation type or input text exceeds focused metadata bounds")
                candidates.append((kind, graphs))
                attempt.update(status="STRICT_DECODED_GRAPH_METADATA", graph_metadata="COMPLETE")
            except (ValueError, TypeError, KeyError, RecursionError) as error:
                attempt.update(status="UNSUPPORTED", error=str(error)[:256])
                attempt["graph_metadata" if attempt["protobuf_decode"] == "PASSED" else "protobuf_decode"] = "FAILED"
            result["attempts"].append(attempt)
        if len(candidates) != 1:
            any_decoded = any(row["protobuf_decode"] == "PASSED" for row in result["attempts"])
            result.update(status=("DECODED_METADATA_UNSUPPORTED" if any_decoded else "OPAQUE_SCHEMA_MISMATCH") if not candidates else "AMBIGUOUS_ROOT_FORMAT",
                          protobuf_decode=("PASSED" if any_decoded else "FAILED") if not candidates else "AMBIGUOUS")
            return result
        kind, graphs = candidates[0]
        dtype_field = _field(schema, schema.package + ".TensorDescriptor", "dtype")
        enum = schema.enums.get(dtype_field.type_name, {})
        matches = [(node, graph, path, data) for graph, data, path in graphs for node in graph["op"] if node["name"] == node_name]
        result.update(protobuf_decode="PASSED", observed_format=kind, graph_count=len(graphs), matching_node_count=len(matches),
            matches_truncated=len(matches) > _MAX_MATCHES,
            matches=[_match(node, graph, path, data, schema, enum) for node, graph, path, data in matches[:_MAX_MATCHES]],
            status="INSPECTED" if matches else "NODE_NOT_FOUND")
        if _stamp(Path(identity["path"])) != tuple(identity["stat"]):
            raise ValueError("input changed during focused inspection")
        result["source_preservation"] = "UNCHANGED_BY_STAT"
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as error:
        result.update(status="UNKNOWN", error=f"{type(error).__name__}: {error}"[:512])
    return result
