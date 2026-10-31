"""Append a GE-built feature branch while preserving original protobuf bytes.

Field numbers come from the selected schema. Only Cast input references and
the original NetOutput's new tail port are written; no parameter data is read.
"""
from __future__ import annotations

import hashlib

from qwen35_moe_dflash.toolchain.omc_wire import _varint
from qwen35_moe_dflash.target.original_air_roundtrip import MAX_AIR_BYTES, _canonical, _decode


PREFIX = "__dflash_features/"


def _v(value):
    if type(value) is not int or value < 0:
        raise ValueError("nonnegative integer wire value required")
    result = bytearray()
    while value >= 128:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def _blob(number, value):
    if isinstance(value, str):
        value = value.encode("utf-8")
    return _v(number << 3 | 2) + _v(len(value)) + value


def _parts(raw):
    """Keep exact encoded spans, including original tag/length encodings."""
    offset, count = 0, 0
    while offset < len(raw):
        count += 1
        if count > 2_000_000:
            raise ValueError("wire field budget exceeded")
        start = offset
        tag, offset = _varint(raw, offset)
        number, wire = tag >> 3, tag & 7
        if not 0 < number < (1 << 29):
            raise ValueError("invalid wire field number")
        payload = None
        if wire == 0:
            _, offset = _varint(raw, offset)
        elif wire in (1, 5):
            offset += 8 if wire == 1 else 4
        elif wire == 2:
            size, offset = _varint(raw, offset)
            end = offset + size
            if end > len(raw):
                raise ValueError("wire payload exceeds file")
            payload, offset = raw[offset:end], end
        else:
            raise ValueError("unsupported wire type")
        if offset > len(raw):
            raise ValueError("wire field exceeds file")
        yield number, wire, raw[start:offset], payload


def _field(schema, message, name, kind, repeated):
    rows = [f for f in schema._messages[schema.package + "." + message].values() if f.name == name]
    if len(rows) != 1 or rows[0].kind != kind or rows[0].repeated is not repeated:
        raise ValueError("selected schema lacks required field: " + message + "." + name)
    return rows[0].number


def _graph_raw(raw, schema, kind):
    if kind == "GraphDef":
        return raw, None
    number = _field(schema, "ModelDef", "graph", 11, True)
    rows = [p for p in _parts(raw) if p[0] == number]
    if len(rows) != 1 or rows[0][1] != 2:
        raise ValueError("one explicit ModelDef graph required")
    return rows[0][3], number


def _node_bytes(graph_raw, schema):
    number = _field(schema, "GraphDef", "op", 11, True)
    result = []
    for field, wire, encoded, payload in _parts(graph_raw):
        if field == number:
            if wire != 2:
                raise ValueError("operation is not a message")
            node = schema.decode(payload, "OpDef")
            result.append((node, payload, encoded))
    return result, number


def _new_id(payload, schema, identifier):
    fields = [f for f in schema._messages[schema.package + ".OpDef"].values() if f.name == "id"]
    if not fields:
        return payload
    if len(fields) != 1 or fields[0].kind not in (3, 4) or fields[0].repeated:
        raise ValueError("unsupported operation id schema")
    field = fields[0].number
    encoded_id = _v(field << 3) + _v(identifier)
    parts, seen = [], False
    for number, wire, encoded, _ in _parts(payload):
        if number == field:
            if seen or wire != 0:
                raise ValueError("ambiguous template operation id")
            encoded, seen = encoded_id, True
        parts.append(encoded)
    if not seen:
        parts.append(encoded_id)
    return b"".join(parts)


def append_feature_branch_wire(original_raw, branch_raw, schema, taps):
    """Return patched bytes and an exact original-operation preservation receipt.

    The caller must independently validate source/tap provenance and the full
    metadata append contract. This function alone proves neither one.
    """
    if any(not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_AIR_BYTES for raw in (original_raw, branch_raw)):
        raise ValueError("bounded raw AIR bytes required")
    old_kind, old_model, old_graph = _decode(schema, original_raw)
    branch_kind, _, branch_graph = _decode(schema, branch_raw)
    old_graph_raw, model_graph_field = _graph_raw(original_raw, schema, old_kind)
    branch_graph_raw, _ = _graph_raw(branch_raw, schema, branch_kind)
    old_nodes, op_field = _node_bytes(old_graph_raw, schema)
    template_nodes, _ = _node_bytes(branch_graph_raw, schema)
    if len(taps) != 8:
        raise ValueError("exactly eight bound taps required")
    by_name = {node["name"]: (node, payload) for node, payload, _ in old_nodes}
    if any(name.startswith(PREFIX) for name in by_name):
        raise ValueError("feature namespace already exists in original graph")
    outputs = [(node, payload) for node, payload, _ in old_nodes if node.get("type") == "NetOutput"]
    if len(outputs) != 1:
        raise ValueError("one original NetOutput required")
    netoutput, netoutput_raw = outputs[0]
    if len(netoutput.get("input", [])) != 81 or len(netoutput.get("input_desc", [])) != 81:
        raise ValueError("original NetOutput must have exactly 81 data ports; control ports unsupported")
    if any(edge.endswith(":-1") for edge in netoutput["input"]):
        raise ValueError("NetOutput control-edge placement requires separate reviewed handling")
    for edge in netoutput["input"]:
        if not isinstance(edge, str) or ":" not in edge:
            raise ValueError("original NetOutput requires explicit data references")
        name, port = edge.rsplit(":", 1)
        if (not port.isascii() or not port.isdecimal() or str(int(port)) != port
                or name not in by_name or int(port) >= len(by_name[name][0].get("output_desc", []))):
            raise ValueError("original NetOutput producer port is unresolved")
    names = {PREFIX + "input_" + str(i) for i in range(8)}
    names |= {PREFIX + "cast_" + str(i) for i in range(8)} | {PREFIX + "concat"}
    actual = {node["name"] for node, _, _ in template_nodes if node.get("type") != "NetOutput"}
    if actual != names or len(template_nodes) != 18:
        raise ValueError("GE template must contain eight Data/Cast pairs, ConcatD and one NetOutput")
    template = {node["name"]: (node, payload) for node, payload, _ in template_nodes}
    input_field = _field(schema, "OpDef", "input", 9, True)
    input_desc_field = _field(schema, "OpDef", "input_desc", 11, True)
    output_desc_field = _field(schema, "OpDef", "output_desc", 11, True)
    old_ids = [node.get("id", 0) for node, _, _ in old_nodes]
    if any(type(value) is not int or value < 0 or value >= (1 << 63) - 10 for value in old_ids):
        raise ValueError("unsupported original operation id")
    new_id_base = max([len(old_nodes) - 1, *old_ids]) + 1
    id_field_available = any(f.name == "id" for f in schema._messages[schema.package + ".OpDef"].values())
    additions = []
    for index, tap in enumerate(taps):
        reference = tap.get("producer_reference")
        if not isinstance(reference, str) or ":" not in reference:
            raise ValueError("tap requires an explicit original producer port")
        name, port = reference.rsplit(":", 1)
        if not port.isascii() or not port.isdecimal() or str(int(port)) != port:
            raise ValueError("tap port must be a canonical nonnegative integer")
        if name not in by_name or int(port) >= len(by_name[name][0].get("output_desc", [])):
            raise ValueError("tap producer does not exist in original graph")
        node, payload = template[PREFIX + "cast_" + str(index)]
        if node.get("type") != "Cast" or node.get("input") != [PREFIX + "input_" + str(index) + ":0"]:
            raise ValueError("GE template Cast connection differs")
        if template[PREFIX + "input_" + str(index)][0].get("type") != "Data":
            raise ValueError("GE template input is not Data")
        pieces, changed = [], False
        for field, wire, encoded, _ in _parts(payload):
            if field == input_field:
                if changed or wire != 2:
                    raise ValueError("GE Cast must have one data input")
                encoded, changed = _blob(field, reference), True
            pieces.append(encoded)
        if not changed:
            raise ValueError("GE Cast input field missing")
        additions.append(_new_id(b"".join(pieces), schema, new_id_base + index))
    concat, concat_raw = template[PREFIX + "concat"]
    if concat.get("type") != "ConcatD" or concat.get("input") != [PREFIX + "cast_" + str(i) + ":0" for i in range(8)]:
        raise ValueError("GE template concat order differs")
    branch_outputs = [node for node, _, _ in template_nodes if node.get("type") == "NetOutput"]
    if (len(branch_outputs) != 1 or branch_outputs[0].get("input") != [PREFIX + "concat:0"]
            or _canonical(branch_outputs[0].get("input_desc")) != _canonical(concat.get("output_desc"))):
        raise ValueError("GE template output differs from its single concat output")
    descs = [payload for number, wire, _, payload in _parts(concat_raw) if number == output_desc_field and wire == 2]
    if len(descs) != 1:
        raise ValueError("GE concat requires one output descriptor")
    additions.append(_new_id(concat_raw, schema, new_id_base + 8))
    feature_ref = PREFIX + "concat:0"
    new_netoutput = netoutput_raw + _blob(input_field, feature_ref) + _blob(input_desc_field, descs[0])
    pieces, node_index = [], 0
    for field, wire, encoded, payload in _parts(old_graph_raw):
        if field == op_field:
            node, old_payload, _ = old_nodes[node_index]
            node_index += 1
            if node["name"] == netoutput["name"]:
                pieces.extend(_blob(op_field, value) for value in additions)
                encoded = _blob(op_field, new_netoutput)
        pieces.append(encoded)
    graph_output = old_graph.get("output", [])
    if graph_output:
        pieces.append(_blob(_field(schema, "GraphDef", "output", 9, True), feature_ref))
    graph_raw = b"".join(pieces)
    if old_kind == "GraphDef":
        patched = graph_raw
    else:
        patched = b"".join(_blob(field, graph_raw) if field == model_graph_field else encoded
                           for field, _, encoded, _ in _parts(original_raw))
    if len(patched) > MAX_AIR_BYTES:
        raise ValueError("appended AIR exceeds 64 MiB")
    _, _, patched_graph = _decode(schema, patched)
    patched_graph_raw, _ = _graph_raw(patched, schema, old_kind)
    patched_nodes, _ = _node_bytes(patched_graph_raw, schema)
    patched_by_name = {node["name"]: payload for node, payload, _ in patched_nodes}
    preserved = [node["name"] for node, payload, _ in old_nodes
                 if node["name"] != netoutput["name"] and patched_by_name.get(node["name"]) == payload]
    if len(preserved) != len(old_nodes) - 1:
        raise ValueError("an original computation operation changed encoded bytes")
    receipt = {"status": "ORIGINAL_COMPUTATION_BYTES_PRESERVED", "format": old_kind,
               "original_nodes": len(old_nodes), "appended_nodes": len(patched_graph["op"]),
               "original_computation_nodes_byte_identical": len(preserved), "new_nodes": 9,
               "netoutput": netoutput["name"], "original_outputs": 81, "new_outputs": 82,
               "original_sha256": hashlib.sha256(original_raw).hexdigest(),
               "patched_sha256": hashlib.sha256(patched).hexdigest(),
               "id_field_available": id_field_available,
               "new_operation_ids": list(range(new_id_base, new_id_base + 9)) if id_field_available else [],
               "id_policy": ("New node ids exceed both original explicit ids and original node-index range; old ids unchanged"
                             if id_field_available else "SCHEMA_HAS_NO_ID_FIELD_NOT_WRITTEN"),
               "scope": "Wire preservation only; independent source/tap/metadata and GE serialization gates still required"}
    return patched, receipt
