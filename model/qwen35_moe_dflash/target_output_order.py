"""Schema-bound output reordering of an original TorchAir target AIR.

Only the confirmed [logits, features, 80 states] boundary is supported. This
module has no filesystem, GE, Torch or payload access. Original computation
nodes retain their complete encoded bytes, including FileConstant references.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import re

from .existing_air_profile import get_existing_air_profile
from .omc_wire import _varint


MAX_AIR_BYTES = 64 << 20
OUTPUT_PERMUTATION = (0, *range(2, 82), 1)
_UNCHANGED_ATTRS = frozenset({"true_branch_stream", "_inner_net_output", "_datadump_original_op_names"})
_SUPPORTED_ATTRS = _UNCHANGED_ATTRS | {"_input_name_key", "_input_name_value", "is_input_const"}
_NET_FIELDS = frozenset({"name", "type", "input", "input_desc", "output_desc", "attr", "has_out_attr",
                         "id", "stream_id", "is_input_const"})


def _v(value):
    if type(value) is not int or not 0 <= value < 1 << 64:
        raise ValueError("nonnegative uint64 wire value required")
    result = bytearray()
    while value >= 128:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def _blob(number, value):
    return _v(number << 3 | 2) + _v(len(value)) + value


def _parts(raw):
    offset, count = 0, 0
    while offset < len(raw):
        count += 1
        if count > 2_000_000:
            raise ValueError("wire field budget exceeded")
        start = offset
        tag, offset = _varint(raw, offset)
        number, wire = tag >> 3, tag & 7
        if not 0 < number < 1 << 29:
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
    qualified = message if message.startswith(schema.package + ".") else schema.package + "." + message
    rows = [row for row in schema._messages.get(qualified, {}).values() if row.name == name]
    if len(rows) != 1 or rows[0].kind != kind or rows[0].repeated is not repeated:
        raise ValueError("selected schema lacks required field: " + qualified + "." + name)
    return rows[0]


def _replace(raw, replacements):
    """Replace exact field spans without changing other bytes or field order."""
    positions = Counter()
    result = []
    for number, _, encoded, _ in _parts(raw):
        if number in replacements:
            index = positions[number]
            if index >= len(replacements[number]):
                raise ValueError("wire replacement count differs")
            encoded = replacements[number][index]
            positions[number] += 1
        result.append(encoded)
    if any(positions[number] != len(values) for number, values in replacements.items()):
        raise ValueError("wire replacement field missing")
    return b"".join(result)


def _spans(raw, field, wire=2):
    rows = [(kind, encoded, payload) for number, kind, encoded, payload in _parts(raw) if number == field]
    if any(kind != wire for kind, _, _ in rows):
        raise ValueError("mistyped repeated wire field")
    return rows


def _attributes(node):
    result = {}
    for row in node.get("attr", []):
        if not isinstance(row, dict) or not isinstance(row.get("key"), str) or row["key"] in result:
            raise ValueError("missing or duplicate operation attribute key")
        result[row["key"]] = row.get("value", {})
    return result


def _descriptor(raw, enum):
    shape = raw.get("shape") if isinstance(raw, dict) else None
    dimensions = shape.get("dim", []) if isinstance(shape, dict) else None
    dtype = raw.get("dtype") if isinstance(raw, dict) else None
    if (not isinstance(dimensions, list) or len(dimensions) > 16
            or any(type(dim) is not int or dim < -1 for dim in dimensions)
            or type(dtype) is not int or dtype not in enum.values()
            or not isinstance(raw.get("layout"), str) or not raw["layout"]):
        raise ValueError("missing or unsupported boundary dtype/shape/layout")
    return {"shape": dimensions, "raw_dtype": dtype, "layout": raw["layout"]}


def _input_check(ops, enum, sequence_length):
    inputs = [node for node in ops if node.get("type") in {"Data", "RefData"}]
    if len(inputs) != 89 or Counter(node["type"] for node in inputs) != {"Data": 9, "RefData": 80}:
        raise ValueError("target requires exactly 9 Data and 80 RefData inputs")
    rows, states = [], {}
    for node in inputs:
        if len(node.get("output_desc", [])) != 1:
            raise ValueError("every original input requires one output descriptor")
        desc = _descriptor(node["output_desc"][0], enum)
        index = _attributes(node).get("index", {}).get("i")
        if index is not None and (type(index) is not int or index < 0):
            raise ValueError("invalid serialized input index")
        rows.append({"name": node["name"], "type": node["type"], "serialized_index": index, **desc})
        if node["type"] == "RefData":
            states[node["name"]] = desc
    if set(states) != {f"RefData_{index}" for index in range(80)}:
        raise ValueError("exact RefData_0..79 names are required")
    profile = get_existing_air_profile("cpro-40block-64-1")
    expected = {entry.split(":", 1)[0]: [int(dim) for dim in entry.split(":", 1)[1].split(",")]
                for entry in profile["input_shape"].split(";")}
    for index in range(80):
        name = f"RefData_{index}"
        if states[name] != {"shape": expected[name], "raw_dtype": enum["DT_FLOAT16"], "layout": "ND"}:
            raise ValueError("FP16 ND state geometry differs: " + name)
    # Ordinary argument numbers can vary between TorchAir export revisions.
    # Match all nine visible geometries without assigning roles by names or
    # claiming that equal-shaped scalar arguments have runtime identities.
    geometries = [value for name, value in expected.items() if not name.startswith("RefData_")]
    for row in (row for row in rows if row["type"] == "Data"):
        shape = row["shape"]
        matches = [index for index, template in enumerate(geometries) if len(shape) == len(template)
            and all(actual == fixed if fixed != -1 else actual in {-1, sequence_length}
                    for actual, fixed in zip(shape, template))]
        if not matches or row["layout"] != "ND":
            raise ValueError("ordinary input geometry/layout differs from the 89-input profile: " + row["name"])
        del geometries[matches[0]]
    return states, {"status": "MATCHED_TYPED_INPUT_GEOMETRIES", "input_count": 89,
        "input_type_counts": {"Data": 9, "RefData": 80}, "inputs": rows,
        "serialized_indices": "OBSERVED_ONLY_NOT_ASSUMED_TO_BE_RUNTIME_ORDER",
        "runtime_input_order": "UNVERIFIED"}


def _edge_descriptor(reference, by_name, enum, declared):
    if not isinstance(reference, str):
        raise ValueError("output producer reference is not text")
    name, separator, port = reference.rpartition(":")
    if not separator or not re.fullmatch(r"0|[1-9][0-9]*", port):
        raise ValueError("control edges or noncanonical output producer ports are unsupported")
    producer = by_name.get(name)
    if producer is None or int(port) >= len(producer.get("output_desc", [])):
        raise ValueError("output producer or selected descriptor is missing")
    descriptor = _descriptor(producer["output_desc"][int(port)], enum)
    if any(descriptor[key] != declared[key] for key in ("raw_dtype", "layout")):
        raise ValueError("output producer dtype/layout contradicts its NetOutput descriptor")
    if descriptor["shape"] and descriptor["shape"] != declared["shape"]:
        raise ValueError("known output producer shape contradicts its NetOutput descriptor")


def _boundary_check(graph, schema, sequence_length, feature_index):
    ops = graph.get("op", [])
    if not isinstance(ops, list) or not 1 <= len(ops) <= 32768:
        raise ValueError("one bounded nonempty root operation list is required")
    names = [node.get("name") for node in ops]
    if any(not isinstance(name, str) or not name or len(name) > 1024 for name in names) or len(set(names)) != len(names):
        raise ValueError("root operation names must be present, bounded and unique")
    by_name = dict(zip(names, ops))
    enum = schema.enums.get(schema.package + ".DataType", {})
    if "DT_FLOAT16" not in enum or "DT_FLOAT" not in enum:
        raise ValueError("selected schema requires explicit DT_FLOAT and DT_FLOAT16 enum names")
    states, input_receipt = _input_check(ops, enum, sequence_length)
    nets = [node for node in ops if node.get("type") == "NetOutput"]
    if len(nets) != 1:
        raise ValueError("exactly one root NetOutput is required")
    net = nets[0]
    references, descriptors = net.get("input", []), net.get("input_desc", [])
    if len(references) != 82 or len(descriptors) != 82:
        raise ValueError("NetOutput must expose exactly 82 explicit data edges and descriptors")
    declared = [_descriptor(desc, enum) for desc in descriptors]
    for reference, desc in zip(references, declared):
        _edge_descriptor(reference, by_name, enum, desc)
    if len(set(references)) != 82:
        raise ValueError("target output producer references must be unique")
    if declared[0] != {"raw_dtype": enum["DT_FLOAT"], "shape": [1, 1, 248320], "layout": "ND"}:
        raise ValueError("output 0 must declare FP32 ND logits [1,1,248320]")
    feature = {"raw_dtype": enum["DT_FLOAT16"], "shape": [1, sequence_length, 16384], "layout": "ND"}
    matches = [index for index, desc in enumerate(declared) if desc == feature]
    if matches != [feature_index]:
        raise ValueError(f"exactly one FP16 ND feature output is required at physical port {feature_index}")
    state_slots = [index for index in range(1, 82) if index != feature_index]
    for index, slot in enumerate(state_slots):
        if declared[slot] != states[f"RefData_{index}"]:
            raise ValueError(f"state output at physical port {slot} differs from RefData_{index} geometry/dtype/layout")
    graph_outputs = graph.get("output", [])
    if graph_outputs and graph_outputs != references:
        raise ValueError("GraphDef.output must be absent/empty or exactly the ordered NetOutput producer references")
    return net, input_receipt


def _primitive_replacements(raw, field, values):
    rows = [(wire, encoded) for number, wire, encoded, _ in _parts(raw) if number == field.number]
    if len(rows) == len(values) and all(wire == 0 for wire, _ in rows):
        return [_v(field.number << 3) + _v(value) for value in values]
    if len(rows) == 1 and rows[0][0] == 2:
        return [_blob(field.number, b"".join(_v(value) for value in values))]
    raise ValueError("ambiguous packed/unpacked port metadata encoding")


def _attr_list(value, member, *, length=82):
    if not isinstance(value, dict) or set(value) != {"list"}:
        raise ValueError("port attribute requires an explicit list value")
    listing = value["list"]
    if not isinstance(listing, dict) or set(listing) - {member, "val_type"}:
        raise ValueError("port attribute list contains unsupported fields")
    entries = listing.get(member)
    if not isinstance(entries, list) or len(entries) != length:
        raise ValueError("port attribute list must contain exactly 82 entries")
    return entries


def _update_list_attr(entry_raw, schema, entry_type, member, values):
    value_field = _field(schema, entry_type, "value", 11, False)
    value_rows = _spans(entry_raw, value_field.number)
    if len(value_rows) != 1:
        raise ValueError("attribute map entry requires one value")
    value_raw = value_rows[0][2]
    list_field = _field(schema, value_field.type_name, "list", 11, False)
    list_rows = _spans(value_raw, list_field.number)
    if len(list_rows) != 1:
        raise ValueError("port metadata requires one explicit list")
    list_raw = list_rows[0][2]
    member_field = _field(schema, list_field.type_name, member, 8 if member == "b" else 3, True)
    list_raw = _replace(list_raw, {member_field.number: _primitive_replacements(list_raw, member_field, values)})
    value_raw = _replace(value_raw, {list_field.number: [_blob(list_field.number, list_raw)]})
    return _replace(entry_raw, {value_field.number: [_blob(value_field.number, value_raw)]})


def _net_replacements(net, raw, schema):
    unsupported_fields = sorted(key for key, value in net.items()
                                if key not in _NET_FIELDS and value not in ([], {}, None))
    if unsupported_fields or net.get("output_desc"):
        raise ValueError("unreviewed NetOutput port fields: " + ",".join(unsupported_fields or ["output_desc"]))
    attrs = _attributes(net)
    unknown = sorted(set(attrs) - _SUPPORTED_ATTRS)
    if unknown:
        raise ValueError("unreviewed NetOutput attributes: " + ",".join(unknown))
    paired = {"_input_name_key", "_input_name_value"} & set(attrs)
    if paired and len(paired) != 2:
        raise ValueError("NetOutput input name/port mapping is incomplete")
    inverse = {old: new for new, old in enumerate(OUTPUT_PERMUTATION)}
    updates = {}
    if paired:
        keys = _attr_list(attrs["_input_name_key"], "s")
        values = _attr_list(attrs["_input_name_value"], "i")
        if (any(not isinstance(key, str) or not key or len(key) > 1024 for key in keys)
                or len(set(keys)) != 82 or any(type(value) is not int for value in values)
                or set(values) != set(range(82))):
            raise ValueError("NetOutput name mapping must bind 82 unique names to all 82 ports")
        updates["_input_name_value"] = ("i", [inverse[value] for value in values])
    if "is_input_const" in attrs:
        values = _attr_list(attrs["is_input_const"], "b")
        if any(type(value) is not bool for value in values):
            raise ValueError("NetOutput is_input_const attribute is not boolean")
        updates["is_input_const"] = ("b", [int(values[index]) for index in OUTPUT_PERMUTATION])
    replacements = {}
    for name, kind in (("input", 9), ("input_desc", 11)):
        field = _field(schema, "OpDef", name, kind, True)
        spans = _spans(raw, field.number)
        if len(spans) != 82:
            raise ValueError("NetOutput repeated wire port count differs")
        replacements[field.number] = [spans[index][1] for index in OUTPUT_PERMUTATION]
    if "is_input_const" in net:
        values = net["is_input_const"]
        if len(values) != 82 or any(type(value) is not bool for value in values):
            raise ValueError("NetOutput top-level is_input_const must contain 82 boolean entries")
        field = _field(schema, "OpDef", "is_input_const", 8, True)
        replacements[field.number] = _primitive_replacements(raw, field,
            [int(values[index]) for index in OUTPUT_PERMUTATION])
    if updates:
        field = _field(schema, "OpDef", "attr", 11, True)
        spans = _spans(raw, field.number)
        if len(spans) != len(net.get("attr", [])):
            raise ValueError("NetOutput attribute wire count differs")
        new_attrs = []
        for row, (_, encoded, payload) in zip(net["attr"], spans):
            if row["key"] in updates:
                member, values = updates[row["key"]]
                encoded = _blob(field.number, _update_list_attr(payload, schema, field.type_name, member, values))
            new_attrs.append(encoded)
        replacements[field.number] = new_attrs
    return replacements, sorted(updates)


def reorder_target_output_bytes(raw, schema, sequence_length=64):
    """Return canonical AIR bytes and an output-only preservation receipt.

    Equal-shaped state K/V identities are not inferred: the existing 80 state
    outputs retain their exact source-relative order and encoded descriptors.
    This is a host metadata operation, not target compilation or execution.
    """
    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_AIR_BYTES:
        raise ValueError("bounded nonempty raw AIR bytes required")
    if type(sequence_length) is not int or not 1 <= sequence_length <= 64:
        raise ValueError("sequence_length must be an integer in 1..64")
    if schema.package != "ge.proto":
        raise ValueError("the explicitly selected TorchAir ge.proto schema is required")
    model = schema.decode(raw, "ModelDef")
    graphs = model.get("graph", [])
    if len(graphs) != 1:
        raise ValueError("one strict raw ModelDef graph is required")
    graph = graphs[0]
    net, inputs = _boundary_check(graph, schema, sequence_length, 1)
    graph_field = _field(schema, "ModelDef", "graph", 11, True)
    graph_rows = _spans(raw, graph_field.number)
    if len(graph_rows) != 1:
        raise ValueError("one encoded ModelDef graph is required")
    graph_raw = graph_rows[0][2]
    op_field = _field(schema, "GraphDef", "op", 11, True)
    op_rows = _spans(graph_raw, op_field.number)
    if len(op_rows) != len(graph["op"]):
        raise ValueError("operation wire count differs")
    net_index = next(index for index, node in enumerate(graph["op"]) if node is net)
    net_raw = op_rows[net_index][2]
    replacements, attrs_updated = _net_replacements(net, net_raw, schema)
    new_net_raw = _replace(net_raw, replacements)
    new_op_spans = [encoded for _, encoded, _ in op_rows]
    new_op_spans[net_index] = _blob(op_field.number, new_net_raw)
    graph_replacements = {op_field.number: new_op_spans}
    if graph.get("output"):
        output_field = _field(schema, "GraphDef", "output", 9, True)
        spans = _spans(graph_raw, output_field.number)
        if len(spans) != 82:
            raise ValueError("GraphDef.output wire count differs")
        graph_replacements[output_field.number] = [spans[index][1] for index in OUTPUT_PERMUTATION]
    new_graph_raw = _replace(graph_raw, graph_replacements)
    patched = _replace(raw, {graph_field.number: [_blob(graph_field.number, new_graph_raw)]})
    if len(patched) > MAX_AIR_BYTES:
        raise ValueError("canonical AIR exceeds byte budget")
    patched_model = schema.decode(patched, "ModelDef")
    patched_graph = patched_model["graph"][0]
    new_net, new_inputs = _boundary_check(patched_graph, schema, sequence_length, 81)
    if inputs != new_inputs:
        raise ValueError("original input metadata changed")
    for key in ("input", "input_desc"):
        if new_net[key] != [net[key][index] for index in OUTPUT_PERMUTATION]:
            raise ValueError("output permutation proof failed")
    new_nodes = _spans(new_graph_raw, op_field.number)
    preserved = sum(old[1] == new[1] for index, (old, new) in enumerate(zip(op_rows, new_nodes)) if index != net_index)
    if len(new_nodes) != len(op_rows) or preserved != len(op_rows) - 1:
        raise ValueError("an original computation operation changed encoded bytes")
    receipt = {"schema_version": 1, "artifact_kind": "qwen35-target-output-order-canonicalization",
        "status": "OUTPUT_ORDER_CANONICALIZED", "original_sha256": hashlib.sha256(raw).hexdigest(),
        "patched_sha256": hashlib.sha256(patched).hexdigest(), "original_bytes": len(raw), "patched_bytes": len(patched),
        "schema": dict(schema.evidence), "sequence_length": sequence_length,
        "output_permutation": list(OUTPUT_PERMUTATION), "input_validation": inputs,
        "source_order": "LOGITS_FEATURES_80_STATES", "canonical_order": "LOGITS_80_STATES_FEATURES",
        "netoutput": net["name"], "netoutput_attributes_updated": attrs_updated,
        "graph_output_policy": "EXACT_REFERENCES_REORDERED" if graph.get("output") else "ABSENT_OR_EMPTY_UNCHANGED",
        "computation_ops_byte_identical": preserved, "original_op_count": len(op_rows), "new_op_count": len(new_nodes),
        "file_constant_ops_byte_identical": sum(node.get("type") == "FileConstant" for node in graph["op"]),
        "weights": "NOT_OPENED_OR_COPIED_REFERENCES_BYTE_IDENTICAL",
        "state_identity_scope": "All 80 states retain source-relative order and exact descriptors; equal-shaped K/V lineage and device continuity are not proven",
        "model_execution": "NOT_RUN", "device_execution": "NOT_RUN", "numerical_validation": "NOT_RUN",
        "scope": "Only schema-bound ordered output metadata changes; original input and computation operation wire bytes remain unchanged"}
    return patched, receipt
