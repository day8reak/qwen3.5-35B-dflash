"""Independent schema-bound validation of one target output permutation.

Only raw AIR metadata is read from caller-supplied bytes. This validates an
interface permutation, never the numerical identity of same-shaped cache rows,
state continuity, compilation compatibility or device execution.
"""
from __future__ import annotations

from copy import deepcopy
from collections import Counter
import hashlib
import re

from qwen35_moe_dflash.toolchain.existing_air_profile import get_existing_air_profile
from qwen35_moe_dflash.toolchain.omc_wire import _fields
from qwen35_moe_dflash.target.original_air_roundtrip import MAX_AIR_BYTES, _canonical, _decode, _digest


_ORDER = [0, *range(2, 82), 1]
_INVERSE = [0, 81, *range(1, 81)]
_EDGE = re.compile(r"(.+):(0|[1-9][0-9]*)\Z")
_PLAIN_ATTRS = {"true_branch_stream", "_inner_net_output", "_datadump_original_op_names"}
_PORT_ATTRS = {"_input_name_key", "_input_name_value", "is_input_const"}
_EMPTY_FIELDS = {"src_name", "src_index", "dst_name", "dst_index", "input_name",
                 "input_i", "output_i", "input_o", "output_o", "workspace", "workspace_bytes", "output_desc"}
_NET_FIELDS = {"name", "type", "input", "input_desc", "attr", "is_input_const", "id", "has_out_attr", "stream_id"} | _EMPTY_FIELDS


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _same(first, second):
    return _digest(first) == _digest(second)


def _attrs(node):
    rows = node.get("attr", [])
    _require(isinstance(rows, list), "NetOutput attr must be a map-entry list")
    result = {}
    for row in rows:
        _require(isinstance(row, dict) and set(row) == {"key", "value"}
                 and isinstance(row["key"], str) and row["key"] not in result,
                 "NetOutput has duplicate or unsupported map entries")
        result[row["key"]] = row["value"]
    return result


def _edge(value, by_name):
    match = _EDGE.fullmatch(value) if isinstance(value, str) else None
    _require(match is not None and not match[1].startswith("^"), "output must be an explicit non-control canonical producer:port")
    name, port = match[1], int(match[2])
    _require(name in by_name and port < len(by_name[name].get("output_desc", [])), "output producer port is missing")
    return name, port


def _geometry(desc, dtype, shape, where):
    _require(isinstance(desc, dict) and desc.get("dtype", 0) == dtype
             and type(desc.get("dtype", 0)) is int and desc.get("shape", {}).get("dim", []) == shape
             and desc.get("layout") == "ND", where + " declared dtype/shape/layout differs")


def _list(value, kind, *, length, where):
    _require(isinstance(value, dict) and set(value) == {"list"} and isinstance(value["list"], dict)
             and set(value["list"]) <= {kind, "val_type"}, where + " list metadata is unsupported")
    rows = value["list"].get(kind)
    _require(isinstance(rows, list) and len(rows) == length, where + " must contain exactly " + str(length) + " entries")
    return rows


def _permuted_net(original):
    unsupported = {key for key in set(original) - _NET_FIELDS if original[key] not in (None, [], {})}
    _require(not unsupported, "NetOutput has unsupported fields: " + ",".join(sorted(unsupported)))
    _require(all(original.get(key) in (None, [], {}) for key in _EMPTY_FIELDS), "nonempty NetOutput src/dst/offset/output metadata is unsupported")
    result = deepcopy(original)
    result["input"] = [original["input"][index] for index in _ORDER]
    result["input_desc"] = [original["input_desc"][index] for index in _ORDER]
    if "is_input_const" in original:
        flags = original["is_input_const"]
        _require(isinstance(flags, list) and len(flags) == 82 and all(type(value) is bool for value in flags),
                 "NetOutput top-level const flags must align with all 82 inputs")
        result["is_input_const"] = [flags[index] for index in _ORDER]
    attributes = _attrs(original)
    _require(not set(attributes) - (_PLAIN_ATTRS | _PORT_ATTRS), "NetOutput has unsupported attributes: " + ",".join(sorted(set(attributes) - (_PLAIN_ATTRS | _PORT_ATTRS))))
    if ("_input_name_key" in attributes) != ("_input_name_value" in attributes):
        raise ValueError("NetOutput input-name map must provide both key and value lists")
    expected = deepcopy(attributes)
    if "_input_name_key" in attributes:
        names = _list(attributes["_input_name_key"], "s", length=82, where="NetOutput input names")
        indices = _list(attributes["_input_name_value"], "i", length=82, where="NetOutput input-name indices")
        _require(all(isinstance(name, str) and name for name in names) and len(set(names)) == 82,
                 "NetOutput input names must be unique explicit labels")
        _require(all(type(index) is int for index in indices) and set(indices) == set(range(82)),
                 "NetOutput input-name map must be a complete unambiguous port bijection")
        expected["_input_name_value"]["list"]["i"] = [_INVERSE[index] for index in indices]
    if "is_input_const" in attributes:
        flags = _list(attributes["is_input_const"], "b", length=82, where="NetOutput attribute const flags")
        _require(all(type(value) is bool for value in flags), "NetOutput attribute const flags must be booleans")
        _require(attributes["is_input_const"]["list"].get("val_type", 4) == 4,
                 "NetOutput const-flag metadata val_type is unsupported")
        expected["is_input_const"]["list"]["b"] = [flags[index] for index in _ORDER]
    if "attr" in result:
        result["attr"] = [{"key": row["key"], "value": expected[row["key"]]} for row in original["attr"]]
    return result


def _operation_payloads(raw, kind, schema):
    def number(message, name):
        values = [field.number for field in schema._messages[schema.package + "." + message].values() if field.name == name]
        _require(len(values) == 1, "selected schema does not uniquely identify " + message + "." + name)
        return values[0]
    graph = raw
    if kind == "ModelDef":
        graph_field = number("ModelDef", "graph")
        graphs = [bytes(value) for key, wire, value in _fields(raw, [2_000_000]) if key == graph_field and wire == 2]
        _require(len(graphs) == 1, "exactly one raw root graph is required")
        graph = graphs[0]
    op_field = number("GraphDef", "op")
    return [(schema.decode(value, "OpDef")["name"], bytes(value))
            for key, wire, value in _fields(graph, [2_000_000]) if key == op_field and wire == 2]


def _file_constants(value):
    if isinstance(value, dict):
        if value.get("type") == "FileConstant":
            yield value
        for key in sorted(value):
            yield from _file_constants(value[key])
    elif isinstance(value, list):
        for item in value:
            yield from _file_constants(item)


def validate_target_output_order(original_raw, normalized_raw, schema, *, sequence_length=64):
    """Validate [logits,feature,80states] -> [logits,80states,feature]."""
    report = {"schema_version": 1, "artifact_kind": "qwen35-target-output-order-validation",
              "status": "FAILED", "model_compilation": "NOT_RUN", "device_execution": "NOT_RUN",
              "numerical_validation": "NOT_EVALUATED", "state_continuity": "UNVERIFIED",
              "state_identity": "RELATIVE_STATE_OUTPUT_ORDER_ONLY_NOT_PROVEN_BY_SAME_SHAPED_DESCRIPTORS",
              "weight_payload_access": "NONE", "scope": "Metadata/byte preservation and one fixed output permutation only; no cache-update, model accuracy or device pass"}
    try:
        _require(type(sequence_length) is int and 1 <= sequence_length <= 64, "sequence_length must be an explicit integer in 1..64")
        _require(all(isinstance(raw, bytes) and 0 < len(raw) <= MAX_AIR_BYTES for raw in (original_raw, normalized_raw)), "bounded raw AIR bytes are required")
        old_kind, old_model, old_graph = _decode(schema, original_raw)
        new_kind, new_model, new_graph = _decode(schema, normalized_raw)
        _require(old_kind == new_kind, "root protobuf format changed")
        report.update(format=old_kind, original_sha256=hashlib.sha256(original_raw).hexdigest(),
                      normalized_sha256=hashlib.sha256(normalized_raw).hexdigest(), schema=dict(schema.evidence))
        if old_kind == "ModelDef":
            _require(_same({key: value for key, value in old_model.items() if key != "graph"},
                          {key: value for key, value in new_model.items() if key != "graph"}), "unrelated ModelDef metadata changed")
        _require(_same({key: value for key, value in old_graph.items() if key not in {"op", "output"}},
                      {key: value for key, value in new_graph.items() if key not in {"op", "output"}}), "unrelated GraphDef metadata changed")
        old_nodes, new_nodes = old_graph["op"], new_graph["op"]
        _require([node["name"] for node in old_nodes] == [node["name"] for node in new_nodes], "original operation names/order/count changed")
        by_name = {node["name"]: node for node in old_nodes}
        types = schema.enums.get(schema.package + ".DataType", {})
        half, single = types.get("DT_FLOAT16"), types.get("DT_FLOAT")
        _require(type(half) is int and type(single) is int and half != single, "selected schema must explicitly identify FP16 and FP32")
        inputs = [node for node in old_nodes if node.get("type") in {"Data", "RefData"}]
        refs = {node["name"]: node for node in inputs if node.get("type") == "RefData"}
        _require(len(inputs) == 89 and len(refs) == 80 and set(refs) == {f"RefData_{index}" for index in range(80)}, "complete original 89-input/80-RefData interface is required")
        shape_rows = [entry.split(":", 1) for entry in get_existing_air_profile("cpro-40block-64-1")["input_shape"].split(";")]
        profile = {name: [sequence_length if int(dim) == -1 else int(dim) for dim in dims.split(",")] for name, dims in shape_rows}
        ordinary_shapes = []
        for node in inputs:
            if node.get("type") == "Data":
                descriptors = node.get("output_desc", [])
                _require(len(descriptors) == 1, "ordinary Data must expose one declared output")
                _require(type(descriptors[0].get("dtype", 0)) is int and descriptors[0].get("dtype", 0) in types.values()
                         and descriptors[0].get("layout") == "ND", "ordinary Data dtype/layout is unsupported")
                shape = descriptors[0].get("shape", {}).get("dim", [])
                _require(isinstance(shape, list) and shape and all(type(dim) is int and dim != 0 for dim in shape),
                         "ordinary input shape is missing or invalid")
                ordinary_shapes.append(tuple(sequence_length if dim == -1 else dim for dim in shape))
        _require(Counter(ordinary_shapes) == Counter(tuple(shape) for name, shape in profile.items() if not name.startswith("RefData_")),
                 "ordinary input geometries differ from the full original profile")
        for name, node in refs.items():
            descriptors = node.get("output_desc", [])
            _require(len(descriptors) == 1, "RefData must expose one declared output: " + name)
            _geometry(descriptors[0], half, profile[name], name)
        netoutputs = [node for node in old_nodes if node.get("type") == "NetOutput"]
        _require(len(netoutputs) == 1, "one original NetOutput is required")
        net = netoutputs[0]
        edges, descriptors = net.get("input", []), net.get("input_desc", [])
        _require(len(edges) == len(descriptors) == 82, "original NetOutput must expose exactly 82 data ports")
        _require(len(set(edges)) == 82, "duplicate output producer references are unsupported")
        for edge, declared in zip(edges, descriptors):
            producer, port = _edge(edge, by_name)
            observed = by_name[producer]["output_desc"][port]
            _require(observed.get("dtype", 0) == declared.get("dtype", 0)
                     and observed.get("layout") == declared.get("layout"), "output producer dtype/layout contradicts its declared descriptor")
            shape = observed.get("shape", {}).get("dim", [])
            _require(not shape or shape == declared.get("shape", {}).get("dim", []),
                     "known output producer shape contradicts its declared descriptor")
        _geometry(descriptors[0], single, [1, 1, 248320], "original logits output 0")
        _geometry(descriptors[1], half, [1, sequence_length, 16384], "original feature output 1")
        for index in range(80):
            _geometry(descriptors[index + 2], half, profile[f"RefData_{index}"], "original state relative index " + str(index))
        expected_net = _permuted_net(net)
        for before, after in zip(old_nodes, new_nodes):
            _require(_same(expected_net if before is net else before, after), "operation changed outside the approved permutation: " + before["name"])
        if "output" in old_graph:
            outputs = old_graph["output"]
            _require(isinstance(outputs, list) and (not outputs or outputs == edges), "unsupported original GraphDef.output representation")
            expected = [outputs[index] for index in _ORDER] if outputs else outputs
            _require("output" in new_graph and new_graph["output"] == expected, "GraphDef.output does not follow the fixed permutation")
        else:
            _require("output" not in new_graph, "GraphDef.output was introduced")
        original_payloads = _operation_payloads(original_raw, old_kind, schema)
        normalized_payloads = dict(_operation_payloads(normalized_raw, new_kind, schema))
        preserved = 0
        for name, payload in original_payloads:
            if name != net["name"]:
                _require(normalized_payloads[name] == payload, "original computation operation encoded bytes changed: " + name)
                preserved += 1
        old_constants = list(_file_constants(_canonical(old_model)))
        new_constants = list(_file_constants(_canonical(new_model)))
        _require(_same(old_constants, new_constants), "FileConstant metadata changed")
        report.update(status="PASS_METADATA_ORDER_ONLY", permutation=_ORDER, input_count=89,
                      refdata_count=80, output_count=82, original_feature_index=1, normalized_feature_index=81,
                      state_relative_order="PRESERVED_EXACT_REFS_AND_DESCRIPTORS",
                      original_computation_nodes_byte_identical=preserved,
                      file_constants={"count": len(old_constants), "metadata_sha256": _digest(old_constants)})
    except (KeyError, TypeError, ValueError, RecursionError) as error:
        report["error"] = f"{type(error).__name__}: {str(error)[:1200]}"
    return report
