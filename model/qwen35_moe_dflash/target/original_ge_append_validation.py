"""Independent metadata-only validation of the original GE feature append.

Accept strict-decoded ModelDef dictionaries, never AIR or weight paths. Original
metadata is immutable except the one appended NetOutput port and (when present)
the appended GraphDef.output edge. Passing this validator is not an arithmetic,
source-tap provenance, compilation or device-execution result.
"""
from __future__ import annotations

import hashlib
import json
import re


LAYER_IDS = (1, 6, 11, 16, 22, 27, 32, 37)
_PREFIX = "__dflash_features/"
_CAST_NAMES = tuple(_PREFIX + f"cast_{index}" for index in range(8))
_CONCAT = _PREFIX + "concat"
_MAX_METADATA_BYTES = 64 << 20
_MAX_NODES = 32768
_EDGE = re.compile(r"(.+):(0|[1-9][0-9]*)\Z")
_NEW_NODE_FIELDS = {"name", "type", "input", "input_desc", "output_desc", "attr",
                    "has_out_attr", "id", "is_input_const"}
# Observed in the real CANN 9.1 GE branch template, decoded with TorchAir schema
# descriptor 5a1eac27746a7bb5109e819ce6dd7813a3ffcaab04bd0ab2ca3c2cdf1e4084ea.
# Every optional metadata value is checked; unlisted fields/attrs fail closed.
_DESCRIPTOR_METADATA = {
    "origin_shape_initialized": {"b": False}, "origin_format": {"s": "ND"},
    "origin_format_for_int": {"i": 2}, "format_for_int": {"i": 2},
    "placement": {"i": 0}, "shape_range": {"list_list_int": {}},
}


def _canonical(value):
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("strict-decoded metadata requires string dictionary keys")
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, list):
        rows = [_canonical(item) for item in value]
        if rows and all(isinstance(row, dict) and isinstance(row.get("key"), str)
                        and not set(row) - {"key", "value"} for row in rows):
            if len({row["key"] for row in rows}) != len(rows):
                raise ValueError("duplicate protobuf map key")
            rows.sort(key=lambda row: row["key"])
        return rows
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise ValueError("input must contain strict-decoded JSON metadata only")


def _bytes(value):
    raw = json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode()
    if len(raw) > _MAX_METADATA_BYTES:
        raise ValueError("decoded metadata exceeds 64 MiB comparison budget")
    return raw


def _sha(value):
    return hashlib.sha256(_bytes(value)).hexdigest()


def _same(first, second):
    return _bytes(first) == _bytes(second)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _root(model):
    _require(isinstance(model, dict), "strict-decoded ModelDef dictionary is required")
    graphs = model.get("graph")
    _require(isinstance(graphs, list) and len(graphs) == 1 and isinstance(graphs[0], dict),
             "exactly one root graph is required")
    graph = graphs[0]
    nodes = graph.get("op")
    _require(isinstance(nodes, list) and 1 <= len(nodes) <= _MAX_NODES, "bounded root operation list is required")
    _require(all(isinstance(node, dict) and isinstance(node.get("name"), str)
                 and node["name"] and isinstance(node.get("type"), str) for node in nodes),
             "all root nodes require explicit names/types")
    _require(len({node["name"] for node in nodes}) == len(nodes), "duplicate root operation name")
    return graph, nodes


def _edge(value):
    _require(isinstance(value, str), "edge must be one explicit producer:port string")
    match = _EDGE.fullmatch(value)
    _require(match is not None and not match[1].startswith("^"), "unsupported or ambiguous producer edge")
    return match[1], int(match[2])


def _shape(desc):
    value = desc.get("shape", {})
    _require(isinstance(value, dict) and set(value) <= {"dim"}, "unexpected TensorDescriptor shape metadata")
    dims = value.get("dim", [])
    _require(isinstance(dims, list) and all(type(size) is int for size in dims), "invalid tensor shape")
    return dims


def _dtype(desc):
    value = desc.get("dtype", 0)  # Omitted proto3 default remains absent in strict decode.
    _require(type(value) is int, "TensorDescriptor dtype must be a selected-schema enum integer")
    return value


def _descriptor(desc, *, dtype, shape, where):
    _require(isinstance(desc, dict), where + " descriptor is missing")
    _require(_dtype(desc) == dtype and _shape(desc) == shape and desc.get("layout") == "ND",
             where + " must retain the exact ND dtype/shape")
    _require(set(desc) <= {"dtype", "shape", "layout", "attr", "device_type", "real_dim_cnt"},
             where + " contains an unsupported descriptor field")
    if "device_type" in desc:
        _require(desc["device_type"] == "NPU", where + " device_type differs from the observed GE template")
    if "real_dim_cnt" in desc:
        _require(type(desc["real_dim_cnt"]) is int and desc["real_dim_cnt"] == 3,
                 where + " real_dim_cnt must be three")
    for key, value in _attrs(desc).items():
        _require(key in _DESCRIPTOR_METADATA and _same(value, _DESCRIPTOR_METADATA[key]),
                 where + " contains unsupported descriptor metadata: " + key)


def _attrs(node):
    rows = node.get("attr", [])
    _require(isinstance(rows, list), "operation attr must be a protobuf map-entry list")
    result = {}
    for row in rows:
        _require(isinstance(row, dict) and set(row) == {"key", "value"}
                 and isinstance(row["key"], str) and row["key"] not in result,
                 "invalid or duplicate operation attribute")
        result[row["key"]] = row["value"]
    return result


def _attr_int(attributes, key, expected, where):
    value = attributes.get(key)
    _require(isinstance(value, dict) and set(value) == {"i"}
             and type(value["i"]) is int and value["i"] == expected,
             where + " attribute " + key + " differs from its explicit operation contract")


def _new_node_metadata(node, input_count, semantic_attrs):
    _require(set(node) <= _NEW_NODE_FIELDS, "new feature operation has unsupported OpDef fields")
    if "has_out_attr" in node:
        _require(node["has_out_attr"] is True, "new feature operation has_out_attr differs from its GE template")
    if "id" in node:
        _require(type(node["id"]) is int and node["id"] >= 0, "new feature operation id must be a nonnegative integer")
    if "is_input_const" in node:
        _require(_same(node["is_input_const"], [False] * input_count), "new feature operation inputs cannot be constants")
    expected = {
        "_input_name_key": {"list": {"s": ["x"] if input_count == 1 else [f"x{i}" for i in range(input_count)]}},
        "_input_name_value": {"list": {"i": list(range(input_count))}},
        "_output_name_key": {"list": {"s": ["y"]}},
        "_output_name_value": {"list": {"i": [0]}},
        "_ir_outputs_key": {"list": {"s": ["y"]}},
        "_ir_outputs_value": {"list": {"i": [0]}},
        "is_input_const": {"list": {"b": [False] * input_count, "val_type": 4}},
    }
    if input_count == 1:
        expected.update(_ir_inputs_key={"list": {"s": ["x"]}}, _ir_inputs_value={"list": {"i": [0]}})
    else:
        expected["__dynamic_input_x_cnt"] = {"i": 8}
    for key, value in _attrs(node).items():
        if key not in semantic_attrs:
            _require(key in expected and _same(value, expected[key]),
                     "new feature operation contains unsupported GE metadata: " + key)


def _file_constants(value):
    if isinstance(value, dict):
        if value.get("type") == "FileConstant":
            yield value
        for key in sorted(value):
            yield from _file_constants(value[key])
    elif isinstance(value, list):
        for item in value:
            yield from _file_constants(item)


def validate_original_ge_append(original_model, appended_model, *, taps, dtype_enum,
                                sequence_length=None, cast_dst_type=1,
                                expected_file_constants=1153):
    """Validate exactly eight explicit Cast taps followed by one ConcatD.

    ``dtype_enum`` is the caller's selected protobuf schema name-to-number map;
    ``cast_dst_type`` is the independent native GE DataType Cast attribute value
    (native GE DT_FLOAT16 is 1), not a protobuf enum number. No enum is inferred
    from a descriptor's position. Taps must come from independently bound layer
    provenance and contain post_decoder_layer_index/producer_reference/descriptor.
    """
    report = {"schema_version": 1, "artifact_kind": "qwen35-original-ge-append-validation",
              "status": "VALIDATION_FAILED", "stage": "input-metadata",
              "model_compilation": "NOT_RUN", "device_execution": "NOT_RUN",
              "numerical_validation": "NOT_EVALUATED", "weight_payload_access": "NONE",
              "source_tap_provenance": "CALLER_SUPPLIED_NOT_ESTABLISHED_BY_THIS_VALIDATOR",
              "scope": "Exact original decoded metadata preservation and bounded auxiliary feature graph structure only; no model accuracy or device pass"}
    try:
        old_digest, new_digest = _sha(original_model), _sha(appended_model)
        report.update(original_model_metadata_sha256=old_digest, appended_model_metadata_sha256=new_digest)
        original, old_nodes = _root(original_model)
        appended, new_nodes = _root(appended_model)
        _require(isinstance(dtype_enum, dict) and type(dtype_enum.get("DT_FLOAT16")) is int
                 and all(isinstance(key, str) and type(value) is int for key, value in dtype_enum.items()),
                 "selected protobuf dtype_enum must explicitly name DT_FLOAT16")
        float16 = dtype_enum["DT_FLOAT16"]
        _require(type(cast_dst_type) is int and cast_dst_type >= 0, "native GE Cast destination dtype is required")
        _require(type(expected_file_constants) is int and expected_file_constants >= 0,
                 "expected FileConstant count must be a nonnegative integer")
        _require(_same({key: value for key, value in original_model.items() if key != "graph"},
                      {key: value for key, value in appended_model.items() if key != "graph"}),
                 "original ModelDef fields changed")
        _require(_same({key: value for key, value in original.items() if key not in {"op", "output"}},
                      {key: value for key, value in appended.items() if key not in {"op", "output"}}),
                 "original GraphDef metadata changed")
        old_by_name = {node["name"]: node for node in old_nodes}
        new_by_name = {node["name"]: node for node in new_nodes}
        _require(not any(name.startswith(_PREFIX) for name in old_by_name), "reserved feature prefix already exists in original graph")
        required_names = {*_CAST_NAMES, _CONCAT}
        _require(set(new_by_name) - set(old_by_name) == required_names
                 and set(old_by_name) <= set(new_by_name) and len(new_nodes) == len(old_nodes) + 9,
                 "append must add exactly the declared eight Cast nodes and one ConcatD")
        _require([node["name"] for node in new_nodes if node["name"] not in required_names]
                 == [node["name"] for node in old_nodes], "original operation order changed")
        additions = [new_by_name[name] for name in (*_CAST_NAMES, _CONCAT)]
        if any("id" in node for node in old_nodes + additions):
            original_ids = [node.get("id", 0) for node in old_nodes]
            _require(all(type(value) is int and value >= 0 for value in original_ids),
                     "original operation ids must be nonnegative integers")
            base = max([len(old_nodes) - 1, *original_ids]) + 1
            new_ids = [node.get("id") for node in additions]
            _require(all(type(value) is int and value >= base for value in new_ids)
                     and len(set(new_ids)) == 9,
                     "new operation ids must be unique and exceed all original explicit ids and node-index range")
            report["new_operation_ids"] = new_ids
            report["operation_id_policy"] = "NEW_IDS_UNIQUE_ABOVE_ORIGINAL_IDS_AND_INDEX_RANGE"
        else:
            report["operation_id_policy"] = "NO_EXPLICIT_OPERATION_ID_FIELDS_OBSERVED"
        old_inputs = [node for node in old_nodes if node["type"] in {"Data", "RefData"}]
        new_inputs = [node for node in new_nodes if node["type"] in {"Data", "RefData"}]
        refs = [node for node in old_inputs if node["type"] == "RefData"]
        _require(len(old_inputs) == 89 and len(refs) == 80 and _same(old_inputs, new_inputs),
                 "original 89 inputs and 80 RefData must remain unchanged in original order")
        _require([node["name"] for node in refs] == [f"RefData_{index}" for index in range(80)],
                 "original RefData names/order do not match the complete 40-layer state contract")
        embedding = old_by_name.get("arg1_1", {})
        embedding_descs = embedding.get("output_desc", [])
        _require(embedding.get("type") == "Data" and len(embedding_descs) == 1,
                 "original arg1_1 embedding input descriptor is required")
        embedding_shape = _shape(embedding_descs[0])
        _require(len(embedding_shape) == 3 and embedding_shape[0] == 1 and embedding_shape[2] == 2048
                 and 1 <= embedding_shape[1] <= 64, "original embedding input must establish [1,T,2048] with T in 1..64")
        sequence = embedding_shape[1]
        _require(sequence_length is None or (type(sequence_length) is int and sequence_length == sequence),
                 "requested sequence length contradicts the original AIR embedding input")
        report["sequence_length"] = sequence
        netoutputs = [node for node in old_nodes if node["type"] == "NetOutput"]
        _require(len(netoutputs) == 1, "exactly one original NetOutput is required")
        old_net = netoutputs[0]
        new_net = new_by_name[old_net["name"]]
        old_edges, old_descs = old_net.get("input"), old_net.get("input_desc")
        _require(isinstance(old_edges, list) and isinstance(old_descs, list)
                 and len(old_edges) == len(old_descs) == 81, "original NetOutput must retain all 81 logits/state ports")
        feature_edge = _CONCAT + ":0"
        _require(new_net.get("input") == old_edges + [feature_edge], "NetOutput original edges/order changed or extra feature edge differs")
        new_descs = new_net.get("input_desc", [])
        _require(len(new_descs) == 82 and _same(new_descs[:81], old_descs),
                 "NetOutput original descriptors/order changed")
        _require(_same({key: value for key, value in old_net.items() if key not in {"input", "input_desc"}},
                      {key: value for key, value in new_net.items() if key not in {"input", "input_desc"}}),
                 "NetOutput fields changed outside the single appended port")
        _descriptor(new_descs[-1], dtype=float16, shape=[1, sequence, 16384], where="feature NetOutput")
        if "output" in original:
            _require(isinstance(original["output"], list) and appended.get("output") == original["output"] + [feature_edge],
                     "GraphDef.output must preserve the complete original prefix and append only the feature edge")
        else:
            _require("output" not in appended, "GraphDef.output was introduced where original graph had none")
        report["stage"] = "original-preservation"
        for node in old_nodes:
            if node is not old_net:
                _require(_same(node, new_by_name[node["name"]]), "original operation changed: " + node["name"])
        old_files = list(_file_constants(_canonical(original_model)))
        new_files = list(_file_constants(_canonical(appended_model)))
        _require(len(old_files) == expected_file_constants and len(new_files) == expected_file_constants,
                 "FileConstant count changed or original count differs from the explicit expected count")
        for index, (before, after) in enumerate(zip(old_files, new_files)):
            _require(_same(before, after), "FileConstant metadata changed at ordered index " + str(index))
        report["file_constants"] = {"count": len(old_files), "matched": len(old_files),
                                    "ordered_metadata_sha256": _sha(old_files)}
        report["original_preservation"] = {"nodes": len(old_nodes), "inputs": 89, "refdata": 80,
                                           "netoutput_ports": 81, "status": "ALL_ORIGINAL_FIELDS_PRESERVED"}
        report["stage"] = "feature-structure"
        _require(isinstance(taps, (list, tuple)) and len(taps) == 8, "eight explicit layer tap bindings are required")
        _require(all(isinstance(row, dict) for row in taps), "each tap binding must be a dictionary")
        _require(all(type(row.get("post_decoder_layer_index")) is int for row in taps),
                 "tap layer indices must be explicit integers")
        _require([row.get("post_decoder_layer_index") for row in taps] == list(LAYER_IDS),
                 "tap layer indices must retain the locked eight-layer order")
        references = []
        for index, tap in enumerate(taps):
            reference = tap.get("producer_reference")
            name, port = _edge(reference)
            _require(name in old_by_name and old_by_name[name]["type"] not in {"Data", "RefData", "FileConstant", "NetOutput"},
                     "feature tap must bind an original intermediate producer")
            producer_descs = old_by_name[name].get("output_desc", [])
            _require(port < len(producer_descs), "tap producer output port does not exist")
            producer_desc = producer_descs[port]
            declared = tap.get("descriptor", {})
            _require(isinstance(declared, dict) and type(declared.get("raw_dtype")) is int
                     and declared.get("raw_dtype") == _dtype(producer_desc)
                     and declared.get("shape") == _shape(producer_desc) and declared.get("layout") == producer_desc.get("layout"),
                     "caller tap descriptor differs from original producer metadata")
            source_dtype = _dtype(producer_desc)
            selected_names = sorted(key for key, value in dtype_enum.items() if value == source_dtype)
            _require(sorted(declared.get("schema_dtype_names", [])) == selected_names
                     and any(name in {"DT_FLOAT", "DT_FLOAT32", "DT_FLOAT16"} for name in selected_names),
                     "tap dtype names differ from the selected schema or are not FP32/FP16")
            _require(_shape(producer_desc) in ([], [1, sequence, 2048]) and producer_desc.get("layout") == "ND",
                     "original tap descriptor has unsupported shape/layout")
            cast = new_by_name[_CAST_NAMES[index]]
            _require(cast["type"] == "Cast" and cast.get("input") == [reference], "feature Cast input was rewired or changed type")
            _require(len(cast.get("input_desc", [])) == len(cast.get("output_desc", [])) == 1,
                     "feature Cast must have one typed input and output")
            _descriptor(cast["input_desc"][0], dtype=source_dtype, shape=[1, sequence, 2048], where="feature Cast input")
            _descriptor(cast["output_desc"][0], dtype=float16, shape=[1, sequence, 2048], where="feature Cast output")
            attrs = _attrs(cast)
            _attr_int(attrs, "dst_type", cast_dst_type, "feature Cast")
            _new_node_metadata(cast, 1, {"dst_type"})
            references.append(reference)
        _require(len(set(references)) == 8, "eight tap producer references must be distinct")
        concat = new_by_name[_CONCAT]
        _require(concat["type"] == "ConcatD" and concat.get("input") == [name + ":0" for name in _CAST_NAMES],
                 "feature ConcatD inputs/type differ from the ordered eight casts")
        concat_inputs, concat_outputs = concat.get("input_desc", []), concat.get("output_desc", [])
        _require(len(concat_inputs) == 8 and len(concat_outputs) == 1, "feature ConcatD must expose eight inputs and one output")
        for desc in concat_inputs:
            _descriptor(desc, dtype=float16, shape=[1, sequence, 2048], where="feature ConcatD input")
        _descriptor(concat_outputs[0], dtype=float16, shape=[1, sequence, 16384], where="feature ConcatD output")
        _require(_same(new_descs[-1], concat_outputs[0]), "feature NetOutput descriptor differs from ConcatD producer")
        attrs = _attrs(concat)
        _attr_int(attrs, "concat_dim", 2, "feature ConcatD")
        _attr_int(attrs, "N", 8, "feature ConcatD")
        _new_node_metadata(concat, 8, {"concat_dim", "N"})
        report.update(status="METADATA_APPEND_VALIDATED", stage="complete",
                      new_node_metadata_policy="CANN9_1_BRANCH_TEMPLATE_VALUES_CHECKED",
                      feature={"output_index": 81, "reference": feature_edge, "dtype": "float16",
                               "protobuf_dtype": float16, "shape": [1, sequence, 16384],
                               "layer_indices": list(LAYER_IDS), "producer_references": references},
                      added_nodes=9, original_op_count=len(old_nodes), appended_op_count=len(new_nodes))
    except (KeyError, TypeError, ValueError, RecursionError) as error:
        report["error"] = f"{type(error).__name__}: {str(error)[:1000]}"
    return report
