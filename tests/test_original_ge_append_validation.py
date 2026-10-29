"""Host metadata fixtures; no vendor code, weights, AIR writer or device run."""
from copy import deepcopy
import json

import pytest

from qwen35_moe_dflash.existing_air_profile import get_existing_air_profile
from qwen35_moe_dflash.original_ge_append_validation import LAYER_IDS, validate_original_ge_append


ENUM = {"DT_UNDEFINED": 0, "DT_FLOAT": 1, "DT_FLOAT16": 2, "DT_INT32": 7}


def desc(shape, dtype=2):
    return {"dtype": dtype, "shape": {"dim": list(shape)}, "layout": "ND"}


def attr(name, value):
    return {"key": name, "value": {"i": value}}


def models(sequence=55, graph_outputs=False):
    nodes, state_descs = [], []
    profile = get_existing_air_profile("cpro-40block-64-1")
    for index, entry in enumerate(profile["input_shape"].split(";")):
        name, shape = entry.split(":")
        shape = [sequence if int(size) == -1 else int(size) for size in shape.split(",")]
        is_state = name.startswith("RefData_")
        descriptor = desc(shape)
        nodes.append({"name": name, "type": "RefData" if is_state else "Data",
                      "output_desc": [descriptor], "attr": [attr("index", index)]})
        if is_state:
            state_descs.append(descriptor)
    taps = []
    for index, layer in enumerate(LAYER_IDS):
        name = f"Add_{index + 300}"
        nodes.append({"name": name, "type": "Add", "input": ["arg1_1:0", "arg1_1:0"],
                      "input_desc": [desc([], 1), desc([], 1)], "output_desc": [desc([], 1)],
                      "attr": [attr("must_not_change", index)]})
        taps.append({"post_decoder_layer_index": layer, "producer_reference": name + ":0",
                     "descriptor": {"raw_dtype": 1, "schema_dtype_names": ["DT_FLOAT"],
                                    "layout": "ND", "shape": []}})
    for index in range(1153):
        nodes.append({"name": f"file_{index}", "type": "FileConstant", "output_desc": [desc([2048])],
                      "attr": [{"key": "file_path", "value": {"s": f"./old_export/weight_{index}"}},
                               attr("offset", index * 4096), attr("length", 4096)]})
    output_descs = [desc([1, 1, 248320], 1), *state_descs]
    nodes.append({"name": "body", "type": "OriginalOpaqueBody", "output_desc": output_descs})
    output_edges = [f"body:{index}" for index in range(81)]
    nodes.append({"name": "Node_Output", "type": "NetOutput", "input": output_edges,
                  "input_desc": deepcopy(output_descs), "attr": [attr("must_preserve", 19)]})
    original = {"name": "original-model", "version": 1,
                "graph": [{"name": "full-target", "op": nodes,
                           "attr": [attr("keep-b", 2), attr("keep-a", 1)]}]}
    if graph_outputs:
        original["graph"][0]["output"] = deepcopy(output_edges)
    appended = deepcopy(original)
    graph = appended["graph"][0]
    net = graph["op"][-1]
    net["input"].append("__dflash_features/concat:0")
    net["input_desc"].append(desc([1, sequence, 16384]))
    for index, tap in enumerate(taps):
        graph["op"].append({"name": f"__dflash_features/cast_{index}", "type": "Cast",
                            "input": [tap["producer_reference"]], "input_desc": [desc([1, sequence, 2048], 1)],
                            "output_desc": [desc([1, sequence, 2048])], "attr": [attr("dst_type", 1)]})
    graph["op"].append({"name": "__dflash_features/concat", "type": "ConcatD",
                        "input": [f"__dflash_features/cast_{index}:0" for index in range(8)],
                        "input_desc": [desc([1, sequence, 2048]) for _ in range(8)],
                        "output_desc": [desc([1, sequence, 16384])],
                        "attr": [attr("concat_dim", 2), attr("N", 8)]})
    if graph_outputs:
        graph["output"].append("__dflash_features/concat:0")
    return original, appended, taps


def validate(old, new, taps, **options):
    return validate_original_ge_append(old, new, taps=taps, dtype_enum=ENUM, **options)


def test_only_append_is_validated_all_fileconstants_and_original_fields_are_checked():
    old, new, taps = models()
    before = json.dumps((old, new, taps), sort_keys=True)
    result = validate(old, new, taps)
    assert result["status"] == "METADATA_APPEND_VALIDATED"
    assert result["feature"]["shape"] == [1, 55, 16384]
    assert result["file_constants"]["count"] == result["file_constants"]["matched"] == 1153
    assert result["original_preservation"]["inputs"] == 89
    assert result["original_preservation"]["refdata"] == 80
    assert result["appended_op_count"] == result["original_op_count"] + 9
    assert result["model_compilation"] == result["device_execution"] == "NOT_RUN"
    assert result["numerical_validation"] == "NOT_EVALUATED"
    assert result["source_tap_provenance"].startswith("CALLER_SUPPLIED")
    assert json.dumps((old, new, taps), sort_keys=True) == before


@pytest.mark.parametrize("sequence", [1, 55, 64])
def test_sequence_is_inferred_from_original_embedding_input(sequence):
    old, new, taps = models(sequence)
    assert validate(old, new, taps)["sequence_length"] == sequence
    assert validate(old, new, taps, sequence_length=sequence)["status"] == "METADATA_APPEND_VALIDATED"
    result = validate(old, new, taps, sequence_length=64 if sequence != 64 else 55)
    assert result["status"] == "VALIDATION_FAILED" and "sequence length" in result["error"]


def test_protobuf_map_order_is_the_only_ignored_order():
    old, new, taps = models()
    new["graph"][0]["attr"].reverse()
    for node in new["graph"][0]["op"]:
        if "attr" in node:
            node["attr"].reverse()
    assert validate(old, new, taps)["status"] == "METADATA_APPEND_VALIDATED"


@pytest.mark.parametrize("kind", ["model", "graph", "file_path", "file_extent", "node_shape", "node_attr", "net_attr", "net_original_desc", "net_rewire", "input_order", "ref_reorder", "extra_node", "missing_node"])
def test_every_unrelated_original_mutation_is_rejected(kind):
    old, new, taps = models()
    graph = new["graph"][0]
    nodes = {node["name"]: node for node in graph["op"]}
    if kind == "model":
        new["version"] = 2
    elif kind == "graph":
        graph["attr"].append(attr("added", 1))
    elif kind == "file_path":
        nodes["file_923"]["attr"][0]["value"]["s"] = "./other/weight"
    elif kind == "file_extent":
        nodes["file_1152"]["attr"][2]["value"]["i"] = 8192
    elif kind == "node_shape":
        nodes["Add_307"]["output_desc"][0]["shape"]["dim"] = [1, 55, 2048]
    elif kind == "node_attr":
        nodes["Add_303"]["attr"][0]["value"]["i"] = 123
    elif kind == "net_attr":
        nodes["Node_Output"]["attr"].append(attr("new", 1))
    elif kind == "net_original_desc":
        nodes["Node_Output"]["input_desc"][25]["dtype"] = 1
    elif kind == "net_rewire":
        nodes["Node_Output"]["input"][1] = "body:2"
    elif kind == "input_order":
        graph["op"][0], graph["op"][1] = graph["op"][1], graph["op"][0]
    elif kind == "ref_reorder":
        nodes["RefData_79"]["attr"][0]["value"]["i"] = 0
    elif kind == "extra_node":
        graph["op"].append({"name": "unauthorized", "type": "Identity"})
    else:
        graph["op"].remove(nodes["file_0"])
    result = validate(old, new, taps)
    assert result["status"] == "VALIDATION_FAILED" and result["error"]


@pytest.mark.parametrize("kind", ["cast_input", "cast_type", "cast_dtype", "cast_shape", "cast_native_enum", "cast_extra_attr", "concat_order", "concat_axis", "concat_N", "concat_type", "concat_dtype", "concat_shape", "extra_feature_output", "feature_desc", "feature_name"])
def test_feature_operator_changes_and_unknown_rewires_are_rejected(kind):
    old, new, taps = models()
    nodes = {node["name"]: node for node in new["graph"][0]["op"]}
    cast, concat, net = nodes["__dflash_features/cast_0"], nodes["__dflash_features/concat"], nodes["Node_Output"]
    if kind == "cast_input":
        cast["input"] = ["Add_301:0"]
    elif kind == "cast_type":
        cast["type"] = "Identity"
    elif kind == "cast_dtype":
        cast["output_desc"][0]["dtype"] = 1
    elif kind == "cast_shape":
        cast["input_desc"][0]["shape"]["dim"] = [1, 55, 1024]
    elif kind == "cast_native_enum":
        cast["attr"][0]["value"]["i"] = 2
    elif kind == "cast_extra_attr":
        cast["attr"].append(attr("unexpected", 1))
    elif kind == "concat_order":
        concat["input"][0], concat["input"][1] = concat["input"][1], concat["input"][0]
    elif kind == "concat_axis":
        concat["attr"][0]["value"]["i"] = 1
    elif kind == "concat_N":
        concat["attr"][1]["value"]["i"] = 9
    elif kind == "concat_type":
        concat["type"] = "Concat"
    elif kind == "concat_dtype":
        concat["input_desc"][6]["dtype"] = 1
    elif kind == "concat_shape":
        concat["output_desc"][0]["shape"]["dim"] = [1, 16384, 55]
    elif kind == "extra_feature_output":
        net["input"].append("__dflash_features/cast_0:0")
        net["input_desc"].append(desc([1, 55, 2048]))
    elif kind == "feature_desc":
        net["input_desc"][-1]["layout"] = "NCHW"
    else:
        cast["name"] = "__dflash_features/unreviewed"
    result = validate(old, new, taps)
    assert result["status"] == "VALIDATION_FAILED" and result["error"]


def test_graph_output_existing_prefix_and_absent_field_rules():
    old, new, taps = models(graph_outputs=True)
    assert validate(old, new, taps)["status"] == "METADATA_APPEND_VALIDATED"
    new["graph"][0]["output"][0] = "body:80"
    assert validate(old, new, taps)["status"] == "VALIDATION_FAILED"
    old, new, taps = models()
    new["graph"][0]["output"] = ["__dflash_features/concat:0"]
    assert validate(old, new, taps)["status"] == "VALIDATION_FAILED"


@pytest.mark.parametrize("kind", ["duplicate_name", "duplicate_attr", "tap_order", "tap_port", "tap_dtype", "tap_raw_shape", "tap_constant", "same_tap", "tap_bool_layer", "tap_bool_dtype", "wrong_file_count", "no_schema_enum", "non_model_dict"])
def test_ambiguous_or_incomplete_provenance_cannot_pass(kind):
    old, new, taps = models()
    options = {}
    if kind == "duplicate_name":
        new["graph"][0]["op"][-1]["name"] = "arg1_1"
    elif kind == "duplicate_attr":
        new["graph"][0]["op"][1]["attr"] *= 2
    elif kind == "tap_order":
        taps.reverse()
    elif kind == "tap_port":
        taps[0]["producer_reference"] = "Add_300:1"
    elif kind == "tap_dtype":
        taps[0]["descriptor"]["schema_dtype_names"] = ["DT_FLOAT16"]
    elif kind == "tap_raw_shape":
        taps[0]["descriptor"]["shape"] = [1, 55, 2048]
    elif kind == "tap_constant":
        taps[0]["producer_reference"] = "file_0:0"
    elif kind == "same_tap":
        taps[1]["producer_reference"] = taps[0]["producer_reference"]
    elif kind == "tap_bool_layer":
        taps[0]["post_decoder_layer_index"] = True
    elif kind == "tap_bool_dtype":
        taps[0]["descriptor"]["raw_dtype"] = True
    elif kind == "wrong_file_count":
        options["expected_file_constants"] = 1152
    elif kind == "no_schema_enum":
        result = validate_original_ge_append(old, new, taps=taps, dtype_enum={})
        assert result["status"] == "VALIDATION_FAILED"
        return
    else:
        old = old["graph"][0]
    assert validate(old, new, taps, **options)["status"] == "VALIDATION_FAILED"


def test_native_cast_and_protobuf_enum_must_not_be_conflated():
    old, new, taps = models()
    assert ENUM["DT_FLOAT16"] == 2
    assert validate(old, new, taps, cast_dst_type=1)["status"] == "METADATA_APPEND_VALIDATED"
    assert validate(old, new, taps, cast_dst_type=2)["status"] == "VALIDATION_FAILED"


def test_entire_nested_graph_fileconstant_metadata_is_compared():
    old, new, taps = models()
    nested = {"name": "nested", "op": [{"name": "nested_file", "type": "FileConstant",
              "attr": [{"key": "file_path", "value": {"s": "nested.bin"}}]}]}
    old["graph"][0]["attr"].append({"key": "nested", "value": {"g": deepcopy(nested)}})
    new["graph"][0]["attr"].append({"key": "nested", "value": {"g": deepcopy(nested)}})
    # Wire/map entry and dictionary field order are not metadata changes.
    new["graph"][0]["attr"].reverse()
    new["graph"][0] = dict(reversed(list(new["graph"][0].items())))
    assert validate(old, new, taps, expected_file_constants=1154)["status"] == "METADATA_APPEND_VALIDATED"
    next(row for row in new["graph"][0]["attr"] if row["key"] == "nested")["value"]["g"]["op"][0]["attr"][0]["value"]["s"] = "replaced.bin"
    assert validate(old, new, taps, expected_file_constants=1154)["status"] == "VALIDATION_FAILED"


def factory_metadata(new):
    """Metadata values observed in the retained real CANN 9.1 branch template."""
    graph = new["graph"][0]
    base = len(graph["op"]) - 9
    for node in graph["op"]:
        if node["name"].startswith("__dflash_features/"):
            index = int(node["name"].rsplit("_", 1)[-1]) if node["type"] == "Cast" else 8
            node["id"] = base + index
    for node in graph["op"]:
        if node["name"] not in {"__dflash_features/cast_0", "__dflash_features/concat"}:
            continue
        count = 1 if node["type"] == "Cast" else 8
        node.update(has_out_attr=True, is_input_const=[False] * count)
        values = {
            "_output_name_key": {"list": {"s": ["y"]}},
            "_output_name_value": {"list": {"i": [0]}},
            "_ir_outputs_key": {"list": {"s": ["y"]}},
            "_ir_outputs_value": {"list": {"i": [0]}},
            "_input_name_key": {"list": {"s": ["x"] if count == 1 else [f"x{i}" for i in range(8)]}},
            "_input_name_value": {"list": {"i": list(range(count))}},
            "is_input_const": {"list": {"b": [False] * count, "val_type": 4}},
        }
        if count == 1:
            values.update(_ir_inputs_key={"list": {"s": ["x"]}}, _ir_inputs_value={"list": {"i": [0]}})
        else:
            values["__dynamic_input_x_cnt"] = {"i": 8}
        node["attr"].extend({"key": key, "value": value} for key, value in values.items())
        for descriptor in node["input_desc"] + node["output_desc"]:
            descriptor.update(device_type="NPU", real_dim_cnt=3, attr=[
                {"key": "origin_shape_initialized", "value": {"b": False}},
                {"key": "origin_format", "value": {"s": "ND"}},
                {"key": "origin_format_for_int", "value": {"i": 2}},
                {"key": "format_for_int", "value": {"i": 2}},
                {"key": "placement", "value": {"i": 0}},
                {"key": "shape_range", "value": {"list_list_int": {}}},
            ])
    concat = next(node for node in graph["op"] if node["name"] == "__dflash_features/concat")
    net = next(node for node in graph["op"] if node["type"] == "NetOutput")
    net["input_desc"][-1] = deepcopy(concat["output_desc"][0])


def test_real_factory_metadata_is_value_checked_and_canonical_map_order_is_allowed():
    old, new, taps = models()
    factory_metadata(new)
    result = validate(old, new, taps)
    assert result["status"] == "METADATA_APPEND_VALIDATED"
    assert result["new_node_metadata_policy"] == "CANN9_1_BRANCH_TEMPLATE_VALUES_CHECKED"


@pytest.mark.parametrize("mutation", ["unknown_top_field", "has_out_attr", "negative_id", "colliding_id", "duplicate_new_id", "missing_new_id", "const_input", "attribute_const_input", "port_names", "dynamic_count", "unknown_attr", "unknown_desc_field", "origin_format", "storage_format", "placement", "shape_range", "device_type", "real_dim_count"])
def test_factory_metadata_is_not_a_blanket_exception_for_unknown_or_changed_values(mutation):
    old, new, taps = models()
    factory_metadata(new)
    nodes = {node["name"]: node for node in new["graph"][0]["op"]}
    cast, concat = nodes["__dflash_features/cast_0"], nodes["__dflash_features/concat"]
    descriptor = cast["input_desc"][0]
    if mutation == "unknown_top_field":
        cast["engine_name"] = "unexpected-compute-engine"
    elif mutation == "has_out_attr":
        cast["has_out_attr"] = False
    elif mutation == "negative_id":
        cast["id"] = -1
    elif mutation == "colliding_id":
        cast["id"] = 8
    elif mutation == "duplicate_new_id":
        cast["id"] = concat["id"]
    elif mutation == "missing_new_id":
        cast.pop("id")
    elif mutation == "const_input":
        cast["is_input_const"] = [True]
    elif mutation == "attribute_const_input":
        next(row for row in cast["attr"] if row["key"] == "is_input_const")["value"]["list"]["b"] = [True]
    elif mutation == "port_names":
        next(row for row in cast["attr"] if row["key"] == "_input_name_key")["value"]["list"]["s"] = ["wrong"]
    elif mutation == "dynamic_count":
        next(row for row in concat["attr"] if row["key"] == "__dynamic_input_x_cnt")["value"]["i"] = 7
    elif mutation == "unknown_attr":
        cast["attr"].append(attr("new_semantic_attribute", 1))
    elif mutation == "unknown_desc_field":
        descriptor["reuse_input_index"] = 0
    elif mutation == "origin_format":
        next(row for row in descriptor["attr"] if row["key"] == "origin_format")["value"]["s"] = "NCHW"
    elif mutation == "storage_format":
        next(row for row in descriptor["attr"] if row["key"] == "format_for_int")["value"]["i"] = 29
    elif mutation == "placement":
        next(row for row in descriptor["attr"] if row["key"] == "placement")["value"]["i"] = 1
    elif mutation == "shape_range":
        next(row for row in descriptor["attr"] if row["key"] == "shape_range")["value"]["list_list_int"] = {"list_list_i": [55]}
    elif mutation == "device_type":
        descriptor["device_type"] = "CPU"
    else:
        descriptor["real_dim_cnt"] = 4
    assert validate(old, new, taps)["status"] == "VALIDATION_FAILED"
