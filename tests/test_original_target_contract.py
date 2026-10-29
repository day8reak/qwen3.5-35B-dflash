"""Boundary observations are never an original-target runtime acceptance."""
from copy import deepcopy
import json

import pytest

from qwen35_moe_dflash.existing_air_profile import get_existing_air_profile
from qwen35_moe_dflash.original_target_contract import capture_runtime_sources, summarize_boundaries


ENUMS = {"dtype": {"values": {"DT_FLOAT16": 77, "DT_INT64": 91}}}


def graph():
    profile = get_existing_air_profile("cpro-40block-64-1")
    nodes = []
    for index, entry in enumerate(profile["input_shape"].split(";")):
        name, raw = entry.split(":")
        desc = {"dtype": 77, "shape": {"dim": [int(x) for x in raw.split(",")]}, "layout": "ND"}
        nodes.append({"name": name, "type": "RefData" if name.startswith("RefData_") else "Data",
                      "attr": [{"key": "index", "value": {"i": index}}], "output_desc": [desc]})
    desc = {"dtype": 91, "shape": {"dim": [1]}, "layout": "ND"}
    nodes.extend([{"name": "body", "type": "GraphOp", "attr": [{"key": "g", "value": {"g": {"op": [
        {"name": "inner_data", "type": "Data"}, {"name": "inner_ref", "type": "RefData"}]}}}], "output_desc": [desc]},
        {"name": "output", "type": "NetOutput", "input": ["body:0"], "input_desc": [desc]}])
    return {"graph": [{"name": "original", "op": nodes}]}, profile


def test_real_indices_and_selected_schema_numbers_are_retained_without_assuming_runtime_abi():
    document, profile = graph()
    document["graph"][0]["op"][:89] = list(reversed(document["graph"][0]["op"][:89]))
    summary = summarize_boundaries(document, ENUMS, profile)
    assert summary["input_type_counts"] == {"Data": 9, "RefData": 80}
    assert summary["declared_profile_check"]["status"] == "MATCHED_VISIBLE_BOUNDARIES"
    assert summary["inputs_in_serialized_op_order"][0]["name"] == "arg18_1"
    assert summary["inputs_in_serialized_op_order"][0]["serialized_data_index"] == 88
    assert summary["outputs"][0]["schema_dtype_names"] == ["DT_INT64"]
    assert summary["runtime_input_order"] == summary["refdata_persistence"] == "UNVERIFIED"
    assert summary["dynamic_gear_execution"] == "NOT_RUN"


@pytest.mark.parametrize("change", ["missing", "shape", "refdata_type"])
def test_declared_shape_table_cannot_supply_missing_or_different_compiled_boundaries(change):
    document, profile = graph()
    node = document["graph"][0]["op"][3]
    if change == "missing":
        document["graph"][0]["op"].remove(node)
    elif change == "shape":
        node["output_desc"][0]["shape"]["dim"][-1] = 999
    else:
        node["type"] = "Data"
    summary = summarize_boundaries(document, ENUMS, profile)
    assert summary["declared_profile_check"]["status"] == "DIFFERENT_VISIBLE_BOUNDARIES"
    assert summary["runtime_input_order"] == "UNVERIFIED"


def test_unobserved_indices_or_unknown_dtype_are_explicit_not_filled_from_shape_preset():
    document, profile = graph()
    node = document["graph"][0]["op"][0]
    del node["attr"]
    node["output_desc"][0]["dtype"] = 1
    summary = summarize_boundaries(document, ENUMS, profile)
    assert summary["serialized_input_indices"] == "INCOMPLETE_OR_NONCONTIGUOUS"
    observed = summary["inputs_in_serialized_op_order"][0]
    assert observed["serialized_data_index"] is None
    assert observed["raw_dtype"] == 1 and observed["schema_dtype_names"] == []
    assert observed["dtype_status"] == "UNRESOLVED"


@pytest.mark.parametrize("change", ["ambiguous_graph", "duplicate_node", "producer_dtype", "absent_producer"])
def test_ambiguous_outputs_or_graphs_do_not_produce_a_valid_boundary_summary(change):
    document, profile = graph()
    ops = document["graph"][0]["op"]
    if change == "ambiguous_graph":
        document["graph"].append(deepcopy(document["graph"][0]))
    elif change == "duplicate_node":
        ops.append(deepcopy(ops[0]))
    elif change == "producer_dtype":
        ops[-1]["input_desc"] = [{"dtype": 77, "shape": {"dim": [1]}, "layout": "ND"}]
    else:
        ops[-1]["input"] = ["missing:0"]
    with pytest.raises(ValueError):
        summarize_boundaries(document, ENUMS, profile)


def test_runtime_sources_are_read_locally_without_importing_the_original_project(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    source = tmp_path / "original"
    source.mkdir()
    (source / "run.py").write_text("raise RuntimeError('MUST NOT IMPORT')\nexample_inputs = [past_key_values]\n")
    (source / "weights").mkdir()
    (source / "weights/ignored.py").write_text("RefData = 'must not scan weights'\n")
    outside = tmp_path / "outside.py"
    outside.write_text("RefData = 'must not follow symlinks'\n")
    (source / "linked.py").symlink_to(outside)
    include = tmp_path / "include"
    include.mkdir()
    (include / "HiAiModelManagerService.h").write_text("class Manager { void SetDynamicDims(); };\n")
    result = capture_runtime_sources(tmp_path / "observed", include=include, source_root=source)
    assert result["status"] == "INCOMPLETE"  # Other selected headers are absent.
    assert [row["path"] for row in result["sources"]] == ["run.py"]
    assert result["headers"][0]["lines"][0]["text"].endswith("SetDynamicDims(); };")
    assert json.loads((tmp_path / "observed/report.json").read_text()) == result
    assert not (source / "__pycache__").exists()
