"""Exact CPU partition-port provenance; no descriptor-only or NPU shortcuts."""
from copy import deepcopy

import pytest

from qwen35_moe_dflash.draft_precision_edges import PartitionEdgeResolver, summarize_precision_failures


def _desc(shape=(1, 1, 8), dtype=1):
    return {"dtype": dtype, "shape": {"dim": list(shape)}, "layout": "ND"}


def _node(name, kind, inputs=(), input_desc=(), output_desc=()):
    result = {"name": name, "type": kind, "input": list(inputs),
              "input_desc": list(input_desc), "output_desc": list(output_desc)}
    if kind == "GraphOp":
        result["attr"] = [{"key": "cl_name", "value": {"s": "CPUCL"}}]
    return result


def _observe(desc, enums):
    return {"dtype": enums.get(desc.get("dtype")), "shape": desc.get("shape", {}).get("dim")}


@pytest.fixture
def case():
    shape = _desc()
    # Packed ports are deliberately reversed in the main consumer GraphOp.
    # Equal shapes alone cannot distinguish the two original producers.
    source = {"owner": "CPU_source", "cl_name": "CPUCL", "descriptor_enum_fields": {1: "float32", 2: "float16"},
        "graph": {"name": "CPU_source", "op": [
            _node("arbitrary_input_name", "Data", [""], [shape], [shape]),
            _node("first_flat", "Reshape", ["arbitrary_input_name:0"], [shape], [shape]),
            _node("second_flat", "Reshape", ["arbitrary_input_name:0"], [shape], [shape]),
            _node("arbitrary_output_name", "NetOutput", ["first_flat:0", "second_flat:0"], [shape, shape], [shape, shape])]}}
    consumer = {"owner": "CPU_consumer", "cl_name": "CPUCL", "descriptor_enum_fields": {1: "float32", 2: "float16"},
        "graph": {"name": "CPU_consumer", "op": [
            _node("consumer_packed_data", "Data", [""], [shape, shape], [shape, shape]),
            _node("consumer", "ConcatD", ["consumer_packed_data:0", "consumer_packed_data:1"], [shape, shape], [_desc((1, 1, 16))]),
            _node("consumer_output", "NetOutput", ["consumer:0"], [_desc((1, 1, 16))], [_desc((1, 1, 16))])]}}
    main = {"name": "main", "op": [
        _node("external", "Data", [""], [shape], [shape]),
        _node("CPU_source", "GraphOp", ["external:0"], [shape], [shape, shape]),
        _node("CPU_consumer", "GraphOp", ["CPU_source:1", "CPU_source:0"], [shape, shape], [_desc((1, 1, 16))])]}
    expected = {"producer": "second_flat", "producer_output_index": 0, "dtype": "float32", "shape": [1, 1, 8]}
    return main, [source, consumer], expected


def _resolve(case):
    main, cpu, expected = case
    return PartitionEdgeResolver(main, cpu, _observe).resolve("CPU_consumer", "consumer_packed_data", 0, expected)


def test_exact_ordered_cross_partition_path_matches_without_naming_conventions(case):
    report = _resolve(case)
    assert report["status"] == "MATCHED", report
    assert report["producer"] == "second_flat" and report["owner"] == "CPU_source"
    assert [point["kind"] for point in report["path"]] == [
        "CPU_OUTPUT", "MAIN_GRAPHOP_INPUT", "MAIN_GRAPHOP_OUTPUT", "CPU_NETOUTPUT_INPUT", "CPU_OUTPUT"]


def test_identical_descriptor_of_a_different_source_is_rejected(case):
    case[2]["producer"] = "first_flat"
    assert _resolve(case)["status"] == "UNRESOLVED"


@pytest.mark.parametrize("damage", ["missing_shape", "half", "unknown_dtype", "different_shape",
                                     "missing_data", "two_data", "missing_netout", "two_netout",
                                     "missing_owner", "non_graphop", "npu_owner", "wrong_count",
                                     "bad_main_port", "bad_netout_port", "extra_cast", "unresolved_edge",
                                     "stale_metadata", "self_cycle"])
def test_unproven_cross_partition_paths_are_rejected(case, damage):
    main, cpu, expected = case
    source_ops = cpu[0]["graph"]["op"]
    consumer_ops = cpu[1]["graph"]["op"]
    if damage == "missing_shape":
        consumer_ops[0]["output_desc"][0] = {"dtype": 1}
    elif damage == "half":
        consumer_ops[0]["output_desc"][0] = _desc(dtype=2)
    elif damage == "unknown_dtype":
        consumer_ops[0]["output_desc"][0] = _desc(dtype=99)
    elif damage == "different_shape":
        consumer_ops[0]["output_desc"][0] = _desc((1, 8, 1))
    elif damage == "missing_data":
        consumer_ops.pop(0)
    elif damage == "two_data":
        consumer_ops.append(_node("extra_data", "Data", [""], [_desc(), _desc()], [_desc(), _desc()]))
    elif damage == "missing_netout":
        source_ops.pop()
    elif damage == "two_netout":
        source_ops.append(_node("extra_netout", "NetOutput", ["first_flat:0", "second_flat:0"], [_desc(), _desc()], [_desc(), _desc()]))
    elif damage == "missing_owner":
        main["op"].pop(1)
    elif damage == "non_graphop":
        main["op"][1]["type"] = "CastT"
    elif damage == "npu_owner":
        cpu.pop(0)
    elif damage == "wrong_count":
        source_ops[-1]["output_desc"].pop()
    elif damage == "bad_main_port":
        main["op"][-1]["input"][0] = "CPU_source:3"
    elif damage == "bad_netout_port":
        source_ops[-1]["input"][1] = "second_flat:1"
    elif damage == "extra_cast":
        source_ops.append(_node("hidden_cast", "CastT", ["second_flat:0"], [_desc()], [_desc()]))
        source_ops[-2]["input"][1] = "hidden_cast:0"
    elif damage == "unresolved_edge":
        main["op"][-1]["input"][0] = "malformed"
    elif damage == "stale_metadata":
        main["op"][-1].update(input=["malformed"], src_name=["CPU_source", "CPU_source"], src_index=[1, 0])
    elif damage == "self_cycle":
        main["op"][-1]["input"][0] = "CPU_consumer:0"
    report = _resolve(case)
    assert report["status"] == "UNRESOLVED", report
    assert report["reason"]


def test_second_cpu_passthrough_partition_is_followed_with_full_port_checks(case):
    main, cpu, expected = case
    shape = _desc()
    bridge = {"owner": "CPU_bridge", "cl_name": "CPUCL", "descriptor_enum_fields": {1: "float32", 2: "float16"},
              "graph": {"name": "CPU_bridge", "op": [
                  _node("bridge_data", "Data", [""], [shape], [shape]),
                  _node("bridge_netout", "NetOutput", ["bridge_data:0"], [shape], [shape])]}}
    cpu.append(bridge)
    main["op"].append(_node("CPU_bridge", "GraphOp", ["CPU_source:1"], [shape], [shape]))
    main["op"][2]["input"][0] = "CPU_bridge:0"
    report = _resolve(case)
    assert report["status"] == "MATCHED", report
    assert sum(point["kind"] == "MAIN_GRAPHOP_INPUT" for point in report["path"]) == 2


def test_fp32_path_does_not_ignore_a_half_port_inside_an_upstream_interface(case):
    main, cpu, expected = case
    cpu[0]["graph"]["op"][-1]["input_desc"][1] = _desc(dtype=2)
    assert _resolve(case)["status"] == "UNRESOLVED"


def test_interface_validation_is_lazy_for_existing_same_partition_context(case):
    main, cpu, expected = case
    main["op"] = [_node("other", "Data")]
    # Construction remains useful to a caller whose original same-region
    # checks need no cross-partition resolution. A path still fails closed.
    resolver = PartitionEdgeResolver(main, cpu, _observe)
    assert resolver.resolve("CPU_consumer", "consumer_packed_data", 0, expected)["status"] == "UNRESOLVED"


@pytest.mark.parametrize("damage", ["missing_cl", "npu_cl", "duplicate_cl", "different_layout", "missing_layout"])
def test_owned_fp32_path_requires_cpu_library_and_explicit_nd_layout(case, damage):
    main, cpu, expected = case
    if damage == "missing_cl":
        main["op"][1]["attr"] = []
    elif damage == "npu_cl":
        main["op"][1]["attr"][0]["value"]["s"] = "NPUCL"
    elif damage == "duplicate_cl":
        main["op"][1]["attr"].append(deepcopy(main["op"][1]["attr"][0]))
    else:
        desc = cpu[1]["graph"]["op"][0]["output_desc"][0]
        if damage == "different_layout":
            desc["layout"] = "NCHW"
        else:
            desc.pop("layout")
    assert _resolve(case)["status"] == "UNRESOLVED"


@pytest.mark.parametrize("unrelated", ["source_other_output", "consumer_other_input", "consumer_result"])
def test_unrelated_interface_descriptor_does_not_block_a_proven_selected_edge(case, unrelated):
    # This validates one producer edge, not all arithmetic in the model. Other
    # edges retain their own independent precision and provenance checks.
    main, cpu, expected = case
    if unrelated == "source_other_output":
        cpu[0]["graph"]["op"][-1]["input_desc"][0] = _desc(dtype=2)
    elif unrelated == "consumer_other_input":
        cpu[1]["graph"]["op"][0]["output_desc"][1] = _desc((1, 8, 1))
    else:
        desc = _desc((1, 1, 16))
        desc["layout"] = "NCHW"
        cpu[1]["graph"]["op"][-1]["output_desc"][0] = desc
    report = _resolve(case)
    assert report["status"] == "MATCHED", report
    assert report["producer"] == "second_flat"


@pytest.mark.parametrize("shape", [(1, 32, 16, 80), (1, 16, 32, 128), (1, 80, 8, 128)])
@pytest.mark.parametrize("dtype", [1, 2])
def test_nonpath_4d_carrier_layout_does_not_block_flattened_3d_fp32_trace(case, shape, dtype):
    # These mixed-rank geometries model observed attention interfaces. The
    # numeric dtype codes here belong only to this fixture's explicit enum.
    # No interpretation of a vendor dtype number is inferred from this test.
    main, cpu, expected = case
    nd = _desc(shape, dtype=dtype)
    nchw = deepcopy(nd)
    nchw["layout"] = "NCHW"
    source = cpu[0]["graph"]["op"]
    source.insert(-1, _node("other_4d_attention_value", "MatMul", output_desc=[nd]))
    source[-1]["input"].append("other_4d_attention_value:0")
    source[-1]["input_desc"].append(nchw)
    source[-1]["output_desc"].append(nd)
    main["op"][1]["output_desc"].append(nd)
    report = _resolve(case)
    assert report["status"] == "MATCHED", report
    assert all(point["descriptor"] == {"dtype": "float32", "shape": [1, 1, 8], "layout": "ND"}
               for point in report["path"])


@pytest.mark.parametrize("point", ["data_output", "data_input", "main_input", "main_output",
                                    "netout_input", "netout_output", "producer_output"])
@pytest.mark.parametrize("field", ["dtype", "shape", "layout"])
def test_every_selected_path_descriptor_side_remains_strict_and_reports_both_values(case, point, field):
    main, cpu, expected = case
    locations = {
        "data_output": (cpu[1]["graph"]["op"][0], "output_desc", 0, "CPU_consumer"),
        "data_input": (cpu[1]["graph"]["op"][0], "input_desc", 0, "CPU_consumer"),
        "main_input": (main["op"][2], "input_desc", 0, "CPU_consumer"),
        "main_output": (main["op"][1], "output_desc", 1, "CPU_source"),
        "netout_input": (cpu[0]["graph"]["op"][-1], "input_desc", 1, "CPU_source"),
        "netout_output": (cpu[0]["graph"]["op"][-1], "output_desc", 1, "CPU_source"),
        "producer_output": (cpu[0]["graph"]["op"][2], "output_desc", 0, "CPU_source"),
    }
    node, side, port, owner = locations[point]
    wrong = _desc()
    if field == "dtype":
        wrong["dtype"] = 2
    elif field == "shape":
        wrong["shape"] = {"dim": [1, 8, 1]}
    else:
        wrong["layout"] = "NCHW"
    node[side][port] = wrong
    report = _resolve(case)
    assert report["status"] == "UNRESOLVED", report
    details = report["details"]
    assert details["owner"] == owner and details["node"] == node["name"]
    assert details["port"] == port and details["boundary_type"] == node["type"]
    assert details["expected"] == {"dtype": "float32", "shape": [1, 1, 8], "layout": "ND"}
    assert details["actual"] != details["expected"]
    assert details["differing_fields"] == [field]
    assert details["relation_to_path"] == "PATH_PORT"
    assert details["main_direction"] == ("input_desc" if point.startswith("data") or point == "main_input" else "output_desc")
    assert details["child_direction"] == (None if point.startswith("main") else side)


def test_interface_port_count_failure_has_structured_owner_and_both_counts(case):
    case[1][0]["graph"]["op"][-1]["output_desc"].pop()
    report = _resolve(case)
    assert report["status"] == "UNRESOLVED"
    assert report["details"] == {
        "owner": "CPU_source", "boundary_type": "NetOutput", "node": "arbitrary_output_name",
        "main_direction": "output_desc", "child_direction": "output_desc", "port": None,
        "expected": 2, "actual": 1, "differing_fields": ["port_count"],
        "relation_to_path": "OTHER_INTERFACE_PORT"}


def test_identical_interface_failures_are_grouped_without_losing_affected_nodes(case):
    case[1][1]["graph"]["op"][0]["output_desc"][0] = _desc(dtype=2)
    resolution = _resolve(case)
    other_relation = deepcopy(resolution)
    other_relation["details"]["relation_to_path"] = "OTHER_INTERFACE_PORT"
    precision = {"status": "FAILED", "nodes": [
        {"name": "concat_a", "status": "MISMATCH", "errors": [
            {"kind": "FP32_REQUIRED_EDGE_CHANGED", "port": 0, "partition_resolution": resolution},
            {"kind": "FP32_REQUIRED_EDGE_CHANGED", "port": 1, "partition_resolution": resolution}]},
        {"name": "concat_b", "status": "MISMATCH", "errors": [
            {"kind": "FP32_REQUIRED_EDGE_CHANGED", "port": 9, "partition_resolution": other_relation}]},
        {"name": "good", "status": "MATCHED", "errors": []}]}
    summary = summarize_precision_failures(precision)
    assert summary["status"] == "FAILED" and summary["failed_node_count"] == 2
    assert summary["unique_issue_count"] == 1 and summary["error_count"] == 3
    issue = summary["issues"][0]
    assert issue["occurrences"] == 3 and issue["affected_nodes"] == ["concat_a", "concat_b"]
    assert issue["affected_input_ports"] == [{"node": "concat_a", "port": 0}, {"node": "concat_a", "port": 1},
                                              {"node": "concat_b", "port": 9}]
    assert issue["relations_to_path"] == ["PATH_PORT", "OTHER_INTERFACE_PORT"]


def test_distinct_descriptor_mismatches_never_share_a_summary_group(case):
    case[1][1]["graph"]["op"][0]["output_desc"][0] = _desc(dtype=2)
    left = _resolve(case)
    case[1][1]["graph"]["op"][0]["output_desc"][0] = _desc((1, 8, 1))
    right = _resolve(case)
    result = summarize_precision_failures({"status": "FAILED", "nodes": [
        {"name": "same_concat", "status": "MISMATCH", "errors": [
            {"kind": "FP32_REQUIRED_EDGE_CHANGED", "port": 0, "partition_resolution": left},
            {"kind": "FP32_REQUIRED_EDGE_CHANGED", "port": 1, "partition_resolution": right}]}]})
    assert result["unique_issue_count"] == 2
    assert result["issues"][0]["details"]["differing_fields"] == ["dtype"]
    assert result["issues"][1]["details"]["differing_fields"] == ["shape"]


def test_legacy_string_errors_and_nonobject_resolution_are_summarized_without_mutating_report():
    precision = {"status": "FAILED", "nodes": [{"name": "legacy_node", "status": "MISSING", "errors": [
        "missing CPU producer", "missing CPU producer",
        {"kind": "FP32_REQUIRED_EDGE_CHANGED", "partition_resolution": "old free-form observation"}]}]}
    original = deepcopy(precision)
    summary = summarize_precision_failures(precision)
    assert precision == original
    assert summary["status"] == "FAILED" and summary["failed_node_count"] == 1
    assert summary["error_count"] == 3 and summary["unique_issue_count"] == 2
    assert summary["issues"][0]["kind"] == "UNSTRUCTURED_ERROR"
    assert summary["issues"][0]["error"]["message"] == "missing CPU producer"
    assert summary["issues"][0]["occurrences"] == 2
    assert summary["issues"][1]["reason"] is None
