"""Exact CPU partition-port provenance; no descriptor-only or NPU shortcuts."""
from copy import deepcopy

import pytest

from qwen35_moe_dflash.draft_precision_edges import PartitionEdgeResolver


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
