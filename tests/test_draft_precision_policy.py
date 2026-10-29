"""The compile request selects exact recipe nodes; placement needs a later gate."""
import copy
import json

import pytest

from qwen35_moe_dflash.draft_precision_policy import customization_bytes
from test_draft_compile_precision import case


def test_full_draft_maps_every_fp32_operation_to_one_partial_selection_group(case):
    plan = case["plan"]
    before = copy.deepcopy(plan)
    raw = customization_bytes(plan)
    config = json.loads(raw)
    graph = config["npucl_strategy"]["npucl_graphs"]["draft"]
    assert len(graph["nodes_info"]) == 153
    assert graph["groups_info"] == [{"npucl_implementation": 7}]
    assert set(graph) == {"groups_info", "nodes_info"}
    for node in plan["required_nodes"]:
        assert graph["nodes_info"][node["name"]] == {"group_id": 0}
    excluded = {n["name"] for n in plan["boundary_casts"] + plan["fp32_constants"]}
    assert not excluded.intersection(graph["nodes_info"])
    assert plan == before
    assert raw == customization_bytes(copy.deepcopy(plan))
    assert raw.endswith(b"\n")


def test_diagnostic_metadata_never_changes_the_vendor_configuration(case):
    plan = copy.deepcopy(case["plan"])
    raw = customization_bytes(plan)
    plan["diagnostic"] = {"path": "/different/run", "status": "unverified"}
    plan["recipe"]["path"] = "/relocated/graph.recipe"
    plan["counts"] = {"irrelevant_to_selector": True}
    assert customization_bytes(plan) == raw


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(contract="unknown"),
    lambda p: p.update(graph="target"),
    lambda p: p.update(required_nodes=[]),
    lambda p: p["required_nodes"].append(copy.deepcopy(p["required_nodes"][0])),
    lambda p: p["required_nodes"][0].update(type="Cast"),
])
def test_rejects_malformed_or_wrong_scope_plan(case, mutation):
    plan = copy.deepcopy(case["plan"])
    mutation(plan)
    with pytest.raises(ValueError):
        customization_bytes(plan)
