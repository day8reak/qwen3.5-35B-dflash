"""Selected MoE graph semantics and bounded carriers; no device placement claim."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from qwen35_moe_dflash.export_moe import StaticExportExperts
from qwen35_moe_dflash.ge_cpp import emit_cpp
from qwen35_moe_dflash.ge_eval import evaluate
from qwen35_moe_dflash.ge_ir import Graph
from qwen35_moe_dflash.ge_moe import selected_experts
from qwen35_moe_dflash.ge_target import _Builder
from reference.ge_torch import evaluate_torch
from test_ge_select_lowering import RecipeReader


def _baseline(b, flat, indices, weights, gate_up, down, middle):
    """Original direct-GE all-expert primitive loop, without changing the router."""
    tokens, hidden = flat.shape
    result = b.constant(np.zeros(flat.shape, np.float16))
    for expert in range(gate_up.shape[0]):
        selected = b.op("eq", indices, b.constant(expert, "int32"), dtype="bool")
        route = b.op("where", selected, weights, b.constant(0))
        scale = b.reduce("reduce_sum", route, -1)
        bank = b.view(b.slice(gate_up, 0, expert, 1), (2 * middle, hidden))
        projected = b.linear(flat, "unused", 2 * middle, bank)
        gate, up = b.slice(projected, -1, 0, middle), b.slice(projected, -1, middle, middle)
        bank = b.view(b.slice(down, 0, expert, 1), (hidden, middle))
        values = b.linear(b.op("mul", b.op("silu", gate), up), "unused", hidden, bank)
        contribution = b.op("mul", values, scale)
        active = b.op("gt", b.reduce("reduce_sum", b.op("cast", selected, dtype="int32"), -1), b.constant(0, "int32"),
                      dtype="bool")
        result = b.op("add", result, b.op("where", active, contribution, b.constant(0),
                                         shape=flat.shape))
    return result


def _graph(tokens=3, hidden=5, experts=4, top_k=2, middle=3, *, baseline=False, input_banks=False):
    graph = Graph("selected_moe")
    b = _Builder(graph)
    flat = graph.input("hidden", (tokens, hidden))
    indices = graph.input("indices", (tokens, top_k), "int32")
    weights = graph.input("route_weights", (tokens, top_k))
    bank = graph.input if input_banks else graph.weight
    gate_up = bank("gate_up", (experts, 2 * middle, hidden))
    down = bank("down", (experts, hidden, middle))
    graph.outputs = [(_baseline if baseline else selected_experts)(
        b, flat, indices, weights, gate_up, down, middle)]
    return graph.validate()


def _samples(tokens, hidden, experts, top_k, middle, seed):
    rng = np.random.default_rng(seed)
    indices = np.stack([rng.permutation(experts)[:top_k] for _ in range(tokens)]).astype(np.int32)
    logits = rng.normal(size=(tokens, top_k)).astype(np.float32)
    weights = np.exp(logits - logits.max(-1, keepdims=True))
    weights = (weights / weights.sum(-1, keepdims=True)).astype(np.float16)
    inputs = {"hidden": rng.normal(0, .3, (tokens, hidden)).astype(np.float16),
              "indices": indices, "route_weights": weights}
    banks = {"gate_up": rng.normal(0, .2, (experts, 2 * middle, hidden)).astype(np.float16),
             "down": rng.normal(0, .2, (experts, hidden, middle)).astype(np.float16)}
    return inputs, banks


def _static_reference(inputs, banks):
    owner = SimpleNamespace(num_experts=len(banks["gate_up"]),
                            gate_up_proj=torch.from_numpy(banks["gate_up"]),
                            down_proj=torch.from_numpy(banks["down"]), act_fn=F.silu)
    with torch.inference_mode():
        return StaticExportExperts(owner)(torch.from_numpy(inputs["hidden"]),
                                          torch.from_numpy(inputs["indices"]),
                                          torch.from_numpy(inputs["route_weights"]))


@pytest.mark.parametrize("tokens,top_k", [(1, 1), (3, 1), (3, 2), (5, 4)])
def test_routes_remain_dynamic_and_match_original_expert_loop(tokens, top_k):
    params = (tokens, 5, 4, top_k, 3)
    graph, original = _graph(*params), _graph(*params, baseline=True)
    for seed in (182, 900, 312):
        inputs, banks = _samples(*params, seed)
        saved_weights = inputs["route_weights"].copy()
        actual, = evaluate(graph, inputs, banks)
        expected, = evaluate(original, inputs, banks)
        np.testing.assert_array_equal(actual, expected)
        actual_torch, = evaluate_torch(graph, inputs, banks)
        expected_torch, = evaluate_torch(original, inputs, banks)
        # BatchMatMul and Linear may use different internal host reductions.
        torch.testing.assert_close(actual_torch, expected_torch)
        torch.testing.assert_close(actual_torch, _static_reference(inputs, banks))
        assert actual.dtype == np.float16 and actual_torch.dtype == torch.float16
        np.testing.assert_array_equal(inputs["route_weights"], saved_weights)


def test_unselected_nan_experts_cannot_contaminate_a_token():
    params = (3, 5, 4, 2, 3)
    graph, original = _graph(*params), _graph(*params, baseline=True)
    inputs, banks = _samples(*params, 42)
    inputs["indices"][:] = [[3, 1], [1, 3], [0, 1]]
    # Expert 0 is active only for the last token, and expert 2 is never active.
    banks["gate_up"][[0, 2]] = np.nan
    banks["down"][[0, 2]] = np.nan
    for evaluator in (evaluate, evaluate_torch):
        actual, = evaluator(graph, inputs, banks)
        expected, = evaluator(original, inputs, banks)
        actual, expected = np.asarray(actual), np.asarray(expected)
        assert np.isfinite(actual[:2]).all()
        assert np.isnan(actual[2]).all()
        torch.testing.assert_close(torch.from_numpy(actual), torch.from_numpy(expected), equal_nan=True)


def test_fp16_additions_follow_expert_ids_not_router_slot_order():
    graph, original = _graph(1, 1, 3, 3, 1), _graph(1, 1, 3, 3, 1, baseline=True)
    inputs = {"hidden": np.ones((1, 1), np.float16),
              "indices": np.array([[2, 0, 1]], np.int32),
              "route_weights": np.array([[.5, .25, .25]], np.float16)}
    banks = {"gate_up": np.array([[[16], [1]]] * 3, np.float16),
             "down": np.array([[[512]], [[-512]], [[.125]]], np.float16)}
    # Expert-ID order gives (2048 + -2048) + 1 == 1 in FP16. Slot order
    # gives (1 + 2048) + -2048 == 0, so a sorted gather without sorted weights
    # or a routing-order sum cannot pass this regression.
    for evaluator in (evaluate, evaluate_torch):
        actual, = evaluator(graph, inputs, banks)
        expected, = evaluator(original, inputs, banks)
        np.testing.assert_array_equal(np.asarray(actual), np.array([[1]], np.float16))
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    torch.testing.assert_close(_static_reference(inputs, banks), torch.ones((1, 1), dtype=torch.float16),
                               rtol=0, atol=0)


def test_maximum_expert_ids_are_exact_through_fp16_sort():
    graph = _graph(1, 1, 2048, 2, 1)
    inputs = {"hidden": np.ones((1, 1), np.float16),
              "indices": np.array([[2047, 0]], np.int32),
              "route_weights": np.array([[.75, .25]], np.float16)}
    banks = {"gate_up": np.full((2048, 2, 1), np.nan, np.float16),
             "down": np.full((2048, 1, 1), np.nan, np.float16)}
    banks["gate_up"][[0, 2047]] = [[[16], [1]]]
    banks["down"][0] = 1
    banks["down"][2047] = 2
    for evaluator in (evaluate, evaluate_torch):
        actual, = evaluator(graph, inputs, banks)
        np.testing.assert_array_equal(np.asarray(actual), np.array([[28]], np.float16))


def test_carriers_gather_only_selected_banks_per_token_without_normalizing():
    graph = _graph(tokens=3, hidden=2048, experts=256, top_k=8, middle=512)
    gathers = [node for node in graph.nodes if node.kind == "gather" and node.inputs[0].name in graph.weights]
    assert len(gathers) == 2 * 3
    assert {node.output.shape for node in gathers} == {(8, 1024, 2048), (8, 2048, 512)}
    assert sum(math.prod(node.output.shape) * 2 for node in gathers[:2]) == 48 * 1024**2
    # This bounds each token's logical carrier, not compiler-scheduled peak memory.
    assert max(len(value.shape) for value in graph._values.values()) <= 3
    assert sum(node.kind == "matmul" for node in graph.nodes) == 2 * 3
    assert sum(node.kind == "add" for node in graph.nodes) == 8 * 3
    assert not ({node.kind for node in graph.nodes} & {
        "gather_elements", "nonzero", "where", "softmax", "reduce_sum", "reduce_mean", "rsqrt"})
    assert all(node.inputs[0].dtype == "float16" for node in graph.nodes if node.kind == "gather")
    assert graph.weights["gate_up"].shape == (256, 1024, 2048)
    assert graph.weights["down"].shape == (256, 2048, 512)


def test_existing_ge_serializer_accepts_selected_path_without_new_operators(tmp_path):
    graph = _graph(tokens=2, input_banks=True)
    emit_cpp(graph, recipe_dir=tmp_path / "recipe")
    records, _, _ = RecipeReader(tmp_path / "recipe/graph.recipe").read()
    kinds = [record["type"] for record in records]
    assert kinds.count("BatchMatMulV2") == 4
    assert kinds.count("GatherV2") == 6
    assert kinds.count("TopKV2") == 2  # Values/indices share each ID sort.
    assert not ({"GatherElements", "NonZero", "Select"} & set(kinds))
    assert max(len(port["shape"]) for record in records for port in record["outputs"]) <= 3
    # Exercise unrolled emission as well as the recipe above.
    assert '"BatchMatMulV2"' in emit_cpp(graph)


@pytest.mark.parametrize("change,match", [
    ({"middle": 0}, "intermediate"),
    ({"middle": True}, "intermediate"),
    ({"flat": ((0, 5), "float16")}, "nonempty"),
    ({"flat": ((3, 5), "int32")}, "FP16"),
    ({"flat": ((1, 3, 5), "float16")}, "input"),
    ({"indices": ((3, 2), "int64")}, "INT32"),
    ({"indices": ((2, 2), "int32")}, "indices"),
    ({"indices": ((3, 0), "int32"), "weights": ((3, 0), "float16")}, "top_k"),
    ({"indices": ((3, 5), "int32"), "weights": ((3, 5), "float16")}, "top_k"),
    ({"weights": ((3, 1), "float16")}, "weights"),
    ({"weights": ((3, 2), "int32")}, "weights"),
    ({"gate_up": ((4, 6, 5), "int32")}, "banks"),
    ({"gate_up": ((4, 30), "float16")}, "rank-three"),
    ({"gate_up": ((4, 8, 5), "float16")}, "banks"),
    ({"down": ((5, 5, 3), "float16")}, "banks"),
    ({"down": ((4, 3, 5), "float16")}, "banks"),
    ({"gate_up": ((2049, 6, 5), "float16"), "down": ((2049, 5, 3), "float16")}, "2048"),
])
def test_invalid_static_contract_rejected_before_constructing_nodes(change, match):
    specs = {"flat": ((3, 5), "float16"), "indices": ((3, 2), "int32"),
             "weights": ((3, 2), "float16"), "gate_up": ((4, 6, 5), "float16"),
             "down": ((4, 5, 3), "float16")}
    middle = change.get("middle", 3)
    specs.update({name: value for name, value in change.items() if name != "middle"})
    graph = Graph("invalid_moe")
    values = [graph.input(name, shape, dtype) for name, (shape, dtype) in specs.items()]
    with pytest.raises(ValueError, match=match):
        selected_experts(_Builder(graph), *values, middle)
    assert graph.nodes == [] and graph.constants == {}
