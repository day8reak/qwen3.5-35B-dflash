"""Attention matches 4B AIR's default FP16 dots and FP32 score arithmetic."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from qwen35_moe_dflash.config import Qwen35DFlashConfig
from qwen35_moe_dflash.ge_cpp import emit_cpp
from qwen35_moe_dflash.ge_draft import build_draft
from qwen35_moe_dflash.ge_export import configurations
from qwen35_moe_dflash.ge_recipe_read import read_recipe
from reference.ge_torch import evaluate_torch


def _core():
    config = Qwen35DFlashConfig.from_dict(configurations(tiny=True)[1])
    graph = deepcopy(build_draft(config, 8))
    qk = next(n for n in graph.nodes if n.output.name == "layer_0_qk_fp16")
    pv = next(n for n in graph.nodes if n.output.name == "layer_0_pv_fp16")
    softmax = next(n for n in graph.nodes if n.output.name == "layer_0_probability_fp32")
    mask = next(n.inputs[0] for n in graph.nodes if n.kind == "where" and n.output is softmax.inputs[0])
    # Cut the production attention core at its four live inputs; retain all
    # casts, scale, mask and softmax nodes rather than constructing a test graph.
    graph.nodes = graph.nodes[graph.nodes.index(qk):graph.nodes.index(pv) + 1]
    graph.inputs = [*qk.inputs, pv.inputs[1], mask]
    graph.outputs = [softmax.output, pv.output]
    live = {v.name for n in graph.nodes for v in n.inputs}
    graph.constants = {k: v for k, v in graph.constants.items() if k in live}
    graph.weights = {}
    return config, graph


def test_source_core_matches_fp16_qk_fp32_softmax_fp16_probability_pv():
    config, graph = _core()
    rng = np.random.default_rng(117)
    q, k, v = [torch.from_numpy(rng.normal(size=value.shape).astype(np.float16)) for value in graph.inputs[:3]]
    q *= 3
    k *= 3
    mask = torch.ones(graph.inputs[3].shape, dtype=torch.bool)
    mask[..., 2:5] = False
    feeds = dict(zip((value.name for value in graph.inputs), (q, k, v, mask)))
    probability, actual = evaluate_torch(graph, feeds)
    scale = config.head_dim ** -0.5
    qk_half = torch.matmul(q, k.transpose(-1, -2))
    scores = (qk_half.float() * scale).masked_fill(~mask, -float("inf"))
    expected_probability = torch.softmax(scores, -1, dtype=torch.float32)
    expected = torch.matmul(expected_probability.half(), v).float().half()
    torch.testing.assert_close(probability, expected_probability, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.count_nonzero(probability[..., 2:5]) == 0
    old_scores = (qk_half * scale).masked_fill(~mask, -float("inf"))
    old = torch.matmul(torch.softmax(old_scores, -1), v)
    assert torch.any(old != expected), "fixture must expose premature FP16 score scaling"
    late_probability_cast = (expected_probability @ v.float()).half()
    assert torch.any(late_probability_cast != expected), "fixture must expose the probability cast boundary"


@pytest.mark.parametrize("policy", ["native", "matmul"])
def test_recipe_preserves_attention_dtypes_under_both_lowering_policies(tmp_path, policy):
    _, graph = _core()
    emit_cpp(graph, recipe_dir=tmp_path / "recipe", fp16_reductions=policy, fp16_row_mul="compact")
    records = read_recipe(tmp_path / "recipe/graph.recipe")["records"]
    by_name = {r["name"]: r for r in records}
    qk, pv = by_name["layer_0_qk_fp16"], by_name["layer_0_pv_fp16"]

    def without_views(record):
        while record["type"] == "Reshape":
            record = records[record["inputs"][0]["src"]]
        return record

    for boundary in (qk, pv):
        assert boundary["outputs"][0]["dtype"] == "float16"
        matmul = without_views(boundary)
        assert matmul["type"] == "BatchMatMulV2"
        assert {p["dtype"] for p in matmul["inputs"] + matmul["outputs"]} == {"float16"}
    softmax = by_name["layer_0_probability_fp32"]
    assert softmax["type"] == "SoftmaxV2"
    assert {p["dtype"] for p in softmax["inputs"] + softmax["outputs"]} == {"float32"}
    assert softmax["attrs"]["half_to_float"] == {"type": "bool", "value": False}
    probability_cast = without_views(records[without_views(pv)["inputs"][0]["src"]])
    assert probability_cast["type"] == "Cast"
    assert probability_cast["inputs"][0]["src"] == records.index(softmax)
    assert probability_cast["attrs"]["dst_type"] == {"type": "int", "value": 1}
    scale = next(r for r in records if r["type"] == "Power")
    promoted = records[scale["inputs"][0]["src"]]
    assert promoted["type"] == "Cast" and promoted["inputs"][0]["src"] == records.index(qk)
    assert promoted["attrs"]["dst_type"] == {"type": "int", "value": 0}
    select = next(r for r in records if r["type"] == "SelectV2")
    assert select["inputs"][0]["dtype"] == "bool"
    assert all(p["dtype"] == "float32" for p in select["inputs"][1:] + select["outputs"])


def test_all_six_layers_use_the_same_attention_precision_contract():
    config = Qwen35DFlashConfig.from_dict(configurations(tiny=False)[1])
    graph = build_draft(config, 64)
    nodes = {n.output.name: n for n in graph.nodes}
    for layer in range(config.num_hidden_layers):
        qk, pv = nodes[f"layer_{layer}_qk_fp16"], nodes[f"layer_{layer}_pv_fp16"]
        assert qk.kind == pv.kind == "matmul"
        assert {v.dtype for n in (qk, pv) for v in (*n.inputs, n.output)} == {"float16"}
        softmax = nodes[f"layer_{layer}_probability_fp32"]
        assert softmax.kind == "softmax" and softmax.output.dtype == softmax.inputs[0].dtype == "float32"
        probability_cast = nodes[pv.inputs[0].name]
        assert probability_cast.kind == "cast" and probability_cast.inputs[0] is softmax.output
