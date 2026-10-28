"""Draft RoPE follows the pinned 4B AIR path: FP32 until cosine/sine outputs."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from qwen35_moe_dflash.config import Qwen35DFlashConfig
from qwen35_moe_dflash.ge_cpp import emit_cpp
from qwen35_moe_dflash.ge_draft import build_draft
from qwen35_moe_dflash.ge_eval import evaluate
from qwen35_moe_dflash.ge_export import configurations
from qwen35_moe_dflash.ge_recipe_read import read_recipe
from qwen35_moe_dflash.model import DFlashRotaryEmbedding
from reference.ge_torch import evaluate_torch


def _config(tiny=True):
    return Qwen35DFlashConfig.from_dict(configurations(tiny=tiny)[1])


def _position_slice(graph):
    """Cut the real source position/RoPE computation, retaining no weights."""
    graph = deepcopy(graph)
    named = {node.output.name: node for node in graph.nodes}
    graph.outputs = [named[name].output for name in ("positions", "rope_frequency", "rope_cosine", "rope_sine")]
    live = {value.name for value in graph.outputs}
    for node in reversed(graph.nodes):
        if node.output.name in live:
            live.update(value.name for value in node.inputs)
    graph.nodes = [node for node in graph.nodes if node.output.name in live]
    graph.inputs = [value for value in graph.inputs if value.name in live]
    graph.constants = {name: array for name, array in graph.constants.items() if name in live}
    graph.weights = {name: value for name, value in graph.weights.items() if name in live}
    assert not graph.weights
    assert [value.name for value in graph.inputs] == ["attention_mask"]
    return graph


@pytest.mark.parametrize("tiny", [True, False], ids=["tiny", "full-config"])
def test_all_prefix_lengths_match_fp32_rotary_source(tiny):
    config = _config(tiny)
    graph = _position_slice(build_draft(config, 8))
    rotary = DFlashRotaryEmbedding(config)
    for length in range(1, 9):
        mask = np.zeros((1, 8), dtype=np.int32)
        mask[0, :length] = 1
        positions, frequency, cosine, sine = evaluate_torch(graph, {"attention_mask": mask})
        expected_cosine, expected_sine = rotary(positions, torch.float16)
        expected_frequency = positions.float().unsqueeze(-1) * rotary.inv_freq.view(1, 1, -1)
        torch.testing.assert_close(frequency, expected_frequency, rtol=1e-7, atol=0)
        torch.testing.assert_close(cosine, expected_cosine, rtol=1e-3, atol=5e-4)
        torch.testing.assert_close(sine, expected_sine, rtol=1e-3, atol=5e-4)
        assert frequency.dtype == torch.float32
        assert cosine.dtype == sine.dtype == torch.float16
        numpy_result = evaluate(graph, {"attention_mask": mask})
        np.testing.assert_allclose(numpy_result[2], cosine.numpy(), rtol=1e-3, atol=5e-4)
        np.testing.assert_allclose(numpy_result[3], sine.numpy(), rtol=1e-3, atol=5e-4)


@pytest.mark.parametrize("sequence", [2052, 65506])
def test_positions_are_not_rounded_or_overflowed_through_fp16(sequence):
    config = replace(_config(), max_position_embeddings=70000)
    graph = _position_slice(build_draft(config, sequence))
    mask = np.ones((1, sequence), dtype=np.int32)
    positions, frequency, cosine, sine = evaluate_torch(graph, {"attention_mask": mask})
    assert torch.isfinite(frequency).all() and torch.isfinite(cosine).all() and torch.isfinite(sine).all()
    # First inverse frequency is exactly one; values beyond the FP16 exact
    # integer range must remain distinct all the way through trig evaluation.
    torch.testing.assert_close(frequency[..., 0], positions.float(), rtol=0, atol=0)
    assert frequency[positions == 2049][0, 0] == 2049
    assert frequency[positions == 2051][0, 0] == 2051
    if sequence > 65504:
        assert frequency.max() > 65504
    old_frequency = (positions.half().unsqueeze(-1) * DFlashRotaryEmbedding(config).inv_freq.half()).half()
    old_embedding = torch.cat((old_frequency, old_frequency), dim=-1)
    assert torch.any(old_embedding.cos() != cosine)


@pytest.mark.parametrize("tiny", [True, False], ids=["tiny", "full-config"])
@pytest.mark.parametrize("active", [False, True], ids=["native", "matmul-compact"])
def test_recipe_keeps_fp32_rope_and_casts_only_completed_trigonometric_values(tmp_path, tiny, active):
    graph = _position_slice(build_draft(_config(tiny), 8))
    options = {"fp16_reductions": "matmul", "fp16_row_mul": "compact"} if active else {}
    emit_cpp(graph, recipe_dir=tmp_path / "recipe", **options)
    records = read_recipe(tmp_path / "recipe/graph.recipe")["records"]
    frequency = next(r for r in records if r["name"] == "rope_frequency")
    assert frequency["type"] == "Mul"
    assert {p["dtype"] for p in frequency["inputs"] + frequency["outputs"]} == {"float32"}
    inverse = records[frequency["inputs"][1]["src"]]
    assert inverse["type"] == "Const"
    assert inverse["outputs"][0]["dtype"] == "float32"
    assert inverse["constant"]["size"] == (_config(tiny).head_dim // 2) * 4
    for name, kind in (("rope_cosine", "Cos"), ("rope_sine", "Sin")):
        cast = next(r for r in records if r["name"] == name)
        assert cast["type"] == "Cast" and cast["attrs"]["dst_type"] == {"type": "int", "value": 1}
        trig = records[cast["inputs"][0]["src"]]
        assert trig["type"] == kind
        assert {p["dtype"] for p in trig["inputs"] + trig["outputs"]} == {"float32"}
        assert cast["outputs"][0]["dtype"] == "float16"
    assert not any(r["type"] in {"MatMul", "BatchMatMulV2"} for r in records)
