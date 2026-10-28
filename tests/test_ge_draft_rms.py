"""Qwen3 RMS contract through the direct IR and serialized GE recipe.

These host checks establish the requested arithmetic and descriptors, not
target compiler placement, supported kernels, or device numerical parity.
"""
import numpy as np
import pytest
import torch

from qwen35_moe_dflash.config import Qwen35DFlashConfig
from qwen35_moe_dflash.ge_cpp import emit_cpp
from qwen35_moe_dflash.ge_draft import _DraftBuilder, build_draft
from qwen35_moe_dflash.ge_eval import evaluate
from qwen35_moe_dflash.ge_export import configurations
from qwen35_moe_dflash.ge_recipe_read import read_recipe
from qwen35_moe_dflash.ops import TorchDFlashOps
from reference.ge_torch import evaluate_torch


def _norm(shape=(1, 3, 2048)):
    config = Qwen35DFlashConfig.from_dict(configurations(tiny=True)[1])
    builder = _DraftBuilder(config, 8)
    x = builder.g.input("x", shape)
    builder.g.outputs = [builder.norm(x, "draft.norm.weight", name="normalized")]
    return builder.g, config.rms_norm_eps


def test_large_and_small_rows_match_qwen3_without_fp16_square_collapse():
    graph, eps = _norm()
    rng = np.random.default_rng(7401)
    x = rng.normal(size=(1, 3, 2048)).astype(np.float16)
    x[:, 0] *= np.float16(300)
    x[:, 1] *= np.float16(0.0002)
    x[:, 2] = 0
    weight = rng.uniform(0.3, 1.7, 2048).astype(np.float16)
    tx, tw = torch.from_numpy(x), torch.from_numpy(weight)
    expected = TorchDFlashOps().rms_norm(tx, tw, eps)
    with torch.inference_mode():
        old = tx * torch.rsqrt(tx.square().mean(-1, keepdim=True) + eps) * tw
    assert not torch.count_nonzero(old[:, 0])
    assert torch.count_nonzero(expected[:, 0]) > 2000
    assert torch.count_nonzero(expected[:, 1]) > 2000
    assert not torch.count_nonzero(expected[:, 2])
    inputs, weights = {"x": x}, {"draft.norm.weight": weight}
    actual, = evaluate_torch(graph, inputs, weights)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    numpy_actual, = evaluate(graph, inputs, weights)
    assert np.isfinite(numpy_actual).all()
    np.testing.assert_allclose(numpy_actual, expected.numpy(), rtol=1e-3, atol=2e-6)
    # Non-unit gamma distinguishes casting before gamma from casting after it;
    # neither a late cast nor the target model's (1+gamma) is the draft contract.
    normalized_fp32 = tx.float() * torch.rsqrt(tx.float().square().mean(-1, keepdim=True) + eps)
    wrong_late_cast = (normalized_fp32 * tw.float()).half()
    assert torch.any(wrong_late_cast != expected)


def test_large_2048_wide_unit_gamma_rows_preserve_unit_rms():
    graph, eps = _norm((1, 15, 2048))
    x = np.linspace(340, 3074, 15 * 2048, dtype=np.float32).reshape(1, 15, 2048).astype(np.float16)
    x[:, :, 1::2] *= -1
    actual, = evaluate_torch(graph, {"x": x}, {"draft.norm.weight": np.ones(2048, dtype=np.float16)})
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float().square().mean(-1).sqrt(), torch.ones(1, 15),
                               rtol=5e-4, atol=0)
    assert float(actual.double().norm()) == pytest.approx(np.sqrt(15 * 2048), rel=5e-4)


@pytest.mark.parametrize("policy", ["native", "matmul"])
def test_serialized_norm_keeps_fp32_statistics_and_casts_before_direct_scale(tmp_path, policy):
    graph, eps = _norm()
    reduction = {}
    emit_cpp(graph, {"draft.norm.weight": tmp_path / "gamma.bin"}, external_weights=True,
             recipe_dir=tmp_path / "recipe", fp16_reductions=policy, reduction_report=reduction,
             fp16_row_mul="compact")
    recipe = read_recipe(tmp_path / "recipe/graph.recipe")
    records = recipe["records"]
    by_name = {record["name"]: record for record in records}
    norms = [r for r in records if r["type"] == "ReduceMean"]
    assert len(norms) == 1
    mean = norms[0]
    assert mean["inputs"][0]["dtype"] == mean["outputs"][0]["dtype"] == "float32"
    assert mean["inputs"][1]["dtype"] == "int32"
    assert not any(r["type"] == "MatMul" for r in records)
    assert reduction["counts"]["replaced"] == 0
    assert reduction["retained"][0]["reason"] == (
        "non_fp16_source_or_output" if policy == "matmul" else "mode_native")
    gamma_mul = by_name["normalized"]
    assert gamma_mul["type"] == "Mul"
    assert gamma_mul["inputs"][1]["src"] == records.index(by_name["draft.norm.weight"])
    before_gamma = records[gamma_mul["inputs"][0]["src"]]
    assert before_gamma["type"] == "Cast"
    assert before_gamma["attrs"]["dst_type"] == {"type": "int", "value": 1}
    assert before_gamma["inputs"][0]["dtype"] == "float32"
    assert before_gamma["outputs"][0]["dtype"] == "float16"
    promoted = [r for r in records if r["type"] == "Cast" and r["outputs"][0]["dtype"] == "float32"]
    assert len(promoted) == 1
    assert promoted[0]["attrs"]["dst_type"] == {"type": "int", "value": 0}
    for r in records:
        if r["type"] in {"Mul", "Power", "Rsqrt"} and r is not gamma_mul:
            assert {p["dtype"] for p in r["inputs"] + r["outputs"]} == {"float32"}
    epsilon = next(r for r in records if r["type"] == "Power")
    assert epsilon["attrs"]["shift"]["value"] == float(np.float32(eps))
    assert epsilon["attrs"]["scale"]["value"] == 1.0


def test_all_complete_draft_norms_have_fp32_stats_with_unchanged_weight_and_io_types():
    config = Qwen35DFlashConfig.from_dict(configurations(tiny=False)[1])
    graph = build_draft(config, 64)
    consumers = {value.name: [] for value in graph._values.values()}
    for node in graph.nodes:
        for value in node.inputs:
            consumers[value.name].append(node)
    norm_weights = [v for key, v in graph.weights.items() if key.endswith("norm.weight")]
    assert len(norm_weights) == 4 * config.num_hidden_layers + 2
    for weight in norm_weights:
        gamma_mul, = consumers[weight.name]
        assert gamma_mul.kind == "mul" and gamma_mul.output.dtype == "float16"
        cast = next(n for n in graph.nodes if n.output is gamma_mul.inputs[0])
        assert cast.kind == "cast" and cast.inputs[0].dtype == "float32"
    means = [n for n in graph.nodes if n.kind == "reduce_mean"]
    assert len(means) == len(norm_weights)
    assert all(n.inputs[0].dtype == n.output.dtype == "float32" for n in means)
    assert {v.dtype for v in graph.weights.values()} == {"float16"}
    assert [v.dtype for v in graph.inputs] == ["float16", "int32", "int32"]
    assert [v.dtype for v in graph.outputs] == ["int32"]
