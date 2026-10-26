"""Numerical gates for direct target construction; no GE runtime is mocked."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from reference.tiny import create
from reference.ge_torch import evaluate_torch
from qwen35_moe_dflash.export_moe import prepare_export_target
from qwen35_moe_dflash.precision import lower_fp16
from qwen35_moe_dflash.ge_ir import Graph
from qwen35_moe_dflash.ge_eval import evaluate
from qwen35_moe_dflash.ge_target import build_target, _Builder, _chunk_gdn


def test_rms_offset_scalar_lowering_matches_every_finite_fp16_bit_pattern():
    # This validates the chosen scalar expression against the prior tensor Add,
    # including signed zero, independently of the compiler's Power kernel.
    bits = np.arange(1 << 16, dtype=np.uint16)
    values = bits.view(np.float16)
    values = values[np.isfinite(values)].copy()
    assert values.size == 63488
    assert {0x0000, 0x8000}.issubset(set(values.view(np.uint16)))
    graph = Graph("rms_offset_exhaustive")
    x = graph.input("x", values.shape)
    _Builder(graph).rms(x, "target.norm.weight", 1e-6)
    offset = next(node for node in graph.nodes
                  if len(node.inputs) == 1 and node.inputs[0].name == "target.norm.weight")
    assert offset.kind == "scalar_add" and offset.attrs == {"value": 1.0}
    graph.outputs = [offset.output]
    inputs = {"x": np.ones(values.shape, dtype=np.float16)}
    weights = {"target.norm.weight": values}
    actual_numpy = evaluate(graph, inputs, weights)[0]
    actual_torch = evaluate_torch(graph, inputs, weights)[0].numpy()
    expected_numpy = np.add(np.array(1, dtype=np.float16), values)
    expected_torch = (torch.tensor(1, dtype=torch.float16) + torch.from_numpy(values)).numpy()
    for actual, expected in ((actual_numpy, expected_numpy), (actual_torch, expected_torch)):
        np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))
        # -1 + 1 must produce positive zero; numerical equality alone misses it.
        assert actual[values == np.float16(-1)].view(np.uint16).tolist() == [0]


@pytest.fixture(scope="module")
def target_case():
    torch.set_num_threads(1)
    target, _ = create()
    # Include both intermediate layers and the final normalized output.
    target.config = replace(target.config, target_layer_ids=(0, 2, 3))
    config = target.language_model.config.to_dict()
    graph = build_target(config, target.config, 8, head_chunk_rows=4)
    ids = torch.zeros((1, 8), dtype=torch.int32)
    ids[0, :2] = torch.tensor([1, 2])
    reference, audit = lower_fp16(prepare_export_target(target), (ids, (ids != 0).int()))
    assert audit["status"] == "PASS"
    weights = {"target." + key.removeprefix("language_model."): value.detach().numpy()
               for key, value in reference.state_dict().items()}
    return graph, reference, weights, config, target.config


@pytest.mark.parametrize("tokens", [[1], [1, 2, 3], [7, 3, 26, 8, 16, 42, 29, 31]])
@pytest.mark.parametrize("expert_dispatch", ["static", "selected"])
def test_target_matches_existing_fp16_graph_and_final_norm(target_case, tokens, expert_dispatch):
    graph, reference, weights, config, draft_config = target_case
    if expert_dispatch == "selected":
        graph = build_target(config, draft_config, 8, head_chunk_rows=4, expert_dispatch=expert_dispatch)
    ids = np.zeros((1, 8), dtype=np.int32)
    mask = ids.copy()
    ids[0, :len(tokens)] = tokens
    mask[0, :len(tokens)] = 1
    actual = evaluate_torch(graph, {"input_ids": ids, "attention_mask": mask}, weights)
    numpy_actual = evaluate(graph, {"input_ids": ids, "attention_mask": mask}, weights)
    with torch.inference_mode():
        expected = reference(torch.from_numpy(ids), torch.from_numpy(mask))
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    np.testing.assert_array_equal(numpy_actual[0], expected[0].numpy())
    # Standard PyTorch FP16 comparison defaults; no tolerance is widened.
    torch.testing.assert_close(actual[1], expected[1])
    assert actual[1].dtype == torch.float16
    assert np.isfinite(numpy_actual[1]).all()
    assert {v.dtype for v in graph.weights.values()} == {"float16"}


@pytest.mark.parametrize("field,value,match", [
    ("attention_bias", True, "bias-free"),
    ("num_experts_per_tok", 5, "top-k"),
    ("num_attention_heads", 3, "head counts"),
    ("rope_parameters", {"rope_type": "dynamic"}, "default text RoPE"),
    ("layer_types", ["full_attention"], "layer_types"),
])
def test_unsupported_target_config_fails_explicitly(target_case, field, value, match):
    _, _, _, config, draft_config = target_case
    changed = dict(config)
    if field == "num_attention_heads":
        changed["num_key_value_heads"] = 2
    changed[field] = value
    with pytest.raises(ValueError, match=match):
        build_target(changed, draft_config, 8)


def test_chunk_gdn_retains_state_between_64_token_chunks():
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import torch_chunk_gated_delta_rule

    class Chunk(torch.nn.Module):
        config = None

        def forward(self, q, k, v, g, beta):
            return torch_chunk_gated_delta_rule(q, k, v, g, beta,
                                                use_qk_l2norm_in_kernel=True)[0]

    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(41)
    inputs = tuple(t.half() for t in (
        torch.randn(1, 65, 2, 4, generator=generator) * 0.25,
        torch.randn(1, 65, 2, 4, generator=generator) * 0.25,
        torch.randn(1, 65, 2, 3, generator=generator) * 0.25,
        -torch.rand(1, 65, 2, generator=generator),
        torch.rand(1, 65, 2, generator=generator),
    ))
    graph = Graph("gdn_chunk_test")
    symbolic = [graph.input(name, tuple(t.shape), "float16")
                for name, t in zip(("q", "k", "v", "g", "beta"), inputs)]
    graph.outputs = [_chunk_gdn(_Builder(graph), *symbolic)]
    reference, audit = lower_fp16(Chunk(), inputs)
    assert audit["status"] == "PASS"
    with torch.inference_mode():
        expected = reference(*inputs)
    actual = evaluate_torch(graph, {v.name: t for v, t in zip(symbolic, inputs)})[0]
    torch.testing.assert_close(actual, expected)
