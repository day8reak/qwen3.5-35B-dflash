"""Semantic checks for direct graph construction, independent of GE runtime."""
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from qwen35_moe_dflash.config import Qwen35DFlashConfig
from qwen35_moe_dflash.ge_draft import build_draft
from qwen35_moe_dflash.ge_eval import evaluate
from qwen35_moe_dflash.graphs import DraftGraph
from qwen35_moe_dflash.model import DFlashDraftModel
from qwen35_moe_dflash.ops import TorchDFlashOps, _repeat_kv
from reference.ge_torch import evaluate_torch


def _fixture():
    torch.set_num_threads(1)
    torch.manual_seed(739)
    config = Qwen35DFlashConfig(
        hidden_size=16, intermediate_size=32, vocab_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=4, num_target_layers=4, target_layer_ids=(0, 2),
        layer_types=("sliding_attention", "full_attention"), block_size=4,
        mask_token_id=63, rms_norm_eps=1e-6, rope_theta=10000.0,
        max_position_embeddings=128, sliding_window=3, use_sliding_window=True,
        attention_bias=False, attention_dropout=0.0, hidden_act="silu", dtype="float16",
        input_embedding_scale=0.75,
    )
    model = DFlashDraftModel(config, dtype=torch.float16).eval()
    embedding = nn.Embedding(config.vocab_size, config.hidden_size, dtype=torch.float16)
    head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=torch.float16)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "norm.weight" in name:
                parameter.uniform_(0.7, 1.2)
            else:
                parameter.normal_(std=0.08)
        embedding.weight.normal_(std=0.2)
        head.weight.normal_(std=0.1)
    graph = DraftGraph(model, embedding, head).eval()
    graph.config = config
    weights = {f"draft.{name}": value.detach().numpy().copy() for name, value in model.state_dict().items()}
    weights["target.embed_tokens.weight"] = embedding.weight.detach().numpy().copy()
    weights["target.lm_head.weight"] = head.weight.detach().numpy().copy()
    return config, graph, weights


def _inputs(config, length, sequence=8):
    rng = np.random.default_rng(900 + length)
    features = rng.normal(0, 0.2, (1, sequence, config.feature_size)).astype(np.float16)
    ids = np.array([[1, 7, 3, 9, 11, 4, 2, 8]], dtype=np.int32)[:, :sequence]
    mask = np.zeros((1, sequence), dtype=np.int32)
    mask[:, :length] = 1
    return {"target_features": features, "input_ids": ids, "attention_mask": mask}


def _named(graph, *names):
    values = {node.output.name: node.output for node in graph.nodes}
    return [values[name] for name in names]


@pytest.mark.parametrize("length", [1, 4, 8])
def test_draft_runtime_prefix_and_layer_masks(length):
    config, _, weights = _fixture()
    graph = build_draft(config, 8)
    graph.outputs = _named(graph, "block_ids", "context_valid", "positions", "layer_0_visible", "layer_1_visible")
    block, context, positions, sliding, full = evaluate(graph, _inputs(config, length), weights)
    assert block.tolist() == [[_inputs(config, length)["input_ids"][0, length - 1], 63, 63, 63]]
    expected_context = np.arange(8)[None, :] < length - 1
    np.testing.assert_array_equal(context, expected_context)
    expected_positions = np.concatenate((np.maximum(np.cumsum(expected_context, axis=1) - 1, 0),
                                         np.arange(length - 1, length - 1 + 4)[None, :]), axis=1)
    np.testing.assert_array_equal(positions, expected_positions)
    key_valid = np.concatenate((expected_context, np.ones((1, 4), dtype=bool)), axis=1)
    query_pos = expected_positions[:, -4:, None]
    key_pos = expected_positions[:, None, :]
    expected_sliding = key_valid[:, None, :] & (key_pos <= query_pos) & ((query_pos - key_pos) < 3)
    expected_full = np.broadcast_to(key_valid[:, None, :], (1, 4, 12))
    np.testing.assert_array_equal(sliding, expected_sliding[:, None])
    np.testing.assert_array_equal(full, expected_full[:, None])
    assert full[0, 0, 0, -1] and not sliding[0, 0, 0, -1]


class _FP16PrimitiveOps(TorchDFlashOps):
    """FP16 primitives with the source Qwen3 FP32 normalization contract."""

    def attention(self, query, key, value, attention_mask, scale, key_value_groups):
        key = _repeat_kv(key, key_value_groups)
        value = _repeat_kv(value, key_value_groups)
        scores = torch.matmul(query, key.transpose(-1, -2)).float() * scale
        scores = torch.where(attention_mask, scores, torch.tensor(-float("inf"), dtype=torch.float32))
        return torch.matmul(torch.softmax(scores, -1).to(query.dtype), value).float().to(query.dtype)


def test_draft_matches_pytorch_air_precision_and_primitive_hidden():
    config, reference, weights = _fixture()
    graph = build_draft(config, 8)
    expected_keys = {f"draft.{name}" for name in config.required_tensor_shapes()}
    assert set(graph.weights) == expected_keys | {"target.embed_tokens.weight", "target.lm_head.weight"}
    assert [(value.name, value.shape, value.dtype) for value in graph.inputs] == [
        ("target_features", (1, 8, config.feature_size), "float16"),
        ("input_ids", (1, 8), "int32"), ("attention_mask", (1, 8), "int32")]
    assert graph.outputs[0].shape == (1, 3) and graph.outputs[0].dtype == "int32"
    assert {value.dtype for value in graph.weights.values()} == {"float16"}
    assert all(node.output.dtype in {"float16", "float32", "int32", "bool"} for node in graph.nodes)

    reference.draft.set_ops(_FP16PrimitiveOps())
    captured = []
    handle = reference.draft.norm.register_forward_hook(lambda module, args, output: captured.append(output[:, 1:].detach()))
    graph.outputs += _named(graph, "draft_hidden", "draft_logits")
    try:
        for length in (1, 4, 8):
            inputs = _inputs(config, length)
            tensors = tuple(torch.from_numpy(inputs[value.name]) for value in graph.inputs)
            top1, hidden, logits = evaluate(graph, inputs, weights)
            torch_top1, torch_hidden, torch_logits = evaluate_torch(graph, inputs, weights)
            with torch.inference_mode():
                primitive_top1 = reference(*tensors)
                primitive_logits = reference.lm_head(captured[-1])
            np.testing.assert_array_equal(top1, primitive_top1.numpy())
            torch.testing.assert_close(torch_top1, primitive_top1, rtol=0, atol=0)
            # Default FP16 tolerances on the same arithmetic backend separate
            # graph-definition correctness from NumPy and device kernels.
            torch.testing.assert_close(torch_hidden, captured[-1])
            torch.testing.assert_close(torch_logits, primitive_logits)
            # NumPy diagnostic only: these bounds are not a GE/device accuracy
            # gate, nor proof of bitwise equivalence to fused SDPA.
            torch.testing.assert_close(torch.from_numpy(hidden), captured[-1], rtol=1e-2, atol=1e-3)
            torch.testing.assert_close(torch.from_numpy(logits), primitive_logits, rtol=1e-2, atol=1e-3)
    finally:
        handle.remove()


def test_draft_rejects_invalid_static_rope_and_block_contract():
    config, _, _ = _fixture()
    for length in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="sequence_length"):
            build_draft(config, length)
    with pytest.raises(ValueError, match="even head_dim"):
        build_draft(replace(config, head_dim=3), 8)
    with pytest.raises(ValueError, match="at least one draft"):
        build_draft(replace(config, block_size=1), 8)
