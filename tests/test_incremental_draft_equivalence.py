"""Six-layer cached Draft versus independent complete-prefix CPU references.

Only synthetic weights are used. The cached IR is evaluated with the existing
NumPy evaluator; the unchanged complete-prefix IR and original PyTorch Draft
provide separate reference paths. These are mathematical host checks, not AIR,
OMC, HIAI, real-checkpoint numerical acceptance or device evidence.
"""
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from qwen35_moe_dflash.config import Qwen35DFlashConfig
from qwen35_moe_dflash.ge_draft import build_draft
from qwen35_moe_dflash.ge_eval import evaluate
from qwen35_moe_dflash.model import DFlashDraftModel
from qwen35_moe_dflash.ops import _rotate_half
from test_ge_draft import _FP16PrimitiveOps


# Existing NumPy-versus-original-PyTorch FP16 semantic bounds from
# test_ge_draft.py; exact token IDs and old-cache bit patterns remain separate
# requirements. These are not thresholds for a device or real checkpoint.
_RTOL, _ATOL = 1e-2, 1e-3


@pytest.fixture(scope='module')
def case():
    torch.set_num_threads(1)
    torch.manual_seed(314159)
    config = Qwen35DFlashConfig(
        hidden_size=16, intermediate_size=32, vocab_size=64,
        num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2,
        head_dim=4, num_target_layers=40, target_layer_ids=(1, 6, 11, 16, 22, 27, 32, 37),
        layer_types=('sliding_attention',) * 5 + ('full_attention',), block_size=16,
        mask_token_id=63, rms_norm_eps=1e-6, rope_theta=10_000_000.0,
        max_position_embeddings=512, sliding_window=7, use_sliding_window=True,
        attention_bias=False, attention_dropout=0.0, hidden_act='silu', dtype='float16',
        input_embedding_scale=0.75,
    )
    model = DFlashDraftModel(config, ops=_FP16PrimitiveOps(), dtype=torch.float16).eval()
    embedding = nn.Embedding(config.vocab_size, config.hidden_size, dtype=torch.float16)
    head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=torch.float16)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if 'norm.weight' in name: parameter.uniform_(0.7, 1.2)
            else: parameter.normal_(std=0.08)
        embedding.weight.normal_(std=0.2)
        head.weight.normal_(std=0.1)
    weights = {f'draft.{name}': value.detach().numpy().copy() for name, value in model.state_dict().items()}
    weights['target.embed_tokens.weight'] = embedding.weight.detach().numpy().copy()
    weights['target.lm_head.weight'] = head.weight.detach().numpy().copy()
    rng = np.random.default_rng(4242)
    features = rng.normal(0, .2, (1, 96, config.feature_size)).astype(np.float16)
    return dict(config=config, model=model, embedding=embedding, head=head, weights=weights, features=features)


def cached_graph(case, rows=16, capacity=48):
    from qwen35_moe_dflash.ge_incremental_draft import build_incremental_draft
    graph = build_incremental_draft(case['config'], context_rows=rows, cache_capacity=capacity)
    assert set(graph.weights) == set(case['weights'])
    assert graph.weights == build_draft(case['config'], 1).weights
    return graph


def zero_state(case, capacity=48, *, sentinel=False):
    config = case['config']
    shape = (1, config.num_key_value_heads, capacity, config.head_dim)
    return [np.full(shape, np.float16((index + 1) / 64 if sentinel else 0), dtype=np.float16) for index in range(12)]


def run_cached(case, graph, state, new_features, start, anchor, count):
    rows = graph.inputs[0].shape[1]
    # The padded scratch rows are intentionally finite, nonzero and different
    # from the true continuation. valid_rows must hide their contribution.
    features = np.full(graph.inputs[0].shape, np.float16(2.25), dtype=np.float16)
    valid = new_features.shape[1]
    features[:, :valid] = new_features
    inputs = {graph.inputs[0].name: features,
        'start_position': np.array([start], np.int32), 'valid_rows': np.array([valid], np.int32),
        'anchor': np.array([anchor], np.int32), 'proposal_count': np.array([count], np.int32)}
    assert 0 <= valid <= rows and len(graph.inputs) == 17
    inputs.update({value.name: raw for value, raw in zip(graph.inputs[5:], state)})
    before = [raw.copy() for raw in state]
    output = evaluate(graph, inputs, case['weights'])
    assert len(output) == 14 and output[0].shape == (1, 15)
    assert output[1].shape == (1, 15, case['config'].hidden_size)
    for old, raw in zip(before, state): np.testing.assert_array_equal(old.view(np.uint16), raw.view(np.uint16))
    return output[0], output[1], output[2:]


def full_prefix_reference(case, context, anchor, count):
    """No cache, no incremental builder helpers and no replayed block KV."""
    config, model = case['config'], case['model']
    length = context.shape[1]
    # The existing cache-free DSL excludes its last valid feature (anchor).
    # Poison it so accidentally admitting that row would be observable.
    features = np.concatenate((context, np.full((1, 1, config.feature_size), 1.5, np.float16)), axis=1)
    ids = np.zeros((1, length + 1), np.int32); ids[0, -1] = anchor
    mask = np.ones_like(ids)
    full = build_draft(replace(config, block_size=count + 1), length + 1)
    full.outputs.append(next(node.output for node in full.nodes if node.output.name == 'draft_hidden'))
    expected_top1, expected_hidden = evaluate(full,
        {'target_features': features, 'input_ids': ids, 'attention_mask': mask}, case['weights'])
    # The original eager model provides an independent mathematical reference,
    # including the bidirectional last layer over only anchor + active masks.
    block = torch.full((1, count + 1), config.mask_token_id, dtype=torch.long)
    block[0, 0] = anchor
    with torch.inference_mode():
        noise = model.embed_block(block, case['embedding'].weight)
        positions = torch.arange(length + count + 1, dtype=torch.int32).reshape(1, -1)
        hidden = model.draft_hidden(torch.from_numpy(context), noise, positions,
            context_attention_mask=torch.ones((1, length), dtype=torch.bool))
        top1 = case['head'](hidden).argmax(-1).to(torch.int32)
    np.testing.assert_array_equal(expected_top1, top1.numpy())
    np.testing.assert_allclose(expected_hidden, hidden.numpy(), rtol=_RTOL, atol=_ATOL)
    return expected_top1, expected_hidden


def reference_context_kv(case, context):
    """Project complete committed context independently through original layers.

    The shared fc/hidden_norm result is the context input to every Draft layer;
    it is never replaced by the preceding layer's transient block hidden.
    """
    model, config = case['model'], case['config']
    length = context.shape[1]
    with torch.inference_mode():
        projected = model.hidden_norm(model.fc(torch.from_numpy(context)))
        cosine, sine = model.rotary(torch.arange(length, dtype=torch.int32)[None], torch.float16)
        cosine, sine = cosine[:, None], sine[:, None]
        result = []
        for layer in model.layers:
            base = layer.self_attn
            key = base.k_norm(base.k_proj(projected).reshape(1, length, config.num_key_value_heads, config.head_dim)).transpose(1, 2)
            key = key * cosine + _rotate_half(key) * sine
            value = base.v_proj(projected).reshape(1, length, config.num_key_value_heads, config.head_dim).transpose(1, 2)
            result.extend((key.numpy(), value.numpy()))
    return result


def compare_round(case, graph, state, context, start, valid, anchor, count):
    top1, hidden, next_state = run_cached(case, graph, state, context[:, start:start + valid], start, anchor, count)
    committed = context[:, :start + valid]
    expected_top1, expected_hidden = full_prefix_reference(case, committed, anchor, count)
    np.testing.assert_array_equal(top1[:, :count], expected_top1)
    np.testing.assert_allclose(hidden[:, :count], expected_hidden, rtol=_RTOL, atol=_ATOL)
    for old, actual, expected in zip(state, next_state, reference_context_kv(case, committed)):
        np.testing.assert_array_equal(actual[:, :, :start].view(np.uint16), old[:, :, :start].view(np.uint16))
        np.testing.assert_allclose(actual[:, :, :start + valid], expected, rtol=_RTOL, atol=_ATOL)
        np.testing.assert_array_equal(actual[:, :, start + valid:].view(np.uint16), old[:, :, start + valid:].view(np.uint16))
    return top1, hidden, next_state


@pytest.mark.parametrize('count', [1, 3, 7, 15])
def test_six_layer_append_rounds_match_independent_complete_prefix_for_full_and_short_blocks(case, count):
    graph, state = cached_graph(case), zero_state(case, sentinel=True)
    start = 0
    for valid, anchor in ((3, 1), (1, 7), (4, 9), (2, 11), (8, 13), (7, 17)):
        _, _, state = compare_round(case, graph, state, case['features'], start, valid, anchor, count)
        start += valid
    assert start > graph.inputs[0].shape[1]


def test_actual_c16_gear_consumes_prompt_once_and_appends_partial_commits_with_absolute_rope(case):
    graph, state = cached_graph(case, rows=16, capacity=96), zero_state(case, 96, sentinel=True)
    start = 0
    for valid, anchor, count in ((16, 1, 15), (1, 7, 1), (4, 9, 7), (16, 11, 15)):
        _, _, state = compare_round(case, graph, state, case['features'], start, valid, anchor, count)
        start += valid
    assert start == 37


def test_full_prompt64_context_gear_then_c16_share_the_same_committed_cache(case):
    prefill = cached_graph(case, rows=64, capacity=192)
    decode = cached_graph(case, rows=16, capacity=192)
    state = zero_state(case, 192, sentinel=True)
    _, _, state = compare_round(case, prefill, state, case['features'], 0, 64, 1, 15)
    _, _, state = compare_round(case, decode, state, case['features'], 64, 1, 7, 3)
    compare_round(case, decode, state, case['features'], 65, 3, 9, 1)


def test_context_only_primer_and_live_proposals_produce_identical_persistent_cache(case):
    graph, old = cached_graph(case), zero_state(case, sentinel=True)
    _, _, primer = run_cached(case, graph, old, case['features'][:, :8], 0, 7, 0)
    _, _, proposal = run_cached(case, graph, old, case['features'][:, :8], 0, 7, 15)
    for a, b in zip(primer, proposal): np.testing.assert_array_equal(a.view(np.uint16), b.view(np.uint16))
    compare_round(case, graph, primer, case['features'], 8, 0, 9, 3)


def test_zero_new_context_reuses_existing_kv_without_admitting_anchor_or_noise(case):
    graph, original = cached_graph(case), zero_state(case, sentinel=True)
    _, _, state = compare_round(case, graph, original, case['features'], 0, 3, 7, 15)
    first, _, again = compare_round(case, graph, state, case['features'], 3, 0, 9, 3)
    second, _, repeated = compare_round(case, graph, again, case['features'], 3, 0, 9, 3)
    np.testing.assert_array_equal(first, second)
    for a, b in zip(again, repeated): np.testing.assert_array_equal(a.view(np.uint16), b.view(np.uint16))


@pytest.mark.parametrize('count', [1, 15])
def test_initial_empty_context_uses_only_anchor_and_active_masks(case, count):
    graph, state = cached_graph(case), zero_state(case, sentinel=True)
    compare_round(case, graph, state, case['features'], 0, 0, 7, count)


@pytest.mark.parametrize('accepted', [0, 2, 15])
def test_rejection_or_full_acceptance_appends_only_committed_target_features_before_next_anchor(case, accepted):
    graph, initial = cached_graph(case, rows=16, capacity=96), zero_state(case, 96, sentinel=True)
    _, _, prompt_state = compare_round(case, graph, initial, case['features'], 0, 5, 1, 15)
    # A speculative draft may propose all15, but none of its transient K/V
    # becomes committed. Target returns only old anchor + accepted prefix rows.
    _, _, proposed = run_cached(case, graph, prompt_state, case['features'][:, :0], 5, 1, 15)
    for a, b in zip(prompt_state, proposed):
        np.testing.assert_array_equal(a[:, :, :5].view(np.uint16), b[:, :, :5].view(np.uint16))
    committed_rows = accepted + 1
    _, _, committed = compare_round(case, graph, proposed, case['features'], 5, committed_rows, 17, 3)
    # Reusing that boundary with no new context must remain a complete-prefix
    # computation at absolute position 5+committed_rows, not position zero.
    compare_round(case, graph, committed, case['features'], 5 + committed_rows, 0, 17, 3)


def test_last_legal_context_write_and_unused_tail_sentinels_are_preserved(case):
    graph, state = cached_graph(case, rows=16, capacity=24), zero_state(case, 24, sentinel=True)
    _, _, state = compare_round(case, graph, state, case['features'], 0, 16, 3, 1)
    compare_round(case, graph, state, case['features'], 16, 8, 3, 1)
