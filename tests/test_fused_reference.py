"""CPU adapter-oracle tests; no fused device or OMC accuracy claim."""
import pytest
import torch
import torch.nn.functional as F

from reference import fused_ops as ops


@pytest.fixture(autouse=True, scope="module")
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def gdn_inputs(sequence=7, batch=1):
    generator = torch.Generator().manual_seed(798)
    q = torch.randn(batch, sequence, 2, 4, generator=generator).half() * .1
    k = torch.randn(batch, sequence, 2, 4, generator=generator).half() * .1
    v = torch.randn(batch, sequence, 2, 3, generator=generator).half() * .1
    g = -torch.rand(batch, sequence, 2, generator=generator).half() * .25
    beta = torch.rand(batch, sequence, 2, generator=generator).half()
    state = torch.randn(batch, 2, 4, 3, generator=generator) * .01
    return q, k, v, g, beta, state


def call_gdn(inputs, *, lengths=None, normalize=True, chunk_size=64):
    q = inputs[0]
    lengths = torch.full((q.shape[0],), q.shape[1], dtype=torch.int16) if lengths is None else lengths
    return ops.chunk_gated_delta_rule(*inputs, lengths, chunk_size=chunk_size,
                                     use_qk_l2norm_in_kernel=normalize)


@pytest.mark.parametrize("normalize", [False, True])
def test_recurrence_matches_pinned_transformers_fp32_reference(normalize):
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import torch_recurrent_gated_delta_rule

    inputs = gdn_inputs()
    actual, actual_state = call_gdn(inputs, normalize=normalize)
    expected, expected_state = torch_recurrent_gated_delta_rule(
        *(item.float() for item in inputs), output_final_state=True,
        use_qk_l2norm_in_kernel=normalize)
    torch.testing.assert_close(actual, expected.half(), rtol=0, atol=0)
    torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)


@pytest.mark.parametrize("split", [1, 17, 64])
def test_fp32_state_split_matches_full_recurrence_across_chunk_boundary(split):
    inputs = gdn_inputs(65)
    expected, final_state = call_gdn(inputs)
    first = [item[:, :split] for item in inputs[:5]] + [inputs[-1]]
    first_out, intermediate = call_gdn(first)
    second = [item[:, split:] for item in inputs[:5]] + [intermediate]
    second_out, actual_state = call_gdn(second)
    torch.testing.assert_close(torch.cat((first_out, second_out), dim=1), expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_state, final_state, rtol=0, atol=0)


def test_effective_length_does_not_commit_padded_or_empty_batch_rows():
    inputs = gdn_inputs(batch=2)
    saved = inputs[-1].clone()
    output, state = call_gdn(inputs, lengths=torch.tensor([3, 0], dtype=torch.int16))
    prefix = [item[:1, :3] for item in inputs[:5]] + [inputs[-1][:1]]
    expected, expected_state = call_gdn(prefix)
    torch.testing.assert_close(output[:1, :3], expected, rtol=0, atol=0)
    torch.testing.assert_close(state[:1], expected_state, rtol=0, atol=0)
    assert not output[0, 3:].count_nonzero()
    assert not output[1].count_nonzero()
    torch.testing.assert_close(state[1], saved[1], rtol=0, atol=0)
    torch.testing.assert_close(inputs[-1], saved, rtol=0, atol=0)


def test_state_dtype_is_explicit_and_fp16_rounds_at_call_boundary():
    inputs = gdn_inputs()
    inputs = (*inputs[:-1], inputs[-1].half())
    output, state = call_gdn(inputs)
    assert output.dtype == state.dtype == torch.float16


@pytest.mark.parametrize("length", [torch.tensor([8], dtype=torch.int16),
                                  torch.tensor([-1], dtype=torch.int16),
                                  torch.tensor([7], dtype=torch.int32),
                                  torch.tensor(7, dtype=torch.int16)])
def test_invalid_effective_length_is_rejected(length):
    with pytest.raises(ValueError, match="effective_length"):
        call_gdn(gdn_inputs(), lengths=length)


@pytest.mark.parametrize("kernel", [1, 4])
def test_conv_prefill_and_incremental_history_match_full_causal_convolution(kernel):
    generator = torch.Generator().manual_seed(2026)
    projected = torch.randn(1, 9, 3, generator=generator)
    weight = torch.randn(3, 1, kernel, generator=generator)
    history = torch.zeros(1, 3, kernel - 1)
    full_history, padded = ops.conv_state_update(projected, history)
    expected = ops.causal_conv1d(padded, weight)
    chunks = []
    for start, end in ((0, 3), (3, 4), (4, 9)):
        history, padded_chunk = ops.conv_state_update(projected[:, start:end], history)
        chunks.append(ops.causal_conv1d(padded_chunk, weight))
    torch.testing.assert_close(torch.cat(chunks, 1), expected)
    torch.testing.assert_close(history, full_history, rtol=0, atol=0)
    full_history.fill_(99)
    assert not torch.all(projected == 99)


def test_rms_gamma_is_effective_and_never_folded_again():
    x = torch.ones(1, 2, 4).half()
    gamma = torch.tensor([1, 2, 3, 4]).half()
    output = ops.rms_norm(x, gamma, 1e-6)
    torch.testing.assert_close(output, gamma.expand_as(x), rtol=0, atol=0)


def test_causal_gqa_prefill_and_cache_appends_match_without_future_leakage():
    generator = torch.Generator().manual_seed(51)
    q = torch.randn(1, 4, 5, 4, generator=generator).half()
    k = torch.randn(1, 2, 5, 4, generator=generator).half()
    v = torch.randn(1, 2, 5, 3, generator=generator).half()
    expected = ops.attention(q, k, v, query_start=0, scale=.5)
    prefix = ops.attention(q[:, :, :3], k[:, :, :3], v[:, :, :3], query_start=0, scale=.5)
    torch.testing.assert_close(prefix, expected[:, :, :3], rtol=0, atol=0)
    key, value = None, None
    results = []
    for token in range(5):
        key, value = ops.cache_update(key, value, k[:, :, token:token+1], v[:, :, token:token+1])
        results.append(ops.attention(q[:, :, token:token+1], key, value, query_start=token, scale=.5))
    torch.testing.assert_close(torch.cat(results, -2), expected, rtol=0, atol=0)
    key.zero_()
    assert k.count_nonzero() == k.numel()


def test_cache_update_initial_return_does_not_alias_input():
    k = torch.ones(1, 2, 1, 4).half()
    v = torch.ones(1, 2, 1, 3).half()
    key, value = ops.cache_update(None, None, k, v)
    key.zero_()
    value.zero_()
    assert k.count_nonzero() == k.numel() and v.count_nonzero() == v.numel()


def test_sparse_moe_matches_explicit_per_token_expert_evaluation():
    generator = torch.Generator().manual_seed(75)
    hidden = torch.randn(1, 4, 6, generator=generator).half() * .2
    router = torch.randn(3, 6, generator=generator).half() * .2
    gate_up = torch.randn(3, 8, 6, generator=generator).half() * .2
    down = torch.randn(3, 6, 4, generator=generator).half() * .2
    actual = ops.moe(hidden, router, gate_up, down, top_k=2)
    expected = torch.zeros_like(hidden)
    for token in range(4):
        x = hidden[0, token]
        probabilities = torch.softmax((router @ x).float(), -1)
        selected = torch.argsort(probabilities, descending=True)[:2]
        weights = (probabilities[selected] / probabilities[selected].sum()).half()
        for expert in sorted(selected.tolist()):
            gate, up = (gate_up[expert] @ x).chunk(2, -1)
            slot = selected.tolist().index(expert)
            expected[0, token] += ((down[expert] @ (F.silu(gate) * up)) * weights[slot])
    torch.testing.assert_close(actual, expected)


def test_metadata_identifies_only_host_reference():
    assert ops.FUSED_OPS_ABI == 1
    assert ops.BACKEND_ID == "host-reference-only"
    assert ops.IS_REFERENCE_BACKEND is True
