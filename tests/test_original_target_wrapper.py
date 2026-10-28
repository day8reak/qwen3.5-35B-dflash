"""Original-style target inputs with a CPU test backend, not an AIR/RefData ABI."""
from pathlib import Path

import pytest
import torch

from qwen35_moe_dflash.fused import FusedTargetOutput, OriginalCproTargetWrapper
from qwen35_moe_dflash.original_model_config import OriginalModelConfig

from test_fused_original_quant import model_data


def _assert_state_equal(left, right):
    assert left.position == right.position
    assert len(left.layers) == len(right.layers)
    for a, b in zip(left.layers, right.layers):
        for field in ("conv", "recurrent", "key", "value"):
            x, y = getattr(a, field), getattr(b, field)
            if x is None:
                assert y is None
            else:
                torch.testing.assert_close(x, y, rtol=0, atol=0)


def _external_rows(model_data, ids):
    embedding, scales = model_data[3:5]
    indices = ids.cpu().numpy()
    return torch.from_numpy(embedding[indices].copy()), torch.from_numpy(scales[indices].copy())


def test_external_original_quant_rows_match_token_lookup_and_preserve_inputs(model_data):
    model = model_data[0]
    wrapper = OriginalCproTargetWrapper(model)
    ids = torch.tensor([[3, 5]], dtype=torch.int32)
    rows, scales = _external_rows(model_data, ids)
    source_rows, source_scales = rows.clone(), scales.clone()
    initial = model.initial_state()
    saved = initial.clone()
    metadata = dict(position_ids=torch.tensor([[0, 1]], dtype=torch.int64),
                    new_kv_cache_pos=torch.tensor([0, 1], dtype=torch.int32),
                    effective_length=torch.tensor([2], dtype=torch.int16),
                    allQLen=torch.tensor([2], dtype=torch.int64))
    tokens = wrapper(ids, past_key_values=initial, output_pos=None, **metadata)
    external = wrapper(inputs_embeds=rows, embed_scale=scales,
                       past_key_values=initial, output_pos=None, **metadata)
    assert isinstance(external, FusedTargetOutput)
    assert external.logits.shape == (1, 2, 32) and external.logits.dtype == torch.float32
    torch.testing.assert_close(external.logits, tokens.logits, rtol=0, atol=0)
    torch.testing.assert_close(external.top1, tokens.top1, rtol=0, atol=0)
    torch.testing.assert_close(external.features, tokens.features, rtol=0, atol=0)
    torch.testing.assert_close(external.top1, external.logits.argmax(-1).int(), rtol=0, atol=0)
    _assert_state_equal(external.state, tokens.state)
    _assert_state_equal(initial, saved)
    torch.testing.assert_close(rows, source_rows, rtol=0, atol=0)
    torch.testing.assert_close(scales, source_scales, rtol=0, atol=0)


def test_selected_logits_match_full_rows_without_truncating_state(model_data, monkeypatch):
    model = model_data[0]
    wrapper = OriginalCproTargetWrapper(model)
    ids = torch.tensor([[3, 5, 7]], dtype=torch.int32)
    full = wrapper(ids, output_pos=None)
    head_rows = []
    original_linear = model._linear

    def record_head(x, prefix):
        if prefix == "target.lm_head":
            head_rows.append(x.numel() // x.shape[-1])
        return original_linear(x, prefix)

    monkeypatch.setattr(model, "_linear", record_head)
    for output_pos, row in ((-1, 2), (torch.tensor([0], dtype=torch.int32), 0)):
        head_rows.clear()
        selected = wrapper(ids, output_pos=output_pos)
        assert selected.logits.shape == (1, 1, 32)
        torch.testing.assert_close(selected.logits, full.logits[:, row:row + 1], rtol=0, atol=0)
        torch.testing.assert_close(selected.top1, full.top1[:, row:row + 1], rtol=0, atol=0)
        assert sum(head_rows) == 1, "selected output must not project unused vocabulary rows"
        assert selected.state.position == ids.shape[1]
        _assert_state_equal(selected.state, full.state)


def test_prefill_decode_external_continuation_preserves_committed_state(model_data):
    model = model_data[0]
    wrapper = OriginalCproTargetWrapper(model)
    prefill = wrapper(torch.tensor([[3, 5]], dtype=torch.int32))
    assert prefill.logits.shape == (1, 1, 32)
    committed = prefill.state.clone()
    ids = torch.tensor([[7]], dtype=torch.int64)
    rows, scales = _external_rows(model_data, ids)
    decoded = wrapper(inputs_embeds=rows, embed_scale=scales, past_key_values=prefill.state,
                      position_ids=torch.tensor([[2]], dtype=torch.int32),
                      new_kv_cache_pos=torch.tensor([[2]], dtype=torch.int64),
                      effective_length=torch.tensor([1], dtype=torch.int32),
                      allQLen=torch.tensor([1], dtype=torch.int32))
    direct = model(ids, committed, return_logits=True)
    assert decoded.state.position == 3
    torch.testing.assert_close(decoded.logits, direct.logits, rtol=0, atol=0)
    torch.testing.assert_close(decoded.features, direct.features, rtol=0, atol=0)
    _assert_state_equal(decoded.state, direct.state)
    _assert_state_equal(prefill.state, committed)


def test_invalid_wrapper_arguments_fail_before_target_execution(model_data, monkeypatch):
    model = model_data[0]
    wrapper = OriginalCproTargetWrapper(model)
    ids = torch.tensor([[3, 5]], dtype=torch.int32)
    rows, scales = _external_rows(model_data, ids)

    def unexpected(*args, **kwargs):
        pytest.fail("invalid wrapper metadata must fail before target execution")

    monkeypatch.setattr(model, "forward", unexpected)
    monkeypatch.setattr(model, "forward_embeddings", unexpected)
    invalid = [
        {},
        {"input_ids": ids, "inputs_embeds": rows, "embed_scale": scales},
        {"input_ids": ids, "embed_scale": scales},
        {"input_ids": ids, "past_key_values": []},
        {"input_ids": ids, "position_ids": torch.tensor([[1, 2]], dtype=torch.int64)},
        {"input_ids": ids, "position_ids": torch.tensor([0, 1], dtype=torch.int64)},
        {"input_ids": ids, "position_ids": torch.tensor([[0., 1.]])},
        {"input_ids": ids, "new_kv_cache_pos": torch.tensor([1, 2], dtype=torch.int32)},
        {"input_ids": ids, "new_kv_cache_pos": torch.tensor([[0], [1]], dtype=torch.int32)},
        {"input_ids": ids, "new_kv_cache_pos": torch.tensor([0., 1.])},
        {"input_ids": ids, "effective_length": torch.tensor([0], dtype=torch.int16)},
        {"input_ids": ids, "effective_length": torch.tensor([1], dtype=torch.int64)},
        {"input_ids": ids, "effective_length": torch.tensor(2, dtype=torch.int32)},
        {"input_ids": ids, "effective_length": torch.tensor([2.])},
        {"input_ids": ids, "allQLen": torch.tensor([1], dtype=torch.int64)},
        {"input_ids": ids, "allQLen": torch.tensor([2], dtype=torch.int16)},
        {"input_ids": ids, "allQLen": torch.tensor([[2]], dtype=torch.int32)},
    ]
    for kwargs in invalid:
        with pytest.raises((TypeError, ValueError)):
            wrapper(**kwargs)


def test_out_of_range_output_positions_fail_before_quant_ops(model_data):
    model, _, backend, *_ = model_data
    wrapper = OriginalCproTargetWrapper(model)
    ids = torch.tensor([[3, 5]], dtype=torch.int32)
    for position in (2, -3, True, torch.tensor([2], dtype=torch.int32), torch.tensor([0.])):
        backend.calls.clear()
        with pytest.raises((TypeError, ValueError)):
            wrapper(ids, output_pos=position)
        assert backend.calls == []


def test_wrapper_requires_actual_target_model():
    with pytest.raises(TypeError):
        OriginalCproTargetWrapper(object())


def test_yaml_wrapper_factory_reuses_existing_loader(model_data, monkeypatch):
    model = model_data[0]
    config = OriginalModelConfig(
        source=Path("source.yaml"), model_file=Path("hf"), quanted_pth=Path("quant"),
        embedding_weight_path=Path("embedding.bin"), embedding_scale_path=Path("scale.bin"),
        kv_cache_max_len=2048)
    options = dict(feature_layers=(0, 1), ops=model.ops, quant_ops=model.quantization.ops,
                   device="cpu", recurrent_dtype=torch.float16)
    calls = []

    def load(self, **kwargs):
        calls.append((self, kwargs))
        return model

    monkeypatch.setattr(OriginalModelConfig, "load_pytorch", load)
    wrapper = config.load_pytorch_wrapper(**options)
    assert isinstance(wrapper, OriginalCproTargetWrapper)
    assert calls == [(config, options)]
    actual = wrapper(torch.tensor([[3]], dtype=torch.int32))
    expected = model(torch.tensor([[3]], dtype=torch.int32), return_logits=True)
    torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
