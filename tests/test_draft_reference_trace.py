"""CPU RMS diagnosis fixtures; no device or full-checkpoint accuracy claims."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from qwen35_moe_dflash.draft_reference_trace import replay_graph
from qwen35_moe_dflash.model import DFlashRMSNorm
from qwen35_moe_dflash.ops import TorchDFlashOps
from test_draft_reference import prepared


class FP16Primitives(TorchDFlashOps):
    def rms_norm(self, x, weight, eps):
        return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)) * weight


class ControlledGraph(nn.Module):
    """Real RMS modules and head with an intentional FP16-square overflow."""
    def __init__(self, *, pre_norm=False):
        super().__init__()
        self.draft = nn.Module()
        self.draft.config = SimpleNamespace(feature_size=4, hidden_size=4, vocab_size=8, block_size=4)
        self.original_ops = FP16Primitives()
        self.draft.ops = self.original_ops
        if pre_norm:
            self.draft.pre_norm = DFlashRMSNorm(4, 1e-6, self.original_ops, dtype=torch.float16)
        self.draft.norm = DFlashRMSNorm(4, 1e-6, self.original_ops, dtype=torch.float16)
        self.lm_head = nn.Linear(4, 8, bias=False, dtype=torch.float16)
        self.fail_after_norm = False
        self.input_multiplier = 1.0
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, DFlashRMSNorm):
                    module.weight.fill_(1)
            self.lm_head.weight.zero_()
            self.lm_head.weight[1].fill_(1)
        self.eval()

    def forward(self, features, ids, mask):
        value = features * self.input_multiplier
        if hasattr(self.draft, 'pre_norm'):
            value = self.draft.pre_norm(value)
        hidden = self.draft.norm(value)
        if self.fail_after_norm:
            raise RuntimeError('intentional graph failure')
        return self.lm_head(hidden[:, 1:, :]).argmax(-1).to(torch.int32)


@pytest.fixture
def exact_inputs():
    features = np.full((1, 4, 4), 300, dtype='<f2')
    features[:, 0] = 0  # Already-zero row must not count as a new collapse.
    return [features, np.asarray([[1, 2, 3, 4]], dtype='<i4'), np.asarray([[1, 1, 1, 0]], dtype='<i4')]


def _hooks(graph):
    return {name: (len(node._forward_hooks), len(node._forward_pre_hooks))
            for name, node in graph.named_modules()}


def test_exact_legacy_replay_is_retained_and_fp32_only_recovers_overflow(exact_inputs):
    graph = ControlledGraph()
    inputs_before = [array.copy() for array in exact_inputs]
    weights_before = {name: value.detach().clone() for name, value in graph.named_parameters()}
    hook_counts = _hooks(graph)
    updates = []
    legacy, trace = replay_graph(graph, exact_inputs, rms_mode='legacy-fp16', progress=updates.append)
    assert legacy['draft_top1'].dtype == np.dtype('<i4')
    assert legacy['draft_hidden'].dtype == np.dtype('<f2')
    assert not np.any(legacy['draft_top1']) and not np.any(legacy['draft_hidden'])
    assert trace['validation'] == {'status': 'FAILED', 'finite': True, 'nondegenerate': False,
                                    'errors': ['degenerate all-zero draft hidden/logits output']}
    row = trace['rms_nodes'][0]
    assert row['name'] == 'draft.norm'
    assert row['input_all_zero_rows'] == 1 and row['newly_zero_rows'] == 3
    assert row['fp16_square_nonfinite_count'] == 12
    assert row['fp16_mean_square_nonfinite_rows'] == 3
    assert row['fp16_rsqrt_zero_rows'] == 3
    assert row['newly_zero_rows_with_fp16_square_overflow'] == 3
    assert trace['first_collapse']['cause_candidate'] == 'FP16_SQUARE_OVERFLOW_WITH_ZERO_OUTPUT'
    assert trace['first_nonfinite'] is None  # Intermediate overflow, finite boundaries.
    assert trace['logits']['all_zero'] is True
    assert len(updates) == 3
    stable, stable_trace = replay_graph(graph, exact_inputs, rms_mode='fp32-stats')
    np.testing.assert_array_equal(stable['draft_top1'], np.ones((1, 3), dtype='<i4'))
    np.testing.assert_array_equal(stable['draft_hidden'], np.ones((1, 3, 4), dtype='<f2'))
    assert stable_trace['validation']['status'] == 'PASSED'
    assert stable_trace['first_collapse'] is None
    assert stable_trace['rms_nodes'][0]['fp16_square_nonfinite_count'] == 12
    assert stable_trace['rms_nodes'][0]['newly_zero_rows'] == 0
    assert graph.draft.norm.ops is graph.original_ops and graph.draft.ops is graph.original_ops
    assert _hooks(graph) == hook_counts
    restored, _ = replay_graph(graph, exact_inputs, rms_mode='legacy-fp16')
    for key in legacy:
        assert restored[key].tobytes() == legacy[key].tobytes()
    for before, after in zip(inputs_before, exact_inputs):
        np.testing.assert_array_equal(before, after)
    for name, value in graph.named_parameters():
        torch.testing.assert_close(value, weights_before[name], rtol=0, atol=0)
    json.dumps(trace, allow_nan=False)
    json.dumps(stable_trace, allow_nan=False)


def test_first_new_collapse_does_not_blame_downstream_already_zero_input(exact_inputs):
    _, trace = replay_graph(ControlledGraph(pre_norm=True), exact_inputs, rms_mode='legacy-fp16')
    first, last = trace['rms_nodes']
    assert first['name'] == trace['first_collapse']['name'] == 'draft.pre_norm'
    assert first['newly_zero_rows'] == 3
    assert last['name'] == 'draft.norm' and last['input_all_zero_rows'] == 4
    assert last['newly_zero_rows'] == 0


def test_zero_scale_does_not_claim_square_overflow_caused_zero_output(exact_inputs):
    graph = ControlledGraph()
    with torch.no_grad():
        graph.draft.norm.weight.zero_()
    _, trace = replay_graph(graph, exact_inputs, rms_mode='legacy-fp16')
    assert trace['rms_nodes'][0]['fp16_square_nonfinite_count'] == 12
    assert trace['rms_nodes'][0]['newly_zero_rows'] == 0
    assert trace['first_collapse'] is None
    assert trace['validation']['status'] == 'FAILED'


def test_fp32_normalization_matches_exact_cast_then_scale_order(exact_inputs):
    graph = ControlledGraph()
    exact_inputs[0][:] = np.asarray([101, 202, 303, 404], dtype='<f2')
    with torch.no_grad():
        graph.draft.norm.weight.copy_(torch.tensor([.123, .789, 1.357, 2.468], dtype=torch.float16))
    expected = TorchDFlashOps().rms_norm(torch.from_numpy(exact_inputs[0]), graph.draft.norm.weight, 1e-6)
    values, _ = replay_graph(graph, exact_inputs, rms_mode='fp32-stats')
    assert values['draft_hidden'].tobytes() == expected[:, 1:, :].detach().numpy().tobytes()


@pytest.mark.parametrize('failure', ['forward', 'progress'])
def test_ops_and_existing_hooks_are_restored_on_every_failure(exact_inputs, failure):
    graph = ControlledGraph(pre_norm=True)
    called = []
    existing = graph.draft.norm.register_forward_hook(lambda *args: called.append(True))
    baseline = _hooks(graph)
    graph.fail_after_norm = failure == 'forward'
    def progress(message):
        if failure == 'progress' and message.startswith('RMS'):
            raise RuntimeError('intentional progress failure')
    with pytest.raises(RuntimeError, match='intentional'):
        replay_graph(graph, exact_inputs, rms_mode='fp32-stats', progress=progress)
    assert _hooks(graph) == baseline
    assert graph.draft.pre_norm.ops is graph.original_ops
    assert graph.draft.norm.ops is graph.original_ops
    existing.remove()


def test_nonfinite_boundary_summaries_remain_json_safe(exact_inputs):
    graph = ControlledGraph()
    graph.input_multiplier = 1000
    _, trace = replay_graph(graph, exact_inputs, rms_mode='legacy-fp16')
    assert trace['first_nonfinite']['name'] == 'draft.norm'
    assert trace['first_nonfinite']['boundary'] == 'input'
    assert trace['validation']['finite'] is False
    assert trace['validation']['status'] == 'FAILED'
    json.dumps(trace, allow_nan=False)


@pytest.mark.parametrize('fault', ['mode', 'dtype', 'shape', 'mask', 'nonfinite', 'training', 'epsilon'])
def test_invalid_contract_fails_before_attaching_hooks(exact_inputs, fault):
    graph = ControlledGraph()
    mode = 'legacy-fp16'
    if fault == 'mode':
        mode = 'fp32-all-ops'
    elif fault == 'dtype':
        exact_inputs[0] = exact_inputs[0].astype(np.float32)
    elif fault == 'shape':
        exact_inputs[1] = exact_inputs[1][:, :2]
    elif fault == 'mask':
        exact_inputs[2][:] = [[1, 0, 1, 0]]
    elif fault == 'nonfinite':
        exact_inputs[0][0, 1, 0] = np.inf
    elif fault == 'training':
        graph.train()
    elif fault == 'epsilon':
        graph.draft.norm.eps = float('nan')
    before = _hooks(graph)
    with pytest.raises(ValueError):
        replay_graph(graph, exact_inputs, rms_mode=mode)
    assert _hooks(graph) == before
    assert graph.draft.norm.ops is graph.original_ops


def test_named_retained_inputs_use_exact_same_order(exact_inputs):
    names = ['target_features', 'input_ids', 'attention_mask']
    named = dict(reversed(list(zip(names, exact_inputs))))
    arrays, trace = replay_graph(ControlledGraph(), named, rms_mode='fp32-stats')
    assert [row['name'] for row in trace['inputs']] == names
    assert np.all(arrays['draft_hidden'] == 1)


def test_trace_does_not_retain_full_logits_or_input_tensor_values(exact_inputs):
    _, trace = replay_graph(ControlledGraph(), exact_inputs, rms_mode='fp32-stats')
    assert trace['logits']['elements'] == 24
    assert 'values' not in trace['logits']
    assert all(isinstance(row, dict) for row in trace['rms_nodes'])
    assert len(json.dumps(trace, allow_nan=False)) < 20000


def test_six_layer_mmap_replay_reproduces_existing_fixture_bytes_and_restores_dispatch(prepared, tmp_path):
    from qwen35_moe_dflash import draft_reference
    from qwen35_moe_dflash.hashing import sha256_file
    root, manifest, _, _, _ = prepared
    fixture = tmp_path / 'retained-reference'
    original = draft_reference.prepare_draft_reference(root, fixture, valid_length=3, run_reference=False)
    inputs = {row['name']: np.fromfile(fixture / row['path'], dtype=row['dtype']).reshape(row['shape'])
              for row in original['inputs']}
    graph = draft_reference.load_draft_reference(root, rms_mode='legacy-fp16')
    # Build an old-contract retained golden independently. New preparation uses
    # FP32 RMS and must never be mistaken for the legacy replay oracle.
    recorded_hidden = []
    handle = graph.draft.norm.register_forward_hook(
        lambda module, args, output: recorded_hidden.append(output[:, 1:, :].detach().clone()))
    try:
        with torch.inference_mode():
            old_top1 = graph(*(torch.from_numpy(inputs[name]) for name in
                               ('target_features', 'input_ids', 'attention_mask'))).numpy().tobytes()
    finally:
        handle.remove()
    old_hidden = recorded_hidden[0].numpy().tobytes()
    dispatch = {name: module.ops for name, module in graph.named_modules() if hasattr(module, 'ops')}
    legacy, trace = replay_graph(graph, inputs, rms_mode='legacy-fp16')
    assert legacy['draft_top1'].tobytes() == old_top1
    assert legacy['draft_hidden'].tobytes() == old_hidden
    assert trace['validation']['status'] == 'PASSED'
    assert len(trace['rms_nodes']) == 26
    expected_names = {name for name, module in graph.named_modules() if isinstance(module, DFlashRMSNorm)}
    assert {row['name'] for row in trace['rms_nodes']} == expected_names
    altered, altered_trace = replay_graph(graph, inputs, rms_mode='fp32-stats')
    assert altered_trace['validation']['status'] == 'PASSED'
    assert altered['draft_hidden'].shape == legacy['draft_hidden'].shape
    assert all(graph.get_submodule(name).ops is original_ops for name, original_ops in dispatch.items())
    assert all(sha256_file(root / row['path']) == row['sha256'] for row in manifest['weights'].values())
    assert not any(module._forward_hooks for module in graph.modules())
