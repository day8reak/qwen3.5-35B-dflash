"""Structural/ABI guards for the new six-layer cached graph; no GE/DDK runs."""
from dataclasses import replace

import numpy as np
import pytest

from qwen35_moe_dflash.config import Qwen35DFlashConfig
from qwen35_moe_dflash.ge_export import configurations
from qwen35_moe_dflash.ge_draft import build_draft
from qwen35_moe_dflash.ge_eval import evaluate
from qwen35_moe_dflash.ge_incremental_draft import (build_incremental_draft, incremental_draft_abi,
    validate_incremental_draft_controls)


def tiny():
    return replace(Qwen35DFlashConfig.from_dict(configurations()[1]), hidden_size=16,
        intermediate_size=32, vocab_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=4,
        num_target_layers=8, target_layer_ids=tuple(range(8)), mask_token_id=63,
        max_position_embeddings=256, sliding_window=5, dtype='float16')


def fixture(config=None, *, rows=16, capacity=32, start=5, valid=3, proposals=3):
    config = config or tiny()
    graph = build_incremental_draft(config, rows, capacity)
    rng = np.random.default_rng(9271)
    weights = {name: rng.normal(0, .04, value.shape).astype('<f2') for name, value in graph.weights.items()}
    for name in weights:
        if 'norm.weight' in name: weights[name].fill(1)
    inputs = {'target_features': rng.normal(0, .1, (1, rows, config.feature_size)).astype('<f2'),
        'start_position': np.array([start], dtype='<i4'), 'valid_rows': np.array([valid], dtype='<i4'),
        'anchor': np.array([9], dtype='<i4'), 'proposal_count': np.array([proposals], dtype='<i4')}
    for value in graph.inputs[5:]:
        # Distinct finite sentinels, including negative zero, expose any rewrite
        # of the prefix, padding region or future suffix.
        array = rng.normal(0, .1, value.shape).astype('<f2')
        array.reshape(-1)[::19] = np.float16(-0.)
        inputs[value.name] = array
    return config, graph, weights, inputs


@pytest.mark.parametrize('rows', [16, 64])
@pytest.mark.parametrize('hidden', [False, True])
def test_locked_full_graph_has_original71_weights_six_layers_eight_taps_and_explicit17_inputs(rows, hidden):
    config = Qwen35DFlashConfig.from_dict(configurations()[1])
    graph = build_incremental_draft(config, rows, 192, hidden)
    original = build_draft(config, 64)
    assert {name: value.shape for name, value in graph.weights.items()} == {name: value.shape for name, value in original.weights.items()}
    assert len(graph.weights) == 71 and graph.inputs[0].shape == (1, rows, 16384)
    assert [value.name for value in graph.inputs[:5]] == ['target_features', 'start_position', 'valid_rows', 'anchor', 'proposal_count']
    assert all(value.shape == (1,) and value.dtype == 'int32' for value in graph.inputs[1:5])
    assert [(value.name, value.shape, value.dtype) for value in graph.inputs[5:]] == [
        (name, (1, 8, 192, 128), 'float16') for layer in range(6) for name in (f'past_k_{layer}', f'past_v_{layer}')]
    expected = ['draft_top1', *(['draft_hidden'] if hidden else []), *[name for layer in range(6) for name in (f'present_k_{layer}', f'present_v_{layer}')]]
    assert [value.name for value in graph.outputs] == expected
    assert graph.outputs[0].shape == (1, 15) and graph.outputs[0].dtype == 'int32'
    assert graph._values['draft_hidden'].shape == (1, 15, 2048)
    assert not {'maximum', 'minimum', 'scatter', 'cache_update'} & {node.kind for node in graph.nodes}


@pytest.mark.parametrize('start,valid', [(0, 0), (0, 16), (5, 3), (16, 16), (32, 0)])
def test_only_declared_committed_context_interval_changes_in_all12_caches(start, valid):
    _, graph, weights, inputs = fixture(start=start, valid=valid)
    outputs = evaluate(graph, inputs, weights)
    for index, result in enumerate(outputs[2:]):
        original = inputs[graph.inputs[5 + index].name]
        np.testing.assert_array_equal(result[:, :, :start].view('<u2'), original[:, :, :start].view('<u2'))
        np.testing.assert_array_equal(result[:, :, start + valid:].view('<u2'), original[:, :, start + valid:].view('<u2'))
    if valid == 0:
        for index, result in enumerate(outputs[2:]):
            assert result.tobytes() == inputs[graph.inputs[5 + index].name].tobytes()


@pytest.mark.parametrize('proposals', [0, 1, 3, 7, 15])
def test_absolute_positions_write_indices_and_all_layer_masks_include_proposal_tail_guard(proposals):
    _, graph, weights, inputs = fixture(proposals=proposals)
    graph.outputs = [graph._values[name] for name in ('context_positions', 'block_positions', 'positions',
        'cache_gather_indices', 'cache_write_mask', 'block_ids', 'layer_0_visible', 'layer_5_visible')]
    context, block, all_positions, gather, write, ids, sliding, full = evaluate(graph, inputs, weights)
    np.testing.assert_array_equal(context, np.arange(5, 21, dtype=np.int32)[None])
    np.testing.assert_array_equal(block, np.arange(8, 24, dtype=np.int32)[None])
    np.testing.assert_array_equal(all_positions[:, :32], np.arange(32, dtype=np.int32)[None])
    np.testing.assert_array_equal(gather, np.clip(np.arange(32) - 5, 0, 15))
    np.testing.assert_array_equal(write.reshape(-1), (np.arange(32) >= 5) & (np.arange(32) < 8))
    assert ids.tolist() == [[9] + [63] * 15]
    expected_keys = np.concatenate((np.arange(32) < 8, np.arange(16) <= proposals))
    np.testing.assert_array_equal(full, np.broadcast_to(expected_keys, (1, 1, 16, 48)))
    qpos, kpos = block.reshape(1, 1, 16, 1), all_positions.reshape(1, 1, 1, 48)
    expected_sliding = full & (kpos <= qpos) & ((qpos - kpos) < 5)
    expected_sliding[:, :, proposals + 1:, 32] = True
    np.testing.assert_array_equal(sliding, expected_sliding)
    assert not np.any(full[..., 32 + proposals + 1:]), 'last full-attention layer must not attend inactive proposals'


def test_persistent_outputs_do_not_depend_on_temporary_anchor_block_or_proposal_count():
    _, graph, weights, inputs = fixture()
    first = evaluate(graph, inputs, weights)
    changed = dict(inputs, anchor=np.array([31], dtype='<i4'), proposal_count=np.array([15], dtype='<i4'))
    second = evaluate(graph, changed, weights)
    assert all(a.tobytes() == b.tobytes() for a, b in zip(first[2:], second[2:]))
    # History arrives only through the final Where branch, not a key norm or
    # RoPE arithmetic node. Its physical rows already contain rotated keys.
    consumers = {value.name: [] for value in graph.inputs[5:]}
    for node in graph.nodes:
        for value in node.inputs:
            if value.name in consumers: consumers[value.name].append(node)
    assert all(len(nodes) == 1 and nodes[0].kind == 'where' and nodes[0].inputs[2].name == name for name, nodes in consumers.items())


@pytest.mark.parametrize('name,value', [('context_rows', 1), ('context_rows', True), ('context_rows', 32),
    ('cache_capacity', 0), ('cache_capacity', 15), ('cache_capacity', True), ('cache_capacity', (1 << 31)), ('include_hidden', 1)])
def test_static_geometry_rejects_unavailable_gears_and_overflow_before_constructing_graph(name, value):
    options = {'context_rows': 16, 'cache_capacity': 192, 'include_hidden': True}; options[name] = value
    with pytest.raises(ValueError): build_incremental_draft(tiny(), **options)


@pytest.mark.parametrize('change', [{'num_hidden_layers': 5, 'layer_types': ('sliding_attention',) * 4 + ('full_attention',)},
    {'target_layer_ids': tuple(range(7))}, {'block_size': 8}, {'layer_types': ('full_attention',) * 6}, {'head_dim': 3}])
def test_graph_refuses_partial_or_other_checkpoint_topologies(change):
    with pytest.raises(ValueError): build_incremental_draft(replace(tiny(), **change), 16, 32)


@pytest.mark.parametrize('change', [{'start_position': -1}, {'valid_rows': 17}, {'anchor': 64}, {'proposal_count': 16},
    {'start_position': 31, 'valid_rows': 2}, {'start_position': True}, {'anchor': 1.0}, {'proposal_count': (1 << 31)}])
def test_host_controls_are_checked_as_explicit_bounded_integers(change):
    values = {'start_position': 5, 'valid_rows': 3, 'anchor': 9, 'proposal_count': 3}; values.update(change)
    with pytest.raises(ValueError): validate_incremental_draft_controls(tiny(), **values, cache_capacity=32)


def test_host_last_valid_positions_and_abi_scope_are_explicit():
    controls = validate_incremental_draft_controls(tiny(), start_position=16, valid_rows=16, anchor=0,
        proposal_count=0, context_rows=16, cache_capacity=32)
    assert controls['context_end'] == 32
    metadata = incremental_draft_abi(tiny(), 16, 32)
    assert metadata['temporary_block_kv'] == 'NEVER_PERSISTED' and metadata['proposal_count_range'] == [0, 15]
    assert metadata['model_compilation'] == metadata['device_execution'] == metadata['numerical_validation'] == 'NOT_RUN'
    assert metadata['graph']['node_count'] > 500 and len(metadata['graph']['inputs']) == 17
