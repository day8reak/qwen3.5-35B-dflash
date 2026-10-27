"""Host validation of the independent fork; no native kernel success is implied."""
from types import ModuleType

import pytest
import torch

from qwen35_moe_dflash.ge_export import configurations, prepare_pair
from qwen35_moe_dflash.ge_weights import tiny_weights
from qwen35_moe_dflash.fused import FusedTargetModel, FusedTargetSession, ModuleFusedOps, load_prepared_target
from qwen35_moe_dflash.fused import SelectiveFusedOps
from qwen35_moe_dflash.fused.model import required_weight_shapes
from qwen35_moe_dflash.fused.session import LayerState, TargetState


def make_model(*, feature_layers=(0, 2, 3), ops=None, recurrent_dtype=torch.float16):
    from reference import fused_ops
    c, _ = configurations(tiny=True)
    weights = {k: torch.from_numpy(v.copy()) for k, v in tiny_weights(required_weight_shapes(c)).items()}
    ops = ops or ModuleFusedOps(fused_ops, allow_reference=True)
    return FusedTargetModel(c, weights, feature_layers=feature_layers, ops=ops,
                            recurrent_dtype=recurrent_dtype), weights


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(1)


def assert_state_close(a, b):
    assert a.position == b.position
    for la, lb in zip(a.layers, b.layers):
        for name in ("conv", "recurrent", "key", "value"):
            va, vb = getattr(la, name), getattr(lb, name)
            if va is None:
                assert vb is None
            else:
                torch.testing.assert_close(va, vb)


def test_streamed_tokens_match_full_prefill_and_preserve_inputs():
    # Explicit reference-only FP32 state isolates cache continuity from the
    # FP16 state rounding introduced at each different invocation boundary.
    model, weights = make_model(recurrent_dtype=torch.float32)
    snapshots = {k: v.clone() for k, v in weights.items()}
    tokens = torch.tensor([[1, 2, 3, 8, 7]], dtype=torch.int32)
    start = model.initial_state()
    full = model(tokens, start)
    streamed = [model(tokens[:, :2], start)]
    for index in range(2, tokens.shape[1]):
        previous = streamed[-1].state
        frozen = previous.clone()
        streamed.append(model(tokens[:, index:index+1], previous))
        assert_state_close(previous, frozen)
    torch.testing.assert_close(torch.cat([r.top1 for r in streamed], dim=1), full.top1, rtol=0, atol=0)
    torch.testing.assert_close(torch.cat([r.features for r in streamed], dim=1), full.features)
    assert_state_close(streamed[-1].state, full.state)
    assert_state_close(start, model.initial_state())
    for name, value in weights.items():
        torch.testing.assert_close(value, snapshots[name], rtol=0, atol=0)
        assert model.weight(name).data_ptr() != value.data_ptr()
    assert full.features.dtype == torch.float16
    assert full.features.shape == (1, 5, 48)
    assert full.state.layers[3].key.shape == (1, 1, 5, 8)


def test_weight_shapes_match_preserved_direct_graph():
    from qwen35_moe_dflash.ge_target import build_target
    from qwen35_moe_dflash.config import Qwen35DFlashConfig
    for tiny in (True, False):
        c, d = configurations(tiny=tiny)
        # Full graph would needlessly expand 40 layers; compare full shape counts
        # with configured geometry, and test exact old graph mapping on tiny.
        if tiny:
            graph = build_target(c, Qwen35DFlashConfig.from_dict(d), 8)
            assert required_weight_shapes(c) == {k: v.shape for k, v in graph.weights.items()}
        else:
            shapes = required_weight_shapes(c)
            assert shapes['target.layers.0.linear_attn.in_proj_qkv.weight'] == (8192, 2048)
            assert shapes['target.layers.39.self_attn.q_proj.weight'] == (8192, 2048)
            assert len([k for k in shapes if k.endswith('.A_log')]) == 30


def test_prepared_loader_reuses_hash_verified_target_weights(tmp_path, monkeypatch):
    from reference import fused_ops
    monkeypatch.setenv('AI_RUN_DIR', str(tmp_path))
    prepared = tmp_path / 'prepared'
    plan = prepare_pair(prepared, 8, tiny=True)
    model = load_prepared_target(prepared, ops=ModuleFusedOps(fused_ops, allow_reference=True))
    assert model.feature_layers == (0, 2)
    assert model.prepared_source == plan['source']
    assert model(torch.tensor([[1, 2, 3]], dtype=torch.int32)).features.shape == (1, 3, 32)
    item = plan['weights']['target.embed_tokens.weight']
    path = prepared / item['path']
    data = bytearray(path.read_bytes()); data[0] ^= 1; path.write_bytes(data)
    with pytest.raises(ValueError, match='hash|SHA|checksum'):
        load_prepared_target(prepared, ops=ModuleFusedOps(fused_ops, allow_reference=True))


@pytest.mark.parametrize('tokens', [torch.tensor([[1., 2.]]), torch.tensor([[64]]),
                                  torch.tensor([[-1]]), torch.empty((1, 0), dtype=torch.int32),
                                  torch.tensor([[1], [2]])])
def test_invalid_inputs_rejected(tokens):
    model, _ = make_model()
    with pytest.raises(ValueError):
        model(tokens)


def test_state_contract_and_capacity_are_checked():
    model, _ = make_model()
    tokens = torch.tensor([[1]], dtype=torch.int32)
    state = model.initial_state()
    bad = state.layers[0]
    wrong_dtype = LayerState(conv=bad.conv, recurrent=bad.recurrent.float())
    with pytest.raises(ValueError, match='recurrent state'):
        model(tokens, TargetState(0, (wrong_dtype,) + state.layers[1:]))
    with pytest.raises(ValueError, match='layer count'):
        model(tokens, TargetState(0, ()))
    model.config['max_position_embeddings'] = 1
    first = model(tokens)
    with pytest.raises(ValueError, match='capacity'):
        model(tokens, first.state)


def test_gdn_is_dispatched_without_external_l2_and_dtype_mismatch_fails():
    from reference import fused_ops
    module = ModuleType('observed_ops')
    for name in ModuleFusedOps.required_operations:
        setattr(module, name, getattr(fused_ops, name))
    module.FUSED_OPS_ABI = 1
    module.BACKEND_ID = 'test-recording'
    seen = []
    def core(q, k, v, g, beta, initial, effective, **attrs):
        seen.append((q.clone(), effective.clone(), attrs.copy()))
        result, state = fused_ops.chunk_gated_delta_rule(q, k, v, g, beta, initial, effective, **attrs)
        return result, state.float()
    module.chunk_gated_delta_rule = core
    model, _ = make_model(ops=ModuleFusedOps(module))
    with pytest.raises(ValueError, match='ChunkGatedDeltaRule state'):
        model(torch.tensor([[1, 2, 3]], dtype=torch.int32))
    assert seen[0][1].dtype == torch.int16 and seen[0][1].tolist() == [3]
    assert seen[0][2] == {'chunk_size': 64, 'use_qk_l2norm_in_kernel': True}
    assert torch.linalg.vector_norm(seen[0][0].float(), dim=-1).max() < .1


@pytest.mark.parametrize('reject_at', range(5))
def test_real_tiny_fork_rejection_and_continuation_equal_ordinary(reject_at):
    # Production-default FP16 state, same one-token target schedule in ordinary
    # and verification. Every tensor must be identical, not merely close.
    model, _ = make_model()
    ordinary, speculative, planner = [FusedTargetSession(model) for _ in range(3)]
    for session in (ordinary, speculative, planner):
        session.prefill([1, 2, 3])
    proposals = []
    for _ in range(4):
        proposals.append(planner.next_token)
        planner.step(planner.next_token)
    if reject_at < 4:
        proposals[reject_at] = (proposals[reject_at] + 1) % model.config['vocab_size']
    branch = speculative.verify(proposals)
    assert branch.max_accepted_prefix == reject_at
    assert speculative.state.position == 3
    branch.commit(reject_at)
    for token in proposals[:reject_at]:
        ordinary.step(token)
    correction_or_bonus = ordinary.next_token
    assert speculative.next_token == correction_or_bonus
    for session in (ordinary, speculative):
        session.step(correction_or_bonus)
        session.step(session.next_token)
    assert ordinary.next_token == speculative.next_token
    torch.testing.assert_close(ordinary.last_features, speculative.last_features, rtol=0, atol=0)
    a, b = ordinary.state, speculative.state
    assert a.position == b.position
    for la, lb in zip(a.layers, b.layers):
        for name in ('conv', 'recurrent', 'key', 'value'):
            va, vb = getattr(la, name), getattr(lb, name)
            if va is None:
                assert vb is None
            else:
                torch.testing.assert_close(va, vb, rtol=0, atol=0)


def test_backend_state_scratch_reuse_cannot_change_layers_or_previous_output():
    from reference import fused_ops
    module = ModuleType('scratch_reusing_ops')
    for name in ModuleFusedOps.required_operations:
        setattr(module, name, getattr(fused_ops, name))
    module.FUSED_OPS_ABI = 1
    module.BACKEND_ID = 'test-scratch-reuse'
    scratch = {}
    def retain(name, tensor):
        if name not in scratch or scratch[name].shape != tensor.shape:
            scratch[name] = tensor.clone()
        else:
            scratch[name].copy_(tensor)
        return scratch[name]
    def conv(*args):
        history, padded = fused_ops.conv_state_update(*args)
        return retain('conv', history), padded
    def core(*args, **kwargs):
        out, state = fused_ops.chunk_gated_delta_rule(*args, **kwargs)
        return out, retain('recurrent', state)
    def kv(*args):
        key, value = fused_ops.cache_update(*args)
        return retain('key', key), retain('value', value)
    module.conv_state_update = conv
    module.chunk_gated_delta_rule = core
    module.cache_update = kv
    candidate, _ = make_model(ops=ModuleFusedOps(module))
    expected, _ = make_model()
    ids = torch.tensor([[1, 2, 3]], dtype=torch.int32)
    result = candidate(ids)
    assert_state_close(result.state, expected(ids).state)
    frozen = result.state.clone()
    candidate(torch.tensor([[4, 5, 6]], dtype=torch.int32))
    assert_state_close(result.state, frozen)


def test_selective_none_preserves_explicit_reference_model_results():
    from reference import fused_ops
    baseline, _ = make_model()
    selected = SelectiveFusedOps(ModuleFusedOps(fused_ops, allow_reference=True),
                                fused_ops="none", allow_reference=True)
    model, _ = make_model(ops=selected)
    ids = torch.tensor([[1, 2, 3]], dtype=torch.int32)
    expected, actual = baseline(ids), model(ids)
    torch.testing.assert_close(actual.top1, expected.top1, rtol=0, atol=0)
    torch.testing.assert_close(actual.features, expected.features, rtol=0, atol=0)
    assert_state_close(actual.state, expected.state)


def test_selective_documented_gdn_rejects_tiny_geometry_before_weights(monkeypatch):
    from types import SimpleNamespace
    from reference import fused_ops
    monkeypatch.setattr('qwen35_moe_dflash.fused.ops.torch.ops', SimpleNamespace(
        npu=SimpleNamespace(npu_chunk_gated_delta_rule=lambda *args: None)))
    monkeypatch.setattr('qwen35_moe_dflash.fused.ops.importlib.import_module', lambda _: None)
    selected = SelectiveFusedOps(ModuleFusedOps(fused_ops, allow_reference=True),
        fused_ops='gdn', vendor_backend_id='test-only-vendor',
        gdn_initial_state_dtype=torch.float16, gdn_state_casts='allow', allow_reference=True)
    c, _ = configurations(tiny=True)
    with pytest.raises(ValueError, match='documented GDN'):
        FusedTargetModel(c, {}, feature_layers=(0, 2), ops=selected)
