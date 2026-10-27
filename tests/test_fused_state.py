"""Speculative cache lifecycle tests; no vendor kernel or device is involved."""

import pytest
import torch

from qwen35_moe_dflash.fused.session import LayerState, StateStore, TargetState


def initial_state():
    return TargetState(0, (
        LayerState(conv=torch.zeros(1, 3), recurrent=torch.zeros(2, 2)),
        LayerState(key=torch.zeros(1, 0, 2), value=torch.zeros(1, 0, 2)),
    ))


def step(state, token):
    """A nonlinear history-dependent transition that mutates all cache buffers.

    The test intentionally behaves like an in-place vendor implementation. Its
    caller must supply an isolated state, including full-attention KV tensors.
    """
    gdn, attention = state.layers
    gdn.conv.copy_(torch.roll(gdn.conv, shifts=-1, dims=-1))
    gdn.conv[:, -1] = token
    gdn.recurrent.mul_(0.75).add_(gdn.conv.sum() + token * token)
    row = torch.tensor([[[token, state.position + 1.0]]])
    return TargetState(state.position + 1, (
        LayerState(conv=gdn.conv, recurrent=gdn.recurrent),
        LayerState(key=torch.cat((attention.key, row), 1),
                   value=torch.cat((attention.value, row.square()), 1)),
    ))


def assert_state_equal(actual, expected):
    assert actual.position == expected.position
    assert len(actual.layers) == len(expected.layers)
    for got, wanted in zip(actual.layers, expected.layers):
        for name in ("conv", "recurrent", "key", "value"):
            left, right = getattr(got, name), getattr(wanted, name)
            if left is None or right is None:
                assert left is right
            else:
                assert left.dtype == right.dtype
                assert left.device == right.device
                assert torch.equal(left, right), name


@pytest.mark.parametrize("accepted", range(5))
def test_reject_at_every_position_then_continue_matches_ordinary(accepted):
    prefix = initial_state()
    for token in (2, 3):
        prefix = step(prefix, token)
    store = StateStore(prefix)
    proposal = [7, 11, 13, 17]
    branch = store.begin()
    for token in proposal:
        branch.record(step(branch.state, token))
    # Speculation has not published even the accepted portion yet.
    assert_state_equal(store.read(), prefix)
    branch.commit(accepted)
    expected = prefix.clone()
    for token in proposal[:accepted]:
        expected = step(expected, token)
    assert_state_equal(store.read(), expected)
    # A correction (or bonus after all accepted) uses only committed history.
    expected = step(expected, 19)
    continuation = store.begin()
    continuation.record(step(continuation.state, 19))
    continuation.commit(1)
    assert_state_equal(store.read(), expected)


def test_all_cache_tensors_are_copied_at_every_ownership_boundary():
    original = step(initial_state(), 3)
    expected = original.clone()
    store = StateStore(original)
    for layer in original.layers:
        for name in ("conv", "recurrent", "key", "value"):
            tensor = getattr(layer, name)
            if tensor is not None:
                tensor.fill_(99)
    assert_state_equal(store.read(), expected)
    branch = store.begin()
    supplied = branch.state
    produced = step(supplied, 5)
    recorded = produced.clone()
    branch.record(produced)
    for layer in produced.layers:
        for name in ("conv", "recurrent", "key", "value"):
            tensor = getattr(layer, name)
            if tensor is not None:
                tensor.fill_(123)
    assert_state_equal(branch.state, recorded)
    assert_state_equal(store.read(), expected)
    result = branch.commit(1)
    result.layers[0].recurrent.fill_(-1)
    store.read().layers[1].key.fill_(-2)
    assert_state_equal(store.read(), recorded)


def test_abort_and_other_branch_cannot_pollute_committed_state():
    store = StateStore(initial_state())
    first, second = store.begin(), store.begin()
    first.record(step(first.state, 17))
    first.abort()
    assert_state_equal(store.read(), initial_state())
    second.record(step(second.state, 2))
    second.commit(1)
    assert_state_equal(store.read(), step(initial_state(), 2))


def test_stale_branch_rejected_even_when_winner_commits_zero():
    store = StateStore(initial_state())
    winner, stale = store.begin(), store.begin()
    stale.record(step(stale.state, 9))
    winner.commit(0)
    with pytest.raises(RuntimeError, match="stale"):
        stale.commit(1)
    with pytest.raises(RuntimeError, match="stale"):
        _ = stale.state
    with pytest.raises(RuntimeError, match="stale"):
        stale.record(step(initial_state(), 1))
    stale.abort()
    assert_state_equal(store.read(), initial_state())


@pytest.mark.parametrize("accepted", [-1, 2, True, False, 1.0, None, "1"])
def test_invalid_accept_count_does_not_publish(accepted):
    store = StateStore(initial_state())
    branch = store.begin()
    branch.record(step(branch.state, 8))
    with pytest.raises(ValueError, match="accepted_tokens"):
        branch.commit(accepted)
    assert not branch.closed
    assert_state_equal(store.read(), initial_state())
    branch.commit(1)
    assert_state_equal(store.read(), step(initial_state(), 8))


@pytest.mark.parametrize("close", ["abort", "commit"])
def test_closed_transaction_cannot_be_reused(close):
    branch = StateStore(initial_state()).begin()
    branch.abort() if close == "abort" else branch.commit(0)
    assert branch.closed
    for operation in (lambda: branch.state, lambda: branch.commit(0),
                      branch.abort, lambda: branch.record(step(initial_state(), 1))):
        with pytest.raises(RuntimeError, match="closed"):
            operation()


def test_reject_position_jumps_missing_layers_and_dropped_caches():
    store = StateStore(initial_state())
    branch = store.begin()
    next_state = step(branch.state, 3)
    invalid = (
        TargetState(2, next_state.layers),
        TargetState(0, next_state.layers),
        TargetState(1, next_state.layers[:1]),
        TargetState(1, (LayerState(), next_state.layers[1])),
    )
    for state in invalid:
        with pytest.raises(ValueError):
            branch.record(state)
    assert branch.recorded_tokens == 0
    assert_state_equal(store.read(), initial_state())
    branch.record(next_state)
    branch.commit(1)
    assert_state_equal(store.read(), next_state)


def test_empty_caches_can_be_initialized_on_first_transition():
    store = StateStore(TargetState(0, (LayerState(),)))
    branch = store.begin()
    first = TargetState(1, (LayerState(conv=torch.ones(1, 3),
                                      recurrent=torch.ones(2, 2)),))
    branch.record(first)
    branch.commit(1)
    assert_state_equal(store.read(), first)


def test_state_clones_preserve_types_and_detach_autograd():
    original = TargetState(2, (LayerState(
        conv=torch.ones(3, dtype=torch.float16, requires_grad=True),
        recurrent=torch.ones(2, 2, dtype=torch.float32, requires_grad=True)),))
    copied = original.clone()
    assert_state_equal(copied, original)
    assert copied.layers[0].conv.data_ptr() != original.layers[0].conv.data_ptr()
    assert copied.layers[0].recurrent.data_ptr() != original.layers[0].recurrent.data_ptr()
    assert not copied.layers[0].conv.requires_grad
    assert not copied.layers[0].recurrent.requires_grad


@pytest.mark.parametrize("position", [-1, True, 1.5, "1"])
def test_invalid_state_position(position):
    with pytest.raises(ValueError, match="position"):
        TargetState(position, ())


def test_invalid_state_records_fail_early():
    with pytest.raises(TypeError, match="Tensor"):
        LayerState(conv=1)
    with pytest.raises(ValueError, match="together"):
        LayerState(key=torch.zeros(1))
    with pytest.raises(TypeError, match="tuple"):
        TargetState(0, [])
    with pytest.raises(TypeError, match="LayerState"):
        TargetState(0, (None,))
    with pytest.raises(TypeError, match="TargetState"):
        StateStore(None)
    with pytest.raises(TypeError, match="TargetState"):
        StateStore(initial_state()).begin().record(None)
