"""Target verification/state orchestration with an in-place toy backend."""

from types import SimpleNamespace

import pytest
import torch

from qwen35_moe_dflash.fused.session import FusedTargetSession
from qwen35_moe_dflash.fused.session import LayerState, TargetState


class ToyTarget:
    device = torch.device("cpu")

    def __init__(self):
        self.fail_on = None
        self.bad_output = False

    def initial_state(self):
        return TargetState(0, (
            LayerState(conv=torch.zeros(1, 3), recurrent=torch.zeros(2, 2)),
            LayerState(key=torch.zeros(1, 0, 2), value=torch.zeros(1, 0, 2)),
        ))

    def __call__(self, input_ids, state=None):
        assert input_ids.dtype == torch.int32 and input_ids.shape[0] == 1
        state = self.initial_state() if state is None else state
        top1, features = [], []
        for token in input_ids[0].tolist():
            linear, attention = state.layers
            linear.conv.copy_(torch.roll(linear.conv, -1, dims=-1))
            linear.conv[:, -1] = token
            linear.recurrent.mul_(0.75).add_((linear.conv.sum() + token) * 0.125)
            if token == self.fail_on:
                # Simulate a vendor error after writing its supplied buffers.
                linear.recurrent.fill_(999)
                raise RuntimeError("toy operator failure")
            row = torch.tensor([[[token, state.position + 1.0]]])
            state = TargetState(state.position + 1, (
                LayerState(conv=linear.conv, recurrent=linear.recurrent),
                LayerState(key=torch.cat((attention.key, row), 1),
                           value=torch.cat((attention.value, row.square()), 1)),
            ))
            top1.append((token + state.position + int(linear.recurrent.sum())) % 31)
            features.append([linear.conv.sum().item(), linear.recurrent.sum().item(),
                             state.layers[1].key.sum().item(), state.layers[1].value.sum().item()])
        return SimpleNamespace(
            top1=torch.tensor([top1], dtype=torch.float32 if self.bad_output else torch.int32),
            features=torch.tensor([features], dtype=torch.float16), state=state)


def assert_state_equal(actual, expected):
    assert actual.position == expected.position
    assert len(actual.layers) == len(expected.layers)
    for got, wanted in zip(actual.layers, expected.layers):
        for name in ("conv", "recurrent", "key", "value"):
            left, right = getattr(got, name), getattr(wanted, name)
            if left is None or right is None:
                assert left is right
            else:
                assert torch.equal(left, right), name


def make_session(model=None):
    session = FusedTargetSession(model or ToyTarget())
    session.prefill([1, 2, 3])
    return session


def matching_proposals(session, count):
    state, token = session.state, session.next_token
    result = []
    for _ in range(count):
        result.append(token)
        output = session.model(torch.tensor([[token]], dtype=torch.int32), state=state)
        state, token = output.state, output.top1.item()
    return result


@pytest.mark.parametrize("reject_at", range(5))
def test_every_rejection_position_and_bonus_continue_match_ordinary(reject_at):
    ordinary, speculative = make_session(), make_session()
    proposals = matching_proposals(speculative, 4)
    if reject_at < len(proposals):
        proposals[reject_at] = (proposals[reject_at] + 1) % 31
    before_state, before_token = speculative.state, speculative.next_token
    before_features = speculative.last_features
    verification = speculative.verify(proposals)
    assert verification.max_accepted_prefix == reject_at
    assert len(verification.expected_next_tokens) == len(proposals) + 1
    assert verification.expected_next_tokens[0] == before_token
    assert_state_equal(speculative.state, before_state)
    assert speculative.next_token == before_token
    assert torch.equal(speculative.last_features, before_features)
    for token in proposals[:reject_at]:
        ordinary.step(token)
    committed = verification.commit(reject_at)
    assert_state_equal(committed, ordinary.state)
    assert_state_equal(speculative.state, ordinary.state)
    assert speculative.next_token == ordinary.next_token
    assert torch.equal(speculative.last_features, ordinary.last_features)
    # Rejected proposal is never consumed; all-accepted instead consumes bonus.
    correction = verification.expected_next_tokens[reject_at]
    assert correction == ordinary.next_token
    speculative.step(correction)
    ordinary.step(correction)
    assert_state_equal(speculative.state, ordinary.state)
    assert speculative.next_token == ordinary.next_token
    assert torch.equal(speculative.last_features, ordinary.last_features)


def test_accepting_less_than_matching_prefix_is_safe():
    session, ordinary = make_session(), make_session()
    proposals = matching_proposals(session, 4)
    verified = session.verify(proposals)
    assert verified.max_accepted_prefix == 4
    verified.commit(2)
    for token in proposals[:2]:
        ordinary.step(token)
    assert_state_equal(session.state, ordinary.state)
    assert session.next_token == verified.expected_next_tokens[2]


def test_cannot_commit_a_mismatching_proposal_or_invalid_count():
    session = make_session()
    verification = session.verify([(session.next_token + 1) % 31])
    before = session.state
    for count in (1, -1, True, 0.0, None):
        with pytest.raises(ValueError, match="matching proposal prefix"):
            verification.commit(count)
        assert_state_equal(session.state, before)
    verification.commit(0)
    assert_state_equal(session.state, before)


def test_stale_verification_cannot_replace_new_state_or_metadata():
    session = make_session()
    proposal = [session.next_token]
    stale = session.verify(proposal)
    winner = session.verify(proposal)
    winner.commit(1)
    current, token, features = session.state, session.next_token, session.last_features
    with pytest.raises(RuntimeError, match="stale"):
        stale.commit(1)
    stale.abort()
    assert_state_equal(session.state, current)
    assert session.next_token == token
    assert torch.equal(session.last_features, features)


def test_step_invalidates_pending_verification():
    session = make_session()
    verification = session.verify([session.next_token])
    session.step(session.next_token)
    with pytest.raises(RuntimeError, match="stale"):
        verification.commit(0)
    verification.abort()


@pytest.mark.parametrize("operation", ["step", "verify"])
def test_failed_model_call_leaves_session_unchanged(operation):
    model = ToyTarget()
    session = make_session(model)
    before, token, features = session.state, session.next_token, session.last_features
    model.fail_on = 30
    with pytest.raises(RuntimeError, match="toy operator failure"):
        session.step(30) if operation == "step" else session.verify([token, 30])
    assert_state_equal(session.state, before)
    assert session.next_token == token
    assert torch.equal(session.last_features, features)
    model.fail_on = None
    session.step(token)
    assert session.state.position == before.position + 1


def test_invalid_output_and_failed_prefill_do_not_publish():
    model = ToyTarget()
    session = FusedTargetSession(model)
    model.fail_on = 2
    with pytest.raises(RuntimeError, match="toy operator failure"):
        session.prefill([1, 2])
    assert_state_equal(session.state, model.initial_state())
    model.fail_on = None
    model.bad_output = True
    with pytest.raises(ValueError, match="top1"):
        session.prefill([1, 2])
    assert_state_equal(session.state, model.initial_state())
    model.bad_output = False
    session.prefill([1, 2])
    previous = session.state
    model.bad_output = True
    with pytest.raises(ValueError, match="top1"):
        session.step(4)
    assert_state_equal(session.state, previous)


def test_returned_tensors_and_private_branch_cannot_mutate_session():
    session = FusedTargetSession(ToyTarget())
    prefill = session.prefill(torch.tensor([[1, 2]], dtype=torch.int64))
    state, token, features = session.state, session.next_token, session.last_features
    prefill.state.layers[0].conv.fill_(99)
    prefill.features.fill_(99)
    session.state.layers[0].recurrent.fill_(99)
    session.last_features.fill_(99)
    verification = session.verify([token, 12, 13])
    verification.abort()
    assert_state_equal(session.state, state)
    assert session.next_token == token
    assert torch.equal(session.last_features, features)


def test_prefill_once_and_no_decoding_before_prefill():
    session = FusedTargetSession(ToyTarget())
    for call in (lambda: session.verify([1]), lambda: session.step(1),
                 lambda: session.next_token, lambda: session.last_features):
        with pytest.raises(RuntimeError, match="prefill"):
            call()
    session.prefill([1])
    with pytest.raises(RuntimeError, match="once"):
        session.prefill([2])


@pytest.mark.parametrize("proposals", [[], [True], [-1], [2**31], [1.2], "12",
                                      torch.tensor([1, 2]), torch.tensor([[1.0]])])
def test_invalid_proposals_fail_without_changes(proposals):
    session = make_session()
    before = session.state
    with pytest.raises(ValueError):
        session.verify(proposals)
    assert_state_equal(session.state, before)


def test_verification_cannot_be_reused_after_commit_or_abort():
    session = make_session()
    first = session.verify([session.next_token])
    first.commit(1)
    with pytest.raises(RuntimeError, match="closed"):
        first.commit(0)
    second = session.verify([session.next_token])
    second.abort()
    with pytest.raises(RuntimeError, match="closed"):
        second.commit(0)
