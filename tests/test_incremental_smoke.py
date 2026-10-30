"""Serial smoke uses committed state and real-feature receipts, never accuracy PASS."""
from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

from qwen35_moe_dflash.incremental_smoke import (
    CallEvidence, DraftObservation, ModelIdentity, SmokeContract, TargetObservation,
    feature_sha256, prefix_sha256, run_incremental_smoke,
)


def _hash(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def _identity(role, contract):
    return ModelIdentity(role, contract.target_layers if role == "target" else contract.draft_layers,
                         contract.vocab_size, contract.feature_width,
                         _hash(role + " model"), _hash(role + " complete source graph"), "host-fake-hiai")


class Target:
    def __init__(self, contract, *, nonfinite=False):
        self.contract, self.identity = contract, _identity("target", contract)
        self.prefix, self.features, self.state, self.calls, self.advanced = [], None, None, 0, []
        self.nonfinite = nonfinite

    def _rows(self, tokens):
        rows = np.repeat(np.asarray(tokens, dtype="<f2")[None, :, None], self.contract.feature_width, axis=2)
        if self.nonfinite:
            rows[:, :, 0] = np.nan
        return rows

    def _observation(self, chunk, previous_state):
        self.state = _hash(("state", self.prefix))
        evidence = CallEvidence("target-" + str(self.calls), self.identity.model_sha256, _hash(("target receipt", self.calls)),
                                prefix_sha256(self.prefix), self.identity.evidence_kind,
                                input_state_sha256=previous_state, output_state_sha256=self.state,
                                feature_output_sha256=feature_sha256(chunk))
        self.calls += 1
        return TargetObservation((self.prefix[-1] + 1) % self.contract.vocab_size, len(self.prefix),
                                 prefix_sha256(self.prefix), self.features, self.state, evidence,
                                 feature_layer_ids=self.contract.feature_layer_ids)

    def prefill(self, prompt):
        self.prefix = list(prompt)
        self.features = self._rows(self.prefix)
        return self._observation(self.features, None)

    def advance(self, token):
        self.advanced.append(token)
        previous_state = self.state
        self.prefix.append(token)
        chunk = self._rows([token])
        self.features = np.concatenate((self.features, chunk), axis=1)
        return self._observation(chunk, previous_state)


class Draft:
    def __init__(self, contract, *, mismatch_at=None):
        self.contract, self.identity = contract, _identity("draft", contract)
        self.calls, self.seen = 0, []
        self.mismatch_at = mismatch_at

    def propose(self, prefix, features):
        assert not features.flags.writeable
        self.seen.append((tuple(prefix), features.copy()))
        tokens = [(prefix[-1] + 1 + i) % self.contract.vocab_size for i in range(self.contract.draft_width)]
        if self.mismatch_at is not None:
            tokens[self.mismatch_at] = (tokens[self.mismatch_at] + 7) % self.contract.vocab_size
        digest = feature_sha256(features)
        evidence = CallEvidence("draft-" + str(self.calls), self.identity.model_sha256, _hash(("draft receipt", self.calls)),
                                prefix_sha256(prefix), self.identity.evidence_kind, feature_input_sha256=digest)
        self.calls += 1
        return DraftObservation(tokens, len(prefix), prefix_sha256(prefix), digest, evidence)


@pytest.fixture
def backends():
    contract = SmokeContract.for_host_test()
    return contract, Target(contract), Draft(contract)


def test_all_accepts_complete_multiple_rounds_and_commit_every_token(backends):
    contract, target, draft = backends
    result = run_incremental_smoke(target, draft, [2, 3], contract=contract, max_new_tokens=5)
    assert result["status"] == "EXECUTION_COMPLETE" and result["multi_round_observed"]
    assert result["generated_token_ids"] == target.advanced == [4, 5, 6, 7, 8]
    assert target.prefix == [2, 3, 4, 5, 6, 7, 8]
    assert result["counts"] == {"target_calls": 6, "draft_calls": 3, "drafted_tokens": 9,
                                "accepted_proposals": 5, "mismatched_proposals": 0,
                                "discarded_unverified_proposals": 4, "committed_tokens": 5}
    assert result["final_state_sha256"] == target.state
    assert result["final_prefix_sha256"] == prefix_sha256(target.prefix)
    assert result["numerical_validation"] == result["performance"] == "NOT_EVALUATED"
    assert result["device_execution"] == "UNVERIFIED" and result["device_receipts_reported"] == 0
    assert result["evidence_kinds"] == ["host-fake-hiai"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("mismatch_at,accepted,rounds", [(0, 0, 4), (1, 2, 2)])
def test_rejection_never_advances_unaccepted_token(backends, mismatch_at, accepted, rounds):
    contract, target, _ = backends
    draft = Draft(contract, mismatch_at=mismatch_at)
    result = run_incremental_smoke(target, draft, [2, 3], contract=contract, max_new_tokens=4)
    assert result["generated_token_ids"] == target.advanced == [4, 5, 6, 7]
    assert result["counts"]["accepted_proposals"] == accepted
    assert result["counts"]["draft_calls"] == rounds
    assert result["counts"]["target_calls"] == 5
    for event in result["rounds"]:
        wrong = event["proposals"][event["mismatch_index"]]
        assert event["committed"][-1] != wrong
    assert result["state_policy"] == "ADVANCE_ONLY_COMMITTED_TARGET_TOKENS_NO_SPECULATIVE_STATE"


def test_eos_before_two_rounds_is_incomplete_milestone_not_device_pass(backends):
    contract, target, draft = backends
    result = run_incremental_smoke(target, draft, [2, 3], contract=contract, max_new_tokens=4, eos_token_ids=[4])
    assert result["generated_token_ids"] == [4] and result["stop_reason"] == "eos"
    assert result["status"] == "INCOMPLETE_MILESTONE" and not result["multi_round_observed"]
    assert target.advanced == [4] and result["counts"]["target_calls"] == 2


def test_generator_prompt_is_reported_and_features_are_bound_to_each_new_prefix(backends):
    contract, target, draft = backends
    result = run_incremental_smoke(target, draft, (i for i in (2, 3)), contract=contract, max_new_tokens=3)
    assert result["prompt_token_ids"] == [2, 3]
    assert [prefix for prefix, _ in draft.seen] == [(2, 3), (2, 3, 4, 5)]
    for prefix, features in draft.seen:
        assert features.shape == (1, len(prefix), contract.feature_width)
        np.testing.assert_array_equal(features[0, :, 0], prefix)


@pytest.mark.parametrize("fault", ["missing", "dtype", "shape", "prefix", "source", "layers", "feature_hash", "state_hash"])
def test_bad_target_feature_provenance_rejects_before_draft(backends, fault):
    contract, target, draft = backends
    prefill = target.prefill

    def bad(prompt):
        obs = prefill(prompt)
        changes = {"missing": {"features": None}, "dtype": {"features": obs.features.astype("float32")},
                   "shape": {"features": obs.features[:, :, :-1].copy()}, "prefix": {"prefix_sha256": _hash("another prefix")},
                   "source": {"feature_source": "synthetic-zero"}, "layers": {"feature_layer_ids": (1, 0)},
                   "feature_hash": {"evidence": replace(obs.evidence, feature_output_sha256=_hash("other feature bytes"))},
                   "state_hash": {"state_sha256": _hash("other output state")}}[fault]
        return replace(obs, **changes)

    target.prefill = bad
    with pytest.raises(ValueError):
        run_incremental_smoke(target, draft, [2, 3], contract=contract, max_new_tokens=3)
    assert draft.calls == 0 and target.advanced == []


@pytest.mark.parametrize("fault", ["old_features", "input_state", "reused_call"])
def test_advance_must_preserve_committed_history_and_state_chain(backends, fault):
    contract, target, draft = backends
    advance = target.advance

    def bad(token):
        obs = advance(token)
        if fault == "old_features":
            obs.features[0, 0, 1] += 1
            return obs
        return replace(obs, evidence=replace(obs.evidence, **(
            {"input_state_sha256": _hash("different state")} if fault == "input_state" else {"call_id": "target-0"})))

    target.advance = bad
    with pytest.raises(ValueError):
        run_incremental_smoke(target, draft, [2, 3], contract=contract, max_new_tokens=3)


@pytest.mark.parametrize("fault", ["count", "token", "prefix", "features", "receipt"])
def test_bad_draft_binding_rejects_before_state_advance(backends, fault):
    contract, target, draft = backends
    propose = draft.propose

    def bad(prefix, features):
        obs = propose(prefix, features)
        changes = {"count": {"token_ids": [4]}, "token": {"token_ids": [4, contract.vocab_size, 6]},
                   "prefix": {"prefix_length": 1}, "features": {"feature_input_sha256": _hash("wrong input")},
                   "receipt": {"evidence": replace(obs.evidence, model_sha256=_hash("other model"))}}[fault]
        return replace(obs, **changes)

    draft.propose = bad
    with pytest.raises(ValueError):
        run_incremental_smoke(target, draft, [2, 3], contract=contract, max_new_tokens=3)
    assert target.advanced == []


def test_mutating_draft_adapter_cannot_change_committed_features(backends):
    contract, target, draft = backends
    propose = draft.propose

    def bad(prefix, features):
        obs = propose(prefix, features)
        features.setflags(write=True)
        features[0, 0, 0] += 1
        return obs

    draft.propose = bad
    with pytest.raises(ValueError, match="modified"):
        run_incremental_smoke(target, draft, [2, 3], contract=contract, max_new_tokens=3)
    assert target.advanced == []


def test_nonfinite_raw_target_features_are_preserved_as_accuracy_deferred_diagnostics():
    contract = SmokeContract.for_host_test()
    target, draft = Target(contract, nonfinite=True), Draft(contract)
    result = run_incremental_smoke(target, draft, [2, 3], contract=contract, max_new_tokens=3)
    assert result["feature_nonfinite_count"] == 5
    assert all(np.isnan(features[0, :, 0]).all() for _, features in draft.seen)
    assert result["numerical_validation"] == "NOT_EVALUATED"
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("field,value", [("target_layers", 2), ("draft_layers", 5), ("vocab_size", 32),
                                         ("feature_width", 8), ("draft_width", 1), ("sequence_capacity", 128)])
def test_production_contract_cannot_silently_shrink_models(field, value):
    with pytest.raises(ValueError):
        SmokeContract(**{field: value})


def test_host_test_contract_cannot_claim_device_evidence(backends):
    contract, target, draft = backends
    target.identity = replace(target.identity, evidence_kind="device-hiai")
    with pytest.raises(ValueError, match="cannot claim device"):
        run_incremental_smoke(target, draft, [2, 3], contract=contract, max_new_tokens=3)


def test_default_full_contract_is_required_unless_host_test_is_explicit(backends):
    _, target, draft = backends
    with pytest.raises(ValueError, match="complete model"):
        run_incremental_smoke(target, draft, [2, 3], max_new_tokens=3)


def test_full_contract_host_backends_still_never_produce_device_pass():
    contract = SmokeContract()
    target, draft = Target(contract), Draft(contract)
    result = run_incremental_smoke(target, draft, [2, 3], max_new_tokens=2, max_proposals_per_round=1)
    assert result["status"] == "EXECUTION_COMPLETE"
    assert result["contract"]["target_layers"] == 40 and result["contract"]["draft_layers"] == 6
    assert result["counts"]["drafted_tokens"] == 30
    assert result["device_execution"] == "UNVERIFIED" and result["numerical_validation"] == "NOT_EVALUATED"


@pytest.mark.parametrize("kwargs", [{"max_new_tokens": 0}, {"max_new_tokens": 15},
                                    {"max_proposals_per_round": 4}, {"max_proposals_per_round": True},
                                    {"minimum_draft_rounds": 0}, {"eos_token_ids": [32]}])
def test_invalid_capacity_and_token_options_fail_before_execution(backends, kwargs):
    contract, target, draft = backends
    with pytest.raises(ValueError):
        run_incremental_smoke(target, draft, [2, 3], contract=contract, **kwargs)
    assert target.calls == draft.calls == 0


def _ordinary_oracle(prompt, budget, vocab, eos=()):
    # Independent ordinary sequence for this fixture's last-token+1 target;
    # no draft proposals, controller result or target observations are reused.
    values = [(prompt[-1] + index + 1) % vocab for index in range(budget)]
    end = next((index + 1 for index, token in enumerate(values) if token in eos), len(values))
    return values[:end]


def _anchored(backends, **options):
    contract, target, draft = backends
    return run_incremental_smoke(target, draft, [2, 3], contract=contract,
        target_next_anchor=True, **options)


@pytest.mark.parametrize("mismatch_at", [None, 0, 1, 2])
def test_target_next_anchor_serial_verify_matches_independent_ordinary_oracle(backends, mismatch_at):
    contract, target, _ = backends
    draft = Draft(contract, mismatch_at=mismatch_at)
    result = _anchored((contract, target, draft), max_new_tokens=9, max_proposals_per_round=3)
    expected = _ordinary_oracle([2, 3], 9, contract.vocab_size)
    assert result["generated_token_ids"] == target.advanced == expected
    assert target.prefix == [2, 3, *expected]
    assert result["anchor_mode"] == "TARGET_NEXT_ANCHOR"
    assert result["initial_anchor"]["token_id"] == 4 and result["initial_anchor"]["position"] == 2
    assert result["initial_anchor"]["target_call_id"] == "target-1"
    assert result["counts"]["target_calls"] == len(expected) + 1
    assert result["counts"]["initial_anchor_tokens"] == 1
    assert result["counts"]["committed_tokens"] == (
        1 + result["counts"]["accepted_proposals"] + result["counts"]["mismatched_proposals"]
        + result["counts"]["bonus_tokens"])
    assert result["multi_round_observed"] and result["status"] == "EXECUTION_COMPLETE"
    assert result["final_state_sha256"] == target.state
    assert result["final_prefix_sha256"] == prefix_sha256(target.prefix)
    assert result["final_feature_prefix_sha256"] == feature_sha256(target.features)
    assert result["device_execution"] == "UNVERIFIED" and result["numerical_validation"] == "NOT_EVALUATED"
    assert [prefix for prefix, _ in draft.seen][0] == (2, 3, 4)
    for (prefix, features), event in zip(draft.seen, result["rounds"]):
        np.testing.assert_array_equal(features[0, :, 0], prefix)
        assert features.shape == (1, len(prefix), contract.feature_width)
        assert event["anchor"]["token_id"] == prefix[-1]
        assert event["anchor"]["position"] == len(prefix) - 1
        assert event["draft_context_contract"] == "EXCLUDE_COMMITTED_ANCHOR_FEATURE_ROW"
        assert len(event["target_call_ids"]) == len(event["committed"])
        start = len(prefix) - 2
        assert event["committed"] == expected[start:start + len(event["committed"])]
        if event["correction"] is not None:
            correction = event["correction"]
            assert correction["rejected_token_id"] != correction["token_id"]
            assert correction["token_id"] == event["committed"][-1]
            assert correction["rejected_token_id"] not in event["committed"]
            assert event["accepted"] == correction["proposal_index"]
            assert event["bonus"] is None
        elif event["bonus"] is not None:
            assert event["bonus"]["token_id"] == event["committed"][-1]
            assert len(event["committed"]) == event["accepted"] + 1
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("limit", [1, 2, 3])
def test_target_next_anchor_effective_proposal_limit_all_accepts_and_bonus(backends, limit):
    result = _anchored(backends, max_new_tokens=8, max_proposals_per_round=limit)
    assert result["generated_token_ids"] == _ordinary_oracle([2, 3], 8, backends[0].vocab_size)
    for event in result["rounds"]:
        assert event["considered"] <= event["proposal_limit"] <= limit
        assert event["accepted"] == event["considered"]
        assert event["correction"] is None
        assert event["discarded_unverified"] == backends[0].draft_width - event["considered"]
    assert result["counts"]["bonus_tokens"] == sum(event["bonus"] is not None for event in result["rounds"])
    assert result["counts"]["bonus_tokens"] > 0


@pytest.mark.parametrize("budget", [1, 2, 3, 4])
def test_target_next_anchor_checks_budget_at_initial_anchor_proposals_and_bonus(backends, budget):
    result = _anchored(backends, max_new_tokens=budget, max_proposals_per_round=1)
    assert result["generated_token_ids"] == _ordinary_oracle([2, 3], budget, backends[0].vocab_size)
    assert len(backends[1].advanced) == budget and result["stop_reason"] == "length"
    assert result["counts"]["target_calls"] == budget + 1
    assert sum(len(event["committed"]) for event in result["rounds"]) == budget - 1
    if budget == 1:
        assert backends[2].calls == 0 and result["rounds"] == []


@pytest.mark.parametrize("kind,eos,mismatch", [("anchor", 4, None), ("proposal", 5, None),
    ("correction", 5, 0), ("bonus", 6, None)])
def test_target_next_anchor_checks_eos_on_every_commit(backends, kind, eos, mismatch):
    contract, target, _ = backends
    draft = Draft(contract, mismatch_at=mismatch)
    result = _anchored((contract, target, draft), max_new_tokens=8, max_proposals_per_round=1, eos_token_ids=[eos])
    assert result["generated_token_ids"] == target.advanced == _ordinary_oracle([2, 3], 8, contract.vocab_size, [eos])
    assert result["stop_reason"] == "eos" and target.prefix[-1] == eos
    assert result["counts"]["target_calls"] == len(target.advanced) + 1
    if kind == "anchor":
        assert not result["rounds"] and draft.calls == 0
    elif kind == "bonus":
        assert result["rounds"][-1]["bonus"]["token_id"] == eos
    elif kind == "correction":
        assert result["rounds"][-1]["correction"]["token_id"] == eos
        assert target.advanced == [4, 5]
    else:
        assert result["rounds"][-1]["bonus"] is None


def test_target_next_anchor_vocabulary_wrap_and_mismatch_remain_ordinary(backends):
    contract, target, _ = backends
    draft = Draft(contract, mismatch_at=1)
    result = run_incremental_smoke(target, draft, [29, 30], contract=contract,
        max_new_tokens=7, max_proposals_per_round=3, target_next_anchor=True)
    assert result["generated_token_ids"] == target.advanced == _ordinary_oracle([29, 30], 7, contract.vocab_size)
    assert result["generated_token_ids"][:3] == [31, 0, 1]


@pytest.mark.parametrize("fault", ["input_state", "old_features", "reused_call"])
def test_target_next_anchor_warmup_retains_state_history_and_receipt_validation(backends, fault):
    contract, target, draft = backends
    original = target.advance
    def bad(token):
        observation = original(token)
        if fault == "old_features":
            observation.features[0, 0, 0] += 1
            return observation
        return replace(observation, evidence=replace(observation.evidence, **(
            {"input_state_sha256": _hash("wrong parent")} if fault == "input_state" else {"call_id": "target-0"})))
    target.advance = bad
    with pytest.raises(ValueError):
        _anchored(backends, max_new_tokens=4)
    assert target.advanced == [4] and draft.calls == 0


@pytest.mark.parametrize("fault", ["prefix", "features", "receipt", "count"])
def test_target_next_anchor_bad_draft_cannot_advance_any_proposal_after_anchor(backends, fault):
    contract, target, draft = backends
    original = draft.propose
    def bad(prefix, features):
        observation = original(prefix, features)
        changes = {"prefix": {"prefix_sha256": _hash("another prefix")},
            "features": {"feature_input_sha256": _hash("another feature")}, "count": {"token_ids": [5]},
            "receipt": {"evidence": replace(observation.evidence, receipt_sha256="not-a-sha")}}[fault]
        return replace(observation, **changes)
    draft.propose = bad
    with pytest.raises(ValueError):
        _anchored(backends, max_new_tokens=4)
    assert target.advanced == [4]


@pytest.mark.parametrize("value", [1, "true", None])
def test_target_next_anchor_mode_requires_explicit_boolean_before_execution(backends, value):
    contract, target, draft = backends
    with pytest.raises(ValueError, match="explicit boolean"):
        run_incremental_smoke(target, draft, [2, 3], contract=contract, target_next_anchor=value)
    assert target.calls == draft.calls == 0


def test_target_next_anchor_does_not_relax_full_model_or_s64_capacity():
    contract = SmokeContract()
    target, draft = Target(contract), Draft(contract)
    with pytest.raises(ValueError, match="capacity"):
        run_incremental_smoke(target, draft, [3] * 64, max_new_tokens=1, target_next_anchor=True)
    assert target.calls == draft.calls == 0
    with pytest.raises(ValueError):
        SmokeContract(sequence_capacity=128)
