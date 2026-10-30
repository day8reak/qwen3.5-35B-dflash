"""One decode-model target can prefill serially, then drive bounded S64 smoke.

This is a host protocol double, not a HIAI executor. Each simulated decode
retains an individual receipt and an80-component state digest. Prefill exposes
their aggregate receipt to the controller; adapters remain responsible for
validating and publishing all underlying native calls separately.
"""
import hashlib
import json

import numpy as np
import pytest

from qwen35_moe_dflash.incremental_smoke import (
    CallEvidence, ModelIdentity, SmokeContract, TargetObservation,
    feature_sha256, prefix_sha256, run_incremental_smoke,
)
from test_incremental_smoke import Draft


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class SerialDecodeTarget:
    """Simulated single identity with no distinct prefill-model identity."""
    def __init__(self, contract):
        self.contract = contract
        self.identity = ModelIdentity('target', contract.target_layers, contract.vocab_size,
            contract.feature_width, digest('single decode1 model'), digest('complete original target'), 'host-fake-hiai')
        self.prefix, self.native_calls = [], []
        self.features = np.zeros((1, 0, contract.feature_width), dtype='<f2')
        self.state_parts = [digest(('zero-state-component', index)) for index in range(80)]
        self.prefill_receipts = None

    @property
    def state(self):
        return digest(self.state_parts)

    def _decode(self, token):
        previous, position = self.state, len(self.prefix)
        self.prefix.append(token)
        self.state_parts = [digest((old, index, position, token)) for index, old in enumerate(self.state_parts)]
        chunk = np.zeros((1, 1, self.contract.feature_width), dtype='<f2')
        chunk[0, 0, 0], chunk[0, 0, 1] = token, position
        self.features = np.concatenate((self.features, chunk), axis=1)
        receipt = {'call_id': f'decode1-{position}', 'model_sha256': self.identity.model_sha256,
            'token_id': token, 'absolute_position': position, 'input_state_sha256': previous,
            'output_state_sha256': self.state, 'state_component_count': 80,
            'prefix_sha256': prefix_sha256(self.prefix), 'feature_output_sha256': feature_sha256(chunk),
            'evidence_kind': 'host-fake-hiai'}
        self.native_calls.append(receipt)
        evidence = CallEvidence(receipt['call_id'], self.identity.model_sha256, digest(receipt),
            receipt['prefix_sha256'], self.identity.evidence_kind,
            input_state_sha256=previous, output_state_sha256=self.state,
            feature_output_sha256=receipt['feature_output_sha256'])
        return TargetObservation((token + 1) % self.contract.vocab_size, len(self.prefix),
            prefix_sha256(self.prefix), self.features, self.state, evidence,
            feature_layer_ids=self.contract.feature_layer_ids)

    def prefill(self, prompt):
        assert self.prefix == [] and self.native_calls == []
        for token in prompt: self._decode(token)
        self.prefill_receipts = self.native_calls.copy()
        evidence = CallEvidence('serial-prefill-aggregate', self.identity.model_sha256,
            digest(self.prefill_receipts), prefix_sha256(self.prefix), self.identity.evidence_kind,
            output_state_sha256=self.state, feature_output_sha256=feature_sha256(self.features))
        return TargetObservation((self.prefix[-1] + 1) % self.contract.vocab_size, len(self.prefix),
            prefix_sha256(self.prefix), self.features, self.state, evidence,
            feature_layer_ids=self.contract.feature_layer_ids)

    def advance(self, token):
        return self._decode(token)


def execute(prompt, budget, *, mismatch=None, eos=()):
    contract = SmokeContract()
    target, draft = SerialDecodeTarget(contract), Draft(contract, mismatch_at=mismatch)
    report = run_incremental_smoke(target, draft, prompt, max_new_tokens=budget,
        max_proposals_per_round=2, target_next_anchor=True, eos_token_ids=eos)
    return report, target, draft


@pytest.mark.parametrize('budget', [8, 16])
@pytest.mark.parametrize('mismatch', [None, 0, 1])
def test_real_shape_short_prompt_loop_accept_reject_bonus_uses_one_target_identity_and_complete_history(budget, mismatch):
    prompt = list(range(1, 9))
    report, target, draft = execute(prompt, budget, mismatch=mismatch)
    expected = list(range(9, 9 + budget))
    assert report['status'] == 'EXECUTION_COMPLETE' and report['multi_round_observed']
    assert report['generated_token_ids'] == expected
    assert target.prefix == prompt + expected
    assert [row['token_id'] for row in target.native_calls] == target.prefix
    assert [row['absolute_position'] for row in target.native_calls] == list(range(8 + budget))
    assert len(target.native_calls) == 8 + budget
    assert all(row['model_sha256'] == target.identity.model_sha256 and row['state_component_count'] == 80 for row in target.native_calls)
    for previous, current in zip(target.native_calls, target.native_calls[1:]):
        assert current['input_state_sha256'] == previous['output_state_sha256']
    # These are protocol observations, with the first aggregate representing
    # eight individually retained decode calls, not one native invocation.
    assert report['counts']['target_calls'] == budget + 1
    target_evidence = [row for row in report['call_evidence'] if row['role'] == 'target']
    assert len(target_evidence) == budget + 1
    assert target_evidence[0]['receipt_sha256'] == digest(target.native_calls[:8])
    assert target_evidence[0]['call_id'] == 'serial-prefill-aggregate'
    assert [row['call_id'] for row in target_evidence[1:]] == [f'decode1-{position}' for position in range(8, 8 + budget)]
    assert report['initial_anchor']['token_id'] == 9 and report['initial_anchor']['position'] == 8
    assert report['initial_anchor']['target_call_id'] == 'decode1-8'
    assert [prefix for prefix, _ in draft.seen][0] == tuple(prompt + [9])
    for (prefix, features), event in zip(draft.seen, report['rounds']):
        assert len(prefix) <= 64 and len(event['proposals']) == 15
        assert features.shape == (1, len(prefix), 16384)
        np.testing.assert_array_equal(features[0, :, 0], prefix)
        np.testing.assert_array_equal(features[0, :, 1], np.arange(len(prefix)))
        assert event['anchor']['token_id'] == prefix[-1] and event['anchor']['position'] == len(prefix) - 1
        assert event['draft_context_contract'] == 'EXCLUDE_COMMITTED_ANCHOR_FEATURE_ROW'
        assert event['considered'] <= 2 and len(event['target_call_ids']) == len(event['committed'])
        if event['correction'] is not None:
            assert event['bonus'] is None
            assert event['correction']['token_id'] == event['committed'][-1]
            assert event['correction']['rejected_token_id'] != event['committed'][-1]
        elif event['bonus'] is not None:
            assert len(event['committed']) == event['accepted'] + 1
    assert report['final_state_sha256'] == target.native_calls[-1]['output_state_sha256']
    assert report['final_feature_prefix_sha256'] == feature_sha256(target.features)
    assert report['device_execution'] == 'UNVERIFIED' and report['numerical_validation'] == 'NOT_EVALUATED'


def test_complete_prefix_reaches_exact_s64_boundary_without_crop_or_rebase():
    prompt = list(range(1, 57))
    report, target, draft = execute(prompt, 8)
    assert target.prefix == list(range(1, 65))
    assert report['generated_token_ids'] == list(range(57, 65))
    assert target.features.shape == (1, 64, 16384)
    assert all(prefix == tuple(target.prefix[:len(prefix)]) for prefix, _ in draft.seen)
    assert all(len(prefix) <= 64 for prefix, _ in draft.seen)
    assert report['counts']['target_calls'] == 9 and len(target.native_calls) == 64


@pytest.mark.parametrize('prompt_length,budget', [(57, 8), (8, 57)])
def test_any_total_prefix_budget_over_s64_is_rejected_before_serial_prefill(prompt_length, budget):
    contract = SmokeContract(); target, draft = SerialDecodeTarget(contract), Draft(contract)
    with pytest.raises(ValueError, match='capacity'):
        run_incremental_smoke(target, draft, [1] * prompt_length, max_new_tokens=budget, target_next_anchor=True)
    assert target.native_calls == [] and draft.calls == 0


@pytest.mark.parametrize('eos,mismatch,kind', [(9, None, 'anchor'), (10, None, 'proposal'),
    (10, 0, 'correction'), (12, None, 'bonus')])
def test_serial_decode_prefill_stops_on_anchor_proposal_correction_or_bonus_with_complete_receipts(eos, mismatch, kind):
    report, target, draft = execute(list(range(1, 9)), 16, mismatch=mismatch, eos=[eos])
    assert report['generated_token_ids'] == list(range(9, eos + 1))
    assert target.prefix == list(range(1, eos + 1))
    assert len(target.native_calls) == eos and report['stop_reason'] == 'eos'
    assert report['counts']['target_calls'] == eos - 8 + 1
    if kind == 'anchor': assert draft.calls == 0 and report['rounds'] == []
    elif kind == 'correction': assert report['rounds'][-1]['correction']['token_id'] == eos
    elif kind == 'bonus': assert report['rounds'][-1]['bonus']['token_id'] == eos
    else: assert report['rounds'][-1]['bonus'] is None
