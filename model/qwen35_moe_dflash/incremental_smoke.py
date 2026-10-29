"""Serial DFlash execution smoke using real, prefix-aligned target features.

This controller knows no HDC paths or vendor APIs. Adapters must provide
hash-bound native observations; numerical reference acceptance stays separate.
Only committed target tokens enter the target state, so rejection never needs
to undo speculative state in this deliberately serial execution lane.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from numbers import Integral
import re
from typing import Protocol, Sequence

import numpy as np


_FEATURE_LAYERS = (1, 6, 11, 16, 22, 27, 32, 37)
_KINDS = {"device-hiai", "host-fake-hiai"}


def _sha(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(label + " must be a SHA256")
    return value


def _ids(values, vocab, label):
    try:
        result = list(values)
    except TypeError as error:
        raise ValueError(label + " must contain integer token IDs") from error
    if any(not isinstance(value, Integral) or isinstance(value, (bool, np.bool_))
           or not 0 <= int(value) < vocab for value in result):
        raise ValueError(label + " contains a token outside the vocabulary")
    return [int(value) for value in result]


def prefix_sha256(token_ids):
    """Hash the complete logical prefix as little-endian INT32 bytes."""
    values = _ids(token_ids, 1 << 31, "prefix")
    return hashlib.sha256(np.asarray(values, dtype="<i4").tobytes()).hexdigest()


def feature_sha256(features):
    return hashlib.sha256(features.tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class SmokeContract:
    target_layers: int = 40
    draft_layers: int = 6
    vocab_size: int = 248320
    feature_width: int = 16384
    draft_width: int = 15
    sequence_capacity: int = 64
    feature_layer_ids: tuple[int, ...] = _FEATURE_LAYERS
    test_only: bool = False

    def __post_init__(self):
        dimensions = (self.target_layers, self.draft_layers, self.vocab_size,
                      self.feature_width, self.draft_width, self.sequence_capacity)
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise ValueError("smoke contract dimensions must be positive integers")
        if (type(self.test_only) is not bool or not self.feature_layer_ids
                or len(set(self.feature_layer_ids)) != len(self.feature_layer_ids)
                or any(type(i) is not int or not 0 <= i < self.target_layers for i in self.feature_layer_ids)):
            raise ValueError("smoke contract has invalid target feature layers")
        if not self.test_only and (dimensions != (40, 6, 248320, 16384, 15, 64)
                                  or self.feature_layer_ids != _FEATURE_LAYERS):
            raise ValueError("production smoke requires the complete locked target and draft")

    @classmethod
    def for_host_test(cls, **changes):
        values = dict(target_layers=2, draft_layers=2, vocab_size=32, feature_width=8,
                      draft_width=3, sequence_capacity=16, feature_layer_ids=(0, 1))
        values.update(changes)
        values["test_only"] = True
        return cls(**values)


@dataclass(frozen=True)
class ModelIdentity:
    role: str
    layers: int
    vocab_size: int
    feature_width: int
    model_sha256: str
    completeness_receipt_sha256: str
    evidence_kind: str


@dataclass(frozen=True)
class CallEvidence:
    call_id: str
    model_sha256: str
    receipt_sha256: str
    prefix_sha256: str
    evidence_kind: str
    status: str = "EXECUTED"
    input_state_sha256: str | None = None
    output_state_sha256: str | None = None
    feature_input_sha256: str | None = None
    feature_output_sha256: str | None = None


@dataclass(frozen=True)
class TargetObservation:
    next_token_id: int
    prefix_length: int
    prefix_sha256: str
    features: np.ndarray
    state_sha256: str
    evidence: CallEvidence
    feature_source: str = "target-hidden-layer-concat"
    feature_layer_ids: tuple[int, ...] = _FEATURE_LAYERS


@dataclass(frozen=True)
class DraftObservation:
    token_ids: Sequence[int]
    prefix_length: int
    prefix_sha256: str
    feature_input_sha256: str
    evidence: CallEvidence


class TargetBackend(Protocol):
    identity: ModelIdentity

    def prefill(self, prompt: Sequence[int]) -> TargetObservation: ...
    def advance(self, committed_token: int) -> TargetObservation: ...


class DraftBackend(Protocol):
    identity: ModelIdentity

    def propose(self, prefix: Sequence[int], features: np.ndarray) -> DraftObservation: ...


def _identity(identity, role, contract):
    if (not isinstance(identity, ModelIdentity) or identity.role != role
            or any(type(value) is not int for value in (identity.layers, identity.vocab_size, identity.feature_width))
            or identity.layers != (contract.target_layers if role == "target" else contract.draft_layers)
            or identity.vocab_size != contract.vocab_size or identity.feature_width != contract.feature_width
            or identity.evidence_kind not in _KINDS):
        raise ValueError(role + " model identity differs from the complete model contract")
    _sha(identity.model_sha256, role + " model identity")
    _sha(identity.completeness_receipt_sha256, role + " completeness receipt")
    if contract.test_only and identity.evidence_kind != "host-fake-hiai":
        raise ValueError("host test contracts cannot claim device execution")


def run_incremental_smoke(target: TargetBackend, draft: DraftBackend, prompt, *,
                          contract=None, max_new_tokens=8, max_proposals_per_round=2,
                          minimum_draft_rounds=2, eos_token_ids=None):
    """Run bounded serial verification without ordinary/golden equality gates.

    Target prefill returns features for the whole prompt. Each advance returns
    the whole updated feature prefix with unchanged older rows; its evidence
    binds the newly appended feature row. Draft evidence binds all input rows.
    Every committed token, including the final one, is advanced into target
    state, so the final state and feature prefix include all generated tokens.

    A complete protocol result is not a device PASS: artifact retrieval and
    native receipt authenticity remain the integrating adapter's responsibility.
    """
    contract = contract or SmokeContract()
    if not isinstance(contract, SmokeContract):
        raise ValueError("explicit smoke contract required")
    _identity(target.identity, "target", contract)
    _identity(draft.identity, "draft", contract)
    prefix = _ids(prompt, contract.vocab_size, "prompt")
    original_prompt = prefix.copy()
    options = (max_new_tokens, max_proposals_per_round, minimum_draft_rounds)
    if (not prefix or any(type(value) is not int or value <= 0 for value in options)
            or max_proposals_per_round > contract.draft_width
            or len(prefix) + max_new_tokens > contract.sequence_capacity):
        raise ValueError("smoke token budget exceeds the bounded graph capacity")
    eos = set(_ids(eos_token_ids if eos_token_ids is not None else
                   (() if contract.test_only else (248044, 248046)), contract.vocab_size, "EOS"))
    seen, evidence = set(), []

    def record(call, identity, expected_prefix):
        if (not isinstance(call, CallEvidence) or call.status != "EXECUTED"
                or call.model_sha256 != identity.model_sha256
                or call.evidence_kind != identity.evidence_kind
                or call.prefix_sha256 != prefix_sha256(expected_prefix)
                or not isinstance(call.call_id, str) or not 1 <= len(call.call_id) <= 256
                or any(ord(c) < 32 for c in call.call_id)):
            raise ValueError("execution receipt does not bind its model and logical prefix")
        _sha(call.receipt_sha256, "native execution receipt")
        key = identity.role, call.call_id
        if key in seen:
            raise ValueError("one native call receipt was reused for multiple steps")
        seen.add(key)
        evidence.append({"role": identity.role, **asdict(call)})

    def observe(observation, previous=None):
        if not isinstance(observation, TargetObservation):
            raise ValueError("target must return a typed observation")
        if (type(observation.prefix_length) is not int or observation.prefix_length != len(prefix)
                or observation.prefix_sha256 != prefix_sha256(prefix)
                or observation.feature_source != "target-hidden-layer-concat"
                or observation.feature_layer_ids != contract.feature_layer_ids):
            raise ValueError("target features are missing real, aligned target-layer provenance")
        array = observation.features
        if (not isinstance(array, np.ndarray) or array.dtype != np.dtype("<f2")
                or array.shape != (1, len(prefix), contract.feature_width) or not array.flags.c_contiguous):
            raise ValueError("target must return the complete aligned FP16 feature prefix")
        _ids([observation.next_token_id], contract.vocab_size, "target next token")
        _sha(observation.state_sha256, "target state")
        record(observation.evidence, target.identity, prefix)
        if observation.evidence.output_state_sha256 != observation.state_sha256:
            raise ValueError("target state does not bind the native output receipt")
        if previous is not None:
            if (observation.evidence.input_state_sha256 != previous.state_sha256
                    or array[:, :-1].tobytes() != previous.features.tobytes()):
                raise ValueError("target advance changed committed history or used another state")
        chunk = array if previous is None else array[:, -1:]
        if observation.evidence.feature_output_sha256 != feature_sha256(chunk):
            raise ValueError("target feature bytes differ from their execution receipt")
        frozen = array.copy()
        frozen.setflags(write=False)
        return TargetObservation(int(observation.next_token_id), len(prefix), prefix_sha256(prefix), frozen,
                                 observation.state_sha256, observation.evidence,
                                 observation.feature_source, observation.feature_layer_ids)

    current = observe(target.prefill(tuple(prefix)))
    generated, rounds = [], []
    accepted = mismatched = discarded = 0
    target_calls = 1
    stopped = False
    while not stopped and len(generated) < max_new_tokens:
        feature_digest, base = feature_sha256(current.features), len(prefix)
        proposed = draft.propose(tuple(prefix), current.features)
        if feature_sha256(current.features) != feature_digest:
            raise ValueError("draft adapter modified the committed target features")
        if (not isinstance(proposed, DraftObservation) or type(proposed.prefix_length) is not int or proposed.prefix_length != base
                or proposed.prefix_sha256 != prefix_sha256(prefix)
                or proposed.feature_input_sha256 != feature_digest):
            raise ValueError("draft proposal does not bind its actual target feature prefix")
        record(proposed.evidence, draft.identity, prefix)
        if proposed.evidence.feature_input_sha256 != feature_digest:
            raise ValueError("draft native receipt has another feature input")
        tokens = _ids(proposed.token_ids, contract.vocab_size, "draft proposals")
        if len(tokens) != contract.draft_width:
            raise ValueError("draft must return its complete trained proposal block")
        event = {"round": len(rounds), "prefix_length": base, "prefix_sha256": prefix_sha256(prefix),
                 "feature_prefix_sha256": feature_digest, "draft_call_id": proposed.evidence.call_id,
                 "proposals": tokens, "committed": [], "accepted": 0, "mismatch_index": None,
                 "state_before_sha256": current.state_sha256, "target_call_ids": []}
        considered = 0
        for index, proposal in enumerate(tokens[:min(max_proposals_per_round, max_new_tokens - len(generated))]):
            expected = current.next_token_id
            matches = proposal == expected
            considered += 1
            if matches:
                accepted += 1
                event["accepted"] += 1
            else:
                mismatched += 1
                event["mismatch_index"] = index
            # Rejected proposal IDs are never passed to the stateful target.
            generated.append(expected)
            prefix.append(expected)
            event["committed"].append(expected)
            current = observe(target.advance(expected), current)
            target_calls += 1
            event["target_call_ids"].append(current.evidence.call_id)
            stopped = expected in eos
            if stopped or not matches:
                break
        discarded += len(tokens) - considered
        event.update(considered=considered, discarded_unverified=len(tokens) - considered,
                     state_after_sha256=current.state_sha256)
        rounds.append(event)
    multi_round = len(rounds) >= minimum_draft_rounds
    kinds = {row["evidence_kind"] for row in evidence}
    return {"schema_version": 1, "artifact_kind": "qwen35-incremental-dflash-execution-smoke",
            "status": "EXECUTION_COMPLETE" if multi_round else "INCOMPLETE_MILESTONE",
            "scope": "host-test" if contract.test_only else "complete-model-serial-verification-smoke",
            "device_execution": "UNVERIFIED", "evidence_kinds": sorted(kinds),
            "device_receipts_reported": sum(row["evidence_kind"] == "device-hiai" for row in evidence),
            "receipt_authenticity": "REQUIRES_INTEGRATING_ADAPTER_VALIDATION",
            "numerical_validation": "NOT_EVALUATED", "performance": "NOT_EVALUATED",
            "contract": asdict(contract), "models": {"target": asdict(target.identity), "draft": asdict(draft.identity)},
            "prompt_token_ids": original_prompt, "generated_token_ids": generated,
            "stop_reason": "eos" if stopped else "length", "final_prefix_sha256": prefix_sha256(prefix),
            "final_state_sha256": current.state_sha256, "final_feature_prefix_sha256": feature_sha256(current.features),
            "feature_nonfinite_count": int(np.count_nonzero(~np.isfinite(current.features))),
            "counts": {"target_calls": target_calls, "draft_calls": len(rounds),
                       "drafted_tokens": len(rounds) * contract.draft_width, "accepted_proposals": accepted,
                       "mismatched_proposals": mismatched, "discarded_unverified_proposals": discarded,
                       "committed_tokens": len(generated)},
            "minimum_draft_rounds": minimum_draft_rounds, "multi_round_observed": multi_round,
            "state_policy": "ADVANCE_ONLY_COMMITTED_TARGET_TOKENS_NO_SPECULATIVE_STATE",
            "rounds": rounds, "call_evidence": evidence}
