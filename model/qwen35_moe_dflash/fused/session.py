"""Cache ownership and host orchestration for the separate fused target.

These logical PyTorch caches are not a vendor GE or HIAI wire ABI. Readers,
speculative branches and recorded checkpoints receive independent copies; the
store never lends its committed tensors to an operator. Native asynchronous
writes must finish before recording state: clone does not synchronize DMA.

Verification records one checkpoint per proposed token. It does not run a draft
model, infer acceptances from a final state, or implement a device state bank.
The ordinary and speculative paths call the same supplied target model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from threading import RLock
from typing import Any

import torch
from torch import Tensor

_CACHE_FIELDS = ("conv", "recurrent", "key", "value")


def _copy_tensor(value: Tensor | None) -> Tensor | None:
    # Cache ownership is inference-only; do not retain an autograd graph across
    # tokens or checkpoint branches. clone preserves dtype/device and values.
    return None if value is None else value.detach().clone()


@dataclass(frozen=True)
class LayerState:
    """One layer's full state, including both sides of an attention KV cache."""

    conv: Tensor | None = None
    recurrent: Tensor | None = None
    key: Tensor | None = None
    value: Tensor | None = None

    def __post_init__(self) -> None:
        for name in _CACHE_FIELDS:
            value = getattr(self, name)
            if value is not None and not isinstance(value, Tensor):
                raise TypeError(f"{name} cache must be a Tensor or None")
        if (self.key is None) != (self.value is None):
            raise ValueError("key and value caches must be supplied together")

    def clone(self) -> "LayerState":
        return LayerState(**{name: _copy_tensor(getattr(self, name))
                             for name in _CACHE_FIELDS})


@dataclass(frozen=True)
class TargetState:
    """Caches after consuming exactly ``position`` tokens of one prefix.

    Tensor shapes and layouts are validated by the model/backend contract, not
    inferred here from a device descriptor or a model's layer number.
    """

    position: int
    layers: tuple[LayerState, ...]

    def __post_init__(self) -> None:
        if type(self.position) is not int or self.position < 0:
            raise ValueError("state position must be a nonnegative integer")
        if not isinstance(self.layers, tuple):
            raise TypeError("state layers must be a tuple")
        if not all(isinstance(layer, LayerState) for layer in self.layers):
            raise TypeError("state layers must contain LayerState records")

    def clone(self) -> "TargetState":
        return TargetState(self.position, tuple(layer.clone() for layer in self.layers))


class StateStore:
    """Own the committed state and atomically replace the entire cache tuple."""

    def __init__(self, initial: TargetState) -> None:
        if not isinstance(initial, TargetState):
            raise TypeError("initial state must be a TargetState")
        self._state = initial.clone()
        self._generation = 0
        self._lock = RLock()

    def read(self) -> TargetState:
        with self._lock:
            return self._state.clone()

    def begin(self) -> "StateTransaction":
        with self._lock:
            return StateTransaction(self, self._generation, self._state.clone())

    def _check_generation(self, generation: int) -> None:
        with self._lock:
            if generation != self._generation:
                raise RuntimeError("stale state transaction: another branch has committed")

    def _commit(self, generation: int, candidate: TargetState) -> TargetState:
        with self._lock:
            self._check_generation(generation)
            # Prepare both independent copies before publishing anything, so
            # allocation/clone failure cannot publish a partial state tuple.
            stored, result = candidate.clone(), candidate.clone()
            self._state = stored
            self._generation += 1
            return result


class StateTransaction:
    """A speculative branch with a checkpoint after each consumed token.

    ``commit(r)`` selects exactly the first ``r`` recorded token transitions.
    Rejected proposals and any unrecorded correction/bonus token are excluded.
    A correction must be executed from the committed state as a new transition.
    A multi-token result alone cannot recover intermediate recurrent states;
    ``record`` rejects jumps in position rather than pretending it can slice one.

    A transaction is used by one caller. Concurrent branches may share a store;
    once one commits, the others must restart from the new committed state.
    """

    def __init__(self, owner: StateStore, generation: int, base: TargetState) -> None:
        self._owner = owner
        self._generation = generation
        self._checkpoints = [base]
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def recorded_tokens(self) -> int:
        return max(0, len(self._checkpoints) - 1)

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("state transaction is closed")

    def _check_active(self) -> None:
        self._check_open()
        self._owner._check_generation(self._generation)

    @property
    def state(self) -> TargetState:
        self._check_active()
        return self._checkpoints[-1].clone()

    def record(self, candidate: TargetState) -> None:
        self._check_active()
        if not isinstance(candidate, TargetState):
            raise TypeError("candidate state must be a TargetState")
        previous = self._checkpoints[-1]
        if candidate.position != previous.position + 1:
            raise ValueError("record requires the state after exactly one more token")
        if len(candidate.layers) != len(previous.layers):
            raise ValueError("record cannot change the number of layers")
        for old, new in zip(previous.layers, candidate.layers):
            for name in _CACHE_FIELDS:
                if getattr(old, name) is not None and getattr(new, name) is None:
                    raise ValueError(f"record cannot drop an initialized {name} cache")
        self._checkpoints.append(candidate.clone())

    def commit(self, accepted_tokens: int) -> TargetState:
        self._check_active()
        if type(accepted_tokens) is not int:
            raise ValueError("accepted_tokens must be an integer, not bool or float")
        if not 0 <= accepted_tokens <= self.recorded_tokens:
            raise ValueError("accepted_tokens exceeds the recorded prefix")
        result = self._owner._commit(self._generation, self._checkpoints[accepted_tokens])
        self._closed = True
        self._checkpoints.clear()
        return result

    def abort(self) -> None:
        # A stale branch may always release its private checkpoints. It cannot
        # publish them, and abort never changes the owner's generation/state.
        self._check_open()
        self._closed = True
        self._checkpoints.clear()


def _token(value: int) -> int:
    if type(value) is not int or not 0 <= value <= 2**31 - 1:
        raise ValueError("token IDs must be nonnegative INT32 integers, not bool")
    return value


def _tokens(values: Sequence[int] | Tensor) -> list[int]:
    if isinstance(values, Tensor):
        if values.dtype not in (torch.int32, torch.int64):
            raise ValueError("token tensor must have an integer dtype")
        if values.ndim != 2 or values.shape[0] != 1:
            raise ValueError("token tensor must have shape [1, sequence]")
        result = [_token(value) for value in values.detach().cpu().tolist()[0]]
    else:
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise ValueError("tokens must be a sequence of integers or a [1,S] tensor")
        result = [_token(value) for value in values]
    if not result:
        raise ValueError("tokens must contain at least one token; padding is not inferred")
    return result


def _output_metadata(output: Any, ids: Tensor, previous: TargetState) -> tuple[int, Tensor]:
    """Validate the logical output before publishing state or session metadata."""
    sequence = ids.shape[1]
    if not isinstance(output.state, TargetState):
        raise TypeError("model output state must be a TargetState")
    if output.state.position != previous.position + sequence:
        raise ValueError("model output position does not match consumed tokens")
    if len(output.state.layers) != len(previous.layers):
        raise ValueError("model output cannot change the number of cache layers")
    if (not isinstance(output.top1, Tensor) or output.top1.dtype != torch.int32
            or tuple(output.top1.shape) != (1, sequence)):
        raise ValueError("model top1 must be INT32 [1, sequence]")
    features = output.features
    if (not isinstance(features, Tensor) or features.dtype != torch.float16
            or features.ndim != 3 or tuple(features.shape[:2]) != (1, sequence)
            or features.shape[2] <= 0):
        raise ValueError("model features must be FP16 [1, sequence, feature_width]")
    if output.top1.device != ids.device or features.device != ids.device:
        raise ValueError("model output tensors must be on the input device")
    if not torch.isfinite(features).all():
        raise ValueError("model features contain nonfinite values")
    if (output.top1 < 0).any():
        raise ValueError("model top1 contains a negative token ID")
    return int(output.top1[0, -1].item()), features[:, -1:, :].detach().clone()


class FusedTargetSession:
    """An ordinary target session with explicit speculative state transactions.

    ``prefill`` consumes an unpadded batch-one prefix exactly once. ``step``
    consumes one token. ``next_token`` predicts the next token after the current
    committed prefix; ``last_features`` is the last consumed token's feature row
    with shape [1,1,F]. Full prefill features remain in the returned model output.

    Native calls must complete their writes before returning. This source path
    uses cloned PyTorch caches; it is not a device state-bank performance design.
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        initial = model.initial_state()
        if not isinstance(initial, TargetState) or initial.position != 0:
            raise ValueError("a new session requires an empty target state at position zero")
        self._store = StateStore(initial)
        self._prefilled = False
        self._next_token: int | None = None
        self._last_features: Tensor | None = None
        self._lock = RLock()

    def _require_prefill(self) -> None:
        if not self._prefilled:
            raise RuntimeError("prefill must complete before decoding or verification")

    def _ids(self, values: Sequence[int]) -> Tensor:
        return torch.tensor([values], dtype=torch.int32, device=self.model.device)

    @property
    def state(self) -> TargetState:
        with self._lock:
            return self._store.read()

    @property
    def next_token(self) -> int:
        with self._lock:
            self._require_prefill()
            assert self._next_token is not None
            return self._next_token

    @property
    def last_features(self) -> Tensor:
        with self._lock:
            self._require_prefill()
            assert self._last_features is not None
            return self._last_features.clone()

    def prefill(self, ids: Sequence[int] | Tensor) -> Any:
        values = _tokens(ids)
        with self._lock:
            if self._prefilled or self._store.read().position != 0:
                raise RuntimeError("prefill is allowed only once on an empty session")
            inputs, previous = self._ids(values), self._store.read()
            output = self.model(inputs, state=previous)
            next_token, features = _output_metadata(output, inputs, previous)
            # Construct a complete independent store before publishing any of
            # the new state or metadata; model/validation/clone failure is safe.
            stored = StateStore(output.state)
            self._store = stored
            self._next_token, self._last_features = next_token, features
            self._prefilled = True
            return output

    def step(self, token: int) -> Any:
        value = _token(token)
        with self._lock:
            self._require_prefill()
            branch = self._store.begin()
            try:
                inputs, previous = self._ids([value]), branch.state
                output = self.model(inputs, state=previous)
                next_token, features = _output_metadata(output, inputs, previous)
                branch.record(output.state)
                branch.commit(1)
            except BaseException:
                if not branch.closed:
                    branch.abort()
                raise
            self._next_token, self._last_features = next_token, features
            return output

    def verify(self, proposals: Sequence[int] | Tensor) -> "TargetVerification":
        values = _tokens(proposals)
        with self._lock:
            self._require_prefill()
            branch = self._store.begin()
            assert self._next_token is not None and self._last_features is not None
            snapshots = [(self._next_token, self._last_features.clone())]
        accepted = 0
        matching = True
        try:
            for token in values:
                expected = snapshots[-1][0]
                if matching and token == expected:
                    accepted += 1
                else:
                    matching = False
                inputs, previous = self._ids([token]), branch.state
                output = self.model(inputs, state=previous)
                metadata = _output_metadata(output, inputs, previous)
                branch.record(output.state)
                snapshots.append(metadata)
        except BaseException:
            if not branch.closed:
                branch.abort()
            raise
        return TargetVerification(self, branch, tuple(snapshots), accepted)


class TargetVerification:
    """Uncommitted verification of a proposal block against one target prefix.

    ``expected_next_tokens[i]`` is the target prediction *before* consuming
    proposal i; the final entry is the bonus-token prediction after the block.
    Only the prefix through ``max_accepted_prefix`` may be committed. Predictions
    after the first mismatch describe rejected private history and must not be
    used to continue the committed session.
    """

    def __init__(self, session: FusedTargetSession, branch: StateTransaction,
                 snapshots: tuple[tuple[int, Tensor], ...], accepted: int) -> None:
        self._session = session
        self._branch = branch
        self._snapshots = snapshots
        self._expected_next_tokens = tuple(value for value, _ in snapshots)
        self._max_accepted_prefix = accepted

    @property
    def expected_next_tokens(self) -> tuple[int, ...]:
        return self._expected_next_tokens

    @property
    def max_accepted_prefix(self) -> int:
        return self._max_accepted_prefix

    @property
    def closed(self) -> bool:
        return self._branch.closed

    def commit(self, accepted: int) -> TargetState:
        with self._session._lock:
            # State access checks closure/staleness before a bounds error or
            # metadata update; do not accidentally revive a superseded branch.
            self._branch._check_active()
            if type(accepted) is not int or not 0 <= accepted <= self.max_accepted_prefix:
                raise ValueError("accepted count exceeds the matching proposal prefix")
            next_token, features = self._snapshots[accepted]
            retained = features.clone()
            state = self._branch.commit(accepted)
            self._session._next_token = next_token
            self._session._last_features = retained
            self._snapshots = ()
            return state

    def abort(self) -> None:
        self._branch.abort()
        self._snapshots = ()
