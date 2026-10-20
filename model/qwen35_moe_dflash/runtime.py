"""Framework-independent, batch-one strict-greedy DFlash scheduler."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import time


@dataclass
class Result:
    token_ids: list[int]
    stop_reason: str
    target_calls: int = 0
    draft_calls: int = 0
    drafted: int = 0
    accepted: int = 0
    prefill_ms: float = 0
    decode_ms: float = 0
    iterations: list[dict] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def generate(executor, prompt, *, max_new_tokens=32, max_draft_tokens=15,
             eos_token_ids=(248044, 248046), ordinary=False):
    """Executor.target(prefix) returns S Top1 rows and retains draft features.

    target and draft calls are synchronous. Recompute every valid prefix;
    rejected proposals cannot persist in recurrent, convolution or KV state.
    target(prefix) must run with use_cache=False and pad to the same fixed gear.
    """
    prompt = list(prompt)
    eos = frozenset(eos_token_ids)
    if not prompt or max_new_tokens <= 0 or max_draft_tokens <= 0:
        raise ValueError("nonempty prompt and positive generation limits required")
    if len(prompt) + max_new_tokens - 1 > executor.sequence_length:
        raise ValueError("prompt plus output exceeds the fixed sequence gear")
    if any(type(t) is not int or not 0 <= t < executor.vocab_size
           for t in [*prompt, *eos]):
        raise ValueError("token ID outside vocabulary")
    prefix = prompt[:]
    result = Result([], "length")

    def target(tokens):
        values = list(executor.target(tokens))
        result.target_calls += 1
        if len(values) != executor.sequence_length:
            raise ValueError("target output does not match the sequence gear")
        if any(type(t) is not int or not 0 <= t < executor.vocab_size for t in values):
            raise ValueError("invalid target token")
        return values

    def append(tokens):
        for token in tokens:
            result.token_ids.append(token)
            prefix.append(token)
            if token in eos:
                result.stop_reason = "eos"
                return True
        return False

    start = time.perf_counter_ns()
    first = target(prefix)[len(prefix) - 1]
    done = append([first])
    result.prefill_ms = (time.perf_counter_ns() - start) / 1e6
    while not done and len(result.token_ids) < max_new_tokens:
        start = time.perf_counter_ns()
        remaining = max_new_tokens - len(result.token_ids)
        predictions = target(prefix)
        proposals, accepted = [], 0
        if not ordinary and remaining > 1:
            count = min(max_draft_tokens, executor.draft_width, remaining - 1)
            values = list(executor.draft())
            result.draft_calls += 1
            if len(values) != executor.draft_width:
                raise ValueError("draft output does not match block width")
            for token in values[:count]:
                if type(token) is not int or not 0 <= token < executor.vocab_size:
                    raise ValueError("invalid draft token")
                proposals.append(token)
                if token in eos:
                    break
        if proposals:
            verified = target(prefix + proposals)
            if verified[len(prefix) - 1] != predictions[len(prefix) - 1]:
                raise RuntimeError("target changed the anchor prediction during verification")
            for index, token in enumerate(proposals):
                if token != verified[len(prefix) - 1 + index]:
                    break
                accepted += 1
            committed = proposals[:accepted]
            if not committed or committed[-1] not in eos:
                committed.append(verified[len(prefix) - 1 + accepted])
        else:
            committed = [predictions[len(prefix) - 1]]
        if len(committed) > remaining:
            raise RuntimeError("verification exceeded generation budget")
        done = append(committed)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        result.decode_ms += elapsed
        result.drafted += len(proposals)
        result.accepted += accepted
        result.iterations.append({
            "proposals": proposals, "accepted": accepted,
            "committed": committed, "latency_ms": elapsed,
        })
    return result


class TorchExecutor:
    """CPU/CUDA framework reference; never represents X90 device evidence."""

    def __init__(self, target, draft, sequence_length, pad_token_id=248044):
        import torch
        if sequence_length < 2:
            raise ValueError("sequence_length must exceed one")
        self.target_graph, self.draft_graph = target, draft
        self.sequence_length = sequence_length
        self.vocab_size = target.config.vocab_size
        self.draft_width = target.config.block_size - 1
        if not 0 <= pad_token_id < self.vocab_size:
            raise ValueError("invalid pad token")
        self.pad_token_id = pad_token_id
        self.device = next(target.parameters()).device
        self.ids = torch.empty(1, sequence_length, dtype=torch.int32, device=self.device)
        self.mask = torch.empty_like(self.ids)
        self.features = None

    def target(self, prefix):
        import torch
        if not 1 <= len(prefix) <= self.sequence_length:
            raise ValueError("prefix outside fixed gear")
        if any(not 0 <= t < self.vocab_size for t in prefix):
            raise ValueError("prefix token outside vocabulary")
        self.ids.fill_(self.pad_token_id)
        self.mask.zero_()
        self.ids[0, :len(prefix)] = torch.tensor(prefix, dtype=torch.int32, device=self.device)
        self.mask[0, :len(prefix)] = 1
        with torch.inference_mode():
            top1, self.features = self.target_graph(self.ids, self.mask)
        if not torch.isfinite(self.features).all():
            raise FloatingPointError("nonfinite target features")
        return top1[0].tolist()

    def draft(self):
        import torch
        if self.features is None:
            raise RuntimeError("target must execute before the draft")
        with torch.inference_mode():
            return self.draft_graph(self.features, self.ids, self.mask)[0].tolist()
