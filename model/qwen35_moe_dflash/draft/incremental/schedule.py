"""Pure planning of context catch-up; no implicit truncation or cache mutation."""
from __future__ import annotations

from dataclasses import dataclass

from qwen35_moe_dflash.graph.incremental_draft import validate_incremental_draft_controls


@dataclass(frozen=True)
class DraftAppend:
    start_position: int
    valid_rows: int
    anchor: int
    proposal_count: int

    @property
    def context_end(self):
        return self.start_position + self.valid_rows


def plan_incremental_round(config, *, cached_rows, committed_tokens, anchor,
                           context_rows=16, cache_capacity=192, max_draft_tokens=15):
    """Catch up a complete committed prefix, excluding its latest anchor.

    Feature slice for each call is [start_position:context_end]. Pad the rest
    of its C rows with zeros. Earlier chunks have proposal_count=0 (ignore
    token outputs); only the last call proposes. Even a zero-delta round runs
    once for its new anchor. Advance the cache cursor AND swap all 12 output
    buffers only after that call succeeds. Failed calls never advance either.
    This plan does not own buffers or claim compiled/runtime ABI validation.
    """
    if (type(cached_rows) is not int or type(committed_tokens) is not int
            or committed_tokens < 1 or not 0 <= cached_rows <= committed_tokens - 1):
        raise ValueError('cache cursor must cover a prefix before the latest committed anchor')
    end = committed_tokens - 1
    # Validate geometry before using context_rows as a loop stride.
    validate_incremental_draft_controls(config, start_position=cached_rows, valid_rows=0,
        anchor=anchor, proposal_count=max_draft_tokens, context_rows=context_rows,
        cache_capacity=cache_capacity, max_draft_tokens=max_draft_tokens)
    if end > cache_capacity:
        raise ValueError('committed history exceeds cache capacity; truncation is forbidden')
    result = []
    start = cached_rows
    while True:
        valid = min(context_rows, end - start)
        count = max_draft_tokens if start + valid == end else 0
        validate_incremental_draft_controls(config, start_position=start, valid_rows=valid,
            anchor=anchor, proposal_count=count, context_rows=context_rows,
            cache_capacity=cache_capacity, max_draft_tokens=max_draft_tokens)
        result.append(DraftAppend(start, valid, anchor, count))
        start += valid
        if start == end:
            return tuple(result)
