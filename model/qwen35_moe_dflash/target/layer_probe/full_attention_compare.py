"""Array-only checks for the fresh-Prefill whole Attention cache outputs."""
from __future__ import annotations

import numpy as np

from .common import metrics


def cache_partitions(actual, golden, before, contract, *, atol=None, rtol=None):
    table = np.asarray(contract['block_table'])
    length, block = contract['sequence_length'], contract['block_size']
    shape = actual.shape
    if (actual.dtype != before.dtype or actual.dtype != golden.dtype or len(shape) != 4
            or golden.shape != shape or before.shape != shape or shape[2:] != (block, 16)
            or table.ndim != 2 or table.shape[0] != 1 or table.dtype.kind not in 'iu'
            or not 0 < length <= table.shape[1]*block):
        raise ValueError('FullAttention cache partition ABI changed')
    pages = table[0, np.arange(length)//block]
    if (pages < 0).any() or (pages >= shape[0]).any():
        raise ValueError('captured block table points outside cache')
    page_map = table[0, : (length+block-1)//block]
    if len(np.unique(page_map)) != len(page_map):
        raise ValueError('fresh-Prefill active cache pages alias')
    active_rows = np.zeros((shape[0], block), dtype=bool)
    active_rows[pages, np.arange(length) % block] = True
    active = np.broadcast_to(active_rows[:, None, :, None], shape)
    reference_preserved = before[~active].tobytes() == golden[~active].tobytes()
    if not reference_preserved:
        raise ValueError('module reference modified cache outside declared Prefill write rows')
    preserved = before[~active].tobytes() == actual[~active].tobytes()
    active_metrics = metrics(actual[active], golden[active], atol=atol, rtol=rtol)
    return dict(active_elements=int(active.sum()), untouched_elements=int((~active).sum()),
        active_metrics=active_metrics, untouched_equal_input_bits=preserved,
        reference_untouched_equal_input_bits=reference_preserved,
        passed=bool(preserved and active_metrics['passed']))
