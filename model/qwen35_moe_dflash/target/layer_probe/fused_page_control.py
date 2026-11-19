"""Sweep only the live T64 cumulative length, then exercise the two active KV pages."""
from hashlib import sha256

import numpy as np
import torch

from .attention_reference import unpack
from .capture import cpu
from .common import metrics
from .fused_rows_control import validate_t64_call
from .fused_value_control import live_description, same_arguments


def compare_pages(npu, live, saved, original, reference, report):
    validate_t64_call(live, saved, report)
    table = saved['block_table']
    key, value = saved['key'][0], saved['value'][0]
    page0, page1 = table[0, :2].tolist()
    if page0 == page1 or not all(0 <= i < min(key.shape[0], value.shape[0]) for i in (page0, page1)):
        raise ValueError('requires two distinct addressable pages')
    report.update(physical_q_rows=64, block_size=32, physical_pages=[page0, page1],
                  actual_seq_lengths_kv=live['actual_seq_lengths_kv'])

    def invoke(args, length, reproduce=False):
        packed = cpu(npu.adn_fused_infer_attention(**dict(args, all_seq_lengths_q=[length]))).numpy()
        if packed.shape != original.shape or packed.dtype != original.dtype:
            raise ValueError('fused output descriptor changed')
        if reproduce and packed.tobytes() != original.tobytes():
            raise ValueError('original packed T64 output did not reproduce')
        return unpack(packed, live['num_heads'])[:, :, :length]

    def digest(a):
        return sha256(a.tobytes()).digest()

    invoke(live, 64, reproduce=True)
    rows = report['length_sweep'] = []
    hashes = {}
    for length in range(32, 65):
        actual = invoke(live, length, reproduce=length == 64)
        prefix = actual[:, :, :32]
        if length == 32:
            prefix32 = digest(prefix)
        hashes[length] = digest(actual)
        active_score = metrics(actual, reference[:, :, :length])
        prefix_score = metrics(prefix, reference[:, :, :32])
        rows.append(dict(length=length, passed=active_score['passed'], relative_l2=active_score['relative_l2'],
                         prefix_passed=prefix_score['passed'], prefix_relative_l2=prefix_score['relative_l2'],
                         prefix_equal_to_length32=digest(prefix) == prefix32))
    for row in reversed(rows):
        length = row['length']
        row['repeat_equal_bits'] = digest(invoke(live, length)) == hashes[length]
    if not same_arguments(live, saved):
        raise ValueError('original fused inputs changed during the length sweep')

    failed = [r['length'] for r in rows if not r['passed']]
    prefix_failed = [r['length'] for r in rows if not r['prefix_passed']]
    report.update(failed_lengths=failed, first_failed_length=next(iter(failed), None),
                  first_failed_prefix32_length=next(iter(prefix_failed), None),
                  first_changed_prefix32_length=next((r['length'] for r in rows if not r['prefix_equal_to_length32']), None),
                  unstable_lengths=[r['length'] for r in rows if not r['repeat_equal_bits']])
    if report['unstable_lengths']:
        raise ValueError('length sweep did not reproduce in reverse order; see per-length results')

    # Keep allocations fixed for baseline, remap, value perturbations, and restoration.
    scratch = dict(live, key=[live['key'][0].clone()], value=[live['value'][0].clone()],
                   block_table=live['block_table'].clone())
    if live_description(npu, scratch) != live_description(npu, live):
        raise ValueError('private K/V/table descriptors differ from the live call')
    lengths = sorted({32, 64, *failed[:1], *prefix_failed[:1]})
    trials = report['page_trials'] = {}

    def trial(name, k, v, t, *, expect_original=False, same_math=False):
        scratch['key'][0].copy_(k)
        scratch['value'][0].copy_(v)
        scratch['block_table'].copy_(t)
        expected = dict(saved, key=[k], value=[v], block_table=t)
        if not same_arguments(scratch, expected):
            raise ValueError('page-control upload failed: ' + name)
        outputs = {}
        trial_rows = trials[name] = {}
        for length in lengths:
            actual = invoke(scratch, length)
            out_hash = digest(actual)
            stable = digest(invoke(scratch, length)) == out_hash
            outputs[length] = actual[:, :, :32].copy()
            info = trial_rows[str(length)] = dict(repeat_equal_bits=stable, equal_bits_to_sweep=out_hash == hashes[length])
            if not stable or (expect_original and not info['equal_bits_to_sweep']):
                raise ValueError('page-control baseline or repeat drift: ' + name)
            if same_math:
                score = metrics(actual, reference[:, :, :length])
                info.update(passed=score['passed'], relative_l2=score['relative_l2'])
        if not same_arguments(scratch, expected):
            raise ValueError('fused operands changed during ' + name)
        return outputs

    trial('original_copy', key, value, table, expect_original=True)
    moved_table = table.clone()
    moved_table[table == page0] = page1
    moved_table[table == page1] = page0
    moved_key, moved_value = key.clone(), value.clone()
    for source, moved in ((key, moved_key), (value, moved_value)):
        moved[page0], moved[page1] = source[page1], source[page0]
        if not torch.equal(source[table[0, :2].long()], moved[moved_table[0, :2].long()]):
            raise ValueError('logical active pages changed during physical remap')
    trial('same_logical_kv_swapped_physical_pages', moved_key, moved_value, moved_table, same_math=True)

    # For queries 0..31, every key on logical page 1 is causally masked.
    modified = value.clone()
    modified[page1].fill_(1)
    plus = trial('page1_v_plus_one', key, modified, table)
    modified[page1].fill_(-1)
    minus = trial('page1_v_minus_one', key, modified, table)
    dependence = report['prefix32_plus_vs_minus'] = {}
    for length in lengths:
        diff = metrics(plus[length], minus[length])
        dependence[str(length)] = dict(equal_bits=digest(plus[length]) == digest(minus[length]),
                                       max_abs_difference=diff['max_abs'])
    trial('restored_copy', key, value, table, expect_original=True)
    invoke(live, 64, reproduce=True)
    if not same_arguments(live, saved):
        raise ValueError('original fused operands changed during page controls')
    report['status'] = 'INSPECTED'
