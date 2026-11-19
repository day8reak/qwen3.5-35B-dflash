"""Vary the L3 fused query window after the original T64 Prefill has returned."""
import numpy as np
import torch

from .attention_reference import unpack
from .capture import cpu
from .common import metrics
from .fused_value_control import same_arguments


def upload_window(npu, original, host_window, base_shape):
    # The original packed Q is a four-dimensional view of head/lane storage.
    # Upload contiguous words in five dimensions before restoring its public view.
    source = host_window.clone(memory_format=torch.contiguous_format)
    base = source.reshape(base_shape).to(original.device, copy=True)
    fmt = npu.get_npu_format(original)
    if npu.get_npu_format(base) != fmt:
        base = npu.npu_format_cast(base, fmt)
    result = base.view(source.shape)
    if (npu.get_npu_format(result) != fmt or result.dtype != original.dtype
            or result.device != original.device or result.storage_offset() != original.storage_offset()
            or result.stride() != source.stride() or cpu(result).numpy().tobytes() != source.numpy().tobytes()):
        raise ValueError('private row window failed descriptor or byte round-trip')
    return result


def validate_t64_call(live, saved, report):
    heads, kv_heads = live['num_heads'], live['num_key_value_heads']
    query, mask = saved['query'], saved['atten_mask']
    if (tuple(query.shape) != (1, 256, 64, 16) or tuple(mask.shape) != (1, 1, 64, 2048)
            or (heads, kv_heads, live['block_size']) != (16, 2, 32)
            or (live['actual_seq_lengths_q'], live['actual_seq_lengths_kv'], live['all_seq_lengths_q'])
            != ([64], [2048], [64]) or not all(live[k].is_contiguous() for k in ('query', 'atten_mask'))):
        raise ValueError('expected the recorded four-layer T64 BNSD call with list lengths')
    blocked = np.arange(mask.shape[-1])[None, :] > np.arange(64)[:, None]
    bias = mask.numpy()[0, 0]
    if np.any(bias[~blocked] != 0) or not np.all(np.isneginf(bias[blocked]) | (bias[blocked] == -65504)):
        raise ValueError('cannot use slices of the T64 reference without its fresh causal mask')
    if not (report['packed_inputs_match_previous_fixture'] and report['prefill_output_matches_previous_fixture']):
        raise ValueError('live T64 call differs from the selected fixture; inspect the new capture first')


def compare_query_rows(npu, live, saved, original, reference, report):
    validate_t64_call(live, saved, report)
    query, mask, heads = saved['query'], saved['atten_mask'], live['num_heads']

    def window(start, physical):
        q = query[:, :, start:start+physical]
        m = mask[:, :, start:start+physical]
        return dict(live, actual_seq_lengths_q=[physical],
                    query=upload_window(npu, live['query'], q, (1, heads, q.shape[1]//heads, physical, 16)),
                    atten_mask=upload_window(npu, live['atten_mask'], m, m.shape))

    cases = report['cases'] = {}
    bound = float(saved['value'][0].abs().max())
    report.update(reference='SAME_T64_CPU_CAUSAL_ROWS', value_cache_abs_max=bound,
                  padded_rows='RETAINED_T64_QUERY_AND_MASK_ROWS_OUTSIDE_SCORED_PREFIX',
                  actual_seq_lengths_kv=live['actual_seq_lengths_kv'])

    def check(label, args, start, valid, *, reproduce=False):
        physical = args['query'].shape[2]
        call = dict(args, all_seq_lengths_q=[start + valid])
        expected_args = dict(saved, query=query[:, :, start:start+physical],
                             atten_mask=mask[:, :, start:start+physical], actual_seq_lengths_q=[physical],
                             all_seq_lengths_q=[start + valid])
        info = cases[label] = dict(first_query=start, physical_rows=physical,
                                   valid_rows=valid, all_seq_lengths_q=start+valid)
        first = None
        for repeat in range(2):
            packed = cpu(npu.adn_fused_infer_attention(**call)).numpy()
            if packed.shape != tuple(call['query'].shape) or packed.dtype != np.float16:
                raise ValueError('unexpected fused output in ' + label)
            if reproduce and packed.tobytes() != original.tobytes():
                info['equal_bits_to_prefill'] = False
                raise ValueError('T64 control did not reproduce the live packed output: ' + label)
            active = unpack(packed, heads)[:, :, :valid]
            if repeat == 0:
                first = active.copy()
                expected = reference[:, :, start:start+valid]
                score = metrics(first, expected)
                prefix = metrics(first[:, :, :19], expected[:, :, :19])
                info.update({k: score[k] for k in ('passed', 'relative_l2', 'cosine', 'max_abs', 'failed_metrics')})
                info.update(first19_relative_l2=prefix['relative_l2'],
                            finite_elements_above_cache_abs_max=int(np.count_nonzero(np.isfinite(first) & (np.abs(first) > bound))),
                            zero_head_rows=int(np.count_nonzero((first == 0).all(-1) & (expected != 0).any(-1))))
            else:
                info['valid_output_repeat_equal_bits'] = first.tobytes() == active.tobytes()
            if not same_arguments(call, expected_args):
                raise ValueError('fused operands changed during ' + label)
        if reproduce:
            info['equal_bits_to_prefill'] = True
        return first

    check('original64', live, 0, 64, reproduce=True)
    copied64 = window(0, 64)
    check('copy64', copied64, 0, 64, reproduce=True)
    first32 = window(0, 32)
    by_width = report['physical32_vs_physical64'] = {}
    def compare_widths(valid):
        wide = check(f'physical64_valid{valid}', copied64, 0, valid)
        short = check(f'physical32_valid{valid}', first32, 0, valid)
        diff = metrics(short, wide)
        by_width[str(valid)] = dict(equal_bits=short.tobytes() == wide.tobytes(),
                                    relative_l2=diff['relative_l2'], max_abs=diff['max_abs'])
        return short

    compare_widths(19)
    first = compare_widths(32)
    second = check('second32_total64', window(32, 32), 32, 32)
    joined = metrics(np.concatenate((first, second), axis=2), reference)
    report['two_chunks_vs_cpu'] = {k: joined[k] for k in ('passed', 'relative_l2', 'cosine', 'max_abs', 'failed_metrics')}
    check('restored64', live, 0, 64, reproduce=True)
    report['all_valid_outputs_repeat_equal_bits'] = all(row['valid_output_repeat_equal_bits'] for row in cases.values())
    report['status'] = 'INSPECTED'
