"""Independent CPU diagnostic for the source B1 packed paged-attention contract.

Uses captured additive mask and logical KV length; never replaces capture golden.
This is a mathematical reference, not an emulator of either device kernel.
"""
from __future__ import annotations

import json
import math
import numpy as np

from .common import metrics, read_array


def unpack(value, heads):
    b, width, seq, lane = value.shape
    if b != 1 or lane != 16 or heads < 1 or width % heads:
        raise ValueError('requires B1 source packed [1, heads*D/16, S, 16]')
    return value.reshape(b, heads, width//heads, seq, lane).transpose(0, 1, 3, 2, 4).reshape(b, heads, seq, -1)


def paged(cache, table, heads, dim, block_size, length):
    if (cache.ndim != 4 or list(cache.shape[1:]) != [heads*dim//16, block_size, 16]
            or table.ndim != 2 or table.shape[0] != 1 or table.dtype.kind not in 'iu'
            or length < 1 or length > table.shape[1]*block_size):
        raise ValueError('KV cache/table/length differs from source packed contract')
    pages = table[0, :(length+block_size-1)//block_size]
    if np.any(pages < 0) or np.any(pages >= cache.shape[0]):
        raise ValueError('active block table entry out of cache range')
    view = cache.reshape(cache.shape[0], heads, dim//16, block_size, 16)
    view = view.transpose(0, 1, 3, 2, 4).reshape(cache.shape[0], heads, block_size, dim)
    return view[pages].transpose(1, 0, 2, 3).reshape(heads, -1, dim)[:, :length]


def reference_math(query, key, value, mask, table, length, *, heads, kv_heads, block_size, scale):
    """Compute softmax(scale * QK^T + captured_mask) V in FP64, then FP16.

    No inferred causal mask, padding removal, all-masked zero fill or layout search.
    """
    if heads < 1 or kv_heads < 1 or heads % kv_heads or block_size < 1 or not math.isfinite(scale):
        raise ValueError('invalid GQA/block/scale contract')
    if any(a.dtype != np.float16 for a in (query, key, value, mask)):
        raise ValueError('this diagnostic requires captured FP16 Q/K/V and additive mask')
    if any(not np.isfinite(a).all() for a in (query, key, value)):
        raise ValueError('nonfinite Q/K/V; independent finite attention reference unavailable')
    q = unpack(query, heads)[0].astype(np.float64)
    _, seq, dim = q.shape
    k = paged(key, table, kv_heads, dim, block_size, length).astype(np.float64)
    v = paged(value, table, kv_heads, dim, block_size, length).astype(np.float64)
    if mask.ndim != 4 or mask.shape[:3] != (1, 1, seq) or mask.shape[3] < length:
        raise ValueError('requires captured additive mask [1,1,S,KV_stride] covering allQLen')
    bias = mask[0, 0, :, :length].astype(np.float64)
    if np.isnan(mask).any() or np.isposinf(mask).any():
        raise ValueError('NaN/+Inf mask unsupported; do not guess all-masked-row semantics')
    undefined = np.flatnonzero(~np.isfinite(bias).any(axis=1)).tolist()
    finite_bias = bias[np.isfinite(bias)]
    evidence = dict(status='COMPUTED' if not undefined else 'UNDEFINED_ALL_MASKED_ROWS',
        scope='INDEPENDENT_SOURCE_CONTRACT_REFERENCE; NOT_DEVICE_KERNEL_EMULATION_OR_CAPTURE_REPLACEMENT',
        formula='softmax((Q @ K.T) * scale + captured_additive_mask) @ V',
        arithmetic='NUMPY_FLOAT64; FINAL_ROUND_TO_FLOAT16', query_shape=list(query.shape),
        num_heads=heads, num_key_value_heads=kv_heads, head_dim=dim, scale_value=scale,
        kv_length=length, kv_length_basis='CAPTURED_all_seq_lengths_q; USER_PROVIDED_KERNEL_KV_LOOP',
        mask_stride=mask.shape[3], causal_mask_added=False, all_masked_zero_fill=False,
        mask=dict(negative_infinity_elements=int(np.isneginf(bias).sum()),
            finite_min=float(finite_bias.min()) if finite_bias.size else None,
            finite_max=float(finite_bias.max()) if finite_bias.size else None,
            all_negative_infinity_query_rows=undefined,
            rows_16_31=[dict(query=i, zero_bias_keys=int(np.count_nonzero(bias[i] == 0)),
                finite_bias_keys=int(np.isfinite(bias[i]).sum())) for i in range(16, min(seq, 32))]),
        v_bounds=[dict(kv_head=i, min=float(v[i].min()), max=float(v[i].max()),
                      abs_max=float(np.abs(v[i]).max())) for i in range(kv_heads)],
        full_value_cache_abs_max=float(np.abs(value.astype(np.float64)).max()))
    output = None
    if not undefined:
        result = np.empty_like(q)
        for h in range(heads):
            kv = h//(heads//kv_heads)
            for start in range(0, seq, 32):
                scores = (q[h, start:start+32] @ k[kv].T)*scale + bias[start:start+32]
                scores -= scores.max(axis=-1, keepdims=True)
                weights = np.exp(scores)
                weights /= weights.sum(axis=-1, keepdims=True)
                result[h, start:start+32] = weights @ v[kv]
        if not np.isfinite(result).all():
            raise ValueError('independent reference produced nonfinite values')
        output = result.astype(np.float16)[None]
    return dict(evidence=evidence, output=output, v=v, heads=heads, kv_heads=kv_heads,
                valid_rows=np.isfinite(bias).any(axis=1))


def prepare_reference(folder, exported, contract):
    """Bind inputs by source AIR ports, never by equal shapes or argument order."""
    edges, chains = contract['input_edges'], contract['input_chains']
    specs = {s['name']: s for s in exported['source_inputs']}
    aliases = {'query': ('query',), 'key': ('key', 'key0'), 'value': ('value', 'value0'),
               'mask': ('atten_mask',), 'table': ('block_table',), 'length': ('all_seq_lengths_q',)}
    used, values, bindings = set(), {}, {}
    for role, names in aliases.items():
        matches = [n for n in names if edges.get(n)]
        if len(matches) != 1:
            raise ValueError(f'unresolved unique {role} AIR port; available ports: {list(edges)}')
        port = matches[0]
        used.add(port)
        trace = chains[port]['chain']
        if (not trace or trace[-1]['type'] != 'Data' or trace[-1]['output_slot'] != 0
                or any(n['type'] != 'Identity' or n['output_slot'] != 0 for n in trace[:-1])):
            raise ValueError(f'{role} must bind through Data/Identity only; review source chain')
        spec = specs[trace[-1]['name']]
        values[role] = read_array(folder, spec)
        bindings[role] = dict(port=port, source_name=spec['name'], fixture_index=spec['fixture_index'],
                              shape=spec['shape'], dtype=spec['dtype'], sha256=spec['sha256'])
    extra = [name for name, edge in edges.items() if edge and name not in used |
             {'actual_seq_lengths_q', 'actual_seq_lengths_kv'}]
    if extra:
        raise ValueError(f'extra populated attention inputs require reference review: {extra}')
    if contract['query_length_constant']['status'] != 'SOURCE_CONST_BYTES_VERIFIED':
        raise ValueError('query length Const not verified')
    attrs = contract['attributes']
    captured = exported['source_contract']['captured_attributes']
    for name, field in (('block_size','i'), ('num_key_value_heads','i'), ('scale_value','f'), ('inner_precise','i')):
        if attrs[name][field] != captured[name]:
            raise ValueError(f'source AIR and capture disagree on {name}')
    if (captured['num_heads'] != contract['num_heads'] or captured['input_layout'] != 'BNSD'
            or captured['actual_seq_lengths_q'] != [values['query'].shape[2]]
            or captured['actual_seq_lengths_kv'] != [values['mask'].shape[-1]]
            or captured['inner_precise'] != 2):
        raise ValueError('captured query/mask length/layout differs from reviewed B1 source contract')
    # Both length constants must bind directly and retain their literal payload.
    kv_chain = chains['actual_seq_lengths_kv']['chain']
    if len(kv_chain) != 1 or kv_chain[0]['type'] != 'Const':
        raise ValueError('actual_seq_lengths_kv must be a direct source Const')
    # Payload identity is retained by source_call_contract, without trusting readable text.
    if contract.get('kv_length_constant', {}).get('status') != 'SOURCE_CONST_BYTES_VERIFIED':
        raise ValueError('KV stride Const not verified against captured mask width')
    length = values['length']
    if length.shape != (1,) or length.dtype != np.int64:
        raise ValueError('allQLen must be captured INT64[1]')
    result = reference_math(values['query'], values['key'], values['value'], values['mask'], values['table'],
        int(length[0]), heads=contract['num_heads'], kv_heads=captured['num_key_value_heads'],
        block_size=captured['block_size'], scale=captured['scale_value'])
    result['evidence']['bindings'] = bindings
    return result


def compare_reference(actual, golden, prepared):
    heads, kv_heads, v = prepared['heads'], prepared['kv_heads'], prepared['v']
    av, gv = unpack(actual, heads), unpack(golden, heads)
    result = dict(status='DIAGNOSTIC_ONLY; ORIGINAL_GOLDEN_AND_ACCEPTANCE_UNCHANGED',
                  source_layout_only=True, bounds=[], comparisons={})
    rows = prepared['valid_rows']
    for label, array in (('actual', av), ('captured_golden', gv)):
        for h in range(heads):
            kv = h//(heads//kv_heads)
            sample = array[0, h, rows].astype(np.float64)
            excess = np.maximum(np.maximum(v[kv].min(axis=0)-sample, sample-v[kv].max(axis=0)), 0)
            finite = bool(np.isfinite(sample).all())
            result['bounds'].append(dict(side=label, head=h, kv_head=kv,
                checked_query_rows=int(rows.sum()),
                outside_v_channel_range=int(np.count_nonzero(excess > 0)) if finite else None,
                max_excess=float(excess.max()) if finite and excess.size else None,
                note='STRICT_CONVEX_BOUND_DIAGNOSTIC; SMALL_ROUNDING_EXCESS_IS_NOT_AN_ACCURACY_GATE'))
        if prepared['output'] is not None:
            result['comparisons'][label+'_vs_reference'] = metrics(array, prepared['output'])
    return result


def reference_summary(cases):
    lines = ['Independent CPU attention diagnostic; original capture golden unchanged.',
        'Formula: softmax(scale * QK^T + captured additive mask) V; FP64 arithmetic, final FP16.',
        'Source packed layout and allQLen define the reference; no inferred query padding or causal mask.',
        'Five metrics below compare each side against this independent reference, not each other.',
        'Strict V bounds are diagnostics; small rounding excess is not an accuracy gate.']
    for row in cases:
        evidence = row.get('independent_reference', {})
        lines += ['', row['case']+' ('+row['phase']+'): '+evidence.get('status','UNAVAILABLE')]
        if evidence.get('error'):
            lines.append('error='+evidence['error'])
        if 'v_bounds' not in evidence:
            continue
        lines.append('Q_shape='+str(evidence['query_shape'])+'; KV_length='+str(evidence['kv_length'])+
                     '; mask_stride='+str(evidence['mask_stride']))
        lines.append('full_value_cache_abs_max='+str(evidence['full_value_cache_abs_max']))
        lines.append('active_V_bounds='+json.dumps(evidence['v_bounds']))
        lines.append('mask='+json.dumps(evidence['mask']))
        for pattern in row['output_patterns']:
            compared = pattern.get('independent_reference')
            if compared is None:
                continue
            lines.append('invocations='+str(pattern['invocations']))
            for side in ('actual','captured_golden'):
                bounds = [b for b in compared['bounds'] if b['side']==side]
                counts = [b['outside_v_channel_range'] for b in bounds]
                excess = [b['max_excess'] for b in bounds if b['max_excess'] is not None]
                lines.append(side+'_V_bounds='+json.dumps(dict(
                    outside_elements=sum(counts) if all(c is not None for c in counts) else None,
                    max_excess=max(excess) if excess else None,
                    checked_query_rows_per_head=bounds[0]['checked_query_rows'] if bounds else 0)))
                metric = compared['comparisons'].get(side+'_vs_reference')
                lines.append(side+'_vs_reference='+json.dumps({k:metric.get(k) for k in
                    ('max_abs','relative_l2','cosine','kl_divergence','rmse','mean_relative_error',
                     'passed','failed_metrics','max_error_coordinate','actual_at_max','golden_at_max')}
                    if metric else dict(status='UNAVAILABLE_ALL_MASKED_ROWS')))
    return '\n'.join(lines)+'\n'
