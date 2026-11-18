"""Independent CPU diagnostic for the source B1 packed paged-attention contract.

Uses captured additive mask and logical KV length; never replaces capture golden.
This is a mathematical reference, not an emulator of either device kernel.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np

from .common import metrics, read_array


def unpack(value, heads):
    b, width, seq, lane = value.shape
    if b != 1 or lane != 16 or heads < 1 or width % heads:
        raise ValueError('requires B1 source packed [1, heads*D/16, S, 16]')
    return value.reshape(b, heads, width//heads, seq, lane).transpose(0, 1, 3, 2, 4).reshape(b, heads, seq, -1)


def inspect_prefill_audit(fixture, device_math=None):
    """Summarize the original call in its logical layout using the retained CPU reference."""
    root = Path(fixture).resolve(strict=True)
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest.get('artifact_kind') != 'qwen35-target-prefill-fixture-v1':
        raise ValueError('pass the Prefill fixture directory containing the top-level manifest.json')
    audit = manifest.get('attention_audit', {})
    sequence = manifest.get('sequence_length')
    result = dict(status='INSPECTED', sequence_length=sequence, core_audit=audit.get('status'),
                  reference_qualification=manifest.get('reference_qualification'))
    calls = audit.get('calls', [])
    if len(calls) != 1 or calls[0].get('status') not in ('PASSED', 'FAILED'):
        reasons = [c.get('reason') for c in calls if c.get('reason')]
        return dict(result, status='PARTIAL', reason=audit.get('reason') or reasons or 'no completed core audit')
    call = calls[0]
    evidence = call['reference']
    actual = unpack(read_array(root, call['actual']), evidence['num_heads'])
    reference = read_array(root, call['independent_reference'])
    if actual.shape != reference.shape or actual.shape[2] != sequence:
        raise ValueError('original core output and CPU reference must cover the same logical query rows')
    fields = ('relative_l2', 'cosine', 'rmse', 'mean_relative_error', 'kl_divergence', 'max_abs',
              'failed_metrics', 'nonfinite_actual', 'max_error_coordinate', 'actual_at_max', 'golden_at_max')
    result['core_vs_cpu'] = {k: call['metrics'].get(k) for k in fields}
    result['active_value_bounds'] = evidence['v_bounds']
    # Even selecting the wrong in-range page cannot exceed the entire V cache
    # bound in exact normalized attention. Small rounding excess is diagnostic.
    full_bound = evidence.get('full_value_cache_abs_max')
    if full_bound is not None and math.isfinite(full_bound):
        magnitudes = np.abs(actual[np.isfinite(actual)])
        result['full_value_cache_check'] = dict(
            cache_abs_max=full_bound,
            finite_core_abs_max=float(magnitudes.max()) if magnitudes.size else None,
            finite_core_elements_above_cache_abs_max=int(np.count_nonzero(magnitudes > full_bound)),
            kv_length=evidence['kv_length'], mask_stride=evidence['mask_stride'],
            all_negative_infinity_query_rows=evidence['mask']['all_negative_infinity_query_rows'])
    result['query_blocks'] = blocks = []
    for start in range(0, sequence, 16):
        stop = min(start + 16, sequence)
        a, r = actual[:, :, start:stop], reference[:, :, start:stop]
        diff = metrics(a, r)
        zero_rows = (a == 0).all(axis=-1) & (r != 0).any(axis=-1)
        blocks.append(dict(first_query=start, last_query=stop-1, max_abs=diff['max_abs'],
                           relative_l2=diff['relative_l2'], nonfinite_actual=diff['nonfinite_actual'],
                           zero_head_query_rows_with_nonzero_reference=int(zero_rows.sum())))
    boundary = manifest.get('boundary_capture', {})
    if boundary.get('status') == 'CAPTURED':
        if boundary.get('path') != 'capture/manifest.json':
            raise ValueError('unknown boundary manifest path')
        capture = json.loads((root/boundary['path']).read_text())
        original = [out for r in capture['regions']
                    if (r['layer'], r['call'], r['phase'], r['point']) == (3, 0, 'prefill', 'fused_attention')
                    for out in r['outputs'] if out['name'] == 'golden_attention_nz']
        if len(original) != 1 or original[0]['sha256'] != call['actual']['sha256']:
            return dict(result, status='PARTIAL', reason='boundary capture and audit core outputs do not match')
        rows = [r for r in capture.get('attention_math_replays', [])
                if (r.get('layer'), r.get('call'), r.get('phase')) == (3, 0, 'prefill')]
        if len(rows) == 1:
            replay = rows[0]
            hidden = replay.get('captured_vs_reference', {}).get('golden_hidden', {})
            result['module_replay'] = dict(status=replay['status'], error=replay.get('error'),
                cache_replay_bitwise_equal=replay.get('cache_replay_bitwise_equal'),
                original_hidden_vs_replay={k: hidden.get(k) for k in ('relative_l2', 'cosine', 'max_abs', 'passed')})
    if device_math is not None:
        control = result['device_math'] = dict(device=device_math, status='PARTIAL', calls=[],
            scope='DENSE_FP32_TORCH_OPS_ON_HOST_UNPACKED_INPUTS; FINAL_FP16')
        try:
            import torch
            import torch_npu
            device = torch.device(device_math)
            if device.type != 'npu':
                raise ValueError('device math requires an NPU, e.g. npu:0')
            control.update(torch_version=str(torch.__version__), torch_npu_version=str(torch_npu.__version__))
            inputs = {s['name']: read_array(root, s) for s in call['inputs']}
            previous = None
            for output in _torch_math_outputs(inputs, evidence, device):
                diff = metrics(output, reference)
                row = {k: diff[k] for k in ('passed', 'relative_l2', 'cosine', 'max_abs', 'failed_metrics')}
                if previous is not None:
                    row['equal_bits_to_previous'] = output.tobytes() == previous.tobytes()
                control['calls'].append(row)
                previous = output
            passed = len(control['calls']) == 3 and all(
                r['passed'] and r.get('equal_bits_to_previous', True) for r in control['calls'])
            control['status'] = 'PASSED' if passed else 'FAILED'
        except (ImportError, RuntimeError, OSError, ValueError, KeyError) as exc:
            control['error'] = f'{type(exc).__name__}: {exc}'
        if control['status'] != 'PASSED':
            result['status'] = control['status']
    return result


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


def _torch_math_outputs(inputs, evidence, device):
    """Run the split attention arithmetic three times on the same dense tensors."""
    import torch

    heads, kv_heads, length = (evidence[k] for k in ('num_heads', 'num_key_value_heads', 'kv_length'))
    q = unpack(inputs['query'], heads)[0]
    mask = inputs['mask']
    if (evidence['status'] != 'COMPUTED' or kv_heads < 1 or heads % kv_heads
            or length != q.shape[1] or q.shape[2] != evidence['head_dim']
            or mask.shape != (1, 1, length, evidence['mask_stride']) or mask.shape[3] < length):
        raise ValueError('requires a completed fresh Prefill audit with matching head, mask and sequence dimensions')

    def tensor(array):
        return torch.from_numpy(np.ascontiguousarray(array)).to(device=device, dtype=torch.float32)

    def cache(name):
        dense = paged(inputs[name], inputs['table'], kv_heads, q.shape[2], inputs['key'].shape[2], length)
        return tensor(dense.repeat(heads // kv_heads, axis=0))

    with torch.no_grad():
        query, key, value = tensor(q), cache('key'), cache('value')
        key_t = key.transpose(1, 2).contiguous()
        bias = tensor(mask[0, :, :, :length])
        for _ in range(3):
            scores = torch.bmm(query, key_t) * evidence['scale_value'] + bias
            probabilities = torch.softmax(scores, dim=-1)
            context = torch.bmm(probabilities, value)
            if context.device != query.device or context.dtype != torch.float32:
                raise ValueError('unexpected output device or dtype for the FP32 attention control')
            yield context.unsqueeze(0).to(torch.float16).to('cpu', copy=True).contiguous().numpy()


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
