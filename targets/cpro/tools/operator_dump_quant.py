"""Offline checks of DynamicQuant and I4ZN W4 operands in an exported AIR."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from quant_batch_matmul_cpu_golden import cpu_golden
from qwen35_moe_dflash.common.accuracy import floating_metrics
from qwen35_moe_dflash.target.layer_probe.common import save_json
from qwen35_moe_dflash.target.layer_probe.qlinear_math import dynamic_quant
from qwen35_moe_dflash.target.prefill_t64.attention_slice import DTYPES
from qwen35_moe_dflash.target.prefill_t64.decode_cache_control import graph_message
from qwen35_moe_dflash.target.prefill_t64.gdn_linear_operands import VIEWS, static_operand


def path_to_input(edge, nodes, passthrough):
    while True:
        yield edge
        op = nodes[edge.rsplit(':', 1)[0]]
        if not edge.endswith(':0') or op.type not in passthrough:
            return
        edge = op.input[0]


def matrix(array, m, k):
    if tuple(d for d in array.shape if d != 1) != tuple(d for d in (m, k) if d != 1):
        raise ValueError(f'Unexpected dump or reference axes {array.shape} for [{m},{k}]')
    return array.reshape(m, k)


def score(actual, reference):
    stats = floating_metrics(actual, reference)
    return {k: stats[k] for k in ('cosine', 'relative_l2', 'max_abs')}


def quant_readings(edge, variants, expected, m, k, rows):
    """Keep each reading; test its agreement with the same-run input on the requested rows."""
    readings, consistent = [], {}
    for array, sources in variants:
        value = matrix(array, m, k)[:rows]
        if expected.dtype == np.int8:
            if value.dtype != np.int8:
                raise ValueError(f'{edge}: expected INT8 codes, got {value.dtype}')
            delta = np.abs(value.astype(np.int16) - expected.astype(np.int16))
            metrics = dict(max_code_delta=int(delta.max()), mismatch_ratio=float(np.mean(delta != 0)))
            compatible = metrics['max_code_delta'] <= 1
        else:
            if value.dtype.kind != 'f':
                raise ValueError(f'{edge}: expected floating per-token scales, got {value.dtype}')
            metrics = score(value, expected)
            compatible = bool(np.all(value >= 0) and np.allclose(value, expected, rtol=0.002, atol=2**-24))
        readings.append(dict(sources=sources, compatible=compatible, **metrics))
        if compatible:
            consistent[(value.dtype.str, value.tobytes())] = value
    return readings, next(iter(consistent.values())) if len(consistent) == 1 else None


def matmul_inputs(edge, nodes):
    if not edge:
        raise ValueError('Missing mapped projection edge')
    output = list(path_to_input(edge, nodes, VIEWS))[-1]
    op = nodes[output.rsplit(':', 1)[0]]
    if op.type != 'QuantBatchMatmulV4444' or not output.endswith(':0'):
        raise ValueError(f'{output} is not an A8W4 projection output')
    slots = dict(zip((s.decode() for s in op.attr['_input_name_key'].list.s),
                     op.attr['_input_name_value'].list.i, strict=True))
    if any(not 0 <= slot < min(len(op.input), len(op.input_desc))
           for port, slot in slots.items() if port in ('x1', 'x2', 'scale', 'pertoken_scale')):
        raise ValueError(f'{op.name}: invalid W4 port index')
    ports = {p: op.input[slots[p]] for p in ('x1', 'x2', 'scale', 'pertoken_scale')}
    if {i for i, e in enumerate(op.input) if e and not e.endswith(':-1')} != {slots[p] for p in ports}:
        raise ValueError('Unexpected additional matmul operand')
    token_path = list(path_to_input(ports['pertoken_scale'], nodes, VIEWS | {'Cast'}))
    codes_edge = list(path_to_input(ports['x1'], nodes, VIEWS))[-1]
    dq = nodes[codes_edge.rsplit(':', 1)[0]]
    if (dq.type != 'DynamicQuant' or codes_edge != dq.name+':0' or token_path[-1] != dq.name+':1'
            or [e for e in dq.input if e and not e.endswith(':-1')] != [dq.input[0]]):
        raise ValueError('Expected one unmodified input to symmetric per-token DynamicQuant and its two output slots')
    return output, op, ports, slots, dq, codes_edge, token_path


def check_quant_input(profile, nodes, dq, codes_edge, scale_edge, candidates, result):
    codes, bad_q = candidates(codes_edge)
    scales, bad_s = candidates(scale_edge)
    if not codes or not scales or bad_q or bad_s:
        raise ValueError('Incomplete code or scale dump')
    m = scales[0][0].size
    k = codes[0][0].size // m
    rows = profile.get('valid_rows')
    if rows is None:
        rows = m
    if type(rows) != int or not 0 < rows <= m:
        raise ValueError(f'Invalid range of {rows} rows for M={m}')
    for source in path_to_input(dq.input[0], nodes, VIEWS):
        options, skipped = candidates(source)
        if len(options) == 1 and not skipped:
            x = matrix(options[0][0], m, k)[:rows]
            result.update(input_edge=source, input_sources=options[0][1], rows=rows)
            break
    else:
        raise ValueError('No unambiguous same-run input dump on the DynamicQuant input/view path')
    expected_q, expected_s, _ = dynamic_quant(x)
    result['codes'], q = quant_readings(codes_edge, codes, expected_q, m, k, rows)
    result['scales'], s = quant_readings(scale_edge, scales, expected_s, m, 1, rows)
    for label in ('codes', 'scales'):
        for reading in result[label]:
            offsets = sorted({v['offset'] for v in reading['sources']})
            print(f'  {label} offset={offsets}:', {key: value for key, value in reading.items() if key != 'sources'})
    return m, rows, q, s


def projection(profile, name, edge, nodes, types, candidates, tmp, result):
    output, op, ports, slots, dq, codes_edge, token_path = matmul_inputs(edge, nodes)
    result.update(matmul=op.name, ports=ports, pertoken_scale_path=token_path)
    outputs, skipped = candidates(output)
    if skipped or not outputs:
        raise ValueError(f'Incomplete output dump at {output}')
    if name == 'qkv' and profile.get('history_role'):
        cache, cache_path = profile['observed'][profile['history_role']]
        _, width, taps = cache.shape
        result['cache_tail'] = []
        for value, sources in outputs:
            tail = matrix(value, value.size // width, width)[-taps:].T
            with np.errstate(over='ignore', invalid='ignore'):
                tail = tail.astype(cache.dtype)
            match = dict(cache=cache_path, sources=sources, exact=bool(np.array_equal(tail, cache[0])),
                         **score(tail, cache[0]))
            result['cache_tail'].append(match)
            print('  QKV 物理末行 vs 同次返回 conv_state:', match['exact'], match['relative_l2'])
    m, rows, q, s = check_quant_input(profile, nodes, dq, codes_edge, token_path[-1], candidates, result)
    if q is None or s is None:
        raise ValueError('No unique formula-compatible code/scale values over these rows; see all readings')
    # Follow any explicit scale casts in AIR order, then use the declared consumer dtype.
    for scale_edge in reversed(token_path[:-1]):
        node = nodes[scale_edge.rsplit(':', 1)[0]]
        if node.type == 'Cast':
            s = s.astype(types[node.output_desc[0].dtype])
    weight, groups = (static_operand(ports[p], nodes, types, profile['air'].parent, tmp)
                      for p in ('x2', 'scale'))
    scale_type = types[op.input_desc[slots['scale']].dtype]
    token_type = types[op.input_desc[slots['pertoken_scale']].dtype]
    result.update(weight_shape=list(weight.shape), group_scale_shape=list(groups.shape),
                  weight_scale_dtype=scale_type, token_scale_dtype=token_type)
    prediction = cpu_golden(q, weight, groups.astype(scale_type, copy=False), s.astype(token_type, copy=False))
    n = prediction.shape[1]
    result['prediction_vs_cpro'] = [dict(sources=sources, **score(prediction, matrix(value, m, n)[:rows]))
                                    for value, sources in outputs]
    reference = profile['references'][1].get(name)
    if reference is not None:
        result['prediction_vs_310p'] = score(prediction, matrix(reference, m, n)[:rows])
    print('  同次编码和尺度 × AIR 权重，复算 vs CPro:',
          [(r['cosine'], r['relative_l2']) for r in result['prediction_vs_cpro']])
    print('  复算 vs 310P:', result.get('prediction_vs_310p'))


def diagnose(profile, names, candidates, folder):
    if not profile.get('air'):
        raise ValueError('W4 boundary checks need the source AIR and its adjacent external constants')
    _, graph = graph_message(profile['air'].read_bytes(), 'NetOutput')
    enum = graph.DESCRIPTOR.file.enum_types_by_name['DataType'].values_by_name
    types = {enum[v].number: k for k, v in DTYPES.items()}
    nodes = {n.name: n for n in graph.op}
    edges = {p['name']: p['edge'] for p in profile['points']}
    report = dict(attempt=profile['attempt'], projections={})
    with TemporaryDirectory(dir=folder) as tmp:
        for name in dict.fromkeys(names):
            result = report['projections'][name] = {}
            print(f'[{name}] {edges.get(name)}', flush=True)
            try:
                projection(profile, name, edges.get(name), nodes, types, candidates, Path(tmp), result)
            except (KeyError, IndexError, ValueError, TypeError, OSError, FloatingPointError) as exc:
                result['issue'] = str(exc)
                print('  数据不足:', exc)
    save_json(folder/'quant-boundary.json', report)
    print(folder/'quant-boundary.json')
