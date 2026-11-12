"""Opt-in FP32 rotation replay; captured coefficients and golden stay unchanged."""
from __future__ import annotations

from .common import metrics, read_array


def apply_fp32(target, q, k, cos, sin, output_dtypes):
    # Call the supplied model's partial-rotation helper, including its tail.
    # Output dtype is the captured boundary contract, not necessarily q.dtype:
    # original coefficients can be FP32 while q/k are FP16.
    values = target.apply_rotary_pos_emb(q.float(), k.float(), cos.float(), sin.float())
    return tuple(value.to(dtype=dtype) for value, dtype in zip(values, output_dtypes))


def partition_metrics(actual, golden, rotary_dim):
    if actual.shape != golden.shape or not 0 < rotary_dim <= golden.shape[-1]:
        raise ValueError('RoPE comparison requires matching shapes and a valid rotary prefix')
    return dict(rotary_dim=rotary_dim,
        rotated=metrics(actual[..., :rotary_dim], golden[..., :rotary_dim]),
        unrotated=metrics(actual[..., rotary_dim:], golden[..., rotary_dim:]))


def prepare(model, record, inputs, capture, precision):
    """Check the real source helper on CPU; never use its output as golden."""
    import torch
    from .source_regions import definition
    if precision not in ('source', 'fp32'):
        raise ValueError('unknown RoPE precision candidate')
    tensors = dict(zip(model.names, inputs))
    if set(tensors) != {'q', 'k', 'cos', 'sin'}:
        raise ValueError('RoPE requires captured q, k, cos and sin inputs')
    q, k, cos, sin = (tensors[n] for n in ('q', 'k', 'cos', 'sin'))
    if (q.ndim != 4 or k.ndim != 4 or q.shape[0::2] != k.shape[0::2]
            or q.shape[-1] != k.shape[-1] or cos.ndim != 3 or cos.shape != sin.shape
            or cos.shape[0] not in (1, q.shape[0]) or cos.shape[1] != q.shape[2]
            or not 0 < cos.shape[-1] <= q.shape[-1] or cos.shape[-1] % 2
            or any(t.dtype not in (torch.float16, torch.float32) for t in inputs)):
        raise ValueError('RoPE requires B,H,S,D q/k and B,S,even-rotary-dim coefficients, FP16/FP32')
    if [s['name'] for s in record['outputs']] != ['golden_q_rotated', 'golden_k_rotated']:
        raise ValueError('RoPE outputs must identify captured Q then K')
    golden = [read_array(capture, spec) for spec in record['outputs']]
    model.rope_output_dtypes = tuple(torch.from_numpy(a).dtype for a in golden)
    model.rope_precision = precision
    with torch.no_grad():
        source = model.target.apply_rotary_pos_emb(q, k, cos, sin)
        candidate = apply_fp32(model.target, q, k, cos, sin, model.rope_output_dtypes)
    if len(source) != 2:
        raise ValueError('CPro RoPE helper must return Q and K')
    comparisons = []
    for spec, original, changed, expected in zip(record['outputs'], source, candidate, golden):
        if list(original.shape) != spec['shape'] or original.dtype != torch.from_numpy(expected).dtype:
            raise ValueError('CPro RoPE source output shape/dtype differs from captured boundary: ' + spec['name'])
        a, b = original.detach().numpy(), changed.detach().numpy()
        comparisons.append(dict(output=spec['name'], source_vs_golden=metrics(a, expected),
            fp32_vs_golden=metrics(b, expected), fp32_vs_source=metrics(b, a)))
    coefficient_source = {}
    owner = getattr(model.target, 'Qwen3_5MoeRotaryEmbedding', None)
    for name in ('_init_rope_cache', 'forward'):
        if owner is not None and hasattr(owner, name):
            coefficient_source[name] = definition(getattr(owner, name))[1]
    return dict(precision=precision, rotary_dim=cos.shape[-1],
        output_dtype=record['outputs'][model.output_index]['dtype'],
        coefficient_scope='CAPTURED_COS_SIN_UNCHANGED; GENERATOR_OUTSIDE_REPLAY',
        coefficient_source=coefficient_source,
        host_check=dict(scope='CPU_HELPER_DIAGNOSTIC; NOT_NPU_VALIDATION; NOT_NEW_GOLDEN',
                        outputs=comparisons))


def check_graph(graph, schema, output_dtype):
    """Check reachable arithmetic ports, including the actual producer dtype."""
    from qwen35_moe_dflash.target.operator_air_review import _data_edges, _dtype, descriptor
    enums = schema.enums[schema.package + '.DataType']
    nodes = {n['name']: n for n in graph['op']}
    nets = [n for n in nodes.values() if n['type'] == 'NetOutput']
    if len(nets) != 1 or len(_data_edges(nets[0])) != 1:
        raise ValueError('RoPE candidate must expose one output')

    def port(node, direction, slot):
        descs = node.get(direction + '_desc', [])
        if slot >= len(descs) or not descs[slot]:
            raise ValueError(f'RoPE {node["name"]} {direction} {slot} lacks a descriptor')
        # Protobuf may omit the default DT_FLOAT enum (0), even when the
        # descriptor has a name/layout. Do not treat an absent descriptor as FP32.
        raw = dict(dtype=enums['DT_FLOAT'], **descs[slot]) if 'dtype' not in descs[slot] else descs[slot]
        return _dtype(descriptor(raw, enums))

    def producer(edge):
        name, slot = edge.rsplit(':', 1)
        return nodes[name], int(slot)

    edge = _data_edges(nets[0])[0]
    node, slot = producer(edge)
    if port(node, 'output', slot) != output_dtype or port(nets[0], 'input', 0) != output_dtype:
        raise ValueError('RoPE candidate output dtype differs from golden')
    pending, seen, arithmetic = [edge], set(), []
    while pending:
        edge = pending.pop()
        if edge in seen:
            continue
        seen.add(edge)
        node, slot = producer(edge)
        edges = _data_edges(node)
        if node['type'] in ('Sin', 'Cos'):
            raise ValueError('RoPE apply replay unexpectedly contains coefficient generation')
        if node['type'] in ('Mul', 'Muls', 'Add', 'AddV2', 'Sub', 'Neg'):
            if port(node, 'output', slot) != 'float32' or not edges:
                raise ValueError('RoPE arithmetic must output FP32: ' + node['name'])
            for i, source in enumerate(edges):
                upstream, index = producer(source)
                if port(node, 'input', i) != 'float32' or port(upstream, 'output', index) != 'float32':
                    raise ValueError('RoPE arithmetic must consume FP32 on both edge ends: ' + source)
            arithmetic.append(dict(node=node['name'], type=node['type'], dtype='float32'))
        pending.extend(e for e in edges if e)
    if not any(n['type'] in ('Add', 'AddV2') for n in arithmetic) or not any(
            n['type'] in ('Mul', 'Muls') for n in arithmetic):
        raise ValueError('RoPE candidate lacks inspectable rotation multiply/add ports')
    return dict(status='PASSED', scope='SOURCE_AIR_ARITHMETIC_PORTS; NOT_COMPILED_PRECISION',
        output_dtype=output_dtype, arithmetic=arithmetic,
        coefficient_generation='OUTSIDE_CAPTURE; NOT_TESTED')
