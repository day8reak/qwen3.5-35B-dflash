"""Check explicit attention dataflow in source AIR without importing Torch."""
from collections import defaultdict

from qwen35_moe_dflash.target.original_air_roundtrip import _attributes


def inspect_composed_attention(nodes, contract, types, mismatch):
    """Require six distinct QK -> FP32 Softmax -> PV paths and no fused call.

    This checks serialized source operations, not the compiler's opaque kernels.
    Missing intermediate shapes remain unproven; recorded conflicts are errors.
    """
    by_name = {n['name']: n for n in nodes}
    users = defaultdict(list)
    for n in nodes:
        for i, edge in enumerate(n.get('input', [])):
            users[edge].append((n, i))
    matmuls = {'MatMul', 'MatMulV2', 'BatchMatMul', 'BatchMatMulV2', 'BatchMatMulV3'}
    views = {'Cast', 'CastT', 'Identity', 'Reshape', 'Transpose', 'Squeeze', 'Unsqueeze', 'ExpandDims'}
    softmaxes = [n for n in nodes if n['type'] in {'Softmax', 'SoftmaxV2'}]
    fused = [n['name'] for n in nodes if 'FusedInferAttention' in n['type']
             or n['type'] in {'FlashAttentionScore', 'PromptFlashAttention', 'IncreFlashAttention'}]
    if fused:
        mismatch('graph', 'composed_attention.fused_nodes', [], fused)
    if contract.get('layers') != 6 or len(softmaxes) != 6:
        mismatch('graph', 'composed_attention.layers', 6,
                 dict(contract=contract.get('layers'), softmax_nodes=len(softmaxes)))
    if len(by_name) != len(nodes):
        mismatch('graph', 'composed_attention.names', 'UNIQUE', 'DUPLICATE')

    def preceding(edge):
        pending, seen, found = [edge], set(), set()
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            name, _, port = current.rpartition(':')
            n = by_name.get(name)
            if n is None or port != '0' or len(seen) > 256:
                continue
            ins, kind = n.get('input', []), n['type']
            if kind in matmuls:
                found.add(name)
            elif kind in views | {'Muls'}:
                pending.extend(ins[:1])
            elif kind == 'SelectV2':
                pending.extend(ins[1:2])  # visible score, not mask or -inf
            elif kind == 'Mul':
                pending.extend(ins[:2])
        return sorted(found)

    def following(edge):
        pending, seen, found = [edge], set(), set()
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            if len(seen) > 256:
                break
            for n, slot in users[current]:
                if slot != 0:
                    continue
                if n['type'] in matmuls:
                    found.add(n['name'])
                elif n['type'] in views:
                    pending.append(n['name'] + ':0')
        return sorted(found)

    paths, qks, pvs = [], [], []
    for n in softmaxes:
        ins = n.get('input', [])
        qk = preceding(ins[0]) if len(ins) == 1 else []
        pv = following(n['name'] + ':0')
        if len(qk) != 1 or len(pv) != 1:
            mismatch(n['name'], 'composed_attention.matmul_paths', 'ONE_QK_AND_ONE_PV',
                     dict(qk=qk, pv=pv))
        qks.extend(qk)
        pvs.extend(pv)
        descriptors = n.get('input_desc', []) + n.get('output_desc', [])
        if len(descriptors) != 2 or any(d.get('dtype') != types['DT_FLOAT'] for d in descriptors):
            mismatch(n['name'], 'composed_attention.softmax_dtype', 'FP32_INPUT_AND_OUTPUT',
                     [d.get('dtype') for d in descriptors])
        shape = descriptors[0].get('shape', {}).get('dim') if descriptors else None
        axes = _attributes(n).get('axes', {}).get('list', {}).get('i')
        allowed_axes = [[-1]] + ([[len(shape)-1]] if shape else [])
        if axes not in allowed_axes:
            mismatch(n['name'], 'composed_attention.softmax_axes', allowed_axes, axes)
        # A recorded score extent must cover the complete logical KV domain.
        if shape and shape[-1] != contract['logical_kv_rows']:
            mismatch(n['name'], 'composed_attention.kv_extent', contract['logical_kv_rows'], shape[-1])
        paths.append(dict(softmax=n['name'], qk=qk, pv=pv, score_shape=shape,
                          dtype='FP32' if len(descriptors) == 2 and all(
                              d.get('dtype') == types['DT_FLOAT'] for d in descriptors) else 'UNVERIFIED',
                          axes=axes))
    if len(set(qks)) != 6 or len(set(pvs)) != 6 or set(qks) & set(pvs):
        mismatch('graph', 'composed_attention.distinct_matmuls', '6_QK_AND_6_PV',
                 dict(qk=sorted(set(qks)), pv=sorted(set(pvs))))
    return paths
