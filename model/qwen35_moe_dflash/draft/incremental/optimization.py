"""Versioned graph candidates and byte-exact packed-weight provenance."""
import hashlib
import math

from qwen35_moe_dflash.common.artifacts import checked_file

POLICIES = ('baseline', 'packed', 'packed-spatial', 'packed-spatial-4d',
            'packed-spatial-controls', 'packed-spatial-rope-table')


def optimization_contract(policy='baseline'):
    if policy not in POLICIES:
        raise ValueError('unknown Draft optimization: ' + str(policy))
    spatial = policy in ('packed-spatial', 'packed-spatial-4d', 'packed-spatial-controls', 'packed-spatial-rope-table')
    contract = dict(policy=policy, version=1, packed_projections=policy != 'baseline',
                    grouped_query_reshape=policy != 'baseline', shared_rope=policy != 'baseline',
                    projection=('PYTORCH_CONV2D_SPATIAL_4X4_FULL_K' if spatial
                                else 'PYTORCH_CONV2D_4X4_FULL_K'))
    if policy in ('packed-spatial-4d', 'packed-spatial-controls', 'packed-spatial-rope-table'):
        contract.update(feature_pack='TWO_RANK4_TRANSPOSES', shared_visibility_masks=True)
    if policy in ('packed-spatial-controls', 'packed-spatial-rope-table'):
        contract.update(key_valid_concat='INT32_BEFORE_QUERY_GATHER')
    if policy == 'packed-spatial-controls':
        contract.update(position_layout='INT32_UNSQUEEZE_BEFORE_FP32_CAST')
    if policy == 'packed-spatial-rope-table':
        contract.update(rope_phase='PRECOMPUTED_FP32_POSITION_TIMES_INV_FREQ',
                        rope_phase_rows='CACHE_CAPACITY_PLUS_MAX_CONTEXT_BLOCK_ROWS',
                        rope_lookup='INT32_GATHER_BEFORE_EXISTING_SIN_COS')
    return contract


def manifest_policy(manifest):
    contract = manifest.get('optimization', optimization_contract())
    expected = optimization_contract(contract['policy'])
    if contract != expected:
        raise ValueError('unsupported Draft optimization contract')
    return contract['policy']


def weight_groups(weights, policy):
    """Derive packing from original parameter names, never an untrusted map."""
    optimization_contract(policy)
    groups = {}
    for name, record in weights.items():
        parameter = {'target.embed_tokens.weight': 'embedding.weight',
                     'target.lm_head.weight': 'head.weight'}.get(name, name)
        rank = 0
        if policy != 'baseline':
            for before, after, order in (
                    ('.self_attn.k_proj.weight', '.self_attn.kv_proj.weight', 0),
                    ('.self_attn.v_proj.weight', '.self_attn.kv_proj.weight', 1),
                    ('.mlp.gate_proj.weight', '.mlp.gate_up_proj.weight', 0),
                    ('.mlp.up_proj.weight', '.mlp.gate_up_proj.weight', 1)):
                if parameter.endswith(before):
                    parameter, rank = parameter[:-len(before)] + after, order
                    break
        relative = 'weight/' + parameter.replace('.', '_')
        groups.setdefault(relative, []).append((rank, name, record))
    return {p: sorted(rows) for p, rows in groups.items()}


def bind_weight_snapshots(weights, policy, file_constants, snapshots, compilation):
    """Verify complete snapshots and each original weight segment in row order."""
    result = []
    for relative, rows in weight_groups(weights, policy).items():
        packed = relative.endswith(('_kv_proj_weight', '_gate_up_proj_weight'))
        if packed and ([r[0] for r in rows] != [0, 1]
                       or len(rows[0][2]['shape']) != 2
                       or rows[0][2]['shape'] != rows[1][2]['shape']):
            raise ValueError('invalid packed projection pair: ' + relative)
        snap = snapshots.get(relative)
        if (file_constants.count(relative) != 1 or snap is None
                or snap['bytes'] != sum(r['bytes'] for _, _, r in rows)):
            raise ValueError('reference weight extent differs from compiler snapshot: ' + relative)
        if not packed:
            _, name, record = rows[0]
            if len(rows) != 1 or snap['sha256'] != record['sha256']:
                raise ValueError('reference weight differs from compiled PyTorch source: ' + name)
            result.append(dict(name=name, source_relative=relative, sha256=record['sha256']))
            continue
        path = checked_file(compilation, snap)
        offset = 0
        with path.open('rb') as stream:
            for _, name, record in rows:
                size = record['bytes']
                if record['dtype'] != 'float16' or size != math.prod(record['shape']) * 2:
                    raise ValueError('invalid packed weight descriptor: ' + name)
                remaining, digest = size, hashlib.sha256()
                while remaining:
                    block = stream.read(min(remaining, 8 << 20))
                    if not block:
                        raise ValueError('truncated packed weight: ' + name)
                    digest.update(block)
                    remaining -= len(block)
                if digest.hexdigest() != record['sha256']:
                    raise ValueError('packed segment differs from original weight: ' + name)
                result.append(dict(name=name, source_relative=relative, offset=offset,
                                   bytes=size, sha256=record['sha256']))
                offset += size
    return result
