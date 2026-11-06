"""Lossless FP16 static-right-operand carriers with explicit transpose flags.

Storage [K1,N1,N0,K0] is also ZN(B[K,N]) for B=W.T. This module does
not assert that a selected OMG FileConstant importer supports that carrier;
The paired device canary must establish the import contract first.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import stat

import numpy as np

from qwen35_moe_dflash.common.hashing import sha256_file
from qwen35_moe_dflash.target.original_air_roundtrip import _attributes, _decode
from qwen35_moe_dflash.target.original_ge_append_wire import _blob, _field, _graph_raw, _node_bytes, _parts, _v

POLICY = 'fp16-static-rhs-layout-v1'
VARIANTS = ('nd_t', 'nd_n', 'nz_t', 'nz_n', 'zn_t', 'zn_n')


def contract(variant, *, legacy=False):
    if variant not in VARIANTS:
        raise ValueError('unknown offline weight variant: '+str(variant))
    layout, transpose = variant.split('_')
    transposed = transpose == 't'
    result = dict(policy=POLICY, variant=variant,
        storage=dict(nd='ND', nz='FRACTAL_NZ', zn='FRACTAL_ZN')[layout],
        logical_axes=['N', 'K'] if transposed else ['K', 'N'],
        physical_axes=(['N', 'K'] if transposed else ['K', 'N']) if layout == 'nd' else
            (['K1', 'N1', 'N0', 'K0'] if (layout == 'nz') == transposed else ['N1', 'K1', 'K0', 'N0']),
        block=[16, 16], transpose_x2=transposed, arithmetic='NONE_UINT16_PERMUTATION')
    if layout == 'nz' and not legacy:
        result.update(policy='fp16-static-rhs-layout-v2',
                      descriptor='LOGICAL_ND_WITH_STORAGE_SHAPE_AND_FORMAT',
                      fileconstant_shape='LOGICAL_2D')
    return result


CONTRACT = contract('nz_t')
# GE format integers are separate from TensorDescriptor.DataType. This is the
# explicit candidate ABI tested by offline_weight_probe, not inferred support.
NZ_FORMAT = 29


def physical_shape(shape):
    if (not isinstance(shape, (list, tuple)) or len(shape) != 2
            or any(type(d) is not int or d <= 0 or d % 16 for d in shape)):
        raise ValueError('offline FP16 carrier requires positive N,K divisible by 16')
    n, k = shape
    return [k // 16, n // 16, 16, 16]


def pack_words(words):
    """No floating point conversion: preserve signed zero, NaN payloads, etc."""
    if words.dtype != np.dtype('<u2') or words.ndim != 2:
        raise ValueError('expected a rank-two little-endian uint16 view')
    k1, n1, _, _ = physical_shape(list(words.shape))
    return np.ascontiguousarray(words.reshape(n1, 16, k1, 16).transpose(2, 0, 1, 3))


def carrier_shape(shape, variant):
    physical_shape(shape)
    c = contract(variant)
    right = list(shape if c['transpose_x2'] else shape[::-1])
    if c['storage'] == 'ND':
        return right
    r, k = right
    return [k//16, r//16, 16, 16] if c['storage'] == 'FRACTAL_NZ' else [r//16, k//16, 16, 16]


def carrier_words(words, variant):
    if words.dtype != np.dtype('<u2') or words.ndim != 2:
        raise ValueError('expected a rank-two little-endian uint16 view')
    physical_shape(list(words.shape))
    c = contract(variant)
    right = words if c['transpose_x2'] else words.T
    if c['storage'] == 'ND':
        return np.ascontiguousarray(right)
    r, k = right.shape
    axes = (2, 0, 1, 3) if c['storage'] == 'FRACTAL_NZ' else (0, 2, 3, 1)
    return np.ascontiguousarray(right.reshape(r//16, 16, k//16, 16).transpose(axes))


def _file(root, relative):
    p = Path(relative)
    if p.is_absolute() or '..' in p.parts or not p.parts or p.parts[0] != 'weight':
        raise ValueError('expected a relative weight/ payload path')
    path = Path(root) / p
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError('offline weight paths must not contain symlinks')
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError('offline weight must be a regular file')
    return path


def inventory(raw, schema, *, draft=True, variant='nz_t'):
    """Fail closed on partial coverage, aliases, control/public/non-MatMul uses."""
    selected_contract = contract(variant)
    _, _, graph = _decode(schema, raw)
    nodes = graph['op']
    by_name = {n['name']: n for n in nodes}
    if len(nodes) != len(by_name):
        raise ValueError('duplicate graph operation names')
    users, aliases = defaultdict(list), defaultdict(list)
    for node in nodes:
        if (any(node.get(key) for key in ('src_name', 'src_index', 'dst_name', 'dst_index', 'subgraph_name'))
                or any('g' in a or a.get('list', {}).get('g') for a in _attributes(node).values())):
            raise ValueError('offline weights require uncompiled flat source AIR')
        for index, edge in enumerate(node.get('input', [])):
            if edge:
                users[edge.rpartition(':')[0]].append((node, index, edge))
        if node['type'] == 'FileConstant':
            aliases[_attributes(node).get('file_path', {}).get('s')].append(node['name'])
    public = {e.rpartition(':')[0] if ':' in e else e
              for e in (*graph.get('input', []), *graph.get('output', []))}
    half = schema.enums[schema.package + '.DataType']['DT_FLOAT16']
    selected = {}
    for mm in nodes:
        if mm['type'] != 'MatMul':
            continue
        edges, descs = mm.get('input', []), mm.get('input_desc', [])
        attrs = _attributes(mm)
        if (len(edges) not in (2, 3) or len(descs) != len(edges) or any(edges[2:])
                or attrs.get('transpose_x1', {}).get('b') is not False
                or attrs.get('transpose_x2', {}).get('b') is not True):
            raise ValueError('MatMul requires folded FP16 right weight and explicit false/true transpose flags: '+mm['name'])
        if len(edges) == 3:
            optional = _attributes(descs[2]).get('_is_unfed_optional', {}).get('i')
            if optional != 1 or descs[2].get('dtype', 0) != 0:
                raise ValueError('unexpected MatMul optional bias descriptor')
        name, _, port = edges[1].rpartition(':')
        weight = by_name.get(name, {})
        wa = _attributes(weight)
        shape = wa.get('shape', {}).get('list', {}).get('i')
        if (port != '0' or weight.get('type') != 'FileConstant' or weight.get('input')
                or wa.get('dtype', {}).get('i') != 1 or len(weight.get('output_desc', [])) != 1):
            raise ValueError('MatMul right input is not an original FP16 FileConstant: '+mm['name'])
        physical_shape(shape)
        for d in (*descs[:2], *mm.get('output_desc', [])):
            if d.get('dtype') != half or d.get('layout', 'ND') != 'ND':
                raise ValueError('MatMul consumer descriptors must be FP16 ND')
        if len(mm.get('output_desc', [])) != 1:
            raise ValueError('MatMul requires one output')
        if descs[1].get('shape', {}).get('dim') not in (None, [], shape):
            raise ValueError('MatMul logical right shape conflicts with FileConstant')
        path = wa.get('file_path', {}).get('s')
        if not isinstance(path, str) or Path(path).parent != Path('weight'):
            raise ValueError('expected a direct weight/ FileConstant')
        if len(aliases[path]) != 1 or name in public:
            raise ValueError('shared/public weight needs an explicit multi-consumer layout contract: '+str(path))
        if any(c['type'] != 'MatMul' or i != 1 or e != name+':0' for c, i, e in users[name]):
            raise ValueError('weight has a non-right-MatMul consumer: '+path)
        selected.setdefault(name, dict(node=name, source=path, shape=shape,
            physical_shape=carrier_shape(shape, variant), destination='weight/offline_'+variant+'_'+Path(path).name,
            bytes=math.prod(shape)*2, consumers=[]))['consumers'].append(mm['name'])
    rows = list(selected.values())
    if not rows:
        raise ValueError('no constant FP16 MatMul weights found')
    if draft:
        expected = Counter({(4096, 2048): 6, (2048, 2048): 6, (2048, 4096): 6,
                            (12288, 2048): 6, (2048, 6144): 6, (248320, 2048): 1})
        if Counter(tuple(r['shape']) for r in rows) != expected or sum(len(r['consumers']) for r in rows) != 31:
            raise ValueError('offline Draft requires all 31 Q/KV/O/gate-up/down/head weights; no partial rewrite')
    return dict(contract=selected_contract, source_sha256=hashlib.sha256(raw).hexdigest(),
                weight_count=len(rows), matmul_count=sum(len(r['consumers']) for r in rows),
                bytes=sum(r['bytes'] for r in rows), weights=rows)


def _scalar(schema, msg, field, value, kind):
    return _v(_field(schema, msg, field, kind, False) << 3) + _v(int(value))


def _attr(schema, message, key, value):
    """Encode the closed subset of AttrDef needed for the import contract."""
    if type(value) is bool:
        raw = _scalar(schema, 'AttrDef', 'b', value, 8)
    elif type(value) is int:
        raw = _scalar(schema, 'AttrDef', 'i', value, 3)
    elif isinstance(value, str):
        raw = _blob(_field(schema, 'AttrDef', 's', 12, False), value)
    else:
        raw = (_blob(_field(schema, 'AttrDef.ListValue', 'i', 3, True), b''.join(_v(d) for d in value))
               + _scalar(schema, 'AttrDef.ListValue', 'val_type', 2, 14))
        # val_type is enum VT_LIST_INT=2 in the source GE AttrDef.
        raw = _blob(_field(schema, 'AttrDef', 'list', 11, False), raw)
    entry = message+'.AttrEntry'
    return _blob(_field(schema, message, 'attr', 11, True),
                 _blob(_field(schema, entry, 'key', 9, False), key)
                 + _blob(_field(schema, entry, 'value', 11, False), raw))


def _shape(schema, shape):
    return _blob(_field(schema, 'TensorDescriptor', 'shape', 11, False),
                 _blob(_field(schema, 'ShapeDef', 'dim', 3, True), b''.join(_v(d) for d in shape)))


def descriptor(schema, shape, *, variant=None):
    c = contract(variant) if variant else None
    logical = list(shape if not c or c['transpose_x2'] else shape[::-1])
    attrs = dict(origin_shape=logical, origin_shape_initialized=True, origin_format='ND', origin_format_for_int=2)
    layout = c['storage'] if c else 'ND'
    dimensions = carrier_shape(shape, variant) if c else shape
    if c and c.get('descriptor') == 'LOGICAL_ND_WITH_STORAGE_SHAPE_AND_FORMAT':
        # The v1 physical rank-four shape was treated as BatchMatMul batches
        # by the selected HIAI importer. Keep graph algebra rank two. Storage
        # attributes are a separately versioned import candidate, not proof
        # that this DDK honors them for FileConstant / MatMul.
        attrs.update(storage_shape=dimensions, storage_format=NZ_FORMAT)
        dimensions, layout = logical, 'ND'
    if layout != 'FRACTAL_ZN':
        attrs['format_for_int'] = NZ_FORMAT if layout == 'FRACTAL_NZ' else 2
    # ZN is an explicit string-import experiment. No unverified integer enum
    # is invented or confused with FRACTAL_Z / FRACTAL_ZN_RNN.
    return (_scalar(schema, 'TensorDescriptor', 'dtype', schema.enums[schema.package+'.DataType']['DT_FLOAT16'], 14)
            + _shape(schema, dimensions)
            + _blob(_field(schema, 'TensorDescriptor', 'layout', 9, False), layout)
            + _blob(_field(schema, 'TensorDescriptor', 'device_type', 9, False), 'NPU')
            + b''.join(_attr(schema, 'TensorDescriptor', k, v) for k, v in attrs.items()))


def rewrite(raw, schema, plan):
    selected_contract = contract(plan['contract']['variant'])
    if hashlib.sha256(raw).hexdigest() != plan['source_sha256'] or plan['contract'] != selected_contract:
        raise ValueError('offline weight inventory differs from the supplied source AIR')
    kind, original, _ = _decode(schema, raw)
    graph_raw, graph_field = _graph_raw(raw, schema, kind)
    rows, op_field = _node_bytes(graph_raw, schema)
    replacements, expected_nodes = {}, {n['name']: deepcopy(n) for n, _, _ in rows}
    weights = {r['node']: r for r in plan['weights']}
    consumers = {c: r for r in plan['weights'] for c in r['consumers']}
    attr_f = _field(schema, 'OpDef', 'attr', 11, True)
    in_f = _field(schema, 'OpDef', 'input_desc', 11, True)
    out_f = _field(schema, 'OpDef', 'output_desc', 11, True)
    for node, payload, _ in rows:
        name = node['name']
        if name not in weights and name not in consumers:
            continue
        row = weights.get(name, consumers.get(name))
        desc = descriptor(schema, row['shape'], variant=selected_contract['variant'])
        parts, index = [], 0
        updated = expected_nodes[name]
        for f, _, encoded, p in _parts(payload):
            if name in weights and f == attr_f:
                key = schema.decode(p, 'OpDef.AttrEntry')['key']
                if key in ('shape', 'file_path'):
                    dimensions = row['physical_shape']
                    if selected_contract.get('fileconstant_shape') == 'LOGICAL_2D':
                        dimensions = row['shape'] if selected_contract['transpose_x2'] else row['shape'][::-1]
                    value = dimensions if key == 'shape' else row['destination']
                    encoded = _attr(schema, 'OpDef', key, value)
                    for a in updated['attr']:
                        if a['key'] == key:
                            a['value'] = {'list': {'i': value, 'val_type': 2}} if key == 'shape' else {'s': value}
            if name in weights and f == out_f:
                encoded = _blob(f, desc)
                updated['output_desc'] = [schema.decode(desc, 'TensorDescriptor')]
            if name in consumers and f == in_f:
                if index == 1:
                    encoded = _blob(f, desc)
                    updated['input_desc'][1] = schema.decode(desc, 'TensorDescriptor')
                index += 1
            if name in consumers and f == attr_f and schema.decode(p, 'OpDef.AttrEntry')['key'] == 'transpose_x2':
                encoded = _attr(schema, 'OpDef', 'transpose_x2', selected_contract['transpose_x2'])
                for a in updated['attr']:
                    if a['key'] == 'transpose_x2':
                        a['value'] = {'b': selected_contract['transpose_x2']}
            parts.append(encoded)
        replacements[name] = b''.join(parts)
    rebuilt = b''.join(_blob(f, replacements[schema.decode(p, 'OpDef')['name']])
                      if f == op_field and schema.decode(p, 'OpDef')['name'] in replacements else e
                      for f, _, e, p in _parts(graph_raw))
    output = rebuilt if graph_field is None else b''.join(_blob(f, rebuilt) if f == graph_field else e
                                                         for f, _, e, _ in _parts(raw))
    expected = deepcopy(original)
    graph = expected if kind == 'GraphDef' else expected['graph'][0]
    graph['op'] = [expected_nodes[n['name']] for n, _, _ in rows]
    if _decode(schema, output)[1] != expected:
        raise ValueError('offline weight rewrite changed unrelated graph metadata')
    return output


def pack_files(root, plan, *, progress=lambda s: None):
    """Bounded row chunks; no full-size in-memory transpose or FP arithmetic."""
    root, records = Path(root), []
    c = contract(plan['contract']['variant'])
    if plan['contract'] != c:
        raise ValueError('offline weight inventory uses a different import contract')
    for row in plan['weights']:
        source = _file(root, row['source'])
        if source.stat().st_size != row['bytes']:
            raise ValueError('weight file size differs: '+row['source'])
        destination = root / row['destination']
        source_hash = sha256_file(source)
        progress('Offline '+c['variant']+' packing: '+row['source'])
        # Exclusive creation also refuses symlinks and prevents stale reuse.
        with destination.open('xb') as stream:
            stream.truncate(row['bytes'])
        src = np.memmap(source, dtype='<u2', mode='r', shape=tuple(row['shape']))
        dst = np.memmap(destination, dtype='<u2', mode='r+', shape=tuple(row['physical_shape']))
        n, k = row['shape']
        chunk = max(16, (8 << 20) // (k*2*16) * 16)
        try:
            for start in range(0, n, chunk):
                stop = min(start+chunk, n)
                block = carrier_words(src[start:stop], c['variant'])
                if c['storage'] == 'ND':
                    if c['transpose_x2']:
                        dst[start:stop, :] = block
                    else:
                        dst[:, start:stop] = block
                elif c['physical_axes'][0] == 'K1':
                    dst[:, start//16:stop//16, :, :] = block
                else:
                    dst[start//16:stop//16, :, :, :] = block
            dst.flush()
        finally:
            del src, dst
        restored = _restored_hash(destination, row, c['variant'])
        if restored != source_hash or sha256_file(source) != source_hash:
            raise ValueError('offline weight byte roundtrip failed: '+row['source'])
        destination.chmod(0o444)
        records.append(dict(row, source_sha256=source_hash, sha256=sha256_file(destination),
                            restored_sha256=restored))
    return dict(status='PACKED_AND_ROUNDTRIP_VERIFIED', contract=c, weights=records,
                bytes=plan['bytes'], weight_count=len(records),
                source_sha256=plan['source_sha256'], model_compilation='NOT_PROVEN_BY_PACKING',
                device_import='REQUIRES_DEVICE_CANARY_AND_FULL_DRAFT_VALIDATION')


def _restored_hash(path, row, variant='nz_t'):
    n, k = row['shape']
    c = contract(variant)
    data = np.memmap(path, dtype='<u2', mode='r', shape=tuple(carrier_shape(row['shape'], variant)))
    digest = hashlib.sha256()
    try:
        for start in range(0, n, 16):
            # Independent inverse, in original row-major byte order.
            for inner in range(16):
                if c['storage'] == 'ND':
                    words = data[start+inner, :] if c['transpose_x2'] else data[:, start+inner]
                elif c['physical_axes'][0] == 'K1':
                    words = data[:, start//16, inner, :]
                else:
                    words = data[start//16, :, :, inner]
                digest.update(np.ascontiguousarray(words).tobytes())
    finally:
        del data
    return digest.hexdigest()


def verify_binding(root, raw, schema, evidence):
    """Check packed payloads and reconstruct the exact pre-layout graph hash."""
    variant = evidence.get('contract', {}).get('variant')
    if evidence.get('status') != 'PACKED_AND_ROUNDTRIP_VERIFIED' or evidence.get('contract') != contract(variant):
        raise ValueError('missing offline weight byte proof')
    before = _file_air(root, evidence['before_air'])
    if hashlib.sha256(before).hexdigest() != evidence['source_sha256']:
        raise ValueError('offline pre-layout AIR hash differs')
    plan = inventory(before, schema, draft=evidence.get('scope') == 'FULL_DRAFT_31_STATIC_MATMULS', variant=variant)
    if rewrite(before, schema, plan) != raw:
        raise ValueError('compiled AIR is not the verified offline weight rewrite')
    records = evidence['weights']
    if len(records) != len(plan['weights']):
        raise ValueError('offline weight coverage differs')
    for row, expected in zip(records, plan['weights']):
        if any(row[k] != v for k, v in expected.items()):
            raise ValueError('offline weight inventory changed')
        source, packed = _file(root, row['source']), _file(root, row['destination'])
        if (source.stat().st_size != row['bytes'] or packed.stat().st_size != row['bytes']
                or sha256_file(source) != row['source_sha256'] or sha256_file(packed) != row['sha256']
                or _restored_hash(packed, row, variant) != row['source_sha256']):
            raise ValueError('offline weight payload binding failed: '+row['source'])
    return dict(status='VERIFIED', weight_count=len(records), bytes=plan['bytes'], contract=plan['contract'])


def _file_air(root, relative):
    if relative != 'before-offline-weights.air':
        raise ValueError('unexpected offline pre-layout AIR path')
    path = Path(root)/relative
    if path.is_symlink() or path.stat().st_size > 64 << 20:
        raise ValueError('invalid offline pre-layout AIR')
    return path.read_bytes()
