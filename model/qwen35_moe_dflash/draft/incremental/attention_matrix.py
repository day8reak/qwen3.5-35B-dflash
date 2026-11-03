"""Bounded Attention diagnosis across data, masks, lengths, pages and geometry.

These are synthetic interface experiments, not replacements for either model.
Use the audited source node and selected schema; never guess a different ABI.
"""
import fcntl
import hashlib
from pathlib import Path
import re

import numpy as np

from qwen35_moe_dflash.common.artifacts import output_path
from qwen35_moe_dflash.execution.runtime_config import read_json
from qwen35_moe_dflash.target.original_air_roundtrip import _attributes, _decode
from qwen35_moe_dflash.target.original_ge_append_wire import _blob, _field, _graph_raw, _node_bytes, _parts, _v, _new_id
from .torch_export import _save, _disjoint

KIND = 'qwen35-attention-diagnostic-matrix'
SCOPE = 'SYNTHETIC_ATTENTION_INTERFACE_MATRIX; NOT_FULL_DRAFT_OR_TARGET'
BOUNDARIES = (0, 15, 16, 31, 32, 63, 64, 127, 191, 192, 207, 208, 223)
LENGTHS = (1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 192, 208)
BASE_CASES = ('uniform', 'weighted', 'query_changed', 'key_changed', 'mask_rows',
              'mask_rows_inf', 'mask_bias', 'mask_prefix_causal', 'mask_temporary_causal',
              'zero_v', *(f'impulse_{r}' for r in BOUNDARIES), *(f'key_{r}' for r in BOUNDARIES))
SMALL_CASES = ('uniform', 'weighted', 'mask_rows', 'mask_temporary_causal',
               'impulse_16', 'impulse_32', 'impulse_192', 'key_16', 'key_32', 'key_192')
VARIANTS = {
    'constant': dict(rows=16, heads=32, kv_heads=8, dim=128, runtime=False, observed=False, cases=BASE_CASES),
    'runtime': dict(rows=16, heads=32, kv_heads=8, dim=128, runtime=True, observed=False,
                    cases=(*BASE_CASES, *(f'length_{n}' for n in LENGTHS), 'table_rotate', 'table_reverse', 'page_equivalent')),
    'constant_mask': dict(rows=16, heads=32, kv_heads=8, dim=128, runtime=False, observed=False,
                          cases=('mask_rows',), constant_mask=True),
    'q32': dict(rows=32, heads=32, kv_heads=8, dim=128, runtime=False, observed=False, cases=SMALL_CASES),
    'target_prefill': dict(rows=64, heads=16, kv_heads=2, dim=256, runtime=False, observed=False, cases=SMALL_CASES),
    'target_decode': dict(rows=1, heads=16, kv_heads=2, dim=256, runtime=False, observed=False, cases=SMALL_CASES),
    # Last: a readback-specific compile/runtime failure must not hide the core tests.
    'observed': dict(rows=16, heads=32, kv_heads=8, dim=128, runtime=True, observed=True,
                     cases=('uniform', 'mask_rows', 'length_32', 'table_rotate')),
}


def names(variant):
    return tuple('attention_matrix_'+variant+'__'+c for c in VARIANTS[variant]['cases'])


CASES = tuple(c for v in VARIANTS for c in names(v))


def shapes(s):
    return dict(q=[1, s['heads']*s['dim']//16, s['rows'], 16],
                k=[7, s['kv_heads']*s['dim']//16, 32, 16],
                v=[7, s['kv_heads']*s['dim']//16, 32, 16], mask=[1, 1, s['rows'], 224],
                q_length=[1], kv_length=[1], all_q_length=[1], table=[1, 7])


def pack_q(a):
    h, r, d = a.shape
    return a.reshape(h, r, d//16, 16).transpose(0, 2, 1, 3).copy().reshape(1, h*d//16, r, 16)


def unpack_q(a, s):
    return a.reshape(s['heads'], s['dim']//16, s['rows'], 16).transpose(0, 2, 1, 3).copy().reshape(s['heads'], s['rows'], s['dim'])


def pack_kv(a):
    h, _, d = a.shape
    return a.reshape(h, 7, 32, d//16, 16).transpose(1, 0, 3, 2, 4).copy().reshape(7, h*d//16, 32, 16)


def logical_case(variant, name):
    s = VARIANTS[variant]
    if name not in names(variant):
        raise ValueError('unknown matrix fixture')
    case = name.split('__', 1)[1]
    h, kh, r, d = (s[k] for k in ('heads', 'kv_heads', 'rows', 'dim'))
    q = np.zeros((h, r, d), dtype='<f2')
    k = np.zeros((kh, 224, d), dtype='<f2')
    hi, ki, di = np.ogrid[:kh, :224, :d]
    v = (((hi*29+ki*13+di*7) % 31-15)/2).astype('<f2')
    # Binary row/head markers distinguish boundaries without near-equal prefix means.
    for bit in range(8):
        v[:, :, bit] = np.where((np.arange(224) >> bit) & 1, 8, -8)
    for bit in range(3):
        v[:, :, 8+bit] = np.where((np.arange(kh)[:, None] >> bit) & 1, 8, -8)
    v[:, :, 11] = np.arange(224)/16
    v[:, :, 12] = np.arange(kh)[:, None]
    if case in ('weighted', 'query_changed', 'key_changed', 'mask_bias'):
        hh, rr, dd = np.ogrid[:h, :r, :d]
        q[:] = ((hh*3+rr*5+dd*7) % 11-5)/4
        k[:] = ((hi*5+ki*3+di*7) % 13-6)/4
        if case == 'query_changed':
            q[:] = -q
        if case == 'key_changed':
            k[:, 32:64] *= -1
    mask = np.zeros((r, 224), dtype='<f2')
    table = np.arange(7, dtype='<i4')
    length = 224
    if case.startswith('key_') and case != 'key_changed':
        mask.fill(-10000); mask[:, int(case[4:])] = 0
    elif case in ('mask_rows', 'mask_rows_inf'):
        mask.fill(-np.inf if case.endswith('_inf') else -10000)
        mask[np.arange(r), np.resize(np.array(BOUNDARIES), r)] = 0
    elif case == 'mask_bias':
        mask[:] = -((np.arange(r)[:, None]*5+np.arange(224)[None, :]) % 17)/2
    elif case == 'mask_prefix_causal':
        mask[np.arange(224)[None, :] > np.arange(r)[:, None]] = -10000
    elif case == 'mask_temporary_causal':
        mask.fill(-10000)
        for i in range(r):
            mask[i, 192:193+min(i, 15)] = 0
    if case.startswith('impulse_') or case == 'zero_v':
        v.fill(0)
        if case.startswith('impulse_'):
            v[:, int(case[8:]), :] = 8
    if case.startswith('length_'):
        length = int(case[7:])
    if case in ('table_rotate', 'table_reverse', 'page_equivalent'):
        # All-page uniform means are permutation invariant. Select known logical
        # rows so that changing the table has a measurable reference response.
        mask.fill(-10000)
        mask[np.arange(r), np.resize(np.array(BOUNDARIES), r)] = 0
        table = (np.roll(table, -1) if case != 'table_reverse' else table[::-1]).copy()
        if case == 'page_equivalent':
            # Compensated storage permutation: logical K/V must stay unchanged.
            k = k.reshape(kh, 7, 32, d)[:, np.argsort(table)].reshape(kh, 224, d).copy()
            v = v.reshape(kh, 7, 32, d)[:, np.argsort(table)].reshape(kh, 224, d).copy()
    return q, k, v, mask, length, table


def sdpa(q, k, v, mask, length, table, *, group='contiguous'):
    h, r, d = q.shape; kh = k.shape[0]
    kv_indices = (table[:, None]*32+np.arange(32)[None, :]).reshape(-1)[:length]
    heads = np.arange(h)//(h//kh) if group == 'contiguous' else np.arange(h) % kh
    kk, vv = k[:, kv_indices].astype(np.float64), v[:, kv_indices].astype(np.float64)
    score = np.matmul(q.astype(np.float64), kk[heads].transpose(0, 2, 1))/np.sqrt(d)
    score += mask[None, :, :length].astype(np.float64)
    if not np.isfinite(score.max(axis=-1)).all():
        raise ValueError('matrix reference must have a visible key in every query')
    prob = np.exp(score-score.max(axis=-1, keepdims=True)); prob /= prob.sum(axis=-1, keepdims=True)
    return np.matmul(prob, vv[heads]).astype('<f2')


def values(variant, name):
    s = VARIANTS[variant]
    q, k, v, mask, length, table = logical_case(variant, name)
    inputs = dict(q=pack_q(q), k=pack_kv(k), v=pack_kv(v), mask=mask.reshape(shapes(s)['mask']),
                  q_length=np.array([s['rows']], dtype='<i8'), kv_length=np.array([length], dtype='<i8'),
                  all_q_length=np.array([s['rows']], dtype='<i8'), table=table.reshape(1, 7))
    return inputs, pack_q(sdpa(q, k, v, mask, length, table))


def graph(raw, schema, receipt, variant):
    """Build one isolated source-node clone; serialize every change in the receipt."""
    s = VARIANTS[variant]; dims = shapes(s)
    kind, _, _ = _decode(schema, raw)
    graph_raw, graph_field = _graph_raw(raw, schema, kind)
    rows, op_field = _node_bytes(graph_raw, schema)
    parent, payload, _ = next(row for row in rows if row[0]['type'] == 'AdnFusedInferAttention')
    types = schema.enums[schema.package+'.DataType']
    def blob(msg, key, value, typ=9, repeated=False):
        return _blob(_field(schema, msg, key, typ, repeated), value)
    def integer(msg, key, value, typ=3):
        return _v(_field(schema, msg, key, typ, False)<<3)+_v(value)
    def desc(dtype, shape):
        return (integer('TensorDescriptor', 'dtype', types['DT_'+dtype.upper()], 14)
                +blob('TensorDescriptor', 'shape', blob('ShapeDef', 'dim', b''.join(_v(d) for d in shape), 3, True), 11)
                +blob('TensorDescriptor', 'layout', 'ND')+blob('TensorDescriptor', 'device_type', 'NPU'))
    def attr(key, value):
        return blob('OpDef', 'attr', blob('OpDef.AttrEntry', 'key', key)+blob('OpDef.AttrEntry', 'value', value, 11), 11, True)
    nodes = []; identifier = max(n.get('id', 0) for n, _, _ in rows)+1
    def node(name, typ, ins, outs, extra=b''):
        nonlocal identifier
        p = blob('OpDef', 'name', name)+blob('OpDef', 'type', typ)
        for edge, d in ins:
            p += blob('OpDef', 'input', edge, repeated=True)+blob('OpDef', 'input_desc', d, 11, True)
        for d in outs:
            p += blob('OpDef', 'output_desc', d, 11, True)
        nodes.append(_new_id(p+extra, schema, identifier)); identifier += 1
    def const(name, array, dtype):
        d = desc(dtype, list(array.shape))
        tensor = blob('TensorDef', 'desc', d, 11)+blob('TensorDef', 'data', array.tobytes(), 12)
        node(name, 'Const', [], [d], attr('value', blob('AttrDef', 't', tensor, 11)))
    default, _ = values(variant, names(variant)[0])
    ports = {'q':0, 'k':1, 'v':2, 'mask':4, 'q_length':5, 'kv_length':6, 'table':14, 'all_q_length':16}
    inputs = []; descriptors = {}; edges = {}; dtypes = {}
    for role in ('q', 'k', 'v', 'mask', 'q_length', 'kv_length', 'all_q_length', 'table'):
        name = 'matrix_'+role
        dtype = 'int64' if role.endswith('length') else 'int32' if role == 'table' else 'float16'
        dtypes[role] = dtype; descriptors[role] = desc(dtype, dims[role]); edges[ports[role]] = name+':0'
        dynamic = role in ('q', 'k', 'v') or (role == 'mask' and not s.get('constant_mask')) or s['runtime']
        if dynamic:
            node(name, 'Data', [('', descriptors[role])], [descriptors[role]], attr('index', integer('AttrDef', 'i', len(inputs))))
            inputs.append(dict(name=name, role=role, dtype=dtype, shape=dims[role]))
        else:
            const(name, default[role], dtype)
    fields = {k:_field(schema, 'OpDef', k, t, rep) for k,t,rep in
              (('input',9,True), ('input_desc',11,True), ('output_desc',11,True), ('attr',11,True))}
    input_index = desc_index = 0; parts = []
    attr_changes = dict(num_heads=s['heads'], num_key_value_heads=s['kv_heads'], scale_value=s['dim']**-0.5)
    role_by_port = {p:r for r,p in ports.items()}
    import struct
    for f, _, e, p in _parts(payload):
        if f == fields['input']:
            if input_index in edges: e = _blob(f, edges[input_index])
            input_index += 1
        elif f == fields['input_desc']:
            if desc_index in role_by_port: e = _blob(f, descriptors[role_by_port[desc_index]])
            desc_index += 1
        elif f == fields['output_desc']:
            e = _blob(f, descriptors['q'])
        elif f == fields['attr']:
            key = schema.decode(p, 'OpDef.AttrEntry')['key']
            if key in attr_changes:
                v = attr_changes[key]
                value = (integer('AttrDef', 'i', v) if isinstance(v, int) else
                         _v(_field(schema, 'AttrDef', 'f', 2, False)<<3 | 5)+struct.pack('<f', v))
                e = attr(key, value)
        parts.append(e)
    clone = _new_id(b''.join(parts), schema, identifier); identifier += 1
    decoded = schema.decode(clone, 'OpDef')
    # Unexpected optional inputs/attributes cannot silently enter the experiment.
    if (decoded['input'] != [edges.get(i, '') for i in range(18)]
            or set(_attributes(decoded)) != set(_attributes(parent))):
        raise ValueError('matrix node changed undeclared optional ports or attributes')
    nodes.append(clone)
    outputs = [dict(name='attention', dtype='float16', shape=dims['q'], exact_bytes=False)]
    out_edges = [(parent['name']+':0', descriptors['q'])]
    if s['observed']:
        views = dict(k=[7, s['kv_heads']*s['dim']//16, 512], v=[7, s['kv_heads']*s['dim']*2, 16],
                     mask=dims['mask'], q_length=[1], kv_length=[1,1], all_q_length=[1,1,1], table=[1,7])
        for role, shape in views.items():
            name = 'readback_'+role; dtype = dtypes[role]
            const(name+'_shape', np.array(shape, dtype='<i8'), 'int64')
            d = desc(dtype, shape)
            node(name, 'Reshape', [(edges[ports[role]], descriptors[role]),
                 (name+'_shape:0', desc('int64', [len(shape)]))], [d])
            outputs.append(dict(name=name, role=role, dtype=dtype, shape=shape, exact_bytes=True))
            out_edges.append((name+':0', d))
    node('NetOutput', 'NetOutput', out_edges, [])
    reduced = blob('GraphDef', 'name', 'attention_matrix_'+variant)+b''.join(_blob(op_field, n) for n in nodes)
    output = reduced if graph_field is None else b''.join(_blob(f, reduced) if f == graph_field else e for f,_,e,_ in _parts(raw))
    _decode(schema, output)
    result = dict(policy='source-attention-matrix-v1', matrix_variant=variant,
        source_air_sha256=receipt['source_air_sha256'], source_attention_sha256=receipt['attention_before_sha256'],
        probe_air_sha256=hashlib.sha256(output).hexdigest(), source_sidecar_nodes=[],
        graph=dict(inputs=inputs, outputs=outputs), geometry={k:s[k] for k in ('rows','heads','kv_heads','dim')},
        changes=dict(input_ports=ports, attributes=attr_changes, runtime=s['runtime'], readbacks=s['observed'],
                     constant_mask=bool(s.get('constant_mask')), kv_extent=224, block_size=32),
        limitation='ISOLATED_SOURCE_NODE_WITH_DECLARED_CHANGES; TARGET_GEOMETRY_ONLY_NOT_TARGET_AIR_OR_FULL_CACHE; READBACKS_ARE_LOGICAL_EDGES')
    return output, result


def reference_case(variant, name, declared):
    arrays, expected = values(variant, name)
    ins = [arrays[s['role']] for s in declared['inputs']]
    outs = [expected if s['name'] == 'attention' else arrays[s['role']].reshape(s['shape']) for s in declared['outputs']]
    if any(not np.isfinite(o).all() for o in outs):
        raise ValueError('matrix output references must be finite, including readbacks')
    return ins, outs, dict(matrix_variant=variant, experiment=name.split('__',1)[1],
        geometry={k:VARIANTS[variant][k] for k in ('rows','heads','kv_heads','dim')},
        q_length=int(arrays['q_length'][0]), kv_length=int(arrays['kv_length'][0]),
        all_q_length=int(arrays['all_q_length'][0]), table=arrays['table'].tolist(),
        input_scope='HASH_BOUND_HOST_FIXTURE; NOT_PHYSICAL_KERNEL_READBACK')


def compile_matrix(air_export, source, result, output, *, ddk_root, ddk_id, tbe_diagnosis,
                   timeout_seconds=1800, progress=print):
    from .attention_probe import compile_probe
    target = read_json(Path(__file__).resolve().parents[4]/'specs/target-config.json')['text_config']
    if [target[k] for k in ('num_attention_heads','num_key_value_heads','head_dim')] != [16,2,256]:
        raise ValueError('Target configuration differs from the declared matrix geometry')
    root = output_path(output)
    if Path(output).absolute() != root:
        raise ValueError('matrix output must not contain symlink or traversal components')
    _disjoint(root, air_export, source, result, ddk_root, tbe_diagnosis)
    root.parent.mkdir(parents=True, exist_ok=True)
    with (root.parent/('.'+root.name+'.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root.mkdir()  # Never reset a previous execution or an unknown directory.
        state = dict(artifact_kind=KIND, status='COMPILING', scope=SCOPE, ddk_id=ddk_id,
                     device_execution='NOT_RUN', numerical_validation='NOT_RUN', models=[])
        _save(root/'result.json', state)
        for variant in ('source', *VARIANTS):
            progress('matrix '+variant+': independent model; continuing after known host compile failures')
            try:
                r = compile_probe(air_export, source, result, root/variant, ddk_root=ddk_root, ddk_id=ddk_id,
                    tbe_diagnosis=tbe_diagnosis, timeout_seconds=timeout_seconds, progress=progress,
                    matrix_variant=None if variant == 'source' else variant)
                row = dict(variant=variant, path=variant, status=r['status'], model_compilation=r['model_compilation'],
                           cases=[c['case'] for c in r['cases']], model=r.get('model'), error=r.get('error'))
                row['compiler_observations'] = compiler_observations(root/variant/'compilation/compile.log')
            except (OSError, ValueError, RuntimeError, KeyError) as error:
                row = dict(variant=variant, path=variant, status='FAILED', error=f'{type(error).__name__}: {error}')
            state['models'].append(row); _save(root/'result.json', state)
        compiled = sum(r['status'] == 'COMPILED' for r in state['models'])
        state.update(status='COMPILED' if compiled == len(state['models']) else 'PARTIAL' if compiled else 'FAILED',
                     model_compilation='PASSED' if compiled == len(state['models']) else 'PARTIAL' if compiled else 'FAILED',
                     compiled_models=compiled, total_models=len(state['models']))
        _save(root/'result.json', state)
        return state


def compiler_observations(path):
    if not path.is_file():
        return dict(status='NO_LOG')
    with path.open('rb') as stream:
        raw = stream.read((16 << 20)+1)
    limited = len(raw) > 16 << 20
    rows = [dict(line=i, text=line[:400]) for i,line in enumerate(raw[:16 << 20].decode(errors='replace').splitlines(),1)
            if re.search(r'seqlen[QKV]|seqLen[QKV]|Atb\w*|ti+lingkey|ACTUAL_SEQ_LENGTHS|kernelId not registered',line)]
    return dict(status='INSPECTED', scope='BOUNDED_COMPILER_TEXT_NOT_PHYSICAL_KERNEL_READBACK',
                matching_lines=len(rows), limited=limited or len(rows)>48, excerpts=rows[:48])


def prepare_fixture(root, out, ddk, ddk_id, selection=None):
    from .attention_probe import prepare_fixture as prepare_child
    from validate_draft_vector import selected_cases
    root, out = Path(root), Path(out)
    report = read_json(root/'result.json')
    if report.get('artifact_kind') != KIND or report.get('status') not in ('COMPILED','PARTIAL') or report['ddk_id'] != ddk_id:
        raise ValueError('expected selected-DDK matrix compilation')
    if [m['variant'] for m in report['models']] != ['source', *VARIANTS]:
        raise ValueError('matrix model inventory differs')
    available = [c for m in report['models'] if m['status'] == 'COMPILED' for c in m['cases']]
    selected = selected_cases(available if selection is None else selection, available)
    cases = []; children = {}; template = None; failures = []
    for model in report['models']:
        chosen = [c for c in selected if c in model['cases']] if model['status'] == 'COMPILED' else []
        if not chosen: continue
        if model['path'] != model['variant']:
            raise ValueError('matrix child path differs')
        try:
            child_meta = prepare_child(root/model['path'], out, ddk, ddk_id, chosen)
            if child_meta['source_evidence']['model'] != model['model']:
                raise ValueError('matrix child model identity differs')
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            failures.append(dict(variant=model['variant'], cases=chosen, error=f'{type(error).__name__}: {error}'))
            continue
        template = child_meta
        children[model['variant']] = template['source_evidence']
        cases.extend(template['cases'])
    by_name = {c['case']:c for c in cases}
    _save(out/'matrix-preparation.json', dict(prepared_cases=len(cases), failures=failures))
    if not cases:
        raise ValueError('no matrix group passed compiled ABI/fixture checks; see matrix-preparation.json')
    meta = dict(template, cases=[by_name[n] for n in selected if n in by_name], scope=SCOPE,
                source_evidence=dict(report, children=children, preparation_failures=failures))
    _save(out/'fixture/manifest.json', meta)
    return meta


def _metrics(a, b, atol, rtol):
    delta = np.abs(a.astype(np.float64)-b.astype(np.float64))
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    return dict(bytes_equal=a.tobytes()==b.tobytes(), finite=finite,
        failing_elements=int(np.count_nonzero(~np.isfinite(delta) | (delta > atol+rtol*np.abs(b)))),
        max_abs_error=float(delta.max()) if finite else None)


def diagnose(fixture, native, meta, comparison, report):
    """Recompare all calls first; report compact, non-causal experimental evidence."""
    comparisons = {c['case']:c for c in comparison['cases']}; arrays = {}; records = []
    for c in meta['cases']:
        name = c['case']
        if name not in comparisons: continue
        raw = (native/name/'0.bin').read_bytes(); offset = 0
        for spec in c['outputs']:
            if spec['name'] == 'attention':
                a = np.frombuffer(raw[offset:offset+spec['bytes']], dtype='<f2').reshape(spec['shape'])
                e = np.fromfile(fixture/name/f'golden_{spec["index"]}.bin', dtype='<f2').reshape(spec['shape'])
                arrays[name] = (a, e)
            offset += spec['bytes']
        cmp = comparisons[name]
        row = dict(case=name, passed=cmp['passed'], stable=cmp['repeat_bitwise_stable'],
                   max_abs_error=cmp['numerics']['max_abs_error'],
                   failed_outputs=[o['name'] for o in cmp['outputs'] if not o['passed']])
        if name in CASES:
            variant = c['controls']['matrix_variant']; s = VARIANTS[variant]
            q, k, v, mask, length, table = logical_case(variant, name)
            expected_inputs, reproduction = values(variant, name)
            # Check regenerated fixture semantics, not just its saved SHA.
            for spec in c['inputs']:
                actual = (fixture/name/f'input_{spec["index"]}.bin').read_bytes()
                if actual != expected_inputs[spec['role']].tobytes():
                    raise ValueError('matrix fixture recipe differs: '+name+'/'+spec['role'])
            a, e = arrays[name]
            if e.tobytes() != reproduction.tobytes():
                raise ValueError('matrix reference reproduction differs: '+name)
            for spec in c['outputs']:
                expected = reproduction if spec['name']=='attention' else expected_inputs[spec['role']].reshape(spec['shape'])
                saved = (fixture/name/f'golden_{spec["index"]}.bin').read_bytes()
                if saved != expected.tobytes():
                    raise ValueError('matrix output reference recipe differs: '+name+'/'+spec['name'])
            if any(o['name']=='attention' and not o['passed'] for o in cmp['outputs']):
                candidates = []
                for n in sorted({1, 16, 32, 64, 192, 208, 224, s['rows'], length}):
                    pred = pack_q(sdpa(q, k, v, np.zeros_like(mask), n, table))
                    m = _metrics(a, pred, meta['atol'], meta['rtol'])
                    candidates.append(dict(hypothesis='unmasked_prefix_'+str(n), **m))
                logical = unpack_q(e, s)
                for label, pred in (('output_ND_BNSD', logical.reshape(a.shape)),
                    ('output_ND_BSND', logical.transpose(1,0,2).copy().reshape(a.shape)),
                    ('GQA_modulo', pack_q(sdpa(q,k,v,mask,length,table,group='modulo')))):
                    candidates.append(dict(hypothesis=label, **_metrics(a,pred,meta['atol'],meta['rtol'])))
                row['matching_signatures'] = [m for m in candidates if m['failing_elements'] == 0]
                row['closest_signature'] = min(candidates, key=lambda m:(m['failing_elements'], m['max_abs_error'] if m['finite'] else float('inf')))
        records.append(row)
    pairs = []
    def pair(left, right, purpose):
        if left in arrays and right in arrays and arrays[left][0].shape == arrays[right][0].shape:
            pairs.append(dict(purpose=purpose, cases=[left,right],
                actual=_metrics(arrays[left][0],arrays[right][0],meta['atol'],meta['rtol']),
                reference=_metrics(arrays[left][1],arrays[right][1],meta['atol'],meta['rtol'])))
    for case in BASE_CASES:
        pair('attention_matrix_constant__'+case, 'attention_matrix_runtime__'+case, 'constant_vs_runtime_lengths_and_table')
    for case in VARIANTS['observed']['cases']:
        pair('attention_matrix_runtime__'+case, 'attention_matrix_observed__'+case, 'readbacks_change_compilation')
    pair('attention_matrix_constant__mask_rows', 'attention_matrix_constant_mask__mask_rows', 'runtime_vs_constant_mask')
    for variant in VARIANTS:
        base = 'attention_matrix_'+variant+'__'
        for left,right,purpose in [('mask_rows','table_rotate','page_table_response'),('mask_rows','table_reverse','page_table_response'),
                ('mask_rows','page_equivalent','compensated_page_permutation'),('weighted','query_changed','query_response'),
                ('weighted','key_changed','key_response'),('mask_rows','mask_rows_inf','finite_vs_infinite_mask')]:
            pair(base+left,base+right,purpose)
        for n in LENGTHS:
            pair(base+'uniform',base+'length_'+str(n),'kv_length_response')
        for n in BOUNDARIES:
            pair(base+'zero_v',base+'impulse_'+str(n),'value_row_response')
    source = meta['source_evidence']
    from .attention_probe import CASES as SOURCE_CASES
    complete = set(comparison['validated_cases']) == set((*SOURCE_CASES,*CASES))
    return dict(artifact_kind=KIND+'-diagnosis', status='ANALYZED', scope=SCOPE,
        model_compilation='NOT_RUN', model_execution='NOT_RUN', numerical_validation=comparison['numerical_validation'],
        matrix_validation=comparison['numerical_validation'] if complete else 'INCOMPLETE',
        board_attempt=report.get('board_attempt'), coverage=dict(compiled_models=source['compiled_models'],
        total_models=source['total_models'], requested_cases=len(meta['cases']), compared_cases=len(records),
        planned_cases=len(SOURCE_CASES)+len(CASES),
        failed_cases=sum(not c['passed'] for c in records), skipped_models=[m for m in source['models'] if m['status']!='COMPILED'],
        preparation_failures=source.get('preparation_failures',[])),
        models=[dict(variant=m['variant'],model=m.get('model'),compiler_observations=m.get('compiler_observations')) for m in source['models']],
        cases=records, pair_differences=pairs,
        interpretation='SIGNATURES_AND_RESPONSE_TESTS_ARE_OBSERVATIONS; NO_AUTOMATIC_KERNEL_CAUSE_OR_MODEL_PROMOTION',
        target_scope='TARGET_HEAD_GEOMETRY_PREFILL64_DECODE1; SYNTHETIC_KV224; NOT_ACTUAL_TARGET_AIR_OR_FULL_MODEL')
