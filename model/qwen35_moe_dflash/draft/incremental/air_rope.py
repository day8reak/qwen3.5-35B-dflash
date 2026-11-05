"""Move source FP32 RoPE trigonometry before lookup in a compiler copy.

The four original Sin/Cos results must feed only FP16 casts. Constants hold
FP16 coefficients in FP32 carriers so that the original graph descriptors and
cast boundaries remain valid. No angle is rounded to FP16 before Sin/Cos.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np

from qwen35_moe_dflash.target.original_air_roundtrip import _attributes, _decode, MAX_AIR_BYTES
from qwen35_moe_dflash.target.original_ge_append_wire import _blob, _field, _graph_raw, _node_bytes, _parts, _v, _new_id
from .air_zeros import _Shapes

POLICY = 'source-fp32-rope-trig-before-gather-v1'


def fold_rope_tables(raw, schema, source, *, expected_rows):
    kind, original, graph = _decode(schema, raw)
    graph_raw, graph_field = _graph_raw(raw, schema, kind)
    rows, op_field = _node_bytes(graph_raw, schema)
    nodes = {n['name']: n for n, _, _ in rows}
    payloads = {n['name']: p for n, p, _ in rows}
    if len(nodes) != len(rows):
        raise ValueError('duplicate RoPE source node names')
    unsupported = {'src_name', 'src_index', 'dst_name', 'dst_index', 'input_i', 'output_i', 'subgraph_name'}
    if any(any(n.get(k) for k in unsupported) or
           any('g' in a or a.get('list', {}).get('g') for a in _attributes(n).values()) for n in nodes.values()):
        raise ValueError('RoPE preparation requires uncompiled source AIR without nested graphs')
    trigs = [n for n in nodes.values() if n['type'] in ('Sin', 'Cos')]
    if sorted(n['type'] for n in trigs) != ['Cos', 'Cos', 'Sin', 'Sin']:
        raise ValueError('RoPE repair requires the four shared source Sin/Cos nodes')
    tables = [n for n in nodes.values() if n['type'] == 'FileConstant'
              and _attributes(n).get('file_path', {}).get('s') == 'weight/rope_phases']
    if len(tables) != 1:
        raise ValueError('RoPE repair requires one original FP32 phase table')
    table = tables[0]; attrs = _attributes(table)
    dims = attrs.get('shape', {}).get('list', {}).get('i', [])
    if (attrs.get('dtype', {}).get('i') != 0 or len(dims) != 2 or dims[0] != expected_rows
            or not 1 <= expected_rows <= 4096 or not 1 <= dims[1] <= 256 or table.get('input')):
        raise ValueError('source FP32 RoPE phase table contract differs')
    path = (Path(source)/'weight/rope_phases').resolve()
    if Path(source).resolve() not in path.parents or path.stat().st_size != int(np.prod(dims))*4:
        raise ValueError('source RoPE table path or extent differs')
    phase_bytes = path.read_bytes()
    if len(phase_bytes) != int(np.prod(dims))*4:
        raise ValueError('source RoPE phase extent changed during read')
    phases = np.frombuffer(phase_bytes, dtype='<f4').reshape(dims)
    if not np.isfinite(phases).all():
        raise ValueError('nonfinite source RoPE phases')
    types = schema.enums[schema.package+'.DataType']; shapes = _Shapes(rows, schema)
    fields = {name: _field(schema, 'OpDef', name, typ, repeated) for name, typ, repeated in (
        ('name', 9, False), ('type', 9, False), ('input', 9, True),
        ('input_desc', 11, True), ('output_desc', 11, True), ('attr', 11, True))}
    def blob(message, field, value, typ=9, repeated=False):
        return _blob(_field(schema, message, field, typ, repeated), value)
    def scalar(message, field, value, typ):
        return _v(_field(schema, message, field, typ, False) << 3)+_v(value)
    def edit(payload, *, name=None, op=None, inputs=None, drop_attrs=False):
        parts = []
        for f, _, encoded, _ in _parts(payload):
            if inputs is not None and f == fields['input']: continue
            if drop_attrs and f == fields['attr']: continue
            if name is not None and f == fields['name']: encoded = _blob(f, name)
            if op is not None and f == fields['type']: encoded = _blob(f, op)
            parts.append(encoded)
        if inputs is not None: parts.extend(_blob(fields['input'], e) for e in inputs)
        return b''.join(parts)
    added, cache, replacements, changes, gather_names = [], {}, {}, [], set()
    next_id = max((n.get('id', 0) for n in nodes.values()), default=0)+1
    def append(name, payload):
        nonlocal next_id
        if name in nodes or any(n == name for n, _ in added):
            raise ValueError('RoPE generated node name collision: '+name)
        payload = _new_id(payload, schema, next_id); next_id += 1
        added.append((name, payload))
        return name+':0'
    def constant(trig):
        name = 'rope_fp32_'+trig.lower()+'_coefficients'
        fn = np.sin if trig == 'Sin' else np.cos
        coefficients = fn(phases).astype('<f2').astype('<f4')
        desc = (scalar('TensorDescriptor', 'dtype', types['DT_FLOAT'], 14)
                +blob('TensorDescriptor', 'shape', blob('ShapeDef', 'dim', b''.join(_v(d) for d in dims), 3, True), 11)
                +blob('TensorDescriptor', 'layout', 'ND')+blob('TensorDescriptor', 'device_type', 'NPU'))
        tensor = blob('TensorDef', 'desc', desc, 11)+blob('TensorDef', 'data', coefficients.tobytes(), 12)
        attr = blob('OpDef.AttrEntry', 'key', 'value')+blob('OpDef.AttrEntry', 'value', blob('AttrDef', 't', tensor, 11), 11)
        payload = (blob('OpDef', 'name', name)+blob('OpDef', 'type', 'Const')
                   +blob('OpDef', 'output_desc', desc, 11, True)+blob('OpDef', 'attr', attr, 11, True))
        return append(name, payload)
    def clone(edge, trig, active=()):
        key = (edge, trig)
        if key in cache: return cache[key]
        name, sep, port = edge.rpartition(':')
        if not sep or port != '0' or name not in nodes or name in active or len(active) > 16:
            raise ValueError('unsupported RoPE table path: '+edge)
        n = nodes[name]; a = _attributes(n); inputs = list(n.get('input', []))
        if name == table['name']:
            cache[key] = constant(trig)
            return cache[key]
        if len(n.get('output_desc', [])) != 1 or n['output_desc'][0].get('dtype', types['DT_FLOAT']) != types['DT_FLOAT']:
            raise ValueError('RoPE path must preserve source FP32: '+name)
        if n['type'] == 'ConcatV2':
            if len(inputs) < 3: raise ValueError('invalid RoPE Concat')
            floating = range(len(inputs)-1)
        elif n['type'] == 'GatherV2':
            if (len(inputs) != 3 or shapes.integers(inputs[2]) != [0]
                    or a.get('batch_dims', {}).get('i', 0) != 0
                    or len(n.get('input_desc', [])) != 3
                    or n['input_desc'][1].get('dtype') != types['DT_INT32']):
                raise ValueError('RoPE lookup must preserve the original INT32 axis-zero indices')
            gather_names.add(name); floating = [0]
        elif n['type'] in {'Identity', 'Reshape', 'Transpose', 'Squeeze', 'Unsqueeze', 'ExpandDims'}:
            if not inputs: raise ValueError('empty RoPE view: '+name)
            floating = [0]
        elif n['type'] == 'Cast' and a.get('dst_type', {}).get('i', 0) == 0:
            if len(inputs) != 1: raise ValueError('invalid RoPE FP32 cast')
            floating = [0]
        else:
            raise ValueError('arithmetic or unexpected constant in RoPE lookup path: '+name)
        for i in floating: inputs[i] = clone(inputs[i], trig, (*active, name))
        new_name = 'rope_fp32_'+trig.lower()+'_'+name
        cache[key] = append(new_name, edit(payloads[name], name=new_name, inputs=inputs))
        return cache[key]
    public = set(graph.get('output', []))
    for n in trigs:
        edge = n['name']+':0'
        consumers = [r for r in nodes.values() if edge in r.get('input', [])]
        if (edge in public or n['name'] in public or len(n.get('input', [])) != 1
                or len(n.get('input_desc', [])) != 1 or len(n.get('output_desc', [])) != 1
                or n['input_desc'][0].get('dtype') != types['DT_FLOAT']
                or n['output_desc'][0].get('dtype') != types['DT_FLOAT'] or not consumers
                or any(r['type'] != 'Cast' or r.get('input') != [edge]
                       or _attributes(r).get('dst_type', {}).get('i') != 1 for r in consumers)):
            raise ValueError('RoPE Sin/Cos must feed only original FP16 boundaries: '+n['name'])
        incoming = clone(n['input'][0], n['type'])
        replacements[n['name']] = edit(payloads[n['name']], op='Identity', inputs=[incoming], drop_attrs=True)
        changes.append(dict(node=n['name'], source_type=n['type'], phase_edge=n['input'][0],
            coefficient_edge=incoming, retained_fp16_casts=[r['name'] for r in consumers]))
    if len(gather_names) != 2:
        raise ValueError('expected separate block/context RoPE lookups')
    # Preserve every original node's bytes except the four trig replacements;
    # insert generated dependencies with a stable topological traversal.
    all_nodes = [(n['name'], replacements.get(n['name'], p)) for n, p, _ in rows]+added
    decoded = {name: schema.decode(p, 'OpDef') for name, p in all_nodes}
    ordered, active, seen = [], set(), set()
    def visit(name):
        if name in seen: return
        if name in active: raise ValueError('cycle in rewritten RoPE graph')
        active.add(name)
        for e in decoded[name].get('input', []):
            if not e: continue
            parent, sep, port = e.rpartition(':')
            if not sep or parent not in decoded: raise ValueError('unresolved rewritten input: '+e)
            visit(parent)
        active.remove(name); seen.add(name); ordered.append(name)
    for name, _ in all_nodes: visit(name)
    by_name = dict(all_nodes)
    encoded_nodes = b''.join(_blob(op_field, by_name[n]) for n in ordered)
    graph_parts, emitted = [], False
    for f, _, encoded, _ in _parts(graph_raw):
        if f == op_field:
            if not emitted: graph_parts.append(encoded_nodes); emitted = True
        else: graph_parts.append(encoded)
    changed_graph = b''.join(graph_parts)
    updated = changed_graph if graph_field is None else b''.join(
        _blob(f, changed_graph) if f == graph_field else encoded for f, _, encoded, _ in _parts(raw))
    expected = deepcopy(original); expected_graph = expected if kind == 'GraphDef' else expected['graph'][0]
    expected_graph['op'] = [decoded[n] for n in ordered]
    if len(updated) > MAX_AIR_BYTES or _decode(schema, updated)[1] != expected:
        raise ValueError('RoPE preparation changed unrelated graph metadata')
    return updated, dict(policy=POLICY, status='APPLIED', trig_count=4, table_shape=dims,
        phase_file=dict(path='weight/rope_phases', bytes=len(phase_bytes), sha256=hashlib.sha256(phase_bytes).hexdigest()),
        source_sha256=hashlib.sha256(raw).hexdigest(), compiler_sha256=hashlib.sha256(updated).hexdigest(),
        coefficients='NUMPY_SIN_COS_FP32_PHASE_THEN_FP16; FP32_EXACT_CARRIER',
        changes=changes, added_nodes=len(added), added_constant_bytes=phases.nbytes*2,
        coverage='ALL_ORIGINAL_TABLE_POSITIONS; ORIGINAL_INDICES_VIEWS_AND_FP16_CASTS',
        limitation='NUMPY_VS_FRAMEWORK_TRANSCENDENTAL_ROUNDING_AND_COMPILED_GRAPH_REQUIRE_ORIGINAL_FULL_MODEL_GATES')
