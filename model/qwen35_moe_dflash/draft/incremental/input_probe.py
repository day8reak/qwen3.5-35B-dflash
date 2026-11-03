"""Preserve the original candidate ID/embedding/first-RMS prefix for diagnosis."""
import hashlib
import math
import struct

from qwen35_moe_dflash.target.original_air_roundtrip import _attributes, _decode
from qwen35_moe_dflash.target.original_ge_append_wire import _blob, _field, _graph_raw, _node_bytes, _parts, _v, _new_id
from .air_fill import _FillValues
from .air_zeros import _Shapes

GAMMA = 'draft.layers.0.input_layernorm.weight'


def _unary_path(shapes, edge, allowed):
    """Return the root and execution-ordered nodes without discarding Casts."""
    path, seen = [], set()
    while True:
        node = shapes.node(edge)
        if node['type'] not in allowed:
            return node, list(reversed(path))
        if len(path) >= 16 or node['name'] in seen or len(node.get('input', [])) != 1:
            raise ValueError('cyclic/unsupported input-prefix unary chain: '+node['name'])
        seen.add(node['name'])
        path.append(node)
        edge = node['input'][0]


def _float_path(shapes, root, path, shape, *, allow_scale=False):
    """Prove FP16 source -> optional scaling/FP16 rounding -> FP32 RMS.

    Cast destination attributes use GE numbers; descriptors use protobuf enums.
    Replay types forward and retain all nodes. Only placeholder descriptors
    covered by the existing source-AIR rule may omit a concrete dtype.
    """
    values = _FillValues(shapes)
    half, single = values.types['DT_FLOAT16'], values.types['DT_FLOAT']
    names = {half: 'float16', single: 'float32'}
    dtype, placeholder = half, root['type'] == 'FileConstant'
    scale, rounded, proof = None, False, []
    for node in path:
        inputs, outputs = node.get('input_desc', []), node.get('output_desc', [])
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError('float prefix requires one input/output descriptor: '+node['name'])
        for desc in inputs+outputs:
            if desc.get('layout', 'ND') != 'ND':
                raise ValueError('float prefix requires ND descriptors: '+node['name'])
            recorded = desc.get('shape', {}).get('dim')
            if recorded and recorded != shape:
                raise ValueError('float prefix recorded shape conflicts: '+node['name'])
        input_placeholder = values.descriptor_dtype(inputs[0], dtype, 'float prefix input '+node['name'],
                                                    allow_placeholder=placeholder)
        attrs, before = _attributes(node), dtype
        if node['type'] == 'Cast':
            dtype = values.attribute_dtype(node, 'dst_type')
            if dtype not in names:
                raise ValueError('float prefix Cast destination dtype must be FP16 or FP32: '+node['name'])
            placeholder = values.descriptor_dtype(outputs[0], dtype, 'float prefix Cast output '+node['name'],
                                                  allow_placeholder=True)
        elif node['type'] in {'Identity', 'Muls'}:
            if node['type'] == 'Muls':
                if not allow_scale or scale is not None:
                    raise ValueError('float prefix permits at most one embedding Muls')
                scale = attrs.get('value', {}).get('f')
                if not isinstance(scale, (float, int)) or not math.isfinite(scale):
                    raise ValueError('embedding Muls needs a finite scalar')
            # TorchAir's raw Muls API adds an output name without a dtype;
            # aten.mul then casts that internal result to the FX output dtype.
            # Resolve only an untyped/default descriptor from Muls' input type.
            # Source: Ascend/torchair e1118e8b, ge_converter/aten/mul.py and
            # auto_generated_ge_raw_ops.py (Muls).
            placeholder = values.descriptor_dtype(outputs[0], dtype, 'float prefix output '+node['name'],
                                                  allow_placeholder=node['type'] == 'Muls')
        else:
            raise ValueError('unsupported float prefix operator: '+node['type'])
        if scale is not None and dtype == half:
            rounded = True
        proof.append(dict(name=node['name'], type=node['type'], input_dtype=names[before],
                          output_dtype=names[dtype], ge_dst_type=attrs.get('dst_type', {}).get('i'),
                          recorded_input_dtype=inputs[0].get('dtype'),
                          recorded_output_dtype=outputs[0].get('dtype'),
                          placeholder_input_dtype=input_placeholder,
                          placeholder_output_dtype=placeholder))
    if dtype != single:
        raise ValueError('float prefix final destination dtype must be FP32 at RMS')
    if scale is not None and not rounded:
        raise ValueError('embedding scaling must preserve the source FP16 boundary before RMS')
    return (1.0 if scale is None else scale), proof


def extract_prefix(raw, schema, lookup, config, *, capture_inputs=False, bypass_unit_scale=False):
    """Cut after first RMS output 0; keep every original prefix op/weight.

    The original anchor Data.index becomes zero and NetOutput is replaced.
    Optional taps add a runtime unity factor to observe constant gamma through
    a live Mul branch. Original RMS inputs/operations remain byte-identical.
    A separate comparison case may bypass only a proved FP16 Muls(value=1).
    """
    if bypass_unit_scale and not capture_inputs:
        raise ValueError('unit-scale comparison requires the six-output RMS taps')
    if bypass_unit_scale and config.input_embedding_scale != 1.0:
        raise ValueError('unit-scale comparison requires reference scale exactly 1')
    kind, _, _ = _decode(schema, raw)
    graph_raw, graph_field = _graph_raw(raw, schema, kind)
    rows, op_field = _node_bytes(graph_raw, schema)
    nodes = {n['name']: n for n, _, _ in rows}
    payloads = {n['name']: p for n, p, _ in rows}
    shapes = _Shapes(rows, schema)
    gather = nodes[lookup['gather']]
    types = schema.enums[schema.package+'.DataType']

    def strip(edge, dtype):
        chain = []
        while True:
            n = shapes.node(edge)
            if n['type'] not in {'Cast', 'Identity'}:
                return n, chain
            if len(chain) >= 8 or n['name'] in chain or len(n.get('input', [])) != 1:
                raise ValueError('cyclic/unsupported input-prefix cast chain')
            if n['type'] == 'Cast' and _attributes(n).get('dst_type', {}).get('i') != dtype:
                raise ValueError(
                    f'input-prefix Cast has unexpected GE destination dtype: {n["name"]}; '
                    f'expected_ge_dtype={dtype}, actual_dst_type={_attributes(n).get("dst_type")}, '
                    f'input_edges={n.get("input")}, '
                    f'input_descriptor_dtypes={[d.get("dtype") for d in n.get("input_desc", [])]}, '
                    f'output_descriptor_dtypes={[d.get("dtype") for d in n.get("output_desc", [])]}. '
                    'GE attribute numbers and protobuf descriptor numbers use different enums; '
                    'inspect the complete chain before changing the graph.')
            chain.append(n['name'])
            edge = n['input'][0]

    matches = []
    gamma_path = 'weight/'+GAMMA.replace('.', '_')
    for n, _, _ in rows:
        if n['type'] != 'RmsNorm' or len(n.get('input', [])) != 2:
            continue
        gamma, gamma_path_nodes = _unary_path(shapes, n['input'][1], {'Cast', 'Identity'})
        if gamma['type'] == 'FileConstant' and _attributes(gamma).get('file_path', {}).get('s') == gamma_path:
            matches.append((n, gamma, gamma_path_nodes))
    if len(matches) != 1:
        raise ValueError('cannot uniquely identify first candidate RMS from its gamma weight')
    rms, gamma, gamma_path_nodes = matches[0]
    epsilon = _attributes(rms).get('epsilon', {}).get('f')
    if (not isinstance(epsilon, (float, int)) or not math.isfinite(epsilon)
            or not math.isclose(epsilon, config.rms_norm_eps, rel_tol=1e-6)
            or len(rms.get('input_desc', [])) != 2 or not rms.get('output_desc')
            or any(d.get('dtype') != types['DT_FLOAT'] for d in rms['input_desc']+[rms['output_desc'][0]])):
        raise ValueError('first RMS must request FP32 input/gamma/output with source epsilon')
    carrier, embedding_path = _unary_path(shapes, rms['input'][0], {'Cast', 'Identity', 'Muls'})
    if carrier['name'] != gather['name']:
        raise ValueError('unsupported path from embedding to first RMS: '+carrier['name'])
    scale, embedding_types = _float_path(shapes, gather, embedding_path,
                                       [1,16,config.hidden_size], allow_scale=True)
    _, gamma_types = _float_path(shapes, gamma, gamma_path_nodes, [config.hidden_size])
    if (not isinstance(scale, (float, int)) or not math.isfinite(scale)
            or not math.isclose(scale, config.input_embedding_scale, rel_tol=1e-6)):
        raise ValueError('AIR embedding scale conflicts with source reference config')
    for descriptor, expected in ((rms['input_desc'][0], [1,16,config.hidden_size]),
                                 (rms['input_desc'][1], [config.hidden_size]),
                                 (rms['output_desc'][0], [1,16,config.hidden_size])):
        recorded = descriptor.get('shape', {}).get('dim')
        if recorded and recorded != expected:
            raise ValueError('first RMS recorded shape conflicts with embedding prefix')

    ids, _ = strip(gather['input'][1], 3)
    if (ids['type'] != 'ConcatV2' or len(ids.get('input', [])) != 3
            or shapes.integers(ids['input'][2]) != [1]):
        raise ValueError('expected original anchor/mask ConcatV2 along token axis')
    anchor_view = shapes.node(ids['input'][0])
    mask = shapes.node(ids['input'][1])
    if (anchor_view['type'] != 'Reshape' or len(anchor_view.get('input', [])) != 2
            or shapes.integers(anchor_view['input'][1]) != [1, 1]):
        raise ValueError('expected source anchor reshape [1] to [1,1]')
    anchor = shapes.node(anchor_view['input'][0])
    if (anchor['type'] != 'Data' or shapes.shape(anchor['name']+':0') != [1]
            or anchor['output_desc'][0].get('dtype') != types['DT_INT32']
            or mask['type'] != 'FileConstant' or shapes.shape(mask['name']+':0') != [1, 15]
            or _attributes(mask).get('dtype', {}).get('i') != 3
            or shapes.shape(gather['input'][1]) != [1, 16]
            or shapes.shape(gather['name']+':0') != [1, 16, config.hidden_size]
            or shapes.shape(gamma['name']+':0') != [config.hidden_size]
            or _attributes(gamma).get('dtype', {}).get('i') != 1):
        raise ValueError('source anchor/mask/gamma descriptor differs from the bounded prefix')

    selected, visiting = set(), set()
    def visit(edge):
        n = shapes.node(edge)
        name = n['name']
        if name in selected:
            return
        if name in visiting or len(visiting) > 32:
            raise ValueError('cyclic/oversized input prefix')
        if n['type'] not in {'Data', 'Const', 'FileConstant', 'Reshape', 'Cast', 'Identity', 'ConcatV2', 'GatherV2', 'Muls', 'RmsNorm'}:
            raise ValueError('unexpected operator in input prefix: '+n['type'])
        if (any(n.get(k) for k in ('src_name', 'src_index', 'dst_name', 'dst_index', 'subgraph_name'))
                or any('g' in a or a.get('list', {}).get('g') for a in _attributes(n).values())):
            raise ValueError('nested/compiled input-prefix graph is unsupported')
        if any(d.get('layout', 'ND') != 'ND' for d in n.get('input_desc', [])+n.get('output_desc', [])):
            raise ValueError('input prefix requires ND descriptors')
        visiting.add(name)
        for e in n.get('input', []):
            if not e and n['type'] == 'Data':
                continue
            visit(e)
        visiting.remove(name); selected.add(name)
    visit(rms['name']+':0')
    if (len(selected) > 32 or [n for n in selected if nodes[n]['type'] == 'Data'] != [anchor['name']]
            or {n for n in selected if nodes[n]['type'] == 'FileConstant'} != {lookup['table'], mask['name'], gamma['name']}):
        raise ValueError('prefix must have one anchor input and exactly three original weight files')

    def blob(message, field, value, field_type=9, repeated=False):
        return _blob(_field(schema, message, field, field_type, repeated), value)
    def desc(dtype, shape):
        return (_v(_field(schema, 'TensorDescriptor', 'dtype', 14, False) << 3) + _v(types[dtype])
                + blob('TensorDescriptor', 'shape', blob('ShapeDef', 'dim', b''.join(_v(d) for d in shape), 3, True), 11)
                + blob('TensorDescriptor', 'layout', 'ND') + blob('TensorDescriptor', 'device_type', 'NPU'))
    attr_field = _field(schema, 'OpDef', 'attr', 11, True)
    index_value = _v(_field(schema, 'AttrDef', 'i', 3, False) << 3) + _v(0)
    index = blob('OpDef.AttrEntry', 'key', 'index') + blob('OpDef.AttrEntry', 'value', index_value, 11)
    data = b''.join(encoded for f, _, encoded, p in _parts(payloads[anchor['name']])
        if f != attr_field or schema.decode(p, 'OpDef.AttrEntry').get('key') != 'index') + _blob(attr_field, index)
    outputs = [(gather['name']+':0', 'DT_FLOAT16', [1,16,config.hidden_size]),
               (gather['input'][1], 'DT_INT32', [1,16]),
               (rms['name']+':0', 'DT_FLOAT', [1,16,config.hidden_size])]
    added, taps = [], []
    next_id = max(n.get('id', 0) for n, _, _ in rows)+1
    if capture_inputs:
        # A flat view gives x a unique public (shape,dtype) pair. Never infer
        # equal-shaped tensor identities from compiled slot order.
        shape_name, view_name = 'probe_rms_x_shape', 'probe_rms_x_view'
        factor_name, gamma_view = 'probe_gamma_factor', 'probe_gamma_product'
        factor_shape, factor_view = 'probe_gamma_factor_shape', 'probe_gamma_factor_view'
        if {shape_name, view_name, factor_name, gamma_view, factor_shape, factor_view} & set(nodes):
            raise ValueError('RMS tap node names collide with source AIR')
        tensor = (blob('TensorDef', 'desc', desc('DT_INT64', [2]), 11)
                  + blob('TensorDef', 'data', struct.pack('<2q', 16, config.hidden_size), 12))
        value = blob('AttrDef', 't', tensor, 11)
        entry = blob('OpDef.AttrEntry', 'key', 'value') + blob('OpDef.AttrEntry', 'value', value, 11)
        constant = (blob('OpDef', 'name', shape_name) + blob('OpDef', 'type', 'Const')
                    + blob('OpDef', 'output_desc', desc('DT_INT64', [2]), 11, True)
                    + _blob(attr_field, entry))
        view = blob('OpDef', 'name', view_name) + blob('OpDef', 'type', 'Reshape')
        for edge, descriptor in ((rms['input'][0], desc('DT_FLOAT', [1,16,config.hidden_size])),
                                 (shape_name+':0', desc('DT_INT64', [2]))):
            view += blob('OpDef', 'input', edge, repeated=True) + blob('OpDef', 'input_desc', descriptor, 11, True)
        view += blob('OpDef', 'output_desc', desc('DT_FLOAT', [16,config.hidden_size]), 11, True)
        # Direct Const/Cast -> NetOutput gamma was observed with an empty
        # compiled edge. A runtime factor prevents a constant-only output.
        factor_desc = desc('DT_FLOAT', [config.hidden_size])
        factor = blob('OpDef', 'name', factor_name) + blob('OpDef', 'type', 'Data')
        factor += blob('OpDef', 'input_desc', factor_desc, 11, True) + blob('OpDef', 'output_desc', factor_desc, 11, True)
        factor_index = _v(_field(schema, 'AttrDef', 'i', 3, False) << 3) + _v(1)
        factor += _blob(attr_field, blob('OpDef.AttrEntry', 'key', 'index')
                        + blob('OpDef.AttrEntry', 'value', factor_index, 11))
        product = blob('OpDef', 'name', gamma_view) + blob('OpDef', 'type', 'Mul')
        for edge in (rms['input'][1], factor_name+':0'):
            product += blob('OpDef', 'input', edge, repeated=True)
            product += blob('OpDef', 'input_desc', factor_desc, 11, True)
        product += blob('OpDef', 'output_desc', factor_desc, 11, True)
        tensor = (blob('TensorDef', 'desc', desc('DT_INT64', [2]), 11)
                  + blob('TensorDef', 'data', struct.pack('<2q', 1, config.hidden_size), 12))
        factor_constant = (blob('OpDef', 'name', factor_shape) + blob('OpDef', 'type', 'Const')
                           + blob('OpDef', 'output_desc', desc('DT_INT64', [2]), 11, True)
                           + _blob(attr_field, blob('OpDef.AttrEntry', 'key', 'value')
                                   + blob('OpDef.AttrEntry', 'value', blob('AttrDef', 't', tensor, 11), 11)))
        factor_readback = blob('OpDef', 'name', factor_view) + blob('OpDef', 'type', 'Reshape')
        for edge, descriptor in ((factor_name+':0', factor_desc), (factor_shape+':0', desc('DT_INT64', [2]))):
            factor_readback += blob('OpDef', 'input', edge, repeated=True) + blob('OpDef', 'input_desc', descriptor, 11, True)
        factor_readback += blob('OpDef', 'output_desc', desc('DT_FLOAT', [1,config.hidden_size]), 11, True)
        added = [_new_id(n, schema, next_id+i) for i, n in
                 enumerate((constant, view, factor, product, factor_constant, factor_readback))]
        next_id += len(added)
        outputs += [(view_name+':0', 'DT_FLOAT', [16,config.hidden_size]),
                    (gamma_view+':0', 'DT_FLOAT', [config.hidden_size]),
                    (factor_view+':0', 'DT_FLOAT', [1,config.hidden_size])]
        taps = [dict(name='rms_x', source_edge=rms['input'][0], view=view_name),
                dict(name='rms_gamma', source_edge=rms['input'][1], view=gamma_view,
                     operation='MUL_BY_RUNTIME_FACTOR', factor_input=factor_name)]
    net = blob('OpDef', 'name', 'NetOutput') + blob('OpDef', 'type', 'NetOutput')
    for edge, dtype, shape in outputs:
        net += blob('OpDef', 'input', edge, repeated=True) + blob('OpDef', 'input_desc', desc(dtype, shape), 11, True)
    net = _new_id(net, schema, next_id)
    reduced = blob('GraphDef', 'name', 'draft_input_rms_probe') + b''.join(_blob(op_field, data if n['name']==anchor['name'] else p)
        for n,p,_ in rows if n['name'] in selected) + b''.join(_blob(op_field, p) for p in added) + _blob(op_field, net)
    updated = reduced if graph_field is None else b''.join(_blob(f,reduced) if f==graph_field else e for f,_,e,_ in _parts(raw))
    _decode(schema, updated)
    receipt = dict(lookup, policy='source-anchor-embedding-first-rms-prefix-v1', probe_air_sha256=hashlib.sha256(updated).hexdigest(),
        changed='Only anchor Data.index and NetOutput; retained prefix operation bytes unchanged',
        excluded='Attention, residuals, later layers and head; full-graph partition not preserved',
        prefix_nodes=[dict(name=n['name'], type=n['type']) for n,_,_ in rows if n['name'] in selected],
        preserved_payload_sha256={n: hashlib.sha256(payloads[n]).hexdigest() for n in sorted(selected) if n!=anchor['name']},
        anchor=anchor['name'], mask_path=_attributes(mask)['file_path']['s'], gamma_path=gamma_path,
        rms=rms['name'], epsilon=epsilon, scale=scale,
        float_paths=dict(embedding=embedding_types, gamma=gamma_types),
        graph=dict(inputs=[dict(name=anchor['name'], dtype='int32', shape=[1])], outputs=[
            dict(name='embedding', dtype='float16', shape=outputs[0][2], exact_bytes=True),
            dict(name='ids_readback', dtype='int32', shape=outputs[1][2], exact_bytes=True),
            dict(name='input_rms', dtype='float32', shape=outputs[2][2], exact_bytes=False)]))
    if capture_inputs:
        receipt.update(policy='source-first-rms-input-taps-v2', taps=taps,
                       changed='Original prefix bytes preserved except anchor Data.index; add x view, gamma runtime-unity Mul, factor readback and NetOutput taps',
                       limitation='Extra output edges may change compiler partition or fusion; compare original RMS output again')
        receipt['gamma_observation'] = dict(method='SOURCE_GAMMA_TIMES_RUNTIME_UNITY',
            input=factor_name, required_value=1.0, output='rms_gamma', factor_output='gamma_factor_readback',
            scope='AUXILIARY_BRANCH_NOT_PHYSICAL_RMS_KERNEL_INPUT_READBACK')
        receipt['graph']['inputs'].append(dict(name=factor_name, dtype='float32', shape=[config.hidden_size]))
        receipt['graph']['outputs'] += [
            dict(name='rms_x', dtype='float32', shape=outputs[3][2], exact_bytes=True),
            dict(name='rms_gamma', dtype='float32', shape=outputs[4][2], exact_bytes=True),
            dict(name='gamma_factor_readback', dtype='float32', shape=outputs[5][2], exact_bytes=True)]
    if bypass_unit_scale:
        return _bypass_unit_scale(updated, schema, receipt)
    return updated, receipt


def _bypass_unit_scale(raw, schema, receipt):
    """Apply the shared compiler rewrite to the six-output diagnostic copy."""
    from .air_embedding import bypass_unit_scale
    updated, change = bypass_unit_scale(raw, schema, receipt['gather'],
                                        receipt['float_paths']['embedding'], receipt['scale'])
    name, consumer = change['removed_node'], change['consumer']
    receipt = dict(receipt, policy='source-first-rms-unit-scale-bypass-v1',
        baseline_probe_air_sha256=hashlib.sha256(raw).hexdigest(),
        probe_air_sha256=hashlib.sha256(updated).hexdigest(),
        changed='Same six-output taps; remove direct FP16 Muls(value=1) and rewire its sole Cast input to Gather; all other node bytes unchanged',
        prefix_nodes=[n for n in receipt['prefix_nodes'] if n['name'] != name],
        preserved_payload_sha256={n: sha for n, sha in receipt['preserved_payload_sha256'].items()
                                 if n not in {name, consumer}})
    receipt['unit_scale_bypass'] = dict(change,
        scope='DIAGNOSTIC_COPY_ONLY; COMPILED_PARTITION_MAY_CHANGE; NOT_FULL_DRAFT_FIX')
    return updated, receipt
