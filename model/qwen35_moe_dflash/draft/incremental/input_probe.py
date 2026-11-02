"""Preserve the original candidate ID/embedding/first-RMS prefix for diagnosis."""
import hashlib
import math

from qwen35_moe_dflash.target.original_air_roundtrip import _attributes, _decode
from qwen35_moe_dflash.target.original_ge_append_wire import _blob, _field, _graph_raw, _node_bytes, _parts, _v, _new_id
from .air_zeros import _Shapes

GAMMA = 'draft.layers.0.input_layernorm.weight'


def extract_prefix(raw, schema, lookup, config):
    """Cut after first RMS output 0; keep every original prefix op/weight.

    Only the sole Data.index becomes zero and NetOutput is replaced. Shapes
    are proved from source constants/edges, not copied from missing metadata.
    """
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
                raise ValueError('input-prefix Cast has unexpected GE destination dtype: '+n['name'])
            chain.append(n['name'])
            edge = n['input'][0]

    matches = []
    gamma_path = 'weight/'+GAMMA.replace('.', '_')
    for n, _, _ in rows:
        if n['type'] != 'RmsNorm' or len(n.get('input', [])) != 2:
            continue
        gamma, _ = strip(n['input'][1], 0)
        if gamma['type'] == 'FileConstant' and _attributes(gamma).get('file_path', {}).get('s') == gamma_path:
            matches.append((n, gamma))
    if len(matches) != 1:
        raise ValueError('cannot uniquely identify first candidate RMS from its gamma weight')
    rms, gamma = matches[0]
    epsilon = _attributes(rms).get('epsilon', {}).get('f')
    if (not isinstance(epsilon, (float, int)) or not math.isfinite(epsilon)
            or not math.isclose(epsilon, config.rms_norm_eps, rel_tol=1e-6)
            or len(rms.get('input_desc', [])) != 2 or not rms.get('output_desc')
            or any(d.get('dtype') != types['DT_FLOAT'] for d in rms['input_desc']+[rms['output_desc'][0]])):
        raise ValueError('first RMS must request FP32 input/gamma/output with source epsilon')
    carrier, _ = strip(rms['input'][0], 0)
    scale = 1.0
    if carrier['type'] == 'Muls' and len(carrier.get('input', [])) == 1:
        scale = _attributes(carrier).get('value', {}).get('f')
        if carrier['input'][0] != gather['name']+':0':
            raise ValueError('first RMS scaling does not consume the original embedding Gather')
        if (len(carrier.get('input_desc', [])) != 1 or len(carrier.get('output_desc', [])) != 1
                or any(d.get('dtype') != types['DT_FLOAT16'] for d in carrier['input_desc']+carrier['output_desc'])):
            raise ValueError('embedding scaling must preserve the source FP16 boundary')
    elif carrier['name'] != gather['name']:
        raise ValueError('unsupported path from embedding to first RMS: '+carrier['name'])
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
    net = blob('OpDef', 'name', 'NetOutput') + blob('OpDef', 'type', 'NetOutput')
    for edge, dtype, shape in outputs:
        net += blob('OpDef', 'input', edge, repeated=True) + blob('OpDef', 'input_desc', desc(dtype, shape), 11, True)
    net = _new_id(net, schema, max(n.get('id', 0) for n, _, _ in rows)+1)
    reduced = blob('GraphDef', 'name', 'draft_input_rms_probe') + b''.join(_blob(op_field, data if n['name']==anchor['name'] else p)
        for n,p,_ in rows if n['name'] in selected) + _blob(op_field, net)
    updated = reduced if graph_field is None else b''.join(_blob(f,reduced) if f==graph_field else e for f,_,e,_ in _parts(raw))
    _decode(schema, updated)
    receipt = dict(lookup, policy='source-anchor-embedding-first-rms-prefix-v1', probe_air_sha256=hashlib.sha256(updated).hexdigest(),
        changed='Only anchor Data.index and NetOutput; retained prefix operation bytes unchanged',
        excluded='Attention, residuals, later layers and head; full-graph partition not preserved',
        prefix_nodes=[dict(name=n['name'], type=n['type']) for n,_,_ in rows if n['name'] in selected],
        preserved_payload_sha256={n: hashlib.sha256(payloads[n]).hexdigest() for n in sorted(selected) if n!=anchor['name']},
        anchor=anchor['name'], mask_path=_attributes(mask)['file_path']['s'], gamma_path=gamma_path,
        rms=rms['name'], epsilon=epsilon, scale=scale,
        graph=dict(inputs=[dict(name=anchor['name'], dtype='int32', shape=[1])], outputs=[
            dict(name='embedding', dtype='float16', shape=outputs[0][2], exact_bytes=True),
            dict(name='ids_readback', dtype='int32', shape=outputs[1][2], exact_bytes=True),
            dict(name='input_rms', dtype='float32', shape=outputs[2][2], exact_bytes=False)]))
    return updated, receipt
