"""One bounded 2x2 alternative to the retained packed 4x4 feature Conv.

Both compute all 16384 products for every one of the 16x2048 outputs,
with unchanged weight words and one original FP16 result boundary. Arithmetic
is algebraically identical; compiled accumulation order/equivalence is unproven.
This module does not rewrite or promote a full Draft.
"""
import hashlib

import numpy as np

from qwen35_moe_dflash.target.original_air_roundtrip import _attributes
from qwen35_moe_dflash.target.original_ge_append_wire import _blob, _field, _parts, _v
from .air_feature_pack import pack_kernel2_words

SHAPE = [1, 4096, 8, 8]
EXPERIMENT = 'full-dot-kernel2-v1'
VARIANTS = ('packed', 'kernel2')
CASES = tuple(f'feature_projection_c{s:02d}_{v}' for s in (2, 16, 17) for v in VARIANTS)


def extract_kernel2(source):
    from .feature_pack_probe import Projection, PACKED_SHAPE, INPUT_SHAPE, WEIGHT
    raw, receipt = source.extract_projection('packed')
    trace = Projection(raw, source.schema)
    schema, conv = source.schema, trace.nodes[source.conv['name']]
    attrs = _attributes(conv)
    if (attrs.get('strides', {}).get('list', {}).get('i') != [1, 1, 4, 4]
            or attrs.get('pads', {}).get('list', {}).get('i') != [0]*4
            or attrs.get('dilations', {}).get('list', {}).get('i', [1]*4) != [1]*4
            or attrs.get('data_format', {}).get('s') != 'NCHW'
            or attrs.get('groups', {}).get('i', 1) != 1
            or attrs.get('offset_x', {}).get('i', 0) != 0
            or len(trace.live_inputs(conv)) != 2):
        raise ValueError('kernel2 requires the original ungrouped zero-pad stride-4 full-dot Conv')
    weights = [n for n in trace.nodes.values() if n['type'] == 'FileConstant']
    if len(weights) != 1 or _attributes(weights[0]).get('file_path', {}).get('s') != WEIGHT:
        raise ValueError('kernel2 requires only the retained feature weight')
    weight = weights[0]
    # Only reshapes may be bypassed: every weight word retains its linear index.
    edge, weight_views = conv['input'][1], set()
    while edge != weight['name']+':0':
        n = trace.shapes.node(edge)
        if n['name'] in weight_views:
            raise ValueError('cyclic projection weight view')
        weight_views.add(n['name'])
        if n['type'] == 'Identity' and len(n.get('input', [])) == 1:
            pass
        elif n['type'] == 'Reshape' and len(n.get('input', [])) == 2:
            if _attributes(n).get('axis', {}).get('i', 0) != 0:
                raise ValueError('partial projection weight reshape')
            trace.shapes.integers(n['input'][1])  # No dynamic shape carrier.
        else:
            raise ValueError('kernel2 may only bypass row-major weight reshapes')
        edge = n['input'][0]
    if trace.shapes.shape(conv['input'][1]) != [2048, 1024, 4, 4]:
        raise ValueError('source Conv filter must be [2048,1024,4,4]')
    labels = np.arange(16*16384, dtype=np.int32).reshape(INPUT_SHAPE)
    packed = pack_kernel2_words(labels)
    patches = np.lib.stride_tricks.sliding_window_view(packed, (2, 2), axis=(2, 3))[:, :, ::2, ::2]
    dots = patches.transpose(0, 2, 3, 1, 4, 5).reshape(INPUT_SHAPE)
    if not np.array_equal(dots, labels):
        raise ValueError('kernel2 changes a token or full-dot term index')

    def blob(msg, key, val, typ=9, repeated=False):
        return _blob(_field(schema, msg, key, typ, repeated), val)
    def int_list(values):
        tag = _field(schema, 'AttrDef.ListValue', 'val_type', 14, False)
        code = schema.enums[schema.package+'.AttrDef.ListValue.ListValueType']['VT_LIST_INT']
        return blob('AttrDef', 'list', blob('AttrDef.ListValue', 'i', b''.join(_v(v) for v in values), 3, True)
                    +_v(tag << 3)+_v(code), 11)
    def attribute(key, value):
        return blob('OpDef', 'attr', blob('OpDef.AttrEntry', 'key', key)+blob('OpDef.AttrEntry', 'value', value, 11), 11, True)
    def descriptor(shape, name=None):
        desc = (_v(_field(schema, 'TensorDescriptor', 'dtype', 14, False) << 3)+_v(trace.types['DT_FLOAT16'])
            +blob('TensorDescriptor', 'shape', blob('ShapeDef', 'dim', b''.join(_v(v) for v in shape), 3, True), 11)
            +blob('TensorDescriptor', 'layout', 'ND')+blob('TensorDescriptor', 'device_type', 'NPU'))
        return desc if name is None else blob('TensorDescriptor', 'name', name)+desc
    fields = {k: _field(schema, 'OpDef', k, t, True) for k, t in
              (('input', 9), ('input_desc', 11), ('output_desc', 11), ('attr', 11))}
    payloads = {n['name']: p for n, p, _ in trace.rows}
    def edit(name, *, shapes=(), input1=None, attr_values=None):
        parts, input_index, desc_index = [], 0, 0
        for f, _, encoded, p in _parts(payloads[name]):
            if f == fields['input']:
                if input_index == 1 and input1 is not None:
                    encoded = _blob(f, input1)
                input_index += 1
            if f == fields['input_desc']:
                if desc_index < len(shapes):
                    encoded = _blob(f, descriptor(shapes[desc_index], schema.decode(p, 'TensorDescriptor').get('name')))
                desc_index += 1
            if f == fields['attr'] and attr_values and schema.decode(p, 'OpDef.AttrEntry')['key'] in attr_values:
                continue
            if f == fields['output_desc'] and name in (source.input, weight['name']):
                encoded = _blob(f, descriptor(SHAPE if name == source.input else [2048, 4096, 2, 2],
                                              schema.decode(p, 'TensorDescriptor').get('name')))
            parts.append(encoded)
        parts.extend(attribute(k, v) for k, v in (attr_values or {}).items())
        return b''.join(parts)
    replacements = {
        source.input: edit(source.input, shapes=[SHAPE]),
        weight['name']: edit(weight['name'], attr_values={'shape': int_list([2048, 4096, 2, 2])}),
        conv['name']: edit(conv['name'], shapes=[SHAPE, [2048, 4096, 2, 2]], input1=weight['name']+':0',
                           attr_values={'strides': int_list([1, 1, 2, 2])})}
    modified = {name: schema.decode(replacements.get(name, p), 'OpDef') for name, p in payloads.items()}
    live = set()
    def visit(name):
        if name in live: return
        live.add(name)
        for edge in modified[name].get('input', []):
            if edge: visit(edge.rsplit(':', 1)[0])
    visit('NetOutput')
    graph = blob('GraphDef', 'name', 'draft_feature_projection_kernel2')+b''.join(
        _blob(trace.op_field, replacements.get(n['name'], p)) for n, p, _ in trace.rows if n['name'] in live)
    updated = graph if trace.graph_field is None else b''.join(
        _blob(f, graph) if f == trace.graph_field else e for f, _, e, _ in _parts(raw))
    from .feature_pack_probe import input_contract
    contract = input_contract(updated, schema, source.input, SHAPE)
    receipt['graph']['inputs'][0]['shape'] = list(SHAPE)
    receipt.update(experiment=EXPERIMENT, data_descriptor_contract=contract,
        changes='PACKED_DATA_AND_FILTER_SHAPES; STRIDE_2; BYPASS_DEAD_WEIGHT_VIEWS',
        conv_attributes='STRIDES_ONLY; OTHER_ATTRIBUTES_UNCHANGED',
        input_mapping='ALL_262144_PATCH_TERM_INDICES_EQUAL_SOURCE_LOGICAL_ROWS',
        preserved_payload_sha256={n['name']: hashlib.sha256(p).hexdigest() for n, p, _ in trace.rows
                                  if n['name'] in live and n['name'] not in replacements},
        removed_nodes=sorted(set(receipt['removed_nodes']) | (set(payloads)-live)),
        full_dot=dict(terms=16384, output_elements=16*2048, weight_bytes=67108864,
                      source_input_shape=PACKED_SHAPE, candidate_input_shape=SHAPE,
                      output_shape=[1, 16, 2048], weight_words='UNCHANGED',
                      arithmetic='FULL_K_DOT_THEN_ONE_FP16_RESULT',
                      classification='ALGEBRAICALLY_EXACT; DEVICE_ROUNDING_ORDER_UNVERIFIED'))
    return updated, receipt
