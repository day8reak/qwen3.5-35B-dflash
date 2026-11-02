"""Materialize static ZerosLike outputs in the compiler copy of Draft AIR.

Only graph descriptors, literal shape/axis constants and a bounded set of shape
rules are used. No expected Draft dimensions, weights or tensor values are read.
Unknown shapes stop preparation instead of inventing a successful replacement.
"""
from copy import deepcopy
import hashlib
from itertools import zip_longest
import json
import math
import struct

from qwen35_moe_dflash.target.original_air_roundtrip import _attributes, _decode, MAX_AIR_BYTES
from qwen35_moe_dflash.target.original_ge_append_wire import _blob, _field, _graph_raw, _node_bytes, _parts, _v
from .air_reshape import _payload


def _known(shape):
    return (isinstance(shape, list) and len(shape) <= 8
            and all(type(d) is int and 0 <= d <= (1 << 31) for d in shape))


def _broadcast(shapes):
    result = []
    for dims in zip_longest(*(reversed(s) for s in shapes), fillvalue=1):
        nonunit = set(dims) - {1}
        if len(nonunit) > 1:
            raise ValueError('incompatible broadcast shapes: ' + str(shapes))
        result.append(next(iter(nonunit), 1))
    return list(reversed(result))


class _Shapes:
    def __init__(self, rows, schema):
        self.nodes = {n['name']: n for n, _, _ in rows}
        self.payloads = {n['name']: p for n, p, _ in rows}
        self.schema, self.cache, self.active = schema, {}, set()
        self.proofs = {}

    def constant_data(self, node):
        number = _field(self.schema, 'OpDef', 'attr', 11, True)
        for field, _, _, entry in _parts(self.payloads[node['name']]):
            if field == number and self.schema.decode(entry, 'OpDef.AttrEntry').get('key') == 'value':
                attr = _payload(entry, self.schema, 'OpDef.AttrEntry', 'value')
                tensor = _payload(attr, self.schema, 'AttrDef', 't')
                return _payload(tensor, self.schema, 'TensorDef', 'data') if tensor is not None else None
        return None

    def node(self, edge):
        name, separator, port = edge.rpartition(':')
        if separator != ':' or port != '0' or name not in self.nodes:
            raise ValueError('static shape needs a resolved output-0 edge: ' + edge)
        return self.nodes[name]

    def integers(self, edge):
        node = self.node(edge)
        value = _attributes(node).get('value', {}).get('t', {})
        types = self.schema.enums[self.schema.package + '.DataType']
        fmt = {types['DT_INT32']: 'i', types['DT_INT64']: 'q'}.get(value.get('desc', {}).get('dtype'))
        if node['type'] != 'Const' or node.get('input') or not fmt:
            raise ValueError('static shape/axis requires an integer Const: ' + edge)
        shape = value.get('desc', {}).get('shape', {}).get('dim', [])
        if not _known(shape) or math.prod(shape) > 8:
            raise ValueError('shape/axis Const is not a small tensor: ' + edge)
        data = self.constant_data(node)
        if data is not None and len(data) == math.prod(shape) * struct.calcsize(fmt):
            return list(struct.unpack('<' + fmt * math.prod(shape), data))
        raise ValueError('invalid shape/axis Const bytes: ' + edge)

    def shape(self, edge):
        if edge in self.cache:
            return self.cache[edge]
        if edge in self.active or len(self.active) >= 128:
            raise ValueError('cycle or excessive depth in static shape inference: ' + edge)
        self.active.add(edge)
        node = self.node(edge)
        kind, inputs, attrs = node['type'], node.get('input', []), _attributes(node)
        outputs = node.get('output_desc', [])
        # TorchAir serializes unknown intermediate dimensions as shape: {}.
        # An empty descriptor alone is not evidence of a rank-zero tensor.
        recorded = (outputs[0].get('shape', {}).get('dim') or None) if outputs else None
        rule = 'output_descriptor'
        # FileConstant output descriptors can be placeholders in source AIR.
        if kind == 'FileConstant':
            shape, rule = attrs.get('shape', {}).get('list', {}).get('i'), 'FileConstant.shape'
        elif _known(recorded):
            shape = recorded
        elif kind == 'Const':
            desc = attrs.get('value', {}).get('t', {}).get('desc', {})
            shape = desc.get('shape', {}).get('dim', []) if 'shape' in desc else None
            rule = 'Const.value.desc'
            if shape == []:
                types = self.schema.enums[self.schema.package + '.DataType']
                widths = {types[t]: w for t, w in (('DT_BOOL', 1), ('DT_INT32', 4),
                          ('DT_INT64', 8), ('DT_FLOAT16', 2), ('DT_FLOAT', 4))}
                data = self.constant_data(node)
                if data is None or len(data) != widths.get(desc.get('dtype')):
                    raise ValueError('scalar Const requires one typed literal value: ' + edge)
        elif kind in {'Cast', 'CastT', 'Identity', 'ZerosLike', 'LogicalNot'} and len(inputs) == 1:
            shape, rule = self.shape(inputs[0]), kind + '.input_shape'
        elif kind == 'Reshape' and len(inputs) == 2:
            if attrs.get('axis', {}).get('i', 0) != 0:
                raise ValueError('partial Reshape is not supported: ' + edge)
            shape, rule = self.integers(inputs[1]), 'Reshape.literal_shape'
            if -1 in shape:
                if shape.count(-1) != 1 or any(d <= 0 and d != -1 for d in shape):
                    raise ValueError('unsupported inferred reshape: ' + edge)
                count = math.prod(self.shape(inputs[0]))
                product = math.prod(d for d in shape if d != -1)
                if count % product:
                    raise ValueError('reshape element count mismatch: ' + edge)
                shape = [count // product if d == -1 else d for d in shape]
            elif any(d <= 0 for d in shape):
                raise ValueError('zero/dynamic reshape dimensions are not inferred: ' + edge)
        elif kind in {'Add', 'Sub', 'Mul', 'Less', 'LessEqual', 'Greater', 'GreaterEqual',
                      'Equal', 'LogicalAnd', 'LogicalOr', 'SelectV2'} and len(inputs) == (3 if kind == 'SelectV2' else 2):
            shape, rule = _broadcast([self.shape(e) for e in inputs]), kind + '.broadcast'
        elif kind == 'GatherV2' and len(inputs) == 3:
            source, indices = self.shape(inputs[0]), self.shape(inputs[1])
            axis = self.integers(inputs[2])
            if len(axis) != 1 or not -len(source) <= axis[0] < len(source) or attrs.get('batch_dims', {}).get('i', 0):
                raise ValueError('unsupported Gather axis/batch_dims: ' + edge)
            index = axis[0] % len(source)
            shape, rule = source[:index] + indices + source[index+1:], 'GatherV2.axis'
        elif kind == 'ConcatV2' and len(inputs) >= 2:
            values, axis = [self.shape(e) for e in inputs[:-1]], self.integers(inputs[-1])
            rank = len(values[0])
            if len(axis) != 1 or not -rank <= axis[0] < rank or any(len(s) != rank for s in values):
                raise ValueError('unsupported Concat shape/axis: ' + edge)
            index = axis[0] % rank
            shape = list(values[0])
            if any(s[:index]+s[index+1:] != shape[:index]+shape[index+1:] for s in values):
                raise ValueError('inconsistent Concat dimensions: ' + edge)
            shape[index] = sum(s[index] for s in values)
            rule = 'ConcatV2.axis'
        else:
            raise ValueError('cannot prove static shape for ' + edge + ' (' + kind + ')')
        if not _known(shape):
            raise ValueError('unknown/dynamic shape at ' + edge)
        self.cache[edge] = list(shape)
        self.proofs[edge] = dict(rule=rule, shape=list(shape))
        self.active.remove(edge)
        return self.cache[edge]


def _constant_node(payload, schema, shape, data):
    """Replace one-output source op by an inline literal, retaining its name."""
    fields = {f.name: f for f in schema._messages[schema.package + '.OpDef'].values()}
    shape_field = _field(schema, 'TensorDescriptor', 'shape', 11, False)
    dim_field = _field(schema, 'ShapeDef', 'dim', 3, True)
    attr_field = _field(schema, 'OpDef', 'attr', 11, True)
    desc_raw = next(v for f, _, _, v in _parts(payload) if f == fields['output_desc'].number)
    desc_raw = b''.join(e for f, _, e, _ in _parts(desc_raw) if f != shape_field)
    desc_raw += _blob(shape_field, _blob(dim_field, b''.join(_v(d) for d in shape)))
    tensor = (_blob(_field(schema, 'TensorDef', 'desc', 11, False), desc_raw)
              + _blob(_field(schema, 'TensorDef', 'data', 12, False), data))
    value = _blob(_field(schema, 'AttrDef', 't', 11, False), tensor)
    attr = (_blob(_field(schema, 'OpDef.AttrEntry', 'key', 9, False), 'value')
            + _blob(_field(schema, 'OpDef.AttrEntry', 'value', 11, False), value))
    pieces = []
    for number, _, encoded, entry in _parts(payload):
        if number in {fields['input'].number, fields['input_desc'].number}:
            continue
        if number == fields['type'].number:
            encoded = _blob(number, 'Const')
        elif number == fields['output_desc'].number:
            encoded = _blob(number, desc_raw)
        elif number == attr_field:
            key = schema.decode(entry, 'OpDef.AttrEntry')['key']
            if key in {'_input_name_key', '_input_name_value', 'value', '_readable_value'}:
                continue
        pieces.append(encoded)
    return b''.join(pieces) + _blob(attr_field, attr)


def _replace_nodes(raw, schema, replacements):
    if not replacements:
        return raw
    kind, original, graph = _decode(schema, raw)
    graph_raw, graph_field = _graph_raw(raw, schema, kind)
    _, op_field = _node_bytes(graph_raw, schema)
    rebuilt = b''.join(_blob(field, replacements[schema.decode(payload, 'OpDef')['name']])
        if field == op_field and schema.decode(payload, 'OpDef')['name'] in replacements else encoded
        for field, _, encoded, payload in _parts(graph_raw))
    updated = rebuilt if graph_field is None else b''.join(_blob(f, rebuilt) if f == graph_field else encoded
        for f, _, encoded, _ in _parts(raw))
    expected = deepcopy(original)
    expected_graph = expected if kind == 'GraphDef' else expected['graph'][0]
    expected_graph['op'] = [schema.decode(replacements[n['name']], 'OpDef') if n['name'] in replacements else n for n in graph['op']]
    if len(updated) > MAX_AIR_BYTES or _decode(schema, updated)[1] != expected:
        raise ValueError('static constant rewrite changed unrelated AIR metadata')
    return updated


def materialize_zeros(raw, schema):
    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_AIR_BYTES:
        raise ValueError('zero preparation requires AIR at most 64 MiB')
    kind, _, _ = _decode(schema, raw)
    graph_raw, _ = _graph_raw(raw, schema, kind)
    rows, _ = _node_bytes(graph_raw, schema)
    unsupported = {'src_name', 'src_index', 'dst_name', 'dst_index', 'input_i', 'output_i', 'subgraph_name'}
    if any(any(n.get(key) for key in unsupported) or
           any('g' in attr or attr.get('list', {}).get('g') for attr in _attributes(n).values())
           for n, _, _ in rows):
        raise ValueError('zero preparation requires source AIR without compiled adjacency or nested graphs')
    shapes = _Shapes(rows, schema)
    types = schema.enums[schema.package + '.DataType']
    widths = {types[t]: w for t, w in (('DT_BOOL', 1), ('DT_INT32', 4), ('DT_INT64', 8), ('DT_FLOAT16', 2), ('DT_FLOAT', 4))}
    replacements, changes, added = {}, [], 0
    for node, payload, _ in rows:
        if node['type'] != 'ZerosLike':
            continue
        name = node['name']
        if len(node.get('input', [])) != 1 or len(node.get('input_desc', [])) != 1 or len(node.get('output_desc', [])) != 1:
            raise ValueError('ZerosLike needs one data input/output without control edges: ' + name)
        desc = node['output_desc'][0]
        dtype = desc.get('dtype')
        inp = node['input_desc'][0]
        # The observed TorchAir mask has an FP32 shape carrier and an FP16
        # output descriptor (zeros_like(..., dtype=float16)). Zero is exact
        # in both types: materialize the recorded output type, without casting
        # input data or changing any consumer. Other mismatches remain errors.
        compatible = inp.get('dtype') == dtype or (inp.get('dtype'), dtype) == (types['DT_FLOAT'], types['DT_FLOAT16'])
        if (dtype not in widths or not compatible
                or any(d.get('layout', 'ND') != 'ND' for d in (inp, desc))):
            def descriptor(d):
                return dict(dtype=d.get('dtype'),
                    dtype_names=[t for t, v in types.items() if v == d.get('dtype')],
                    layout=d.get('layout'), shape=d.get('shape'))
            raise ValueError('unsupported ZerosLike dtype/layout: ' + json.dumps(dict(
                node=name, input=descriptor(inp), output=descriptor(desc)), ensure_ascii=False))
        shape = shapes.shape(node['input'][0])
        for d in (inp, desc):
            known = d.get('shape', {}).get('dim') or None
            if _known(known) and known != shape:
                raise ValueError('ZerosLike input/output shape mismatch: ' + name)
        size = math.prod(shape) * widths[dtype]
        added += size
        if added > 8 << 20:
            raise ValueError('static zero payloads exceed 8 MiB budget')
        replacements[name] = _constant_node(payload, schema, shape, bytes(size))
        changes.append(dict(node=name, source=node['input'][0], shape=shape,
                            dtype=dtype, input_dtype=inp.get('dtype'), bytes=size, replacement='Const'))
    updated = _replace_nodes(raw, schema, replacements)
    return updated, dict(status='APPLIED' if replacements else 'UNCHANGED',
        zero_count=len(replacements), embedded_bytes=added, changes=changes, shape_proofs=shapes.proofs,
        source_sha256=hashlib.sha256(raw).hexdigest(), compiler_sha256=hashlib.sha256(updated).hexdigest(),
        scope='STATIC_ZEROS_ONLY; NO_INPUT_DATA_OR_WEIGHT_READ; OMG_AND_DEVICE_NOT_VALIDATED')
