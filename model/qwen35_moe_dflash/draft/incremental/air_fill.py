"""Materialize proven static Fill values in the compiler copy of Draft AIR.

The X90 CPUCL can reject Fill during model load. Read dimensions and scalar
bytes from this AIR, never from the expected model configuration. Preserve
literal bits; only exactly representable FP16/FP32 scalar conversions are
accepted. Unknown or conflicting inputs stop preparation before compilation.
"""
import hashlib
import math
import struct

from qwen35_moe_dflash.target.original_air_roundtrip import _attributes, _decode, MAX_AIR_BYTES
from qwen35_moe_dflash.target.original_ge_append_wire import _graph_raw, _node_bytes
from .air_zeros import _Shapes, _constant_node, _known, _replace_nodes


# GE API attributes use DataType, NOT ge_ir.proto's descriptor enum. Keep the
# small supported mapping explicit and separate, with its upstream definition.
_GE_TYPES_SOURCE = ('https://github.com/Ascend/torchair/blob/'
    'e1118e8b0fc1b95190ab53b732c0872989550e20/python/torchair/ge/_ge_graph.py')
_GE_TYPES = {0: 'DT_FLOAT', 1: 'DT_FLOAT16', 3: 'DT_INT32', 9: 'DT_INT64', 12: 'DT_BOOL'}


class _FillValues:
    def __init__(self, shapes):
        self.shapes = shapes
        self.types = shapes.schema.enums[shapes.schema.package + '.DataType']
        self.formats = {self.types[t]: f for t, f in (
            ('DT_BOOL', '?'), ('DT_INT32', 'i'), ('DT_INT64', 'q'),
            ('DT_FLOAT16', 'e'), ('DT_FLOAT', 'f'))}
        self.proofs = {}

    def attribute_dtype(self, node, key):
        value = _attributes(node).get(key, {}).get('i')
        if self.shapes.schema.package != 'ge.proto' or value not in _GE_TYPES:
            raise ValueError(f'unsupported source GE {key} attribute on {node["name"]}: {value}')
        return self.types[_GE_TYPES[value]]

    def descriptor_dtype(self, desc, dtype, context, *, allow_placeholder=False):
        actual = desc.get('dtype')
        if actual == dtype:
            return False
        # Internal Shape/Cast results may not receive FX meta; serialization
        # then retains an untyped/default FP32 descriptor, also copied to the
        # consumer. Resolve only these nodes from explicit semantic attributes.
        placeholder = (not desc.get('shape', {}).get('dim') and '_meta' not in _attributes(desc)
                       and actual in (None, self.types.get('DT_UNDEFINED'), self.types['DT_FLOAT']))
        if allow_placeholder and placeholder:
            return True
        raise ValueError(f'{context} dtype conflicts: descriptor={actual}, resolved={dtype}')

    def _node(self, edge, depth):
        if depth >= 32:
            raise ValueError('excessive depth or cycle in static Fill values: ' + edge)
        node = self.shapes.node(edge)
        if len(node.get('output_desc', [])) != 1 or any(
                e.endswith(':-1') for e in node.get('input', [])):
            raise ValueError('Fill value needs one output and no control inputs: ' + edge)
        if any(d.get('layout', 'ND') != 'ND' for d in
               node.get('input_desc', []) + node['output_desc']):
            raise ValueError('Fill value requires ND descriptors: ' + edge)
        return node

    def exact_cast(self, data, source, target):
        if source == target:
            return data
        integers = {self.types['DT_INT32'], self.types['DT_INT64']}
        if source in integers and target in integers:
            try:
                return struct.pack('<' + self.formats[target], struct.unpack('<' + self.formats[source], data)[0])
            except struct.error as error:
                raise ValueError('Fill integer scalar conversion is not exact') from error
        floats = {self.types['DT_FLOAT16'], self.types['DT_FLOAT']}
        if source not in floats or target not in floats:
            raise ValueError('Fill scalar dtype conversion is not supported')
        value = struct.unpack('<' + self.formats[source], data)[0]
        if math.isnan(value):
            raise ValueError('Fill scalar conversion would change a NaN payload')
        try:
            converted = struct.pack('<' + self.formats[target], value)
        except (OverflowError, struct.error) as error:
            raise ValueError('Fill scalar conversion is not exact') from error
        restored = struct.unpack('<' + self.formats[target], converted)[0]
        if restored != value or math.copysign(1, restored) != math.copysign(1, value):
            raise ValueError('Fill scalar conversion is not exact')
        return converted

    def scalar(self, edge, depth=0):
        node = self._node(edge, depth)
        dtype = node['output_desc'][0].get('dtype')
        attrs, inputs = _attributes(node), node.get('input', [])
        placeholder = False
        if node['type'] == 'Const' and not inputs:
            if dtype not in self.formats:
                raise ValueError('unsupported Fill scalar dtype: ' + edge)
            desc = attrs.get('value', {}).get('t', {}).get('desc', {})
            data = self.shapes.constant_data(node)
            if (desc.get('dtype') != dtype or desc.get('layout', 'ND') != 'ND'
                    or 'shape' not in desc or desc['shape'].get('dim', []) not in ([], [1])
                    or data is None or len(data) != struct.calcsize(self.formats[dtype])):
                raise ValueError('Fill needs one typed scalar literal: ' + edge)
            rule = 'Const.value.bytes'
        elif node['type'] in {'Identity', 'Cast'} and len(inputs) == 1:
            source, data = self.scalar(inputs[0], depth + 1)
            if len(node.get('input_desc', [])) != 1:
                raise ValueError('Fill scalar needs one input descriptor: ' + edge)
            self.descriptor_dtype(node['input_desc'][0], source, 'Fill scalar input ' + edge,
                                  allow_placeholder=self.proofs[inputs[0]]['placeholder_dtype'])
            if node['type'] == 'Cast':
                dtype = self.attribute_dtype(node, 'dst_type')
                data = self.exact_cast(data, source, dtype)
                placeholder = self.descriptor_dtype(node['output_desc'][0], dtype, 'Cast output ' + edge,
                                                     allow_placeholder=True)
                rule = 'Cast.dst_type.exact_scalar'
            else:
                self.descriptor_dtype(node['output_desc'][0], source, 'Identity output ' + edge)
                dtype, rule = source, 'Identity.scalar'
        else:
            raise ValueError('Fill value is not a static scalar: ' + edge)
        known = node['output_desc'][0].get('shape', {}).get('dim')
        if known and known != [1]:
            raise ValueError('Fill scalar output is not scalar or one element: ' + edge)
        self.proofs[edge] = dict(rule=rule, dtype=dtype, scalar_hex=data.hex(),
                                ge_dst_type=attrs.get('dst_type', {}).get('i'),
                                descriptor_dtype=node['output_desc'][0].get('dtype'),
                                placeholder_dtype=placeholder)
        return dtype, data

    def dimensions(self, edge, depth=0):
        node = self._node(edge, depth)
        dtype = node['output_desc'][0].get('dtype')
        placeholder = False
        if node['type'] == 'Shape' and 'dtype' in _attributes(node):
            dtype = self.attribute_dtype(node, 'dtype')
            placeholder = self.descriptor_dtype(node['output_desc'][0], dtype, 'Shape output ' + edge,
                                                 allow_placeholder=True)
        if dtype not in {self.types['DT_INT32'], self.types['DT_INT64']}:
            raise ValueError(f'Fill dimensions need INT32 or INT64: {edge}, descriptor={node["output_desc"][0].get("dtype")}, attrs={_attributes(node)}')
        inputs = node.get('input', [])
        if node['type'] == 'Const' and not inputs:
            literal = _attributes(node).get('value', {}).get('t', {}).get('desc', {})
            shape = self.shapes.integers(edge)
            if literal.get('dtype') != dtype or literal.get('shape', {}).get('dim') != [len(shape)]:
                raise ValueError('Fill dimensions require a literal vector: ' + edge)
            rule = 'Const.integer_vector'
        elif node['type'] == 'Shape' and len(inputs) == 1:
            shape = self.shapes.shape(inputs[0])
            rule = 'Shape.static_input_shape'
            if len(node.get('input_desc', [])) != 1:
                raise ValueError('Fill Shape needs one input descriptor: ' + edge)
            recorded = node['input_desc'][0].get('shape', {}).get('dim') or None
            if recorded is not None and recorded != shape:
                raise ValueError('Fill Shape input descriptor conflicts: ' + edge)
        elif node['type'] == 'Identity' and len(inputs) == 1:
            if len(node.get('input_desc', [])) != 1 or node['input_desc'][0].get('dtype') != dtype:
                raise ValueError('Fill dimensions Identity changes dtype: ' + edge)
            shape = self.dimensions(inputs[0], depth + 1)
            rule = 'Identity.dimensions'
        else:
            raise ValueError('cannot prove static Fill dimensions: ' + edge)
        if not _known(shape) or (dtype == self.types['DT_INT32'] and any(d >= 1 << 31 for d in shape)):
            raise ValueError('Fill has invalid/static dimension range: ' + edge)
        recorded = node['output_desc'][0].get('shape', {}).get('dim') or None
        if recorded is not None and recorded != [len(shape)]:
            raise ValueError('Fill dimensions vector descriptor conflicts: ' + edge)
        self.proofs[edge] = dict(rule=rule, dtype=dtype, dimensions=shape,
                                ge_shape_dtype=_attributes(node).get('dtype', {}).get('i'),
                                descriptor_dtype=node['output_desc'][0].get('dtype'),
                                placeholder_dtype=placeholder)
        return shape


def materialize_fills(raw, schema):
    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_AIR_BYTES:
        raise ValueError('Fill preparation requires AIR at most 64 MiB')
    kind, _, graph = _decode(schema, raw)
    graph_raw, _ = _graph_raw(raw, schema, kind)
    rows, _ = _node_bytes(graph_raw, schema)
    unsupported = {'src_name', 'src_index', 'dst_name', 'dst_index', 'input_i', 'output_i', 'subgraph_name'}
    if any(any(n.get(key) for key in unsupported) or
           any('g' in attr or attr.get('list', {}).get('g') for attr in _attributes(n).values())
           for n, _, _ in rows):
        raise ValueError('Fill preparation requires source AIR without compiled adjacency or nested graphs')
    shapes = _Shapes(rows, schema)
    values = _FillValues(shapes)
    if len(shapes.nodes) != len(rows):
        raise ValueError('duplicate AIR node names')
    refs = list(graph.get('output', [])) + [e for n, _, _ in rows for e in n.get('input', [])]
    replacements, changes, added = {}, [], 0
    for node, payload, _ in rows:
        if node['type'] != 'Fill':
            continue
        name = node['name']
        try:
            inputs, outputs = node.get('input_desc', []), node.get('output_desc', [])
            if len(node.get('input', [])) != 2 or len(inputs) != 2 or len(outputs) != 1:
                raise ValueError('Fill needs two data inputs and one output')
            if any(e.rpartition(':')[0] == name and not e.endswith(':0') for e in refs):
                raise ValueError('Fill has a non-data-output consumer')
            if any(d.get('layout', 'ND') != 'ND' for d in inputs + outputs):
                raise ValueError('Fill requires ND layout')
            shape = values.dimensions(node['input'][0])
            dtype, scalar = values.scalar(node['input'][1])
            for desc, edge in zip(inputs, node['input']):
                proof = values.proofs[edge]
                values.descriptor_dtype(desc, proof['dtype'], 'Fill input',
                                        allow_placeholder=proof['placeholder_dtype'])
            for desc, expected in ((inputs[0], [len(shape)]), (inputs[1], [1]), (outputs[0], shape)):
                recorded = desc.get('shape', {}).get('dim') or None
                if recorded is not None and recorded != expected:
                    raise ValueError('Fill descriptor shape conflicts with static values')
            output_dtype = outputs[0].get('dtype')
            output_scalar = values.exact_cast(scalar, dtype, output_dtype)
            size = math.prod(shape) * len(output_scalar)
            added += size
            if added > 8 << 20:
                raise ValueError('static Fill payloads exceed 8 MiB budget')
            replacements[name] = _constant_node(payload, schema, shape, output_scalar * math.prod(shape))
            changes.append(dict(node=name, inputs=node['input'], shape=shape,
                input_dtype=dtype, dtype=output_dtype, source_scalar_hex=scalar.hex(),
                scalar_hex=output_scalar.hex(), bytes=size, replacement='Const'))
        except ValueError as error:
            raise ValueError(f'static Fill preparation failed for {name}: {error}') from error
    updated = _replace_nodes(raw, schema, replacements)
    return updated, dict(status='APPLIED' if replacements else 'UNCHANGED',
        fill_count=len(replacements), embedded_bytes=added, changes=changes,
        ge_attribute_types_source=_GE_TYPES_SOURCE,
        value_proofs=values.proofs, shape_proofs=shapes.proofs,
        source_sha256=hashlib.sha256(raw).hexdigest(), compiler_sha256=hashlib.sha256(updated).hexdigest(),
        scope='STATIC_FILL_LITERALS_ONLY; NO_WEIGHT_CHANGE; OMG_AND_DEVICE_NOT_VALIDATED')
