"""Absorb literal rank-two FP16 weight transposes into MatMul attributes.

The compiler copy changes X @ transpose(W) into MatMul(X, W, transpose_x2).
External weight bytes, floating point types and graph outputs are untouched.
Only FileConstant weights with a proven [1, 0] permutation are eligible.
"""
from collections import defaultdict
from copy import deepcopy
import hashlib

from qwen35_moe_dflash.target.original_air_roundtrip import _attributes, _decode, MAX_AIR_BYTES
from qwen35_moe_dflash.target.original_ge_append_wire import _blob, _field, _graph_raw, _node_bytes, _parts, _v
from .air_reshape import _payload
from .air_zeros import _Shapes


def fold_weight_transposes(raw, schema):
    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_AIR_BYTES:
        raise ValueError('weight transpose preparation requires AIR at most 64 MiB')
    kind, original, graph = _decode(schema, raw)
    graph_raw, graph_field = _graph_raw(raw, schema, kind)
    rows, op_field = _node_bytes(graph_raw, schema)
    nodes = {n['name']: n for n, _, _ in rows}
    payloads = {n['name']: p for n, p, _ in rows}
    unsupported = {'src_name', 'src_index', 'dst_name', 'dst_index', 'input_i', 'output_i', 'subgraph_name'}
    if any(any(n.get(key) for key in unsupported) or
           any('g' in a or a.get('list', {}).get('g') for a in _attributes(n).values())
           for n in nodes.values()):
        raise ValueError('weight transpose preparation requires source AIR without compiled adjacency or nested graphs')
    users = defaultdict(list)
    for node in nodes.values():
        for index, edge in enumerate(node.get('input', [])):
            if edge:
                users[edge.rpartition(':')[0]].append((node, index, edge))
    public = {edge.rpartition(':')[0] if ':' in edge else edge
              for edge in (*graph.get('input', []), *graph.get('output', []))}
    literals = _Shapes(rows, schema)
    half = schema.enums[schema.package + '.DataType']['DT_FLOAT16']
    input_field = _field(schema, 'OpDef', 'input', 9, True)
    desc_field = _field(schema, 'OpDef', 'input_desc', 11, True)
    attr_field = _field(schema, 'OpDef', 'attr', 11, True)
    shape_field = _field(schema, 'TensorDescriptor', 'shape', 11, False)
    dim_field = _field(schema, 'ShapeDef', 'dim', 3, True)
    desc_attr_field = _field(schema, 'TensorDescriptor', 'attr', 11, True)
    value_field = _field(schema, 'OpDef.AttrEntry', 'value', 11, False)
    bool_field = _field(schema, 'AttrDef', 'b', 8, False)
    replacements, removed, changes, skipped = {}, set(), [], []
    expected_nodes = deepcopy(nodes)

    def fp16_nd(desc):
        return desc.get('dtype') == half and desc.get('layout', 'ND') == 'ND'

    for trans, _, _ in rows:
        if trans['type'] != 'Transpose' or len(trans.get('input', [])) != 2:
            continue
        source, permutation = trans['input']
        weight_name, _, port = source.rpartition(':')
        weight = nodes.get(weight_name, {})
        if port != '0' or weight.get('type') != 'FileConstant':
            continue
        name = trans['name']
        try:
            perm_node = literals.node(permutation)
            perm_desc = _attributes(perm_node).get('value', {}).get('t', {}).get('desc', {})
            if perm_desc.get('shape', {}).get('dim') != [2] or literals.integers(permutation) != [1, 0]:
                raise ValueError('permutation is not a literal [1, 0] vector')
            shape = _attributes(weight).get('shape', {}).get('list', {}).get('i')
            if (not isinstance(shape, list) or len(shape) != 2 or
                    any(type(d) is not int or not 0 < d <= (1 << 31) for d in shape)):
                raise ValueError('FileConstant has no static rank-two shape')
            if weight.get('input') or len(weight.get('output_desc', [])) != 1:
                raise ValueError('weight requires one output and no input/control edges')
            ins, outs = trans.get('input_desc', []), trans.get('output_desc', [])
            if len(ins) != 2 or len(outs) != 1 or not all(fp16_nd(d) for d in (ins[0], outs[0])):
                raise ValueError('Transpose data descriptors are not FP16 ND')
            for desc, required in ((ins[0], shape), (outs[0], shape[::-1])):
                recorded = desc.get('shape', {}).get('dim')
                if recorded and recorded != required:
                    raise ValueError('Transpose descriptor contradicts weight shape')
            if name in public or not users[name]:
                raise ValueError('Transpose is public or has no consumer')
            # All uses must be right-hand MatMul data ports. In particular,
            # never delete control, public, activation or non-MatMul uses.
            for consumer, index, edge in users[name]:
                inputs, descs = consumer.get('input', []), consumer.get('input_desc', [])
                attrs = _attributes(consumer)
                if (consumer['type'] != 'MatMul' or index != 1 or edge != name + ':0'
                        or len(inputs) not in (2, 3) or any(inputs[2:])
                        or len(descs) != len(inputs) or len(consumer.get('output_desc', [])) != 1
                        or not all(fp16_nd(d) for d in (*descs[:2], consumer['output_desc'][0]))
                        or any(type(attrs.get(key, {}).get('b')) is not bool
                               for key in ('transpose_x1', 'transpose_x2'))):
                    raise ValueError('all uses must be FP16 ND MatMul right operands with explicit transpose flags')
                recorded = descs[1].get('shape', {}).get('dim')
                if recorded and recorded != shape[::-1]:
                    raise ValueError('MatMul weight descriptor contradicts transposed weight shape')
        except ValueError as error:
            skipped.append(dict(node=name, reason=str(error)))
            continue

        consumers = []
        for consumer, _, _ in users[name]:
            cname = consumer['name']
            old_flag = _attributes(consumer)['transpose_x2']['b']
            updated = expected_nodes[cname]
            updated['input'][1] = source
            updated['input_desc'][1]['shape'] = dict(dim=shape)
            # _meta is a debug string describing the old transposed tensor.
            # Keep the consumer's dtype/layout/device/name and other metadata.
            desc = updated['input_desc'][1]
            if 'attr' in desc:
                desc['attr'] = [a for a in desc['attr'] if a.get('key') != '_meta']
                if not desc['attr']:
                    del desc['attr']
            for attr in updated['attr']:
                if attr['key'] == 'transpose_x2':
                    attr['value']['b'] = not old_flag
            pieces, input_index, desc_index = [], 0, 0
            for field, _, encoded, payload in _parts(payloads[cname]):
                if field == input_field:
                    if input_index == 1:
                        encoded = _blob(field, source)
                    input_index += 1
                elif field == desc_field:
                    if desc_index == 1:
                        parts = []
                        for f, _, e, p in _parts(payload):
                            if f == shape_field:
                                continue
                            if f == desc_attr_field and schema.decode(p, 'TensorDescriptor.AttrEntry').get('key') == '_meta':
                                continue
                            parts.append(e)
                        parts.append(_blob(shape_field, _blob(dim_field, b''.join(_v(d) for d in shape))))
                        encoded = _blob(field, b''.join(parts))
                    desc_index += 1
                elif field == attr_field and schema.decode(payload, 'OpDef.AttrEntry').get('key') == 'transpose_x2':
                    value = _payload(payload, schema, 'OpDef.AttrEntry', 'value')
                    value = b''.join(e for f, _, e, _ in _parts(value) if f != bool_field)
                    value += _v(bool_field << 3) + _v(int(not old_flag))
                    encoded = _blob(field, b''.join(_blob(f, value) if f == value_field else e
                        for f, _, e, _ in _parts(payload)))
                pieces.append(encoded)
            replacements[cname] = b''.join(pieces)
            consumers.append(dict(node=cname, transpose_x2_before=old_flag, transpose_x2_after=not old_flag))
        removed.add(name)
        changes.append(dict(removed=name, source=source, weight_shape=shape,
                            permutation=[1, 0], consumers=consumers))

    output = raw
    if removed:
        pieces = []
        for field, _, encoded, payload in _parts(graph_raw):
            if field == op_field:
                name = schema.decode(payload, 'OpDef')['name']
                if name in removed:
                    continue
                if name in replacements:
                    encoded = _blob(field, replacements[name])
            pieces.append(encoded)
        rebuilt = b''.join(pieces)
        output = rebuilt if graph_field is None else b''.join(_blob(f, rebuilt) if f == graph_field else e
            for f, _, e, _ in _parts(raw))
        expected = deepcopy(original)
        expected_graph = expected if kind == 'GraphDef' else expected['graph'][0]
        expected_graph['op'] = [expected_nodes[n['name']] for n, _, _ in rows if n['name'] not in removed]
        if len(output) > MAX_AIR_BYTES or _decode(schema, output)[1] != expected:
            raise ValueError('MatMul transpose rewrite changed unrelated AIR metadata')
    return output, dict(policy='fp16-fileconstant-matmul-transpose-v1',
        status='APPLIED' if removed else 'UNCHANGED', weight_transpose_count=len(removed),
        matmul_count=len(replacements), changes=changes, skipped=skipped,
        source_sha256=hashlib.sha256(raw).hexdigest(), compiler_sha256=hashlib.sha256(output).hexdigest(),
        scope='COMPILER_COPY_ONLY; NO_WEIGHT_DATA_OR_DTYPE_CHANGE; OMG_AND_DEVICE_NOT_VALIDATED')
