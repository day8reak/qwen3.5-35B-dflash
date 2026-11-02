"""Fold proven static Reshape chains in a compiler-only PyTorch AIR copy.

This avoids asking OMG's ReshapeFusion pass to delete those intermediate
nodes. It is an exact graph simplification, not a claim about the cause of an
OMG ByPassNode failure. No layout conversion, dtype or arithmetic is changed.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import math
import struct

from qwen35_moe_dflash.target.original_air_roundtrip import _attributes, _decode, MAX_AIR_BYTES
from qwen35_moe_dflash.target.original_ge_append_wire import _blob, _field, _graph_raw, _node_bytes, _parts


def _payload(raw, schema, message, field):
    definition = next(f for f in schema._messages[schema.package + '.' + message].values() if f.name == field)
    values = [value for number, wire, _, value in _parts(raw) if number == definition.number and wire == 2]
    return values[0] if len(values) == 1 else None


def _shape_constant(node, raw, schema):
    if node.get('type') != 'Const' or node.get('input'):
        return None
    tensor = _attributes(node).get('value', {}).get('t', {})
    desc = tensor.get('desc', {})
    dims = desc.get('shape', {}).get('dim', [])
    types = schema.enums[schema.package + '.DataType']
    formats = {types['DT_INT32']: 'i', types['DT_INT64']: 'q'}
    fmt = formats.get(desc.get('dtype'))
    if not fmt or len(dims) != 1 or not 1 <= dims[0] <= 8:
        return None
    number = _field(schema, 'OpDef', 'attr', 11, True)
    for field, _, _, entry in _parts(raw):
        if field != number:
            continue
        if schema.decode(entry, 'OpDef.AttrEntry').get('key') != 'value':
            continue
        value = _payload(entry, schema, 'OpDef.AttrEntry', 'value')
        tensor_raw = _payload(value, schema, 'AttrDef', 't')
        data = _payload(tensor_raw, schema, 'TensorDef', 'data')
        if data is None or len(data) != dims[0] * struct.calcsize(fmt):
            return None
        shape = list(struct.unpack('<' + fmt * dims[0], data))
        if shape.count(-1) > 1 or any(d == 0 or d < -1 for d in shape):
            return None
        return shape
    return None


def _compatible_shapes(a, b):
    # A fully specified shape anchors the element count. A single inferred
    # dimension is allowed only when that count divides its known dimensions.
    count = next((math.prod(s) for s in (a, b) if -1 not in s), None)
    if count is None:
        return False
    for shape in (a, b):
        product = math.prod(d for d in shape if d > 0)
        if (count % product if -1 in shape else count != product):
            return False
    return True


def fold_reshape_chains(raw, schema):
    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_AIR_BYTES:
        raise ValueError('reshape preparation requires a nonempty AIR at most 64 MiB')
    kind, original, graph = _decode(schema, raw)
    graph_raw, graph_field = _graph_raw(raw, schema, kind)
    rows, op_field = _node_bytes(graph_raw, schema)
    nodes = {n['name']: deepcopy(n) for n, _, _ in rows}
    payloads = {n['name']: p for n, p, _ in rows}
    # Compiled adjacency metadata and nested graphs need a different rewrite.
    # Source TorchAir AIR uses explicit name:port edges in a single root graph.
    unsupported = {'src_name', 'src_index', 'dst_name', 'dst_index', 'input_i', 'output_i', 'subgraph_name'}
    if any(any(n.get(key) for key in unsupported) or
           any('g' in attr or attr.get('list', {}).get('g') for attr in _attributes(n).values())
           for n in nodes.values()):
        raise ValueError('reshape preparation requires source AIR without compiled adjacency or nested graphs')
    uses = Counter(edge for node in nodes.values() for edge in node.get('input', []) if edge)
    producer_uses = Counter()
    for edge, count in uses.items():
        producer_uses[edge.rpartition(':')[0]] += count
    public = set(graph.get('input', [])) | set(graph.get('output', []))
    input_field = _field(schema, 'OpDef', 'input', 9, True)
    desc_field = _field(schema, 'OpDef', 'input_desc', 11, True)
    shapes = {}
    for name, node in nodes.items():
        if node.get('type') != 'Reshape' or len(node.get('input', [])) != 2:
            continue
        # Full reshape only; reject shape-copy (zero), dynamic shape inputs,
        # extra/control ports, dtype changes and physical layout descriptors.
        if _attributes(node).get('axis', {}).get('i', 0) != 0:
            continue
        ins, outs = node.get('input_desc', []), node.get('output_desc', [])
        if (len(ins) != 2 or len(outs) != 1 or ins[0].get('dtype') is None
                or ins[0].get('dtype') != outs[0].get('dtype')
                or any(d.get('layout', 'ND') != 'ND' for d in (ins[0], outs[0]))):
            continue
        shape_name, _, port = node['input'][1].rpartition(':')
        if port == '0' and shape_name in nodes:
            shape = _shape_constant(nodes[shape_name], payloads[shape_name], schema)
            if shape:
                shapes[name] = shape
    descriptors = {name: next(v for f, _, _, v in _parts(payloads[name]) if f == desc_field) for name in shapes}
    name_field = _field(schema, 'TensorDescriptor', 'name', 9, False)
    removed, changes = set(), []
    # Walk from each child towards its parent: order independent, and consumes
    # a chain completely without retaining stale references to removed nodes.
    for name in shapes:
        if name in removed:
            continue
        child = nodes[name]
        seen = {name}
        while True:
            edge = child['input'][0]
            parent_name, _, port = edge.rpartition(':')
            if port != '0' or parent_name not in shapes or parent_name in removed:
                break
            if parent_name in seen:
                raise ValueError('cycle in AIR Reshape chain')
            seen.add(parent_name)
            parent = nodes[parent_name]
            if (producer_uses[parent_name] != 1 or edge in public or parent_name in public
                    or not _compatible_shapes(shapes[parent_name], shapes[name])
                    or parent['output_desc'][0]['dtype'] != child['input_desc'][0]['dtype']):
                break
            source = parent['input'][0]
            source_name, _, source_port = source.rpartition(':')
            if (not source_port.isdecimal() or source_name not in nodes or source_name in removed
                    or int(source_port) >= len(nodes[source_name].get('output_desc', []))):
                break
            # Retain the child's final shape, output descriptor, name and all
            # attributes. Its new data descriptor comes from the parent input.
            descriptor = deepcopy(parent['input_desc'][0])
            if 'name' in child['input_desc'][0]:
                descriptor['name'] = child['input_desc'][0]['name']
            elif 'name' in descriptor:
                descriptor.pop('name')
            child['input'][0] = source
            child['input_desc'][0] = descriptor
            desc_raw = b''.join(encoded for f, _, encoded, _ in _parts(descriptors[parent_name]) if f != name_field)
            if 'name' in descriptor:
                desc_raw += _blob(name_field, descriptor['name'])
            descriptors[name] = desc_raw
            uses[edge] -= 1
            producer_uses[parent_name] -= 1
            # The source use moves from parent to child, so its count is stable.
            uses[parent['input'][1]] -= 1
            producer_uses[parent['input'][1].rpartition(':')[0]] -= 1
            removed.add(parent_name)
            changes.append(dict(removed=parent_name, retained=name, source=source,
                                intermediate_shape=shapes[parent_name], final_shape=shapes[name]))

    changed_names = {c['retained'] for c in changes} - removed
    replacements = {}
    for name in changed_names:
        child = nodes[name]
        pieces, input_index, desc_index = [], 0, 0
        for field, _, encoded, _ in _parts(payloads[name]):
            if field == input_field:
                if input_index == 0:
                    encoded = _blob(field, child['input'][0])
                input_index += 1
            elif field == desc_field:
                if desc_index == 0:
                    encoded = _blob(field, descriptors[name])
                desc_index += 1
            pieces.append(encoded)
        replacements[name] = b''.join(pieces)

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
        updated_graph = b''.join(pieces)
        output = updated_graph if graph_field is None else b''.join(
            _blob(field, updated_graph) if field == graph_field else encoded
            for field, _, encoded, _ in _parts(raw))
        expected = deepcopy(original)
        expected_graph = expected if kind == 'GraphDef' else expected['graph'][0]
        expected_graph['op'] = [nodes[n['name']] for n, _, _ in rows if n['name'] not in removed]
        if _decode(schema, output)[1] != expected:
            raise ValueError('AIR reshape rewrite changed metadata outside the declared edges/descriptors')
    return output, dict(policy='static-single-consumer-reshape-chain-v1',
        status='APPLIED' if removed else 'UNCHANGED', removed_count=len(removed), changes=changes,
        source_sha256=hashlib.sha256(raw).hexdigest(), compiler_sha256=hashlib.sha256(output).hexdigest(),
        scope='COMPILER_COPY_ONLY; STATIC_RESHAPE_COMPOSITION; OMG_AND_DEVICE_NOT_VALIDATED')
