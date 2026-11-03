"""Bypass the proved FP16 unit scale on the Draft embedding compiler path."""
from copy import deepcopy
import hashlib

from qwen35_moe_dflash.target.original_air_roundtrip import _attributes, _decode, MAX_AIR_BYTES
from qwen35_moe_dflash.target.original_ge_append_wire import _blob, _field, _graph_raw, _node_bytes, _parts
from .air_zeros import _Shapes


def bypass_unit_scale(raw, schema, gather, path, scale):
    """Shared probe/full-graph rewrite; retain all other node/graph bytes."""
    scales = [row for row in path if row['type'] == 'Muls']
    if (len(scales) != 1 or scale != 1.0
            or scales[0]['input_dtype'] != 'float16' or scales[0]['output_dtype'] != 'float16'):
        raise ValueError('unit-scale comparison needs exactly one FP16 Muls with scale exactly 1')
    kind, original, graph = _decode(schema, raw)
    graph_raw, graph_field = _graph_raw(raw, schema, kind)
    rows, op_field = _node_bytes(graph_raw, schema)
    nodes = {n['name']: (n, payload) for n, payload, _ in rows}
    if len(nodes) != len(rows):
        raise ValueError('duplicate source AIR node names')
    unsupported = {'src_name', 'src_index', 'dst_name', 'dst_index', 'input_i', 'output_i', 'subgraph_name'}
    if any(any(n.get(k) for k in unsupported) or
           any('g' in a or a.get('list', {}).get('g') for a in _attributes(n).values())
           for n, _, _ in rows):
        raise ValueError('unit-scale preparation requires source AIR without compiled adjacency or nested graphs')
    name = scales[0]['name']
    mul, mul_payload = nodes[name]
    source_edge, removed_edge = gather+':0', name+':0'
    if (mul['type'] != 'Muls' or mul.get('input') != [source_edge]
            or _attributes(mul).get('value', {}).get('f') != 1.0):
        raise ValueError('unit-scale comparison must bypass the direct Gather scalar-one Muls')
    public = {e.rpartition(':')[0] if ':' in e else e
              for e in (*graph.get('input', []), *graph.get('output', []))}
    if name in public:
        raise ValueError('cannot bypass a public unit-scale node')
    # Count every port, including control uses, before removing the node.
    consumers = [(n, payload) for n, payload, _ in rows
                 for edge in n.get('input', [])
                 if edge == name or edge.rpartition(':')[0] == name]
    if (len(consumers) != 1 or consumers[0][0]['type'] != 'Cast'
            or consumers[0][0].get('input') != [removed_edge]
            or consumers[0][0]['name'] not in {n['name'] for n in path}):
        raise ValueError('unit-scale comparison requires one downstream prefix Cast consumer')
    consumer, old_payload = consumers[0]
    input_field = _field(schema, 'OpDef', 'input', 9, True)
    new_payload = b''.join(_blob(f, source_edge) if f == input_field else encoded
                           for f, _, encoded, _ in _parts(old_payload))
    pieces, index = [], 0
    for f, _, encoded, _ in _parts(graph_raw):
        if f == op_field:
            node, payload, _ = rows[index]
            index += 1
            if node['name'] == name:
                continue
            encoded = _blob(f, new_payload if node['name'] == consumer['name'] else payload)
        pieces.append(encoded)
    changed_graph = b''.join(pieces)
    updated = changed_graph if graph_field is None else b''.join(
        _blob(f, changed_graph) if f == graph_field else encoded for f, _, encoded, _ in _parts(raw))
    expected = deepcopy(original)
    expected_graph = expected if kind == 'GraphDef' else expected['graph'][0]
    expected_graph['op'] = [n for n in expected_graph['op'] if n['name'] != name]
    next(n for n in expected_graph['op'] if n['name'] == consumer['name'])['input'] = [source_edge]
    if _decode(schema, updated)[1] != expected:
        raise ValueError('unit-scale rewrite changed unrelated AIR metadata')
    return updated, dict(
        removed_node=name, removed_payload_sha256=hashlib.sha256(mul_payload).hexdigest(),
        source_edge=source_edge, consumer=consumer['name'], old_input=removed_edge, new_input=source_edge,
        consumer_before_sha256=hashlib.sha256(old_payload).hexdigest(),
        consumer_after_sha256=hashlib.sha256(new_payload).hexdigest(),
        scale=1.0, dtype='float16', equivalence='IDENTITY_FOR_FINITE_FP16_VALUES_INCLUDING_SIGNED_ZERO',
        effective_embedding_path=[n for n in path if n['name'] != name])


def fold_embedding_unit_scale(raw, schema):
    """Select only the embedding path into layer-zero candidate RMS.

    Non-unit or already direct paths stay byte-identical. Never rewrite other
    scalar operators or infer GE attribute types from protobuf enum numbers.
    """
    from .input_probe import GAMMA, _float_path, _unary_path
    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_AIR_BYTES:
        raise ValueError('embedding preparation requires AIR at most 64 MiB')
    kind, _, _ = _decode(schema, raw)
    graph_raw, _ = _graph_raw(raw, schema, kind)
    rows, _ = _node_bytes(graph_raw, schema)
    shapes = _Shapes(rows, schema)
    receipt = dict(policy='fp16-embedding-unit-scale-v1', status='UNCHANGED', unit_scale_count=0,
        source_sha256=hashlib.sha256(raw).hexdigest(), compiler_sha256=hashlib.sha256(raw).hexdigest(),
        scope='COMPILER_COPY_ONLY; NO_WEIGHT_REFERENCE_OR_ABI_CHANGE; FULL_DRAFT_DEVICE_VALIDATION_PENDING')
    matches = []
    for n, _, _ in rows:
        if n['type'] == 'RmsNorm' and len(n.get('input', [])) == 2:
            gamma, _ = _unary_path(shapes, n['input'][1], {'Cast', 'Identity'})
            if (gamma['type'] == 'FileConstant' and _attributes(gamma).get('file_path', {}).get('s')
                    == 'weight/'+GAMMA.replace('.', '_')):
                matches.append(n)
    if not matches:
        return raw, dict(receipt, reason='NO_CANDIDATE_FIRST_RMS')
    if len(matches) != 1:
        raise ValueError('cannot uniquely identify first candidate RMS for unit-scale preparation')
    gather, path = _unary_path(shapes, matches[0]['input'][0], {'Cast', 'Identity', 'Muls'})
    if not any(n['type'] == 'Muls' for n in path):
        return raw, dict(receipt, reason='NO_EMBEDDING_SCALAR')
    half = schema.enums[schema.package+'.DataType']['DT_FLOAT16']
    int32 = schema.enums[schema.package+'.DataType']['DT_INT32']
    if gather['type'] != 'GatherV2' or len(gather.get('input', [])) != 3:
        raise ValueError('candidate unit-scale path must begin at embedding GatherV2')
    table = shapes.node(gather['input'][0])
    attrs = _attributes(table)
    ins, outs = gather.get('input_desc', []), gather.get('output_desc', [])
    if (table['type'] != 'FileConstant' or table.get('input')
            or attrs.get('file_path', {}).get('s') != 'weight/embedding_weight'
            or attrs.get('dtype', {}).get('i') != 1
            or len(ins) != 3 or len(outs) != 1
            or ins[1].get('dtype') != int32
            or any(d.get('dtype') != half or d.get('layout', 'ND') != 'ND' for d in (ins[0], outs[0]))
            or shapes.integers(gather['input'][2]) != [0]
            or _attributes(gather).get('batch_dims', {}).get('i', 0) != 0):
        raise ValueError('candidate unit-scale input must be FP16 ND full-table axis-zero embedding')
    if len(shapes.shape(gather['input'][0])) != 2:
        raise ValueError('candidate embedding table must be rank two')
    shape = shapes.shape(gather['name']+':0')
    scale, proof = _float_path(shapes, gather, path, shape, allow_scale=True)
    if scale != 1.0:
        return raw, dict(receipt, reason='NON_UNIT_EMBEDDING_SCALE', scale=scale)
    updated, change = bypass_unit_scale(raw, schema, gather['name'], proof, scale)
    receipt.update(status='APPLIED', unit_scale_count=1, change=change,
                   compiler_sha256=hashlib.sha256(updated).hexdigest())
    return updated, receipt
