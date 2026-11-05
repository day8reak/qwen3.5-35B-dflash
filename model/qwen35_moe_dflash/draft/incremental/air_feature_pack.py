"""Exact C16 feature-input permutation, restricted to a diagnostic full Draft."""
from copy import deepcopy
import hashlib

POLICY = 'cpu-feature-pack-c16-v1'
LOGICAL_SHAPE = [1, 16, 16384]
PHYSICAL_SHAPE = [1, 1024, 16, 16]
CONTRACT = dict(policy=POLICY, input='target_features', dtype='float16', bytes=524288,
    logical_shape=LOGICAL_SHAPE, physical_shape=PHYSICAL_SHAPE,
    fixture_storage='ORIGINAL_LOGICAL_ROW_MAJOR_BYTES', arithmetic='NONE; FP16_WORD_PERMUTATION')


def check_contract(manifest):
    packing = manifest.get('feature_pack')
    if packing is None:
        return False
    if (packing != CONTRACT or manifest['context_rows'] != 16
            or manifest.get('optimization', {}).get('policy') != 'packed-spatial-host-embedding'):
        raise ValueError('unsupported full Draft feature-pack contract')
    return True


def physical_inputs(manifest):
    """Derive physical slots only for the explicit, exact packing contract."""
    rows = deepcopy(manifest['inputs'])
    if not check_contract(manifest):
        return rows
    selected = [r for r in rows if r['name'] == 'target_features']
    if len(selected) != 1 or selected[0]['dtype'] != 'float16' or selected[0]['shape'] != LOGICAL_SHAPE:
        raise ValueError('feature packing requires the original logical C16 input')
    selected[0]['shape'] = list(PHYSICAL_SHAPE)
    return rows


def rewrite(raw, schema, input_name):
    # Reuse the measured prefix's complete element-index proof and descriptor
    # writer. Import lazily: the probe also uses the compiler's schema loader.
    from .feature_pack_probe import Projection
    from qwen35_moe_dflash.target.original_air_roundtrip import _attributes, _decode
    from qwen35_moe_dflash.target.original_ge_append_wire import _blob, _field, _graph_raw, _parts
    trace = Projection(raw, schema)
    trace.select_projection(input_name)
    prefix, evidence = trace.extract_projection('packed')
    packed = Projection(prefix, schema)
    replacements = {n['name']: p for n, p, _ in packed.rows if n['name'] in (input_name, trace.conv['name'])}
    # Prefixes use Data.index=0. Full Draft must retain its original external
    # slot, including every original index attr byte.
    attr = _field(schema, 'OpDef', 'attr', 11, True)
    original = next(p for n, p, _ in trace.rows if n['name'] == input_name)
    def is_index(f, p):
        return f == attr and schema.decode(p, 'OpDef.AttrEntry')['key'] == 'index'
    indices = [e for f, _, e, p in _parts(original) if is_index(f, p)]
    if len(indices) != 1:
        raise ValueError('full feature Data needs one original index')
    replacements[input_name] = b''.join(e for f, _, e, p in _parts(replacements[input_name])
                                      if not is_index(f, p))+indices[0]
    removed = set(evidence['removed_nodes'])
    layout = {n for n in removed if trace.nodes[n]['type'] != 'Const'}
    # Sharing a layout value or the original feature Data elsewhere would
    # change semantics when its physical storage changes. Reject that topology.
    shared_constants = set()
    for n in trace.nodes.values():
        if n['name'] in removed:
            continue
        for i, edge in enumerate(n.get('input', [])):
            if not edge:
                continue
            parent = edge.rsplit(':', 1)[0]
            if n['name'] == trace.conv['name'] and i == 0:
                continue
            if parent in layout or parent == input_name:
                raise ValueError('feature layout has an additional consumer: '+n['name'])
            if parent in removed:
                shared_constants.add(parent)
    removed -= shared_constants
    graph_raw, _ = _graph_raw(raw, schema, trace.kind)
    ops = iter(trace.rows)
    parts = []
    for f, _, encoded, _ in _parts(graph_raw):
        if f != trace.op_field:
            parts.append(encoded)
            continue
        n, payload, _ = next(ops)
        if n['name'] not in removed:
            parts.append(_blob(f, replacements[n['name']]) if n['name'] in replacements else encoded)
    graph_raw = b''.join(parts)
    result = graph_raw if trace.graph_field is None else b''.join(
        _blob(f, graph_raw) if f == trace.graph_field else e for f, _, e, _ in _parts(raw))
    _, _, graph = _decode(schema, result)
    actual = next(n for n in graph['op'] if n['name'] == input_name)
    if _attributes(actual)['index'] != _attributes(trace.nodes[input_name])['index']:
        raise ValueError('feature Data index changed')
    return result, dict(contract=deepcopy(CONTRACT), status='APPLIED', input_data=input_name,
        conv=trace.conv['name'], removed_nodes=sorted(removed),
        input_mapping=evidence['input_mapping'], data_descriptor_contract=evidence['data_descriptor_contract'],
        source_sha256=hashlib.sha256(raw).hexdigest(), compiler_sha256=hashlib.sha256(result).hexdigest(),
        scope='FULL_GRAPH; ONLY_FEATURE_DATA_DESCRIPTORS_CONV_INPUT_AND_DEAD_INPUT_LAYOUT_CHANGED')
