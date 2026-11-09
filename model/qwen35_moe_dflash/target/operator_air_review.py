"""Review source AIR connections without inferring compiled tensor storage.

Empty descriptors and FileConstant metadata can be placeholders before shape
inference. Retain both ends of an edge; never replace either with an expectation.
"""
import json

from qwen35_moe_dflash.target.original_air_inspect import _metadata_desc
from qwen35_moe_dflash.target.original_air_roundtrip import _attributes

AIR_ONLY_FAMILIES = {'moe_reduce_sum': ('Tensor.sum', ('ReduceSumCpro',)),
                    'attention_conv': ('Conv1d', ('Conv2D',))}
CHAIN_TYPES = {'Reshape', 'Identity', 'Cast', 'CastT', 'Bitcast', 'Transpose',
               'Squeeze', 'Unsqueeze'}
FOCUS_TYPES = {'ChunkGatedDeltaRule', 'LinearAttentionConvstateUpdate', 'Conv2D',
               'RmsNorm', 'CacheUpdate', 'AdnFusedInferAttention'}
DTYPES = {'DT_FLOAT': 'float32', 'DT_FLOAT32': 'float32', 'DT_FLOAT16': 'float16',
          'DT_INT8': 'int8', 'DT_INT16': 'int16', 'DT_INT32': 'int32',
          'DT_INT64': 'int64', 'DT_UINT64': 'uint64', 'DT_BOOL': 'bool',
          'DT_BF16': 'bfloat16', 'DT_BFLOAT16': 'bfloat16'}


def descriptor(raw, enums):
    result = _metadata_desc(raw, enums)
    shape = result.get('shape')
    result['shape_evidence'] = ('SOURCE_DESCRIPTOR_DIMENSIONS' if shape and all(d >= 0 for d in shape)
        else 'UNRESOLVED_EMPTY_OR_DYNAMIC_DESCRIPTOR')
    result['dtype_evidence'] = 'SOURCE_DESCRIPTOR_ONLY_NOT_COMPILED_STORAGE'
    return result


def port_names(attributes, direction):
    keys = attributes.get(f'_{direction}_name_key', {}).get('list', {}).get('s', [])
    slots = attributes.get(f'_{direction}_name_value', {}).get('list', {}).get('i', [])
    if (len(keys) != len(slots) or any(type(s) is not int or s < 0 for s in slots)
            or len(set(keys)) != len(keys) or len(set(slots)) != len(slots)):
        return {}
    return dict(zip(slots, keys))


def _edge(edge):
    if not edge or ':' not in edge:
        return None
    name, slot = edge.rsplit(':', 1)
    return (name, int(slot)) if slot.isdecimal() else None


def _data_edges(node):
    # Control edges do not consume destination data slots; empty optionals do.
    return [edge for edge in node.get('input', node.get('inputs', [])) if not edge.endswith(':-1')]


def trace_input(edge, by_name, enums, limit=8):
    """Trace data input 0 through named shape/cast operations, not arithmetic."""
    chain, seen = [], set()
    for _ in range(limit):
        parsed = _edge(edge)
        if parsed is None or parsed[0] not in by_name:
            return dict(chain=chain, stop='UNRESOLVED_EDGE')
        if parsed in seen:
            return dict(chain=chain, stop='CYCLE')
        seen.add(parsed)
        name, slot = parsed
        node = by_name[name]
        descs = node.get('output_desc', [])
        attrs = _attributes(node)
        row = dict(name=name, type=node['type'], output_slot=slot,
                   descriptor=descriptor(descs[slot] if slot < len(descs) else {}, enums))
        row['attributes'] = {k: v for k, v in attrs.items()
            if not k.startswith('_') and set(v) <= {'i', 'f', 'b', 's', 'list'}
            and len(json.dumps(v)) <= 2048}
        chain.append(row)
        if node['type'] not in CHAIN_TYPES or slot != 0:
            return dict(chain=chain, stop='PRODUCER_REACHED')
        edges = _data_edges(node)
        if not edges or not edges[0]:
            return dict(chain=chain, stop='MISSING_DATA_INPUT')
        edge = edges[0]
    return dict(chain=chain, stop='DEPTH_LIMIT')


def graph_evidence(nodes, selected, enums):
    by_name = {n['name']: n for n in nodes}
    examples, seen_types = [], set()
    consumers = {}
    for raw in nodes:
        for slot, edge in enumerate(_data_edges(raw)):
            consumers.setdefault(edge, []).append((raw, slot))
    for node in selected:
        if node['type'] not in FOCUS_TYPES or node['type'] in seen_types:
            continue
        raw = by_name[node['name']]
        names = port_names(node['attributes'], 'input')
        bindings = []
        for slot, edge in enumerate(_data_edges(raw)):
            if not edge or _edge(edge) is None:  # absent optionals and control edges
                continue
            descs = node['input_desc']
            bindings.append(dict(slot=slot, name=names.get(slot), edge=edge,
                descriptor=descs[slot] if slot < len(descs) else {},
                producer=trace_input(edge, by_name, enums)))
        seen_types.add(node['type'])
        downstream = []
        for slot in range(len(node['output_desc'])):
            uses = consumers.get(f"{node['name']}:{slot}", [])
            downstream.append(dict(output_slot=slot, count=len(uses), consumers=[
                dict(name=n['name'], type=n['type'], input_slot=i,
                     input_descriptor=descriptor(n.get('input_desc', [])[i]
                        if i < len(n.get('input_desc', [])) else {}, enums),
                     output_descriptors=[descriptor(d,enums) for d in n.get('output_desc',[])])
                for n,i in uses[:8]]))
        examples.append(dict(name=node['name'], type=node['type'], inputs=bindings,
            outputs=node['output_desc'], output_consumers=downstream,
            attributes={k:v for k,v in node['attributes'].items() if not k.startswith('_')}))
    return dict(focused_inputs=examples,
        scope='FIRST_REACHABLE_NODE_PER_ATTENTION_TYPE; SOURCE_CONNECTIONS_ONLY; NO_COMPILED_ABI_PROOF')


def _dtype(desc):
    values = {DTYPES[x] for x in desc.get('schema_dtype_names', []) if x in DTYPES}
    return next(iter(values)) if len(values) == 1 else None


def _variants(value):
    return [s.strip().lower().replace('float32', 'float').replace('bf16', 'bfloat16')
            for s in value.split(',')] if isinstance(value, str) and value else []


def declaration_differences(node, declaration):
    """Compare columns together; report declaration differences, not ABI faults."""
    fields, differences = [], []
    attrs = node['attributes']
    edges = _data_edges(node)
    input_names = port_names(attrs, 'input')
    declared_inputs = declaration.get('inputs', [])
    declared_names = {p.get('name') for p in declared_inputs}
    for slot, edge in enumerate(edges):
        name = input_names.get(slot)
        if edge and _edge(edge) and name and name not in declared_names:
            # TorchAir lists use x0/weight0/scale0; do not invent dynamic slots.
            if not any(name == f'{n}0' for n in declared_names if n):
                differences.append(dict(kind='CONNECTED_INPUT_NOT_DECLARED', port=name, slot=slot))
    for direction, metadata_key in (('input', 'input_desc'), ('output', 'output_desc')):
        names = port_names(attrs, direction)
        if not names:
            continue
        for port in declaration.get(direction + 's', []):
            slots = [i for i, n in names.items() if n == port.get('name') or n == str(port.get('name')) + '0']
            if len(slots) != 1:
                continue
            slot = slots[0]
            if direction == 'input' and (slot >= len(edges) or not edges[slot]):
                continue
            descs = node[metadata_key]
            if slot >= len(descs):
                continue
            desc = descs[slot]
            for field, actual in (('dtype', _dtype(desc)), ('format', desc.get('layout'))):
                options = _variants(port.get(field))
                if not actual or actual == 'FORMAT_RESERVED' or not options:
                    continue
                actual = _variants(actual)[0]
                record = dict(direction=direction, port=names[slot], field=field,
                    observed=actual, declared=options, shape_evidence=desc.get('shape_evidence', 'UNRESOLVED'))
                fields.append((record, options))
                if actual not in options:
                    carrier = (direction == 'input' and slot in (1, 2)
                               and node.get('family') in {'quant_matmul', 'moe_gmm'})
                    differences.append(dict(kind='PACKED_CARRIER_DECLARATION_REVIEW' if carrier
                        else 'DESCRIPTOR_DECLARATION_DIFFERENCE', **record))
    widths = {len(options) for _, options in fields} - {1}
    if len(widths) == 1 and not differences:
        width = next(iter(widths))
        compatible = [i for i in range(width) if all(r['observed'] == values[0 if len(values) == 1 else i]
                      for r, values in fields)]
        if not compatible:
            differences.append(dict(kind='NO_CORRELATED_REGISTRATION_VARIANT', fields=[r for r,_ in fields]))
    # Unknown names/missing descriptors/dynamic lists are deliberately not a compatibility PASS.
    return differences


def review_report(report):
    """Compact whole-AIR review; detailed producer traces remain in inventory."""
    registered = (report.get('ddk') or {}).get('registrations', [])
    airs = {}
    for label, air in report['airs'].items():
        groups = {}
        for node in air.get('operators', []):
            if node['type'] not in FOCUS_TYPES:
                continue
            for declaration in registered:
                if declaration['ge_type'] != node['type']:
                    continue
                diff = declaration_differences(node, declaration)
                if not diff:
                    continue
                key = json.dumps([node['type'], diff], sort_keys=True)
                group = groups.setdefault(key, dict(ge_type=node['type'], nodes=set(), declarations=set(), differences=diff))
                group['nodes'].add(node['name'])
                group['declarations'].add(declaration['path'])
        differences = [dict(ge_type=g['ge_type'], count=len(g['nodes']), examples=sorted(g['nodes'])[:3],
            declarations=sorted(g['declarations']), differences=g['differences']) for g in groups.values()]
        evidence = air.get('connection_evidence', {})
        airs[label] = dict(status=air['status'], path=air.get('path'), sha256=air.get('sha256'),
            schema_source=air.get('schema_source'), error=air.get('error'),
            reachable_type_counts=air.get('reachable_type_counts'), declaration_differences=differences,
            attention_examples=[dict(name=n['name'],type=n['type'],
                inputs=[dict(slot=p['slot'],name=p['name'],descriptor=p['descriptor'],
                    producer_chain=[dict(name=r['name'],type=r['type'],output_slot=r['output_slot'],
                        shape=r['descriptor'].get('shape'), dtype=_dtype(r['descriptor']),attributes=r['attributes'])
                        for r in p['producer']['chain']],trace_stop=p['producer']['stop']) for p in n['inputs']],
                outputs=n['outputs'],output_consumers=n['output_consumers'],attributes=n['attributes'])
                for n in evidence.get('focused_inputs',[])],
            additional_families={family: {t: air.get('reachable_type_counts', {}).get(t, 0) for t in types}
                                 for family, (_,types) in AIR_ONLY_FAMILIES.items()},
            unclassified_types=air.get('unclassified_types', []))
    return dict(schema_version=1, artifact_kind='qwen35-target-air-contract-review', airs=airs,
        status=('PARTIAL' if any(a['status']!='GRAPH_OBSERVED' for a in airs.values())
                else 'REVIEW_REQUIRED' if airs else 'AIR_NOT_REQUESTED'),
        scope='SOURCE_AIR_VS_SCANNED_DDK_DECLARATIONS; NOT_COMPILED_ABI_OR_NUMERICAL_VALIDATION',
        priority='LINEAR_ATTENTION_AND_SELF_ATTENTION_NON_MATMUL_OPERATORS',
        deferred=['quant_matmul', 'moe_gmm', 'moe_router', 'moe_init', 'moe_finalize', 'moe_reduce_sum'],
        ddk_evidence='SAVED_INVENTORY_DECLARATIONS; CURRENT_COMPILER_SELECTION_UNVERIFIED',
        packed_weight_policy='INT8/INT32/FLOAT BIT CONTAINERS DO NOT DEFINE ARITHMETIC WIDTH; CHECK UNPACKING, LAYOUT, LOGICAL K/N AND GROUP SCALES',
        scale_policy=dict(user_described_format='FP32_LIKE_WITH_10_MANTISSA_BITS',
            integer_dtype_meaning='ENCODED_PARAMETER_CARRIER_NOT_INTEGER_MAGNITUDE',
            exact_bit_layout_and_rounding='UNVERIFIED_REQUIRES_PACKER_AND_CONSUMER_CODE',
            payload_decoding='NOT_PERFORMED; DO_NOT_USE_NUMERIC_FLOAT_CAST_AS_DECODING'),
        numerical_validation='NOT_RUN', all_operators_passed=False,
        next='Inspect compiled interfaces/selected kernels and run isolated numerical probes. Empty source shapes remain unresolved.')
