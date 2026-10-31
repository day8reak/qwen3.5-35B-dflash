"""A narrow state-owner naming contract for this original CPro target pair.

The contract uses the parameter FQN spelling retained in FileConstant paths,
same whole-file content pins and actual named data paths. It does not attest
the original Python exporter, interpret integer FileConstant dtype domains,
establish numerical equivalence, or bind compiled/runtime state files.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from pathlib import PurePosixPath
import re

from qwen35_moe_dflash.target import state_lineage as lineage


CONTRACT = 'QWEN35_CPRO_PARAMETER_FQN_OWNER_V1'
_FQN = re.compile(r'model_model_layers_(0|[1-9]|[12][0-9]|3[0-9])_(linear_attn|self_attn)_(in_proj_qkv|in_proj_b|k_proj|v_proj)_(W_q|scale)\Z')
_LIMITS = {'maximum_depth': 32, 'maximum_edges': 1024, 'maximum_total_enqueued_edges': 4096}
_ROLES = {'LinearAttentionConvstateUpdate': ('x1', ('in_proj_qkv',), 'conv', (8192, 4)),
    'ChunkGatedDeltaRule': ('beta', ('in_proj_b',), 'recurrent', (32, 128, 128)),
    'CacheUpdate': ('updates', ('k_proj', 'v_proj'), None, (64, 32, 32, 16))}


def _file_label(anchor):
    match = _FQN.fullmatch(PurePosixPath(anchor['normalized_relative_path']).name)
    if match is None:
        return None
    layer, module, family, part = int(match[1]), match[2], match[3], match[4]
    expected_module = 'self_attn' if layer % 4 == 3 else 'linear_attn'
    if module != expected_module or (module == 'self_attn') != (family in {'k_proj', 'v_proj'}):
        return None
    return layer, module, family, part


def _raw_tensor(node):
    attrs = lineage._attributes(node)
    if any(re.search(r'offset|slice|length|byte_range', key, re.I) for key in attrs):
        raise ValueError('FileConstant offset/range attributes are unsupported')
    semantic = {key: value for key, value in attrs.items() if not key.startswith('_') and key != 'file_path'}
    if set(semantic) - {'shape', 'dtype', 'file_id'}:
        raise ValueError('unsupported FileConstant semantic fields or offsets: ' + ','.join(sorted(set(semantic) - {'shape', 'dtype', 'file_id'})))
    shape = attrs.get('shape', {}).get('list', {}).get('i')
    dtype = attrs.get('dtype', {}).get('i')
    if (not isinstance(shape, list) or not 1 <= len(shape) <= 8
            or any(type(value) is not int or value <= 0 for value in shape)
            or type(dtype) is not int or dtype < 0):
        raise ValueError('FileConstant requires explicit positive raw shape and integer raw dtype attributes')
    file_id = attrs.get('file_id')
    if file_id is not None and (not isinstance(file_id, dict) or set(file_id) - {'s'} or file_id.get('s', '') != ''):
        raise ValueError('FileConstant file_id indirection is unsupported')
    return {'shape_attribute': attrs['shape'], 'dtype_attribute': attrs['dtype'],
        'file_id_attribute': {'present': 'file_id' in attrs, 'value': file_id},
        'offset_or_slice': 'NONE_DECLARED', 'integer_dtype_domain': 'NOT_RESOLVED_NOT_TENSOR_DESCRIPTOR_ENUM',
        'tensor_extent': 'UNVERIFIED_RAW_ATTRIBUTES_COMPARED_ONLY'}


def _common_projection(weight_path, scale_path):
    """A single actual operator must name both the weight and scale branches."""
    selected = []
    for weight in weight_path:
        if weight.get('input_port_name') != 'x2':
            continue
        for scale in scale_path:
            if (scale.get('input_port_name') == 'scale' and weight['node'] == scale['node']
                    and weight['output_port'] == scale['output_port']):
                selected.append({'node': weight['node'], 'type': weight['type'],
                    'output_port': weight['output_port'], 'weight_port_name': 'x2', 'scale_port_name': 'scale'})
    if len(selected) != 1:
        raise ValueError('parameter pair does not share one explicit x2/scale projection')
    return selected[0]


def _search(nodes, constants, edge, family):
    queue = deque([(edge, [])])
    seen, found, repeated = set(), defaultdict(dict), 0
    enqueued, best, frontier, truncated = 1, None, None, False
    terminals = Counter()
    while queue:
        current, path = queue.popleft()
        depth = len(path) + 1
        if best is not None and depth > best:
            break
        if current in seen:
            repeated += 1
            continue
        if len(seen) >= _LIMITS['maximum_edges']:
            frontier, truncated = depth, True
            break
        seen.add(current)
        node, port = lineage._edge(current, nodes)
        step = {'node': node['name'], 'type': node['type'], 'output_port': port}
        if node['type'] == 'FileConstant':
            anchor = constants[node['name']]
            label = _file_label(anchor)
            if label is not None and label[2] == family:
                key, part = label[:3], label[3]
                found[key].setdefault(part, {'anchor': anchor, 'path': path + [step], 'depth': depth})
                if set(found[key]) == {'W_q', 'scale'}:
                    distance = max(row['depth'] for row in found[key].values())
                    best = distance if best is None else min(best, distance)
            continue
        if node['type'] in {'Data', 'RefData', *lineage._PAIRS}:
            terminals['ROOT_OR_OTHER_STATE_UPDATER'] += 1
            continue
        if depth >= _LIMITS['maximum_depth']:
            terminals['DEPTH_LIMIT'] += 1
            frontier = depth + 1 if frontier is None else min(frontier, depth + 1)
            truncated = True
            continue
        try:
            named = {index: name for name, index in lineage._port_map(node, 'input').items()}
        except ValueError:
            named = {}
        for index, source in enumerate(lineage._inputs(node)):
            if not source:
                continue
            if enqueued >= _LIMITS['maximum_total_enqueued_edges']:
                terminals['ENQUEUE_LIMIT'] += 1
                frontier = depth + 1 if frontier is None else min(frontier, depth + 1)
                truncated = True
                break
            queue.append((source, path + [dict(step, input_port=index, input_port_name=named.get(index))]))
            enqueued += 1
        if terminals['ENQUEUE_LIMIT']:
            # Earlier queued paths may be shallower than the edge that failed
            # to enqueue. Their uncertainty cannot be hidden by a known pair.
            if queue: frontier = min(frontier, len(queue[0][1]) + 1)
            break
    candidates, rejected = [], []
    for key, parts in found.items():
        if set(parts) != {'W_q', 'scale'}:
            continue
        distance = max(row['depth'] for row in parts.values())
        if best != distance:
            continue
        try:
            projection = _common_projection(parts['W_q']['path'], parts['scale']['path'])
        except ValueError as error:
            if len(rejected) < 8: rejected.append({'layer': key[0], 'family': key[2], 'reason': str(error)})
            continue
        candidates.append({'layer': key[0], 'module': key[1], 'family': key[2], 'distance': distance,
            'projection': projection, 'parameters': parts})
    return {'family': family, 'candidates': candidates, 'rejected_pairs': rejected, 'truncated': truncated,
        'minimum_unexpanded_distance': frontier, 'visited_edges': len(seen), 'enqueued_edges': enqueued,
        'repeated_edges': repeated, 'terminals': dict(terminals), 'budgets': dict(_LIMITS)}


def _search_summary(search):
    return {**{key: value for key, value in search.items() if key != 'candidates'},
        'candidate_count': len(search['candidates']), 'candidate_summaries': [
            {'layer': row['layer'], 'family': row['family'], 'distance': row['distance'], 'projection': row['projection'],
             'references': {part: value['anchor']['reference'] for part, value in row['parameters'].items()}}
            for row in search['candidates']]}


def analyze_parameter_fqn_owners(model_nodes, models, constants, enums):
    """Bind source owners under this explicit naming/port contract only."""
    labels, contents, raw, invalid = {}, {}, {}, {}
    for model, selected in constants.items():
        labels[model], contents[model], raw[model], invalid[model] = defaultdict(list), defaultdict(set), {}, {}
        for node_name, anchor in selected.items():
            label = _file_label(anchor)
            contents[model][(anchor['sha256'], anchor['bytes'])].add(anchor['normalized_relative_path'])
            if label is None: continue
            labels[model][label].append(node_name)
            try: raw[model][node_name] = _raw_tensor(model_nodes[model][node_name])
            except (ValueError, TypeError, KeyError) as error: invalid[model][node_name] = str(error)
    observed = {}
    for model, description in models.items():
        nodes, rows = model_nodes[model], []
        for state in description['states']:
            row = {'source_refdata': state.get('source_refdata'), 'status': 'UNVERIFIED'}
            try:
                if state.get('status') != 'NAMED_PORT_STATE_LINEAGE_OBSERVED':
                    raise ValueError('source named-port state owner is unverified')
                update = state['updater']; control, families, role, shape = _ROLES[update['type']]
                expected_view = ('FLOAT16', shape)
                actual_view = (state['source_netoutput_view']['dtype'], tuple(state['source_netoutput_view']['shape_without_edge_singletons']))
                if actual_view != expected_view or lineage._view(nodes[state['source_refdata']]['output_desc'][0], enums[model]) != expected_view:
                    raise ValueError('source state input/output view differs from the model-specific cache role')
                index = update['serialized_input_port_map'].get(control)
                if type(index) is not int:
                    raise ValueError('missing explicit model-specific updater input port: ' + control)
                edge = lineage._inputs(nodes[update['node']])[index]
                if not edge: raise ValueError('model-specific updater input is empty')
                searches = [_search(nodes, constants[model], edge, family) for family in families]
                row.update(updater_node=update['node'], updater_type=update['type'], owner_input_port=control,
                    searches=[_search_summary(search) for search in searches])
                candidates = [candidate for search in searches for candidate in search['candidates']]
                nearest = min((value['distance'] for value in candidates), default=None)
                candidates = [value for value in candidates if value['distance'] == nearest]
                if len(candidates) != 1:
                    raise ValueError('no unique nearest complete learned parameter family pair')
                if any(search['truncated'] and search['minimum_unexpanded_distance'] <= nearest for search in searches):
                    raise ValueError('a search budget can hide a nearer or equal learned parameter family')
                selected = candidates[0]
                cache_role = role or ('key' if selected['family'] == 'k_proj' else 'value')
                expected_module = 'self_attn' if cache_role in {'key', 'value'} else 'linear_attn'
                if selected['module'] != expected_module:
                    raise ValueError('learned family module conflicts with the state cache role')
                for part, value in selected['parameters'].items():
                    anchor, node_name = value['anchor'], value['anchor']['node']
                    label = (selected['layer'], selected['module'], selected['family'], part)
                    content = (anchor['sha256'], anchor['bytes'])
                    if (len(labels[model][label]) != 1 or node_name not in raw[model]
                            or any(len(contents[name].get(content, ())) != 1 for name in models)):
                        raise ValueError(invalid[model].get(node_name, 'parameter filename or whole-file content is not unique/common'))
                    value['raw_tensor_attributes'] = raw[model][node_name]
                row.update(status='MODEL_PARAMETER_FQN_SOURCE_OWNER_OBSERVED', layer=selected['layer'],
                    cache_role=cache_role, owner_key=f"layer.{selected['layer']}.{cache_role}", binding=selected)
            except (ValueError, TypeError, KeyError, IndexError) as error:
                row['reason'] = str(error)
            rows.append(row)
        observed[model] = rows
    grouped = {name: defaultdict(list) for name in models}
    for name, rows in observed.items():
        for row in rows:
            if row['status'] == 'MODEL_PARAMETER_FQN_SOURCE_OWNER_OBSERVED': grouped[name][row['owner_key']].append(row)
    source_preconditions = {name: (model['source_owner_count'] == model['unique_source_owner_count'] == 80
        and len(model['states']) == 80 and {row.get('source_refdata') for row in model['states']} == lineage._STATES
        and all(row.get('status') == 'NAMED_PORT_STATE_LINEAGE_OBSERVED' for row in model['states']))
        for name, model in models.items()}
    mapping, conflicts = [], []
    for key in grouped['prefill'].keys() & grouped['decode'].keys():
        left, right = grouped['prefill'][key], grouped['decode'][key]
        if len(left) != 1 or len(right) != 1:
            conflicts.append({'owner_key': key, 'reason': 'duplicate model-specific source owners'})
            continue
        a, b = left[0], right[0]
        same = a['binding']['projection']['type'] == b['binding']['projection']['type']
        for part in ('W_q', 'scale'):
            aa, bb = a['binding']['parameters'][part], b['binding']['parameters'][part]
            same = same and aa['raw_tensor_attributes'] == bb['raw_tensor_attributes'] and all(aa['anchor'][field] == bb['anchor'][field] for field in ('bytes', 'sha256'))
        if not same:
            conflicts.append({'owner_key': key, 'reason': 'cross-export raw FileConstant attributes, whole-file content or projection types differ'})
            continue
        mapping.append({'owner_key': key, 'layer': a['layer'], 'cache_role': a['cache_role'],
            'prefill_refdata': a['source_refdata'], 'decode_refdata': b['source_refdata'],
            'status': 'SOURCE_OWNER_MATCHED_UNDER_PARAMETER_FQN_CONTRACT'})
    if not all(source_preconditions.values()):
        mapping = []
        conflicts.append({'reason': 'complete unique named-port source owner preconditions are not satisfied'})
    expected = {f'layer.{layer}.{role}' for layer in range(40) for role in (('key', 'value') if layer % 4 == 3 else ('conv', 'recurrent'))}
    complete = (len(mapping) == 80 and {row['owner_key'] for row in mapping} == expected
        and all(model['source_owner_count'] == model['unique_source_owner_count'] == 80 for model in models.values()))
    return {'contract': CONTRACT, 'status': 'ALL80_SOURCE_OWNERS_MATCHED_UNDER_PARAMETER_FQN_CONTRACT' if complete else 'PARAMETER_FQN_OWNER_CONTRACT_OBSERVED_WITH_GAPS',
        'matched_owner_count': len(mapping), 'expected_owner_count': 80, 'mapping': sorted(mapping, key=lambda row: (row['layer'], row['cache_role'])),
        'complete_named_port_source_owner_preconditions': source_preconditions,
        'observed_owner_count_by_model': {name: sum(row['status'] == 'MODEL_PARAMETER_FQN_SOURCE_OWNER_OBSERVED' for row in rows) for name, rows in observed.items()},
        'models': observed, 'conflicts': conflicts, 'missing_owner_keys': sorted(expected - {row['owner_key'] for row in mapping}),
        'tensor_numeric_dtype_and_extent': 'UNVERIFIED_INTEGER_ATTRIBUTE_DOMAIN_NOT_RESOLVED',
        'original_python_fqn_to_filename_attestation': 'NOT_AVAILABLE', 'compiled_runtime_file_binding': 'NOT_EVALUATED',
        'feedback_ready': False, 'numerical_validation': 'NOT_RUN', 'weight_payload_access': 'NONE',
        'scope': 'Model-specific retained FileConstant filename convention, raw attributes, whole-file pins and actual named updater/projection data ports. This does not attest Python module FQNs, tensor numeric layout, numerical equivalence, or state feedback through compiled/runtime files.'}
