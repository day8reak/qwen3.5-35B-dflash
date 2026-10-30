"""Named role/parameter-FQN owners, with actual-style unmaterialized constants."""
from copy import deepcopy
import hashlib

import pytest

from qwen35_moe_dflash import target_state_correspondence as audit
from qwen35_moe_dflash import target_parameter_fqn_owner as owner
from qwen35_moe_dflash import target_state_lineage as lineage
from test_target_state_lineage import graph, desc, named, attr, ENUMS


def source(sequence, prefix, *, permute=False, deep_other_roles=True):
    document = graph(sequence)
    ops = document['graph'][0]['op']; nodes = {row['name']: row for row in ops}
    files = []
    for index in range(80):
        group, slot = divmod(index, 8)
        layer = group * 4 + (slot // 2 if slot < 6 else 3)
        family = ('in_proj_qkv' if slot % 2 == 0 else 'in_proj_b') if slot < 6 else ('k_proj' if slot == 6 else 'v_proj')
        module = 'linear_attn' if slot < 6 else 'self_attn'
        constants = []
        for part in ('W_q', 'scale'):
            reference = f'{prefix}/model_model_layers_{layer}_{module}_{family}_{part}'
            node = {'name': f'{prefix}_tensor_{index}_{part}', 'type': 'FileConstant',
                'attr': [attr('file_path', {'s': reference}), attr('shape', {'list': {'i': [16, 32]}}),
                    attr('dtype', {'i': 3}), attr('file_id', {'s': ''})],
                # Observed source TensorDescriptor placeholders are deliberately
                # FLOAT32+shape{}, independent from integer attr dtype3.
                'output_desc': [{'dtype': 78, 'shape': {}}]}
            ops.append(node); constants.append(node)
            files.append({'reference': reference, 'relative_path': reference,
                'nodes': [node['name']], 'bytes': 2048 if part == 'W_q' else 128,
                'sha256': hashlib.sha256(f'{layer}:{family}:{part}'.encode()).hexdigest()})
        projection = named({'name': f'{prefix}_projection_{index}', 'type': 'QuantBatchMatmulV4444',
            'input': ['arg1_1:0', constants[0]['name'] + ':0', constants[1]['name'] + ':0'],
            'output_desc': [desc([1])]}, {'x1': 0, 'x2': 1, 'scale': 2}, {'y': 0})
        ops.append(projection)
        edge = projection['name'] + ':0'
        length = 18 if slot == 6 else 2 if slot % 2 and slot < 6 else 1
        for step in range(length):
            carrier = {'name': f'{prefix}_carrier_{index}_{step}', 'type': 'Sigmoid' if step == 1 and slot < 6 and slot % 2 else 'Reshape',
                'input': [edge], 'output_desc': [desc([1])]}
            ops.append(carrier); edge = carrier['name'] + ':0'
        update = nodes[f'update_{index}']; state_in, state_out = lineage._PAIRS[update['type']]
        control = 'x1' if slot < 6 and slot % 2 == 0 else 'beta' if slot < 6 else 'updates'
        update['input'][0] = edge
        named(update, {control: 0, state_in: 1}, {state_out: 0})
        if deep_other_roles:
            noise = 'arg1_1:0'
            for step in range(40):
                node = {'name': f'{prefix}_unrelated_{index}_{step}', 'type': 'Identity', 'input': [noise], 'output_desc': [desc([1])]}
                ops.append(node); noise = node['name'] + ':0'
            update['input'].append(noise)
            named(update, {control: 0, state_in: 1, 'unrelated_query_g_or_offset': 2}, {state_out: 0})
    if permute:
        labels = {f'RefData_{index}': f'RefData_{(9-index//8)*8+index%8}' for index in range(80)}
        for node in ops:
            node['name'] = labels.get(node['name'], node['name'])
            if 'input' in node:
                node['input'] = [labels.get(edge.rsplit(':', 1)[0], edge.rsplit(':', 1)[0]) + ':' + edge.rsplit(':', 1)[1] if edge else edge for edge in node['input']]
    return document, {'files': files}


def inspect(left=None, right=None):
    p, pi = source(64, 'prefill_quant') if left is None else left
    d, di = source(1, 'decode_quant', permute=True) if right is None else right
    return audit.analyze_state_correspondence(p, d, pi, di, ENUMS, ENUMS)


def test_all80_parameter_fqn_owners_follow_named_roles_and_content_without_promoting_runtime_feedback():
    result = inspect()
    contract = result['parameter_fqn_owner_contract']
    assert contract['status'] == 'ALL80_SOURCE_OWNERS_MATCHED_UNDER_PARAMETER_FQN_CONTRACT'
    assert contract['matched_owner_count'] == 80 and contract['contract'] == owner.CONTRACT
    assert contract['observed_owner_count_by_model'] == {'prefill': 80, 'decode': 80}
    mapping = {row['owner_key']: row for row in contract['mapping']}
    assert mapping['layer.0.conv']['prefill_refdata'] == 'RefData_0'
    assert mapping['layer.0.conv']['decode_refdata'] == 'RefData_72'
    assert mapping['layer.3.key']['prefill_refdata'] == 'RefData_6'
    assert mapping['layer.3.key']['decode_refdata'] == 'RefData_78'
    assert result['gap_summary']['prefill']['context_budget_exhausted'] > 0
    assert result['unique_state_context_candidate_count'] < 80
    assert result['proven_cross_model_state_owner_count'] == 0 and not result['feedback_ready']
    assert contract['tensor_numeric_dtype_and_extent'] == 'UNVERIFIED_INTEGER_ATTRIBUTE_DOMAIN_NOT_RESOLVED'
    assert contract['original_python_fqn_to_filename_attestation'] == 'NOT_AVAILABLE'
    assert contract['compiled_runtime_file_binding'] == 'NOT_EVALUATED' and not contract['feedback_ready']
    recurrent = next(row for row in contract['models']['prefill'] if row.get('owner_key') == 'layer.0.recurrent')
    assert recurrent['owner_input_port'] == 'beta'
    assert recurrent['binding']['parameters']['W_q']['raw_tensor_attributes']['dtype_attribute'] == {'i': 3}


@pytest.mark.parametrize('fault', ['wrong_role', 'raw_dtype', 'raw_shape', 'file_id', 'offset', 'hidden_offset',
    'duplicate_fqn_node', 'duplicate_content_reference', 'mismatched_sha', 'wrong_family_layer', 'projection_port', 'projection_type'])
def test_unproven_or_conflicting_parameter_labels_and_raw_metadata_block_owner_mapping(fault):
    d, di = source(1, 'decode_quant', permute=True)
    nodes = {row['name']: row for row in d['graph'][0]['op']}
    constant = nodes['decode_quant_tensor_0_W_q']
    if fault == 'wrong_role': nodes['update_0']['attr'][0]['value']['list']['s'][0] = 'unreviewed_input'
    elif fault == 'raw_dtype': constant['attr'][2]['value']['i'] = 4
    elif fault == 'raw_shape': constant['attr'][1]['value']['list']['i'] = [8, 64]
    elif fault == 'file_id': constant['attr'][3]['value']['s'] = 'indirect'
    elif fault in {'offset', 'hidden_offset'}: constant['attr'].append(attr('_offset' if fault == 'hidden_offset' else 'offset', {'i': 0}))
    elif fault == 'duplicate_fqn_node':
        duplicate = deepcopy(constant); duplicate['name'] = 'same_fqn_twice'; d['graph'][0]['op'].append(duplicate)
        di['files'][0]['nodes'].append(duplicate['name'])
    elif fault == 'duplicate_content_reference':
        duplicate = deepcopy(constant); duplicate['name'] = 'unrelated_duplicate'; duplicate['attr'][0]['value']['s'] = 'decode_quant/other.weight'
        d['graph'][0]['op'].append(duplicate)
        row = dict(di['files'][0], reference='decode_quant/other.weight', relative_path='decode_quant/other.weight', nodes=[duplicate['name']]); di['files'].append(row)
    elif fault == 'mismatched_sha': di['files'][0]['sha256'] = 'f' * 64
    elif fault == 'wrong_family_layer':
        old = constant['attr'][0]['value']['s']; new = old.replace('layers_0_linear_attn', 'layers_3_linear_attn')
        constant['attr'][0]['value']['s'] = new; di['files'][0].update(reference=new, relative_path=new)
    elif fault == 'projection_port': nodes['decode_quant_projection_0']['attr'][0]['value']['list']['s'][1] = 'different_weight_port'
    else: nodes['decode_quant_projection_0']['type'] = 'DifferentProjection'
    result = inspect(right=(d, di))['parameter_fqn_owner_contract']
    assert result['matched_owner_count'] < 80
    assert result['status'] == 'PARAMETER_FQN_OWNER_CONTRACT_OBSERVED_WITH_GAPS'
    assert 'layer.0.conv' in result['missing_owner_keys'] and not result['feedback_ready']


def test_matching_integer_attribute_domain_is_not_reinterpreted_as_placeholder_descriptor_float():
    p, pi = source(64, 'p'); d, di = source(1, 'd', permute=True)
    for document in (p, d):
        for node in document['graph'][0]['op']:
            if node['type'] == 'FileConstant':
                node['output_desc'] = [{'dtype': 78, 'shape': {}, 'layout': 'ND'}]
    result = inspect((p, pi), (d, di))['parameter_fqn_owner_contract']
    assert result['matched_owner_count'] == 80 and 'UNVERIFIED' in result['tensor_numeric_dtype_and_extent']


def test_role_budget_exhaustion_at_or_before_a_possible_nearest_projection_is_not_ignored(monkeypatch):
    monkeypatch.setitem(owner._LIMITS, 'maximum_depth', 3)
    result = inspect()['parameter_fqn_owner_contract']
    assert result['matched_owner_count'] < 80
    assert any(row.get('searches') and any(search['truncated'] for search in row['searches']) for row in result['models']['prefill'])


def test_unique_parameter_keys_cannot_replace_missing_or_duplicate_actual_source_state_owners():
    p, pi = source(64, 'prefill_quant')
    update = next(row for row in p['graph'][0]['op'] if row['name'] == 'update_2')
    update['input'][1] = 'RefData_0:0'
    result = inspect(left=(p, pi))['parameter_fqn_owner_contract']
    assert result['complete_named_port_source_owner_preconditions']['prefill'] is False
    assert result['matched_owner_count'] == 0 and not result['feedback_ready']
