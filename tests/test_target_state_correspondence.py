"""Cross-export contexts are facts/candidates, never invented state owners."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from qwen35_moe_dflash import target_state_correspondence as audit
from qwen35_moe_dflash.target_compile_failure import _read_small
from test_target_state_lineage import graph, desc, attr, ENUMS


def pair(sequence, *, label, permute=False):
    document = graph(sequence)
    ops = document['graph'][0]['op']
    nodes = {row['name']: row for row in ops}
    files = []
    for index in range(80):
        reference = f'{label}_parameters/content_{index}.weight'
        constant = {'name': f'{label}_parameter_{index}', 'type': 'FileConstant',
            'attr': [attr('file_path', {'s': reference})], 'output_desc': [desc([2048])]}
        projection = {'name': f'{label}_projection_{index}', 'type': 'MatMul',
            'input': ['arg1_1:0', constant['name'] + ':0'], 'output_desc': [desc([2048])]}
        nodes[f'update_{index}']['input'][0] = projection['name'] + ':0'
        ops.extend([constant, projection])
        files.append({'reference': reference, 'relative_path': reference, 'basename': f'content_{index}.weight',
            'nodes': [constant['name']], 'bytes': 4096, 'sha256': hashlib.sha256(f'parameter {index}'.encode()).hexdigest()})
    if permute:
        # Preserve only type-compatible state geometries. Actual correspondence
        # must follow parameter edges, not RefData or output numeric spellings.
        labels = {f'RefData_{index}': f'RefData_{(9-index//8)*8+index%8}' for index in range(80)}
        for node in ops:
            node['name'] = labels.get(node['name'], node['name'])
            if 'input' in node:
                node['input'] = [labels.get(edge.rsplit(':', 1)[0], edge.rsplit(':', 1)[0]) + ':' + edge.rsplit(':', 1)[1]
                    if edge else edge for edge in node['input']]
    inventory = {'artifact_kind': 'qwen35-target-export-inventory', 'status': 'INVENTORIED', 'weight_identity': 'SHA256_COMPLETE',
        'air': {'bytes': 123, 'sha256': 'a' * 64}, 'schema': {'descriptor_sha256': 'b' * 64},
        'source_relative_prefix': label + '_parameters',
        'declared_abi': {'status': 'MATCHED_INTERFACE_METADATA', 'input_count': 89, 'output_count': 82,
            'feature_output_index': 81, 'feature_shape': [1, sequence, 16384], 'feature_dtype': 'DT_FLOAT16'}, 'files': files}
    return document, inventory


def analyze(left=None, right=None):
    p, pi = pair(64, label='prefill') if left is None else left
    d, di = pair(1, label='decode', permute=True) if right is None else right
    return audit.analyze_state_correspondence(p, d, pi, di, ENUMS, ENUMS)


def test_all80_context_candidates_follow_actual_edges_and_content_not_numeric_labels():
    result = analyze()
    assert result['status'] == 'ALL80_UNIQUE_WEIGHT_CONTEXT_CANDIDATES_OBSERVED'
    assert result['unique_state_context_candidate_count'] == 80
    mapping = {row['prefill_refdata']: row['decode_refdata'] for row in result['state_context_candidates']}
    assert mapping['RefData_0'] == 'RefData_72' and mapping['RefData_79'] == 'RefData_7'
    assert all(name != match for name, match in mapping.items())
    assert result['proven_cross_model_state_owner_count'] == 0
    assert result['cross_model_state_identity'] == 'UNVERIFIED_WEIGHT_CONTEXT_CANDIDATES_ONLY'
    assert result['feedback_ready'] is False and result['weight_payload_access'] == 'NONE'
    state = next(row for row in result['models']['prefill']['states'] if row['source_refdata'] == 'RefData_0')
    path = state['context']['paths'][0]
    assert path['context_input_port_name'] == 'other_input'
    assert [(row['type'], row.get('input_port')) for row in path['edges_from_updater_input']] == [('MatMul', 1), ('FileConstant', None)]


@pytest.mark.parametrize('fault', ['different_content', 'duplicate_content_reference', 'changed_context_role',
    'different_projection_input', 'missing_source_owner', 'ambiguous_state_context'])
def test_unproven_ambiguous_or_different_context_never_becomes_full80(fault):
    p, pi = pair(64, label='prefill')
    d, di = pair(1, label='decode', permute=True)
    nodes = {row['name']: row for row in d['graph'][0]['op']}
    if fault == 'different_content': di['files'][0]['sha256'] = 'f' * 64
    elif fault == 'duplicate_content_reference':
        duplicate = deepcopy(di['files'][0]); duplicate.update(reference='decode_parameters/duplicate.weight',
            relative_path='decode_parameters/duplicate.weight', basename='duplicate.weight', nodes=['duplicate'])
        di['files'].append(duplicate)
        d['graph'][0]['op'].append({'name': 'duplicate', 'type': 'FileConstant',
            'attr': [attr('file_path', {'s': duplicate['reference']})], 'output_desc': [desc([2048])]})
    elif fault == 'changed_context_role': nodes['update_0']['attr'][0]['value']['list']['s'][0] = 'different_role'
    elif fault == 'different_projection_input': nodes['decode_projection_0']['input'].reverse()
    elif fault == 'missing_source_owner': nodes['update_0']['attr'] = []
    else: nodes['update_8']['input'][0] = nodes['update_0']['input'][0]
    result = analyze((p, pi), (d, di))
    assert result['unique_state_context_candidate_count'] < 80
    assert result['status'] == 'STATE_CORRESPONDENCE_EVIDENCE_OBSERVED_WITH_GAPS'
    assert not result['feedback_ready'] and result['proven_cross_model_state_owner_count'] == 0


def test_depth_budget_reports_missing_evidence_without_traversing_state_ancestry():
    d, di = pair(1, label='decode', permute=True)
    nodes = {row['name']: row for row in d['graph'][0]['op']}
    edge = nodes['update_0']['input'][0]
    for index in range(12):
        node = {'name': f'long_{index}', 'type': 'Identity', 'input': [edge], 'output_desc': [desc([2048])]}
        d['graph'][0]['op'].append(node); edge = node['name'] + ':0'
    nodes['update_0']['input'][0] = edge
    result = analyze(right=(d, di))
    state = next(row for row in result['models']['decode']['states'] if row.get('updater', {}).get('node') == 'update_0')
    assert state['context']['truncated'] and state['context']['terminals']['DEPTH_LIMIT']
    assert 'candidate_signature' not in state and result['unique_state_context_candidate_count'] == 79


def test_total_queued_paths_are_bounded_even_with_repeated_high_fanout_edges():
    d, di = pair(1, label='decode', permute=True)
    nodes = {row['name']: row for row in d['graph'][0]['op']}
    branches = []
    for index in range(5):
        branch = {'name': f'fanout_{index}', 'type': 'Add', 'input': ['arg1_1:0'] * 1024, 'output_desc': [desc([2048])]}
        branches.append(branch); d['graph'][0]['op'].append(branch)
    root = {'name': 'fanout_root', 'type': 'Add', 'input': [row['name'] + ':0' for row in branches], 'output_desc': [desc([2048])]}
    d['graph'][0]['op'].append(root); nodes['update_0']['input'][0] = root['name'] + ':0'
    result = analyze(right=(d, di))
    state = next(row for row in result['models']['decode']['states'] if row.get('updater', {}).get('node') == 'update_0')
    assert state['context']['truncated'] and state['context']['terminals']['ENQUEUE_LIMIT'] == 1
    assert state['context']['enqueued_edge_count'] == 4096 and 'candidate_signature' not in state


def test_unreviewed_fqn_and_fx_metadata_remain_literal_diagnostics_not_owner_proof():
    p, pi = pair(64, label='prefill'); d, di = pair(1, label='decode', permute=True)
    for document in (p, d):
        for node in document['graph'][0]['op']:
            if node['name'].startswith('RefData_'):
                node['attr'] = [attr('nn_module_stack', {'s': 'model.layers.0.fake'})]
    result = analyze((p, pi), (d, di))
    assert result['unique_state_context_candidate_count'] == 80 and not result['feedback_ready']
    provenance = result['models']['prefill']['states'][0]['state_input_provenance']
    assert provenance['selected']['nn_module_stack'] == {'s': 'model.layers.0.fake'}
    assert 'UNREVIEWED' in provenance['acceptance']


def test_actual_unmaterialized_source_producer_shapes_do_not_replace_named_owner_or_context_evidence():
    p, pi = pair(64, label='prefill'); d, di = pair(1, label='decode', permute=True)
    for document in (p, d):
        for node in document['graph'][0]['op']:
            if node['name'].startswith('state_cast_'):
                node['output_desc'][0]['shape'] = {}
    result = analyze((p, pi), (d, di))
    assert result['unique_state_context_candidate_count'] == 80
    assert all(row['source_owner_count'] == row['unique_source_owner_count'] == 80 for row in result['models'].values())
    assert not result['feedback_ready'] and result['proven_cross_model_state_owner_count'] == 0


@pytest.mark.parametrize('fault', ['absolute_reference', 'missing_reference', 'unretained_node'])
def test_parameter_anchor_requires_exact_retained_fileconstant_reference(fault):
    p, pi = pair(64, label='prefill')
    constant = next(row for row in p['graph'][0]['op'] if row['type'] == 'FileConstant')
    if fault == 'absolute_reference': constant['attr'][0]['value']['s'] = '/weights/unbound.weight'
    elif fault == 'missing_reference': constant['attr'][0]['value']['s'] = 'prefill_parameters/unbound.weight'
    else: pi['files'][0]['nodes'] = ['not-this-actual-constant']
    with pytest.raises(ValueError, match='FileConstant'):
        analyze(left=(p, pi))


def test_explicit_leading_dot_alias_uses_exact_inventory_normalization_and_preserves_literal_tensor_fields():
    p, pi = pair(64, label='prefill')
    constant = next(row for row in p['graph'][0]['op'] if row['type'] == 'FileConstant')
    reference = constant['attr'][0]['value']['s']
    constant['attr'][0]['value']['s'] = './' + reference
    constant['attr'].append(attr('offset', {'i': 128}))
    pi['files'][0]['references'] = [reference, './' + reference]
    result = analyze(left=(p, pi))
    state = next(row for row in result['models']['prefill']['states'] if row['source_refdata'] == 'RefData_0')
    anchor = state['context']['paths'][0]['parameter']
    assert anchor['reference'] == './' + reference and anchor['normalized_relative_path'] == reference
    assert anchor['literal_tensor_metadata']['attributes']['offset'] == {'i': 128}
    assert result['unique_state_context_candidate_count'] == 80 and result['proven_cross_model_state_owner_count'] == 0
    assert 'WHOLE_SIDECAR' in result['parameter_identity_scope'] and not result['feedback_ready']


def test_unretained_leading_dot_spelling_is_not_rescued_by_normalization_alone():
    p, pi = pair(64, label='prefill')
    constant = next(row for row in p['graph'][0]['op'] if row['type'] == 'FileConstant')
    constant['attr'][0]['value']['s'] = './' + constant['attr'][0]['value']['s']
    with pytest.raises(ValueError, match='raw reference is not retained'):
        analyze(left=(p, pi))


def test_two_retained_raw_aliases_for_one_sidecar_do_not_falsely_make_its_content_nonunique():
    p, pi = pair(64, label='prefill')
    constant = next(row for row in p['graph'][0]['op'] if row['type'] == 'FileConstant')
    duplicate = deepcopy(constant); duplicate['name'] = 'same_file_second_constant'
    duplicate['attr'][0]['value']['s'] = './' + duplicate['attr'][0]['value']['s']
    pi['files'][0]['references'] = [pi['files'][0]['reference'], duplicate['attr'][0]['value']['s']]
    pi['files'][0]['nodes'].append(duplicate['name']); p['graph'][0]['op'].append(duplicate)
    result = analyze(left=(p, pi))
    assert result['unique_common_parameter_content_count'] == 80
    assert result['unique_state_context_candidate_count'] == 80 and not result['feedback_ready']


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')
    return _read_small(path, 32 << 20)[1]


@pytest.fixture
def saved_pair(tmp_path, monkeypatch):
    monkeypatch.setenv('AI_RUN_DIR', str(tmp_path))
    records, calls = {}, []
    for label, sequence in [('prefill', 64), ('decode', 1)]:
        document, inventory = pair(sequence, label=label, permute=sequence == 1)
        job = tmp_path / label
        inventory_id = save(job / 'inventory.json', inventory)
        report_id = save(job / 'compilation/report.json', {'input_snapshots': [dict(row, source_relative=row['relative_path']) for row in inventory['files']]})
        metadata = {'job': str(job), 'air': inventory['air'], 'model': {'sha256': 'c' * 64, 'bytes': 123},
            'source_schema': inventory['schema'], 'source_schema_binding': {'inventory': inventory_id},
            'source_enum_fields': ENUMS, 'source_result_sha256': label + '-outer', 'compilation_report_sha256': report_id['sha256']}
        records[str(job)] = (metadata, document)
    p = records[str(tmp_path / 'prefill')][0]; d = records[str(tmp_path / 'decode')][0]
    d['borrowed_prefill_inputs'] = {'artifact_kind': 'qwen35-decode-input-reference-view', 'status': 'PASSED',
        'source_job': p['job'], 'source_binding': {'source_compilation_report_sha256': p['compilation_report_sha256'],
            'source_result_sha256': p['source_result_sha256']}, 'reference_method': 'READONLY_SOURCE_REFERENCE_SYMLINK',
        'explicit_weight_copy_bytes': 0, 'audit_errors': []}
    def load(job, profile, bindings, **options):
        calls.append((str(job), profile, options))
        meta, document = records[str(job)]
        return deepcopy(meta), deepcopy(document), None, ENUMS
    monkeypatch.setattr(audit.lineage, 'load_target_graph', load)
    return records, calls


def test_inspect_binds_all_snapshot_hashes_and_schema_without_any_weight_access(saved_pair, tmp_path, monkeypatch):
    records, calls = saved_pair
    actual_read = audit._read_small
    def read(path, maximum):
        assert Path(path).suffix != '.weight'
        return actual_read(path, maximum)
    monkeypatch.setattr(audit, '_read_small', read)
    result = audit.inspect_target_state_correspondence(tmp_path / 'prefill', tmp_path / 'decode', tmp_path / 'result',
        prefill_source_torchair_pb2='/copied/prefill.py', decode_source_torchair_pb2='/copied/decode.py',
        prefill_source_export_receipt='/retained/prefill-canonical.json')
    assert result['stage'] == 'complete' and result['unique_state_context_candidate_count'] == 80
    assert result['target_pair_binding']['status'] == 'MATCHED_SAVED_PREFILL_INPUT_BORROW_DECLARATION'
    assert len(result['source_bindings']) == 4
    assert calls[0][2] == {'source_torchair_pb2': '/copied/prefill.py', 'source_export_receipt': '/retained/prefill-canonical.json'}
    assert calls[1][2]['source_export_receipt'] is None
    assert (tmp_path / 'result/report.json').is_file() and not result['feedback_ready']


def test_decode_canonical_receipt_inventory_is_followed_only_by_its_exact_pin(saved_pair, tmp_path):
    records, _ = saved_pair
    metadata = records[str(tmp_path / 'decode')][0]
    inventory = metadata['source_schema_binding'].pop('inventory')
    # Canonical receipts historically omit inventory bytes and retain path/SHA.
    pin = {'path': inventory['path'], 'sha256': inventory['sha256'], 'status': 'INVENTORIED'}
    receipt = save(tmp_path / 'canonical/result.json', {'inventory': pin})
    metadata['source_schema_binding']['receipt'] = receipt
    result = audit.inspect_target_state_correspondence(tmp_path / 'prefill', tmp_path / 'decode', tmp_path / 'result')
    assert result['stage'] == 'complete' and result['unique_state_context_candidate_count'] == 80
    assert len(result['source_bindings']) == 5
    assert result['decode']['inventory']['path'] == inventory['path']


@pytest.mark.parametrize('fault', ['receipt_hash', 'inventory_pin', 'missing_inventory'])
def test_declared_canonical_receipt_failure_has_no_inventory_discovery_fallback(saved_pair, tmp_path, fault):
    records, _ = saved_pair
    metadata = records[str(tmp_path / 'decode')][0]
    inventory = metadata['source_schema_binding'].pop('inventory')
    if fault == 'inventory_pin': inventory = dict(inventory, sha256='f' * 64)
    receipt = save(tmp_path / 'canonical/result.json', {} if fault == 'missing_inventory' else {'inventory': inventory})
    if fault == 'receipt_hash': receipt['sha256'] = 'f' * 64
    metadata['source_schema_binding']['receipt'] = receipt
    result = audit.inspect_target_state_correspondence(tmp_path / 'prefill', tmp_path / 'decode', tmp_path / 'result')
    assert result['status'] == 'FAILED' and not result['feedback_ready']


def test_large_unreviewed_provenance_is_bounded_and_never_used_as_a_signature():
    p, pi = pair(64, label='prefill')
    for node in p['graph'][0]['op']:
        if node['name'].startswith('RefData_'):
            node['attr'] = [attr('nn_module_stack', {'s': 'x' * 20000}),
                *[attr('source_meta_' + str(index), {'list': {'s': ['y' * 500] * 100}}) for index in range(20)]]
    result = analyze(left=(p, pi))
    assert result['unique_state_context_candidate_count'] == 80 and not result['feedback_ready']
    assert result['models']['prefill']['states'][0]['state_input_provenance']['truncated']
    assert len(json.dumps(result).encode()) < 4 << 20


@pytest.mark.parametrize('fault', ['inventory_pin', 'inventory_air', 'inventory_schema', 'compiled_snapshot', 'active_inventory_change'])
def test_inspect_rejects_unbound_or_actively_changed_metadata(saved_pair, tmp_path, monkeypatch, fault):
    records, _ = saved_pair
    meta, _ = records[str(tmp_path / 'decode')]
    if fault == 'inventory_pin': meta['source_schema_binding']['inventory']['sha256'] = 'f' * 64
    elif fault in {'inventory_air', 'inventory_schema'}:
        path = Path(meta['source_schema_binding']['inventory']['path']); value = json.loads(path.read_bytes())
        value['air' if fault == 'inventory_air' else 'schema']['sha256' if fault == 'inventory_air' else 'descriptor_sha256'] = 'f' * 64
        meta['source_schema_binding']['inventory'] = save(path, value)
    elif fault == 'compiled_snapshot':
        path = tmp_path / 'decode/compilation/report.json'; value = json.loads(path.read_bytes())
        value['input_snapshots'][0]['sha256'] = 'f' * 64
        meta['compilation_report_sha256'] = save(path, value)['sha256']
    else:
        original = audit.analyze_state_correspondence
        def mutate(*args):
            result = original(*args)
            path = tmp_path / 'decode/inventory.json'; path.write_bytes(path.read_bytes() + b' ')
            return result
        monkeypatch.setattr(audit, 'analyze_state_correspondence', mutate)
    result = audit.inspect_target_state_correspondence(tmp_path / 'prefill', tmp_path / 'decode', tmp_path / 'result')
    assert result['status'] == 'FAILED' and not result['feedback_ready']
    assert 'error' in result and (tmp_path / 'result/report.json').is_file()


@pytest.fixture
def relocated_decode_inventory(saved_pair, tmp_path):
    records, _ = saved_pair
    metadata = records[str(tmp_path / 'decode')][0]
    local = metadata['source_schema_binding'].pop('inventory')
    original_pin = {'path': str(tmp_path / 'missing-origin/qwen35-target-pytorch-run/export/inventory/inventory.json'),
        'sha256': local['sha256'], 'status': 'INVENTORIED'}
    receipt = save(tmp_path / 'decode-transfer/result.json', {'inventory': original_pin})
    metadata['source_schema_binding']['receipt'] = receipt
    report_path = tmp_path / 'decode/compilation/report.json'
    report = json.loads(report_path.read_text())
    report['decode_input_view'] = {'transfer': {'inventory': local, 'export_receipt': receipt}}
    metadata['compilation_report_sha256'] = save(report_path, report)['sha256']
    return {'metadata': metadata, 'local': local, 'original': original_pin,
        'receipt': receipt, 'report': report, 'report_path': report_path, 'records': records}


def test_relocated_inventory_uses_exact_compiler_transfer_pin_without_opening_missing_origin_or_changing_old_receipts(relocated_decode_inventory, tmp_path, monkeypatch):
    case = relocated_decode_inventory
    assert not Path(case['original']['path']).exists()
    watched = [Path(case['local']['path']), Path(case['receipt']['path']), case['report_path']]
    before = {path: (path.read_bytes(), path.stat().st_ctime_ns) for path in watched}
    old_binding = deepcopy(case['metadata']['source_schema_binding'])
    actual = audit._read_small
    def guarded(path, maximum):
        assert str(path) != case['original']['path'] and Path(path).suffix != '.weight'
        return actual(path, maximum)
    monkeypatch.setattr(audit, '_read_small', guarded)
    result = audit.inspect_target_state_correspondence(tmp_path / 'prefill', tmp_path / 'decode', tmp_path / 'result')
    assert result['stage'] == 'complete' and result['unique_state_context_candidate_count'] == 80, result
    assert result['decode']['inventory']['path'] == case['local']['path']
    resolution = result['decode']['inventory_resolution']
    assert resolution['status'] == 'PINNED_LOCAL_TRANSFER_CONTENT_MATCH'
    assert resolution['original_producer_inventory_pin'] == case['original']
    assert resolution['selected_transfer_inventory_pin'] == case['local']
    assert resolution['origin_path_access'] == 'NOT_ATTEMPTED' and resolution['root_rebasing'] == 'NOT_PERFORMED'
    assert case['metadata']['source_schema_binding'] == old_binding
    assert before == {path: (path.read_bytes(), path.stat().st_ctime_ns) for path in watched}
    assert not result['feedback_ready'] and result['proven_cross_model_state_owner_count'] == 0


@pytest.mark.parametrize('fault', ['local_bytes', 'local_hash', 'pin_sha', 'pin_bytes', 'pin_bool', 'null_pin',
    'relative_pin', 'canonical_sha', 'canonical_bytes', 'canonical_path', 'receipt_pin', 'compiler_sha',
    'rebound_wrong_air', 'rebound_wrong_schema', 'rebound_wrong_abi', 'rebound_wrong_snapshot',
    'missing_inventory', 'empty_transfer', 'null_transfer'])
def test_declared_local_inventory_faults_never_probe_origin_or_fall_back(relocated_decode_inventory, tmp_path, monkeypatch, fault):
    case = relocated_decode_inventory
    report, meta = case['report'], case['metadata']
    transfer = report['decode_input_view']['transfer']
    if fault == 'missing_inventory': transfer.pop('inventory')
    elif fault == 'empty_transfer': report['decode_input_view']['transfer'] = {}
    elif fault == 'null_transfer': report['decode_input_view']['transfer'] = None
    elif fault in {'local_bytes', 'local_hash'}:
        path = Path(case['local']['path'])
        if fault == 'local_bytes': path.write_bytes(path.read_bytes() + b' ')
        else:
            raw = bytearray(path.read_bytes()); raw[5] = ord('x'); path.write_bytes(raw)
    elif fault == 'pin_sha': transfer['inventory']['sha256'] = 'f' * 64
    elif fault == 'pin_bytes': transfer['inventory']['bytes'] += 1
    elif fault == 'pin_bool': transfer['inventory']['bytes'] = True
    elif fault == 'null_pin': transfer['inventory'] = None
    elif fault == 'relative_pin': transfer['inventory']['path'] = 'inventory.json'
    elif fault == 'receipt_pin': transfer['export_receipt']['sha256'] = 'f' * 64
    elif fault.startswith('canonical_'):
        receipt_value = json.loads(Path(case['receipt']['path']).read_text())
        receipt_value['inventory'][fault.removeprefix('canonical_')] = None if fault.endswith('path') else ('f' * 64 if fault.endswith('sha') else 1)
        if fault == 'canonical_sha':
            receipt_value['inventory']['sha256'] = receipt_value['inventory'].pop('sha')
        receipt_id = save(Path(case['receipt']['path']), receipt_value)
        meta['source_schema_binding']['receipt'] = receipt_id
        transfer['export_receipt'] = receipt_id
    elif fault.startswith('rebound_'):
        path = Path(case['local']['path']); value = json.loads(path.read_text())
        if fault == 'rebound_wrong_air': value['air']['sha256'] = 'f' * 64
        elif fault == 'rebound_wrong_schema': value['schema']['descriptor_sha256'] = 'f' * 64
        elif fault == 'rebound_wrong_abi': value['declared_abi']['feature_shape'] = [1, 64, 16384]
        else: report['input_snapshots'][0]['sha256'] = 'f' * 64
        transfer['inventory'] = save(path, value)
        receipt_value = json.loads(Path(case['receipt']['path']).read_text())
        receipt_value['inventory']['sha256'] = transfer['inventory']['sha256']
        transfer['export_receipt'] = save(Path(case['receipt']['path']), receipt_value)
        meta['source_schema_binding']['receipt'] = transfer['export_receipt']
    current_report_id = save(case['report_path'], report)
    meta['compilation_report_sha256'] = 'f' * 64 if fault == 'compiler_sha' else current_report_id['sha256']
    actual = audit._read_small
    def guarded(path, maximum):
        assert str(path) != case['original']['path'] and Path(path).suffix != '.weight'
        return actual(path, maximum)
    monkeypatch.setattr(audit, '_read_small', guarded)
    result = audit.inspect_target_state_correspondence(tmp_path / 'prefill', tmp_path / 'decode', tmp_path / 'result')
    assert result['status'] == 'FAILED' and result['error'] and not result['feedback_ready'], result
