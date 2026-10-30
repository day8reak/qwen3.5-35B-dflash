"""A successful static64 inventory pins the source AST independently of DDK."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import builtins

import pytest

from qwen35_moe_dflash import target_source_schema as source
from qwen35_moe_dflash.air_schema import load_torchair_schema
from qwen35_moe_dflash.target_compile_failure import _read_small
from test_air_schema import pb2
from test_target_state_lineage import jobs, record, save, attach_torchair_source
from qwen35_moe_dflash import target_state_lineage as lineage


def identity(path):
    raw, current = _read_small(path, 32 << 20)
    return current


@pytest.fixture
def case(tmp_path):
    path = pb2(tmp_path / 'producer/ge_ir_pb2.py')
    schema = load_torchair_schema(path)
    air = {'path': '/bound/snapshot/export.air', 'bytes': 123, 'sha256': 'b' * 64}
    inventory_path = tmp_path / 'staged/inventory.json'
    inventory = {'artifact_kind': 'qwen35-target-export-inventory', 'status': 'INVENTORIED', 'weight_identity': 'SHA256_COMPLETE',
        'air': air, 'schema': schema.evidence, 'source_relative_prefix': 'data',
        'declared_abi': {'status': 'MATCHED_INTERFACE_METADATA', 'input_count': 89, 'output_count': 82,
            'feature_output_index': 81, 'feature_shape': [1, 64, 16384], 'feature_dtype': 'DT_FLOAT16'},
        'files': [{'reference': 'data/never-open.weight', 'relative_path': 'data/never-open.weight',
            'basename': 'never-open.weight', 'bytes': 999, 'sha256': 'c' * 64}]}
    save(inventory_path, inventory)
    report = {'external_weights': {'reuse_binding': {'status': 'RECEIPT_BOUND_SELECTION',
        'payload_verification': 'ALL_SELECTED_SNAPSHOT_SHA256_MATCHED_RECEIPT', 'source_relative_prefix': 'data',
        'inventory': identity(inventory_path)}}}
    reads = []
    def metadata(path, role):
        reads.append((str(path), role))
        raw, current = _read_small(path, 32 << 20)
        return json.loads(raw), current
    def read_code(path):
        reads.append((str(path), 'pb2'))
        return identity(path)
    return {'path': path, 'schema': schema, 'air': air, 'inventory': inventory, 'inventory_path': inventory_path,
        'report': report, 'metadata': metadata, 'pb2': read_code, 'reads': reads}


def load(case, **options):
    return source.load_bound_source_schema(case['report'], case['air'], read_metadata=case['metadata'],
        read_pb2=case['pb2'], **options)


def rebind(case):
    save(case['inventory_path'], case['inventory'])
    case['report']['external_weights']['reuse_binding']['inventory'] = identity(case['inventory_path'])


def test_prefill_inventory_ast_source_is_explicit_complete_and_never_imports_vendor_or_reads_weights(case, monkeypatch):
    imported = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name.split('.')[0] not in {'torch', 'torchair', 'torch_npu'}
        return imported(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    current, binding = load(case)
    assert current.package == 'ge.proto' and current.evidence == case['schema'].evidence
    assert binding['status'] == 'PREFILL_INVENTORY_AND_AST_DESCRIPTOR_MATCHED'
    assert binding['inventory'] == identity(case['inventory_path'])
    assert binding['selected_pb2'] == identity(case['path'])
    assert binding['sequence_length'] == 64 and binding['pb2_execution'] == 'NOT_EXECUTED'
    assert case['reads'] == [(str(case['inventory_path']), 'source-prefill-export-inventory'), (str(case['path']), 'pb2')]


def test_identical_explicit_pb2_copy_is_accepted_after_original_machine_path_disappears(case, tmp_path):
    copied = tmp_path / 'copied-producer.py'; copied.write_bytes(case['path'].read_bytes()); case['path'].unlink()
    current, binding = load(case, source_torchair_pb2=copied)
    assert binding['selection'] == 'EXPLICIT_IDENTICAL_PB2_COPY'
    assert binding['selected_pb2']['path'] == str(copied)
    assert current.evidence['pb2_sha256'] == case['schema'].evidence['pb2_sha256']


@pytest.mark.parametrize('fault', ['selection', 'unfinished_payload', 'pin_hash', 'pin_bytes', 'air_hash', 'air_bytes',
    'abi_sequence', 'abi_bool', 'output_count', 'feature_dtype', 'weight_hash_incomplete', 'sidecar_hash',
    'schema_package', 'schema_filename', 'schema_pb2_sha', 'schema_descriptor_sha', 'schema_pb2_bytes', 'prefix', 'null_binding', 'invalid_external'])
def test_declared_prefill_schema_failures_never_fall_back_to_compiler_or_discovery(case, fault):
    reuse = case['report']['external_weights']['reuse_binding']
    if fault == 'null_binding': case['report']['external_weights']['reuse_binding'] = None
    elif fault == 'invalid_external': case['report']['external_weights'] = []
    elif fault == 'selection': reuse['status'] = 'UNVERIFIED'
    elif fault == 'unfinished_payload': reuse['payload_verification'] = 'REQUIRES_EXISTING_SNAPSHOT_SHA256_MATCH_NO_EXTRA_PAYLOAD_HASH'
    elif fault == 'pin_hash': reuse['inventory']['sha256'] = 'f' * 64
    elif fault == 'pin_bytes': reuse['inventory']['bytes'] += 1
    else:
        manifest = case['inventory']
        if fault == 'air_hash': manifest['air'] = dict(manifest['air'], sha256='f' * 64)
        elif fault == 'air_bytes': manifest['air'] = dict(manifest['air'], bytes=124)
        elif fault == 'abi_sequence': manifest['declared_abi']['feature_shape'] = [1, 1, 16384]
        elif fault == 'abi_bool': manifest['declared_abi']['feature_shape'] = [True, 64, 16384]
        elif fault == 'output_count': manifest['declared_abi']['output_count'] = 81
        elif fault == 'feature_dtype': manifest['declared_abi']['feature_dtype'] = 'DT_FLOAT'
        elif fault == 'weight_hash_incomplete': manifest['weight_identity'] = 'STAT_ONLY'
        elif fault == 'sidecar_hash': manifest['files'][0]['sha256'] = None
        elif fault == 'schema_package': manifest['schema']['schema_package'] = 'hiai.proto'
        elif fault == 'schema_filename': manifest['schema']['schema_file'] = 'other.proto'
        elif fault == 'schema_pb2_sha': manifest['schema']['pb2_sha256'] = 'f' * 64
        elif fault == 'schema_descriptor_sha': manifest['schema']['descriptor_sha256'] = 'f' * 64
        elif fault == 'schema_pb2_bytes': manifest['schema']['pb2_bytes'] = True
        else: reuse['source_relative_prefix'] = 'other'
        rebind(case)
    with pytest.raises((ValueError, OSError)):
        load(case)


def test_existing_decode_canonical_branch_retains_precedence_and_refuses_missing_declared_transfer(case):
    case['report']['decode_input_view'] = {'decode_export_receipt': '/declared/missing/result.json'}
    with pytest.raises(ValueError, match='missing its exact transfer binding'):
        load(case)
    assert case['reads'] == []


def test_no_producer_declaration_preserves_legacy_lane_and_cannot_select_an_unbound_pb2(case):
    assert source.load_bound_source_schema({}, case['air'], read_metadata=case['metadata'], read_pb2=case['pb2']) is None
    with pytest.raises(ValueError, match='hash-bound canonical producer'):
        source.load_bound_source_schema({}, case['air'], read_metadata=case['metadata'], read_pb2=case['pb2'], source_torchair_pb2=case['path'])


def test_real_prefill_graph_loader_uses_inventory_ast_and_separate_compiler_schema(jobs):
    # The existing source fixture creates a genuine protobuf source with enum7,
    # and an independent selected compiler-schema fixture with enum77.
    original_decode = attach_torchair_source(jobs)
    prefill_job, report = jobs['prefill_job'], jobs['prefill_report']
    root = prefill_job / 'compilation'
    air = root / report['input_snapshots'][0]['path']
    air.write_bytes(original_decode['air'].read_bytes())
    report['input_snapshots'][0] = dict(record(air, root, stat_key='snapshot_stat'), role='air')
    air_id = {'path': str(air), 'bytes': air.stat().st_size, 'sha256': hashlib.sha256(air.read_bytes()).hexdigest()}
    inventory_path = jobs['tmp'] / 'prefill-staging/inventory.json'
    manifest = {'artifact_kind': 'qwen35-target-export-inventory', 'status': 'INVENTORIED', 'weight_identity': 'SHA256_COMPLETE',
        'air': air_id, 'schema': original_decode['schema'].evidence, 'source_relative_prefix': 'data',
        'declared_abi': {'status': 'MATCHED_INTERFACE_METADATA', 'input_count': 89, 'output_count': 82,
            'feature_output_index': 81, 'feature_shape': [1, 64, 16384], 'feature_dtype': 'DT_FLOAT16'},
        'files': [{'reference': 'data/never-open.weight', 'relative_path': 'data/never-open.weight',
            'basename': 'never-open.weight', 'bytes': 999, 'sha256': 'c' * 64}]}
    save(inventory_path, manifest)
    report['external_weights'] = {'reuse_binding': {'status': 'RECEIPT_BOUND_SELECTION',
        'payload_verification': 'ALL_SELECTED_SNAPSHOT_SHA256_MATCHED_RECEIPT', 'source_relative_prefix': 'data',
        'inventory': identity(inventory_path)}}
    save(root / 'input-inventory.json', {'files': report['input_snapshots']}); save(root / 'report.json', report)
    outer = json.loads((prefill_job / 'result.json').read_text())
    outer['compilation_report_sha256'] = identity(root / 'report.json')['sha256']; save(prefill_job / 'result.json', outer)
    metadata, graph, compiled, enums = lineage.load_target_graph(prefill_job, 'cpro-40block-prefill-64', [])
    assert metadata['source_schema_binding']['status'] == 'PREFILL_INVENTORY_AND_AST_DESCRIPTOR_MATCHED'
    assert metadata['source_enum_fields']['dtype']['values']['DT_FLOAT16'] == 7
    assert enums['dtype']['values']['DT_FLOAT16'] == 77
    assert lineage.analyze_state_lineage(graph, metadata['source_enum_fields'])['unique_state_owner_count'] == 80


def explicit_receipt(case, tmp_path, *, include_inventory=True):
    schema = deepcopy(case['inventory']['schema'])
    receipt = {'artifact_kind': 'qwen35-canonical-target-export', 'status': 'EXPORT_READY', 'air': case['air'],
        'schema': schema, 'normalization': {'status': 'OUTPUT_ORDER_CANONICALIZED', 'sequence_length': 64,
            'patched_sha256': case['air']['sha256'], 'schema': deepcopy(schema)}}
    if include_inventory:
        receipt['inventory'] = deepcopy(case['report']['external_weights']['reuse_binding']['inventory'])
    path = tmp_path / 'canonical64/result.json'; save(path, receipt)
    return path, receipt


def test_explicit_canonical64_and_implicit_inventory_are_both_bound_when_present(case, tmp_path):
    path, _ = explicit_receipt(case, tmp_path)
    schema, binding = load(case, source_export_receipt=path)
    assert binding['status'] == 'PREFILL_INVENTORY_AND_CANONICAL_RECEIPT_AST_DESCRIPTOR_MATCHED'
    assert binding['receipt'] == identity(path) and binding['inventory'] == identity(case['inventory_path'])
    assert schema.evidence['pb2_sha256'] == case['schema'].evidence['pb2_sha256']


def test_explicit_same_air_canonical64_without_staging_inventory_can_use_identical_pb2_copy(case, tmp_path):
    path, _ = explicit_receipt(case, tmp_path, include_inventory=False)
    copied = tmp_path / 'owned/ge_ir_pb2.py'; copied.parent.mkdir(); copied.write_bytes(case['path'].read_bytes()); case['path'].unlink()
    case['report'] = {}
    schema, binding = load(case, source_export_receipt=path, source_torchair_pb2=copied)
    assert binding['status'] == 'PREFILL_CANONICAL_EXPORT_RECEIPT_AND_AST_DESCRIPTOR_MATCHED'
    assert binding['selection'] == 'EXPLICIT_IDENTICAL_PB2_COPY' and binding['selected_pb2']['path'] == str(copied)
    assert schema.evidence['pb2_sha256'] == case['schema'].evidence['pb2_sha256']


@pytest.mark.parametrize('fault', ['missing', 'sequence1', 'sequence_bool', 'air', 'normalization_sha', 'schema', 'inventory_pin', 'different_pb2_source'])
def test_explicit_receipt_failure_or_dual_source_disagreement_never_falls_back(case, tmp_path, fault):
    path, receipt = explicit_receipt(case, tmp_path)
    if fault == 'missing': path.unlink()
    else:
        if fault == 'sequence1': receipt['normalization']['sequence_length'] = 1
        elif fault == 'sequence_bool': receipt['normalization']['sequence_length'] = True
        elif fault == 'air': receipt['air'] = dict(receipt['air'], sha256='f' * 64)
        elif fault == 'normalization_sha': receipt['normalization']['patched_sha256'] = 'f' * 64
        elif fault == 'schema': receipt['normalization']['schema']['descriptor_sha256'] = 'f' * 64
        elif fault == 'inventory_pin': receipt['inventory']['sha256'] = 'f' * 64
        else:
            alternative = tmp_path / 'alternative/ge_ir_pb2.py'; alternative.parent.mkdir()
            alternative.write_bytes(case['path'].read_bytes() + b'\n# different producer bytes, same AST descriptor\n')
            receipt['schema'] = load_torchair_schema(alternative).evidence
            receipt['normalization']['schema'] = deepcopy(receipt['schema'])
        save(path, receipt)
    with pytest.raises((ValueError, OSError)):
        load(case, source_export_receipt=path)


def test_explicit_prefill_receipt_cannot_override_an_existing_decode_producer_declaration(case, tmp_path):
    path, _ = explicit_receipt(case, tmp_path)
    case['report']['decode_input_view'] = {'transfer': {}}
    with pytest.raises(ValueError, match='cannot replace a declared decode'):
        load(case, source_export_receipt=path)


def test_shared_source_air_wrapper_forwards_explicit_receipt_and_pb2_without_profile_or_call_changes(monkeypatch):
    captured = {}
    schema = object()
    def loader(job, profile, bindings, **options):
        captured.update(job=job, profile=profile, bindings=bindings, **options)
        options['source_capture'].update(schema=schema, raw=b'exact-source', document={'graph': []}, kind='ModelDef')
        return {'identity': 'bound'}, None, None, None
    monkeypatch.setattr(lineage, 'load_target_graph', loader)
    bindings = []
    result = lineage.load_bound_source_air('job', 'cpro-40block-prefill-64', bindings,
        source_torchair_pb2='copy.py', source_export_receipt='canonical/result.json')
    assert result == ({'identity': 'bound'}, schema, b'exact-source', {'graph': []}, 'ModelDef')
    assert captured['source_torchair_pb2'] == 'copy.py' and captured['source_export_receipt'] == 'canonical/result.json'
    assert captured['bindings'] is bindings


def test_explicit_prefill_receipt_kwarg_cannot_change_decode_profile_loader():
    with pytest.raises(ValueError, match='only for the static64 prefill'):
        lineage.load_target_graph('/absent', 'cpro-40block-decode-1', [], source_export_receipt='/absent/result.json')
