"""Real graph/schema loading keeps old bindings and selects a pinned local JSON."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from qwen35_moe_dflash import target_state_correspondence as cross
from qwen35_moe_dflash import target_state_lineage as lineage
from qwen35_moe_dflash.target_compile_failure import _read_small
from test_target_state_lineage import jobs, attach_torchair_source, save


def test_real_canonical_decode_schema_survives_missing_origin_inventory_and_uses_exact_local_transfer(jobs, monkeypatch):
    source = attach_torchair_source(jobs)
    job, report = jobs['decode_job'], source['report']
    compilation = job / 'compilation'
    air_identity = {'bytes': source['air'].stat().st_size, 'sha256': hashlib.sha256(source['air'].read_bytes()).hexdigest()}
    row = report['input_snapshots'][1]
    row['source_relative'] = 'data/never-open.weight'
    inventory = {'artifact_kind': 'qwen35-target-export-inventory', 'status': 'INVENTORIED', 'weight_identity': 'SHA256_COMPLETE',
        'air': air_identity, 'schema': source['schema'].evidence, 'source_relative_prefix': 'data',
        'declared_abi': {'status': 'MATCHED_INTERFACE_METADATA', 'input_count': 89, 'output_count': 82,
            'feature_output_index': 81, 'feature_shape': [1, 1, 16384], 'feature_dtype': 'DT_FLOAT16'},
        'files': [{'reference': 'data/never-open.weight', 'relative_path': 'data/never-open.weight',
            'basename': 'never-open.weight', 'bytes': row['bytes'], 'sha256': row['sha256']}]}
    local_path = jobs['tmp'] / 'decode-transfer/inventory.json'; save(local_path, inventory)
    local_identity = _read_small(local_path, 32 << 20)[1]
    origin = jobs['tmp'] / 'absent-origin/qwen35-target-pytorch-run/canonical/inventory/inventory.json'
    assert not origin.exists()
    receipt = json.loads(source['receipt'].read_text())
    receipt['inventory'] = {'path': str(origin), 'sha256': local_identity['sha256'], 'status': 'INVENTORIED'}
    save(source['receipt'], receipt)
    view = report['decode_input_view']
    view['transfer'].update(inventory=local_identity, export_receipt=_read_small(source['receipt'], 32 << 20)[1])
    save(compilation / 'decode-input-view.json', view)
    save(compilation / 'input-inventory.json', {'files': report['input_snapshots']})
    save(compilation / 'report.json', report)
    outer = json.loads((job / 'result.json').read_text())
    outer['compilation_report_sha256'] = _read_small(compilation / 'report.json', 32 << 20)[1]['sha256']
    save(job / 'result.json', outer)
    watched = [source['receipt'], local_path, compilation / 'report.json']
    before = {path: (path.read_bytes(), path.stat().st_ctime_ns) for path in watched}
    opened = Path.open
    def guarded(path, *args, **kwargs):
        assert path != origin and path.suffix != '.weight', 'only graph/schema/JSON payloads are allowed'
        return opened(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)
    bindings = []
    metadata, _, _, _ = lineage.load_target_graph(job, 'cpro-40block-decode-1', bindings)
    original_schema_binding = deepcopy(metadata['source_schema_binding'])
    assert original_schema_binding['status'] == 'CANONICAL_EXPORT_RECEIPT_AND_AST_DESCRIPTOR_MATCHED'
    assert 'inventory' not in original_schema_binding
    manifest, current = cross._bound_inventory(metadata, bindings, 1)
    assert manifest == inventory and current == local_identity
    assert metadata['source_schema_binding'] == original_schema_binding
    assert metadata['inventory_resolution']['status'] == 'PINNED_LOCAL_TRANSFER_CONTENT_MATCH'
    assert metadata['inventory_resolution']['original_producer_inventory_pin']['path'] == str(origin)
    assert metadata['inventory_resolution']['current_inventory']['path'] == str(local_path)
    assert before == {path: (path.read_bytes(), path.stat().st_ctime_ns) for path in watched}
