"""Re-evaluate hash-verified retained draft outputs without another device run."""
import json
from pathlib import Path
import re

import numpy as np

from .draft_board import _file, compare_draft_result
from .hashing import sha256_file


def _json(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 << 20:
        raise ValueError('invalid retained draft metadata: '+str(path))
    return json.loads(path.read_text())


def _reference_audit(job, bundle_manifest):
    """Inspect only small normalization parameters, without loading the model."""
    components = job.get('components', {})
    if not all(name in components for name in ('prepared', 'fixture')):
        return {'status': 'UNAVAILABLE', 'reason': 'prepared/fixture component receipts missing'}
    prepared, fixture = components['prepared'], components['fixture']
    prepared_dir, fixture_dir = Path(prepared['directory']).resolve(), Path(fixture['directory']).resolve()
    prepared_file = _file(prepared_dir, prepared['manifest'])
    fixture_file = _file(fixture_dir, fixture['manifest'])
    plan, reference = _json(prepared_file), _json(fixture_file)
    if (bundle_manifest.get('fixture_report_sha256') != sha256_file(fixture_file)
            or reference.get('prepared_manifest_sha256') != sha256_file(prepared_file)):
        raise ValueError('reference audit manifests differ from the executed bundle')
    selected = [(name, row) for name, row in plan.get('weights', {}).items()
                if name.startswith('draft.') and 'norm' in name and name.endswith('.weight')]
    if len(selected) > 64:
        raise ValueError('normalization parameter audit exceeds bounded tensor count')
    stats = []
    for name, row in selected:
        if row.get('dtype') != 'float16' or type(row.get('bytes')) is not int or not 0 < row['bytes'] <= 65536:
            raise ValueError('invalid small normalization parameter: '+name)
        values = np.fromfile(_file(prepared_dir, row), dtype='<f2').astype(np.float64)
        finite = np.isfinite(values)
        stats.append({'name': name, 'sha256': row['sha256'], 'elements': int(values.size),
                      'nonfinite_count': int((~finite).sum()), 'nonzero_count': int(np.count_nonzero(values[finite])),
                      'finite_min': float(values[finite].min()) if finite.any() else None,
                      'finite_max': float(values[finite].max()) if finite.any() else None})
    return {'status': 'INSPECTED', 'manifest_binding': 'MATCHED',
            'reference_kind': reference.get('reference'), 'normalization_tensors_checked': len(stats),
            'all_zero_weights': [s['name'] for s in stats if not s['nonfinite_count'] and not s['nonzero_count']],
            'nonfinite_weights': [s['name'] for s in stats if s['nonfinite_count']],
            'final_norms': [s for s in stats if s['name'] in ('draft.norm.weight', 'draft.hidden_norm.weight')],
            'scope': 'Small prepared norm tensors only; no reference replay or intermediate-activation diagnosis.'}


def review_draft_result(job_dir):
    """Return a fresh comparison; leave all original job/board files untouched.

    A prior comparison/JSON error is recoverable. Incomplete transport, changed
    files, conflicting identities and nonzero exits are not promoted to success.
    """
    root = Path(job_dir).resolve()
    job_file = root/'result.json'
    job = _json(job_file)
    if job.get('artifact_kind') != 'qwen35-full-draft-job':
        raise ValueError('compare-only requires a retained full-draft job')
    component = job.get('components', {}).get('bundle')
    if not isinstance(component, dict) or not isinstance(component.get('directory'), str):
        raise ValueError('retained job has no completed bundle')
    bundle = Path(component['directory']).resolve()
    manifest_file = _file(bundle, component['manifest'])
    if manifest_file.name != 'manifest.json':
        raise ValueError('expected the retained bundle manifest')
    manifest = _json(manifest_file)
    if manifest.get('source_identity') != 'MATCHED_EXPORT_AND_REFERENCE':
        raise ValueError('comparison requires matched export and reference identities')
    board = root/'board'
    if job.get('board') and Path(job['board']).resolve() != board:
        raise ValueError('retained board path differs from this job')
    board_file = board/'result.json'
    execution = _json(board_file)
    if execution.get('artifact_kind') != 'qwen35-standalone-draft-execution':
        raise ValueError('expected retained standalone draft execution evidence')
    if execution.get('bundle_manifest_sha256') != sha256_file(manifest_file):
        raise ValueError('retained execution belongs to a different bundle')
    if execution.get('transport_error') or execution.get('retrieval_errors'):
        raise ValueError('retained execution/transport incomplete; do not reclassify it')
    if type(execution.get('remote_exitcode')) is not int or execution['remote_exitcode'] != 0:
        raise ValueError('retained draft process did not exit successfully')
    names = ('draft.stdout', 'draft.stderr', 'draft.exitcode', 'results/result.json',
             'results/output_0000.bin', 'results/output_0001.bin')
    hashes = execution.get('retrieved_sha256', {})
    sources = []
    for name in names:
        digest = hashes.get(name)
        if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
            raise ValueError('missing verified retrieval hash: '+name)
        path = board/name
        # _file enforces non-symlink ancestry and source SHA/size checks.
        info = {'path': name, 'bytes': path.stat().st_size, 'sha256': digest}
        _file(board, info)
        sources.append(info)
    if (board/'draft.exitcode').read_bytes().strip() != b'0':
        raise ValueError('retained exit file conflicts with successful process exit')
    native = _json(board/'results/result.json')
    if native.get('status') != 'EXECUTED':
        raise ValueError('compare-only requires a completed native draft execution')
    for key in ('runtime_version', 'firmware_id'):
        if not isinstance(execution.get(key), str) or not execution[key] or native.get(key) != execution[key]:
            raise ValueError('native '+key+' differs from the retained execution')
    if execution.get('ddk_id') != manifest.get('ddk_id'):
        raise ValueError('retained execution DDK differs from its bundle')
    api = execution.get('model_load_api', 'legacy-buffer')
    comparison = compare_draft_result(bundle, board/'results', expected_model_load_api=api)
    result = {'schema_version': 1, 'artifact_kind': 'qwen35-retained-draft-comparison',
              'status': comparison['status'], 'device_execution': comparison['device_execution'],
              'numerical_validation': comparison['numerical_validation'],
              'runtime_placement': 'UNVERIFIED', 'operation': 'HOST_COMPARISON_ONLY',
              'new_device_execution': 'NOT_RUN', 'new_compilation': 'NOT_RUN',
              'job': str(root), 'bundle_manifest_sha256': sha256_file(manifest_file),
              'job_result_sha256': sha256_file(job_file), 'board_result_sha256': sha256_file(board_file),
              'retrieved_files': sources, 'comparison': comparison,
              'native_status': native['status'], 'native_process_id': native.get('process_id'),
              'runtime_stage': native.get('runtime_stage'), 'model_load_api': api,
              'performance': 'NOT_EVALUATED',
              'performance_reason': 'The retained standalone runner did not record inference duration; log timestamps are not a benchmark.',
              'reference_audit': _reference_audit(job, manifest),
              'scope': 'Rechecks retained file identities and outputs; no HDC, model loading, execution or reference regeneration.'}
    if execution.get('comparison_error'):
        result['previous_comparison_error'] = execution['comparison_error']
    return result
