"""Retained comparison tests use fabricated device markers, never board evidence."""
import json
from pathlib import Path
import runpy
import subprocess

import numpy as np
import pytest

from qwen35_moe_dflash import draft_board, draft_review
from qwen35_moe_dflash.hashing import sha256_file
from qwen35_moe_dflash.draft_workspace import DraftWorkspace
from test_draft_board import bundle, native_result, record, write_json


@pytest.fixture
def retained(bundle, tmp_path, monkeypatch):
    original, manifest = bundle
    job = tmp_path / 'full-draft'
    job.mkdir()
    root = job / 'bundle'
    original.rename(root)
    manifest['source_identity'] = 'MATCHED_EXPORT_AND_REFERENCE'
    write_json(root / 'manifest.json', manifest)
    board = job / 'board'
    board.mkdir()
    native = native_result((root, manifest), board / 'results', evidence='device-hiai')
    native.update(runtime_version='test-runtime', firmware_id='test-firmware',
                  process_id=4321, model_load_api='legacy-buffer', runtime_stage='COMPLETE')
    write_json(board / 'results/result.json', native)
    (board / 'draft.stdout').write_bytes(b'')
    (board / 'draft.stderr').write_bytes(b'completed test fixture\n')
    (board / 'draft.exitcode').write_bytes(b'0\n')
    # The historical comparison write itself was incomplete. Review must use
    # the bound native outputs, never parse or repair this obsolete report.
    (board / 'comparison.json').write_bytes(b'{"hidden_diagnostic": unfinished')
    hashes = {name: sha256_file(board / name) for name in (
        'draft.stdout', 'draft.stderr', 'draft.exitcode', 'results/result.json',
        'results/output_0000.bin', 'results/output_0001.bin')}
    execution = {'artifact_kind': 'qwen35-standalone-draft-execution', 'status': 'FAILED',
                 'bundle_manifest_sha256': sha256_file(root / 'manifest.json'),
                 'retrieved_sha256': hashes, 'retrieval_errors': [], 'remote_exitcode': 0,
                 'runtime_version': 'test-runtime', 'firmware_id': 'test-firmware',
                 'ddk_id': 'fixture', 'model_load_api': 'legacy-buffer',
                 'comparison_error': 'ValueError: Out of range float values are not JSON compliant'}
    write_json(board / 'result.json', execution)
    write_json(job / 'result.json', {'schema_version': 1, 'artifact_kind': 'qwen35-full-draft-job',
        'status': 'FAILED', 'stage': 'board', 'board': str(board), 'components': {
            'bundle': {'directory': str(root), 'manifest': record(root / 'manifest.json', root)}}})

    def no_execution(*args, **kwargs):
        pytest.fail('retained comparison must not invoke HDC, compilation or execution')
    monkeypatch.setattr(subprocess, 'run', no_execution)
    monkeypatch.setattr(draft_board, 'Hdc', no_execution)
    monkeypatch.setattr(draft_board, '_build_runner', no_execution)
    monkeypatch.setattr(draft_board, 'execute_draft_bundle', no_execution)
    return {'job': job, 'bundle': root, 'board': board, 'manifest': manifest,
            'native': native, 'execution': execution}


def _snapshot(root):
    return {path.relative_to(root).as_posix(): sha256_file(path)
            for path in root.rglob('*') if path.is_file()}


def _store_execution(case):
    write_json(case['board'] / 'result.json', case['execution'])


def _store_native(case):
    path = case['board'] / 'results/result.json'
    write_json(path, case['native'])
    case['execution']['retrieved_sha256']['results/result.json'] = sha256_file(path)
    _store_execution(case)


def _rebind_output(case, index):
    name = f'output_{index:04d}.bin'
    path = case['board'] / 'results' / name
    digest = sha256_file(path)
    case['native']['outputs'][index]['sha256'] = digest
    case['execution']['retrieved_sha256']['results/' + name] = digest
    _store_native(case)


def _cli():
    script = Path(__file__).resolve().parents[1] / 'targets/cpro/scripts/run_full_draft.py'
    return runpy.run_path(str(script))['main']


@pytest.mark.parametrize('changed', [False, True])
def test_small_norm_audit_binds_reference_and_reads_no_other_weights(tmp_path, changed):
    prepared, fixture = tmp_path/'prepared', tmp_path/'fixture'
    prepared.mkdir(); fixture.mkdir()
    norm = prepared/'norm.bin'
    np.asarray([0, 1, 2, 3], dtype='<f2').tofile(norm)
    write_json(prepared/'prepared.json', {'weights': {
        'draft.norm.weight': record(norm, prepared, dtype='float16'),
        'draft.fc.weight': {'path': 'deliberately-unavailable.bin'},
    }})
    write_json(fixture/'report.json', {'prepared_manifest_sha256': sha256_file(prepared/'prepared.json'),
                                     'reference': 'PYTORCH_FP16_PRIMITIVES'})
    job = {'components': {name: {'directory': str(folder), 'manifest': record(folder/file, folder)}
                          for name, folder, file in [('prepared', prepared, 'prepared.json'),
                                                     ('fixture', fixture, 'report.json')]}}
    manifest = {'fixture_report_sha256': sha256_file(fixture/'report.json')}
    if changed:
        norm.write_bytes(b'changed!')
        with pytest.raises(ValueError, match='SHA256/size mismatch'):
            draft_review._reference_audit(job, manifest)
    else:
        audit = draft_review._reference_audit(job, manifest)
        assert audit['manifest_binding'] == 'MATCHED'
        assert audit['normalization_tensors_checked'] == 1
        assert audit['all_zero_weights'] == audit['nonfinite_weights'] == []
        assert audit['final_norms'][0]['nonzero_count'] == 3
        json.dumps(audit, allow_nan=False)


def test_recovers_old_comparison_error_with_real_strict_comparison_without_writes(retained):
    before = _snapshot(retained['job'])
    report = draft_review.review_draft_result(retained['job'])
    assert report['artifact_kind'] == 'qwen35-retained-draft-comparison'
    assert report['status'] == 'EXECUTION_COMPLETE'
    assert report['device_execution'] == 'PASSED'
    assert report['numerical_validation'] == 'TOP1_MATCHED'
    assert report['comparison']['token_mismatches'] == 0
    assert report['comparison']['host_input_readback'] == 'MATCHED'
    assert report['new_device_execution'] == report['new_compilation'] == 'NOT_RUN'
    assert report['operation'] == 'HOST_COMPARISON_ONLY'
    assert report['native_status'] == 'EXECUTED'
    assert report['native_process_id'] == 4321
    assert report['runtime_stage'] == 'COMPLETE'
    assert report['model_load_api'] == 'legacy-buffer'
    assert report['performance'] == 'NOT_EVALUATED'
    assert 'did not record inference duration' in report['performance_reason']
    assert report['previous_comparison_error'] == retained['execution']['comparison_error']
    assert report['job_result_sha256'] == before['result.json']
    assert report['board_result_sha256'] == before['board/result.json']
    assert len(report['retrieved_files']) == 6
    json.dumps(report, allow_nan=False)
    assert _snapshot(retained['job']) == before


@pytest.mark.parametrize('name', ['draft.stdout', 'draft.stderr', 'draft.exitcode',
                                 'results/result.json', 'results/output_0000.bin', 'results/output_0001.bin'])
def test_tampered_retrieved_bytes_are_never_promoted(retained, name):
    path = retained['board'] / name
    path.write_bytes(path.read_bytes() + b'changed')
    before = _snapshot(retained['job'])
    with pytest.raises(ValueError, match='SHA256/size mismatch'):
        draft_review.review_draft_result(retained['job'])
    assert _snapshot(retained['job']) == before


@pytest.mark.parametrize('fault', ['missing-hash', 'bad-hash', 'transport', 'retrieval',
    'nonzero-exit', 'bool-exit', 'exit-conflict', 'bundle-binding', 'wrong-board-kind', 'ddk'])
def test_incomplete_or_conflicting_transport_evidence_is_rejected(retained, fault):
    execution = retained['execution']
    if fault == 'missing-hash':
        execution['retrieved_sha256'].pop('draft.stderr')
    elif fault == 'bad-hash':
        execution['retrieved_sha256']['draft.stderr'] = 'invalid'
    elif fault == 'transport':
        execution['transport_error'] = 'HDC disconnected'
    elif fault == 'retrieval':
        execution['retrieval_errors'] = ['receive failed']
    elif fault == 'nonzero-exit':
        execution['remote_exitcode'] = 1
    elif fault == 'bool-exit':
        execution['remote_exitcode'] = False
    elif fault == 'exit-conflict':
        path = retained['board'] / 'draft.exitcode'
        path.write_bytes(b'7\n')
        execution['retrieved_sha256']['draft.exitcode'] = sha256_file(path)
    elif fault == 'bundle-binding':
        execution['bundle_manifest_sha256'] = '0' * 64
    elif fault == 'wrong-board-kind':
        execution['artifact_kind'] = 'unrelated-execution'
    elif fault == 'ddk':
        execution['ddk_id'] = 'wrong-ddk'
    _store_execution(retained)
    before = _snapshot(retained['job'])
    with pytest.raises(ValueError):
        draft_review.review_draft_result(retained['job'])
    assert _snapshot(retained['job']) == before


@pytest.mark.parametrize('fault', ['runtime', 'firmware', 'model', 'plan', 'host-fake',
    'loader', 'input-readback', 'output-identity', 'native-failed'])
def test_hash_bound_native_still_must_pass_identity_and_io_gates(retained, fault):
    native = retained['native']
    if fault == 'runtime':
        native['runtime_version'] = 'other-runtime'
    elif fault == 'firmware':
        native['firmware_id'] = 'other-firmware'
    elif fault == 'model':
        native['model_sha256'] = '0' * 64
    elif fault == 'plan':
        native['step_plan_sha256'] = '0' * 64
    elif fault == 'host-fake':
        native['evidence_kind'] = 'host-fake-hiai'
    elif fault == 'loader':
        native['model_load_api'] = 'file-v2'
    elif fault == 'input-readback':
        native['io_diagnostic']['inputs'][0]['readback_matches_source'] = False
    elif fault == 'output-identity':
        native['io_diagnostic']['outputs'][0]['buffer_same'] = False
    elif fault == 'native-failed':
        native['status'] = 'RUNTIME_FAILED'
    _store_native(retained)
    before = _snapshot(retained['job'])
    with pytest.raises(ValueError):
        draft_review.review_draft_result(retained['job'])
    assert _snapshot(retained['job']) == before


@pytest.mark.parametrize('fault', ['job-kind', 'bundle-manifest', 'board-path', 'missing-bundle'])
def test_current_job_must_bind_retained_bundle_and_board(retained, fault):
    path = retained['job'] / 'result.json'
    job = json.loads(path.read_text())
    if fault == 'job-kind':
        job['artifact_kind'] = 'unrelated-job'
    elif fault == 'bundle-manifest':
        job['components']['bundle']['manifest']['sha256'] = '0' * 64
    elif fault == 'board-path':
        job['board'] = str(retained['job'] / 'other-board')
    elif fault == 'missing-bundle':
        job['components'].pop('bundle')
    write_json(path, job)
    with pytest.raises(ValueError):
        draft_review.review_draft_result(retained['job'])


def test_bound_top1_mismatch_preserves_execution_success_and_fails_numerics(retained):
    path = retained['board'] / 'results/output_0000.bin'
    np.zeros((1, 15), dtype='<i4').tofile(path)
    _rebind_output(retained, 0)
    report = draft_review.review_draft_result(retained['job'])
    assert report['status'] == 'FAILED'
    assert report['device_execution'] == 'PASSED'
    assert report['numerical_validation'] == 'FAILED'
    assert report['comparison']['token_mismatches'] == 15
    json.dumps(report, allow_nan=False)


def test_bound_nonfinite_hidden_returns_json_safe_numeric_failure(retained):
    path = retained['board'] / 'results/output_0001.bin'
    np.full((1, 15, 2048), np.inf, dtype='<f2').tofile(path)
    _rebind_output(retained, 1)
    report = draft_review.review_draft_result(retained['job'])
    assert report['status'] == 'FAILED'
    assert report['device_execution'] == 'PASSED'
    assert report['numerical_validation'] == 'FAILED'
    json.dumps(report, allow_nan=False)


def test_compare_only_cli_repeats_locally_without_reset_or_original_mutation(retained):
    main = _cli()
    before = _snapshot(retained['job'])
    for _ in range(2):
        assert main(['--compare-only', '--resume', str(retained['job'])]) == 0
        report = json.loads((retained['job'] / 'comparison-review.json').read_text())
        assert report['status'] == 'EXECUTION_COMPLETE'
        assert report['new_device_execution'] == 'NOT_RUN'
        assert all(sha256_file(retained['job'] / name) == digest for name, digest in before.items())
    created = set(_snapshot(retained['job'])) - set(before)
    assert created == {'comparison-review.json', '.draft-workspace.json', '.draft-workspace.lock'}


def test_compare_only_cli_returns_failure_for_numeric_mismatch_without_reexecution(retained):
    np.zeros((1, 15), dtype='<i4').tofile(retained['board'] / 'results/output_0000.bin')
    _rebind_output(retained, 0)
    before = _snapshot(retained['job'])
    assert _cli()(['--compare-only']) == 1
    report = json.loads((retained['job'] / 'comparison-review.json').read_text())
    assert report['status'] == 'FAILED' and report['device_execution'] == 'PASSED'
    assert all(sha256_file(retained['job'] / name) == digest for name, digest in before.items())


def test_compare_only_cli_obeys_existing_workspace_lock(retained):
    main = _cli()
    with DraftWorkspace(retained['job']):
        before = _snapshot(retained['job'])
        assert main(['--compare-only']) == 1
        assert _snapshot(retained['job']) == before
    assert not (retained['job'] / 'comparison-review.json').exists()


@pytest.mark.parametrize('extra', [['--weight-storage', 'embedded-stream'], ['--model-load-api', 'file-v2'],
                                 ['--sequence-length', '8']])
def test_compare_only_cli_rejects_changed_configuration_without_touching_job(retained, extra):
    before = _snapshot(retained['job'])
    assert _cli()(['--compare-only', *extra]) == 1
    assert _snapshot(retained['job']) == before
