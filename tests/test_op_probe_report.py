import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np
import pytest

from qwen35_moe_dflash.op_probe_report import compare_outputs, compile_evidence, write_report


def test_compiler_candidates_are_not_placement_and_scopes_remain_separate():
    log = '''start InferShape for graph[constNodeGraph] ....
opDeviceSupport: op=constmm, devices=[1,0,]
partition type NPU:0, CPU:1, GPU:0, ISP:0.
TCH ConstantFolding end..
partition type NPU:1, CPU:0, GPU:0, ISP:0.
start InferShape for graph[probe_main] ....
[CPUCL] gemm.cpp CheckSupport(90):input1 dtype should be fp32, but now it is fp16
partition type NPU:1, CPU:0, GPU:0, ISP:0.
subGraph SubGraph_0 run_by_cl NPUCL
'''
    result = compile_evidence(log)
    assert [e['scope'] for e in result['partition_events']] == ['constant_folding', 'unknown', 'main']
    assert result['main_partition_events'][0]['counts']['NPU'] == 1
    assert result['constant_folding_partition_events'][0]['counts']['CPU'] == 1
    assert result['runtime_placement'] == 'UNVERIFIED'
    assert result['support_candidate_lines'] == 2
    assert result['diagnostics'][0]['category'] == 'candidate_check'
    assert len(result['subgraph_events']) == 1


def test_compile_diagnostics_are_bounded_and_count_repeated_failures():
    log = '\n'.join(['ModuleNotFoundError: No module named te_fusion'] * 100
                    + [f'node{i} dtype must be fp16' for i in range(70)])
    result = compile_evidence(log)
    assert len(result['diagnostics']) == 32
    assert result['diagnostics'][0]['occurrences'] == 100
    assert result['diagnostic_matching_lines'] == 170
    assert result['diagnostic_omitted_distinct'] == 39


def test_specific_rejections_survive_generic_store_flood():
    noise = [f'CheckSupported node v{i} not supported in npucl store custom_ascendc' for i in range(90)]
    signals = ['Neg CheckSupport dimCnt 5 > 4', 'Slice not support realDimCnt 5',
               "MatMul doesn't exist in tbe ops store", 'ModuleNotFoundError: te_fusion',
               'input1 dtype should be fp32 but now it is fp16',
               'BeforeCompile check ir model compatibility failed']
    result = compile_evidence('\n'.join(noise + signals))
    texts = {record['text'] for record in result['diagnostics']}
    assert set(signals) <= texts
    assert len(texts) == 32
    assert result['diagnostic_matching_lines'] == 96


def _fixture(tmp_path, dtype='float16'):
    case_dir = tmp_path / 'case'
    board_dir = tmp_path / 'board' / 'case'
    case_dir.mkdir()
    board_dir.mkdir(parents=True)
    values = np.array([[1.0, -2.0, 0.125, 4.0]], dtype=dtype)
    values.tofile(case_dir / 'expected_0.bin')
    values.tofile(board_dir / 'output_0.bin')
    case = {'name': 'case', 'description': 'test', 'directory': str(case_dir),
            'compile': {'status': 'COMPILED', 'evidence': compile_evidence('')},
            'outputs': [{'path': 'output_0.bin', 'expected': 'expected_0.bin', 'dtype': dtype,
                         'shape': [1, 4], 'rtol': .01, 'atol': .002}]}
    return case, board_dir, values


def test_comparison_fp16_tolerance_and_mismatch_details(tmp_path):
    case, output, values = _fixture(tmp_path)
    assert compare_outputs(case, output)['status'] == 'PASSED'
    values[0, 0] = np.float16(1.007)
    values.tofile(output / 'output_0.bin')
    assert compare_outputs(case, output)['status'] == 'PASSED'
    values[0, 1] = -1
    values.tofile(output / 'output_0.bin')
    result = compare_outputs(case, output)
    assert result['status'] == 'FAILED'
    assert result['outputs'][0]['mismatches'] == 1
    assert result['outputs'][0]['examples'] == [{'flat_index': 1, 'actual': -1.0, 'expected': -2.0}]


def test_integer_comparison_is_exact_even_with_large_tolerance(tmp_path):
    case, output, values = _fixture(tmp_path, 'int32')
    case['outputs'][0].update(rtol=10, atol=10000)
    values[0, 0] += 1
    values.tofile(output / 'output_0.bin')
    result = compare_outputs(case, output)
    assert result['status'] == 'FAILED'
    assert result['outputs'][0]['rtol'] == result['outputs'][0]['atol'] == 0


def test_reference_and_retrieved_output_hashes_are_checked(tmp_path):
    case, output, values = _fixture(tmp_path)
    expected = Path(case['directory']) / 'expected_0.bin'
    original = hashlib.sha256(expected.read_bytes()).hexdigest()
    case['outputs'][0]['reference'] = {'sha256': original}
    actual_metadata = [{'path': str(output / 'output_0.bin'), 'sha256': original}]
    assert compare_outputs(case, output, actual_metadata)['status'] == 'PASSED'
    values[0, 0] = 3
    values.tofile(expected)
    values.tofile(output / 'output_0.bin')
    result = compare_outputs(case, output, actual_metadata)
    assert result['outputs'][0]['reason'] == 'REFERENCE_HASH_MISMATCH'
    case['outputs'][0]['reference']['sha256'] = hashlib.sha256(expected.read_bytes()).hexdigest()
    result = compare_outputs(case, output, actual_metadata)
    assert result['outputs'][0]['reason'] == 'OUTPUT_HASH_MISMATCH'


@pytest.mark.parametrize('fault,reason', [('missing', 'MISSING_OUTPUT'), ('short', 'WRONG_OUTPUT_SIZE'),
                                         ('nan', 'NONFINITE'), ('inf', 'NONFINITE'),
                                         ('escape', 'INVALID_SPEC_OR_IO')])
def test_invalid_output_is_never_reported_as_numeric_success(tmp_path, fault, reason):
    case, output, values = _fixture(tmp_path)
    path = output / 'output_0.bin'
    if fault == 'missing':
        path.unlink()
    elif fault == 'short':
        path.write_bytes(b'\x00')
    elif fault in {'nan', 'inf'}:
        values[0, 0] = float(fault)
        values.tofile(path)
    else:
        case['outputs'][0]['path'] = '../outside.bin'
    result = compare_outputs(case, output)
    assert result['status'] == 'FAILED'
    assert result['outputs'][0]['reason'] == reason


def test_report_stage_status_and_archive_exclude_binaries_and_unrelated_hilog(tmp_path):
    case, board, _ = _fixture(tmp_path)
    (board / 'stdout').write_text(json.dumps({'status': 'EXECUTED', 'runtime_version': 'test', 'placement': 'NPU'}) + '\n')
    (board / 'stderr').write_text('')
    (board / 'exitcode').write_text('0\n')
    (board / 'pid').write_text('123\n')
    (board / 'hilog.txt').write_text('03-06 12:30:00.123 123 124 I A0FFFF/probe/HIAI: own log\n'
                                   '03-06 12:30:00.124 456 789 I A0FFFF/other/HIAI: unrelated secret\n')
    (board / 'model.omc').write_bytes(b'not for archive')
    (tmp_path / 'raw-hilog.txt').write_text('unrelated secret')
    report = write_report(tmp_path, {'cases': [case]}, {'status': 'TRANSPORT_COMPLETE', 'cases': {}})
    row = report['cases'][0]
    assert row['load'] == row['run'] == row['numerical']['status'] == 'PASSED'
    assert row['placement'] == 'UNVERIFIED'
    with tarfile.open(tmp_path / 'results.tar.gz') as archive:
        names = archive.getnames()
        assert all(not name.endswith(('.bin', '.omc')) and 'raw-hilog' not in name for name in names)
        text = archive.extractfile('board/case/hilog.pid.txt').read().decode()
        assert 'own log' in text and 'unrelated secret' not in text
        assert 'report-manifest.json' in names and 'report.json' in names


@pytest.mark.parametrize('stage,load,run', [('load', 'FAILED', 'NOT_RUN'), ('io', 'PASSED', 'NOT_RUN'),
                                           ('process', 'PASSED', 'FAILED'), ('output', 'PASSED', 'PASSED')])
def test_native_failure_keeps_load_run_and_numerical_distinct(tmp_path, stage, load, run):
    case, board, _ = _fixture(tmp_path)
    (board / 'stdout').write_text(json.dumps({'status': 'FAILED', 'stage': stage, 'error': 'test failure'}))
    (board / 'exitcode').write_text('1')
    report = write_report(tmp_path, {'cases': [case]})
    row = report['cases'][0]
    assert (row['load'], row['run']) == (load, run)
    assert row['numerical']['status'] == 'NOT_RUN'


def test_success_without_remote_zero_exit_is_unverified(tmp_path):
    case, board, _ = _fixture(tmp_path)
    (board / 'stdout').write_text('{"status":"EXECUTED"}')
    report = write_report(tmp_path, {'cases': [case]})
    assert report['cases'][0]['run'] == 'UNVERIFIED'
    assert report['cases'][0]['numerical']['status'] == 'NOT_RUN'


def test_loader_exit_without_native_json_has_no_execution_claim(tmp_path):
    case, board, _ = _fixture(tmp_path)
    (board / 'stdout').write_text('')
    (board / 'stderr').write_text('Error loading shared library libhiai.so')
    (board / 'exitcode').write_text('127')
    report = write_report(tmp_path, {'cases': [case]})
    row = report['cases'][0]
    assert row['load'] == row['run'] == row['placement'] == 'UNVERIFIED'
    assert row['numerical']['status'] == 'NOT_RUN'
    assert 'exited 127' in row['runtime']['error']


@pytest.mark.parametrize('saved_name,status', [('failure.json', 'FAILED'), ('result.json', 'TRANSPORT_COMPLETE')])
def test_saved_transport_evidence_preserves_partial_results_and_identity(tmp_path, saved_name, status):
    case, board, _ = _fixture(tmp_path)
    (board / 'stdout').write_text('{"status":"EXECUTED"}')
    (board / 'exitcode').write_text('0')
    record = {'status': status, 'stage': 'hilog', 'error': 'transport example',
              'serial': 'test-device', 'hdc_server': 'example:1234', 'firmware': {'value': 'test'},
              'cases': {'case': {'directory': str(board)}}}
    (tmp_path / 'board' / saved_name).write_text(json.dumps(record))
    report = write_report(tmp_path, {'cases': [case], 'status': 'INFRASTRUCTURE_FAILED', 'error': 'workflow example'})
    assert report['cases'][0]['numerical']['status'] == 'PASSED'
    assert report['board_transport']['status'] == status
    assert report['board_transport']['serial'] == 'test-device'
    assert report['board_transport']['error'] == 'transport example'
    assert report['manifest_error'] == 'workflow example'
    assert 'Transport error: transport example' in (tmp_path / 'report.txt').read_text()


def test_reports_refuse_overwrite_without_modifying_existing_evidence(tmp_path):
    case, _, _ = _fixture(tmp_path)
    write_report(tmp_path, {'cases': [case]})
    before = {path: path.read_bytes() for path in (tmp_path / 'report.json', tmp_path / 'results.tar.gz')}
    with pytest.raises(FileExistsError, match='refusing to overwrite'):
        write_report(tmp_path, {'cases': []})
    assert all(path.read_bytes() == value for path, value in before.items())


def test_text_report_distinguishes_compile_partition_from_runtime_placement(tmp_path):
    case, _, _ = _fixture(tmp_path)
    case['compile']['evidence'] = compile_evidence('partition type NPU:1, CPU:0, GPU:0, ISP:0.')
    write_report(tmp_path, {'cases': [case]})
    text = (tmp_path / 'report.txt').read_text()
    assert 'compiler partition (unknown, log line 1): NPU=1, CPU=0' in text
    assert 'runtime placement unverified' in text


def test_archive_log_bounds_do_not_modify_local_logs(tmp_path):
    case, board, _ = _fixture(tmp_path)
    oversized = 'HEAD\n' + 'm' * (300 * 1024) + '\nTAIL'
    (board / 'stderr').write_text(oversized)
    report = write_report(tmp_path, {'cases': [case]})
    assert (board / 'stderr').read_text() == oversized
    assert report['archive_notes']
    with tarfile.open(tmp_path / 'results.tar.gz') as archive:
        excerpt = archive.extractfile('board/case/stderr').read().decode()
        assert len(excerpt) < 257 * 1024
        assert excerpt.startswith('HEAD') and excerpt.endswith('TAIL')
        assert 'middle bytes' in excerpt
