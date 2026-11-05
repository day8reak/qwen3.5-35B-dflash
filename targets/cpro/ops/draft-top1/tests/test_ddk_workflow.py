#!/usr/bin/env python3
"""Host workflow tests with forbidden Torch imports and mocked SDK calls."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import importlib.abc
import io
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'torch_npu', 'torchair'):
            raise AssertionError('DDK path imported a forbidden framework: ' + fullname)


def check(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    sys.meta_path.insert(0, NoTorch())
    import prepare_allops as staging
    import stable_arg_max_ge as probe
    import numpy as np
    sys.path.insert(0, str(probe.REPO / 'targets/cpro/tools'))
    import validate_draft_vector as board
    checks = []

    def rejects(fn, name):
        try:
            fn()
        except (ValueError, FileExistsError):
            checks.append(name)
        else:
            raise AssertionError(name + ': did not reject')

    allops = output / 'allops'
    for path in ('op_host', 'op_kernel'):
        (allops / path).mkdir(parents=True)
    for name in ('CMakeLists.txt', 'op_host/other.cpp', 'op_kernel/other.cpp'):
        (allops / name).write_text('existing allops file')
    staging.prepare(allops, output / 'copy')
    assert sorted(str(p.relative_to(allops)) for p in allops.rglob('*') if p.is_file()) == sorted(
        [*staging.FILES, 'CMakeLists.txt', 'op_host/other.cpp', 'op_kernel/other.cpp'])
    assert (allops / 'CMakeLists.txt').read_text() == 'existing allops file'
    checks.append('copy_exactly_three_sources_preserve_other_ops_and_build')
    staging.prepare(allops, output / 'identical')
    checks.append('identical_source_idempotent')
    dest = allops / staging.FILES[0]
    dest.write_text('previous source')
    rejects(lambda: staging.prepare(allops, output / 'conflict'), 'conflict_without_overwrite')
    staging.prepare(allops, output / 'replace', replace=True)
    assert (output / 'replace/previous' / staging.FILES[0]).read_text() == 'previous source'
    checks.append('replace_retains_previous_source')
    dest.unlink()
    dest.symlink_to(allops / 'CMakeLists.txt')
    rejects(lambda: staging.prepare(allops, output / 'symlink', True), 'reject_linked_destination')
    rejects(lambda: staging.prepare(allops, allops / 'report'), 'report_outside_allops')

    def args(folder, shape=(1, 15, 248320), plan=True, export=False):
        return argparse.Namespace(output=str(output / folder), shape=list(shape), plan_only=plan,
            export_only=export, ge_root=None, ddk_root=None, ddk_id=None,
            cxx='c++', cxx11_abi=0, timeout_seconds=30, tbe_diagnosis=None, reuse_export=None)

    qwen = probe.run(args('qwen'))
    assert len(qwen['cases']) == 7 and qwen['model_compilation'] == 'NOT_RUN'
    assert qwen['graph']['custom_ops'][0]['type'] == 'StableArgMax'
    source = (output / 'qwen/probe.cpp').read_text()
    assert '"StableArgMax", {"x"}, {"indices"}' in source
    assert '"ArgMaxExt2"' not in source and '"Cast"' not in source
    checks.append('single_named_custom_ge_node_no_builtin_or_cast_qwen_seven_fixtures')
    for shape in ((17, 8193), (1,), (1, 1, 1, 1, 1, 1, 1, 7)):
        row = probe.run(args('rank-' + str(len(shape)), shape))
        assert row['graph']['outputs'][0]['shape'] == list(shape[:-1])
        checks.append('static_shape_export_plan/' + str(shape))
    for shape in ((), (0, 5), (1,) * 9, (2147483648,)):
        rejects(lambda: probe.checked_shape(shape), 'reject_invalid_shape/' + str(shape))
    rejects(lambda: probe.run(args('too-large', (15, 1000000))), 'bounded_fixture_io')
    rejects(lambda: probe.run(args('qwen')), 'preserve_existing_output')
    rejects(lambda: probe.run(args('missing-ge', plan=False)), 'require_ge_sdk_not_torchair')

    # Exercise export/OMG and board adapter wiring without claiming SDK evidence.
    fakege = output / 'ge'
    for name in ('include/graph/graph.h', 'lib64/libgraph.so'):
        path = fakege / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('mock GE SDK')
    calls = []
    compiler_kwargs = []

    def logged(argv, **kwargs):
        calls.append(argv)
        if str(argv[-1]).endswith('.air'):
            Path(argv[-1]).write_bytes(b'MOCK AIR')

    def compile_air(air, folder, ddk, **kwargs):
        calls.append(['OMG', str(air)])
        compiler_kwargs.append(kwargs)
        kwargs['progress'](dict(stage='omg-wait', message='Waiting for OMG: elapsed=30s; log=123 bytes'))
        folder.mkdir()
        (folder / 'model.omc').write_bytes(b'MOCK OMC')
        report = dict(status='COMPILED', model_compilation='PASSED', platform='kirinx90',
            ddk_id='mock-ddk', omc=dict(path='model.omc'))
        if kwargs.get('tbe_diagnosis'):
            report['tbe_compatibility'] = dict(diagnosis=kwargs['tbe_diagnosis'], audit={'status': 'APPLIED'})
        probe.write(folder / 'report.json', report)
        return report

    def inspect(model, ddk, graph):
        io = {}
        for direction in ('inputs', 'outputs'):
            spec = dict(graph[direction][0], index=0, fixture_index=0)
            spec['bytes'] = math.prod(spec['shape']) * (2 if direction == 'inputs' else 4)
            io[direction] = [spec]
        return dict(placement=dict(npu_only_candidate=True, graphop_counts={'NPUCL': 1}), io=io, schema={})

    actual_args = args('compiled', (17, 8193), plan=False)
    actual_args.ge_root, actual_args.ddk_root, actual_args.ddk_id = str(fakege), str(output / 'ddk'), 'mock-ddk'
    with patch('qwen35_moe_dflash.graph.export._logged', logged), \
         patch('qwen35_moe_dflash.toolchain.existing_air.compile_existing_air', compile_air), \
         patch.object(probe, 'inspect_model', inspect):
        result = probe.run(actual_args)
        assert result['status'] == 'COMPILED' and sum(c[0] == 'OMG' for c in calls) == 1
        assert compiler_kwargs[-1]['tbe_diagnosis'] is None
        checks.append('mock_sdk_one_compile_reused_for_all_fixtures')
        compatible = args('compatible-fresh', (2, 3), plan=False)
        compatible.ge_root, compatible.ddk_root, compatible.ddk_id = str(fakege), str(output / 'ddk'), 'mock-ddk'
        compatible.tbe_diagnosis = str(output / 'retained-diagnosis')
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            compatible_report = probe.run(compatible)
        assert 'preflight:' in stderr.getvalue() and 'omg-prepare:' in stderr.getvalue()
        assert 'omg-wait: Waiting for OMG: elapsed=30s; log=123 bytes' in stderr.getvalue()
        assert 'inspect:' in stderr.getvalue()
        checks.append('stage_and_existing_compiler_wait_callback_are_visible_on_stderr')
        assert compiler_kwargs[-1]['tbe_diagnosis'] == compatible.tbe_diagnosis
        assert compatible_report['tbe_compatibility']['audit_status'] == 'APPLIED'
        checks.append('fresh_compile_forwards_evidence_bound_tbe_option_and_reports_worker_audit')
        old = output / 'compatible-fresh'
        # Match the user's old failed report after a CLI-only source update.
        saved_report = json.loads((old / 'result.json').read_text())
        saved_report['status'] = 'FAILED'; saved_report['stage'] = 'omg'
        saved_report['model_compilation'] = 'FAILED'
        saved_report['source_sha256']['scripts/stable_arg_max_ge.py'] = 'old-diagnostic-script'
        probe.write(old / 'result.json', saved_report)
        snapshot = {str(p): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in old.rglob('*') if p.is_file()}
        reuse = args('reuse', (2, 3), plan=False)
        reuse.shape = None
        reuse.ge_root = None
        reuse.ddk_root, reuse.ddk_id = str(output / 'ddk'), 'mock-ddk'
        reuse.reuse_export, reuse.tbe_diagnosis = str(old), compatible.tbe_diagnosis
        before = len(calls)
        with patch('qwen35_moe_dflash.graph.export._logged', side_effect=AssertionError('re-exported')):
            reused = probe.run(reuse)
        assert len(calls) == before + 1 and compiler_kwargs[-1]['tbe_diagnosis'] == reuse.tbe_diagnosis
        assert reused['shape'] == [2, 3] and reused['reused_export']['air_sha256'] == saved_report['air']['sha256']
        assert (output / 'reuse/air/stable_arg_max.air').read_bytes() == (old / 'air/stable_arg_max.air').read_bytes()
        assert snapshot == {str(p): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in old.rglob('*') if p.is_file()}
        checks.append('failed_export_reuse_without_ge_sdk_or_reexport_old_bytes_preserved_one_omg')
        reuse.shape = [2, 4]
        reuse.output = str(output / 'shape-mismatch')
        rejects(lambda: probe.run(reuse), 'reject_reuse_shape_mismatch_before_compile')
        reuse.shape = None
        for field, value in (('graph_cpp_sha256', 'changed'), ('graph', {})):
            edited = dict(saved_report); edited[field] = value
            probe.write(old / 'result.json', edited)
            rejects(lambda: probe.retained_export(old), 'reject_reuse_changed_' + field)
        edited = json.loads(json.dumps(saved_report))
        edited['source_sha256']['op_kernel/stable_arg_max.cpp'] = 'changed'
        probe.write(old / 'result.json', edited)
        rejects(lambda: probe.retained_export(old), 'reject_reuse_changed_kernel')
        probe.write(old / 'result.json', saved_report)
        air_path = old / 'air/stable_arg_max.air'
        original = air_path.read_bytes(); air_path.write_bytes(b'changed AIR')
        rejects(lambda: probe.retained_export(old), 'reject_reuse_changed_air')
        air_path.write_bytes(original)
        inp = old / 'fixture/stable_arg_max__first_last/input_0.bin'
        original = inp.read_bytes(); inp.write_bytes(b'changed fixture')
        reuse.output = str(output / 'reuse-bad-fixture')
        before = len(calls)
        try:
            probe.run(reuse)
        except probe.ProbeFailure as error:
            assert 'fixture' in error.report['error']
        else:
            raise AssertionError('changed fixture reused')
        assert len(calls) == before
        inp.write_bytes(original)
        checks.append('reject_reuse_changed_fixture_before_compilation')
        root, execution = output / 'compiled', output / 'execution'
        metadata = board.prepare(root, execution, actual_args.ddk_root, 'mock-ddk')
        assert len(metadata['cases']) == 4 and metadata['exact_bytes']
        assert metadata['warmup'] == 3 and metadata['repetitions'] == 10
        assert board.selected_cases(c['case'] for c in metadata['cases'])
        checks.append('board_adapter_no_torch_exact_bytes_3_warmup_10_repeats_recovery_names')
        native = output / 'native'
        for case in metadata['cases']:
            name = case['case']
            folder = native / name
            folder.mkdir(parents=True)
            golden = (execution / 'fixture' / name / 'golden_0.bin').read_bytes()
            for i in range(10):
                (folder / f'{i}.bin').write_bytes(golden)
            probe.write(folder / 'result.json', dict(evidence_kind='host-fake-hiai', status='EXECUTED',
                warmup=3, repetitions=10, api_us=[1] * 10, timing_scope='SYNCHRONOUS_API_WALL_NOT_NPU_ONLY'))
        result = board.compare(execution / 'fixture', native, allow_host_fake=True)
        assert result['numerical_validation'] == 'PASSED'
        name = metadata['cases'][0]['case']
        path = native / name / '9.bin'
        data = bytearray(path.read_bytes()); data[0] ^= 1; path.write_bytes(data)
        assert board.compare(execution / 'fixture', native, allow_host_fake=True)['numerical_validation'] == 'FAILED'
        checks.append('synthetic_comparison_rejects_one_wrong_byte_in_tenth_repeat')
        rejects(lambda: board.prepare(root, output / 'wrong-ddk', actual_args.ddk_root, 'other'), 'reject_wrong_ddk')
        model = root / 'omc/model.omc'
        model.write_bytes(b'changed')
        rejects(lambda: board.prepare(root, output / 'changed-omc', actual_args.ddk_root, 'mock-ddk'), 'reject_changed_omc')
        model.write_bytes(b'MOCK OMC')
        inp = root / 'fixture' / name / 'input_0.bin'
        old = inp.read_bytes(); inp.write_bytes(b'changed')
        rejects(lambda: board.prepare(root, output / 'changed-input', actual_args.ddk_root, 'mock-ddk'), 'reject_changed_fixture')
        inp.write_bytes(old)
        with patch.object(probe, 'inspect_model', return_value={'placement': {'npu_only_candidate': False}}):
            rejects(lambda: board.prepare(root, output / 'cpu-placement', actual_args.ddk_root, 'mock-ddk'), 'reject_cpu_placement')
        export_args = args('export-only', (2, 3), plan=False, export=True)
        export_args.ge_root = str(fakege)
        before = len(calls)
        assert probe.run(export_args)['status'] == 'EXPORTED'
        assert len(calls) == before + 2
        checks.append('export_only_does_not_require_ddk_or_run_omg')
        failure_args = args('failed-export', (2, 3), plan=False, export=True)
        failure_args.ge_root = str(fakege)
        with patch('qwen35_moe_dflash.graph.export._logged', side_effect=RuntimeError('mock GE error')):
            try:
                probe.run(failure_args)
            except RuntimeError:
                pass
        saved = json.loads((output / 'failed-export/result.json').read_text())
        assert saved['status'] == 'FAILED' and saved['stage'] == 'ge-export'
        assert saved['model_compilation'] == 'NOT_RUN'
        checks.append('retain_failed_export_stage_no_false_compilation_pass')
        def compile_failure(air, folder, ddk, **kwargs):
            kwargs['progress'](dict(stage='omg', message='Compiling existing AIR'))
            folder.mkdir()
            saved = dict(status='FAILED', stage='omg', model_compilation='FAILED', returncode=1,
                error='CalledProcessError: synthetic OMG exited 1')
            probe.write(folder / 'report.json', saved)
            (folder / 'compile.log').write_text('E/AI_NPUCL: node top1 type StableArgMax is not supported\n')
            return saved

        with patch('qwen35_moe_dflash.toolchain.existing_air.compile_existing_air', compile_failure):
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = probe.main(['--shape', '2', '3', '--output', str(output / 'failed-omg'),
                    '--ge-root', str(fakege), '--ddk-root', str(output / 'ddk'), '--ddk-id', 'mock-ddk'])
            failed = json.loads(stdout.getvalue())
            assert code == 1 and failed['status'] == 'FAILED' and failed['stage'] == 'omg'
            assert failed['model_compilation'] == 'FAILED'
            assert 'StableArgMax' in failed['compilation_diagnostics']['log_excerpt']
            assert 'synthetic OMG exited 1' in failed['error']
            assert json.loads((output / 'failed-omg/result.json').read_text()) == failed
            assert 'Compiling existing AIR' in stderr.getvalue()
            checks.append('progress_does_not_pollute_final_stdout_json_or_mask_compiler_failure')
            checks.append('failed_omg_json_includes_original_error_and_concrete_compiler_log')
        snapshot = {str(p): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in (output / 'failed-omg').rglob('*') if p.is_file()}
        # Old failures are inspectable after a tooling/source update, without retry.
        with patch.object(probe, 'run', side_effect=AssertionError('inspection recompiled')), \
             patch.object(probe, 'source_hashes', side_effect=AssertionError('inspection requires current sources')):
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                assert probe.main(['--inspect-compilation', str(output / 'failed-omg')]) == 0
            inspected = json.loads(stdout.getvalue())
            assert inspected['source_model_compilation'] == 'FAILED'
            assert inspected['model_compilation'] == inspected['device_execution'] == 'NOT_RUN'
            assert 'StableArgMax' in inspected['compilation_diagnostics']['log_excerpt']
        assert snapshot == {str(p): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in (output / 'failed-omg').rglob('*') if p.is_file()}
        checks.append('inspect_old_failed_run_read_only_no_source_gate_sdk_or_retry')
        (output / 'failed-omg/omc/compile.log').unlink()
        assert probe.inspect_compilation(output / 'failed-omg')['compilation_diagnostics']['log_excerpt'] == ''
        checks.append('inspect_missing_log_retains_compiler_report')
        (output / 'failed-omg/omc/report.json').write_text('invalid JSON')
        assert 'report_error' in probe.inspect_compilation(output / 'failed-omg')['compilation_diagnostics']
        checks.append('inspect_malformed_compiler_report_is_diagnostic_not_retry')
        probe.write(output / 'foreign.json', {'artifact_kind': 'other'})
        foreign = output / 'foreign'; foreign.mkdir()
        probe.write(foreign / 'result.json', {'artifact_kind': 'other'})
        rejects(lambda: probe.inspect_compilation(foreign), 'reject_foreign_probe_report')
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            assert probe.main(['--output', str(output / 'compiled'), '--plan-only']) == 1
        preflight = json.loads(stdout.getvalue())
        assert preflight['stage'] == 'preflight' and preflight['model_compilation'] == 'NOT_RUN'
        assert json.loads((output / 'compiled/result.json').read_text())['status'] == 'COMPILED'
        checks.append('preflight_json_does_not_relabel_or_overwrite_old_run')
    assert not any(n.split('.')[0] in ('torch', 'torch_npu', 'torchair') for n in sys.modules)
    checks.append('all_ddk_paths_and_board_preparation_without_framework_imports')
    report = dict(status='PASSED', checks=checks, count=len(checks),
        scope='HOST_WORKFLOW_WITH_MOCK_SDK_AND_SYNTHETIC_OUTPUTS; NO_REAL_GE_OMG_DEVICE_EXECUTION',
        source_sha256=probe.source_hashes())
    probe.write(output / 'validation.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    print(json.dumps(check(parser.parse_args().output)))
