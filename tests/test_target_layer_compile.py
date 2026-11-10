"""Host regressions for separate AIR/OMC batches; no device/compiler success claim."""
from contextlib import ExitStack
import os
from pathlib import Path
import shutil
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from qwen35_moe_dflash.target.layer_probe.batch import compile_batch, prepare_fixture
from qwen35_moe_dflash.target.layer_probe.common import KIND, identity, save_json, sha


class CompileBatchTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ['AI_RUN_DIR'])
        self.root = Path(self.temp.name)
        self.air = self.root/'air-batch'
        self.ddk = self.root/'ddk'
        self.ddk.mkdir()
        self.output = self.root/'omc-batch-001'
        cases = []
        for call, phase in enumerate(('prefill', 'decode')):
            name = f'l00_c{call:02d}_input_layernorm_o0'
            folder = self.air/'models'/name
            (folder/'fixture').mkdir(parents=True)
            (folder/'export.air').write_bytes(b'host AIR fixture')
            # A neighboring sidecar must never be moved or rewritten by the batch.
            (folder/'weight').mkdir()
            (folder/'weight/tensor').write_bytes(b'host weight fixture')
            specs = {}
            for tensor in ('input', 'golden'):
                path = folder/'fixture'/f'{tensor}.bin'
                path.write_bytes(struct.pack('<2e', call + 1, call + 2))
                specs[tensor] = dict(identity(path, folder), name=tensor, dtype='float16', shape=[2])
            save_json(folder/'export.json', dict(status='EXPORTED',
                air=identity(folder/'export.air', folder), source_inputs=[specs['input']], golden=specs['golden']))
            cases.append(dict(id=name, status='EXPORTED', export=identity(folder/'export.json', self.air),
                layer=0, call=call, point='input_layernorm', kind='norm', phase=phase,
                ordinal=0, output_name='y'))
        save_json(self.air/'manifest.json', dict(artifact_kind=KIND, status='EXPORTED', cases=cases,
            scope='HOST_TEST', golden_source='host fixture', reference_model_layers=4))
        # Reproduce the user's old failed report without asking them to delete it.
        save_json(self.air/'compile.json', dict(artifact_kind=KIND, status='PARTIAL',
            cases={c['id']: dict(status='FAILED', error='compilation output must be outside the source and DDK directories')
                   for c in cases}))
        self.names = [c['id'] for c in cases]

    def tearDown(self):
        self.temp.cleanup()

    def fake_compiler(self, air, output, ddk, **kwargs):
        self.assertNotIn(air.parent, output.parents)
        self.assertNotIn(ddk, output.parents)
        self.assertEqual(ddk, self.ddk)
        self.assertFalse(kwargs['save_weights_as_external_data'])
        self.assertIsNone(kwargs['limit_partition_size'])
        self.assertEqual(kwargs['compiler_profile'], 'isolated-kirinx90')
        output.mkdir(parents=True, exist_ok=False)
        model = output/'model.omc'
        model.write_bytes(b'host OMC fixture')
        return dict(status='COMPILED', omc=identity(model, output))

    def compile(self, output=None, **kwargs):
        with patch('qwen35_moe_dflash.toolchain.existing_air.compile_existing_air', side_effect=self.fake_compiler) as compiler:
            result = compile_batch(self.air, output or self.output, self.ddk, 'test-ddk', **kwargs)
        return result, compiler

    def fixtures(self, compiled=None, output=None, **kwargs):
        doc = {'graph': [{'input': ['input:0'], 'op': [
            dict(type='Data', name='input', output_desc=[{}]),
            dict(type='NetOutput', name='out', input_desc=[{}])]}]}
        schema = SimpleNamespace(decode=lambda *args: doc, evidence={'kind': 'host fixture'})
        with ExitStack() as stack:
            for target, value in (
                ('toolchain.omc_container.read_container', {'partitions': [{'type': 0}]}),
                ('toolchain.omc_container.read_partition', b'host metadata'),
                ('toolchain.omc_wire.load_schema', schema),
                ('draft.ir_inspect._enum_fields', {}),
                ('draft.compile_precision._observed_desc', {'shape': [2], 'dtype': 'float16'}),
                ('draft.incremental.torch_export._compiled_observation', {'graphop_counts': {}}),
            ):
                stack.enter_context(patch('qwen35_moe_dflash.' + target, return_value=value))
            return prepare_fixture(compiled or self.output, output or self.root/'execution',
                                   self.ddk, 'test-ddk', **kwargs)

    def test_compile_keeps_source_and_failed_evidence_unchanged(self):
        before = {str(p.relative_to(self.air)): sha(p) for p in self.air.rglob('*') if p.is_file()}
        report, compiler = self.compile(timeout=123, use_origin_format=True, tbe_diagnosis='diagnosis')
        self.assertEqual(report['status'], 'COMPILED')
        self.assertEqual(compiler.call_count, 2)
        for call in compiler.call_args_list:
            self.assertEqual(call.kwargs['timeout_seconds'], 123)
            self.assertTrue(call.kwargs['use_origin_format'])
            self.assertEqual(call.kwargs['tbe_diagnosis'], 'diagnosis')
        after = {str(p.relative_to(self.air)): sha(p) for p in self.air.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual((self.output/report['source_batch']).resolve(), self.air)
        self.assertEqual(sha(self.output/'manifest.json'), sha(self.air/'manifest.json'))
        for name in self.names:
            self.assertTrue((self.output/report['cases'][name]['omc']['path']).is_file())
            self.assertFalse((self.output/'models'/name/'export.air').exists())
            self.assertFalse((self.output/'models'/name/'weight').exists())

    def test_existing_output_is_not_overwritten_and_fresh_retry_reuses_air(self):
        self.compile()
        digest = sha(self.output/'compile.json')
        with patch('qwen35_moe_dflash.toolchain.existing_air.compile_existing_air') as compiler:
            with self.assertRaises(FileExistsError):
                compile_batch(self.air, self.output, self.ddk, 'test-ddk')
            compiler.assert_not_called()
        self.assertEqual(sha(self.output/'compile.json'), digest)
        retried, _ = self.compile(self.root/'omc-batch-002')
        self.assertEqual(retried['status'], 'COMPILED')

    def test_overlapping_paths_fail_before_batch_writes_or_compiler_calls(self):
        alias = self.root/'air-alias'
        alias.symlink_to(self.air, target_is_directory=True)
        for output in (self.air, self.air/'omc', self.ddk, self.ddk/'out', self.root, alias/'out'):
            with self.subTest(output=output), patch('qwen35_moe_dflash.toolchain.existing_air.compile_existing_air') as compiler:
                with self.assertRaisesRegex(ValueError, 'fresh sibling directory'):
                    compile_batch(self.air, output, self.ddk, 'test-ddk')
                compiler.assert_not_called()
        self.assertFalse((self.air/'omc').exists())
        self.assertFalse((self.ddk/'out').exists())
        self.assertFalse((self.root/'compile.json').exists())

    def test_real_compiler_path_guard_is_passed_before_mock_environment(self):
        # Exercise the actual helper's AIR/output checks, stopping before any DDK use.
        with patch('qwen35_moe_dflash.toolchain.existing_air.omg_environment',
                   side_effect=RuntimeError('HOST_TEST_REACHED_OMG_ENVIRONMENT')) as environment:
            result = compile_batch(self.air, self.output, self.ddk, 'test-ddk')
        self.assertEqual(environment.call_count, 2)
        self.assertEqual(result['status'], 'PARTIAL')
        self.assertTrue(all(r['error'] == 'RuntimeError: HOST_TEST_REACHED_OMG_ENVIRONMENT'
                            for r in result['cases'].values()))

    def test_fixture_uses_source_golden_and_compiled_models_after_relocation(self):
        self.compile()
        relocated = self.root/'relocated'
        relocated.mkdir()
        shutil.move(self.air, relocated/'air-batch')
        shutil.move(self.output, relocated/'omc-batch-001')
        compiled = relocated/'omc-batch-001'
        out = self.root/'execution'
        meta = self.fixtures(compiled, out)
        self.assertEqual(len(meta['cases']), 2)
        self.assertEqual(meta['uncompiled_cases'], [])
        self.assertEqual(meta['reference']['reference_model_layers'], 4)
        for name in self.names:
            source = relocated/'air-batch/models'/name/'fixture'
            self.assertEqual((out/'fixture'/name/'input_0.bin').read_bytes(), (source/'input.bin').read_bytes())
            self.assertEqual((out/'fixture'/name/'golden_0.bin').read_bytes(), (source/'golden.bin').read_bytes())

    def test_source_manifest_changes_are_rejected_before_fixture_creation(self):
        self.compile()
        with (self.air/'manifest.json').open('a') as stream:
            stream.write('\n')
        with self.assertRaisesRegex(ValueError, 'AIR source manifest is missing or changed'):
            self.fixtures()
        self.assertFalse((self.root/'execution').exists())

    def test_source_payload_and_compiled_omc_hashes_remain_enforced(self):
        self.compile()
        for payload in (self.air/'models'/self.names[0]/'fixture/input.bin',
                        self.output/'models'/self.names[0]/'omc/model.omc'):
            original = payload.read_bytes()
            payload.write_bytes(original + b'changed')
            with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, 'fixture changed'):
                self.fixtures(output=self.root/f'execution-{payload.stem}')
            payload.write_bytes(original)

    def test_ddk_identity_mismatch_remains_rejected(self):
        self.compile()
        with self.assertRaisesRegex(ValueError, 'compile source/DDK identity changed'):
            prepare_fixture(self.output, self.root/'execution', self.ddk, 'different-ddk')
        self.assertFalse((self.root/'execution').exists())

    def test_partial_batch_prepares_only_compiled_cases(self):
        def compiler(air, output, ddk, **kwargs):
            if air.parent.name == self.names[1]:
                return dict(status='FAILED', error='host fixture compile failure')
            return self.fake_compiler(air, output, ddk, **kwargs)
        with patch('qwen35_moe_dflash.toolchain.existing_air.compile_existing_air', side_effect=compiler):
            report = compile_batch(self.air, self.output, self.ddk, 'test-ddk')
        self.assertEqual(report['status'], 'PARTIAL')
        meta = self.fixtures()
        self.assertEqual([c['case'] for c in meta['cases']], [self.names[0]])
        self.assertEqual(meta['uncompiled_cases'], [self.names[1]])
        with self.assertRaisesRegex(ValueError, 'select unique compiled regions'):
            self.fixtures(output=self.root/'invalid-selection', selection=[self.names[1]])

    def test_selected_cases_keep_full_manifest_and_do_not_compile_other_cases(self):
        report, compiler = self.compile(cases=[self.names[1]])
        self.assertEqual(compiler.call_count, 1)
        self.assertEqual(compiler.call_args.args[0].parent.name, self.names[1])
        self.assertEqual(report['status'], 'COMPILED')
        self.assertEqual(report['coverage'], 'SELECTED_CASES_ONLY')
        self.assertEqual(report['selected_cases'], [self.names[1]])
        self.assertEqual(report['cases'][self.names[0]]['status'], 'NOT_SELECTED')
        self.assertEqual(sha(self.output/'manifest.json'), sha(self.air/'manifest.json'))
        meta = self.fixtures()
        self.assertEqual(meta['uncompiled_cases'], [self.names[0]])

    def test_invalid_selection_or_profile_fails_before_writes(self):
        for kwargs in ({'cases': []}, {'cases': ['unknown']}, {'cases': self.names * 2},
                       {'compiler_profile': 'unknown'},
                       {'compiler_profile': 'sourced-wrapper-default', 'tbe_diagnosis': 'diagnosis'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.compile(**kwargs)
            self.assertFalse(self.output.exists())

    def test_diagnostic_profile_cannot_prepare_board_fixture(self):
        def compiler(air, output, ddk, **kwargs):
            self.assertTrue(kwargs['save_weights_as_external_data'])
            self.assertFalse(kwargs['limit_partition_size'])
            self.assertEqual(kwargs['compiler_profile'], 'sourced-wrapper-default')
            output.mkdir(parents=True)
            model = output/'model.omc'
            model.write_bytes(b'host diagnostic OMC fixture')
            return dict(status='COMPILED', omc=identity(model, output))
        with patch('qwen35_moe_dflash.target.layer_probe.batch.sourced_wrapper_environment'), \
             patch('qwen35_moe_dflash.toolchain.existing_air.compile_existing_air', side_effect=compiler):
            report = compile_batch(self.air, self.output, self.ddk, 'test-ddk',
                cases=[self.names[0]], compiler_profile='sourced-wrapper-default')
        self.assertEqual(report['platform'], 'UNSPECIFIED_COMPILER_DEFAULT')
        self.assertTrue(report['diagnostic_only'])
        with self.assertRaisesRegex(ValueError, 'unverified target'):
            self.fixtures()
        self.assertFalse((self.root/'execution').exists())


if __name__ == '__main__':
    unittest.main()
