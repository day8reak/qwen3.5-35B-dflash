"""Host-only compiler-entry contract tests, without executing any DDK tool."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from qwen35_moe_dflash.common.hashing import sha256_file
from qwen35_moe_dflash.toolchain.compiler_entry import (
    SCRIPT_HASHES, SOURCED_PROFILE, sourced_wrapper_environment)
from qwen35_moe_dflash.toolchain.existing_air import compile_existing_air


class CompilerEntryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ['AI_RUN_DIR'])
        self.root = Path(self.temp.name)
        self.ddk = self.root/'ddk'
        self.hashes = {}
        for relative in SCRIPT_HASHES:
            path = self.ddk/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('#!/bin/bash\nexit 97 # host fixture; must never be executed\n')
            self.hashes[relative] = sha256_file(path)
        self.omg = self.ddk/'tools/tools_omg/master/omg'
        self.omg.parent.mkdir(parents=True)
        self.omg.write_bytes(b'\x7fELFhost fixture, not executable machine code')
        self.omg.chmod(0o755)
        self.air = self.root/'source/export.air'
        self.air.parent.mkdir()
        self.air.write_bytes(b'unchanged source AIR fixture')
        self.inherited = dict(DDK_PATH=str(self.ddk), HIAI_VERSION=str(self.omg.parent),
            PATH='/host/bin', LD_LIBRARY_PATH='/inherited/lib',
            PYTHONPATH='/sourced/python:/other/python', SOC_VERSION='inherited-soc',
            ASCEND_CUSTOM_OPP_PATH='/sourced/custom', PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python')
        self.patches = [patch('qwen35_moe_dflash.toolchain.compiler_entry.SCRIPT_HASHES', self.hashes),
                        patch.dict(os.environ, self.inherited)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.temp.cleanup()

    def test_wrapper_replaces_libraries_and_preserves_sourced_registration(self):
        before = self.inherited.copy()
        omg, env, evidence = sourced_wrapper_environment(self.ddk, self.inherited)
        self.assertEqual(omg, self.omg)
        self.assertEqual(env['LD_LIBRARY_PATH'].split(':'), [str(self.ddk/p)+'/' for p in
            ('tools/tools_omg/master/lib64', 'ddk/tbe/lib64', 'ddk/ascendcpp/lib64')])
        self.assertEqual(env['HIAI_VERSION'], 'master')
        self.assertEqual(env['PYTHONPATH'], before['PYTHONPATH'])
        self.assertEqual(env['SOC_VERSION'], before['SOC_VERSION'])
        self.assertEqual(env['ASCEND_CUSTOM_OPP_PATH'], before['ASCEND_CUSTOM_OPP_PATH'])
        self.assertEqual(self.inherited, before)
        self.assertEqual(evidence['script_execution'], 'NOT_RUN')
        no_soc = {k: v for k, v in before.items() if k not in ('SOC_VERSION', 'HIAI_VERSION')}
        _, clean, _ = sourced_wrapper_environment(self.ddk, no_soc)
        self.assertNotIn('SOC_VERSION', clean)
        self.assertNotIn('HIAI_VERSION', clean)

    def test_changed_entry_unsourced_shell_and_shell_executable_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'sourced shell'):
            sourced_wrapper_environment(self.ddk, {})
        with self.assertRaisesRegex(ValueError, 'sourced shell'):
            sourced_wrapper_environment(self.ddk, {'DDK_PATH': str(self.root/'another-ddk')})
        script = self.ddk/'tools/tools_omg/omg'
        original = script.read_bytes()
        script.write_bytes(original+b'# changed\n')
        with self.assertRaisesRegex(ValueError, 'entry changed'):
            sourced_wrapper_environment(self.ddk, self.inherited)
        script.write_bytes(original)
        self.omg.write_bytes(b'#!/bin/bash\nexit 0\n')
        with self.assertRaisesRegex(ValueError, 'ELF'):
            sourced_wrapper_environment(self.ddk, self.inherited)

    def fake_run(self, argv, **kwargs):
        self.assertEqual(argv[0], str(self.omg))
        if argv[1:] == ['--help']:
            kwargs['stdout'].write('Usage --model --limit_partition_size --save_weights_as_external_data\n')
        else:
            air_arg = next(v for v in argv if v.startswith('--model='))
            self.assertEqual(Path(air_arg.split('=', 1)[1]).read_bytes(), self.air.read_bytes())
            self.assertFalse(any(v.startswith(('--target', '--platform', '--dynamic_dims', '--input_shape')) for v in argv))
            self.assertIn('--limit_partition_size=false', argv)
            self.assertIn('--save_weights_as_external_data=true', argv)
            self.assertEqual(kwargs['env']['SOC_VERSION'], 'inherited-soc')
            self.assertTrue(kwargs['env']['PYTHONPATH'].endswith(self.inherited['PYTHONPATH']))
            target = Path(kwargs['cwd'])/'model'
            target.mkdir()
            (target/'model.omc').write_bytes(b'host OMC fixture')
            (target/'SubGraph_0.weight').write_bytes(b'host external weight fixture')
        return SimpleNamespace(returncode=0)

    def compile(self, output, **kwargs):
        return compile_existing_air(self.air, output, self.ddk, ddk_id='host-fixture',
            compiler_profile=SOURCED_PROFILE, limit_partition_size=False,
            save_weights_as_external_data=True, **kwargs)

    def test_same_air_environment_receipt_external_outputs_and_no_shared_writes(self):
        before = {str(p): (p.stat().st_mode, sha256_file(p))
                  for p in self.ddk.rglob('*') if p.is_file()}
        output = self.root/'compiled'
        with patch('qwen35_moe_dflash.toolchain.existing_air.subprocess.run', side_effect=self.fake_run) as run:
            report = self.compile(output)
        self.assertEqual(report['status'], 'COMPILED', report.get('error'))
        self.assertEqual(run.call_count, 2)  # master --help + master compile, never shell scripts
        self.assertEqual(report['platform'], 'UNSPECIFIED_COMPILER_DEFAULT')
        self.assertEqual(report['device_execution'], 'NOT_RUN')
        self.assertFalse(report['device_ready'])
        self.assertEqual(report['omc']['path'], 'compile-work/generated/model/model.omc')
        self.assertTrue(any(p['path'].endswith('SubGraph_0.weight') for p in report['external_data_candidates']))
        saved = json.loads((output/'omg-environment.json').read_text())['environment']
        self.assertEqual(saved['ASCEND_CUSTOM_OPP_PATH'], '/sourced/custom')
        self.assertEqual(saved['HIAI_VERSION'], 'master')
        self.assertIn('UNSPECIFIED_COMPILER_DEFAULT', (output/'report.txt').read_text())
        self.assertEqual(before, {str(p): (p.stat().st_mode, sha256_file(p))
                                for p in self.ddk.rglob('*') if p.is_file()})
        self.assertEqual(self.air.read_bytes(), b'unchanged source AIR fixture')

    def test_compiler_crash_keeps_returncode_and_does_not_promote(self):
        def failing(argv, **kwargs):
            if argv[1:] == ['--help']:
                return self.fake_run(argv, **kwargs)
            raise subprocess.CalledProcessError(-11, argv)
        with patch('qwen35_moe_dflash.toolchain.existing_air.subprocess.run', side_effect=failing):
            report = self.compile(self.root/'failed')
        self.assertEqual(report['status'], 'FAILED')
        self.assertEqual(report['returncode'], -11)
        self.assertEqual(report['stage'], 'omg')

    def test_default_entry_keeps_explicit_target_without_reading_wrapper(self):
        (self.ddk/'tools/tools_omg/omg').unlink()
        def default_run(argv, **kwargs):
            if argv[1:] == ['--help']:
                return self.fake_run(argv, **kwargs)
            self.assertIn('--target=omc', argv)
            self.assertIn('--platform=kirinx90', argv)
            self.assertIn('--save_weights_as_external_data=false', argv)
            self.assertFalse(any(v.startswith('--limit_partition_size') for v in argv))
            (Path(kwargs['cwd'])/'model.omc').write_bytes(b'host default OMC fixture')
            return SimpleNamespace(returncode=0)
        with patch('qwen35_moe_dflash.toolchain.existing_air.omg_environment',
                   return_value=(self.omg, self.inherited)), \
             patch('qwen35_moe_dflash.toolchain.existing_air.subprocess.run', side_effect=default_run):
            report = compile_existing_air(self.air, self.root/'default', self.ddk,
                ddk_id='host-fixture', save_weights_as_external_data=False)
        self.assertEqual(report['status'], 'COMPILED', report.get('error'))
        self.assertEqual(report['platform'], 'kirinx90')
        self.assertEqual(report['compiler_profile'], 'isolated-kirinx90')
        self.assertNotIn('compiler_entry', report)

    def test_incompatible_profile_options_fail_before_output_creation(self):
        for kwargs in ({'tbe_diagnosis': 'diagnosis'}, {'omg_path': self.omg},
                       {'air_transform': lambda p: (b'changed', {})}):
            output = self.root/'rejected'
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.compile(output, **kwargs)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
