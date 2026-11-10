"""Exercise Target partial-batch recovery with synthetic archives and mocked HDC."""
import argparse
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import Mock, patch

from qwen35_moe_dflash.target.layer_probe.common import KIND, save_json, sha

TOOLS = Path(__file__).resolve().parents[1]/'targets/cpro/tools'
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location('target_probe_board_test', TOOLS/'validate_draft_vector.py')
board = importlib.util.module_from_spec(spec)
spec.loader.exec_module(board)


class RecoveryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ['AI_RUN_DIR'])
        self.root = Path(self.temp.name)
        self.execution = self.root/'execution'
        self.name = 'l00_c00_conv_silu_o0'
        self.missing = 'l00_c00_gdn_o0'
        self.token = 'a'*32
        fixture = self.execution/'fixture'/self.name
        fixture.mkdir(parents=True)
        value = struct.pack('<2e', 1, 2)
        (fixture/'golden_0.bin').write_bytes(value)
        output = dict(index=0, name='golden_0', shape=[2], dtype='float16', golden_dtype='float16',
            bytes=len(value), fixture_bytes=len(value), fixture_sha256=sha(fixture/'golden_0.bin'))
        self.meta = dict(artifact_kind=KIND, warmup=3, repetitions=10, uncompiled_cases=[self.missing],
            cases=[dict(case=self.name, layer=0, call=0, phase='prefill', point='conv_silu', kind='silu',
                ordinal=0, output_name='y', inputs=[], outputs=[output], source_dtype_preserved=True,
                graphop_counts={})])
        save_json(fixture.parent/'manifest.json', self.meta)
        save_json(self.execution/'request.json', dict(artifact_kind=board.KIND, token=self.token,
            board_attempt=board.PREFIX+self.token, serial='host-test', hdc_server='host-server',
            transport='archive-v1', cases=[self.name], fixture_sha256=sha(fixture.parent/'manifest.json')))
        self.remote = self.root/'mock-remote'
        self.remote.mkdir()
        self.archive(self.remote/'logs.tar', {f'{self.name}.exitcode': b'0',
            'native.exitcode': b'0', 'archive.exitcode': b'0'})
        # A synthetic receipt exercises the production recovery/comparison path;
        # it is not hardware evidence and the HDC transport is always mocked.
        receipt = dict(status='EXECUTED', evidence_kind='device-hiai', warmup=3, repetitions=10,
            input_reset='copy-each-call', api_us=[1]*10)
        entries = {f'{self.name}/result.json': json.dumps(receipt).encode()}
        entries.update({f'{self.name}/{i}.bin': value for i in range(10)})
        self.archive(self.remote/'outputs.tar', entries)

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def archive(path, entries):
        with tarfile.open(path, 'w') as stream:
            for name, value in entries.items():
                item = tarfile.TarInfo(name)
                item.size = len(value)
                stream.addfile(item, io.BytesIO(value))

    def recover(self):
        def shell(command, stage, **kwargs):
            values = {'vector-completion': self.token,
                      'vector-log-size': '0\n'+str((self.remote/'logs.tar').stat().st_size),
                      'vector-output-size': str((self.remote/'outputs.tar').stat().st_size),
                      'vector-lock-owner': 'ABSENT'}
            return 0, values[stage]
        def receive(hdc, remote, out, name, **kwargs):
            return shutil.copyfile(self.remote/name, out/name)
        hdc = Mock()
        hdc.shell.side_effect = shell
        with patch.object(board, 'Hdc', return_value=hdc) as transport, \
             patch.object(board, '_environment', return_value=('mock-hdc', 'host-test', 'host-server')), \
             patch.object(board, '_receive', side_effect=receive), \
             patch.object(board, 'local_status', return_value={}):
            result = board.recover(self.execution, self.root/'recovery')
        return result, transport

    def test_partial_target_batch_recovers_and_compares_completed_cases(self):
        report, transport = self.recover()
        transport.assert_called_once()
        self.assertEqual(report['device_execution'], 'EXECUTED_OUTPUTS_RETRIEVED')
        self.assertEqual(report['status'], 'PARTIAL')
        self.assertEqual(report['numerical_validation'], 'METRICS_RECORDED')
        self.assertEqual(report['validated_cases'], [self.name])
        self.assertEqual(report['uncompiled_cases'], [self.missing])
        self.assertFalse(report['coverage_complete'])
        self.assertEqual(report['cases'][0]['invocations'][0]['max_abs'], 0)
        self.assertTrue((self.root/'recovery/native'/self.name/'9.bin').is_file())

    def test_unbound_requested_case_is_rejected_before_transport(self):
        path = self.execution/'request.json'
        plan = json.loads(path.read_text())
        plan['cases'] = [self.missing]
        save_json(path, plan)
        report, transport = self.recover()
        self.assertEqual(report['status'], 'FAILED')
        transport.assert_not_called()

    def test_changed_fixture_is_rejected_before_transport(self):
        with (self.execution/'fixture/manifest.json').open('a') as stream:
            stream.write('\n')
        report, transport = self.recover()
        self.assertIn('retained fixture manifest changed', report['error'])
        transport.assert_not_called()

    def test_unsafe_and_duplicate_target_names_are_rejected(self):
        for names in ([self.name, self.name], ['../'+self.name], [self.name+';id']):
            with self.subTest(names=names), self.assertRaises(ValueError):
                board.fixture_cases(dict(artifact_kind=KIND, cases=[dict(case=n) for n in names]))
        with self.assertRaises(ValueError):
            board.fixture_cases(dict(artifact_kind=board.KIND, cases=[dict(case=self.name)]))
        self.assertEqual(board.selected_cases([board.CASES[0]]), [board.CASES[0]])

    def test_archive_cannot_inject_an_unrequested_target_case(self):
        path = self.root/'unexpected.tar'
        self.archive(path, {f'{self.missing}/0.bin': b'1234'})
        with self.assertRaisesRegex(ValueError, 'unexpected probe members'):
            board.extract(path, self.root/'extracted', self.meta)
        self.assertFalse((self.root/'extracted').exists())

    def test_cli_accepts_target_names_but_rejects_path_or_shell_tokens(self):
        self.assertEqual(board.case_argument(self.name), self.name)
        for name in ('../case', 'not_a_probe', self.name+';id'):
            with self.subTest(name=name), self.assertRaises(argparse.ArgumentTypeError):
                board.case_argument(name)
        args = ['validate_draft_vector.py', '--compiled', str(self.root/'compiled'),
                '--config', str(self.root/'runtime.json'), '--ddk-root', str(self.root/'ddk'),
                '--output', str(self.root/'run'), '--cases', self.name]
        with patch.object(sys, 'argv', args), patch.object(board, 'start', return_value={'status':'PREPARED'}) as start:
            self.assertEqual(board.main(), 0)
        self.assertEqual(start.call_args.args[0].cases, [self.name])


if __name__ == '__main__':
    unittest.main()
