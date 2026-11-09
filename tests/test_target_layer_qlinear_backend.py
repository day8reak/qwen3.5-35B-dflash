import functools
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from qwen35_moe_dflash.target.layer_probe import qlinear_backend as backend


class BackendTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ['AI_RUN_DIR'])
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def test_mapped_custom_and_lazy_api_libraries_are_not_dropped(self):
        paths = ['/vendor/custom ops/libcust_opapi.so', '/cann/libopapi_math.so',
                 '/cann/libaclnn_math.so', '/cann/libascendcl.so', '/torch/libtorch_npu.so']
        maps = '\n'.join('100-200 r-xp 000 00:00 1 ' + p for p in paths)
        maps += '\n100-200 r-xp 000 00:00 1 /other/libc.so\n100-200 rw-p 000 00:00 0 [heap]'
        maps += '\n100-200 r-xp 000 00:00 1 /vendor/custom ops/libcust_opapi.so (deleted)'
        self.assertEqual(backend.loaded_libraries(maps), sorted(paths))

    def test_wrapped_schema_is_recovered(self):
        def op():
            pass
        op._schemas = {'': 'saved schema'}
        @functools.wraps(op, updated=())
        def wrapped():
            pass
        with patch.object(Path, 'read_text', side_effect=OSError('maps unavailable')):
            record = backend.runtime(SimpleNamespace(npu_quant_matmul=wrapped))
        self.assertIn('saved schema', record['quant_matmul_schema'])
        self.assertEqual(record['loaded_libraries'], [])
        self.assertIn('maps unavailable', record['maps_error'])

    def test_undefined_hidden_and_local_symbols_are_not_exports(self):
        symbols = '''
  1: 00000000 0 FUNC GLOBAL DEFAULT UND aclnnQuantMatmulV5
  2: 00000001 8 FUNC GLOBAL DEFAULT 12 aclnnQuantMatmulV4
  3: 00000002 8 FUNC WEAK DEFAULT 12 aclnnQuantMatmulV5GetWorkspaceSize@@V1
  4: 00000003 8 FUNC GLOBAL HIDDEN 12 aclnnQuantMatmulV5
  5: 00000004 8 FUNC LOCAL DEFAULT 12 aclnnQuantMatmulV5
  6: 00000005 8 FUNC GLOBAL PROTECTED 12 aclnnTransData
  7: 00000006 8 FUNC GLOBAL DEFAULT 12 unrelated
'''
        self.assertEqual(backend.exported_symbols(symbols),
            ['aclnnQuantMatmulV4', 'aclnnQuantMatmulV5GetWorkspaceSize', 'aclnnTransData'])

    def test_inventory_reads_only_declared_candidates_and_mapped_api_files(self):
        vendor = self.root/'vendor with spaces'
        library = vendor/'op_api/lib/libcust_opapi.so'
        library.parent.mkdir(parents=True)
        library.write_bytes(b'mock elf')
        missing = self.root/'absent'
        record = dict(environment={'ASCEND_CUSTOM_OPP_PATH': f'{vendor}:{missing}:',
            'ASCEND_OPP_PATH': None}, loaded_libraries=[str(library)])
        symbols = '1: 01 8 FUNC GLOBAL DEFAULT 12 aclnnQuantMatmulV4\n'
        with patch.object(backend.shutil, 'which', return_value='/usr/bin/readelf'), \
             patch.object(backend.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout=symbols)) as run:
            result = backend.inventory(record)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0], ['/usr/bin/readelf', '--dyn-syms', '--wide', str(library)])
        self.assertEqual(len(result['libraries']), 2)
        row = result['libraries'][0]
        self.assertTrue(row['mapped'])
        self.assertFalse(row['defines_v5_pair'])
        self.assertEqual(row['owner_uid'], library.stat().st_uid)
        self.assertEqual(row['origins'], ['ASCEND_CUSTOM_OPP_PATH[0]', 'PROCESS_MAPS'])
        self.assertIn('NOT_DISPATCH_BINDING_PROOF', result['scope'])
        self.assertEqual(result['libraries'][1]['symbol_status'], 'UNAVAILABLE')

    def test_diagnostic_timeout_does_not_claim_missing_v5(self):
        library = self.root/'libcust_opapi.so'
        library.write_bytes(b'mock elf')
        with patch.object(backend.subprocess, 'run', side_effect=subprocess.TimeoutExpired('readelf', 10)):
            result = backend.library_record(library, set(), '/usr/bin/readelf')
        self.assertEqual(result['symbol_status'], 'UNAVAILABLE')
        self.assertNotIn('defines_v5_pair', result)

    def test_format_read_never_casts_or_calls_getter_for_cpu(self):
        def getter(value):
            raise AssertionError('should not initialize NPU for a CPU fixture')
        result = backend.tensor_formats(SimpleNamespace(get_npu_format=getter), {'x1': torch.ones(1)})
        self.assertEqual(result['x1']['status'], 'NOT_NPU')


if __name__ == '__main__':
    unittest.main()
