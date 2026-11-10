"""Host preparation/capability gates and the actual C++ plan parser; no device."""
import copy
import importlib.util
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from qwen35_moe_dflash.target.layer_probe.batch import check_io_budget

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT/'targets/cpro/tools'
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location('probe_budget_board', TOOLS/'validate_draft_vector.py')
board = importlib.util.module_from_spec(spec)
spec.loader.exec_module(board)


def conv_case():
    def tensor(shape):
        return dict(shape=shape, dtype='float32', bytes=math.prod(shape)*4)
    return dict(case='l00_c00_conv1d_o0',
                inputs=[tensor([1, 8192, 259])], outputs=[tensor([1, 8192, 256])])


def capabilities(limit=None):
    result = dict(tensor_probe_dtypes=['float32'], tensor_probe_input_reset=['copy-each-call'])
    if limit is not None:
        result['tensor_probe_limits'] = dict(max_io_bytes=limit)
    return result


class BudgetTest(unittest.TestCase):
    def test_fp32_prefill_conv_exceeds_old_limit_and_fits_new_limit(self):
        case = conv_case()
        total = sum(s['bytes'] for d in ('inputs', 'outputs') for s in case[d])
        self.assertEqual(total, (16 << 20) + 96*1024)
        check_io_budget(case['case'], case)
        meta = dict(cases=[case], input_reset='copy-each-call')
        for old in (None, 16 << 20):
            with self.subTest(old=old), self.assertRaisesRegex(
                    ValueError, 'l00_c00_conv1d_o0.*16875520.*16777216.*Rebuild'):
                board._check_probe_dtypes(meta, capabilities(old))
        board._check_probe_dtypes(meta, capabilities(32 << 20))

    def test_budget_is_per_case_and_legacy_runner_still_accepts_small_cases(self):
        case = conv_case()
        for d in ('inputs', 'outputs'):
            case[d] = [dict(shape=[1024, 1024], dtype='float32', bytes=4 << 20)]
        board._check_probe_dtypes(dict(cases=[copy.deepcopy(case) for _ in range(26)]), capabilities())

    def test_total_bound_includes_outputs_but_not_golden_storage_width(self):
        case = dict(case='limit', inputs=[dict(bytes=16 << 20)],
                    outputs=[dict(bytes=16 << 20, fixture_bytes=32 << 20)])
        check_io_budget(case['case'], case)
        case['outputs'][0]['bytes'] += 4
        with self.assertRaisesRegex(ValueError, 'limit:.*33554436.*33554432'):
            check_io_budget(case['case'], case)
        case['outputs'] = [dict(bytes=2)]*9
        with self.assertRaisesRegex(ValueError, 'at most 8 slots'):
            check_io_budget(case['case'], case)


@unittest.skipUnless(shutil.which('g++'), 'requires host g++ for production C++ parser')
class NativePlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(dir=os.environ['AI_RUN_DIR'])
        cls.root = Path(cls.temp.name)
        # Compile the complete production runner, keeping only ReadPlan and its
        # dependencies at link time. Use the production byte-count function too;
        # HIAI allocation/execution is deliberately not linked or invoked.
        executor = (ROOT/'targets/cpro/runtime/src/hiai_executor.cpp').read_text()
        byte_function = executor.split('std::size_t HiaiTensorBytes(', 1)[1].split(
            '\nstruct HiaiStatefulExecutor::Impl', 1)[0]
        harness = cls.root/'plan_test.cpp'
        harness.write_text(
            '#define main unused_probe_main\n'
            f'#include "{TOOLS / "tensor_probe_main.cpp"}"\n'
            '#undef main\nnamespace qwen35::dflash {\n'
            'constexpr std::size_t kStatefulTensorLimit = 512ULL << 20;\n'
            'std::size_t HiaiTensorBytes(' + byte_function + '\n}\n'
            'int main(int argc, char** argv) {\n'
            '  try { if (argc != 2) return 2; ReadPlan(argv[1]); return 0; }\n'
            '  catch (const std::exception& e) { std::cerr << e.what(); return 1; }\n}\n')
        cls.binary = cls.root/'plan_test'
        subprocess.run(['g++', '-std=c++17', '-ffunction-sections', '-fdata-sections',
                        '-I', str(ROOT/'targets/cpro/runtime/include'), str(harness),
                        '-Wl,--gc-sections', '-o', str(cls.binary)], check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def plan(self, inputs, outputs):
        path = self.root/'plan.txt'
        path.write_text('QWEN35_TENSOR_PROBE 1\niterations 3 10\ninput_reset copy-each-call\n'
                        f'input 0 x float32 {inputs}\noutput 0 y float32 {outputs}\n')
        return subprocess.run([str(self.binary), str(path)], capture_output=True, text=True)

    def test_actual_fp32_prefill_conv_plan(self):
        result = self.plan('1,8192,259', '1,8192,256')
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_exact_cap_and_over_cap(self):
        result = self.plan('4194304', '4194304')  # 16 MiB + 16 MiB
        self.assertEqual(result.returncode, 0, result.stderr)
        for x, y in [('4194304', '4194305'), ('8388609', '1')]:
            with self.subTest(x=x, y=y):
                result = self.plan(x, y)
                self.assertEqual(result.returncode, 1)
                self.assertIn('probe IO exceeds 32 MiB', result.stderr)


if __name__ == '__main__':
    unittest.main()
