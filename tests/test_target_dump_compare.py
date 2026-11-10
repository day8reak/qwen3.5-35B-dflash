import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from qwen35_moe_dflash.target.dump_compare import prepare, compare, tensor_metrics
from qwen35_moe_dflash.target.layer_probe.common import KIND, identity, save_json


class DumpComparisonTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ['AI_RUN_DIR'])
        self.root = Path(self.temp.name)
        self.golden, self.actual = self.root/'golden', self.root/'actual'
        self.golden.mkdir()
        self.actual.mkdir()
        x = self.tensor(self.golden, 'input', [1., 2.])
        y = self.tensor(self.golden, 'golden_y', [3., 4.])
        for spec in (x, y):
            spec.pop('format')
            spec.pop('layout')
        self.capture = dict(artifact_kind=KIND, status='CAPTURED', regions=[
            dict(id='l00_c00_add', layer=0, call=0, phase='prefill', point='add',
                 ordinal=0, inputs=[x], outputs=[y])])
        save_json(self.golden/'manifest.json', self.capture)
        self.plan_path = Path(prepare(self.golden, self.root/'plan')['mapping'])
        self.plan = json.loads(self.plan_path.read_text())

    def tearDown(self):
        self.temp.cleanup()

    def tensor(self, root, name, values, dtype='float32', fmt='raw'):
        a = np.asarray(values, dtype=dtype)
        path = root/(name + ('.npy' if fmt == 'npy' else '.bin'))
        if fmt == 'npy':
            np.save(path, a, allow_pickle=False)
        else:
            a.tofile(path)
        return dict(identity(path, root), name=name, dtype=dtype,
                    shape=list(a.shape), format=fmt, layout='logical')

    def mapped(self, actual_x=(1., 2.), actual_y=(3., 4.)):
        row = self.plan['regions'][0]
        self.plan.update(contract_confirmed=True, contract_evidence='Host synthetic same-weight/state fixture')
        row.update(inputs_complete=True, mapping_evidence='Synthetic region has one input; no state/control')
        row['inputs'][0]['actual'] = self.tensor(self.actual, 'x', actual_x)
        row['outputs'][0]['actual'] = self.tensor(self.actual, 'y', actual_y)
        for direction in ('inputs', 'outputs'):
            row[direction][0]['golden']['layout'] = 'logical'
            row[direction][0]['gate'] = dict(mode='allclose', atol=0., rtol=0.)

    def run_compare(self, name='comparison'):
        save_json(self.plan_path, self.plan)
        summary = compare(self.plan_path, self.golden, self.actual, self.root/name)
        report = json.loads(Path(summary['report']).read_text())
        self.assertTrue(Path(summary['csv']).is_file())
        return report

    def test_template_does_not_guess_mapping_or_claim_pass(self):
        report = self.run_compare()
        self.assertEqual(report['status'], 'PARTIAL')
        self.assertEqual(report['tensor_status_counts'], {'UNMAPPED': 2})
        self.assertEqual(report['local_candidates'], [])

    def test_generated_mapping_cannot_modify_capture_tree(self):
        with self.assertRaisesRegex(ValueError, 'outside the capture'):
            prepare(self.golden, self.golden/'new_mapping')
        self.assertFalse((self.golden/'new_mapping').exists())

    def test_mapped_boundaries_pass_without_claiming_full_model(self):
        self.mapped()
        report = self.run_compare()
        self.assertEqual(report['status'], 'MAPPED_BOUNDARIES_PASSED')
        self.assertIn('no model execution', report['validation_scope'])

    def test_capture_shape_alone_does_not_prove_logical_layout(self):
        self.mapped()
        self.plan['regions'][0]['inputs'][0]['golden']['layout'] = 'UNVERIFIED'
        report = self.run_compare()
        self.assertEqual(report['regions'][0]['inputs'][0]['status'], 'INVALID_TENSOR')
        self.assertEqual(report['local_candidates'], [])

    def test_output_failure_distinguishes_local_candidate_from_input_propagation(self):
        self.mapped(actual_y=(3., 8.))
        report = self.run_compare()
        self.assertEqual(report['local_candidates'], ['l00_c00_add'])
        self.assertEqual(report['first_observed_output_failure'], 'l00_c00_add')
        self.plan['regions'][0]['inputs'][0]['actual'] = self.tensor(self.actual, 'x2', [1., 5.])
        report = self.run_compare('comparison2')
        self.assertEqual(report['local_candidates'], [])
        self.assertEqual(report['regions'][0]['diagnosis'], 'OUTPUT_FAILURE_WITH_INPUT_DIFFERENCES')

    def test_unknown_control_coverage_or_contract_blocks_local_claim(self):
        self.mapped(actual_y=(4., 8.))
        self.plan['regions'][0]['inputs_complete'] = False
        self.assertEqual(self.run_compare()['local_candidates'], [])
        self.plan['regions'][0]['inputs_complete'] = True
        self.plan['contract_confirmed'] = False
        self.assertEqual(self.run_compare('comparison2')['local_candidates'], [])

    def test_allclose_inputs_are_not_exact_inputs(self):
        self.mapped(actual_x=(1.001, 2.), actual_y=(4., 8.))
        self.plan['regions'][0]['inputs'][0]['gate']['atol'] = .1
        r = self.run_compare()
        self.assertEqual(r['regions'][0]['inputs'][0]['status'], 'PASSED')
        self.assertEqual(r['local_candidates'], [])

    def test_signed_zero_input_difference_blocks_local_claim(self):
        self.mapped(actual_y=(3., 8.))
        spec = self.tensor(self.golden, 'input', [0., 2.])
        for key in ('format', 'layout'):
            spec.pop(key)
        self.capture['regions'][0]['inputs'] = [spec]
        save_json(self.golden/'manifest.json', self.capture)
        from qwen35_moe_dflash.target.layer_probe.common import sha
        self.plan['capture_manifest_sha256'] = sha(self.golden/'manifest.json')
        pair = self.plan['regions'][0]['inputs'][0]
        pair['golden'] = dict(spec, format='raw', layout='logical')
        pair['actual'] = self.tensor(self.actual, 'signed_zero', [-0., 2.])
        result = self.run_compare()
        self.assertTrue(result['regions'][0]['inputs'][0]['exact_values_equal'])
        self.assertFalse(result['regions'][0]['inputs'][0]['bitwise_equal'])
        self.assertEqual(result['local_candidates'], [])

    def test_integer_above_float64_precision_uses_exact_values_despite_tolerance(self):
        a = np.array([2**60+1], dtype=np.int64)
        g = np.array([2**60], dtype=np.int64)
        r = tensor_metrics(a, g, dict(mode='allclose', atol=1e20, rtol=1e20))
        self.assertEqual(r['status'], 'FAILED')
        self.assertEqual(r['exact_mismatch_count'], 1)
        self.assertEqual(r['gate'], 'exact_integer')

    def test_nan_fails_even_without_gate_and_zero_reference_is_not_divided(self):
        r = tensor_metrics(np.array([np.nan]), np.array([np.nan]), {'mode': 'metrics'})
        self.assertEqual(r['status'], 'FAILED')
        r = tensor_metrics(np.ones(2), np.zeros(2), {'mode': 'metrics'})
        self.assertIsNone(r['relative_l2'])
        self.assertEqual(r['rmse'], 1.)
        self.assertEqual(r['status'], 'METRICS_RECORDED')

    def test_bitwise_distinguishes_signed_zero(self):
        r = tensor_metrics(np.array([-0.], dtype=np.float32), np.array([0.], dtype=np.float32), {'mode': 'bitwise'})
        self.assertEqual(r['status'], 'FAILED')
        self.assertEqual(r['exact_mismatch_count'], 0)
        self.assertEqual(r['gate_mismatch_count'], 1)

    def test_metrics_across_chunks_have_logical_coordinates(self):
        a = np.zeros((2, 3), dtype=np.float32)
        a[1, 2] = 3.
        r = tensor_metrics(a, np.zeros_like(a), dict(mode='allclose', atol=0, rtol=0), chunk=2)
        self.assertEqual(r['max_error_coordinate'], [1, 2])
        self.assertEqual(r['first_mismatch_coordinates'], [[1, 2]])
        self.assertAlmostEqual(r['rmse'], np.sqrt(9/6))
        self.assertEqual(r['max_abs'], 3.)

    def test_missing_tensor_does_not_hide_other_output_failure(self):
        self.mapped(actual_y=(3., 8.))
        (self.actual/'x.bin').unlink()
        r = self.run_compare()
        self.assertEqual(r['status'], 'PARTIAL')
        self.assertEqual(r['tensor_status_counts'], {'MISSING_FILE': 1, 'FAILED': 1})
        self.assertEqual(r['local_candidates'], [])

    def test_stale_hash_and_path_escape_do_not_get_compared(self):
        self.mapped()
        np.array([9., 8.], dtype=np.float32).tofile(self.actual/'y.bin')
        r = self.run_compare()
        self.assertEqual(r['regions'][0]['outputs'][0]['status'], 'INVALID_TENSOR')
        self.plan['regions'][0]['inputs'][0]['actual']['path'] = '../golden/input.bin'
        r = self.run_compare('comparison2')
        self.assertEqual(r['regions'][0]['inputs'][0]['status'], 'INVALID_TENSOR')

    def test_golden_mutation_or_port_omission_is_rejected(self):
        self.mapped()
        self.plan['regions'][0]['inputs'][0]['golden']['shape'] = [1, 2]
        with self.assertRaisesRegex(ValueError, 'golden descriptor'):
            self.run_compare()
        self.plan['regions'][0]['inputs'][0]['golden']['shape'] = [2]
        self.capture['status'] = 'changed'
        save_json(self.golden/'manifest.json', self.capture)
        with self.assertRaisesRegex(ValueError, 'capture manifest'):
            self.run_compare()

    def test_npy_explicit_view_and_unsupported_nz(self):
        self.mapped()
        spec = self.tensor(self.actual, 'padded', [[3., 99.], [4., 99.]], fmt='npy')
        spec.update(view={'permute': [1, 0], 'slice': [[0, 1], [0, 2]]},
                    view_evidence='Synthetic tensor: row zero is useful after transpose')
        # A view must match logical rank exactly; no silent squeeze/reshape.
        self.plan['regions'][0]['outputs'][0]['actual'] = spec
        self.assertEqual(self.run_compare()['regions'][0]['outputs'][0]['status'], 'INVALID_TENSOR')
        spec.update(layout='FRACTAL_NZ')
        self.assertEqual(self.run_compare('comparison2')['regions'][0]['outputs'][0]['status'], 'INVALID_TENSOR')

    def test_explicit_slice_supports_npy_without_copying_or_guessing(self):
        self.mapped()
        spec = self.tensor(self.actual, 'padded', [3., 4., 99.], fmt='npy')
        spec.update(view={'slice': [[0, 2]]}, view_evidence='Synthetic trailing padding')
        self.plan['regions'][0]['outputs'][0]['actual'] = spec
        self.assertEqual(self.run_compare()['status'], 'MAPPED_BOUNDARIES_PASSED')

    def test_empty_tensor_and_unset_threshold_never_pass(self):
        r = tensor_metrics(np.empty(0), np.empty(0), dict(mode='allclose', atol=0, rtol=0))
        self.assertEqual(r['status'], 'EMPTY_TENSOR')
        self.mapped()
        for d in ('inputs', 'outputs'):
            self.plan['regions'][0][d][0]['gate'] = {'mode': 'metrics'}
        self.assertEqual(self.run_compare()['status'], 'METRICS_RECORDED')

    def test_overflow_and_invalid_tolerance_are_not_success(self):
        with self.assertRaises(ValueError):
            tensor_metrics(np.array([1e308]), np.array([-1e308]), dict(mode='allclose', atol=0., rtol=0.))
        with self.assertRaises(ValueError):
            tensor_metrics(np.ones(1), np.ones(1), dict(mode='allclose', atol=float('nan'), rtol=0.))


if __name__ == '__main__':
    unittest.main()
