import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from qwen35_moe_dflash.target.layer_probe.capture import Capture
from qwen35_moe_dflash.target.layer_probe.common import checked
from qwen35_moe_dflash.target.layer_probe.qlinear import QuantMatmulAudit, contract, explicit_group256, replay_failure


def inputs():
    x = torch.ones(3, 256, dtype=torch.int8)
    w = torch.arange(512, dtype=torch.int32).reshape(256, 2)
    scale = torch.full((1, 16), 0.125, dtype=torch.float32)
    return (x, w, scale), dict(pertoken_scale=torch.ones(3, 1), output_dtype=torch.float16)


class QLinearTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ['AI_RUN_DIR'])
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def test_group256_changes_only_keyword_and_preserves_input_objects(self):
        args, kwargs = inputs()
        actual = explicit_group256(args, kwargs)
        self.assertEqual(actual['group_sizes'], [0, 0, 256])
        self.assertNotIn('group_sizes', kwargs)
        for k, v in kwargs.items():
            self.assertIs(actual[k], v)
        self.assertEqual(contract(args, kwargs)['inferred_weight_group_size'], 256)

    def test_wrong_scale_axis_or_group128_is_never_reshaped(self):
        args, kwargs = inputs()
        for scale in (torch.ones(2, 16), torch.ones(16, 1)):
            with self.assertRaisesRegex(ValueError, 'do not describe'):
                explicit_group256((*args[:2], scale), kwargs)
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            explicit_group256(args, dict(kwargs, group_sizes=[0, 0, 128]))

    def test_source_call_unchanged_by_default_and_patch_restored(self):
        self.exercise(None, False)

    def test_explicit_group256_marked_unverified_not_original_golden(self):
        self.exercise(256, False)

    def exercise(self, group, fail):
        args, kwargs = inputs()
        seen = []
        def op(*a, **k):
            seen.append((a, k))
            if fail:
                raise RuntimeError('groupSize equal to 256')
            return torch.zeros(3, 16, dtype=torch.float16)
        npu = SimpleNamespace(npu_quant_matmul=op)
        class QLinear(torch.nn.Module):
            def forward(self, x):
                return npu.npu_quant_matmul(x, args[1], args[2], **kwargs)
        wrapper = torch.nn.Module()
        wrapper.lm_head = QLinear()
        capture = Capture(self.root/'capture', layers=[0], calls=2)
        audit = QuantMatmulAudit(capture, npu, group)
        audit.attach(wrapper)
        original = QLinear.forward
        audit.install(QLinear)
        try:
            if fail:
                with self.assertRaisesRegex(RuntimeError, 'groupSize'):
                    wrapper.lm_head(args[0])
            else:
                wrapper.lm_head(args[0])
        finally:
            capture.close('expected toy failure' if fail else None)
        self.assertIs(npu.npu_quant_matmul, op)
        self.assertIs(QLinear.forward, original)
        self.assertEqual(len(seen), 1)  # No automatic failed-call retry.
        for a, b in zip(seen[0][0], args):
            self.assertIs(a, b)
        if group:
            self.assertEqual(seen[0][1]['group_sizes'], [0, 0, 256])
            self.assertEqual(audit.record['baseline_equivalence'], 'NOT_VERIFIED')
            self.assertIn('UNVERIFIED', capture.report['reference'])
        else:
            self.assertNotIn('group_sizes', seen[0][1])
            self.assertEqual(capture.report['reference'], 'ORIGINAL_310P_INFERENCE')
        return capture, audit, args, kwargs

    def test_failure_fixture_replays_only_real_matmul_without_source_model(self):
        capture, audit, args, kwargs = self.exercise(None, True)
        saved = torch.load(checked(capture.root, audit.record['failed_call']['reproducer']), weights_only=True)
        for a, b in zip(saved['args'], args):
            self.assertTrue(torch.equal(a, b))
            self.assertEqual(a.dtype, b.dtype)
        self.assertNotIn('group_sizes', saved['kwargs'])
        seen = []
        def op(*a, **kw):
            seen.append((a, kw))
            return torch.zeros(3, 16, dtype=torch.float16)
        result = replay_failure(capture.root, self.root/'replay', 'cpu', 256,
            npu=SimpleNamespace(npu_quant_matmul=op))
        self.assertEqual(result['status'], 'EXECUTED')
        self.assertIn('NOT_GOLDEN', result['scope'])
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][1]['group_sizes'], [0, 0, 256])
        self.assertEqual(seen[0][0][2].dtype, torch.float32)
        self.assertTrue((self.root/'replay/report.json').is_file())
        self.assertEqual(result['stage'], 'COMPLETE')
        self.assertIn('runtime_at_end', result)

    def test_failure_in_explicit_mode_saves_original_call_for_ab_replay(self):
        capture, audit, _, _ = self.exercise(256, True)
        saved = torch.load(checked(capture.root, audit.record['failed_call']['reproducer']), weights_only=True)
        self.assertNotIn('group_sizes', saved['kwargs'])
        row = json.loads((capture.root/'qlinear-audit.json').read_text())['failed_call']
        self.assertEqual(row['effective_kwargs']['group_sizes'], [0, 0, 256])

    def test_failure_keeps_post_dispatch_libraries_and_original_exception(self):
        capture, _, _, _ = self.exercise(None, True)
        before = dict(environment={}, loaded_libraries=['/cann/libopapi.so'])
        after = dict(environment={}, loaded_libraries=['/cann/libopapi.so', '/vendor/libcust_opapi.so'])
        seen = []
        def op(*args, **kwargs):
            seen.append(kwargs)
            raise RuntimeError('Cannot find binary for op TransData')
        with patch('qwen35_moe_dflash.target.layer_probe.qlinear.runtime', side_effect=[before, after]), \
             patch('qwen35_moe_dflash.target.layer_probe.qlinear.inventory', side_effect=OSError('diagnostic failed')):
            result = replay_failure(capture.root, self.root/'replay', 'cpu', 256,
                npu=SimpleNamespace(npu_quant_matmul=op))
        self.assertEqual(len(seen), 1)
        self.assertEqual(result['status'], 'FAILED')
        self.assertEqual(result['stage'], 'OP_DISPATCH')
        self.assertIn('TransData', result['error'])
        self.assertEqual(result['libraries_added'], ['/vendor/libcust_opapi.so'])
        self.assertEqual(result['runtime_at_end'], after)
        self.assertIn('diagnostic failed', result['backend_diagnostic_error'])
        self.assertFalse((self.root/'replay/output.pt').exists())


if __name__ == '__main__':
    unittest.main()
