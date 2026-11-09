import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from qwen35_moe_dflash.target.layer_probe.steps import build_four_layers, execute_steps, import_source, initial_state, run_steps


def config():
    return SimpleNamespace(num_hidden_layers=40, num_experts=256, layer_types=['linear_attention']*3+['full_attention'],
        linear_num_key_heads=16, linear_key_head_dim=128, linear_num_value_heads=32,
        linear_value_head_dim=128, linear_conv_kernel_dim=4, num_key_value_heads=2, head_dim=256)


class DirectStepsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ['AI_RUN_DIR'])
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def test_import_reuses_functions_without_running_main(self):
        (self.root/'inference.py').write_text('''
def prefill(*args): return "prefill"
def decoder(*args): return "decode"
def create_strict_causal_mask(*args): pass
def quant_model_inplace(*args): pass
if __name__ == "__main__":
    raise AssertionError("must not run original main")
''')
        cwd, paths = Path.cwd(), list(sys.path)
        source = import_source(self.root)
        self.assertEqual(source.prefill(), 'prefill')
        self.assertEqual(source.decoder(), 'decode')
        self.assertEqual(Path.cwd(), cwd)
        self.assertEqual(sys.path, paths)

    def test_only_four_layers_constructed_and_factory_restored(self):
        module = ModuleType('test_original_wrapper')
        calls = []
        def load(model_path, device, dtype, layer_num=None, start_layer_idx=0):
            calls.append((device, layer_num, dtype))
            self.assertEqual(layer_num, 4, 'must never create 40 and slice')
            model = torch.nn.Module()
            model.config = config()
            model.model = torch.nn.Module()
            model.model.layers = torch.nn.ModuleList([torch.nn.Linear(1, 1) for _ in range(layer_num)])
            return model
        module.load_npu_model = load
        class Wrapper(torch.nn.Module):
            def __init__(self, model_path, device, dtype, embedding_config, quant_enable):
                super().__init__()
                self.model = module.load_npu_model(model_path, device, dtype).to(device)
        Wrapper.__module__ = module.__name__
        record = {}
        with patch.dict(sys.modules, {module.__name__: module}):
            wrapper = build_four_layers(SimpleNamespace(Qwen3_5MoeForCausalLMWrapper=Wrapper), 'unused', record)
        self.assertIs(module.load_npu_model, load)
        self.assertEqual(calls, [('cpu', 4, torch.float16)])
        self.assertEqual(wrapper.model.config.num_hidden_layers, 4)
        self.assertEqual(record['checkpoint_layers'], 40)

    def test_live_cache_mask_and_argmax_pass_unchanged_to_original_decode(self):
        ids = torch.tensor([[1, 2, 3]])
        live_cache = [('actual_state_from_prefill',)]
        events = []
        source = SimpleNamespace(device='cpu', real_seq_len=3, config_data={'kv_cache_max_len': 32})
        mask = torch.ones(32, 32)
        source.create_strict_causal_mask = lambda *args: mask
        def prefill(wrapper, passed_ids, passed_mask, caches, max_len):
            self.assertIs(passed_ids, ids)
            self.assertEqual(len(caches), 4)
            self.assertEqual(caches[3][0].shape, (1, 32, 32, 16))
            self.assertEqual(Path(source.output_path), self.root/'prefill')
            passed_mask[..., 0, 0] = 7
            events.append(('prefill', passed_mask))
            Path(source.output_path, 'golden_logits.bin').write_bytes(b'prefill')
            return torch.tensor([[1., 8., 3.]]), torch.tensor([99]), live_cache
        def decoder(wrapper, first_token, pos, passed_mask, caches, count, max_len):
            self.assertEqual(first_token.item(), 1)  # argmax, NOT auxiliary token 99.
            self.assertIs(caches, live_cache)
            self.assertIs(passed_mask, events[0][1])
            self.assertEqual(passed_mask[..., 0, 0].item(), 7)
            self.assertEqual((pos, count, max_len), (3, 1, 32))
            self.assertIs(source.input_ids, ids)
            self.assertEqual(Path.cwd(), self.root/'decode')
            Path(source.output_path, 'golden_logits.bin').write_bytes(b'decode')
            events.append(('decode', passed_mask))
            return torch.tensor([1, 2])
        source.prefill, source.decoder = prefill, decoder
        cwd = Path.cwd()
        result = execute_steps(source, SimpleNamespace(model=SimpleNamespace(config=config())), ids, self.root, 1)
        self.assertEqual([e[0] for e in events], ['prefill', 'decode'])
        self.assertEqual(result.tolist(), [1, 2])
        self.assertEqual((self.root/'prefill/golden_logits.bin').read_bytes(), b'prefill')
        self.assertEqual(Path.cwd(), cwd)

    def test_prefill_failure_propagates_without_decode_or_retry(self):
        source = SimpleNamespace(device='cpu', real_seq_len=3, config_data={'kv_cache_max_len': 32},
            create_strict_causal_mask=lambda *a: torch.zeros(32, 32))
        def prefill(*args):
            raise RuntimeError('original V5 error')
        source.prefill = prefill
        source.decoder = lambda *a: self.fail('must not decode a failed prefill')
        cwd = Path.cwd()
        with self.assertRaisesRegex(RuntimeError, 'original V5'):
            execute_steps(source, SimpleNamespace(model=SimpleNamespace(config=config())), torch.ones(1, 3), self.root, 1)
        self.assertEqual(Path.cwd(), cwd)

    def test_unsupported_cache_shape_is_not_silently_reinterpreted(self):
        cfg = config()
        cfg.linear_num_key_heads = 8
        with self.assertRaisesRegex(ValueError, 'cache dimensions'):
            initial_state(cfg, 32, 'cpu')

    def test_baseline_driver_calls_original_steps_without_observers_or_backend_override(self):
        import json
        import numpy as np
        for filename in ('inference.py', 'utils.py', 'models/modeling_qwen3_5_moe_hiai_nz.py'):
            path = self.root/filename
            path.parent.mkdir(exist_ok=True)
            path.write_text('# synthetic test source\n')
        np.ones((4, 2), dtype=np.int8).tofile(self.root/'embedding.bin')
        np.ones((4, 1), dtype=np.float32).tofile(self.root/'scale.bin')
        cfg = dict(model_file='model', quanted_pth='quant', embedding_weight_path='embedding.bin',
            embedding_scale_path='scale.bin', prompts=['hello'], kv_cache_max_len=32)
        (self.root/'config.json').write_text(json.dumps(cfg))
        events = []
        class Wrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                c = config()
                c.num_hidden_layers, c.vocab_size, c.kv_cache_max_len = 4, 4, 32
                self.model = SimpleNamespace(config=c)
            def to(self, *args, **kwargs):
                self.assert_transfer = (args, kwargs)
                events.append('move')
                return self
        wrapper = Wrapper()
        class Tokenizer:
            def __call__(self, texts, **kwargs):
                self.texts = texts
                return {'input_ids': torch.tensor([[1, 2, 3]]), 'attention_mask': torch.ones(1, 3)}
            def decode(self, ids, **kwargs):
                return 'synthetic result'
        tokenizer = Tokenizer()
        source = SimpleNamespace(Qwen3_5MoeForCausalLMWrapper=Wrapper,
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a: tokenizer),
            create_strict_causal_mask=lambda *a: torch.zeros(32, 32))
        def quant(*args, **kwargs):
            self.assertIs(args[0], wrapper)
            self.assertEqual(kwargs, dict(start_layer_idx=0, device='cpu', pseudo_quant=False, quant_bits=4))
            events.append('quant')
        def prefill(model, ids, mask, caches, max_len):
            self.assertEqual(source.quant_embedding_weights.dtype, torch.int8)
            self.assertEqual(source.quant_embedding_scale.dtype, torch.float32)
            events.append('prefill')
            return torch.tensor([[1., 2., 3., 0.]]), None, caches
        def decoder(*args):
            events.append('decode')
            return torch.tensor([2, 1])
        source.quant_model_inplace, source.prefill, source.decoder = quant, prefill, decoder
        npu = SimpleNamespace(npu_quant_matmul=lambda: None)
        with patch.dict(sys.modules, {'torch_npu': npu}), \
             patch('qwen35_moe_dflash.target.layer_probe.steps.import_source', return_value=source), \
             patch('qwen35_moe_dflash.target.layer_probe.steps.build_four_layers', return_value=wrapper), \
             patch('qwen35_moe_dflash.target.layer_probe.capture.Capture.install', side_effect=AssertionError('no observers')):
            result = run_steps(self.root, 'config.json', self.root/'direct', device='cpu', baseline_only=True)
        self.assertEqual(result['status'], 'EXECUTED', result)
        self.assertEqual(events, ['quant', 'move', 'prefill', 'decode'])
        self.assertEqual(wrapper.assert_transfer, (('cpu',), {}))  # No dtype cast.
        self.assertEqual(tokenizer.texts, ['hello'*32])
        self.assertIn('ORIGINAL_PREFILL', result['steps'])
        self.assertIn('ORIGINAL_DECODE', result['steps'])
        self.assertFalse((self.root/'direct/capture').exists())
        self.assertEqual(result['numerical_validation'], 'NOT_VERIFIED_AGAINST_EXISTING_GOLDEN')


if __name__ == '__main__':
    unittest.main()
