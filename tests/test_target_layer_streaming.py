"""Host checks for reference placement; no NPU numerical-pass claim."""
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from qwen35_moe_dflash.target.layer_probe.capture import Capture, cpu
from qwen35_moe_dflash.target.layer_probe.placement import LayerStreaming


class QLinear(torch.nn.Module):
    """Toy quantized module: retain carrier bits while testing placement only."""
    def __init__(self, linear):
        super().__init__()
        self.register_buffer('logical_weight', linear.weight.detach().clone())
        self.register_buffer('W_q', torch.tensor([-2147483648, 2147483647], dtype=torch.int32))
        self.register_buffer('scale', torch.tensor([0.000000123], dtype=torch.float32))
        self.register_buffer('encoded_scale', torch.tensor([2**60 + 33], dtype=torch.int64))

    def forward(self, x):
        return torch.nn.functional.linear(x, self.logical_weight)


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(4, 4, bias=False)
        self.fail = False

    def forward(self, hidden_states, past_key_values):
        if self.fail:
            raise RuntimeError('original kernel failed')
        state = past_key_values[0]
        state.add_(hidden_states.mean(1) * 0.01)
        return hidden_states + self.projection(hidden_states) * 0.02 + state[:, None], past_key_values


def wrapper_type():
    class Wrapper(torch.nn.Module):
        def __init__(self, device, count=40):
            super().__init__()
            self.construct_device = device
            self.embedding_in_omc = False
            self.model = torch.nn.Module()
            self.model.config = SimpleNamespace(num_hidden_layers=count)
            self.model.model = torch.nn.Module()
            self.model.model.start_layer_idx = 0
            self.model.model.layers = torch.nn.ModuleList([Layer() for _ in range(count)])
            self.model.model.norm = torch.nn.LayerNorm(4)
            self.model.lm_head = torch.nn.Linear(4, 8, bias=False)
            # Reproduces the failing original constructor order, before quant.
            self.model.to(device)

        def forward(self, inputs_embeds, embed_scale, past_key_values):
            x = inputs_embeds * embed_scale
            caches = []
            for layer, cache in zip(self.model.model.layers, past_key_values):
                x, cache = layer(x, past_key_values=cache)
                caches.extend(cache)
            return self.model.lm_head(self.model.model.norm(x[:, -1:])), *caches
    return Wrapper


def quantize(wrapper):
    for layer in wrapper.model.model.layers:
        layer.projection = QLinear(layer.projection)
    wrapper.model.lm_head = QLinear(wrapper.model.lm_head)


def generation(wrapper):
    cache = [(torch.zeros(1, 4),) for _ in wrapper.model.model.layers]
    embedding = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 32
    hidden = embedding[torch.tensor([[1, 2, 3]])]
    rows = []
    with torch.no_grad():
        for _ in range(2):
            output = wrapper(inputs_embeds=hidden, embed_scale=torch.ones(1), past_key_values=cache)
            rows.append(cpu(output))
            # Decode token is selected by the FULL model, with continuous state.
            hidden = embedding[output[0][:, -1].argmax(-1).unsqueeze(1)]
    return rows


class StreamingTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ['AI_RUN_DIR'])
        self.root = Path(self.temp.name)
        self.cls = wrapper_type()
        self.module = SimpleNamespace(Qwen3_5MoeForCausalLMWrapper=self.cls)
        self.report = {}
        self.stream = LayerStreaming(self.report)
        self.addCleanup(self.temp.cleanup)

    def test_all_40_layers_prefill_decode_match_with_one_resident_unit_and_capture(self):
        torch.manual_seed(37)
        baseline = self.cls('cpu')
        quantize(baseline)
        expected = generation(baseline)
        npu = SimpleNamespace(npu_chunk_gated_delta_rule=lambda *a, **k: None,
            npu_cache_update_=lambda *a, **k: None, adn_fused_infer_attention=lambda *a, **k: None)
        nz = SimpleNamespace(apply_rotary_pos_emb=lambda *a: a[:2])
        capture = Capture(self.root/'capture', layers=[0, 1, 2, 3], calls=2)
        capture.install(npu, nz, self.module)
        self.stream.install(self.module)
        try:
            torch.manual_seed(37)
            wrapper = self.cls('npu')  # Would fail on this CPU host without interception.
            self.assertEqual(wrapper.construct_device, 'cpu')
            self.assertEqual(self.stream.record['wrappers'], [])
            quantize(wrapper)
            original_weights = cpu(wrapper.state_dict())
            resident, events = set(), []
            def move(module, device):
                if id(module) in resident:
                    resident.remove(id(module))
                else:
                    self.assertFalse(resident, 'a previous unit is still resident')
                    resident.add(id(module))
                events.append(id(module))
                module.to(device)  # Device only, as in production; CPU execution here.
            with patch.object(self.stream, 'move', side_effect=move):
                actual = generation(wrapper)
            self.assertFalse(resident)
            self.assertEqual(len(events), (40+2)*2*2)
            for got, want in zip(actual, expected):
                for a, b in zip(got, want):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            for name, tensor in wrapper.state_dict().items():
                self.assertEqual(tensor.dtype, original_weights[name].dtype)
                self.assertTrue(torch.equal(tensor, original_weights[name]))
            row = self.stream.record['wrappers'][0]
            self.assertEqual(row['total_decoder_layers'], 40)
            self.assertTrue(all(n == 2 for n in row['units'].values()))
            self.assertEqual(self.stream.record['max_active_units'], 1)
            self.assertEqual({r['layer'] for r in capture.report['regions']}, {0, 1, 2, 3})
            self.assertEqual({r['phase'] for r in capture.report['regions']}, {'prefill', 'decode'})
        finally:
            self.stream.close()
            capture.close()
        # Toy layers deliberately lack the real attention boundaries.
        self.assertEqual(capture.report['status'], 'INCOMPLETE')
        self.assertNotIn('to', vars(wrapper))
        self.assertNotIn('to', vars(wrapper.model))

    def test_initial_bulk_transfer_and_later_bulk_transfer_are_blocked(self):
        init, forward = self.cls.__init__, self.cls.forward
        self.stream.install(self.module)
        try:
            wrapper = self.cls('npu', count=2)
            self.assertEqual(wrapper.construct_device, 'cpu')
            for obj in (wrapper, wrapper.model):
                with self.assertRaisesRegex(RuntimeError, 'whole-model'):
                    obj.to('npu')
                obj.to('cpu')
        finally:
            self.stream.close()
        self.assertIs(self.cls.__init__, init)
        self.assertIs(self.cls.forward, forward)

    def test_unquantized_model_rejected_before_any_transfer(self):
        self.stream.install(self.module)
        try:
            wrapper = self.cls('npu', count=2)
            with patch.object(self.stream, 'move') as move:
                with self.assertRaisesRegex(ValueError, 'W4 QLinear'):
                    generation(wrapper)
                move.assert_not_called()
        finally:
            self.stream.close()

    def test_truncated_model_rejected_instead_of_using_prefix_logits(self):
        self.stream.install(self.module)
        try:
            wrapper = self.cls('npu', count=4)
            quantize(wrapper)
            wrapper.model.model.layers = wrapper.model.model.layers[:2]
            with self.assertRaisesRegex(ValueError, 'not a truncated prefix'):
                generation(wrapper)
        finally:
            self.stream.close()

    def test_kernel_exception_offloads_active_unit_and_restores_hooks(self):
        self.stream.install(self.module)
        try:
            wrapper = self.cls('npu', count=4)
            quantize(wrapper)
            wrapper.model.model.layers[1].fail = True
            with patch.object(self.stream, 'move', wraps=self.stream.move) as move:
                with self.assertRaisesRegex(RuntimeError, 'original kernel failed'):
                    generation(wrapper)
            self.assertIsNone(self.stream.active)
            self.assertEqual(move.call_count, 4)  # Enter/exit layers 0 and 1 only.
            self.assertIs(move.call_args.args[0], wrapper.model.model.layers[1])
            self.assertEqual(str(move.call_args.args[1]), 'cpu')
        finally:
            self.stream.close()
        self.assertFalse(wrapper.model.model.layers[0]._forward_pre_hooks)
        self.assertFalse(wrapper.model.model.layers[1]._forward_hooks)

    def test_partial_transfer_failure_also_attempts_offload(self):
        self.stream.install(self.module)
        try:
            wrapper = self.cls('npu', count=2)
            quantize(wrapper)
            with patch.object(self.stream, 'move', side_effect=[RuntimeError('allocation failed'), None]) as move:
                with self.assertRaisesRegex(RuntimeError, 'allocation failed'):
                    generation(wrapper)
            self.assertEqual(move.call_count, 2)
            self.assertEqual(str(move.call_args.args[1]), 'cpu')
            self.assertIsNone(self.stream.active)
        finally:
            self.stream.close()

    def test_missing_captures_do_not_hide_original_failure_status(self):
        capture = Capture(self.root/'failed', layers=[0, 1, 2, 3], calls=2)
        capture.close('NPU out of memory during original constructor')
        self.assertEqual(capture.report['status'], 'FAILED')


if __name__ == '__main__':
    unittest.main()
