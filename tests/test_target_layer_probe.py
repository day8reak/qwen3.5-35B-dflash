import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from qwen35_moe_dflash.target.layer_probe.common import KIND, DTYPES, identity, metrics, read_array, save_json
from qwen35_moe_dflash.target.layer_probe.capture import Capture, cpu
from qwen35_moe_dflash.target.layer_probe.replay import Region, source_bindings, export_input_identity, check_adjusted_graph
from qwen35_moe_dflash.target.layer_probe.batch import bind_compiled, compare


torch_npu = SimpleNamespace(npu_rms_norm=lambda x, gamma, eps: (F.rms_norm(x, [gamma.numel()], gamma, eps), None))


class CProRMSNorm(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.variance_epsilon = 1e-6

    def forward(self, hidden_states):
        output = torch_npu.npu_rms_norm(hidden_states, self.weight + 1.0, self.variance_epsilon)[0]
        return output.type_as(hidden_states)


class ChangedCProRMSNorm(CProRMSNorm):
    def forward(self, hidden_states):
        return torch_npu.npu_rms_norm(hidden_states, self.weight.float() + 1.0, self.variance_epsilon)[0]


class FixtureTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ['AI_RUN_DIR'])
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_cpu_snapshot_before_inplace_is_independent(self):
        x = torch.arange(6.).reshape(2, 3)
        snapshot = cpu({'cache': x})
        x.add_(100)
        torch.testing.assert_close(snapshot['cache'], torch.arange(6.).reshape(2, 3))

    def test_emit_preserves_integer_weight_bits_and_reuses_only_weights(self):
        capture = Capture(self.root/'capture', layers=[0], calls=2)
        weights = {'W_q': torch.tensor([[-2147483648, 2147483647]], dtype=torch.int32),
                   'scale': torch.tensor([2**60+33], dtype=torch.int64)}
        for call in range(2):
            capture.active = dict(layer=0, call=call, seq_len=1)
            capture.emit('q', 'qlinear', {'x': torch.ones(1, 2)}, {'y': torch.ones(1, 2)*call}, weights=weights)
        rows = capture.report['regions']
        self.assertEqual(rows[0]['weights'], rows[1]['weights'])
        saved = torch.load(capture.root/rows[0]['weights']['path'], weights_only=True)
        for name in weights:
            self.assertTrue(torch.equal(saved[name], weights[name]))
            self.assertEqual(saved[name].dtype, weights[name].dtype)
        self.assertFalse(np.array_equal(read_array(capture.root, rows[0]['outputs'][0]), read_array(capture.root, rows[1]['outputs'][0])))

    def test_metrics_zero_reference_and_nonfinite(self):
        r = metrics(np.ones(2), np.zeros(2))
        self.assertIsNone(r['relative_l2'])
        self.assertIsNone(r['passed'])
        r = metrics(np.array([np.nan]), np.ones(1), atol=0, rtol=0)
        self.assertFalse(r['passed'])
        self.assertEqual(r['nonfinite_actual'], 1)

    def test_integer_comparison_never_uses_float_tolerance(self):
        r = metrics(np.array([2], np.int64), np.array([1], np.int64), atol=10, rtol=10)
        self.assertFalse(r['passed'])
        self.assertEqual(r['gate'], 'EXACT_INTEGER')

    def test_explicit_source_names_allow_pruned_inputs(self):
        specs = [dict(name='k'), dict(name='v')]
        graph = {'op': [dict(name='l_probe_input_001_', type='Data'),
                        dict(name='out', type='NetOutput', input=['node:0'])]}
        self.assertEqual(source_bindings(graph, specs)[0]['fixture_index'], 1)
        graph['op'][0]['name'] = 'mystery'
        with self.assertRaises(ValueError):
            source_bindings(graph, specs)

    def test_compiled_binding_reorders_equal_shape_inputs_by_edges(self):
        specs = [dict(name='k', shape=[1, 2], dtype='float16'), dict(name='v', shape=[1, 2], dtype='float16')]
        desc = dict(shape={'dim': [1, 2]}, dtype=1)
        doc = {'graph': [{'input': ['v:0', 'k:0'], 'op': [
            dict(name='k', type='Data', output_desc=[desc]), dict(name='v', type='Data', output_desc=[desc]),
            dict(name='out', type='NetOutput', input_desc=[desc])]}]}
        with patch('qwen35_moe_dflash.draft.compile_precision._observed_desc', return_value=dict(shape=[1,2], dtype='float16')):
            result = bind_compiled(doc, specs, dict(name='golden', shape=[1,2], dtype='float16'), {})
            self.assertEqual([r['name'] for r in result['inputs']], ['v','k'])
            doc['graph'][0]['input'] = ['v:0', 'v:0']
            with self.assertRaises(ValueError):
                bind_compiled(doc, specs, dict(shape=[1,2], dtype='float16'), {})

    def test_compiled_binding_decodes_all_fixture_types_from_schema_names(self):
        # Deliberately nonstandard enum numbers: source AIR codes are not the
        # compiled schema. Exercise the real decoder, not a mocked dtype result.
        aliases = dict(float16=('DT_FLOAT16', 'HALF'), float32=('DT_FLOAT', 'FLOAT'),
                       int16=('DT_INT16', 'INT16'), int32=('DT_INT32', 'INT32'),
                       int64=('DT_INT64', 'INT64'))
        self.assertEqual(set(aliases), set(DTYPES))
        for dtype, names in aliases.items():
            for alias in names:
                with self.subTest(dtype=dtype, alias=alias):
                    desc = dict(shape={'dim': [1]}, dtype=51)
                    golden_desc = dict(shape={'dim': [1]}, dtype=71)
                    doc = {'graph': [{'input': ['arg6_1:0'], 'op': [
                        dict(name='arg6_1', type='Data', output_desc=[desc]),
                        dict(name='out', type='NetOutput', input_desc=[golden_desc])]}]}
                    enums = {'dtype': {'values': {alias: 51, 'FP16': 71}}}
                    source = dict(name='arg6_1', shape=[1], dtype=dtype)
                    result = bind_compiled(doc, [source], dict(shape=[1], dtype='float16'), enums)
                    self.assertEqual(result['inputs'], [dict(source, index=0)])

    def test_compiled_binding_rejects_real_mismatches_and_unknown_enum(self):
        desc = dict(shape={'dim': [1]}, dtype=51)
        golden_desc = dict(shape={'dim': [1]}, dtype=71)
        doc = {'graph': [{'input': ['arg6_1:0'], 'op': [
            dict(name='arg6_1', type='Data', output_desc=[desc]),
            dict(name='out', type='NetOutput', input_desc=[golden_desc])]}]}
        source = dict(name='arg6_1', shape=[1], dtype='int16')
        for alias, shape in (('DT_INT32', [1]), ('DT_INT16', [2]), ('UNRECOGNIZED', [1])):
            with self.subTest(alias=alias, shape=shape):
                desc['shape']['dim'] = shape
                enums = {'dtype': {'values': {alias: 51, 'FP16': 71}}}
                with self.assertRaises(ValueError) as caught:
                    bind_compiled(doc, [source], dict(shape=[1], dtype='float16'), enums)
                message = str(caught.exception)
                for evidence in ('arg6_1', 'expected=', 'observed=', alias, 'raw_dtype', '51'):
                    self.assertIn(evidence, message)

    def make_native(self, *, runtime_dtype='float16', delta=0, stable=True):
        fixture, native = self.root/'fixture', self.root/'native'
        fixture.mkdir(); native.mkdir()
        name = 'l00_c00_norm_o0'
        (fixture/name).mkdir(); (native/name).mkdir()
        golden = np.array([1, 2], dtype=np.float16)
        golden.tofile(fixture/name/'golden_0.bin')
        spec = dict(name='golden', index=0, shape=[2], dtype=runtime_dtype, golden_dtype='float16',
                    bytes=2*np.dtype(runtime_dtype).itemsize, fixture_bytes=4,
                    fixture_sha256=identity(fixture/name/'golden_0.bin', fixture)['sha256'])
        case = dict(case=name, layer=0, call=0, point='norm', phase='prefill', kind='norm', ordinal=0,
                    output_name='norm_y', inputs=[], outputs=[spec], source_dtype_preserved=runtime_dtype=='float16', graphop_counts={})
        save_json(fixture/'manifest.json', dict(artifact_kind=KIND, warmup=3, repetitions=10, cases=[case]))
        save_json(native/name/'result.json', dict(status='EXECUTED', evidence_kind='device-hiai',
            warmup=3, repetitions=10, api_us=[1]*10, input_reset='copy-each-call'))
        for i in range(10):
            (golden.astype(runtime_dtype) + delta + (i if not stable else 0)).tofile(native/name/f'{i}.bin')
        return fixture, native

    def test_compare_records_promoted_dtype_without_reinterpreting_bytes(self):
        fixture, native = self.make_native(runtime_dtype='float32')
        result = compare(fixture, native, atol=0, rtol=0)
        self.assertEqual(result['status'], 'PASSED')
        self.assertFalse(result['cases'][0]['source_dtype_preserved'])
        self.assertFalse(result['all_model_layers_passed'])

    def test_compare_does_not_claim_pass_without_thresholds(self):
        fixture, native = self.make_native()
        self.assertEqual(compare(fixture, native)['status'], 'METRICS_RECORDED')

    def test_compare_detects_repeat_instability(self):
        fixture, native = self.make_native(stable=False)
        self.assertEqual(compare(fixture, native)['first_failed_boundary'], 'l00_c00_norm_o0')

    def test_missing_output_is_failure_not_skipped(self):
        fixture, native = self.make_native()
        (native/'l00_c00_norm_o0/9.bin').unlink()
        self.assertEqual(compare(fixture, native)['status'], 'FAILED')

    def test_export_input_identity_handles_reordering_and_restores_hook(self):
        k, v, w = torch.zeros(2), torch.zeros(2), torch.ones(2)
        def original(inputs, weight_name, export_graph):
            return False, len(weight_name)
        module = SimpleNamespace(_is_weight_externalized=original)
        graph = SimpleNamespace(op=[SimpleNamespace(type='Data', name=n) for n in ('arg0_1','arg1_1','arg2_1')])
        with patch('qwen35_moe_dflash.target.layer_probe.replay.importlib.import_module', return_value=module):
            with export_input_identity([k,v]) as evidence:
                self.assertEqual(module._is_weight_externalized([v,w,k], {id(w):'weight'}, graph), (False,1))
            self.assertEqual(evidence['nodes'], {'arg0_1':1,'arg2_1':0})
            self.assertIs(module._is_weight_externalized, original)
            with self.assertRaises(ValueError):
                with export_input_identity([k,v]):
                    module._is_weight_externalized([k.clone()], {}, graph)
            self.assertIs(module._is_weight_externalized, original)

    def test_source_binding_uses_actual_identity_even_for_equal_values(self):
        specs=[dict(name='k'),dict(name='v')]
        graph={'op':[dict(name='arg0_1',type='Data'),dict(name='arg2_1',type='Data'),
                     dict(name='out',type='NetOutput',input=['y:0'])]}
        bound=source_bindings(graph,specs,{'arg0_1':1,'arg2_1':0})
        self.assertEqual([r['fixture_index'] for r in bound],[1,0])

    def test_replay_convolution_matches_executed_valid_suffix(self):
        torch.manual_seed(10)
        before = torch.randn(1, 4, 4)
        x = torch.randn(1, 4, 3)
        weight = torch.randn(4, 1, 4)
        full = torch.cat((before, x), -1)
        golden = F.conv1d(full, weight, groups=4)[..., -3:]
        row = dict(kind='conv', attrs={'groups':4}, inputs=[dict(name='padded')])
        region = Region(row, {'weight': weight}, None, None, 0)
        actual = region(full[..., -6:])
        torch.testing.assert_close(actual, golden)
        # Trace the exact generated forward signature on CPU; no complete model involved.
        exported = torch.export.export(region, (full[..., -6:].contiguous(),))
        torch.testing.assert_close(exported.module()(full[..., -6:].contiguous()), golden)

    def test_folded_norm_preserves_half_rounding_and_export_has_no_weight_add(self):
        bits = np.arange(65536, dtype=np.uint16).view(np.float16)
        weight = torch.from_numpy(bits[np.isfinite(bits)].copy())
        before = weight.clone()
        row = dict(kind='norm', attrs={'epsilon':1e-6}, inputs=[dict(name='x')])
        target = SimpleNamespace(Qwen3_5MoeRMSNorm=CProRMSNorm)
        region = Region(row, {'weight':weight}, target, torch_npu, 0)
        # Exhaust all finite FP16 weights, including ties, subnormals and -1 cancellation.
        torch.testing.assert_close(region.gamma, (weight.double()+1).half(), rtol=0, atol=0)
        torch.testing.assert_close(weight, before, rtol=0, atol=0)
        self.assertNotEqual(region.gamma.data_ptr(), weight.data_ptr())
        self.assertEqual(region.export_adjustments[0]['gamma']['dtype'], 'float16')
        for dtype in (torch.float16, torch.float32):
            w = torch.tensor([-1., -0.9995, -0.5, 0., 2**-11, 2**-10, 1., 3.], dtype=dtype)
            region = Region(row, {'weight':w}, target, torch_npu, 0)
            original = CProRMSNorm(w)
            for shape in ((1,1,8), (1,256,8), (1,4,2,8)):
                x = torch.randn(shape, dtype=dtype)
                torch.testing.assert_close(region(x), original(x), rtol=0, atol=0)
            exported = torch.export.export(region, (x,))
            self.assertFalse(any(n.target == torch.ops.aten.add.Tensor for n in exported.graph.nodes))
            torch.testing.assert_close(exported.module()(x), original(x), rtol=0, atol=0)

    def test_unknown_norm_formula_is_not_silently_folded(self):
        row = dict(kind='norm', attrs={'epsilon':1e-6}, inputs=[dict(name='x')])
        with self.assertRaisesRegex(ValueError, 'forward changed'):
            Region(row, {'weight':torch.ones(8)}, SimpleNamespace(Qwen3_5MoeRMSNorm=ChangedCProRMSNorm), torch_npu, 0)

    def test_explicit_conv2d_preserves_prefill_decode_bias_dtype_and_rank(self):
        torch.manual_seed(24)
        row = dict(kind='conv', attrs={'groups':4}, inputs=[dict(name='padded')])
        for dtype in (torch.float32, torch.float16):
            for tokens in (1, 256):
                for with_bias in (False, True):
                    with self.subTest(dtype=dtype, tokens=tokens, bias=with_bias):
                        x = torch.randn(1,4,tokens+3, dtype=dtype)
                        w = torch.randn(4,1,4, dtype=dtype)
                        b = torch.randn(4, dtype=dtype) if with_bias else None
                        region = Region(row, dict(weight=w, **({'bias':b} if with_bias else {})), None, None, 0)
                        golden = F.conv1d(x, w, b, groups=4)
                        actual = region(x)
                        self.assertEqual(actual.dtype, dtype)
                        torch.testing.assert_close(actual, golden, rtol=0, atol=0)
        exported = torch.export.export(region, (x,))
        conv, = [n for n in exported.graph.nodes if n.target == torch.ops.aten.conv2d.default]
        self.assertEqual(list(conv.args[0].meta['val'].shape), [1,4,1,259])
        self.assertEqual(list(conv.args[1].meta['val'].shape), [4,1,1,4])
        torch.testing.assert_close(exported.module()(x), golden, rtol=0, atol=0)

    def test_air_check_rejects_old_conv_rank_and_norm_add(self):
        x = dict(name='x', type='Data', output_desc=[dict(shape={'dim':[1,8192,259]})])
        w = dict(name='w', type='Const', output_desc=[dict(shape={'dim':[8192,1,4]})])
        def unsqueeze(name, source):
            return dict(name=name, type='Unsqueeze', input=[source+':0'], output_desc=[dict(shape={})],
                        attr=[dict(key='axes', value={'list':{'i':[2]}})])
        conv = dict(name='conv', type='Conv2D', input=['x:0','wu:0'])
        graph = dict(op=[x,w,unsqueeze('wu','w'),conv])
        with self.assertRaisesRegex(ValueError, 'singleton height'):
            check_adjusted_graph('conv',graph)
        graph['op'].append(unsqueeze('xu','x'))
        conv['input'][0] = 'xu:0'
        result = check_adjusted_graph('conv',graph)
        self.assertEqual(result['input_shape'], [1,8192,1,259])
        self.assertEqual(result['filter_shape'], [8192,1,1,4])
        norm = dict(name='norm', type='RmsNorm', input=['x:0','add:0'])
        graph = dict(op=[x,w,dict(name='add',type='Add',input=['w:0']),norm])
        with self.assertRaisesRegex(ValueError, 'constant gamma'):
            check_adjusted_graph('norm',graph)
        graph['op'].pop(2)
        norm['input'][1] = 'w:0'
        self.assertEqual(check_adjusted_graph('norm',graph)['status'], 'PASSED')

    def test_replay_attention_nz_layout_and_gate(self):
        b, h, t, d = 1, 2, 3, 32
        nd = torch.arange(b*h*t*d, dtype=torch.float32).reshape(b,h,t,d)
        nz = nd.reshape(b,h,t,d//16,16).transpose(2,3).contiguous().reshape(b,h*d//16,t,16)
        gate = torch.randn(b,t,h*d)
        row = dict(kind='attention_post', attrs={'num_heads':h,'head_dim':d},
                   inputs=[dict(name='attention_nz'),dict(name='gate')])
        region = Region(row, {}, None, None, 0)
        torch.testing.assert_close(region(nz,gate), nd.transpose(1,2).reshape(b,t,h*d)*gate.sigmoid())


class Qwen3_5MoeRMSNorm(torch.nn.Module):
    def __init__(self, width, eps=1e-6):
        super().__init__(); self.weight=torch.nn.Parameter(torch.zeros(width)); self.variance_epsilon=eps
    def forward(self,x):
        return x * torch.rsqrt(x.square().mean(-1,keepdim=True)+self.variance_epsilon)*(1+self.weight)


class LinearMixer(torch.nn.Module):
    def __init__(self, npu):
        super().__init__(); self.npu=npu; self.num_k_heads=2; self.num_v_heads=4
        self.in_proj_qkv=torch.nn.Linear(8,16,bias=False)
        self.in_proj_a=torch.nn.Linear(8,4,bias=False); self.in_proj_b=torch.nn.Linear(8,4,bias=False)
        self.conv1d=torch.nn.Conv1d(16,16,4,groups=16,bias=False)
        self.A_log=torch.nn.Parameter(torch.zeros(4)); self.dt_bias=torch.nn.Parameter(torch.ones(4))
        self.out_proj=torch.nn.Linear(8,8,bias=False)
    def forward(self, hidden_states, past_key_values, effective_length):
        x=self.in_proj_qkv(hidden_states); a=self.in_proj_a(hidden_states); b=self.in_proj_b(hidden_states)
        old, state=past_key_values; t=x.shape[1]
        all_x=torch.cat((old,x.transpose(1,2)),-1)
        old.copy_(all_x[...,-4:])
        out=F.silu(F.conv1d(all_x,self.conv1d.weight,groups=16)[...,-t:]).transpose(1,2)
        q,k,v=torch.split(out,[4,4,8],-1)
        q=q.reshape(1,t,2,2).repeat_interleave(2,2); k=k.reshape(1,t,2,2).repeat_interleave(2,2)
        y,s=self.npu.npu_chunk_gated_delta_rule(q,k,v.reshape(1,t,4,2),
            g=-self.A_log.exp()*F.softplus(a+self.dt_bias),beta=b.sigmoid(),initial_state=state.float(),
            effective_length=effective_length,chunk_size=64,output_final_state=True,use_qk_l2norm_in_kernel=True)
        state.copy_(s.to(torch.float16))
        return self.out_proj(y.reshape(1,t,8)),past_key_values


class Decoder(torch.nn.Module):
    def __init__(self,npu):
        super().__init__(); self.input_layernorm=Qwen3_5MoeRMSNorm(8)
        self.linear_attn=LinearMixer(npu); self.post_attention_layernorm=Qwen3_5MoeRMSNorm(8); self.mlp=torch.nn.Identity()
    def forward(self,hidden_states,past_key_values,effective_length):
        branch,cache=self.linear_attn(self.input_layernorm(hidden_states),past_key_values,effective_length)
        x=hidden_states+branch
        return x+self.mlp(self.post_attention_layernorm(x)),cache


class HookTest(unittest.TestCase):
    setUp = FixtureTest.setUp
    tearDown = FixtureTest.tearDown
    def test_original_execution_unchanged_and_cpro_conv_boundary_from_actual_suffix(self):
        npu=SimpleNamespace(npu_chunk_gated_delta_rule=lambda q,k,v,**kw:(v,kw['initial_state']+0.25),
            npu_cache_update_=lambda *a,**k:None, adn_fused_infer_attention=lambda **k:None)
        layer=Decoder(npu).eval()
        wrapper=SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace(layers=[layer],start_layer_idx=0)))
        x=torch.randn(1,3,8); cache=(torch.randn(1,16,4),torch.zeros(1,4,2,2))
        with torch.no_grad(): expected=layer(x,cpu(cache),torch.tensor([3],dtype=torch.int16))
        capture=Capture(self.root/'hooks',layers=[0],calls=1)
        fake_cls=type('Wrapper',(),{'forward':lambda self,*a,**k:None})
        nz=SimpleNamespace(apply_rotary_pos_emb=lambda *a: a[:2])
        original_conv=F.conv1d
        capture.install(npu,nz,SimpleNamespace(Qwen3_5MoeForCausalLMWrapper=fake_cls))
        capture.attach(wrapper)
        try:
            with torch.no_grad(): actual=layer(x,cpu(cache),torch.tensor([3],dtype=torch.int16))
        finally:
            capture.close()
        torch.testing.assert_close(actual[0],expected[0]); self.assertIs(F.conv1d,original_conv)
        self.assertEqual(capture.report['status'],'CAPTURED',capture.report['errors'])
        rows={r['point']:r for r in capture.report['regions']}
        state=read_array(capture.root,rows['convstate_update']['outputs'][0])
        np.testing.assert_array_equal(state,actual[1][0].numpy())
        old=read_array(capture.root,rows['convstate_update']['inputs'][1])
        np.testing.assert_array_equal(old,cache[0].numpy())
        self.assertEqual(rows['gdn']['inputs'][-1]['dtype'],'int16')


class AttentionMixer(torch.nn.Module):
    def __init__(self,npu,nz):
        super().__init__(); self.npu=npu; self.nz=nz; self.num_heads=2; self.head_dim=32
        self.q_proj=torch.nn.Linear(8,128,bias=False); self.k_proj=torch.nn.Linear(8,32,bias=False)
        self.v_proj=torch.nn.Linear(8,32,bias=False); self.o_proj=torch.nn.Linear(64,8,bias=False)
        self.q_norm=Qwen3_5MoeRMSNorm(32); self.k_norm=Qwen3_5MoeRMSNorm(32)
    def forward(self,hidden_states,past_key_values,effective_length):
        b,t,_=hidden_states.shape
        q,gate=self.q_proj(hidden_states).reshape(b,t,2,64).chunk(2,-1)
        q=self.q_norm(q).transpose(1,2)
        k=self.k_norm(self.k_proj(hidden_states).reshape(b,t,1,32)).transpose(1,2)
        v=self.v_proj(hidden_states).reshape(b,t,1,32)
        q,k=self.nz.apply_rotary_pos_emb(q,k,torch.ones(b,t,8),torch.zeros(b,t,8))
        for fresh,cache in zip((k.transpose(1,2),v),past_key_values):
            self.npu.npu_cache_update_(cache,fresh.reshape(t,2,16),torch.tensor([0],dtype=torch.int32),torch.tensor([0],dtype=torch.int32))
        nzq=q.reshape(b,2,t,2,16).transpose(2,3).contiguous().reshape(b,4,t,16)
        out=self.npu.adn_fused_infer_attention(query=nzq,key=[past_key_values[0]],value=[past_key_values[1]],
            atten_mask=torch.zeros(b,1,t,64),block_table=torch.tensor([[0,1]],dtype=torch.int32),
            all_seq_lengths_q=[t],num_heads=2,num_key_value_heads=1,block_size=32,input_layout='BNSD',
            actual_seq_lengths_q=[t],actual_seq_lengths_kv=[64],scale_value=0.1,inner_precise=2)
        out=out.reshape(b,2,2,t,16).transpose(2,3).contiguous().reshape(b,2,t,32).transpose(1,2).reshape(b,t,64)
        return self.o_proj(out*gate.reshape(b,t,64).sigmoid()),past_key_values


class AttentionDecoder(torch.nn.Module):
    def __init__(self,npu,nz):
        super().__init__(); self.input_layernorm=Qwen3_5MoeRMSNorm(8)
        self.self_attn=AttentionMixer(npu,nz); self.post_attention_layernorm=Qwen3_5MoeRMSNorm(8); self.mlp=torch.nn.Identity()
    def forward(self,hidden_states,past_key_values,effective_length):
        branch,cache=self.self_attn(self.input_layernorm(hidden_states),past_key_values,effective_length)
        x=hidden_states+branch
        return x+self.mlp(self.post_attention_layernorm(x)),cache


class AttentionHookTest(unittest.TestCase):
    setUp = FixtureTest.setUp
    tearDown = FixtureTest.tearDown
    def test_full_attention_capture_has_distinct_real_kv_and_replay_gate(self):
        def update(cache,fresh,blocks,offsets):
            for i in range(fresh.shape[0]): cache[0,:,i,:].copy_(fresh[i])
        npu=SimpleNamespace(npu_chunk_gated_delta_rule=lambda *a,**k:None,npu_cache_update_=update,
                            adn_fused_infer_attention=lambda **k:k['query'])
        nz=SimpleNamespace(apply_rotary_pos_emb=lambda q,k,c,s:(q,k))
        layers=[AttentionDecoder(npu,nz),AttentionDecoder(npu,nz)]
        wrapper=SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace(layers=layers,start_layer_idx=3)))
        x=torch.randn(1,3,8); cache=(torch.zeros(2,2,32,16),torch.zeros(2,2,32,16))
        capture=Capture(self.root/'attention',layers=[3,4],calls=1)
        cls=type('Wrapper',(),{'forward':lambda self,*a,**k:None})
        capture.install(npu,nz,SimpleNamespace(Qwen3_5MoeForCausalLMWrapper=cls)); capture.attach(wrapper)
        try:
            with torch.no_grad():
                for layer in layers: layer(x,cpu(cache),torch.tensor([3],dtype=torch.int16))
        finally: capture.close()
        self.assertEqual(capture.report['status'],'CAPTURED',capture.report['errors'])
        rows=[r for r in capture.report['regions'] if r['layer']==3]
        bypoint={r['point']:r for r in rows}
        k=read_array(capture.root,bypoint['cache_k']['outputs'][0]); v=read_array(capture.root,bypoint['cache_v']['outputs'][0])
        self.assertFalse(np.array_equal(k,v))
        row=bypoint['attention_layout_gate']
        region=Region(row,{},None,None,0)
        inputs=[torch.from_numpy(read_array(capture.root,s).copy()) for s in row['inputs']]
        expected=torch.from_numpy(read_array(capture.root,row['outputs'][0]))
        torch.testing.assert_close(region(*inputs),expected)


if __name__=='__main__': unittest.main()
