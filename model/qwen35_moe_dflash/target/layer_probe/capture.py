"""Observe original inference; map observations to CPro regions, never compute a new golden."""
from __future__ import annotations
import functools
import inspect
from pathlib import Path
import re
import sys

import torch
import torch.nn.functional as F
from .common import KIND, DTYPES, identity, new_output, save_json


def cpu(value):
    # Copy directly off device. Do not allocate a second device-sized snapshot.
    if isinstance(value, torch.Tensor):
        return value.detach().to('cpu', copy=True).contiguous()
    if isinstance(value, tuple):
        return tuple(cpu(x) for x in value)
    if isinstance(value, list):
        return [cpu(x) for x in value]
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    return value


def bound(module, args, kwargs):
    values = inspect.signature(module.forward).bind(*args, **kwargs)
    values.apply_defaults()
    return dict(values.arguments)


class Capture:
    def __init__(self, output, layers=(0, 1, 2, 3), calls=2):
        self.root = new_output(output)
        self.layers, self.calls = set(layers), calls
        if not self.layers or min(self.layers) < 0 or calls < 1:
            raise ValueError('nonnegative layers and positive calls required')
        self.handles, self.patches, self.active, self.saved_weights = [], [], None, {}
        self.counts, self.attached = {}, set()
        self.quant_audit = None
        self.report = dict(artifact_kind=KIND, status='CAPTURING', reference='ORIGINAL_310P_INFERENCE',
            layers=sorted(self.layers), calls_per_layer=calls, regions=[], errors=[], sources=[],
            torch_version=torch.__version__,
            scope='REAL_INPUT_LOCAL_REGION_REPLAY; NOT_FULL_MODEL_OR_CPRO_STATE_FEEDBACK',
            numerical_gate='METRICS_ONLY_UNLESS_EXPLICIT_THRESHOLDS_SUPPLIED')
        self.flush()

    def flush(self):
        save_json(self.root / 'manifest.json', self.report)

    def source(self, path):
        path = Path(path).resolve()
        from .common import sha
        row = dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))
        if row not in self.report['sources']:
            self.report['sources'].append(row)

    def tensor(self, folder, name, tensor):
        tensor = cpu(tensor)
        dtype = str(tensor.dtype).removeprefix('torch.')
        if dtype not in DTYPES:
            raise ValueError('unsupported public replay dtype: ' + dtype)
        path = folder / (name + '.bin')
        tensor.numpy().tofile(path)
        return dict(identity(path, self.root), name=name, shape=list(tensor.shape), dtype=dtype)

    def emit(self, name, kind, inputs, outputs, *, weights=None, attrs=None, provenance='DIRECT_EXECUTED_TENSOR'):
        ctx = self.active
        if ctx is None:
            return
        base = f'l{ctx["layer"]:02d}_c{ctx["call"]:02d}_{name}'
        if any(r['id'] == base for r in self.report['regions']):
            raise ValueError('duplicate replay point: ' + base)
        folder = self.root / 'regions' / base
        folder.mkdir(parents=True)
        inputs, outputs = dict(inputs), dict(outputs)
        row = dict(id=base, layer=ctx['layer'], call=ctx['call'], point=name, kind=kind,
            phase='prefill' if ctx['seq_len'] > 1 else 'decode', sequence_length=ctx['seq_len'],
            execution_device=ctx.get('execution_device', 'UNSPECIFIED_TEST_CONTEXT'),
            ordinal=len(self.report['regions']), attrs=attrs or {}, provenance=provenance,
            inputs=[self.tensor(folder, k, v) for k, v in inputs.items()],
            outputs=[self.tensor(folder, 'golden_' + k, v) for k, v in outputs.items()])
        # One CPU weight copy per boundary, reused across captured decode steps.
        if weights:
            key = (ctx['layer'], name)
            if key not in self.saved_weights:
                path = self.root / 'weights' / f'l{ctx["layer"]:02d}_{name}.pt'
                path.parent.mkdir(exist_ok=True)
                torch.save(cpu(weights), path)
                self.saved_weights[key] = identity(path, self.root)
            row['weights'] = self.saved_weights[key]
        self.report['regions'].append(row)
        self.flush()

    def attach(self, wrapper):
        if id(wrapper) in self.attached:
            return
        self.attached.add(id(wrapper))
        if self.quant_audit is not None:
            self.quant_audit.attach(wrapper)
        model = wrapper.model.model
        start = getattr(model, 'start_layer_idx', 0)
        for local, layer in enumerate(model.layers):
            index = local + start
            if index not in self.layers:
                continue
            self.source(inspect.getfile(type(layer)))
            key = (id(wrapper), index)
            self.counts[key] = 0
            def before(module, args, kwargs, key=key, index=index):
                call = self.counts[key]
                self.counts[key] += 1
                if call >= self.calls:
                    return
                values = bound(module, args, kwargs)
                hidden = values['hidden_states']
                self.active = dict(layer=index, call=call, seq_len=hidden.shape[1], execution_device=str(hidden.device),
                    module=module, input_hidden=cpu(hidden), cache_before=cpu(values['past_key_values']),
                    cache_live=values['past_key_values'], values=values,
                    seen={}, mixer=None, cache_ordinal=0)
            def after(module, args, kwargs, output):
                ctx = self.active
                if ctx is None:
                    return
                seen = ctx['seen']
                if 'post_attention_layernorm' in seen and 'mlp' in seen:
                    self.emit('residual_mlp', 'add',
                        {'residual': seen['post_attention_layernorm']['args'][0], 'branch': seen['mlp']['out']},
                        {'hidden': output[0]})
                self.active = None
            self.handles.extend([layer.register_forward_pre_hook(before, with_kwargs=True),
                                 layer.register_forward_hook(after, with_kwargs=True)])
            for name, module in layer.named_modules():
                if not name:
                    continue
                # MoE is observed for residual alignment but never exported in this first batch.
                if name.startswith('mlp.'):
                    continue
                def pre(m, args, kwargs, name=name):
                    if self.active is None:
                        return
                    self.active['seen'][name] = dict(args=cpu(args), kwargs=cpu(kwargs))
                    if name in ('linear_attn', 'self_attn'):
                        self.active['mixer'] = name
                def post(m, args, kwargs, output, name=name, wrapper=wrapper, index=index, start=start):
                    ctx = self.active
                    if ctx is None:
                        return
                    record = ctx['seen'][name]
                    record['out'] = cpu(output)
                    label = name.replace('.', '_')
                    if type(m).__name__ == 'QLinear':
                        self.emit(label, 'qlinear', {'x': record['args'][0]}, {'y': output},
                                  weights=dict(W_q=m.W_q, scale=m.scale))
                    elif isinstance(m, torch.nn.Linear):
                        self.emit(label, 'linear', {'x': record['args'][0]}, {'y': output},
                                  weights=dict(m.state_dict()))
                    elif type(m).__name__ in ('Qwen3_5MoeRMSNorm', 'Qwen3_5MoeRMSNormGated'):
                        ins = {'x': record['args'][0]}
                        if len(record['args']) > 1:
                            ins['gate'] = record['args'][1]
                        self.emit(label, 'norm_gated' if 'gate' in ins else 'norm', ins, {'y': output},
                                  weights={'weight': m.weight}, attrs={'epsilon': m.variance_epsilon})
                    if name == 'post_attention_layernorm':
                        mixer = ctx['seen'].get('linear_attn', ctx['seen'].get('self_attn'))
                        if mixer is not None:
                            self.emit('residual_attention', 'add',
                                {'residual': ctx['input_hidden'], 'branch': mixer['out'][0]},
                                {'hidden': record['args'][0]})
                    if name == 'self_attn.o_proj' and 'attention_raw' in ctx:
                        qgate = ctx['seen']['self_attn.q_proj']['out']
                        attn = layer_module(wrapper, index, start).self_attn
                        gate = qgate.reshape(*qgate.shape[:-1], -1, 2*attn.head_dim).chunk(2, dim=-1)[1]
                        self.emit('attention_layout_gate', 'attention_post',
                            {'attention_nz': ctx['attention_raw'], 'gate': gate.reshape(*qgate.shape[:-1], -1)},
                            {'before_o_proj': record['args'][0]},
                            attrs={'num_heads': attn.num_heads, 'head_dim': attn.head_dim})
                    if name in ('linear_attn', 'self_attn'):
                        if name == 'linear_attn' and 'gdn_state' in ctx:
                            self.emit('recurrent_writeback', 'state_cast', {'state': ctx['gdn_state']},
                                {'written_state': ctx['cache_live'][1]}, attrs={'dtype': str(ctx['cache_live'][1].dtype).removeprefix('torch.')})
                        ctx['mixer'] = None
                self.handles.extend([module.register_forward_pre_hook(pre, with_kwargs=True),
                                     module.register_forward_hook(post, with_kwargs=True)])
        self.flush()

    def patch(self, obj, name, callback):
        original = getattr(obj, name)
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            return callback(original, args, kwargs)
        setattr(obj, name, wrapped)
        self.patches.append((obj, name, original))

    def install(self, npu, nz_module, wrapper_module, *, qlinear_group_size=None):
        self.report['torch_npu_version'] = getattr(npu, '__version__', 'unknown')
        if hasattr(npu, 'npu_quant_matmul') and hasattr(nz_module, 'QLinear'):
            from .qlinear import QuantMatmulAudit
            self.quant_audit = QuantMatmulAudit(self, npu, qlinear_group_size)
            self.quant_audit.install(nz_module.QLinear)
        elif qlinear_group_size is not None:
            raise ValueError('cannot apply group256 without the original QLinear and npu_quant_matmul')
        # Attach after the original program has loaded and quantized its model.
        cls = wrapper_module.Qwen3_5MoeForCausalLMWrapper
        def wrapper_call(original, args, kwargs):
            self.attach(args[0])
            result = original(*args, **kwargs)
            rows = self.report.setdefault('original_wrapper_outputs', [])
            if len(rows) < self.calls:
                folder = self.root/'wrapper_outputs'/str(len(rows))
                folder.mkdir(parents=True)
                rows.append(dict(output0=self.tensor(folder, 'output0', result[0]),
                    topK=kwargs.get('topK'), provenance='ORIGINAL_WRAPPER_RETURN; NOT_AN_EXPORTED_MODEL'))
                self.flush()
            return result
        self.patch(cls, 'forward', wrapper_call)

        def conv(original, args, kwargs):
            ctx = self.active
            enabled = ctx is not None and ctx['mixer'] == 'linear_attn'
            x, weight = (cpu(args[0]), cpu(args[1])) if enabled else (None, None)
            result = original(*args, **kwargs)
            if enabled:
                length, width = ctx['seq_len'], weight.shape[-1]
                padded = x[..., -(length + width - 1):]
                qkv = ctx['seen']['linear_attn.in_proj_qkv']['out']
                self.emit('convstate_update', 'convstate',
                    {'mixed_qkv': qkv, 'old_conv_state': ctx['cache_before'][0],
                     'linear_update_tensor': torch.zeros(128, dtype=torch.int32)},
                    {'new_conv_state': ctx['cache_live'][0], 'padded': padded},
                    provenance='STATE_FROM_EXECUTED_UPDATE; PADDED_IS_VALID_SUFFIX_OF_EXECUTED_CONV_INPUT',
                    attrs={'kernel_size': width, 'auxiliary_policy': 'ORIGINAL_CPRO_EXPORT_ZERO_INT32_128'})
                weights = {'weight': weight}
                bias = args[2] if len(args) > 2 else kwargs.get('bias')
                if bias is not None:
                    weights['bias'] = bias
                self.emit('conv1d', 'conv', {'padded': padded}, {'conv': result[..., -length:]},
                    weights=weights, attrs={'groups': weight.shape[0]},
                    provenance='VALID_SUFFIX_OF_EXECUTED_CONV_OUTPUT')
                ctx['conv_seen'] = True
            return result
        self.patch(F, 'conv1d', conv)

        def silu(original, args, kwargs):
            ctx = self.active
            enabled = ctx is not None and ctx['mixer'] == 'linear_attn' and ctx.pop('conv_seen', False)
            x = cpu(args[0]) if enabled else None
            result = original(*args, **kwargs)
            if enabled:
                self.emit('conv_silu', 'silu', {'x': x}, {'y': result})
            return result
        self.patch(F, 'silu', silu)

        def gdn(original, args, kwargs):
            ctx = self.active
            if ctx is None or ctx['mixer'] != 'linear_attn':
                return original(*args, **kwargs)
            names = ('query', 'key', 'value', 'g', 'beta')
            values = dict(zip(names, args))
            values.update(kwargs)
            inputs = {name: cpu(values[name]) for name in names}
            # CPro passes the cache boundary dtype directly; NZ casts it before the kernel.
            inputs['initial_state'] = ctx['cache_before'][1]
            if values.get('effective_length') is None:
                raise ValueError('golden GDN call must supply effective_length')
            inputs['effective_length'] = cpu(values['effective_length'])
            linear = ctx['module'].linear_attn
            self.emit('decay_beta', 'gates',
                {'a': ctx['seen']['linear_attn.in_proj_a']['out'],
                 'b': ctx['seen']['linear_attn.in_proj_b']['out']},
                {'g': inputs['g'], 'beta': inputs['beta']},
                weights={'A_log': linear.A_log, 'dt_bias': linear.dt_bias})
            ratio = linear.num_v_heads // linear.num_k_heads
            if ratio > 1:
                self.emit('repeat_qk', 'repeat_qk',
                    {'q': inputs['query'][:, :, ::ratio, :].contiguous(),
                     'k': inputs['key'][:, :, ::ratio, :].contiguous()},
                    {'q_repeated': inputs['query'], 'k_repeated': inputs['key']},
                    attrs={'ratio': ratio}, provenance='INPUTS_ARE_HEAD_SLICES_OF_EXECUTED_GDN_INPUTS')
            result = original(*args, **kwargs)
            self.emit('gdn', 'gdn', inputs, {'core': result[0], 'state_raw': result[1]},
                attrs={k: values.get(k, default) for k, default in (
                    ('chunk_size', 64), ('output_final_state', True), ('use_qk_l2norm_in_kernel', True))})
            ctx['gdn_state'] = cpu(result[1])
            return result
        self.patch(npu, 'npu_chunk_gated_delta_rule', gdn)

        def cache(original, args, kwargs):
            ctx = self.active
            if ctx is None or ctx['mixer'] != 'self_attn':
                return original(*args, **kwargs)
            names = ('cache', 'updates', 'target_block', 'offset_in_block')
            if len(args) != 4 or kwargs:
                raise ValueError('expected original positional CacheUpdate call')
            inputs = dict(zip(names, cpu(args)))
            result = original(*args, **kwargs)
            label = 'cache_k' if ctx['cache_ordinal'] == 0 else 'cache_v'
            self.emit(label, 'cache', inputs, {'updated_cache': args[0]})
            ctx['cache_ordinal'] += 1
            return result
        self.patch(npu, 'npu_cache_update_', cache)

        def attention(original, args, kwargs):
            ctx = self.active
            if ctx is None or ctx['mixer'] != 'self_attn':
                return original(*args, **kwargs)
            if args:
                raise ValueError('expected original keyword fused-attention call')
            inputs = {k: cpu(kwargs[k]) for k in ('query', 'atten_mask', 'block_table')}
            inputs.update(key=cpu(kwargs['key'][0]), value=cpu(kwargs['value'][0]))
            inputs['all_seq_lengths_q'] = torch.as_tensor(kwargs['all_seq_lengths_q'], dtype=torch.int64).cpu()
            attrs = {k: v for k, v in kwargs.items() if k not in ('query', 'key', 'value', 'atten_mask', 'block_table', 'all_seq_lengths_q')}
            result = original(*args, **kwargs)
            self.emit('fused_attention', 'attention', inputs, {'attention_nz': result}, attrs=attrs)
            ctx['attention_raw'] = cpu(result)
            return result
        self.patch(npu, 'adn_fused_infer_attention', attention)

        def rope(original, args, kwargs):
            enabled = self.active is not None and self.active['mixer'] == 'self_attn'
            inputs = dict(zip(('q', 'k', 'cos', 'sin'), cpu(args[:4]))) if enabled else None
            result = original(*args, **kwargs)
            if enabled:
                self.emit('rope', 'rope', inputs, {'q_rotated': result[0], 'k_rotated': result[1]})
            return result
        self.patch(nz_module, 'apply_rotary_pos_emb', rope)

    def close(self, error=None):
        for handle in self.handles:
            handle.remove()
        for obj, name, original in reversed(self.patches):
            setattr(obj, name, original)
        self.active = None
        self.report['status'] = 'FAILED' if error else 'CAPTURED'
        if error:
            self.report['errors'].append(str(error))
        observed = {r['layer'] for r in self.report['regions']}
        missing = self.layers - observed
        if missing:
            if not error:
                self.report['status'] = 'INCOMPLETE'
            self.report['errors'].append('unobserved requested layers: ' + str(sorted(missing)))
        for layer in self.layers:
            rows = [r for r in self.report['regions'] if r['layer'] == layer]
            calls = {r['call'] for r in rows}
            if calls != set(range(self.calls)):
                self.report['errors'].append(f'layer {layer}: captured calls {sorted(calls)}, requested {self.calls}')
            for call in calls:
                points = {r['point'] for r in rows if r['call'] == call}
                if 'linear_attn_in_proj_qkv' in points:
                    required = {'convstate_update', 'conv1d', 'conv_silu', 'gdn', 'recurrent_writeback', 'decay_beta'}
                else:
                    required = {'cache_k', 'cache_v', 'fused_attention', 'rope', 'attention_layout_gate'}
                if required - points:
                    self.report['errors'].append(f'layer {layer} call {call}: missing CPro boundaries {sorted(required-points)}')
        if self.report['errors'] and not error:
            self.report['status'] = 'INCOMPLETE'
        self.flush()


def layer_module(wrapper, index, start):
    return wrapper.model.model.layers[index-start]
