"""Replay the original FullAttention module with only its attention math replaced.

Original observations are never overwritten. QLinear, norms, RoPE, cache update,
gate and output projection still use the original module and original backend;
this is an independent attention-core reference, not an all-CPU module reference.
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import numpy as np
import torch

from .attention_reference import reference_math
from .capture import cpu
from .common import identity, metrics


POLICY = 'ORIGINAL_FULL_ATTENTION_MODULE_WITH_INDEPENDENT_ATTENTION_MATH'
OUTPUTS = ('golden_hidden', 'golden_key_cache', 'golden_value_cache')


def _tree(value, tensor_fn, path=()):
    if isinstance(value, torch.Tensor):
        return tensor_fn(value, path)
    if isinstance(value, (tuple, list)):
        return type(value)(_tree(v, tensor_fn, path+(i,)) for i, v in enumerate(value))
    if isinstance(value, dict):
        return {k: _tree(v, tensor_fn, path+(k,)) for k, v in value.items()}
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise ValueError(f'unsupported source argument at {path}: {type(value).__name__}')


def _outputs(result):
    if (not isinstance(result, (tuple, list)) or len(result) != 3 or result[1] is not None
            or not isinstance(result[2], (tuple, list)) or len(result[2]) != 2):
        raise ValueError('requires original (hidden, None, (key_cache, value_cache)) output')
    values = (result[0], result[2][0], result[2][1])
    if any(not isinstance(v, torch.Tensor) for v in values):
        raise ValueError('FullAttention outputs must be three tensors')
    return values


def _length(value):
    if isinstance(value, torch.Tensor):
        if value.dtype != torch.int64 or tuple(value.shape) != (1,):
            raise ValueError('allQLen tensor must be INT64[1]')
        value = value.tolist()
    if not isinstance(value, (tuple, list)) or len(value) != 1 or type(value[0]) is not int:
        raise ValueError('requires one integer allQLen')
    return value[0]


class FullAttentionReplay:
    def __init__(self, root, record, npu, layer=3):
        self.root, self.record, self.npu, self.layer = Path(root), record, npu, layer
        self.handles, self.devices, self.modes = [], {}, {}
        self.module = self.arguments = self.actual = self.error = None
        self.pre_calls = self.post_calls = 0
        self.closed = self.done = False

    def attach(self, wrapper):
        if self.module is not None or self.closed or self.layer != 3:
            raise ValueError('only one L3 FullAttention attachment is supported')
        self.module = wrapper.model.model.layers[self.layer].self_attn
        self.signature = inspect.signature(self.module.forward)
        self.handles = [self.module.register_forward_pre_hook(self._before, with_kwargs=True),
                        self.module.register_forward_hook(self._after, with_kwargs=True)]
        return self

    def _before(self, module, args, kwargs):
        self.pre_calls += 1
        try:
            if self.pre_calls != 1:
                raise ValueError('requires exactly one original Prefill call')
            bound = self.signature.bind(*args, **kwargs)
            bound.apply_defaults()
            self.modes = {name: child.training for name, child in module.named_modules()}

            def snapshot(value, path):
                self.devices[path] = value.device
                return cpu(value)

            self.arguments = _tree(dict(bound.arguments), snapshot)
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'

    def _after(self, module, args, kwargs, result):
        self.post_calls += 1
        try:
            if self.post_calls != 1:
                raise ValueError('requires exactly one original Prefill result')
            self.actual = cpu(_outputs(result))
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.closed = True

    def _validate(self):
        if not self.closed or self.pre_calls != 1 or self.post_calls != 1 or self.error:
            raise ValueError(self.error or 'close hooks after exactly one complete original call before replay')
        module = self.module
        if any(m._forward_pre_hooks or m._forward_hooks for m in module.modules()):
            raise ValueError('replay requires all observation hooks removed')
        if module.training or self.modes != {name: child.training for name, child in module.named_modules()}:
            raise ValueError('replay requires the same captured module modes and an eval FullAttention module')
        if module.forward.__globals__.get('torch_npu') is not self.npu:
            raise ValueError('source module does not call the supplied torch_npu namespace')
        values = self.arguments
        expected = {'hidden_states', 'attention_mask', 'position_ids', 'past_key_values',
                    'new_kv_cache_pos', 'output_attentions', 'use_cache', 'layer_idx',
                    'allQLen', 'token_count', 'export_flag', 'kwargs'}
        if set(values) != expected or values['kwargs']:
            raise ValueError('FullAttention forward arguments differ from the reviewed source')
        if (values['export_flag'] is not False or values['output_attentions'] is not False
                or values['use_cache'] is not False or values['layer_idx'] is not None
                or values['token_count'] != 0):
            raise ValueError('FullAttention flags differ from the original Prefill defaults')
        hidden, mask = values['hidden_states'], values['attention_mask']
        length = _length(values['allQLen'])
        if (not isinstance(hidden, torch.Tensor) or hidden.ndim != 3 or hidden.shape[0] != 1
                or hidden.shape[1] != length or length <= 1 or not torch.isfinite(hidden).all()):
            raise ValueError('requires one finite B1 Prefill hidden sequence with T > 1')
        if (not isinstance(mask, torch.Tensor) or tuple(mask.shape[:3]) != (1, 1, length)
                or mask.ndim != 4 or mask.shape[-1] < length):
            raise ValueError('requires captured additive mask [1,1,T,KV_capacity]')
        for name, expected_pos in (('position_ids', torch.arange(length)[None]),
                                    ('new_kv_cache_pos', torch.arange(length)[:, None])):
            value = values[name]
            if (not isinstance(value, torch.Tensor) or value.dtype != torch.int64
                    or value.shape != expected_pos.shape or not torch.equal(value, expected_pos)):
                raise ValueError(f'{name} must describe fresh Prefill positions 0..T-1 as INT64')
        caches = values['past_key_values']
        if (not isinstance(caches, (tuple, list)) or len(caches) != 2
                or any(not isinstance(v, torch.Tensor) or v.dtype != torch.float16
                       or v.ndim != 4 or torch.count_nonzero(v).item() for v in caches)):
            raise ValueError('requires two independent snapshots of zero FP16 initial paged caches')
        return length

    def replay(self):
        if self.done:
            raise ValueError('FullAttention replay may only run once')
        self.done = True
        row = dict(layer=self.layer, call=0, phase='prefill', status='UNAVAILABLE', policy=POLICY,
            scope='COMPLETE_SELF_ATTN_MODULE; NOT_DECODER_LAYER_OR_ALL_CPU_REFERENCE',
            input_policy='CPU_SNAPSHOT_BEFORE_ORIGINAL_STATE_UPDATE; INDEPENDENT_REPLAY_CLONES',
            original_golden='UNCHANGED', outputs=[])
        self.record.setdefault('attention_math_replays', []).append(row)
        try:
            length = self._validate()
            row['source_module_training_flags'] = self.modes.copy()
            row['source_forward_arguments'] = _tree(self.arguments, lambda v, p: dict(
                shape=list(v.shape), dtype=str(v.dtype).removeprefix('torch.'), device=str(self.devices[p]),
                sha256=hashlib.sha256(v.numpy().tobytes()).hexdigest()))
            cloned = _tree(self.arguments, lambda v, p: v.to(self.devices[p], copy=True).contiguous())
            bound = inspect.BoundArguments(self.signature, cloned)
            original = getattr(self.npu, 'adn_fused_infer_attention', None)
            if not callable(original):
                raise ValueError('original attention namespace is unavailable')
            calls = []

            def independent(*args, **kwargs):
                if args or calls:
                    raise ValueError('requires exactly one keyword-only attention-core call')
                calls.append(None)
                allowed = {'query', 'key', 'value', 'actual_seq_lengths_q', 'actual_seq_lengths_kv',
                    'block_table', 'num_heads', 'num_key_value_heads', 'block_size', 'input_layout',
                    'scale_value', 'inner_precise', 'atten_mask', 'all_seq_lengths_q'}
                if set(kwargs) != allowed or kwargs['input_layout'] != 'BNSD' or kwargs['inner_precise'] != 2:
                    raise ValueError('attention core arguments differ from the reviewed source contract')
                values = cpu(kwargs)
                query, mask = values['query'], values['atten_mask']
                if (query.ndim != 4 or mask.ndim != 4 or query.shape[2] != length
                        or _length(values['all_seq_lengths_q']) != length
                        or values['actual_seq_lengths_q'] != [length]
                        or values['actual_seq_lengths_kv'] != [mask.shape[-1]]):
                    raise ValueError('attention lengths differ from this fresh Prefill invocation')
                for name in ('key', 'value'):
                    if not isinstance(values[name], (tuple, list)) or len(values[name]) != 1:
                        raise ValueError(f'{name} must contain one paged cache tensor')
                prepared = reference_math(query.numpy(), values['key'][0].numpy(), values['value'][0].numpy(),
                    mask.numpy(), values['block_table'].numpy(), length, heads=values['num_heads'],
                    kv_heads=values['num_key_value_heads'], block_size=values['block_size'], scale=values['scale_value'])
                row['attention_reference'] = prepared['evidence']
                logical = prepared['output']
                if logical is None:
                    raise ValueError(prepared['evidence']['status'])
                b, heads, seq, dim = logical.shape
                packed = logical.reshape(b, heads, seq, dim//16, 16).transpose(0, 1, 3, 2, 4)
                packed = np.ascontiguousarray(packed.reshape(query.shape))
                return torch.from_numpy(packed).to(kwargs['query'].device, copy=True)

            self.npu.adn_fused_infer_attention = independent
            try:
                with torch.no_grad():
                    result = cpu(_outputs(self.module(*bound.args, **bound.kwargs)))
            finally:
                self.npu.adn_fused_infer_attention = original
            if len(calls) != 1:
                raise ValueError('source module did not invoke exactly one independent attention core')
            for name, actual, reference in zip(OUTPUTS, self.actual, result):
                if reference.shape != actual.shape or reference.dtype != actual.dtype or not torch.isfinite(reference).all():
                    raise ValueError(f'independent module result shape/dtype/nonfinite mismatch: {name}')
            row['cache_replay_bitwise_equal'] = {
                name: actual.numpy().tobytes() == reference.numpy().tobytes()
                for name, actual, reference in zip(OUTPUTS[1:], self.actual[1:], result[1:])}
            if not all(row['cache_replay_bitwise_equal'].values()):
                raise ValueError('replayed K/V differ from original K/V; attention-only substitution is not isolated')
            folder = self.root/'attention-math-replay'/f'l{self.layer:02d}_c00'
            folder.mkdir(parents=True, exist_ok=False)
            for name, actual, reference in zip(OUTPUTS, self.actual, result):
                path = folder/(name+'.bin')
                with path.open('xb') as stream:
                    stream.write(reference.numpy().tobytes())
                row['outputs'].append(dict(name=name, shape=list(reference.shape),
                    dtype=str(reference.dtype).removeprefix('torch.'), **identity(path, self.root)))
                row.setdefault('captured_vs_reference', {})[name] = metrics(actual.numpy(), reference.numpy())
            row['status'] = 'COMPUTED'
        except Exception as exc:
            row['error'] = f'{type(exc).__name__}: {exc}'
        return row
