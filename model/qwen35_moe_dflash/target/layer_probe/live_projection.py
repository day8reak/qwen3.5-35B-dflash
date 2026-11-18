"""Observe L3 V inside the original Prefill, then compare the same call on CPU."""
import json
from pathlib import Path
from threading import get_ident

from .common import checked, identity, metrics, read_array, save_json, sha
from .qlinear import arguments, describe, snapshot
from .qlinear_backend import runtime, tensor_formats
from .qlinear_golden import _array_identity, _replay, load_capture, tensor_spec
from .qlinear_math import group_matmul
from ..original_quant import OriginalQuantWeights, pack_i4zn


class LiveProjection:
    def __init__(self, out):
        self.out = out
        self.handles = []
        self.original = None
        self.calls = 0
        self.report = {'status': 'WAITING_FOR_L3_V'}

    def attach(self, frame):
        import torch
        import torch_npu

        code = frame.f_code
        params = code.co_varnames[:code.co_argcount + code.co_kwonlyargcount]
        models = [frame.f_locals[key] for key in params if isinstance(frame.f_locals[key], torch.nn.Module)]
        matches = [(model, name, module) for model in models for name, module in model.named_modules()
                   if name == 'layers.3.self_attn.v_proj' or name.endswith('.layers.3.self_attn.v_proj')]
        if len(matches) != 1:
            raise ValueError(f'expected one L3 self_attn.v_proj in Prefill arguments, found {len(matches)}')
        model, name, module = matches[0]
        owner = model.get_submodule(name.removesuffix('layers.3.self_attn.v_proj').rstrip('.'))
        offset = getattr(owner, 'start_layer_idx', 0)
        if offset != 0:
            raise ValueError(f'local layer 3 starts at a nonzero model offset {offset}')
        self.npu = torch_npu
        self.report.update(module=name, module_class=f'{type(module).__module__}.{type(module).__qualname__}',
                           model_layer_count=len(owner.layers), layer_offset=offset,
                           entry=str(Path(code.co_filename).resolve()), entry_sha256=sha(code.co_filename))
        self.handles.append(module.register_forward_pre_hook(self.before, with_kwargs=True))
        self.handles.append(module.register_forward_hook(self.after))
        return module

    def replay_captured_hidden(self, module, capture):
        import torch

        root, _, (region,) = load_capture(capture, ['self_attn_v_proj'])
        spec = tensor_spec(region, 'inputs', 'x')
        hidden = torch.from_numpy(read_array(root, spec))
        weight = getattr(module, 'W_q', None)
        if (not isinstance(weight, torch.Tensor) or weight.ndim != 2
                or weight.device.type != 'npu' or weight.dtype != torch.int32):
            raise ValueError('initialized L3 V requires the original packed INT32 W_q on NPU')
        if hidden.dtype != torch.float16 or tuple(hidden.shape) != (1, region['sequence_length'], weight.shape[0]):
            raise ValueError('capture hidden dtype/shape does not match the L3 Prefill input')
        self.report.update(input_source='CAPTURED_LOGICAL_HIDDEN', input_tensor=spec)
        with torch.no_grad():
            module(hidden.to(weight.device))

    def before(self, module, args, kwargs):
        import torch

        hidden = args[0] if args else kwargs.get('x')
        self.report['runtime_before'] = runtime(self.npu)
        original = getattr(self.npu, 'npu_quant_matmul', None)
        if not callable(original):
            self.report['observation_error'] = 'torch_npu.npu_quant_matmul is unavailable'
            return
        self.original = original
        self.report['python_operator'] = dict(module=getattr(original, '__module__', None),
                                              name=getattr(original, '__qualname__', None))
        owner = get_ident()

        def observe(*op_args, **op_kwargs):
            if get_ident() != owner:
                return original(*op_args, **op_kwargs)
            self.calls += 1
            operands = dict(arguments(op_args, op_kwargs), pertoken_scale=op_kwargs.get('pertoken_scale'))
            self.report.update(inputs=describe(operands), kwargs=describe(op_kwargs), positional_count=len(op_args))
            # Pass exactly the live objects through the original entrypoint.
            try:
                result = original(*op_args, **op_kwargs)
            except Exception as exc:
                self.report.update(status='CALL_FAILED', op_error=f'{type(exc).__name__}: {exc}',
                                   formats=tensor_formats(self.npu, operands), runtime_after=runtime(self.npu))
                self.close()
                raise
            try:
                if self.calls != 1:
                    return result
                self.report.update(formats=tensor_formats(self.npu, operands), runtime_after=runtime(self.npu))
                torch.save(snapshot(dict(operands, hidden=hidden, matmul_y=result)), self.out / 'call.pt')
                self.report['call'] = identity(self.out / 'call.pt', self.out)
            except Exception as exc:
                self.report['observation_error'] = f'{type(exc).__name__}: {exc}'
            return result

        self.npu.npu_quant_matmul = observe

    def after(self, module, args, result):
        import torch

        self.close()
        self.report['calls'] = self.calls
        try:
            if self.calls != 1 or 'call' not in self.report:
                self.report['status'] = 'INCOMPLETE'
                return
            torch.save(snapshot(result), self.out / 'module_y.pt')
            self.report.update(status='CAPTURED', module_y=identity(self.out / 'module_y.pt', self.out))
            save_json(self.out / 'live.json', self.report)
        except Exception as exc:
            self.report.update(status='INCOMPLETE', observation_error=f'{type(exc).__name__}: {exc}')

    def close(self):
        if self.original is not None:
            self.npu.npu_quant_matmul = self.original
            self.original = None
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def compare_live(out, capture, cpu_report):
    import torch
    import numpy as np

    root, _, (region,) = load_capture(capture, ['self_attn_v_proj'])
    cpu = json.loads((cpu_report / 'report.json').read_text())
    if (cpu.get('scope') != 'SELECTED_QLINEAR_GOLDEN_MATH_ONLY'
            or cpu['capture_sha256'] != sha(root / 'manifest.json')):
        raise ValueError('CPU reference belongs to a different capture')
    case, = [c for c in cpu['cases'] if c['id'] == region['id'] and c['status'] == 'COMPUTED']
    binding = case['weight_binding']
    if binding['status'] != 'EXACT_RAW_TO_CAPTURE' or binding['capture'] != region['weights']:
        raise ValueError('CPU reference weight identity differs')
    live = json.loads((out / 'live.json').read_text())

    def load_tensor(spec):
        return torch.load(checked(out, spec), map_location='cpu', weights_only=True)

    tensors = load_tensor(live['call'])
    module_y = load_tensor(live['module_y']).numpy()
    hidden, q, w, s, ps, y = (tensors[key].numpy() for key in
                              ('hidden', 'x1', 'x2', 'scale', 'pertoken_scale', 'matmul_y'))
    history_x = read_array(root, tensor_spec(region, 'inputs', 'x'))
    history_y = read_array(root, tensor_spec(region, 'outputs', 'golden_y'))

    def same(a, b):
        return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()

    if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[1] <= 1:
        raise ValueError('need a batch-one Prefill activation')
    m, k = q.shape
    n = y.shape[-1]
    if (q.shape != hidden.shape[1:] or y.shape != (m, n) or y.dtype != np.float16
            or module_y.shape != (1, m, n) or s.shape != (k // 256, n)):
        raise ValueError('unexpected projection dimensions or output dtype')
    result = dict(live_hidden_shape=list(hidden.shape), capture_hidden_shape=list(history_x.shape),
                  hidden_equal_bits=same(hidden, history_x), module_equals_matmul=same(module_y[0], y))
    weights = OriginalQuantWeights(Path(cpu['quanted_pth']))
    raw, scales = weights.logical('target.layers.3.self_attn.v_proj.weight', (n, k))
    if _array_identity(raw) != binding['raw_weight'] or _array_identity(scales) != binding['raw_scale']:
        raise ValueError('reference checkpoint contents changed')
    result['weight_equal_bits'] = same(w, pack_i4zn(raw))
    reference_scales = scales[:, :, 0].astype(np.float32).T
    result['scale_equal_bits'] = same(s, reference_scales)
    if result['hidden_equal_bits'] and history_y.shape == module_y.shape:
        result['module_vs_four_layer'] = metrics(module_y, history_y)
        result['module_vs_four_layer']['equal_bits'] = same(module_y, history_y)
        replay = cpu.get('dynamic_quant_replay')
        if replay:
            try:
                replay_root, _, rows = _replay(replay['root'], cpu['capture_sha256'], [region])
                checked(replay_root, replay)
                dq = rows[region['id']]
                aq = read_array(replay_root, dq['q'])
                token_scale = read_array(replay_root, dq['token_scale'])
                result['a8_equals_replay'] = same(q, aq)
                result['token_scale_equals_replay'] = same(ps, token_scale)
            except (ValueError, KeyError, OSError) as exc:
                result['quant_replay_unavailable'] = str(exc)
    kwargs = live['kwargs']
    audited = (live['positional_count'] == 3 and kwargs.get('output_dtype') == 'torch.float16'
               and kwargs.get('group_sizes') in (None, [0, 0, 256])
               and all(v is None for key, v in kwargs.items()
                       if key not in ('output_dtype', 'group_sizes', 'pertoken_scale')))
    if result['weight_equal_bits'] and result['scale_equal_bits'] and audited:
        expected, _ = group_matmul(q, ps, raw, scales)
        result['matmul_vs_same_call_cpu'] = metrics(y, expected)
    else:
        result['same_call_cpu_skipped'] = 'weight bytes, scale or call signature differs from group256 reference'
    return result
