"""Observe L3 V inside the original Prefill, then compare the same call on CPU."""
import json
from pathlib import Path
from threading import get_ident

from .common import checked, identity, metrics, read_array, save_json, sha
from .qlinear import arguments, describe, snapshot
from .qlinear_backend import runtime, tensor_formats
from .qlinear_golden import _array_identity, _audit, _replay, load_capture, tensor_spec
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

    def replay_captured_hidden(self, module, capture, input_ids=None, token_arg='input_ids'):
        import torch

        root, _, (region,) = load_capture(capture, ['self_attn_v_proj'])
        spec = tensor_spec(region, 'inputs', 'x')
        hidden = torch.from_numpy(read_array(root, spec))
        tensors = dict(module.named_parameters())
        tensors.update(module.named_buffers())
        tensors.update((k, v) for k, v in vars(module).items() if isinstance(v, torch.Tensor))
        weight = getattr(module, 'W_q', None)
        if isinstance(weight, torch.Tensor):
            tensors['W_q'] = weight
        self.report.update(module_tensors=describe(tensors), module_formats=tensor_formats(self.npu, tensors),
                           W_q_type=type(weight).__name__ if hasattr(module, 'W_q') else 'MISSING',
                           prefill_input_ids=describe(input_ids),
                           input_source='CAPTURED_LOGICAL_HIDDEN', input_tensor=spec)
        if (hidden.dtype != torch.float16 or hidden.ndim != 3 or hidden.shape[-1] < 1
                or hidden.shape[:2] != (1, region['sequence_length'])):
            raise ValueError('capture hidden dtype/shape does not match the L3 Prefill input')
        npu_types = ('npu', 'privateuseone')
        if (isinstance(input_ids, torch.Tensor) and input_ids.device.type in npu_types
                and (token_arg == 'input_ids' or input_ids.dtype in (torch.int16, torch.int32, torch.int64))):
            device = input_ids.device
            source = f'prefill.{token_arg}'
        else:
            devices = {t.device for t in tensors.values() if t.device.type in npu_types}
            if len(devices) != 1:
                raise ValueError(f'cannot identify L3 replay NPU from Prefill input_ids or module tensors: {devices}')
            device, = devices
            source = 'module_tensors'
        self.report.update(replay_device=str(device), replay_device_source=source)
        forward = getattr(module.forward, '__code__', None)
        if forward is not None:
            self.report['module_forward'] = dict(path=forward.co_filename, line=forward.co_firstlineno)
        save_json(self.out / 'live.json', self.report)
        print(f'[target-trace] replay {list(hidden.shape)} on {device} from {source}; '
              f'module tensors={json.dumps(self.report["module_tensors"])}', flush=True)
        with torch.no_grad():
            device_hidden = hidden.to(device)
            try:
                module(device_hidden)
            except Exception as exc:
                self.report['module_forward_error'] = f'{type(exc).__name__}: {exc}'
                raise

    def before(self, module, args, kwargs):
        import torch

        hidden = args[0] if args else kwargs.get('x')
        # Retain the module boundary even when this backend bypasses the Python matmul hook.
        try:
            torch.save(snapshot(hidden), self.out / 'module_x.pt')
            self.report['module_x'] = identity(self.out / 'module_x.pt', self.out)
        except Exception as exc:
            self.report['input_snapshot_error'] = f'{type(exc).__name__}: {exc}'
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
            torch.save(snapshot(result), self.out / 'module_y.pt')
            status = 'CAPTURED' if self.calls == 1 and 'call' in self.report else 'MODULE_ONLY'
            self.report.update(status=status, module_y=identity(self.out / 'module_y.pt', self.out))
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


def _layout_report(live, manifest, source_call, tensors, checkpoint_y, historical_y):
    """Compare two fixed W4 interpretations using the saved call operands."""
    from .qlinear_matmul_probe import nd_weight_hypothesis

    q, weight, scales, token_scale, live_y = (
        tensors[key].numpy() for key in ('x1', 'x2', 'scale', 'pertoken_scale', 'matmul_y'))
    nd_y, _ = group_matmul(q, token_scale, nd_weight_hypothesis(weight), scales.T[:, :, None])
    comparisons = []
    for layout, predicted in [('checkpoint_i4zn', checkpoint_y), ('nd_output_channel_nibbles', nd_y)]:
        for pair, actual, reference in [('live_vs_prediction', live_y, predicted),
                                        ('prediction_vs_capture', predicted, historical_y)]:
            score = metrics(actual, reference)
            comparisons.append(dict(layout=layout, pair=pair, relative_l2=score['relative_l2'],
                                    cosine=score['cosine'], max_abs=score['max_abs'], passed=score['passed']))

    def call_info(inputs, formats, kwargs, runtime):
        runtime = runtime or {}
        return dict(
            operands={key: dict(dtype=desc.get('dtype'), shape=desc.get('shape'),
                                stride=desc.get('stride'), format=(formats or {}).get(key, {}).get('format'))
                      for key, desc in inputs.items()},
            kwargs={k: v for k, v in kwargs.items() if k != 'pertoken_scale'},
            python=runtime.get('python'), torch=runtime.get('torch_version'),
            torch_npu=runtime.get('torch_npu_version'), torch_npu_file=runtime.get('torch_npu_file'),
            custom_opp_path=runtime.get('environment', {}).get('ASCEND_CUSTOM_OPP_PATH'),
            mapped_opapi=[p for p in runtime.get('loaded_libraries', [])
                          if Path(p).name.startswith(('libcust_opapi.so', 'libopapi.so'))])

    old_kwargs = source_call['original_kwargs']
    return dict(
        comparisons=comparisons,
        capture=call_info(dict(source_call['inputs'], pertoken_scale=old_kwargs['pertoken_scale']),
                          source_call.get('npu_formats'), old_kwargs, manifest['quant_matmul'].get('runtime')),
        live=call_info(live['inputs'], live.get('formats'), live['kwargs'], live.get('runtime_after')))


def compare_live(out, capture, cpu_report, *, inspect_layouts=False):
    import torch
    import numpy as np

    root, manifest, (region,) = load_capture(capture, ['self_attn_v_proj'])
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

    tensors = load_tensor(live['call']) if live['status'] == 'CAPTURED' else {}
    module_y = load_tensor(live['module_y']).numpy()
    hidden = (load_tensor(live['module_x']) if 'module_x' in live else tensors['hidden']).numpy()
    history_x = read_array(root, tensor_spec(region, 'inputs', 'x'))
    history_y = read_array(root, tensor_spec(region, 'outputs', 'golden_y'))

    def same(a, b):
        return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()

    def operand_matches(tensor, array):
        return (isinstance(tensor, torch.Tensor) and tensor.dtype == torch.from_numpy(array).dtype
                and tuple(tensor.shape) == array.shape and same(tensor.numpy(), array))

    if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[1] <= 1:
        raise ValueError('need a batch-one Prefill activation')
    m, k = hidden.shape[1:]
    n = history_y.shape[-1]
    if module_y.shape != (1, m, n):
        raise ValueError(f'unexpected L3 V output shape {module_y.shape}, expected {(1, m, n)}')
    result = dict(live_hidden_shape=list(hidden.shape), capture_hidden_shape=list(history_x.shape),
                  hidden_equal_bits=same(hidden, history_x))
    if result['hidden_equal_bits'] and history_y.shape == module_y.shape:
        result['module_vs_four_layer'] = metrics(module_y, history_y)
        result['module_vs_four_layer']['equal_bits'] = same(module_y, history_y)
        cpu_tensor = case.get('cpu_reference', {}).get('tensor')
        if cpu_tensor:
            result['module_vs_cpu_reference'] = metrics(module_y, read_array(cpu_report, cpu_tensor))
    if not tensors:
        result['same_call_cpu_skipped'] = (live.get('observation_error') or
            f"observed {live.get('calls', 0)} torch_npu.npu_quant_matmul calls; need one complete call")
        return result
    q, ps, y = (tensors.get(name) for name in ('x1', 'pertoken_scale', 'matmul_y'))
    result['module_equals_matmul'] = operand_matches(y, module_y[0])
    weights = OriginalQuantWeights(Path(cpu['quanted_pth']))
    raw, scales = weights.logical('target.layers.3.self_attn.v_proj.weight', (n, k))
    if _array_identity(raw) != binding['raw_weight'] or _array_identity(scales) != binding['raw_scale']:
        raise ValueError('reference checkpoint contents changed')
    result['weight_equal_bits'] = operand_matches(tensors.get('x2'), pack_i4zn(raw))
    reference_scales = scales[:, :, 0].astype(np.float32).T
    result['scale_equal_bits'] = operand_matches(tensors.get('scale'), reference_scales)
    if result['hidden_equal_bits'] and history_y.shape == module_y.shape:
        replay = cpu.get('dynamic_quant_replay')
        if replay:
            try:
                replay_root, _, rows = _replay(replay['root'], cpu['capture_sha256'], [region])
                checked(replay_root, replay)
                dq = rows[region['id']]
                aq = read_array(replay_root, dq['q'])
                token_scale = read_array(replay_root, dq['token_scale'])
                result['a8_equals_replay'] = operand_matches(q, aq)
                result['token_scale_equals_replay'] = operand_matches(ps, token_scale)
            except (ValueError, KeyError, OSError) as exc:
                result['quant_replay_unavailable'] = str(exc)
    kwargs = live['kwargs']
    audited = (live['positional_count'] == 3 and kwargs.get('output_dtype') == 'torch.float16'
               and kwargs.get('group_sizes') in (None, [0, 0, 256])
               and all(v is None for key, v in kwargs.items()
                       if key not in ('output_dtype', 'group_sizes', 'pertoken_scale')))
    if (result['weight_equal_bits'] and result['scale_equal_bits'] and audited
            and isinstance(q, torch.Tensor) and q.dtype == torch.int8 and q.shape == (m, k)
            and isinstance(ps, torch.Tensor) and ps.dtype == torch.float32 and ps.shape == (m, 1)
            and isinstance(y, torch.Tensor) and y.dtype == torch.float16 and y.shape == (m, n)):
        expected, _ = group_matmul(q.numpy(), ps.numpy(), raw, scales)
        result['matmul_vs_same_call_cpu'] = metrics(y.numpy(), expected)
        if (inspect_layouts and result['hidden_equal_bits'] and result['module_equals_matmul']
                and same(tensors['hidden'].numpy(), history_x) and history_y.shape == module_y.shape):
            source_call = _audit(manifest, region, m, k, n)
            result['layout_report'] = _layout_report(live, manifest, source_call, tensors, expected, history_y[0])
    else:
        result['same_call_cpu_skipped'] = 'weight bytes, scale, shape or call signature differs from group256 reference'
    return result
