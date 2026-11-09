"""Audit the real QLinear call and retain a bounded failure reproducer."""
import json
from pathlib import Path

import torch

from .common import checked, identity, new_output, save_json
from .qlinear_backend import inventory, runtime, tensor_formats


def describe(value):
    if isinstance(value, torch.Tensor):
        return dict(shape=list(value.shape), dtype=str(value.dtype), device=str(value.device),
                    stride=list(value.stride()), contiguous=value.is_contiguous())
    if isinstance(value, dict):
        return {k: describe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [describe(v) for v in value]
    return value if value is None or isinstance(value, (str, int, float, bool)) else str(value)


def snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to('cpu', copy=True)
    if isinstance(value, dict):
        return {k: snapshot(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(snapshot(v) for v in value)
    return value


def arguments(args, kwargs):
    names = ('x1', 'x2', 'scale')
    return {name: args[i] if len(args) > i else kwargs.get(name) for i, name in enumerate(names)}


def contract(args, kwargs):
    tensors = arguments(args, kwargs)
    x, w, s = (tensors[k] for k in ('x1', 'x2', 'scale'))
    errors = []
    result = dict(expected_group_size=256, errors=errors)
    if not all(isinstance(t, torch.Tensor) and t.ndim == 2 for t in (x, w, s)):
        errors.append('expected 2D x1, packed x2 and group scale')
        return result
    m, k = x.shape
    n = w.shape[1] * 8
    groups = s.shape[0]
    result.update(M=m, K=k, N=n, scale_groups=groups,
        inferred_weight_group_size=k // groups if groups and k % groups == 0 else None,
        expected_weight_shape=[k, n // 8], expected_scale_shape=[k // 256, n])
    if x.dtype != torch.int8 or w.dtype != torch.int32:
        errors.append('expected INT8 activations and INT32 packed W4 carrier')
    if w.shape[0] != k or k % 256 or tuple(s.shape) != (k // 256, n):
        errors.append('weight/scale shapes do not describe K-axis group256; no reshape or requantization allowed')
    if not x.is_contiguous() or not w.is_contiguous() or not s.is_contiguous():
        errors.append('this original I4ZN carrier path requires contiguous tensors')
    ps = kwargs.get('pertoken_scale')
    if not isinstance(ps, torch.Tensor) or tuple(ps.shape) != (m, 1):
        errors.append('original pertoken_scale must have shape [M,1]')
    # Scale dtype is recorded, not converted: a vendor ABI mismatch must remain visible.
    return result


def explicit_group256(args, kwargs):
    check = contract(args, kwargs)
    if check['errors']:
        raise ValueError('; '.join(check['errors']))
    groups = kwargs.get('group_sizes')
    if groups is not None and list(groups) != [0, 0, 256]:
        raise ValueError(f'original group_sizes={groups} conflicts with explicit group256; refusing override')
    return dict(kwargs, group_sizes=[0, 0, 256])


class QuantMatmulAudit:
    def __init__(self, capture, npu, group_size=None):
        if group_size not in (None, 256):
            raise ValueError('only explicit group256 is supported; weights are never requantized')
        self.capture, self.npu, self.group_size = capture, npu, group_size
        self.module, self.names, self.calls = None, {}, {}
        self.record = dict(policy='ORIGINAL_CALL' if group_size is None else 'EXPLICIT_GROUP256',
            runtime=runtime(npu), calls=[], failed_call=None,
            baseline_equivalence='NOT_VERIFIED' if group_size else 'ORIGINAL_CALL_UNCHANGED')
        capture.report['quant_matmul'] = self.record
        if group_size:
            capture.report['reference'] = '310P_EXPLICIT_QMATMUL_GROUP256_BASELINE_UNVERIFIED'

    def attach(self, wrapper):
        self.names.update({id(m): name for name, m in wrapper.named_modules()})

    def install(self, cls):
        def forward(original, args, kwargs):
            previous = self.module
            self.module = args[0]
            try:
                return original(*args, **kwargs)
            finally:
                self.module = previous
        def matmul(original, args, kwargs):
            if self.module is None:
                return original(*args, **kwargs)
            name = self.names.get(id(self.module), type(self.module).__qualname__)
            count = self.calls.get(name, 0)
            self.calls[name] = count + 1
            row = dict(module=name, call=count, inputs=describe(arguments(args, kwargs)),
                original_kwargs=describe(kwargs), contract=contract(args, kwargs), status='CALLING')
            row['npu_formats'] = tensor_formats(self.npu, dict(arguments(args, kwargs), **kwargs))
            retained = count < self.capture.calls and (self.capture.active is not None or name.endswith('lm_head'))
            if retained:
                self.record['calls'].append(row)
            # Persist before dispatch: parameter validation can fail before any output hook.
            self.write()
            effective = kwargs
            try:
                if self.group_size is not None:
                    effective = explicit_group256(args, kwargs)
                row['effective_kwargs'] = describe(effective)
                result = original(*args, **effective)
                row['status'] = 'EXECUTED'
                return result
            except Exception as exc:
                row.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
                self.record['failed_call'] = row
                if not retained:
                    self.record['calls'].append(row)
                # Exact real tensors, one failing matmul only; no synthetic golden.
                try:
                    path = self.capture.root/'qlinear-failure.pt'
                    torch.save(dict(args=snapshot(args), kwargs=snapshot(kwargs)), path)
                    row['reproducer'] = identity(path, self.capture.root)
                except Exception as dump_error:
                    row['dump_error'] = f'{type(dump_error).__name__}: {dump_error}'
                try:
                    self.record['runtime_at_failure'] = runtime(self.npu)
                    self.record['backend_inventory'] = inventory(self.record['runtime_at_failure'])
                except Exception as diagnostic_error:
                    self.record['backend_diagnostic_error'] = str(diagnostic_error)
                print('[target-probe] QLinear failure: ' + json.dumps(row, ensure_ascii=False), flush=True)
                raise
            finally:
                self.write()
        self.capture.patch(cls, 'forward', forward)
        self.capture.patch(self.npu, 'npu_quant_matmul', matmul)

    def write(self):
        save_json(self.capture.root/'qlinear-audit.json', self.record)


def replay_failure(capture_dir, output, device='npu:0', group_size=None, *, npu=None):
    root = new_output(output)
    capture_dir = Path(capture_dir).resolve()
    result = dict(status='STARTING', stage='LOAD_FIXTURE',
                  scope='SINGLE_FAILED_MATMUL; NOT_GOLDEN_OR_ACCURACY_VALIDATION')
    try:
        audit = json.loads((capture_dir/'qlinear-audit.json').read_text())
        failed = audit['failed_call']
        payload = torch.load(checked(capture_dir, failed['reproducer']), map_location='cpu', weights_only=True)
        result['reproducer'] = failed['reproducer']
        result['captured_npu_formats'] = failed.get('npu_formats', 'NOT_RECORDED_BY_OLDER_CAPTURE')
        def move(value):
            if isinstance(value, torch.Tensor):
                return value.to(device)
            if isinstance(value, dict):
                return {k: move(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return type(value)(move(v) for v in value)
            return value
        if npu is None:
            import torch_npu as npu
        result['stage'] = 'PREPARE_INPUTS'
        args, kwargs = move(payload['args']), move(payload['kwargs'])
        result.update(module=failed['module'], runtime=runtime(npu), inputs=describe(arguments(args, kwargs)),
                      original_kwargs=describe(kwargs), contract=contract(args, kwargs),
                      npu_formats=tensor_formats(npu, dict(arguments(args, kwargs), **kwargs)))
        if group_size is not None:
            if group_size != 256:
                raise ValueError('only explicit group256 is supported')
            kwargs = explicit_group256(args, kwargs)
        result['effective_kwargs'] = describe(kwargs)
        result['stage'] = 'OP_DISPATCH'
        with torch.no_grad():
            actual = npu.npu_quant_matmul(*args, **kwargs)
            result['stage'] = 'OUTPUT_COPY_TO_CPU'
            actual = snapshot(actual)
        result['stage'] = 'SAVE_OUTPUT'
        path = root/'output.pt'
        torch.save(actual, path)
        result.update(status='EXECUTED', stage='COMPLETE', output=identity(path, root), output_tensor=describe(actual))
    except Exception as exc:
        result.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
    finally:
        if npu is not None:
            try:
                # Dispatch loads API libraries lazily, even when GetWorkspaceSize fails.
                result['runtime_at_end'] = runtime(npu)
                before = set(result.get('runtime', {}).get('loaded_libraries', []))
                result['libraries_added'] = sorted(set(result['runtime_at_end']['loaded_libraries']) - before)
                result['backend_inventory'] = inventory(result['runtime_at_end'])
            except Exception as diagnostic_error:
                result['backend_diagnostic_error'] = f'{type(diagnostic_error).__name__}: {diagnostic_error}'
        save_json(root/'report.json', result)
    return result
