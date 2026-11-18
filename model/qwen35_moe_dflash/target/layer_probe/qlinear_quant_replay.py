"""Replay only original DynamicQuant on frozen logical projection inputs.

These are new quantizer calls, not retained intermediates of the original
successful model invocation. No QuantMatmul or model source is executed.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from .common import checked, identity, read_array, save_json, sha

KIND = 'qwen35-original-dynamic-quant-replay-v1'
PROVENANCE = 'NEW_DYNAMIC_QUANT_CALL_ON_FROZEN_LOGICAL_INPUT; ORIGINAL_SUCCESSFUL_A8_NOT_CAPTURED'


def _module_identity(module):
    row = dict(version=str(getattr(module, '__version__', 'UNAVAILABLE')))
    filename = getattr(module, '__file__', None)
    if filename and Path(filename).is_file():
        path = Path(filename).resolve()
        row['entrypoint_file'] = dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))
    else:
        row['entrypoint_file'] = dict(status='UNAVAILABLE')
    # The entrypoint hash identifies this Python module, not every linked library.
    extension = getattr(module, '_C', None)
    filename = getattr(extension, '__file__', None)
    if filename and Path(filename).is_file():
        path = Path(filename).resolve()
        row['extension_file'] = dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))
    return row


def _describe(tensor, npu):
    row = dict(shape=list(tensor.shape), dtype=str(tensor.dtype), device=str(tensor.device),
               stride=list(tensor.stride()), contiguous=tensor.is_contiguous())
    getter = getattr(npu, 'get_npu_format', None)
    if tensor.device.type != 'npu' or getter is None:
        row['npu_format'] = 'UNAVAILABLE'
    else:
        try:
            row['npu_format'] = int(getter(tensor))
        except Exception as exc:
            row.update(npu_format='UNAVAILABLE', npu_format_error=f'{type(exc).__name__}: {exc}')
    return row


def _save_tensor(root, folder, name, tensor):
    array = tensor.detach().to('cpu', copy=True).contiguous().numpy()
    path = folder / (name + '.bin')
    with path.open('xb') as stream:
        stream.write(array.tobytes(order='C'))
    spec = dict(identity(path, root), dtype=str(array.dtype), shape=list(array.shape))
    checked(root, spec)
    return spec


def replay(capture, output, device='npu:0', points=None):
    """Record new A8/scale pairs; failure remains explicit in manifest.json."""
    from .qlinear_golden import load_capture, separate_output, tensor_spec

    capture_root = Path(capture).resolve(strict=True)
    capture_sha256 = sha(capture_root / 'manifest.json')
    capture_root, manifest, regions = load_capture(capture_root, points)
    if sha(capture_root / 'manifest.json') != capture_sha256:
        raise ValueError('capture manifest changed while selecting projections')
    root = separate_output(output, capture_root)
    report = dict(artifact_kind=KIND, status='RECORDING', capture=str(capture_root),
        capture_sha256=capture_sha256, provenance=PROVENANCE, requested_device=str(device),
        scope='DYNAMIC_QUANT_ONLY; NO_MODEL_EXECUTION; NO_QUANT_MATMUL; NO_GOLDEN_REPLACEMENT',
        input_reconstruction='CONTIGUOUS_LOGICAL_VALUES; ORIGINAL_NPU_FORMAT_AND_STRIDE_NOT_RETAINED',
        numerical_validation='NOT_RUN', cases=[], errors=[])
    save_json(root / 'manifest.json', report)
    try:
        import torch
        import torch_npu

        report['runtime'] = dict(torch=_module_identity(torch), torch_npu=_module_identity(torch_npu))
        for region in regions:
            case = dict(id=region['id'], status='RECORDING',
                        original_execution_device=region.get('execution_device', 'UNAVAILABLE'))
            report['cases'].append(case)
            save_json(root / 'manifest.json', report)
            try:
                spec = tensor_spec(region, 'inputs', 'x')
                case['input'] = deepcopy(spec)
                x = read_array(capture_root, spec)
                if (x.dtype not in (np.dtype('float16'), np.dtype('float32')) or x.ndim < 2
                        or not x.size or not np.isfinite(x).all()):
                    raise ValueError('DynamicQuant replay needs nonempty finite FP16/FP32 input of rank >= 2')
                m, k = x.size // x.shape[-1], x.shape[-1]
                flat = torch.from_numpy(x.copy()).reshape(m, k).to(device)
                case['call_input'] = _describe(flat, torch_npu)
                with torch.no_grad():
                    result = torch_npu.npu_dynamic_quant(flat)
                if not isinstance(result, (tuple, list)) or len(result) != 2:
                    raise ValueError('DynamicQuant must return exactly q and per-token scale')
                q, scale = result
                if not isinstance(q, torch.Tensor) or not isinstance(scale, torch.Tensor):
                    raise ValueError('DynamicQuant results must both be tensors')
                case['call_outputs'] = dict(q=_describe(q, torch_npu), token_scale=_describe(scale, torch_npu))
                if q.dtype != torch.int8 or list(q.shape) != [m, k]:
                    raise ValueError('DynamicQuant q must be INT8[M,K]; no dtype/shape conversion allowed')
                if scale.dtype != torch.float32 or list(scale.shape) not in ([m], [m, 1]):
                    raise ValueError('DynamicQuant scale must be FP32[M] or FP32[M,1]; no dtype conversion allowed')
                if q.device != flat.device or scale.device != flat.device:
                    raise ValueError('DynamicQuant output device differs from its input')
                scale_array = scale.detach().to('cpu', copy=True).contiguous().numpy()
                if not np.isfinite(scale_array).all() or (scale_array < 0).any():
                    raise ValueError('DynamicQuant returned nonfinite or negative scale')
                if flat.detach().to('cpu', copy=True).contiguous().numpy().tobytes() != x.tobytes():
                    raise ValueError('DynamicQuant changed the frozen input values')
                # Recheck the recorded source bytes after the asynchronous device work.
                checked(capture_root, spec)
                folder = root / region['id']
                folder.mkdir(mode=0o700)
                case['q'] = _save_tensor(root, folder, 'q', q)
                case['token_scale'] = _save_tensor(root, folder, 'token_scale', scale.reshape(m, 1))
                case['status'] = 'RECORDED'
            except Exception as exc:
                case.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
            save_json(root / 'manifest.json', report)
        report['status'] = 'RECORDED' if all(c['status'] == 'RECORDED' for c in report['cases']) else 'PARTIAL'
        if sha(capture_root / 'manifest.json') != capture_sha256:
            raise ValueError('capture manifest changed during quantizer replay; recorded pairs are unqualified')
    except Exception as exc:
        report['status'] = 'FAILED'
        report['errors'].append(f'{type(exc).__name__}: {exc}')
    save_json(root / 'manifest.json', report)
    lines = [PROVENANCE, 'status=' + report['status']]
    lines.extend(f"{c['id']}: {c['status']}" + ('; ' + c['error'] if 'error' in c else '') for c in report['cases'])
    lines.extend('ERROR: ' + error for error in report['errors'])
    (root / 'summary.txt').write_text('\n'.join(lines) + '\n')
    return report
