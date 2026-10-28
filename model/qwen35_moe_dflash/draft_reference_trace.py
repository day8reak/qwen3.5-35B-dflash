"""Bounded CPU traces of the retained draft and an isolated RMS intervention.

This helper never loads weights, generates inputs, writes reference files or
executes a target/device model. The caller binds graph/inputs to retained hashes.
FP32 statistics are a counterfactual diagnostic, not a promoted golden.
"""
import math
import os

import numpy as np


_CHUNK_ELEMENTS = 1 << 18
_MAX_NORM_CALLS = 512
_MAX_HIDDEN_ELEMENTS = 4 << 20
_INPUT_NAMES = ('target_features', 'input_ids', 'attention_mask')


def _summary(tensor):
    import torch
    flat = tensor.detach().reshape(-1)
    result = {'shape': list(tensor.shape), 'dtype': str(tensor.dtype).removeprefix('torch.'),
              'elements': tensor.numel(), 'finite_count': 0, 'nonfinite_count': 0,
              'nan_count': 0, 'inf_count': 0, 'zero_count': 0, 'nonzero_count': 0,
              'finite_min': None, 'finite_max': None, 'finite_abs_max': None}
    total, squared = 0.0, 0.0
    for start in range(0, flat.numel(), _CHUNK_ELEMENTS):
        values = flat[start:start + _CHUNK_ELEMENTS]
        finite = torch.isfinite(values)
        count = int(finite.sum())
        result['finite_count'] += count
        result['nonfinite_count'] += values.numel() - count
        result['nan_count'] += int(torch.isnan(values).sum())
        result['inf_count'] += int(torch.isinf(values).sum())
        result['zero_count'] += int((values == 0).sum())
        if count:
            selected = values[finite].double()
            low, high = float(selected.min()), float(selected.max())
            result['finite_min'] = low if result['finite_min'] is None else min(result['finite_min'], low)
            result['finite_max'] = high if result['finite_max'] is None else max(result['finite_max'], high)
            total += float(selected.sum())
            squared += float(selected.square().sum())
    result['nonzero_count'] = result['elements'] - result['zero_count']
    result['all_zero'] = result['elements'] > 0 and result['zero_count'] == result['elements']
    if result['finite_count']:
        result['finite_abs_max'] = max(abs(result['finite_min']), abs(result['finite_max']))
        result['finite_mean'] = total / result['finite_count']
        result['finite_rms'] = math.sqrt(squared / result['finite_count'])
    else:
        result.update(finite_mean=None, finite_rms=None)
    return result


def _rms_diagnostics(x, weight, output, eps):
    """Count hypothetical FP16 intermediate failures without retaining tensors."""
    import torch
    width = x.shape[-1]
    rows = x.detach().reshape(-1, width)
    out = output.detach().reshape(-1, width)
    result = {'rows': rows.shape[0], 'fp16_square_nonfinite_count': 0,
              'fp16_mean_square_nonfinite_rows': 0, 'fp16_rsqrt_zero_rows': 0,
              'input_all_zero_rows': 0, 'output_all_zero_rows': 0,
              'newly_zero_rows': 0, 'newly_zero_rows_with_fp16_square_overflow': 0}
    chunk_rows = max(1, _CHUNK_ELEMENTS // width)
    scale = weight.detach().view(1, width)
    for start in range(0, rows.shape[0], chunk_rows):
        value, actual = rows[start:start + chunk_rows], out[start:start + chunk_rows]
        square = value.half().square()
        mean = square.mean(-1, keepdim=True)
        inverse = torch.rsqrt(mean + eps)
        square_bad = ~torch.isfinite(square)
        output_zero = (actual == 0).all(-1)
        # Padding/already-zero rows and zero scale do not establish collapse.
        weighted_nonzero = ((value != 0) & (scale != 0)).any(-1)
        finite_input = torch.isfinite(value).all(-1) & torch.isfinite(scale).all(-1)
        collapsed = finite_input & weighted_nonzero & output_zero
        result['fp16_square_nonfinite_count'] += int(square_bad.sum())
        result['fp16_mean_square_nonfinite_rows'] += int((~torch.isfinite(mean)).sum())
        result['fp16_rsqrt_zero_rows'] += int((inverse == 0).sum())
        result['input_all_zero_rows'] += int((value == 0).all(-1).sum())
        result['output_all_zero_rows'] += int(output_zero.sum())
        result['newly_zero_rows'] += int(collapsed.sum())
        result['newly_zero_rows_with_fp16_square_overflow'] += int((collapsed & square_bad.any(-1)).sum())
    return result


class _RMSOnly:
    def __init__(self, original, rms):
        self.original, self.rms = original, rms

    def rms_norm(self, x, weight, eps):
        return self.rms(x, weight, eps)

    def __getattr__(self, name):
        return getattr(self.original, name)


def replay_graph(graph, inputs, *, rms_mode, progress=None):
    """Return ``({'draft_top1': array, 'draft_hidden': array}, JSON-safe trace)``.

    Inputs are the three retained NumPy arrays in ABI order or a name-keyed dict.
    The same loaded CPU FP16 graph can be used for both modes. ``legacy-fp16``
    leaves all operators unchanged; ``fp32-stats`` replaces only RMS arithmetic
    with TorchDFlashOps.rms_norm (FP32 normalization, cast, direct scale).
    Degenerate legacy outputs are returned with FAILED validation so callers can
    reproduce old goldens byte-for-byte. They must not be promoted as a golden.
    """
    os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
    import torch
    from .model import DFlashRMSNorm
    from .ops import TorchDFlashOps
    if rms_mode not in {'legacy-fp16', 'fp32-stats'}:
        raise ValueError('rms_mode must be legacy-fp16 or fp32-stats')
    if graph.training or any(p.device.type != 'cpu' or p.dtype != torch.float16 for p in graph.parameters()):
        raise ValueError('replay requires the loaded eval-mode CPU FP16 draft graph')
    if isinstance(inputs, dict):
        if set(inputs) != set(_INPUT_NAMES):
            raise ValueError('retained input names differ from the draft ABI')
        arrays = [inputs[name] for name in _INPUT_NAMES]
    else:
        arrays = list(inputs)
    if len(arrays) != 3 or any(not isinstance(array, np.ndarray) for array in arrays):
        raise ValueError('replay requires three retained NumPy input arrays')
    if [array.dtype for array in arrays] != [np.dtype('<f2'), np.dtype('<i4'), np.dtype('<i4')]:
        raise ValueError('retained inputs must keep exact float16/int32/int32 dtypes')
    if any(not array.flags.c_contiguous for array in arrays):
        raise ValueError('retained input arrays must be contiguous')
    config = graph.draft.config
    features, ids, mask = arrays
    if (features.ndim != 3 or features.shape[0] != 1 or features.shape[-1] != config.feature_size
            or ids.shape != features.shape[:2] or mask.shape != ids.shape or ids.shape[1] < 1):
        raise ValueError('retained input shapes differ from the loaded draft')
    if (not np.isfinite(features).all() or np.any(ids < 0) or np.any(ids >= config.vocab_size)
            or not np.isin(mask, [0, 1]).all() or not mask[0, 0]
            or np.any(np.diff(mask.astype(np.int64), axis=-1) > 0)):
        raise ValueError('retained inputs violate finite features/token/right-padded mask contract')
    norms = [(name, node) for name, node in graph.named_modules() if isinstance(node, DFlashRMSNorm)]
    if not norms or len(norms) > _MAX_NORM_CALLS or not isinstance(graph.draft.norm, DFlashRMSNorm):
        raise ValueError('missing or unbounded draft RMS norm topology')
    if any(type(node.eps) not in (int, float) or not math.isfinite(node.eps) or node.eps < 0
           for _, node in norms):
        raise ValueError('RMS epsilon must be finite and nonnegative')
    trace = {'schema_version': 1, 'artifact_kind': 'qwen35-draft-reference-rms-trace',
             'rms_mode': rms_mode, 'rms_nodes': [], 'first_collapse': None,
             'first_nonfinite': None, 'inputs': [], 'outputs': {}, 'logits': None,
             'target_execution': 'NOT_RUN', 'device_execution': 'NOT_RUN',
             'precision_scope': ('Unchanged retained FP16 primitive graph' if rms_mode == 'legacy-fp16' else
                'Counterfactual RMS-only intervention: normalize in FP32, cast to input dtype, multiply unchanged scale'),
             'scope': 'CPU arithmetic trace on supplied graph/inputs; provenance and legacy reproduction must be checked by caller.'}
    tensors = [torch.from_numpy(array) for array in arrays]
    trace['inputs'] = [{'name': name, **_summary(value)} for name, value in zip(_INPUT_NAMES, tensors)]
    handles, originals, captured, head_summaries = [], [], [], []
    def norm_hook(name):
        def observe(node, arguments, output):
            if len(trace['rms_nodes']) >= _MAX_NORM_CALLS:
                raise ValueError('RMS trace call budget exceeded')
            x = arguments[0]
            row = {'index': len(trace['rms_nodes']), 'name': name,
                   'input': _summary(x), 'weight': _summary(node.weight), 'output': _summary(output),
                   'epsilon': float(node.eps), **_rms_diagnostics(x, node.weight, output, node.eps)}
            trace['rms_nodes'].append(row)
            if trace['first_collapse'] is None and row['newly_zero_rows']:
                overflow = row['newly_zero_rows_with_fp16_square_overflow'] > 0
                trace['first_collapse'] = {'index': row['index'], 'name': name,
                    'newly_zero_rows': row['newly_zero_rows'],
                    'cause_candidate': ('FP16_SQUARE_OVERFLOW_WITH_ZERO_OUTPUT' if overflow and rms_mode == 'legacy-fp16'
                                        else 'ZERO_OUTPUT_FROM_FINITE_NONZERO_INPUT'),
                    'scope': 'Observed intermediate coincidence; exact legacy reproduction is required for attribution.'}
            if trace['first_nonfinite'] is None:
                for boundary in ('input', 'output'):
                    if row[boundary]['nonfinite_count']:
                        trace['first_nonfinite'] = {'index': row['index'], 'name': name, 'boundary': boundary}
                        break
            if node is graph.draft.norm:
                hidden = output[:, 1:, :]
                if hidden.numel() > _MAX_HIDDEN_ELEMENTS or captured:
                    raise ValueError('final draft hidden capture exceeds bounded single-call contract')
                captured.append(hidden.detach().clone())
            if progress:
                progress(f"RMS {row['index'] + 1}: {name}; input_abs_max={row['input']['finite_abs_max']}; "
                         f"fp16_square_nonfinite={row['fp16_square_nonfinite_count']}; newly_zero_rows={row['newly_zero_rows']}")
        return observe
    def head_hook(node, arguments, output):
        if head_summaries:
            raise ValueError('draft replay expects one full-vocabulary head call')
        head_summaries.append(_summary(output))
    try:
        fp32_rms = TorchDFlashOps().rms_norm
        for name, node in norms:
            originals.append((node, node.ops))
            if rms_mode == 'fp32-stats':
                node.ops = _RMSOnly(node.ops, fp32_rms)
            handles.append(node.register_forward_hook(norm_hook(name)))
        handles.append(graph.lm_head.register_forward_hook(head_hook))
        if progress:
            progress(f"Replaying retained draft with RMS mode {rms_mode}")
        with torch.inference_mode():
            top1 = graph(*tensors)
        if len(captured) != 1 or len(head_summaries) != 1:
            raise ValueError('missing complete draft hidden/head capture')
        hidden = captured[0]
        if (top1.dtype != torch.int32 or list(top1.shape) != [1, config.block_size - 1]
                or hidden.dtype != torch.float16 or list(hidden.shape) != [1, config.block_size - 1, config.hidden_size]):
            raise ValueError('replayed draft output ABI differs from the loaded graph')
        trace['outputs'] = {'draft_top1': _summary(top1), 'draft_hidden': _summary(hidden)}
        trace['logits'] = head_summaries[0]
        finite = not any(row['nonfinite_count'] for row in [*trace['outputs'].values(), trace['logits']])
        nondegenerate = not (trace['outputs']['draft_hidden']['all_zero'] or trace['logits']['all_zero'])
        errors = []
        if not finite:
            errors.append('nonfinite complete draft hidden/logits output')
        if not nondegenerate:
            errors.append('degenerate all-zero draft hidden/logits output')
        if bool((top1 < 0).any()) or bool((top1 >= config.vocab_size).any()):
            errors.append('draft Top1 outside full vocabulary')
        trace['validation'] = {'status': 'FAILED' if errors else 'PASSED', 'finite': finite,
                               'nondegenerate': nondegenerate, 'errors': errors}
        trace['rms_calls'] = len(trace['rms_nodes'])
        trace['torch_version'] = str(torch.__version__)
        arrays = {'draft_top1': top1.detach().numpy().copy(), 'draft_hidden': hidden.numpy().copy()}
        if progress:
            progress(f"Replay {rms_mode} complete: {trace['validation']['status']}")
        return arrays, trace
    finally:
        for handle in handles:
            handle.remove()
        for node, original in originals:
            node.ops = original
