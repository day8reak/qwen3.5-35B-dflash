"""Exercise the live source attention callable after the four-layer Prefill has returned."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils._pytree import tree_flatten

from .attention_reference import unpack
from .capture import cpu
from .common import checked, read_array


def previous_call(fixture):
    root = Path(fixture).resolve(strict=True)
    manifest = json.loads((root / 'manifest.json').read_text())
    calls = manifest.get('attention_audit', {}).get('calls', [])
    if (manifest.get('artifact_kind') != 'qwen35-target-prefill-fixture-v1'
            or len(calls) != 1 or calls[0].get('status') not in ('PASSED', 'FAILED')):
        raise ValueError('expected a Prefill fixture with one completed source attention audit')
    return checked(root, manifest['tokens']), calls[0]


def same_arguments(live, saved):
    values, structure = tree_flatten(cpu(live))
    originals, original_structure = tree_flatten(saved)
    if structure != original_structure:
        return False
    for a, b in zip(values, originals):
        if type(a) is not type(b):
            return False
        if isinstance(a, torch.Tensor):
            if a.shape != b.shape or a.dtype != b.dtype or a.numpy().tobytes() != b.numpy().tobytes():
                return False
        elif a != b:
            return False
    return True


@torch.no_grad()
def run_control(npu, live_calls, root, audit, historical, report):
    if len(live_calls) != 1 or len(audit['calls']) != 1:
        raise ValueError('expected one captured live source attention call')
    kwargs, saved = live_calls[0]
    call = audit['calls'][0]

    def inputs(row):
        return {s['name']: (s['shape'], s['dtype'], s['sha256']) for s in row['inputs']}

    report['packed_inputs_match_previous_fixture'] = inputs(call) == inputs(historical)
    report['prefill_output_matches_previous_fixture'] = call['actual']['sha256'] == historical['actual']['sha256']
    if not same_arguments(kwargs, saved):
        raise ValueError('live attention arguments changed since capture')
    original = read_array(root, call['actual'])
    reference = read_array(root, call['independent_reference'])
    value = kwargs['value'][0]
    scratch = value.clone(memory_format=torch.preserve_format)
    if (value.stride() != scratch.stride() or value.storage_offset() != scratch.storage_offset()
            or value.device != scratch.device or value.dtype != scratch.dtype):
        raise ValueError('private V copy differs from source V descriptor')
    if value.device.type == 'npu':
        report['value_formats'] = [int(npu.get_npu_format(t)) for t in (value, scratch)]
        if report['value_formats'][0] != report['value_formats'][1]:
            raise ValueError('private V copy differs from the source NPU format')
    # Reuse one private allocation for all four calls, preserving the source
    # list/tuple signature and keeping the original cache and golden intact.
    args = dict(kwargs, value=type(kwargs['value'])([scratch]))
    results = report['calls'] = {}
    first = None
    for label, constant in (('original_v', None), ('plus_one', 1), ('minus_one', -1), ('restored_v', None)):
        if constant is None:
            scratch.copy_(value)
        else:
            scratch.fill_(constant)
        staged = cpu(scratch).numpy()
        if constant is None:
            payload_matches = staged.tobytes() == saved['value'][0].numpy().tobytes()
        else:
            payload_matches = (staged == constant).all()
        if not payload_matches:
            raise ValueError('V payload readback failed before ' + label)
        y = cpu(npu.adn_fused_infer_attention(**args)).numpy()
        if y.dtype != original.dtype or y.shape != original.shape:
            raise ValueError('source fused output dtype/shape changed')
        logical = unpack(y, kwargs['num_heads']).astype(np.float32)
        expected = reference if constant is None else constant
        delta = np.abs(logical - expected)
        blocks = []
        for start in range(0, logical.shape[2], 16):
            part = logical[:, :, start:start+16]
            err = delta[:, :, start:start+16]
            finite = np.isfinite(err)
            blocks.append(dict(first_query=start, zero_head_rows=int((part == 0).all(-1).sum()),
                nonfinite_elements=int((~np.isfinite(part)).sum()),
                max_abs_error=float(err[finite].max()) if finite.any() else None))
        results[label] = dict(blocks=blocks)
        if constant is None:
            bits = y.tobytes()
            results[label]['equal_bits_to_prefill'] = bits == original.tobytes()
            if first is None:
                first = bits
            else:
                results[label]['equal_bits_to_first_control'] = bits == first
        if not same_arguments(kwargs, saved):
            raise ValueError('source arguments changed during ' + label)
    report['status'] = 'INSPECTED'
