"""Read whole-GDN outputs and test whether state outputs retain their old inputs.

No source export, device access, tensor repair, or replacement golden. Input
semantics follow WholeGDN's explicit five-argument contract and fixture_index,
not runtime slot numbers or matching tensor sizes.
"""
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from .common import KIND, DTYPES, checked, new_output, save_json
from qwen35_moe_dflash.common.artifacts import output_path

OUTPUTS = ('golden_hidden', 'golden_new_conv_state', 'golden_recurrent_state_fp32')


def stats(value):
    finite = value[np.isfinite(value)]
    return dict(shape=list(value.shape), dtype=str(value.dtype), elements=value.size,
        nonzero=int(np.count_nonzero(value)), nonfinite=int(value.size - finite.size),
        all_zero=bool(value.size and np.all(value == 0)),
        abs_max=float(np.max(np.abs(finite.astype(np.float64)))) if finite.size else None)


def difference(actual, reference):
    if actual.shape != reference.shape:
        raise ValueError('diagnostic comparison requires identical shapes')
    a, b = actual.astype(np.float64), reference.astype(np.float64)
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    norm = float(np.linalg.norm(b.ravel())) if finite else None
    delta = float(np.linalg.norm((a-b).ravel())) if finite else None
    return dict(equal_values=bool(finite and np.array_equal(a, b)),
        equal_bits=bool(actual.dtype == reference.dtype and actual.tobytes() == reference.tobytes()),
        mismatched_elements=int(np.count_nonzero(a != b)), elements=a.size,
        max_abs=float(np.max(np.abs(a-b))) if finite and a.size else None,
        relative_l2=delta/norm if norm else (0.0 if delta == 0 else None))


def _array(fixture, name, spec, output=False):
    dtype = spec['golden_dtype'] if output else spec['dtype']
    path = checked(fixture, dict(path=f'{name}/{"golden" if output else "input"}_{spec["index"]}.bin',
        bytes=spec['fixture_bytes'], sha256=spec['fixture_sha256']))
    expected = math.prod(spec['shape']) * np.dtype(DTYPES[dtype]).itemsize
    if expected != path.stat().st_size:
        raise ValueError('fixture extent differs from its tensor descriptor')
    return np.fromfile(path, dtype=DTYPES[dtype]).reshape(spec['shape'])


def _inspect_case(case, meta, fixture, native):
    name = case['case']
    if not re.fullmatch(r'l[0-9]{2,}_c[0-9]{2,}_gdn_layer_all', name):
        raise ValueError('invalid whole-GDN case name')
    outputs = case['outputs']
    if ([s['index'] for s in outputs] != [0, 1, 2]
            or tuple(s['semantic_name'] for s in outputs) != OUTPUTS
            or any(s['dtype'] not in ('float16', 'float32') for s in outputs)):
        raise ValueError('whole-GDN tuple binding differs from hidden/conv/state contract')
    inputs = {s['fixture_index']: s for s in case['inputs']}
    if len(inputs) != len(case['inputs']) or not {1, 2, 3, 4} <= inputs.keys():
        raise ValueError('old conv/state and controls must bind to distinct source arguments')
    sizes = [math.prod(s['shape'])*np.dtype(DTYPES[s['dtype']]).itemsize for s in outputs]
    if any(s['bytes'] != size for s, size in zip(outputs, sizes)) or not 0 < sum(sizes) <= 32 << 20:
        raise ValueError('compiled output extent is invalid or exceeds 32 MiB')
    for spec in (*inputs.values(), *outputs):
        if (not spec['shape'] or any(type(d) is not int or d <= 0 for d in spec['shape'])
                or not 0 < spec['fixture_bytes'] <= 32 << 20):
            raise ValueError('requires bounded positive fixture dimensions and byte counts')
    old = {1: _array(fixture, name, inputs[1]), 2: _array(fixture, name, inputs[2])}
    controls = []
    for source_index, semantic in ((3, 'linear_update_tensor'), (4, 'effective_length')):
        spec = inputs[source_index]
        if spec['dtype'] not in ('int16', 'int32', 'int64') or math.prod(spec['shape']) > 256:
            raise ValueError('unexpected GDN control tensor')
        value = _array(fixture, name, spec)
        controls.append(dict(name=semantic, source_argument=source_index, runtime_slot=spec['index'],
                             **stats(value), values=value.tolist()))
    golden = [_array(fixture, name, s, output=True) for s in outputs]
    receipt = json.loads((native/name/'result.json').read_text())
    if (receipt.get('status') != 'EXECUTED' or receipt.get('evidence_kind') != 'device-hiai'
            or receipt.get('repetitions') != meta['repetitions'] or receipt.get('warmup') != meta['warmup']
            or len(receipt.get('api_us', [])) != meta['repetitions']
            or receipt.get('input_reset') != 'copy-each-call'):
        raise ValueError('native receipt does not match the completed fixture invocation contract')
    hashes, first, old_matches = [[], [], []], [], {1: [], 2: []}
    for invocation in range(meta['repetitions']):
        path = native/name/f'{invocation}.bin'
        if path.stat().st_size != sum(sizes):
            raise ValueError('native dump byte extent differs from the compiled three-output tuple')
        payload, offset = path.read_bytes(), 0
        for slot, (spec, size) in enumerate(zip(outputs, sizes)):
            data = payload[offset:offset+size]
            actual = np.frombuffer(data, dtype=DTYPES[spec['dtype']]).reshape(spec['shape'])
            hashes[slot].append(hashlib.sha256(data).hexdigest())
            if invocation == 0:
                first.append(actual.copy())
            if slot in old_matches:
                old_matches[slot].append(difference(actual, old[slot])['equal_bits'])
            offset += size
    rows = []
    for slot, (actual, expected) in enumerate(zip(first, golden)):
        row = dict(slot=slot, semantic_name=OUTPUTS[slot], actual=stats(actual), golden=stats(expected),
                   actual_vs_golden=difference(actual, expected),
                   repeated_bitwise_stable=len(set(hashes[slot])) == 1)
        if slot in old:
            row.update(old_input=stats(old[slot]), old_input_runtime_slot=inputs[slot]['index'],
                actual_vs_old_input=difference(actual, old[slot]),
                golden_vs_old_input=difference(expected, old[slot]),
                matches_old_input_bitwise_invocations=[i for i, equal in enumerate(old_matches[slot]) if equal],
                all_invocations_equal_old_input_bits=all(old_matches[slot]))
        rows.append(row)
    conv, previous, expected = first[1], old[1], golden[1]
    if conv.ndim != 3 or not 1 <= conv.shape[-1] <= 16:
        raise ValueError('ConvState history diagnostics require captured [B, C, K], K <= 16')
    lanes = [dict(actual_history_slot=i,
        actual_vs_old_slots=[difference(conv[..., i], previous[..., j]) for j in range(conv.shape[-1])],
        actual_vs_golden_slots=[difference(conv[..., i], expected[..., j]) for j in range(conv.shape[-1])])
        for i in range(conv.shape[-1])]
    return dict(case=name, layer=case['layer'], call=case['call'], phase=case['phase'], status='INSPECTED',
        controls=controls, outputs=rows, conv_history_slots=lanes,
        conv_history_scope='CAPTURED_SOURCE_BCK_COORDINATES; NO_LAYOUT_REPAIR_OR_SLOT_REBINDING',
        observation=('CONV_OUTPUT_EQUALS_OLD_INPUT_BITS' if all(old_matches[1])
                     else 'CONV_OUTPUT_DIFFERS_FROM_OLD_INPUT_BITS'),
        cause='UNDETERMINED; OUTPUT_EQUALITY_DOES_NOT_PROVE_KERNEL_WRITE_OR_ALIAS_BEHAVIOR')


def inspect_gdn_outputs(fixture, native, output):
    fixture, native, out = Path(fixture).resolve(), Path(native).resolve(), output_path(output)
    if any(out == p or out in p.parents or p in out.parents for p in (fixture, native)):
        raise ValueError('inspection output must be separate from its retained inputs')
    meta = json.loads((fixture/'manifest.json').read_text())
    if meta.get('artifact_kind') != KIND or not 0 < meta.get('repetitions', 0) <= 100:
        raise ValueError('expected a completed Target boundary fixture')
    selected = [c for c in meta['cases'] if c['kind'] == 'gdn_layer']
    if not selected or len({c['case'] for c in selected}) != len(selected):
        raise ValueError('expected distinct whole-GDN fixture cases')
    out = new_output(out)
    report = dict(status='INSPECTED', cases=[], numerical_validation='NOT_RUN', model_compilation='NOT_RUN',
        device_execution='NOT_RUN', scope='RETAINED_DUMPS_AND_FIXTURE_INPUTS; NOT_A_NEW_ACCURACY_VERDICT',
        source_binding='WholeGDN source arguments 1=old_conv_state, 2=initial_state; fixture_index to runtime slot',
        invocation_scope=f'All {meta["repetitions"]} dumps checked; detailed values from invocation 0')
    lines = ['Read-only whole-GDN state-output inspection; no export, compilation or device execution.',
             'No values, layouts, output order, golden or accuracy thresholds changed.',
             'Equality to old input is an observation; kernel writes and storage aliases remain unverified.', '']
    for case in selected:
        try:
            row = _inspect_case(case, meta, fixture, native)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            row = dict(case=case['case'], status='FAILED', error=f'{type(exc).__name__}: {exc}')
            report['status'] = 'PARTIAL'
        report['cases'].append(row)
        lines.append(f'{row["case"]}: {row["status"]}')
        if row['status'] == 'INSPECTED':
            lines.append('controls=' + json.dumps(row['controls']))
            for s in row['outputs']:
                lines.append(json.dumps(s))
            lines.append('observation=' + row['observation'])
        else:
            lines.append(row['error'])
        lines.append('')
    save_json(out/'report.json', report)
    (out/'summary.txt').write_text('\n'.join(lines)+'\n')
    return dict(status=report['status'], cases=len(selected), report=str(out/'report.json'),
        summary=str(out/'summary.txt'), numerical_validation='NOT_RUN', model_compilation='NOT_RUN', device_execution='NOT_RUN')
