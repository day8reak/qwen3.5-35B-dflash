"""Offline check of retained single-token whole-GDN board outputs."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def _board_scores(actual, reference):
    from qwen35_moe_dflash.target.layer_probe.common import metrics

    row = metrics(actual, reference)
    return {key: row[key] for key in ('relative_l2', 'cosine', 'max_abs', 'golden_l2',
                                      'error_l2', 'passed', 'failed_metrics')}


def _board_hidden_rows(actual, reference):
    if actual.shape != reference.shape or actual.ndim != 3 or actual.shape[:2] != (1, 64):
        raise ValueError('Expected CPro hidden and captured out_proj output of shape [1,64,C]')
    whole = _board_scores(actual, reference)
    rows = []
    for token in range(64):
        detail = _board_scores(actual[0, token], reference[0, token])
        energy_fraction = (detail['error_l2'] / whole['error_l2'])**2 if whole['error_l2'] else 0.0
        rows.append(dict(token=token, relative_l2=detail['relative_l2'],
                         golden_l2=detail['golden_l2'], error_l2=detail['error_l2'],
                         squared_error_fraction=energy_fraction))
    defined = [r for r in rows if r['relative_l2'] is not None]
    return dict(whole=whole, per_token=rows, last_token=rows[-1],
                largest_relative_l2_token=max(defined, key=lambda r: r['relative_l2']) if defined else None,
                largest_error_energy_token=max(rows, key=lambda r: r['error_l2']))


def audit_board(group, capture_hash, layer, captured_hidden, captured_state, cpu_state, rounded_state, *, return_hidden=False):
    """Accept optional whole-GDN group and arrays in their original logical shapes."""
    if group is None:
        return None
    from qwen35_moe_dflash.target.layer_probe.common import DTYPES, KIND, checked, sha

    group = Path(group).resolve(strict=True)
    fixture, omc = group/'execution/fixture', group/'omc'
    case_name = f'l{layer:02d}_c00_gdn_layer_all'
    device = group/'recovery/native'/case_name
    meta = json.loads((fixture/'manifest.json').read_text())
    manifest = json.loads((omc/'manifest.json').read_text())
    compiled = json.loads((omc/'compile.json').read_text())
    expected = meta['source_evidence']
    manifest_hash = sha(omc/'manifest.json')
    if (meta.get('artifact_kind') != KIND or manifest.get('capture_sha256') != capture_hash
            or expected['manifest_sha256'] != manifest_hash or compiled['manifest_sha256'] != manifest_hash
            or expected['compile_sha256'] != sha(omc/'compile.json')):
        raise ValueError('Capture, OMC batch and execution fixture identities differ')
    case, = (r for r in meta['cases'] if r['case'] == case_name)
    build = compiled['cases'][case_name]
    if ((case['kind'], case['layer'], case['phase'], case['call'], case.get('gdr_mode'))
            != ('gdn_layer', layer, 'prefill', 0, 'single-token') or build['status'] != 'COMPILED'):
        raise ValueError('Expected a compiled T64 single-token Prefill GDN case')
    model = build['omc']
    if any(model[k] != case['model'][k] for k in ('sha256', 'bytes')):
        raise ValueError('Compiled OMC identity differs from the execution fixture')
    model_present = (omc/model['path']).is_file()
    if model_present:
        checked(omc, model)
    receipt = json.loads((device/'result.json').read_text())
    reps = meta['repetitions']
    if (receipt.get('evidence_kind') != 'device-hiai' or receipt.get('status') != 'EXECUTED'
            or not 2 <= reps <= 100 or receipt.get('repetitions') != reps
            or receipt.get('warmup') != meta['warmup'] or len(receipt.get('api_us', [])) != reps
            or receipt.get('input_reset') != 'copy-each-call'):
        raise ValueError('Need completed repeated board calls on the same fixtures')
    specs = case['outputs']
    names = ['golden_hidden', 'golden_new_conv_state', 'golden_recurrent_state_fp32']
    if ([s['index'] for s in specs] != [0, 1, 2] or [s['semantic_name'] for s in specs] != names
            or any(s['dtype'] != 'float16' for s in specs)
            or [s['golden_dtype'] for s in specs] != ['float16', 'float16', 'float32']):
        raise ValueError('Unexpected GDN output order or dtype')
    for s in specs:
        n = math.prod(s['shape'])
        if not 0 < s['bytes'] == n * np.dtype(DTYPES[s['dtype']]).itemsize <= (32 << 20):
            raise ValueError('Unexpected native tensor byte size')
        checked(fixture, dict(path=f"{case_name}/golden_{s['index']}.bin",
                                     bytes=s['fixture_bytes'], sha256=s['fixture_sha256']))
        expected_array = {0: captured_hidden, 2: captured_state}.get(s['index'])
        if expected_array is not None:
            array = np.asarray(expected_array)
            if (array.shape != tuple(s['shape']) or array.dtype != np.dtype(DTYPES[s['golden_dtype']])
                    or hashlib.sha256(array.tobytes()).hexdigest() != s['fixture_sha256']):
                raise ValueError(f'Board fixture {s["semantic_name"]} differs from capture')
    verified = json.loads((group/'comparison/report.json').read_text())
    previous = {r['output_name']: r for r in verified['cases'] if r['case'] == case_name}
    if set(previous) != set(names):
        raise ValueError('Prior comparison lacks all three GDN outputs')
    for row in previous.values():
        if (not row.get('stable') or len(row['output_sha256']) != reps
                or len(set(row['output_sha256'])) != 1):
            raise ValueError('Prior GDN comparison reported unstable or missing samples')
    size = sum(s['bytes'] for s in specs)
    for i in range(reps):
        sample = device/f'{i}.bin'
        if sample.stat().st_size != size:
            raise ValueError('Native output extent differs from the compiled tuple')
        with sample.open('rb') as stream:
            for s in specs:
                raw = stream.read(s['bytes'])
                if hashlib.sha256(raw).hexdigest() != previous[s['semantic_name']]['output_sha256'][i]:
                    raise ValueError(f'Native sample differs from comparison: {i} {s["semantic_name"]}')
    def read_sample(index):
        s = specs[index]
        return np.fromfile(device/'0.bin', dtype=DTYPES[s['dtype']], count=math.prod(s['shape']),
                           offset=sum(x['bytes'] for x in specs[:index])).reshape(s['shape'])
    hidden, state = read_sample(0), read_sample(2)
    for arr in (hidden, state, captured_hidden, captured_state, cpu_state, rounded_state):
        if not np.isfinite(arr).all():
            raise ValueError('Non-finite board or reference array')
    report = dict(case=case_name, model_sha256=model['sha256'], model_file_checked=model_present,
                capture_manifest_sha256=capture_hash, stable_repetitions=reps,
                hidden_vs_captured_out_proj=_board_hidden_rows(hidden, captured_hidden),
                final_state_vs_capture=_board_scores(state, captured_state),
                final_state_vs_cpu=_board_scores(state, cpu_state),
                final_state_vs_cpu_rounded=_board_scores(state, rounded_state),
                scope='WHOLE_LAYER_FINAL_OUTPUTS_ONLY')
    return (report, hidden) if return_hidden else report
