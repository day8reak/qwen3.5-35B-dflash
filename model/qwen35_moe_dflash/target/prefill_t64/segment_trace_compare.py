"""Locate valid19 errors only when the traced GDN reproduces the original board hidden."""
import hashlib
import json

import numpy as np

from ..layer_probe.common import checked, metrics, read_array, sha
from ..layer_probe.qlinear_math import dynamic_quant, group_matmul
from .gdn_tail import _gate_norm
from .gdr_math import compute
from .segment_gdn_trace import BASIS, OPERANDS, TAPS


def _json(path):
    return json.loads(path.read_text())


def board_case(root, name):
    fixture = root / 'execution/fixture'
    manifest = _json(fixture / 'manifest.json')
    case, = [c for c in manifest['cases'] if c['case'] == name]
    report = _json(root / 'recovery/result.json')
    rows = {r['output_name']: r for r in report['cases'] if r['case'] == name}
    device = root / 'recovery/native' / name
    receipt = _json(device / 'result.json')
    if (report.get('device_execution') != 'EXECUTED_OUTPUTS_RETRIEVED' or receipt['status'] != 'EXECUTED'
            or receipt['evidence_kind'] != 'device-hiai' or receipt['input_reset'] != 'copy-each-call'
            or receipt['repetitions'] != manifest['repetitions'] or receipt['repetitions'] < 2):
        raise ValueError(f'{name}: need a recovered repeated device execution')
    inputs = {}
    for s in case['inputs']:
        checked(fixture, dict(path=f"{name}/input_{s['index']}.bin", bytes=s['fixture_bytes'], sha256=s['fixture_sha256']))
        inputs[s['semantic_name']] = (s['dtype'], s['shape'], s['fixture_sha256'])
    outputs, references = {}, {}
    with (device / '0.bin').open('rb') as stream:
        for s in case['outputs']:
            key = s['semantic_name']
            raw = stream.read(s['bytes'])
            row = rows[key]
            if (not row['stable'] or len(row['output_sha256']) != receipt['repetitions']
                    or row['output_sha256'][0] != hashlib.sha256(raw).hexdigest()):
                raise ValueError(f'{name}/{key}: unstable result or mismatched readback')
            outputs[key] = np.frombuffer(raw, dtype=s['dtype']).reshape(s['shape'])
            references[key] = read_array(fixture, dict(path=f"{name}/golden_{s['index']}.bin", dtype=s['golden_dtype'],
                shape=s['shape'], bytes=s['fixture_bytes'], sha256=s['fixture_sha256']))
        if stream.read(1):
            raise ValueError(f'{name}: unexpected output extent')
    return inputs, references, outputs, case


def summarize_hidden(baseline):
    print('layer  valid19_rel_L2  error_RMS  reference_RMS  token18_rel_L2  largest_error_token  share')
    for layer in range(3):
        _, refs, actual, _ = board_case(baseline, f'l{layer:02}_c00_gdn_layer_hidden_o0')
        golden = refs['golden_hidden'][0, :19].astype(np.float64)
        delta = actual['golden_hidden'][0, :19].astype(np.float64) - golden
        error_energy = np.sum(delta**2, axis=1)
        peak = int(error_energy.argmax())
        def ratio(delta, reference):
            denominator = np.linalg.norm(reference)
            return f'{np.linalg.norm(delta)/denominator:.5f}' if denominator else 'undefined'
        share = error_energy[peak] / error_energy.sum() if error_energy.sum() else 0.
        print(f'L{layer:<4} {ratio(delta, golden):>13}  {np.sqrt(np.mean(delta**2)):9.5g}'
              f'  {np.sqrt(np.mean(golden**2)):13.5g}  {ratio(delta[18], golden[18]):>14}'
              f'  {peak:19d}  {share:.1%}')


def compare(baseline, trace):
    omc = trace / 'omc'
    build = _json(omc / 'compile.json')
    air = omc / build['source_batch']
    if sha(air / 'manifest.json') != build['manifest_sha256']:
        raise ValueError('Mismatched trace compilation and source AIR batch')
    entry, = _json(air / 'manifest.json')['cases']
    folder = checked(air, entry['export']).parent
    exported = _json(folder / 'export.json')
    spec = exported['trace_replay']
    layer = entry['layer']
    name = f'l{layer:02}_c00_gdn_layer_hidden_o0'
    base_inputs, base_refs, base, _ = board_case(baseline, name)
    inputs, refs, board, case = board_case(trace, f'l{layer:02}_c00_gdn_trace_o0')
    if (case['reference_basis'] != BASIS or base_inputs != inputs
            or refs['golden_hidden'].tobytes() != base_refs['golden_hidden'].tobytes()):
        raise ValueError('Baseline and trace must have the same module inputs and original hidden golden')
    fixture_manifest = _json(trace / 'execution/fixture/manifest.json')
    if fixture_manifest['source_evidence']['compile_sha256'] != sha(omc / 'compile.json'):
        raise ValueError('The recovered fixture was prepared with a different compilation')
    for ref in exported['goldens']:
        key = ref['name']
        if read_array(folder, ref).tobytes() != refs[key].tobytes():
            raise ValueError(f'{key}: recovered reference differs from its exported capture')
    valid_equal = board['golden_hidden'][:, :19].tobytes() == base['golden_hidden'][:, :19].tobytes()
    full_equal = board['golden_hidden'].tobytes() == base['golden_hidden'].tobytes()
    print(f'L{layer} CPro hidden vs original board run: valid19 equal bits={valid_equal}, full64={full_equal}')
    if not valid_equal:
        raise ValueError('Tracing changed the valid hidden. These boundaries cannot locate the original failure.')

    def show(name, actual, expected):
        all_rows = metrics(actual, expected)
        last = metrics(actual[:, 18], expected[:, 18])
        def fmt(value):
            return f'{value:.6g}' if value is not None else 'undefined'
        print(f"{name:42s} {fmt(all_rows['relative_l2']):>11} {fmt(last['relative_l2']):>11}"
              f"  {'PASS' if all_rows['passed'] and last['passed'] else 'FAIL'}")

    print('comparison                                  valid19_L2  token18_L2  five_metrics')
    for key in (*TAPS, 'golden_hidden'):
        show('board vs 310P / ' + key, board[key][:, :19], refs[key][:, :19])

    def shaped(d, key):
        return d[key].reshape(1, 19, 32, -1)
    def core(d):
        operands = [shaped(d, n) if n in ('query', 'key', 'value') else d[n] for n in OPERANDS]
        value, _ = compute(*operands, np.zeros((1, 32, 128, 128), np.float32))
        return value

    gamma = read_array(folder, spec['gamma'])
    def norm(d, gates):
        return _gate_norm(shaped(d, 'core'), shaped(gates, 'norm_gate'), gamma, spec['epsilon'], np.float16)

    weight, scale = (read_array(folder, spec[n]) for n in ('weight', 'scale'))
    def project(d):
        x = d['out_proj_input'].reshape(19, 4096)
        quant, token_scale, _ = dynamic_quant(x)
        return group_matmul(quant, token_scale, weight, scale)[1].reshape(1, 19, -1)

    show('310P core vs CPU on 310P GDR inputs', shaped(refs, 'core'), core(refs))
    show('board core vs CPU on board GDR inputs', shaped(board, 'core'), core(board))
    show('310P proj input vs CPU on 310P core/gate', shaped(refs, 'out_proj_input'), norm(refs, refs))
    show('board proj input vs CPU on board core/gate', shaped(board, 'out_proj_input'), norm(board, board))
    show('CPU norm / change core only', norm(board, refs), norm(refs, refs))
    show('CPU norm / change gate only', norm(refs, board), norm(refs, refs))
    show('310P hidden vs CPU on 310P proj input', refs['golden_hidden'][:, :19], project(refs))
    show('board hidden vs CPU on board proj input', board['golden_hidden'][:, :19], project(board))
