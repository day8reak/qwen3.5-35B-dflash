"""CPU-only propagation of a measured standalone GDR core through the captured gated norm."""
import csv
import hashlib
import tempfile
from pathlib import Path

import numpy as np

from ...common.artifacts import share_for_transfer
from ..layer_probe.common import identity, metrics, read_array, save_json, sha
from .gdn_tail import gated_norm_stages
from .gdr_stage_data import logical


def save_context(case, context, origin):
    z, gamma, eps = context
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError('Missing or invalid gated RMSNorm epsilon')
    destination = case / 'norm-context'
    with tempfile.TemporaryDirectory(dir=case) as tmp:
        folder = Path(tmp) / destination.name
        folder.mkdir()
        meta = dict(bundle_manifest_sha256=sha(case / 'manifest.json'), layer=0, epsilon=float(eps), origin=origin)
        for name, role, value in [('z', 'gate', z), ('gamma', 'gamma', gamma)]:
            value = logical(value, role)
            if value.dtype not in (np.float16, np.float32):
                raise ValueError(f'Unexpected L0 norm {name} dtype {value.dtype}')
            path = folder / (name + '.bin')
            value.tofile(path)
            meta[name] = dict(identity(path, folder), shape=list(value.shape), dtype=value.dtype.name)
        save_json(folder / 'manifest.json', meta)
        share_for_transfer(folder)
        folder.rename(destination)


def ratio(value, base):
    if value is None or base is None or base == 0:
        return None
    return value / base


def stages(core, z, gamma, eps):
    norm, product = gated_norm_stages(core, z, gamma, eps)
    return dict(core_attn=core, rms_output=norm, mul_output=product, gated_return=product.astype(core.dtype))


def replay(recovery, report, context=None):
    from .standalone_gdr import CASE, FIELDS, read

    execution = Path(report.get('recovery_source') or recovery.parent / 'execution')
    fixture = execution / 'fixture'
    if context is None and not (execution / 'request.json').is_file():
        print('CPU gated RMSNorm replay skipped: the original execution request is unavailable')
        return
    request = read(execution / 'request.json')
    if sha(fixture / 'manifest.json') != request['fixture_sha256'] or request['board_attempt'] != report['board_attempt']:
        raise ValueError('Recovered outputs and execution fixture differ')
    meta = read(fixture / 'manifest.json')
    evidence = meta['source_evidence']
    if context is None:
        compiled = execution.parent / 'omc'
        context = compiled / evidence['bundle'] / 'norm-context'
        if not context.exists():
            print('CPU gated RMSNorm replay skipped: use export_gdr_t64.py --add-norm-context CASE_DIR, then report --norm-context CASE_DIR/norm-context')
            return
    ctx = read(context / 'manifest.json')
    if ctx['layer'] != 0 or ctx['bundle_manifest_sha256'] != evidence['manifest_sha256']:
        raise ValueError('Norm context belongs to a different GDR bundle or layer')
    case, = [c for c in meta['cases'] if c['case'] == CASE]
    valid = case['effective_length']
    if valid == 0:
        print('No valid core rows for gated RMSNorm replay')
        return
    z = logical(read_array(context, ctx['z']), 'gate')[:, :valid]
    gamma = logical(read_array(context, ctx['gamma']), 'gamma')
    eps = ctx['epsilon']
    spec = case['outputs'][0]
    if spec['index'] != 0 or spec['semantic_name'] != 'core_attn' or spec['golden_shape'][1] != valid:
        raise ValueError('Unexpected standalone GDR output order or valid length')
    ref = read_array(fixture, dict(path=f'{CASE}/golden_0.bin', dtype=spec['golden_dtype'],
                     shape=spec['golden_shape'], bytes=spec['fixture_bytes'], sha256=spec['fixture_sha256']))
    expected = stages(ref, z, gamma, eps)
    core_row, = [r for r in report['cases'] if r['output_name'] == 'core_attn']
    scopes = [(f'valid{valid}', slice(None)), *((f'token{i}', slice(i, i+1)) for i in range(valid))]
    destination = recovery / 'cpu-tail-accuracy.csv'
    print(f"CPU gated RMSNorm replay: same captured z, gamma, epsilon={eps:g}; {valid} valid tokens")
    print('stage          scope      relative_L2       cosine     L2_vs_core    error_ratio')
    with tempfile.TemporaryDirectory(dir=recovery) as tmp:
        path = Path(tmp) / destination.name
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['stage', 'scope', 'call', *FIELDS,
                                        'relative_l2_vs_core', 'rmse_vs_core', 'relative_l2_vs_previous'])
            writer.writeheader()
            call_hashes = core_row['output_sha256']
            if len(call_hashes) != meta['repetitions']:
                raise ValueError('Incomplete GDR core readback hashes')
            for call, digest in enumerate(call_hashes):
                native = recovery / 'native' / CASE / f'{call}.bin'
                with native.open('rb') as handle:
                    raw = handle.read(spec['bytes'])
                if hashlib.sha256(raw).hexdigest() != digest:
                    raise ValueError(f'{native}: core differs from the recovered comparison')
                actual = np.frombuffer(raw, dtype=spec['dtype']).reshape(spec['shape'])[:, :valid]
                observed = stages(actual, z, gamma, eps)
                for scope, extent in scopes:
                    previous = base = None
                    for stage, tensor in observed.items():
                        score = metrics(tensor[:, extent], expected[stage][:, extent])
                        if base is None:
                            base = previous = score
                        gain = ratio(score['relative_l2'], base['relative_l2'])
                        writer.writerow(dict(call=call, scope=scope, stage=stage,
                            **{k: score[k] for k in FIELDS}, relative_l2_vs_core=gain,
                            rmse_vs_core=ratio(score['rmse'], base['rmse']),
                            relative_l2_vs_previous=ratio(score['relative_l2'], previous['relative_l2'])))
                        if call == 0 and scope in (scopes[0][0], scopes[-1][0]):
                            values = [score['relative_l2'], score['cosine'], gain, score['error_ratio']]
                            print(f'{stage:13} {scope:9}', ' '.join(f'{x:12.6g}' if x is not None else '         n/a' for x in values))
                        previous = score
        path.replace(destination)
    print(destination)
