"""Locate new-KV regressions in a retained candidate replay, without execution."""
from pathlib import Path

import numpy as np

from qwen35_moe_dflash.common.artifacts import checked_file
from qwen35_moe_dflash.common.hashing import sha256_file
from qwen35_moe_dflash.execution.runtime_config import read_json
from .accuracy import analyze_result, metrics, read_snapshot
from .accuracy_diagnose import _controls, initial_call_comparison, tensor_difference
from .kv_row_audit import RowAudit


def region_details(actual, golden, start, atol, rtol):
    """ND [batch, head, appended row, channel] coordinates; original gates."""
    if actual.shape != golden.shape or actual.ndim != 4 or not actual.size:
        raise ValueError('KV diagnosis requires equal nonempty rank-four regions')
    a, g = actual.astype(np.float64), golden.astype(np.float64)
    finite = bool(np.isfinite(a).all() and np.isfinite(g).all())
    missing = (a == 0) & (g != 0)
    detail = dict(elements=int(a.size), zero_elements=int(np.count_nonzero(a == 0)),
        reference_zero_elements=int(np.count_nonzero(g == 0)),
        zero_where_reference_nonzero=int(missing.sum()), nonfinite_elements=int((~np.isfinite(a)).sum()),
        reference_nonfinite_elements=int((~np.isfinite(g)).sum()))
    if not finite:
        return detail
    delta = np.abs(a-g); failed = delta > atol+rtol*np.abs(g)
    point = tuple(int(i) for i in np.unravel_index(int(delta.argmax()), delta.shape))
    coordinate = list(point); coordinate[2] += start
    detail.update(failed_elements=int(failed.sum()),
        worst_element=dict(coordinate=coordinate, actual=float(a[point]), reference=float(g[point]),
                           abs_error=float(delta[point]), tolerance=float(atol+rtol*abs(g[point]))))
    errors = np.linalg.norm(a-g, axis=-1); norms = np.linalg.norm(g, axis=-1)
    relative = np.divide(errors, norms, out=np.zeros_like(errors), where=norms != 0)
    ranking = np.where(norms != 0, relative, errors)
    rows = []
    for flat in np.argsort(ranking.reshape(-1), kind='stable')[-3:][::-1]:
        p = tuple(int(i) for i in np.unravel_index(int(flat), norms.shape))
        rows.append(dict(batch=p[0], head=p[1], cache_position=start+p[2],
            relative_l2_error=float(relative[p]) if norms[p] else None,
            absolute_l2_error=float(errors[p]), reference_l2_norm=float(norms[p]),
            failed_elements=int(failed[p].sum()), zero_where_reference_nonzero=int(missing[p].sum())))
    detail['worst_rows'] = rows
    return detail


def _load(path):
    path = Path(path).resolve(); path = path/'result.json' if path.is_dir() else path
    analysis = analyze_result(path)
    result = read_json(path, 4 << 20)
    fixture = (path.parent/Path(result['fixture'])).resolve()
    meta = read_json(fixture/'manifest.json')
    if len(meta['cases']) > 64 or meta['cache_capacity'] > 512:
        raise ValueError('KV diagnosis requires bounded standalone accuracy fixtures')
    return path, result, fixture, meta, analysis


def _same_inputs(fixture, meta, baseline, old):
    for key in ('cases', 'context_rows', 'cache_capacity', 'max_draft_tokens', 'comparison', 'outputs', 'host_embedding'):
        if meta.get(key) != old.get(key):
            raise ValueError('candidate and baseline replay contract differs: '+key)
    inventories = []
    for directory, manifest in ((fixture, meta), (baseline, old)):
        _controls(directory, manifest)
        inventory = {}
        for r in manifest['input_files']:
            if (type(r['case']) is not int or not 0 <= r['case'] < len(manifest['cases'])
                    or type(r['index']) is not int or not 0 <= r['index'] < len(manifest['inputs'])
                    or r['path'] != f"inputs/{r['case']}/{r['index']}.bin"):
                raise ValueError('retained input case, slot or path is inconsistent')
            spec = manifest['inputs'][r['index']]
            key = (r['case'], spec['name'])
            if key in inventory or r['bytes'] != spec['bytes']:
                raise ValueError('duplicate or inconsistent retained KV input record')
            checked_file(directory, r)
            inventory[key] = (spec['dtype'], spec['shape'], r['bytes'], r['sha256'])
        if len(inventory) != len(manifest['cases'])*len(manifest['inputs']):
            raise ValueError('retained input inventory is incomplete')
        inventories.append(inventory)
    if inventories[0] != inventories[1]:
        raise ValueError('candidate replay inputs differ from baseline by semantic name')
    for step in meta['cases']:
        names = [f'kv_{j}' for j in range(12)]+(['top1', 'hidden'] if step['proposals'] else [])
        for name in names:
            relative = f'golden/{step["index"]}/{name}.bin'
            if sha256_file(fixture/relative) != sha256_file(baseline/relative):
                raise ValueError('candidate golden differs from original: '+relative)


def diagnose_kv(path):
    path, result, fixture, meta, analysis = _load(path)
    if not result.get('replay_source') or not meta.get('candidate_replay'):
        raise ValueError('KV regression diagnosis requires a retained candidate replay and its original result')
    old_path, old_result, baseline, old, old_analysis = _load(path.parent/Path(result['replay_source']))
    if (result.get('baseline_model') != old['model']
            or meta['candidate_replay'].get('baseline_model') != old['model']
            or meta['candidate_replay'].get('source_fixture_sha256') != sha256_file(baseline/'manifest.json')
            or result.get('output_initialization') != old_result.get('output_initialization')):
        raise ValueError('baseline identity or output initialization differs')
    _same_inputs(fixture, meta, baseline, old)
    atol, rtol = (meta['comparison'][k] for k in ('atol', 'rtol'))
    rows = []
    audit = RowAudit(meta['cases'], atol, rtol, meta['context_rows'])
    for step in meta['cases']:
        i, start, valid = (step[k] for k in ('index', 'start', 'valid'))
        if not valid:
            continue
        current = {mode: read_snapshot(path.parent/'native'/f'{mode}-{i}.bin', meta['outputs'])
                   for mode in ('isolated', 'feedback')}
        previous = {mode: read_snapshot(old_path.parent/'native'/f'{mode}-{i}.bin', old['outputs'])
                    for mode in current}
        goldens = [np.fromfile(fixture/f'golden/{i}/kv_{j}.bin', dtype='<f2').reshape(spec['shape'])
                   for j, spec in enumerate(meta['outputs'][2:])]
        audit.observe(step, current, previous, goldens, meta['outputs'])
        for j, spec in enumerate(meta['outputs'][2:]):
            expected = goldens[j][:,:,start:start+valid]
            regions = {mode: values[j+2][:,:,start:start+valid] for mode, values in current.items()}
            between = tensor_difference(regions['isolated'], regions['feedback'])
            for mode, actual in regions.items():
                before = previous[mode][j+2][:,:,start:start+valid]
                measured, original = metrics(actual, expected, atol, rtol), metrics(before, expected, atol, rtol)
                delta = (measured['relative_l2_error']-original['relative_l2_error']
                         if measured.get('relative_l2_error') is not None and original.get('relative_l2_error') is not None else None)
                rows.append(dict(mode=mode, index=i, label=step['label'], start=start, valid=valid,
                    call_ordinal_zero_based=i+(len(meta['cases']) if mode == 'feedback' else 0),
                    tensor=spec['name'], layer=j//2, kind='K' if j%2 == 0 else 'V',
                    candidate=measured, baseline=original, relative_l2_increase=delta,
                    details=region_details(actual, expected, start, atol, rtol),
                    candidate_vs_baseline=tensor_difference(actual, before), between_modes=between))
    def score(r):
        return float('inf') if r['candidate'].get('nonfinite') else r['candidate'].get('relative_l2_error') or 0.
    worst = {mode: max((r for r in rows if r['mode'] == mode), key=score, default=None)
             for mode in ('isolated', 'feedback')}
    regressions = sorted((r for r in rows if r['relative_l2_increase'] is not None and r['relative_l2_increase'] > 0),
                         key=lambda r:r['relative_l2_increase'], reverse=True)
    keys = ('mode', 'index', 'label', 'tensor', 'start', 'valid', 'candidate', 'baseline', 'relative_l2_increase')
    return dict(artifact_kind='qwen35-retained-kv-regression-diagnosis', status='ANALYZED',
        numerical_validation=analysis['numerical_validation'], model_execution='NOT_RUN', model_compilation='NOT_RUN',
        candidate_model=meta['model'], baseline_model=old['model'], board_attempt=result.get('board_attempt'),
        input_verification='ALL_RETAINED_INPUT_HASHES_MATCH_BY_SEMANTIC_NAME; ORIGINAL_GOLDEN_BYTES_MATCH',
        scope='ALL_APPENDED_KV_REGIONS_IN_BOTH_MODES; ORIGINAL_GATES_UNCHANGED',
        summary=dict(compared_regions=len(rows), worst_by_mode=worst,
            nonfinite_regions=[{k:r[k] for k in keys} for r in rows if r['candidate'].get('nonfinite')],
            failed_zero_reference_regions=[{k:r[k] for k in keys} for r in rows
                if not r['candidate']['passed'] and not r['candidate'].get('nonfinite')
                and r['candidate'].get('relative_l2_error') is None],
            largest_regressions=[{k:r[k] for k in keys} for r in regressions[:8]],
            candidate_case_counts=analysis['case_counts'], baseline_case_counts=old_analysis['case_counts']),
        initial_call_comparison=initial_call_comparison(fixture, path.parent/'native', meta), rows=rows,
        row_audit=audit.report(fixture, meta),
        limitations=['Retained inputs are not device input readback.',
            'Later feedback calls reuse device KV; hidden is not a same-input mode comparison.',
            'Source new-context KV rows have no mathematical dependence on past KV; packed/fused execution may differ.',
            'Zero/mode/location patterns cannot alone prove compute, input visibility or output visibility causes.'])
