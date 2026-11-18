"""Locate the earliest observed change between two retained 310P Prefill fixtures."""
import json
from pathlib import Path

from .common import checked, metrics, read_array, sha
from .qlinear_golden import load_capture


def by_name(specs):
    indexed = {spec['name']: spec for spec in specs}
    if len(indexed) != len(specs):
        raise ValueError('duplicate tensor names in a captured boundary')
    return indexed


def differences(roots, old_specs, new_specs):
    old, new = by_name(old_specs), by_name(new_specs)
    if old.keys() != new.keys():
        raise ValueError(f'tensor names changed: {sorted(old)} vs {sorted(new)}')
    changed = {}
    for name, first in old.items():
        second = new[name]
        descriptors = [(s['dtype'], s['shape']) for s in (first, second)]
        if descriptors[0] != descriptors[1] or first['sha256'] == second['sha256']:
            checked(roots[0], first)
            checked(roots[1], second)
            if descriptors[0] != descriptors[1]:
                changed[name] = dict(passed=False, old_descriptor=descriptors[0], new_descriptor=descriptors[1])
            continue
        score = metrics(read_array(roots[1], second), read_array(roots[0], first))
        changed[name] = {k: score[k] for k in ('passed', 'exact_elements', 'total_elements',
                                               'relative_l2', 'cosine', 'max_abs', 'failed_metrics') if k in score}
    return changed


def weight_comparison(roots, old, new):
    specs = (old.get('weights'), new.get('weights'))
    if not all(specs):
        return 'UNRECORDED' if not any(specs) else 'ONLY_ONE_CAPTURE'
    import torch
    from .prefill_matmul_repeat import equal_bits

    a, b = (torch.load(checked(root, spec), weights_only=True, map_location='cpu')
            for root, spec in zip(roots, specs))
    if a.keys() != b.keys():
        return 'DIFFERENT_WEIGHT_NAMES'
    return {name: equal_bits(a[name], b[name]) for name in a}


def ordered_regions(manifest):
    rows = sorted(manifest['regions'], key=lambda r: r['ordinal'])
    if (len({r['id'] for r in rows}) != len(rows) or
            any(r['ordinal'] != i or r['phase'] != 'prefill' or r['call'] != 0 for i, r in enumerate(rows))):
        raise ValueError('expected contiguous unique single-Prefill observations')
    return rows


def inspect_drift(fixture, check):
    fixture, check = Path(fixture).resolve(strict=True), Path(check).resolve(strict=True)
    repeat = json.loads((check/'report.json').read_text())
    fixtures = (fixture, check/'fixture')
    records = [json.loads((f/'manifest.json').read_text()) for f in fixtures]
    roots = [f/'capture' for f in fixtures]
    if sha(roots[0]/'manifest.json') != repeat['historical_capture_sha256']:
        raise ValueError('repeat report belongs to a different historical capture')
    manifests = [load_capture(root, ['self_attn_v_proj'])[1] for root in roots]
    old, new = map(ordered_regions, manifests)
    if [r['id'] for r in old] != [r['id'] for r in new]:
        raise ValueError('capture order or region coverage differs between the two runs')
    token_specs = [r['tokens'] for r in records]
    for root, spec in zip(fixtures, token_specs):
        checked(root, spec)
    if any(token_specs[0][k] != token_specs[1][k] for k in ('shape', 'dtype', 'sha256')):
        raise ValueError('the two Prefills used different token inputs')
    wrapper = differences(fixtures, records[0]['inputs'], records[1]['inputs'])
    result = dict(status='INSPECTED', order='CAPTURE_ORDINAL', total_regions=len(old),
                  wrapper_input_differences=wrapper, first_changed_boundary=None,
                  first_output_outside_gate=None, observations=[], new_fixture_status=records[1]['status'],
                  attention_module_reference=records[1].get('attention_module_reference'))
    for a, b in zip(old, new):
        if (a['kind'], a['sequence_length']) != (b['kind'], b['sequence_length']):
            raise ValueError(f'{a["id"]}: incompatible region descriptors')
        ins = differences(roots, a['inputs'], b['inputs'])
        outs = differences(roots, a['outputs'], b['outputs'])
        result['scanned_regions'] = a['ordinal'] + 1
        first_change = result['first_changed_boundary'] is None and bool(ins or outs)
        outside = any(not v['passed'] for v in outs.values())
        if first_change or outside:
            row = dict(id=a['id'], ordinal=a['ordinal'], input_differences=ins, output_differences=outs,
                       attrs_equal=a.get('attrs') == b.get('attrs'), recorded_weight_equal_bits=weight_comparison(roots, a, b))
            result['observations'].append(row)
            if first_change:
                result['first_changed_boundary'] = a['id']
            if outside:
                result['first_output_outside_gate'] = a['id']
                break
    return result
