"""Offline comparison of captured 310P boundaries and explicitly mapped CPro dumps."""
from __future__ import annotations

from collections import Counter
import csv
import json
import math
from pathlib import Path

import numpy as np
from qwen35_moe_dflash.common.accuracy import POLICY, THRESHOLDS, floating_metrics

from .layer_probe.common import KIND, identity, new_output, save_json, sha

PLAN_KIND = 'qwen35-target-dump-comparison-plan-v1'
DTYPES = {name: np.dtype(code) for name, code in {
    'float16': '<f2', 'float32': '<f4', 'float64': '<f8',
    'int8': 'i1', 'uint8': 'u1', 'int16': '<i2', 'uint16': '<u2',
    'int32': '<i4', 'uint32': '<u4', 'int64': '<i8', 'uint64': '<u8', 'bool': '?',
}.items()}


def read_json(path):
    path = Path(path)
    if path.stat().st_size > 32 << 20:
        raise ValueError('metadata exceeds 32 MiB')
    return json.loads(path.read_text())


def prepare(capture, output):
    capture = Path(capture).resolve()
    destination = Path(output).resolve()
    if destination == capture or capture in destination.parents:
        raise ValueError('mapping output must be outside the capture root')
    manifest = read_json(capture / 'manifest.json')
    if manifest.get('artifact_kind') != KIND or manifest.get('status') != 'CAPTURED':
        raise ValueError('requires completed original 310P capture')
    regions = []
    for region in manifest['regions']:
        row = {k: region[k] for k in ('id', 'layer', 'call', 'phase', 'point', 'ordinal')}
        row.update(inputs_complete=False, mapping_evidence='', inputs=[], outputs=[])
        for direction in ('inputs', 'outputs'):
            for tensor in region[direction]:
                row[direction].append(dict(
                    name=tensor['name'],
                    golden=dict(tensor, format='raw', layout='UNVERIFIED'),
                    actual=None, gate={'mode': 'metrics'}))
        regions.append(row)
    plan = dict(artifact_kind=PLAN_KIND, schema_version=1,
        capture_manifest_sha256=sha(capture / 'manifest.json'),
        contract_confirmed=False, contract_evidence='',
        scope='Only listed captured boundaries; not all graph operators or model acceptance',
        regions=regions)
    validate_plan(plan)
    root = new_output(output)
    save_json(root / 'mapping.json', plan)
    return dict(status='MAPPING_REQUIRED', mapping=str(root / 'mapping.json'),
        regions=len(regions), tensors=sum(len(r[d]) for r in regions for d in ('inputs', 'outputs')))


def validate_plan(plan):
    if plan.get('artifact_kind') != PLAN_KIND or plan.get('schema_version') != 1:
        raise ValueError('unsupported comparison plan')
    if type(plan.get('contract_confirmed')) is not bool:
        raise ValueError('contract_confirmed must be boolean')
    if plan['contract_confirmed'] and not str(plan.get('contract_evidence', '')).strip():
        raise ValueError('confirmed contract requires evidence')
    regions = plan.get('regions')
    if not isinstance(regions, list) or not 0 < len(regions) <= 4096:
        raise ValueError('requires 1..4096 regions')
    ids, ordinals = set(), set()
    for row in regions:
        if not isinstance(row.get('id'), str) or not row['id'] or row['id'] in ids:
            raise ValueError('region IDs must be distinct nonempty strings')
        ids.add(row['id'])
        if type(row.get('ordinal')) is not int or row['ordinal'] < 0 or row['ordinal'] in ordinals:
            raise ValueError('captured ordinals must be distinct nonnegative integers')
        ordinals.add(row['ordinal'])
        if type(row.get('inputs_complete')) is not bool:
            raise ValueError('inputs_complete must be boolean')
        if row['inputs_complete'] and not str(row.get('mapping_evidence', '')).strip():
            raise ValueError('complete input mapping requires evidence including state/control coverage')
        for direction in ('inputs', 'outputs'):
            pairs = row.get(direction)
            if not isinstance(pairs, list) or not pairs or len(pairs) > 128:
                raise ValueError('each region requires 1..128 inputs and outputs')
            names = [p['name'] for p in pairs]
            if len(names) != len(set(names)):
                raise ValueError('duplicate tensor names within a direction')


def _shape(value):
    if not isinstance(value, list) or len(value) > 16 or any(type(d) is not int or d < 0 for d in value):
        raise ValueError('shape must contain nonnegative integer dimensions')
    return tuple(value)


def _load(root, spec, identities):
    if not isinstance(spec, dict):
        raise ValueError('tensor specification must be an object')
    relative = Path(spec['path'])
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if relative.is_absolute() or root not in path.parents:
        raise ValueError('tensor path escapes its declared root')
    if not path.is_file():
        raise FileNotFoundError(str(relative))
    if path not in identities:
        identities[path] = identity(path, root)
    observed = identities[path]
    if 'bytes' in spec and spec['bytes'] != observed['bytes']:
        raise ValueError('file size differs from manifest')
    if 'sha256' in spec and spec['sha256'] != observed['sha256']:
        raise ValueError('file hash differs from manifest')
    dtype = DTYPES[spec['dtype']]
    shape = _shape(spec['shape'])
    if spec.get('format') == 'raw':
        if math.prod(shape) * dtype.itemsize != observed['bytes']:
            raise ValueError('raw file extent differs from physical shape/dtype')
        array = np.memmap(path, mode='r', dtype=dtype, shape=shape) if observed['bytes'] else np.empty(shape, dtype)
    elif spec.get('format') == 'npy':
        array = np.load(path, allow_pickle=False, mmap_mode='r')
        if not isinstance(array, np.ndarray):
            if hasattr(array, 'close'):
                array.close()
            raise ValueError('expected one NPY array, not an archive')
        if array.dtype != dtype or array.shape != shape:
            raise ValueError('NPY header differs from declared shape/dtype')
    else:
        raise ValueError('supported formats are raw (headerless little-endian) and npy')
    if spec.get('layout') not in ('logical', 'ND'):
        raise ValueError('layout requires a verified external decoder; no NZ or packed-INT4 guessing')
    view = spec.get('view', {})
    if view:
        if not str(spec.get('view_evidence', '')).strip():
            raise ValueError('explicit view requires mapping evidence')
        if not set(view) <= {'permute', 'slice'}:
            raise ValueError('only explicit permute and slice views are supported')
        if 'permute' in view:
            axes = view['permute']
            if any(type(x) is not int for x in axes) or sorted(axes) != list(range(array.ndim)):
                raise ValueError('invalid axis permutation')
            array = array.transpose(axes)
        if 'slice' in view:
            bounds = view['slice']
            if len(bounds) != array.ndim:
                raise ValueError('slice must specify every permuted dimension')
            slices = []
            for size, bound in zip(array.shape, bounds):
                if (not isinstance(bound, list) or len(bound) != 2 or
                    any(type(x) is not int for x in bound) or not 0 <= bound[0] <= bound[1] <= size):
                    raise ValueError('slice bounds outside physical extent')
                slices.append(slice(*bound))
            array = array[tuple(slices)]
    return array, observed


def _gate(gate):
    mode = gate.get('mode', 'metrics')
    if mode not in ('metrics', 'allclose', 'bitwise'):
        raise ValueError('gate must be metrics, allclose or bitwise')
    atol = rtol = 0.0
    if mode == 'allclose':
        atol, rtol = float(gate['atol']), float(gate['rtol'])
        if not all(math.isfinite(v) and v >= 0 for v in (atol, rtol)):
            raise ValueError('atol/rtol must be finite and nonnegative')
    return mode, atol, rtol


def tensor_metrics(actual, golden, gate, chunk=65536):
    if actual.shape != golden.shape:
        raise ValueError('logical shapes differ after explicit views')
    if actual.dtype.kind not in 'biuf' or golden.dtype.kind not in 'biuf':
        raise ValueError('only real numeric tensors are supported')
    mode, atol, rtol = _gate(gate)
    integer = actual.dtype.kind in 'biu' and golden.dtype.kind in 'biu'
    if (actual.dtype.kind in 'biu') != (golden.dtype.kind in 'biu'):
        raise ValueError('integer/float comparison requires explicit semantic decoding')
    if integer and actual.dtype != golden.dtype:
        raise ValueError('integer dtype mismatch; implicit promotion is not a verified mapping')
    if integer:
        mode = 'exact_integer'  # Float tolerance never relaxes an index/control comparison.
    same_dtype = actual.dtype == golden.dtype
    if mode == 'bitwise' and not same_dtype:
        raise ValueError('bitwise comparison requires identical dtypes')
    total = actual.size
    result = dict(shape=list(actual.shape), actual_dtype=str(actual.dtype),
        golden_dtype=str(golden.dtype), dtype_equal=same_dtype, total_elements=total, gate=mode,
        exact_mismatch_count=0, nonfinite_actual=0, nonfinite_golden=0,
        max_abs=None, rmse=None, relative_l2=None, cosine=None,
        first_mismatch_coordinates=[], max_error_coordinate=None)
    bad_count, bit_bad_count, offset, maximum = 0, 0, 0, -1.
    for av, gv in np.nditer([actual, golden], flags=['external_loop', 'buffered', 'zerosize_ok'],
                           op_flags=[['readonly'], ['readonly']], order='C', buffersize=chunk):
        equal = av == gv  # Preserve int64/uint64 precision for exact comparison.
        result['exact_mismatch_count'] += int(np.count_nonzero(~equal))
        fa, fg = np.isfinite(av), np.isfinite(gv)
        finite = fa & fg
        result['nonfinite_actual'] += int(np.count_nonzero(~fa))
        result['nonfinite_golden'] += int(np.count_nonzero(~fg))
        if same_dtype:
            a_bytes = np.ascontiguousarray(av).view(np.uint8).reshape(-1, av.dtype.itemsize)
            g_bytes = np.ascontiguousarray(gv).view(np.uint8).reshape(-1, gv.dtype.itemsize)
            bit_bad = np.any(a_bytes != g_bytes, axis=1)
            bit_bad_count += int(np.count_nonzero(bit_bad))
        if not integer:
            af, gf = av.astype(np.float64), gv.astype(np.float64)
            with np.errstate(over='ignore', invalid='ignore'):
                delta = af - gf
                absdiff = np.abs(delta)
            if finite.any():
                j = int(np.argmax(np.where(finite, absdiff, -1)))
                if float(absdiff[j]) > maximum:
                    maximum = float(absdiff[j])
                    result.update(max_error_coordinate=[int(i) for i in np.unravel_index(offset+j, actual.shape)],
                        actual_at_max=float(af[j]), golden_at_max=float(gf[j]))
        if mode == 'bitwise':
            bad = bit_bad | ~finite
        elif mode == 'allclose':
            with np.errstate(over='ignore', invalid='ignore'):
                if np.any(finite & ~np.isfinite(absdiff)):
                    raise ValueError('floating difference overflow; no tolerance verdict')
                bad = ~finite | (absdiff > atol + rtol * np.abs(gf))
        else:
            bad = ~equal | ~finite
        bad_count += int(np.count_nonzero(bad))
        needed = 8 - len(result['first_mismatch_coordinates'])
        for index in np.flatnonzero(bad)[:needed]:
            result['first_mismatch_coordinates'].append(
                [int(i) for i in np.unravel_index(offset + int(index), actual.shape)])
        offset += av.size
    nonfinite = result['nonfinite_actual'] + result['nonfinite_golden']
    result['exact_values_equal'] = bool(total and not result['exact_mismatch_count'] and not nonfinite)
    result['bitwise_equal'] = bool(total and same_dtype and not bit_bad_count and not nonfinite)
    result['bitwise_mismatch_count'] = bit_bad_count if same_dtype else None
    result['gate_mismatch_count'] = bad_count if mode != 'metrics' else None
    if not integer:
        result.update(floating_metrics(actual, golden, chunk=chunk))
        result['gate'] = POLICY if mode == 'metrics' else POLICY+'+'+mode
    # Overflowing metric reductions remain unavailable, never emitted as nonstandard JSON.
    for key, value in list(result.items()):
        if isinstance(value, float) and not math.isfinite(value):
            result[key] = None
    if not total:
        result['status'] = 'EMPTY_TENSOR'
    elif nonfinite:
        result['status'] = 'FAILED'
    else:
        failed = (not integer and not result['reference_metrics_passed']) or (mode != 'metrics' and bad_count)
        result['status'] = 'FAILED' if failed else 'PASSED'
    if mode == 'allclose':
        result.update(atol=atol, rtol=rtol)
    return result


def compare_pair(pair, golden_root, actual_root, identities):
    result = dict(name=pair['name'])
    if pair.get('actual') is None:
        return dict(result, status='UNMAPPED')
    try:
        golden, gi = _load(golden_root, pair['golden'], identities)
        actual, ai = _load(actual_root, pair['actual'], identities)
        result.update(golden=gi, actual=ai)
        result.update(tensor_metrics(actual, golden, pair.get('gate', {})))
    except FileNotFoundError as exc:
        result.update(status='MISSING_FILE', error=str(exc))
    except (ValueError, KeyError, TypeError, OSError, OverflowError) as exc:
        result.update(status='INVALID_TENSOR', error=f'{type(exc).__name__}: {exc}')
    return result


def classify(row, contract):
    ins, outs = row['inputs'], row['outputs']
    complete = row['inputs_complete'] and all(
        r['status'] in ('PASSED', 'FAILED', 'METRICS_RECORDED') for r in ins)
    if any(r['status'] == 'FAILED' for r in outs):
        if not contract or not complete:
            return 'OUTPUT_FAILURE_INPUTS_OR_CONTRACT_UNVERIFIED'
        if all(r.get('bitwise_equal') for r in ins):
            return 'LOCAL_CANDIDATE_EXACT_INPUTS'
        return 'OUTPUT_FAILURE_WITH_INPUT_DIFFERENCES'
    if any(r['status'] not in ('PASSED', 'FAILED', 'METRICS_RECORDED') for r in ins + outs):
        return 'INCOMPLETE'
    if any(r['status'] == 'FAILED' for r in ins):
        return 'INPUT_FAILURE'
    if not contract or not complete:
        return 'CONTRACT_OR_INPUT_COVERAGE_UNVERIFIED'
    if all(r['status'] == 'PASSED' for r in ins + outs):
        return 'MAPPED_BOUNDARY_PASSED'
    return 'METRICS_RECORDED'


def compare(plan_path, golden_root, actual_root, output):
    plan = read_json(plan_path)
    validate_plan(plan)
    golden_root, actual_root = Path(golden_root).resolve(), Path(actual_root).resolve()
    if not golden_root.is_dir() or not actual_root.is_dir():
        raise ValueError('both dump roots must exist')
    if sha(golden_root / 'manifest.json') != plan['capture_manifest_sha256']:
        raise ValueError('capture manifest differs from mapping plan')
    # The plan may select regions but must not invent a captured region or alter its golden descriptors.
    capture = read_json(golden_root / 'manifest.json')
    source = {r['id']: r for r in capture['regions']}
    for region in plan['regions']:
        original = source[region['id']]
        for key in ('layer', 'call', 'phase', 'point', 'ordinal'):
            if region[key] != original[key]:
                raise ValueError('captured region metadata changed in plan')
        for direction in ('inputs', 'outputs'):
            expected = {t['name']: t for t in original[direction]}
            if {p['name'] for p in region[direction]} != set(expected):
                raise ValueError('all captured region ports must remain listed')
            for pair in region[direction]:
                for key, value in expected[pair['name']].items():
                    if pair['golden'].get(key) != value:
                        raise ValueError('golden descriptor differs from capture')
    destination = Path(output).resolve()
    if any(destination == p or p in destination.parents for p in (golden_root, actual_root)):
        raise ValueError('comparison output must be outside both dump roots')
    identities, rows = {}, []
    for region in sorted(plan['regions'], key=lambda x: x['ordinal']):
        row = {k: region[k] for k in ('id', 'layer', 'call', 'phase', 'point', 'ordinal', 'inputs_complete')}
        row['mapping_evidence'] = region.get('mapping_evidence', '')
        for direction in ('inputs', 'outputs'):
            row[direction] = [compare_pair(p, golden_root, actual_root, identities) for p in region[direction]]
        row['diagnosis'] = classify(row, plan['contract_confirmed'])
        rows.append(row)
    tensors = [r for row in rows for d in ('inputs', 'outputs') for r in row[d]]
    counts = dict(Counter(r['status'] for r in tensors))
    incomplete = sum(v for k, v in counts.items() if k not in ('PASSED', 'FAILED', 'METRICS_RECORDED'))
    status = ('PARTIAL' if incomplete else 'FAILED' if counts.get('FAILED') else
              'MAPPED_BOUNDARIES_PASSED' if all(r['diagnosis'] == 'MAPPED_BOUNDARY_PASSED' for r in rows)
              else 'METRICS_RECORDED')
    report = dict(artifact_kind=PLAN_KIND + '-result', status=status,
        plan_sha256=sha(plan_path), capture_manifest_sha256=plan['capture_manifest_sha256'],
        contract_confirmed=plan['contract_confirmed'], contract_evidence=plan['contract_evidence'],
        region_count=len(rows), omitted_capture_regions=len(source) - len(rows), tensor_status_counts=counts,
        first_observed_output_failure=next((r['id'] for r in rows if any(t['status'] == 'FAILED' for t in r['outputs'])), None),
        local_candidates=[r['id'] for r in rows if r['diagnosis'] == 'LOCAL_CANDIDATE_EXACT_INPUTS'],
        ordering_scope='Original capture ordinal, not verified compiler topology or proof of root cause',
        validation_scope='Explicitly mapped regions only; no model execution, repeat stability or full-model PASS',
        regions=rows, numerical_policy=POLICY, thresholds=THRESHOLDS)
    root = new_output(output)
    save_json(root / 'report.json', report)
    with (root / 'summary.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        keys = ['status', 'gate', 'max_abs', 'rmse', 'relative_l2', 'cosine', 'kl_divergence', 'mean_relative_error',
                'failed_metrics', 'exact_mismatch_count', 'bitwise_mismatch_count',
                'nonfinite_actual', 'nonfinite_golden', 'error']
        writer.writerow(['region', 'layer', 'call', 'phase', 'direction', 'tensor', 'diagnosis'] + keys)
        for row in rows:
            for direction in ('inputs', 'outputs'):
                for tensor in row[direction]:
                    writer.writerow([row['id'], row['layer'], row['call'], row['phase'], direction,
                                     tensor['name'], row['diagnosis']] + [tensor.get(k) for k in keys])
    return dict(status=status, report=str(root / 'report.json'), csv=str(root / 'summary.csv'),
                tensor_status_counts=counts, local_candidates=report['local_candidates'],
                first_observed_output_failure=report['first_observed_output_failure'])
