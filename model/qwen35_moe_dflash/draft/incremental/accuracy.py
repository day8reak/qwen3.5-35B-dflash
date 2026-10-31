"""Deterministic multi-step accuracy fixtures, independent of GE graph evaluation."""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np

from qwen35_moe_dflash.common.hashing import sha256_file
from qwen35_moe_dflash.execution.runtime_config import read_json
from .runtime import validate_abi

KIND = 'qwen35-incremental-draft-accuracy'
CONTRACT = 'torch-prefix-recompute-and-context-kv-fp16-v1'


def cases(context, capacity, candidates):
    """One growing committed prefix; repeated/rejected proposals never append KV."""
    rows = []
    cursor = 0
    def add(valid, count, label, anchor=17):
        nonlocal cursor
        rows.append(dict(index=len(rows), start=cursor, valid=valid, anchor=anchor,
                         proposals=count, label=label))
        cursor += valid
    add(0, candidates, 'empty-context')
    add(context, 0, 'initial-catchup')
    add(min(2, capacity-cursor), candidates, 'partial-context')
    add(0, candidates, 'identical-repeat')
    for count in sorted({min(k, candidates) for k in (1, 3, 7)}):
        add(0, count, 'rejected-proposals-new-anchor', 31+count)
    while cursor < capacity:
        valid = min(context, capacity-cursor)
        add(valid, candidates if cursor+valid == capacity else 0,
            'capacity-boundary' if cursor+valid == capacity else 'context-growth')
    add(0, candidates, 'full-cache-repeat')
    return rows


def prefix_outputs(graph, features, anchor, count):
    """Original cache-free Torch modules; inactive proposal rows are not evaluated."""
    import torch
    from qwen35_moe_dflash.ops import _rotate_half
    draft, c = graph.draft, graph.config
    length = features.shape[1]
    positions = torch.arange(length, dtype=torch.long).reshape(1, -1)
    projected = draft.hidden_norm(draft.fc(features))
    cosine, sine = draft.rotary(positions, features.dtype)
    banks = []
    for layer in draft.layers:
        attention = layer.self_attn
        key = attention.k_norm(attention.k_proj(projected).reshape(
            1, length, c.num_key_value_heads, c.head_dim)).transpose(1, 2)
        key = key*cosine.unsqueeze(1) + _rotate_half(key)*sine.unsqueeze(1)
        value = attention.v_proj(projected).reshape(
            1, length, c.num_key_value_heads, c.head_dim).transpose(1, 2)
        banks.extend((key, value))
    if not count:
        return None, None, banks
    block = torch.tensor([[anchor] + [c.mask_token_id]*count], dtype=torch.long)
    noise = draft.embed_block(block, graph.embedding.weight)
    positions = torch.arange(length+count+1, dtype=torch.long).reshape(1, -1)
    hidden = draft(features, noise, positions)[:, 1:, :]
    top1 = graph.lm_head(hidden).argmax(-1).to(torch.int32)
    return top1, hidden, banks


def _write(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.ascontiguousarray(array)
    if array.dtype.kind == 'f' and not np.isfinite(array).all():
        raise ValueError('nonfinite reference: ' + str(path))
    array.tofile(path)


def prepare(compiled, source, air, output, *, seed=17, atol=0.01, rtol=0.01):
    from qwen35_moe_dflash.draft.reference import load_draft_reference
    if any(not math.isfinite(v) or v < 0 for v in (atol, rtol)):
        raise ValueError('atol/rtol must be finite and nonnegative')
    root, source, air, out = map(Path, (compiled, source, air, output))
    meta = read_json(root / 'manifest.json')
    geometry = validate_abi(meta)
    c, capacity, k = (geometry[n] for n in ('context_rows', 'cache_capacity', 'max_draft_tokens'))
    schedule = cases(c, capacity, k)
    if capacity > 512 or len(schedule) > 64:
        raise ValueError('standalone diagnostic supports capacity<=512 and at most 64 steps')
    out.mkdir(parents=True, exist_ok=True)
    required = len(schedule)*(2*sum(r['bytes'] for r in meta['inputs'])
                               + 6*sum(r['bytes'] for r in meta['outputs'])) + (256 << 20)
    if shutil.disk_usage(out).free < required:
        raise OSError(f'accuracy fixture and retrieval need {required} free host bytes')
    if (meta.get('status') != 'DRAFT_INCREMENTAL_OMC_COMPILED_ABI_BOUND'
            or meta.get('compiled_precision') != 'PASSED' or meta.get('compiled_abi') != 'PASSED'):
        raise ValueError('accuracy requires the completed incremental OMC')
    # Explicit numerical diagnosis binds the reference to the original prepared
    # weights. No dependency on an old smoke or its absolute paths is introduced.
    export = read_json(air / 'manifest.json')
    if sha256_file(air / 'manifest.json') != meta['export_manifest']['sha256']:
        raise ValueError('AIR manifest does not match this compiled OMC')
    if sha256_file(source / 'prepared.json') != export['source_prepared']['sha256']:
        raise ValueError('reference preparation does not match the compiled source weights')
    if read_json(source / 'prepared.json')['draft_config'] != meta['draft_config']:
        raise ValueError('reference/model configurations differ')
    model = root / meta['model']['path']
    if model.stat().st_size != meta['model']['bytes'] or sha256_file(model) != meta['model']['sha256']:
        raise ValueError('accuracy OMC differs from its compilation receipt')
    print('draft-accuracy: loading the independent CPU reference and prepared FP16 weights', flush=True)
    graph = load_draft_reference(source)
    import torch
    rng = np.random.default_rng(seed)
    all_features = rng.normal(0, 0.2, (1, capacity, graph.config.feature_size)).astype('<f2')
    cache_shape = (1, graph.config.num_key_value_heads, capacity, graph.config.head_dim)
    bank = [np.zeros(cache_shape, dtype='<f2') for _ in range(12)]
    input_files = []
    rows = ['QWEN35_INCREMENTAL_ACCURACY 1', f'geometry {c} {capacity} {k}']
    for direction in ('inputs', 'outputs'):
        for spec in meta[direction]:
            rows.append(' '.join((direction[:-1], str(spec['index']), spec['name'], spec['dtype'],
                                  ','.join(map(str, spec['shape'])))))
    out.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for step in schedule:
            i, start, valid, count = (step[n] for n in ('index', 'start', 'valid', 'proposals'))
            end = start+valid
            print(f'draft-accuracy: reference step {i+1}/{len(schedule)} context={end} proposals={count}', flush=True)
            folder = out / 'inputs' / str(i)
            feature = np.zeros((1, c, graph.config.feature_size), dtype='<f2')
            feature[:, :valid] = all_features[:, start:end]
            values = dict(target_features=feature, **{
                name: np.asarray([step[key]], dtype='<i4') for name, key in (
                    ('start_position', 'start'), ('valid_rows', 'valid'), ('anchor', 'anchor'),
                    ('proposal_count', 'proposals'))})
            values.update({f'past_{kind}_{layer}': bank[2*layer+j]
                           for layer in range(6) for j, kind in enumerate(('k', 'v'))})
            for spec in meta['inputs']:
                path = folder / f"{spec['index']}.bin"
                _write(path, values[spec['name']])
                input_files.append(dict(case=i, index=spec['index'], path=str(path.relative_to(out)),
                                        bytes=path.stat().st_size, sha256=sha256_file(path)))
            top1, hidden, prefix = prefix_outputs(graph, torch.from_numpy(all_features[:, :end]),
                                                   step['anchor'], count)
            if count:
                _write(out / 'golden' / str(i) / 'top1.bin', top1.numpy())
                _write(out / 'golden' / str(i) / 'hidden.bin', hidden.numpy())
            for j in range(12):
                bank[j][:, :, start:end] = prefix[j].numpy()[:, :, start:end]
                _write(out / 'golden' / str(i) / f'kv_{j}.bin', bank[j])
            rows.append(f"case {i} {start} {valid} {step['anchor']} {count}")
    (out / 'plan.txt').write_text('\n'.join(rows)+'\n')
    result = dict(schema_version=1, artifact_kind=KIND, status='REFERENCE_PREPARED',
        device_execution='NOT_RUN', model_compilation='NOT_RUN', reference_contract=CONTRACT,
        reference_scope='SYNTHETIC_FEATURE_COMPUTATIONAL_ACCURACY_NOT_TARGET_OR_LANGUAGE_QUALITY',
        model=dict(path=str(model.resolve()), **{n: meta['model'][n] for n in ('bytes', 'sha256')}),
        inputs=meta['inputs'], outputs=meta['outputs'], input_files=input_files, cases=schedule, seed=seed, **geometry,
        comparison=dict(atol=atol, rtol=rtol, top1_mismatches=0,
            cache_preservation='BIT_EXACT_OUTSIDE_WRITE_INTERVAL'),
        profile_case=2, ddk_id=meta['ddk_id'])
    (out / 'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def metrics(actual, golden, atol, rtol):
    a, g = actual.astype(np.float64), golden.astype(np.float64)
    finite = bool(np.isfinite(a).all() and np.isfinite(g).all())
    if not finite:
        return dict(passed=False, nonfinite=True)
    delta = a-g
    norm = float(np.linalg.norm(g))
    return dict(passed=bool(np.all(np.abs(delta) <= atol+rtol*np.abs(g))),
        max_abs_error=float(np.max(np.abs(delta), initial=0)),
        rmse=float(np.sqrt(np.mean(delta*delta))) if delta.size else 0.,
        relative_l2_error=float(np.linalg.norm(delta)/norm) if norm else None)


def read_snapshot(path, outputs):
    raw = Path(path).read_bytes()
    if len(raw) != sum(s['bytes'] for s in outputs):
        raise ValueError('incomplete native output snapshot: '+str(path))
    values, offset = [], 0
    for spec in outputs:
        dt = '<f2' if spec['dtype'] == 'float16' else '<i4'
        values.append(np.frombuffer(raw, dtype=dt, count=math.prod(spec['shape']),
                                    offset=offset).reshape(spec['shape']))
        offset += spec['bytes']
    return values


def compare(fixture, native):
    fixture, native = Path(fixture), Path(native)
    meta = read_json(fixture / 'manifest.json')
    validate_abi(meta)
    report = read_json(native / 'result.json')
    if report.get('status') != 'EXECUTED' or report.get('evidence_kind') != 'device-hiai':
        raise ValueError('a completed real HIAI run is required for device accuracy')
    rows, timings, repeats = [], report['calls'], []
    expected_keys = {(mode, s['index']) for mode in ('isolated', 'feedback') for s in meta['cases']}
    if len(timings) != len(expected_keys) or {(r['mode'], r['index']) for r in timings} != expected_keys:
        raise ValueError('native calls do not cover both complete validation sequences')
    for row in timings:
        step = meta['cases'][row['index']]
        if (row.get('commit_readback_matches') is not True
                or row.get('commit_bytes') != step['valid']*12*8*128*2
                or type(row.get('api_us')) is not int or row['api_us'] < 0):
            raise ValueError('native commit/readback or timing evidence is incomplete')
    atol, rtol = (meta['comparison'][n] for n in ('atol', 'rtol'))
    for mode in ('isolated', 'feedback'):
        previous = None
        for step in meta['cases']:
            i, start, valid, count = (step[n] for n in ('index', 'start', 'valid', 'proposals'))
            values = read_snapshot(native / f'{mode}-{i}.bin', meta['outputs'])
            result = dict(mode=mode, index=i, label=step['label'], passed=True, tensors=[])
            if count:
                ids = np.fromfile(fixture / 'golden' / str(i) / 'top1.bin', dtype='<i4')
                got = values[0].reshape(-1)[:count]
                result['top1_mismatches'] = int(np.count_nonzero(got != ids))
                result['actual_top1'], result['expected_top1'] = got.tolist(), ids.tolist()
                result['passed'] &= result['top1_mismatches'] == 0
                golden = np.fromfile(fixture / 'golden' / str(i) / 'hidden.bin', dtype='<f2').reshape(1,count,-1)
                row = dict(name='draft_hidden', **metrics(values[1][:,:count],golden,atol,rtol))
                result['tensors'].append(row); result['passed'] &= row['passed']
            for j, got in enumerate(values[2:]):
                spec = meta['outputs'][j+2]
                golden = np.fromfile(fixture / 'golden' / str(i) / f'kv_{j}.bin', dtype='<f2').reshape(got.shape)
                old_index = next(r['index'] for r in meta['inputs'] if r['name'] == spec['name'].replace('present_', 'past_'))
                old = (np.fromfile(fixture / 'inputs' / str(i) / f'{old_index}.bin', dtype='<f2').reshape(got.shape)
                       if mode == 'isolated' or previous is None else previous[j+2])
                outside = np.ones(got.shape[2], dtype=bool); outside[start:start+valid] = False
                unchanged = bool(np.array_equal(got.view('<u2')[:,:,outside], old.view('<u2')[:,:,outside]))
                row = dict(name=spec['name'], **metrics(got,golden,atol,rtol), preserved=unchanged)
                # Whole-bank errors can be diluted by zeros or golden history.
                # Additional regions describe the error; the frozen gate is unchanged.
                row['regions'] = dict(
                    appended=(metrics(got[:,:,start:start+valid],golden[:,:,start:start+valid],atol,rtol)
                              if valid else None),
                    committed=(metrics(got[:,:,:start+valid],golden[:,:,:start+valid],atol,rtol)
                               if start+valid else None))
                row['passed'] &= unchanged
                result['tensors'].append(row); result['passed'] &= row['passed']
            if mode == 'feedback' and previous is not None and step['label'] in ('identical-repeat','full-cache-repeat'):
                repeats.append(dict(index=i, label=step['label'],
                    scope='SAME_COMMITTED_CONTEXT_AND_PROPOSALS_ZERO_APPEND_FEEDBACK_ONLY',
                    outputs_bitwise_equal=all(a.tobytes()==b.tobytes() for a,b in zip(values,previous))))
            rows.append(result)
            previous = values
    passed = all(r['passed'] for r in rows)
    return dict(artifact_kind=KIND, status='PASSED' if passed else 'NUMERICAL_FAILED',
        numerical_validation='PASSED' if passed else 'FAILED', comparison=meta['comparison'],
        scope=meta['reference_scope'], reference_contract=CONTRACT, cases=rows,
        first_failure=next((dict(mode=r['mode'],index=r['index'],label=r['label']) for r in rows if not r['passed']),None),
        native_calls=timings, native_api_latency=latency_summary(timings,meta['cases']),
        feedback_repeats=repeats,
        performance='DIAGNOSTIC_CALL_TIMINGS_NOT_PRODUCTION_BENCHMARK')


def latency_summary(calls, schedule):
    result = {'timing_scope':'SYNCHRONOUS_API_WALL_NOT_NPU_ONLY'}
    for mode in ('isolated', 'feedback'):
        result[mode] = {}
        for phase, proposal in (('catchup', False), ('proposal', True)):
            values = [r['api_us'] for r in calls if r['mode']==mode
                      and bool(schedule[r['index']]['proposals'])==proposal]
            result[mode][phase] = dict(calls=len(values),
                mean_ms=sum(values)/len(values)/1000 if values else None,
                min_ms=min(values)/1000 if values else None,
                max_ms=max(values)/1000 if values else None)
    return result


def summarize_result(result):
    """Expose existing comparisons without rereading tensors or changing gates."""
    keys=('status','stage','device_execution','model_compilation','numerical_validation',
          'first_failure','native_api_latency','board_attempt','lock_state','error',
          'lock_release_errors','next')
    summary = {k:result[k] for k in keys if k in result}
    rows = result.get('cases', [])
    if not rows:
        return summary
    summary['comparison'] = result['comparison']
    summary['case_counts'] = {}
    for mode in ('isolated','feedback'):
        selected = [r for r in rows if r['mode']==mode]
        count = sum(len(r.get('expected_top1', [])) for r in selected)
        mismatches = sum(r.get('top1_mismatches',0) for r in selected)
        summary['case_counts'][mode] = dict(
            total=len(selected), passed=sum(r['passed'] is True for r in selected),
            top1_tokens=count, top1_matches=count-mismatches,
            top1_failed=sum(r.get('top1_mismatches',0)>0 for r in selected),
            hidden_failed=sum(any(t['name']=='draft_hidden' and not t['passed'] for t in r['tensors']) for r in selected),
            kv_failed=sum(any(t['name'].startswith('present_') and not t['passed'] for t in r['tensors']) for r in selected),
            preservation_failed=sum(any(t.get('preserved') is False for t in r['tensors']) for r in selected))
    summary['top1_differences'] = [dict(mode=r['mode'],index=r['index'],position=p,
        actual=a,expected=e) for r in rows for p,(a,e) in
        enumerate(zip(r.get('actual_top1',[]),r.get('expected_top1',[]))) if a!=e]
    summary['relative_l2_ranges'] = {}
    for mode in ('isolated','feedback'):
        tensors = [t for r in rows if r['mode']==mode for t in r['tensors']]
        groups = {'hidden':[t for t in tensors if t['name']=='draft_hidden'],
                  'kv_whole_bank':[t for t in tensors if t['name'].startswith('present_')]}
        for region in ('appended','committed'):
            groups['kv_'+region] = [t['regions'][region] for t in tensors
                if t.get('regions',{}).get(region) is not None]
        summary['relative_l2_ranges'][mode] = {}
        for name, group in groups.items():
            values = [t['relative_l2_error'] for t in group if t.get('relative_l2_error') is not None]
            summary['relative_l2_ranges'][mode][name] = dict(
                compared_tensors=len(group), nonfinite_tensors=sum(t.get('nonfinite',False) for t in group),
                finite_zero_reference_norm=sum(not t.get('nonfinite') and t.get('relative_l2_error') is None for t in group),
                minimum=min(values) if values else None, maximum=max(values) if values else None)
    summary['relative_l2_definition'] = 'norm(actual-reference)/norm(reference); 0.01 means 1%; no L2 acceptance gate'
    if 'feedback_repeats' in result:
        summary['feedback_repeats'] = result['feedback_repeats']
    first = next((r for r in rows if not r['passed']), None)
    if first:
        summary['first_failure'] = {k:first[k] for k in
            ('mode','index','label','top1_mismatches','actual_top1','expected_top1') if k in first}
        summary['first_failure']['bad_tensors'] = [t for t in first['tensors'] if not t['passed']]
    return summary


def analyze_result(path):
    """Recompute diagnostics from retained host tensors; never rewrite acceptance."""
    path = Path(path)
    path = path/'result.json' if path.is_dir() else path
    result = read_json(path, 4 << 20)
    if (result.get('artifact_kind') != KIND or result.get('stage') != 'complete'
            or result.get('status') not in ('PASSED','NUMERICAL_FAILED')
            or result.get('device_execution') != 'EXECUTED_OUTPUTS_RETRIEVED'):
        raise ValueError('offline analysis requires a completed, retrieved accuracy result')
    fixture = Path(result['fixture'])
    if not fixture.is_absolute():
        fixture = path.parent/fixture
    if sha256_file(fixture/'manifest.json') != result['fixture_sha256']:
        raise ValueError('accuracy fixture changed after comparison')
    updated = compare(fixture, path.parent/'native')
    # This command cannot silently reinterpret a changed reference or result.
    if (updated['status'] != result['status'] or updated['comparison'] != result['comparison']
            or [(r['mode'],r['index'],r['passed'],r.get('top1_mismatches')) for r in updated['cases']]
            != [(r['mode'],r['index'],r['passed'],r.get('top1_mismatches')) for r in result['cases']]):
        raise ValueError('retained tensors no longer reproduce the original comparison')
    summary = summarize_result(dict(result, **updated))
    summary['analysis_scope'] = 'HOST_RETAINED_TENSORS_NO_BOARD_RUN_NO_GATE_CHANGE'
    return summary
