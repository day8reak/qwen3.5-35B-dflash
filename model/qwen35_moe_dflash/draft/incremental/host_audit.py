"""Compare retained device outputs with CPU executions of the source math.

This diagnostic never rewrites goldens, exports a graph, or invokes a device.
The alternative RMS rounding is CPU-only and cannot promote a model.
"""
import json
import os
from pathlib import Path
from types import MethodType

import numpy as np

from qwen35_moe_dflash.common.artifacts import checked_file, output_path
from qwen35_moe_dflash.common.hashing import sha256_file
from qwen35_moe_dflash.execution.runtime_config import read_json
from .accuracy import analyze_result, metrics, read_snapshot
from .accuracy_diagnose import _controls
from .runtime import GEOMETRY, validate_abi

KIND = 'qwen35-draft-host-reference-audit'


def reference_boundary_norm(self, x, norm):
    """Original oracle: normalize FP32 -> FP16 -> direct gamma multiply."""
    import torch
    normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + norm.eps)
    return normalized.to(x.dtype) * norm.weight


def compare_outputs(actual, expected, step, meta):
    atol, rtol = (meta['comparison'][n] for n in ('atol', 'rtol'))
    count, start, valid = (step[n] for n in ('proposals', 'start', 'valid'))
    row = dict(passed=True, top1_mismatches=0, top1_differences=[], hidden=None, kv=[])
    if count:
        a, e = actual[0].reshape(-1)[:count], expected[0].reshape(-1)[:count]
        row['top1_differences'] = [dict(position=i, actual=int(x), expected=int(y))
                                 for i, (x, y) in enumerate(zip(a, e)) if x != y]
        row['top1_mismatches'] = len(row['top1_differences'])
        row['hidden'] = metrics(actual[1][:, :count], expected[1][:, :count], atol, rtol)
        row['passed'] &= not row['top1_mismatches'] and row['hidden']['passed']
    for spec, a, e in zip(meta['outputs'][2:], actual[2:], expected[2:]):
        item = dict(name=spec['name'], whole=metrics(a, e, atol, rtol),
            appended=metrics(a[:, :, start:start+valid], e[:, :, start:start+valid], atol, rtol) if valid else None)
        row['kv'].append(item)
        row['passed'] &= item['whole']['passed']
    return row


def compact_comparison(row):
    appended = [x['appended'] for x in row['kv'] if x['appended'] is not None]
    return dict(passed=row['passed'], top1_mismatches=row['top1_mismatches'],
        top1_differences=row['top1_differences'], hidden=row['hidden'],
        kv_failed=[x['name'] for x in row['kv'] if not x['whole']['passed']],
        kv_appended_max_abs=max((x.get('max_abs_error', 0) for x in appended), default=None))


def summarize_audit(report):
    return {**{k: report[k] for k in (
        'artifact_kind', 'status', 'source_numerical_validation', 'scope', 'device_execution',
        'model_compilation', 'model_sha256', 'comparison', 'report', 'limitations')},
        'cases': [dict(index=r['index'], label=r['label'], variant=r['variant'],
            **{key: compact_comparison(r[key]) for key in
               ('host_vs_reference', 'device_vs_host', 'device_vs_reference')})
            for r in report['cases']],
        'retained_kv_appended_by_layer': report['retained_kv_appended_by_layer']}


def _kv_summary(result):
    groups = {}
    for case in result['cases']:
        for tensor in case['tensors']:
            region = tensor.get('regions', {}).get('appended')
            if region is None:
                continue
            key = (case['mode'], tensor['name'])
            groups.setdefault(key, []).append(region)
    return [dict(mode=mode, name=name, compared=len(rows),
        failed=sum(not x['passed'] for x in rows),
        nonfinite=sum(x.get('nonfinite', False) for x in rows),
        max_abs_error=max((x.get('max_abs_error', 0) for x in rows), default=None),
        relative_l2_min=min((x['relative_l2_error'] for x in rows if x.get('relative_l2_error') is not None), default=None),
        relative_l2_max=max((x['relative_l2_error'] for x in rows if x.get('relative_l2_error') is not None), default=None))
        for (mode, name), rows in sorted(groups.items())]


def audit_reference(path, source, air_export, output, progress=lambda x: None):
    path = Path(path).resolve()
    path = path/'result.json' if path.is_dir() else path
    # Verify the original complete two-sequence acceptance before any CPU replay.
    original = analyze_result(path)
    result = read_json(path, 4 << 20)
    fixture = Path(result['fixture'])
    fixture = fixture if fixture.is_absolute() else path.parent/fixture
    meta = read_json(fixture/'manifest.json')
    validate_abi(meta)
    source, air = Path(source).resolve(), Path(air_export).resolve()
    destination = output_path(output)
    if any(destination == p or p in destination.parents for p in (path.parent, fixture.resolve(), source, air)):
        raise ValueError('host audit report must be separate from all retained source/result bundles')
    if destination.exists() and read_json(destination).get('artifact_kind') != KIND:
        raise ValueError('refusing to overwrite an unrelated audit output')
    binding = read_json(fixture/'pytorch-abi.json')
    plan, exported = read_json(source/'prepared.json'), read_json(air/'manifest.json')
    from .optimization import manifest_policy, optimization_contract
    policy = manifest_policy(exported)
    if (meta.get('frontend') != 'pytorch'
            or binding.get('packed_model_sha256') != meta['model']['sha256']
            or binding.get('prepared_sha256') != sha256_file(source/'prepared.json')
            or binding.get('source_air_sha256') != sha256_file(air/'export.air')):
        raise ValueError('retained model/source/AIR binding differs')
    if (exported.get('artifact_kind') != 'qwen35-incremental-draft-pytorch-air'
            or exported.get('status') != 'PYTORCH_DRAFT_AIR_EXPORTED'
            or exported.get('attention', {}).get('policy') != 'composed-attention-v1'
            or exported.get('projection') != optimization_contract(policy)['projection']
            or any(exported[k] != meta[k] for k in GEOMETRY)):
        raise ValueError('host audit requires the matching composed-attention full-dot Conv export')
    bound = {r['name']: r['sha256'] for r in binding['reference_weights']}
    if (len(bound) != len(binding['reference_weights']) or set(bound) != set(plan['weights'])
            or any(bound[n] != r['sha256'] for n, r in plan['weights'].items())):
        raise ValueError('prepared weights differ from retained compiler/reference binding')
    progress('verifying prepared weights and retained controls; CPU only')
    for record in plan['weights'].values():
        checked_file(source, record)
    control_evidence = _controls(fixture, meta)
    selected = [r for r in meta['cases'] if r['label'] in
                ('empty-context', 'initial-catchup', 'partial-context', 'capacity-boundary')]
    if not 1 <= len(selected) <= 4:
        raise ValueError('unexpected bounded diagnostic schedule')
    # Avoid loading torch_npu automatically on a host with an installed backend.
    os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
    import torch
    from . import torch_model
    model = torch_model.load_prepared(source, optimization=policy, **{k: meta[k] for k in GEOMETRY})
    if model.attention_contract() != exported['attention']:
        raise ValueError('current host source attention contract differs from the retained export')
    roles = [r['name'] for r in model.abi()['inputs']]
    records = {(r['case'], r['index']): r for r in meta['input_files']}
    if len(records) != len(meta['input_files']):
        raise ValueError('duplicate retained input records')
    report = dict(artifact_kind=KIND, status='ANALYZED',
        source_numerical_validation=original['numerical_validation'],
        scope='CPU_SOURCE_MATH_REPLAY_OF_SELECTED_ISOLATED_CASES_NOT_TARGET_EXECUTION',
        device_execution='NOT_RUN', model_compilation='NOT_RUN', production_promotion='NOT_PERFORMED',
        optimization=optimization_contract(policy),
        model_sha256=meta['model']['sha256'], board_attempt=result.get('board_attempt'),
        comparison=meta['comparison'], controls=control_evidence,
        host_source_sha256=sha256_file(Path(torch_model.__file__)), torch_version=torch.__version__,
        host_sources={name: sha256_file(Path(__file__).resolve().parents[2]/name) for name in (
            'draft/incremental/torch_model.py', 'draft/incremental/host_audit.py',
            'draft/incremental/optimization.py',
            'model.py', 'ops.py', 'config.py')},
        source_air_sha256=binding['source_air_sha256'], prepared_sha256=binding['prepared_sha256'],
        rms_variants=dict(source_fused_gamma='FP32_NORMALIZE_AND_GAMMA_THEN_FP16',
                          reference_gamma_boundary='FP32_NORMALIZE_THEN_FP16_THEN_FP16_GAMMA; CPU_ONLY'),
        cases=[], retained_kv_appended_by_layer=_kv_summary(result),
        report=str(destination),
        limitations='CPU source RMS uses the declared formula, not vendor implementation. Reduction order, internal compiled precision, source/compiled rewrite effects and untested feedback remain unresolved. No golden or acceptance changes.')
    original_norm = model.norm
    try:
        with torch.inference_mode():
            for step in selected:
                i, count = step['index'], step['proposals']
                values = {}
                for spec in meta['inputs']:
                    record = records[(i, spec['index'])]
                    if record['bytes'] != spec['bytes'] or record['path'] != f"inputs/{i}/{spec['index']}.bin":
                        raise ValueError('retained input identity/extent differs')
                    dt = '<f2' if spec['dtype'] == 'float16' else '<i4'
                    a = np.fromfile(checked_file(fixture, record), dtype=dt).reshape(spec['shape'])
                    values[spec['name']] = torch.from_numpy(a)
                args = tuple(values[n] for n in roles)
                before = [t.numpy().tobytes() for t in args]
                device = read_snapshot(path.parent/'native'/f'isolated-{i}.bin', meta['outputs'])
                golden = [None, None]
                if count:
                    golden = [np.fromfile(fixture/'golden'/str(i)/'top1.bin', dtype='<i4'),
                        np.fromfile(fixture/'golden'/str(i)/'hidden.bin', dtype='<f2').reshape(1,count,-1)]
                golden += [np.fromfile(fixture/'golden'/str(i)/f'kv_{j}.bin', dtype='<f2').reshape(s['shape'])
                           for j, s in enumerate(meta['outputs'][2:])]
                for variant in report['rms_variants']:
                    progress(f"CPU case {i} {step['label']}: {variant}")
                    model.norm = original_norm if variant == 'source_fused_gamma' else MethodType(reference_boundary_norm, model)
                    actual = [t.detach().numpy() for t in model(*args)]
                    if [t.numpy().tobytes() for t in args] != before:
                        raise ValueError('CPU source replay mutated retained inputs')
                    outside = np.ones(meta['cache_capacity'], dtype=bool)
                    outside[step['start']:step['start']+step['valid']] = False
                    for spec, a in zip(meta['outputs'][2:], actual[2:]):
                        old = values[spec['name'].replace('present_', 'past_')].numpy()
                        if a[:,:,outside].tobytes() != old[:,:,outside].tobytes():
                            raise ValueError('CPU replay changed KV outside write interval')
                    row = dict(index=i, label=step['label'], variant=variant,
                        controls={k: step[k] for k in ('start','valid','anchor','proposals')},
                        host_vs_reference=compare_outputs(actual, golden, step, meta),
                        device_vs_host=compare_outputs(device, actual, step, meta),
                        device_vs_reference=compare_outputs(device, golden, step, meta))
                    report['cases'].append(row)
    finally:
        model.norm = original_norm
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix+'.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(destination)
    return summarize_audit(report)
