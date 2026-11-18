"""Independent CPU group-256 arithmetic against retained original projection outputs.

No model import, NPU operation, export or golden replacement. Agreement here
qualifies only selected projections under the declared quantization convention.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from ..original_quant import OriginalQuantWeights, pack_i4zn
from .common import KIND, checked, identity, metrics, new_output, read_array, save_json, sha
from .qlinear_math import dynamic_quant, group_matmul

POINTS = {'linear_attn_in_proj_qkv': (0, 1, 2), 'self_attn_v_proj': (3,)}


def load_capture(capture, points=None):
    root = Path(capture).resolve(strict=True)
    manifest = json.loads((root/'manifest.json').read_text())
    if (manifest.get('artifact_kind') != KIND or manifest.get('status') != 'CAPTURED'
            or manifest.get('reference') != 'ORIGINAL_310P_INFERENCE' or manifest.get('errors')
            or manifest.get('entrypoint') != 'DIRECT_ORIGINAL_SINGLE_PREFILL'
            or set(manifest.get('layers', [])) != {0, 1, 2, 3}):
        raise ValueError('requires original L0-L3 single-call Prefill capture')
    points = list(POINTS) if points is None else list(points)
    if not points or len(points) != len(set(points)) or set(points) - POINTS.keys():
        raise ValueError('select unique QKV and/or V projection points')
    selected = []
    for point in points:
        for layer in POINTS[point]:
            rows = [r for r in manifest['regions'] if r['point'] == point and r['layer'] == layer]
            if len(rows) != 1:
                raise ValueError(f'need exactly one L{layer} {point}')
            row = rows[0]
            if (row['kind'] != 'qlinear' or row['phase'] != 'prefill' or row['call'] != 0
                    or type(row['sequence_length']) is not int or row['sequence_length'] <= 1
                    or row['id'] != f'l{layer:02d}_c00_{point}'):
                raise ValueError('requires original fresh Prefill QLinear regions')
            selected.append(row)
    if len({r['sequence_length'] for r in selected}) != 1:
        raise ValueError('selected regions have different sequence lengths')
    return root, manifest, selected


def tensor_spec(region, direction, name):
    rows = region[direction]
    if len(rows) != 1 or rows[0]['name'] != name:
        raise ValueError(f'{region["id"]}: expected only {direction}/{name}')
    return rows[0]


def separate_output(output, *inputs):
    output = Path(output).resolve()
    for source in inputs:
        source = Path(source).resolve()
        if output == source or source in output.parents or output in source.parents:
            raise ValueError('diagnostic output must be separate from capture/quantizer replay')
    return new_output(output)


def _audit(manifest, region, m, k, n):
    audit = manifest.get('quant_matmul', {})
    if audit.get('policy') != 'ORIGINAL_CALL' or audit.get('baseline_equivalence') != 'ORIGINAL_CALL_UNCHANGED':
        raise ValueError('capture does not establish unchanged original QuantMatmul calls')
    member = 'linear_attn.in_proj_qkv' if region['point'] == 'linear_attn_in_proj_qkv' else 'self_attn.v_proj'
    suffix = f'layers.{region["layer"]}.{member}'
    calls = [c for c in audit.get('calls', []) if (c['module'] == suffix or c['module'].endswith('.'+suffix))
             and c['call'] == region['call']]
    if len(calls) != 1 or calls[0].get('status') != 'EXECUTED':
        raise ValueError('need exactly one successful source QuantMatmul audit for ' + suffix)
    call = calls[0]
    kwargs = call.get('effective_kwargs', {})
    if kwargs != call.get('original_kwargs') or call.get('contract', {}).get('errors'):
        raise ValueError('original QuantMatmul call changed or failed its shape contract')
    if any(v is not None for key, v in kwargs.items() if key not in ('pertoken_scale', 'output_dtype', 'group_sizes')):
        raise ValueError('CPU reference does not cover additional bias/offset/transpose options')
    if kwargs.get('group_sizes') not in (None, [0, 0, 256]) or kwargs.get('output_dtype') != 'torch.float16':
        raise ValueError('CPU reference requires FP16 output and group256 weight semantics')
    for desc, dtype, shape in ((call['inputs']['x1'], 'torch.int8', [m, k]),
                               (call['inputs']['x2'], 'torch.int32', [k, n//8]),
                               (call['inputs']['scale'], 'torch.float32', [k//256, n]),
                               (kwargs['pertoken_scale'], 'torch.float32', [m, 1])):
        if desc.get('dtype') != dtype or desc.get('shape') != shape:
            raise ValueError('original QuantMatmul audit dtype/shape differs from group256 reference')
    return dict(module=call['module'], original_kwargs=call['original_kwargs'], inputs=call['inputs'],
        npu_formats=call.get('npu_formats'), contract=call['contract'],
        group_sizes_argument=kwargs.get('group_sizes'),
        backend_group_interpretation='NOT_VERIFIED', original_a8_values='NOT_CAPTURED')


def _array_identity(value):
    data = np.ascontiguousarray(value)
    return dict(shape=list(data.shape), dtype=str(data.dtype), bytes=data.nbytes,
                sha256=hashlib.sha256(data.view(np.uint8)).hexdigest())


def _source_contract(manifest, source_root):
    path = Path(source_root).resolve(strict=True)/'models/modeling_qwen3_5_moe_hiai_nz.py'
    digest = sha(path)
    originals = [s for s in manifest.get('sources', []) if Path(s['path']).name == path.name]
    if not originals or any(s['sha256'] != digest or s['bytes'] != path.stat().st_size for s in originals):
        raise ValueError('supply the original NZ source matching capture.sources; CPro source is not this reference')
    classes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'QLinear']
    forwards = [n for c in classes for n in c.body if isinstance(n, ast.FunctionDef) and n.name == 'forward']
    if len(classes) != 1 or len(forwards) != 1:
        raise ValueError('need original QLinear.forward source')
    calls = [n for n in ast.walk(forwards[0]) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == 'npu_dynamic_quant']
    if (len(calls) != 1 or ast.unparse(calls[0].func) != 'torch_npu.npu_dynamic_quant'
            or len(calls[0].args) != 1 or ast.unparse(calls[0].args[0]) != 'x' or calls[0].keywords):
        raise ValueError('reference covers only the original npu_dynamic_quant(x) call without smooth/group options')
    return dict(path=str(path), sha256=digest, dynamic_quant_call=ast.unparse(calls[0]),
        scope='SOURCE_IDENTITY_AND_DYNAMIC_QUANT_ARGUMENTS_ONLY; NOT_BACKEND_IMPLEMENTATION')


def _comparison(reference, golden):
    result = metrics(reference, golden)
    return dict(agreement='AGREES' if result['passed'] else 'DIFFERS', metrics=result)


def _replay(path, capture_hash, selected):
    root = Path(path).resolve(strict=True)
    manifest = json.loads((root/'manifest.json').read_text())
    if (manifest.get('artifact_kind') != 'qwen35-original-dynamic-quant-replay-v1'
            or manifest.get('capture_sha256') != capture_hash
            or manifest.get('status') not in ('RECORDED', 'PARTIAL') or manifest.get('errors')):
        raise ValueError('dynamic quant replay belongs to a different capture or schema')
    rows = {}
    for region in selected:
        matches = [c for c in manifest['cases'] if c['id'] == region['id']]
        if len(matches) != 1 or matches[0].get('status') != 'RECORDED':
            raise ValueError('missing successful dynamic quant replay: ' + region['id'])
        if matches[0]['input'] != tensor_spec(region, 'inputs', 'x'):
            raise ValueError('dynamic quant replay input identity differs: ' + region['id'])
        rows[region['id']] = matches[0]
    return root, manifest, rows


def verify(capture, config_path, source_root, output, *, points=None, quant_replay=None):
    # Only torch.load is needed for the captured .pt. Never autoload torch_npu here.
    os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
    import torch
    import yaml

    root, manifest, selected = load_capture(capture, points)
    source_contract = _source_contract(manifest, source_root)
    config_path = Path(config_path).resolve(strict=True)
    config = yaml.safe_load(config_path.read_text())
    weights_path = Path(config['quanted_pth']).expanduser()
    weights_path = (weights_path if weights_path.is_absolute() else Path(source_root)/weights_path).resolve(strict=True)
    weights = OriginalQuantWeights(weights_path)
    cap_hash = sha(root/'manifest.json')
    replay_data = _replay(quant_replay, cap_hash, selected) if quant_replay else None
    out = separate_output(output, root, *([quant_replay] if quant_replay else []))
    report = dict(status='PARTIAL', scope='SELECTED_QLINEAR_GOLDEN_MATH_ONLY',
        whole_model_golden='UNVERIFIED', capture_sha256=cap_hash,
        capture_sources=manifest.get('sources', []), config=dict(path=str(config_path), sha256=sha(config_path)),
        source_contract=source_contract,
        quanted_pth=str(weights_path), selected_cases=[r['id'] for r in selected], cases=[],
        computation='EXACT_INTEGER_GROUP256_DOTS; FP64_SCALE_SUM; FINAL_FP16',
        cpu_quantizer='FP32_ABSMAX_DIV127; FP32_DIV; RINT_TIES_TO_EVEN; ZERO_ROWS_ZERO',
        qualification='CPU_QUANTIZER_CONVENTION_NOT_VERIFIED_AGAINST_ORIGINAL_INVOCATION',
        model_compilation='NOT_RUN', device_execution='NOT_RUN', original_golden='UNCHANGED')
    if replay_data:
        report['dynamic_quant_replay'] = dict(identity(replay_data[0]/'manifest.json', replay_data[0]),
            root=str(replay_data[0]), scope='NEW_CALL_ON_FROZEN_INPUT; NOT_ORIGINAL_CALL_A8',
            runtime=replay_data[1].get('runtime'))
    for region in selected:
        row = dict(id=region['id'], layer=region['layer'], point=region['point'], status='UNAVAILABLE')
        report['cases'].append(row)
        try:
            x = read_array(root, tensor_spec(region, 'inputs', 'x'))
            golden = read_array(root, tensor_spec(region, 'outputs', 'golden_y'))
            if (x.ndim != 3 or x.shape[:2] != (1, region['sequence_length'])
                    or golden.ndim != 3 or golden.shape[:2] != x.shape[:2]
                    or x.dtype not in (np.float16, np.float32) or golden.dtype != np.float16
                    or not np.isfinite(x).all() or not np.isfinite(golden).all()):
                raise ValueError('need finite X[1,T,K] and FP16 golden[1,T,N]')
            m, k, n = x.shape[1], x.shape[2], golden.shape[2]
            row['source_call'] = _audit(manifest, region, m, k, n)
            member = 'linear_attn.in_proj_qkv' if region['point'] == 'linear_attn_in_proj_qkv' else 'self_attn.v_proj'
            key = f'target.layers.{region["layer"]}.{member}.weight'
            meta = weights.describe(key, (n, k))
            raw, scales = weights.logical(key, (n, k))
            saved = torch.load(checked(root, region['weights']), map_location='cpu', weights_only=True)
            if set(saved) != {'W_q', 'scale'}:
                raise ValueError('captured QLinear must have only W_q and scale')
            packed, expected_scale = pack_i4zn(raw), np.ascontiguousarray(scales[:, :, 0].astype(np.float32).T)
            for name, expected, dtype in (('W_q', packed, torch.int32), ('scale', expected_scale, torch.float32)):
                actual = saved[name]
                if (not isinstance(actual, torch.Tensor) or actual.dtype != dtype
                        or tuple(actual.shape) != expected.shape
                        or actual.contiguous().numpy().tobytes() != expected.tobytes()):
                    raise ValueError('raw checkpoint and captured tensor differ: ' + name)
            row['weight_binding'] = dict(status='EXACT_RAW_TO_CAPTURE', capture=region['weights'],
                source=meta, raw_weight=_array_identity(raw), raw_scale=_array_identity(scales),
                effective_fp32_scale=_array_identity(expected_scale),
                device_packing_interpretation='NOT_VERIFIED')
            del saved, packed
            q, token_scale, quant_info = dynamic_quant(x.reshape(m, k))
            ref64, ref32 = group_matmul(q, token_scale, raw, scales)
            row['cpu_dynamic_quant'] = quant_info
            row['cpu_reference'] = _comparison(ref64.reshape(golden.shape), golden)
            row['fp32_accumulation_vs_fp64'] = _comparison(ref32, ref64)
            folder = out/region['id']
            folder.mkdir()
            ref64.tofile(folder/'cpu_reference_fp16.bin')
            row['cpu_reference']['tensor'] = dict(identity(folder/'cpu_reference_fp16.bin', out),
                shape=list(golden.shape), dtype='float16')
            if replay_data:
                rr, _, records = replay_data
                qr = records[region['id']]
                dq, ds = read_array(rr, qr['q']), read_array(rr, qr['token_scale'])
                if dq.shape != q.shape or dq.dtype != np.int8 or ds.shape != token_scale.shape or ds.dtype != np.float32:
                    raise ValueError('replayed quantization dtype or shape differs')
                replay64, replay32 = group_matmul(dq, ds, raw, scales)
                row['quantizer_comparison'] = dict(a8=metrics(dq, q), token_scale=metrics(ds, token_scale),
                    a8_equal_bits=dq.tobytes() == q.tobytes(), scale_equal_bits=ds.tobytes() == token_scale.tobytes(),
                    max_abs_a8_delta=int(np.max(np.abs(dq.astype(np.int16)-q.astype(np.int16)))))
                row['replayed_quant_reference'] = _comparison(replay64.reshape(golden.shape), golden)
                row['replayed_fp32_accumulation_vs_fp64'] = _comparison(replay32, replay64)
                replay64.tofile(folder/'replayed_quant_reference_fp16.bin')
                row['replayed_quant_reference']['tensor'] = dict(identity(folder/'replayed_quant_reference_fp16.bin', out),
                    shape=list(golden.shape), dtype='float16')
            weights.describe(key, (n, k))  # Detect source shard replacement during computation.
            row['status'] = 'COMPUTED'
        except (ValueError, KeyError, OSError, TypeError, RuntimeError, ArithmeticError) as exc:
            row['error'] = f'{type(exc).__name__}: {exc}'
    report['status'] = 'INSPECTED' if all(c['status'] == 'COMPUTED' for c in report['cases']) else 'PARTIAL'
    save_json(out/'report.json', report)
    fields = ['case', 'reference', 'agreement', 'max_abs', 'relative_l2', 'cosine', 'rmse', 'error']
    with (out/'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in report['cases']:
            if row['status'] != 'COMPUTED':
                writer.writerow(dict(case=row['id'], agreement='UNAVAILABLE', error=row.get('error')))
            for label in ('cpu_reference', 'replayed_quant_reference'):
                if label in row:
                    value = row[label]
                    writer.writerow(dict(case=row['id'], reference=label, agreement=value['agreement'],
                        **{k: value['metrics'].get(k) for k in ('max_abs', 'relative_l2', 'cosine', 'rmse')}))
    text = [report['scope'], 'whole_model_golden=UNVERIFIED; original_golden=UNCHANGED',
            'CPU reference uses an explicit dynamic quantization convention, not captured A8.',
            'Optional replay is a new DynamicQuant call; all matrix multiplication is CPU arithmetic.',
            (out/'summary.csv').read_text()]
    (out/'summary.txt').write_text('\n'.join(text))
    return dict(status=report['status'], report=str(out/'report.json'), summary=str(out/'summary.txt'),
                csv=str(out/'summary.csv'), whole_model_golden='UNVERIFIED')
