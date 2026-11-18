"""Diagnose one original 310P QLinear call with frozen A8, scales and W4 bytes."""
import json
from pathlib import Path

import numpy as np

from .common import checked, metrics, read_array, save_json, sha
from .qlinear_golden import _audit, _replay, load_capture, separate_output, tensor_spec
from .qlinear_math import group_matmul


def nd_weight_hypothesis(carrier):
    """Interpret each INT32[K,N/8] word as eight ND output-channel nibbles."""
    if carrier.ndim != 2 or carrier.dtype != np.dtype('<i4'):
        raise ValueError('requires little-endian INT32[K,N/8]')
    words = carrier.view('<u4')
    k, packed_n = words.shape
    decoded = np.empty((k, packed_n * 8), np.int8)
    for lane in range(8):
        nibble = ((words >> (4 * lane)) & 15).astype(np.int8)
        decoded[:, lane::8] = (nibble ^ 8) - 8
    return decoded.T


def run_probe(capture, quant_replay, cpu_report, output, layer=3, device='npu:0'):
    import torch
    from .qlinear import describe, explicit_group256
    from .qlinear_backend import runtime, tensor_formats

    point = 'self_attn_v_proj' if layer == 3 else 'linear_attn_in_proj_qkv'
    root, manifest, regions = load_capture(capture, [point])
    region, = [r for r in regions if r['layer'] == layer]
    name = region['id']
    cap_sha = sha(root / 'manifest.json')
    replay_dir, _, replay_cases = _replay(quant_replay, cap_sha, [region])
    cpu_dir = Path(cpu_report).resolve(strict=True)
    reference_report = json.loads((cpu_dir / 'report.json').read_text())
    replay_sha = sha(replay_dir / 'manifest.json')
    if (reference_report.get('scope') != 'SELECTED_QLINEAR_GOLDEN_MATH_ONLY'
            or reference_report['capture_sha256'] != cap_sha
            or (reference_report.get('dynamic_quant_replay') or {}).get('sha256') != replay_sha):
        raise ValueError('use with-dynamic-quant report from the same capture and replay')
    ref_case, = [c for c in reference_report['cases'] if c['id'] == name and c['status'] == 'COMPUTED']
    if (ref_case['weight_binding']['status'] != 'EXACT_RAW_TO_CAPTURE'
            or ref_case['weight_binding']['capture'] != region['weights']):
        raise ValueError('original CPU weight binding differs')
    q = read_array(replay_dir, replay_cases[name]['q'])
    ps = read_array(replay_dir, replay_cases[name]['token_scale'])
    expected = read_array(cpu_dir, ref_case['replayed_quant_reference']['tensor'])
    original = read_array(root, tensor_spec(region, 'outputs', 'golden_y'))
    m, k = q.shape
    n = original.shape[-1]
    if (expected.shape != (1, m, n) or original.shape != expected.shape
            or original.dtype != np.float16 or expected.dtype != np.float16):
        raise ValueError('expected FP16 projection [1,T,N]')
    source_call = _audit(manifest, region, m, k, n)
    tensors = torch.load(checked(root, region['weights']), weights_only=True, map_location='cpu')
    w, scale = tensors['W_q'].contiguous(), tensors['scale'].contiguous()
    if (w.dtype != torch.int32 or tuple(w.shape) != (k, n // 8) or scale.dtype != torch.float32
            or tuple(scale.shape) != (k // 256, n)):
        raise ValueError('captured carrier or scales differ from audited inputs')
    out = separate_output(output, root, replay_dir, cpu_dir)
    report = dict(
        status='IN_PROGRESS', case=name, device=device, variants=[],
        capture_sha256=cap_sha, quant_replay_sha256=replay_sha,
        cpu_report_sha256=sha(cpu_dir / 'report.json'), source_call=source_call,
        scope='SINGLE_FROZEN_INPUT_QUANT_MATMUL_REPLAY', whole_model_golden='UNVERIFIED',
        quantizer_comparison=ref_case.get('quantizer_comparison'),
    )
    summaries = []

    def record_comparison(label, actual, reference):
        result = metrics(actual, reference)
        result['equal_bits'] = actual.dtype == reference.dtype and actual.tobytes() == reference.tobytes()
        summaries.append(f"{label}: rel_l2={result['relative_l2']} cosine={result['cosine']} "
                         f"max_abs={result['max_abs']} equal_bits={result['equal_bits']} passed={result['passed']}")
        return result

    save_json(out/'report.json', report)
    try:
        references = dict(cpu=expected[0], capture=original[0])
        try:
            nd, _ = group_matmul(q, ps, nd_weight_hypothesis(w.numpy()), scale.numpy().T[:, :, None])
            references['nd_hypothesis'] = nd
            report['nd_hypothesis_vs_capture'] = record_comparison('nd_hypothesis_vs_capture', nd, original[0])
        except (ArithmeticError, ValueError) as exc:
            report['nd_hypothesis_error'] = str(exc)
            summaries.append(f'nd_hypothesis unavailable: {exc}')
        import torch_npu
        report['capture_runtime'] = manifest['quant_matmul'].get('runtime')
        report['runtime_before'] = runtime(torch_npu)
        args = (torch.from_numpy(q).to(device), w.to(device), scale.to(device))
        replay_token_scale = torch.from_numpy(ps).to(device)
        original_kwargs = source_call['original_kwargs']
        kwargs = dict(original_kwargs, pertoken_scale=replay_token_scale, output_dtype=torch.float16)
        replay_inputs = dict(x1=args[0], x2=args[1], scale=args[2], pertoken_scale=replay_token_scale)
        replay_desc = describe(replay_inputs)
        replay_formats = tensor_formats(torch_npu, replay_inputs)
        capture_desc = dict(source_call['inputs'], pertoken_scale=original_kwargs['pertoken_scale'])
        capture_formats = source_call['npu_formats'] or {}
        report['replay_inputs'] = replay_desc
        report['replay_npu_formats'] = replay_formats
        for key in replay_inputs:
            before, after = capture_desc[key], replay_desc[key]
            old_fmt = capture_formats.get(key, {}).get('format')
            new_fmt = replay_formats[key].get('format')
            summaries.append(f"{key} (capture -> replay): dtype {before.get('dtype')} -> {after['dtype']}, "
                             f"stride {before.get('stride')} -> {after['stride']}, "
                             f"NPU format {old_fmt} -> {new_fmt}")
        variants = ['original']
        if kwargs.get('group_sizes') is None:
            variants.append('explicit_group256')
        for label in variants:
            row = dict(name=label, status='DISPATCHING')
            report['variants'].append(row)
            save_json(out/'report.json', report)
            try:
                options = kwargs if label == 'original' else explicit_group256(args, kwargs)
                row['group_sizes'] = options.get('group_sizes')
                with torch.no_grad():
                    device_y = torch_npu.npu_quant_matmul(*args, **options)
                    if device_y.device != args[0].device:
                        raise ValueError(f'unexpected output device {device_y.device}')
                    y = device_y.detach().cpu().contiguous()
                if y.dtype != torch.float16 or tuple(y.shape) != (m, n):
                    raise ValueError(f'unexpected output {y.dtype} {tuple(y.shape)}')
                values = y.numpy()
                for reference, array in references.items():
                    row['vs_' + reference] = record_comparison(f'{label}_vs_{reference}', values, array)
                row['status'] = 'EXECUTED'
            except Exception as exc:
                row.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
                summaries.append(f"{label}: FAILED {row['error']}")
            save_json(out/'report.json', report)
        report['runtime_after'] = runtime(torch_npu)
        executed = all(row['status'] == 'EXECUTED' for row in report['variants'])
        report['status'] = 'INSPECTED' if executed else 'PARTIAL'
    except Exception as exc:
        report.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        summaries.append(report['error'])
    save_json(out/'report.json', report)
    (out/'summary.txt').write_text('\n'.join([name, *summaries])+'\n')
    return dict(status=report['status'], case=name, summary=str(out/'summary.txt'), report=str(out/'report.json'))
