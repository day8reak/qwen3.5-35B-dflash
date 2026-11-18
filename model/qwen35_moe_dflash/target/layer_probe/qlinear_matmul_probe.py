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


def repeat_dynamic_quant(npu, hidden, saved_q, saved_scale, device, rows):
    import torch
    from .prefill_matmul_repeat import equal_bits
    from .qlinear import snapshot

    source = torch.from_numpy(hidden).reshape(saved_q.shape)
    x = source.to(device)
    if not equal_bits(snapshot(x), source):
        raise ValueError('device transfer changed the captured hidden')
    previous = None
    with torch.no_grad():
        for i in range(3):
            q, scale = npu.npu_dynamic_quant(x)
            torch.npu.synchronize(device)
            if (q.device != x.device or scale.device != x.device or q.dtype != torch.int8
                    or q.shape != saved_q.shape or scale.dtype != torch.float32
                    or tuple(scale.shape) not in ((x.shape[0],), (x.shape[0], 1))):
                raise ValueError('unexpected DynamicQuant output descriptors')
            current = snapshot(dict(a8=q, token_scale=scale.reshape(-1, 1)))
            vs_saved = dict(a8=equal_bits(current['a8'], saved_q),
                            token_scale=equal_bits(current['token_scale'], saved_scale))
            row = dict(call=i + 1, vs_saved=vs_saved, hidden_equal_after=equal_bits(snapshot(x), source))
            if previous is not None:
                row['vs_previous'] = {key: equal_bits(value, previous[key]) for key, value in current.items()}
            rows.append(row)
            if not row['hidden_equal_after']:
                raise ValueError('hidden changed during DynamicQuant')
            previous = current


def dequantized_device_matmul(reference_report, ref_case, region, q, ps, n, device):
    """Ordinary FP32 matmul on the same A8 and logical W4 as the CPU reference."""
    import torch
    from ..original_quant import OriginalQuantWeights
    from .qlinear_golden import _array_identity

    member = 'self_attn.v_proj' if region['point'] == 'self_attn_v_proj' else 'linear_attn.in_proj_qkv'
    weights = OriginalQuantWeights(Path(reference_report['quanted_pth']))
    raw, scales = weights.logical(f'target.layers.{region["layer"]}.{member}.weight', (n, q.shape[1]))
    binding = ref_case['weight_binding']
    if _array_identity(raw) != binding['raw_weight'] or _array_identity(scales) != binding['raw_scale']:
        raise ValueError('logical checkpoint W4 or group scales changed since the CPU reference')
    with torch.no_grad():
        x = torch.from_numpy(q).to(device=device, dtype=torch.float32)
        w = torch.from_numpy(raw.copy()).to(device=device, dtype=torch.float32)
        s = torch.from_numpy(scales.copy()).to(device=device, dtype=torch.float32)
        w = (w.reshape(n, -1, 256) * s).reshape(n, q.shape[1])
        token_scale = torch.from_numpy(ps).to(device)
        for _ in range(2):
            y = (torch.matmul(x, w.T) * token_scale).to(torch.float16)
            torch.npu.synchronize(device)
            yield y.detach().to('cpu', copy=True).contiguous().numpy()


def run_probe(capture, quant_replay, cpu_report, output, layer=3, device='npu:0', *,
              repeat_original=False, dequant_control=False, compare_group256=False):
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
    if compare_group256 and (repeat_original or source_call['group_sizes_argument'] is not None):
        raise ValueError('group256 comparison requires a separate run and an original call without explicit groups')
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
        repeated = repeat_original or compare_group256
        if repeat_original:
            report['scope'] = 'REPEATED_DYNAMIC_QUANT_AND_SEPARATE_FROZEN_ORIGINAL_MATMUL'
            quant_calls = report['dynamic_quant_calls'] = []
            try:
                hidden = read_array(root, tensor_spec(region, 'inputs', 'x'))
                repeat_dynamic_quant(torch_npu, hidden, torch.from_numpy(q), torch.from_numpy(ps), device, quant_calls)
            except Exception as exc:
                report['dynamic_quant_error'] = f'{type(exc).__name__}: {exc}'
                summaries.append('DynamicQuant: ' + report['dynamic_quant_error'])
            for call in quant_calls:
                summaries.append(f'DynamicQuant {call}')
        schedule = ['original']
        if repeat_original:
            schedule *= 3
        elif compare_group256:
            report['scope'] = 'FROZEN_ORIGINAL_VS_EXPLICIT_GROUP256'
            schedule = ['original', 'explicit_group256', 'explicit_group256',
                        'original', 'original', 'explicit_group256']
        elif kwargs.get('group_sizes') is None:
            schedule.append('explicit_group256')
        if repeated:
            from .qlinear import snapshot
            from .prefill_matmul_repeat import compare_operands

            saved = dict(x1=torch.from_numpy(q), x2=w, scale=scale, pertoken_scale=torch.from_numpy(ps))
            report['operands_equal_before'] = compare_operands(snapshot(replay_inputs), saved)
            if not all(report['operands_equal_before'].values()):
                raise ValueError('restored matmul operands differ from the saved logical values')
        counts = {}
        previous = {}
        for signature in schedule:
            count = counts[signature] = counts.get(signature, 0) + 1
            label = signature if count == 1 else f'{signature}_repeat_{count}'
            row = dict(name=label, signature=signature, status='DISPATCHING')
            report['variants'].append(row)
            save_json(out/'report.json', report)
            try:
                options = explicit_group256(args, kwargs) if signature == 'explicit_group256' else kwargs
                row['group_sizes'] = options.get('group_sizes')
                with torch.no_grad():
                    device_y = torch_npu.npu_quant_matmul(*args, **options)
                    if device_y.device != args[0].device:
                        raise ValueError(f'unexpected output device {device_y.device}')
                    if repeated:
                        torch.npu.synchronize(device)
                    y = device_y.detach().to('cpu', copy=True).contiguous()
                if y.dtype != torch.float16 or tuple(y.shape) != (m, n):
                    raise ValueError(f'unexpected output {y.dtype} {tuple(y.shape)}')
                values = y.numpy()
                for reference, array in references.items():
                    row['vs_' + reference] = record_comparison(f'{label}_vs_{reference}', values, array)
                if repeated:
                    row['operands_equal_after'] = compare_operands(snapshot(replay_inputs), saved)
                    if not all(row['operands_equal_after'].values()):
                        raise ValueError('a frozen operand changed during QuantMatmul')
                    if signature in previous:
                        row['previous_call'], previous_values = previous[signature]
                        row['vs_previous'] = record_comparison(f'{label}_vs_previous', values, previous_values)
                    previous[signature] = (label, values)
                row['status'] = 'EXECUTED'
            except Exception as exc:
                row.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
                summaries.append(f"{label}: FAILED {row['error']}")
            save_json(out/'report.json', report)
            if repeated and row['status'] == 'FAILED':
                break
        if compare_group256:
            verdicts = report['group_comparison'] = {}
            for signature in ('original', 'explicit_group256'):
                rows = [r for r in report['variants'] if r['signature'] == signature and r['status'] == 'EXECUTED']
                verdict = 'INCOMPLETE'
                if len(rows) == 3:
                    accurate = all(r['vs_cpu']['passed'] for r in rows)
                    stable = all(r['vs_previous']['equal_bits'] for r in rows[1:])
                    verdict = 'PASSED' if accurate and stable else 'FAILED'
                verdicts[signature] = verdict
                summaries.append(f'{signature} group check: {verdict} ({len(rows)}/3 completed)')
        report['runtime_after'] = runtime(torch_npu)
        executed = all(row['status'] == 'EXECUTED' for row in report['variants']) and 'dynamic_quant_error' not in report
        if dequant_control:
            control = report['dequant_control'] = dict(calls=[], input='SAVED_DYNAMIC_QUANT_REPLAY', output_dtype='float16')
            last = None
            try:
                for i, values in enumerate(dequantized_device_matmul(reference_report, ref_case, region, q, ps, n, device), 1):
                    row = dict(call=i, vs_cpu=record_comparison(f'dequant_{i}_vs_cpu', values, expected[0]))
                    if last is not None:
                        row['vs_previous'] = record_comparison('dequant_2_vs_previous', values, last)
                    control['calls'].append(row)
                    last = values
            except Exception as exc:
                control['error'] = f'{type(exc).__name__}: {exc}'
                summaries.append('ordinary device matmul: ' + control['error'])
                executed = False
        report['status'] = 'INSPECTED' if executed else 'PARTIAL'
    except Exception as exc:
        report.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        summaries.append(report['error'])
    save_json(out/'report.json', report)
    (out/'summary.txt').write_text('\n'.join([name, *summaries])+'\n')
    return dict(status=report['status'], case=name, summary=str(out/'summary.txt'), report=str(out/'report.json'))
