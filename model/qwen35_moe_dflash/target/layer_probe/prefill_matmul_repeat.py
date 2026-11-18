"""Repeat one observed L3 V matmul after the original four-layer Prefill returns."""
import json
from pathlib import Path

import torch

from .common import checked, identity, metrics, read_array, save_json, sha
from .qlinear import arguments, describe, snapshot
from .qlinear_golden import load_capture, separate_output, tensor_spec


OPERANDS = ('x1', 'x2', 'scale', 'pertoken_scale')


def equal_bits(a, b):
    return (a.shape == b.shape and a.dtype == b.dtype and
            a.numpy().tobytes() == b.numpy().tobytes())


def compare_output(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError(f'projection descriptors differ: {describe(a)}, {describe(b)}')
    score = metrics(a.numpy(), b.numpy())
    return dict(equal_bits=equal_bits(a, b), relative_l2=score['relative_l2'],
                cosine=score['cosine'], max_abs=score['max_abs'])


def compare_operands(a, b):
    return {key: equal_bits(a[key], b[key]) for key in OPERANDS}


class MatmulRepeat:
    def __init__(self, out):
        self.out = out
        self.module = None
        self.original = None
        self.report = {}

    def save(self, name, tensors):
        path = self.out / (name + '.pt')
        torch.save(tensors, path)
        self.report[name] = identity(path, self.out)

    def first_call(self, original, args, kwargs):
        if self.original is not None:
            raise ValueError('multiple selected L3 V calls')
        if (len(args) != 3 or not isinstance(kwargs.get('pertoken_scale'), torch.Tensor)
                or kwargs.get('output_dtype') != torch.float16
                or any(v is not None for k, v in kwargs.items()
                       if k not in ('pertoken_scale', 'output_dtype', 'group_sizes'))):
            raise ValueError('L3 repeat expects the original four-operand QuantMatmul signature')
        self.original, self.args, self.kwargs = original, args, kwargs
        self.operands = dict(arguments(args, kwargs), pertoken_scale=kwargs['pertoken_scale'])
        self.first = snapshot(self.operands)
        self.report['kwargs'] = describe(kwargs)
        self.report['positional_count'] = len(args)
        result = original(*args, **kwargs)
        self.first_y = snapshot(result)
        self.save('first_call', dict(self.first, matmul_y=self.first_y))
        return result

    def repeat(self):
        if self.original is None:
            raise ValueError('no L3 V matmul was captured during original Prefill')
        before = snapshot(self.operands)
        same = compare_operands(before, self.first)
        self.report['operands_equal_before_repeat'] = same
        if not all(same.values()):
            self.save('repeat_inputs', before)
        with torch.no_grad():
            repeat_y = snapshot(self.original(*self.args, **self.kwargs))
        self.save('repeat_y', repeat_y)
        self.report['repeat_vs_first'] = compare_output(repeat_y, self.first_y)


def projection(capture, side, name):
    root, _, (region,) = capture
    return torch.from_numpy(read_array(root, tensor_spec(region, side, name)))


def replay_hidden(probe, hidden, prefix, golden):
    """Control for a changed upstream trajectory using this run's initialized V."""
    import torch_npu
    from .live_projection import LiveProjection

    root = probe.out / 'frozen-hidden'
    root.mkdir()
    observer = LiveProjection(root)
    observer.npu = torch_npu
    observer.handles.append(probe.module.register_forward_pre_hook(observer.before, with_kwargs=True))
    observer.handles.append(probe.module.register_forward_hook(observer.after))
    try:
        with torch.no_grad():
            result = snapshot(probe.module(hidden.to(probe.operands['x1'].device)))
    finally:
        observer.close()
    if observer.report.get('status') != 'CAPTURED':
        raise ValueError('frozen-hidden control did not capture one complete L3 V call')
    call = torch.load(checked(root, observer.report['call']), map_location='cpu', weights_only=True)
    return dict(hidden_equal_bits=equal_bits(call['hidden'], hidden),
                operands_equal_to_prefix=compare_operands(call, prefix),
                module_equals_matmul=equal_bits(result[0], call['matmul_y']),
                vs_prefix=compare_output(result[0], prefix['matmul_y']),
                vs_historical_golden=compare_output(result, golden))


def source_hashes(record):
    return {(Path(s['path']).name, s['sha256']) for s in record['sources']}


def check_prefill_matmul(fixture, trace, source_root, config, output, device):
    from ..prefill_fixture import generate

    fixture, trace = Path(fixture).resolve(strict=True), Path(trace).resolve(strict=True)
    old = json.loads((fixture / 'manifest.json').read_text())
    if (old.get('artifact_kind') != 'qwen35-target-prefill-fixture-v1' or
            old.get('boundary_capture', {}).get('status') != 'CAPTURED'):
        raise ValueError('supply the original Prefill fixture with its completed boundary capture')
    tokens = checked(fixture, old['tokens'])
    captured = load_capture(fixture/'capture', ['self_attn_v_proj'])
    hidden = projection(captured, 'inputs', 'x')
    golden = projection(captured, 'outputs', 'golden_y')
    live = json.loads((trace / 'live.json').read_text())
    if (live.get('status') != 'CAPTURED' or live.get('calls') != 1 or
            not ('.' + live['module']).endswith('.layers.3.self_attn.v_proj') or
            live.get('input_tensor') != tensor_spec(captured[2][0], 'inputs', 'x')):
        raise ValueError('supply the L3 prefix trace replayed on this captured hidden')
    prefix = torch.load(checked(trace, live['call']), map_location='cpu', weights_only=True)
    if not equal_bits(hidden, prefix['hidden']):
        raise ValueError('prefix hidden differs from original capture')

    out = separate_output(output, fixture, trace)
    probe = MatmulRepeat(out)
    report = dict(case=captured[2][0]['id'], status='PARTIAL', report=str(out/'report.json'),
                  historical_capture_sha256=sha(captured[0]/'manifest.json'),
                  prefix_call_sha256=live['call']['sha256'], observation=probe.report)
    try:
        fresh = generate(source_root, config, tokens, old['sequence_length'], out/'fixture', device=device,
                         intermediates=bool(old.get('intermediates')), capture_boundaries=True, l3_repeat=probe)
        report['new_fixture_status'] = fresh['status']
        if fresh.get('error'):
            report['fixture_error'] = fresh['error']
        if 'first_call' in probe.report:
            report['operands_equal_to_prefix'] = compare_operands(probe.first, prefix)
            report['first_vs_prefix'] = compare_output(probe.first_y, prefix['matmul_y'])
            report['first_vs_historical_golden'] = compare_output(probe.first_y, golden[0])
        new = load_capture(out/'fixture/capture', ['self_attn_v_proj'])
        new_hidden = projection(new, 'inputs', 'x')
        report['hidden_equal_to_historical'] = equal_bits(new_hidden, hidden)
        report['tokens_equal_to_historical'] = fresh['tokens']['sha256'] == old['tokens']['sha256']
        report['source_hashes_equal'] = source_hashes(fresh) == source_hashes(old)
        report['signature_equal_to_prefix'] = (probe.report['kwargs'] == live['kwargs']
                                               and probe.report['positional_count'] == live['positional_count'])
        new_y = projection(new, 'outputs', 'golden_y')
        report['module_equals_first_matmul'] = equal_bits(new_y[0], probe.first_y)
        if 'repeat_vs_first' in probe.report and not fresh.get('error'):
            if not report['hidden_equal_to_historical']:
                report['frozen_hidden_control'] = replay_hidden(probe, hidden, prefix, golden)
            report['status'] = 'INSPECTED'
    except (OSError, KeyError, ValueError, RuntimeError, AttributeError) as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        probe.module = None
        save_json(out/'report.json', report)
    return report
