#!/usr/bin/env python3
"""Audit the first NZ L3 Prefill in the user's normal inference_segment.py process."""
import argparse
from contextlib import ExitStack
from functools import wraps
import hashlib
import json
from pathlib import Path
import runpy
import sys

sys.path.append(str(Path(__file__).resolve().parents[3] / 'model'))


def tensor_hash(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def single_int(value):
    import torch

    if torch.is_tensor(value):
        value = value.tolist()
    if isinstance(value, (tuple, list)) and len(value) == 1:
        value = value[0]
    if type(value) is not int:
        raise ValueError('expected one integer length')
    return value


def compare_call(args, packed, wrapper, baseline, report):
    import numpy as np
    from qwen35_moe_dflash.target.layer_probe.attention_reference import reference_math, unpack
    from qwen35_moe_dflash.target.layer_probe.common import metrics

    query = args['query']
    steps = query.shape[2]
    valid = single_int(wrapper['effective_length'])
    report.update(physical_query_rows=steps, valid_query_rows=valid)
    fields = {'query', 'key', 'value', 'atten_mask', 'block_table', 'all_seq_lengths_q',
              'actual_seq_lengths_q', 'actual_seq_lengths_kv', 'num_heads', 'num_key_value_heads',
              'scale_value', 'inner_precise', 'input_layout', 'block_size'}
    if (set(args) != fields or args['input_layout'] != 'BNSD' or args['inner_precise'] != 2
            or not 0 < valid <= steps or single_int(args['all_seq_lengths_q']) != valid
            or single_int(wrapper['allQLen']) != valid
            or single_int(args['actual_seq_lengths_q']) not in (valid, steps)
            or single_int(args['actual_seq_lengths_kv']) < valid
            or any(len(args[k]) != 1 for k in ('key', 'value'))
            or packed.shape != query.shape or packed.dtype != query.dtype):
        raise ValueError('call differs from the reviewed fresh B1 packed Prefill contract')

    tensors = dict(query=query, key=args['key'][0], value=args['value'][0],
                   mask=args['atten_mask'], table=args['block_table'])
    if baseline is not None:
        _, old = baseline
        previous = {s['name']: s for s in old['inputs']}
        report['vs_fixture'] = {
            name: {'same_shape': list(t.shape) == previous[name]['shape'],
                   'same_dtype': str(t.numpy().dtype) == previous[name]['dtype'],
                   'same_bytes': tensor_hash(t) == previous[name]['sha256']}
            for name, t in tensors.items()}
        report['packed_output_matches_fixture_bytes'] = tensor_hash(packed) == old['actual']['sha256']

    # A short last/only chunk may have padded query rows. Certify the valid prefix
    # from both the wrapper length and its actual causal mask before scoring it.
    mask = tensors['mask'].numpy()
    if mask.ndim != 4 or mask.shape[:3] != (1, 1, steps) or mask.shape[-1] < valid:
        raise ValueError('mask shape does not cover the physical and valid query rows')
    active = mask[0, 0, :valid]
    blocked = np.arange(active.shape[1])[None, :] > np.arange(valid)[:, None]
    if np.any(active[~blocked] != 0) or not np.all(np.isneginf(active[blocked]) | (active[blocked] == -65504)):
        raise ValueError('valid query rows do not have a fresh causal mask starting at position zero')
    heads = args['num_heads']
    ref = reference_math(query.numpy()[:, :, :valid], tensors['key'].numpy(), tensors['value'].numpy(),
                         mask[:, :, :valid], tensors['table'].numpy(), valid, heads=heads,
                         kv_heads=args['num_key_value_heads'], block_size=args['block_size'], scale=args['scale_value'])
    if ref['output'] is None:
        raise ValueError(ref['evidence']['status'])
    actual = unpack(packed.numpy(), heads)[:, :, :valid]
    diff = metrics(actual, ref['output'])
    report['core_vs_cpu'] = {key: diff.get(key) for key in
                             ('passed', 'relative_l2', 'cosine', 'rmse', 'kl_divergence',
                              'mean_relative_error', 'max_abs', 'failed_metrics', 'max_error_coordinate')}
    last = metrics(actual[:, :, -1], ref['output'][:, :, -1])
    report['last_valid_query_vs_cpu'] = dict(query_index=valid - 1, passed=last['passed'],
        relative_l2=last['relative_l2'], cosine=last['cosine'], max_abs=last['max_abs'])
    finite = np.abs(actual[np.isfinite(actual)])
    bound = ref['evidence']['full_value_cache_abs_max']
    report['value_cache_abs_max'] = bound
    report['finite_core_abs_max'] = float(finite.max()) if finite.size else None
    report['finite_elements_above_cache_abs_max'] = int((finite > bound).sum())
    report['zero_head_query_rows_with_nonzero_reference'] = int(
        ((actual == 0).all(-1) & (ref['output'] != 0).any(-1)).sum())
    report['status'] = 'PASSED' if diff['passed'] else 'FAILED'


def arm(wrapper, baseline, report, cleanup):
    import torch
    import torch_npu as npu
    from qwen35_moe_dflash.target.layer_probe.capture import bound, cpu
    from qwen35_moe_dflash.target.layer_probe.fused_value_control import live_description

    candidates = [(k, v) for k, v in wrapper.named_modules() if ('.' + k).endswith('.layers.3.self_attn')]
    if len(candidates) != 1:
        raise ValueError('could not identify one layers.3.self_attn in the first segment')
    name, attn = candidates[0]
    report.update(module=name, module_class=type(attn).__module__ + '.' + type(attn).__qualname__)
    wrapper_args = {}
    snapshot = {}

    def wrapper_entry(module, pargs, kwargs):
        wrapper_handle.remove()
        try:
            values = dict(kwargs)
            if pargs:
                values.update(bound(module, pargs, kwargs))
            for key in ('allQLen', 'effective_length', 'output_pos'):
                if key in values:
                    value = cpu(values[key])
                    wrapper_args[key] = value.tolist() if torch.is_tensor(value) else value
            report['wrapper_arguments'] = wrapper_args
            ids = values.get('input_ids')
            if torch.is_tensor(ids):
                tokens = cpu(ids).numpy().astype('<i4', copy=False).tobytes()
                report['input_ids_shape'] = list(ids.shape)
                report['input_ids_i32_sha256'] = hashlib.sha256(tokens).hexdigest()
                if baseline is not None:
                    report['input_ids_match_fixture'] = tokens == baseline[0].read_bytes()
        except Exception as exc:
            report['wrapper_observation_error'] = str(exc)

    def enter_layer(module, pargs):
        layer_handle.remove()
        report['entered_l3'] = True
        original = getattr(npu, 'adn_fused_infer_attention', None)
        if not callable(original):
            report['reason'] = 'torch_npu.adn_fused_infer_attention unavailable'
            return

        @wraps(original)
        def observed(*pos, **kwargs):
            if torch.compiler.is_compiling():
                return original(*pos, **kwargs)
            report['fused_calls'] += 1
            if report['fused_calls'] != 1:
                return original(*pos, **kwargs)
            try:
                names = ('query', 'key', 'value')
                if len(pos) > 3 or set(names[:len(pos)]).intersection(kwargs):
                    raise ValueError('unrecognized positional fused arguments')
                values = dict(zip(names, pos), **kwargs)
                if values['query'].device.type != 'npu':
                    raise ValueError('fused query is not an NPU tensor')
                report['live_arguments'] = live_description(npu, values)
                snapshot['inputs'] = cpu(values)
            except Exception as exc:
                report['reason'] = str(exc)
            result = original(*pos, **kwargs)
            if 'inputs' in snapshot:
                try:
                    snapshot['output'] = cpu(result)
                except Exception as exc:
                    report['reason'] = str(exc)
            return result

        replacements = ExitStack()
        cleanup.callback(replacements.close)
        npu.adn_fused_infer_attention = observed
        replacements.callback(setattr, npu, 'adn_fused_infer_attention', original)
        namespace = getattr(module.forward, '__globals__', {})
        for key in [k for k, v in namespace.items() if v is original]:
            namespace[key] = observed
            replacements.callback(namespace.__setitem__, key, original)

        def leave_layer(_module, _inputs, _output):
            replacements.close()
            end_handle.remove()

        end_handle = module.register_forward_hook(leave_layer)
        cleanup.callback(end_handle.remove)

    wrapper_handle = wrapper.register_forward_pre_hook(wrapper_entry, with_kwargs=True)
    cleanup.callback(wrapper_handle.remove)
    layer_handle = attn.register_forward_pre_hook(enter_layer)
    cleanup.callback(layer_handle.remove)
    return snapshot, wrapper_args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, help='previous four-layer fixture for byte comparisons')
    parser.add_argument('inference_args', nargs=argparse.REMAINDER, help='-- followed by the normal inference arguments')
    args = parser.parse_args()
    entry = (Path.cwd() / 'inference_segment.py').resolve(strict=True)
    forwarded = args.inference_args[1:] if args.inference_args[:1] == ['--'] else args.inference_args
    baseline = None
    report = dict(status='UNOBSERVED', entry=str(entry), argv=forwarded, entered_l3=False, fused_calls=0)
    captured = None
    previous = sys.getprofile()
    with ExitStack() as cleanup:
        def on_entry(frame, event, arg):
            nonlocal baseline, captured
            if (event != 'call' or frame.f_code.co_name not in ('prefill_segment1', 'prefill_chunk_segment1')
                    or frame.f_code.co_filename != str(entry)):
                return
            sys.setprofile(previous)
            report['segment_entry'] = frame.f_code.co_name
            try:
                from qwen35_moe_dflash.target.layer_probe.capture import cpu
                from qwen35_moe_dflash.target.layer_probe.fused_value_control import previous_call
                ids = cpu(frame.f_locals['input_ids'])
                report['segment_input_ids_shape'] = list(ids.shape)
                if args.fixture:
                    baseline = previous_call(args.fixture)
                    report['segment_input_ids_match_fixture'] = (
                        ids.numpy().astype('<i4', copy=False).tobytes() == baseline[0].read_bytes())
                wrapper = frame.f_locals['model_wrapper']
                report['first_segment_layers'] = getattr(wrapper, 'layer_num', None)
                captured = arm(wrapper, baseline, report, cleanup)
            except Exception as exc:
                report['reason'] = f'{type(exc).__name__}: {exc}'

        cleanup.callback(sys.setprofile, previous)
        sys.setprofile(on_entry)
        sys.argv = [str(entry), *forwarded]
        sys.path.insert(0, str(entry.parent))
        try:
            runpy.run_path(str(entry), run_name='__main__')
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
        finally:
            if captured is not None:
                snapshot, wrapper_args = captured
                if report['fused_calls'] == 1 and 'output' in snapshot:
                    try:
                        compare_call(snapshot['inputs'], snapshot['output'], wrapper_args, baseline, report)
                    except Exception as exc:
                        report.update(status='UNAVAILABLE', reason=f'{type(exc).__name__}: {exc}')
                elif report['fused_calls']:
                    report['status'] = 'UNAVAILABLE'
                    if report['fused_calls'] > 1:
                        report['reason'] = 'multiple fused calls inside the first L3 forward'
            if report['status'] == 'UNOBSERVED' and 'reason' not in report:
                report['reason'] = 'no eager fused call observed at the selected first-segment L3 boundary'
            print('[segment-attention] ' + json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if report['status'] == 'PASSED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
