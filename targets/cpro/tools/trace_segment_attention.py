#!/usr/bin/env python3
"""Audit the first NZ L3 Prefill in the user's normal inference_segment.py process."""
import argparse
import ast
from contextlib import ExitStack
from functools import wraps
import hashlib
import inspect
import json
from pathlib import Path
import runpy
import sys
from types import ModuleType

sys.path.append(str(Path(__file__).resolve().parents[3] / 'model'))


def tensor_hash(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def run_segment_with_chunk_size(entry, size):
    """Override only the two Prefill chunkers' use of chunk_size in this process."""
    class ChunkWidth(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == 'chunk_size' and isinstance(node.ctx, ast.Load):
                return ast.copy_location(ast.Constant(size), node)
            return node

    tree = ast.parse(entry.read_bytes(), filename=str(entry))
    names = ('prefill_chunk_segment1', 'prefill_chunk_segment2')
    for name in names:
        functions = [f for f in tree.body if isinstance(f, ast.FunctionDef) and f.name == name]
        if len(functions) != 1:
            raise ValueError(f'expected one {name} in {entry}')
        fn = functions[0]
        loops = [n for n in fn.body if isinstance(n, ast.For)
                 and ast.unparse(n.iter) == 'range(0, seq_len, chunk_size)']
        if len(loops) != 1:
            raise ValueError(f'{name}: cannot identify the source Prefill chunk loop')
        # Include any allocations before the loop as well as the padding in it.
        # actual_chunk_len, positions and the valid lengths stay source-derived.
        transform = ChunkWidth()
        fn.body = [transform.visit(stmt) for stmt in fn.body]
    code = compile(ast.fix_missing_locations(tree), str(entry), 'exec', dont_inherit=True)
    module = ModuleType('__main__')
    module.__file__ = str(entry)
    previous = sys.modules['__main__']
    sys.modules['__main__'] = module
    try:
        exec(code, module.__dict__)
    finally:
        sys.modules['__main__'] = previous


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
    if 'prefill_chunk_size_override' in report and (
            steps != report['prefill_chunk_size_override'] or single_int(args['actual_seq_lengths_q']) != steps):
        raise ValueError('the fused query did not execute at the requested physical chunk size')
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
    return ref['output']


def save_valid19(output, snapshot, reference, report):
    import torch
    from qwen35_moe_dflash.target.layer_probe.capture import Capture

    args = snapshot['inputs']
    ids = snapshot['wrapper_ids']
    physical = report['physical_query_rows']
    if (report['status'] != 'PASSED' or not report['last_valid_query_vs_cpu']['passed']
            or report['fused_calls'] != 1 or report['segment_entry'] != 'prefill_chunk_segment1'
            or physical not in (32, 64) or report['valid_query_rows'] != 19
            or physical != report.get('prefill_chunk_size_override', physical)
            or single_int(args['actual_seq_lengths_q']) != physical
            or list(ids.shape) != [1, physical] or list(snapshot['prompt_ids'].shape) != [1, 19]
            or not torch.equal(snapshot['prompt_ids'], ids[:, :19])):
        raise ValueError('expected a CPU-checked first segmented Prefill with 19 valid tokens at the selected physical size')

    capture = Capture(output, layers=[3], calls=1)
    for path in (report['entry'], *report['module_sources']):
        capture.source(path)
    capture.active = dict(layer=3, call=0, seq_len=physical, execution_device=report['live_arguments']['query']['device'])
    inputs = {name: args[name] for name in ('query', 'atten_mask', 'block_table')}
    inputs.update(key=args['key'][0], value=args['value'][0],
                  all_seq_lengths_q=torch.as_tensor(args['all_seq_lengths_q'], dtype=torch.int64))
    attrs = {k: v for k, v in args.items() if k not in inputs}
    capture.emit('fused_attention', 'attention', inputs, {'attention_nz': snapshot['output']}, attrs=attrs)
    capture.report['regions'][0]['audited_valid_query_rows'] = 19
    capture.report.update(status='CAPTURED', scope='L3_FUSED_ATTENTION_FIRST_CHUNK_VALID_ROWS_ONLY',
        segment_audit=dict(report), wrapper_ids=capture.tensor(capture.root, 'wrapper_ids', ids),
        cpu_valid_bnsd=capture.tensor(capture.root, 'cpu_valid_bnsd', torch.from_numpy(reference)))
    capture.flush()
    return str(capture.root / 'manifest.json')


def observe_l1_gdr_length(wrapper, report, cleanup, snapshot=None):
    """Read the actual length argument inside the first L1 prefill forward."""
    import torch
    import torch_npu as npu

    result = report['l1_gdr'] = dict(status='UNOBSERVED', calls=[])
    matches = [m for name, m in wrapper.named_modules() if ('.' + name).endswith('.layers.1.linear_attn')]
    if len(matches) != 1:
        result['reason'] = 'could not identify the first segment L1 linear_attn'
        return

    def enter(layer, _args):
        handle.remove()
        original = getattr(npu, 'npu_chunk_gated_delta_rule', None)
        if not callable(original):
            result['reason'] = 'torch_npu.npu_chunk_gated_delta_rule unavailable'
            return
        @wraps(original)
        def observed(*args, **kwargs):
            if torch.compiler.is_compiling():
                return original(*args, **kwargs)
            entry = {}
            result['calls'].append(entry)
            try:
                query = args[0] if args else kwargs['query']
                length = kwargs['effective_length']
                entry.update(physical_q_rows=int(query.shape[1]), effective_length=length.detach().cpu().tolist(),
                             length_dtype=str(length.dtype), chunk_size=kwargs.get('chunk_size'))
                result['status'] = 'OBSERVED'
                if snapshot is not None and len(result['calls']) == 1:
                    from qwen35_moe_dflash.target.layer_probe.capture import cpu
                    from qwen35_moe_dflash.target.prefill_t64.segment_gdr import OPERANDS
                    if query.device.type != 'npu' or len(args) > 5:
                        raise ValueError('unrecognized GDR device or positional arguments')
                    values = dict(zip(OPERANDS, args), **kwargs)
                    snapshot['gdr'] = dict(inputs={n: cpu(values[n]) for n in OPERANDS},
                        attrs={n: values[n] for n in ('chunk_size', 'output_final_state', 'use_qk_l2norm_in_kernel')},
                        device=str(query.device), source=inspect.getfile(type(layer)))
            except Exception as exc:
                entry['error'] = str(exc)
            outputs = original(*args, **kwargs)
            if snapshot is not None and 'gdr' in snapshot and len(result['calls']) == 1:
                try:
                    snapshot['gdr']['outputs'] = cpu(outputs)
                except Exception as exc:
                    entry['error'] = str(exc)
            return outputs
        restore = ExitStack()
        cleanup.callback(restore.close)
        npu.npu_chunk_gated_delta_rule = observed
        restore.callback(setattr, npu, 'npu_chunk_gated_delta_rule', original)
        namespace = getattr(layer.forward, '__globals__', {})
        for key in [k for k, v in namespace.items() if v is original]:
            namespace[key] = observed
            restore.callback(namespace.__setitem__, key, original)
        def leave(_module, _args, _output):
            restore.close()
            end.remove()
        end = layer.register_forward_hook(leave)
        cleanup.callback(end.remove)

    handle = matches[0].register_forward_pre_hook(enter)
    cleanup.callback(handle.remove)


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
    report['module_sources'] = [inspect.getfile(type(m)) for m in (wrapper, attn)]
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
                snapshot['wrapper_ids'] = cpu(ids)
                tokens = snapshot['wrapper_ids'].numpy().astype('<i4', copy=False).tobytes()
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
    parser.add_argument('--output', type=Path, help='save the CPU-checked 19-valid-token L3 core at its observed physical size')
    parser.add_argument('--prefill-chunk-size', type=int, choices=(64,),
                        help='run both original Prefill chunk loops at 64 rows for the T64 CPro comparison')
    parser.add_argument('--gdr-length', action='store_true', help='also observe effective_length at the L1 GDR call')
    parser.add_argument('--gdr-output', type=Path, help='capture and CPU-check the L1 GDR core and state at T64/valid19')
    parser.add_argument('inference_args', nargs=argparse.REMAINDER, help='-- followed by the normal inference arguments')
    args = parser.parse_args()
    if args.gdr_output and args.prefill_chunk_size != 64:
        parser.error('--gdr-output requires --prefill-chunk-size 64')
    if args.output or args.gdr_output:
        from qwen35_moe_dflash.common.artifacts import output_path
        args.output = output_path(args.output) if args.output else None
        args.gdr_output = output_path(args.gdr_output) if args.gdr_output else None
        for path in (args.output, args.gdr_output):
            if path and path.exists():
                parser.error(f'capture directory already exists: {path}')
        if args.output and args.gdr_output and (
                args.output.is_relative_to(args.gdr_output) or args.gdr_output.is_relative_to(args.output)):
            parser.error('L1 and L3 capture directories must be separate')
    entry = (Path.cwd() / 'inference_segment.py').resolve(strict=True)
    forwarded = args.inference_args[1:] if args.inference_args[:1] == ['--'] else args.inference_args
    baseline = None
    report = dict(status='UNOBSERVED', entry=str(entry), argv=forwarded, entered_l3=False, fused_calls=0)
    if args.prefill_chunk_size:
        report['prefill_chunk_size_override'] = args.prefill_chunk_size
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
                captured[0]['prompt_ids'] = ids
                if args.gdr_length or args.gdr_output:
                    observe_l1_gdr_length(wrapper, report, cleanup, captured[0] if args.gdr_output else None)
            except Exception as exc:
                report['reason'] = f'{type(exc).__name__}: {exc}'

        cleanup.callback(sys.setprofile, previous)
        sys.setprofile(on_entry)
        sys.argv = [str(entry), *forwarded]
        sys.path.insert(0, str(entry.parent))
        completed = False
        try:
            if args.prefill_chunk_size:
                run_segment_with_chunk_size(entry, args.prefill_chunk_size)
            else:
                runpy.run_path(str(entry), run_name='__main__')
            completed = True
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
            completed = True
        finally:
            if captured is not None:
                snapshot, wrapper_args = captured
                if args.gdr_output and completed:
                    try:
                        from qwen35_moe_dflash.target.prefill_t64.segment_gdr import save
                        report['gdr_reference'] = save(args.gdr_output, snapshot, report)
                    except Exception as exc:
                        report['gdr_reference'] = dict(status='UNAVAILABLE', error=f'{type(exc).__name__}: {exc}')
                if report['fused_calls'] == 1 and 'output' in snapshot:
                    try:
                        reference = compare_call(snapshot['inputs'], snapshot['output'], wrapper_args, baseline, report)
                        if args.output and completed and report['status'] == 'PASSED':
                            report['capture'] = save_valid19(args.output, snapshot, reference, report)
                    except Exception as exc:
                        report.update(status='UNAVAILABLE', reason=f'{type(exc).__name__}: {exc}')
                elif report['fused_calls']:
                    report['status'] = 'UNAVAILABLE'
                    if report['fused_calls'] > 1:
                        report['reason'] = 'multiple fused calls inside the first L3 forward'
            if report['status'] == 'UNOBSERVED' and 'reason' not in report:
                report['reason'] = 'no eager fused call observed at the selected first-segment L3 boundary'
            print('[segment-attention] ' + json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
    length_observed = not args.gdr_length or report.get('l1_gdr', {}).get('status') == 'OBSERVED'
    qualified = not args.gdr_output or report.get('gdr_reference', {}).get('status') == 'CPU_CHECKED'
    return 0 if report['status'] == 'PASSED' and length_observed and qualified else 1


if __name__ == '__main__':
    raise SystemExit(main())
