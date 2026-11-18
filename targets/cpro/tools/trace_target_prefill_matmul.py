#!/usr/bin/env python3
"""Run the normal inference.py and observe the first Prefill L3 V QuantMatmul."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import re
import runpy
import sys


class ProjectionReplayFinished(BaseException):
    """Stop inference at Prefill entry after the requested standalone call."""


def list_configs(source_root, selected):
    from qwen35_moe_dflash.target.layer_probe.reference_config import read_trace_config
    from qwen35_moe_dflash.target.original_quant import OriginalQuantWeights
    import yaml

    candidates = {selected.resolve()} if selected else set()
    folders = {source_root, source_root / 'config'}
    if selected:
        folders.add(selected.resolve().parent)
    for folder in folders:
        for pattern in ('*.yaml', '*.yml'):
            candidates.update(p.resolve() for p in folder.glob(pattern))
    quant_dirs = {}
    found = 0
    for path in sorted(candidates):
        try:
            _, info = read_trace_config(path)
        except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
            print(f'{path}: {exc}', file=sys.stderr)
            continue
        w4 = info['quanted_pth']
        if w4 and w4 not in quant_dirs:
            try:
                keys = OriginalQuantWeights(w4).mapping
                layers = {int(m[1]) for key in keys
                          if (m := re.fullmatch(r'model\.layers\.(\d+)\..+_quant_weight', key))}
                quant_dirs[w4] = sorted(layers)
            except (OSError, ValueError) as exc:
                quant_dirs[w4] = f'unavailable: {exc}'
        print(json.dumps(dict(info, w4_layer_ids=quant_dirs.get(w4)), ensure_ascii=False))
        found += 1
    return 0 if found else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--capture', type=Path, help='previous four-layer Prefill capture')
    parser.add_argument('--cpu', type=Path, help='previous verify-qlinear-golden directory')
    parser.add_argument('--output', type=Path, help='new observation directory')
    parser.add_argument('--list-configs', action='store_true', help='inspect local model YAMLs and W4 headers, then exit')
    parser.add_argument('--reference-layers', type=int, choices=(4, 40),
                        help='check checkpoint depth and, for a natural Prefill trace, observed depth')
    parser.add_argument('--replay-l3-prefix', action='store_true',
                        help='construct layers 0..3 from the 40-layer checkpoint with original placement; '
                             'call the initialized L3 V on captured hidden at Prefill entry and exit')
    parser.add_argument('inference_args', nargs=argparse.REMAINDER, help='-- followed by normal inference.py arguments')
    args = parser.parse_args()
    if args.replay_l3_prefix and (args.reference_layers != 40 or args.list_configs):
        parser.error('--replay-l3-prefix requires --reference-layers 40 in a trace run')
    forwarded = args.inference_args[1:] if args.inference_args[:1] == ['--'] else args.inference_args
    sys.path.append(str(Path(__file__).resolve().parents[3] / 'model'))
    info = None
    adapt_four = False
    if args.list_configs or args.reference_layers:
        from qwen35_moe_dflash.target.layer_probe.reference_config import read_trace_config, trace_layer_policy
        import yaml

        config_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        config_parser.add_argument('--config', type=Path)
        options, _ = config_parser.parse_known_args(forwarded)
        if args.list_configs:
            return list_configs(args.source_root.resolve(), options.config)
        if not options.config:
            parser.error('pass the original --config YAML after -- when using --reference-layers')
        try:
            raw, info = read_trace_config(options.config)
            adapt_four = trace_layer_policy(raw, info, args.reference_layers)
        except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
            parser.error(str(exc))
    if not all((args.capture, args.cpu, args.output)):
        parser.error('--capture, --cpu and --output are required to run inference')
    entry = (args.source_root / 'inference.py').resolve(strict=True)
    capture = args.capture.resolve(strict=True)
    cpu = args.cpu.resolve(strict=True)
    from qwen35_moe_dflash.target.layer_probe.qlinear_golden import separate_output
    from qwen35_moe_dflash.target.layer_probe.common import save_json

    out = separate_output(args.output, capture, cpu)
    status = dict(entry=str(entry), argv=forwarded, cwd=str(Path.cwd()), status='WAITING_FOR_PREFILL')
    if info:
        status.update(requested_layers=args.reference_layers, reference_config=info)
    if args.replay_l3_prefix:
        status.update(scope='L3_PROJECTION_REPLAY_AT_PREFILL_ENTRY', requested_layers=4,
                      checkpoint_layers=info['num_hidden_layers'], placement={})
    save_json(out / 'report.json', status)
    observer = None
    previous_profile = sys.getprofile()
    patches = ExitStack()
    wrapper_path = entry.parent / 'models/export_model_wrapper_qwen3_5_moe.py'
    prefix_installed = False

    def on_prefill(frame, event, arg):
        nonlocal observer, prefix_installed
        if (args.replay_l3_prefix and not prefix_installed and event == 'call'
                and frame.f_code.co_name == '__init__'
                and frame.f_globals.get('__name__') == 'models.export_model_wrapper_qwen3_5_moe'
                and Path(frame.f_code.co_filename).resolve() == wrapper_path.resolve()):
            from qwen35_moe_dflash.target.layer_probe.reference_config import load_layer_prefix
            namespace = frame.f_globals
            loader = namespace['load_npu_model']

            def load_prefix(*args, **kwargs):
                print('[target-trace] loading layers 0..3 with original device and dtype', flush=True)
                return load_layer_prefix(loader, args, kwargs, 4, status['placement'])

            patches.callback(namespace.__setitem__, 'load_npu_model', loader)
            namespace['load_npu_model'] = load_prefix
            prefix_installed = True
        if (event == 'call' and frame.f_code.co_name == 'prefill'
                and Path(frame.f_code.co_filename).resolve() == entry):
            sys.setprofile(previous_profile)
            try:
                # Import torch/torch_npu observers only after the original model has initialized.
                from qwen35_moe_dflash.target.layer_probe.live_projection import LiveProjection
                observer = LiveProjection(out)
                module = observer.attach(frame)
                if args.replay_l3_prefix:
                    if (status['placement'].get('constructed_layers') != 4
                            or observer.report['model_layer_count'] != 4):
                        raise ValueError('replay requires original loader to construct layers 0..3 from 40')
                    observer.replay_captured_hidden(module, capture)
            except Exception as exc:
                status['attach_error'] = f'{type(exc).__name__}: {exc}'
                if args.replay_l3_prefix:
                    raise
            if args.replay_l3_prefix:
                raise ProjectionReplayFinished()

    sys.path.insert(0, str(entry.parent))
    sys.argv = [str(entry), *forwarded]
    sys.setprofile(on_prefill)
    try:
        try:
            with patches:
                if adapt_four:
                    from qwen35_moe_dflash.target.layer_probe.reference_config import hf_config_prefix
                    patches.enter_context(hf_config_prefix(info['model_file'], 4, status))
                runpy.run_path(str(entry), run_name='__main__')
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
        status['inference'] = 'FINISHED'
    except ProjectionReplayFinished:
        status['inference'] = 'STOPPED_AFTER_L3_REPLAY'
    except BaseException as exc:
        status.update(inference='INTERRUPTED', inference_error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        sys.setprofile(previous_profile)
        if observer is not None:
            observer.close()
            status['observed_layers'] = observer.report.get('model_layer_count')
            save_json(out / 'live.json', observer.report)
        status['status'] = 'PARTIAL'
        if observer is not None and observer.report['status'] == 'CAPTURED':
            try:
                expected_layers = 4 if args.replay_l3_prefix else args.reference_layers
                if expected_layers and status['observed_layers'] != expected_layers:
                    raise ValueError(f"requested {expected_layers} layers, observed {status['observed_layers']}")
                from qwen35_moe_dflash.target.layer_probe.live_projection import compare_live
                status['comparison'] = compare_live(out, capture, cpu)
                if status['inference'] in ('FINISHED', 'STOPPED_AFTER_L3_REPLAY'):
                    status['status'] = 'INSPECTED'
            except Exception as exc:
                status['comparison_error'] = f'{type(exc).__name__}: {exc}'
        else:
            status['comparison_error'] = (status.get('attach_error')
                or (observer and observer.report.get('observation_error'))
                or 'no complete Prefill L3 V call observed')
        save_json(out / 'report.json', status)

    summary = {}
    for key, value in status.get('comparison', {}).items():
        if isinstance(value, dict) and 'relative_l2' in value:
            summary[key] = {k: value[k] for k in ('relative_l2', 'cosine', 'equal_bits', 'passed') if k in value}
        else:
            summary[key] = value
    print(json.dumps(dict(status=status['status'], inference=status.get('inference'), observed_layers=status.get('observed_layers'), report=str(out / 'report.json'),
                          error=status.get('comparison_error'), **summary), indent=2))
    return 0 if status['status'] == 'INSPECTED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
