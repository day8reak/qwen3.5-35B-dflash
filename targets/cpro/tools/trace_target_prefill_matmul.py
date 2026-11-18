#!/usr/bin/env python3
"""Run the normal inference.py and observe the first Prefill L3 V QuantMatmul."""
import argparse
import json
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--capture', type=Path, required=True, help='previous four-layer Prefill capture')
    parser.add_argument('--cpu', type=Path, required=True, help='previous verify-qlinear-golden directory')
    parser.add_argument('--output', type=Path, required=True, help='new observation directory')
    parser.add_argument('inference_args', nargs=argparse.REMAINDER, help='-- followed by normal inference.py arguments')
    args = parser.parse_args()
    entry = (args.source_root / 'inference.py').resolve(strict=True)
    capture = args.capture.resolve(strict=True)
    cpu = args.cpu.resolve(strict=True)
    forwarded = args.inference_args[1:] if args.inference_args[:1] == ['--'] else args.inference_args
    sys.path.append(str(Path(__file__).resolve().parents[3] / 'model'))
    from qwen35_moe_dflash.target.layer_probe.qlinear_golden import separate_output
    from qwen35_moe_dflash.target.layer_probe.common import save_json

    out = separate_output(args.output, capture, cpu)
    status = dict(entry=str(entry), argv=forwarded, cwd=str(Path.cwd()), status='WAITING_FOR_PREFILL')
    save_json(out / 'report.json', status)
    observer = None
    previous_profile = sys.getprofile()

    def on_prefill(frame, event, arg):
        nonlocal observer
        if (event == 'call' and frame.f_code.co_name == 'prefill'
                and Path(frame.f_code.co_filename).resolve() == entry):
            sys.setprofile(previous_profile)
            try:
                # Import torch/torch_npu observers only after the original model has initialized.
                from qwen35_moe_dflash.target.layer_probe.live_projection import LiveProjection
                observer = LiveProjection(out)
                observer.attach(frame)
            except Exception as exc:
                status['attach_error'] = f'{type(exc).__name__}: {exc}'

    sys.path.insert(0, str(entry.parent))
    sys.argv = [str(entry), *forwarded]
    sys.setprofile(on_prefill)
    try:
        try:
            runpy.run_path(str(entry), run_name='__main__')
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
        status['inference'] = 'FINISHED'
    except BaseException as exc:
        status.update(inference='INTERRUPTED', inference_error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        sys.setprofile(previous_profile)
        if observer is not None:
            observer.close()
            save_json(out / 'live.json', observer.report)
        status['status'] = 'PARTIAL'
        if observer is not None and observer.report['status'] == 'CAPTURED':
            try:
                from qwen35_moe_dflash.target.layer_probe.live_projection import compare_live
                status['comparison'] = compare_live(out, capture, cpu)
                if status['inference'] == 'FINISHED':
                    status['status'] = 'INSPECTED'
            except Exception as exc:
                status['comparison_error'] = f'{type(exc).__name__}: {exc}'
        else:
            status['comparison_error'] = status.get('attach_error') or 'no complete Prefill L3 V call observed'
        save_json(out / 'report.json', status)

    summary = {}
    for key, value in status.get('comparison', {}).items():
        if isinstance(value, dict) and 'relative_l2' in value:
            summary[key] = {k: value[k] for k in ('relative_l2', 'cosine', 'passed')}
        else:
            summary[key] = value
    print(json.dumps(dict(status=status['status'], report=str(out / 'report.json'),
                          error=status.get('comparison_error'), **summary), indent=2))
    return 0 if status['status'] == 'INSPECTED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
