#!/usr/bin/env python3
"""Run the user's original 310P inference with CPro-boundary observations."""
import argparse
import importlib
import os
from pathlib import Path
import runpy
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', required=True, help='original project containing inference.py and models/')
    p.add_argument('--output', required=True, help='new capture directory below AI_RUN_DIR')
    p.add_argument('--layers', default='0,1,2,3')
    p.add_argument('--calls', type=int, default=2, help='first calls per layer; 2 normally means prefill + first decode')
    p.add_argument('inference_args', nargs=argparse.REMAINDER, help='-- followed by the ORIGINAL working inference arguments')
    a = p.parse_args()
    root = Path(a.source_root).resolve()
    entry = root / 'inference.py'
    if not entry.is_file():
        p.error('source-root must contain the original inference.py')
    forwarded = a.inference_args[1:] if a.inference_args[:1] == ['--'] else a.inference_args
    if not forwarded:
        p.error('supply the original working inference command arguments after --')
    # Do not silently switch a W8/pseudo model into the claimed W4 golden.
    reference_parser = argparse.ArgumentParser(add_help=False)
    reference_parser.add_argument('--quant_bits', type=int)
    reference_parser.add_argument('--quant_mode', default='enable')
    reference_parser.add_argument('--pseudo_quant', action='store_true')
    reference_parser.add_argument('--config')
    reference, _ = reference_parser.parse_known_args(forwarded)
    if reference.quant_bits != 4 or reference.quant_mode != 'enable':
        p.error('this diagnosis requires the original enabled W4 path: --quant_mode enable --quant_bits 4')
    if reference.pseudo_quant:
        p.error('pseudo quant is not the selected trusted 310P reference')
    sys.path.insert(0, str(root))
    from qwen35_moe_dflash.target.layer_probe.capture import Capture
    import torch_npu
    nz = importlib.import_module('models.modeling_qwen3_5_moe_hiai_nz')
    wrappers = importlib.import_module('models.export_model_wrapper_qwen3_5_moe')
    capture = Capture(a.output, [int(x) for x in a.layers.split(',')], a.calls)
    capture.report['original_argv'] = [str(entry), *forwarded]
    for path in (entry, Path(nz.__file__), Path(wrappers.__file__), root/'utils.py'):
        capture.source(path)
    # Record configuration identity, without rewriting relative paths or prompts.
    if reference.config:
        capture.source(Path(reference.config).resolve())
    capture.install(torch_npu, nz, wrappers)
    old_argv = sys.argv
    error = None
    try:
        sys.argv = [str(entry), *forwarded]
        runpy.run_path(str(entry), run_name='__main__')
    except SystemExit as exc:
        if exc.code not in (0, None):
            error = f'original inference exited {exc.code}'
            raise
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        sys.argv = old_argv
        capture.close(error)
    print('capture:', capture.root/'manifest.json', capture.report['status'])
    return 0 if capture.report['status'] == 'CAPTURED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
