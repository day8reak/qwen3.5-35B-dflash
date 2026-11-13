#!/usr/bin/env python3
"""Run the user's original 310P inference with CPro-boundary observations."""
import argparse
from contextlib import nullcontext
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
    p.add_argument('--model-layers', type=int, default=4,
        help='construct only this layer prefix; default 4 matches the working inference.py golden')
    p.add_argument('--calls', type=int, default=2, help='first calls per layer; 2 normally means prefill + first decode')
    p.add_argument('--qlinear-group-size', type=int, choices=[256],
        help='explicit QLinear group_sizes=[0,0,256], after shape checks; baseline equivalence remains unverified')
    p.add_argument('inference_args', nargs=argparse.REMAINDER, help='-- followed by the ORIGINAL working inference arguments')
    a = p.parse_args()
    selected = [int(x) for x in a.layers.split(',')]
    if a.model_layers < 1 or not selected or min(selected) < 0 or max(selected) >= a.model_layers:
        p.error('capture layers must lie within the positive --model-layers prefix')
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
    reference_parser.add_argument('--use_origin_model', action='store_true')
    reference_parser.add_argument('--config')
    reference, _ = reference_parser.parse_known_args(forwarded)
    if reference.quant_bits != 4 or reference.quant_mode != 'enable':
        p.error('this diagnosis requires the original enabled W4 path: --quant_mode enable --quant_bits 4')
    if reference.pseudo_quant:
        p.error('pseudo quant is not the selected trusted 310P reference')
    if reference.use_origin_model:
        p.error('use_origin_model bypasses the original NPU wrapper and its bounded weight placement')
    sys.path.insert(0, str(root))
    from qwen35_moe_dflash.target.layer_probe.capture import Capture
    from qwen35_moe_dflash.target.layer_probe.placement import LayerStreaming
    import torch_npu
    nz = importlib.import_module('models.modeling_qwen3_5_moe_hiai_nz')
    wrappers = importlib.import_module('models.export_model_wrapper_qwen3_5_moe')
    capture = Capture(a.output, selected, a.calls)
    capture.report['original_argv'] = [str(entry), *forwarded]
    for path in (entry, Path(nz.__file__), Path(wrappers.__file__), root/'utils.py'):
        capture.source(path)
    # Record configuration identity, without rewriting relative paths or prompts.
    if reference.config:
        capture.source(Path(reference.config).resolve())
    placement = LayerStreaming(capture.report, model_layers=a.model_layers)
    old_argv = sys.argv
    error = None
    try:
        capture.install(torch_npu, nz, wrappers, qlinear_group_size=a.qlinear_group_size)
        placement.install(wrappers)
        capture.flush()
        sys.argv = [str(entry), *forwarded]
        config_guard = nullcontext()
        if reference.config:
            import yaml
            from qwen35_moe_dflash.target.layer_probe.reference_config import hf_config_prefix
            model_path = yaml.safe_load(Path(reference.config).read_text())['model_file']
            config_guard = hf_config_prefix(model_path, a.model_layers, capture.report)
        with config_guard:
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
        try:
            placement.close()
        except Exception as exc:
            if error is None:
                error = f'placement cleanup failed: {exc}'
                raise
        finally:
            capture.close(error)
    print('capture:', capture.root/'manifest.json', capture.report['status'])
    return 0 if capture.report['status'] == 'CAPTURED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
