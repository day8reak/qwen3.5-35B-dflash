#!/usr/bin/env python3
"""Export one cached six-layer Draft AIR from shared prepared weights."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--job', help='legacy full-draft job containing prepared and residual-fix')
    source.add_argument('--prepared', help='shared prepared weight directory; no previous AIR/OMC required')
    source.add_argument('--import-air', help='move a completed PyTorch AIR bundle to --output without exporting or compiling')
    source.add_argument('--inspect-air', help='read an existing successful or failed PyTorch AIR export; no model or SDK execution')
    source.add_argument('--finalize-air', help='recheck a saved Draft AIR after export failure and move it to --output; no re-export')
    parser.add_argument('--frontend', choices=('ge', 'pytorch'), default='ge')
    from qwen35_moe_dflash.draft.incremental.optimization import POLICIES
    parser.add_argument('--optimization', choices=POLICIES, default='baseline',
                        help='PyTorch candidate; packed-spatial-host-embedding keeps the RoPE phase table and replaces the anchor ID with original FP16 block embeddings')
    parser.add_argument('--torchair-pb2', help='installed TorchAir literal schema; normally discovered automatically')
    parser.add_argument('--replace', action='store_true', help='replace a finished AIR bundle after successful PyTorch export/import')
    parser.add_argument('--context-rows', type=int, choices=(16, 64), default=16)
    parser.add_argument('--cache-capacity', type=int, default=192)
    parser.add_argument('--max-draft-tokens', type=int, choices=(15, 7, 3, 1), default=15)
    parser.add_argument('--ge-root', help='GE SDK; defaults to CANN_ROOT with --prepared, recorded SDK with --job')
    parser.add_argument('--cxx', default='c++')
    parser.add_argument('--cxx11-abi', type=int, choices=(0, 1), default=0)
    parser.add_argument('--timeout-seconds', type=int, default=3600)
    parser.add_argument('--output')
    args = parser.parse_args(argv)
    if args.optimization != 'baseline' and (args.frontend != 'pytorch' or not args.prepared):
        parser.error('--optimization requires --frontend pytorch --prepared')
    if not args.inspect_air and not args.output:
        parser.error('--output is required for export/import')
    if args.inspect_air and (args.output or args.replace):
        parser.error('--inspect-air is read-only; omit --output and --replace')
    if args.timeout_seconds <= 0:
        parser.error('--timeout-seconds must be positive')
    if args.frontend == 'pytorch' and not (args.prepared or args.import_air or args.inspect_air or args.finalize_air):
        parser.error('PyTorch export requires --prepared')
    if args.frontend == 'ge' and not (args.import_air or args.finalize_air) and (args.replace or args.torchair_pb2):
        parser.error('--replace and --torchair-pb2 belong to the PyTorch candidate export')
    try:
        if args.inspect_air:
            from qwen35_moe_dflash.draft.incremental.torch_export import inspect_pytorch_air
            print(json.dumps(inspect_pytorch_air(args.inspect_air), ensure_ascii=False, indent=2, allow_nan=False), flush=True)
            return 0
        if args.finalize_air:
            from qwen35_moe_dflash.draft.incremental.torch_export import finalize_pytorch_air
            print(json.dumps(finalize_pytorch_air(args.finalize_air, args.output, replace=args.replace),
                             ensure_ascii=False, allow_nan=False), flush=True)
            return 0
        if args.import_air:
            from qwen35_moe_dflash.draft.incremental.torch_export import import_pytorch_air
            result = import_pytorch_air(args.import_air, args.output, replace=args.replace)
            print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
            return 0
        if args.frontend == 'pytorch':
            from qwen35_moe_dflash.draft.incremental.torch_export import export_pytorch
            result = export_pytorch(args.prepared, args.output, context_rows=args.context_rows,
                cache_capacity=args.cache_capacity, max_draft_tokens=args.max_draft_tokens,
                replace=args.replace, torchair_pb2=args.torchair_pb2, optimization=args.optimization,
                progress=lambda value: print('pytorch-draft: ' + value, flush=True))
            shown = {k: v for k, v in result.items() if k not in ('inputs', 'outputs')}
            shown.update(input_count=len(result.get('inputs', [])), output_count=len(result.get('outputs', [])))
            print(json.dumps(shown, ensure_ascii=False, allow_nan=False), flush=True)
            return 0 if result['status'] == 'PYTORCH_DRAFT_AIR_EXPORTED' else 1
        from qwen35_moe_dflash.draft.incremental.export import export_incremental_draft
        result = export_incremental_draft(args.job, args.output, context_rows=args.context_rows,
            cache_capacity=args.cache_capacity, max_draft_tokens=args.max_draft_tokens,
            ge_root=args.ge_root or (os.environ.get('CANN_ROOT') if args.prepared else None),
            prepared=args.prepared, cxx11_abi=args.cxx11_abi, cxx=args.cxx, timeout_seconds=args.timeout_seconds,
            progress=lambda value: print('incremental-draft: ' + value, flush=True))
        shown = {key: result[key] for key in ('status', 'context_rows', 'cache_capacity', 'max_draft_tokens',
            'draft_layers', 'draft_width', 'cache_bank_bytes', 'weight_files_copied', 'weight_links_created',
            'observed_new_file_bytes', 'model_compilation', 'device_execution', 'numerical_validation',
            'compilation_ready', 'runtime_integration', 'next')}
        shown.update(input_count=len(result['inputs']), output_count=len(result['outputs']),
            manifest=str(Path(args.output).absolute() / 'manifest.json'))
        print(json.dumps(shown, ensure_ascii=False, allow_nan=False), flush=True)
        return 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        print(json.dumps({'status': 'FAILED', 'model_compilation': 'NOT_RUN', 'device_execution': 'NOT_RUN',
            'output': str(Path(args.output).absolute()) if args.output else None,
            'error': f'{type(error).__name__}: {error}'}, ensure_ascii=False), flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
