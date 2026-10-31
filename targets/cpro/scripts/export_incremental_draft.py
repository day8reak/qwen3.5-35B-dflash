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
    parser.add_argument('--context-rows', type=int, choices=(16, 64), default=16)
    parser.add_argument('--cache-capacity', type=int, default=192)
    parser.add_argument('--max-draft-tokens', type=int, choices=(15, 7, 3, 1), default=15)
    parser.add_argument('--ge-root', help='GE SDK; defaults to CANN_ROOT with --prepared, recorded SDK with --job')
    parser.add_argument('--cxx', default='c++')
    parser.add_argument('--cxx11-abi', type=int, choices=(0, 1), default=0)
    parser.add_argument('--timeout-seconds', type=int, default=3600)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    from qwen35_moe_dflash.draft.incremental.export import export_incremental_draft
    try:
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
            'output': str(Path(args.output).absolute()), 'error': f'{type(error).__name__}: {error}'}, ensure_ascii=False), flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
