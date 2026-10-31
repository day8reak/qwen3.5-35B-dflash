#!/usr/bin/env python3
"""Export a real larger six-layer draft AIR without copying its prepared weights."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True)
    parser.add_argument('--sequence-length', type=int, default=128)
    parser.add_argument('--prompt-length', type=int, default=64)
    parser.add_argument('--max-new-tokens', type=int, default=32)
    parser.add_argument('--ge-root', help='optional identical GE library copy; default recorded source SDK')
    parser.add_argument('--cxx', default='c++')
    parser.add_argument('--timeout-seconds', type=int, default=3600)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    from qwen35_moe_dflash.draft.capacity_export import export_draft_capacity
    try:
        result = export_draft_capacity(args.job, args.output, sequence_length=args.sequence_length,
            prompt_length=args.prompt_length, max_new_tokens=args.max_new_tokens, ge_root=args.ge_root,
            cxx=args.cxx, timeout_seconds=args.timeout_seconds,
            progress=lambda value: print('draft-capacity: ' + value, flush=True))
        shown = {key: result[key] for key in ('status', 'sequence_length', 'original_sequence_length', 'draft_layers',
            'weight_files_copied', 'weight_links_created', 'observed_new_file_bytes', 'model_compilation',
            'device_execution', 'numerical_validation', 'compilation_ready', 'next')}
        shown['manifest'] = str(Path(args.output).absolute() / 'manifest.json')
        print(json.dumps(shown, ensure_ascii=False, allow_nan=False), flush=True)
        return 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        print(json.dumps({'status': 'FAILED', 'model_compilation': 'NOT_RUN', 'device_execution': 'NOT_RUN',
            'output': str(Path(args.output).absolute()), 'error': f'{type(error).__name__}: {error}'}, ensure_ascii=False), flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
