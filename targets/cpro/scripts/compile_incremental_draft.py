#!/usr/bin/env python3
"""Compile one exported incremental Draft gear using its original selected DDK."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--air-export', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--timeout-seconds', type=int, default=7200)
    parser.add_argument('--tbe-diagnosis', help='optional existing selected-DDK TBE compatibility diagnosis')
    args = parser.parse_args(argv)
    from qwen35_moe_dflash.draft.incremental.compile import compile_incremental_draft
    try:
        result = compile_incremental_draft(args.air_export, args.output, timeout_seconds=args.timeout_seconds,
            tbe_diagnosis=args.tbe_diagnosis, progress=lambda value: print('incremental-draft: ' + value, flush=True))
        keys = ('status', 'stage', 'context_rows', 'cache_capacity', 'max_draft_tokens', 'model_compilation',
            'compiled_precision', 'compiled_abi', 'device_execution', 'numerical_validation',
            'runtime_integration', 'full_loop_smoke', 'weight_files_copied', 'error', 'next')
        shown = {k: result[k] for k in keys if k in result}
        shown.update(input_count=len(result.get('inputs', [])), output_count=len(result.get('outputs', [])),
            manifest=str(Path(args.output).absolute() / 'manifest.json'))
        print(json.dumps(shown, ensure_ascii=False, allow_nan=False), flush=True)
        return 0 if result['status'] == 'DRAFT_INCREMENTAL_OMC_COMPILED_ABI_BOUND' else 1
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        print(json.dumps({'status': 'FAILED', 'model_compilation': 'NOT_RUN', 'device_execution': 'NOT_RUN',
            'error': f'{type(error).__name__}: {error}'}, ensure_ascii=False), flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
