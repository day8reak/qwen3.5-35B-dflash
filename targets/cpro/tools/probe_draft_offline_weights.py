#!/usr/bin/env python3
"""Compile ND/NZ/ZN × transpose on/off probes over nine matrix geometries."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'model'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'ops/draft-top1/scripts'))
from qwen35_moe_dflash.draft.incremental.offline_weight_probe import compile_probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--ddk-root', default=os.environ.get('CPRO_DDK_ROOT'))
    parser.add_argument('--ddk-id', default=os.environ.get('CPRO_DDK_ID'))
    parser.add_argument('--torchair-pb2', help='installed literal TorchAir GE descriptor; auto-discovered when unique')
    parser.add_argument('--tbe-diagnosis')
    parser.add_argument('--timeout-seconds', type=int, default=1800)
    parser.add_argument('--replace-failed', action='store_true')
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error('--timeout-seconds must be positive')
    try:
        result = compile_probe(args.output, ddk_root=args.ddk_root, ddk_id=args.ddk_id,
            torchair_pb2=args.torchair_pb2, tbe_diagnosis=args.tbe_diagnosis,
            timeout_seconds=args.timeout_seconds, replace=args.replace_failed,
            progress=lambda s: print('offline-weights: '+s, flush=True))
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        result = dict(status='FAILED', error=f'{type(error).__name__}: {error}')
    shown = {k: v for k, v in result.items() if k != 'cases'}
    if 'cases' in result:
        first = {}
        for row in result['cases']:
            if row['status'] == 'FAILED' and row['variant'] not in first:
                first[row['variant']] = {k: row[k] for k in ('case', 'error', 'compilation_diagnostics') if k in row}
        shown.update(case_count=len(result['cases']), first_failure_by_variant=first,
                     report=str(Path(args.output)/'result.json'))
    print(json.dumps(shown, ensure_ascii=False, allow_nan=False))
    return int(result['status'] not in ('COMPILED', 'PARTIAL'))


if __name__ == '__main__':
    raise SystemExit(main())
