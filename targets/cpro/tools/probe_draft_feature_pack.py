#!/usr/bin/env python3
"""Compile paired C16 feature prefixes, optionally testing the full-dot 2x2 Conv."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'model'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'ops/draft-top1/scripts'))
from qwen35_moe_dflash.draft.incremental.feature_pack_probe import compile_probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('air-export', 'result', 'ddk-root', 'ddk-id', 'tbe-diagnosis', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--timeout-seconds', type=int, default=1800)
    parser.add_argument('--projection-kernel', type=int, choices=[2],
        help='compare accepted packed 4x4 Conv with full-dot 2x2; requires its full-model result')
    parser.add_argument('--retry-failed', action='store_true',
        help='retry known failed OMG cases in this directory; keep successful OMCs and retained fixtures')
    args = parser.parse_args()
    try:
        result = compile_probe(args.air_export, args.result, args.output,
            ddk_root=args.ddk_root, ddk_id=args.ddk_id, tbe_diagnosis=args.tbe_diagnosis,
            timeout_seconds=args.timeout_seconds, retry_failed=args.retry_failed,
            projection_kernel=args.projection_kernel,
            progress=lambda s: print('feature-pack: '+s, flush=True))
        summary = {k: v for k, v in result.items() if k not in ('cases', 'models', 'source_cases')}
        summary['models'] = [{k: v for k, v in r.items() if k != 'extraction'} for r in result['models']]
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        summary = dict(status='FAILED', error=f'{type(error).__name__}: {error}')
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    return int(summary['status'] != 'COMPILED')


if __name__ == '__main__':
    raise SystemExit(main())
