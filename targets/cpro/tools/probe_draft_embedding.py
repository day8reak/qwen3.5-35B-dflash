#!/usr/bin/env python3
"""Extract/compile a PyTorch AIR lookup or input-RMS probe on the selected DDK server."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'model'))
from qwen35_moe_dflash.draft.incremental.embedding_probe import compile_probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('air-export', 'source', 'result', 'ddk-root', 'ddk-id', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--timeout-seconds', type=int, default=1800)
    parser.add_argument('--stage', choices=('lookup', 'input-rms'), default='lookup',
                        help='explicit IDs at Gather, or original anchor/mask construction through first RMS')
    args = parser.parse_args()
    try:
        result = compile_probe(args.air_export, args.source, args.output, result=args.result, ddk_root=args.ddk_root,
            ddk_id=args.ddk_id, stage=args.stage, timeout_seconds=args.timeout_seconds,
            progress=lambda text: print('embedding-probe: '+text, flush=True))
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        result = dict(status='FAILED', error=f'{type(error).__name__}: {error}')
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return int(result['status'] != 'COMPILED')


if __name__ == '__main__':
    raise SystemExit(main())
