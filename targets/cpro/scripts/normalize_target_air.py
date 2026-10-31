#!/usr/bin/env python3
"""Use weight/ in target AIR bundles and their FileConstant references."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument('--root', help='Normalize prefill/air and decode/air')
    location.add_argument('--air-dir', help='One exported or shared AIR directory')
    parser.add_argument('--torchair-pb2', help='Optional source schema override; default uses bundled offline GE descriptor')
    parser.add_argument('--apply', action='store_true', help='Apply renames and AIR updates; default previews only')
    args = parser.parse_args()
    try:
        from qwen35_moe_dflash.target.air_weights import normalize_air_weights
        from qwen35_moe_dflash.toolchain.full_target import target_source_lock
        from contextlib import ExitStack
        folders = [Path(args.air_dir)] if args.air_dir else [Path(args.root) / role / 'air' for role in ('prefill', 'decode')]
        os.environ.setdefault('AI_RUN_DIR', args.root or str(Path(args.air_dir).resolve()))
        with ExitStack() as locks:
            for folder in folders:
                locks.enter_context(target_source_lock(folder / 'export.air'))
            # Validate both before either directory is changed.
            rows = [normalize_air_weights(p, torchair_pb2=args.torchair_pb2) for p in folders]
            if args.apply:
                rows = [normalize_air_weights(p, apply=True, torchair_pb2=args.torchair_pb2) for p in folders]
        print(json.dumps(dict(status='NORMALIZED' if args.apply else 'PLANNED', models=rows), ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        print(json.dumps(dict(status='FAILED', error=f'{type(error).__name__}: {error}'), ensure_ascii=False))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
