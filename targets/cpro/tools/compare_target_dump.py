#!/usr/bin/env python3
"""Prepare an explicit mapping or compare whole-graph dumps offline; never runs a model."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare', help='create unbound mapping from completed original 310P capture')
    p.add_argument('--capture', required=True)
    p.add_argument('--output', required=True)
    p = sub.add_parser('compare', help='batch compare mapped raw/NPY files without DDK or device')
    p.add_argument('--mapping', required=True)
    p.add_argument('--golden-root', required=True)
    p.add_argument('--actual-root', required=True)
    p.add_argument('--output', required=True)
    args = parser.parse_args()
    from qwen35_moe_dflash.target.dump_compare import compare, prepare
    try:
        if args.command == 'prepare':
            result = prepare(args.capture, args.output)
        else:
            result = compare(args.mapping, args.golden_root, args.actual_root, args.output)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(json.dumps(dict(status='INVALID_REQUEST', error=f'{type(exc).__name__}: {exc}'), ensure_ascii=False))
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 1 if result['status'] in ('FAILED', 'PARTIAL') else 0


if __name__ == '__main__':
    raise SystemExit(main())
