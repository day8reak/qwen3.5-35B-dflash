#!/usr/bin/env python3
"""Group model dependencies and remove completed tests after saving compact results."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--config', help='Default: ROOT/runtime.json')
    parser.add_argument('--clean-completed', action='store_true', help='Keep summary/CSV and remove confirmed completed tests')
    parser.add_argument('--apply', action='store_true', help='Move files and apply cleanup; default only prints the plan')
    args = parser.parse_args()
    from qwen35_moe_dflash.run_layout import organize
    try:
        result = organize(args.root, args.config or Path(args.root) / 'runtime.json',
                          apply=args.apply, clean_completed=args.clean_completed)
        result = {k: v for k, v in result.items() if k != 'original_config'}
    except Exception as error:
        result = dict(status='FAILED', error=f'{type(error).__name__}: {error}')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(result['status'] == 'FAILED')


if __name__ == '__main__':
    raise SystemExit(main())
