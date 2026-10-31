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
    parser.add_argument('--closed-attempt', action='append', default=[],
                        help='Exact old board attempt path the operator has confirmed exited; never kills processes or removes board locks')
    parser.add_argument('--confirm-idle', action='store_true',
                        help='Operator confirms all old requests have exited and no board job is running; retire stale local test records')
    parser.add_argument('--details', action='store_true', help='Print every move, retained item and cleanup record')
    args = parser.parse_args()
    from qwen35_moe_dflash.execution.run_layout import organize
    try:
        result = organize(args.root, args.config or Path(args.root) / 'runtime.json',
                          apply=args.apply, clean_completed=args.clean_completed,
                          closed_attempts=args.closed_attempt, confirm_idle=args.confirm_idle)
        result = {k: v for k, v in result.items() if k != 'original_config'}
        if not args.details:
            result = dict(status=result['status'],
                moved=len(result['completed_moves']) if args.apply else len(result['moves']),
                cleaned=len(result['completed_deletions']) if args.apply else len(result['delete']),
                retained=result['keep'], unresolved_board_attempts=result['unresolved_board_attempts'],
                journal=str(Path(args.root) / '.run-layout.json') if args.apply else None)
    except Exception as error:
        result = dict(status='FAILED', error=f'{type(error).__name__}: {error}')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(result['status'] == 'FAILED')


if __name__ == '__main__':
    raise SystemExit(main())
