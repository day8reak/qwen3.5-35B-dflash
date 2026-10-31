#!/usr/bin/env python3
"""Read bounded stdout/stderr tails from one completed failed board recovery."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recovery', required=True, help='Existing completed recovery directory with checkpoint/result and results/loop.json')
    parser.add_argument('--output', required=True, help='Separate new diagnostic directory below AI_RUN_DIR')
    parser.add_argument('--timeout-seconds', type=int, default=30)
    parser.add_argument('--hilog', action='store_true', help='Also read the current HiLog ring for the recorded native PID; this is a diagnostic snapshot, not exact-attempt evidence')
    args = parser.parse_args(argv)
    try:
        from qwen35_moe_dflash.board_serial_failure_logs import collect_board_serial_failure_logs
        result = collect_board_serial_failure_logs(args.recovery, args.output,
            timeout_seconds=args.timeout_seconds, hilog=args.hilog)
    except Exception as error:
        result = dict(status='FAILED', error=f'{type(error).__name__}: {error}')
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if result.get('status') == 'BOARD_SERIAL_FAILURE_LOGS_COLLECTED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
