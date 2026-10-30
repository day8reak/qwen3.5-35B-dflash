#!/usr/bin/env python3
"""One cache-only target Init-before-InitWeights experiment; no inference."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--prepared', help='Existing board-loop job or its immutable prepared directory')
    choice.add_argument('--recover', help='Existing load-probe execution; never relaunches the model')
    parser.add_argument('--output', required=True)
    parser.add_argument('--timeout-seconds', type=int, default=600)
    parser.add_argument('--lock-wait-seconds', type=int, default=1800)
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--apply', action='store_true', help='Run exactly one declared cached target load experiment')
    args = parser.parse_args(argv)
    if args.recover and (args.detach or args.apply):
        parser.error('--recover cannot change the original launch mode')
    if args.detach and not args.apply:
        parser.error('--detach requires --apply')
    try:
        from qwen35_moe_dflash.target_load_probe import run_target_load_probe, recover_target_load_probe
        if args.recover:
            result = recover_target_load_probe(args.recover, args.output, timeout_seconds=args.timeout_seconds)
        else:
            result = run_target_load_probe(args.prepared, args.output, apply=args.apply, detach=args.detach,
                timeout_seconds=args.timeout_seconds, lock_wait_seconds=args.lock_wait_seconds)
    except Exception as error:
        result = dict(status='FAILED', error=f'{type(error).__name__}: {error}')
    keys = ('status', 'stage', 'output', 'device_execution', 'model_compilation', 'model_inference', 'load_order',
        'runtime_stage', 'lifecycle', 'elapsed_ms', 'counts', 'error', 'execution_failure_stage', 'board_attempt', 'lock_state', 'next')
    print(json.dumps({key: result[key] for key in keys if key in result}, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if result.get('status') in {'TARGET_LOAD_PROBE_PLANNED', 'TARGET_LOAD_PROBE_PASSED', 'TARGET_LOAD_PROBE_STARTED'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
