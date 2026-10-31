#!/usr/bin/env python3
"""Collect one S64 Draft operator profile using the existing board vendor tools."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--prepared', help='existing real-feature-draft-plan; unchanged S64 OMC')
    source.add_argument('--recover', help='same retained profiling execution, no model relaunch')
    parser.add_argument('--tool-inspection', help='existing target-board-tool-inspection-writable')
    parser.add_argument('--data-proc-tool', default='/data/local/tmp/tools_sysdbg/data_proc_tool_internal')
    parser.add_argument('--output', required=True, help='separate new directory below AI_RUN_DIR')
    parser.add_argument('--apply', action='store_true', help='start one detached profiling invocation')
    parser.add_argument('--timeout-seconds', type=int, default=600)
    parser.add_argument('--lock-wait-seconds', type=int, default=1800)
    args = parser.parse_args(argv)
    if args.prepared and (not args.apply or not args.tool_inspection):
        parser.error('--prepared requires --apply and --tool-inspection')
    if args.recover and (args.apply or args.tool_inspection):
        parser.error('--recover does not accept --apply or --tool-inspection')
    from qwen35_moe_dflash.draft_profile_board import start_profile, recover_profile
    try:
        if args.recover:
            result = recover_profile(args.recover, args.output)
        else:
            result = start_profile(args.prepared, args.tool_inspection, args.output,
                parser_tool=args.data_proc_tool, timeout_seconds=args.timeout_seconds,
                lock_wait_seconds=args.lock_wait_seconds)
        keys = ('status', 'stage', 'device_execution', 'model_compilation', 'numerical_validation',
            'board_attempt', 'model_process_id', 'model_exitcode', 'parser_exitcode', 'operator_count',
            'model_profiles', 'summary', 'table', 'lock_state', 'error', 'lock_release_errors', 'next')
        print(json.dumps({k: result[k] for k in keys if k in result}, ensure_ascii=False, allow_nan=False), flush=True)
        return 1 if result['status'] == 'FAILED' else 0
    except Exception as error:
        print(json.dumps({'status': 'FAILED', 'error': f'{type(error).__name__}: {error}'}, ensure_ascii=False), flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
