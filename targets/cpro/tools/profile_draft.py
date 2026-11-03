#!/usr/bin/env python3
"""Inspect Draft partition boundaries or collect a vendor operator profile."""
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
    source.add_argument('--accuracy', help='completed incremental accuracy recovery directory')
    source.add_argument('--recover', help='same retained profiling execution, no model relaunch')
    source.add_argument('--inspect-boundaries', metavar='PYTORCH_OMC_DIR',
                        help='read current compiled partition metadata locally; no compiler or board access')
    source.add_argument('--analyze-boundaries', metavar='BOUNDARIES_JSON',
                        help='trace partition dependencies in saved full boundary JSON; no SDK or board access')
    parser.add_argument('--accuracy-result', help='retained accuracy result binding for --inspect-boundaries')
    parser.add_argument('--ddk-root', help='selected DDK schema for --inspect-boundaries; no library execution')
    parser.add_argument('--tool-inspection', help='existing target-board-tool-inspection-writable')
    parser.add_argument('--model-tool', default='/data/local/tmp/tools_sysdbg/model_run_tool_internal')
    parser.add_argument('--modelmanager-itf', choices=('V1', 'V2'),
                        help='--accuracy only: compare the same embedded OMC through V1 (default) or V2 file restore')
    parser.add_argument('--data-proc-tool', default='/data/local/tmp/tools_sysdbg/data_proc_tool_internal')
    parser.add_argument('--output', required=True, help='separate fixed test slot below AI_RUN_DIR')
    parser.add_argument('--apply', action='store_true', help='start one detached profiling invocation')
    parser.add_argument('--allow-numerical-failure', action='store_true',
                        help='profile a completed NUMERICAL_FAILED result for diagnosis; keep accuracy failed')
    parser.add_argument('--timeout-seconds', type=int, default=600)
    parser.add_argument('--lock-wait-seconds', type=int, default=1800)
    args = parser.parse_args(argv)
    if args.analyze_boundaries:
        if (args.accuracy_result or args.ddk_root or args.apply or args.tool_inspection
                or args.allow_numerical_failure or args.modelmanager_itf):
            parser.error('--analyze-boundaries needs only the saved JSON and --output; omit execution flags')
        try:
            from qwen35_moe_dflash.draft.boundary_dependencies import analyze_boundaries
            result = analyze_boundaries(args.analyze_boundaries, args.output)
            print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
            return 0
        except Exception as error:
            print(json.dumps(dict(status='FAILED', model_compilation='NOT_RUN', device_execution='NOT_RUN',
                error=f'{type(error).__name__}: {error}'), ensure_ascii=False), flush=True)
            return 1
    if args.inspect_boundaries:
        if (not args.accuracy_result or not args.ddk_root or args.apply or args.tool_inspection
                or args.allow_numerical_failure or args.modelmanager_itf):
            parser.error('--inspect-boundaries requires --accuracy-result, --ddk-root and --output JSON; omit execution flags')
        try:
            from qwen35_moe_dflash.draft.runtime_boundaries import inspect_boundaries
            result = inspect_boundaries(args.inspect_boundaries, args.accuracy_result, args.ddk_root, args.output)
            print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
            return 0
        except Exception as error:
            print(json.dumps(dict(status='FAILED', model_compilation='NOT_RUN', device_execution='NOT_RUN',
                error=f'{type(error).__name__}: {error}'), ensure_ascii=False), flush=True)
            return 1
    if args.accuracy_result or args.ddk_root:
        parser.error('--accuracy-result and --ddk-root require --inspect-boundaries')
    if args.allow_numerical_failure and not args.accuracy:
        parser.error('--allow-numerical-failure requires --accuracy')
    if args.modelmanager_itf and not args.accuracy:
        parser.error('--modelmanager-itf requires --accuracy; recovery uses the retained interface')
    if args.prepared and (not args.apply or not args.tool_inspection):
        parser.error('--prepared requires --apply and --tool-inspection')
    if args.accuracy and (not args.apply or args.tool_inspection):
        parser.error('--accuracy requires --apply; help is inspected directly on the selected board')
    if args.recover and (args.apply or args.tool_inspection):
        parser.error('--recover does not accept --apply or --tool-inspection')
    from qwen35_moe_dflash.draft.profile_board import start_profile, recover_profile
    try:
        if args.recover:
            result = recover_profile(args.recover, args.output)
        else:
            result = start_profile(args.prepared, args.tool_inspection, args.output,
                parser_tool=args.data_proc_tool, timeout_seconds=args.timeout_seconds,
                lock_wait_seconds=args.lock_wait_seconds,accuracy=args.accuracy,model_tool=args.model_tool,
                allow_numerical_failure=args.allow_numerical_failure,
                model_manager_interface=args.modelmanager_itf or 'V1')
        keys = ('status', 'stage', 'device_execution', 'model_compilation', 'numerical_validation',
            'source_numerical_validation', 'profiling_scope', 'model_manager_interface', 'model_restore_mode',
            'board_attempt', 'model_process_id', 'model_exitcode', 'parser_exitcode',
            'csv_directory', 'csv_count', 'csv_files', 'csv_analysis', 'input_revalidation',
            'lock_state', 'error', 'lock_release_errors', 'next')
        print(json.dumps({k: result[k] for k in keys if k in result}, ensure_ascii=False, allow_nan=False), flush=True)
        return 1 if result['status'] == 'FAILED' else 0
    except Exception as error:
        print(json.dumps({'status': 'FAILED', 'error': f'{type(error).__name__}: {error}'}, ensure_ascii=False), flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
