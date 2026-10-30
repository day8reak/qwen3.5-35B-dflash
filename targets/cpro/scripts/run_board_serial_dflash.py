#!/usr/bin/env python3
"""Run the entire short-prompt DFlash loop on board using existing OMCs."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    required = ('decode-job', 'runtime-probe', 'state-lineage', 'source-torchair-pb2',
                'draft-bundle', 'weight-table', 'scale-table', 'prompt-input-ids')
    for name in required:
        parser.add_argument('--' + name)
    parser.add_argument('--prefill-feedback', help='Completed all80 prefill/decode mapping directory; enables one actual prefill64 call with explicit short-prompt padding controls')
    parser.add_argument('--output', required=True)
    parser.add_argument('--recover', help='Existing board-loop job or its board execution directory; never launches inference')
    parser.add_argument('--prompt-length', type=int, default=8)
    parser.add_argument('--max-new-tokens', type=int, default=8)
    parser.add_argument('--max-proposals-per-round', type=int, default=2)
    parser.add_argument('--minimum-draft-rounds', type=int, default=2)
    parser.add_argument('--timeout-seconds', type=int, default=7200)
    parser.add_argument('--lock-wait-seconds', type=int, default=1800)
    parser.add_argument('--board-model-root', default='/data/local/tmp/qwen3_5_moe')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--detach', action='store_true', help='Return after launch; the board finishes autonomously, then use --recover')
    parser.add_argument('--dump', action='store_true', default=None,
        help='Retain debug tensors/receipts on board and retrieve detailed evidence; default returns final tokens/status only')
    parser.add_argument('--resident', action='store_true', default=None,
        help='Explicit file-v2 same-process decode+draft residency; prefill remains a single original-tool call')
    args = parser.parse_args(argv)
    if args.recover:
        if args.apply or args.detach or args.dump is not None or args.resident is not None or args.prefill_feedback is not None or any(getattr(args, name.replace('-', '_')) is not None for name in required):
            parser.error('--recover uses the original prepared dump mode; omit --dump, --apply and model arguments')
    elif any(getattr(args, name.replace('-', '_')) is None for name in required):
        parser.error('new runs require ' + ', '.join('--' + name for name in required))
    elif args.detach and not args.apply:
        parser.error('--detach requires --apply')
    try:
        if args.recover:
            from qwen35_moe_dflash.board_serial_transport import recover_board_serial
            selected = Path(args.recover)
            if (selected / 'board').is_dir():
                selected /= 'board'
            result = recover_board_serial(selected, args.output, timeout_seconds=args.timeout_seconds)
        else:
            from qwen35_moe_dflash.board_serial_dflash import run_board_serial_dflash
            options = vars(args).copy(); options.pop('recover')
            options['dump'] = bool(args.dump)
            options['resident'] = bool(args.resident)
            result = run_board_serial_dflash(**options)
    except Exception as error:
        result = {'status': 'FAILED', 'error': f'{type(error).__name__}: {error}'}
    keys = ('status', 'stage', 'output', 'mode', 'device_execution', 'model_compilation',
            'numerical_validation', 'performance', 'generated_token_ids', 'counts', 'board_attempt',
            'lock_state', 'error', 'next', 'prepared', 'board_result')
    keys += ('dump', 'retrieval_mode', 'evidence_scope', 'acceptance', 'phase_latency', 'supervisor_timing', 'prompt_ingestion')
    keys += ('execution_mode', 'resident_lifecycle', 'resident_memory', 'target_lifecycle_policy')
    keys += ('execution_failure_stage', 'resident_failure_stage')
    print(json.dumps({key: result[key] for key in keys if key in result}, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if result.get('status') in {'BOARD_SERIAL_DFLASH_PLANNED', 'BOARD_SERIAL_DFLASH_STARTED',
        'BOARD_SERIAL_DFLASH_EXECUTED_UNVALIDATED', 'BOARD_SERIAL_DFLASH_RECOVERED_UNVALIDATED'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
