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
                'draft-bundle', 'weight-table', 'scale-table')
    for name in required:
        parser.add_argument('--' + name)
    prompt = parser.add_mutually_exclusive_group()
    prompt.add_argument('--prompt-input-ids')
    prompt.add_argument('--text', help='UTF-8 user text; encoded on board with the original tokenizer')
    prompt.add_argument('--text-file', help='UTF-8 text file, useful for multiline prompts')
    parser.add_argument('--tokenizer-dir', help='Pinned original tokenizer source directory (prepare_board_tokenizer.py output/source)')
    parser.add_argument('--prompt-format', choices=('chat', 'raw'), default='chat', help='chat: single user, thinking disabled; raw: no template')
    parser.add_argument('--prefill-feedback', help='Completed all80 prefill/decode mapping directory; enables one actual prefill64 call with explicit short-prompt padding controls')
    parser.add_argument('--output', required=True)
    parser.add_argument('--recover', help='Existing board-loop job or its board execution directory; never launches inference')
    parser.add_argument('--prompt-length', type=int, help='Token-file prefix length, default8; text length is derived automatically')
    parser.add_argument('--max-new-tokens', type=int, default=8)
    parser.add_argument('--max-proposals-per-round', type=int, default=2)
    parser.add_argument('--minimum-draft-rounds', type=int, default=2)
    parser.add_argument('--timeout-seconds', type=int, default=7200)
    parser.add_argument('--lock-wait-seconds', type=int, default=1800)
    parser.add_argument('--board-model-root', default='/data/local/tmp/qwen3_5_moe')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--detach', action='store_true', help='Return after launch; the board finishes autonomously, then use --recover')
    parser.add_argument('--dump', action='store_true', default=None,
        help='Retain debug tensors/receipts on board and retrieve detailed evidence; default uses in-memory state feedback without tensor diagnostics and returns final tokens/timing only')
    parser.add_argument('--resident', action='store_true', default=None,
        help='Required for new runs: V2 prefill once, resident V2 decode and V1 embedded draft; historical policies are recover-only')
    parser.add_argument('--draft-preread', action='store_true', default=None,
        help='Startup experiment: sequentially read the mapped draft OMC with a 1MiB buffer before Load; time is included in total load latency')
    args = parser.parse_args(argv)
    if args.recover:
        if any(v is not None for v in (args.text, args.text_file, args.prompt_input_ids, args.tokenizer_dir, args.draft_preread)) or args.prompt_format != 'chat' or args.apply or args.detach or args.dump is not None or args.resident is not None or args.prefill_feedback is not None or any(getattr(args, name.replace('-', '_')) is not None for name in required):
            parser.error('--recover uses the original prepared dump mode; omit --dump, --apply and model arguments')
    elif any(getattr(args, name.replace('-', '_')) is None for name in required):
        parser.error('new runs require ' + ', '.join('--' + name for name in required))
    elif all(v is None for v in (args.text, args.text_file, args.prompt_input_ids)):
        parser.error('provide --text, --text-file, or --prompt-input-ids')
    elif (args.text is not None or args.text_file is not None) and args.prompt_length is not None:
        parser.error('--prompt-length is for token files; text uses its full encoded length')
    elif not args.resident:
        parser.error('new runs require --resident for explicit mixed-API residency; old modes support historical --recover only')
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
            options['draft_preread'] = bool(args.draft_preread)
            result = run_board_serial_dflash(**options)
    except Exception as error:
        result = {'status': 'FAILED', 'error': f'{type(error).__name__}: {error}'}
    keys = ('status', 'stage', 'output', 'mode', 'device_execution', 'model_compilation',
            'numerical_validation', 'performance', 'generated_token_ids', 'counts', 'board_attempt',
            'lock_state', 'error', 'next', 'prepared', 'board_result')
    keys += ('dump', 'retrieval_mode', 'evidence_scope', 'acceptance', 'phase_latency', 'supervisor_timing', 'prompt_ingestion')
    keys += ('execution_mode', 'model_load_api', 'model_load_apis', 'resident_lifecycle', 'resident_memory', 'target_lifecycle_policy')
    keys += ('external_weight_release', 'runtime_profile', 'native_api_latency',
             'diagnostics', 'intermediate_tensor_io', 'startup_payload_validation')
    keys += ('generated_text', 'generated_text_with_special_tokens', 'text_io', 'text_latency')
    keys += ('draft_preread',)
    keys += ('execution_failure_stage', 'resident_failure_stage', 'prefill_failure_stage', 'prefill_lifecycle')
    print(json.dumps({key: result[key] for key in keys if key in result}, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if result.get('status') in {'BOARD_SERIAL_DFLASH_PLANNED', 'BOARD_SERIAL_DFLASH_STARTED',
        'BOARD_SERIAL_DFLASH_EXECUTED_UNVALIDATED', 'BOARD_SERIAL_DFLASH_RECOVERED_UNVALIDATED'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
