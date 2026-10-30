#!/usr/bin/env python3
"""Retrieve and verify one completed original continuation; never rerun it."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))


def main(argv=None):
    from qwen35_moe_dflash.target_decode_continuation_recovery import recover_decode_continuation
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--timeout-seconds', type=int, default=300)
    args = parser.parse_args(argv)
    try:
        result = recover_decode_continuation(args.execution, args.output, timeout_seconds=args.timeout_seconds)
        keys = ('status', 'stage', 'output', 'board_attempt', 'tool_process_id', 'model_inference', 'input_upload',
            'model_upload', 'state_feedback', 'feedback_input_count', 'bound_child_output_count', 'lock_state',
            'numerical_validation', 'state_continuity', 'error')
        print(json.dumps({key: result[key] for key in keys if key in result}, ensure_ascii=False, allow_nan=False), flush=True)
        return 0 if result['status'] == 'DECODE_CONTINUATION_OUTPUTS_RECOVERED_UNVALIDATED' else 1
    except Exception as error:
        print(json.dumps({'status': 'FAILED', 'error': f'{type(error).__name__}: {error}'}), flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
