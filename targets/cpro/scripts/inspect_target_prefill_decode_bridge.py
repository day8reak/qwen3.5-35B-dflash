#!/usr/bin/env python3
"""Bind later prefill names to old retained files; diagnose bridge gaps only."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))


def main(argv=None):
    from qwen35_moe_dflash.target_prefill_decode_bridge import inspect_prefill_decode_bridge
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution', required=True)
    parser.add_argument('--named-runtime-probe', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    try:
        result = inspect_prefill_decode_bridge(args.execution, args.named_runtime_probe, args.output)
        keys = ('status', 'output', 'input_count', 'output_count', 'executed_input_byte_binding', 'feedback_prepared',
            'state_continuity', 'numerical_validation', 'error')
        summary = {key: result[key] for key in keys if key in result}
        summary['named_probe_witness'] = result.get('named_probe_witness', {}).get('status', 'NOT_BOUND')
        summary['required_next_evidence'] = [row['id'] for row in result.get('required_next_evidence', [])]
        summary['report'] = str(Path(args.output).resolve() / 'report.json')
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)
        return 0 if result['status'] == 'PREFILL_NAMED_SLOTS_AND_RETAINED_FILES_BOUND_BRIDGE_GAPS_REMAIN' else 1
    except Exception as error:
        print(json.dumps({'status': 'FAILED', 'error': f'{type(error).__name__}: {error}'}), flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
