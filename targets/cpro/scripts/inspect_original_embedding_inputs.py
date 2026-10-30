#!/usr/bin/env python3
"""Compare original ID/INT8 embedding/FP32 scale inputs using selected table rows."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))


def main(argv=None):
    from qwen35_moe_dflash.original_embedding_lookup import inspect_original_embedding_inputs
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weight-table', required=True)
    parser.add_argument('--scale-table', required=True)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--sequence-length', required=True, type=int, choices=(1, 64))
    parser.add_argument('--output', required=True)
    parser.add_argument('--expected-weight-sha256', help='Explicit one-time full weight-table SHA256 verification at initialization')
    parser.add_argument('--expected-scale-sha256', help='Explicit one-time full scale-table SHA256 verification at initialization')
    args = parser.parse_args(argv)
    try:
        result = inspect_original_embedding_inputs(args.weight_table, args.scale_table, args.input_dir, args.sequence_length,
            args.output, expected_weight_sha256=args.expected_weight_sha256, expected_scale_sha256=args.expected_scale_sha256)
        keys = ('status', 'output', 'sequence_length', 'position_count', 'matched_position_count', 'mismatch_positions',
            'embedding_mismatch_positions', 'scale_mismatch_positions', 'device_execution', 'input_mutation',
            'numerical_validation', 'source_authenticity', 'error')
        summary = {key: result[key] for key in keys if key in result}
        observation = result.get('lookup_observation', {})
        summary['selected_table_read_bytes'] = observation.get('selected_table_read_bytes', 0)
        summary['full_table_hash'] = observation.get('initialization', {}).get('full_table_hash', 'NOT_PERFORMED')
        summary['report'] = str(Path(args.output).resolve() / 'report.json')
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)
        return 0 if result['status'] == 'ORIGINAL_EMBEDDING_INPUT_ROWS_MATCHED' else 1
    except Exception as error:
        print(json.dumps({'status': 'FAILED', 'error': f'{type(error).__name__}: {error}'}), flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
