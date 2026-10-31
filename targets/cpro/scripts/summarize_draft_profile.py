#!/usr/bin/env python3
"""Build sorted operator/type/device tables from retained vendor profiling CSVs."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    from qwen35_moe_dflash.draft_profile_csv import summarize_profile
    try:
        result = summarize_profile(args.profile_dir, args.output)
        print(json.dumps({k: result[k] for k in ('status', 'operator_count', 'operator_duration_sum_ms',
            'by_operator_type', 'by_device_type', 'model_profiles', 'scope')}, ensure_ascii=False, allow_nan=False))
        print('完整表：' + str(Path(args.output) / 'operators.csv'))
        return 0
    except (OSError, ValueError) as error:
        print(json.dumps({'status': 'FAILED', 'error': str(error)}, ensure_ascii=False))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
