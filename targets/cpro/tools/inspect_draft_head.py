#!/usr/bin/env python3
"""Inspect retained full-vocabulary head layout; no compilation or device access."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'model'))
from qwen35_moe_dflash.draft.incremental.head_layout import inspect_head_layout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('air-export', 'compiled', 'ddk-root'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    try:
        report = inspect_head_layout(args.air_export, args.compiled, args.ddk_root)
    except (OSError, ValueError, KeyError, TypeError) as error:
        report = dict(status='FAILED', model_compilation='NOT_RUN', device_execution='NOT_RUN',
                      error=f'{type(error).__name__}: {error}')
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return int(report['status'] == 'FAILED')


if __name__ == '__main__':
    raise SystemExit(main())
