#!/usr/bin/env python3
"""Retain the measured packed Draft baseline in shared air/omc directories."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'model'))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--compiled', required=True, help='measured baseline, not the nd_n candidate')
    p.add_argument('--air-export', required=True)
    p.add_argument('--source', required=True, help='original prepared weights outside draft/tests')
    p.add_argument('--comparison', required=True, help='completed nd_n-result; reuses retained comparison outputs')
    p.add_argument('--ddk-root', default=os.environ.get('CPRO_DDK_ROOT'))
    p.add_argument('--apply', action='store_true')
    a = p.parse_args(argv)
    try:
        from qwen35_moe_dflash.draft.incremental.publication import archive_baseline
        result = archive_baseline(a.root, a.compiled, a.air_export, a.source, a.comparison,
                                  a.ddk_root, apply=a.apply)
        shown = {k: v for k, v in result.items() if k != 'comparison'}
        print(json.dumps(shown, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(json.dumps(dict(status='FAILED', error=f'{type(error).__name__}: {error}',
            model_compilation='NOT_RUN', device_execution='NOT_RUN'), ensure_ascii=False))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
