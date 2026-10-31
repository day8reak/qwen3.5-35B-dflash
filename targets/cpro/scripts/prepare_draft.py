#!/usr/bin/env python3
"""Prepare shared six-layer Draft weights, without AIR export or target inference."""
import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target-weights', required=True)
    parser.add_argument('--draft-weights', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    from qwen35_moe_dflash.draft.standalone import prepare_draft
    try:
        result = prepare_draft(args.output, target_path=args.target_weights,
                               draft_path=args.draft_weights)
        print(json.dumps(dict(status=result['status'], prepared=args.output,
                              model_compilation='NOT_RUN', device_execution='NOT_RUN')))
        return int(result['status'] != 'PREPARED')
    except Exception as error:
        print(json.dumps(dict(status='FAILED', error=str(error))))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

