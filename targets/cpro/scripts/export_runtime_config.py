#!/usr/bin/env python3
"""Import model files, ABI and state mappings into a portable runtime config."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', required=True, help='Existing resident-mixed prepared directory')
    parser.add_argument('--output', required=True, help='Destination runtime.json')
    args = parser.parse_args()
    from qwen35_moe_dflash.runtime_config import export_config
    from qwen35_moe_dflash.artifacts import output_path
    try:
        export_config(args.prepared, output_path(args.output))
        result = dict(status='RUNTIME_CONFIG_EXPORTED', config=str(Path(args.output).resolve()),
            model_compilation='NOT_RUN', device_execution='NOT_RUN', file_validation='size-and-abi-only')
    except Exception as error:
        result = dict(status='FAILED', error=str(error))
    print(json.dumps(result, ensure_ascii=False))
    return int(result['status'] == 'FAILED')


if __name__ == '__main__':
    raise SystemExit(main())
