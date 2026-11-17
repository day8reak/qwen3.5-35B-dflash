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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--prepared', help='Existing resident-mixed prepared directory')
    source.add_argument('--config', help='Existing runtime.json to connect to a packed Draft')
    parser.add_argument('--packed-draft', help='Published packed C16/cap192/K15 OMC directory')
    parser.add_argument('--draft-source', help='Original FP16 Draft prepared directory containing prepared.json')
    parser.add_argument('--output', required=True, help='Destination runtime.json')
    args = parser.parse_args()
    from qwen35_moe_dflash.execution.runtime_config import export_config, configure_packed_draft
    from qwen35_moe_dflash.common.artifacts import output_path
    try:
        if args.config:
            if not args.packed_draft or not args.draft_source:
                raise ValueError('--config requires --packed-draft and --draft-source')
            configure_packed_draft(args.config, args.packed_draft, args.draft_source, output_path(args.output))
        else:
            if args.packed_draft or args.draft_source:
                raise ValueError('use --config when connecting a packed Draft')
            export_config(args.prepared, output_path(args.output))
        result = dict(status='RUNTIME_CONFIG_EXPORTED', config=str(Path(args.output).resolve()),
            model_compilation='NOT_RUN', device_execution='NOT_RUN', file_validation='size-and-abi-only')
    except Exception as error:
        result = dict(status='FAILED', error=str(error))
    print(json.dumps(result, ensure_ascii=False))
    return int(result['status'] == 'FAILED')


if __name__ == '__main__':
    raise SystemExit(main())
