#!/usr/bin/env python3
"""Extract/compile a PyTorch AIR lookup or input-RMS probe on the selected DDK server."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'model'))
from qwen35_moe_dflash.draft.incremental.embedding_probe import compile_probe, inspect_compiled_probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('air-export', 'source', 'result', 'ddk-root', 'ddk-id', 'output'):
        parser.add_argument('--'+name, required=name in ('ddk-root', 'ddk-id'))
    parser.add_argument('--inspect-compiled', metavar='DIRECTORY',
                        help='Read an existing probe OMC interface; no compilation, fixture writes or board access')
    parser.add_argument('--timeout-seconds', type=int)
    parser.add_argument('--stage', choices=('lookup', 'input-rms', 'input-rms-taps'),
                        help='explicit IDs, original first-RMS prefix, or prefix with x/gamma readback')
    args = parser.parse_args()
    if args.inspect_compiled:
        if any(getattr(args, name) is not None for name in ('air_export', 'source', 'result', 'output', 'stage', 'timeout_seconds')):
            parser.error('--inspect-compiled accepts only --ddk-root and --ddk-id')
    elif any(getattr(args, name) is None for name in ('air_export', 'source', 'result', 'output')):
        parser.error('compilation requires --air-export, --source, --result and --output')
    try:
        if args.inspect_compiled:
            result = inspect_compiled_probe(args.inspect_compiled, args.ddk_root, args.ddk_id)
        else:
            result = compile_probe(args.air_export, args.source, args.output, result=args.result, ddk_root=args.ddk_root,
                ddk_id=args.ddk_id, stage=args.stage or 'lookup',
                timeout_seconds=1800 if args.timeout_seconds is None else args.timeout_seconds,
                progress=lambda text: print('embedding-probe: '+text, flush=True))
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        result = dict(status='FAILED', error=f'{type(error).__name__}: {error}')
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return int(result['status'] not in ('COMPILED', 'INSPECTED'))


if __name__ == '__main__':
    raise SystemExit(main())
