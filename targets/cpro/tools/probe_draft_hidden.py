#!/usr/bin/env python3
"""Prepare retained-input hidden checkpoint probes on the DDK server."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'model'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'ops/draft-top1/scripts'))
from qwen35_moe_dflash.draft.incremental.hidden_probe import compile_probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('air-export', 'source', 'result', 'ddk-root', 'ddk-id', 'tbe-diagnosis', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--timeout-seconds', type=int, default=1800)
    parser.add_argument('--suite', choices=('first-query', 'layer0'), default='first-query',
                        help='layer0 batches single-output checkpoints on three retained contexts')
    parser.add_argument('--replace', action='store_true', help='reuse a completed failed host compilation; retain its error summary')
    args = parser.parse_args()
    try:
        build = compile_probe
        if args.suite == 'layer0':
            from qwen35_moe_dflash.draft.incremental.hidden_layer_probe import compile_suite
            build = compile_suite
        result = build(args.air_export, args.source, args.result, args.output,
            ddk_root=args.ddk_root, ddk_id=args.ddk_id, tbe_diagnosis=args.tbe_diagnosis,
            timeout_seconds=args.timeout_seconds, replace=args.replace,
            progress=lambda s: print('hidden-probe: '+s, flush=True))
        summary = {k: v for k, v in result.items() if k not in ('cases', 'source_weights', 'input_record')}
        if args.suite == 'layer0':
            fields = ('case', 'status', 'model_compilation', 'checkpoint', 'error', 'compilation_diagnostics')
            summary['cases'] = [{k: r[k] for k in fields if k in r} for r in result['cases']]
        else:
            summary['cases'] = [{k: v for k, v in r.items() if k not in ('fixtures', 'extraction')}
                                for r in result['cases']]
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        summary = dict(status='FAILED', error=f'{type(error).__name__}: {error}')
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    return int(summary['status'] != 'COMPILED')


if __name__ == '__main__':
    raise SystemExit(main())
