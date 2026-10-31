#!/usr/bin/env python3
"""Prepare, execute and compare standalone incremental Draft without Target inference."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'model'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--compiled')
    parser.add_argument('--source',help='prepared FP16 weights directory containing prepared.json')
    parser.add_argument('--air-export')
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--recover')
    mode.add_argument('--inspect',help='print existing comparison details locally; no board access')
    mode.add_argument('--analyze',help='recompute new-KV and repeat diagnostics from retained host tensors')
    parser.add_argument('--output')
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--atol',type=float)
    parser.add_argument('--rtol',type=float)
    parser.add_argument('--timeout-seconds',type=int,default=1800)
    args=parser.parse_args()
    from qwen35_moe_dflash.draft.incremental.accuracy import summarize_result
    if args.inspect or args.analyze:
        if (args.apply or any((args.config,args.compiled,args.source,args.air_export,args.output))
                or args.atol is not None or args.rtol is not None):
            parser.error('--inspect/--analyze only read existing results; comparison gates cannot be changed')
        if args.analyze:
            from qwen35_moe_dflash.draft.incremental.accuracy import analyze_result
            print(json.dumps(analyze_result(args.analyze),ensure_ascii=False))
            return 0
        from qwen35_moe_dflash.execution.runtime_config import read_json
        path=Path(args.inspect)
        result=read_json(path/'result.json' if path.is_dir() else path, 4 << 20)
        if result.get('artifact_kind') != 'qwen35-incremental-draft-accuracy':
            parser.error('expected an incremental Draft accuracy report')
        print(json.dumps(summarize_result(result),ensure_ascii=False))
        return 0
    if not args.output:
        parser.error('preparation/recovery requires --output')
    from qwen35_moe_dflash.draft.incremental.accuracy_board import start,recover
    if args.recover:
        if args.apply or any((args.config,args.compiled,args.source,args.air_export)):
            parser.error('--recover only accepts the original execution and a separate output')
        result=recover(args.recover,args.output)
    else:
        if not all((args.config,args.compiled,args.source,args.air_export)):
            parser.error('preparation requires --config, --compiled, --source and --air-export')
        result=start(args.config,args.compiled,args.source,args.air_export,args.output,
                     apply=args.apply,atol=0.01 if args.atol is None else args.atol,
                     rtol=0.01 if args.rtol is None else args.rtol,timeout_seconds=args.timeout_seconds)
    print(json.dumps(summarize_result(result),ensure_ascii=False))
    return 1 if result['status'] in ('FAILED','NUMERICAL_FAILED') else 0


if __name__=='__main__':
    try:
        code=main()
    except Exception as error:
        print(json.dumps({'status':'FAILED','error':f'{type(error).__name__}: {error}'}))
        code=1
    raise SystemExit(code)
