#!/usr/bin/env python3
"""Compile source first-mask/Attention with small synthetic packed QKV; no board access."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'model'))
from qwen35_moe_dflash.draft.incremental.attention_probe import compile_probe, inspect_compiled_probe
from qwen35_moe_dflash.draft.incremental.attention_inspect import inspect_attention_lengths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('air-export','source','result','ddk-root','ddk-id','tbe-diagnosis','output'):
        parser.add_argument('--'+name,required=name in ('ddk-root','ddk-id'))
    inspection = parser.add_mutually_exclusive_group()
    inspection.add_argument('--inspect-compiled',metavar='DIRECTORY',
                        help='read existing Attention OMC input-order metadata; no compilation or board access')
    inspection.add_argument('--inspect-lengths',metavar='DIRECTORY',
                        help='read selected DDK Attention sources and retained compiler records; no compilation or board access')
    parser.add_argument('--timeout-seconds',type=int)
    parser.add_argument('--replace',action='store_true',help='reuse a known failed host compilation in draft/tests; no archive')
    experiments = parser.add_mutually_exclusive_group()
    experiments.add_argument('--matrix',action='store_true',
                        help='compile independent mask/length/page/layout/geometry experiments, including Target geometry')
    experiments.add_argument('--controls',action='store_true',
                        help='retain source Attention and add four fixed mask/actual-KV-length control branches')
    experiments.add_argument('--runtime-lengths',action='store_true',
                        help='compare constant and runtime length sources with unchanged query16/KV224 values')
    args = parser.parse_args()
    if args.inspect_compiled or args.inspect_lengths:
        if args.replace or args.controls or args.runtime_lengths or args.matrix or any(getattr(args,name) is not None for name in
                ('air_export','source','result','tbe_diagnosis','output','timeout_seconds')):
            parser.error('inspection accepts only --ddk-root and --ddk-id')
    elif any(getattr(args,name) is None for name in ('air_export','source','result','tbe_diagnosis','output')):
        parser.error('compilation requires --air-export, --source, --result, --tbe-diagnosis and --output')
    if args.matrix and args.replace:
        parser.error('--matrix requires a fresh functional test directory')
    try:
        if args.inspect_lengths:
            result = inspect_attention_lengths(args.inspect_lengths,args.ddk_root,args.ddk_id)
        elif args.inspect_compiled:
            result = inspect_compiled_probe(args.inspect_compiled,args.ddk_root,args.ddk_id)
        elif args.matrix:
            from qwen35_moe_dflash.draft.incremental.attention_matrix import compile_matrix
            result = compile_matrix(args.air_export,args.source,args.result,args.output,
                ddk_root=args.ddk_root,ddk_id=args.ddk_id,tbe_diagnosis=args.tbe_diagnosis,
                timeout_seconds=1800 if args.timeout_seconds is None else args.timeout_seconds,
                progress=lambda s:print('attention-matrix: '+s,flush=True))
        else:
            result = compile_probe(args.air_export,args.source,args.result,args.output,
                ddk_root=args.ddk_root,ddk_id=args.ddk_id,tbe_diagnosis=args.tbe_diagnosis,
                timeout_seconds=1800 if args.timeout_seconds is None else args.timeout_seconds,replace=args.replace,
                controls=args.controls,runtime_lengths=args.runtime_lengths,
                progress=lambda s:print('attention-probe: '+s,flush=True))
    except (OSError,ValueError,KeyError,RuntimeError) as error:
        result = dict(status='FAILED',error=f'{type(error).__name__}: {error}')
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
    return int(result['status'] not in ('COMPILED','INSPECTED','PARTIAL'))


if __name__=='__main__':
    raise SystemExit(main())
