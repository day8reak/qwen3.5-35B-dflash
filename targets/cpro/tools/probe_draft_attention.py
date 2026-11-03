#!/usr/bin/env python3
"""Compile source first-mask/Attention with small synthetic packed QKV; no board access."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'model'))
from qwen35_moe_dflash.draft.incremental.attention_probe import compile_probe, inspect_compiled_probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('air-export','source','result','ddk-root','ddk-id','tbe-diagnosis','output'):
        parser.add_argument('--'+name,required=name in ('ddk-root','ddk-id'))
    parser.add_argument('--inspect-compiled',metavar='DIRECTORY',
                        help='read existing Attention OMC input-order metadata; no compilation or board access')
    parser.add_argument('--timeout-seconds',type=int)
    parser.add_argument('--replace',action='store_true',help='reuse a known failed host compilation in draft/tests; no archive')
    parser.add_argument('--controls',action='store_true',
                        help='retain source Attention and add four fixed mask/actual-KV-length control branches')
    args = parser.parse_args()
    if args.inspect_compiled:
        if args.replace or args.controls or any(getattr(args,name) is not None for name in
                ('air_export','source','result','tbe_diagnosis','output','timeout_seconds')):
            parser.error('inspection accepts only --ddk-root and --ddk-id')
    elif any(getattr(args,name) is None for name in ('air_export','source','result','tbe_diagnosis','output')):
        parser.error('compilation requires --air-export, --source, --result, --tbe-diagnosis and --output')
    try:
        if args.inspect_compiled:
            result = inspect_compiled_probe(args.inspect_compiled,args.ddk_root,args.ddk_id)
        else:
            result = compile_probe(args.air_export,args.source,args.result,args.output,
                ddk_root=args.ddk_root,ddk_id=args.ddk_id,tbe_diagnosis=args.tbe_diagnosis,
                timeout_seconds=1800 if args.timeout_seconds is None else args.timeout_seconds,replace=args.replace,
                controls=args.controls,
                progress=lambda s:print('attention-probe: '+s,flush=True))
    except (OSError,ValueError,KeyError,RuntimeError) as error:
        result = dict(status='FAILED',error=f'{type(error).__name__}: {error}')
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
    return int(result['status'] not in ('COMPILED','INSPECTED'))


if __name__=='__main__':
    raise SystemExit(main())
