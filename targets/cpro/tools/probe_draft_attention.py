#!/usr/bin/env python3
"""Compile source first-mask/Attention with small synthetic packed QKV; no board access."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'model'))
from qwen35_moe_dflash.draft.incremental.attention_probe import compile_probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('air-export','source','result','ddk-root','ddk-id','tbe-diagnosis','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--timeout-seconds',type=int,default=1800)
    parser.add_argument('--replace',action='store_true',help='reuse a known failed host compilation in draft/tests; no archive')
    args = parser.parse_args()
    try:
        result = compile_probe(args.air_export,args.source,args.result,args.output,
            ddk_root=args.ddk_root,ddk_id=args.ddk_id,tbe_diagnosis=args.tbe_diagnosis,
            timeout_seconds=args.timeout_seconds,replace=args.replace,
            progress=lambda s:print('attention-probe: '+s,flush=True))
    except (OSError,ValueError,KeyError,RuntimeError) as error:
        result = dict(status='FAILED',error=f'{type(error).__name__}: {error}')
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
    return int(result['status']!='COMPILED')


if __name__=='__main__':
    raise SystemExit(main())
