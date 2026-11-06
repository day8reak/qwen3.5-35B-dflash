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
    parser.add_argument('--ddk-root',help='selected DDK schema, required for a PyTorch OMC candidate')
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--recover')
    mode.add_argument('--replay', help='reuse a completed fixture; optional compiled/source/air-export/ddk-root binds a RoPE, feature-pack or offline-weight candidate; no reference regeneration')
    mode.add_argument('--inspect',help='print existing comparison details locally; no board access')
    mode.add_argument('--analyze',help='recompute new-KV and repeat diagnostics from retained host tensors')
    mode.add_argument('--diagnose',help='inspect retained hidden zeros/repeated rows and controls; optional --source checks embedding rows locally')
    mode.add_argument('--diagnose-kv',help='compare all appended KV regions with the retained replay baseline; host-only')
    mode.add_argument('--compare-feature-pack',help='compare full packed candidate outputs/timings with its retained device baseline; host-only')
    mode.add_argument('--compare-offline-weights',help='compare all full-Draft outputs/timings for offline nd_n weights against the packed baseline; host-only')
    mode.add_argument('--audit-reference',help='CPU source-math replay against retained reference and board tensors; requires --source, --air-export and an --output JSON file')
    mode.add_argument('--audit-head',help='NumPy-only full-vocabulary ranking of retained hidden at Top1 mismatch positions; requires --source and --output JSON')
    parser.add_argument('--output')
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--atol',type=float)
    parser.add_argument('--rtol',type=float)
    parser.add_argument('--timeout-seconds',type=int,default=1800)
    parser.add_argument('--output-initialization', choices=('zero', 'untouched'),
                        help='--replay only: caller output initialization, default zero; SDK allocation behavior is unchanged')
    parser.add_argument('--kv-reproduction-case', type=int,
                        help='untouched same-OMC replay only: append 10 rewrite/reuse pairs for this case, without KV commits')
    args=parser.parse_args()
    if args.output_initialization is not None and not args.replay:
        parser.error('--output-initialization requires --replay')
    if args.kv_reproduction_case is not None and (not args.replay or args.compiled or args.output_initialization != 'untouched'):
        parser.error('--kv-reproduction-case requires --replay --output-initialization untouched without --compiled')
    from qwen35_moe_dflash.draft.incremental.accuracy import summarize_result
    if args.compare_feature_pack or args.compare_offline_weights:
        if (args.apply or any((args.config,args.compiled,args.source,args.air_export,args.output,args.ddk_root))
                or args.atol is not None or args.rtol is not None):
            parser.error('candidate comparison only reads retained candidate and baseline results')
        from qwen35_moe_dflash.draft.incremental.feature_pack_compare import compare_result
        report = compare_result(args.compare_feature_pack or args.compare_offline_weights,
                                offline_weights=bool(args.compare_offline_weights))
        print(json.dumps(report,ensure_ascii=False,allow_nan=False))
        return 0 if report['baseline_equivalence'] == 'PASSED' else 1
    if args.diagnose_kv:
        if (args.apply or any((args.config,args.compiled,args.source,args.air_export,args.output,args.ddk_root))
                or args.atol is not None or args.rtol is not None):
            parser.error('--diagnose-kv reads only the candidate result and its retained replay baseline')
        from qwen35_moe_dflash.draft.incremental.kv_diagnose import diagnose_kv
        print(json.dumps(diagnose_kv(args.diagnose_kv),ensure_ascii=False,allow_nan=False))
        return 0
    if args.audit_head:
        if (args.apply or any((args.config,args.compiled,args.ddk_root,args.air_export))
                or args.atol is not None or args.rtol is not None
                or not all((args.source,args.output))):
            parser.error('--audit-head requires only retained result, --source and --output JSON')
        from qwen35_moe_dflash.draft.incremental.head_audit import audit_head
        print(json.dumps(audit_head(args.audit_head,args.source,args.output,
            progress=lambda x: print('draft-head-audit: '+x,flush=True)),ensure_ascii=False,allow_nan=False))
        return 0
    if args.audit_reference:
        if (args.apply or any((args.config,args.compiled,args.ddk_root))
                or args.atol is not None or args.rtol is not None
                or not all((args.source,args.air_export,args.output))):
            parser.error('--audit-reference requires only retained result, --source, --air-export and --output JSON')
        from qwen35_moe_dflash.draft.incremental.host_audit import audit_reference
        print(json.dumps(audit_reference(args.audit_reference,args.source,args.air_export,args.output,
            progress=lambda x: print('draft-host-audit: '+x,flush=True)),ensure_ascii=False,allow_nan=False))
        return 0
    if args.diagnose:
        if (args.apply or any((args.config,args.compiled,args.air_export,args.output,args.ddk_root))
                or args.atol is not None or args.rtol is not None):
            parser.error('--diagnose reads retained results and optional --source only')
        from qwen35_moe_dflash.draft.incremental.accuracy_diagnose import diagnose_result
        print(json.dumps(diagnose_result(args.diagnose,source=args.source),ensure_ascii=False,allow_nan=False))
        return 0
    if args.inspect or args.analyze:
        if (args.apply or any((args.config,args.compiled,args.source,args.air_export,args.output,args.ddk_root))
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
    if args.replay:
        candidate = (args.compiled,args.source,args.air_export,args.ddk_root)
        if (not args.config or (any(candidate) and not all(candidate))
                or args.atol is not None or args.rtol is not None):
            parser.error('--replay requires config/output and unchanged gates; a candidate needs compiled/source/air-export/ddk-root together')
        result=start(args.config,args.compiled,args.source,args.air_export,args.output,apply=args.apply,
                     timeout_seconds=args.timeout_seconds,replay=args.replay,
                     ddk_root=args.ddk_root,
                     output_initialization=args.output_initialization or 'zero',
                     kv_reproduction_case=args.kv_reproduction_case)
    elif args.recover:
        if args.apply or any((args.config,args.compiled,args.source,args.air_export,args.ddk_root)):
            parser.error('--recover only accepts the original execution and a separate output')
        result=recover(args.recover,args.output)
    else:
        if not all((args.config,args.compiled,args.source,args.air_export)):
            parser.error('preparation requires --config, --compiled, --source and --air-export')
        result=start(args.config,args.compiled,args.source,args.air_export,args.output,
                     apply=args.apply,atol=0.01 if args.atol is None else args.atol,
                     rtol=0.01 if args.rtol is None else args.rtol,timeout_seconds=args.timeout_seconds,
                     ddk_root=args.ddk_root)
    print(json.dumps(summarize_result(result),ensure_ascii=False))
    return 1 if (result['status'] in ('FAILED','NUMERICAL_FAILED')
                 or result.get('kv_reproduction', {}).get('native_status') == 'FAILED'
                 or result.get('kv_reproduction', {}).get('numerical_validation') == 'FAILED') else 0


if __name__=='__main__':
    try:
        code=main()
    except Exception as error:
        print(json.dumps({'status':'FAILED','error':f'{type(error).__name__}: {error}'}))
        code=1
    raise SystemExit(code)
