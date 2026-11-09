#!/usr/bin/env python3
"""Collect all original Target vendor-call contracts in one bounded inspection."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode=True
REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'model'))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',required=True,help='private master-derived qwen3_5_moe source directory')
    parser.add_argument('--config',default=str(REPO/'specs/target-config.json'),help='model geometry JSON, not the private path YAML')
    parser.add_argument('--ddk-root',default=os.environ.get('CPRO_DDK_ROOT'))
    parser.add_argument('--prefill-air',help='existing AIR file, or directory containing exactly one .air file')
    parser.add_argument('--decode-air',help='existing AIR file, or directory containing exactly one .air file')
    parser.add_argument('--torchair-pb2')
    parser.add_argument('--output',required=True,help='report directory below AI_RUN_DIR; no tensor/model payload is generated')
    args=parser.parse_args(argv)
    try:
        from qwen35_moe_dflash.common.artifacts import output_path
        from qwen35_moe_dflash.target.operator_inventory import KIND,inspect,concise_report
        output=output_path(args.output)
        output.mkdir(parents=True,exist_ok=True)
        existing=output/'inventory.json'
        if existing.exists() and json.loads(existing.read_text()).get('artifact_kind')!=KIND:
            raise ValueError('report belongs to another task')
        receipt=output/'summary.json'
        if receipt.exists() and json.loads(receipt.read_text()).get('artifact_kind')!=KIND+'-summary':
            raise ValueError('summary belongs to another task')
        airs={name:path for name,path in (('prefill64',args.prefill_air),('decode1',args.decode_air)) if path}
        report=inspect(args.source_root,json.loads(Path(args.config).read_text()),
            ddk_root=args.ddk_root,airs=airs,torchair_pb2=args.torchair_pb2)
        temporary=output/'inventory.json.tmp'
        temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        temporary.replace(existing)
        temporary=output/'summary.json.tmp'
        temporary.write_text(json.dumps(concise_report(report),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        temporary.replace(receipt)
        summary={key:report[key] for key in ('status','model_compilation','device_execution','numerical_validation')}
        summary.update(families=len(report['operator_families']),logical_signatures=len(report['cases']),
            source_status=report['source']['status'],unknown_calls=report['source']['unknown_selected_calls'],
            source_observations=report['source']['observations'],
            contract_review=report['contract_review'],contract_summary=str(receipt),
            air_status={name:r['status'] for name,r in report['airs'].items()},
            air_inspection='REQUESTED' if airs else 'NOT_REQUESTED',
            partial_reasons=report['ddk']['partial_reasons'] if report['ddk'] else [],
            ddk_status=report['ddk']['status'] if report['ddk'] else 'NOT_INSPECTED',report=str(existing))
        print(json.dumps(summary,ensure_ascii=False,indent=2))
        return 0 if report['status']=='INVENTORIED' else 1
    except (OSError,ValueError,KeyError,RuntimeError,SyntaxError) as error:
        print(json.dumps(dict(status='FAILED',error=f'{type(error).__name__}: {error}'),ensure_ascii=False))
        return 1


if __name__=='__main__':
    raise SystemExit(main())
