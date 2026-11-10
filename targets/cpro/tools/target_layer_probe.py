#!/usr/bin/env python3
"""Export one CPro region at a time, compile batches, and compare board dumps."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'model'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for command in ('export', 'export-one'):
        s = sub.add_parser(command)
        s.add_argument('--capture', required=True)
        s.add_argument('--output', required=True)
        s.add_argument('--source-root', required=True)
        s.add_argument('--converter-script', required=True)
        s.add_argument('--torchair-pb2')
        if command == 'export':
            s.add_argument('--points', nargs='+')
            s.add_argument('--kinds', nargs='+')
            s.add_argument('--timeout', type=int, default=900)
        else:
            s.add_argument('--region', required=True)
            s.add_argument('--output-index', required=True, type=int)
    s = sub.add_parser('compile')
    s.add_argument('--batch', required=True, help='existing AIR batch (read-only)')
    s.add_argument('--output', required=True, help='fresh OMC batch below AI_RUN_DIR, outside the AIR batch and DDK')
    s.add_argument('--ddk-root', default=os.environ.get('CPRO_DDK_ROOT'), required=not bool(os.environ.get('CPRO_DDK_ROOT')))
    s.add_argument('--ddk-id', default=os.environ.get('CPRO_DDK_ID'), required=not bool(os.environ.get('CPRO_DDK_ID')))
    s.add_argument('--tbe-diagnosis')
    s.add_argument('--timeout', type=int, default=900)
    s.add_argument('--use-origin-format', action='store_const', const=True, default=None)
    s = sub.add_parser('inspect-compile', help='read retained AIR/IR metadata without compiling or running models')
    s.add_argument('--batch', required=True, help='existing OMC batch')
    s.add_argument('--output', required=True)
    s.add_argument('--cases', nargs='+', required=True)
    s.add_argument('--ddk-root', default=os.environ.get('CPRO_DDK_ROOT'))
    s = sub.add_parser('compare')
    s.add_argument('--fixture', required=True)
    s.add_argument('--native', required=True)
    s.add_argument('--output', required=True)
    s.add_argument('--atol', type=float)
    s.add_argument('--rtol', type=float)
    s = sub.add_parser('list')
    s.add_argument('--capture', required=True)
    s = sub.add_parser('replay-qlinear-failure')
    s.add_argument('--capture', required=True)
    s.add_argument('--output', required=True)
    s.add_argument('--device', default='npu:0')
    s.add_argument('--group-size', type=int, choices=[256])
    a = p.parse_args()
    from qwen35_moe_dflash.target.layer_probe import batch
    from qwen35_moe_dflash.target.layer_probe.common import new_output, save_json
    if a.command == 'replay-qlinear-failure':
        from qwen35_moe_dflash.target.layer_probe.qlinear import replay_failure
        result = replay_failure(a.capture, a.output, a.device, a.group_size)
    elif a.command == 'export':
        result = batch.export_batch(a.capture, a.output, a.source_root, a.converter_script,
            points=a.points, kinds=a.kinds, timeout=a.timeout, torchair_pb2=a.torchair_pb2)
    elif a.command == 'export-one':
        from qwen35_moe_dflash.target.layer_probe.replay import export_one
        result = export_one(a.capture, a.region, a.output_index, a.output, a.source_root, a.converter_script, a.torchair_pb2)
    elif a.command == 'compile':
        result = batch.compile_batch(a.batch, a.output, a.ddk_root, a.ddk_id, tbe_diagnosis=a.tbe_diagnosis,
                                     timeout=a.timeout, use_origin_format=a.use_origin_format)
    elif a.command == 'inspect-compile':
        from qwen35_moe_dflash.target.layer_probe.inspection import inspect_compilation
        report = inspect_compilation(a.batch, a.output, a.cases, ddk_root=a.ddk_root)
        result = dict(status=report['status'], report=str(Path(a.output).resolve()/'report.json'),
            model_compilation='NOT_RUN', device_execution='NOT_RUN',
            cases={name: dict(compilation=row['compilation'].get('status'),
                source_error=row.get('source_error'), ir_dumps=len(row['ir_dumps']))
                for name, row in report['cases'].items()})
    elif a.command == 'compare':
        result = batch.compare(a.fixture, a.native, atol=a.atol, rtol=a.rtol)
        root = new_output(a.output)
        save_json(root/'report.json', result)
        import csv
        with (root/'summary.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['layer', 'call', 'phase', 'point', 'output', 'status', 'max_abs', 'relative_l2', 'dtype_preserved', 'stable', 'error'])
            for row in result['cases']:
                metric = row.get('invocations', [{}])[0]
                writer.writerow([row.get(k) for k in ('layer', 'call', 'phase', 'point', 'output_name', 'status')]
                    + [metric.get('max_abs'), metric.get('relative_l2'), row.get('source_dtype_preserved'), row.get('stable'), row.get('error')])
        result = dict(status=result['status'], first_failed_boundary=result['first_failed_boundary'], report=str(root/'report.json'), csv=str(root/'summary.csv'))
    else:
        manifest = json.loads((Path(a.capture)/'manifest.json').read_text())
        result = dict(status=manifest['status'], regions=[{k: r[k] for k in ('id', 'layer', 'call', 'phase', 'point', 'kind')} for r in manifest['regions']])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result.get('status') in ('FAILED', 'PARTIAL', 'INCOMPLETE') else 0


if __name__ == '__main__':
    raise SystemExit(main())
