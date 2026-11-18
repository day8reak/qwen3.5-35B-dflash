#!/usr/bin/env python3
"""Export one CPro region at a time, compile batches, and compare board dumps."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'model'))


def check_directory(value):
    if not value.strip():
        raise argparse.ArgumentTypeError('empty path: pass the report path printed by check-prefill-matmul or its directory')
    path = Path(value).expanduser()
    if path.name == 'report.json' and path.is_file():
        return path.parent
    if (path / 'report.json').is_file():
        return path
    raise argparse.ArgumentTypeError(f'no check-prefill-matmul report at {path}; pass report.json or its directory')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for command in ('export', 'export-one'):
        s = sub.add_parser(command)
        s.add_argument('--capture', required=True)
        s.add_argument('--output', required=True)
        s.add_argument('--source-root', required=True)
        s.add_argument('--converter-script', required=True)
        s.add_argument('--target-config', help='original Target YAML (model_file/quanted_pth); required for weighted regions')
        s.add_argument('--torchair-pb2')
        s.add_argument('--gdn-state-dtype', choices=('source', 'float16', 'float32'), default='source',
                       help='whole gdn_layer: explicit recurrent input/output/writeback dtype; float16 also casts g at the GE operator input; golden unchanged')
        s.add_argument('--gdr-mode', choices=('chunk', 'single-token'), default='chunk',
                       help='whole Prefill GDN: one original GDR call, or one state-chained S=1 call per token')
        s.add_argument('--gdn-convstate', choices=('kernel', 'reference'), default='kernel',
                       help='whole gdn_layer only: reference uses explicit concat/tail after exact capture checks; full-length calls, one-token Decode; golden unchanged')
        s.add_argument('--qlinear-scales', choices=('source', 'fp16'), default='source',
                       help='opt-in candidate: numeric FP16 weight scale and per-token scale, original golden')
        s.add_argument('--rope-precision', choices=('source', 'fp32'), default='source',
                       help='opt-in FP32 rotation arithmetic; captured cos/sin and golden unchanged, output dtype preserved')
        s.add_argument('--full-attention-reference', choices=('math-replay', 'captured-audited'), default='math-replay',
                       help='whole FullAttention reference: separate math-core replay (default), or same-call original outputs after rechecking the captured core math audit')
        s.add_argument('--cache-indexing', choices=('kernel', 'float', 'integer', 'both') if command == 'export' else ('kernel', 'float', 'integer'),
                       default='kernel', help='expand cache boundary to position/table indexing; both exports float control and integer candidate')
        if command == 'export':
            s.add_argument('--points', nargs='+')
            s.add_argument('--kinds', nargs='+')
            s.add_argument('--phases', nargs='+', choices=('prefill', 'decode'),
                           help='export only these captured phases; omitted keeps all phases')
            s.add_argument('--timeout', type=int, default=900)
        else:
            s.add_argument('--region', required=True)
            s.add_argument('--output-index', type=int, help='required except for gdn_layer/full_attention_layer, which return all three outputs')
    s = sub.add_parser('compile')
    s.add_argument('--batch', required=True, help='existing AIR batch (read-only)')
    s.add_argument('--output', required=True, help='fresh OMC batch below AI_RUN_DIR, outside the AIR batch and DDK')
    s.add_argument('--ddk-root', default=os.environ.get('CPRO_DDK_ROOT'), required=not bool(os.environ.get('CPRO_DDK_ROOT')))
    s.add_argument('--ddk-id', default=os.environ.get('CPRO_DDK_ID'), required=not bool(os.environ.get('CPRO_DDK_ID')))
    s.add_argument('--tbe-diagnosis', help='reuse the completed selected-DDK TBE diagnosis for run-local worker compatibility; no new diagnosis or target change')
    s.add_argument('--cases', nargs='+', help='compile only these retained AIR cases; other cases remain NOT_SELECTED')
    from qwen35_moe_dflash.toolchain.compiler_entry import COMPILER_PROFILES, DEFAULT_PROFILE
    s.add_argument('--compiler-profile', choices=COMPILER_PROFILES, default=DEFAULT_PROFILE,
                   help='sourced-wrapper-default compares the inspected entry using the current sourced shell; diagnostic only')
    s.add_argument('--timeout', type=int, default=900)
    s.add_argument('--use-origin-format', action='store_const', const=True, default=None)
    s = sub.add_parser('inspect-compile', help='read retained AIR/IR metadata without compiling or running models')
    s.add_argument('--batch', required=True, help='existing OMC batch')
    s.add_argument('--output', required=True)
    selection = s.add_mutually_exclusive_group(required=True)
    selection.add_argument('--cases', nargs='+')
    selection.add_argument('--failed', action='store_true', help='inspect every FAILED case in the retained compile.json; never rerun OMG')
    s.add_argument('--ddk-root', default=os.environ.get('CPRO_DDK_ROOT'))
    s.add_argument('--compiler-entry', action='store_true', help='also read OMG/env entry scripts and compare current shell with probe environment preparation; executes neither script nor compiler')
    s.add_argument('--state-chain', action='store_true', help='also decode compiled OMC and trace whole-GDN output 2 through visible Cast/shape nodes; writes state-summary.txt without changing the requested ABI')
    s.add_argument('--reference-ir', nargs='+', help='up to 8 existing IR files; strictly compare selected DDK and retained AIR producer schemas, without compiling')
    s = sub.add_parser('inspect-qlinear', help='read independent/whole-module QLinear boundaries, constant pairs and kernel build arguments; no compilation')
    s.add_argument('--batch', required=True, help='existing OMC batch; includes QLinear, whole GDN and whole FullAttention cases')
    s.add_argument('--output', required=True, help='fresh inspection directory, separate from AIR/OMC')
    s.add_argument('--ddk-root', default=os.environ.get('CPRO_DDK_ROOT'), required=not bool(os.environ.get('CPRO_DDK_ROOT')))
    s.add_argument('--fixture', help='optional prepared execution fixture directory containing manifest.json and per-case inputs')
    s = sub.add_parser('inspect-attention', help='read attention extents, source lengths and per-head/block/row errors; no compile or device run')
    s.add_argument('--reference', action='store_true', help='also compute independent NumPy FP64 attention from the retained source-bound inputs; leaves original golden unchanged')
    s.add_argument('--batch', required=True, help='OMC batch that produced the attention results')
    s.add_argument('--fixture', required=True, help='matching execution/fixture directory')
    s.add_argument('--native', required=True, help='matching recovered native directory')
    s.add_argument('--capture', help='optional original capture for hash-bound argument names and sequence metadata')
    s.add_argument('--output', required=True, help='fresh inspection directory separate from retained inputs')
    s = sub.add_parser('inspect-gdn-output', help='read whole-GDN dumps and compare state outputs with old state inputs; no export or device run')
    s.add_argument('--fixture', required=True)
    s.add_argument('--native', required=True)
    s.add_argument('--output', required=True)
    s = sub.add_parser('inspect-prefill-links', help='check captured QKV/ConvState and V projection/cache links offline; no export, compilation or device execution')
    s.add_argument('--capture', required=True, help='original fresh-Prefill boundary capture directory')
    s.add_argument('--output', required=True, help='fresh read-only inspection report directory')
    for command in ('verify-qlinear-golden', 'replay-dynamic-quant'):
        s = sub.add_parser(command, help='independent CPU QKV/V reference' if command == 'verify-qlinear-golden'
                           else 'only original DynamicQuant on frozen inputs; no QuantMatmul or whole-model replay')
        s.add_argument('--capture', required=True)
        s.add_argument('--output', required=True)
        s.add_argument('--points', nargs='+', choices=('linear_attn_in_proj_qkv', 'self_attn_v_proj'),
                       help='omitted selects L0-L2 QKV and L3 V')
        if command == 'verify-qlinear-golden':
            s.add_argument('--source-root', required=True, help='original NZ/310P source matching capture; also resolves relative YAML resource paths')
            s.add_argument('--target-config', required=True, help='original YAML with quanted_pth')
            s.add_argument('--quant-replay', help='optional replay-dynamic-quant output directory')
        else:
            s.add_argument('--device', default='npu:0')
    s = sub.add_parser('probe-310p-matmul', help='diagnose one original QLinear call using frozen DynamicQuant results')
    s.add_argument('--capture', required=True)
    s.add_argument('--quant-replay', required=True, help='completed replay-dynamic-quant directory')
    s.add_argument('--cpu', required=True, help='verify-qlinear-golden output obtained with this quant replay')
    s.add_argument('--layer', type=int, choices=range(4), default=3, help='L3 V by default, or L0-L2 QKV')
    repeat = s.add_mutually_exclusive_group()
    repeat.add_argument('--repeat-original', action='store_true',
                        help='repeat DynamicQuant on captured hidden and original QuantMatmul on frozen operands three times each')
    repeat.add_argument('--compare-group256', action='store_true',
                        help='compare the original and explicit group256 signatures three times each on identical frozen operands')
    s.add_argument('--dequant-control', action='store_true', help='also compare ordinary device matmul on dequantized checkpoint W4 to the frozen-A8 CPU reference')
    s.add_argument('--device', default='npu:0')
    s.add_argument('--output', required=True)
    s = sub.add_parser('inspect-live-layout', help='compare two W4 layouts on the saved L3 V call using CPU arithmetic')
    s.add_argument('--trace', required=True, type=Path, help='completed trace_target_prefill_matmul.py directory')
    s.add_argument('--capture', required=True, type=Path)
    s.add_argument('--cpu', required=True, type=Path, help='matching verify-qlinear-golden directory')
    s = sub.add_parser('inspect-prefill-drift', help='find the first changed recorded boundary in the existing historical and repeat Prefills')
    s.add_argument('--fixture', required=True, help='historical Prefill fixture with capture/')
    s.add_argument('--check', required=True, type=check_directory, help='check-prefill-matmul report.json or its directory')
    s = sub.add_parser('replay-saved-matmul', help='run three consecutive L3 V matmuls on the restored operands from first_call.pt')
    s.add_argument('--check', required=True, type=check_directory, help='check-prefill-matmul report.json or its directory')
    s.add_argument('--device', default='npu:0')
    s = sub.add_parser('check-prefill-matmul', help='recapture L3 V in original four-layer Prefill and repeat its matmul on the same device tensors')
    s.add_argument('--fixture', required=True, help='historical generate_target_prefill.py directory containing input_ids.bin and capture/')
    s.add_argument('--trace', required=True, help='successful trace_target_prefill_matmul.py prefix trace directory')
    s.add_argument('--source-root', required=True, help='original NZ/310P source directory')
    s.add_argument('--target-config', required=True, help='original Prefill YAML')
    s.add_argument('--device', default='npu:0')
    s.add_argument('--output', required=True, help='new diagnostic directory separate from fixture and trace')
    s = sub.add_parser('compare-projection-values', help='compare retained Prefill state values directly to independent CPU projections')
    s.add_argument('--cpu', required=True, help='completed verify-qlinear-golden directory')
    s.add_argument('--group', required=True, help='linear or full-fp32-pos directory containing omc, execution, recovery and comparison')
    s = sub.add_parser('compare')
    s.add_argument('--fixture', required=True)
    s.add_argument('--native', required=True)
    s.add_argument('--output', required=True)
    s.add_argument('--cases', nargs='+', help='compare only these retained fixture cases; full-fixture coverage remains explicit')
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
            points=a.points, kinds=a.kinds, phases=a.phases, timeout=a.timeout, torchair_pb2=a.torchair_pb2, target_config=a.target_config,
            qlinear_scales=a.qlinear_scales, cache_indexing=a.cache_indexing, rope_precision=a.rope_precision,
            gdn_state_dtype=a.gdn_state_dtype, gdn_convstate=a.gdn_convstate, gdr_mode=a.gdr_mode,
            full_attention_reference=a.full_attention_reference)
    elif a.command == 'export-one':
        from qwen35_moe_dflash.target.layer_probe.replay import export_one
        result = export_one(a.capture, a.region, a.output_index, a.output, a.source_root, a.converter_script, a.torchair_pb2, a.target_config,
                            qlinear_scales=a.qlinear_scales, cache_indexing=a.cache_indexing, rope_precision=a.rope_precision,
                            gdn_state_dtype=a.gdn_state_dtype, gdn_convstate=a.gdn_convstate, gdr_mode=a.gdr_mode,
                            full_attention_reference=a.full_attention_reference)
    elif a.command == 'compile':
        result = batch.compile_batch(a.batch, a.output, a.ddk_root, a.ddk_id, tbe_diagnosis=a.tbe_diagnosis,
                                     timeout=a.timeout, use_origin_format=a.use_origin_format,
                                     cases=a.cases, compiler_profile=a.compiler_profile)
    elif a.command == 'probe-310p-matmul':
        from qwen35_moe_dflash.target.layer_probe.qlinear_matmul_probe import run_probe
        result = run_probe(a.capture, a.quant_replay, a.cpu, a.output, a.layer, a.device,
                           repeat_original=a.repeat_original, dequant_control=a.dequant_control,
                           compare_group256=a.compare_group256)
    elif a.command == 'inspect-live-layout':
        from qwen35_moe_dflash.target.layer_probe.live_projection import compare_live
        evidence = compare_live(a.trace, a.capture, a.cpu, inspect_layouts=True)
        layout = evidence.pop('layout_report', None)
        result = dict(status='INSPECTED' if layout else 'PARTIAL',
                      case='l03_c00_self_attn_v_proj', **(layout or {}))
        for key, value in evidence.items():
            if not isinstance(value, dict):
                result[key] = value
        if not layout:
            result['error'] = evidence.get('same_call_cpu_skipped') or 'need matching captured hidden and module/matmul output'
    elif a.command == 'inspect-prefill-drift':
        from qwen35_moe_dflash.target.layer_probe.prefill_drift import inspect_drift
        result = inspect_drift(a.fixture, a.check)
    elif a.command == 'replay-saved-matmul':
        from qwen35_moe_dflash.target.layer_probe.prefill_matmul_repeat import replay_saved_matmul
        result = replay_saved_matmul(a.check, a.device)
    elif a.command == 'check-prefill-matmul':
        from qwen35_moe_dflash.target.layer_probe.prefill_matmul_repeat import check_prefill_matmul
        result = check_prefill_matmul(a.fixture, a.trace, a.source_root, a.target_config, a.output, a.device)
    elif a.command == 'compare-projection-values':
        import csv
        from qwen35_moe_dflash.target.layer_probe.projection_values import compare_values
        rows = compare_values(a.cpu, a.group)
        writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        return 0
    elif a.command == 'inspect-prefill-links':
        from qwen35_moe_dflash.target.layer_probe.prefill_links import inspect_prefill_links
        result = inspect_prefill_links(a.capture, a.output)
        result = dict(status=result['status'], report=str(Path(a.output).resolve()/'report.json'),
            summary=str(Path(a.output).resolve()/'summary.txt'),
            model_compilation='NOT_RUN', device_execution='NOT_RUN')
    elif a.command == 'verify-qlinear-golden':
        from qwen35_moe_dflash.target.layer_probe.qlinear_golden import verify
        result = verify(a.capture, a.target_config, a.source_root, a.output,
                        points=a.points, quant_replay=a.quant_replay)
    elif a.command == 'replay-dynamic-quant':
        from qwen35_moe_dflash.target.layer_probe.qlinear_quant_replay import replay
        result = replay(a.capture, a.output, a.device, points=a.points)
        result = dict(status=result['status'], manifest=str(Path(a.output).resolve()/'manifest.json'),
                      summary=str(Path(a.output).resolve()/'summary.txt'),
                      cases={c['id']: c['status'] for c in result['cases']}, errors=result['errors'])
    elif a.command == 'inspect-qlinear':
        from qwen35_moe_dflash.target.layer_probe.contracts import inspect_qlinear
        result = inspect_qlinear(a.batch, a.output, a.ddk_root, a.fixture)
    elif a.command == 'inspect-attention':
        from qwen35_moe_dflash.target.layer_probe.attention_inspect import inspect_attention
        result = inspect_attention(a.batch, a.fixture, a.native, a.output, a.capture, a.reference)
    elif a.command == 'inspect-gdn-output':
        from qwen35_moe_dflash.target.layer_probe.gdn_output_inspect import inspect_gdn_outputs
        result = inspect_gdn_outputs(a.fixture, a.native, a.output)
    elif a.command == 'inspect-compile':
        from qwen35_moe_dflash.target.layer_probe.inspection import inspect_compilation
        report = inspect_compilation(a.batch, a.output, a.cases, ddk_root=a.ddk_root,
                                     compiler_entry=a.compiler_entry, reference_ir=a.reference_ir,
                                     state_chain=a.state_chain)
        result = dict(status=report['status'], report=str(Path(a.output).resolve()/'report.json'),
            summary=str(Path(a.output).resolve()/'summary.txt'),
            model_compilation='NOT_RUN', device_execution='NOT_RUN',
            cases={name: dict(compilation=row['compilation'].get('status'),
                source_error=row.get('source_error'), ir_dumps=len(row['ir_dumps']),
                ir_inventory=row['ir_inventory']['status'])
                for name, row in report['cases'].items()})
        if a.state_chain:
            result['state_summary'] = str(Path(a.output).resolve()/'state-summary.txt')
    elif a.command == 'compare':
        result = batch.compare(a.fixture, a.native, selection=a.cases, atol=a.atol, rtol=a.rtol)
        root = new_output(a.output)
        save_json(root/'report.json', result)
        import csv
        with (root/'summary.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            metric_keys = ['max_abs', 'relative_l2', 'cosine', 'kl_divergence', 'rmse', 'mean_relative_error']
            writer.writerow(['layer', 'call', 'phase', 'point', 'output', 'status'] + metric_keys
                            + ['failed_metrics', 'dtype_preserved', 'stable', 'error', 'actual_dtype', 'golden_dtype',
                               'reference_basis', 'captured_relative_l2', 'captured_cosine',
                               'active_cache_relative_l2', 'untouched_cache_equal_bits'])
            for row in result['cases']:
                metric = row.get('invocations', [{}])[0]
                writer.writerow([row.get(k) for k in ('layer', 'call', 'phase', 'point', 'output_name', 'status')]
                    + [metric.get(k) for k in metric_keys]
                    + [';'.join(sorted({k for m in row.get('invocations', []) for k in m.get('failed_metrics', [])})),
                       row.get('source_dtype_preserved'), row.get('stable'), row.get('error'),
                       row.get('actual_dtype'), row.get('golden_dtype'), row.get('reference_basis'),
                       metric.get('original_capture_comparison', {}).get('relative_l2'),
                       metric.get('original_capture_comparison', {}).get('cosine'),
                       metric.get('full_attention_cache', {}).get('active_metrics', {}).get('relative_l2'),
                       metric.get('full_attention_cache', {}).get('untouched_equal_input_bits')])
        result = dict(status=result['status'], first_failed_boundary=result['first_failed_boundary'], report=str(root/'report.json'), csv=str(root/'summary.csv'))
    else:
        manifest = json.loads((Path(a.capture)/'manifest.json').read_text())
        result = dict(status=manifest['status'], regions=[{k: r[k] for k in ('id', 'layer', 'call', 'phase', 'point', 'kind')} for r in manifest['regions']])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result.get('status') in ('FAILED', 'PARTIAL', 'INCOMPLETE') else 0


if __name__ == '__main__':
    raise SystemExit(main())
