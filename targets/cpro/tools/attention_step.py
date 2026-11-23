#!/usr/bin/env python3
"""Extract one operation from L0/L3 attention, or replay one measured L0 operator on its native 310P operand."""
import argparse
from functools import partial
import json
import os
from pathlib import Path
import sys
import tempfile

from gdn_tail_accuracy import board, board_arguments
from qwen35_moe_dflash.common.artifacts import output_path, share_for_transfer
from qwen35_moe_dflash.target.layer_probe.common import checked, save_json
from qwen35_moe_dflash.target.prefill_t64 import attention_slice as slices
from qwen35_moe_dflash.target.prefill_t64.attention_catalog import Catalog, write_catalog
from qwen35_moe_dflash.target.prefill_t64 import qkv_silu
from qwen35_moe_dflash.target.prefill_t64 import gdn_op_matrix as ops
from qwen35_moe_dflash.target.prefill_t64.chain_compile import _pull


NATIVE_STEPS = {qkv_silu.STAGE: qkv_silu.STAGE, 'gated_rmsnorm': 'rms_output',
                'gate_silu': 'silu_output', 'out_proj': 'out_proj_only'}


def extract(args, run):
    folder, original, msg, graph = slices.source_graph(args.source_air, args.layer)
    ports = {}
    for i, (edge, dtype, dims) in enumerate(args.input or []):
        if edge in ports:
            raise ValueError(f'Duplicate input {edge}')
        shape = [] if dims == 'scalar' else [int(n) for n in dims.split(',')]
        ports[edge] = (f'step_input_{i}', dtype, shape)
    target_edge = args.output_edge
    if args.step:
        selected = Catalog(graph, original.get('source_inputs', ())).select(args.layer, args.step)
        target_edge = selected['output_edge']
        inputs = selected['runtime_inputs']
        if set(ports) - {s['edge'] for s in inputs}:
            raise ValueError('An input override is outside the selected step. Use --output-edge for custom boundaries.')
        bindings = {}
        for i, s in enumerate(inputs):
            edge = s['edge']
            if edge in ports:
                _, dtype, shape = ports[edge]
            elif edge in selected['incomplete_inputs']:
                raise ValueError(f'{edge}: source lacks a static ABI; supply --input EDGE DTYPE DIMS')
            else:
                dtype, shape = s['dtype'], s['shape']
            bindings[edge] = (f'step_input_{i}', dtype, shape)
        ports = bindings
    if not ports:
        raise ValueError('Supply --input for every runtime operand of the requested slice')
    run.mkdir(exist_ok=True, parents=True)
    target = run/'slice'
    if target.exists():
        raise FileExistsError(target)
    with tempfile.TemporaryDirectory(dir=run) as tmp:
        dest = Path(tmp)/'slice'
        dest.mkdir()
        spec = slices.extract(msg, graph, folder, dest, ports, target_edge)
        if args.step:
            spec['step'] = args.step
        spec.update(source_case=slices.CASE_IDS[args.layer], source_air=original['air'])
        save_json(dest/'slice.json', spec)
        dest.rename(target)
    share_for_transfer(target)
    print(target/'export.air')


def compile_slice(args, run):
    from qwen35_moe_dflash.toolchain.existing_air import compile_existing_air
    source = run/'slice'
    if not source.exists():
        _pull(source, source, args.remote_slice or source)
    receipt = json.loads((source/'slice.json').read_text())
    if args.step and args.step != receipt.get('step'):
        raise ValueError(f"Requested {args.step}, but {source} contains {receipt.get('step')}")
    if args.layer is not None and slices.CASE_IDS[args.layer] != receipt.get('source_case'):
        raise ValueError(f"Requested L{args.layer}, but {source} belongs to {receipt.get('source_case')}")
    air = checked(source, receipt['air'])
    for weight in receipt['weight_evidence'].values():
        checked(source, weight)
    dest = run/'omc'
    report = compile_existing_air(air, dest, args.ddk_root.resolve(), ddk_id=args.ddk_id,
                                  compiler_profile='isolated-kirinx90', use_origin_format=True,
                                  tbe_diagnosis=args.tbe_diagnosis)
    if report['status'] != 'COMPILED':
        raise RuntimeError(f"OMG {report['status']}: {report.get('error', dest/'report.json')}")
    print(checked(dest, report['omc']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    nodes = actions.add_parser('list', help='list original GE nodes, their input/output edges, shapes and dtypes')
    nodes.add_argument('--match', default='', help='case-insensitive substring of node name or type')
    catalog = actions.add_parser('catalog', help='78: export selectable L0/L3 steps and their inputs to CSV')
    catalog.add_argument('--source-air', required=True, type=Path)
    catalog.add_argument('--layer', type=int, choices=slices.CASE_IDS, action='append', help='default: both 0 and 3')
    catalog.add_argument('--presets-only', action='store_true', help='only source-level steps, in forward order')
    catalog.add_argument('--output', type=Path, required=True)
    cut = actions.add_parser('extract', help='78: generate independent AIR from named graph edges')
    for sub in (nodes, cut):
        sub.add_argument('--layer', type=int, choices=slices.CASE_IDS, required=True)
        sub.add_argument('--source-air', required=True, type=Path, help='original four-attention AIR batch')
    cut.add_argument('--input', nargs=3, action='append', metavar=('EDGE', 'DTYPE', 'DIMS'),
                     help='new runtime operand, e.g. MyConv:0 float16 1,8192,64; repeat for additional inputs')
    select = cut.add_mutually_exclusive_group(required=True)
    select.add_argument('--output-edge')
    select.add_argument('--step', help='name or producer:port from the catalog')
    compile_cmd = actions.add_parser('compile', help='35: compile the parameterized slice; pull from 78 if needed')
    compile_cmd.add_argument('--ddk-root', type=Path, required=True)
    compile_cmd.add_argument('--ddk-id', default=os.environ.get('CPRO_DDK_ID') or '100.520.020.010')
    compile_cmd.add_argument('--remote-slice', type=Path)
    compile_cmd.add_argument('--tbe-diagnosis', type=Path, required=True)
    compile_cmd.add_argument('--step', help='check that the pulled slice matches this step')
    compile_cmd.add_argument('--layer', type=int, choices=slices.CASE_IDS)
    prep = actions.add_parser('prepare', aliases=['prepare-qkv'], help='78: prepare one original L0 operator')
    prep.add_argument('--reuse', action='store_true', help='verify and reuse the selected inputs and references')
    prep.add_argument('--source-air', type=Path, required=True, help='original four-attention AIR batch')
    prep.add_argument('--source-root', type=Path, required=True, help='original 310P segmented driver')
    prep.add_argument('--config', type=Path, default=Path('config/qwen3_5_moe.yaml'))
    execute = actions.add_parser('board', help='35: compile and execute the selected L0 operator and print matrices')
    board_arguments(execute)
    execute.add_argument('--runner', choices=('model-tool', 'probe'), default='model-tool',
                         help='board model_run_tool_internal V2 (default) or repeated tensor probe')
    execute.add_argument('--local-air', type=Path, help='reuse an existing single- or multi-step AIR batch on 35')
    execute.add_argument('--cpu-silu', action='store_true', help='run FP32 CPUCL SiLU with the original input bytes')
    report = actions.add_parser('report', help='35: recompute the selected matrices from retained outputs')
    report.add_argument('--runner', choices=('model-tool', 'probe'), help='default: V2 if present, otherwise probe')
    report.add_argument('--step', choices=NATIVE_STEPS, help='default: read the step from the saved AIR manifest')
    for sub in (prep, execute):
        sub.add_argument('--step', choices=NATIVE_STEPS, default=qkv_silu.STAGE)
    for sub in (cut, compile_cmd, prep, execute, report):
        sub.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'catalog':
        write_catalog(args.source_air, tuple(dict.fromkeys(args.layer or slices.CASE_IDS)), output_path(args.output),
                      presets_only=args.presets_only)
        return 0
    if args.action == 'list':
        _, _, _, graph = slices.source_graph(args.source_air, args.layer)
        for row in slices.listing(graph):
            if args.match.lower() in (row['name'] + ' ' + row['type']).lower():
                print(json.dumps(row))
        return 0
    run = output_path(args.run)
    if args.action == 'extract':
        extract(args, run)
    elif args.action == 'compile':
        compile_slice(args, run)
    elif args.action in ('prepare', 'prepare-qkv'):
        source_air, source = args.source_air.resolve(), args.source_root.resolve()
        stage = NATIVE_STEPS[args.step]
        if stage == qkv_silu.STAGE:
            qkv_silu.prepare(run, source_air, source, args.config, reuse=args.reuse)
        elif args.reuse and (run/'air').is_dir():
            ops.reuse_fixture(run/'air', source_air, source, args.config, stage, ops.BASIS)
        else:
            ops.prepare(run, source_air, source, args.config, stages=(stage,))
    else:
        if args.action == 'report' and args.step is None:
            manifest = json.loads((run/'air/manifest.json').read_text())
            points = [case['point'] for case in manifest['cases']]
            if len(points) != 1 or points[0] not in NATIVE_STEPS.values():
                raise ValueError(f'{run}: expected one supported attention step, found {points}')
            stage, = points
        else:
            stage = NATIVE_STEPS[args.step]
        role, basis = ('conv_out', qkv_silu.BASIS) if stage == qkv_silu.STAGE else (ops.ROLES[stage], ops.BASIS)
        scores = partial(ops.summarize, roles={stage: role}, basis=basis, audit_ratios=True)
        if args.action == 'report':
            v2 = args.runner == 'model-tool' if args.runner else (run/'v2/result.json').is_file()
            ok = scores(run, v2=v2)
        else:
            from attention_step_fixture import ensure_fixture
            ensure_fixture(run, args.step, stage, basis, local=args.local_air, remote=args.remote_air)
            ok = board(run, args, stages=(stage,), summarize=scores, basis=basis)
        return 0 if ok else 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
