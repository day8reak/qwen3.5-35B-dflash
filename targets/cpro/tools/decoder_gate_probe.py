"""Find which prefix of a retained four-layer RMS gate loads through the native V2 runner."""
import shutil
import subprocess

import numpy as np

from decoder_cases_board import compiled, named_shapes, read
from qwen35_moe_dflash.target.layer_probe.common import identity, save_json, sha, sized_file
from qwen35_moe_dflash.target.prefill_t64.attention_step_v2 import run_v2
from qwen35_moe_dflash.target.prefill_t64.decode_cache_control import graph_message
from qwen35_moe_dflash.target.prefill_t64.decoder_four_boundary import data_edges
from qwen35_moe_dflash.target.prefill_t64.decoder_four_silu import gate_product
from qwen35_moe_dflash.target.prefill_t64.l0_split import slice_air, wire
from qwen35_moe_dflash.target.prefill_t64.whole_run import hiai_failure_stage


SHAPE = (2048, 128)
KIND = 'four-layer-rms-gate-probes'


def gate_cases(raw, meta):
    _, graph = graph_message(raw, 'SiluPatch')
    gates = {row['name'] for row in meta.get('silu_patch', {}).get('nodes', []) if row.get('shape') == list(SHAPE)}
    gate = next((op for op in graph.op if op.name in gates and op.type == 'SiluPatch'), None)
    if gate is None:
        raise ValueError('The selected four-layer AIR has no [2048,128] RMS gate SiluPatch')
    if ([d.dtype for d in gate.input_desc], [d.dtype for d in gate.output_desc]) != ([2], [2]):
        raise ValueError(f'{gate.name}: expected FP16 SiluPatch ports')
    gate_input, = data_edges(gate)
    gate_out = gate.name + ':0'
    mul, peer = gate_product(graph, gate_out)
    cast_out, = [edge for edge in data_edges(mul) if edge != peer]
    cast = cast_out.rsplit(':', 1)[0]
    mul_out = mul.name + ':0'
    wires = {edge: wire(edge, SHAPE, dtype) for edge, dtype in (
        (gate_input, 'float16'), (gate_out, 'float16'), (cast_out, 'float32'),
        (peer, 'float32'), (mul_out, 'float32'))}
    for label, inputs, nodes, output in (
        ('cast-mul', [gate_out, peer], [cast, mul.name], mul_out),
        ('silu', [gate_input], [gate.name], gate_out),
        ('silu-cast', [gate_input], [gate.name, cast], cast_out),
        ('silu-cast-mul', [gate_input, peer], [gate.name, cast, mul.name], mul_out),
    ):
        step = dict(inputs=inputs, nodes=nodes, outputs=[output])
        yield label, nodes, slice_air(raw, step, wires)


def compile_cases(folder, case, meta, record, ddk):
    raw = sized_file(case, record).read_bytes()
    index = folder/'compile.json'
    if folder.is_symlink():
        raise ValueError(f'Unexpected diagnostic directory link: {folder}')
    if folder.exists():
        if not index.is_file() or read(index).get('kind') != KIND:
            raise ValueError(f'Existing directory is not a gate probe: {folder}')
        shutil.rmtree(folder)
    folder.mkdir()
    report = dict(kind=KIND, air=record, complete=False, cases={})
    save_json(index, report)
    for label, nodes, graph in gate_cases(raw, meta):
        base = folder/label
        air = base/'weight/export.air'
        air.parent.mkdir(parents=True)
        air.write_bytes(graph)
        source = identity(air, base)
        save_json(base/'case.json', dict(native_tool_args=True, files=[source]))
        print(f'[{label}] 编译 {nodes}', flush=True)
        row = dict(nodes=nodes, source=source['sha256'])
        try:
            compiled(base, base, ddk, None, row['source'])
            row['status'] = 'COMPILED'
        except (RuntimeError, ValueError) as err:
            row.update(status='BUILD_FAILED', error=str(err))
            print(f'[{label}] {err}', flush=True)
        report['cases'][label] = row
        save_json(index, report)
    report['complete'] = True
    save_json(index, report)


def execute_case(base, source, runtime, config, *, replace_inputs):
    build, model, weights = compiled(base, base, None, None, source)
    replay = base/'replay'
    if replay.exists():
        shutil.rmtree(replay)
    replay.mkdir()
    operands, paths = [], []
    for spec in build['inputs']:
        expected = {'l0_feed_0': 'float16', 'l0_feed_1': 'float32'}[spec['name']]
        if spec['dtype'] != expected or tuple(d for d in spec['shape'] if d != 1) != SHAPE:
            raise ValueError(f'{base.name}: unexpected gate input {spec}')
        low, high = (-2, 2) if expected == 'float16' else (0.5, 1.5)
        path = replay/(spec['name'] + '.bin')
        np.linspace(low, high, num=SHAPE[0]*SHAPE[1], dtype=expected).tofile(path)
        operands.append(dict(spec, sha256=sha(path)))
        paths.append(path)
    try:
        run_v2(replay, base.name, model, operands, paths, build['outputs'], runtime,
               board_case='4-prefill-t64-gate-' + base.name, decoder_shapes=(named_shapes(operands), build['output_shape']),
               shape_config=config, original_board_args=True, replace_inputs=replace_inputs, external_weights=weights,
               build_records=(build['model'], build['weights']))
    except RuntimeError as err:
        receipt_path = replay/'result.json'
        if not receipt_path.is_file():
            raise
        receipt = read(receipt_path)
        if not receipt.get('exitcode'):
            raise
        stage = hiai_failure_stage(str(err))
        result = dict(status=stage.upper() + '_FAILED', exitcode=receipt['exitcode'], error=str(err))
        from split_stage_log import diagnose
        try:
            diagnose([replay])
            for path in base.glob('load-diagnostics-*/result.json'):
                for row in read(path):
                    if row.get('attempt') == receipt['attempt']:
                        result['diagnosis'] = str(path)
                        result['kernel_args_error'] = any(
                            f['op_update_kernel_args_seen'] for f in row.get('service_failures', []))
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as diag_err:
            print(f'[{base.name}] HiLog: {diag_err}', flush=True)
        return result
    return dict(status='EXECUTED', exitcode=0)


def probe(root, action, ddk, config, *, replace_inputs=False):
    case = root/'board/package/L00-L03'
    meta = read(case/'case.json')
    record, = [r for r in meta['files'] if r['path'] == meta['air_directory'] + '/export.air']
    folder = root/'board/L00-L03/gate-probes'
    if action == 'compile':
        if not (case/record['path']).is_file():
            case = root/'export/package/L00-L03'
            if record not in read(case/'case.json')['files']:
                raise ValueError('The local AIR export differs from the model on the board host')
        compile_cases(folder, case, meta, record, ddk)
        return True
    report = read(folder/'compile.json')
    if report['kind'] != KIND or not report['complete'] or report['air'] != record:
        raise ValueError('Compile gate probes for the current four-layer AIR before executing')
    runtime = read(config.parent/read(config)['assets'])['environment']
    summary = {}
    save_json(folder/'result.json', summary)
    for label, row in report['cases'].items():
        summary[label] = row
        if row['status'] == 'COMPILED':
            summary[label] = execute_case(folder/label, row['source'], runtime, config, replace_inputs=replace_inputs)
        save_json(folder/'result.json', summary)
    print(f'门控上板结果 {folder}/result.json', flush=True)
    for label, row in summary.items():
        matched = '  OpUpdateKernelArgs failed' if row.get('kernel_args_error') else ''
        print(f'{label:16} {row["status"]}{matched}', flush=True)
    return all(row['status'] == 'EXECUTED' for row in summary.values())
