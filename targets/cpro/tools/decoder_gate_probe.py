"""Export and replay L0 attention or a gate slice from an existing four-layer AIR."""
from math import prod
from pathlib import Path
import errno
import shutil
import subprocess

import numpy as np

from decoder_cases_board import compiled, named_shapes, read
from qwen35_moe_dflash.target.layer_probe.common import identity, save_json, sha, sized_file
from qwen35_moe_dflash.target.prefill_t64.attention_step_v2 import run_v2
from qwen35_moe_dflash.target.prefill_t64.attention_steps import AttentionSteps, MATMUL, NORMS
from qwen35_moe_dflash.target.prefill_t64.decode_cache_control import graph_message
from qwen35_moe_dflash.target.prefill_t64.decoder_four_boundary import data_edges
from qwen35_moe_dflash.target.prefill_t64.decoder_four_silu import activation_port_shape, gate_product
from qwen35_moe_dflash.target.prefill_t64.l0_split import slice_air, slice_dependencies, wire
from qwen35_moe_dflash.target.prefill_t64.whole_run import hiai_failure_stage


SHAPE = (2048, 128)
KIND = 'four-layer-rms-gate-probes'


def input_wire(node, fallback_shape):
    desc = node.input_desc[0]
    dtype = {1: 'float32', 2: 'float16'}.get(desc.dtype)
    shapes = activation_port_shape(desc)
    static = [s for s in (shapes['ge_dims'], shapes['fx_dims']) if s and min(s) > 0]
    shape = static[0] if static else fallback_shape
    if (not dtype or any(s != shape for s in static)
            or shape[-1] != fallback_shape[-1] or prod(shape) != prod(fallback_shape)):
        raise ValueError(f'{node.name}: unexpected input descriptor {desc.dtype} {shapes}')
    return wire(node.input[0], shape, dtype)


def probe_cases(raw, meta, mode):
    if mode in ('gdn-export', 'gdn-whole'):
        from qwen35_moe_dflash.target.prefill_t64.gdn_layer_air import four_layer_gdn
        step = four_layer_gdn(raw, meta)
        yield 'linear-attention', step, step['wires']
        return
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
    cases = [
        ('cast-mul', [gate_out, peer], [cast, mul.name], mul_out),
        ('silu', [gate_input], [gate.name], gate_out),
        ('silu-cast', [gate_input], [gate.name, cast], cast_out),
        ('silu-cast-mul', [gate_input, peer], [gate.name, cast, mul.name], mul_out),
    ]
    by_name = {op.name: op for op in graph.op}
    if mode == 'upstream':
        walk = AttentionSteps(graph, by_name, [])
        z = walk.expect(walk.strip(gate_input), {MATMUL})
        quant = walk.expect(walk.strip(z.input[0]), {'DynamicQuant'})
        rms = walk.expect(peer, NORMS)
        hidden, core = quant.input[0], rms.input[0]
        wires[hidden] = input_wire(quant, (64, 2048))
        wires[core] = input_wire(rms, SHAPE)
        cases = []
        for label, inputs, producers in (
                ('z-branch', [hidden, peer], {z.name, quant.name}),
                ('rms-branch', [gate_input, core], {rms.name}),
                ('both-branches', [hidden, core], {z.name, quant.name, rms.name})):
            kept, used = slice_dependencies(by_name, [mul_out], set(inputs), label)
            if used != set(inputs) or (kept & {z.name, quant.name, rms.name}) != producers:
                raise ValueError(f'{label}: slice crosses unexpected projection or RMSNorm boundaries')
            cases.append((label, inputs, [name for name in by_name if name in kept], mul_out))
    for label, inputs, nodes, output in cases:
        step = dict(inputs=inputs, nodes=nodes, outputs=[output])
        yield label, step, wires


def compile_cases(folder, case, meta, record, ddk, mode):
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
    inventory = {r['path']: r for r in meta['files']}
    _, original = graph_message(raw, 'SiluPatch')
    external = {op.name: op.attr['file_path'].s.decode() for op in original.op if op.type == 'FileConstant'}
    for label, step, wires in probe_cases(raw, meta, mode):
        base = folder/label
        directory = 'weight' if mode == 'slices' else meta['air_directory']
        air = base/directory/'export.air'
        air.parent.mkdir(parents=True)
        air.write_bytes(slice_air(raw, step, wires))
        source = identity(air, base)
        files = [source]
        nodes = step['nodes']
        weights = {external[n] for n in nodes if n in external}
        for filename in sorted(weights):
            rel = Path(filename)
            if rel.is_absolute() or '..' in rel.parts or not rel.parts:
                raise ValueError(f'{label}: unexpected FileConstant path {filename!r}')
            if rel.parts[0] != directory:
                rel = Path(directory)/rel
            item = inventory[rel.as_posix()]
            target = base/rel
            target.parent.mkdir(exist_ok=True, parents=True)
            weight = sized_file(case, item)
            if mode in ('gdn-export', 'gdn-whole'):
                try:
                    target.hardlink_to(weight)
                except OSError as exc:
                    if exc.errno not in (errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP):
                        raise
                    shutil.copyfile(weight, target)
            else:
                target.symlink_to(weight)
            files.append(item)
        feeds = {f'l0_feed_{i}': wires[e] for i, e in enumerate(step['inputs'])}
        outputs = [wires[e] for e in step['outputs']]
        save_json(base/'case.json', dict(native_tool_args=True, files=files, air_directory=directory,
                                         inputs=feeds, outputs=outputs, source_air=record))
        print(f'[{label}] 已导出 AIR：{air}（{len(nodes)} 个节点）', flush=True)
        row = dict(nodes=nodes, source=source['sha256'], feeds=feeds, outputs=outputs)
        if mode == 'gdn-export':
            report['cases'][label] = dict(row, status='EXPORTED')
            continue
        print(f'[{label}] 编译 OMC', flush=True)
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


def execute_case(base, row, runtime, config, *, replace_inputs, replace_weights, valid_rows=None):
    build, model, weights = compiled(base, base, None, None, row['source'])
    replay = base/'replay'
    if replay.exists():
        shutil.rmtree(replay)
    replay.mkdir()
    operands, paths = [], []
    feeds = row.get('feeds')  # Older isolated gate records have fixed FP16/FP32 ports.
    if feeds is not None and set(feeds) != {s['name'] for s in build['inputs']}:
        raise ValueError(f'{base.name}: compiled input names differ from the AIR cut')
    for spec in build['inputs']:
        name = spec['name']
        if feeds is None:
            dtype, shape = {'l0_feed_0': 'float16', 'l0_feed_1': 'float32'}[name], SHAPE
        else:
            dtype, shape = feeds[name]['dtype'], feeds[name]['shape']
        if spec['dtype'] != dtype or [d for d in spec['shape'] if d != 1] != [d for d in shape if d != 1]:
            raise ValueError(f'{base.name}: unexpected gate input {spec}')
        path = replay/(name + '.bin')
        role = feeds[name]['name'] if feeds else None
        if role in ('history', 'state'):
            data = np.zeros(shape, dtype=dtype)
        elif role == 'length':
            data = np.full(shape, valid_rows, dtype=dtype)
        else:
            low, high = (-2, 2) if name == 'l0_feed_0' else (0.5, 1.5)
            data = np.linspace(low, high, num=prod(shape), dtype=dtype)
        data.tofile(path)
        operands.append(dict(spec, sha256=sha(path)))
        paths.append(path)
    try:
        run_v2(replay, base.name, model, operands, paths, build['outputs'], runtime,
               board_case='4-prefill-t64-gate-' + base.name, decoder_shapes=(named_shapes(operands), build['output_shape']),
               shape_config=config, original_board_args=True, replace_inputs=replace_inputs, external_weights=weights,
               replace_weights=replace_weights,
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


def probe(root, action, ddk, config, *, mode='slices', replace_inputs=False, replace_weights=False):
    case = root/'board/package/L00-L03'
    meta = read(case/'case.json')
    record, = [r for r in meta['files'] if r['path'] == meta['air_directory'] + '/export.air']
    whole = mode in ('gdn-export', 'gdn-whole')
    directory = 'gdn' if whole else ('gate-upstream' if mode == 'upstream' else 'gate-probes')
    folder = root/'board/L00-L03'/directory
    if action == 'compile':
        if not (case/record['path']).is_file():
            case = root/'export/package/L00-L03'
            if record not in read(case/'case.json')['files']:
                raise ValueError('The local AIR export differs from the model on the board host')
        compile_cases(folder, case, meta, record, ddk, mode)
        return True
    report = read(folder/'compile.json')
    if report['kind'] != KIND or not report['complete'] or report['air'] != record:
        raise ValueError('Compile gate probes for the current four-layer AIR before executing')
    runtime = read(config.parent/read(config)['assets'])['environment']
    valid_rows = read(case.parent/'fixture/reference.json')['valid_rows'] if whole else None
    if whole and not 1 <= valid_rows <= 64:
        raise ValueError(f'Unexpected T64 prefill length: {valid_rows}')
    summary = {}
    save_json(folder/'result.json', summary)
    for label, row in report['cases'].items():
        summary[label] = row
        if row['status'] == 'COMPILED':
            summary[label] = execute_case(folder/label, row, runtime, config, replace_inputs=replace_inputs,
                                          replace_weights=replace_weights, valid_rows=valid_rows)
        save_json(folder/'result.json', summary)
    print(f'上板结果 {folder}/result.json', flush=True)
    for label, row in summary.items():
        matched = '  OpUpdateKernelArgs failed' if row.get('kernel_args_error') else ''
        print(f'{label:16} {row["status"]}{matched}', flush=True)
    return all(row['status'] == 'EXECUTED' for row in summary.values())
