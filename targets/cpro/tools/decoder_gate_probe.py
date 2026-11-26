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
from qwen35_moe_dflash.target.prefill_t64.decoder_four_silu import activation_port_shape, rms_gate_path
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
    if mode == 'l0-steps':
        from qwen35_moe_dflash.target.prefill_t64.l0_accuracy_air import readouts
        yield from readouts(raw, meta)
        return
    if mode == 'gdn-bisect':
        from qwen35_moe_dflash.target.prefill_t64.gdn_layer_air import gdn_load_sections
        yield from gdn_load_sections(raw, meta)
        return
    if mode in ('gdn-export', 'gdn-whole'):
        from qwen35_moe_dflash.target.prefill_t64.gdn_layer_air import four_layer_gdn
        step = four_layer_gdn(raw, meta)
        yield 'linear-attention', step, step['wires']
        return
    _, graph = graph_message(raw, 'SiluPatch')
    kernel_shape = (64, 4096) if meta.get('rms_silu_token_view') else SHAPE
    gates = {row['name'] for row in meta.get('silu_patch', {}).get('nodes', []) if row.get('shape') == list(kernel_shape)}
    gate = next((op for op in graph.op if op.name in gates and op.type == 'SiluPatch'), None)
    if gate is None:
        raise ValueError(f'The selected four-layer AIR has no {kernel_shape} RMS gate SiluPatch')
    if ([d.dtype for d in gate.input_desc], [d.dtype for d in gate.output_desc]) != ([2], [2]):
        raise ValueError(f'{gate.name}: expected FP16 SiluPatch ports')
    gate_input, = data_edges(gate)
    gate_out = gate.name + ':0'
    chain, peer = rms_gate_path(graph, gate_out)
    mul = chain[-1]
    gate_operand, = [edge for edge in data_edges(mul) if edge != peer]
    mul_out = mul.name + ':0'
    wires = {edge: wire(edge, kernel_shape, 'float16') for edge in (gate_input, gate_out)}
    wires.update({e: wire(e, SHAPE, 'float32') for e in (peer, gate_operand, mul_out)})
    by_name = {op.name: op for op in graph.op}
    cases = []
    if mode == 'upstream':
        walk = AttentionSteps(graph, by_name, [])
        z = walk.expect(walk.strip(gate_input), {MATMUL})
        quant = walk.expect(walk.strip(z.input[0]), {'DynamicQuant'})
        rms = walk.expect(peer, NORMS)
        hidden, core = quant.input[0], rms.input[0]
        wires[hidden] = input_wire(quant, (64, 2048))
        wires[core] = input_wire(rms, SHAPE)
        for label, inputs, producers in (
                ('z-branch', [hidden, peer], {z.name, quant.name}),
                ('rms-branch', [gate_input, core], {rms.name}),
                ('both-branches', [hidden, core], {z.name, quant.name, rms.name})):
            kept, used = slice_dependencies(by_name, [mul_out], set(inputs), label)
            if used != set(inputs) or (kept & {z.name, quant.name, rms.name}) != producers:
                raise ValueError(f'{label}: slice crosses unexpected projection or RMSNorm boundaries')
            cases.append((label, inputs, [name for name in by_name if name in kept], mul_out))
    else:
        for label, inputs, output in (
                ('cast-mul', [gate_out, peer], mul_out), ('silu', [gate_input], gate_out),
                ('silu-cast', [gate_input], gate_operand), ('silu-cast-mul', [gate_input, peer], mul_out)):
            kept, _ = slice_dependencies(by_name, [output], set(inputs), label)
            cases.append((label, inputs, [name for name in by_name if name in kept], output))
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
        if 'error' in step:
            report['cases'][label] = dict(status='CUT_FAILED', error=step['error'], nodes=[],
                                          input_edges=step['inputs'], output_edges=step['outputs'])
            save_json(index, report)
            print(f'[{label}] CUT_FAILED {step["error"]}', flush=True)
            continue
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
        case_info = dict(native_tool_args=True, files=files, air_directory=directory, inputs=feeds,
                         outputs=outputs, source_air=record)
        if meta.get('rms_silu_token_view'):
            case_info['rms_silu_token_view'] = True
        save_json(base/'case.json', case_info)
        print(f'[{label}] 已导出 AIR：{air}（{len(nodes)} 个节点）', flush=True)
        row = dict(nodes=nodes, source=source['sha256'], feeds=feeds, outputs=outputs)
        if mode == 'gdn-bisect':
            row.update(input_edges=step['inputs'], output_edges=step['outputs'])
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


def execute_case(base, row, runtime, config, *, replace_inputs, replace_weights, valid_rows=None, board_case=None, real_inputs=None):
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
        if real_inputs is not None:
            source, records = real_inputs
            record = records[role]
            if dtype != record['dtype'] or spec['bytes'] != record['bytes']:
                raise ValueError(f'{base.name}: {name} and original {role} input descriptors differ')
            paths.append(sized_file(source, record))
            operands.append(dict(spec, sha256=record['sha256']))
            continue
        if role in ('history', 'state'):
            data = np.zeros(shape, dtype=dtype)
        elif role == 'length':
            data = np.full(shape, valid_rows, dtype=dtype)
        else:
            low, high = {'g': (-1, -0.01), 'beta': (0.1, 0.9)}.get(
                role, (-2, 2) if name == 'l0_feed_0' else (0.5, 1.5))
            data = np.linspace(low, high, num=prod(shape), dtype=dtype)
        data.tofile(path)
        operands.append(dict(spec, sha256=sha(path)))
        paths.append(path)
    try:
        run_v2(replay, base.name, model, operands, paths, build['outputs'], runtime,
               board_case=board_case or '4-prefill-t64-gate-' + base.name, decoder_shapes=(named_shapes(operands), build['output_shape']),
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


def localize_gdn(folder, case, report, summary):
    """Compare observations for nested cuts, including the retained whole-module control."""
    builds = dict(report['cases'])
    observations = dict(summary)
    baseline = case.name
    whole_build, whole_result = case.parent/'compile.json', case.parent/'result.json'
    if whole_build.is_file() and whole_result.is_file():
        parent = read(whole_build)['cases'].get(baseline, {})
        observed = read(whole_result).get(baseline)
        if observed and parent.get('source') == report['air']['sha256']:
            builds[baseline], observations[baseline] = parent, observed

    matched = {k: set(builds[k]['nodes']) for k, row in observations.items()
               if row['status'] == 'LOAD_MODEL_FAILED' and row.get('kernel_args_error')}
    minimal = sorted((k for k, nodes in matched.items() if not any(other < nodes for other in matched.values())),
                     key=lambda k: (len(matched[k]), k))
    splits = {baseline: ('gdn-front', 'gdn-tail'),
              'gdn-front': ('pre-gdr', 'gdr-state'), 'pre-gdr': ('qkv-path', 'decay-beta'),
              'qkv-path': ('qkv-project', 'conv-to-qkv'), 'decay-beta': ('decay-path', 'beta-path'),
              'gdn-tail': ('gate-product', 'out-proj'), 'gate-product': ('z-gate', 'rms-mul'),
              'z-gate': ('z-project', 'gate-activation')}
    joins = {parent: children for parent, children in splits.items() if parent in matched
             and all(observations.get(k, {}).get('status') == 'EXECUTED' for k in children)}
    cases = {}
    for label, row in observations.items():
        base = case if label == baseline else folder/label
        cases[label] = dict(status=row['status'], exitcode=row.get('exitcode'),
                            matched_kernel_args=label in matched, nodes=builds[label]['nodes'],
                            inputs=builds[label].get('input_edges'), outputs=builds[label].get('output_edges'),
                            case=str(base/'case.json') if (base/'case.json').is_file() else None,
                            diagnosis=row.get('diagnosis'))
    path = folder/'localization.json'
    save_json(path, dict(source_air=str(case/report['air']['path']), baseline_observed=baseline in observations,
                         cases=cases, minimal_reproductions=minimal, passing_halves=joins))
    if minimal:
        print('复现同一加载错误的最小已测切片：' + '、'.join(minimal), flush=True)
        for k in minimal:
            print(f'  {k}：{len(matched[k])} 个源图节点，{cases[k]["case"]}', flush=True)
    else:
        print('已有记录中没有同时匹配 load_model 和本次 hiaiserver 的 OpUpdateKernelArgs failed。', flush=True)
    for k, children in joins.items():
        print(f'  {k} 复现，两段各自执行成功：{" + ".join(children)}', flush=True)
    print(f'二分定位报告 {path}', flush=True)


def air_bundle(case, alternate=None):
    meta = read(case/'case.json')
    record, = [r for r in meta['files'] if r['path'] == meta['air_directory'] + '/export.air']
    if alternate is not None and not (case/record['path']).is_file():
        case = alternate
        if record not in read(case/'case.json')['files']:
            raise ValueError('The local AIR export differs from the model on the board host')
    return case, meta, record


def probe(root, action, ddk, config, *, mode='slices', replace_inputs=False, replace_weights=False):
    parent = root/'board/L00-L03'
    case = root/'board/package/L00-L03'
    alternate = root/'export/package/L00-L03' if action == 'compile' else None
    bisect = mode == 'gdn-bisect'
    make_control = bisect and not (parent/'gdn/linear-attention/case.json').is_file()
    if bisect:
        source = parent/('gdn-bisect-source' if make_control else 'gdn')
        if make_control and action == 'compile':
            print('从四层包切出完整 L0 GDN，作为本批对照', flush=True)
            compile_cases(source, *air_bundle(case, alternate), ddk, 'gdn-whole')
        case, alternate = source/'linear-attention', None
    case, meta, record = air_bundle(case, alternate)
    whole = mode in ('gdn-export', 'gdn-whole')
    directory = {'gdn-bisect': 'gdn-bisect', 'upstream': 'gate-upstream',
                 'gdn-whole': 'gdn', 'gdn-export': 'gdn'}.get(mode, 'gate-probes')
    folder = parent/directory
    if action == 'compile':
        compile_cases(folder, case, meta, record, ddk, mode)
        return True
    report = read(folder/'compile.json')
    if report['kind'] != KIND or not report['complete'] or report['air'] != record:
        raise ValueError('Compile gate probes for the current four-layer AIR before executing')
    runtime = read(config.parent/read(config)['assets'])['environment']
    valid_rows = read(root/'board/package/fixture/reference.json')['valid_rows'] if whole or bisect else None
    if (whole or bisect) and not 1 <= valid_rows <= 64:
        raise ValueError(f'Unexpected T64 prefill length: {valid_rows}')

    def observe(base, row, board_case=None):
        if row['status'] != 'COMPILED':
            return row
        try:
            return execute_case(base, row, runtime, config, replace_inputs=replace_inputs,
                                replace_weights=replace_weights, valid_rows=valid_rows, board_case=board_case)
        except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as err:
            if not bisect or 'BOARD_MODEL_IO_BUSY' in str(err):
                raise
            print(f'[{base.name}] PROBE_ERROR {err}', flush=True)
            return dict(status='PROBE_ERROR', error=str(err))

    if make_control:
        control = read(case.parent/'compile.json')['cases'][case.name]
        save_json(case.parent/'result.json', {case.name: observe(case, control, '4-prefill-t64-gdn-bisect-control')})
    summary = {}
    save_json(folder/'result.json', summary)
    for label, row in report['cases'].items():
        summary[label] = observe(folder/label, row)
        save_json(folder/'result.json', summary)
    print(f'上板结果 {folder}/result.json', flush=True)
    for label, row in summary.items():
        matched = '  OpUpdateKernelArgs failed' if row.get('kernel_args_error') else ''
        print(f'{label:16} {row["status"]}{matched}', flush=True)
    if bisect:
        localize_gdn(folder, case, report, summary)
        # Reproducing a model failure is a completed diagnostic observation.
        return all(row['status'] in ('EXECUTED', 'LOAD_MODEL_FAILED') for row in summary.values())
    return all(row['status'] == 'EXECUTED' for row in summary.values())
