#!/usr/bin/env python3
"""DDK-server GE/AIR -> OMG probe. Requires NumPy and GE C++, never TorchAir."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[3]
sys.path.insert(0, str(REPO / 'model'))
import numpy as np
import reference

KIND = 'stable-arg-max-ge-probe'
QWEN_SCENARIOS = ('index_boundaries', 'ties_across_boundaries', 'zeros_equal_extremes',
    'signed_subnormal', 'adjacent_representable', 'random_seed350248320', 'finite_binary16_alphabet')
GENERAL_SCENARIOS = ('first_last', 'equal', 'tile_ties', 'subnormal')
CASES = tuple('stable_arg_max__' + s for s in (*QWEN_SCENARIOS, *GENERAL_SCENARIOS))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def checked_shape(shape):
    shape = tuple(shape)
    if (not 1 <= len(shape) <= 8 or any(type(d) is not int or d <= 0 for d in shape)
            or math.prod(shape) > 2147483647):
        raise ValueError('requires static rank 1..8, positive dimensions, numel <= INT32_MAX')
    return shape


def source_hashes():
    from prepare_allops import FILES
    return {p: sha(ROOT / p) for p in (*FILES, 'scripts/stable_arg_max_ge.py', 'scripts/reference.py')}


def graph_for(shape):
    from qwen35_moe_dflash.graph.fused.ir import Graph
    shape = checked_shape(shape)
    graph = Graph('stable_arg_max_probe')
    x = graph.input('x', shape, 'float16')
    result = graph.custom('StableArgMax', {'x': x},
        {'indices': (shape[:-1], 'int32')}, name='top1')
    graph.outputs = [result['indices']]
    return graph


def fixtures(shape):
    shape = checked_shape(shape)
    if shape == reference.SHAPE:
        yield from reference.fixtures()
        return
    x = np.full(shape, -2, dtype='<f2')
    rows = x.reshape(-1, shape[-1])
    rows[::2, 0] = 2
    rows[1::2, -1] = 2
    yield 'first_last', x.copy()
    x.fill(-65504)
    yield 'equal', x.copy()
    x.fill(-2)
    # Uneven final tile; ties span the 8192-value internal tile boundary.
    a, b = min(8191, shape[-1] - 1), min(8192, shape[-1] - 1)
    rows[:, [a, b]] = 2
    yield 'tile_ties', x.copy()
    x.fill(0)
    rows[:, -1] = np.array([1], dtype='<u2').view('<f2')[0]
    yield 'subnormal', x.copy()


def inspect_model(model, ddk, graph):
    from qwen35_moe_dflash.toolchain.omc_wire import load_schema
    from qwen35_moe_dflash.toolchain.omc_container import read_container, read_partition
    from qwen35_moe_dflash.draft.ir_inspect import _enum_fields
    from qwen35_moe_dflash.draft.incremental.top1_ir_inspect import _compiled_model, placement_summary
    # Reuse the same strict public ABI binding as the existing board runner.
    sys.path.insert(0, str(REPO / 'targets/cpro/tools'))
    from validate_draft_vector import bind_io
    schema = load_schema(Path(ddk) / 'tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so')
    decoded = _compiled_model(model, schema, compact=True)
    decoded.update(case='stable_arg_max', status='INSPECTED')
    placement = placement_summary(dict(status='INSPECTED', cases=[decoded]))['cases'][0]
    main = [p for p in read_container(model)['partitions'] if p['type'] == 0]
    if len(main) != 1 or any(model.parent.glob('*.weight')):
        raise ValueError('expected one embedded model without external weights')
    io = bind_io(schema.decode(read_partition(model, main[0]), 'ModelDef'), graph, _enum_fields(schema))
    return dict(placement=placement, io=io, schema=schema.evidence)


def run(args):
    from qwen35_moe_dflash.graph.fused.emitter import emit_cpp
    from qwen35_moe_dflash.graph.export import _logged
    from qwen35_moe_dflash.toolchain.existing_air import compile_existing_air
    shape = checked_shape(args.shape)
    if math.prod(shape) * 2 + math.prod(shape[:-1]) * 4 > 16 << 20:
        raise ValueError('standalone probe IO is bounded to 16 MiB; this is not the kernel shape limit')
    output = Path(args.output).absolute()
    if output.is_symlink():
        raise ValueError('output must not be a symlink')
    output = output.resolve()
    protected = [REPO]
    for selected in (args.ge_root, args.ddk_root):
        if selected:
            protected.append(Path(selected).resolve())
    if any(output == p or p in output.parents or output in p.parents for p in protected):
        raise ValueError('output must be separate from repository and SDKs')
    if not args.plan_only and not args.ge_root:
        raise ValueError('provide --ge-root/CANN_ROOT with GE C++ include/graph and lib64/libgraph.so')
    if not args.plan_only and not args.export_only and not (args.ddk_root and args.ddk_id):
        raise ValueError('provide CPRO_DDK_ROOT and CPRO_DDK_ID (or corresponding options)')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(artifact_kind=KIND, status='PREPARING', stage='fixtures', shape=list(shape),
        model_compilation='NOT_RUN', device_execution='NOT_RUN', numerical_validation='NOT_RUN',
        frontend='GE_CPP_NO_TORCH_TORCHAIR_TORCH_NPU', source_sha256=source_hashes(),
        contract='FINITE_FP16_FIRST_MAXIMUM_EXACT_INT32', cases=[])
    try:
        graph = graph_for(shape)
        report['graph'] = graph.manifest()
        cpp = output / 'probe.cpp'
        cpp.write_text(emit_cpp(graph, fold_constant_arithmetic=False))
        report['graph_cpp_sha256'] = sha(cpp)
        for scenario, x in fixtures(shape):
            name = 'stable_arg_max__' + scenario
            folder = output / 'fixture' / name
            folder.mkdir(parents=True)
            y = np.argmax(x, axis=-1).astype('<i4')
            x.tofile(folder / 'input_0.bin')
            y.tofile(folder / 'golden_0.bin')
            report['cases'].append(dict(case=name, input_sha256=sha(folder / 'input_0.bin'),
                golden_sha256=sha(folder / 'golden_0.bin')))
        report['reference'] = 'DIRECT_NUMPY_ARGMAX; NO_ON_SERVER_TORCH_CROSS_CHECK'
        if args.plan_only:
            report.update(status='PLANNED', stage='complete')
            return report
        ge = Path(args.ge_root).resolve()
        for path in (ge / 'include/graph/graph.h', ge / 'lib64/libgraph.so'):
            if not path.is_file():
                raise FileNotFoundError(path)
        report['stage'] = 'ge-export'
        env = dict(os.environ)
        library = ge / 'lib64'
        env['LD_LIBRARY_PATH'] = str(library) + os.pathsep + env.get('LD_LIBRARY_PATH', '')
        exporter = output / 'exporter'
        command = [args.cxx, '-std=c++17', f'-D_GLIBCXX_USE_CXX11_ABI={args.cxx11_abi}',
            '-I', str(ge / 'include'), str(cpp), '-L', str(library),
            f'-Wl,-rpath,{library}', '-lgraph']
        if (library / 'libgraph_base.so').is_file():
            command.append('-lgraph_base')
        command += ['-o', str(exporter)]
        write(output / 'build-command.json', dict(argv=command, ge_root=str(ge),
            libgraph_sha256=sha(library / 'libgraph.so')))
        _logged(command, folder=output, env=env, log=output / 'build.log', timeout=args.timeout_seconds)
        (output / 'air').mkdir()
        air = output / 'air/stable_arg_max.air'
        _logged([str(exporter), str(air)], folder=output, env=env,
            log=output / 'export.log', timeout=args.timeout_seconds)
        report.update(status='EXPORTED', air=dict(path=str(air.relative_to(output)), sha256=sha(air)))
        if args.export_only:
            report['stage'] = 'complete'
            return report
        report['stage'] = 'omg'
        write(output / 'result.json', report)
        compiled = compile_existing_air(air, output / 'omc', args.ddk_root, ddk_id=args.ddk_id,
            timeout_seconds=args.timeout_seconds, use_origin_format=True, save_weights_as_external_data=False)
        report['model_compilation'] = compiled['model_compilation']
        if compiled['status'] != 'COMPILED':
            raise RuntimeError(str(compiled.get('error', compiled['status'])))
        model = output / 'omc' / compiled['omc']['path']
        report.update(omc=str(model.relative_to(output)), omc_sha256=sha(model), ddk_id=args.ddk_id)
        report['stage'] = 'inspect'
        inspection = inspect_model(model, args.ddk_root, report['graph'])
        write(output / 'placement-inspection.json', inspection)
        report['placement'] = inspection['placement']
        report.update(status='COMPILED' if inspection['placement']['npu_only_candidate'] else 'PLACEMENT_REJECTED',
            stage='complete', next='Use validate_draft_vector.py with this directory; no model recompilation.')
        return report
    except Exception as error:
        report.update(status='FAILED', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write(output / 'result.json', report)


def prepare_fixture(root, out, ddk, ddk_id, selection):
    """Adapter for existing ten-repeat board launcher, including recover/hash checks."""
    from validate_draft_vector import member, selected_cases
    from qwen35_moe_dflash.execution.hdc_run import _file_record
    report = json.loads((root / 'result.json').read_text())
    shape = checked_shape(report['shape'])
    graph = graph_for(shape).manifest()
    if (report.get('artifact_kind') != KIND or report.get('status') != 'COMPILED'
            or report.get('model_compilation') != 'PASSED' or report.get('ddk_id') != ddk_id
            or report.get('source_sha256') != source_hashes() or report.get('graph') != graph):
        raise ValueError('StableArgMax source/graph/compiler identity differs or compilation not accepted')
    # Scalar and rank>4 runner support is a separate ABI gate, not a kernel limitation.
    if len(shape) == 1 or len(shape) > 4:
        raise ValueError('existing tensor runner requires rank 2..4 input for this probe; AIR export supports rank 1..8')
    compilation = json.loads(member(root, 'omc/report.json').read_text())
    model = member(root, report['omc'])
    if (compilation.get('status') != 'COMPILED' or compilation.get('platform') != 'kirinx90'
            or compilation.get('ddk_id') != ddk_id
            or model != member(root, 'omc/' + compilation['omc']['path'])
            or model.stat().st_size > 64 << 20 or sha(model) != report['omc_sha256']):
        raise ValueError('StableArgMax compiled model identity differs')
    inspection = inspect_model(model, ddk, graph)
    if not inspection['placement']['npu_only_candidate']:
        raise ValueError('StableArgMax is not NPU-only at decoded graph level')
    generated = list(fixtures(shape))
    available = ['stable_arg_max__' + s for s, _ in generated]
    if [r['case'] for r in report['cases']] != available:
        raise ValueError('fixture suite differs')
    selected = selected_cases(available if selection is None else selection, available)
    saved = {r['case']: r for r in report['cases']}
    cases = []
    for scenario, x in generated:
        name = 'stable_arg_max__' + scenario
        if name not in selected:
            continue
        io = json.loads(json.dumps(inspection['io']))
        folder = out / 'fixture' / name
        folder.mkdir(parents=True)
        plan = ['QWEN35_TENSOR_PROBE 1', 'iterations 3 10']
        y = np.argmax(x, axis=-1).astype('<i4')
        for direction, prefix, array, hash_key in (
                ('inputs', 'input', x, 'input_sha256'), ('outputs', 'golden', y, 'golden_sha256')):
            spec = io[direction][0]
            path = member(root, f'fixture/{name}/{prefix}_0.bin')
            if sha(path) != saved[name][hash_key] or path.read_bytes() != array.tobytes():
                raise ValueError('fixture changed or differs from independent reference: ' + name)
            shutil.copyfile(path, folder / f'{prefix}_0.bin')
            spec['fixture_sha256'] = sha(path)
            plan.append(' '.join((direction[:-1], '0', spec['name'], spec['dtype'],
                ','.join(map(str, spec['shape'])))))
        (folder / 'plan.txt').write_text('\n'.join(plan) + '\n')
        cases.append(dict(case=name, **io, model=_file_record(model, name + '/model.omc'),
            compiler_options=compilation.get('requested_options', {}),
            graphop_counts=inspection['placement']['graphop_counts'],
            top1_contract=dict(acceptance='EXACT_INT32_BYTES_ALL_TEN_REPEATS',
                input_contract='FINITE_FP16', tie_rule='SMALLEST_INDEX')))
    cases.sort(key=lambda row: selected.index(row['case']))
    result = dict(cases=cases, warmup=3, repetitions=10, atol=0.0, rtol=0.0, exact_bytes=True,
        ddk_id=ddk_id, schema=inspection['schema'], scope='STABLE_ARG_MAX_GE_SINGLE_OPERATOR_NOT_FULL_DRAFT')
    write(out / 'fixture/manifest.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shape', nargs='+', type=int, default=[1, 15, 248320])
    parser.add_argument('--output', required=True)
    parser.add_argument('--ge-root', default=os.environ.get('CANN_ROOT'))
    parser.add_argument('--ddk-root', default=os.environ.get('CPRO_DDK_ROOT'))
    parser.add_argument('--ddk-id', default=os.environ.get('CPRO_DDK_ID'))
    parser.add_argument('--cxx', default='c++')
    parser.add_argument('--cxx11-abi', type=int, choices=(0, 1), default=0)
    parser.add_argument('--timeout-seconds', type=int, default=1800)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--plan-only', action='store_true', help='source + NumPy fixtures, no GE/OMG needed')
    modes.add_argument('--export-only', action='store_true', help='GE C++ -> AIR without OMG')
    report = run(parser.parse_args())
    print(json.dumps(report))
    return 0 if report['status'] in ('COMPILED', 'EXPORTED', 'PLANNED') else 1


if __name__ == '__main__':
    raise SystemExit(main())
