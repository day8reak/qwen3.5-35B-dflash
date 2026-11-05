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


class ProbeFailure(RuntimeError):
    """A failed attempt whose original error and bounded diagnostics were saved."""
    def __init__(self, report):
        super().__init__(report['error'])
        self.report = report


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def progress(row):
    """Keep stdout as one result JSON; immediately show existing compiler events."""
    try:
        print(f"stable-arg-max: {row['stage']}: {row['message']}", file=sys.stderr, flush=True)
    except OSError:
        # A closed diagnostic stream must not change the compilation result.
        pass


def failure_diagnostics(output, stage):
    from qwen35_moe_dflash.common.tool_logs import compilation_diagnostics, log_excerpt
    try:
        if stage == 'omg':
            return {'compilation_diagnostics': compilation_diagnostics(output / 'omc')}
        if stage == 'ge-export':
            return {'ge_diagnostics': {name: log_excerpt(output / name)
                for name in ('build.log', 'export.log') if (output / name).is_file()}}
    except (OSError, ValueError) as error:
        return {'diagnostics_error': str(error)}
    return {}


def inspect_compilation(directory):
    """Inspect even older failed attempts; never require source hashes or an SDK."""
    output = Path(directory).resolve()
    path = output / 'result.json'
    with path.open('rb') as stream:
        raw = stream.read((8 << 20) + 1)
    if len(raw) > 8 << 20:
        raise ValueError('probe report exceeds 8 MiB')
    report = json.loads(raw)
    if not isinstance(report, dict) or report.get('artifact_kind') != KIND:
        raise ValueError('expected an existing StableArgMax GE probe directory')
    return dict(artifact_kind=KIND + '-inspection', status='INSPECTED', source=str(output),
        source_report_sha256=hashlib.sha256(raw).hexdigest(),
        source_status=report.get('status'), source_stage=report.get('stage'),
        source_model_compilation=report.get('model_compilation'), error=report.get('error'),
        air=report.get('air'), model_compilation='NOT_RUN', device_execution='NOT_RUN',
        numerical_validation='NOT_RUN', scope='READ_ONLY_EXISTING_REPORT_AND_LOGS_NOT_A_RETRY',
        **failure_diagnostics(output, 'omg' if (output / 'omc').is_dir() else report.get('stage')))


def retained_export(directory):
    """Validate an existing export, allowing tooling-only changes since failure."""
    root = Path(directory).resolve()
    path = root / 'result.json'
    with path.open('rb') as stream:
        raw = stream.read((8 << 20) + 1)
    if len(raw) > 8 << 20:
        raise ValueError('retained probe report exceeds 8 MiB')
    report = json.loads(raw)
    if (not isinstance(report, dict) or report.get('artifact_kind') != KIND
            or report.get('status') not in ('FAILED', 'EXPORTED', 'COMPILED', 'PLACEMENT_REJECTED')
            or not isinstance(report.get('air'), dict)):
        raise ValueError('reuse requires an existing completed GE export with recorded AIR hash')
    shape = checked_shape(report['shape'])
    from qwen35_moe_dflash.graph.fused.emitter import emit_cpp
    graph = graph_for(shape)
    cpp = emit_cpp(graph, fold_constant_arithmetic=False).encode()
    source = root / 'probe.cpp'
    if (report.get('graph') != graph.manifest()
            or report.get('graph_cpp_sha256') != hashlib.sha256(cpp).hexdigest()
            or source.is_symlink() or not source.resolve().is_relative_to(root)
            or source.stat().st_size != len(cpp) or source.read_bytes() != cpp):
        raise ValueError('retained GE graph/source differs from the current recipe')
    # Diagnostic/CLI changes may differ; the actual op and reference must not.
    for name, digest in source_hashes().items():
        if name != 'scripts/stable_arg_max_ge.py' and report.get('source_sha256', {}).get(name) != digest:
            raise ValueError('retained operator/reference source changed: ' + name)
    relative = Path(report['air']['path'])
    if relative != Path('air/stable_arg_max.air'):
        raise ValueError('unexpected retained AIR path')
    air = root / relative
    if (air.is_symlink() or not air.is_file() or not air.resolve().is_relative_to(root)
            or not 0 < air.stat().st_size <= 64 << 20 or sha(air) != report['air'].get('sha256')):
        raise ValueError('retained AIR bytes/hash differ or file is outside the probe')
    return root, report, dict(source=str(root), source_report_sha256=hashlib.sha256(raw).hexdigest(),
        air_sha256=report['air']['sha256'], ge_export='REUSED_HASH_CHECKED_AIR',
        graph_cpp_sha256=report['graph_cpp_sha256'])


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
    progress(dict(stage='preflight', message='Checking export inputs and retained AIR identity'))
    from qwen35_moe_dflash.graph.fused.emitter import emit_cpp
    from qwen35_moe_dflash.graph.export import _logged
    from qwen35_moe_dflash.toolchain.existing_air import compile_existing_air
    reused = retained_export(args.reuse_export) if getattr(args, 'reuse_export', None) else None
    shape = checked_shape(args.shape if args.shape is not None else
        reused[1]['shape'] if reused else reference.SHAPE)
    if reused and tuple(reused[1]['shape']) != shape:
        raise ValueError('--shape differs from retained export')
    if math.prod(shape) * 2 + math.prod(shape[:-1]) * 4 > 16 << 20:
        raise ValueError('standalone probe IO is bounded to 16 MiB; this is not the kernel shape limit')
    output = Path(args.output).absolute()
    if output.is_symlink():
        raise ValueError('output must not be a symlink')
    output = output.resolve()
    protected = [REPO]
    if reused:
        protected.append(reused[0])
    for selected in (args.ge_root, args.ddk_root):
        if selected:
            protected.append(Path(selected).resolve())
    if any(output == p or p in output.parents or output in p.parents for p in protected):
        raise ValueError('output must be separate from repository and SDKs')
    if not args.plan_only and not reused and not args.ge_root:
        raise ValueError('provide --ge-root/CANN_ROOT with GE C++ include/graph and lib64/libgraph.so')
    if not args.plan_only and not args.export_only and not (args.ddk_root and args.ddk_id):
        raise ValueError('provide CPRO_DDK_ROOT and CPRO_DDK_ID (or corresponding options)')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(artifact_kind=KIND, status='PREPARING', stage='fixtures', shape=list(shape),
        model_compilation='NOT_RUN', device_execution='NOT_RUN', numerical_validation='NOT_RUN',
        frontend='GE_CPP_NO_TORCH_TORCHAIR_TORCH_NPU', source_sha256=source_hashes(),
        contract='FINITE_FP16_FIRST_MAXIMUM_EXACT_INT32', cases=[])
    try:
        progress(dict(stage='fixtures', message=f'Preparing and verifying reference inputs; output={output}'))
        graph = graph_for(shape)
        report['graph'] = graph.manifest()
        cpp = output / 'probe.cpp'
        cpp.write_text(emit_cpp(graph, fold_constant_arithmetic=False))
        report['graph_cpp_sha256'] = sha(cpp)
        if reused:
            report['reused_export'] = reused[2]
            expected_cases = ['stable_arg_max__' + name for name, _ in fixtures(shape)]
            if [r['case'] for r in reused[1].get('cases', [])] != expected_cases:
                raise ValueError('retained fixture suite differs')
            saved_cases = {r['case']: r for r in reused[1]['cases']}
        for scenario, x in fixtures(shape):
            name = 'stable_arg_max__' + scenario
            folder = output / 'fixture' / name
            folder.mkdir(parents=True)
            y = np.argmax(x, axis=-1).astype('<i4')
            if reused:
                for prefix, array, key in (('input', x, 'input_sha256'), ('golden', y, 'golden_sha256')):
                    old = reused[0] / 'fixture' / name / (prefix + '_0.bin')
                    if (old.is_symlink() or not old.resolve().is_relative_to(reused[0])
                            or old.stat().st_size != array.nbytes or sha(old) != saved_cases[name][key]
                            or old.read_bytes() != array.tobytes()):
                        raise ValueError('retained fixture differs from direct reference: ' + name)
            x.tofile(folder / 'input_0.bin')
            y.tofile(folder / 'golden_0.bin')
            report['cases'].append(dict(case=name, input_sha256=sha(folder / 'input_0.bin'),
                golden_sha256=sha(folder / 'golden_0.bin')))
        report['reference'] = 'DIRECT_NUMPY_ARGMAX; NO_ON_SERVER_TORCH_CROSS_CHECK'
        if args.plan_only:
            report.update(status='PLANNED', stage='complete')
            return report
        (output / 'air').mkdir()
        air = output / 'air/stable_arg_max.air'
        if reused:
            report['stage'] = 'reuse-air'
            progress(dict(stage='reuse-air', message='Reusing verified AIR; GE export is skipped'))
            shutil.copyfile(reused[0] / 'air/stable_arg_max.air', air)
            if sha(air) != reused[2]['air_sha256']:
                raise ValueError('AIR changed during reuse copy')
        else:
            ge = Path(args.ge_root).resolve()
            for path in (ge / 'include/graph/graph.h', ge / 'lib64/libgraph.so'):
                if not path.is_file():
                    raise FileNotFoundError(path)
            report['stage'] = 'ge-export'
            progress(dict(stage='ge-export', message=f'Building GE exporter; logs={output}/build.log, {output}/export.log'))
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
            _logged([str(exporter), str(air)], folder=output, env=env,
                log=output / 'export.log', timeout=args.timeout_seconds)
        report.update(status='EXPORTED', air=dict(path=str(air.relative_to(output)), sha256=sha(air)))
        if args.export_only:
            report['stage'] = 'complete'
            return report
        report['stage'] = 'omg'
        write(output / 'result.json', report)
        progress(dict(stage='omg-prepare', message=f'Preparing selected DDK/TBE; OMG log={output}/omc/compile.log; '
            f'OMG timeout={args.timeout_seconds:g}s; wait updates every 30s once OMG starts'))
        compiled = compile_existing_air(air, output / 'omc', args.ddk_root, ddk_id=args.ddk_id,
            timeout_seconds=args.timeout_seconds, use_origin_format=True, save_weights_as_external_data=False,
            tbe_diagnosis=getattr(args, 'tbe_diagnosis', None), progress=progress)
        report['model_compilation'] = compiled['model_compilation']
        compatibility = compiled.get('tbe_compatibility')
        if compatibility is not None:
            report['tbe_compatibility'] = dict(diagnosis=compatibility.get('diagnosis'),
                audit_status=compatibility.get('audit', {}).get('status', 'NOT_OBSERVED'),
                report='omc/report.json')
        if compiled['status'] != 'COMPILED':
            raise RuntimeError(str(compiled.get('error', compiled['status'])))
        model = output / 'omc' / compiled['omc']['path']
        report.update(omc=str(model.relative_to(output)), omc_sha256=sha(model), ddk_id=args.ddk_id)
        report['stage'] = 'inspect'
        progress(dict(stage='inspect', message='OMG completed; inspecting compiled placement and interfaces'))
        inspection = inspect_model(model, args.ddk_root, report['graph'])
        write(output / 'placement-inspection.json', inspection)
        report['placement'] = inspection['placement']
        report.update(status='COMPILED' if inspection['placement']['npu_only_candidate'] else 'PLACEMENT_REJECTED',
            stage='complete', next='Use validate_draft_vector.py with this directory; no model recompilation.')
        return report
    except Exception as error:
        report.update(status='FAILED', error=f'{type(error).__name__}: {error}')
        # Keep the original failure; do not replace it with an excerpt or retry.
        report.update(failure_diagnostics(output, report['stage']))
        raise ProbeFailure(report) from error
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shape', nargs='+', type=int, help='default Qwen shape, or the retained export shape')
    parser.add_argument('--output')
    parser.add_argument('--ge-root', default=os.environ.get('CANN_ROOT'))
    parser.add_argument('--ddk-root', default=os.environ.get('CPRO_DDK_ROOT'))
    parser.add_argument('--ddk-id', default=os.environ.get('CPRO_DDK_ID'))
    parser.add_argument('--tbe-diagnosis', help='existing selected-DDK diagnosis for run-local TBE compatibility')
    parser.add_argument('--cxx', default='c++')
    parser.add_argument('--cxx11-abi', type=int, choices=(0, 1), default=0)
    parser.add_argument('--timeout-seconds', type=int, default=1800)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--plan-only', action='store_true', help='source + NumPy fixtures, no GE/OMG needed')
    modes.add_argument('--export-only', action='store_true', help='GE C++ -> AIR without OMG')
    modes.add_argument('--reuse-export', metavar='DIRECTORY',
        help='reuse a hash-checked existing AIR/recipe in a new compile directory; no GE export')
    modes.add_argument('--inspect-compilation', metavar='DIRECTORY',
        help='read a retained result/report/log; no compilation, writes, SDK or device')
    args = parser.parse_args(argv)
    if args.inspect_compilation:
        if args.output:
            parser.error('--inspect-compilation does not take --output or write a new run')
    elif not args.output:
        parser.error('export/compile requires --output')
    try:
        report = inspect_compilation(args.inspect_compilation) if args.inspect_compilation else run(args)
    except ProbeFailure as error:
        report = error.report
    except (OSError, ValueError, RuntimeError) as error:
        report = dict(artifact_kind=KIND, status='FAILED',
            stage='inspect' if args.inspect_compilation else 'preflight',
            error=f'{type(error).__name__}: {error}', model_compilation='NOT_RUN',
            device_execution='NOT_RUN', numerical_validation='NOT_RUN')
    print(json.dumps(report))
    return 0 if report['status'] in ('COMPILED', 'EXPORTED', 'PLANNED', 'INSPECTED') else 1


if __name__ == '__main__':
    raise SystemExit(main())
