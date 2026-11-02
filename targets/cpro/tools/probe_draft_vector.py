#!/usr/bin/env python3
"""Inspect the selected DDK or compile small FP32 Draft regions; no board access."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import shutil
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))

KIND = 'qwen35-draft-vector-probe'
SCHEMA = ('https://github.com/Ascend/torchair/blob/'
          'e1118e8b0fc1b95190ab53b732c0872989550e20/'
          'python/torchair/_ge_concrete_graph/auto_generated_ge_raw_ops.py')
CASES = ('rms_primitive_hidden', 'rms_builtin_hidden', 'rms_builtin_heads',
         'softmax_fp32', 'residual_fp32', 'residual_addcustom_fp32')


def graph_case(case):
    """Keep FP32 IO explicit; support does not prove internal precision."""
    import numpy as np
    from qwen35_moe_dflash.graph.fused.ir import Graph
    if case not in CASES:
        raise ValueError('unknown probe case')
    shape = ((1, 16, 32, 128) if case.endswith('heads') else
             (1, 32, 16, 208) if case == 'softmax_fp32' else (1, 16, 2048))
    g = Graph('draft_vector_probe')
    x = g.input('x', shape, 'float32')
    rng = np.random.default_rng(17)
    data = rng.standard_normal(shape).astype('<f4')
    rows = data.reshape(-1, shape[-1])
    rows[0] = 0
    rows[1] *= 1024  # Expose a compiler that silently squares in FP16.
    rows[2] *= 1e-4
    inputs = [data]
    if case.startswith('rms_'):
        eps = float(np.float32(1e-6))
        mean = np.mean(data * data, axis=-1, keepdims=True, dtype=np.float32)
        rstd = (np.float32(1) / np.sqrt(mean + np.float32(eps))).astype('<f4')
        golden = [(data * rstd).astype('<f4')]
        if case.startswith('rms_builtin'):
            gamma = g.const(np.ones(shape[-1], dtype='<f4'), dtype='float32', name='unit_gamma')
            result = g.custom('RmsNorm', {'x': x, 'gamma': gamma},
                {'y': (shape, 'float32'), 'rstd': ((*shape[:-1], 1), 'float32')},
                {'epsilon': {'type': 'float', 'value': eps}}, name='norm')
            g.outputs = [result['y'], result['rstd']]
            golden.append(rstd)
        else:
            square = g.op('mul', x, x, shape=shape, dtype='float32', name='square')
            reduced = g.op('reduce_mean', square, shape=(*shape[:-1], 1), dtype='float32',
                attrs={'axes': [-1], 'keepdims': True}, name='mean')
            shifted = g.op('scalar_add', reduced, shape=reduced.shape, dtype='float32',
                attrs={'value': eps}, name='epsilon')
            reciprocal = g.op('rsqrt', shifted, shape=shifted.shape, dtype='float32', name='rstd')
            g.outputs = [g.op('mul', x, reciprocal, shape=shape, dtype='float32', name='normalized')]
    elif case == 'softmax_fp32':
        rows[:, shape[-1] // 2:] = -np.inf  # Every row retains visible keys.
        exp = np.exp(data - np.max(data, axis=-1, keepdims=True))
        golden = [(exp / np.sum(exp, axis=-1, keepdims=True, dtype=np.float32)).astype('<f4')]
        g.outputs = [g.op('softmax', x, shape=shape, dtype='float32', attrs={'axis': -1}, name='probability')]
    else:
        right = rng.standard_normal(shape).astype('<f4')
        right.reshape(-1, shape[-1])[1] = -rows[1] + np.float32(0.01)
        inputs.append(right)
        y = g.input('residual', shape, 'float32')
        golden = [(data + right).astype('<f4')]
        if case == 'residual_addcustom_fp32':
            # Selected X90 DDK registration: AddCustom(x, y) -> z, ND FP32.
            # It also permits precision_reduce; declaration alone is not a gate.
            result = g.custom('AddCustom', {'x': x, 'y': y},
                              {'z': (shape, 'float32')}, name='sum')
            g.outputs = [result['z']]
        else:
            g.outputs = [g.op('add', x, y, shape=shape, dtype='float32', name='sum')]
    return g.validate(), inputs, golden


def precision_boundaries(nodes):
    """Summarize actual descriptors without claiming internal arithmetic dtype."""
    casts, graphs = [], []
    for node in nodes:
        ins = [d.get('dtype') for d in node['inputs']]
        outs = [d.get('dtype') for d in node['outputs']]
        if node['type'] in {'Cast', 'CastT'} and ins == ['float32'] and outs == ['float16']:
            casts.append(node['name'])
        if node['type'] == 'GraphOp':
            graphs.append({'name': node['name'], 'owner': node['owner'],
                           'input_dtypes': ins, 'output_dtypes': outs})
    npu = [g for g in graphs if g['owner'] == 'NPUCL']
    return {'fp32_to_fp16_casts': casts, 'graph_interfaces': graphs,
            'all_npu_boundaries_fp32': bool(npu) and all(
                g['input_dtypes'] and g['output_dtypes']
                and all(d == 'float32' for d in g['input_dtypes'] + g['output_dtypes']) for g in npu),
            'scope': 'COMPILED_BOUNDARY_DESCRIPTORS_NOT_INTERNAL_INSTRUCTIONS'}


def inspect_placement(omc, schema_path):
    from qwen35_moe_dflash.toolchain.omc_container import read_container, read_partition
    from qwen35_moe_dflash.toolchain.omc_wire import load_schema
    from qwen35_moe_dflash.toolchain.omc_cpu_inspect import _task
    from qwen35_moe_dflash.draft.ir_inspect import _enum_fields
    from qwen35_moe_dflash.draft.compile_precision import _observed_desc
    container = read_container(omc)
    schema = load_schema(schema_path)
    parts = [p for p in container['partitions'] if p['type'] == 0]
    if len(parts) != 1:
        raise ValueError('expected one model partition for placement inspection')
    part = parts[0]
    document = schema.decode(read_partition(omc, part), 'ModelDef')
    enums, counts, nodes = _enum_fields(schema), Counter(), []
    for graph in document.get('graph', []):
        for node in graph.get('op', []):
            kind = node.get('type')
            row = {'name': node.get('name'), 'type': kind,
                'inputs': [_observed_desc(d, enums) for d in node.get('input_desc', [])],
                'outputs': [_observed_desc(d, enums) for d in node.get('output_desc', [])]}
            if kind == 'GraphOp':
                row['owner'] = _task(node)['cl_name']
                counts[row['owner']] += 1
            nodes.append(row)
    if not counts:
        raise ValueError('no decoded GraphOp ownership; cannot establish placement')
    return {'graphop_counts': dict(counts), 'nodes': nodes,
        'precision_boundaries': precision_boundaries(nodes),
        'status': 'NPU_GRAPH_PRESENT' if counts['NPUCL'] else 'NO_NPU_GRAPH',
        'internal_precision': 'NOT_VERIFIED_BY_GRAPH_BOUNDARY_DTYPES',
        'scope': 'Compiler task ownership and boundary descriptors only; no device accuracy or latency claim'}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def inspect_ddk(compiled, ddk_root):
    """Read registration records and locate sources without loading vendor code."""
    if not ddk_root or not Path(ddk_root).is_dir():
        raise ValueError('set CPRO_DDK_ROOT or pass an existing --ddk-root')
    root, compiled = Path(ddk_root).resolve(), Path(compiled).resolve()
    if not compiled.is_dir():
        raise FileNotFoundError(compiled)
    limits = {'file_bytes': 2 * 1024 * 1024, 'total_bytes': 32 * 1024 * 1024,
              'entries': 10000, 'matches': 32, 'record_characters': 6000}
    report = {'artifact_kind': 'qwen35-draft-ddk-inspection', 'status': 'INSPECTED',
        'scope': 'READ_ONLY_REGISTRATION_AND_FILE_PRESENCE_NOT_FP32_SUPPORT_PROOF',
        'model_compilation': 'NOT_RUN', 'device_execution': 'NOT_RUN',
        'ddk_root': str(root), 'limits': limits, 'compile_environments': [],
        'roots': [], 'registrations': [], 'source_candidates': [], 'notes': []}
    used = {'bytes': 0, 'entries': 0, 'files_read': 0, 'symlinks_skipped': 0}
    seen_files, seen_dirs = set(), set()

    def note(path, error):
        if len(report['notes']) < 32:
            report['notes'].append({'path': str(path), 'error': str(error)})
        report['status'] = 'PARTIAL'

    def read(path):
        try:
            size = path.stat().st_size
            if size > limits['file_bytes'] or used['bytes'] + size > limits['total_bytes']:
                raise ValueError('read budget exceeded')
            with path.open('rb') as stream:
                raw = stream.read(size + 1)
            if len(raw) > size:
                raise ValueError('file grew while reading')
            used['bytes'] += len(raw)
            used['files_read'] += 1
            return raw.decode('utf-8', errors='replace')
        except (OSError, ValueError) as error:
            note(path, error)
            return None

    # The compile record includes KIRIN_CONF_JSON_PATH; it need not be under OMG.
    roots = []
    for case in CASES:
        path = compiled / case / 'omc/omg-environment.json'
        if not path.is_file():
            continue
        raw = read(path)
        try:
            record = json.loads(raw) if raw is not None else {}
            env = record.get('environment', {})
            conf = env.get('KIRIN_CONF_JSON_PATH')
            report['compile_environments'].append({'case': case, 'record': str(path),
                'ddk_root': record.get('ddk_root'), 'KIRIN_CONF_JSON_PATH': conf})
            if conf:
                roots.append(Path(conf))
        except (ValueError, AttributeError, TypeError) as error:
            note(path, error)
    if not report['compile_environments']:
        note(compiled, 'no saved compiler environment found; only selected DDK defaults will be scanned')
    roots.extend(root / name for name in ('ddk/ascendcpp', 'tools/platform/kirinx90',
                 'tools/tools_ascendc', 'tools/tools_omg/master'))
    match = re.compile(r'add_?custom', re.I)

    def records(value, location='$'):
        if isinstance(value, dict):
            if any(isinstance(v, str) and match.fullmatch(v) for v in value.values()):
                yield location, value
                return
            for key, item in value.items():
                child = location + '[' + json.dumps(key) + ']'
                if match.fullmatch(key):
                    yield child, item
                else:
                    yield from records(item, child)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                yield from records(item, f'{location}[{index}]')

    for folder in dict.fromkeys(roots):
        report['roots'].append({'path': str(folder), 'exists': folder.is_dir()})
        if not folder.is_dir():
            continue
        # Explicit config roots can be links. Do not recursively follow child links.
        for directory, dirs, files in os.walk(folder, followlinks=False,
                onerror=lambda error: note(error.filename, error)):
            resolved = Path(directory).resolve()
            if resolved in seen_dirs:
                dirs[:] = []
                continue
            seen_dirs.add(resolved)
            used['entries'] += len(dirs) + len(files)
            if used['entries'] > limits['entries']:
                report['status'] = 'PARTIAL'
                break
            used['symlinks_skipped'] += sum((Path(directory) / name).is_symlink() for name in dirs)
            dirs[:] = sorted(name for name in dirs if not (Path(directory) / name).is_symlink())
            for name in sorted(files):
                path = Path(directory) / name
                if path.is_symlink():
                    used['symlinks_skipped'] += 1
                    continue
                if path in seen_files:
                    continue
                seen_files.add(path)
                if match.search(str(path.relative_to(folder))):
                    if len(report['source_candidates']) < 40:
                        report['source_candidates'].append(str(path))
                    else:
                        note(folder, 'source candidate list truncated at 40 files')
                if path.suffix.lower() != '.json':
                    continue
                raw = read(path)
                if raw is None or not match.search(raw):
                    continue
                try:
                    found = records(json.loads(raw))
                    for location, record in found:
                        if len(report['registrations']) >= limits['matches']:
                            note(path, 'registration list truncated at match limit')
                            break
                        serialized = json.dumps(record, ensure_ascii=False)
                        row = {'path': str(path), 'json_path': location}
                        if len(serialized) <= limits['record_characters']:
                            row['record'] = record
                        else:
                            row['record_excerpt'] = serialized[:limits['record_characters']]
                            row['truncated'] = True
                        report['registrations'].append(row)
                except (ValueError, RecursionError) as error:
                    note(path, error)
        if used['entries'] > limits['entries']:
            report['notes'].append({'error': 'entry budget exceeded; remaining roots not scanned'})
            break
    report['files'] = [{'path': str(root / name), 'exists': (root / name).is_file()}
        for name in ('tools/tools_omg/master/lib64/libai_npucore_ascendc_kernel.so',
                     'tools/tools_ascendc/bisheng/bin/bisheng', 'ddk/ccec_compiler/bin/ccec')]
    report['coverage'] = used
    report['next'] = ('Inspect AddCustom input/output types, registration and JIT build sources. '
                      'Missing files or zero matches do not establish lack of platform support.')
    return report


def inspect_rms(airs, pb2=None):
    from qwen35_moe_dflash.target.air_weights import weight_schema
    from qwen35_moe_dflash.draft.incremental.torch_export import inspect_air
    schema = weight_schema(pb2)
    reports = []
    for air in airs:
        observation = inspect_air(air, schema)
        groups = {}
        for node in observation['rms']:
            signature = {k: node[k] for k in ('type', 'inputs', 'outputs', 'epsilon')}
            key = json.dumps(signature, sort_keys=True)
            group = groups.setdefault(key, dict(**signature, count=0, examples=[]))
            group['count'] += 1
            if len(group['examples']) < 2:
                group['examples'].append(dict(name=node['name'], input=node['input']))
        reports.append(dict(air=str(air), rms_count=len(observation['rms']),
            groups=list(groups.values()), input_count=observation['input_count'],
            output_counts=observation['output_counts']))
    return dict(status='INSPECTED', schema_source=pb2 or 'specs/torchair-ge-ir.pb',
                scope='SOURCE_AIR_RMS_DESCRIPTORS_ONLY_NOT_COMPILED_KERNEL_OR_FP32_PROOF',
                model_compilation='NOT_RUN', device_execution='NOT_RUN', models=reports)


def compilation_diagnostics(folder):
    """Read existing compiler errors; never retry compilation or infer a cause."""
    from qwen35_moe_dflash.common.tool_logs import compilation_diagnostics as read_diagnostics
    return read_diagnostics(Path(folder) / 'omc')


def inspect_compilation(directory):
    root = Path(directory)
    report = json.loads((root / 'result.json').read_text())
    if report.get('artifact_kind') != KIND:
        raise ValueError('expected an existing Draft vector probe directory')
    cases = []
    for row in report['cases']:
        name = row['case']
        if name not in CASES:
            raise ValueError('unknown probe case in saved report')
        if row.get('status') != 'FAILED':
            continue
        cases.append(dict(case=name, error=row.get('error'),
                          compilation_diagnostics=compilation_diagnostics(root / name)))
    return dict(status='INSPECTED', source_batch_status=report.get('status'), cases=cases,
                scope='EXISTING_COMPILER_REPORTS_AND_LOGS_ONLY_ROOT_CAUSE_NOT_CLASSIFIED',
                model_compilation='NOT_RUN', device_execution='NOT_RUN')


def _run(args, root):
    from qwen35_moe_dflash.graph.fused.emitter import emit_cpp
    from qwen35_moe_dflash.graph.export import _logged
    from qwen35_moe_dflash.toolchain.existing_air import compile_existing_air
    if not args.plan_only:
        if not all((args.ge_root, args.ddk_root, args.ddk_id)):
            raise ValueError('set CANN_ROOT, CPRO_DDK_ROOT, CPRO_DDK_ID or pass the corresponding options')
        ge, ddk = Path(args.ge_root).resolve(), Path(args.ddk_root).resolve()
        schema = ddk / 'tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so'
        for required in (ge / 'include/graph/graph.h', ge / 'lib64/libgraph.so', schema):
            if not required.is_file():
                raise FileNotFoundError(required)
        if any(root == p or p in root.parents or root in p.parents for p in (ge, ddk)):
            raise ValueError('probe output must not contain or be inside an SDK')
    if root.exists():
        if not args.replace or root.is_symlink():
            raise FileExistsError('probe directory exists; use --replace after its completed run')
        old = json.loads((root / 'result.json').read_text())
        if old.get('artifact_kind') != KIND or old.get('status') not in {'COMPILED', 'PLANNED', 'FAILED', 'PARTIAL'}:
            raise ValueError('existing output is foreign or still running; retained')
        shutil.rmtree(root)
    root.mkdir(parents=True)
    result = {'artifact_kind': KIND, 'status': 'RUNNING', 'stage': 'prepare',
        'device_execution': 'NOT_RUN', 'numerical_validation': 'NOT_RUN',
        'model_compilation': 'NOT_RUN', 'full_draft_changed': False, 'cases': [],
        'precision_request': 'SOURCE_FP32_GRAPH_WITH_DEFAULT_OMG_ARITHMETIC_POLICY_NO_CPU_CUSTOMIZATION',
        'weight_data_type': args.weight_data_type or 'OMG_DEFAULT',
        'schema_source': SCHEMA, 'platform': 'kirinx90'}
    write_json(root / 'result.json', result)
    try:
        for case in args.cases:
            folder = root / case
            folder.mkdir()
            air_dir = folder / 'air'
            air_dir.mkdir()
            graph, inputs, golden = graph_case(case)
            source = folder / 'probe.cpp'
            source.write_text(emit_cpp(graph, fold_constant_arithmetic=False))
            fixture = folder / 'fixture'
            fixture.mkdir()
            for prefix, arrays in (('input', inputs), ('golden', golden)):
                for index, array in enumerate(arrays):
                    array.tofile(fixture / f'{prefix}_{index}.bin')
            row = {'case': case, 'status': 'PLANNED', 'model_compilation': 'NOT_RUN', 'graph': graph.manifest(),
                   'reference': ('NUMPY_FP32_ADD_WITH_CANCELLATION_ROWS_NOT_DEVICE_VALIDATION'
                       if case.startswith('residual_') else 'NUMPY_FP32_ROW_REDUCTION_NOT_DEVICE_VALIDATION')}
            if case == 'residual_addcustom_fp32':
                row['schema_source'] = 'USER_OBSERVED_SELECTED_DDK_AddCustom_REGISTRATION_x_y_TO_z'
                row['implementation_request'] = 'AddCustom_FP32_ND_DEFAULT_DDK_SELECTION_NO_FORCED_IMPLEMENTATION'
            result['cases'].append(row)
            if args.plan_only:
                continue
            print('draft-vector: compiling ' + case, flush=True)
            try:
                env = dict(os.environ)
                library = ge / 'lib64'
                env['LD_LIBRARY_PATH'] = str(library) + os.pathsep + env.get('LD_LIBRARY_PATH', '')
                exporter, air = folder / 'exporter', air_dir / 'probe.air'
                command = [args.cxx, '-std=c++17', f'-D_GLIBCXX_USE_CXX11_ABI={args.cxx11_abi}',
                    '-I', str(ge / 'include'), str(source), '-L', str(library),
                    f'-Wl,-rpath,{library}', '-lgraph']
                if (library / 'libgraph_base.so').is_file():
                    command.append('-lgraph_base')
                command += ['-o', str(exporter)]
                write_json(folder / 'build-command.json', {'argv': command})
                _logged(command, folder=folder, env=env, log=folder / 'build.log', timeout=args.timeout_seconds)
                _logged([str(exporter), str(air)], folder=folder, env=env,
                        log=folder / 'export.log', timeout=args.timeout_seconds)
                compiled = compile_existing_air(air, folder / 'omc', ddk, ddk_id=args.ddk_id,
                    timeout_seconds=args.timeout_seconds, use_origin_format=True,
                    weight_data_type=args.weight_data_type,
                    tbe_diagnosis=args.tbe_diagnosis, save_weights_as_external_data=False)
                row['model_compilation'] = compiled['model_compilation']
                if compiled['status'] != 'COMPILED':
                    row['compilation_diagnostics'] = compilation_diagnostics(folder)
                    raise RuntimeError(str(compiled.get('error', compiled['status'])))
                row['omc'] = str(Path(case) / 'omc' / compiled['omc']['path'])
                placement = inspect_placement(root / row['omc'], schema)
                write_json(folder / 'placement.json', placement)
                row.update(status='COMPILED', placement=placement['status'], graphop_counts=placement['graphop_counts'],
                           precision_boundaries=placement['precision_boundaries'])
            except (OSError, ValueError, KeyError, RuntimeError) as error:
                row.update(status='FAILED', error=f'{type(error).__name__}: {error}')
            write_json(root / 'result.json', result)
        statuses = {row['status'] for row in result['cases']}
        compilation = {row['model_compilation'] for row in result['cases']}
        result.update(stage='complete', status=('PLANNED' if args.plan_only else
                      'COMPILED' if statuses == {'COMPILED'} else 'FAILED' if statuses == {'FAILED'} else 'PARTIAL'),
            model_compilation=('NOT_RUN' if compilation == {'NOT_RUN'} else
                               'PASSED' if compilation == {'PASSED'} else
                               'FAILED' if compilation == {'FAILED'} else 'PARTIAL'),
            next=('No SDK compilation requested; inspect the planned source and fixtures.' if args.plan_only else
                  'Inspect placement.json for FP32 descriptors and CPU/NPU owners; device precision and performance are still required before Draft integration.'
                  if statuses == {'COMPILED'} else
                  'Inspect failed cases and compilation_diagnostics; failed compilation has no validated placement. Do not run failed cases on device.'))
    except Exception as error:
        result.update(status='FAILED', error=f'{type(error).__name__}: {error}')
    finally:
        write_json(root / 'result.json', result)
    return result


def run(args):
    import fcntl
    from qwen35_moe_dflash.common.artifacts import output_path
    if Path(args.output).is_symlink():
        raise ValueError('probe output must not be a symlink')
    root = output_path(args.output)
    root.parent.mkdir(parents=True, exist_ok=True)
    # Keep this sibling lock inode stable while replacing the owned test slot.
    with (root.parent / ('.' + root.name + '.lock')).open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('this probe slot is still running; retained') from None
        return _run(args, root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    parser.add_argument('--inspect-ddk', metavar='COMPILED_PROBES',
                        help='read saved compiler environment and selected DDK AddCustom registration; no writes or compilation')
    parser.add_argument('--inspect-rms', nargs='+', metavar='AIR',
                        help='read normalized Target/PyTorch AIR RMS descriptors without compiling or executing')
    parser.add_argument('--inspect-compile', metavar='COMPILED_PROBES',
                        help='read existing failed compiler reports and error excerpts; no writes or retries')
    parser.add_argument('--torchair-pb2', help='optional actual TorchAir schema for --inspect-rms')
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    parser.add_argument('--plan-only', action='store_true', help='write source and fixtures without invoking any SDK')
    parser.add_argument('--replace', action='store_true', help='replace this completed probe slot; never shared models')
    parser.add_argument('--ge-root', default=os.environ.get('CANN_ROOT'))
    parser.add_argument('--ddk-root', default=os.environ.get('CPRO_DDK_ROOT'))
    parser.add_argument('--ddk-id', default=os.environ.get('CPRO_DDK_ID'))
    parser.add_argument('--cxx', default='c++')
    parser.add_argument('--cxx11-abi', type=int, choices=(0, 1), default=0)
    parser.add_argument('--timeout-seconds', type=int, default=600)
    parser.add_argument('--tbe-diagnosis')
    parser.add_argument('--weight-data-type', choices=('FP16', 'FP32'),
                        help='Explicit OMG weight conversion; omitted uses its default. Does not set arithmetic precision.')
    args = parser.parse_args(argv)
    if args.timeout_seconds <= 0 or len(args.cases) != len(set(args.cases)):
        parser.error('timeout must be positive and cases must be distinct')
    if sum(bool(value) for value in (args.inspect_ddk, args.inspect_rms, args.inspect_compile)) > 1:
        parser.error('select one read-only inspection mode')
    if args.torchair_pb2 and not args.inspect_rms:
        parser.error('--torchair-pb2 requires --inspect-rms')
    if args.inspect_ddk or args.inspect_rms or args.inspect_compile:
        if args.output or args.replace or args.plan_only:
            parser.error('inspection cannot be combined with --output, --replace or --plan-only')
    elif not args.output:
        parser.error('--output is required for compilation or --plan-only')
    try:
        if args.inspect_compile:
            print(json.dumps(inspect_compilation(args.inspect_compile), ensure_ascii=False, indent=2))
            return 0
        if args.inspect_rms:
            print(json.dumps(inspect_rms(args.inspect_rms, args.torchair_pb2), ensure_ascii=False, indent=2))
            return 0
        if args.inspect_ddk:
            print(json.dumps(inspect_ddk(args.inspect_ddk, args.ddk_root), ensure_ascii=False, indent=2))
            return 0
        result = run(args)
        shown = {key: value for key, value in result.items() if key != 'cases'}
        shown['cases'] = [{key: value for key, value in row.items() if key != 'graph'} for row in result['cases']]
        print(json.dumps(shown, ensure_ascii=False, allow_nan=False))
        return 0 if result['status'] in {'COMPILED', 'PLANNED'} else 1
    except (OSError, ValueError, RuntimeError) as error:
        print(json.dumps({'status': 'FAILED', 'error': f'{type(error).__name__}: {error}'}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
