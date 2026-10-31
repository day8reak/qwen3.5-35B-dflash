#!/usr/bin/env python3
"""Compile small FP32 Draft regions on the selected X90 DDK; no board access."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))

KIND = 'qwen35-draft-vector-probe'
SCHEMA = ('https://github.com/Ascend/torchair/blob/'
          'e1118e8b0fc1b95190ab53b732c0872989550e20/'
          'python/torchair/_ge_concrete_graph/auto_generated_ge_raw_ops.py')
CASES = ('rms_primitive_hidden', 'rms_builtin_hidden', 'rms_builtin_heads',
         'softmax_fp32', 'residual_fp32')


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
        g.outputs = [g.op('add', x, y, shape=shape, dtype='float32', name='sum')]
    return g.validate(), inputs, golden


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
        'status': 'NPU_GRAPH_PRESENT' if counts['NPUCL'] else 'NO_NPU_GRAPH',
        'internal_precision': 'NOT_VERIFIED_BY_GRAPH_BOUNDARY_DTYPES',
        'scope': 'Compiler task ownership and boundary descriptors only; no device accuracy or latency claim'}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


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
        'precision_request': 'FP32_INPUT_OUTPUT_AND_ARITHMETIC_NO_CPU_CUSTOMIZATION_NO_FP16_WEIGHT_OVERRIDE',
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
                   'reference': 'NUMPY_FP32_ROW_REDUCTION_NOT_DEVICE_VALIDATION'}
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
                    tbe_diagnosis=args.tbe_diagnosis, save_weights_as_external_data=False)
                row['model_compilation'] = compiled['model_compilation']
                if compiled['status'] != 'COMPILED':
                    raise RuntimeError(str(compiled.get('error', compiled['status'])))
                row['omc'] = str(Path(case) / 'omc' / compiled['omc']['path'])
                placement = inspect_placement(root / row['omc'], schema)
                write_json(folder / 'placement.json', placement)
                row.update(status='COMPILED', placement=placement['status'], graphop_counts=placement['graphop_counts'])
            except (OSError, ValueError, KeyError, RuntimeError) as error:
                row.update(status='FAILED', error=f'{type(error).__name__}: {error}')
            write_json(root / 'result.json', result)
        statuses = {row['status'] for row in result['cases']}
        compilation = {row['model_compilation'] for row in result['cases']}
        result.update(stage='complete', status=('PLANNED' if args.plan_only else
                      'COMPILED' if statuses == {'COMPILED'} else 'FAILED' if statuses == {'FAILED'} else 'PARTIAL'),
            model_compilation=('NOT_RUN' if compilation == {'NOT_RUN'} else
                               'PASSED' if compilation == {'PASSED'} else 'FAILED_OR_PARTIAL'),
            next='Inspect placement.json for FP32 descriptors and CPU/NPU owners; device precision and performance are still required before Draft integration.')
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
    parser.add_argument('--output', required=True)
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
    args = parser.parse_args(argv)
    if args.timeout_seconds <= 0 or len(args.cases) != len(set(args.cases)):
        parser.error('timeout must be positive and cases must be distinct')
    try:
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
