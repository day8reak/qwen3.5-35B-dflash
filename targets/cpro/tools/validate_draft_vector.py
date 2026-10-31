#!/usr/bin/env python3
"""Validate existing X90 FP32 probe OMCs; never compile or replace Draft models."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import sys
import tarfile
import uuid

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))
from probe_draft_vector import CASES, KIND as COMPILE_KIND, graph_case
from qwen35_moe_dflash.common.artifacts import output_path
from qwen35_moe_dflash.execution.runtime_config import read_json
from qwen35_moe_dflash.execution.run_layout import _save, reuse_test_output
from qwen35_moe_dflash.execution.serial_transport import _environment, _build_payloads
from qwen35_moe_dflash.execution.hdc_run import Hdc, _file_record
from qwen35_moe_dflash.execution.lock import _acquire_global, _release_global, _space
from qwen35_moe_dflash.draft.profile_board import GLOBAL_LOCK, _guard, _identity_guard
from qwen35_moe_dflash.draft.board import _receive
from qwen35_moe_dflash.common.hashing import sha256_file

KIND = 'qwen35-draft-vector-validation'
PREFIX = '/data/local/tmp/qwen35_vector_'
q = shlex.quote
WARMUP, REPETITIONS = 3, 10
ATOL, RTOL = 1e-5, 1e-4


def member(root, name):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('probe member must be relative')
    path = root / relative
    if root.resolve() not in path.resolve().parents or path.is_symlink() or not path.is_file():
        raise ValueError('probe member is missing or outside its directory: ' + name)
    return path


def bind_io(document, declared, enums):
    from qwen35_moe_dflash.draft.compile_precision import _observed_desc
    from qwen35_moe_dflash.draft.direct_abi import _ordered_edges
    graphs = document.get('graph', [])
    if len(graphs) != 1:
        raise ValueError('probe needs one compiled main graph')
    nodes = graphs[0]['op']
    operations = {n['name']: n for n in nodes}
    if len(operations) != len(nodes):
        raise ValueError('duplicate compiled node name')
    expected = {r['name']: (i, r) for i, r in enumerate(declared['inputs'])}
    inputs = []
    for node in nodes:
        if node.get('type') != 'Data':
            continue
        source, spec = expected[node['name']]
        indices = [a['value']['i'] for a in node.get('attr', []) if a['key'] == 'index']
        if len(indices) != 1 or type(indices[0]) is not int:
            raise ValueError('missing or ambiguous compiled input index')
        descs = node['output_desc']
        if len(descs) != 1 or any(_observed_desc(descs[0], enums)[k] != spec[k] for k in ('shape', 'dtype')):
            raise ValueError('compiled FP32 input descriptor changed')
        inputs.append(dict(spec, index=indices[0], fixture_index=source, bytes=math.prod(spec['shape'])*4))
    if (len(inputs) != len(expected) or {r['name'] for r in inputs} != set(expected)
            or sorted(r['index'] for r in inputs) != list(range(len(expected)))):
        raise ValueError('compiled inputs are not a complete indexed mapping')
    net = [n for n in nodes if n.get('type') == 'NetOutput']
    if len(net) != 1:
        raise ValueError('missing or ambiguous compiled NetOutput')
    descriptors = [_observed_desc(d, enums) for d in net[0]['input_desc']]
    connections = _ordered_edges(net[0], operations, len(descriptors), enums, descriptors)
    outputs = []
    for index, desc in enumerate(descriptors):
        # These small probes have unique output shapes (RMS y versus rstd).
        # Never extend shape-only matching to equal-shaped model KV outputs.
        matches = [(i, r) for i, r in enumerate(declared['outputs'])
                   if all(desc[k] == r[k] for k in ('shape', 'dtype'))]
        if len(matches) != 1:
            raise ValueError('compiled output descriptor changed or is ambiguous')
        source, spec = matches[0]
        outputs.append(dict(spec, index=index, fixture_index=source, bytes=math.prod(spec['shape'])*4,
                            connection=connections[index]))
    if sorted(r['fixture_index'] for r in outputs) != list(range(len(declared['outputs']))):
        raise ValueError('compiled outputs do not cover every probe output once')
    return dict(inputs=sorted(inputs, key=lambda r: r['index']), outputs=outputs)


def prepare(compiled, out, ddk, ddk_id):
    from qwen35_moe_dflash.toolchain.omc_container import read_container, read_partition
    from qwen35_moe_dflash.toolchain.omc_wire import load_schema
    from qwen35_moe_dflash.draft.ir_inspect import _enum_fields
    root = Path(compiled).resolve()
    report = read_json(root/'result.json')
    if report.get('artifact_kind') != COMPILE_KIND or report.get('status') != 'COMPILED':
        raise ValueError('expected completed FP32 probe compilation')
    rows = report['cases']
    if len(rows) != len(CASES) or {r['case'] for r in rows} != set(CASES):
        raise ValueError('expected all five distinct compiled probes')
    schema = load_schema(Path(ddk)/'tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so')
    enums = _enum_fields(schema)
    cases = []
    for row in rows:
        name = row['case']
        compilation = read_json(member(root, f'{name}/omc/report.json'))
        if (compilation.get('status') != 'COMPILED' or compilation.get('platform') != 'kirinx90'
                or compilation.get('ddk_id') != ddk_id):
            raise ValueError('probe compiler DDK differs from runtime configuration')
        graph, inputs, golden = graph_case(name)
        if row['status'] != 'COMPILED' or row['graph'] != graph.manifest():
            raise ValueError('probe recipe changed; recompile this probe before execution')
        model = member(root, row['omc'])
        if model.stat().st_size > 64 << 20:
            raise ValueError('small probe model exceeds 64 MiB')
        parts = read_container(model)['partitions']
        main = [p for p in parts if p['type'] == 0]
        if len(main) != 1 or any(model.parent.glob('*.weight')):
            raise ValueError('probe needs one embedded model without external weight sidecars')
        io = bind_io(schema.decode(read_partition(model, main[0]), 'ModelDef'), row['graph'], enums)
        folder = out/'fixture'/name
        folder.mkdir(parents=True)
        for prefix, arrays in (('input', inputs), ('golden', golden)):
            for i, array in enumerate(arrays):
                path = member(root, f'{name}/fixture/{prefix}_{i}.bin')
                if path.stat().st_size != array.nbytes or path.read_bytes() != array.tobytes():
                    raise ValueError('compiled probe fixture differs: ' + str(path))
        plan = ['QWEN35_TENSOR_PROBE 1', f'iterations {WARMUP} {REPETITIONS}']
        for direction, prefix in (('inputs', 'input'), ('outputs', 'golden')):
            for spec in io[direction]:
                plan.append(' '.join((direction[:-1], str(spec['index']), spec['name'], 'float32',
                                      ','.join(map(str, spec['shape'])))))
                destination = folder/f'{prefix}_{spec["index"]}.bin'
                shutil.copyfile(root/f'{name}/fixture/{prefix}_{spec["fixture_index"]}.bin', destination)
                spec['fixture_sha256'] = sha256_file(destination)
        (folder/'plan.txt').write_text('\n'.join(plan)+'\n')
        cases.append(dict(case=name, **io, model=_file_record(model, name+'/model.omc'),
                          graphop_counts=row['graphop_counts']))
    result = dict(cases=cases, warmup=WARMUP, repetitions=REPETITIONS, atol=ATOL, rtol=RTOL,
                  scope='FP32_REGIONS_ONLY_NOT_FULL_DRAFT_VALIDATION', ddk_id=ddk_id, schema=schema.evidence)
    _save(out/'fixture/manifest.json', result)
    return result


def compare(fixture, native, *, allow_host_fake=False):
    import numpy as np
    meta = read_json(fixture/'manifest.json')
    cases = []
    for case in meta['cases']:
        name = case['case']
        result = read_json(native/name/'result.json')
        expected_kind = 'host-fake-hiai' if allow_host_fake else 'device-hiai'
        timings = result.get('api_us', [])
        if (result.get('status') != 'EXECUTED' or result.get('evidence_kind') != expected_kind
                or result.get('warmup') != WARMUP or result.get('repetitions') != REPETITIONS
                or len(timings) != REPETITIONS or any(type(t) is not int or t < 0 for t in timings)
                or result.get('timing_scope') != 'SYNCHRONOUS_API_WALL_NOT_NPU_ONLY'):
            raise ValueError('native report does not establish ten device calls after warmup')
        total = sum(r['bytes'] for r in case['outputs'])
        tensors, stable, first = [], True, None
        for iteration in range(REPETITIONS):
            snapshot = member(native, f'{name}/{iteration}.bin')
            if snapshot.stat().st_size != total:
                raise ValueError('probe snapshot has incorrect byte extent')
            raw = snapshot.read_bytes()
            if first is None:
                first = raw
            stable = stable and raw == first
            offset = 0
            for spec in case['outputs']:
                actual = np.frombuffer(raw[offset:offset+spec['bytes']], dtype='<f4').astype(np.float64)
                offset += spec['bytes']
                expected = np.fromfile(fixture/name/f'golden_{spec["index"]}.bin', dtype='<f4').astype(np.float64)
                if expected.size*4 != spec['bytes'] or not np.isfinite(expected).all():
                    raise ValueError('invalid retained reference')
                finite = bool(np.isfinite(actual).all())
                delta = actual-expected
                passed = finite and bool(np.all(np.abs(delta) <= meta['atol']+meta['rtol']*np.abs(expected)))
                tensors.append(dict(iteration=iteration, name=spec['name'], passed=passed, finite=finite,
                    max_abs_error=float(np.max(np.abs(delta))) if finite else None,
                    relative_l2_error=float(np.linalg.norm(delta)/max(float(np.linalg.norm(expected)), 1e-30)) if finite else None))
        passed = all(r['passed'] for r in tensors)
        cases.append(dict(case=name, passed=passed, repeat_bitwise_stable=stable,
            tensors=tensors, graphop_counts=case['graphop_counts'], api_us=timings,
            mean_ms=sum(timings)/REPETITIONS/1000, min_ms=min(timings)/1000, max_ms=max(timings)/1000))
    passed = all(r['passed'] for r in cases)
    return dict(status='PASSED' if passed else 'NUMERICAL_FAILED', numerical_validation='PASSED' if passed else 'FAILED',
        cases=cases, warmup=WARMUP, repetitions=REPETITIONS, atol=meta['atol'], rtol=meta['rtol'],
        repeat_bitwise_stable=all(r['repeat_bitwise_stable'] for r in cases),
        timing_scope='SYNCHRONOUS_API_WALL_NOT_NPU_ONLY; SNAPSHOTS_BETWEEN_CALLS_EXCLUDED',
        full_draft_validation='NOT_RUN', internal_instruction_precision='NOT_PROVEN_BY_OUTPUT_MATCH')


def supervisor(meta, board, token, env, search):
    lines = ['#!/bin/sh', 'cd '+q(str(board))+' || exit 91', _guard(board, token)+' || exit 92',
             _guard(GLOBAL_LOCK, token)+' || exit 93', 'failed=0', 'for batch in once; do']
    for case in meta['cases']:
        name = case['case']
        argv = ['timeout', '-k', '5', '600', 'env', 'LD_LIBRARY_PATH='+':'.join([str(board/'lib'), *search]),
                str(board/'tensor-probe'), str(board/name/'model.omc'), env['runtime_version'],
                str(board/name), str(board/'outputs'/name)]
        lines += [shlex.join(argv)+' > '+name+'.stdout 2> '+name+'.stderr', 'rc=$?',
                  'printf "%s\\n" "$rc" > '+name+'.exitcode',
                  'if test "$rc" -ne 0; then failed=1; break; fi']
    # Do not start another SDK process after a failed/timed-out case.
    return '\n'.join([*lines, 'done', 'printf "%s\\n" "$failed" > native.exitcode',
        'tar -cf outputs.tar -C outputs .', 'printf "%s\\n" "$?" > archive.exitcode',
        'printf "%s\\n" '+q(token)+' > complete.tmp', 'mv complete.tmp complete', ''])


def start(args, out):
    config = Path(args.config).resolve(); source = Path(args.compiled).resolve()
    for path in (config, source):
        if path == out or out in path.parents or source in out.parents:
            raise ValueError('execution slot must be separate from config and compiled probes')
    reuse_test_output(out, config=config)
    out.mkdir(parents=True)
    state = dict(artifact_kind=KIND, status='FAILED', stage='prepare', device_execution='NOT_RUN',
                 model_compilation='NOT_RUN', numerical_validation='NOT_RUN', lock_state='NO_KNOWN_OWNED_LOCK')
    hdc = None; held = uncertain = launched = False
    try:
        cfg = read_json(config)
        env = read_json(config.parent/cfg['assets'])['environment']
        with (source.parent/('.'+source.name+'.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            meta = prepare(source, out, args.ddk_root, env['ddk_id'])
        state.update(status='PREPARED', stage='prepared')
        if not args.apply:
            return state
        native = (config.parent/cfg.get('native_directory', 'native')).resolve()
        binary = native/'build/qwen35_tensor_probe'
        built = read_json(native/'result.json')
        if not binary.is_file() or built.get('diagnostic_binaries', {}).get(binary.name) != {
                'bytes': binary.stat().st_size, 'sha256': sha256_file(binary)}:
            raise ValueError('rebuild runtime/native with build_hiai.py --replace for tensor probes')
        uploads, search, _ = _build_payloads(out, 600, resident='resident-mixed', native_directory=native)
        uploads = [r for r in uploads if r['remote'].startswith('lib/')]
        uploads.append(_file_record(binary, 'tensor-probe'))
        for case in meta['cases']:
            uploads.append(case['model'])
            for path in sorted((out/'fixture'/case['case']).iterdir()):
                if path.name.startswith('input_') or path.name == 'plan.txt':
                    uploads.append(_file_record(path, case['case']+'/'+path.name))
        exe, serial, server = _environment(); token = uuid.uuid4().hex
        board = PurePosixPath(PREFIX+token)
        plan = dict(artifact_kind=KIND, token=token, board_attempt=str(board), serial=serial,
                    hdc_server=server, runtime=env, fixture_sha256=sha256_file(out/'fixture/manifest.json'))
        _save(out/'request.json', plan)
        state.update(board_attempt=str(board), stage='board-preflight')
        hdc = Hdc(exe, serial, server, out, 180)
        uncertain = True
        try:
            _acquire_global(hdc, GLOBAL_LOCK, token, 0)
        except ValueError as error:
            if 'BOARD_MODEL_IO_BUSY:' in str(error):
                uncertain = False
            raise
        held = True; uncertain = False
        _, firmware = hdc.shell('param get const.ohos.fullname', 'vector-firmware')
        if firmware.strip() != env['firmware_id'].strip():
            raise ValueError('board firmware differs from runtime configuration')
        _, space = hdc.shell('df -Pk /data/local/tmp', 'vector-space')
        output_bytes = REPETITIONS*sum(s['bytes'] for c in meta['cases'] for s in c['outputs'])
        if _space(space) < sum(r['bytes'] for r in uploads)+2*output_bytes+(64 << 20):
            raise OSError('insufficient board space for small probe request')
        hdc.shell('mkdir '+q(str(board))+' && printf "%s\\n" '+q(token)+' > '+q(str(board/'owner'))
                  +' && mkdir '+shlex.join(str(board/p) for p in ['lib', 'outputs', *CASES]), 'vector-directory')
        for row in uploads:
            hdc.send(row['local'], board/row['remote'])
            hdc.shell(_identity_guard(board/row['remote'], row), 'vector-upload')
        hdc.shell('chmod +x '+q(str(board/'tensor-probe')), 'vector-executable')
        (out/'launch.sh').write_text(supervisor(meta, board, token, env, search))
        hdc.send(out/'launch.sh', board/'launch.sh')
        hdc.shell(_identity_guard(board/'launch.sh', _file_record(out/'launch.sh', 'launch.sh')), 'vector-launcher')
        launched = True
        state.update(launch_requested=True, device_execution='RUNNING_OR_COMPLETION_PENDING')
        _save(out/'result.json', state)
        hdc.shell('cd '+q(str(board))+' && '+_guard(board, token)
                  +' && { nohup sh launch.sh < /dev/null > supervisor.stdout 2> supervisor.stderr & }', 'vector-launch')
        state.update(status='VECTOR_VALIDATION_STARTED', stage='running-detached', next='Recover the same execution; no relaunch.')
    except Exception as error:
        state.update(status='FAILED', error=f'{type(error).__name__}: {error}')
    finally:
        if hdc and held and not launched and not uncertain:
            try:
                _release_global(hdc, GLOBAL_LOCK, token); held = False
            except Exception as error:
                state['lock_release_error'] = str(error)
        state['lock_state'] = 'RETAINED_OR_UNCERTAIN' if held or uncertain else 'NO_KNOWN_OWNED_LOCK'
        _save(out/'result.json', state)
    return state


def extract(archive, destination, meta):
    expected = {}
    for case in meta['cases']:
        expected[case['case']+'/result.json'] = None
        for i in range(REPETITIONS):
            expected[f'{case["case"]}/{i}.bin'] = sum(r['bytes'] for r in case['outputs'])
    with tarfile.open(archive) as stream:
        members = [m for m in stream.getmembers() if not m.isdir()]
        names = [m.name.removeprefix('./') for m in members]
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ValueError('archive does not contain exactly five complete probe results')
        for m, name in zip(members, names):
            size = expected[name]
            if not m.isfile() or (size is None and not 0 < m.size < 65536) or (size is not None and m.size != size):
                raise ValueError('invalid archive member extent or type')
            path = destination/name; path.parent.mkdir(parents=True, exist_ok=True)
            with stream.extractfile(m) as src, path.open('xb') as dst:
                shutil.copyfileobj(src, dst)


def recover(root, out):
    root = Path(root).resolve()
    if root == out or root in out.parents or out in root.parents:
        raise ValueError('recovery needs a separate output slot')
    reuse_test_output(out, recovery_source=root); out.mkdir(parents=True)
    state = dict(artifact_kind=KIND, status='FAILED', stage='recover', model_compilation='NOT_RUN',
        numerical_validation='NOT_RUN', recovery_source=str(root), recovery_model_execution='FORBIDDEN',
        lock_state='RETAINED_OR_UNCERTAIN')
    complete = False; hdc = plan = None
    try:
        plan = read_json(root/'request.json'); token = plan['token']
        if plan['artifact_kind'] != KIND or not re.fullmatch('[0-9a-f]{32}', token) or plan['board_attempt'] != PREFIX+token:
            raise ValueError('invalid vector request')
        board = PurePosixPath(plan['board_attempt']); state['board_attempt'] = str(board)
        exe, serial, server = _environment()
        if (serial, server) != (plan['serial'], plan['hdc_server']):
            raise ValueError('recover requires the same board')
        hdc = Hdc(exe, serial, server, out, 180)
        _, done = hdc.shell(_guard(board, token)+' && if test -f '+q(str(board/'complete'))
            +'; then cat '+q(str(board/'complete'))+'; else echo PENDING; fi', 'vector-completion')
        if done == 'PENDING':
            state.update(status='VECTOR_VALIDATION_PENDING', stage='running'); return state
        if done != token:
            raise ValueError('completion belongs to another request')
        complete = True; state['device_execution'] = 'FAILED_OUTPUTS_RETRIEVED'
        for name in ('native.exitcode', 'archive.exitcode', 'supervisor.stdout', 'supervisor.stderr'):
            _receive(hdc, board, out, name)
        for name in CASES:
            for suffix in ('stdout', 'stderr', 'exitcode'):
                _receive(hdc, board, out, name+'.'+suffix)
        if (out/'native.exitcode').read_text().strip() != '0' or (out/'archive.exitcode').read_text().strip() != '0':
            raise ValueError('probe execution/archive failed; inspect per-case stderr and exitcode')
        fixture = root/'fixture'
        if sha256_file(fixture/'manifest.json') != plan['fixture_sha256']:
            raise ValueError('retained fixture manifest changed')
        meta = read_json(fixture/'manifest.json')
        for case in meta['cases']:
            for direction, prefix in (('inputs', 'input'), ('outputs', 'golden')):
                for spec in case[direction]:
                    path = fixture/case['case']/f'{prefix}_{spec["index"]}.bin'
                    if path.stat().st_size != spec['bytes'] or sha256_file(path) != spec['fixture_sha256']:
                        raise ValueError('retained input/reference changed after launch')
        maximum = 2*REPETITIONS*sum(r['bytes'] for c in meta['cases'] for r in c['outputs'])+(2 << 20)
        _, size = hdc.shell('wc -c < '+q(str(board/'outputs.tar')), 'vector-output-size')
        if not size.isdigit() or not 0 < int(size) < maximum:
            raise ValueError('unexpected output archive size')
        if _receive(hdc, board, out, 'outputs.tar') is None:
            raise ValueError('missing output archive')
        extract(out/'outputs.tar', out/'native', meta)
        state.update(compare(fixture, out/'native'), stage='complete', device_execution='EXECUTED_OUTPUTS_RETRIEVED')
    except Exception as error:
        state.update(status='FAILED', error=f'{type(error).__name__}: {error}')
    finally:
        if complete and hdc and plan:
            try:
                _, owner = hdc.shell('if test ! -e '+q(str(GLOBAL_LOCK))+'; then echo ABSENT; elif '
                    +_guard(GLOBAL_LOCK, plan['token'])+'; then echo OWNED; else echo OTHER; fi', 'vector-lock-owner')
                if owner == 'OWNED':
                    _release_global(hdc, GLOBAL_LOCK, plan['token'])
                elif owner not in ('ABSENT', 'OTHER'):
                    raise ValueError('unknown lock owner')
                state['lock_state'] = 'NO_OWNED_LOCK_AFTER_KNOWN_COMPLETION'
            except Exception as error:
                state['lock_release_error'] = str(error)
        _save(out/'result.json', state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiled')
    parser.add_argument('--config')
    parser.add_argument('--ddk-root', default=os.environ.get('CPRO_DDK_ROOT'))
    parser.add_argument('--recover')
    parser.add_argument('--output', required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if args.recover and (args.apply or args.config or args.compiled):
        parser.error('recovery only accepts the original execution and separate output')
    if not args.recover and not all((args.config, args.compiled, args.ddk_root)):
        parser.error('start requires --config, --compiled and CPRO_DDK_ROOT/--ddk-root')
    if Path(args.output).is_symlink():
        raise ValueError('output must not be a symlink')
    out = output_path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with (out.parent/('.'+out.name+'.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = recover(args.recover, out) if args.recover else start(args, out)
    shown = {k: v for k, v in result.items() if k != 'cases'}
    if 'cases' in result:
        shown['cases'] = [{k: v for k, v in c.items() if k not in ('tensors', 'api_us')} for c in result['cases']]
    print(json.dumps(shown, ensure_ascii=False, allow_nan=False))
    return 1 if result['status'] in ('FAILED', 'NUMERICAL_FAILED') else 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(json.dumps({'status': 'FAILED', 'error': f'{type(error).__name__}: {error}'}))
        raise SystemExit(1)
