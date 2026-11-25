"""Compile, execute, resume and compare whole-decoder OMCs on the OMG host."""
import json
from pathlib import Path
import tempfile
import shutil

import numpy as np

from decoder4_original import compile_log_text, compile_omg
from decoder4_prefill import pull_air, ssh_file
from l0_silu_omcs import write_matrices
from parse_model_json import GE_PROTO, inspect, read_omc
from qwen35_moe_dflash.execution.serial_transport import _environment
from qwen35_moe_dflash.target.layer_probe.common import checked, identity, metrics, read_array, save_json, sha
from qwen35_moe_dflash.target.prefill_t64.attention_step_v2 import run_v2
from qwen35_moe_dflash.target.prefill_t64.decoder_cases import case_name, comparison_view, groups, preserved_cache
from qwen35_moe_dflash.target.prefill_t64.decoder_four_boundary import view


def read(path):
    return json.loads(path.read_bytes())


def transfer_size(meta, dest):
    total = sum(row['bytes'] for row in meta['files'])
    free = shutil.disk_usage(dest).free
    print(f'Package {total / 2**30:.2f} GiB, available {free / 2**30:.2f} GiB before OMG', flush=True)
    if total > free:
        raise OSError(f'{dest}: insufficient space for the {total:,}-byte transfer')


def transfer(client, remote, dest):
    dest.mkdir()
    remote = Path(remote)
    ssh_file(client, remote/'manifest.json', dest/'manifest.json')
    meta = read(dest/'manifest.json')
    transfer_size(meta, dest)
    for row in meta['files']:
        rel = Path(row['path'])
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError(f'Invalid artifact path {rel}')
        (dest/rel).parent.mkdir(parents=True, exist_ok=True)
        ssh_file(client, remote/rel, dest/rel, row)


def compiled(case, base, ddk, diagnosis, bundle_stamp):
    dest = base/'omc'
    stamp = dict(bundle=bundle_stamp, omg=sha(ddk/'tools/tools_omg/master/omg'))
    if not dest.exists():
        graph_size = sum(r['bytes'] for r in read(case/'case.json')['files'] if r['path'].startswith('weight/'))
        print(f'{case.name}: AIR with weights {graph_size / 2**30:.2f} GiB, '
              f'available before OMG {shutil.disk_usage(base).free / 2**30:.2f} GiB', flush=True)
        work = Path(tempfile.mkdtemp(dir=base, prefix='omg-'))
        (work/'weight').symlink_to(case/'weight', target_is_directory=True)
        code, argv, _ = compile_omg(work, 'weight/export.air', ddk, diagnosis, None)
        if code:
            raise RuntimeError(compile_log_text(work/'compile.log'))
        model, = (work/'model').rglob('*.omc')
        io = inspect(read_omc(model, ddk_root=ddk)[0], enum=GE_PROTO)
        data = dict(stamp=stamp, argv=argv, model=identity(model, work),
                    weights=[identity(p, work) for p in sorted(model.parent.rglob('*.weight'))])
        for direction in ('inputs', 'outputs'):
            data[direction] = []
            for row in io[direction]:
                if row['slot'] != len(data[direction]) or row['issues'] or not row['logical_bytes']:
                    raise ValueError(f'Invalid compiled descriptor {row}')
                dtype = {'FP16': 'float16', 'FP32': 'float32'}.get(row['dtype'], row['dtype'].lower())
                data[direction].append(dict(name=row['node' if direction == 'inputs' else 'edge'],
                                            shape=row['shape'], dtype=dtype, bytes=row['logical_bytes']))
        save_json(work/'build.json', data)
        work.rename(dest)
    build = read(dest/'build.json')
    if build['stamp'] != stamp:
        raise ValueError(f'{dest}: bundle or OMG changed. Select a new run directory.')
    return build, checked(dest, build['model']), [checked(dest, r) for r in build['weights']]


def output_order(descriptors, source, golden):
    if len(descriptors) != len(source):
        raise ValueError('Compiled output count differs from AIR')
    # Use surviving AIR edges when present. GraphOp outputs otherwise retain the public slot.
    names = [None] * len(source)
    by_edge = {row['edge']: row['name'] for row in source}
    for i, spec in enumerate(descriptors):
        names[i] = by_edge.get(spec['name'])
    for i, spec in enumerate(descriptors):
        if names[i] is None:
            candidates = [r['name'] for r in source if r['name'] not in names
                          and view(golden[r['name']]['shape']) == view(spec['shape'])]
            if len(candidates) == 1:
                names[i] = candidates[0]
            elif source[i]['name'] in candidates:
                names[i] = source[i]['name']
            else:
                raise ValueError(f'Cannot bind compiled output {i} {spec} to the AIR slots {candidates}')
    if sorted(names) != sorted(by_edge.values()):
        raise ValueError('Conflicting compiled output names and public AIR order')
    return names


def replay(base, case, build, model, weights, feeds, golden, runtime, stamp, recovery):
    meta = read(case/'case.json')
    if {r['name'] for r in build['inputs']} != set(meta['inputs']):
        raise ValueError(f'{case}: compiled input names differ from AIR')
    operands, paths = [], []
    for spec in build['inputs']:
        role = meta['inputs'][spec['name']]
        root, row = feeds[role]
        if (row['dtype'], view(row['shape']), row['bytes']) != (spec['dtype'], view(spec['shape']), spec['bytes']):
            raise ValueError(f'{case.name}: {role} compiled input differs from feed: {spec} vs {row}')
        paths.append(checked(root, row))
        operands.append(dict(spec, sha256=row['sha256']))
    names = output_order(build['outputs'], meta['outputs'], golden)
    for role, spec in zip(names, build['outputs'], strict=True):
        ref = golden[role]
        if view(ref['shape']) != view(spec['shape']) or spec['dtype'] not in ('float16', 'float32'):
            raise ValueError(f'{case.name}: invalid compiled {role} descriptor {spec}')
    _, serial, server = _environment()
    key = dict(bundle=stamp, model=build['model']['sha256'], weights=[w['sha256'] for w in build['weights']],
               inputs=[r['sha256'] for r in operands], runtime=runtime, serial=serial, server=server)
    saved = base/'completed.json'
    if recovery and saved.exists():
        record = read(saved)
        if record['binding'] != key:
            raise ValueError(f'{base}: saved result has different inputs, runtime, board or weights')
        folder = base/record['replay']
        result = read(checked(folder, record['receipt']))
        if result['status'] != 'EXECUTED' or result['exitcode'] != 0:
            raise ValueError(f'{folder}: incomplete replay')
        print(f'{case.name}: resumed verified board outputs', flush=True)
    else:
        folder = Path(tempfile.mkdtemp(dir=base, prefix='replay-'))
        result = run_v2(folder, case.name, model, operands, paths, build['outputs'], runtime,
                        fixture_sha256=stamp, external_weights=weights, release_after_pull=True, hdc_timeout=7200)
        save_json(saved, dict(binding=key, replay=folder.name, receipt=identity(folder/'result.json', folder)))
    values = {}
    for role, spec, row in zip(names, build['outputs'], result['outputs'], strict=True):
        checked(folder, row)
        values[role] = (folder, dict(row, dtype=spec['dtype'], shape=golden[role]['shape']))
    return values


def board(run, mode, phase, remote, ddk, config, diagnosis, recovery):
    bundle = run/'package'
    if not bundle.exists():
        pull_air(run, remote/'package', destination_name='package', ssh_download=transfer, before_download=transfer_size)
    manifest = read(bundle/'manifest.json')
    if (manifest['mode'], manifest['phase']) != (mode, phase):
        raise ValueError(f'{bundle}: mode/phase mismatch')
    for row in manifest['files']:
        checked(bundle, row)
    fixture = bundle/'fixture'
    ref = read(fixture/'reference.json')
    stamp = sha(bundle/'manifest.json')
    runtime = read(config.parent/read(config)['assets'])['environment']
    pieces = read(fixture/'tokens.json')
    feeds = {k: (fixture, v) for k, v in ref['inputs'].items()}
    scores = {b: {r: {} for r in ('cpu', '310p', 'cpro')} for b in ('cpu', '310p', 'cpro')}
    failures, unchanged, predictions = [], {}, {}
    for group in groups(mode):
        name = case_name(group)
        case = bundle/name
        base = run/'board'/name
        base.mkdir(parents=True, exist_ok=True)
        progress = dict(mode=mode, phase=phase, bundle_sha256=stamp, layer=group[0], case=name)
        save_json(run/'board/comparison.json', dict(status='RUNNING', **progress))
        try:
            build, model, weights = compiled(case, base, ddk, diagnosis, stamp)
            values = replay(base, case, build, model, weights, feeds, ref['outputs'], runtime, stamp, recovery)
        except Exception as exc:
            save_json(run/'board/comparison.json', dict(status='INTERRUPTED', message=str(exc), **progress))
            raise
        feeds['hidden'] = values[f'hidden_{group[-1]}']
        for role, (folder, row) in values.items():
            arrays = {'cpu': read_array(fixture, ref['cpu'][role]), '310p': read_array(fixture, ref['outputs'][role]),
                      'cpro': read_array(folder, row)}
            if role.startswith(('key_cache_', 'value_cache_')):
                before = read_array(fixture, ref['inputs'][role])
                unchanged[role] = {b: preserved_cache(x, before, phase) for b, x in arrays.items()}
                if not all(unchanged[role].values()):
                    failures.append(role + '/unwritten_slots')
            if role == 'logits':
                for backend, logits in arrays.items():
                    token = int(logits.argmax())
                    predictions[backend] = dict(id=token, text=pieces[token]) if np.isfinite(logits).all() else None
            for b, x in arrays.items():
                scope, actual = comparison_view(role, x, phase)
                for r, y in arrays.items():
                    scores[b][r][role] = {scope: metrics(actual, comparison_view(role, y, phase)[1])}
            if not scores['cpro']['310p'][role][scope]['passed']:
                failures.append(role)
        # A later failed layer still leaves the completed layer comparisons accessible.
        write_matrices(base/'matrices.csv', {b: {r: {k: v for k, v in s.items() if k in values}
                                                for r, s in row.items()} for b, row in scores.items()}, display=False)
    write_matrices(run/'board/matrices.csv', scores)
    save_json(run/'board/comparison.json', dict(mode=mode, phase=phase, bundle_sha256=stamp, failed=failures, unchanged_kv=unchanged,
                                               top1=predictions, status='FAILED_ACCURACY' if failures else 'PASSED'))
    print(f'Top1 {predictions}; matrices and comparison: {run / "board"}', flush=True)
    return not failures
