"""Compile the 40-layer chain on the receiving DDK host and release each layer's AIR."""
import fcntl
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from ...common.artifacts import output_path
from ...toolchain.compiler_entry import DEFAULT_PROFILE
from ...toolchain.existing_air import compile_existing_air, select_primary_omc
from ...toolchain.omc_container import read_container
from ..layer_probe.common import KIND, checked, identity, save_json, sha
from .chain_reference import BASIS, LAYERS, installed_vscale


def _load(path):
    return json.loads(path.read_text())


def _pull(path):
    tool = Path(__file__).resolve().parents[4] / 'targets/cpro/scripts/pull_artifacts.py'
    subprocess.run([sys.executable, str(tool), '--from', 'air', str(path)], check=True)


def _local(path):
    if path.resolve() != path:
        raise ValueError(f'Expected a local path without symlinks: {path}')
    return path


def _fixtures(source, folder, case):
    # A completed run only needs the small metadata, input and golden files.
    if not _local(folder / 'export.json').is_file():
        _pull(folder / 'export.json')
    export = _load(checked(source, case['export']))
    for spec in [*export['source_inputs'], export['golden']]:
        if not re.fullmatch(r'[A-Za-z0-9_]+\.bin', spec['path']):
            raise ValueError(f'Unexpected chain fixture name in {folder}')
        path = _local(folder / spec['path'])
        if not path.is_file():
            _pull(path)
        checked(folder, spec)
    return export


def _payloads(folder, export):
    air = export['air']
    weights = export.get('weight_files', {})
    if air['path'] != 'export.air' or any(not re.fullmatch(r'weight/[A-Za-z0-9_]+', p) for p in weights):
        raise ValueError(f'Unexpected AIR or FileConstant path in {folder}')
    protected = {s['path'] for s in [export['golden'], *export['source_inputs']]}
    files = {air['path']: air['bytes'], **weights}
    if protected.intersection(files):
        raise ValueError(f'Compiler payload overlaps a board fixture in {folder}')
    for path in files:
        _local(folder / path)
    return files


def _release(folder, sizes, inputs):
    if set(inputs) != set(sizes):
        raise ValueError(f'Compile receipt does not cover the AIR and all its weights in {folder}')
    present = []
    for name, size in sizes.items():
        path = _local(folder / name)
        if inputs[name]['bytes'] != size:
            raise ValueError(f'Compile input size differs from export: {path}')
        if path.exists():
            checked(folder, dict(inputs[name], path=name))
            present.append(path)
    for path in present:
        path.unlink()
    if present:
        print(f'{folder.name}: released {sum(sizes[p.relative_to(folder).as_posix()] for p in present):,} AIR bytes', flush=True)


def _verified_model(result_dir, result, export, sizes):
    model = select_primary_omc(result_dir)
    if checked(result_dir, result['omc']) != model or any(model.parent.glob('*.weight')):
        raise ValueError(f'{model}: expected the hash-bound embedded OMC')
    parts = read_container(model)['partitions']
    if not any(p['type'] == 1 and p['bytes'] > 0 for p in parts):
        raise ValueError(f'{model}: missing embedded weight partition')
    inputs = {r['source_relative']: {k: r[k] for k in ('bytes', 'sha256')}
              for r in result['input_snapshots'] if r['source_relative'] in sizes}
    if ({name: s['bytes'] for name, s in inputs.items()} != sizes
            or inputs['export.air']['sha256'] != export['air']['sha256']):
        raise ValueError(f'{result_dir}: source differs from the compiler snapshot')
    return model, inputs


def _compile_case(air, export, sizes, dest, ddk, ddk_id, diagnosis, gdr):
    # Persist compiler reports and logs. Only this invocation writes into the new attempt.
    attempt = Path(tempfile.mkdtemp(dir=dest, prefix='attempt-'))
    result_dir = attempt / 'omc'
    try:
        result = compile_existing_air(air, result_dir, ddk, ddk_id=ddk_id, timeout_seconds=900,
            compiler_profile=DEFAULT_PROFILE, save_weights_as_external_data=False, use_origin_format=True,
            tbe_diagnosis=diagnosis)
        if result['status'] != 'COMPILED':
            raise RuntimeError(f"{result.get('error', 'OMG failed')}; see {result_dir}/compile.log")
        model, inputs = _verified_model(result_dir, result, export, sizes)
        if installed_vscale(ddk) != gdr:
            raise ValueError('DDK GDR source changed during compilation')
        model.replace(dest / 'model.omc')
        return dict(status='COMPILED', omc=identity(dest / 'model.omc', dest.parents[1]), air_inputs=inputs)
    finally:
        # OMG's snapshots and generated scratch can be as large as both the AIR and OMC.
        # Failure diagnostics, logs, and the source AIR survive a failed attempt.
        work = result_dir / 'compile-work'
        if work.is_dir():
            shutil.rmtree(work)


def compile_chain(run, ddk, ddk_id, diagnosis):
    run = output_path(run)
    run.mkdir(parents=True, exist_ok=True)
    with (run / '.ddk-compile.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _compile_chain(run, ddk.resolve(), ddk_id, diagnosis)


def _compile_chain(run, ddk, ddk_id, diagnosis):
    source, out = _local(run / 'air'), _local(run / 'omc')
    if (run / '.air-export.lock').exists():
        raise ValueError('Use the DDK host with its own transferred AIR copy; this is the export directory')
    manifest_path = _local(source / 'manifest.json')
    if not manifest_path.is_file():
        _pull(manifest_path)
    manifest = _load(manifest_path)
    names = ['l00_c00_embedding_o0', *[f'l{i:02d}_c00_decoder_o0' for i in range(LAYERS)], 'l39_c00_head_o0']
    if (manifest.get('scope') != BASIS or manifest.get('artifact_kind') != KIND or manifest.get('status') != 'EXPORTED'
            or [c['id'] for c in manifest['cases']] != names
            or any(c.get('kind') != 'prefill_chain' or c.get('status') != 'EXPORTED'
                   or c['export']['path'] != f"models/{c['id']}/export.json" for c in manifest['cases'])):
        raise ValueError('Expected the exported 42-case prefill chain')
    if manifest['token_text_map']['path'] != 'token-text.json':
        raise ValueError('Unexpected token map path')
    if not _local(source / 'token-text.json').is_file():
        _pull(source / 'token-text.json')
    checked(source, manifest['token_text_map'])

    manifest_hash = sha(manifest_path)
    checkpoint = _local(out / 'compile.json')
    report = _load(checkpoint) if checkpoint.exists() else None
    if not ddk_id:
        if report is None or report.get('status') != 'COMPILED':
            raise ValueError('Set CPRO_DDK_ID before compiling the chain')
        ddk_id = report['ddk_id']
    meta = dict(artifact_kind=KIND, ddk_id=ddk_id, platform='kirinx90', compiler_profile=DEFAULT_PROFILE,
                selected_cases=names, tbe_compatibility_requested=diagnosis is not None, coverage='ALL_MANIFEST_CASES',
                status_scope='SELECTED_CASES', diagnostic_only=False, manifest_sha256=manifest_hash, source_batch='../air')
    if report is not None:
        if any(report.get(k) != v for k, v in meta.items()) or sha(out/'manifest.json') != manifest_hash:
            raise ValueError('Compilation checkpoint differs from the source or compiler configuration')
        if report['status'] == 'COMPILED':
            for case in manifest['cases']:
                _fixtures(source, _local(source/'models'/case['id']), case)
            return
        if report.get('storage') not in (None, 'per-layer-v1'):
            raise ValueError('Unrecognized OMC storage mode')
        report['storage'] = 'per-layer-v1'
    else:
        # Only a manifest snapshot can exist after interruption during initialization.
        out.mkdir(exist_ok=True)
        if set(p.name for p in out.iterdir()) - {'manifest.json', 'compile.json.tmp'}:
            raise ValueError(f'Unrecognized OMC directory without a compile checkpoint: {out}')
        shutil.copyfile(manifest_path, _local(out/'manifest.json'))
        report = dict(meta, status='COMPILING', storage='per-layer-v1', cases={}, gdr_source_sha256=installed_vscale(ddk))
        save_json(checkpoint, report)
    gdr = report['gdr_source_sha256']
    for n, case in enumerate(manifest['cases'], 1):
        name = case['id']
        folder = _local(source / 'models' / name)
        dest = _local(out / 'models' / name)
        row = report['cases'].get(name, {})
        done = row.get('status') == 'COMPILED'
        if not done and not folder.exists():
            _pull(folder)
        export = _fixtures(source, folder, case)
        sizes = _payloads(folder, export)
        if not done:
            for p in sizes:
                if not (folder/p).is_file():
                    _pull(folder/p)
        if done:
            model = checked(out, row['omc'])
            if model != dest/'model.omc':
                # Older PARTIAL batches kept their OMC inside compile-work. Keep its
                # original path and release only the explicitly inventoried snapshots.
                legacy = _local(dest / 'omc')
                receipt = _load(legacy/'report.json')
                if (receipt['status'] != 'COMPILED' or receipt['ddk_id'] != ddk_id
                        or receipt['compiler_profile'] != DEFAULT_PROFILE
                        or receipt['requested_options'] != dict(save_weights_as_external_data='false', use_origin_format='true')):
                    raise ValueError(f'{name}: previous OMG recipe differs')
                original, inputs = _verified_model(legacy, receipt, export, sizes)
                if original != model:
                    raise ValueError(f'{name}: invalid OMC checkpoint')
                row['air_inputs'] = inputs
                _release(legacy / 'compile-work/generated', sizes, inputs)
        else:
            if installed_vscale(ddk) != gdr:
                raise ValueError('DDK GDR source changed since the start of the batch')
            print(f'[{n}/{len(names)}] compile {name}', flush=True)
            dest.mkdir(parents=True, exist_ok=True)
            try:
                row = _compile_case(checked(folder, export['air']), export, sizes, dest, ddk, ddk_id, diagnosis, gdr)
            except Exception as e:
                report['status'] = 'PARTIAL'
                report['cases'][name] = dict(status='FAILED', error=str(e))
                save_json(checkpoint, report)
                raise
            report['cases'][name] = row
            save_json(checkpoint, report)
        # Publication precedes deletion. A crash during unlink is resumable without OMG.
        _release(folder, sizes, row['air_inputs'])
    report['status'] = 'COMPILED'
    save_json(checkpoint, report)
