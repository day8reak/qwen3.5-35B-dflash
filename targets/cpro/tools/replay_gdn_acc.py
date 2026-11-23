#!/usr/bin/env python3
"""Replay the existing T64 GDR AIR with the AIR host's GDN-acc/input files and return device outputs."""
import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import sys
import uuid

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'scripts'))
from pull_artifacts import DEFAULT_CONFIG, connection
from export_gdr_t64 import INPUTS, write_case
from qwen35_moe_dflash.common.artifacts import output_path
from qwen35_moe_dflash.target.layer_probe.common import checked, identity, new_output, save_json, sha
from qwen35_moe_dflash.target.prefill_t64 import standalone_gdr as gdr
from qwen35_moe_dflash.target.prefill_t64.gdr_stage_data import shapes


def remote_hash(sftp, path):
    digest = hashlib.sha256()
    with sftp.open(str(path), 'rb') as stream:
        for data in iter(lambda: stream.read(128 << 10), b''):
            digest.update(data)
    return digest.hexdigest()


def find_air(client, sftp, tests):
    command = f'find {shlex.quote(str(tests))} -name gdr_t64.air -type f -print 2>/dev/null'
    _, stdout, _ = client.exec_command(command, timeout=120)
    candidates = sorted(PurePosixPath(p) for p in stdout.read().decode().splitlines() if p)
    if not candidates:
        raise ValueError(f'No gdr_t64.air found under {tests} on the AIR host. Supply --air with its full path.')
    if len(candidates) > 1 and len({remote_hash(sftp, p) for p in candidates}) != 1:
        raise ValueError('Found different AIRs on the AIR host. Select one with --air:\n' + '\n'.join(map(str, candidates)))
    return candidates[0]


def input_names(available, explicit):
    if explicit:
        if len(set(explicit)) != len(INPUTS) or not set(explicit) <= set(available):
            raise ValueError(f'Expected seven distinct files in the input directory; found {available}')
        return explicit
    result = []
    for role in INPUTS:
        stems = {role.replace('_', ''), {'query': 'q', 'key': 'k', 'value': 'v'}.get(role, role)}
        matches = [n for n in available if n.lower().endswith('.bin') and n[:-4].lower().replace('_', '') in stems]
        if len(matches) != 1:
            raise ValueError(f'{role}: cannot identify one named bin in {available}. '
                             'Supply --input-files in AIR order: ' + ' '.join(INPUTS))
        result.append(matches[0])
    return result


def prepare(run, config, tests, air_path, input_dir, explicit):
    incoming = new_output(run / 'download')
    values, bindings = {}, {}
    with connection(config, 'air') as client, client.open_sftp() as sftp:
        sftp.get_channel().settimeout(120)
        air_path = air_path or find_air(client, sftp, tests)
        local_air = incoming / 'gdr_t64.air'
        print(f'Pulling AIR {air_path}', flush=True)
        sftp.get(str(air_path), str(local_air))
        if sha(local_air) != remote_hash(sftp, air_path):
            raise ValueError('AIR digest changed during transfer')
        names = input_names(sorted(sftp.listdir(str(input_dir))), explicit)
        for role, name in zip(INPUTS, names, strict=True):
            remote = input_dir / name
            local = incoming / (role + '.bin')
            shape = shapes(role)[0]
            dtype = np.dtype('<i2' if role == 'effective_length' else '<f2')
            size = int(np.prod(shape)) * dtype.itemsize
            if sftp.stat(str(remote)).st_size != size:
                raise ValueError(f'{remote}: expected {size} bytes for {dtype} {shape}')
            sftp.get(str(remote), str(local))
            digest = sha(local)
            if digest != remote_hash(sftp, remote):
                raise ValueError(f'{remote}: input digest changed during transfer')
            values[role] = np.fromfile(local, dtype=dtype).reshape(shape)
            bindings[role] = dict(path=str(remote), sha256=digest)
            print(f'{role} <- {remote} ({size} bytes)', flush=True)
    write_case(run / 'case', values, dict(kind='supplied_gdr_inputs', air=str(air_path), files=bindings),
               existing_air=local_air)
    gdr.bundle(run / 'case')


def collect(run):
    recovery = run / 'recovery'
    report = gdr.read(recovery / 'result.json')
    if report.get('stage') != 'complete' or report.get('device_execution') != 'EXECUTED_OUTPUTS_RETRIEVED':
        raise RuntimeError(f"No completed device output: {report.get('error', report.get('status'))}; see {recovery}")
    fixture = run / 'execution/fixture'
    request = gdr.read(run / 'execution/request.json')
    if request['fixture_sha256'] != sha(fixture / 'manifest.json') or request['board_attempt'] != report['board_attempt']:
        raise ValueError('Recovery does not belong to the requested board fixture')
    meta = gdr.read(fixture / 'manifest.json')
    case, = meta['cases']
    original = gdr.read(run / 'case/manifest.json')
    if (case['case'] != gdr.CASE or meta['source_evidence']['manifest_sha256'] != sha(run / 'case/manifest.json')
            or {s['semantic_name']: s['fixture_sha256'] for s in case['inputs']}
            != {s['name']: s['sha256'] for s in original['inputs']}):
        raise ValueError('Board inputs differ from the selected seven files')
    device = recovery / 'native' / gdr.CASE
    receipt = gdr.read(device / 'result.json')
    if receipt['status'] != 'EXECUTED' or receipt['evidence_kind'] != 'device-hiai':
        raise ValueError('Expected a completed HIAI probe on the board')
    gdr.summarize(recovery)
    result = run / 'outputs'
    result.mkdir(exist_ok=True)
    rows = {r['output_name']: r for r in report['cases']}
    outputs = []
    with (device / '0.bin').open('rb') as stream:
        for i, (spec, name) in enumerate(zip(case['outputs'], gdr.OUTPUTS, strict=True)):
            raw = stream.read(spec['bytes'])
            row = rows[name]
            if (spec['index'] != i or spec['semantic_name'] != name or len(raw) != spec['bytes']
                    or hashlib.sha256(raw).hexdigest() != row['output_sha256'][0]):
                raise ValueError(f'{name}: board output hash, size or slot mismatch')
            path = result / (name + '.bin')
            path.write_bytes(raw)
            outputs.append(dict(identity(path, result), dtype=spec['dtype'], shape=spec['shape'],
                                stable=row['stable'], full_output_stable=row.get('full_output_stable', row['stable'])))
        if stream.read(1):
            raise ValueError('Unexpected trailing bytes in measured call 0')
    shutil.copyfile(recovery / 'accuracy.csv', result / 'accuracy.csv')
    save_json(result / 'manifest.json', dict(source=original['source'], air=original['air'],
        omc_sha256=meta['source_evidence']['omc_sha256'], gdr_source_sha256=meta['source_evidence']['gdr_source_sha256'],
        board_attempt=request['board_attempt'], measured_call=0, effective_length=case['effective_length'],
        numerical_validation=report['numerical_validation'], outputs=outputs, accuracy=identity(result / 'accuracy.csv', result)))
    return result


def publish(config, result, dest, run_name):
    manifest = gdr.read(result / 'manifest.json')
    files = [checked(result, spec) for spec in [*manifest['outputs'], manifest['accuracy']]] + [result / 'manifest.json']
    with connection(config, 'air') as client, client.open_sftp() as sftp:
        sftp.get_channel().settimeout(120)
        for directory in (dest, dest / run_name):
            try:
                previous = sftp.listdir(str(directory))
            except FileNotFoundError:
                previous = None
            if not previous:
                dest = directory
                break
            if all(p.name in previous and remote_hash(sftp, directory / p.name) == sha(p) for p in files):
                print(f'Already verified on 78: {directory}', flush=True)
                return
        else:
            raise FileExistsError(f'{directory} already contains a different result')
        stage = dest.parent / ('.gdr-upload-' + uuid.uuid4().hex)
        sftp.mkdir(str(stage))
        for path in files:
            remote = stage / path.name
            sftp.put(str(path), str(remote))
            if remote_hash(sftp, remote) != sha(path):
                raise IOError(f'Upload SHA256 mismatch: {remote}')
        if previous == []:  # POSIX rename can atomically replace an empty directory only.
            sftp.posix_rename(str(stage), str(dest))
        else:
            sftp.rename(str(stage), str(dest))
        print(f'Verified device outputs on 78: {dest}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='new working directory on the DDK host')
    parser.add_argument('--air', type=PurePosixPath, help='explicit path on the AIR host')
    parser.add_argument('--input-files', nargs=7, metavar='BIN', help='filenames in query key value g beta initial_state effective_length order')
    parser.add_argument('--transfer-config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--publish-only', action='store_true', help='retry extraction/upload of an already recovered run')
    parser.add_argument('--gdr-source', type=Path, default=os.getenv('GDR_SOURCE'))
    parser.add_argument('--ddk-root', type=Path, default=os.getenv('CPRO_DDK_ROOT'))
    parser.add_argument('--runtime-config', type=Path, default=os.getenv('RUNTIME_CONFIG'))
    parser.add_argument('--tbe-diagnosis', type=Path, default=os.getenv('TBE_DIAGNOSIS'))
    args = parser.parse_args()
    run = output_path(args.run)
    tests = Path(os.environ['AI_RUN_DIR']).resolve() / 'target/tests'
    remote = PurePosixPath(tests) / 'GDN-acc'
    if not args.publish_only:
        if not args.ddk_root or not args.gdr_source:
            parser.error('set CPRO_DDK_ROOT and GDR_SOURCE or pass --ddk-root and --gdr-source')
        runtime = args.runtime_config or tests.parents[1] / 'runtime/runtime.json'
        config = gdr.read(runtime)
        ddk_id = gdr.read(runtime.parent / config['assets'])['environment']['ddk_id']
        diagnosis = args.tbe_diagnosis or tests.parents[1] / 'common/tbe-diagnosis'
        new_output(run)
        prepare(run, args.transfer_config, tests, args.air, remote / 'input', args.input_files)
        ddk = args.ddk_root.resolve()
        gdr.compile_case(run / 'case', run / 'omc', ddk, ddk_id, args.gdr_source.resolve(), diagnosis)
        validate = [sys.executable, str(HERE / 'validate_draft_vector.py')]
        subprocess.run([*validate, '--compiled', run / 'omc', '--config', runtime, '--ddk-root', ddk,
                        '--output', run / 'execution', '--apply'], check=True)
        recovery = subprocess.run([*validate, '--recover', run / 'execution', '--wait', '--output', run / 'recovery'])
        if recovery.returncode and not (run / 'recovery/result.json').is_file():
            recovery.check_returncode()
    publish(args.transfer_config, collect(run), remote / 'gdr', run.name)


if __name__ == '__main__':
    try:
        main()
    except (OSError, KeyError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        sys.exit(f'gdn-acc: {exc}')
