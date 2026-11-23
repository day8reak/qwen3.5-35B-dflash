#!/usr/bin/env python3
"""Replay the existing T64 GDR AIR with the AIR host's GDN-acc/input files and return device outputs."""
import argparse
import fcntl
import hashlib
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'scripts'))
from pull_artifacts import DEFAULT_CONFIG, connection
from export_gdr_t64 import INPUTS, write_case
from qwen35_moe_dflash.common.artifacts import output_path
from qwen35_moe_dflash.target.layer_probe.common import checked, identity, save_json, sha
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


def absolute_remote_path(value):
    path = PurePosixPath(value)
    if not path.is_absolute() or '..' in path.parts or path == PurePosixPath('/'):
        raise argparse.ArgumentTypeError('expected an absolute path on the AIR host')
    return path


def fetch(sftp, remote, local, expected_bytes=None):
    print(f'78 {remote} -> 35 {local}', flush=True)
    try:
        size = sftp.stat(str(remote)).st_size
        if expected_bytes is not None and size != expected_bytes:
            raise ValueError(f'78 {remote}: expected {expected_bytes} bytes, found {size}')
        digest = remote_hash(sftp, remote)
        if local.is_file() and local.stat().st_size == size and sha(local) == digest:
            print(f'Reusing verified {local.name} ({size} bytes)', flush=True)
            return digest
        with tempfile.TemporaryDirectory(dir=local.parent) as tmp:
            downloaded = Path(tmp) / local.name
            sftp.get(str(remote), str(downloaded))
            if downloaded.stat().st_size != size or sha(downloaded) != digest:
                raise ValueError(f'{remote}: size or SHA256 differs after transfer to {downloaded}')
            downloaded.replace(local)
    except OSError as exc:
        raise RuntimeError(f'Transfer from 78 {remote} to 35 {local} failed: {exc}') from exc
    print(f'Verified {local.name} ({size} bytes)', flush=True)
    return digest


def list_inputs(sftp, folder):
    print(f'Listing input directory on 78: {folder}', flush=True)
    try:
        return sorted(sftp.listdir(str(folder)))
    except OSError as exc:
        raise RuntimeError(f'Cannot list input directory on 78: {folder} ({exc}). '
                           'Set --input-dir to the directory containing the seven bins.') from exc


def prepare(run, config, tests, air_path, input_dir, explicit, output_dir):
    incoming = run / 'download'
    incoming.mkdir(mode=0o700, exist_ok=True)
    values, bindings = {}, {}
    with connection(config, 'air') as client, client.open_sftp() as sftp:
        sftp.get_channel().settimeout(120)
        names = input_names(list_inputs(sftp, input_dir), explicit)
        air_path = air_path or find_air(client, sftp, tests)
        local_air = incoming / 'gdr_t64.air'
        fetch(sftp, air_path, local_air)
        for role, name in zip(INPUTS, names, strict=True):
            remote = input_dir / name
            local = incoming / (role + '.bin')
            shape = shapes(role)[0]
            dtype = np.dtype('<i2' if role == 'effective_length' else '<f2')
            digest = fetch(sftp, remote, local, int(np.prod(shape)) * dtype.itemsize)
            values[role] = np.fromfile(local, dtype=dtype).reshape(shape)
            bindings[role] = dict(path=str(remote), sha256=digest)
    write_case(run / 'case', values, dict(kind='supplied_gdr_inputs', air=str(air_path), files=bindings,
               input_dir=str(input_dir), output_dir=str(output_dir)), existing_air=local_air)
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


def ensure_remote_directory(sftp, directory):
    try:
        attrs = sftp.stat(str(directory))
    except FileNotFoundError:
        if directory == directory.parent:
            raise
        ensure_remote_directory(sftp, directory.parent)
        try:
            sftp.mkdir(str(directory))
            return
        except OSError as exc:
            # An SFTP server may report a concurrent mkdir as a generic failure.
            try:
                attrs = sftp.stat(str(directory))
            except OSError:
                raise OSError(f'Cannot create output directory on 78: {directory} ({exc})') from exc
    if not stat.S_ISDIR(attrs.st_mode):
        raise NotADirectoryError(f'Expected a directory on 78: {directory}')


def publish(config, result, dest, run_name):
    manifest = gdr.read(result / 'manifest.json')
    files = [checked(result, spec) for spec in [*manifest['outputs'], manifest['accuracy']]] + [result / 'manifest.json']
    with connection(config, 'air') as client, client.open_sftp() as sftp:
        sftp.get_channel().settimeout(120)
        print(f'Publishing board results to 78: {dest}', flush=True)
        ensure_remote_directory(sftp, dest.parent)
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


def prepare_download(run, config, tests, air, inputs, files, destination):
    run.mkdir(mode=0o700, parents=True, exist_ok=True)
    token = os.open(run, os.O_RDONLY | os.O_DIRECTORY)
    try:
        try:
            fcntl.flock(token, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f'A download is already active in {run}') from exc
        if any(p.name != 'download' for p in run.iterdir()) or (run / 'download').is_symlink():
            raise ValueError(f'{run} already contains more than a partial download. '
                             'Use --publish-only for a recovered run or select a fresh --run for new inputs.')
        prepare(run, config, tests, air, inputs, files, destination)
    finally:
        os.close(token)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='new or download-only working directory on the DDK host')
    parser.add_argument('--air', type=absolute_remote_path, help='absolute path to the standalone GDR AIR file on 78')
    parser.add_argument('--input-dir', type=absolute_remote_path, help='absolute directory of seven input bins on 78')
    parser.add_argument('--output-dir', type=absolute_remote_path, help='absolute destination directory on 78 '
                        '(default: AI_RUN_DIR/target/tests/GDN-acc/gdr)')
    parser.add_argument('--input-files', nargs=7, metavar='BIN', help='filenames in query key value g beta initial_state effective_length order')
    parser.add_argument('--transfer-config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--publish-only', action='store_true', help='retry extraction/upload of an already recovered run')
    parser.add_argument('--gdr-source', type=Path, default=os.getenv('GDR_SOURCE'))
    parser.add_argument('--ddk-root', type=Path, default=os.getenv('CPRO_DDK_ROOT'))
    parser.add_argument('--runtime-config', type=Path, default=os.getenv('RUNTIME_CONFIG'))
    parser.add_argument('--tbe-diagnosis', type=Path, default=os.getenv('TBE_DIAGNOSIS'))
    args = parser.parse_args()
    if args.publish_only and (args.air or args.input_dir or args.input_files):
        parser.error('--publish-only reads the recovered run; use --output-dir to change its destination')
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
        prepare_download(run, args.transfer_config, tests, args.air, args.input_dir or remote / 'input',
                         args.input_files, args.output_dir or remote / 'gdr')
        ddk = args.ddk_root.resolve()
        gdr.compile_case(run / 'case', run / 'omc', ddk, ddk_id, args.gdr_source.resolve(), diagnosis)
        validate = [sys.executable, str(HERE / 'validate_draft_vector.py')]
        subprocess.run([*validate, '--compiled', run / 'omc', '--config', runtime, '--ddk-root', ddk,
                        '--output', run / 'execution', '--apply'], check=True)
        recovery = subprocess.run([*validate, '--recover', run / 'execution', '--wait', '--output', run / 'recovery'])
        if recovery.returncode and not (run / 'recovery/result.json').is_file():
            recovery.check_returncode()
    result = collect(run)
    recorded = gdr.read(result / 'manifest.json')['source'].get('output_dir')
    destination = args.output_dir or (PurePosixPath(recorded) if recorded else remote / 'gdr')
    publish(args.transfer_config, result, destination, run.name)


if __name__ == '__main__':
    try:
        main()
    except (OSError, KeyError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        sys.exit(f'gdn-acc: {exc}')
