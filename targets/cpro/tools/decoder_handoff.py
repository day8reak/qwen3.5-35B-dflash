"""Publish and receive only the files required by the next decoder stage."""
from contextlib import ExitStack, contextmanager, nullcontext
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
import uuid

from qwen35_moe_dflash.common.artifacts import share_for_transfer
from qwen35_moe_dflash.common.hashing import sha256_file
from qwen35_moe_dflash.target.prefill_t64.decoder_cases import case_name, groups


def receipt(root, role):
    return root/f'.handoff-{role}.json'


def runtime_file(path):
    return path.startswith(('fixture/', 'L00-L03/inputs/')) or path.endswith('/case.json')


def inventory(root, role, mode, phase):
    native = (mode, phase) == ('40', 'prefill')
    site = root/('board' if role == 'omc' else 'export')
    bundle = site/('air' if native else 'package')
    if role == 'golden':
        folders = [site/'fixture']
        if native:
            folders = [site/'reference', *(site/f'L{i:02d}/cpu' for i in range(40))]
        for folder in folders:
            marker = 'reference.json' if folder.name == 'fixture' else 'manifest.json'
            if not (folder/marker).is_file():
                raise ValueError(f'Missing golden receipt {folder/marker}')
            for path in sorted(folder.rglob('*')):
                if path.is_file():
                    yield path, None
        return

    def listed(folder, index, records):
        yield folder/index, sha256_file(folder/index)
        for row in records:
            rel = Path(row['path'])
            if rel.is_absolute() or '..' in rel.parts:
                raise ValueError(f'Invalid manifest path {rel}')
            path = folder/rel
            if path.stat().st_size != row['bytes']:
                raise ValueError(f'Compiled or exported artifact changed: {path}')
            yield path, row['sha256']

    meta = json.loads((bundle/'manifest.json').read_bytes())
    files = meta['files']
    if role == 'omc':
        if native:
            files = [r for r in files if r['path'].startswith(('inputs/', 'golden/'))]
        else:
            files = [r for r in files if runtime_file(r['path'])]
    yield from listed(bundle, 'manifest.json', files)
    if role == 'omc':
        for group in groups(mode):
            folder = (site if native else site/case_name(group))/'omc'
            index = 'manifest.json' if native else 'build.json'
            build = json.loads((folder/index).read_bytes())
            yield from listed(folder, index, [build['primary' if native else 'model'], *build['weights']])


def publish(root, role, args):
    site = root/('board' if role == 'omc' else 'export')
    records = []
    for path, digest in inventory(root, role, args.mode, args.phase):
        row = dict(path=path.relative_to(site).as_posix(), bytes=path.stat().st_size)
        if digest is not None:
            row['sha256'] = digest
        records.append(row)
    header = dict(stage=role, case=root.name, files=records)
    if role == 'golden':
        header['generation'] = uuid.uuid4().hex
    else:
        header['gdn_conv'] = args.gdn_conv
    dest = receipt(root, role)
    with tempfile.TemporaryDirectory(prefix='.publish-', dir=root) as staging:
        tmp = Path(staging)/dest.name
        tmp.write_text(json.dumps(header, ensure_ascii=False) + '\n')
        if role in ('golden', 'air'):
            print(f'[{role}] 发布前设置本次运行目录和清单权限 chmod -R 777 {root}', flush=True)
            share_for_transfer(root)
        tmp.replace(dest)
    print(f'{role} transfer manifest: {len(records)} files → {dest}', flush=True)


def complete_file(path, row):
    return path.is_file() and path.stat().st_size == row['bytes']


def previous_files(dest, role, data, marker):
    if (dest/marker).is_file():
        prior = json.loads((dest/marker).read_bytes())
        if prior == data:
            return True, {}
        if prior['stage'] == role:
            return False, {r['path']: r.get('sha256') for r in prior['files']}
    # Older receivers kept the AIR bundle's manifest but no transfer marker.
    if role == 'air':
        for name in ('air', 'package'):
            manifest = dest/name/'manifest.json'
            if manifest.is_file():
                try:
                    rows = json.loads(manifest.read_bytes())['files']
                    return False, {f"{name}/{r['path']}": r.get('sha256') for r in rows}
                except (ValueError, KeyError, TypeError):
                    continue
    return False, {}


SSH_CHUNK = 256 << 20
SSH_SESSION_BYTES = 512 << 20


@contextmanager
def ssh_reader(role):
    import paramiko
    from decoder4_prefill import ssh_file
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
    from pull_artifacts import DEFAULT_CONFIG, connection

    with ExitStack() as session:
        client = None
        transferred = 0

        def fetch(src, dst, size=None):
            nonlocal client, transferred
            failures = 0
            while True:
                offset = dst.stat().st_size if size is not None and dst.exists() else 0
                length = None if size is None else min(SSH_CHUNK, size - offset)
                if client is not None and transferred + (length or 0) > SSH_SESSION_BYTES:
                    session.close()
                    client = None
                try:
                    if client is None:
                        client = session.enter_context(connection(DEFAULT_CONFIG, role))
                        transferred = 0
                    ssh_file(client, src, dst, offset=offset, length=length)
                except (paramiko.AuthenticationException, paramiko.BadHostKeyException):
                    raise
                except (paramiko.SSHException, paramiko.ssh_exception.NoValidConnectionsError,
                        EOFError, ConnectionError, TimeoutError) as error:
                    session.close()
                    client = None
                    if size is None:
                        dst.unlink(missing_ok=True)
                    failures += 1
                    if failures == 3:
                        raise
                    print(f'SSH interrupted ({error}); reconnecting to read {src}', flush=True)
                    time.sleep(failures)
                    continue
                received = dst.stat().st_size
                if length is not None and received != offset + length:
                    dst.unlink()
                    raise ValueError(f'Unexpected byte count for {src} at offset {offset}')
                transferred += received - offset
                if size is None or received == size:
                    return
                failures = 0
        yield fetch


def resume_directory(root, role, manifest):
    # The outer case lock serializes receivers. A different manifest has its own payload.
    cache = root/f'.receive-{role}'
    if cache.is_symlink():
        raise ValueError(f'Unexpected transfer directory link: {cache}')
    cache.mkdir(mode=0o700, exist_ok=True)
    staged = cache/sha256_file(manifest)
    for old in cache.iterdir():
        if old.is_symlink() or not old.is_dir() or not re.fullmatch('[0-9a-f]{64}', old.name):
            raise ValueError(f'Unexpected entry in transfer directory: {old}')
        if old != staged:
            shutil.rmtree(old)
    return staged


def receive(root, remote, role, args):
    local = args.local_source
    configured = os.getenv(f'DECODER_{role.upper()}_LOCAL_ROOT')
    if local is None and configured:
        local = Path(configured).expanduser()/root.name
    if local is None and remote == root and receipt(root, role).is_file():
        local = root
    site_name = 'board' if role == 'omc' else 'export'
    dest = root/site_name
    if dest.is_symlink():
        raise ValueError(f'Unexpected stage directory link: {dest}')
    if local:
        local = local.resolve(strict=True)
        reader = nullcontext(shutil.copyfile)
        source = local
    else:
        reader = ssh_reader(role)
        source = remote
    with reader as fetch, tempfile.TemporaryDirectory(dir=root, prefix='.handoff-') as temp:
        temp = Path(temp)
        index = receipt(source, role)
        fetch(index, temp/'manifest')
        manifest = (temp/'manifest').read_bytes()
        data = json.loads(manifest)
        if (data['stage'], data['case']) != (role, root.name) or data.get('gdn_conv', args.gdn_conv) != args.gdn_conv:
            raise ValueError(f'{index}: stage, case or GDN convolution mismatch')
        staged = temp/'incoming' if local else resume_directory(root, role, temp/'manifest')
        staged.mkdir(exist_ok=True)
        roles = {'golden': 'air', 'air': 'omc', 'omc': 'board'}
        print(f'[{role} → {roles[role]}] {source/site_name} → {dest}, {len(data["files"])} files, '
              f'{sum(r["bytes"] for r in data["files"])/2**30:.2f} GiB', flush=True)
        if not data['files']:
            raise ValueError(f'Empty transfer manifest {index}')
        marker = '.decoder-transfer.json'
        same_generation, previous_ids = previous_files(dest, role, data, marker)
        seen = set()
        missing = []
        needed = 0
        for row in data['files']:
            rel = Path(row['path'])
            if rel.is_absolute() or '..' in rel.parts or not rel.parts or rel.parts[0] == marker or rel in seen:
                raise ValueError(f'Invalid or duplicate transfer path {rel}')
            seen.add(rel)
            previous = dest/rel
            if local == root:
                if not complete_file(previous, row):
                    raise ValueError(f'Shared artifact changed: {previous}')
                continue
            target = staged/rel
            if target.is_symlink() or target.resolve() != target:
                raise ValueError(f'Unexpected transfer path: {target}')
            target.parent.mkdir(exist_ok=True, parents=True)
            if complete_file(target, row):
                continue
            if target.exists() and (target.stat().st_size >= row['bytes'] or target.stat().st_nlink > 1):
                target.unlink()
            reuse = same_generation or (row.get('sha256') and previous_ids.get(row['path']) == row['sha256'])
            if reuse and previous.resolve().is_relative_to(root) and complete_file(previous, row):
                target.unlink(missing_ok=True)
                os.link(previous.resolve(), target)
            else:
                missing.append((rel, target, row))
                needed += row['bytes'] - (target.stat().st_size if target.exists() else 0)
        free = shutil.disk_usage(root).free
        print(f'To receive: {needed/2**30:.2f} GiB, free: {free/2**30:.2f} GiB', flush=True)
        if needed > free:
            raise OSError(f'{root}: insufficient space for {needed:,} bytes')
        for number, (rel, target, row) in enumerate(missing, 1):
            if local:
                fetch(source/site_name/rel, target)
            else:
                fetch(source/site_name/rel, target, row['bytes'])
            if not complete_file(target, row):
                target.unlink(missing_ok=True)
                raise ValueError(f'Incomplete file from {index}: {rel}')
            if number % 64 == 0 or number == len(missing):
                print(f'已接收 {len(seen) - len(missing) + number}/{len(seen)} 个文件', flush=True)
        if local == root:
            print('Producer and receiver share this case directory.', flush=True)
            return
        # Confirm a producer that restarted during the transfer did not replace the manifest.
        fetch(index, temp/'confirmed')
        if manifest != (temp/'confirmed').read_bytes():
            raise ValueError(f'{source}: upstream stage changed during transfer')
        (staged/marker).write_bytes(manifest)
        previous_dir = temp/'previous'
        if dest.exists():
            dest.rename(previous_dir)
        try:
            staged.rename(dest)
        except BaseException:
            if previous_dir.exists():
                previous_dir.rename(dest)
            raise
        if not local:
            staged.parent.rmdir()
        for later in {'golden': ('golden', 'air', 'omc'), 'air': ('golden', 'air', 'omc'), 'omc': ('omc',)}[role]:
            receipt(root, later).unlink(missing_ok=True)
        print(f'Received {dest}', flush=True)
