"""Publish and receive only the files required by the next decoder stage."""
from contextlib import nullcontext
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

from qwen35_moe_dflash.common.artifacts import share_for_transfer
from qwen35_moe_dflash.common.hashing import sha256_file
from qwen35_moe_dflash.target.prefill_t64.decoder_cases import case_name, groups


def receipt(root, role):
    return root/f'.handoff-{role}.json'


def runtime_file(path):
    return path.startswith('fixture/') or path.endswith('/case.json')


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
                    yield path, sha256_file(path)
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
        records.append(dict(path=path.relative_to(site).as_posix(), bytes=path.stat().st_size, sha256=digest))
    header = dict(stage=role, case=root.name, files=records)
    if role != 'golden':
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


def valid_file(path, row):
    return path.is_file() and path.stat().st_size == row['bytes'] and sha256_file(path) == row['sha256']


def receive(root, remote, role, args):
    from decoder4_prefill import ssh_file
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
        channel = nullcontext(None)
        source = local
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
        from pull_artifacts import DEFAULT_CONFIG, connection
        channel = connection(DEFAULT_CONFIG, role)
        source = remote
    with channel as client, tempfile.TemporaryDirectory(dir=root, prefix='.handoff-') as temp:
        temp = Path(temp)
        def fetch(src, dst):
            if local:
                shutil.copyfile(src, dst)
            else:
                ssh_file(client, src, dst)
        index = receipt(source, role)
        fetch(index, temp/'manifest')
        data = json.loads((temp/'manifest').read_bytes())
        if (data['stage'], data['case']) != (role, root.name) or data.get('gdn_conv', args.gdn_conv) != args.gdn_conv:
            raise ValueError(f'{index}: stage, case or GDN convolution mismatch')
        staged = temp/'incoming'
        staged.mkdir()
        roles = {'golden': 'air', 'air': 'omc', 'omc': 'board'}
        print(f'[{role} → {roles[role]}] {source/site_name} → {dest}, {len(data["files"])} files, '
              f'{sum(r["bytes"] for r in data["files"])/2**30:.2f} GiB', flush=True)
        if not data['files']:
            raise ValueError(f'Empty transfer manifest {index}')
        seen = set()
        missing = []
        for row in data['files']:
            rel = Path(row['path'])
            if rel.is_absolute() or '..' in rel.parts or not rel.parts or rel in seen:
                raise ValueError(f'Invalid or duplicate transfer path {rel}')
            seen.add(rel)
            previous = dest/rel
            if local == root:
                if not valid_file(previous, row):
                    raise ValueError(f'Shared artifact changed: {previous}')
                continue
            target = staged/rel
            target.parent.mkdir(exist_ok=True, parents=True)
            if previous.exists() and previous.resolve().is_relative_to(root) and valid_file(previous, row):
                os.link(previous.resolve(), target)
            else:
                missing.append((rel, target, row))
        needed = sum(row['bytes'] for _, _, row in missing)
        free = shutil.disk_usage(root).free
        print(f'To receive: {needed/2**30:.2f} GiB, free: {free/2**30:.2f} GiB', flush=True)
        if needed > free:
            raise OSError(f'{root}: insufficient space for {needed:,} bytes')
        for rel, target, row in missing:
            fetch(source/site_name/rel, target)
            if not valid_file(target, row):
                raise ValueError(f'Size or SHA-256 differs from {index}: {rel}')
        if local == root:
            print('Producer and receiver share this verified case directory.', flush=True)
            return
        # Confirm a producer that restarted during the transfer did not replace the manifest.
        fetch(index, temp/'confirmed')
        if (temp/'manifest').read_bytes() != (temp/'confirmed').read_bytes():
            raise ValueError(f'{source}: upstream stage changed during transfer')
        previous_dir = temp/'previous'
        if dest.exists():
            dest.rename(previous_dir)
        try:
            staged.rename(dest)
        except BaseException:
            if previous_dir.exists():
                previous_dir.rename(dest)
            raise
        for later in {'golden': ('golden', 'air', 'omc'), 'air': ('golden', 'air', 'omc'), 'omc': ('omc',)}[role]:
            receipt(root, later).unlink(missing_ok=True)
        print(f'Checked and received {dest}', flush=True)
