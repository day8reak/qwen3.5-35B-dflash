#!/usr/bin/env python3
"""Pull exported files or directories to the same absolute paths on the receiving host."""
import argparse
from contextlib import contextmanager
import fcntl
import getpass
import inspect
import json
import os
from pathlib import Path
import posixpath
import shutil
import stat
import sys
import tempfile


DEFAULT_CONFIG = Path(os.getenv('CPRO_TRANSFER_CONFIG') or
                      Path.home() / '.config' / 'qwen35-cpro' / 'transfer.json').expanduser()


def initialize(config, hosts):
    if config.exists():
        raise FileExistsError(f'Edit the existing settings in {config}')
    endpoints = {}
    roles = ('air', 'omc') if len(hosts) == 2 else ('golden', 'air', 'omc')
    for role, host in zip(roles, hosts, strict=True):
        user = input(f'{role} ({host}) SSH user [{getpass.getuser()}]: ').strip() or getpass.getuser()
        password = getpass.getpass(f'{role} ({host}) SSH password (Enter for key or local login): ')
        endpoints[role] = dict(host=host, port=22, user=user, password=password)
    config.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with os.fdopen(os.open(config, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as stream:
        json.dump(endpoints, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(f'Settings saved in {config}')


def destination(name, roots):
    path = Path(name).expanduser()
    if not path.is_absolute():
        path = roots[0] / path
    if '..' in path.parts or not any(root in path.parents for root in roots) or path in roots:
        raise ValueError(f'Select an artifact under AI_RUN_DIR or REPO_DIR: {path}')
    if path.is_symlink() or not any(root in path.parents and path.parent.resolve().is_relative_to(root.resolve())
                                    for root in roots):
        raise ValueError(f'Destination traverses a symbolic link: {path}')
    return path


def download(sftp, remote, local, ancestors=()):
    attrs = sftp.stat(remote)
    if stat.S_ISREG(attrs.st_mode):
        if 'max_concurrent_prefetch_requests' in inspect.signature(sftp.get).parameters:
            sftp.get(remote, str(local), max_concurrent_prefetch_requests=64)
        else:
            with sftp.open(remote, 'rb') as src, local.open('wb') as dst:
                shutil.copyfileobj(src, dst, length=1024*1024)
        if local.stat().st_size != attrs.st_size:
            raise OSError(f'File size changed during transfer: {remote}')
        local.chmod(stat.S_IMODE(attrs.st_mode) & 0o777)
    elif stat.S_ISDIR(attrs.st_mode):
        canonical = sftp.normalize(remote)
        if canonical in ancestors:
            raise ValueError(f'Remote directory link cycle: {remote}')
        local.mkdir()
        for name in sftp.listdir(remote):
            if name in ('', '.', '..') or '/' in name:
                raise ValueError(f'Invalid SFTP directory entry under {remote}')
            download(sftp, posixpath.join(remote, name), local / name, (*ancestors, canonical))
    else:
        raise ValueError(f'Expected a regular file or directory: {remote}')


def pull(sftp, path, remote=None, *, download_to=download):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.pull-', dir=path.parent) as temporary:
        staged = Path(temporary) / 'incoming'
        remote = path if remote is None else remote
        print(f'Pulling {remote} to {path}', flush=True)
        download_to(sftp, str(remote), staged)
        # Serialize publication of completed downloads in this directory.
        parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            fcntl.flock(parent_fd, fcntl.LOCK_EX)
            if path.is_symlink() or (path.exists() and staged.is_dir() != path.is_dir()):
                raise ValueError(f'Destination type differs from source: {path}')
            if path.is_dir():
                previous = Path(temporary) / 'previous'
                path.rename(previous)
                try:
                    staged.rename(path)
                except BaseException:
                    previous.rename(path)
                    raise
            else:
                staged.replace(path)
        finally:
            os.close(parent_fd)
    print(f'Received {path}', flush=True)


@contextmanager
def connection(config, role):
    import paramiko
    if config.stat().st_mode & 0o077:
        raise ValueError(f'Settings contain SSH credentials. Set permissions with chmod 600 {config}')
    with config.open() as stream:
        endpoint = json.load(stream)[role]
    # Reuse OpenSSH's known keys. Remember new hosts alongside the private settings.
    known = config.with_name('transfer.known_hosts')
    known.touch(mode=0o600, exist_ok=True)
    with paramiko.SSHClient() as client:
        client.load_system_host_keys()
        client.load_host_keys(str(known))
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        password = endpoint.get('password') or None
        client.connect(endpoint['host'], port=int(endpoint.get('port', 22)), username=endpoint['user'],
                       password=password, look_for_keys=not password, allow_agent=not password, timeout=20)
        yield client


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare', action='store_true', help='on the producer, grant mode 0777 to the selected run artifacts')
    mode.add_argument('--init', nargs='+', metavar='HOST',
                        help='create private SSH settings: [GOLDEN_HOST] AIR_HOST OMC_HOST')
    parser.add_argument('--from', dest='source', choices=('air', 'omc'), default='air')
    parser.add_argument('--to', type=Path, help='local destination for a single remote artifact')
    parser.add_argument('paths', nargs='*', help='absolute artifact paths or paths relative to AI_RUN_DIR')
    args = parser.parse_args()
    config = args.config.expanduser().absolute()
    if args.to is not None and (len(args.paths) != 1 or args.prepare or args.init):
        parser.error('--to requires a single remote artifact')
    if args.init:
        if args.paths:
            parser.error('--init takes two hostnames and no artifact paths')
        if len(args.init) not in (2, 3):
            parser.error('--init expects [GOLDEN_HOST] AIR_HOST OMC_HOST')
        initialize(config, args.init)
        return
    if not args.paths:
        parser.error('supply the files or directories to receive')
    if args.prepare:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))
        from qwen35_moe_dflash.common.artifacts import share_for_transfer
        root = Path(os.environ.get('AI_RUN_DIR', ''))
        if not root.is_absolute():
            parser.error('--prepare requires an absolute AI_RUN_DIR')
        for name in args.paths:
            path = destination(name, [root])
            if path.resolve() == config.resolve() or path.resolve() in config.resolve().parents:
                parser.error('selected output includes the private SSH settings')
            share_for_transfer(path)
            print(f'Transfer permissions ready: {path}')
        return
    roots = []
    for variable in ('AI_RUN_DIR', 'REPO_DIR'):
        value = os.environ.get(variable)
        if not value or not Path(value).is_absolute():
            raise ValueError(f'Set {variable} to its absolute path on both hosts')
        roots.append(Path(value))
    paths = [destination(name, roots) for name in args.paths]
    targets = paths if args.to is None else [destination(args.to, roots)]
    if any(path == config or path in config.parents for path in [*paths, *targets]):
        raise ValueError('The destination includes the private transfer settings')
    with connection(config, args.source) as client:
        with client.open_sftp() as sftp:
            sftp.get_channel().settimeout(120)
            for path, target in zip(paths, targets):
                pull(sftp, target, path)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        sys.exit(f'transfer failed: {error}')
