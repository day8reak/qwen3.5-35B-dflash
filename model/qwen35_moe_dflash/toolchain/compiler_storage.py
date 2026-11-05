"""Release disposable compiler copies after validating a self-contained OMC.

Callers hold the candidate's writer lock. Reports, diagnostics, packed models
and original export inputs remain in place; this never contacts a device.
"""
import os
from pathlib import Path
import shutil
import stat

from qwen35_moe_dflash.common.hashing import sha256_file
from qwen35_moe_dflash.execution.runtime_config import read_json
from qwen35_moe_dflash.execution.run_layout import _save
from .omc_container import read_container

SCRATCH = ('compilation/compile-work', 'compilation/tbe-compat/wheel-python')


def require_idle(root):
    """Reject live local processes using this candidate, including vendor workers."""
    prefix = str(root).encode()
    for process in Path('/proc').glob('[0-9]*'):
        if int(process.name) == os.getpid():
            continue
        try:
            args = (process/'cmdline').read_bytes().split(b'\0')
        except FileNotFoundError:
            continue  # Exited process.
        except PermissionError:
            raise ValueError('cannot inspect local process ownership; scratch retained') from None
        try:
            cwd = (process/'cwd').resolve(strict=True)
        except FileNotFoundError:
            cwd = Path('/')  # Missing/deleted cwd does not hide command references.
        except PermissionError:
            raise ValueError('cannot inspect local process cwd; scratch retained') from None
        if (cwd == root or root in cwd.parents or any(prefix in arg for arg in args)):
            raise ValueError(f'local process {process.name} references candidate; scratch retained')


def _plain(path, *, directory=False):
    if path.resolve() != path or path.is_symlink():
        raise ValueError(f'symlink path is not cleanup-owned: {path}')
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f'expected {"directory" if directory else "file"}: {path}')


def _tree_bytes(root):
    """Count allocated bytes once per inode, without following scratch links."""
    _plain(root, directory=True)
    if root.is_mount():
        raise ValueError(f'scratch root is a mount; retained: {root}')
    device, seen, size = root.stat().st_dev, set(), 0
    for base, dirs, files in os.walk(root, followlinks=False):
        for name in dirs+files:
            path = Path(base)/name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                continue  # rmtree unlinks this entry; never follows its target.
            if (info.st_dev != device or path.is_mount() or
                    not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode))):
                raise ValueError(f'unexpected scratch entry; retained: {path}')
            key = (info.st_dev, info.st_ino)
            if stat.S_ISREG(info.st_mode) and key not in seen:
                size += info.st_blocks*512
                seen.add(key)
    return size


def release_packed_scratch(root, *, apply=False, protected=()):
    root = Path(root).absolute()
    _plain(root, directory=True)
    require_idle(root)
    for relative in ('model.omc', 'weight-packing.json', 'compilation/report.json'):
        _plain(root/relative)
    packing = read_json(root/'weight-packing.json')
    build = read_json(root/'compilation/report.json', 32 << 20)
    model = root/'model.omc'
    source = root/'compilation'/build.get('omc', {}).get('path', '')
    if (packing.get('artifact_kind') != 'qwen35-omc-embedded-stream'
            or packing.get('status') != 'EMBEDDED' or build.get('status') != 'COMPILED'
            or build.get('model_compilation') != 'PASSED' or build.get('returncode') != 0
            or root/'compilation/compile-work' not in source.parents
            or source.resolve() != source
            or packing.get('source', {}).get('path') != str(source)
            or packing['source'].get('sha256') != build['omc'].get('sha256')
            or packing.get('output', {}).get('path') != str(model)
            or packing['output'].get('bytes') != model.stat().st_size):
        raise ValueError('completed compiler and embedded model binding required; scratch retained')
    before = model.stat()
    digest = sha256_file(model)
    parts = read_container(model)['partitions']
    if (digest != packing['output'].get('sha256') or [p['type'] for p in parts] != [0, 1, 2]
            or parts[1]['bytes'] <= 0
            or [(p['type'], p['bytes']) for p in parts] !=
               [(p['type'], p['bytes']) for p in packing.get('partitions', [])]):
        raise ValueError('packed OMC content/embedded weight partition differs; scratch retained')
    selected = []
    for relative in SCRATCH:
        path = root/relative
        if not path.exists() and not path.is_symlink():
            continue
        if any(asset == path or path in asset.parents or asset in path.parents for asset in protected):
            raise ValueError(f'runtime configuration references scratch: {path}')
        selected.append(dict(path=relative, allocated_bytes=_tree_bytes(path)))
    result = dict(status='PLANNED', model_sha256=digest, directories=selected,
        allocated_bytes_selected=sum(r['allocated_bytes'] for r in selected),
        scope='COMPLETED_PACKED_COMPILER_COPIES_ONLY; REPORTS_MODELS_AND_SOURCE_EXPORT_RETAINED',
        limitation='Hardlinks/reflinks can reduce actual freed space; historical inventories describe removed copies.')
    if apply and selected:
        require_idle(root)
        now = model.stat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                now.st_dev, now.st_ino, now.st_size, now.st_mtime_ns, now.st_ctime_ns):
            raise ValueError('packed model changed during cleanup checks')
        free = shutil.disk_usage(root).free
        result.update(status='REMOVING', removed=[])
        _save(root/'compilation/storage-retention.json', result)
        for row in selected:
            shutil.rmtree(root/row['path'])
            result['removed'].append(row['path'])
            _save(root/'compilation/storage-retention.json', result)
        result.update(status='CLEANED', free_bytes_after=shutil.disk_usage(root).free,
                      observed_free_bytes_change=shutil.disk_usage(root).free-free)
        _save(root/'compilation/storage-retention.json', result)
    elif apply:
        result['status'] = 'ALREADY_CLEAN'
    return result
