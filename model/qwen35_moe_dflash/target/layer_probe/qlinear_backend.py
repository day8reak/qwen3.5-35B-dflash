"""Read-only backend evidence; never load a candidate library or alter dispatch."""
import inspect
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import torch

from .common import sha


def loaded_libraries(maps):
    """Include custom opapi and lazily loaded feature libraries, preserving paths."""
    libraries = set()
    for line in maps.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith('/'):
            continue
        path = fields[5].removesuffix(' (deleted)')
        name = Path(path).name
        if name.startswith(('libopapi', 'libcust_opapi', 'libaclnn', 'libascendcl', 'libtorch_npu')):
            libraries.add(path)
    return sorted(libraries)


def runtime(npu):
    record = dict(python=sys.executable, python_version=sys.version, torch_version=torch.__version__,
        torch_npu_version=getattr(npu, '__version__', 'unknown'), torch_npu_file=getattr(npu, '__file__', None),
        uid=os.getuid(), euid=os.geteuid(), loaded_libraries=[],
        environment={k: os.environ.get(k) for k in ('CONDA_DEFAULT_ENV', 'CONDA_PREFIX',
            'ASCEND_HOME_PATH', 'ASCEND_OPP_PATH', 'ASCEND_CUSTOM_OPP_PATH', 'LD_LIBRARY_PATH',
            'ASCEND_RT_VISIBLE_DEVICES', 'ASCEND_LAUNCH_BLOCKING',
            'ASCEND_GLOBAL_LOG_LEVEL', 'ASCEND_MODULE_LOG_LEVEL', 'ASCEND_SLOG_PRINT_TO_STDOUT')})
    try:
        # The capture hook uses functools.wraps; its wrapper has no _schemas attribute.
        op = inspect.unwrap(npu.npu_quant_matmul)
        record['quant_matmul_schema'] = str(getattr(op, '_schemas', 'unavailable'))
    except Exception as exc:
        record['schema_error'] = f'{type(exc).__name__}: {exc}'
    try:
        record['loaded_libraries'] = loaded_libraries(Path('/proc/self/maps').read_text())
    except OSError as exc:
        record['maps_error'] = f'{type(exc).__name__}: {exc}'
    return record


def tensor_formats(npu, tensors):
    """NPU descriptor only: does not infer the manual I4ZN ordering of W4 bytes."""
    records = {}
    for name, value in tensors.items():
        if not isinstance(value, torch.Tensor):
            continue
        if value.device.type != 'npu':
            records[name] = dict(status='NOT_NPU', device=str(value.device))
            continue
        try:
            records[name] = dict(status='READ', format=int(npu.get_npu_format(value)))
        except Exception as exc:
            records[name] = dict(status='UNAVAILABLE', error=f'{type(exc).__name__}: {exc}')
    return records


def exported_symbols(text):
    """Parse defined, externally visible ELF symbols, excluding UND imports."""
    names = set()
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 8 or fields[4] not in ('GLOBAL', 'WEAK'):
            continue
        if fields[5] not in ('DEFAULT', 'PROTECTED') or fields[6] == 'UND':
            continue
        name = fields[7].split('@', 1)[0]
        if re.fullmatch(r'aclnn(?:QuantMatmul|TransData)\w*', name):
            names.add(name)
    return sorted(names)


def library_record(path, mapped, readelf):
    row = dict(path=str(path), resolved_path=str(path.resolve()), mapped=str(path.resolve()) in mapped)
    try:
        stat = path.stat()
        row.update(bytes=stat.st_size, owner_uid=stat.st_uid, owner_gid=stat.st_gid,
                   readable=os.access(path, os.R_OK), sha256=sha(path))
        uid = os.getuid()
        # Source comparison only, NOT proof that this installed binary applies this check.
        row['owner_compatible_with_op_plugin_7_1'] = stat.st_uid == uid or (uid != 0 and stat.st_uid == 0)
        if readelf is None:
            row.update(symbol_status='UNAVAILABLE', symbol_error='readelf is not on PATH')
            return row
        completed = subprocess.run([readelf, '--dyn-syms', '--wide', str(path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors='replace', timeout=10)
        if completed.returncode:
            row.update(symbol_status='FAILED', symbol_error=completed.stderr[-2000:])
            return row
        symbols = exported_symbols(completed.stdout)
        row.update(symbol_status='READ', exported_symbols=symbols,
            defines_v5_pair=all(s in symbols for s in ('aclnnQuantMatmulV5', 'aclnnQuantMatmulV5GetWorkspaceSize')))
    except Exception as exc:
        row.update(symbol_status='UNAVAILABLE', symbol_error=f'{type(exc).__name__}: {exc}')
    return row


def inventory(runtime_record):
    """Scope inspection to declared vendor opapi files and already mapped API libs.

    No dlopen/dlsym: probing a candidate must not initialize it or change lookup order.
    Symbol definitions and mappings do not identify cached dispatch function pointers.
    """
    env = runtime_record['environment']
    candidates = {}
    def add(path, source):
        candidates.setdefault(Path(path), []).append(source)
    entries = env.get('ASCEND_CUSTOM_OPP_PATH') or ''
    for i, entry in enumerate(entries.split(':')):
        if entry:  # An empty entry is not permission to scan /op_api or the cwd.
            add(Path(entry)/'op_api/lib/libcust_opapi.so', f'ASCEND_CUSTOM_OPP_PATH[{i}]')
    result = dict(scope='READ_ONLY_LIBRARY_INVENTORY; NOT_DISPATCH_BINDING_PROOF', libraries=[])
    opp = env.get('ASCEND_OPP_PATH')
    if opp:
        config = Path(opp)/'vendors/config.ini'
        try:
            for line in config.read_text().splitlines():
                if line.startswith('load_priority='):
                    result['default_vendor_priority'] = line
                    for vendor in line.split('=', 1)[1].split(','):
                        vendor = vendor.strip()
                        if vendor and Path(vendor).name == vendor and vendor not in ('.', '..'):
                            add(config.parent/vendor/'op_api/lib/libcust_opapi.so', 'vendors/config.ini')
                    break
        except OSError as exc:
            result['default_vendor_config_error'] = str(exc)
    mapped = {str(Path(p).resolve()) for p in runtime_record['loaded_libraries']}
    for path in runtime_record['loaded_libraries']:
        if Path(path).name.startswith(('libopapi', 'libcust_opapi', 'libaclnn')):
            add(path, 'PROCESS_MAPS')
    readelf = shutil.which('readelf')
    # Avoid repeatedly hashing/reading the same symlink target, retain each declared origin.
    examined = {}
    for path, origins in candidates.items():
        real = str(path.resolve())
        if real not in examined:
            examined[real] = library_record(path, mapped, readelf)
        result['libraries'].append(dict(examined[real], path=str(path), origins=origins))
    return result
