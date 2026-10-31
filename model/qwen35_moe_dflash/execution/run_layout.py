"""Relocate runtime assets and retire completed tests without following old audits."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

from qwen35_moe_dflash.execution.runtime_config import read_json

JOURNAL = '.run-layout.json'
ZONES = {'prefill', 'decode', 'draft', 'common', 'work', 'results'}
RELEASED = {'NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT',
            'NO_OWNED_LOCK_AFTER_KNOWN_COMPLETION'}


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def relocated_path(path, anchor):
    """Resolve old build references through a move journal, without editing AIR metadata."""
    original = Path(path)
    for root in (Path(anchor).absolute(), *Path(anchor).absolute().parents):
        journal = root / JOURNAL
        if not journal.is_file():
            continue
        layout = read_json(journal)
        old = Path(layout['original_root'])
        try:
            relative = original.relative_to(old)
        except ValueError:
            return original
        for row in layout['moves']:
            try:
                tail = relative.relative_to(row['source'])
            except ValueError:
                continue
            destination = root / row['destination'] / tail
            return destination if destination.exists() else original
        return root / relative
    return original


def _refs(config):
    yield ('assets',), config['assets']
    for role, model in config['models'].items():
        key = 'directory' if 'directory' in model else 'path'
        yield ('models', role, key), model[key]
    for name, row in config['embedding'].items():
        yield ('embedding', name, 'path'), row['path']
    if 'tokenizer' in config:
        yield ('tokenizer', 'directory'), config['tokenizer']['directory']


def _runtime_paths(config_path):
    config = read_json(config_path)
    return [(config_path.parent / value).resolve() for _, value in _refs(config)]


def _reports(folder):
    for relative in ('result.json', 'board/result.json', 'board/checkpoint.json', 'manifest.json', 'report.json'):
        path = folder / relative
        if path.is_file() and not path.is_symlink():
            try:
                yield read_json(path)
            except (OSError, ValueError):
                pass


def _closed(row):
    if row.get('uncertain') or row.get('active_child_count', 0):
        return False
    return (row.get('lock_state') in RELEASED or
            row.get('lock_state') == 'NO_KNOWN_OWNED_LOCK'
            and row.get('device_execution') == 'NOT_RUN'
            and not row.get('launch_requested'))


def _host_only(rows):
    return bool(rows) and all(r.get('device_execution') == 'NOT_RUN'
        and not r.get('launch_requested') and not r.get('board_attempt')
        and r.get('lock_state') in {None, 'NO_KNOWN_OWNED_LOCK', *RELEASED} for r in rows)


def _test(folder):
    # Only recognized execution families are eligible for automatic removal.
    return (folder.name.startswith(('dflash-', 'draft-s64-profile-', 'target-resident-load-probe-',
                                    'prefill64-embedding-row-check'))
            or any(r.get('artifact_kind') in {'qwen35-board-serial-dflash-job',
                'qwen35-board-serial-dflash-execution', 'qwen35-original-target-tool-execution'}
                   for r in _reports(folder)))


def _role(name):
    if name == 'deleted-target-compile-stop':
        return 'work'
    if name.startswith(('target-prefill', 'target-pytorch-prefill')):
        return 'prefill'
    if name.startswith(('target-decode', 'target-pytorch-decode')):
        return 'decode'
    if name == 'full-draft' or name.startswith(('draft-kv-', 'draft-capacity-')):
        return 'draft'
    if name.startswith('real-feature-draft-'):
        return 'draft'
    if name.startswith('prefill64-'):
        return 'prefill'
    if name in {'weights', 'original-embedding-tables', 'original-tokenizer', 'tokenizer-offline',
                'qwen35-tokenizer-offline.tar.gz'}:
        return 'common'
    if name.startswith(('target-ge', 'target-pytorch-', 'target-board-tool-', 'fused-all-')):
        return 'common'
    return None


def _active_processes(root):
    found = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = path.read_bytes().split(b'\0')
        except OSError:
            continue
        if int(path.parent.name) == os.getpid():
            continue
        # Ignore editors and this organizer; model runners/compilers retain their inputs.
        scripts = [v.decode(errors='replace') for v in args[:3]]
        if not any(('targets/cpro/scripts/' in a or 'targets/cpro/tools/' in a) and Path(a).name.startswith(
                ('run_', 'compile_', 'export_', 'profile_', 'prepare_', 'build_')) for a in scripts):
            continue
        if any(str(root).encode() in a for a in args):
            found.append(int(path.parent.name))
    return found


def _summary(root, source, report):
    parts = Path(source).parts
    base = root / parts[0] / 'results' if parts[0] in {'fused', 'draft', 'prefill', 'decode'} else root / 'results'
    destination = base / parts[-1]
    if destination.is_symlink() or base.is_symlink():
        raise ValueError('results directory must not be a symlink')
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    keys = ('status', 'stage', 'error', 'device_execution', 'numerical_validation', 'board_attempt',
            'lock_state', 'generated_token_ids', 'generated_text', 'counts', 'acceptance',
            'phase_latency', 'native_api_latency', 'supervisor_timing', 'draft_kv')
    reports = list(_reports(root / source))
    selected = report or next((r for r in reports if _closed(r)), reports[0] if reports else {})
    compact = {k: selected[k] for k in keys if k in selected}
    compact['source_directory'] = source
    _save(destination / 'result.json', compact)
    for path in (root / source).rglob('*.csv'):
        if path.is_file() and not path.is_symlink() and not any(p.is_symlink() for p in path.parents):
            output = destination / 'profiling' / path.relative_to(root / source)
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, output)


def reuse_test_output(output, *, config=None, recovery_source=None):
    """Reuse a named test slot only after completion, never as a process reset."""
    path = Path(output).absolute()
    if not path.exists():
        return
    root = Path(os.environ['AI_RUN_DIR']).resolve()
    allowed = {root, root / 'work', root / 'fused',
               *(root / role / 'tests' for role in ('draft', 'prefill', 'decode'))}
    if path.is_symlink() or path.parent.resolve() not in allowed:
        raise ValueError('reusable test outputs must be in fused/ or draft/prefill/decode/tests/')
    if _active_processes(root):
        raise ValueError('another local model job is running; output retained')
    rows = list(_reports(path))
    if not rows or any(r.get('artifact_kind') not in {None, 'qwen35-board-serial-dflash-job',
            'qwen35-board-serial-dflash-execution', 'qwen35-draft-operator-profile'} for r in rows):
        raise ValueError('output is not a recognized test request; retained: ' + str(path))
    protected = _runtime_paths(Path(config).resolve()) if config else []
    if any(p == path or path in p.parents for p in protected):
        raise ValueError('output contains runtime assets; retained')
    attempts = {r['board_attempt'] for r in rows if r.get('board_attempt')}
    siblings = [p for p in path.parent.iterdir() if p.is_dir() and not p.is_symlink()]
    proofs = [r for p in siblings for r in _reports(p) if _closed(r)]
    closed = {r['board_attempt'] for r in proofs if r.get('board_attempt')}
    host_only = _host_only(rows)
    readonly_recovery = recovery_source is not None and Path(recovery_source).exists() and all(
        r.get('recovery_model_execution') == 'FORBIDDEN'
        and Path(r.get('recovery_source', '')).absolute() == Path(recovery_source).absolute() for r in rows)
    if not (host_only or any(_closed(r) for r in rows) and not attempts
            or attempts and attempts <= closed or readonly_recovery):
        raise ValueError('previous attempt still running or exit unknown; recover it before reusing: ' + str(path))
    # One latest summary per test, overwritten by the next completed attempt.
    proof = next((r for r in proofs if r.get('board_attempt') in attempts), {})
    _summary(root, str(path.relative_to(root)), proof)
    shutil.rmtree(path)
