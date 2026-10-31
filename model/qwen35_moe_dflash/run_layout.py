"""Relocate runtime assets and retire completed tests without following old audits."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil

from .runtime_config import read_json

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
    for relative in ('result.json', 'board/result.json', 'board/checkpoint.json', 'manifest.json'):
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
    return (folder.name.startswith(('dflash-', 'draft-s64-profile-', 'target-resident-load-probe-'))
            or any(r.get('artifact_kind') in {'qwen35-board-serial-dflash-job',
                'qwen35-board-serial-dflash-execution', 'qwen35-original-target-tool-execution'}
                   for r in _reports(folder)))


def _role(name):
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
        if not any('/targets/cpro/scripts/' in a and Path(a).name.startswith(
                ('run_', 'compile_', 'export_', 'profile_', 'prepare_')) for a in scripts):
            continue
        if any(str(root).encode() in a for a in args):
            found.append(int(path.parent.name))
    return found


def plan_layout(root, config_path, *, clean_completed=False):
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    if not root.is_dir() or config_path.parent != root:
        raise ValueError('runtime config must be directly inside the existing run root')
    active = _active_processes(root)
    if active:
        raise ValueError('local model jobs are still running; finish them before moving assets: ' + str(active))
    paths = _runtime_paths(config_path)
    protected = {p.relative_to(root).parts[0] for p in paths if p.is_relative_to(root)}
    roles = {}
    config = read_json(config_path)
    for key, value in _refs(config):
        path = (root / value).resolve()
        if path.is_relative_to(root) and path != root:
            role = key[1] if key[0] == 'models' else 'common'
            top = path.relative_to(root).parts[0]
            if top in roles and roles[top] != role:
                roles[top] = 'common'
            else:
                roles[top] = role
    folders = [p for p in root.iterdir() if p.is_dir() and not p.is_symlink() and p.name not in ZONES]
    folders += [p for p in (root / 'work').iterdir() if p.is_dir() and not p.is_symlink()] if (root / 'work').is_dir() else []
    reports = {p: list(_reports(p)) for p in folders}
    closed = {r['board_attempt'] for rows in reports.values() for r in rows if _closed(r) and r.get('board_attempt')}
    unresolved = {r['board_attempt'] for rows in reports.values() for r in rows
                  if r.get('board_attempt') and r['board_attempt'] not in closed}
    moves, delete, keep = [], [], []
    for path in sorted(root.iterdir()):
        name = path.name
        if name in ZONES or name.startswith('.') or path in (config_path, root / config['assets']):
            continue
        if path.is_symlink():
            keep.append(dict(path=name, reason='symlink; retain without following'))
            continue
        if path.is_dir() and _test(path):
            rows = reports[path]
            completed = any(_closed(r) or r.get('board_attempt') in closed for r in rows)
            host_only = _host_only(rows)
            if name in protected or not (completed or host_only):
                keep.append(dict(path=name, reason='runtime dependency or completion unconfirmed'))
            elif clean_completed:
                delete.append(name)
            else:
                moves.append(dict(source=name, destination='work/' + name))
            continue
        role = roles.get(name) or _role(name)
        if role and not unresolved:
            moves.append(dict(source=name, destination=role + '/' + name))
        else:
            keep.append(dict(path=name, reason='pending device request; assets stay in place' if role else 'unknown purpose'))
    if clean_completed:
        for path in folders:
            if path.parent != root / 'work':
                continue
            rows = reports[path]
            if any(_closed(r) or r.get('board_attempt') in closed for r in rows):
                if not any(p == path or path in p.parents for p in paths):
                    delete.append(str(path.relative_to(root)))
    for row in moves:
        if os.path.lexists(root / row['destination']):
            raise FileExistsError('destination already exists: ' + row['destination'])
    csv_bytes = sum(p.stat().st_size for name in delete for p in (root / name).rglob('*.csv')
                    if p.is_file() and not p.is_symlink())
    if shutil.disk_usage(root).free < csv_bytes + (1 << 20):
        raise OSError('insufficient space to save profiling CSV and compact results before cleanup')
    return dict(schema_version=1, original_root=str(root), config=config_path.name,
        status='PLANNED', moves=moves, delete=delete, keep=keep,
        retained_csv_bytes=csv_bytes,
        unresolved_board_attempts=sorted(unresolved), completed_moves=[], completed_deletions=[])


def _summary(root, source, report):
    destination = root / 'results' / source.replace('/', '__')
    if destination.is_symlink() or (root / 'results').is_symlink():
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


def organize(root, config_path, *, apply=False, clean_completed=False):
    root = Path(root).resolve()
    journal = root / JOURNAL
    prior = read_json(journal) if journal.exists() else None
    if prior and prior['status'] == 'MOVING':
        plan = prior
    else:
        plan = plan_layout(root, config_path, clean_completed=clean_completed)
        # Keep all past aliases for AIR/build references, including after a new cleanup.
        if prior:
            plan['original_root'] = prior['original_root']
            plan['moves'] = prior['moves'] + plan['moves']
            plan['completed_moves'] = [r['source'] for r in prior['moves']]
    if not apply:
        return plan
    for row in plan['moves']:
        src, dst = Path(row['source']), Path(row['destination'])
        if (src.is_absolute() or len(src.parts) != 1 or src.name in ZONES | {'.', '..'}
                or dst.is_absolute() or len(dst.parts) != 2 or dst.parts[0] not in ZONES
                or dst.name != src.name):
            raise ValueError('invalid directory move in journal')
    for name in plan['delete']:
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or path.name in ZONES or not (len(path.parts) == 1 or
                len(path.parts) == 2 and path.parts[0] == 'work'):
            raise ValueError('invalid cleanup path in journal')
    if _active_processes(root):
        raise ValueError('local model job started; retry after it finishes')
    for name in ZONES:
        path = root / name
        if path.is_symlink() or path.exists() and not path.is_dir():
            raise ValueError('layout zone is not a regular directory: ' + name)
        path.mkdir(exist_ok=True)
    config_path = root / plan['config']
    if 'original_config' not in plan:
        plan['original_config'] = read_json(config_path)
    plan['status'] = 'MOVING'; _save(journal, plan)
    for row in plan['moves']:
        if row['source'] in plan['completed_moves']:
            continue
        source, destination = root / row['source'], root / row['destination']
        if source.exists():
            if destination.exists():
                raise FileExistsError(str(destination))
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
        elif not destination.exists():
            raise FileNotFoundError(str(source))
        plan['completed_moves'].append(row['source']); _save(journal, plan)
    config = deepcopy(plan['original_config'])
    for keys, value in _refs(config):
        old = Path(value)
        absolute = old if old.is_absolute() else Path(plan['original_root']) / old
        new = relocated_path(absolute, root)
        parent = config
        for key in keys[:-1]:
            parent = parent[key]
        parent[keys[-1]] = os.path.relpath(new, root)
    _save(config_path, config)
    for path in _runtime_paths(config_path):
        if not path.exists():
            raise FileNotFoundError('runtime dependency missing after move: ' + str(path))
    for relative in plan['delete']:
        if relative in plan['completed_deletions']:
            continue
        path = root / relative
        if any(p == path or path in p.parents for p in _runtime_paths(config_path)):
            raise ValueError('cleanup would remove a current runtime dependency: ' + relative)
        if path.exists():
            archived = plan.setdefault('archived_tests', [])
            if (path.is_symlink() or path.parent not in {root, root / 'work'}
                    or relative not in archived and not _test(path)):
                raise ValueError('cleanup path is no longer an owned test directory: ' + relative)
            if relative not in archived:
                _summary(root, relative, {})
                archived.append(relative); _save(journal, plan)
            shutil.rmtree(path)
        plan['completed_deletions'].append(relative); _save(journal, plan)
    plan['status'] = 'COMPLETE'; _save(journal, plan)
    return plan


def reuse_test_output(output, *, config=None, recovery_source=None):
    """Reuse a named test slot only after completion, never as a process reset."""
    path = Path(output).absolute()
    if not path.exists():
        return
    root = Path(os.environ['AI_RUN_DIR']).resolve()
    if path.is_symlink() or path.parent not in {root, root / 'work'}:
        raise ValueError('reusable test outputs must be directly in AI_RUN_DIR or AI_RUN_DIR/work')
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
