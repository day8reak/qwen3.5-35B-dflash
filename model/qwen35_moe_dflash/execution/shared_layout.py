"""Select current shared model assets, then retire superseded job directories."""
from copy import deepcopy
import os
from pathlib import Path
import shutil

from .runtime_config import read_json, validate_assets, file_record
from .run_layout import (_save, _active_processes, _reports, _closed, _host_only,
                         _role, _test, relocated_path)

JOURNAL = '.shared-layout.json'
ROLES = ('prefill', 'decode', 'draft')
RESERVED = {'air', 'omc', 'tests', 'source', 'results'}


def _source_path(path, root):
    """Read old aliases through the previous move journal, without retargeting them.

    resolve(strict=True) cannot follow a borrowed weight link after its source
    job moved. Relocate each link target before resolving the next component.
    Only recorded moves are used; a similar basename is never a substitute.
    """
    path = Path(os.path.abspath(path))
    seen = set()
    for _ in range(64):
        if path in seen:
            raise ValueError('cyclic artifact reference: ' + str(path))
        seen.add(path)
        if not path.exists():
            moved = relocated_path(path, root, include_missing=True)
            if moved != path:
                path = moved
                continue
        for component in (*reversed(path.parents), path):
            if component.is_symlink():
                target = component.parent / os.readlink(component)
                path = Path(os.path.abspath(target / path.relative_to(component)))
                break
        else:
            return path.resolve(strict=True)
    raise ValueError('too many artifact link/move redirects: ' + str(path))


def _relative(path, root):
    path = Path(path).absolute()
    if path == root or not path.is_relative_to(root):
        raise ValueError('shared assets must be inside AI_RUN_DIR: ' + str(path))
    return path.relative_to(root).as_posix()


def _check_path(root, relative):
    p = Path(relative)
    if p.is_absolute() or '..' in p.parts or not p.parts:
        raise ValueError('invalid migration path: ' + str(relative))
    path = root / p
    if any(x.is_symlink() for x in (path, *path.parents) if x.is_relative_to(root)):
        raise ValueError('migration destination/source must not traverse symlinks: ' + str(path))
    return path


def _compiler_report(directory):
    for parent in (directory, *directory.parents):
        report = parent / 'compilation/report.json'
        if report.is_file():
            return report
        if parent.name == 'compilation' and (parent / 'report.json').is_file():
            return parent / 'report.json'
    return None


def plan(root, config_path, *, draft_air=None, clean_completed=False, confirm_idle=False, closed_attempts=()):
    root = Path(root).resolve()
    if _active_processes(root):
        raise ValueError('finish active local model jobs before reorganizing')
    config_path = _source_path(config_path, root)
    config = read_json(config_path)
    asset_file = _source_path(config_path.parent / config['assets'], root)
    assets = read_json(asset_file)
    validate_assets(assets)
    moves, files, writes, retained = [], {}, [], []
    selected = set()
    new_config = deepcopy(config)
    new_config['assets'] = 'runtime-assets.json'
    new_config['native_directory'] = 'native'
    new_assets = deepcopy(assets)

    def select(source, destination):
        # Existing compiler input aliases are resolved once. Only the owned
        # source file moves; old aliases disappear with the retired build job.
        source = _source_path(source, root)
        src, dst = _relative(source, root), str(destination)
        _check_path(root, dst)
        selected.add(src)
        if src == dst:
            return
        if os.path.lexists(root / dst):
            raise FileExistsError('destination already exists: ' + dst)
        if src in files:
            moves.append(dict(source=files[src], destination=dst, action='link'))
        else:
            files[src] = dst
            moves.append(dict(source=src, destination=dst, action='move'))

    def small(path, value):
        writes.append(dict(path=str(path), value=value))

    for role in ('prefill', 'decode'):
        model = config['models'][role]
        old_root = _source_path(config_path.parent / model['directory'], root)
        destination = Path(role) / 'omc/current'
        for row in assets[role]['payloads']:
            relative = Path(row['path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('invalid model payload path')
            source = _source_path(old_root / relative, root)
            file_record(source, size=row['bytes'])
            select(source, destination / relative)
        new_config['models'][role]['directory'] = '../' + str(destination)
        # Keep only the AIR and source files actually selected for this model.
        receipt = _compiler_report(old_root)
        if receipt:
            report = read_json(receipt)
            air_root = Path(role) / 'air/current'
            snapshots = report.get('input_snapshots', [])
            primary = [r for r in snapshots if r.get('role') == 'air']
            if len(primary) != 1:
                raise ValueError(role + ' current compiler receipt has no unique AIR')
            for row in snapshots:
                relative = Path(row['path']).relative_to('compile-work/generated')
                select(receipt.parent / row['path'], air_root / relative)
            small(air_root / 'build.json', dict(
                air=str(Path(primary[0]['path']).relative_to('compile-work/generated')),
                input_layout='air-directory', ddk_id=report.get('ddk_id'),
                ddk_root=report.get('ddk_root'), sequence_length=64 if role == 'prefill' else 1))
        elif not (root / role / 'air').exists():
            retained.append(dict(path=role + '/air', reason='current AIR source not found; OMC remains runnable'))

    old_draft = _source_path(config_path.parent / config['models']['draft']['path'], root)
    select(old_draft, 'draft/omc/current/model.omc')
    new_config['models']['draft']['path'] = '../draft/omc/current/model.omc'
    for key, row in config['embedding'].items():
        source = _source_path(config_path.parent / row['path'], root)
        destination = Path('common/embedding') / source.name
        file_record(source, size=row['bytes'])
        select(source, destination)
        new_config['embedding'][key]['path'] = '../' + str(destination)
    if config.get('tokenizer'):
        source = _source_path(config_path.parent / config['tokenizer']['directory'], root)
        # Runtime preparation accepts original local tokenizer assets. Keep
        # only this selected source directory, not old generated bundles.
        select(source, 'common/tokenizer')
        new_config['tokenizer']['directory'] = '../common/tokenizer'

    if draft_air:
        from qwen35_moe_dflash.draft.incremental.inputs import bind_incremental_export
        bound = bind_incremental_export(_source_path(draft_air, root))
        meta = deepcopy(bound['manifest'])
        air_root = Path('draft/air') / ('kv-c%d-cap%d-k%d' % tuple(meta[k] for k in
            ('context_rows', 'cache_capacity', 'max_draft_tokens')))
        prepared_root = root / 'draft/source/prepared'
        if meta.get('source_policy') == 'portable-prepared-weights-v1':
            original_prepared = (bound['root'] / meta['source_prepared_root']).resolve()
        else:
            original_prepared = relocated_path(meta['source_prepared_root'], bound['root'])
        select(original_prepared, 'draft/source/prepared')
        select(bound['root'], air_root)
        prepared = read_json(original_prepared / 'prepared.json')
        # The portable AIR depends on its weights, never on a former S64 run.
        source_record = {k: meta['source_prepared'][k] for k in ('bytes', 'sha256')}
        source_record['path'] = os.path.relpath(prepared_root / 'prepared.json', root / air_root)
        meta.update(source_policy='portable-prepared-weights-v1',
            source_prepared_root=os.path.relpath(prepared_root, root / air_root),
            file_constant_root=os.path.relpath(prepared_root, root / air_root),
            source_prepared=source_record, source_bindings=[source_record],
            source_preservation='CURRENT_PREPARED_WEIGHTS_CONTENT_AND_ABI', output='.')
        meta['source_weight_identities'] = [
            dict(path=os.path.relpath(prepared_root / row['path'], root / air_root), identity={})
            for row in prepared['weights'].values()]
        # Preserve selected toolchain options as small metadata.
        bundle = meta.get('source_bundle_manifest', {}).get('path')
        if bundle and not meta.get('compiler'):
            bundle = relocated_path(bundle, bound['root'])
            receipt = bundle.parent.parent / 'compilation/report.json'
            compiler = read_json(receipt)
            meta['compiler'] = {k: compiler[k] for k in ('ddk_root', 'ddk_id')}
        for key in ('source_job', 'source_air_manifest', 'source_bundle_manifest'):
            meta.pop(key, None)
        small(air_root / 'manifest.json', meta)

    small('runtime/runtime-assets.json', new_assets)
    small('runtime/runtime.json', new_config)
    protected = set(selected)
    targets = {m['destination'] for m in moves} | {w['path'] for w in writes}
    candidates = []
    for path in sorted(root.iterdir()):
        if path.name in ROLES + ('common',):
            candidates += sorted(p for p in path.iterdir() if p.name not in RESERVED | {'embedding', 'tokenizer'})
        elif path.name in ('work', 'results'):
            candidates += sorted(path.iterdir())
        elif path.name not in {'runtime', 'fused', '.run-layout.json', JOURNAL}:
            candidates.append(path)
    # The retained operator CSV is a useful result, not another model version.
    csv_names = {}
    for folder in candidates:
        if folder.is_dir() and folder.name.startswith('draft-s64-profile'):
            for csv in sorted(folder.rglob('*.csv')):
                if csv.is_symlink():
                    continue
                if csv.name in csv_names:
                    if csv.read_bytes() != csv_names[csv.name].read_bytes():
                        raise ValueError('different profiling files share a name: ' + csv.name)
                    continue
                csv_names[csv.name] = csv
                destination = root / 'draft/results/profiling' / csv.name
                if destination.exists() and destination.read_bytes() == csv.read_bytes():
                    continue
                select(csv, destination.relative_to(root))
    targets = {m['destination'] for m in moves} | {w['path'] for w in writes}
    confirmed = set(closed_attempts)
    if any(not isinstance(p, str) or not p.startswith('/data/local/tmp/') or '..' in Path(p).parts for p in confirmed):
        raise ValueError('closed attempts must be exact board paths below /data/local/tmp/')
    confirmed.update(r['board_attempt'] for path in candidates if path.is_dir()
                     for r in _reports(path) if _closed(r) and r.get('board_attempt'))
    delete, unresolved = [], set()
    for path in candidates:
        relative = _relative(path, root)
        if relative in targets or path.name.startswith('.'):
            continue
        # Selected source subtrees are removed by the moves. If only some files
        # are selected, the former build directory becomes obsolete afterward.
        if relative in selected:
            continue
        if path.name == 'weights':
            retained.append(dict(path=relative, reason='source weights; not identified as an obsolete build copy'))
            continue
        known = _role(path.name) or _test(path) or path in (config_path, asset_file)
        if not known:
            retained.append(dict(path=relative, reason='unknown purpose'))
            continue
        rows = list(_reports(path)) if path.is_dir() else []
        if not draft_air and any(r.get('status') == 'DRAFT_INCREMENTAL_AIR_EXPORTED' for r in rows):
            raise ValueError('select the current incremental AIR with --draft-air before removing its source jobs')
        if any(r.get('status') == 'DRAFT_INCREMENTAL_OMC_COMPILED_ABI_BOUND' for r in rows):
            retained.append(dict(path=relative, reason='unselected successful incremental build; not deleted'))
            continue
        pending = {r['board_attempt'] for r in rows if r.get('board_attempt') and r['board_attempt'] not in confirmed}
        if pending and not confirm_idle:
            unresolved.update(pending)
            raise ValueError('unfinished request must be recovered before moving its inputs: ' + relative)
        if _test(path) and not (confirm_idle or _host_only(rows) or any(_closed(r)
                or r.get('board_attempt') in confirmed for r in rows)):
            retained.append(dict(path=relative, reason='test completion unconfirmed'))
        elif clean_completed:
            delete.append(relative)
        else:
            retained.append(dict(path=relative, reason='obsolete; add --clean-completed to remove'))
    # Moving a selected directory must not swallow another selected asset.
    # Sort by depth so individual OMC files are extracted before a prepared root.
    moves.sort(key=lambda r: (r['action'] == 'link', -len(Path(r['source']).parts)))
    return dict(schema_version=2, original_root=str(root), status='PLANNED',
        moves=moves, writes=writes, delete=delete, keep=retained,
        config='runtime/runtime.json', draft_air=str(air_root) if draft_air else None,
        cleanup_attempts={name: sorted({r['board_attempt'] for r in _reports(root / name)
                                      if r.get('board_attempt')}) for name in delete},
        unresolved_board_attempts=sorted(unresolved), completed_moves=[], completed_deletions=[])


def organize_shared(root, config_path, *, apply=False, draft_air=None, clean_completed=False,
                    confirm_idle=False, closed_attempts=()):
    root = Path(root).resolve()
    journal = root / JOURNAL
    saved = read_json(journal) if journal.is_file() else None
    if saved and draft_air and not Path(draft_air).exists():
        old = _relative(Path(draft_air), root)
        for row in saved['moves']:
            if row['source'] == old and row['destination'] == saved.get('draft_air'):
                draft_air = root / row['destination']
                break
    if saved and saved['status'] == 'MOVING':
        result = saved
    else:
        result = plan(root, config_path, draft_air=draft_air, clean_completed=clean_completed,
                      confirm_idle=confirm_idle, closed_attempts=closed_attempts)
    if not apply:
        return result
    if _active_processes(root):
        raise ValueError('local model job started; no cleanup performed')
    result['status'] = 'MOVING'
    _save(journal, result)
    for i, row in enumerate(result['moves']):
        if i in result['completed_moves']:
            continue
        source = _check_path(root, row['source'])
        destination = _check_path(root, row['destination'])
        destination.parent.mkdir(parents=True, exist_ok=True)
        if row['action'] == 'move':
            if source.exists():
                if destination.exists():
                    raise FileExistsError(str(destination))
                source.rename(destination)
            elif not destination.exists():
                raise FileNotFoundError(str(source))
        elif row['action'] == 'link':
            if not destination.exists():
                os.link(source, destination)
            elif not os.path.samefile(source, destination):
                raise FileExistsError(str(destination))
        else:
            raise ValueError('invalid migration action')
        result['completed_moves'].append(i)
        _save(journal, result)
    for write in result['writes']:
        _save(_check_path(root, write['path']), write['value'])
    for role in ROLES:
        for part in ('air', 'omc', 'tests'):
            (root / role / part).mkdir(parents=True, exist_ok=True)
    (root / 'runtime/native').mkdir(parents=True, exist_ok=True)
    (root / 'fused').mkdir(exist_ok=True)
    # Validate the new portable references before any removal.
    config = read_json(root / result['config'])
    assets = read_json(root / 'runtime' / config['assets'])
    validate_assets(assets)
    required = []
    for role in ('prefill', 'decode'):
        base = (root / 'runtime' / config['models'][role]['directory']).resolve()
        for row in assets[role]['payloads']:
            file_record(base / row['path'], size=row['bytes'])
            required.append(base / row['path'])
    required.append((root / 'runtime' / config['models']['draft']['path']).resolve())
    for row in config['embedding'].values():
        path = (root / 'runtime' / row['path']).resolve()
        file_record(path, size=row['bytes'])
        required.append(path)
    if config.get('tokenizer'):
        required.append((root / 'runtime' / config['tokenizer']['directory']).resolve())
    for path in required:
        if not path.exists():
            raise FileNotFoundError(str(path))
    if result['draft_air']:
        from qwen35_moe_dflash.draft.incremental.inputs import bind_incremental_export
        bind_incremental_export(root / result['draft_air'])
        required += [root / result['draft_air'], root / 'draft/source/prepared']
    if _active_processes(root):
        raise ValueError('local model job started; assets moved, cleanup deferred')
    for relative in result['delete']:
        if relative in result['completed_deletions']:
            continue
        path = _check_path(root, relative)
        observed = {r['board_attempt'] for r in _reports(path) if r.get('board_attempt')}
        if not observed <= set(result['cleanup_attempts'].get(relative, [])):
            raise ValueError('cleanup slot now belongs to a newer request; retained: ' + relative)
        if any(path == p or path in p.parents for p in required):
            raise ValueError('cleanup overlaps a current asset: ' + relative)
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
        result['completed_deletions'].append(relative)
        _save(journal, result)
    for zone in ('work', 'results'):
        path = root / zone
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    result['status'] = 'PARTIAL' if result['keep'] else 'COMPLETE'
    _save(journal, result)
    return result

