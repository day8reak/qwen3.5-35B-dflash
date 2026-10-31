"""Bind incremental AIR FileConstants to unchanged original prepared weights.

Only individual file aliases are created under the new compiler directory.
No source chmod, hardlink, graph rewrite, or payload copy is performed. Aliases
are not OS write protection; source contents and full identities are audited.
"""
from __future__ import annotations

import os
import math
from pathlib import Path

from .config import Qwen35DFlashConfig
from .draft_capacity_export import _current, _metadata, _stable
from .draft_standalone import CURRENT_DRAFT_EXPORT_CONTRACT
from .ge_export import configurations, weight_identity
from .ge_incremental_draft import build_incremental_draft, incremental_draft_abi
from .ge_recipe_read import read_recipe
from .ge_residual import EXPORT_SUFFIX
from .hashing import sha256_file
from .target_compile_failure import _safe
from .run_layout import relocated_path


def _five(identity):
    return [identity[k] for k in ('device', 'inode', 'bytes', 'mtime_ns', 'ctime_ns')]


def _payload(path, row):
    path, before = _current(path)
    if before['bytes'] != row.get('bytes') or sha256_file(path) != row.get('sha256') or _current(path)[1] != before:
        raise ValueError('incremental file differs from retained identity: ' + str(path))
    return before


def bind_incremental_export(folder):
    root, _ = _safe(folder, directory=True)
    bindings = []
    manifest, identity = _metadata(root / 'manifest.json', bindings)
    if (manifest.get('artifact_kind') != 'qwen35-incremental-draft-air'
            or manifest.get('status') != 'DRAFT_INCREMENTAL_AIR_EXPORTED'
            or manifest.get('export_contract') != CURRENT_DRAFT_EXPORT_CONTRACT + EXPORT_SUFFIX + '-context-kv-v2'
            or manifest.get('file_constant_policy') != 'cpro-relative-int'
            or manifest.get('file_constant_path_base') != 'ORIGINAL_PREPARED_ROOT_EXPLICIT_REFERENCE'
            or manifest.get('residual_policy') != 'cpu-fp32-add'
            or manifest.get('draft_config') != configurations()[1]):
        raise ValueError('requires a successful new six-layer incremental AIR export')
    config = Qwen35DFlashConfig.from_dict(manifest['draft_config'])
    options = {key: manifest[key] for key in ('context_rows', 'cache_capacity', 'max_draft_tokens')}
    graph = build_incremental_draft(config, **options)
    expected_abi = incremental_draft_abi(config, **options)
    if manifest.get('incremental_abi') != expected_abi:
        raise ValueError('incremental host ABI differs from the selected graph gear')
    for kind in ('inputs', 'outputs'):
        expected = [{'name': r.name, 'shape': list(r.shape), 'dtype': r.dtype} for r in getattr(graph, kind)]
        if manifest.get(kind) != expected:
            raise ValueError('incremental ' + kind + ' differ from the selected graph')
    locate = lambda path: relocated_path(path, root)
    prepared_root, _ = _safe(locate(manifest['source_prepared_root']), directory=True)
    if locate(manifest['file_constant_root']) != prepared_root:
        raise ValueError('FileConstant root differs from original preparation')
    prepared, prepared_id = _metadata(locate(manifest['source_prepared']['path']), bindings)
    if (Path(prepared_id['path']).parent != prepared_root
            or any(prepared_id[k] != manifest['source_prepared'].get(k) for k in ('bytes', 'sha256', 'full_identity'))
            or prepared.get('status') != 'PREPARED' or prepared.get('draft_config') != manifest['draft_config']
            or prepared.get('source', {}).get('fp16_payload_sha256') != weight_identity(prepared['weights'])
            or set(prepared['weights']) != set(graph.weights)):
        raise ValueError('original prepared weights/configuration changed')
    # Retain the source receipts the exporter actually checked, without loading
    # vendor GE libraries or touching the successful S64 OMC.
    guards = manifest.get('source_bindings')
    if not isinstance(guards, list) or not guards:
        raise ValueError('incremental export lacks original metadata bindings')
    for row in guards:
        _, actual = _metadata(locate(row['path']), bindings)
        if any(actual[k] != row.get(k) for k in ('bytes', 'sha256', 'full_identity')):
            raise ValueError('original export metadata changed: ' + row['path'])
    rows = {}
    retained = manifest.get('source_weight_identities', [])
    retained_ids = {str(locate(row['path'])): row['identity'] for row in retained}
    if len(retained_ids) != len(graph.weights) or len(retained) != len(retained_ids):
        raise ValueError('incomplete or duplicate original weight identities')
    for key, tensor in graph.weights.items():
        row = prepared['weights'][key]
        relative = Path(row['path'])
        if (len(relative.parts) != 2 or relative.parts[0] != 'weights' or '..' in relative.parts
                or row['shape'] != list(tensor.shape) or row['dtype'] != 'float16'
                or row['bytes'] != 2 * math.prod(tensor.shape)):
            raise ValueError('invalid prepared weight path/descriptor: ' + key)
        path = prepared_root / relative
        before = _payload(path, row)
        if retained_ids.get(str(path)) != before or str(path) in rows:
            raise ValueError('original weight full identity changed or duplicated: ' + key)
        rows[str(path)] = dict(row, identity=before, relative=relative.as_posix())
    if manifest.get('source_weight_bytes') != sum(row['bytes'] for row in rows.values()):
        raise ValueError('original weight extent budget changed')
    if manifest['air'].get('path') != 'draft.air':
        raise ValueError('incremental AIR path must be draft.air')
    air = root / 'draft.air'
    air_id = _payload(air, manifest['air'])
    generated = {row['path']: row for row in manifest['generated_files']}
    recipe_path = root / 'draft/recipe/graph.recipe'
    recipe_id = _payload(recipe_path, generated['draft/recipe/graph.recipe'])
    recipe = read_recipe(recipe_path)
    constants = [n for n in recipe['records'] if n['type'] == 'FileConstant']
    declared = {row['relative']: row for row in rows.values()}
    seen = set()
    for node in constants:
        relative = node['attrs']['file_path']['value']
        if relative not in declared or relative in seen:
            raise ValueError('FileConstant recipe paths must match every original weight exactly once')
        seen.add(relative)
    if seen != set(declared) or len(constants) != len(graph.weights):
        raise ValueError('FileConstant recipe coverage is incomplete')
    rows = {str(air): dict(manifest['air'], identity=air_id, relative=air.name), **rows}
    sources = {path: row['identity'] for path, row in rows.items()}
    sources[str(recipe_path)] = recipe_id
    _stable(bindings, sources)
    return dict(root=root, manifest=manifest, identity=identity, bindings=bindings,
                air=air, recipe=recipe_path, rows=rows, sources=sources)


class IncrementalInputView:
    def __init__(self, bound, output):
        self.bound, self.output = bound, Path(output)
        self.air, self.rows = bound['air'], bound['rows']
        self.sources = [Path(path) for path in self.rows]
        self.attempted, self.aliases, self.snapshots = set(), {}, {}
        self.evidence = {'status': 'BOUND', 'artifact_kind': 'qwen35-incremental-compiler-input-view',
            'export_manifest': bound['identity'], 'weight_files_copied': 0, 'weight_hardlinks_created': 0,
            'weight_file_aliases_created': 0, 'filesystem_write_protection': 'NOT_ENFORCED_BY_ALIASES',
            'source_policy': 'NO_WRITES_REQUESTED_FULL_IDENTITIES_AND_SHA256_AUDITED', 'files': []}

    def source_relative(self, source):
        return Path(self.rows[str(source)]['relative'])

    def snapshot(self, source, destination, normal_snapshot):
        key = str(source)
        if key not in self.rows or key in self.attempted:
            raise ValueError('unknown or repeated incremental input')
        row = self.rows[key]
        if destination != self.output / 'compile-work/generated' / row['relative']:
            raise ValueError('incremental input destination changed')
        self.attempted.add(key)
        if _current(source)[1] != row['identity']:
            raise ValueError('incremental source identity changed before staging')
        if source == self.air:
            result = normal_snapshot(source, destination)
            if any(result[k] != row[k] for k in ('bytes', 'sha256')):
                raise ValueError('incremental AIR snapshot content changed')
            self.snapshots[str(destination)] = dict(row, identity=_current(destination)[1])
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            _safe(destination.parent, directory=True)
            destination.symlink_to(source)
            self.aliases[str(destination)] = {'source': key, 'stat': _five_alias(destination)}
            result = {'source': key, 'source_stat': _five(row['identity']), 'snapshot_stat': _five(row['identity']),
                'bytes': row['bytes'], 'sha256': row['sha256'], 'snapshot_method': 'ORIGINAL_WEIGHT_FILE_REFERENCE'}
            self.evidence['weight_file_aliases_created'] += 1
        self.evidence['files'].append({'source': key, 'relative': row['relative'], 'bytes': row['bytes'],
            'sha256': row['sha256'], 'identity': row['identity'], 'method': result['snapshot_method']})
        return result

    def audit(self):
        errors = []
        try:
            _stable(self.bound['bindings'], self.bound['sources'])
            for path, row in self.rows.items():
                if _payload(path, row) != row['identity']:
                    raise ValueError('incremental source full identity changed')
            for path, row in self.snapshots.items():
                if _payload(path, row) != row['identity']:
                    raise ValueError('incremental AIR snapshot changed')
            for path, row in self.aliases.items():
                if os.readlink(path) != row['source'] or _five_alias(Path(path)) != row['stat']:
                    raise ValueError('incremental weight alias changed')
            if self.attempted != set(self.rows) or len(self.snapshots) + len(self.aliases) != len(self.rows):
                raise ValueError('incremental input view did not complete')
            _stable(self.bound['bindings'], self.bound['sources'])
        except (OSError, ValueError) as error:
            errors.append(str(error))
        self.evidence.update(status='FAILED' if errors else 'PASSED', audit_errors=errors)
        return self.evidence


def _five_alias(path):
    s = path.lstat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def prepare_incremental_input_view(folder, *, air, output):
    bound = bind_incremental_export(folder)
    if Path(air) != bound['air']:
        raise ValueError('compiler AIR differs from incremental export')
    if bound['root'] in Path(output).parents or Path(output) == bound['root']:
        raise ValueError('compile output must not be inside incremental export')
    return IncrementalInputView(bound, output)
