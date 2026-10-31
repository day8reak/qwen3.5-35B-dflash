"""Normalize a target AIR bundle to export.air + weight/, without weight copies."""
from copy import deepcopy
import os
from pathlib import Path, PurePosixPath

from qwen35_moe_dflash.execution.runtime_config import read_json
from qwen35_moe_dflash.execution.run_layout import _save
from qwen35_moe_dflash.toolchain.air_schema import load_torchair_schema
from qwen35_moe_dflash.toolchain.omc_wire import Schema
from .original_air_roundtrip import _attributes, _canonical, _decode, MAX_AIR_BYTES
from .original_ge_append_wire import _parts, _blob


def weight_schema(pb2=None):
    if pb2:
        return load_torchair_schema(pb2)
    # Offline descriptor compiled from the pinned public TorchAir source.
    # Provenance: specs/torchair-ge-ir.json. Never import TorchAir or execute pb2.
    source = Path(__file__).resolve().parents[3] / 'specs/torchair-ge-ir.pb'
    return Schema(source.read_bytes(), expected_file='ge_ir.proto', expected_package='ge.proto')


def rewrite_weight_paths(raw, schema):
    kind, decoded, _ = _decode(schema, raw)
    references = []

    def rewrite(payload, message, overrides=None, depth=0):
        if depth > 64:
            raise ValueError('AIR message nesting exceeds limit')
        fields = schema._messages[message]
        constant = None
        if message == schema.package + '.OpDef':
            node = schema.decode(payload, message)
            if node.get('type') == 'FileConstant':
                old = _attributes(node).get('file_path', {}).get('s')
                if (not isinstance(old, str) or '\\' in old or '\x00' in old
                        or PurePosixPath(old).is_absolute() or '..' in PurePosixPath(old).parts
                        or len(PurePosixPath(old).parts) != 2):
                    raise ValueError('FileConstant requires a relative directory/basename: ' + str(old))
                constant = './weight/' + PurePosixPath(old).name
                references.append(dict(node=node['name'], old=old, new=constant))
        pieces = []
        for number, wire, encoded, value in _parts(payload):
            field = fields[number]  # Strict schema decoding above already rejected unknown fields.
            change = (overrides or {}).get(field.name)
            if constant and field.name == 'attr':
                entry = schema.decode(value, field.type_name)
                if entry.get('key') == 'file_path':
                    change = {'value': {'s': constant.encode()}}
            if field.kind == 11:
                updated = rewrite(value, field.type_name, change, depth + 1)
            elif isinstance(change, bytes):
                if wire != 2 or field.kind not in (9, 12):
                    raise ValueError('FileConstant path is not a string field')
                updated = change
            else:
                updated = value
            pieces.append(_blob(number, updated) if updated != value else encoded)
        return b''.join(pieces)

    normalized = rewrite(raw, schema.package + '.' + kind)
    if not references:
        raise ValueError('AIR has no FileConstant weights')
    prefixes = {PurePosixPath(row['old']).parts[0] for row in references}
    if len(prefixes) != 1:
        raise ValueError('AIR weights must share one directory before normalization')
    expected = deepcopy(decoded)

    def update(value):
        if isinstance(value, dict):
            if value.get('type') == 'FileConstant':
                attr = _attributes(value)['file_path']
                attr['s'] = './weight/' + PurePosixPath(attr['s']).name
            for child in value.values():
                update(child)
        elif isinstance(value, list):
            for child in value:
                update(child)

    update(expected)
    new_kind, observed, _ = _decode(schema, normalized)
    if kind != new_kind or _canonical(expected) != _canonical(observed):
        raise ValueError('AIR changed outside FileConstant.file_path')
    return normalized, prefixes.pop(), references


def normalize_air_weights(directory, *, apply=False, torchair_pb2=None):
    """Rename existing sidecars or collect flat exporter weights; resume in place."""
    directory = Path(directory).resolve(strict=True)
    build = read_json(directory / 'build.json') if (directory / 'build.json').is_file() else {}
    name = build.get('air', 'export.air')
    if Path(name).name != name or not name.endswith('.air'):
        raise ValueError('AIR must be directly inside its bundle directory')
    air = directory / name
    if air.is_symlink() or not air.is_file() or not 0 < air.stat().st_size <= MAX_AIR_BYTES:
        raise ValueError('AIR must be a regular file at most 64 MiB')
    raw = air.read_bytes()
    normalized, prefix, references = rewrite_weight_paths(raw, weight_schema(torchair_pb2))
    if (directory / 'weight').is_symlink() or (directory / prefix).is_symlink():
        raise ValueError('weight directory must not be a symlink')
    names = sorted({PurePosixPath(row['old']).name for row in references})
    journal = directory / '.weight-layout.json'
    pending = read_json(journal) if journal.exists() else None
    if pending:
        if (pending.get('air') != name or pending.get('files') != names
                or prefix not in (pending.get('prefix'), 'weight')):
            raise ValueError('unfinished weight-directory migration belongs to another AIR')
        prefix, moves = pending['prefix'], pending['moves']
    else:
        if prefix == 'weight':
            moves = []
        elif (directory / prefix).is_dir():
            if (directory / prefix).is_symlink():
                raise ValueError('weight directory must not be a symlink')
            moves = [dict(source=prefix, destination='weight')]
        elif directory.name == prefix:
            moves = [dict(source=value, destination='weight/' + value) for value in names]
        else:
            raise FileNotFoundError(directory / prefix)
        if moves and os.path.lexists(directory / 'weight'):
            raise FileExistsError('weight/ already exists; no existing files were replaced')

    for move in moves:
        for key in ('source', 'destination'):
            part = Path(move[key])
            if part.is_absolute() or '..' in part.parts or not part.parts:
                raise ValueError('invalid saved weight migration path')
        source, target = directory / move['source'], directory / move['destination']
        if source.is_symlink() or target.is_symlink():
            raise ValueError('weight migration cannot traverse symlinks')
        if source.exists() and target.exists():
            raise FileExistsError(target)
        if not source.exists() and not target.exists():
            raise FileNotFoundError(source)
    for value in names:
        candidates = ([directory / prefix / value] if prefix != 'weight' else [])
        if directory.name == prefix:
            candidates.append(directory / value)
        candidates.append(directory / 'weight' / value)
        path = next((p for p in candidates if p.is_file()), None)
        if path is None or path.is_symlink() or path.stat().st_size <= 0:
            raise ValueError('missing or invalid weight: ' + value)
    result = dict(status='PLANNED', directory=str(directory), air=name, weight_directory='weight',
                  file_constants=len(references), weight_files=len(names), weight_files_copied=0,
                  model_compilation='NOT_RUN', device_execution='NOT_RUN')
    if not apply:
        return result
    if moves or raw != normalized or pending:
        _save(journal, dict(air=name, prefix=prefix, files=names, moves=moves))
        for move in moves:
            source, target = directory / move['source'], directory / move['destination']
            if source.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                source.rename(target)
        if raw != normalized:
            temporary = directory / '.weight-layout.air.tmp'
            with temporary.open('wb') as stream:
                stream.write(normalized)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(air)
    build.update(air=name, input_layout='air-directory', weight_directory='weight')
    _save(directory / 'build.json', build)
    journal.unlink(missing_ok=True)
    result['status'] = 'NORMALIZED'
    return result
