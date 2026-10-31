"""Bind a packed incremental Draft to the existing board-local resident loop."""
from __future__ import annotations

import json
import math
from pathlib import Path

from .artifacts import save_json
from .target_compile_failure import _read_small

VARIANT = 'incremental-context-kv-v1'
GEOMETRY = ('context_rows', 'cache_capacity', 'max_draft_tokens')


def validate_abi(value):
    c, capacity, k = (value[key] for key in GEOMETRY)
    if (any(type(v) is not int for v in (c, capacity, k)) or c not in (16, 64)
            or not c <= capacity <= 4096 or k not in (1, 3, 7, 15)):
        raise ValueError('incremental runtime requires C16/C64, capacity<=4096, K1/3/7/15')
    inputs = [('target_features', 'float16', [1, c, 16384])]
    inputs += [(name, 'int32', [1]) for name in ('start_position', 'valid_rows', 'anchor', 'proposal_count')]
    outputs = [('draft_top1', 'int32', [1, k]), ('draft_hidden', 'float16', [1, k, 2048])]
    banks = [(f'{kind}_{layer}', 'float16', [1, 8, capacity, 128]) for layer in range(6) for kind in ('k', 'v')]
    inputs += [('past_' + n, dt, shape) for n, dt, shape in banks]
    outputs += [('present_' + n, dt, shape) for n, dt, shape in banks]
    for direction, expected in (('inputs', inputs), ('outputs', outputs)):
        rows = value[direction]
        if not isinstance(rows, list) or len(rows) != len(expected):
            raise ValueError('incremental ABI count differs')
        remaining = {n: (dt, shape) for n, dt, shape in expected}
        for index, row in enumerate(rows):
            name = row.get('name')
            if (name not in remaining or type(row.get('index')) is not int or row['index'] != index
                    or (row.get('dtype'), row.get('shape')) != remaining[name]
                    or direction == 'outputs' and name != expected[index][0]):
                raise ValueError('incremental ABI ordered slot differs')
            dt, shape = remaining.pop(name)
            if row.get('bytes') != math.prod(shape) * (2 if dt == 'float16' else 4):
                raise ValueError('incremental ABI byte extent differs')
    return {key: value[key] for key in GEOMETRY}


def bind_compiled(directory, output, *, ddk_id, max_draft_tokens, max_proposals, total_tokens,
                  verify_sources=True):
    # Import here to keep preparation's helpers as the one identity policy.
    from .board_serial_dflash import _hash_source
    root = Path(directory).resolve()
    pins = {}
    def read(path, declared=None):
        if not verify_sources:
            from .runtime_config import read_json, file_record
            return read_json(path, 4 << 20), file_record(path)
        raw, observed = _read_small(path, 4 << 20)
        record = _hash_source(path)
        if any(record[k] != observed[k] for k in ('sha256', 'bytes')) or declared and any(record[k] != declared[k] for k in ('sha256', 'bytes')):
            raise ValueError('incremental compilation receipt changed: ' + str(path))
        pins[str(path)] = record['identity']
        return json.loads(raw), record
    source, source_id = read(root / 'manifest.json')
    if (source.get('artifact_kind') != 'qwen35-incremental-draft-compiled'
            or source.get('status') != 'DRAFT_INCREMENTAL_OMC_COMPILED_ABI_BOUND'
            or source.get('compiled_abi') != 'PASSED' or source.get('compiled_precision') != 'PASSED'
            or source.get('weight_storage') != 'embedded-stream' or source.get('ddk_id') != ddk_id):
        raise ValueError('expected successful same-DDK incremental compiled model/ABI/precision receipts')
    geometry = validate_abi(source)
    if (type(max_draft_tokens) is not int or type(max_proposals) is not int
            or max_draft_tokens != geometry['max_draft_tokens'] or not 1 <= max_proposals <= max_draft_tokens):
        raise ValueError('--max-draft-tokens must match the compiled gear and cover considered proposals')
    if total_tokens - 1 > geometry['cache_capacity']:
        raise ValueError('incremental cache cannot cover this complete request')
    def child(record):
        relative = Path(record['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('incremental receipt must stay inside compilation output')
        return root / relative
    abi, _ = read(child(source['compiled_abi_file']), source['compiled_abi_file'])
    packing, _ = read(child(source['weight_packing']), source['weight_packing'])
    if (abi.get('status') != 'PASSED' or any(abi.get(key) != source[key] for key in (*GEOMETRY, 'inputs', 'outputs'))
            or packing.get('status') != 'EMBEDDED'
            or verify_sources and packing.get('source', {}).get('sha256') != abi.get('omc', {}).get('sha256')
            or any(packing.get('output', {}).get(key) != source['model'][key]
                   for key in (('bytes', 'sha256') if verify_sources else ('bytes',)))):
        raise ValueError('incremental ABI/packing/model receipts disagree')
    if verify_sources:
        model = _hash_source(child(source['model']))
    else:
        from .runtime_config import file_record
        model = file_record(child(source['model']), size=source['model']['bytes'],
                            cache_key=source['model']['sha256'])
    if verify_sources and any(model[key] != source['model'][key] for key in ('sha256', 'bytes')):
        raise ValueError('incremental packed model changed')
    pins[model['path']] = model['identity']
    native = dict(schema_version=1, variant=VARIANT, model_sha256=model['sha256'],
                  inputs=source['inputs'], outputs=source['outputs'], **geometry)
    native_path = Path(output) / 'incremental-draft-abi.json'
    save_json(native_path, native)
    record = _hash_source(native_path) if verify_sources else file_record(native_path)
    # Recheck metadata identities after reading the complete model payload.
    from .original_embedding_lookup import _current
    if verify_sources and any(_current(path)[1] != identity for path, identity in pins.items()):
        raise ValueError('incremental source changed during binding')
    return (dict(model_source=model, abi=record, source_bundle=str(root), variant=VARIANT,
                 inputs=source['inputs'], outputs=source['outputs'], ddk_id=ddk_id,
                 compiled_manifest=source_id, **geometry), pins)
