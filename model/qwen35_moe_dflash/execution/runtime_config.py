"""Portable runtime configuration, independent of historical execution evidence.

Only file presence/extent and tensor contracts are admission requirements.
Cache keys are deployment labels; they are not file integrity attestations.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import uuid

POLICY = 'size-and-abi-only'
ADAPTER = 'qwen35-cpro-resident'


def read_json(path, maximum=64 << 20):
    path = Path(path)
    if not path.is_file() or not 0 < path.stat().st_size <= maximum:
        raise ValueError('missing or oversized JSON: ' + str(path))
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError('expected JSON object: ' + str(path))
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def file_record(path, *, size=None, cache_key=None):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError('missing runtime file: ' + str(path))
    actual = path.stat().st_size
    if actual <= 0 or size is not None and actual != size:
        raise ValueError('runtime file extent differs: ' + str(path))
    # The old wire format calls this field sha256. In this policy it is only
    # a persistent cache label, never compared with a payload digest.
    key = cache_key or (uuid.uuid4().hex + uuid.uuid4().hex)
    if len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
        raise ValueError('cache_key must be 64 lowercase hexadecimal characters')
    return dict(path=str(path), bytes=actual, sha256=key, identity={})


def _relative(path, base):
    return os.path.relpath(Path(path).resolve(), Path(base).resolve())


def _io(rows):
    fields = ('index', 'name', 'runtime_tensor_name', 'dtype', 'shape', 'bytes',
              'tool_input_type', 'tool_output_type')
    return [{k: row[k] for k in fields if k in row} for row in rows]


def export_config(prepared, output):
    """Import an existing ABI/mapping once, without replaying its old audits."""
    from qwen35_moe_dflash.execution.serial_dflash import MIXED_MODEL_LOAD_APIS, TARGET_LIFECYCLE_POLICY
    source = read_json(Path(prepared) / 'manifest.json')
    if (source.get('execution_mode') != 'resident-mixed' or 'prefill' not in source
            or source.get('model_load_apis') != MIXED_MODEL_LOAD_APIS
            or source.get('target_lifecycle_policy') != TARGET_LIFECYCLE_POLICY):
        raise ValueError('import a mixed resident preparation with one prefill and explicit state maps')
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError('runtime config already exists: ' + str(destination))
    assets_path = destination.with_name(destination.stem + '-assets.json')
    if assets_path.exists():
        raise FileExistsError(str(assets_path))
    parent = destination.parent
    models, assets = {}, {}
    for role in ('prefill', 'decode'):
        prefill = role == 'prefill'
        base = source['prefill']['base_plan'] if prefill else source['target_binding']['base_plan']
        model = source['prefill']['model'] if prefill else source['target_model']
        rows = source['prefill']['payloads'] if prefill else source['target_payloads']
        primary = [r for r in rows if r.get('primary')]
        if len(primary) != 1:
            raise ValueError(role + ' must contain one primary model')
        model_root = Path(primary[0]['local'])
        for _ in Path(primary[0]['remote']).parts:
            model_root = model_root.parent
        payloads = []
        for row in rows:
            relative = Path(row['remote'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('invalid model sidecar path')
            file_record(model_root / relative, size=row['bytes'])
            payloads.append(dict(path=str(relative), bytes=row['bytes']))
        models[role] = dict(directory=_relative(model_root, parent),
            model=primary[0]['remote'], cache_key=base['compilation_report_sha256'])
        abi = source['prefill_abi' if prefill else 'resident_abi']
        assets[role] = dict(inputs=_io(base['inputs']), outputs=_io(base['outputs']),
            runtime_inputs=abi['inputs'], runtime_outputs=abi['outputs'], payloads=payloads,
            model_cache_key=model['sha256'],
            state_mapping=deepcopy(source['prefill' if prefill else 'target_binding']['state_mapping']))
    draft = source['draft']
    model = file_record(draft['model_source']['path'], size=draft['model_source']['bytes'],
                        cache_key=draft['model_source']['sha256'])
    models['draft'] = dict(path=_relative(model['path'], parent), cache_key=model['sha256'])
    assets['draft'] = {k: deepcopy(draft[k]) for k in ('outputs', 'direct_spec', 'variant',
        'inputs', 'context_rows', 'cache_capacity', 'max_draft_tokens') if k in draft}
    assets['draft']['abi'] = read_json(draft['abi']['path'], 1 << 20)
    assets['environment'] = {k: source['target_binding']['base_plan'][k]
                            for k in ('ddk_id', 'runtime_version', 'firmware_id')}
    tables = {}
    for role, row in source['tables'].items():
        checked = file_record(row['path'], size=row['bytes'], cache_key=row['sha256'])
        tables[role] = dict(path=_relative(checked['path'], parent), bytes=row['bytes'], cache_key=row['sha256'])
    config = dict(schema_version=1, adapter=ADAPTER, assets=assets_path.name,
        models=models, embedding=tables,
        defaults=dict(max_new_tokens=8, max_proposals_per_round=2, minimum_draft_rounds=2,
            prompt_format='chat', draft_preread=True, timeout_seconds=7200, lock_wait_seconds=1800),
        board_model_root=source['board_model_root'])
    if 'text_io' in source:
        config['tokenizer'] = dict(directory=_relative(source['text_io']['source_directory'], parent))
    validate_assets(assets)
    write_json(assets_path, assets)
    write_json(destination, config)
    return config


def validate_assets(assets):
    """Structural validation remains mandatory; file identity is irrelevant."""
    from qwen35_moe_dflash.target.original_target_smoke_backend import _mapping
    widths = {'FLOAT16': 2, 'FLOAT32': 4, 'INT8': 1, 'INT16': 2, 'INT32': 4, 'INT64': 8}
    for role in ('prefill', 'decode'):
        spec = assets[role]
        for direction, count in (('inputs', 89), ('outputs', 82)):
            rows, native = spec[direction], spec['runtime_' + direction]
            if len(rows) != count or len(native) != count:
                raise ValueError(role + ' ABI count differs')
            names = set()
            for i, (row, desc) in enumerate(zip(rows, native)):
                name = row['runtime_tensor_name']
                shape = row['shape']
                if (row['index'] != i or desc['index'] != i or name in names or desc['name'] != name
                        or desc['dtype'] != row['dtype'] or desc['shape'] != shape
                        or not shape or any(type(n) is not int or n <= 0 for n in shape)
                        or row['bytes'] != math.prod(shape) * widths[row['dtype']]
                        or desc['bytes'] != row['bytes']):
                    raise ValueError(role + ' tensor descriptor differs')
                names.add(name)
    base = dict(assets['decode'], profile='cpro-40block-decode-1', gear=1, sequence_length=1)
    _mapping(base, assets['decode']['state_mapping'])
    mappings = assets['prefill']['state_mapping']
    decode = {r['source_refdata']: r for r in assets['decode']['state_mapping']}
    if (len(mappings) != 80 or {r['decode']['source_refdata'] for r in mappings} != set(decode)
            or len({r['prefill']['runtime_output_index'] for r in mappings}) != 80):
        raise ValueError('prefill/decode map must cover 80 unique states')
    for row in mappings:
        for role in ('prefill', 'decode'):
            side = row[role]
            for direction, key in (('inputs', 'runtime_input_index'), ('outputs', 'runtime_output_index')):
                index = side[key]
                if type(index) is not int or not 0 <= index < len(assets[role][direction]):
                    raise ValueError('state slot outside runtime ABI')
                tensor = assets[role][direction][index]
                if tensor['bytes'] != row['bytes'] or tensor['dtype'] != 'FLOAT16':
                    raise ValueError('prefill/decode state extent or dtype differs')
            if role == 'decode' and any(side[k] != decode[side['source_refdata']][k]
                    for k in ('runtime_input_index', 'runtime_output_index')):
                raise ValueError('prefill map differs from decode state slots')


def _native_abi(spec, prefix, path):
    from qwen35_moe_dflash.execution.serial_dflash import _field
    lines = ['QWEN35_RESIDENT_ABI\t1']
    for direction in ('inputs', 'outputs'):
        for row in spec['runtime_' + direction]:
            lines.append('\t'.join(map(_field, (prefix + '_' + direction[:-1], row['index'], row['name'],
                row['dtype'], row['raw_dtype'], row['raw_format'], ','.join(map(str, row['shape'])), row['bytes']))))
    path.write_text('\n'.join(lines) + '\n')
    return dict(file_record(path), inputs=spec['runtime_inputs'], outputs=spec['runtime_outputs'])


def validate_prepared_config(root, manifest):
    from qwen35_moe_dflash.execution.serial_dflash import KIND, _options, mixed_load_contract
    if (manifest.get('artifact_kind') != KIND or manifest.get('status') != 'PREPARED'
            or manifest.get('dump') is not False or manifest.get('runtime_profile') != 'compact-memory-v1'):
        raise ValueError('expected compact runtime preparation')
    mixed_load_contract(manifest)
    prompt = manifest['prompt_token_ids']
    if not isinstance(prompt, list) or any(type(v) is not int or not 0 <= v < 248320 for v in prompt):
        raise ValueError('invalid prompt token')
    _options(len(prompt), *[manifest[k] for k in ('max_new_tokens', 'max_proposals_per_round',
        'minimum_draft_rounds', 'timeout_seconds', 'lock_wait_seconds')])
    if manifest['draft'].get('variant'):
        from qwen35_moe_dflash.draft.incremental.runtime import validate_abi
        validate_abi(manifest['draft'])
        if (manifest['max_proposals_per_round'] > manifest['draft']['max_draft_tokens']
                or len(prompt) + manifest['max_new_tokens'] - 1 > manifest['draft']['cache_capacity']):
            raise ValueError('incremental gear/capacity cannot cover this request')
    for role, abi_key in (('decode', 'resident_abi'), ('prefill', 'prefill_abi')):
        abi = manifest[abi_key]
        # Local generated files are relative to the preparation, even after a
        # whole run directory is moved. No historical absolute path comparison.
        path = Path(root) / (abi_key.replace('_', '-') + '.tsv')
        if role == 'decode':
            path = Path(root) / 'resident-abi.tsv'
        abi['path'] = str(path)
        file_record(path, size=abi['bytes'], cache_key=abi['sha256'])
    generated = [(manifest['draft']['abi'], 'incremental-draft-abi.json' if manifest['draft'].get('variant') else 'draft-abi.json')]
    if 'text_io' in manifest:
        generated.extend([(manifest['text_io']['bundle'], 'tokenizer.bin'),
                          (manifest['text_io']['prompt'], 'prompt.txt')])
    for record, name in generated:
        record['path'] = str(Path(root) / name)
        file_record(record['path'], size=record['bytes'], cache_key=record['sha256'])
    # Static payloads are checked by extent when needed for a cache upload.
    # Completed recovery can run after static assets have been archived.
    manifest['output'] = str(root)
    return manifest


def prepare_config(config_file, output, *, text=None, text_file=None, prompt_input_ids=None,
                   prompt_length=None, incremental_draft=None, max_draft_tokens=None, **overrides):
    from qwen35_moe_dflash.common.artifacts import output_path
    from qwen35_moe_dflash.execution.serial_dflash import KIND, _options, TARGET_LIFECYCLE_POLICY, MIXED_MODEL_LOAD_API, MIXED_MODEL_LOAD_APIS
    from qwen35_moe_dflash.execution.prefill_binding import RECIPE, control_assumption
    config_file = Path(config_file).resolve()
    config = read_json(config_file)
    if config.get('schema_version') != 1 or config.get('adapter') != ADAPTER:
        raise ValueError('unsupported runtime configuration')
    parent = config_file.parent
    assets = read_json(parent / config['assets'])
    validate_assets(assets)
    options = dict(config['defaults'])
    for key, value in overrides.items():
        if value is not None:
            if key not in options:
                raise ValueError('unsupported runtime option: ' + key)
            options[key] = value
    out = output_path(output); out.mkdir(parents=True, exist_ok=False)
    selected = sum(v is not None for v in (text, text_file, prompt_input_ids))
    if selected != 1:
        raise ValueError('select exactly one text or token input')
    text_io = None
    if prompt_input_ids is not None:
        import struct
        raw = Path(prompt_input_ids).read_bytes()
        n = prompt_length if prompt_length is not None else len(raw) // 4
        if len(raw) % 4 or not 1 <= n <= 63 or len(raw) < 4 * n:
            raise ValueError('invalid INT32 prompt token file')
        prompt = list(struct.unpack('<' + 'i' * n, raw[:4*n]))
    else:
        if prompt_length is not None:
            raise ValueError('text input uses its complete encoded prompt')
        from qwen35_moe_dflash.execution.tokenizer import prepare_text
        if text_file is not None:
            text = Path(text_file).read_text(encoding='utf-8')
        prompt, _, text_io = prepare_text(parent / config['tokenizer']['directory'], text,
            options['prompt_format'], out, verify_sources=False)
    _options(len(prompt), *[options[k] for k in ('max_new_tokens', 'max_proposals_per_round',
        'minimum_draft_rounds', 'timeout_seconds', 'lock_wait_seconds')])
    if any(type(t) is not int or not 0 <= t < 248320 for t in prompt):
        raise ValueError('prompt token outside vocabulary')
    bound = {}
    for role, gear in (('prefill', 64), ('decode', 1)):
        spec, model = assets[role], config['models'][role]
        directory = (parent / model['directory']).resolve()
        payloads = []
        for entry in spec['payloads']:
            relative = Path(entry['path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('invalid model sidecar path')
            record = file_record(directory / relative, size=entry['bytes'], cache_key=model['cache_key'])
            declared = dict(path='compile-work/generated/' + str(relative), bytes=record['bytes'], sha256=record['sha256'])
            payloads.append(dict(local=record['path'], remote=str(relative), bytes=record['bytes'],
                sha256=record['sha256'], record=declared, primary=str(relative) == model['model']))
        primary = [r for r in payloads if r['primary']]
        if len(primary) != 1:
            raise ValueError(role + ' requires one primary model')
        file = file_record(primary[0]['local'], cache_key=spec['model_cache_key'])
        base = dict(inputs=spec['inputs'], outputs=spec['outputs'], gear=gear, sequence_length=gear,
            profile='cpro-40block-' + ('prefill-64' if gear == 64 else 'decode-1'),
            model_sha256=file['sha256'], compilation_report_sha256=model['cache_key'],
            compilation=str(directory), primary_model_binding=file,
            tool_binding=dict(path='/data/local/tmp/tools_sysdbg/model_run_tool_internal', sha256='0'*64),
            **assets['environment'])
        compiled = dict(generated_artifacts=[r['record'] for r in payloads], omc=primary[0]['record'])
        bound[role] = dict(base_plan=base, model=file, payloads=payloads, compiled=compiled,
                           state_mapping=spec['state_mapping'], source_bindings=[])
    draft = deepcopy(assets['draft'])
    draft_model = config['models']['draft']
    draft['model_source'] = file_record(parent / draft_model['path'], cache_key=draft_model['cache_key'])
    draft['ddk_id'] = assets['environment']['ddk_id']
    abi_path = out / 'draft-abi.json'; write_json(abi_path, draft.pop('abi'))
    draft['abi'] = file_record(abi_path)
    if incremental_draft is not None:
        from qwen35_moe_dflash.draft.incremental.runtime import bind_compiled
        draft, _ = bind_compiled(incremental_draft, out, ddk_id=assets['environment']['ddk_id'],
            max_draft_tokens=max_draft_tokens, max_proposals=options['max_proposals_per_round'],
            total_tokens=len(prompt) + options['max_new_tokens'], verify_sources=False)
    elif max_draft_tokens is not None and max_draft_tokens != draft.get('max_draft_tokens', 15):
        raise ValueError('candidate gear differs from configured Draft model')
    tables = {k: file_record(parent / r['path'], size=r['bytes'], cache_key=r['cache_key'])
              for k, r in config['embedding'].items()}
    if {k: r['bytes'] for k, r in tables.items()} != {'weight': 508559360, 'scale': 993280}:
        raise ValueError('embedding table extents differ')
    decode, prefill = bound['decode'], bound['prefill']
    prefill.update(control_recipe=RECIPE, experimental_control_assumption=control_assumption(len(prompt)))
    manifest = dict(schema_version=1, artifact_kind=KIND, status='PREPARED', output=str(out),
        file_validation=POLICY, prompt_token_ids=prompt, dump=False, execution_mode='resident-mixed',
        target_binding=dict(base_plan=decode['base_plan'], state_mapping=decode['state_mapping'], active_source_identities=[]),
        target_model=decode['model'], target_payloads=decode['payloads'], target_compiled=decode['compiled'],
        draft=draft, tables=tables, prefill=prefill, prompt_ingestion=RECIPE, source_bindings=[],
        resident_abi=_native_abi(assets['decode'], 'target', out / 'resident-abi.tsv'),
        prefill_abi=_native_abi(assets['prefill'], 'prefill', out / 'prefill-abi.tsv'),
        runtime_profile='compact-memory-v1', target_lifecycle_policy=TARGET_LIFECYCLE_POLICY,
        model_load_api=MIXED_MODEL_LOAD_API, model_load_apis=dict(MIXED_MODEL_LOAD_APIS),
        board_model_root=config['board_model_root'], **{k: options[k] for k in (
            'max_new_tokens', 'max_proposals_per_round', 'minimum_draft_rounds', 'timeout_seconds', 'lock_wait_seconds')})
    if options['draft_preread']:
        manifest['draft_preread'] = 'sequential-1m-v1'
    if text_io:
        manifest['text_io'] = text_io
    validate_prepared_config(out, manifest)
    write_json(out / 'manifest.json', manifest)
    return manifest
