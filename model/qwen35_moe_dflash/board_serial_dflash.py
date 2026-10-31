"""Prepare an autonomous board-local loop using the existing complete OMCs.

The supervisor moves orchestration, original embedding row lookup, and all
state/feature feedback onto the board. Resident mixed-API execution is an explicit
separate mode; neither mode is a speed/accuracy acceptance gate.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path, PurePosixPath
import re
import shutil

from .artifacts import output_path, save_json
from .hashing import sha256_file
from .original_embedding_lookup import OriginalEmbeddingLookup, _current
from .serial_dflash_smoke import _backends, _prompt
from .target_compile_failure import _read_small


KIND = 'qwen35-board-serial-dflash-prepared'
MAX_MANIFEST_BYTES = 64 << 20
TARGET_LIFECYCLE_POLICY = 'v2-target-v1-draft-resident-v1'
V1_TARGET_LIFECYCLE_POLICY = 'legacy-v1-file-target-buffer-draft-v1'
HISTORICAL_V1_LIFECYCLE_POLICY = 'legacy-v1-load-process-unload-v1'
V1_MODEL_LOAD_API = 'legacy-v1-mixed'
V1_MODEL_LOAD_APIS = {'prefill': 'legacy-file', 'target': 'legacy-file', 'draft': 'legacy-buffer'}
MIXED_MODEL_LOAD_API = 'target-v2-draft-v1'
MIXED_MODEL_LOAD_APIS = {'prefill': 'file-v2', 'target': 'file-v2', 'draft': 'legacy-buffer'}
V2_TARGET_LIFECYCLE_POLICY = 'init-before-weights-close-weights-first-v1'


def v1_load_contract(manifest):
    """Bind source representation to preparation, including exact old recovery."""
    if manifest.get('execution_mode') != 'resident-v1':
        raise ValueError('V1 model source contract belongs only to resident-v1')
    policy = manifest.get('target_lifecycle_policy')
    if policy == HISTORICAL_V1_LIFECYCLE_POLICY:
        if 'model_load_api' in manifest or 'model_load_apis' in manifest:
            raise ValueError('historical V1 preparation cannot adopt a new model source representation')
        return 'legacy-buffer', dict.fromkeys(('prefill', 'target', 'draft'), 'legacy-buffer')
    if (policy != V1_TARGET_LIFECYCLE_POLICY or manifest.get('model_load_api') != V1_MODEL_LOAD_API
            or manifest.get('model_load_apis') != V1_MODEL_LOAD_APIS):
        raise ValueError('prepared V1 model source policy requires exact file target and buffer draft bindings')
    return V1_MODEL_LOAD_API, dict(V1_MODEL_LOAD_APIS)


def mixed_load_contract(manifest):
    if (manifest.get('execution_mode') != 'resident-mixed'
            or manifest.get('target_lifecycle_policy') != TARGET_LIFECYCLE_POLICY
            or manifest.get('model_load_api') != MIXED_MODEL_LOAD_API
            or manifest.get('model_load_apis') != MIXED_MODEL_LOAD_APIS):
        raise ValueError('mixed residency requires explicit V2 prefill/target and V1 embedded-draft APIs')
    return MIXED_MODEL_LOAD_API, dict(MIXED_MODEL_LOAD_APIS)


def _resident_abi(binding, runtime_probe, output, *, prefix='target', runtime_api='legacy-v1'):
    """Pin actual selected-HIAI descriptors; never infer shapes from byte counts."""
    from .target_prefill_smoke_plan import _runtime_keys, _WIDTHS, _TOOL_TYPES
    base = binding['base_plan']; records = {row['path']: row['identity'] for row in binding['active_source_identities']}
    source = Path(runtime_probe).resolve() / 'model-io.json'
    raw, observed = _read_small(source, 32 << 20)
    if records.get(str(source)) != _current(source)[1]:
        raise ValueError('resident ABI source is not the freshly bound native descriptor file')
    native = json.loads(raw)
    if (native.get('artifact_kind') != 'qwen35-hiai-model-io-probe'
            or native.get('evidence_kind') != 'device-hiai' or native.get('status') != 'IO_DESCRIPTORS_OBSERVED'
            or native.get('model_sha256') != base['model_sha256']
            or native.get('model_bytes') != base['primary_model_binding']['bytes']
            or any(native.get(key) != base[key] for key in ('runtime_version', 'ddk_id', 'firmware_id'))):
        raise ValueError('resident ABI source differs from the executed target model/runtime')
    lines = ['QWEN35_RESIDENT_ABI\t1']; descriptors = {}
    for direction, type_key in (('inputs', 'tool_input_type'), ('outputs', 'tool_output_type')):
        views = _runtime_keys(native, direction); rows = native[direction]; planned = base[direction]
        if len(rows) != len(planned):
            raise ValueError('resident ABI source count differs from the bound interface')
        result = []
        for index, (row, (dtype, shape), expected) in enumerate(zip(rows, views, planned)):
            name = row.get('tensor_name'); actual_name = expected['runtime_tensor_name']
            elements = 1
            for dimension in shape: elements *= dimension
            if (row.get('tensor_name_status') != 'OBSERVED' or name != actual_name
                    or _TOOL_TYPES[dtype] != expected[type_key]
                    or elements * _WIDTHS[dtype] != expected['bytes']):
                raise ValueError('resident ABI literal name/type/extent differs from the bound runtime interface')
            result.append(dict(index=index, name=name, dtype=dtype, raw_dtype=row['raw_dtype'],
                raw_format=row['raw_format'], shape=list(shape), bytes=expected['bytes']))
            lines.append('\t'.join(map(_field, (prefix + '_' + direction[:-1], index, name, dtype,
                row['raw_dtype'], row['raw_format'], ','.join(map(str, shape)), expected['bytes']))))
        descriptors[direction] = result
    if records[str(source)] != _current(source)[1] or observed['sha256'] != sha256_file(source):
        raise ValueError('resident ABI source changed during serialization')
    Path(output).write_bytes(('\n'.join(lines) + '\n').encode('ascii'))
    return {**_hash_source(output), 'descriptor_source': dict(observed, identity=records[str(source)]),
        'inputs': descriptors['inputs'], 'outputs': descriptors['outputs'],
        'descriptor_provenance': 'HISTORICAL_FILE_V2_NAMED_MODEL_IO_PROBE',
        'runtime_validation': ('V2_NAMED_RAW_DESCRIPTOR_AND_BYTE_EXTENT_REQUIRED' if runtime_api == 'file-v2'
            else 'V1_SLOT_SHAPE_AND_BYTE_EXTENT_REQUIRED_NO_SDK_TENSOR_NAME_CLAIM'),
        'scope': 'ACTUAL_NAMED_HIAI_RAW_ENUM_AND_FIXED_SHAPE_DESCRIPTORS'}


def _hash_source(path, expected_size=None):
    path, identity = _current(path)
    if expected_size is not None and identity['bytes'] != expected_size:
        raise ValueError('source extent differs from the complete model contract: ' + str(path))
    digest = sha256_file(path)
    if _current(path)[1] != identity:
        raise ValueError('source changed during full SHA256 binding: ' + str(path))
    return {'path': str(path), 'bytes': identity['bytes'], 'sha256': digest, 'identity': identity}


def _options(prompt_length, max_new_tokens, max_proposals_per_round, minimum_draft_rounds,
             timeout_seconds, lock_wait_seconds):
    if (any(type(value) is not int for value in (prompt_length, max_new_tokens,
            max_proposals_per_round, minimum_draft_rounds, timeout_seconds, lock_wait_seconds))
            or not 1 <= prompt_length <= 63 or not 1 <= max_new_tokens
            or prompt_length + max_new_tokens > 64 or not 1 <= max_proposals_per_round <= 15
            or not 2 <= minimum_draft_rounds <= max_new_tokens - 1
            or not 1 <= timeout_seconds <= 7200 or not 0 <= lock_wait_seconds <= 7200):
        raise ValueError('board loop requires complete prefix<=64, K1..15 and bounded positive options')


def prepare_board_serial(*, decode_job, runtime_probe, state_lineage, source_torchair_pb2,
        draft_bundle, weight_table, scale_table, prompt_input_ids, output,
        prompt_length=8, max_new_tokens=8, max_proposals_per_round=2,
        minimum_draft_rounds=2, timeout_seconds=7200, lock_wait_seconds=1800,
        board_model_root='/data/local/tmp/qwen3_5_moe', dump=False, prefill_feedback=None, resident=False):
    _options(prompt_length, max_new_tokens, max_proposals_per_round, minimum_draft_rounds,
             timeout_seconds, lock_wait_seconds)
    if type(dump) is not bool or type(resident) is not bool:
        raise ValueError('dump/resident must be explicit booleans')
    if resident and prefill_feedback is None:
        raise ValueError('resident-mixed requires the bound original prefill feedback mapping')
    if (not isinstance(board_model_root, str)
            or not re.fullmatch(r'/[A-Za-z0-9_./-]+', board_model_root)
            or any(part in {'.', '..'} for part in PurePosixPath(board_model_root).parts)):
        raise ValueError('board model root must be a safe absolute path')
    out = output_path(output)
    out.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(out).free < 512 << 20:
        raise ValueError('at least512MiB free host space is required for preparation and final evidence')
    prompt, prompt_source = _prompt(prompt_input_ids, prompt_length)
    lookup = OriginalEmbeddingLookup(weight_table, scale_table)
    Target, Draft = _backends()
    target = Target(decode_job, runtime_probe, state_lineage, lookup, out / 'target-binding',
        source_torchair_pb2=source_torchair_pb2, board_model_root=board_model_root,
        timeout_seconds=timeout_seconds, lock_wait_seconds=lock_wait_seconds)
    target_preflight = target.preflight(prompt)
    resident_abi = _resident_abi(target.binding, runtime_probe, out / 'resident-abi.tsv', runtime_api='file-v2') if resident else None
    draft = Draft(draft_bundle, out / 'draft-binding', runtime_version=target.runtime_version,
        firmware_id=target.firmware_id, timeout_seconds=timeout_seconds,
        lock_wait_seconds=lock_wait_seconds)
    draft_preflight = draft.preflight()
    base = target.binding['base_plan']
    prefill = None
    if prefill_feedback is not None:
        from .board_prefill_binding import prepare_prefill_binding
        prefill = prepare_prefill_binding(prefill_feedback, target.binding, source_torchair_pb2,
            prompt_length, out / 'prefill-binding')
    prefill_abi = None
    if resident:
        prefill_abi = _resident_abi({'base_plan': prefill['base_plan'],
            'active_source_identities': prefill['source_bindings']},
            prefill['base_plan']['runtime_probe'], out / 'prefill-abi.tsv', prefix='prefill', runtime_api='file-v2')
    if (draft._source['ddk_id'] != base['ddk_id']
            or draft._source['outputs'] != [
                {'index': 0, 'name': 'draft_top1', 'dtype': 'int32', 'shape': [1, 15]},
                {'index': 1, 'name': 'draft_hidden', 'dtype': 'float16', 'shape': [1, 15, 2048]}]):
        raise ValueError('board loop requires the matching original complete S64 two-output draft ABI')
    compiled_path = Path(base['compilation']) / 'report.json'
    compiled_raw, compiled_record = _read_small(compiled_path, 32 << 20)
    if compiled_record['sha256'] != base['compilation_report_sha256']:
        raise ValueError('compiled target inventory differs from its same-model binding')
    compiled = json.loads(compiled_raw)
    payloads = []
    for row in compiled.get('generated_artifacts', []):
        relative = row.get('path', '')
        if (not isinstance(relative, str) or not relative.startswith('compile-work/generated/')
                or any(part in {'', '.', '..'} for part in relative.split('/'))
                or not re.fullmatch(r'[A-Za-z0-9_./-]+', relative)
                or type(row.get('bytes')) is not int or row['bytes'] <= 0
                or re.fullmatch(r'[0-9a-f]{64}', str(row.get('sha256'))) is None):
            raise ValueError('compiled target payload inventory contains an invalid declaration')
        payloads.append({'local': str(Path(base['compilation']) / relative),
            'remote': str(PurePosixPath(relative).relative_to('compile-work/generated')),
            'bytes': row['bytes'], 'sha256': row['sha256'], 'record': deepcopy(row),
            'primary': row == compiled.get('omc')})
    if (not 1 <= len(payloads) <= 8192 or sum(row['primary'] for row in payloads) != 1
            or len({row['remote'] for row in payloads}) != len(payloads)):
        raise ValueError('target payload inventory must contain one unique primary OMC')
    tables = {'weight': _hash_source(weight_table, 508559360),
              'scale': _hash_source(scale_table, 993280)}
    model = draft_preflight['model_source']
    abi = _hash_source(Path(draft_bundle) / draft._source['draft_abi']['path'])
    if abi['sha256'] != draft._source['draft_abi']['sha256']:
        raise ValueError('draft ABI changed after the complete source-bundle validation')
    primary = _hash_source(base['primary_model_binding']['path'], base['primary_model_binding']['bytes'])
    if primary['sha256'] != base['model_sha256']:
        raise ValueError('actual target primary differs from the executed same-model SHA')
    pins = {}
    for row in target.binding['active_source_identities']:
        pins[row['path']] = deepcopy(row['identity'])
    if prefill is not None:
        for row in prefill['source_bindings']:
            if row['path'] in pins and pins[row['path']] != row['identity']:
                raise ValueError('prefill/decode source identity differs during request preparation')
            pins[row['path']] = deepcopy(row['identity'])
    for path, identity in draft._source_identities.items():
        pins[path] = deepcopy(identity)
    for row in [*tables.values(), primary, abi, prompt_source,
            *([resident_abi, prefill_abi] if resident_abi else [])]:
        if row['path'] in pins and pins[row['path']] != row['identity']:
            raise ValueError('source identity changed across board-loop preparation')
        pins[row['path']] = deepcopy(row['identity'])
    manifest = {'schema_version': 1, 'artifact_kind': KIND, 'status': 'PREPARED',
        'output': str(out), 'prompt_token_ids': prompt, 'input_snapshot': prompt_source,
        'max_new_tokens': max_new_tokens, 'max_proposals_per_round': max_proposals_per_round,
        'minimum_draft_rounds': minimum_draft_rounds, 'timeout_seconds': timeout_seconds,
        'dump': dump,
        'lock_wait_seconds': lock_wait_seconds, 'board_model_root': board_model_root,
        'target_binding': target.binding, 'target_preflight': target_preflight,
        'target_compiled': compiled, 'target_payloads': payloads, 'target_model': primary,
        'draft': {'model_source': model, 'abi': abi, 'source_bundle': str(Path(draft_bundle).resolve()),
            'outputs': deepcopy(draft._source['outputs']), 'direct_spec': deepcopy(draft._source.get('direct_spec')),
            'ddk_id': draft._source['ddk_id']},
        'tables': tables, 'source_bindings': [{'path': path, 'identity': identity} for path, identity in sorted(pins.items())],
        'model_compilation': 'NOT_RUN', 'device_execution': 'NOT_RUN',
        'numerical_validation': 'NOT_EVALUATED', 'performance': 'NOT_EVALUATED',
        'eos_policy': 'DISABLED_FOR_BOUNDED_EXECUTION_MILESTONE',
        'orchestration': 'BOARD_LOCAL_FRESH_CHILD_PROCESSES_NO_INTER_TOKEN_HOST_DATA',
        'initial_state_policy': 'ALL80_FP16_ZERO_EXPERIMENTAL_CONTROLS',
        'embedding_authenticity': 'FULL_CURRENT_BYTES_BOUND_ORIGINAL_PROVENANCE_NOT_ATTESTED',
        'compiled_weight_access': 'DECLARED_CACHE_HASHES_ONLY_LOCAL_SEND_REQUIRES_FRESH_VALIDATION',
        'model_residency': 'NOT_IMPLEMENTED_FRESH_TARGET_OR_DRAFT_PROCESS_PER_CALL'}
    if prefill is not None:
        manifest['prefill'] = prefill
        manifest['prompt_ingestion'] = prefill['control_recipe']
        manifest['initial_state_policy'] = 'PREFILL_ZERO_STATES_THEN_ALL80_EXPLICITLY_MAPPED_PREFILL_OUTPUTS'
    else:
        manifest['prompt_ingestion'] = 'DECODE1_SERIAL_PROMPT_INGESTION'
    if resident:
        manifest.update(execution_mode='resident-mixed', resident_abi=resident_abi, prefill_abi=prefill_abi,
            target_lifecycle_policy=TARGET_LIFECYCLE_POLICY,
            model_load_api=MIXED_MODEL_LOAD_API, model_load_apis=dict(MIXED_MODEL_LOAD_APIS),
            external_weight_release='NOT_INDEPENDENTLY_VALIDATED_MANAGER_DEINIT_ONLY',
            orchestration='BOARD_LOCAL_SINGLE_PROCESS_V2_PREFILL_TARGET_V1_DRAFT',
            model_residency='REQUESTED_V2_PREFILL_ONCE_V2_DECODE_AND_V1_DRAFT_LOADED_ONCE_UNVALIDATED')
    save_json(out / 'manifest.json', manifest)
    (out / 'manifest.sha256').write_text(sha256_file(out / 'manifest.json') + '\n')
    return validate_prepared(out)


def validate_prepared(directory):
    root = Path(directory).resolve()
    raw, record = _read_small(root / 'manifest.json', MAX_MANIFEST_BYTES)
    pin, _ = _read_small(root / 'manifest.sha256', 65)
    if pin != (record['sha256'] + '\n').encode('ascii'):
        raise ValueError('prepared board-loop manifest SHA256 changed')
    manifest = json.loads(raw)
    if (manifest.get('schema_version') != 1 or manifest.get('artifact_kind') != KIND
            or manifest.get('status') != 'PREPARED' or manifest.get('output') != str(root)):
        raise ValueError('expected the original prepared board-loop manifest')
    prompt = manifest['prompt_token_ids']
    if not isinstance(prompt, list) or any(type(value) is not int or not 0 <= value < 248320 for value in prompt):
        raise ValueError('prepared prompt IDs are invalid')
    _options(len(prompt), manifest['max_new_tokens'], manifest['max_proposals_per_round'],
             manifest['minimum_draft_rounds'], manifest['timeout_seconds'], manifest['lock_wait_seconds'])
    # Existing prepared requests predate the switch and retain their full dump.
    # Recovery never changes a request's original retention/validation policy.
    if type(manifest.get('dump', True)) is not bool:
        raise ValueError('prepared dump mode must be boolean')
    mode = manifest.get('execution_mode', 'fresh-process')
    if mode not in {'fresh-process', 'resident-v2', 'resident-v1', 'resident-mixed'}:
        raise ValueError('prepared execution mode is unsupported')
    if mode in {'resident-v2', 'resident-v1', 'resident-mixed'}:
        if mode == 'resident-v1':
            v1_load_contract(manifest)
        elif mode == 'resident-mixed':
            mixed_load_contract(manifest)
        elif manifest.get('target_lifecycle_policy', V2_TARGET_LIFECYCLE_POLICY) != V2_TARGET_LIFECYCLE_POLICY:
            raise ValueError('prepared resident target lifecycle policy is unsupported')
        abi = manifest.get('resident_abi', {})
        if ('prefill' not in manifest or not isinstance(abi, dict) or abi.get('path') != str(root / 'resident-abi.tsv')
                or _hash_source(abi['path']) != {key: abi[key] for key in ('path', 'bytes', 'sha256', 'identity')}
                or len(abi.get('inputs', [])) != 89 or len(abi.get('outputs', [])) != 82):
            raise ValueError('prepared resident ABI changed or is incomplete')
        if mode in {'resident-v1', 'resident-mixed'}:
            abi = manifest.get('prefill_abi', {})
            if (abi.get('path') != str(root / 'prefill-abi.tsv')
                    or _hash_source(abi['path']) != {key: abi[key] for key in ('path', 'bytes', 'sha256', 'identity')}
                    or len(abi.get('inputs', [])) != 89 or len(abi.get('outputs', [])) != 82):
                raise ValueError('prepared same-process prefill ABI changed or is incomplete')
    elif 'resident_abi' in manifest:
        raise ValueError('resident ABI cannot be attached to a legacy request')
    if 'prefill_abi' in manifest and mode not in {'resident-v1', 'resident-mixed'}:
        raise ValueError('V1 prefill ABI cannot be attached to a historical request')
    if 'target_lifecycle_policy' in manifest and mode not in {'resident-v2', 'resident-v1', 'resident-mixed'}:
        raise ValueError('target lifecycle policy belongs only to an explicit resident request')
    if mode not in {'resident-v1', 'resident-mixed'} and any(key in manifest for key in ('model_load_api', 'model_load_apis')):
        raise ValueError('V1 model source representation cannot be attached to another execution mode')
    for row in manifest['source_bindings']:
        if _current(row['path'])[1] != row['identity']:
            raise ValueError('board-loop source changed after preparation: ' + row['path'])
    return manifest


def _field(value):
    value = str(value)
    if not value or len(value) > 4096 or not value.isascii() or any(char in value for char in '\x00\t\r\n'):
        raise ValueError('invalid bounded TSV field')
    return value


def make_config(manifest, remote_paths):
    """Serialize only explicit validated native ABI rows, never inferred order."""
    base = manifest['target_binding']['base_plan']
    dump = manifest.get('dump', True)
    if type(dump) is not bool:
        raise ValueError('prepared dump mode must be boolean')
    lines = ['QWEN35_BOARD_SERIAL\t1']
    files = {'target_model': manifest['target_model'], 'draft_model': manifest['draft']['model_source'],
        'draft_abi': manifest['draft']['abi'],
        'weight_table': manifest['tables']['weight'], 'scale_table': manifest['tables']['scale']}
    if manifest.get('execution_mode') not in {'resident-v1', 'resident-mixed'}:
        files.update(target_tool=remote_paths['target_tool_record'], draft_runner=remote_paths['draft_runner_record'])
    if 'prefill' in manifest:
        files['prefill_model'] = manifest['prefill']['model']
    resident = manifest.get('execution_mode') in {'resident-v2', 'resident-v1', 'resident-mixed'}
    if resident:
        files['resident_abi'] = manifest['resident_abi']
    if manifest.get('execution_mode') in {'resident-v1', 'resident-mixed'}:
        files['prefill_abi'] = manifest['prefill_abi']
    for key, record in files.items():
        path = str(remote_paths[key])
        if (not path.startswith('/') or any(part in {'.', '..'} for part in PurePosixPath(path).parts)
                or not re.fullmatch(r'[0-9a-f]{64}', str(record['sha256']))
                or type(record['bytes']) is not int or record['bytes'] <= 0):
            raise ValueError('invalid pinned native file configuration: ' + key)
        lines.append('\t'.join(map(_field, (key, path, record['sha256'], record['bytes']))))
    for key, value in {'target_library_path': remote_paths['target_library_path'],
            'draft_library_path': remote_paths['draft_library_path'],
            'runtime_version': base['runtime_version'], 'ddk_id': base['ddk_id'],
            'firmware_id': base['firmware_id'], 'prompt': ','.join(map(str, manifest['prompt_token_ids'])),
            'max_new_tokens': manifest['max_new_tokens'], 'max_proposals': manifest['max_proposals_per_round'],
            'min_draft_rounds': manifest['minimum_draft_rounds'],
            'dump': int(dump),
            'timeout_ms': min(manifest['timeout_seconds'] * 1000, 2**32 - 1)}.items():
        lines.append('\t'.join(map(_field, (key, value))))
    if resident:
        lines.append('execution_mode\t' + manifest['execution_mode'])
        if 'target_lifecycle_policy' in manifest:
            policy = manifest['target_lifecycle_policy']
            if manifest['execution_mode'] == 'resident-v1':
                v1_load_contract(manifest)
            elif manifest['execution_mode'] == 'resident-mixed':
                mixed_load_contract(manifest)
            elif policy != V2_TARGET_LIFECYCLE_POLICY:
                raise ValueError('unsupported resident target lifecycle policy')
            lines.append('target_lifecycle_policy\t' + policy)
    for role, type_key in (('input', 'tool_input_type'), ('output', 'tool_output_type')):
        for row in base[role + 's']:
            name = row['name'] if role == 'input' else row['runtime_tensor_name']
            lines.append('\t'.join(map(_field, ('target_' + role, row['index'], name, row[type_key], row['bytes']))))
    for row in manifest['target_binding']['state_mapping']:
        size = base['inputs'][row['runtime_input_index']]['bytes']
        lines.append('\t'.join(map(_field, ('state', row['source_refdata'], row['runtime_input_index'],
                                            row['runtime_output_index'], size))))
    if 'prefill' in manifest:
        prefill = manifest['prefill']
        from .board_prefill_binding import RECIPE, control_assumption
        if (prefill.get('control_recipe') != RECIPE
                or prefill.get('experimental_control_assumption') != control_assumption(len(manifest['prompt_token_ids']))):
            raise ValueError('prepared padded prefill controls differ from the explicit native recipe')
        for role, type_key in (('input', 'tool_input_type'), ('output', 'tool_output_type')):
            for row in prefill['base_plan'][role + 's']:
                name = row['name'] if role == 'input' else row['runtime_tensor_name']
                lines.append('\t'.join(map(_field, ('prefill_' + role, row['index'], name, row[type_key], row['bytes']))))
        for row in prefill['state_mapping']:
            lines.append('\t'.join(map(_field, ('prefill_state', row['decode']['source_refdata'],
                row['prefill']['source_refdata'], row['prefill']['runtime_input_index'],
                row['prefill']['runtime_output_index'], row['bytes']))))
    return ('\n'.join(lines) + '\n').encode('utf-8')


def run_board_serial_dflash(*, output, apply=False, detach=False, **options):
    if type(apply) is not bool or type(detach) is not bool or detach and not apply:
        raise ValueError('apply/detach must be explicit booleans; detach requires apply')
    out = output_path(output)
    out.mkdir(parents=True, exist_ok=False)
    result = {'schema_version': 1, 'artifact_kind': 'qwen35-board-serial-dflash-job',
        'status': 'FAILED', 'stage': 'host-preflight', 'output': str(out),
        'mode': 'APPLY' if apply else 'PLAN_ONLY', 'model_compilation': 'NOT_RUN',
        'device_execution': 'NOT_RUN', 'numerical_validation': 'NOT_EVALUATED', 'performance': 'NOT_EVALUATED'}
    try:
        manifest = prepare_board_serial(output=out / 'prepared', **options)
        result['prepared'] = manifest['output']
        result['dump'] = manifest.get('dump', True)
        result['prompt_ingestion'] = manifest['prompt_ingestion']
        if manifest.get('execution_mode') in {'resident-v2', 'resident-v1', 'resident-mixed'}:
            result['execution_mode'] = manifest['execution_mode']
        if manifest.get('execution_mode') == 'resident-v1':
            result['target_lifecycle_policy'] = manifest['target_lifecycle_policy']
            result['model_load_api'], result['model_load_apis'] = v1_load_contract(manifest)
        elif manifest.get('execution_mode') == 'resident-mixed':
            result['target_lifecycle_policy'] = manifest['target_lifecycle_policy']
            result['model_load_api'], result['model_load_apis'] = mixed_load_contract(manifest)
            result['external_weight_release'] = 'NOT_INDEPENDENTLY_VALIDATED_MANAGER_DEINIT_ONLY'
        result['retrieval_mode'] = 'FULL_DEBUG' if result['dump'] else 'FINAL_TOKENS_ONLY'
        if apply:
            from .board_serial_transport import execute_board_serial
            result['stage'] = 'board-local-loop'
            execution = execute_board_serial(manifest['output'], out / 'board',
                timeout_seconds=manifest['timeout_seconds'], lock_wait_seconds=manifest['lock_wait_seconds'],
                detach=detach)
            result.update(execution)
            result.update(output=str(out), board_result=str(out / 'board' / 'result.json'),
                          prepared=manifest['output'], mode='APPLY')
        else:
            result.update(status='BOARD_SERIAL_DFLASH_PLANNED', stage='complete',
                next='Run the same arguments with --apply and a new output; no model compilation')
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
        result['next'] = 'Retain this execution; inspect board checkpoint before any retry or lock release'
    save_json(out / 'result.json', result)
    return result
