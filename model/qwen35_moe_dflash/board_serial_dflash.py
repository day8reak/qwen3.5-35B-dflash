"""Prepare an autonomous board-local loop using the existing complete OMCs.

The new supervisor moves orchestration, original embedding row lookup, and all
state/feature feedback onto the board. It still invokes fresh single-model
processes; it is neither model residency nor a speed/accuracy acceptance gate.
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
        board_model_root='/data/local/tmp/qwen3_5_moe'):
    _options(prompt_length, max_new_tokens, max_proposals_per_round, minimum_draft_rounds,
             timeout_seconds, lock_wait_seconds)
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
    draft = Draft(draft_bundle, out / 'draft-binding', runtime_version=target.runtime_version,
        firmware_id=target.firmware_id, timeout_seconds=timeout_seconds,
        lock_wait_seconds=lock_wait_seconds)
    draft_preflight = draft.preflight()
    base = target.binding['base_plan']
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
    for path, identity in draft._source_identities.items():
        pins[path] = deepcopy(identity)
    for row in [*tables.values(), primary, abi, prompt_source]:
        if row['path'] in pins and pins[row['path']] != row['identity']:
            raise ValueError('source identity changed across board-loop preparation')
        pins[row['path']] = deepcopy(row['identity'])
    manifest = {'schema_version': 1, 'artifact_kind': KIND, 'status': 'PREPARED',
        'output': str(out), 'prompt_token_ids': prompt, 'input_snapshot': prompt_source,
        'max_new_tokens': max_new_tokens, 'max_proposals_per_round': max_proposals_per_round,
        'minimum_draft_rounds': minimum_draft_rounds, 'timeout_seconds': timeout_seconds,
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
    lines = ['QWEN35_BOARD_SERIAL\t1']
    files = {'target_model': manifest['target_model'], 'draft_model': manifest['draft']['model_source'],
        'draft_abi': manifest['draft']['abi'], 'target_tool': remote_paths['target_tool_record'],
        'draft_runner': remote_paths['draft_runner_record'],
        'weight_table': manifest['tables']['weight'], 'scale_table': manifest['tables']['scale']}
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
            'timeout_ms': min(manifest['timeout_seconds'] * 1000, 2**32 - 1)}.items():
        lines.append('\t'.join(map(_field, (key, value))))
    for role, type_key in (('input', 'tool_input_type'), ('output', 'tool_output_type')):
        for row in base[role + 's']:
            name = row['name'] if role == 'input' else row['runtime_tensor_name']
            lines.append('\t'.join(map(_field, ('target_' + role, row['index'], name, row[type_key], row['bytes']))))
    for row in manifest['target_binding']['state_mapping']:
        size = base['inputs'][row['runtime_input_index']]['bytes']
        lines.append('\t'.join(map(_field, ('state', row['source_refdata'], row['runtime_input_index'],
                                            row['runtime_output_index'], size))))
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
