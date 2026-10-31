"""Original decode1 TargetBackend for an explicit zero-state serial experiment.

Prompt prefill is a sequence of real decode1 calls, using one unchanged OMC.
Only committed IDs enter it. Zero initialization and local-query controls are
experimental conventions, not validated kernel semantics or accuracy claims.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import struct

import numpy as np

from qwen35_moe_dflash.common.artifacts import output_path, save_json
from qwen35_moe_dflash.toolchain.existing_air_profile import get_existing_air_profile
from qwen35_moe_dflash.draft.incremental_smoke import CallEvidence, ModelIdentity, TargetObservation, feature_sha256, prefix_sha256, _ids
from qwen35_moe_dflash.execution.original_embedding_lookup import OriginalEmbeddingLookup
from qwen35_moe_dflash.common.file_io import _read_small, _safe
from qwen35_moe_dflash.target.decode_smoke_plan import validate_decode_smoke_plan
from qwen35_moe_dflash.target.decode_state_lineage import validate_target_decode_state_lineage
from qwen35_moe_dflash.target.prefill_smoke_plan import _identity, _nd_view_key, _source
from qwen35_moe_dflash.target.prefill_output_inventory import _hash_retrieved


ZERO_STATE_SERIAL_ASSUMPTION = 'ZERO_FP16_STATES_SERIAL_DECODE1_ABSOLUTE_CAUSAL_LOCAL_QUERY_V1'
SOURCE_PRESERVATION = 'UNCHANGED_SMOKE_BACKEND_PROOF_AND_ALL89_PREPARED_INPUT_SHA256'
_BINDING_KIND = 'qwen35-original-target-serial-smoke-binding'
_STEP_KIND = 'qwen35-original-target-serial-smoke-step'
_SCOPE = 'ZERO_STATE_ORIGINAL_DECODE1_SERIAL_COMMITTED_TOKEN_EXPERIMENT'
_MAX_METADATA = 8 << 20


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file(path):
    path, stamp = _safe(path)
    full = _identity(path.stat())
    if [full[key] for key in ('device', 'inode', 'bytes', 'mtime_ns', 'ctime_ns')] != stamp:
        raise ValueError('serial target source changed before its current snapshot')
    return {'path': str(path), 'identity': full}


def _stable(rows):
    for row in rows:
        if _file(row['path']) != row:
            raise ValueError('serial target source changed during this invocation: ' + row['path'])


def _metadata(path):
    raw, identity = _read_small(path, _MAX_METADATA)
    value = json.loads(raw)
    full = _file(path)
    if [full['identity'][key] for key in ('device', 'inode', 'bytes', 'mtime_ns', 'ctime_ns')] != identity['stat']:
        raise ValueError('serial target metadata changed after its fresh read')
    return value, dict(identity, identity=full['identity'])


def _state_digest(states):
    return _digest([{'source_refdata': row['source_refdata'], 'bytes': row['output_file']['bytes'],
        'sha256': row['output_file']['sha256']} for row in sorted(states, key=lambda value: value['source_refdata'])])


def _controls(position):
    if type(position) is not int or not 0 <= position < 64:
        raise ValueError('serial smoke supports only the bounded old S64 draft prefix')
    return {'arg107_1': struct.pack('<i', position), 'arg2_1': struct.pack('<i', position),
        'arg113_1': bytes(4 * (position + 1)) + struct.pack('<f', float('-inf')) * (2047 - position),
        'arg1238_1': struct.pack('<i', 0), 'arg114_1': struct.pack('<q', 1),
        'arg18_1': struct.pack('<h', 1), 'arg14_1': struct.pack('<128i', 1, *([0] * 127))}


def _mapping(base, mapping):
    if (base.get('profile') != 'cpro-40block-decode-1' or base.get('gear') != 1
            or base.get('sequence_length') != 1 or len(base.get('inputs', [])) != 89 or len(base.get('outputs', [])) != 82
            or len(mapping) != 80 or {row.get('source_refdata') for row in mapping} != {f'RefData_{i}' for i in range(80)}
            or len({row.get('runtime_input_index') for row in mapping}) != 80
            or len({row.get('runtime_output_index') for row in mapping}) != 80):
        raise ValueError('serial target requires the complete original static1 ABI and all 80 unique state owners')
    for row in mapping:
        a, b = base['inputs'][row['runtime_input_index']], base['outputs'][row['runtime_output_index']]
        if (a['name'] != row['source_refdata'] or a['runtime_tensor_name'] != row['runtime_input_name']
                or b['runtime_tensor_name'] != row['runtime_output_name'] or a['dtype'] != 'FLOAT16'
                or b['dtype'] != 'FLOAT16' or a['bytes'] != b['bytes']
                or _nd_view_key((a['dtype'], tuple(a['shape']))) != _nd_view_key((b['dtype'], tuple(b['shape'])))):
            raise ValueError('serial target state mapping differs from the exact compiled native ABI')


def _copy(parent, destination):
    expected = parent['output_file']
    fresh = _source(expected['path'], expected['bytes'])
    if (fresh['sha256'] != expected['sha256'] or fresh['identity'] != expected['identity']):
        raise ValueError('committed parent state bytes or current full identity changed')
    fd = os.open(expected['path'], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    digest, count = hashlib.sha256(), 0
    with os.fdopen(fd, 'rb') as source, destination.open('xb') as target:
        if _identity(os.fstat(source.fileno())) != fresh['identity']:
            raise ValueError('committed state changed before copy')
        remaining = expected['bytes']
        while remaining:
            block = source.read(min(64 << 10, remaining))
            if not block:
                raise ValueError('committed state truncated during copy')
            count += len(block); digest.update(block); target.write(block)
            remaining -= len(block)
        if source.read(1):
            raise ValueError('committed state grew beyond its fixed byte extent')
        if _identity(os.fstat(source.fileno())) != fresh['identity']:
            raise ValueError('committed state changed during copy')
    if count != expected['bytes'] or digest.hexdigest() != expected['sha256']:
        raise ValueError('committed state copy differs from its parent SHA256')
    _stable([{'path': expected['path'], 'identity': fresh['identity']}])
    return _source(destination, count)


def _make_plan(binding, directory, sources, states, prefix, token):
    plan = deepcopy(binding['base_plan'])
    for key in ('control_observations', 'warnings', 'input_preflight', 'trial_input_receipt',
                'trial_input_preservation', 'experimental_input_assumption'):
        plan.pop(key, None)
    inputs = [dict(row, source=sources[row['name']]['path'],
        source_filename=Path(sources[row['name']]['path']).name, sha256=sources[row['name']]['sha256'],
        source_semantics=_SCOPE) for row in plan['inputs']]
    plan.update(artifact_kind='qwen35-original-target-serial-smoke-plan', status='ORIGINAL_TARGET_SMOKE_STEP_PREPARED',
        input_dir=str(directory), inputs=inputs, source_inputs=sorted(sources.values(), key=lambda row: row['path']),
        state_feedback_bindings=deepcopy(states), execution_scope=_SCOPE,
        source_semantics='EXPERIMENTAL_ZERO_STATE_OR_REAL_PREVIOUS_CALL_OUTPUTS',
        experimental_input_assumption={'id': ZERO_STATE_SERIAL_ASSUMPTION,
            'contract_status': 'EXPERIMENTAL_NOT_A_VALIDATED_KERNEL_RECIPE', 'absolute_position': len(prefix),
            'output_pos': 0, 'allQLen': 1, 'effective_length': 1, 'conv_state_pos': [1] + [0] * 127,
            'visible_mask_columns': len(prefix) + 1, 'initial_states': 'ALL80_FP16_ZERO_BYTES',
            'history': 'ONLY_ACTUAL_SERIAL_CALLS_FROM_THE_DECLARED_SHORT_PROMPT'},
        prefix_before=list(prefix), committed_token_id=token, prefix_after=[*prefix, token],
        logical_input_bytes=sum(row['bytes'] for row in inputs), unique_source_input_bytes=sum(row['bytes'] for row in sources.values()),
        state_input_layer_identity='ALL80_SAME_MODEL_STRUCTURAL_OWNERS_BOUND',
        state_output_layer_identity='ALL80_SAME_MODEL_STRUCTURAL_OWNERS_BOUND',
        numerical_validation='NOT_RUN_NO_REFERENCE', state_continuity='BYTE_CHAIN_ONLY_EXPERIMENTAL_CONTROLS',
        model_compilation='NOT_RUN', device_execution='NOT_RUN', weight_payload_access='NO_COMPILED_EXTERNAL_WEIGHTS_OPENED',
        compiled_weight_payload_access='NONE', embedding_table_payload_access='SELECTED_ORIGINAL_ROWS_ONLY',
        state_feedback='PREPARED_ZERO_STATE_OR_SAME_MODEL_CHILD_STATE_BYTES_NOT_EXECUTED',
        scope='Original complete 40-layer decode1 step. Original INT8 embedding plus separate FP32 scale, 80 explicit zero or preceding child states, and newly generated experimental local-query controls. No accuracy or validated kernel/control contract.')
    plan['control_observations'] = deepcopy(plan['experimental_input_assumption'])
    plan['warnings'] = ['Zero initialization and local-query controls are experimental; no numerical accuracy accepted']
    return plan


def validate_smoke_backend_step(prepared):
    """Validate one immutable 89-file step for the board's cached payload path."""
    root, _ = _safe(prepared, directory=True)
    manifest, manifest_id = _metadata(root / 'manifest.json')
    plan, plan_id = _metadata(root / 'plan.json')
    if (manifest.get('artifact_kind') != _STEP_KIND or manifest.get('status') != 'PREPARED'
            or manifest.get('assumption') != ZERO_STATE_SERIAL_ASSUMPTION or manifest.get('output') != str(root)
            or manifest.get('plan') != plan_id):
        raise ValueError('expected an unchanged explicitly experimental serial target step')
    binding, binding_id = _metadata(manifest['backend_binding']['path'])
    if (binding_id != manifest['backend_binding'] or binding.get('artifact_kind') != _BINDING_KIND
            or binding.get('status') != 'BOUND_COMPLETE40_DECODE1_ZERO_STATE_EXPERIMENT'
            or binding.get('assumption') != ZERO_STATE_SERIAL_ASSUMPTION
            or binding.get('target_layers') != 40 or binding.get('vocab_size') != 248320
            or binding.get('feature_width') != 16384):
        raise ValueError('serial step backend binding changed or differs from the complete model')
    baselines = binding['active_source_identities']
    _stable(baselines)
    prefix = _ids(manifest['prefix_before'], 248320, 'serial prefix')
    token = _ids([manifest['token_id']], 248320, 'committed serial token')[0]
    controls = _controls(len(prefix))
    mapping = binding['state_mapping']; _mapping(binding['base_plan'], mapping)
    states = manifest['parent_states']
    if (len(states) != 80 or {row['source_refdata'] for row in states} != {row['source_refdata'] for row in mapping}
            or _state_digest(states) != manifest['input_state_sha256']):
        raise ValueError('serial step parent state set changed')
    parent_metadata = []
    parent_call = manifest.get('parent_native_receipt')
    if prefix:
        if not isinstance(parent_call, dict):
            raise ValueError('serial history requires its preceding actual native call receipt')
        receipt, receipt_id = _metadata(parent_call['path'])
        if (receipt_id != parent_call or receipt.get('artifact_kind') != 'qwen35-original-target-serial-native-call'
                or receipt.get('status') != 'EXECUTED_UNVALIDATED' or receipt.get('remote_exitcode') != 0
                or receipt.get('model_sha256') != binding['base_plan']['model_sha256']
                or receipt.get('prefix_after') != prefix or receipt.get('output_state_sha256') != manifest['input_state_sha256']):
            raise ValueError('serial history differs from the preceding same-model actual call')
        chain, chain_id = _metadata(receipt['state_chain_identity']['path'])
        previous_board, board_id = _metadata(receipt['board_result_identity']['path'])
        if (chain_id != receipt['state_chain_identity'] or board_id != receipt['board_result_identity']
                or chain.get('status') != 'ALL80_PARENT_OUTPUT_BYTES_USED_BY_CHILD_INPUTS'
                or previous_board.get('status') != 'EXECUTED_UNVALIDATED' or previous_board.get('remote_exitcode') != 0
                or previous_board.get('tool_process_id') != receipt.get('tool_process_id')):
            raise ValueError('preceding native call no longer binds its actual board/state-chain receipts')
        children = {row['source_refdata']: row['child_output'] for row in chain['states']}
        if len(children) != 80:
            raise ValueError('preceding native call state-chain is incomplete')
        for state in states:
            child = children[state['source_refdata']]; parent = state['output_file']
            if (parent['path'] != str(Path(board_id['path']).parent / child['path'])
                    or any(parent[key] != child[key] for key in ('sha256', 'bytes'))
                    or [parent['identity'][key] for key in ('device', 'inode', 'bytes', 'mtime_ns', 'ctime_ns')] != child['stat']):
                raise ValueError('serial parent state is not the preceding actual child output file')
        parent_metadata = [{'path': row['path'], 'identity': row['identity']} for row in (receipt_id, chain_id, board_id)]
    elif parent_call is not None:
        raise ValueError('zero-state first step must not borrow an unrelated decode history')
    by_ref = {row['source_refdata']: row for row in mapping}
    payloads = []
    for state in states:
        proven = by_ref[state['source_refdata']]
        if any(state.get(key) != proven[key] for key in ('runtime_input_index', 'runtime_output_index')):
            raise ValueError('serial parent state uses another runtime slot')
        parent = state['output_file']; observed = _source(parent['path'], parent['bytes'])
        if observed['sha256'] != parent['sha256'] or observed['identity'] != parent['identity']:
            raise ValueError('serial parent state changed before the actual board input handoff')
        payloads.append({'path': parent['path'], 'identity': parent['identity']})
        if not prefix and parent['sha256'] != hashlib.sha256(bytes(parent['bytes'])).hexdigest():
            raise ValueError('serial prompt must start from all 80 explicit FP16 zero states')
        if not prefix and parent['path'] != str(Path(binding_id['path']).parent / 'initial-states' / (state['source_refdata'] + '.bin')):
            raise ValueError('zero-state first step must use this fresh backend initialization')
    lookup_observation = manifest['embedding_lookup']
    tables = {row['role']: row for row in lookup_observation['tables']}
    lookup = OriginalEmbeddingLookup(tables['weight']['path'], tables['scale']['path'])
    selected = lookup.lookup([token])
    if (selected.embedding_bytes.hex() != manifest['embedding_row_hex'] or selected.scale_bytes.hex() != manifest['scale_hex']
            or any(selected.observation['tables'][i]['identity'] != lookup_observation['tables'][i]['identity'] for i in range(2))):
        raise ValueError('serial embedding row/scale or declared table identity changed')
    table_baselines = [{'path': row['path'], 'identity': row['identity']} for row in selected.observation['tables']]
    raw = {'arg1_1': selected.embedding_bytes, 'arg0_1': selected.scale_bytes, **controls}
    sources = {}
    expected_sources = {row['path']: row for row in plan['source_inputs']}
    if len(expected_sources) != 89 or len(plan['inputs']) != 89:
        raise ValueError('serial step must retain 89 distinct prepared input files')
    for row in plan['inputs']:
        path = root / 'inputs' / row['source_filename']
        if row['source'] != str(path) or path.name != row['source_filename']:
            raise ValueError('serial input escaped its unique owned input directory')
        actual = _source(path, row['bytes']); expected = expected_sources.get(str(path))
        if actual != expected or actual['identity']['nlink'] != 1:
            raise ValueError('serial prepared input bytes or independent full identity changed')
        name = row['name']
        if name in raw:
            if actual['sha256'] != hashlib.sha256(raw[name]).hexdigest():
                raise ValueError('serial token embedding or declared experimental controls differ')
        else:
            state = next(item for item in states if item['source_refdata'] == name)
            if actual['sha256'] != state['output_file']['sha256'] or actual['bytes'] != state['output_file']['bytes']:
                raise ValueError('serial prepared state differs from its actual parent output')
        sources[name] = actual; payloads.append({'path': actual['path'], 'identity': actual['identity']})
    if len({(row['identity']['device'], row['identity']['inode']) for row in sources.values()}) != 89:
        raise ValueError('serial prepared input files must be independently owned')
    if _make_plan(binding, root / 'inputs', sources, states, prefix, token) != plan:
        raise ValueError('serial step plan changed from its exact bound model/inputs/controls')
    _stable([*baselines, *payloads, *table_baselines, *parent_metadata, {'path': manifest_id['path'], 'identity': manifest_id['identity']},
        {'path': plan_id['path'], 'identity': plan_id['identity']}, {'path': binding_id['path'], 'identity': binding_id['identity']}])
    return plan


class OriginalTargetSmokeBackend:
    """One original 40-layer decode1 OMC, zero-state serial prompt and advance."""

    def __init__(self, decode_job, runtime_probe, state_lineage, embedding_lookup, output, *,
            source_torchair_pb2=None, assumption=ZERO_STATE_SERIAL_ASSUMPTION,
            board_model_root='/data/local/tmp/qwen3_5_moe', timeout_seconds=7200, lock_wait_seconds=1800,
            evidence_kind='device-hiai'):
        if assumption != ZERO_STATE_SERIAL_ASSUMPTION or evidence_kind not in {'device-hiai', 'host-fake-hiai'}:
            raise ValueError('explicit original zero-state serial assumption and valid receipt kind required')
        if not isinstance(embedding_lookup, OriginalEmbeddingLookup):
            raise ValueError('the original INT8 row and separate FP32 scale lookup is required')
        self.output = output_path(output); self.output.mkdir(parents=True, exist_ok=False)
        self.lookup = embedding_lookup
        self.board_options = dict(board_model_root=board_model_root, timeout_seconds=timeout_seconds, lock_wait_seconds=lock_wait_seconds)
        current = validate_target_decode_state_lineage(state_lineage)
        if (current.get('status') != 'ALL80_SAME_MODEL_STATE_BINDINGS_OBSERVED'
                or current['decode']['job'] != str(Path(decode_job).resolve())):
            raise ValueError('serial target requires the freshly validated same-model decode state lineage')
        base, base_id = _metadata(Path(current['execution']) / 'plan.json')
        base = validate_decode_smoke_plan(base)
        if (base['runtime_probe'] != str(Path(runtime_probe).resolve())
                or base['model_sha256'] != current['decode']['model_sha256']):
            raise ValueError('serial target must use the identical named decode1 runtime probe and compiled model')
        mapping = [{key: row[key] for key in ('source_refdata', 'runtime_input_index', 'runtime_input_name',
            'runtime_output_index', 'runtime_output_name')} for row in current['decode']['state_file_mapping']['states']]
        _mapping(base, mapping)
        baselines = []
        for row in [*current['source_bindings'], *base['metadata_bindings'], base_id, current['saved_report_identity']]:
            full = _file(row['path'])
            if [full['identity'][key] for key in ('device', 'inode', 'bytes', 'mtime_ns', 'ctime_ns')] != row['stat']:
                raise ValueError('serial source proof changed after its fresh same-model validation')
            if full not in baselines: baselines.append(full)
        if source_torchair_pb2 is not None:
            _, identity = _read_small(source_torchair_pb2, 4 << 20)
            candidates = [row for row in current['source_bindings'] if row['path'].endswith('_pb2.py')]
            if not candidates or identity['sha256'] not in {row['sha256'] for row in candidates}:
                raise ValueError('explicit producer schema copy differs from the validated decode source schema')
            baselines.append(_file(source_torchair_pb2))
        self.binding = {'schema_version': 1, 'artifact_kind': _BINDING_KIND,
            'status': 'BOUND_COMPLETE40_DECODE1_ZERO_STATE_EXPERIMENT', 'assumption': assumption,
            'target_layers': 40, 'vocab_size': 248320, 'feature_width': 16384,
            'decode_job': str(Path(decode_job).resolve()), 'runtime_probe': str(Path(runtime_probe).resolve()),
            'base_plan': base, 'state_mapping': mapping, 'active_source_identities': baselines,
            'source_lineage': current['saved_report_identity'], 'old_execution_state_values': 'DISCARDED_NOT_USED_AS_PROMPT_HISTORY',
            'numerical_validation': 'NOT_RUN_NO_REFERENCE', 'embedding_table_authenticity': 'CALLER_DECLARED_PATHS_AND_OPTIONAL_HASH_PINS'}
        _stable(baselines)
        save_json(self.output / 'backend-binding.json', self.binding)
        _, self.binding_id = _metadata(self.output / 'backend-binding.json')
        self.identity = ModelIdentity('target', 40, 248320, 16384, base['model_sha256'],
            self.binding_id['sha256'], evidence_kind)
        self._runtime_version = base['runtime_version']
        self._firmware_id = base['firmware_id']
        self.states = []
        initial = self.output / 'initial-states'; initial.mkdir()
        for row in mapping:
            size = base['inputs'][row['runtime_input_index']]['bytes']
            path = initial / (row['source_refdata'] + '.bin')
            with path.open('xb') as stream: stream.write(bytes(size))
            item = _source(path, size)
            self.states.append({**row, 'output_file': {key: item[key] for key in ('path', 'bytes', 'sha256', 'identity')},
                'initialization': 'EXPLICIT_FP16_ZERO_BYTES'})
        self.prefix, self.native_calls = [], []
        self.features = np.empty((1, 0, 16384), dtype='<f2')
        self.failed = False
        self._failed_step, self._retained_board_result, self._failure = None, None, None
        self.state_sha256 = _state_digest(self.states)
        _stable(baselines)

    @property
    def runtime_version(self):
        return self._runtime_version

    @property
    def firmware_id(self):
        return self._firmware_id

    @property
    def native_call_count(self):
        return len(self.native_calls)

    def execution_summary(self):
        return {'native_calls': len(self.native_calls), 'receipt_paths': [row['path'] for row in self.native_calls],
            'validated_native_calls': bool(self.native_calls) and not self.failed,
            'model_sha256': self.identity.model_sha256, 'failed': self.failed,
            'failed_step': self._failed_step, 'retained_board_result': self._retained_board_result, 'error': self._failure,
            'numerical_validation': 'NOT_RUN_NO_REFERENCE', 'assumption': ZERO_STATE_SERIAL_ASSUMPTION}

    def preflight(self, prompt):
        tokens = _ids(prompt, 248320, 'short original prompt')
        if not 1 <= len(tokens) <= 64:
            raise ValueError('short prompt must contain 1..64 actual token IDs')
        selected = self.lookup.lookup(tokens)
        _stable(self.binding['active_source_identities'])
        return {'status': 'ORIGINAL_TARGET_SERIAL_SMOKE_HOST_PREPARED', 'device_execution': 'NOT_RUN',
            'prompt_token_ids': tokens, 'prompt_length': len(tokens), 'planned_prompt_native_calls': len(tokens),
            'initial_state_sha256': self.state_sha256, 'initialization': 'ALL80_FP16_ZERO_BYTES',
            'assumption': ZERO_STATE_SERIAL_ASSUMPTION, 'model_sha256': self.identity.model_sha256,
            'model_compilation': 'NOT_RUN', 'weight_payload_access': 'NO_COMPILED_EXTERNAL_WEIGHTS_OPENED',
            'compiled_weight_payload_access': 'NONE', 'embedding_table_payload_access': 'SELECTED_ORIGINAL_ROWS_ONLY',
            'embedding_lookup': selected.observation, 'numerical_validation': 'NOT_RUN_NO_REFERENCE'}

    def _step(self, token):
        if self.failed:
            raise RuntimeError('serial target request failed; no automatic model retry is allowed')
        token = _ids([token], 248320, 'committed original token')[0]
        root = self.output / ('step_' + str(len(self.native_calls)).zfill(4)); root.mkdir()
        directory = root / 'prepared'; directory.mkdir(); inputs = directory / 'inputs'; inputs.mkdir()
        try:
            selected = self.lookup.lookup([token]); source_states = deepcopy(self.states)
            sources = {row['source_refdata']: _copy(row, inputs / (row['source_refdata'] + '.bin')) for row in source_states}
            for name, raw in {'arg1_1': selected.embedding_bytes, 'arg0_1': selected.scale_bytes, **_controls(len(self.prefix))}.items():
                path = inputs / (name + '.bin')
                with path.open('xb') as stream: stream.write(raw)
                sources[name] = _source(path, len(raw))
            plan = _make_plan(self.binding, inputs, sources, source_states, self.prefix, token)
            save_json(directory / 'plan.json', plan); _, plan_id = _metadata(directory / 'plan.json')
            manifest = {'schema_version': 1, 'artifact_kind': _STEP_KIND, 'status': 'PREPARED', 'output': str(directory),
                'assumption': ZERO_STATE_SERIAL_ASSUMPTION, 'backend_binding': self.binding_id, 'plan': plan_id,
                'token_id': token, 'prefix_before': list(self.prefix), 'parent_states': source_states,
                'input_state_sha256': self.state_sha256, 'embedding_lookup': selected.observation,
                'parent_native_receipt': deepcopy(self.native_calls[-1]) if self.native_calls else None,
                'embedding_row_hex': selected.embedding_bytes.hex(), 'scale_hex': selected.scale_bytes.hex()}
            save_json(directory / 'manifest.json', manifest)
            if validate_smoke_backend_step(directory) != plan:
                raise ValueError('prepared serial target plan changed before execution')
            from qwen35_moe_dflash.target.original_target_board import execute_target_tool
            from qwen35_moe_dflash.target.decode_continuation_run import inspect_continuation_outputs
            base = self.binding['base_plan']
            observed = execute_target_tool(base['compilation'], Path(base['compilation']).parent / 'target-contract/report.json', root / 'board',
                profile=get_existing_air_profile('cpro-40block-decode-1'), gear=1, host_input_dir=str(inputs),
                target_features=True, runtime_smoke_backend_prepared=directory, tool=base['tool_binding']['path'], **self.board_options)
            save_json(root / 'board-result.json', observed)
            if (observed.get('status') != 'EXECUTED_UNVALIDATED' or observed.get('remote_exitcode') != 0
                    or observed.get('device_execution') != 'EXECUTED_OUTPUTS_RETRIEVED'):
                detail = {key: observed[key] for key in ('stage', 'status', 'remote_exitcode', 'error', 'evidence_error', 'source_error') if key in observed}
                raise RuntimeError('original decode1 execution failed: ' + json.dumps(detail, ensure_ascii=False)
                    + '; board_result=' + str(root / 'board/result.json'))
            chain = inspect_continuation_outputs(plan, root / 'board', observed, expected_source_preservation=SOURCE_PRESERVATION)
            save_json(root / 'state-chain.json', chain)
            output_baselines = []
            for row in [*[item['child_output'] for item in chain['states']], *chain['bound_tensors'].values()]:
                full = _file(root / 'board' / row['path'])
                if [full['identity'][key] for key in ('device', 'inode', 'bytes', 'mtime_ns', 'ctime_ns')] != row['stat']:
                    raise ValueError('actual child output changed before the final active snapshot')
                output_baselines.append(full)
            for name in ('tool.stdout', 'tool.stderr', 'tool.exitcode', 'tool.pid'):
                actual, _ = _hash_retrieved(root / 'board' / name, observed['retrieved'][name])
                full = _file(root / 'board' / name)
                if [full['identity'][key] for key in ('device', 'inode', 'bytes', 'mtime_ns', 'ctime_ns')] != actual['stat']:
                    raise ValueError('actual child log changed after its final hash')
                output_baselines.append(full)
            logits, feature = chain['bound_tensors']['logits'], chain['bound_tensors']['feature']
            argmax = logits['statistics'].get('argmax', {})
            if argmax.get('status') != 'OBSERVED_ALL_FINITE' or type(argmax.get('token_id')) is not int:
                raise ValueError('actual original logits provide no all-finite next-token observation')
            path = root / 'board' / feature['path']
            raw, identity = _read_small(path, 32768)
            if len(raw) != 32768 or identity['sha256'] != feature['sha256'] or identity['stat'] != feature['stat']:
                raise ValueError('actual original feature row changed after output inspection')
            chunk = np.frombuffer(raw, dtype='<f2').reshape(1, 1, 16384).copy()
            after = []
            by_ref = {row['source_refdata']: row for row in self.binding['state_mapping']}
            for row in chain['states']:
                child = row['child_output']; file = root / 'board' / child['path']
                full = _file(file)
                if [full['identity'][key] for key in ('device', 'inode', 'bytes', 'mtime_ns', 'ctime_ns')] != child['stat']:
                    raise ValueError('actual original child state changed after output inspection')
                after.append({**by_ref[row['source_refdata']], 'output_file': {
                    'path': str(file), 'bytes': child['bytes'], 'sha256': child['sha256'], 'identity': full['identity']}})
            if validate_smoke_backend_step(directory) != plan:
                raise ValueError('serial input evidence changed during actual output inspection')
            _, chain_id = _metadata(root / 'state-chain.json')
            _, board_id = _metadata(root / 'board/result.json')
            output_baselines.extend({'path': row['path'], 'identity': row['identity']} for row in (chain_id, board_id))
            receipt = {'schema_version': 1, 'artifact_kind': 'qwen35-original-target-serial-native-call',
                'status': 'EXECUTED_UNVALIDATED', 'evidence_kind': self.identity.evidence_kind,
                'model_sha256': self.identity.model_sha256, 'token_id': token,
                'prefix_before': list(self.prefix), 'prefix_after': [*self.prefix, token],
                'input_state_sha256': self.state_sha256, 'output_state_sha256': _state_digest(after),
                'feature_output_sha256': feature_sha256(chunk), 'next_token_id': argmax['token_id'],
                'board_result': str(root / 'board/result.json'), 'state_chain': str(root / 'state-chain.json'),
                'board_result_identity': board_id, 'state_chain_identity': chain_id,
                'tool_process_id': observed['tool_process_id'], 'remote_exitcode': observed['remote_exitcode'],
                'assumption': ZERO_STATE_SERIAL_ASSUMPTION, 'numerical_validation': 'NOT_RUN_NO_REFERENCE'}
            save_json(root / 'receipt.json', receipt); _, receipt_id = _metadata(root / 'receipt.json')
            _stable(output_baselines)
            self.prefix.append(token); self.features = np.concatenate((self.features, chunk), axis=1)
            before = self.state_sha256; self.states, self.state_sha256 = after, receipt['output_state_sha256']
            self.native_calls.append(receipt_id)
            evidence = CallEvidence('original-target-' + str(len(self.native_calls)).zfill(4), self.identity.model_sha256,
                receipt_id['sha256'], prefix_sha256(self.prefix), self.identity.evidence_kind,
                input_state_sha256=before, output_state_sha256=self.state_sha256, feature_output_sha256=feature_sha256(chunk))
            return TargetObservation(argmax['token_id'], len(self.prefix), prefix_sha256(self.prefix), self.features.copy(),
                self.state_sha256, evidence)
        except Exception as error:
            self.failed = True
            self._failed_step, self._failure = str(root), f'{type(error).__name__}: {error}'
            board_result = root / 'board/result.json'
            self._retained_board_result = str(board_result) if board_result.exists() else None
            save_json(root / 'failure.json', {'status': 'FAILED', 'error': f'{type(error).__name__}: {error}',
                'model_retry': 'NOT_REQUESTED', 'native_calls_before_failure': len(self.native_calls),
                'failed_step': self._failed_step, 'retained_board_result': self._retained_board_result})
            raise

    def prefill(self, prompt):
        if self.prefix or self.native_calls:
            raise ValueError('serial prefill requires one fresh zero-state backend')
        tokens = self.preflight(prompt)['prompt_token_ids']
        initial = self.state_sha256
        for token in tokens:
            result = self._step(token)
        receipt = {'schema_version': 1, 'artifact_kind': 'qwen35-original-target-serial-prefill-composite',
            'status': 'EXECUTED_UNVALIDATED', 'evidence_kind': self.identity.evidence_kind,
            'model_sha256': self.identity.model_sha256, 'native_call_count': len(tokens), 'native_calls': deepcopy(self.native_calls),
            'prefix_sha256': prefix_sha256(tokens), 'input_state_sha256': initial, 'output_state_sha256': self.state_sha256,
            'feature_output_sha256': feature_sha256(self.features), 'feature_binding': 'CONCATENATION_OF_ALL_ACTUAL_NATIVE_FEATURE_ROWS',
            'next_token_id': result.next_token_id, 'numerical_validation': 'NOT_RUN_NO_REFERENCE'}
        save_json(self.output / 'prefill-receipt.json', receipt); _, identity = _metadata(self.output / 'prefill-receipt.json')
        evidence = CallEvidence('original-target-serial-prefill', self.identity.model_sha256, identity['sha256'],
            prefix_sha256(tokens), self.identity.evidence_kind, input_state_sha256=initial,
            output_state_sha256=self.state_sha256, feature_output_sha256=feature_sha256(self.features))
        return TargetObservation(result.next_token_id, len(tokens), prefix_sha256(tokens), self.features.copy(), self.state_sha256, evidence)

    def advance(self, committed_token):
        if not self.prefix:
            raise ValueError('serial target advance requires an actual completed prompt prefill')
        return self._step(committed_token)
