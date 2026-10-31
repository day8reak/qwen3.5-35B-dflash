"""Stage once and supervise one detached board-local serial DFlash request.

No HDC call advances an individual token.  A lost launch response is never
replayed: recovery only reads the recorded attempt and releases its exact locks
after the supervisor and every recorded child have completed.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import struct
import time
import uuid

from .artifacts import output_path, save_json
from .draft_board import _build_runner, _receive
from .elf_dynamic import dynamic_info
from .hashing import sha256_file
from .hdc_run import Hdc, _check_local, _file_record, _relative, _text
from .hiai_deployment import BOARD_LIBRARY_SEARCH_PATHS, DDK_HCL_EXCLUDED, candidate_blockers, discover_libraries
from .target_compile_failure import _read_small, _safe
from .target_prefill_smoke_plan import _identity


_KIND = 'qwen35-board-serial-dflash-execution'
_GLOBAL = PurePosixPath('/data/local/tmp/.qwen35-target-model-io-lock')
_SHA = re.compile(r'[0-9a-f]{64}')
_TOKEN = re.compile(r'[0-9a-f]{32}')
_ATTEMPT_LOGS = ('supervisor.stdout', 'supervisor.stderr', 'tool.pid', 'tool.exitcode', 'run.sh')
_MAX_FILE = 128 << 20
_MAX_TOTAL = 2 << 30
_PREFILL_RECIPE = 'PREFILL64_PADDED_SHORT_PROMPT_EXPERIMENTAL'


def _has_prefill(manifest):
    value = manifest.get('prefill')
    if value is None:
        return False
    if not isinstance(value, dict) or value.get('control_recipe') != _PREFILL_RECIPE:
        raise ValueError('prefill requires the explicit padded-short-prompt experimental contract')
    return True


def _resident_mode(manifest):
    mode = manifest.get('execution_mode', 'fresh-process')
    if mode not in {'fresh-process', 'resident-v2', 'resident-v1'}:
        raise ValueError('unsupported pinned board execution mode')
    return mode in {'resident-v2', 'resident-v1'}


def _v1_mode(manifest):
    _resident_mode(manifest)
    return manifest.get('execution_mode') == 'resident-v1'


def _new_target_policy(manifest):
    value = manifest.get('target_lifecycle_policy')
    if value is None:
        return False
    if _v1_mode(manifest):
        from .board_serial_dflash import v1_load_contract
        v1_load_contract(manifest)
        return True
    expected = 'init-before-weights-close-weights-first-v1'
    if value != expected or not _resident_mode(manifest):
        raise ValueError('unsupported pinned target lifecycle policy')
    return True


def _dump_mode(manifest):
    # Original prepared/config records predate the flag and retain full audit.
    value = manifest.get('dump', True)
    if type(value) is not bool:
        raise ValueError('prepared dump mode must be an explicit boolean')
    return value


def _result_scope(state, manifest):
    dump = _dump_mode(manifest)
    state.update(dump=dump, retrieval_mode='FULL_DEBUG' if dump else 'FINAL_TOKENS_ONLY',
        evidence_scope=('FULL_RETRIEVED_NATIVE_IO_AND80_STATE_BYTE_CHAIN' if dump
        else 'TOKENS_AND_SUPERVISOR_COMPLETION_ONLY_NO_HOST_STATE_CHAIN_REPLAY'))
    if not dump:
        state['state_continuity'] = 'NOT_REPLAYED_COMPACT_RESULT'
    if _has_prefill(manifest):
        state['prompt_ingestion'] = _PREFILL_RECIPE
    if _resident_mode(manifest):
        state['execution_mode'] = manifest['execution_mode']
        if _new_target_policy(manifest):
            state['target_lifecycle_policy'] = manifest['target_lifecycle_policy']
        if _v1_mode(manifest):
            from .board_serial_dflash import v1_load_contract
            state['model_load_api'], state['model_load_apis'] = v1_load_contract(manifest)


def _next_action(state):
    if state.get('status') == 'BOARD_SERIAL_DFLASH_EXECUTED_UNVALIDATED':
        return 'Completed request recovered and owned locks released; retain this result. Numerical accuracy and performance remain unvalidated.'
    if state.get('stage') == 'complete' and state.get('completion', {}).get('remote_exitcode', 0) != 0:
        return 'Completed failed request retained and owned locks released; inspect its failure evidence before a separate corrected request.'
    if state.get('launch_requested'):
        return 'Recover this exact execution directory; never relaunch or delete uncertain locks'
    return 'Inspect the retained deployment phase and original command completion before any retry or lock cleanup'


def _validate_summaries(loop, manifest, *, required=False):
    """Acceptance arithmetic and wall-clock phases are summaries, not accuracy gates."""
    if not required and 'acceptance' not in loop and 'phase_latency' not in loop:
        return
    counts = loop['counts']; acceptance = loop.get('acceptance', {})
    if (not isinstance(acceptance, dict) or any(type(acceptance.get(key)) is not int for key in ('accepted', 'considered'))
            or acceptance.get('accepted') != counts['accepted']
            or acceptance.get('considered') != counts['considered']):
        raise ValueError('acceptance summary differs from considered-only counts')
    rate = acceptance.get('rate')
    if (counts['considered'] == 0 and rate is not None or counts['considered'] > 0 and (
            type(rate) not in {int, float} or not math.isfinite(rate)
            or abs(rate - counts['accepted'] / counts['considered']) > 1e-12)):
        raise ValueError('acceptance rate is not the finite accepted/considered ratio')
    latency = loop.get('phase_latency', {}); timing = loop.get('supervisor_timing', {})
    resident = _resident_mode(manifest)
    if (not isinstance(latency, dict) or latency.get('timing_scope') != (
            'RESIDENT_RUN_WALL_EXCLUDES_MODEL_LOAD_AND_CLOSE' if resident else 'CHILD_WALL_INCLUDES_LOAD_RUN_UNLOAD')
            or latency.get('inference_only_status', 'NOT_PROVIDED') != 'NOT_PROVIDED'
            or not isinstance(timing, dict) or timing.get('scope') != 'MONOTONIC_WALL_CLOCK_NOT_NPU_INFERENCE'):
        raise ValueError('phase latency must identify child wall-clock timing, not NPU inference')
    prefill = _has_prefill(manifest)
    prompt_calls = counts['prefill_native_calls'] if prefill else min(len(manifest['prompt_token_ids']), counts['target_native_calls'])
    expected = {'prompt_ingest': prompt_calls, 'decode': counts['target_native_calls'] - (0 if prefill else prompt_calls),
        'draft': counts['draft_native_calls'], **({'prefill': prompt_calls} if prefill else {})}
    for name, calls in expected.items():
        row = latency.get(name, {})
        if (not isinstance(row, dict) or type(row.get('calls')) is not int or row['calls'] != calls
                or type(row.get('total_ms')) is not int or row['total_ms'] < 0):
            raise ValueError('phase latency calls or milliseconds differ from actual child counts')
        mean = row.get('mean_ms')
        if (calls == 0 and mean is not None or calls > 0 and (
                type(mean) not in {int, float} or not math.isfinite(mean)
                or abs(mean - row['total_ms'] / calls) > max(1e-9, abs(mean) * 1e-12))):
            raise ValueError('phase latency mean differs from its finite total/call count')
    if latency['prompt_ingest'].get('mechanism') != (_PREFILL_RECIPE if prefill else 'DECODE1_SERIAL_PROMPT_INGESTION'):
        raise ValueError('serial prompt ingestion cannot be reported as one prefill model call')
    if resident:
        life = loop['resident_lifecycle']
        pairs = {'resident_target_load_ms': 'target_load_ms', 'resident_draft_load_ms': 'draft_load_ms',
            'resident_target_run_ms': 'target_run_ms', 'resident_draft_run_ms': 'draft_run_ms', 'resident_close_ms': 'close_ms'}
        if (any(type(timing.get(key)) is not int or timing[key] != life[value] for key, value in pairs.items())
                or latency['decode']['total_ms'] + (0 if prefill else latency['prompt_ingest']['total_ms']) != life['target_run_ms']
                or latency['draft']['total_ms'] != life['draft_run_ms']
                or any(timing.get(key) != 0 for key in ('target_child_process_wall_ms', 'draft_child_process_wall_ms',
                    'target_child_processes_timed', 'draft_child_processes_timed'))
                or type(timing.get('total_wall_ms')) is not int
                or timing['total_wall_ms'] < sum(life[key] for key in ('target_load_ms', 'draft_load_ms', 'target_run_ms', 'draft_run_ms', 'close_ms'))
                    + (latency['prefill']['total_ms'] if prefill else 0)):
            raise ValueError('resident Run/load/close wall timing differs from the single-process lifecycle')
        if prefill and (latency['prompt_ingest']['total_ms'] != latency['prefill']['total_ms']
                or latency['prompt_ingest'].get('latency_alias') != 'prefill'):
            raise ValueError('resident prefill child timing differs from its distinct phase')
        if _v1_mode(manifest):
            if (not prefill or any(type(timing.get(key)) is not int or timing[key] < 0 for key in
                    ('prefill_model_load_ms', 'prefill_process_ms', 'prefill_model_unload_ms', 'prefill_v1_total_ms'))
                    or timing['prefill_v1_total_ms'] != latency['prefill']['total_ms']
                    or timing['prefill_v1_total_ms'] < sum(timing[key] for key in
                        ('prefill_model_load_ms', 'prefill_process_ms', 'prefill_model_unload_ms'))
                    or any(timing.get(key) != 0 for key in ('prefill_child_processes_timed', 'prefill_child_process_wall_ms'))
                    or any(latency[key].get('native_api') != 'Process' for key in ('decode', 'draft'))):
                raise ValueError('V1 prefill Load/Process/Unload timing cannot be reported as child-process or NPU-only timing')
        elif prefill and (timing.get('prefill_child_processes_timed') != prompt_calls
                or timing.get('prefill_child_process_wall_ms') != latency['prefill']['total_ms']):
            raise ValueError('resident prefill child timing differs from its distinct phase')
        if (any(latency[key].get('timing_scope') != 'RESIDENT_RUN_INCLUDES_TENSOR_COPIES_READBACK_HASH_AND_BUFFER_CHECKS_NOT_NPU_ONLY' for key in ('decode', 'draft'))
                or prefill and any(latency[key].get('timing_scope') != ('PREFILL_V1_LOAD_PROCESS_UNLOAD_WALL'
                    if _v1_mode(manifest) else 'CHILD_WALL_INCLUDES_LOAD_RUN_UNLOAD') for key in ('prompt_ingest', 'prefill'))):
            raise ValueError('resident per-phase timing must distinguish Run copies/hash from prefill child load/run/unload')
        return
    if (any(type(timing.get(key)) is not int or timing[key] < 0 for key in
            ('startup_source_hash_ms', 'target_child_process_wall_ms', 'draft_child_process_wall_ms', 'total_wall_ms'))
            or any(type(timing.get(key)) is not int for key in ('target_child_processes_timed', 'draft_child_processes_timed'))
            or timing.get('target_child_processes_timed') != counts['target_native_calls']
            or timing.get('draft_child_processes_timed') != counts['draft_native_calls']
            or (0 if prefill else latency['prompt_ingest']['total_ms']) + latency['decode']['total_ms'] != timing['target_child_process_wall_ms']
            or latency['draft']['total_ms'] != timing['draft_child_process_wall_ms']
            or timing['total_wall_ms'] < timing['target_child_process_wall_ms'] + timing['draft_child_process_wall_ms']):
        raise ValueError('phase timing does not reconcile with supervisor monotonic totals')
    if prefill and (latency['prompt_ingest']['total_ms'] != latency['prefill']['total_ms']
            or latency['prompt_ingest'].get('latency_alias') != 'prefill'
            or type(timing.get('prefill_child_process_wall_ms')) is not int
            or timing.get('prefill_child_processes_timed') != prompt_calls
            or timing['prefill_child_process_wall_ms'] != latency['prefill']['total_ms']
            or timing['total_wall_ms'] < timing['target_child_process_wall_ms'] + timing['draft_child_process_wall_ms'] + timing['prefill_child_process_wall_ms']):
        raise ValueError('prefill phase timing differs from its distinct child process')


def _legacy_v1_lifecycle(value, *, name, processes, success, model_load_api='legacy-buffer'):
    """Validate V1 facts without importing V2 InitWeights or SDK-name claims."""
    flags = ('client_init_attempted', 'client_initialized', 'load_attempted', 'loaded', 'unload_called', 'closed')
    statuses = ('client_init_status', 'load_status', 'unload_status')
    if (model_load_api not in {'legacy-file', 'legacy-buffer'}
            or not isinstance(value, dict) or value.get('model_load_api') != model_load_api
            or value.get('lifecycle_policy') != 'legacy-v1-load-process-unload-v1'
            or value.get('model_name') != name
            or value.get('runtime_tensor_names') != 'NOT_QUERIED_V1_SAVED_NAMED_ABI_ONLY'
            or value.get('runtime_dtypes') != 'DECLARED_BOUND_ABI_NOT_QUERIED'
            or value.get('external_weight_resolution') != 'MODEL_PATH_PARENT_SIDECARS_UNVALIDATED_ON_DEVICE'
            or any(type(value.get(key)) is not bool for key in flags)
            or any(value.get(key) is not None and (type(value[key]) is not int or not -(1 << 63) <= value[key] < 1 << 63) for key in statuses)
            or any(type(value.get(key)) is not int or value[key] < 0 for key in ('process_attempts', 'successful_processes'))
            or value['successful_processes'] < processes or value['process_attempts'] < value['successful_processes']
            or value['client_init_attempted'] and value['closed'] is not True
            or not isinstance(value.get('failure_stage'), str) or not isinstance(value.get('cleanup_error'), str)
            or value['client_initialized'] != (value['client_init_attempted'] and value.get('client_init_status') == 0)
            or value['loaded'] != (value['load_attempted'] and value.get('load_status') == 0)
            or value['load_attempted'] and not value['client_initialized']
            or value['process_attempts'] and not value['loaded']
            or not value['client_init_attempted'] and value.get('client_init_status') is not None
            or not value['load_attempted'] and value.get('load_status') is not None
            or not value['unload_called'] and value.get('unload_status') is not None
            or value['loaded'] and not value['unload_called']):
        raise ValueError('V1 Init/Load/Process/Unload lifecycle is incomplete or inconsistent; locks retained')
    if success and (any(value[key] is not True for key in flags)
            or any(value.get(key) != 0 for key in statuses)
            or value['process_attempts'] != processes or value['successful_processes'] != processes
            or value['failure_stage'] or value['cleanup_error']):
        raise ValueError('successful V1 execution requires one Init/Load/Unload and all Process calls returned successfully')
    return value


def _resident_lifecycle(loop, manifest, *, success):
    life = loop.get('resident_lifecycle', {}); counts = loop['counts']
    integers = ('target_load_attempts', 'draft_load_attempts', 'target_loads', 'draft_loads', 'target_run_attempts', 'target_runs', 'draft_run_attempts', 'draft_runs',
        'target_close_calls', 'draft_close_calls', 'target_load_ms', 'draft_load_ms', 'target_run_ms', 'draft_run_ms', 'close_ms')
    v1 = _v1_mode(manifest)
    if v1:
        from .board_serial_dflash import v1_load_contract, TARGET_LIFECYCLE_POLICY
        session_api, model_apis = v1_load_contract(manifest)
        if ((manifest['target_lifecycle_policy'] == TARGET_LIFECYCLE_POLICY
                and loop.get('model_load_apis') != model_apis)
                or manifest['target_lifecycle_policy'] != TARGET_LIFECYCLE_POLICY
                and 'model_load_apis' in loop):
            raise ValueError('resident V1 per-model source APIs differ from the original prepared policy')
    else:
        session_api, model_apis = 'file-v2', {}
    if v1 and (any(type(counts.get(key)) is not int or counts[key] < 0 for key in
            ('target_native_calls', 'draft_native_calls', 'prefill_native_calls', 'accepted', 'considered', 'corrections', 'bonuses'))
            or counts['prefill_native_calls'] > 1
            or counts['considered'] != counts['accepted'] + counts['corrections']):
        raise ValueError('V1 completed-call counts require exact bounded integer observations')
    if (not isinstance(life, dict) or life.get('model_load_api') != session_api
            or _SHA.fullmatch(str(life.get('session_id'))) is None
            or type(loop.get('process_id')) is not int or not 0 < loop['process_id'] <= (1 << 31) - 1
            or type(life.get('process_id')) is not int or life['process_id'] != loop.get('process_id')
            or life['session_id'] != hashlib.sha256((str(loop.get('config_sha256')) + ':' + str(loop['process_id'])).encode('ascii')).hexdigest()
            or type(life.get('closed')) is not bool or life['closed'] is not True
            or type(life.get('poisoned')) is not bool
            or any(type(life.get(key)) is not int or life[key] < 0 for key in integers)
            or any(life[key] > 1 for key in ('target_load_attempts', 'draft_load_attempts', 'target_loads', 'draft_loads', 'target_close_calls', 'draft_close_calls'))
            or any(life[kind + '_close_calls'] != life[kind + '_loads'] for kind in ('target', 'draft'))
            or life['target_runs'] != counts['target_native_calls'] or life['draft_runs'] != counts['draft_native_calls']
            or any(life[kind + '_run_attempts'] < life[kind + '_runs'] for kind in ('target', 'draft'))):
        raise ValueError('resident lifecycle is incomplete, unclosed or inconsistent; locks retained')
    if success and (life['poisoned'] or life.get('stage') != 'CLOSED' or life.get('status') != 'CLOSED' or life.get('cleanup_error') not in {None, ''}
            or any(life[key] != 1 for key in ('target_load_attempts', 'draft_load_attempts', 'target_loads', 'draft_loads', 'target_close_calls', 'draft_close_calls'))
            or any(life[kind + '_run_attempts'] != life[kind + '_runs'] for kind in ('target', 'draft'))):
        raise ValueError('successful residency requires one load/close per model and no failed or retried Run')
    if v1:
        if (loop.get('target_lifecycle_policy') != manifest.get('target_lifecycle_policy')
                or 'target_lifecycle' in life
                or any(life.get(kind + '_model_name') != 'qwen35_resident_' + kind
                    or life.get(kind + '_model_name_scope') != 'V1_MODEL_DESCRIPTION_LABEL_NOT_RUNTIME_TENSOR_NAME'
                    for kind in ('target', 'draft'))):
            raise ValueError('resident V1 cannot adopt V2 external-weight lifecycle evidence')
        for kind in ('target', 'draft'):
            observed = _legacy_v1_lifecycle(life.get(kind + '_v1_lifecycle'),
                name='qwen35_resident_' + kind, processes=counts[kind + '_native_calls'],
                success=success, model_load_api=model_apis[kind])
            if (life[kind + '_load_attempts'] < int(observed['client_init_attempted'])
                    or life[kind + '_loads'] > int(observed['loaded'])
                    or success and life[kind + '_loads'] != int(observed['loaded'])
                    or life[kind + '_run_attempts'] < observed['process_attempts']
                    or success and life[kind + '_run_attempts'] != observed['process_attempts']):
                raise ValueError('resident V1 model counters differ from actual Load/Process lifecycle')
        _legacy_v1_lifecycle(loop.get('prefill_lifecycle'), name='qwen35_resident_prefill',
            processes=counts['prefill_native_calls'], success=success, model_load_api=model_apis['prefill'])
    elif any(key in life for key in ('target_v1_lifecycle', 'draft_v1_lifecycle')) or 'prefill_lifecycle' in loop:
        raise ValueError('historical resident V2 cannot inherit V1 lifecycle evidence')
    if success and not v1:
        target = life.get('target_lifecycle', {})
        if (any(life.get(kind + '_model_name') != 'qwen35_resident_' + kind
                    or life.get(kind + '_model_name_scope') != 'SDK_SETNAME_GETNAME_VALIDATED_BEFORE_INIT' for kind in ('target', 'draft'))
                or target.get('weight_init_policy') != ('INIT_BEFORE_WEIGHTS_NO_FALLBACK' if _new_target_policy(manifest)
                    else 'EXPERIMENTAL_INIT_WEIGHTS_BEFORE_INIT_NO_FALLBACK')
                or (_new_target_policy(manifest) and target.get('weight_cleanup_policy') != 'DEINIT_WEIGHTS_BEFORE_DEINIT')
                or target.get('ordering_validation') != 'NOT_VALIDATED_ON_DEVICE'
                or any(target.get(key) is not True for key in ('weights_init_attempted', 'weights_initialized',
                    'manager_init_attempted', 'manager_initialized', 'manager_deinit_called', 'weights_deinit_called', 'closed'))
                or target.get('weights_deinit_status') != 0
                or target.get('run_attempts') != counts['target_native_calls'] or target.get('successful_runs') != counts['target_native_calls']):
            raise ValueError('resident model names or external-weight Init/DeInit lifecycle is unverified')
    target = life.get('target_lifecycle', {})
    if v1:
        pass
    elif _new_target_policy(manifest):
        if (loop.get('target_lifecycle_policy') != manifest['target_lifecycle_policy']
                or target.get('weight_init_policy') != 'INIT_BEFORE_WEIGHTS_NO_FALLBACK'
                or target.get('weight_cleanup_policy') != 'DEINIT_WEIGHTS_BEFORE_DEINIT'):
            raise ValueError('resident lifecycle differs from the explicitly prepared Init/cleanup policy')
    elif (loop.get('target_lifecycle_policy') is not None
            or target.get('weight_init_policy') != 'EXPERIMENTAL_INIT_WEIGHTS_BEFORE_INIT_NO_FALLBACK'
            or target.get('weight_cleanup_policy') is not None):
        raise ValueError('historical resident preparation cannot adopt a new target lifecycle policy')
    memory = loop.get('resident_memory')
    if memory is not None:
        if (not isinstance(memory, dict) or memory.get('scope') != 'PROCESS_RSS_NOT_NPU_ALLOCATED_MEMORY'
                or not isinstance(memory.get('snapshots'), list) or len(memory['snapshots']) > 8):
            raise ValueError('resident process-memory telemetry is malformed or unbounded')
        for snapshot in memory['snapshots']:
            if (not isinstance(snapshot, dict) or snapshot.get('phase') not in {'before-load', 'after-both-load', 'after-close', 'after-failure-close'}
                    or snapshot.get('status') not in {'OBSERVED', 'UNAVAILABLE'}
                    or any(type(snapshot[key]) is not int or not 0 <= snapshot[key] <= 1 << 50
                        for key in ('vm_rss_kib', 'vm_hwm_kib', 'mem_available_kib') if key in snapshot)):
                raise ValueError('resident memory telemetry requires bounded nonnegative process RSS values')
    return life


def _resident_call(receipt, event, loop, calls, kind, position):
    expected = {'schema_version': 2, 'execution_mode': loop['execution_mode'],
        'session_id': loop['resident_lifecycle']['session_id'], 'process_id': loop['process_id']}
    index = receipt.get('call_index')
    if (any(receipt.get(key) != value or event.get(key) != value for key, value in expected.items() if key != 'schema_version')
            or receipt.get('schema_version') != 2 or type(index) is not int or index < 0
            or event.get('call_index') != index or index in calls or any(key in receipt or key in event for key in ('pid', 'exitcode'))):
        raise ValueError('resident call does not bind the exact session/process/unique call index')
    calls[index] = (position, -1 if kind == 'prefill' else 0 if kind == 'draft' else 1)
    return loop['process_id']


def _resident_io(diagnostic, inputs, outputs):
    if (not isinstance(diagnostic, dict) or diagnostic.get('enabled') is not True
            or diagnostic.get('process_called') is not True or diagnostic.get('output_fill_byte') != 0
            or diagnostic.get('input_readback_point') != 'AFTER_ALL_INPUT_COPIES_AND_OUTPUT_FILL_BEFORE_PROCESS'
            or len(diagnostic.get('inputs', [])) != len(inputs) or len(diagnostic.get('outputs', [])) != len(outputs)):
        raise ValueError('resident actual input/output buffer diagnostics are incomplete')
    for index, (row, actual) in enumerate(zip(inputs, diagnostic['inputs'])):
        if (actual.get('index') != index or actual.get('logical_bytes') != row['bytes']
                or type(actual.get('allocated_bytes')) is not int or actual['allocated_bytes'] < row['bytes']
                or actual.get('readback_matches_source') is not True or actual.get('readback_sha256') != row['sha256']):
            raise ValueError('resident actual input buffer differs from the declared file bytes')
    for index, (row, actual) in enumerate(zip(outputs, diagnostic['outputs'])):
        if (actual.get('index') != index or actual.get('logical_bytes') != row['bytes']
                or actual.get('checked_after_process') is not True or actual.get('object_same') is not True
                or actual.get('buffer_same') is not True):
            raise ValueError('resident output buffer identity differs through native Run')


def _v1_observed_abi(receipt, declaration):
    """Check real V1 dimensions/extents; names and dtypes remain saved provenance."""
    if (receipt.get('runtime_tensor_names') != 'NOT_QUERIED_V1_SAVED_NAMED_ABI_ONLY'
            or receipt.get('runtime_dtypes') != 'DECLARED_BOUND_ABI_NOT_QUERIED'):
        raise ValueError('V1 cannot claim observed SDK tensor names or runtime dtypes')
    observed = receipt.get('runtime_abi_observed_on_load', {})
    for direction in ('inputs', 'outputs'):
        actual, expected = observed.get(direction), declaration[direction]
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError('V1 actual loaded interface dimensions are incomplete')
        for index, (value, desc) in enumerate(zip(actual, expected)):
            shape, wanted = value.get('shape'), desc['shape']
            if (type(value.get('index')) is not int or value['index'] != index
                    or not isinstance(shape, list) or not 1 <= len(shape) <= 8
                    or any(type(part) is not int or not 0 < part < 1 << 31 for part in shape)
                    or type(value.get('bytes')) is not int or value['bytes'] != desc['bytes']
                    or type(value.get('raw_format')) is not int or value['raw_format'] < 0
                    or 'raw_format' in desc and value['raw_format'] != desc['raw_format']
                    or not (shape == wanted or len(shape) == 4 and (
                        1 <= len(wanted) <= 3 and shape == wanted + [1] * (4 - len(wanted))
                        or len(wanted) == 3 and shape == [1] + wanted))):
                raise ValueError('V1 actual slot shape/extent differs from the bound descriptor provenance')


def _resident_progress(loop, progress, *, success):
    if (progress.get('status') != loop['status'] or progress.get('supervisor_pid') != loop['process_id']
            or progress.get('resident_lifecycle') != loop['resident_lifecycle']
            or progress.get('resident_call_status') != ('RETURNED' if success else 'FAILED_CLOSED')):
        raise ValueError('resident completion progress is unresolved or differs from its current lifecycle')


def _v1_no_children(loop, progress):
    ledger = loop.get('child_lifecycle', {})
    expected = dict(launched_child_count=0, completed_child_count=0, active_child_count=0,
        last_child_pid=None, last_child_exitcode=None, child_status='NONE')
    if (not isinstance(ledger, dict) or ledger.get('children') != []
            or any(ledger.get(key) != value or progress.get(key) != value
                or type(ledger.get(key)) is not type(value) or type(progress.get(key)) is not type(value)
                for key, value in expected.items())):
        raise ValueError('same-process V1 cannot claim child PID/exit records or unresolved child launches')


def _json(path, maximum=64 << 20):
    raw, identity = _read_small(path, maximum)
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError('board serial metadata must be an object: ' + str(path))
    return value, identity


def _full(path):
    path, stamp = _safe(path)
    full = _identity(path.stat())
    if [full[k] for k in ('device', 'inode', 'bytes', 'mtime_ns', 'ctime_ns')] != stamp:
        raise ValueError('board serial file changed while snapshotting: ' + str(path))
    return full


def _checkpoint(out, state, phase, **updates):
    """Durable before-operation journal, including when Python is interrupted."""
    state.update(updates, phase=phase)
    temporary = out / ('checkpoint-' + uuid.uuid4().hex + '.pending')
    raw = json.dumps(state, sort_keys=True, indent=2, allow_nan=False).encode() + b'\n'
    with temporary.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, out / 'checkpoint.json')
    descriptor = os.open(out, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remote(value):
    _text(str(value), 'remote path')
    path = PurePosixPath(value)
    if (not path.is_absolute() or str(path) != str(value) or str(path) == '/'
            or any(part in {'', '.', '..'} for part in str(value).split('/')[1:])
            or re.fullmatch(r'/[A-Za-z0-9_./-]+', str(path)) is None):
        raise ValueError('board serial remote path must be a normalized absolute safe path')
    return path


def _regular_guard(path, *, size=None):
    quoted = shlex.quote(str(path))
    guard = 'test -f ' + quoted + ' && test ! -L ' + quoted
    if size is not None:
        guard += ' && test "$(wc -c < ' + quoted + ')" -eq ' + str(size)
    return guard


def _owner_guard(directory, filename, owner):
    path = directory / filename
    return ('test -d ' + shlex.quote(str(directory)) + ' && test ! -L ' + shlex.quote(str(directory))
        + ' && ' + _regular_guard(path, size=len(owner) + 1)
        + ' && read -r serial_owner < ' + shlex.quote(str(path))
        + ' && test "$serial_owner" = ' + shlex.quote(owner))


def _mutate(hdc, out, state, command, label, *, timeout=None, **updates):
    _checkpoint(out, state, label + '-requested', uncertain=True, **updates)
    result = hdc.shell(command, label, timeout=min(60, hdc.timeout) if timeout is None else timeout)
    _checkpoint(out, state, label + '-completed', uncertain=False)
    return result


def _batches(commands, budget=12000):
    batch, size = [], 0
    for command in commands:
        if len(command.encode()) > budget:
            raise ValueError('one remote inventory operation exceeds the bounded shell command budget')
        if batch and size + len(command.encode()) + 4 > budget:
            yield batch; batch, size = [], 0
        batch.append(command); size += len(command.encode()) + 4
    if batch:
        yield batch


def _environment():
    serial, server = _text(os.environ.get('SN'), 'SN'), _text(os.environ.get('BIP'), 'BIP')
    if (serial.startswith('-') or server.startswith('-') or re.fullmatch(r'[^\s:]+:[0-9]+', server) is None
            or not 0 < int(server.rsplit(':', 1)[1]) < 65536):
        raise ValueError('SN and the selected HDC server BIP host:port are required')
    return _text(os.environ.get('HDC', 'hdc'), 'HDC'), serial, server


def _owned_directory(hdc, out, state, directory, filename, owner, label):
    parent = directory.parent
    command = ('test -d ' + shlex.quote(str(parent)) + ' && test ! -L ' + shlex.quote(str(parent))
        + ' && if mkdir ' + shlex.quote(str(directory)) + ' 2>/dev/null; then printf \'%s\\n\' '
        + shlex.quote(owner) + ' > ' + shlex.quote(str(directory / filename))
        + '; else ' + _owner_guard(directory, filename, owner) + '; fi')
    _mutate(hdc, out, state, command, label)


def _directories(hdc, out, state, root, paths, label):
    children = set()
    for path in paths:
        path = _remote(path)
        if root not in path.parents and path != root:
            raise ValueError('remote directory escapes its owned cache')
        while path != root:
            children.add(path); path = path.parent
    command = []
    for path in sorted(children, key=lambda item: (len(item.parts), str(item))):
        quoted = shlex.quote(str(path))
        command.append('test ! -L ' + quoted + ' && if test -e ' + quoted
            + '; then test -d ' + quoted + '; else mkdir ' + quoted + '; fi')
    if command:
        _mutate(hdc, out, state, ' && '.join(command), label)


def _cache_inventory(hdc, rows):
    """One actual remote SHA pass over the whole request's static payloads."""
    commands = []
    for row in rows:
        path = shlex.quote(row['remote'])
        commands.append('test ! -L ' + path + ' || exit 1; if test -e ' + path + '; then '
            + 'test -f ' + path + ' || exit 1; serial_hash=$(sha256sum ' + path + ') || exit 1; '
            + 'serial_hash=${serial_hash%% *}; test "${#serial_hash}" -eq 64 || exit 1; '
            + 'case "$serial_hash" in *[!0-9a-f]*) exit 1;; esac; '
            + 'test "$serial_hash" = ' + shlex.quote(row['sha256']) + ' || exit 1; '
            + 'test "$(wc -c < ' + path + ')" -eq ' + str(row['bytes']) + ' || exit 1; '
            + 'printf \'OK %s\\n\' ' + path + '; else printf \'SEND %s\\n\' ' + path + '; fi')
    responses = []
    for batch in _batches(commands):
        _, text = hdc.shell('\n'.join(batch), 'inspect-whole-loop-cache', timeout=hdc.timeout)
        responses.append(text)
    text = '\n'.join(responses)
    decisions = {}
    for line in text.splitlines():
        match = re.fullmatch(r'(OK|SEND) (/[A-Za-z0-9_./-]+)', line)
        if not match or match[2] in decisions:
            raise ValueError('remote static-payload inventory is malformed or duplicated')
        decisions[match[2]] = match[1]
    if set(decisions) != {row['remote'] for row in rows}:
        raise ValueError('remote static-payload inventory is incomplete')
    return decisions


def _local_upload(row, manifest):
    if row['kind'] in {'target', 'prefill'}:
        from .original_target_board import _recorded_board_payload
        base = manifest['prefill']['base_plan'] if row['kind'] == 'prefill' else manifest['target_binding']['base_plan']
        compiled = manifest['prefill']['compiled'] if row['kind'] == 'prefill' else manifest['target_compiled']
        return _recorded_board_payload(Path(base['compilation']), row['record'], compiled, smoke_proof=base)
    source = Path(row['local'])
    before = _full(source)
    if row.get('identity') is not None and before != row['identity']:
        raise ValueError('uncached source differs from the immutable preparation identity: ' + str(source))
    if before['bytes'] != row['bytes'] or sha256_file(source) != row['sha256'] or _full(source) != before:
        raise ValueError('uncached source changed or differs from its SHA256: ' + str(source))
    return source


def _upload_cached(hdc, out, state, rows, decisions, manifest):
    from .target_model_io_probe import _space
    _, disk = hdc.shell('df -Pk ' + shlex.quote(state['board_attempt']), 'whole-loop-board-disk')
    missing = sum(row['bytes'] for row in rows if decisions[row['remote']] == 'SEND')
    free = _space(disk)
    # Final states plus intermediate per-step files coexist until successful retrieval.
    steps = manifest['max_new_tokens'] + (0 if _has_prefill(manifest) else len(manifest['prompt_token_ids']))
    output_bytes = sum(row['bytes'] for row in manifest['target_binding']['base_plan']['outputs'])
    input_bytes = sum(row['bytes'] for row in manifest['target_binding']['base_plan'].get('inputs', []))
    draft_bytes = manifest['max_new_tokens'] * ((2 << 20) + 61440 + (1 << 20))
    feature_bytes = steps * 32768
    reserve = (128 << 20) + output_bytes * (steps + 1) + input_bytes * steps + draft_bytes + feature_bytes
    if _has_prefill(manifest):
        reserve += sum(row['bytes'] for direction in ('inputs', 'outputs') for row in manifest['prefill']['base_plan'][direction])
        reserve += len(manifest['prompt_token_ids']) * 32768
    state['board_disk_budget'] = {'free_bytes': free, 'missing_upload_bytes': missing,
        'retained_target_input_bytes': input_bytes * steps, 'retained_target_output_bytes': output_bytes * steps,
        'draft_input_output_metadata_reserve_bytes': draft_bytes, 'final_feature_bytes': feature_bytes,
        'retained_output_reserve_bytes': reserve, 'required_bytes': missing + reserve}
    print(f'board-serial: whole-request cache verified; missing upload={missing} bytes; retained-data reserve={reserve} bytes', flush=True)
    if free < missing + reserve:
        raise ValueError('board disk budget is insufficient for static payloads and retained loop outputs')
    for row in rows:
        if decisions[row['remote']] == 'OK':
            continue
        local = _local_upload(row, manifest)
        remote = PurePosixPath(row['remote']); partial = remote.with_name(remote.name + '.serial-partial')
        _mutate(hdc, out, state, 'test ! -e ' + shlex.quote(str(partial)) + ' && test ! -L '
            + shlex.quote(str(partial)), 'check-static-partial-absent')
        _checkpoint(out, state, 'send-static-payload-requested', uncertain=True,
            pending_payload={'remote': str(remote), 'bytes': row['bytes'], 'sha256': row['sha256']})
        hdc.send(local, partial)
        _mutate(hdc, out, state, _regular_guard(partial, size=row['bytes'])
            + ' && serial_hash=$(sha256sum ' + shlex.quote(str(partial)) + ') && serial_hash=${serial_hash%% *}'
            + ' && test "$serial_hash" = ' + shlex.quote(row['sha256'])
            + ' && test ! -e ' + shlex.quote(str(remote)) + ' && test ! -L ' + shlex.quote(str(remote))
            + ' && mv ' + shlex.quote(str(partial)) + ' ' + shlex.quote(str(remote)), 'publish-static-payload',
            timeout=hdc.timeout)
    state['cached_payloads'] = [{key: row[key] for key in ('remote', 'bytes', 'sha256', 'kind')}
        | {'decision': decisions[row['remote']]} for row in rows]


def _build_payloads(out, timeout, *, resident=False):
    runner = _build_runner(out, timeout)
    supervisor = runner.parent / ('qwen35_board_resident_v1' if resident == 'resident-v1'
        else 'qwen35_board_resident' if resident else 'qwen35_board_serial')
    _safe(supervisor); dynamic_info(supervisor)
    uploads = [*([_file_record(runner, 'draft-runner')] if resident != 'resident-v1' else []),
        _file_record(supervisor, 'board-serial')]
    paths = list(BOARD_LIBRARY_SEARCH_PATHS)
    private = os.environ.get('BOARD_HIAI_LIB_DIR')
    if private:
        _remote(private); paths.insert(0, private)
        return uploads, paths, {'mode': 'existing-board-directory', 'path': private}
    libraries = {}
    discoveries = []
    # discover_libraries binds result.binary_sha256 to the original HIAI
    # executable. The sibling supervisor uses the same selected SDK C++ runtime
    # and is separately ELF/SHA bound; it is not that recorded HIAI executable.
    for binary in (runner,):
        found = discover_libraries(binary, out / 'native')
        discoveries.append(found['metadata'])
        for path in found['libraries']:
            row = _file_record(path.resolve(), 'lib/' + str(_relative(path.name)))
            previous = libraries.get(path.name)
            if previous and (previous['sha256'], previous['bytes']) != (row['sha256'], row['bytes']):
                raise ValueError('runner dependency basenames have conflicting library identities')
            libraries[path.name] = row
    dependencies = [{'name': PurePosixPath(row['remote']).name,
        'kind': 'runner' if row in uploads else 'library', 'sha256': row['sha256'], **dynamic_info(row['local'])}
        for row in [*uploads, *libraries.values()]]
    blockers = candidate_blockers(dependencies, require_pair=False)
    if blockers:
        raise ValueError('cannot select board libraries: ' + '; '.join(blockers))
    uploads.extend(row for name, row in libraries.items() if name not in DDK_HCL_EXCLUDED)
    return uploads, paths, {'mode': 'candidate-without-ddk-hcl', 'discoveries': discoveries,
        'dependencies': dependencies, 'backend_selection': 'UNVERIFIED'}


def _static_rows(manifest, target_cache, draft_cache, table_cache, prefill_cache=None):
    rows = []
    for row in manifest['target_payloads']:
        relative = _relative(row['remote'])
        rows.append({'local': row['local'], 'remote': str(target_cache / 'payload' / relative),
            'bytes': row['bytes'], 'sha256': row['sha256'], 'record': row['record'], 'kind': 'target'})
    if _has_prefill(manifest):
        if prefill_cache is None:
            raise ValueError('prefill payloads require their separately owned compiled cache')
        for row in manifest['prefill']['payloads']:
            rows.append({'local': row['local'], 'remote': str(prefill_cache / 'payload' / _relative(row['remote'])),
                'bytes': row['bytes'], 'sha256': row['sha256'], 'record': row['record'], 'kind': 'prefill'})
    model = manifest['draft']['model_source']
    rows.append({'local': model['path'], 'remote': str(draft_cache / 'model/model.omc'),
        'bytes': model['bytes'], 'sha256': model['sha256'], 'identity': model.get('identity'), 'kind': 'draft'})
    for name in ('weight', 'scale'):
        source = manifest['tables'][name]
        rows.append({'local': source['path'], 'remote': str(table_cache / (name + '.bin')),
            'bytes': source['bytes'], 'sha256': source['sha256'], 'identity': source['identity'], 'kind': 'embedding-' + name})
    if (not rows or len(rows) > 16384 or len({row['remote'] for row in rows}) != len(rows)
            or any(type(row['bytes']) is not int or row['bytes'] <= 0 or _SHA.fullmatch(row['sha256']) is None for row in rows)):
        raise ValueError('whole-loop static payload inventory is invalid or ambiguous')
    return rows


def _launcher(state):
    board = PurePosixPath(state['board_attempt'])
    argv = [str(board / 'board-serial'), '--config', str(board / 'config.tsv'), '--output', str(board / 'results')]
    return ('#!/system/bin/sh\nset -Cu\ncd ' + shlex.quote(str(board)) + ' || exit 125\n'
        + 'exec 3> launch.guard\n'  # Retained, attempt-local launch evidence.
        + 'printf \'%s\\n\' "$$" > tool.pid\n'
        + 'LD_LIBRARY_PATH=' + shlex.quote(state['supervisor_library_path']) + ' ' + shlex.join(argv)
        + ' > supervisor.stdout 2> supervisor.stderr\n'
        + 'serial_rc=$?\nprintf \'%s\\n\' "$serial_rc" > tool.exitcode\nexit "$serial_rc"\n')


def _completion(hdc, state):
    board = PurePosixPath(state['board_attempt'])
    # A PID can be reused: any live /proc entry is conservative uncertainty.
    command = ('test -d ' + shlex.quote(str(board)) + ' && test ! -L ' + shlex.quote(str(board))
        + ' && if test -e ' + shlex.quote(str(board / 'tool.exitcode')) + ' || test -L '
        + shlex.quote(str(board / 'tool.exitcode')) + '; then '
        + _regular_guard(board / 'tool.exitcode') + ' && test "$(wc -c < '
        + shlex.quote(str(board / 'tool.exitcode')) + ')" -le 4 && '
        + _regular_guard(board / 'tool.pid') + ' && test "$(wc -c < '
        + shlex.quote(str(board / 'tool.pid')) + ')" -le 11'
        + ' && printf \'EXIT=\' && cat ' + shlex.quote(str(board / 'tool.exitcode'))
        + ' && printf \'PID=\' && cat ' + shlex.quote(str(board / 'tool.pid'))
        + ' && read -r serial_pid < ' + shlex.quote(str(board / 'tool.pid'))
        + ' && case "$serial_pid" in \'\'|*[!0-9]*) exit 1;; esac'
        + ' && if test -d "/proc/$serial_pid"; then printf \'PROCESS_PRESENT\\n\'; else printf \'PROCESS_ABSENT\\n\'; fi'
        + '; else printf \'PENDING\\n\'; fi')
    _, text = hdc.shell(command, 'poll-whole-loop-completion', timeout=min(30, hdc.timeout))
    if text == 'PENDING':
        return None
    match = re.fullmatch(r'EXIT=(0|[1-9][0-9]{0,2})\nPID=([1-9][0-9]{0,9})\nPROCESS_(ABSENT|PRESENT)', text.replace('\r', ''))
    if not match or int(match[1]) > 255 or int(match[2]) > (1 << 31) - 1:
        raise ValueError('supervisor completion is active, malformed, or uncertain: ' + repr(text))
    if match[3] == 'PRESENT':
        return None
    return {'remote_exitcode': int(match[1]), 'launcher_process_id': int(match[2]),
        'completion_proof': 'RETAINED_EXITCODE_AND_PID_ABSENT_IN_ONE_FRAMED_SHELL'}


def _lock_checks(hdc, state, *, release=False):
    target, draft = PurePosixPath(state['target_cache']), PurePosixPath(state['draft_cache'])
    board, token = PurePosixPath(state['board_attempt']), state['board_lock_token']
    target_lock, draft_lock = target / '.execution-lock', draft / '.execution-lock'
    prefill = PurePosixPath(state['prefill_cache']) if state.get('prefill_cache') else None
    locks = [_GLOBAL, target_lock, draft_lock, *([prefill / '.execution-lock'] if prefill else [])]
    both_absent = ' && '.join('test ! -e ' + shlex.quote(str(path)) + ' && test ! -L ' + shlex.quote(str(path)) for path in locks)
    owned = (_owner_guard(_GLOBAL, 'owner', token) + ' && '
        + _owner_guard(target, '.compilation-sha256', state['compilation_report_sha256']) + ' && '
        + _owner_guard(draft, '.model-sha256', state['draft_model_sha256']) + ' && '
        + _owner_guard(draft_lock, 'owner', token)
        + ' && ' + _regular_guard(board / 'target-lock.owner', size=33)
        + ' && read -r serial_target_token < ' + shlex.quote(str(board / 'target-lock.owner'))
        + ' && test "$serial_target_token" = ' + shlex.quote(token)
        + ' && test -d ' + shlex.quote(str(target_lock)) + ' && test ! -L ' + shlex.quote(str(target_lock))
        + ' && test "$(find ' + shlex.quote(str(target_lock)) + ' | wc -l)" -eq 1'
        + ' && test "$(find ' + shlex.quote(str(_GLOBAL)) + ' | wc -l)" -eq 2'
        + ' && test "$(find ' + shlex.quote(str(draft_lock)) + ' | wc -l)" -eq 2')
    if prefill:
        plock = prefill / '.execution-lock'
        owned += (' && ' + _owner_guard(prefill, '.compilation-sha256', state['prefill_compilation_report_sha256'])
            + ' && ' + _regular_guard(board / 'prefill-lock.owner', size=33)
            + ' && read -r serial_prefill_token < ' + shlex.quote(str(board / 'prefill-lock.owner'))
            + ' && test "$serial_prefill_token" = ' + shlex.quote(token)
            + ' && test -d ' + shlex.quote(str(plock)) + ' && test ! -L ' + shlex.quote(str(plock))
            + ' && test "$(find ' + shlex.quote(str(plock)) + ' | wc -l)" -eq 1')
    if release:
        completion = _completion(hdc, state)
        if completion is None or completion != state.get('completion'):
            raise ValueError('supervisor completion changed before exact-owner lock release')
        # Repeat completion inside the actual release shell, not only in a previous query.
        owned += (' && ' + _regular_guard(board / 'tool.exitcode', size=len(str(completion['remote_exitcode'])) + 1)
            + ' && test "$(cat ' + shlex.quote(str(board / 'tool.exitcode')) + ')" = '
            + shlex.quote(str(completion['remote_exitcode'])) + ' && '
            + _regular_guard(board / 'tool.pid', size=len(str(completion['launcher_process_id'])) + 1)
            + ' && test "$(cat ' + shlex.quote(str(board / 'tool.pid')) + ')" = '
            + shlex.quote(str(completion['launcher_process_id']))
            + ' && test ! -d /proc/' + str(completion['launcher_process_id']))
        for pid in state.get('completed_native_pids', []):
            if type(pid) is not int or not 0 < pid <= (1 << 31) - 1:
                raise ValueError('invalid completed native PID before lock release')
            owned += ' && test ! -d /proc/' + str(pid)
        action = (('rmdir ' + shlex.quote(str(prefill / '.execution-lock')) + ' && ' if prefill else '')
            + 'rmdir ' + shlex.quote(str(target_lock)) + ' && rm ' + shlex.quote(str(draft_lock / 'owner'))
            + ' && rmdir ' + shlex.quote(str(draft_lock)) + ' && rm ' + shlex.quote(str(_GLOBAL / 'owner'))
            + ' && rmdir ' + shlex.quote(str(_GLOBAL)) + ' && printf \'RELEASED\\n\'')
    else:
        action = 'printf \'OWNED\\n\''
    _, text = hdc.shell('if ' + both_absent + '; then printf \'ABSENT\\n\'; else '
        + owned + ' && ' + action + '; fi', 'release-whole-loop-locks' if release else 'inspect-whole-loop-locks',
        timeout=min(60, hdc.timeout))
    if text not in ({'ABSENT', 'RELEASED'} if release else {'ABSENT', 'OWNED'}):
        raise ValueError('whole-loop lock ownership is foreign, partial, or uncertain')
    return text


def _inventory(raw):
    lines = raw.decode('utf-8').splitlines()
    if not lines or lines[0] != 'QWEN35_BOARD_SERIAL_RETRIEVAL\t1':
        raise ValueError('supervisor retrieval inventory has no supported header')
    rows, seen, total = [], set(), 0
    for line in lines[1:]:
        fields = line.split('\t')
        if len(fields) != 4 or fields[0] != 'file' or _SHA.fullmatch(fields[2]) is None or not fields[3].isdigit():
            raise ValueError('supervisor retrieval inventory row is malformed')
        relative = str(_relative(fields[1])); size = int(fields[3])
        if relative in seen or size > _MAX_FILE or len(rows) >= 16384:
            raise ValueError('supervisor retrieval inventory is duplicate or exceeds bounded file budgets')
        seen.add(relative); total += size
        if total > _MAX_TOTAL:
            raise ValueError('supervisor retrieval inventory exceeds its total byte budget')
        rows.append({'path': relative, 'sha256': fields[2], 'bytes': size})
    if not {'loop.json', 'progress.json'} <= seen:
        raise ValueError('supervisor retrieval inventory omits loop/progress reports')
    return rows


def _table_rows(table, tokens, width):
    before = _full(table['path'])
    if before != table['identity']:
        raise ValueError('original embedding table changed after preparation')
    descriptor = os.open(table['path'], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        if _identity(os.fstat(descriptor)) != before:
            raise ValueError('original embedding table changed before row read')
        rows = [os.pread(descriptor, width, token * width) for token in tokens]
        if any(len(row) != width for row in rows) or _identity(os.fstat(descriptor)) != before or _full(table['path']) != before:
            raise ValueError('original embedding table changed during row read')
    finally:
        os.close(descriptor)
    return b''.join(rows)


def _prefill_controls(manifest):
    tokens = manifest['prompt_token_ids']; length = len(tokens)
    if not 1 <= length <= 63:
        raise ValueError('padded prefill needs1..63 original prompt tokens')
    positions = struct.pack('<64i', *range(64)); negative = struct.pack('<f', float('-inf'))
    mask = b''.join(bytes(4 * (query + 1 if query < length else 1))
        + negative * (2048 - (query + 1 if query < length else 1)) for query in range(64))
    return {'arg1_1': _table_rows(manifest['tables']['weight'], tokens, 2048) + bytes((64 - length) * 2048),
        'arg0_1': _table_rows(manifest['tables']['scale'], tokens, 4) + bytes((64 - length) * 4),
        'arg107_1': positions, 'arg2_1': positions, 'arg1238_1': struct.pack('<i', length - 1),
        'arg114_1': struct.pack('<q', length), 'arg18_1': struct.pack('<h', length),
        'arg14_1': struct.pack('<128i', length, *([0] * 127)), 'arg113_1': mask}


def _validate_prefill_seed(manifest, root, loop, record, load_record, logs, zeros, all_features, evidence,
                           resident_calls=None):
    """Bind the actual prefill's original inputs,80owners and valid feature rows."""
    declaration = manifest['prefill']; base = declaration['base_plan']; length = len(manifest['prompt_token_ids'])
    events = loop.get('prefill_steps', [])
    if len(events) != 1:
        raise ValueError('full prefill audit requires exactly one actual prefill receipt')
    event = events[0]; receipt = load_record(event['receipt']); pid = event.get('pid'); v1 = _v1_mode(manifest)
    if v1:
        pid = _resident_call(receipt, event, loop, resident_calls, 'prefill', -1)
    expected = {'schema_version': 2 if v1 else 1, 'artifact_kind': 'qwen35-board-serial-prefill-step',
        **({} if v1 else {'pid': pid, 'exitcode': 0}), 'status': 'EXECUTED_OUTPUTS_BOUND', 'evidence_kind': evidence,
        'model_sha256': declaration['model']['sha256'], 'prompt_token_ids': manifest['prompt_token_ids'],
        'logical_prompt_length': length, 'physical_sequence_length': 64, 'selected_output_pos': length - 1,
        'control_semantics': 'EXPERIMENTAL_PADDED_PREFILL_VALID_LENGTH_NOT_NUMERICALLY_VALIDATED'}
    if (type(pid) is not int or not 0 < pid <= (1 << 31) - 1
            or any(receipt.get(key) != value for key, value in expected.items())
            or event.get('prompt_length') != length or event.get('next_token_id') != receipt.get('next_token_id')):
        raise ValueError('prefill invocation does not bind this original model, prompt and experimental controls')
    if not v1:
        logs(event['receipt'], pid)
    inputs, outputs = receipt.get('inputs', []), receipt.get('outputs', [])
    if len(inputs) != 89 or len(outputs) != 82 or len(base['inputs']) != 89 or len(base['outputs']) != 82:
        raise ValueError('prefill receipt omits its complete89/82 native interface')
    if v1:
        _resident_io(receipt.get('io_diagnostic'), inputs, outputs)
        _v1_observed_abi(receipt, manifest['prefill_abi'])
        if receipt.get('prefill_lifecycle') != loop['prefill_lifecycle']:
            raise ValueError('V1 prefill receipt differs from its actual Init/Load/Process/Unload lifecycle')
    controls = _prefill_controls(manifest)
    for index, (value, desc) in enumerate(zip(inputs, base['inputs'])):
        record(value)
        if (value.get('index') != index or value.get('name') != desc['name']
                or value['bytes'] != desc['bytes'] or value.get('tool_type') != desc['tool_input_type']):
            raise ValueError('prefill input differs from its named typed compiled interface')
        if value['name'] in controls and value['sha256'] != hashlib.sha256(controls[value['name']]).hexdigest():
            raise ValueError('prefill original-table embedding, padding or declared query controls differ')
    for index, (value, desc) in enumerate(zip(outputs, base['outputs'])):
        record(value, actual=True)
        if (value.get('index') != index or value.get('runtime_name') != desc['runtime_tensor_name']
                or value['bytes'] != desc['bytes'] or value.get('tool_type') != desc['tool_output_type']):
            raise ValueError('prefill output differs from its named typed compiled interface')
    mapped = receipt.get('state_mapping', [])
    by_decode = {row.get('decode_source_refdata'): row for row in mapped}
    if len(mapped) != 80 or len(by_decode) != 80 or len(declaration['state_mapping']) != 80:
        raise ValueError('prefill-to-decode owner map is incomplete or duplicated')
    owners = declaration['state_mapping']; decoded = {row['source_refdata']: row for row in manifest['target_binding']['state_mapping']}
    if (len({row['decode']['source_refdata'] for row in owners}) != 80
            or len({row['prefill']['source_refdata'] for row in owners}) != 80
            or len({row['prefill']['runtime_input_index'] for row in owners}) != 80
            or len({row['prefill']['runtime_output_index'] for row in owners}) != 80):
        raise ValueError('bound cross-model prefill owner mapping is not a bijection')
    for owner in declaration['state_mapping']:
        decode, prefill = owner['decode'], owner['prefill']; row = by_decode.get(decode['source_refdata'], {})
        old = decoded.get(decode['source_refdata'], {})
        if any(old.get(key) != decode[key] for key in ('runtime_input_index', 'runtime_output_index')):
            raise ValueError('prefill cross-model owner differs from active decode state identity')
        keys = {'decode_source_refdata': decode['source_refdata'], 'decode_runtime_input_index': decode['runtime_input_index'],
            'decode_runtime_output_index': decode['runtime_output_index'], 'prefill_source_refdata': prefill['source_refdata'],
            'prefill_runtime_input_index': prefill['runtime_input_index'], 'prefill_runtime_output_index': prefill['runtime_output_index'],
            'bytes': owner['bytes'], 'initialization': 'ZERO_FP16'}
        if any(row.get(key) != value for key, value in keys.items()):
            raise ValueError('actual prefill state owner differs from the bound cross-model map')
        source = inputs[prefill['runtime_input_index']]; output = outputs[prefill['runtime_output_index']]
        if source['bytes'] not in zeros:
            zeros[source['bytes']] = hashlib.sha256(bytes(source['bytes'])).hexdigest()
        if (source['bytes'] != owner['bytes'] or output['bytes'] != owner['bytes']
                or manifest['target_binding']['base_plan']['inputs'][decode['runtime_input_index']]['bytes'] != owner['bytes']
                or source['sha256'] != zeros[source['bytes']] or source['name'] != prefill['source_refdata']
                or record(row.get('prefill_output'), actual=True) != record(output, actual=True)):
            raise ValueError('prefill zero initialization or actual state output bytes differ')
    feature = receipt.get('feature_prefix', {})
    raw, _ = _read_small(root / record(outputs[81], actual=True)['path'], 4 << 20)
    valid = raw[:length * 32768]
    if (record(feature.get('source'), actual=True) != record(outputs[81], actual=True)
            or feature.get('row_start') != 0 or feature.get('row_count') != length or feature.get('bytes') != len(valid)
            or len(raw) != 64 * 32768 or feature.get('sha256') != hashlib.sha256(valid).hexdigest()
            or all_features[:len(valid)] != valid):
        raise ValueError('actual prefill valid feature rows are not the cumulative draft prefix')
    logits, _ = _read_small(root / record(outputs[0], actual=True)['path'], 2 << 20)
    values = [value[0] for value in struct.iter_unpack('<f', logits)]
    if len(values) != 248320 or not all(math.isfinite(value) for value in values):
        raise ValueError('prefill logits are not the complete finite vocabulary')
    if receipt.get('next_token_id') != max(range(len(values)), key=values.__getitem__):
        raise ValueError('prefill anchor differs from actual full-vocabulary greedy logits')
    return receipt, outputs


def validate_retrieved_loop(manifest, directory, *, _expected_evidence_kind='device-hiai',
                            _expected_config_sha256=None):
    """Select the pinned compact or complete native-evidence validation contract."""
    if not _dump_mode(manifest):
        return validate_retrieved_tokens(manifest, directory, _expected_evidence_kind=_expected_evidence_kind,
            _expected_config_sha256=_expected_config_sha256)
    root = Path(directory)
    inventory_before = _full(root / 'retrieval-manifest.tsv')
    raw, _ = _read_small(root / 'retrieval-manifest.tsv', 8 << 20)
    if _full(root / 'retrieval-manifest.tsv') != inventory_before:
        raise ValueError('retrieval inventory changed before current audit baselines')
    rows = _inventory(raw)
    by_path = {row['path']: row for row in rows}
    snapshots = {}
    for row in rows:
        path = root / row['path']; before = _full(path)
        if before['bytes'] != row['bytes'] or sha256_file(path) != row['sha256'] or _full(path) != before:
            raise ValueError('retrieved supervisor file differs from its final inventory: ' + row['path'])
        snapshots[str(path)] = before
    loop, _ = _json(root / 'loop.json')
    if (loop.get('status') != 'BOARD_SERIAL_DFLASH_EXECUTED_UNVALIDATED'
            or loop.get('evidence_kind') != _expected_evidence_kind
            or loop.get('prompt') != manifest['prompt_token_ids']
            or len(loop.get('generated_tokens', [])) != manifest['max_new_tokens']
            or loop.get('full_prefix') != [*manifest['prompt_token_ids'], *loop['generated_tokens']]):
        raise ValueError('supervisor loop is not this complete real request')
    counts = loop.get('counts', {})
    expected_target = len(manifest['prompt_token_ids']) + manifest['max_new_tokens']
    has_prefill = _has_prefill(manifest)
    expected_decode = manifest['max_new_tokens'] if has_prefill else expected_target
    if (counts.get('target_native_calls') != expected_decode
            or has_prefill and (counts.get('prefill_native_calls') != 1
                or loop.get('prefill_model_sha256') != manifest['prefill']['model']['sha256']
                or loop.get('prompt_ingestion') != _PREFILL_RECIPE)
            or type(counts.get('draft_native_calls')) is not int
            or counts['draft_native_calls'] < manifest['minimum_draft_rounds']
            or len(loop.get('target_steps', [])) != expected_decode
            or len(loop.get('draft_steps', [])) != counts['draft_native_calls']
            or len(loop.get('final_state_files', [])) != 80 or len(loop.get('final_target_outputs', [])) != 82):
        raise ValueError('supervisor did not retain all actual calls, states, and outputs')
    for token in loop['full_prefix']:
        if type(token) is not int or not 0 <= token < 248320:
            raise ValueError('supervisor token is outside the complete vocabulary')
    base = manifest['target_binding']['base_plan']
    resident = _resident_mode(manifest)
    expected = {'schema_version': 3 if resident else 1, 'artifact_kind': 'qwen35-board-local-serial-dflash',
        'target_model_sha256': manifest['target_model']['sha256'],
        'draft_model_sha256': manifest['draft']['model_source']['sha256'],
        'draft_abi_sha256': manifest['draft']['abi']['sha256'],
        'runtime_version': base['runtime_version'], 'ddk_id': base['ddk_id'], 'firmware_id': base['firmware_id'],
        'state_policy': 'ADVANCE_ONLY_COMMITTED_TOKENS_NO_SPECULATIVE_STATE',
        'control_semantics': ('EXPERIMENTAL_PADDED_PREFILL_VALID_LENGTH_NOT_NUMERICALLY_VALIDATED'
            if has_prefill else 'EXPERIMENTAL_SINGLE_QUERY_ABSOLUTE_POSITION')}
    if any(loop.get(key) != value for key, value in expected.items()):
        raise ValueError('supervisor model, environment, or scheduling identity differs from preparation')
    if resident:
        if (loop.get('execution_mode') != manifest['execution_mode']
                or loop.get('resident_abi_sha256') != manifest['resident_abi']['sha256']
                or _v1_mode(manifest) and loop.get('prefill_abi_sha256') != manifest['prefill_abi']['sha256']
                or loop.get('dump') is not True or loop.get('model_compilation') != 'NOT_RUN'
                or loop.get('numerical_validation') not in {'NOT_RUN', 'NOT_EVALUATED'}
                or loop.get('performance') != 'NOT_EVALUATED'):
            raise ValueError('resident result does not bind the selected raw descriptor ABI')
        _resident_lifecycle(loop, manifest, success=True)
    elif loop.get('execution_mode') in {'resident-v2', 'resident-v1'} or 'resident_lifecycle' in loop:
        raise ValueError('legacy request cannot inherit resident execution evidence')
    if (_SHA.fullmatch(str(loop.get('config_sha256'))) is None
            or _expected_config_sha256 is not None and loop['config_sha256'] != _expected_config_sha256):
        raise ValueError('supervisor did not execute this pinned native config')

    def record(value, *, actual=False):
        if (not isinstance(value, dict) or type(value.get('bytes')) is not int or value['bytes'] <= 0
                or _SHA.fullmatch(str(value.get('sha256'))) is None):
            raise ValueError('invalid supervisor file record')
        name = str(_relative(value.get('path')))
        core = {key: value[key] for key in ('path', 'bytes', 'sha256')}
        if actual and by_path.get(name) != core:
            raise ValueError('native evidence is absent from the verified retrieval inventory: ' + name)
        return core

    def load_record(value):
        row = record(value, actual=True)
        obj, identity = _json(root / row['path'])
        if any(identity[key] != row[key] for key in ('bytes', 'sha256')):
            raise ValueError('native receipt changed after retrieval')
        return obj

    def logs(receipt_record, pid):
        folder = PurePosixPath(receipt_record['path']).parent
        for name in ('tool.stdout', 'tool.stderr', 'tool.pid', 'tool.exitcode', 'command.json'):
            if str(folder / name) not in by_path:
                raise ValueError('native child logs or command are incomplete')
        if ((root / folder / 'tool.pid').read_text() != str(pid) + '\n'
                or (root / folder / 'tool.exitcode').read_text() != '0\n'):
            raise ValueError('native child PID/exit differs from its successful receipt')
        return _json(root / folder / 'command.json')[0]

    mapping = manifest['target_binding']['state_mapping']
    if (len(mapping) != 80 or len({row['source_refdata'] for row in mapping}) != 80
            or len({row['runtime_input_index'] for row in mapping}) != 80
            or len({row['runtime_output_index'] for row in mapping}) != 80):
        raise ValueError('prepared state mapping is not an80-owner bijection')
    previous, target_receipts, child_pids, zeros, resident_calls = None, {}, set(), {}, {}
    all_features, _ = _read_small(root / record(loop['final_features'], actual=True)['path'], 4 << 20)
    if len(all_features) != expected_target * 32768:
        raise ValueError('retained final features do not cover every committed target step')
    if has_prefill:
        seed, source_outputs = _validate_prefill_seed(manifest, root, loop, record, load_record, logs, zeros,
            all_features, _expected_evidence_kind, resident_calls)
        if not _v1_mode(manifest):
            child_pids.add(seed['pid'])
        target_receipts[len(manifest['prompt_token_ids']) - 1] = seed
        previous = [None] * 82
        for owner in manifest['prefill']['state_mapping']:
            previous[owner['decode']['runtime_output_index']] = source_outputs[owner['prefill']['runtime_output_index']]
    elif loop.get('prefill_steps') or loop.get('prefill_model_sha256'):
        raise ValueError('serial request cannot inherit unrequested prefill evidence')
    for index, event in enumerate(loop['target_steps']):
        position = index + (len(manifest['prompt_token_ids']) if has_prefill else 0)
        receipt = load_record(event['receipt']); pid = event.get('pid')
        if resident:
            pid = _resident_call(receipt, event, loop, resident_calls, 'target', position)
        if ((not resident and (type(pid) is not int or not 0 < pid <= (1 << 31) - 1 or pid in child_pids))
                or any(receipt.get(key) != value for key, value in {
                    'schema_version': 2 if resident else 1, 'artifact_kind': 'qwen35-board-serial-target-step',
                    'evidence_kind': _expected_evidence_kind, 'status': 'EXECUTED_OUTPUTS_BOUND',
                    **({} if resident else {'exitcode': 0, 'pid': pid}), 'absolute_position': position,
                    'model_sha256': manifest['target_model']['sha256'],
                    'token_id': loop['full_prefix'][position], 'next_token_id': event.get('next_token_id'),
                    'control_semantics': 'EXPERIMENTAL_SINGLE_QUERY_ABSOLUTE_POSITION'}.items())
                or event.get('token_id') != receipt['token_id']):
            raise ValueError('target native step identity/token order is inconsistent')
        if not resident:
            child_pids.add(pid); logs(event['receipt'], pid)
        inputs, outputs = receipt.get('inputs', []), receipt.get('outputs', [])
        if len(inputs) != 89 or len(outputs) != 82:
            raise ValueError('target native receipt omits ordered89/82 IO')
        if resident:
            _resident_io(receipt.get('io_diagnostic'), inputs, outputs)
            if _v1_mode(manifest):
                _v1_observed_abi(receipt, manifest['resident_abi'])
        for index, (value, descriptor) in enumerate(zip(inputs, base['inputs'])):
            record(value)
            if (value.get('index') != index or value.get('name') != descriptor['name']
                    or value['bytes'] != descriptor['bytes'] or value.get('tool_type') != descriptor['tool_input_type']):
                raise ValueError('target native input differs from its named compiled ABI')
        for index, (value, descriptor) in enumerate(zip(outputs, base['outputs'])):
            record(value)
            if (value.get('index') != index or value.get('runtime_name') != descriptor['runtime_tensor_name']
                    or value['bytes'] != descriptor['bytes'] or value.get('tool_type') != descriptor['tool_output_type']):
                raise ValueError('target native output differs from its named compiled ABI')
        feedback = {row.get('source_refdata'): row for row in receipt.get('state_feedback', [])}
        if len(receipt.get('state_feedback', [])) != 80 or len(feedback) != 80:
            raise ValueError('target native state feedback is incomplete or duplicated')
        for owner in mapping:
            row = feedback.get(owner['source_refdata'], {})
            a, b = owner['runtime_input_index'], owner['runtime_output_index']
            if (row.get('runtime_input_index') != a or row.get('runtime_output_index') != b
                    or row.get('input') != inputs[a]):
                raise ValueError('target native feedback differs from the proved state-owner mapping')
            if previous is None:
                size = inputs[a]['bytes']
                if size not in zeros:
                    zeros[size] = hashlib.sha256(bytes(size)).hexdigest()
                if (row.get('initialization') != 'ZERO_FP16' or 'parent_output' in row
                        or inputs[a]['sha256'] != zeros[size]):
                    raise ValueError('target initial state is not the explicit80 zero buffers')
            elif (record(row.get('parent_output')) != record(previous[b])
                    or inputs[a]['sha256'] != previous[b]['sha256'] or inputs[a]['bytes'] != previous[b]['bytes']):
                raise ValueError('target input bytes do not match the exact preceding80 outputs')
        from .original_target_smoke_backend import _controls
        controls = _controls(position)
        for name, width, table in (('arg1_1', 2048, manifest['tables']['weight']),
                                   ('arg0_1', 4, manifest['tables']['scale'])):
            before = _full(table['path'])
            if before != table['identity']:
                raise ValueError('original embedding table changed since whole-request preparation')
            fd = os.open(table['path'], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            try:
                if _identity(os.fstat(fd)) != before:
                    raise ValueError('original embedding table changed before row read')
                data = os.pread(fd, width, receipt['token_id'] * width)
                if len(data) != width or _identity(os.fstat(fd)) != before or _full(table['path']) != before:
                    raise ValueError('original embedding table changed during row read')
            finally:
                os.close(fd)
            controls[name] = data
        for value in inputs:
            if value['name'] in controls and hashlib.sha256(controls[value['name']]).hexdigest() != value['sha256']:
                raise ValueError('target input differs from original embedding row or declared query controls')
        feature = all_features[position * 32768:(position + 1) * 32768]
        if outputs[81]['sha256'] != hashlib.sha256(feature).hexdigest():
            raise ValueError('draft cumulative features differ from an actual target output chunk')
        if position and receipt['token_id'] != target_receipts[position - 1]['next_token_id']:
            if position >= len(manifest['prompt_token_ids']):
                raise ValueError('a committed generated token differs from current target greedy logits')
        previous = outputs; target_receipts[position] = receipt
    if [record(row, actual=True) for row in loop['final_target_outputs']] != [record(row) for row in previous]:
        raise ValueError('final82 retained outputs differ from the final native step')
    final_states = {row.get('source_refdata'): row for row in loop['final_state_files']}
    if len(final_states) != 80:
        raise ValueError('final state owners are duplicated')
    for owner in mapping:
        row = final_states.get(owner['source_refdata'], {})
        if (row.get('runtime_input_index') != owner['runtime_input_index']
                or row.get('runtime_output_index') != owner['runtime_output_index']
                or record(row, actual=True) != record(previous[owner['runtime_output_index']])):
            raise ValueError('final80 retained files differ from the proved state-owner outputs')
    for event in loop['draft_steps']:
        receipt = load_record(event['receipt']); pid = event.get('pid')
        if resident:
            pid = _resident_call(receipt, event, loop, resident_calls, 'draft', len(receipt.get('prefix', [])))
        if ((not resident and (type(pid) is not int or not 0 < pid <= (1 << 31) - 1 or pid in child_pids))
                or any(receipt.get(key) != value for key, value in {'schema_version': 2 if resident else 1,
                    'artifact_kind': 'qwen35-board-serial-draft-step', 'evidence_kind': _expected_evidence_kind,
                    **({} if resident else {'pid': pid, 'exitcode': 0}), 'model_sha256': manifest['draft']['model_source']['sha256']}.items())):
            raise ValueError('draft native step identity is inconsistent')
        if not resident:
            child_pids.add(pid); logs(event['receipt'], pid)
        prefix = receipt.get('prefix', [])
        if not 1 <= len(prefix) <= 64 or prefix != loop['full_prefix'][:len(prefix)]:
            raise ValueError('draft did not receive the complete ordered committed prefix')
        raw_feature = all_features[:len(prefix) * 32768]
        if receipt.get('feature_prefix_sha256') != hashlib.sha256(raw_feature).hexdigest():
            raise ValueError('draft logical feature prefix differs from actual target history')
        native = load_record(receipt['native_receipt'])
        if (native.get('status') != 'EXECUTED' or native.get('evidence_kind') != _expected_evidence_kind
                or native.get('process_id') != pid or (not resident and (native.get('target_calls') != 0 or native.get('draft_calls') != 1))
                or native.get('model_sha256') != manifest['draft']['model_source']['sha256']
                or native.get('draft_abi_sha256') != manifest['draft']['abi']['sha256']
                or native.get('runtime_version') != base['runtime_version'] or native.get('sequence_length') != 64
                or native.get('ddk_id') != base['ddk_id'] or native.get('firmware_id') != base['firmware_id']
                or native.get('model_load_api') != ('file-v2' if resident and not _v1_mode(manifest) else 'legacy-buffer')):
            raise ValueError('draft native receipt does not bind this exact full model call')
        if resident and (native.get('schema_version') != 2 or native.get('artifact_kind') != 'qwen35-resident-draft-call'
                or native.get('session_id') != receipt['session_id'] or native.get('call_index') != receipt['call_index']
                or native.get('execution_mode') != manifest['execution_mode']):
            raise ValueError('draft buffer evidence differs from its resident call identity')
        if _v1_mode(manifest):
            _v1_observed_abi(native, {'inputs': [dict(shape=shape, bytes=size) for shape, size in
                [([1, 64, 16384], 2097152), ([1, 64], 256), ([1, 64], 256)]],
                'outputs': [dict(shape=[1, 15], bytes=60), dict(shape=[1, 15, 2048], bytes=61440)]})
        expected_inputs = [raw_feature + bytes((64 - len(prefix)) * 32768),
            struct.pack('<64i', *(prefix + [0] * (64 - len(prefix)))),
            struct.pack('<64i', *([1] * len(prefix) + [0] * (64 - len(prefix))))]
        if len(receipt.get('inputs', [])) != 3 or len(native.get('inputs', [])) != 3:
            raise ValueError('draft complete-prefix input evidence is incomplete')
        for index, data in enumerate(expected_inputs):
            row = receipt['inputs'][index]; actual = native['inputs'][index]
            if (row.get('index') != index or row.get('bytes') != len(data)
                    or row.get('sha256') != hashlib.sha256(data).hexdigest()
                    or any(actual.get(key) != row[key] for key in ('index', 'bytes', 'sha256'))):
                raise ValueError('draft native inputs differ from the full right-padded actual prefix')
        diagnostic = native.get('io_diagnostic', {})
        if (diagnostic.get('enabled') is not True or diagnostic.get('process_called') is not True
                or diagnostic.get('output_fill_byte') != 0
                or diagnostic.get('input_readback_point') != 'AFTER_ALL_INPUT_COPIES_AND_OUTPUT_FILL_BEFORE_PROCESS'
                or len(diagnostic.get('inputs', [])) != 3 or len(diagnostic.get('outputs', [])) != 2):
            raise ValueError('draft native input buffer readback evidence is incomplete')
        for index, data in enumerate(expected_inputs):
            readback = diagnostic['inputs'][index]
            if (readback.get('index') != index or readback.get('readback_matches_source') is not True
                    or readback.get('readback_sha256') != hashlib.sha256(data).hexdigest()):
                raise ValueError('draft actual buffer readback differs from full-prefix input bytes')
        for index, value in enumerate(diagnostic['outputs']):
            if (value.get('index') != index or value.get('checked_after_process') is not True
                    or value.get('object_same') is not True or value.get('buffer_same') is not True):
                raise ValueError('draft output buffers changed through native Process')
        step_path = str(PurePosixPath(event['receipt']['path']).parent / 'bundle/step.tsv')
        if not resident and (step_path not in by_path or native.get('step_plan_sha256') != by_path[step_path]['sha256']):
            raise ValueError('draft native step recipe differs from the actual retained invocation')
        native_folder = PurePosixPath(receipt['native_receipt']['path']).parent
        top = native_folder / 'output_0000.bin'; hidden = native_folder / 'output_0001.bin'
        if str(top) not in by_path or str(hidden) not in by_path:
            raise ValueError('draft actual two output files are missing')
        if len(native.get('outputs', [])) != 2:
            raise ValueError('draft native output evidence is incomplete')
        for index, name in enumerate((top, hidden)):
            row = native['outputs'][index]; retained = by_path[str(name)]
            if (row.get('index') != index or row.get('path') != name.name
                    or any(row.get(key) != retained[key] for key in ('sha256', 'bytes'))):
                raise ValueError('draft actual output file differs from native result bytes')
        proposals = list(struct.unpack('<15i', (root / top).read_bytes()))
        if (proposals != event.get('proposals') or proposals != receipt.get('proposals')
                or any(not 0 <= token < 248320 for token in proposals)):
            raise ValueError('draft proposals differ from actual native Top1 bytes')
    if len(loop.get('commits', [])) != manifest['max_new_tokens']:
        raise ValueError('generated-token commit evidence is incomplete')
    for index, row in enumerate(loop['commits']):
        position = len(manifest['prompt_token_ids']) + index
        if (row.get('position') != position or row.get('target_step_index') != (index if has_prefill else position)
                or row.get('token_id') != loop['generated_tokens'][index]
                or row.get('role') != target_receipts[position]['role']):
            raise ValueError('generated-token commits differ from actual target calls')
    rounds = loop.get('rounds', [])
    if len(rounds) != counts['draft_native_calls']:
        raise ValueError('draft rounds differ from actual native proposal calls')
    committed = len(manifest['prompt_token_ids']) + 1
    if loop['commits'][0]['role'] != 'initial_target_next_anchor':
        raise ValueError('first generated token is not the eager target-next anchor')
    totals = dict(considered=0, accepted=0, corrections=0, bonuses=0)
    for number, (round_row, draft_event) in enumerate(zip(rounds, loop['draft_steps'])):
        draft = load_record(draft_event['receipt'])
        prefix_before = loop['full_prefix'][:committed]
        proposals = draft_event['proposals']
        if (round_row.get('round') != number or round_row.get('prefix_before') != prefix_before
                or draft['prefix'] != prefix_before or round_row.get('proposals') != proposals
                or round_row.get('anchor_position') != committed - 1
                or round_row.get('anchor_token_id') != prefix_before[-1]):
            raise ValueError('draft round prefix/anchor differs from current committed state')
        limit = min(manifest['max_proposals_per_round'], expected_target - committed)
        used = matched = 0; mismatch = False; new = []
        for index in range(limit):
            token = target_receipts[committed - 1]['next_token_id']
            okay = proposals[index] == token
            role = 'accepted_proposal' if okay else 'target_correction'
            if target_receipts[committed]['role'] != role or loop['full_prefix'][committed] != token:
                raise ValueError('round acceptance/correction differs from actual target greedy output')
            new.append(token); committed += 1; used += 1
            totals['considered'] += 1
            if okay:
                matched += 1; totals['accepted'] += 1
            else:
                mismatch = True; totals['corrections'] += 1
                if round_row.get('mismatch_index') != index or round_row.get('correction_token_id') != token:
                    raise ValueError('round mismatch evidence is inconsistent')
                break
        if not mismatch and used == limit and committed < expected_target:
            token = target_receipts[committed - 1]['next_token_id']
            if (target_receipts[committed]['role'] != 'all_considered_accepted_bonus'
                    or loop['full_prefix'][committed] != token or round_row.get('bonus_token_id') != token):
                raise ValueError('round bonus differs from actual target greedy output')
            new.append(token); committed += 1; totals['bonuses'] += 1
        if (round_row.get('considered') != used or round_row.get('accepted') != matched
                or round_row.get('discarded_unverified') != 15 - used or round_row.get('committed') != new
                or round_row.get('prefix_after') != loop['full_prefix'][:committed]):
            raise ValueError('round accounting differs from actual committed token evidence')
    if committed != expected_target or any(counts.get(key) != value for key, value in totals.items()):
        raise ValueError('loop totals or considered-only acceptance accounting is inconsistent')
    supervisor = loop.get('process_id')
    if type(supervisor) is not int or not 0 < supervisor <= (1 << 31) - 1 or supervisor in child_pids:
        raise ValueError('native supervisor PID is missing or aliases a child')
    if resident and (sorted(resident_calls) != list(range(counts['target_native_calls'] + counts['draft_native_calls']
                + (counts['prefill_native_calls'] if _v1_mode(manifest) else 0)))
            or [resident_calls[index] for index in sorted(resident_calls)] != sorted(resident_calls.values())):
        raise ValueError('resident call indices do not follow the actual committed-token scheduling order')
    if resident:
        progress, _ = _json(root / 'progress.json')
        _resident_progress(loop, progress, success=True)
        children = loop.get('child_lifecycle', {}); expected_children = [{'pid': pid, 'exitcode': 0} for pid in child_pids]
        expected_count = 0 if _v1_mode(manifest) else 1
        if (children.get('children') != expected_children or len(expected_children) != expected_count
                or any(children.get(key) != expected_count or progress.get(key) != expected_count for key in ('launched_child_count', 'completed_child_count'))
                or children.get('active_child_count') != 0 or progress.get('active_child_count') != 0
                or children.get('child_status') != ('NONE' if _v1_mode(manifest) else 'EXITED')
                or progress.get('child_status') != children.get('child_status')
                or any(children.get(key) != value or progress.get(key) != value for key, value in
                    ({'last_child_pid': None, 'last_child_exitcode': None} if _v1_mode(manifest)
                     else {'last_child_pid': expected_children[0]['pid'], 'last_child_exitcode': 0}).items())):
            raise ValueError('resident full audit child ledger differs from the pinned runtime mode')
    _validate_summaries(loop, manifest)
    for path, before in snapshots.items():
        if _full(path) != before:
            raise ValueError('retrieved native evidence changed during the final audit')
    if _full(root / 'retrieval-manifest.tsv') != inventory_before:
        raise ValueError('retrieval inventory changed during the final audit')
    return loop


def validate_retrieved_tokens(manifest, directory, *, _expected_evidence_kind='device-hiai',
                              _expected_config_sha256=None, _remote_exitcode=0):
    """Check compact token/completion evidence, without replaying state or proposals."""
    if _dump_mode(manifest):
        raise ValueError('compact token validation requires the pinned dump=false mode')
    root = Path(directory)
    inventory_path = root / 'retrieval-manifest.tsv'; baseline = _full(inventory_path)
    raw, _ = _read_small(inventory_path, 8 << 20); rows = _inventory(raw)
    if {row['path'] for row in rows} != {'loop.json', 'progress.json'} or len(rows) != 2:
        raise ValueError('compact retrieval must contain only loop.json and progress.json')
    pins = {str(root / row['path']): _full(root / row['path']) for row in rows}
    values = {}
    for row in rows:
        if not 0 < row['bytes'] <= 1 << 20:
            raise ValueError('compact token/status report exceeds its small metadata budget')
        value, identity = _json(root / row['path'], 1 << 20)
        if any(identity[key] != row[key] for key in ('bytes', 'sha256')):
            raise ValueError('compact report differs from its completed inventory SHA/extent')
        values[row['path']] = value
    loop, progress = values['loop.json'], values['progress.json']
    base = manifest['target_binding']['base_plan']
    success = _remote_exitcode == 0
    resident = _resident_mode(manifest)
    expected = {'schema_version': 3 if resident else 2, 'artifact_kind': 'qwen35-board-local-serial-dflash',
        'dump': False, 'retrieval_mode': 'FINAL_TOKENS_ONLY', 'evidence_kind': _expected_evidence_kind,
        'model_compilation': 'NOT_RUN',
        'status': 'BOARD_SERIAL_DFLASH_EXECUTED_UNVALIDATED' if success else 'FAILED',
        'target_model_sha256': manifest['target_model']['sha256'],
        'draft_model_sha256': manifest['draft']['model_source']['sha256'],
        'draft_abi_sha256': manifest['draft']['abi']['sha256'],
        'runtime_version': base['runtime_version'], 'ddk_id': base['ddk_id'], 'firmware_id': base['firmware_id'],
        'state_policy': 'ADVANCE_ONLY_COMMITTED_TOKENS_NO_SPECULATIVE_STATE',
        'control_semantics': 'EXPERIMENTAL_SINGLE_QUERY_ABSOLUTE_POSITION'}
    prefill = _has_prefill(manifest)
    if prefill:
        expected.update(prefill_model_sha256=manifest['prefill']['model']['sha256'], prompt_ingestion=_PREFILL_RECIPE,
            control_semantics='EXPERIMENTAL_PADDED_PREFILL_VALID_LENGTH_NOT_NUMERICALLY_VALIDATED')
    if resident:
        expected.update(execution_mode=manifest['execution_mode'], resident_abi_sha256=manifest['resident_abi']['sha256'])
        if _v1_mode(manifest):
            expected['prefill_abi_sha256'] = manifest['prefill_abi']['sha256']
    if (type(_remote_exitcode) is not int or not 0 <= _remote_exitcode <= 255
            or type(loop.get('schema_version')) is not int or type(loop.get('dump')) is not bool
            or any(loop.get(key) != value for key, value in expected.items())
            or _SHA.fullmatch(str(loop.get('config_sha256'))) is None
            or _expected_config_sha256 is not None and loop['config_sha256'] != _expected_config_sha256):
        raise ValueError('compact result does not bind this exact config/model/runtime and completed exit')
    if (set(loop) & {'target_steps', 'draft_steps', 'rounds', 'commits', 'final_state_files',
            'final_target_outputs', 'final_features', 'prefill_steps'}
            or loop.get('numerical_validation') not in {'NOT_RUN', 'NOT_EVALUATED'}
            or loop.get('performance') != 'NOT_EVALUATED'):
        raise ValueError('compact tokens-only report contains full-audit or unsupported accuracy/performance claims')
    prompt, generated, prefix = loop.get('prompt'), loop.get('generated_tokens'), loop.get('full_prefix')
    if (not isinstance(prompt, list) or prompt != manifest['prompt_token_ids'] or not isinstance(generated, list) or not isinstance(prefix, list)
            or len(generated) > manifest['max_new_tokens']
            or any(type(token) is not int or not 0 <= token < 248320 for token in [*prompt, *generated, *prefix])
            or prefix[:len(prompt)] != prompt[:len(prefix)]
            or (generated != prefix[len(prompt):] if len(prefix) >= len(prompt) else bool(generated))):
        raise ValueError('compact token history differs from the pinned prompt and ordered request')
    counts = loop.get('counts', {})
    keys = ('target_native_calls', 'draft_native_calls', 'considered', 'accepted', 'corrections', 'bonuses')
    if prefill:
        keys = (*keys, 'prefill_native_calls')
    elif counts.get('prefill_native_calls', 0) or loop.get('prefill_model_sha256') or loop.get('prompt_ingestion') == _PREFILL_RECIPE:
        raise ValueError('serial request cannot inherit an unrequested prefill variant')
    if (not isinstance(counts, dict) or any(type(counts.get(key)) is not int or counts[key] < 0 for key in keys)
            or counts['target_native_calls'] > len(prompt) + manifest['max_new_tokens']
            or counts['draft_native_calls'] > manifest['max_new_tokens']
            or prefill and not 0 <= counts['prefill_native_calls'] <= 1
            or counts['considered'] != counts['accepted'] + counts['corrections']):
        raise ValueError('compact native/acceptance counts are malformed or inconsistent')
    if resident:
        _resident_lifecycle(loop, manifest, success=success)
        _resident_progress(loop, progress, success=success)
        if _v1_mode(manifest):
            _v1_no_children(loop, progress)
    elif loop.get('execution_mode') in {'resident-v2', 'resident-v1'} or 'resident_lifecycle' in loop:
        raise ValueError('legacy compact request cannot inherit resident execution evidence')
    total = (0 if resident else counts['target_native_calls'] + counts['draft_native_calls']) + (
        counts['prefill_native_calls'] if prefill and not _v1_mode(manifest) else 0)
    lifecycle = loop.get('child_lifecycle', {})
    lifecycle_keys = ('launched_child_count', 'completed_child_count', 'active_child_count',
        'last_child_pid', 'last_child_exitcode', 'child_status')
    if (not isinstance(lifecycle, dict) or any(progress.get(key) != lifecycle.get(key)
                or type(progress.get(key)) is not type(lifecycle.get(key)) for key in lifecycle_keys)
            or any(type(lifecycle.get(key)) is not int or lifecycle[key] < 0
                for key in lifecycle_keys[:3])
            or lifecycle['launched_child_count'] != total or lifecycle['completed_child_count'] != total
            or lifecycle['active_child_count'] != 0 or lifecycle.get('child_status') in {'STARTING', 'RUNNING'}):
        raise ValueError('compact supervisor has active or unverified child completion; locks retained')
    children = lifecycle.get('children')
    if not isinstance(children, list) or len(children) != total or total > 128:
        raise ValueError('compact result omits its bounded child PID/exit completion ledger')
    pids = set()
    for child in children:
        pid, status = child.get('pid'), child.get('exitcode')
        if (type(pid) is not int or not 0 < pid <= (1 << 31) - 1 or pid in pids
                or type(status) is not int or not 0 <= status <= 255 or success and status != 0):
            raise ValueError('compact child PID/exit ledger is incomplete or inconsistent')
        pids.add(pid)
    if children:
        if (lifecycle.get('child_status') != 'EXITED'
                or lifecycle.get('last_child_pid') != children[-1]['pid']
                or lifecycle.get('last_child_exitcode') != children[-1]['exitcode']):
            raise ValueError('compact final child is not proven exited')
    elif lifecycle.get('last_child_pid') is not None or lifecycle.get('last_child_exitcode') is not None:
        raise ValueError('compact no-child result contains an unbound process')
    supervisor = loop.get('process_id')
    if (type(supervisor) is not int or not 0 < supervisor <= (1 << 31) - 1 or supervisor in pids
            or type(progress.get('supervisor_pid')) is not int or progress.get('supervisor_pid') != supervisor or progress.get('status') != loop['status']
            or progress.get('counts') != counts or any(type(progress.get('counts', {}).get(key)) is not int for key in keys)
            or progress.get('generated_tokens') != generated
            or progress.get('supervisor_timing') != loop.get('supervisor_timing')):
        raise ValueError('compact progress differs from its completed supervisor result')
    if success and (len(generated) != manifest['max_new_tokens'] or prefix != prompt + generated
            or counts['target_native_calls'] != (len(generated) if prefill else len(prompt) + len(generated))
            or prefill and counts['prefill_native_calls'] != 1
            or not manifest['minimum_draft_rounds'] <= counts['draft_native_calls'] <= counts['considered']
            or counts['considered'] > counts['draft_native_calls'] * manifest['max_proposals_per_round']
            or counts['corrections'] > counts['draft_native_calls'] or counts['bonuses'] > counts['draft_native_calls']
            or len(generated) != 1 + counts['considered'] + counts['bonuses']
            or type(loop.get('completed_rounds')) is not int or loop.get('completed_rounds') != counts['draft_native_calls']
            or loop.get('eos_policy') != 'DISABLED_FIXED_BUDGET_SMOKE'
            or loop.get('acceptance_scope') != 'CONSIDERED_K_ONLY_NOT_ALL15'):
        raise ValueError('compact fixed-budget request or considered-only accounting is incomplete')
    if not success and (not isinstance(loop.get('error'), str) or not loop['error'].strip()
            or loop.get('failed_child_pid') is not None and (not children
                or loop['failed_child_pid'] != children[-1]['pid']
                or loop.get('failed_child_exitcode') != children[-1]['exitcode'])):
        raise ValueError('compact failure lacks its bound error/last-child exit evidence')
    _validate_summaries(loop, manifest, required=True)
    for path, before in pins.items():
        if _full(path) != before:
            raise ValueError('compact result changed during validation')
    if _full(inventory_path) != baseline:
        raise ValueError('compact inventory changed during validation')
    return loop


def _validate_completed_failure(manifest, directory, state):
    """Known failed execution is recoverable evidence, never a successful loop."""
    root = Path(directory); loop, _ = _json(root / 'loop.json'); progress, _ = _json(root / 'progress.json')
    base = manifest['target_binding']['base_plan']
    expected = {'status': 'FAILED', 'evidence_kind': 'device-hiai', 'config_sha256': state['config_sha256'],
        'target_model_sha256': manifest['target_model']['sha256'],
        'draft_model_sha256': manifest['draft']['model_source']['sha256'],
        'draft_abi_sha256': manifest['draft']['abi']['sha256'],
        'runtime_version': base['runtime_version'], 'ddk_id': base['ddk_id'], 'firmware_id': base['firmware_id']}
    if _has_prefill(manifest):
        expected.update(prefill_model_sha256=manifest['prefill']['model']['sha256'], prompt_ingestion=_PREFILL_RECIPE)
    resident = _resident_mode(manifest)
    if resident:
        expected.update(schema_version=3, execution_mode=manifest['execution_mode'], resident_abi_sha256=manifest['resident_abi']['sha256'])
        if _v1_mode(manifest):
            expected['prefill_abi_sha256'] = manifest['prefill_abi']['sha256']
    if any(loop.get(key) != value for key, value in expected.items()):
        raise ValueError('failed supervisor lacks the complete bound model/config identity; locks retained')
    raw, _ = _read_small(root / 'retrieval-manifest.tsv', 8 << 20)
    rows = _inventory(raw); names = {row['path'] for row in rows}; pids = []; exits = {}
    for row in rows:
        path = root / row['path']
        if _full(path)['bytes'] != row['bytes'] or sha256_file(path) != row['sha256']:
            raise ValueError('failed supervisor evidence differs from its retained inventory')
        if PurePosixPath(row['path']).name != 'tool.pid':
            continue
        exit_path = str(PurePosixPath(row['path']).with_name('tool.exitcode'))
        if exit_path not in names:
            raise ValueError('a failed supervisor child has no known exit; locks retained')
        pid_text, exit_text = path.read_text(), (root / exit_path).read_text()
        if (re.fullmatch(r'[1-9][0-9]{0,9}\n', pid_text) is None or int(pid_text) > (1 << 31) - 1
                or re.fullmatch(r'(0|[1-9][0-9]{0,2})\n', exit_text) is None or int(exit_text) > 255):
            raise ValueError('failed supervisor child PID/exit evidence is malformed')
        pids.append(int(pid_text))
        if int(pid_text) in exits:
            raise ValueError('failed supervisor reused a child PID ambiguously')
        exits[int(pid_text)] = int(exit_text)
    child_present = any(key in progress for key in ('child_status', 'last_child_pid', 'last_child_exitcode'))
    counts = loop.get('counts', {})
    if resident:
        _resident_lifecycle(loop, manifest, success=False)
        _resident_progress(loop, progress, success=False)
        if _v1_mode(manifest):
            _v1_no_children(loop, progress)
    counted = (0 if resident else counts.get('target_native_calls', 0) + counts.get('draft_native_calls', 0)) + (
        counts.get('prefill_native_calls', 0) if _has_prefill(manifest) and not _v1_mode(manifest) else 0)
    if (progress.get('status') != 'FAILED' or progress.get('supervisor_pid') != loop.get('process_id')
            or counted != len(pids)
            or child_present and not _v1_mode(manifest) and (progress.get('child_status') != 'EXITED'
                or progress.get('last_child_pid') not in exits
                or type(progress.get('last_child_exitcode')) is not int
                or exits[progress['last_child_pid']] != progress['last_child_exitcode'])
            or loop.get('failed_child_pid') is not None and (loop['failed_child_pid'] not in exits
                or loop.get('failed_child_exitcode') != exits[loop['failed_child_pid']])):
        raise ValueError('failed supervisor has an active or uncertain final child; locks retained')
    supervisor = loop.get('process_id')
    if type(supervisor) is not int or not 0 < supervisor <= (1 << 31) - 1 or supervisor in pids:
        raise ValueError('failed supervisor PID is unverified')
    if _v1_mode(manifest):
        _validate_summaries(loop, manifest, required=True)
    return loop, [supervisor, *pids]


def _receive_verified_inventory_file(hdc, remote_root, local_root, row):
    """Receive against the completed inventory already checked in remote batches.

    This deliberately sends no additional per-file remote hash query.  A file
    changed after the batch check must still match the original inventory's
    exact bytes/SHA when received. Hdc.receive separately verifies publication
    of a new regular file, including zero-exit-without-artifact failures.
    """
    relative = _relative(row['path'])
    if (type(row.get('bytes')) is not int or not 0 <= row['bytes'] <= _MAX_FILE
            or _SHA.fullmatch(str(row.get('sha256'))) is None):
        raise ValueError('invalid bound retrieval-inventory file')
    local = Path(local_root) / relative
    local.parent.mkdir(parents=True, exist_ok=True)
    _safe(local.parent, directory=True)
    if os.path.lexists(local):
        raise FileExistsError('completed-loop retrieval never overwrites a local artifact: ' + str(local))
    hdc.receive(PurePosixPath(remote_root) / relative, local, timeout=min(30, hdc.timeout))
    before = _full(local)
    if before['bytes'] != row['bytes']:
        raise ValueError('retrieved loop file extent differs from the completed supervisor inventory: ' + row['path'])
    descriptor = os.open(local, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    digest = hashlib.sha256()
    with os.fdopen(descriptor, 'rb') as stream:
        if _identity(os.fstat(stream.fileno())) != before:
            raise ValueError('retrieved loop file changed before its bounded SHA256 read')
        remaining = row['bytes']
        while remaining:
            block = stream.read(min(1 << 20, remaining))
            if not block:
                raise ValueError('retrieved loop file truncated during SHA256 verification')
            digest.update(block); remaining -= len(block)
        if (stream.read(1) or _identity(os.fstat(stream.fileno())) != before
                or _full(local) != before):
            raise ValueError('retrieved loop file changed during SHA256 verification')
    if digest.hexdigest() != row['sha256']:
        raise ValueError('retrieved loop file SHA differs from the completed supervisor inventory: ' + row['path'])
    return before


def _retrieve(hdc, out, state, manifest):
    board = PurePosixPath(state['board_attempt'])
    dump = _dump_mode(manifest)
    state['stage'] = 'retrieve-completed-loop'
    # Recompute these summaries only from this call's verified native reports.
    # A recovery journal can contain an older CLOSED session summary.
    for key in ('prefill_lifecycle', 'prefill_failure_stage', 'resident_failure_stage'):
        state.pop(key, None)
    print('board-serial: supervisor completed; retrieving ' + ('full debug evidence' if dump else 'tokens and small completion status'), flush=True)
    for name in (_ATTEMPT_LOGS if dump else ('tool.pid', 'tool.exitcode')):
        if _receive(hdc, board, out, name) is None:
            raise ValueError('completed supervisor attempt is missing ' + name)
    if dump and (out / 'run.sh').read_text() != state['launcher']:
        raise ValueError('retained launcher differs from this one recorded launch')
    if (out / 'tool.pid').read_text() != str(state['completion']['launcher_process_id']) + '\n' or (
            out / 'tool.exitcode').read_text() != str(state['completion']['remote_exitcode']) + '\n':
        raise ValueError('retrieved completion files differ from the read-only process proof')
    remote_results = board / 'results'; local_results = out / 'results'
    if _receive(hdc, remote_results, local_results, 'retrieval-manifest.tsv') is None:
        raise ValueError('completed supervisor has no retrieval inventory')
    raw, inventory_id = _read_small(local_results / 'retrieval-manifest.tsv', 8 << 20)
    rows = _inventory(raw)
    if not dump and ({row['path'] for row in rows} != {'loop.json', 'progress.json'} or len(rows) != 2):
        raise ValueError('compact request returned unexpected per-step or tensor artifacts')
    # Independently compare the entire declared final inventory on the board.
    checks = []
    for row in rows:
        path = remote_results / row['path']
        checks.append(_regular_guard(path, size=row['bytes']) + ' && serial_hash=$(sha256sum '
            + shlex.quote(str(path)) + ') && serial_hash=${serial_hash%% *} && test "$serial_hash" = '
            + shlex.quote(row['sha256']))
    hash_batches = 0
    for batch in _batches(checks):
        hdc.shell(' && '.join(batch), 'verify-final-loop-inventory', timeout=hdc.timeout)
        hash_batches += 1
    state['retrieval_transport'] = {'inventory_files': len(rows), 'bounded_remote_hash_batches': hash_batches,
        'duplicate_per_file_remote_hash_queries_avoided': 0, 'file_receive_requests': 0,
        'inventory_payload_bytes': sum(row['bytes'] for row in rows),
        'hash_binding': 'COMPLETED_SUPERVISOR_INVENTORY_AND_BATCHED_REMOTE_SHA256_THEN_STABLE_LOCAL_SHA256',
        'scope': 'Final-result transport only; native execution and model residency are unchanged. Receive retries remain read-only.'}
    for row in rows:
        state['retrieval_transport']['file_receive_requests'] += 1
        state['retrieval_transport']['duplicate_per_file_remote_hash_queries_avoided'] += 1
        _receive_verified_inventory_file(hdc, remote_results, local_results, row)
    if not dump:
        loop = validate_retrieved_tokens(manifest, local_results, _expected_config_sha256=state['config_sha256'],
            _remote_exitcode=state['completion']['remote_exitcode'])
        completed = [loop['process_id'], *[row['pid'] for row in loop['child_lifecycle']['children']]]
    elif state['completion']['remote_exitcode'] == 0:
        loop = validate_retrieved_loop(manifest, local_results, _expected_config_sha256=state['config_sha256'])
        completed = [loop['process_id'],
            *([] if _resident_mode(manifest) else [row['pid'] for row in loop['target_steps']] + [row['pid'] for row in loop['draft_steps']]),
            *([] if _v1_mode(manifest) else [row['pid'] for row in loop.get('prefill_steps', [])])]
    else:
        loop, completed = _validate_completed_failure(manifest, local_results, state)
    if _full(local_results / 'retrieval-manifest.tsv')['bytes'] != inventory_id['bytes'] or sha256_file(
            local_results / 'retrieval-manifest.tsv') != inventory_id['sha256']:
        raise ValueError('retrieval inventory changed during verification')
    state.update(loop_report=str(local_results / 'loop.json'), loop=loop,
        retrieval_inventory=inventory_id,
        device_execution='EXECUTED_OUTPUTS_RETRIEVED' if state['completion']['remote_exitcode'] == 0 else 'FAILED_OUTPUTS_RETRIEVED',
        completed_native_pids=completed)
    _result_scope(state, manifest)
    # A completed failure is still an actual execution observation. Preserve
    # its counts and original failure location separately from retrieval's
    # outer stage (which becomes "complete" after owned-lock release).
    if 'counts' in loop:
        state['counts'] = loop['counts']
    if 'generated_tokens' in loop:
        state.update(generated_tokens=loop['generated_tokens'], generated_token_ids=loop['generated_tokens'])
    if _v1_mode(manifest):
        state['prefill_lifecycle'] = loop['prefill_lifecycle']
    if state['completion']['remote_exitcode'] != 0:
        if isinstance(loop.get('error'), str) and loop['error'].strip():
            state.update(error=loop['error'], execution_error=loop['error'])
        if isinstance(loop.get('stage'), str) and loop['stage']:
            state['execution_failure_stage'] = loop['stage']
        if _resident_mode(manifest):
            life = loop['resident_lifecycle']
            native_stage = life.get('failure_stage')
            if not native_stage and (life.get('status') == 'FAILED' or life.get('poisoned') is True):
                native_stage = life.get('stage')
            if isinstance(native_stage, str) and native_stage:
                state['resident_failure_stage'] = native_stage
        if _v1_mode(manifest):
            prefill_stage = loop['prefill_lifecycle'].get('failure_stage')
            if isinstance(prefill_stage, str) and prefill_stage:
                state['prefill_failure_stage'] = prefill_stage
    for key in ('acceptance', 'phase_latency', 'supervisor_timing'):
        if key in loop:
            state[key] = loop[key]
    if _resident_mode(manifest):
        state['resident_lifecycle'] = loop['resident_lifecycle']
        if 'resident_memory' in loop:
            state['resident_memory'] = loop['resident_memory']
    return loop


def _wait(hdc, out, state, timeout_seconds, poll_seconds):
    started = time.monotonic()
    while True:
        completion = _completion(hdc, state)
        if completion is not None:
            _checkpoint(out, state, 'remote-completed', completion=completion, uncertain=False)
            return completion
        if time.monotonic() - started >= timeout_seconds:
            _checkpoint(out, state, 'running-detached', uncertain=True)
            return None
        # This is coarse request-level polling, never an inter-token trigger.
        time.sleep(min(poll_seconds, max(.001, timeout_seconds - (time.monotonic() - started))))


def execute_board_serial(prepared_dir, output_dir, *, timeout_seconds=7200, lock_wait_seconds=1800,
                         poll_seconds=15, detach=False):
    """Stage once, detach once, and retrieve once; disconnects retain the journal."""
    from .board_serial_dflash import validate_prepared, make_config, TARGET_LIFECYCLE_POLICY
    if (type(timeout_seconds) not in {int, float} or not math.isfinite(timeout_seconds) or timeout_seconds <= 0
            or type(lock_wait_seconds) is not int or lock_wait_seconds < 0
            or type(poll_seconds) not in {int, float} or not 0 < poll_seconds <= 60 or type(detach) is not bool):
        raise ValueError('positive finite whole-request timeout/poll and nonnegative lock wait required')
    prepared, out = Path(prepared_dir).resolve(), output_path(output_dir)
    if prepared == out or prepared in out.parents:
        raise ValueError('execution must be outside immutable preparation')
    out.mkdir(parents=True, exist_ok=False)
    state = {'schema_version': 1, 'artifact_kind': _KIND, 'status': 'FAILED', 'prepared': str(prepared),
        'output': str(out), 'device_execution': 'NOT_RUN', 'model_compilation': 'NOT_RUN',
        'numerical_validation': 'NOT_EVALUATED', 'state_continuity': 'BYTE_CHAIN_ONLY_EXPERIMENTAL_CONTROLS',
        'board_lock_token': uuid.uuid4().hex, 'global_lock': str(_GLOBAL), 'global_held': False,
        'target_lock_held': False, 'prefill_lock_held': False, 'draft_lock_held': False, 'launch_requested': False, 'uncertain': False,
        'automatic_model_retry': 'FORBIDDEN', 'intertoken_hdc_calls': 0}
    hdc = None
    try:
        manifest = validate_prepared(prepared)
        if manifest.get('execution_mode') == 'resident-v2':
            raise ValueError('historical resident-v2 preparation is recover-only; prepare a new explicit all-V1 request')
        if _v1_mode(manifest) and (not _new_target_policy(manifest)
                or manifest.get('target_lifecycle_policy') != TARGET_LIFECYCLE_POLICY):
            raise ValueError('historical resident-v1 preparation is recover-only; prepare an explicit file-target/buffer-draft request')
        _result_scope(state, manifest)
        _, prep_id = _json(prepared / 'manifest.json')
        base = manifest['target_binding']['base_plan']
        executable, serial, server = _environment()
        token = state['board_lock_token']; digest = base['compilation_report_sha256']
        target = _remote(manifest['board_model_root']) / ('codex-target-' + digest[:20])
        prefill = None
        if _has_prefill(manifest):
            pdigest = manifest['prefill']['base_plan']['compilation_report_sha256']
            prefill = _remote(manifest['board_model_root']) / ('codex-target-' + pdigest[:20])
            if prefill == target:
                raise ValueError('prefill and decode require distinct compiled variant caches')
            state.update(prefill_cache=str(prefill), prefill_compilation_report_sha256=pdigest,
                prefill_model_sha256=manifest['prefill']['model']['sha256'])
        draft = PurePosixPath('/data/local/tmp/qwen35_real_draft_' + manifest['draft']['model_source']['sha256'])
        table_digest = hashlib.sha256((manifest['tables']['weight']['sha256'] + manifest['tables']['scale']['sha256']).encode()).hexdigest()
        tables = PurePosixPath('/data/local/tmp/qwen35_embedding_' + table_digest)
        board = PurePosixPath('/data/local/tmp/qwen35_board_serial_' + token)
        state.update(prepared_manifest=prep_id, serial=serial, hdc_server=server, hdc_executable=executable,
            board_attempt=str(board), target_cache=str(target), draft_cache=str(draft), tables_cache=str(tables),
            compilation_report_sha256=digest, draft_model_sha256=manifest['draft']['model_source']['sha256'],
            tables_cache_sha256=table_digest, runtime_version=base['runtime_version'], firmware_id=base['firmware_id'])
        _checkpoint(out, state, 'host-preflight')
        print('board-serial: building current native runtime; existing OMCs are reused', flush=True)
        uploads, search, deployment = (_build_payloads(out, timeout_seconds, resident=manifest['execution_mode'])
            if _resident_mode(manifest) else _build_payloads(out, timeout_seconds))
        state['libraries'] = deployment
        hdc = Hdc(executable, serial, server, out, timeout_seconds)
        tool = _remote(base['tool_binding']['path'])
        preflight = 'command -v sha256sum >/dev/null && command -v nohup >/dev/null && command -v find >/dev/null && '
        if not _v1_mode(manifest):
            preflight += (_regular_guard(tool) + ' && serial_hash=$(sha256sum ' + shlex.quote(str(tool))
                + ') && serial_hash=${serial_hash%% *} && test "$serial_hash" = ' + shlex.quote(base['tool_binding']['sha256']) + ' && ')
        preflight += 'param get const.ohos.fullname'
        if not _v1_mode(manifest):
            preflight += ' && printf \'TOOL_BYTES=\' && wc -c < ' + shlex.quote(str(tool))
        _, firmware = hdc.shell(preflight, 'whole-loop-device-preflight')
        observed = firmware.splitlines()
        if (len(observed) != (1 if _v1_mode(manifest) else 2) or observed[0].strip() != base['firmware_id'].strip()
                or not _v1_mode(manifest) and re.fullmatch(r'TOOL_BYTES=[1-9][0-9]*', observed[1]) is None):
            raise ValueError('board tool or firmware differs from the immutable decode proof')
        _mutate(hdc, out, state, 'mkdir ' + shlex.quote(str(board)), 'create-whole-loop-attempt')
        from .target_model_io_probe import _acquire_global
        print('board-serial: waiting for the shared board model lock', flush=True)
        _checkpoint(out, state, 'acquire-global-requested', uncertain=True)
        _acquire_global(hdc, _GLOBAL, token, lock_wait_seconds)
        _checkpoint(out, state, 'acquire-global-completed', global_held=True, uncertain=False)
        _owned_directory(hdc, out, state, target, '.compilation-sha256', digest, 'own-target-cache')
        _mutate(hdc, out, state, 'mkdir ' + shlex.quote(str(target / '.execution-lock'))
            + ' && printf \'%s\\n\' ' + shlex.quote(token) + ' > ' + shlex.quote(str(board / 'target-lock.owner')),
            'acquire-target-cache-lock')
        _checkpoint(out, state, 'target-cache-locked', target_lock_held=True)
        if prefill:
            _owned_directory(hdc, out, state, prefill, '.compilation-sha256', state['prefill_compilation_report_sha256'], 'own-prefill-cache')
            _mutate(hdc, out, state, 'mkdir ' + shlex.quote(str(prefill / '.execution-lock'))
                + ' && printf \'%s\\n\' ' + shlex.quote(token) + ' > ' + shlex.quote(str(board / 'prefill-lock.owner')),
                'acquire-prefill-cache-lock')
            _checkpoint(out, state, 'prefill-cache-locked', prefill_lock_held=True)
        _owned_directory(hdc, out, state, draft, '.model-sha256', state['draft_model_sha256'], 'own-draft-cache')
        _checkpoint(out, state, 'acquire-draft-cache-lock-requested', uncertain=True)
        _acquire_global(hdc, draft / '.execution-lock', token, 0)
        _checkpoint(out, state, 'draft-cache-locked', draft_lock_held=True, uncertain=False)
        _owned_directory(hdc, out, state, tables, '.tables-sha256', table_digest, 'own-embedding-cache')
        rows = _static_rows(manifest, target, draft, tables, prefill)
        for cache in (target, draft, tables, *([prefill] if prefill else [])):
            _directories(hdc, out, state, cache, [PurePosixPath(row['remote']).parent for row in rows
                if cache in PurePosixPath(row['remote']).parents], 'create-static-cache-directories')
        _checkpoint(out, state, 'inspect-static-cache-requested', uncertain=True)
        print('board-serial: checking target, draft and embedding cache SHA256 once for the whole request', flush=True)
        decisions = _cache_inventory(hdc, rows)
        _checkpoint(out, state, 'static-cache-inspected', uncertain=False)
        _upload_cached(hdc, out, state, rows, decisions, manifest)
        target_model = next(row['remote'] for row in rows if row['kind'] == 'target' and row['record'] == manifest['target_compiled']['omc'])
        draft_model = draft / 'model/model.omc'; attempt_model = board / 'draft-model.omc'
        _mutate(hdc, out, state, _regular_guard(draft_model, size=manifest['draft']['model_source']['bytes'])
            + ' && ln ' + shlex.quote(str(draft_model)) + ' ' + shlex.quote(str(attempt_model))
            + ' && test ' + shlex.quote(str(draft_model)) + ' -ef ' + shlex.quote(str(attempt_model)), 'link-cached-draft-model')
        abi = manifest['draft']['abi']; uploads.append({'local': abi['path'], 'remote': 'draft-abi.tsv',
            'bytes': abi['bytes'], 'sha256': abi['sha256']})
        if _resident_mode(manifest):
            abi = manifest['resident_abi']; uploads.append({'local': abi['path'], 'remote': 'resident-abi.tsv',
                'bytes': abi['bytes'], 'sha256': abi['sha256']})
        if _v1_mode(manifest):
            abi = manifest['prefill_abi']; uploads.append({'local': abi['path'], 'remote': 'prefill-abi.tsv',
                'bytes': abi['bytes'], 'sha256': abi['sha256']})
        library_paths = [str(board / 'lib'), *search]
        state['supervisor_library_path'] = ':'.join(library_paths)
        remote_paths = {'attempt': str(board), 'target_model': target_model,
            'target_weight_dir': str(PurePosixPath(target_model).parent),
            'draft_model': str(attempt_model),
            'board_runner': str(board / 'board-serial'), 'weight_table': str(tables / 'weight.bin'),
            'scale_table': str(tables / 'scale.bin'), 'draft_abi': str(board / 'draft-abi.tsv'),
            'config': str(board / 'config.tsv'), 'results': str(board / 'results'),
            'target_library_path': ':'.join(library_paths) if _v1_mode(manifest) else str(tool.parent),
            'draft_library_path': ':'.join(library_paths)}
        if not _v1_mode(manifest):
            remote_paths.update(target_tool=str(tool), draft_runner=str(board / 'draft-runner'),
                draft_runner_record={key: uploads[0][key] for key in ('bytes', 'sha256')},
                target_tool_record={'sha256': base['tool_binding']['sha256'], 'bytes': int(observed[1].split('=')[1])})
        if prefill:
            remote_paths['prefill_model'] = next(row['remote'] for row in rows if row['kind'] == 'prefill'
                and row['record'] == manifest['prefill']['compiled']['omc'])
        if _resident_mode(manifest):
            remote_paths['resident_abi'] = str(board / 'resident-abi.tsv')
        if _v1_mode(manifest):
            remote_paths['prefill_abi'] = str(board / 'prefill-abi.tsv')
        config = make_config(manifest, remote_paths)
        if not isinstance(config, bytes):
            raise TypeError('board serial config serializer must return bytes')
        (out / 'config.tsv').write_bytes(config)
        uploads.append(_file_record(out / 'config.tsv', 'config.tsv'))
        state['config_sha256'] = uploads[-1]['sha256']
        state['remote_paths'] = remote_paths
        state['launcher'] = _launcher(state)
        (out / 'run.sh').write_text(state['launcher'])
        uploads.append(_file_record(out / 'run.sh', 'run.sh'))
        _directories(hdc, out, state, board, [board / 'lib', *[(board / row['remote']).parent for row in uploads
            if PurePosixPath(row['remote']).parent != PurePosixPath('.')]], 'create-whole-loop-upload-directories')
        _checkpoint(out, state, 'upload-runtime-requested', uncertain=True, runtime_uploads=uploads)
        for entry in uploads:
            _check_local(entry)
            _checkpoint(out, state, 'upload-runtime-requested', uncertain=True, pending_runtime_payload=entry)
            hdc.send(entry['local'], board / entry['remote'])
        checksum = out / 'payloads.sha256'
        checksum.write_text(''.join(entry['sha256'] + '  ' + entry['remote'] + '\n' for entry in uploads))
        _checkpoint(out, state, 'upload-runtime-requested', uncertain=True, pending_runtime_payload={'path': str(checksum)})
        hdc.send(checksum, board / checksum.name, 'send-checksums')
        hdc.shell('cd ' + shlex.quote(str(board)) + ' && sha256sum -c ' + shlex.quote(checksum.name), 'verify-checksums')
        _checkpoint(out, state, 'runtime-uploaded', uncertain=False)
        executables = [board / 'board-serial', board / 'run.sh']
        if not _v1_mode(manifest):
            executables.insert(0, board / 'draft-runner')
        _mutate(hdc, out, state, 'chmod 755 ' + ' '.join(shlex.quote(str(path)) for path in executables), 'set-runtime-permissions')
        if validate_prepared(prepared) != manifest:
            raise ValueError('immutable preparation changed before the one launch')
        command = ('cd ' + shlex.quote(str(board)) + ' && test ! -e tool.pid && test ! -L tool.pid'
            + ' && test ! -e tool.exitcode && test ! -L tool.exitcode && test ! -e launch.guard && test ! -L launch.guard'
            + ' && { nohup sh ' + shlex.quote(str(board / 'run.sh'))
            + ' </dev/null > launcher.stdout 2> launcher.stderr & printf \'LAUNCHED\\n\'; }')
        _checkpoint(out, state, 'launch-requested', launch_requested=True, uncertain=True,
            device_execution='ATTEMPTED', launch_command=command)
        print('board-serial: starting detached autonomous board loop; host polling does not drive tokens', flush=True)
        _, launch_text = hdc.shell(command, 'launch-detached-board-serial', timeout=min(30, hdc.timeout))
        if launch_text != 'LAUNCHED':
            raise ValueError('detached launch lacks the exact acknowledgement; completion remains uncertain')
        _checkpoint(out, state, 'launch-acknowledged', uncertain=False)
        completion = None if detach else _wait(hdc, out, state, timeout_seconds, poll_seconds)
        if completion is None:
            state.update(status='BOARD_SERIAL_DFLASH_STARTED' if detach else 'BOARD_SERIAL_RUNNING_OR_UNCERTAIN',
                stage='running-detached', device_execution='RUNNING_OR_COMPLETION_PENDING')
        else:
            _lock_checks(hdc, state)
            _retrieve(hdc, out, state, manifest)
            _checkpoint(out, state, 'release-locks-requested', uncertain=True)
            state['lock_release'] = _lock_checks(hdc, state, release=True)
            state.update(status='BOARD_SERIAL_DFLASH_EXECUTED_UNVALIDATED' if completion['remote_exitcode'] == 0 else 'FAILED', stage='complete',
                global_held=False, target_lock_held=False, prefill_lock_held=False, draft_lock_held=False, uncertain=False)
    except BaseException as error:
        state.update(error=f'{type(error).__name__}: {error}', stage=state.get('phase', 'preflight'))
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            state['interruption'] = 'LOCAL_CONTROLLER_INTERRUPTED; NO_REMOTE_CANCELLATION_OR_RELAUNCH'
    finally:
        state['lock_state'] = ('RETAINED_OR_UNCERTAIN' if state['global_held'] or state['target_lock_held'] or state.get('prefill_lock_held')
            or state['draft_lock_held'] or state['uncertain'] else 'NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT')
        state['hdc_invocations'] = hdc.count if hdc else 0
        state['next'] = _next_action(state)
        _checkpoint(out, state, state.get('phase', 'failed'))
        save_json(out / 'result.json', state)
    return state


def recover_board_serial(execution_dir, output_dir, *, timeout_seconds=60, poll_seconds=15):
    """Read/poll/retrieve the identical detached request; never launch a model."""
    from .board_serial_dflash import validate_prepared
    if (type(timeout_seconds) not in {int, float} or not math.isfinite(timeout_seconds) or timeout_seconds <= 0
            or type(poll_seconds) not in {int, float} or not 0 < poll_seconds <= 60):
        raise ValueError('positive finite request recovery timeout and bounded poll interval required')
    execution, _ = _safe(execution_dir, directory=True)
    out = output_path(output_dir)
    if execution == out or execution in out.parents:
        raise ValueError('recovery output must be a separate new directory')
    out.mkdir(parents=True, exist_ok=False)
    state, identity = _json(execution / 'checkpoint.json')
    state = dict(state, recovery_source=str(execution), recovery_source_identity=identity, output=str(out),
        status='FAILED', device_execution='NOT_RUN_BY_RECOVERY', recovery_model_execution='FORBIDDEN')
    hdc = None
    try:
        if (state.get('artifact_kind') != _KIND or _TOKEN.fullmatch(str(state.get('board_lock_token'))) is None
                or not state.get('launch_requested') or state.get('global_lock') != str(_GLOBAL)
                or state.get('board_attempt') != '/data/local/tmp/qwen35_board_serial_' + state['board_lock_token']):
            raise ValueError('recovery requires this exact journaled detached launch and ownership token')
        for key in ('target_cache', 'draft_cache', 'tables_cache', 'board_attempt'):
            _remote(state[key])
        if (state['target_cache'].split('/')[-1] != 'codex-target-' + state['compilation_report_sha256'][:20]
                or state['draft_cache'] != '/data/local/tmp/qwen35_real_draft_' + state['draft_model_sha256']
                or state['launcher'] != _launcher(state)):
            raise ValueError('recovery journal cache/model/launcher identities are inconsistent')
        manifest = validate_prepared(state['prepared'])
        if manifest.get('execution_mode', 'fresh-process') != state.get('execution_mode', 'fresh-process'):
            raise ValueError('recovery cannot change the pinned resident execution mode')
        if state.get('target_lifecycle_policy') != manifest.get('target_lifecycle_policy'):
            raise ValueError('recovery cannot change the pinned target lifecycle policy')
        if _v1_mode(manifest):
            from .board_serial_dflash import v1_load_contract, TARGET_LIFECYCLE_POLICY
            session_api, model_apis = v1_load_contract(manifest)
            if ((manifest['target_lifecycle_policy'] == TARGET_LIFECYCLE_POLICY
                    and (state.get('model_load_api') != session_api or state.get('model_load_apis') != model_apis))
                    or any(key in state and state[key] != expected for key, expected in
                        (('model_load_api', session_api), ('model_load_apis', model_apis)))):
                raise ValueError('recovery cannot change the pinned V1 model source representations')
        if _has_prefill(manifest):
            pbase = manifest['prefill']['base_plan']
            expected_cache = str(_remote(manifest['board_model_root']) / ('codex-target-' + pbase['compilation_report_sha256'][:20]))
            if (state.get('prefill_cache') != expected_cache
                    or state.get('prefill_compilation_report_sha256') != pbase['compilation_report_sha256']
                    or state.get('prefill_model_sha256') != manifest['prefill']['model']['sha256']):
                raise ValueError('recovery prefill cache/model differs from the pinned request')
        elif state.get('prefill_cache'):
            raise ValueError('recovery cannot add a prefill variant to a serial request')
        if state.get('dump', True) != _dump_mode(manifest):
            raise ValueError('recovery cannot change the pinned dump/config mode')
        _result_scope(state, manifest)
        base = manifest['target_binding']['base_plan']
        if (state['compilation_report_sha256'] != base['compilation_report_sha256']
                or state['draft_model_sha256'] != manifest['draft']['model_source']['sha256']
                or state['runtime_version'] != base['runtime_version'] or state['firmware_id'] != base['firmware_id']):
            raise ValueError('recovery checkpoint differs from the exact prepared model and environment identities')
        _, current = _json(Path(state['prepared']) / 'manifest.json')
        if current != state['prepared_manifest']:
            raise ValueError('immutable preparation changed after the original launch')
        executable, serial, server = _environment()
        if (serial, server) != (state['serial'], state['hdc_server']):
            raise ValueError('recovery must use the same device and HDC server')
        hdc = Hdc(executable, serial, server, out, timeout_seconds)
        _checkpoint(out, state, 'recover-read-only')
        print('board-serial: recovering the same retained attempt; checking completion without relaunch', flush=True)
        _lock_checks(hdc, state)
        completion = _wait(hdc, out, state, timeout_seconds, poll_seconds)
        if completion is None:
            state.update(status='BOARD_SERIAL_RUNNING_OR_UNCERTAIN', stage='running-detached')
        else:
            _retrieve(hdc, out, state, manifest)
            _checkpoint(out, state, 'release-locks-requested', uncertain=True)
            state['lock_release'] = _lock_checks(hdc, state, release=True)
            state.update(status='BOARD_SERIAL_DFLASH_EXECUTED_UNVALIDATED' if completion['remote_exitcode'] == 0 else 'FAILED', stage='complete',
                global_held=False, target_lock_held=False, prefill_lock_held=False, draft_lock_held=False, uncertain=False)
    except BaseException as error:
        state.update(error=f'{type(error).__name__}: {error}', stage=state.get('phase', 'recovery-preflight'))
    finally:
        state['lock_state'] = ('RETAINED_OR_UNCERTAIN' if state.get('global_held') or state.get('target_lock_held') or state.get('prefill_lock_held')
            or state.get('draft_lock_held') or state.get('uncertain') else 'NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT')
        state['hdc_invocations'] = hdc.count if hdc else 0
        state['next'] = _next_action(state)
        _checkpoint(out, state, state.get('phase', 'recovery-failed'))
        save_json(out / 'result.json', state)
    return state
