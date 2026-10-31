"""Host-only evidence for a future prefill64 to decode1 bridge.

A later named metadata probe can add a same-model ordered-descriptor witness to
old retained outputs. It does not establish cross-export state ownership or a
valid next-token/control recipe, and never prepares or executes feedback.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

from qwen35_moe_dflash.common.artifacts import output_path, save_json
from qwen35_moe_dflash.toolchain.existing_air_profile import get_existing_air_profile
from qwen35_moe_dflash.target.original_target_board import original_input_files
from qwen35_moe_dflash.common.file_io import _relative, _safe
from qwen35_moe_dflash.target.prefill_output_inventory import _bound_execution, _descriptors, _hash_retrieved, _logged_slots, _LOGS, _MAX_TOTAL, _json
from qwen35_moe_dflash.target.prefill_smoke_plan import _identity, _nd_view_key, _runtime_keys, _source, _INPUT_TYPES
from qwen35_moe_dflash.target.state_lineage import _probe


_KIND = 'qwen35-target-prefill-decode-bridge-diagnosis'


def _same(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def bind_prefill_named_probe(plan, old_probe, named_probe):
    """Return side-witness names only when every ordered raw descriptor agrees."""
    for key in ('job', 'profile', 'compilation_report_sha256', 'model_sha256', 'model_bytes', 'runtime_version', 'firmware_id', 'ddk_id'):
        if not _same(old_probe.get(key), named_probe.get(key)):
            raise ValueError('old/new prefill metadata probes differ in bound identity: ' + key)
    inputs, outputs = [], []
    for direction, target in (('inputs', inputs), ('outputs', outputs)):
        old_keys, new_keys = _runtime_keys(old_probe, direction), _runtime_keys(named_probe, direction)
        if not _same(old_keys, new_keys) or len(old_keys) != (89 if direction == 'inputs' else 82):
            raise ValueError('old/new prefill ordered dtype/shape descriptors differ: ' + direction)
        if len(plan[direction]) != len(old_keys):
            raise ValueError('saved prefill plan descriptor count differs from its original probe')
        names = []
        for index, (old, new, row, key) in enumerate(zip(old_probe[direction], named_probe[direction], plan[direction], old_keys)):
            name = new.get('tensor_name')
            if (type(new.get('index')) is not int or new['index'] != index or type(old.get('index')) is not int or old['index'] != index
                    or old.get('raw_dtype') != new.get('raw_dtype') or old.get('raw_format') != new.get('raw_format')
                    or new.get('tensor_name_status') != 'OBSERVED' or not isinstance(name, str) or not 0 < len(name) <= 1024
                    or any(char in name for char in ('\0', '\r', '\n'))
                    or _nd_view_key((row.get('dtype'), tuple(row.get('shape', [])))) != _nd_view_key(key)):
                raise ValueError('named side witness does not retain the exact ordered raw prefill ABI')
            names.append(name)
            target.append({'runtime_index': index, 'tensor_name': name, 'dtype': key[0], 'shape': list(key[1]),
                'raw_dtype': new['raw_dtype'], 'raw_format': new['raw_format'],
                'binding': 'POST_HOC_SAME_MODEL_ORDERED_DESCRIPTOR_NAME_WITNESS'})
        if len(names) != len(set(names)):
            raise ValueError('named prefill metadata witness contains duplicate tensor names')
    if {row['tensor_name'] for row in inputs if row['tensor_name'].startswith('RefData_')} != {f'RefData_{i}' for i in range(80)}:
        raise ValueError('named prefill side witness must expose all80 distinct literal RefData inputs')
    return {'status': 'POST_HOC_SAME_MODEL_ORDERED_DESCRIPTOR_NAMES_BOUND', 'inputs': inputs, 'outputs': outputs,
        'scope': 'Names were queried by the later metadata-only probe for this identical OMC/runtime and ordered raw ABI; they were not observed during the older inference. Cross-export layer owners still require graph provenance.'}


def inspect_prefill_decode_bridge(prefill_execution, prefill_named_probe, output):
    out = output_path(output); out.mkdir(parents=True, exist_ok=False)
    result = {'schema_version': 1, 'artifact_kind': _KIND, 'status': 'FAILED', 'output': str(out),
        'device_execution': 'NOT_RUN', 'device_calls': 'NONE', 'model_compilation': 'NOT_RUN',
        'model_payload_access': 'NONE', 'weight_payload_access': 'NONE', 'feedback_prepared': False,
        'state_feedback': 'NOT_RUN', 'state_continuity': 'NOT_VALIDATED', 'numerical_validation': 'NOT_RUN'}
    bindings, snapshots = [], {}
    def pin(path, expected=None):
        path, stamp = _safe(path); current = _identity(path.stat())
        if [current[k] for k in ('device', 'inode', 'bytes', 'mtime_ns', 'ctime_ns')] != stamp:
            raise ValueError('bridge source changed before current baseline')
        if expected is not None and stamp != expected:
            raise ValueError('bridge file changed after its fresh read')
        if str(path) in snapshots and snapshots[str(path)] != current:
            raise ValueError('bridge source changed within this invocation')
        snapshots[str(path)] = current
    try:
        execution, _ = _safe(prefill_execution, directory=True)
        # First pin metadata/retained paths before any nested hash replay.
        for relative in ('result.json', 'plan.json', 'board/result.json', 'board/plan.json',
                'board/runtime-prefill-proof.json', 'board/command.json'):
            pin(execution / relative)
        _, board, plan = _bound_execution(execution, bindings)
        descriptors = _descriptors(plan)
        old_dir, _ = _safe(plan['runtime_probe'], directory=True)
        named_dir, _ = _safe(prefill_named_probe, directory=True)
        for directory in (old_dir, named_dir):
            for name in ('result.json', 'model-io.json', 'runtime.json', 'model-io-command.json', 'runtime.exitcode', 'model-io.exitcode'):
                pin(directory / name)
        metadata = {'job': str(Path(plan['compilation']).parent), 'profile': 'cpro-40block-prefill-64',
            'compilation_report_sha256': plan['compilation_report_sha256'], 'model_sha256': plan['model_sha256'],
            'model_bytes': plan['primary_model_binding']['bytes'], 'ddk_id': plan['ddk_id']}
        old_probe, new_probe = _probe(old_dir, metadata, bindings), _probe(named_dir, metadata, bindings)
        witness = bind_prefill_named_probe(plan, old_probe, new_probe)
        original_plan = _json(execution / 'plan.json', bindings)
        evidence_path = execution / 'board/input-evidence.json'; pin(evidence_path)
        evidence = _json(evidence_path, bindings)
        expected_inputs = [dict(row, host_source=row['source'],
            source=str(Path(board['board_attempt']) / 'source-inputs' / row['source_filename'])) for row in original_plan['inputs']]
        if (not _same(plan['inputs'], expected_inputs) or not _same(board.get('inputs'), expected_inputs)
                or not _same(evidence, {'inputs': expected_inputs})):
            raise ValueError('older prefill execution lacks all89 exact uploaded/private-slot input byte hashes')
        sources = original_plan.get('source_inputs')
        if not isinstance(sources, list) or len(sources) != 13 or len({row['path'] for row in sources}) != 13:
            raise ValueError('older first-prefill proof must retain its13 exact original source files')
        source_by_path = {row['path']: row for row in sources}
        widths = {'FLOAT16': 2, 'FLOAT32': 4, 'INT8': 1, 'INT16': 2, 'INT32': 4, 'INT64': 8}
        expected_sources = {}
        for row, key in zip(original_plan['inputs'], _runtime_keys(old_probe, 'inputs')):
            size = math.prod(key[1]) * widths[key[0]]
            path = str(Path(original_plan['input_dir']) / row['source_filename'])
            recorded = source_by_path.get(path)
            if (row['source'] != path or row['bytes'] != size or recorded is None
                    or recorded['bytes'] != size or recorded['sha256'] != row['sha256']):
                raise ValueError('prefill original source extent/hash differs from its actual ordered runtime descriptor')
            if path in expected_sources and expected_sources[path] != size:
                raise ValueError('one prefill source file has conflicting runtime byte extents')
            expected_sources[path] = size
        # The original first-call planner may choose one representative for
        # equal-byte typed groups (K/V, or position/new-KV). Its13 originals
        # remain pinned even when fewer source basenames were actually used.
        frozen = {}
        profile_rows = get_existing_air_profile('cpro-40block-prefill-64')['input_shape'].split(';')
        for entry, (_, filename) in zip(profile_rows, original_input_files()):
            name, dimensions = entry.split(':')
            dtype = 'FLOAT16' if name.startswith('RefData_') else _INPUT_TYPES[name]
            width = math.prod(int(value) for value in dimensions.split(',')) * widths[dtype]
            path = str(Path(original_plan['input_dir']) / filename)
            if path in frozen and frozen[path] != width:
                raise ValueError('frozen original source filename has conflicting extents')
            frozen[path] = width
        if (set(frozen) != set(source_by_path) or any(source_by_path[path]['bytes'] != size for path, size in frozen.items())
                or any(frozen.get(path) != size for path, size in expected_sources.items())):
            raise ValueError('original prefill sources differ from the exact13 declared original files')
        expected_sources = frozen
        for path in expected_sources: pin(path)
        current_sources = []
        for path, size in expected_sources.items():
            current = _source(path, size)
            if current['sha256'] != source_by_path[path]['sha256']:
                raise ValueError('current original prefill source bytes differ from the actually executed input hashes')
            pin(path, [current['identity'][key] for key in ('device', 'inode', 'bytes', 'mtime_ns', 'ctime_ns')])
            current_sources.append(current)
        retrieved = board.get('retrieved')
        if (not isinstance(retrieved, dict) or len(retrieved) != 86 or set(retrieved) != {*[name for name in retrieved if name.startswith('outputs/')], *_LOGS}
                or sum(row['bytes'] for row in retrieved.values()) > _MAX_TOTAL):
            raise ValueError('bridge requires exactly82 retained data files and4 complete logs within budget')
        for name in retrieved:
            pin(_relative(execution / 'board', name))
        files, logs = [], {}
        for name, recorded in retrieved.items():
            path = _relative(execution / 'board', name)
            identity, captured = _hash_retrieved(path, recorded, capture=name in _LOGS)
            pin(path, identity['stat'])
            if name in _LOGS:
                logs[name] = captured
            else:
                files.append(dict(identity, path=name))
        if logs['tool.exitcode'].strip() != b'0' or logs['tool.pid'].strip() != str(board['tool_process_id']).encode():
            raise ValueError('retained prefill completion logs differ from the successful inference receipt')
        mapping = _logged_slots(logs['tool.stdout'], board, descriptors, files)
        if mapping['status'] != 'BOUND_TO_COMPLETE_EXPLICIT_TOOL_LOG' or mapping.get('unlogged_retrieved_files'):
            raise ValueError('retained prefill files require all82 explicit logged output slots')
        by_path = {row['path']: row for row in files}
        outputs = [dict(witness['outputs'][row['runtime_index']], actual_file=dict(by_path[row['path']],
            path=str(_relative(execution / 'board', row['path'])))) for row in mapping['entries']]
        result.update(status='PREFILL_NAMED_SLOTS_AND_RETAINED_FILES_BOUND_BRIDGE_GAPS_REMAIN', execution=str(execution),
            original_probe=str(old_dir), named_probe=str(named_dir), input_count=89, output_count=82,
            named_probe_witness=witness, retained_outputs=outputs, logged_slot_mapping=mapping, source_bindings=bindings,
            original_source_inputs=current_sources, executed_input_byte_binding='ALL89_SAVED_PRIVATE_SLOT_SHA256_MATCHED_CURRENT13_ORIGINAL_SOURCES',
            historical_input_identity_scope='RECORDED_BYTE_SHA256_WITH_CURRENT_STABLE_FULL9; HISTORICAL_MODE_NOT_INFERRED',
            required_next_evidence=[
                {'id': 'CROSS_EXPORT_STATE_CORRESPONDENCE', 'status': 'NOT_EVALUATED',
                 'requirement': 'All80 source-layer owners must join prefill to decode through stable original graph/parameter provenance, not matching RefData ordinals or shapes'},
                {'id': 'PREFILL_LOGIT_SELECTION', 'status': 'NOT_EVALUATED',
                 'requirement': 'Trace executed arg1238_1/output_pos64 through the actual AIR Gather dependency; do not relabel the current argmax as a validated last-row next token'},
                {'id': 'ORIGINAL_TOKEN_EMBEDDING_AND_NEXT_ROW', 'status': 'NOT_EVALUATED',
                 'requirement': 'Prove all64 executed INT8 rows and FP32 scales against original input IDs/table rows, then obtain only the legitimately selected next original row'},
                {'id': 'DECODE_CONTROL_AND_INITIAL_HISTORY', 'status': 'UNVERIFIED',
                 'requirement': 'Positions/mask/current-query controls and original initial-state history need explicit evidence or a separately declared experiment'}],
            scope='Read-only same-OMC name side witness and old actual output-file slot bindings only; no cross-model state join, greedy-token acceptance, prepared inputs or board run')
        for identity in bindings:
            pin(identity['path'], identity['stat'])
        for path, current in snapshots.items():
            if _identity(_safe(path)[0].stat()) != current:
                raise ValueError('bridge source changed within this invocation: ' + path)
        result['source_preservation'] = 'CURRENT_METADATA_AND_ALL86_RETAINED_FILES_UNCHANGED'
        result['active_source_identities'] = [{'path': path, 'identity': deepcopy(identity)} for path, identity in sorted(snapshots.items())]
    except (OSError, ValueError, KeyError, TypeError, IndexError, RecursionError) as error:
        result.update(status='FAILED', error=f'{type(error).__name__}: {error}')
    save_json(out / 'report.json', result)
    return result
