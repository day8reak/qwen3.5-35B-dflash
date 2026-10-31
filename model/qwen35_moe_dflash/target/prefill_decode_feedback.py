"""Plan experimental prefill state feedback through exact graph/file joins.

The retained parameter filename/port contract is explicit. It does not attest
Python FQNs or numerical equivalence. This tool writes a mapping only; it does
not create decode inputs, select controls, load weights or execute a model.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import struct

from qwen35_moe_dflash.common.artifacts import output_path, save_json
from qwen35_moe_dflash.target import state_lineage as lineage
from qwen35_moe_dflash.common.file_io import _read_small, _safe
from qwen35_moe_dflash.target.parameter_fqn_owner import CONTRACT
from qwen35_moe_dflash.target.prefill_decode_bridge import inspect_prefill_decode_bridge
from qwen35_moe_dflash.target.prefill_last_row_trial import _pin, _stable
from qwen35_moe_dflash.target.prefill_output_inventory import _json, _numerical
from qwen35_moe_dflash.target.prefill_smoke_plan import _nd_view_key, _runtime_keys
from qwen35_moe_dflash.target.state_correspondence import _bound_inventory, analyze_state_correspondence


_KIND = 'qwen35-target-prefill-decode-feedback-plan'
_STATUS = 'PREFILL_DECODE_PARAMETER_CONTRACT_FEEDBACK_PLANNED'
_BRIDGE = 'PREFILL_NAMED_SLOTS_AND_RETAINED_FILES_BOUND_BRIDGE_GAPS_REMAIN'
_MAX_PLAN = 4 << 20


def _unique(rows, key, count, label):
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError(label + ' must contain exactly ' + str(count) + ' rows')
    result = {}
    for row in rows:
        if not isinstance(row, dict) or key not in row or row[key] in result:
            raise ValueError(label + ' has a missing or duplicate ' + key)
        result[row[key]] = row
    return result


def bind_prefill_decode_feedback(contract, prefill, decode, retained_outputs, prefill_probe, decode_probe):
    """Pure mapping; the caller binds both graphs, native probes and file bytes."""
    expected = {f'layer.{layer}.{role}' for layer in range(40)
        for role in (('key', 'value') if layer % 4 == 3 else ('conv', 'recurrent'))}
    owners = _unique(contract.get('mapping'), 'owner_key', 80, 'parameter owner mapping')
    if (contract.get('contract') != CONTRACT
            or contract.get('status') != 'ALL80_SOURCE_OWNERS_MATCHED_UNDER_PARAMETER_FQN_CONTRACT'
            or contract.get('matched_owner_count') != 80 or set(owners) != expected
            or contract.get('complete_named_port_source_owner_preconditions') != {'prefill': True, 'decode': True}
            or contract.get('conflicts') or contract.get('missing_owner_keys')):
        raise ValueError('all 80 unique source owners must match the explicit parameter FQN contract')
    bound = {}
    for label, probe in (('prefill', prefill_probe), ('decode', decode_probe)):
        for direction, count in (('inputs', 89), ('outputs', 82)):
            entries = probe.get(direction)
            if (not isinstance(entries, list) or len(entries) != count
                    or any(row.get('tensor_name_status') != 'OBSERVED'
                        or not isinstance(row.get('tensor_name'), str) or not row['tensor_name'] for row in entries)
                    or len({row['tensor_name'] for row in entries}) != count):
                raise ValueError(label + ' requires complete unique native names for all ' + direction)
    for label, job in (('prefill', prefill), ('decode', decode)):
        source, compiled, runtime = (job[key] for key in ('source_lineage', 'compiled_lineage', 'runtime_bindings'))
        if (source.get('observed_state_owner_count') != 80 or source.get('unique_state_owner_count') != 80
                or compiled.get('matched_count') != 80 or runtime.get('matched_count') != 80):
            raise ValueError(label + ' requires all 80 source, compiled FX and runtime name bindings')
        source_rows = _unique(source.get('states'), 'source_refdata', 80, label + ' source states')
        if set(source_rows) != lineage._STATES or any(row.get('status') != 'NAMED_PORT_STATE_LINEAGE_OBSERVED' for row in source_rows.values()):
            raise ValueError(label + ' source named-port owners are incomplete')
        rows = _unique(runtime.get('states'), 'source_refdata', 80, label + ' runtime states')
        if set(rows) != lineage._STATES or any(row.get('status') != 'EXPLICIT_TENSOR_NAME_AND_DESCRIPTOR_MATCHED' for row in rows.values()):
            raise ValueError(label + ' runtime state identities are incomplete')
        compiled_rows = _unique(compiled.get('states'), 'source_refdata', 80, label + ' compiled states')
        if any(row.get('status') != 'EXPLICIT_FX_TENSOR_IDENTITY_MATCHED' for row in compiled_rows.values()):
            raise ValueError(label + ' compiled state identities are incomplete')
        for field in ('runtime_input_index', 'runtime_output_index'):
            if len({row[field] for row in rows.values()}) != 80:
                raise ValueError(label + ' has duplicate state runtime indices')
        bound[label] = rows
    outputs = _unique(retained_outputs, 'runtime_index', 82, 'retained prefill output slots')
    if set(outputs) != set(range(82)):
        raise ValueError('retained prefill output slots must be the complete literal native index set')
    pi, po = _runtime_keys(prefill_probe, 'inputs'), _runtime_keys(prefill_probe, 'outputs')
    di, do = _runtime_keys(decode_probe, 'inputs'), _runtime_keys(decode_probe, 'outputs')
    result = []
    for key in sorted(owners, key=lambda value: (owners[value]['layer'], owners[value]['cache_role'])):
        owner = owners[key]
        if owner.get('status') != 'SOURCE_OWNER_MATCHED_UNDER_PARAMETER_FQN_CONTRACT':
            raise ValueError('parameter owner row is not matched')
        a, b = bound['prefill'][owner['prefill_refdata']], bound['decode'][owner['decode_refdata']]
        views = [_nd_view_key(values[row[field]]) for values, row, field in
            ((pi, a, 'runtime_input_index'), (po, a, 'runtime_output_index'),
             (di, b, 'runtime_input_index'), (do, b, 'runtime_output_index'))]
        if any(value != views[0] for value in views) or views[0][0] != 'FLOAT16':
            raise ValueError('cross-model state physical FP16 views differ')
        actual = outputs[a['runtime_output_index']]
        if (actual.get('tensor_name') != a['runtime_output_name']
                or _nd_view_key((actual.get('dtype'), tuple(actual.get('shape', [])))) != views[0]):
            raise ValueError('prefill retained file slot does not bind its exact native state name/view')
        file = actual['actual_file']
        if file.get('bytes') != 2 * math.prod(views[0][1]) or not Path(file['path']).is_absolute():
            raise ValueError('prefill state file extent or absolute source path differs')
        result.append({'owner_key': key, 'layer': owner['layer'], 'cache_role': owner['cache_role'],
            'source_owner_contract': CONTRACT,
            'prefill': deepcopy(a), 'decode': deepcopy(b),
            'output_file': deepcopy(file),
            'decode_runtime_input_index': b['runtime_input_index'], 'decode_runtime_input_name': b['runtime_input_name'],
            'dtype': 'FLOAT16', 'shape_without_edge_singletons': list(views[0][1]), 'bytes': file['bytes'],
            'status': 'PARAMETER_CONTRACT_SOURCE_COMPILED_NATIVE_FILE_JOINED'})
    for key, values in (('prefill refs', [row['prefill']['source_refdata'] for row in result]),
                        ('decode refs', [row['decode']['source_refdata'] for row in result]),
                        ('source files', [row['output_file']['path'] for row in result])):
        if len(set(values)) != 80:
            raise ValueError('feedback mapping is not bijective: ' + key)
    return result


def _seed_observation(directory, execution, bridge, plan, bindings, snapshots):
    """Describe a separately requested output observation; never accept a seed."""
    sources = {Path(row['path']).name: row for row in bridge['original_source_inputs']}
    pos = sources['output_pos.bin']
    raw, identity = _read_small(pos['path'], 4)
    if len(raw) != 4 or identity['sha256'] != pos['sha256']:
        raise ValueError('executed prefill output_pos bytes changed')
    _pin(snapshots, pos['path'], pos['identity'])
    position = struct.unpack('<i', raw)[0]
    result = {'status': 'NOT_REQUESTED', 'seed_accepted': False, 'executed_output_pos': position,
        'selection_semantics': 'UNVERIFIED_PRIVATE_INDEX_OPERATOR', 'numerical_validation': 'NOT_RUN',
        'history_and_token_embedding_equivalence': 'UNVERIFIED',
        'next': 'Run inspect_target_prefill_outputs for the exact chosen prefill execution and supply its --prefill-output-inspection'}
    index = plan['logits_runtime_output_index']
    output = next(row for row in bridge['retained_outputs'] if row['runtime_index'] == index)
    file = output['actual_file']
    if directory is not None:
        root, _ = _safe(directory, directory=True)
        report = _json(root / 'report.json', bindings)
        _pin(snapshots, bindings[-1]['path'])
        if (report.get('artifact_kind') != 'qwen35-target-prefill-output-inventory'
                or report.get('status') != 'OUTPUT_FILES_BOUND_TO_LOGGED_SLOTS'
                or report.get('execution') != str(execution)
                or report.get('slot_binding') != 'EXPLICIT_COMPLETE_TOOL_LOG_AND_VERIFIED_BYTES'):
            raise ValueError('next-token observation must bind this exact successful prefill output inspection')
        claimed = report.get('bound_tensors', {}).get('logits', {})
        relative = Path(file['path']).relative_to(execution / 'board').as_posix()
        if (claimed.get('runtime_index') != index or claimed.get('path') != relative
                or any(claimed.get(key) != file[key] for key in ('sha256', 'bytes'))):
            raise ValueError('prefill logits observation differs from the freshly verified logged file')
        result['inspection'] = bindings[-1]
    desc = plan['outputs'][index]
    statistics = _numerical(file['path'], desc, file)
    _pin(snapshots, file['path'])
    result.update(status='BOUND_PREFILL_LOGITS_OBSERVED_UNVALIDATED',
        logits_file=deepcopy(file), runtime_index=index, statistics=statistics,
        trial_use=('SOURCE_PYTHON_LAST_ROW63_EXPERIMENT_ONLY' if position == 63
                   else 'NOT_AN_ACCEPTED_LAST_ROW_SEED'),
        next='Any use as an experimental decode seed must explicitly retain the unverified selection, input-history and numerical assumptions')
    return result


def prepare_prefill_decode_feedback(prefill_job, decode_job, prefill_execution, named_prefill_probe,
        decode_runtime_probe, output, *, prefill_source_torchair_pb2=None, decode_source_torchair_pb2=None,
        prefill_source_export_receipt=None, prefill_output_inspection=None):
    """Write an independent experimental mapping, with no prepared input files."""
    execution, _ = _safe(prefill_execution, directory=True)
    bindings, snapshots = [], {}
    initial_plan = _json(execution / 'plan.json', bindings)
    _pin(snapshots, execution / 'plan.json')
    out = output_path(output)
    roots = [execution, prefill_job, decode_job, named_prefill_probe, decode_runtime_probe, initial_plan['input_dir']]
    if prefill_output_inspection is not None:
        roots.append(prefill_output_inspection)
    for root in roots:
        root, _ = _safe(root, directory=True)
        if out == root or root in out.parents or out in root.parents:
            raise ValueError('feedback plan output must be separate from retained evidence, jobs and original inputs')
    out.mkdir(parents=True, exist_ok=False)
    result = {'schema_version': 1, 'artifact_kind': _KIND, 'status': 'BLOCKED', 'stage': 'validate-prefill-files',
        'output': str(out), 'device_execution': 'NOT_RUN', 'device_calls': 'NONE', 'model_compilation': 'NOT_RUN',
        'weight_payload_access': 'NONE', 'model_payload_access': 'BOUNDED_SMALL_AIR_OMC_AND_SCHEMA_ONLY',
        'state_feedback': 'NOT_RUN', 'state_continuity': 'NOT_VALIDATED', 'numerical_validation': 'NOT_RUN',
        'feedback_ready': False, 'decode_inputs_created': False, 'source_bindings': bindings}
    try:
        bridge = inspect_prefill_decode_bridge(execution, named_prefill_probe, out / 'prefill-file-validation')
        if bridge.get('status') != _BRIDGE or bridge.get('input_count') != 89 or bridge.get('output_count') != 82:
            raise ValueError('prefill feedback requires all 89 executed inputs and 82 logged files plus same-model named witness: ' + str(bridge.get('error', bridge.get('status'))))
        for row in bridge['active_source_identities']:
            _pin(snapshots, row['path'], row['identity'])
        result['stage'] = 'bind-source-compiled-runtime-graphs'
        jobs, documents, inventories, enums, probes = {}, {}, {}, {}, {}
        for name, job, seq, probe_path, pb2 in (
            ('prefill', prefill_job, 64, named_prefill_probe, prefill_source_torchair_pb2),
            ('decode', decode_job, 1, decode_runtime_probe, decode_source_torchair_pb2)):
            meta, source, compiled, domain = lineage.load_target_graph(job,
                'cpro-40block-prefill-64' if seq == 64 else 'cpro-40block-decode-1', bindings,
                source_torchair_pb2=pb2, source_export_receipt=prefill_source_export_receipt if seq == 64 else None)
            inventories[name], inventory_id = _bound_inventory(meta, bindings, seq)
            meta['source_lineage'] = lineage.analyze_state_lineage(source, meta['source_enum_fields'])
            meta['compiled_lineage'] = lineage._join_compiled(meta['source_lineage'], compiled, domain)
            probes[name] = lineage._probe(probe_path, meta, bindings)
            meta['runtime_bindings'] = lineage._runtime_bindings(meta['compiled_lineage'], probes[name], domain)
            jobs[name], documents[name], enums[name] = meta, source, domain
            result[name] = {key: deepcopy(meta[key]) for key in ('job', 'profile', 'air', 'model', 'source_schema_binding',
                'compilation_report_sha256', 'source_lineage', 'compiled_lineage', 'runtime_bindings')}
            result[name]['inventory'] = inventory_id
            for row in bindings:
                _, current = _pin(snapshots, row['path'])
                if [current[key] for key in ('device', 'inode', 'bytes', 'mtime_ns', 'ctime_ns')] != row['stat']:
                    raise ValueError('bound graph metadata changed after its fresh read')
        if any(probes['prefill'][key] != probes['decode'][key] for key in ('runtime_version', 'firmware_id', 'ddk_id')):
            raise ValueError('prefill and decode native metadata have different runtime identities')
        plan = _json(execution / 'plan.json', bindings)
        for key in ('compilation_report_sha256', 'model_sha256'):
            if plan.get(key) != jobs['prefill'][key]:
                raise ValueError('retained prefill execution differs from the bound prefill graph: ' + key)
        # The actual prefill smoke planner records `gear`; some earlier
        # fixtures/receipts also carried `sequence_length`. Require every
        # supplied declaration to agree, instead of rejecting the real schema
        # or allowing a contradictory alias to hide a different model length.
        lengths = [plan[key] for key in ('gear', 'sequence_length') if key in plan]
        if (plan.get('compilation') != str(Path(jobs['prefill']['job']) / 'compilation')
                or not lengths or any(type(value) is not int or value != 64 for value in lengths)):
            raise ValueError('retained prefill execution must bind the exact static64 job')
        result['stage'] = 'join-parameter-contract-state-files'
        correspondence = analyze_state_correspondence(documents['prefill'], documents['decode'], inventories['prefill'], inventories['decode'],
            jobs['prefill']['source_enum_fields'], jobs['decode']['source_enum_fields'])
        contract = correspondence['parameter_fqn_owner_contract']
        mapping = bind_prefill_decode_feedback(contract, jobs['prefill'], jobs['decode'], bridge['retained_outputs'], probes['prefill'], probes['decode'])
        for row in mapping:
            _, current = _pin(snapshots, row['output_file']['path'])
            row['output_file']['identity'] = current
        ids_path = Path(plan['input_dir']) / 'input_ids.bin'
        raw, ids_id = _read_small(ids_path, 256)
        if len(raw) != 256:
            raise ValueError('prefill prefix IDs must contain exactly 64 original INT32 words')
        ids_path, ids_full = _pin(snapshots, ids_path)
        if [ids_full[key] for key in ('device', 'inode', 'bytes', 'mtime_ns', 'ctime_ns')] != ids_id['stat']:
            raise ValueError('prefill prefix IDs changed after their bounded read')
        bindings.append(ids_id)
        prefix = {'length': 64, 'input_ids': list(struct.unpack('<64i', raw)), 'source': dict(ids_id, identity=ids_full),
            'scope': 'ORIGINAL_AUXILIARY_IDS_ONLY_NOT_A_90TH_TARGET_INPUT', 'token_to_embedding_equivalence': 'UNVERIFIED'}
        if any(value < 0 or value >= 248320 for value in prefix['input_ids']):
            raise ValueError('prefill prefix IDs are outside the declared vocabulary')
        seed = _seed_observation(prefill_output_inspection, execution, bridge, plan, bindings, snapshots)
        for row in bindings:
            _, current = _pin(snapshots, row['path'])
            if [current[key] for key in ('device', 'inode', 'bytes', 'mtime_ns', 'ctime_ns')] != row['stat']:
                raise ValueError('feedback source metadata changed during preparation')
        result.update(status=_STATUS, stage='complete', prefill_execution=str(execution),
            named_prefill_probe=str(Path(named_prefill_probe).absolute()), decode_runtime_probe=str(Path(decode_runtime_probe).absolute()),
            prefix=prefix, next_seed=seed, state_feedback_bindings=mapping, mapped_state_count=80,
            parameter_owner_contract={key: deepcopy(contract[key]) for key in ('contract', 'status', 'matched_owner_count',
                'tensor_numeric_dtype_and_extent', 'original_python_fqn_to_filename_attestation', 'mapping')},
            compiled_runtime_file_binding='ALL80_EXPERIMENTAL_PARAMETER_CONTRACT_GRAPH_NATIVE_FILE_JOINS',
            required_execution_assumptions=['PARAMETER_FQN_OWNER_CONTRACT_NOT_PYTHON_FQN_ATTESTATION',
                'RAW_PARAMETER_INTEGER_DTYPE_DOMAIN_UNRESOLVED', 'PREFILL_TO_DECODE_CONTROLS_AND_INITIAL_HISTORY_UNVERIFIED',
                'PREFIX_TOKEN_EMBEDDING_EQUIVALENCE_UNVERIFIED', 'PREFILL_NEXT_SEED_SELECTION_AND_NUMERICS_UNVALIDATED'],
            source_preservation='CURRENT_FULL9_AND_ALL_BOUND_SMALL_GRAPH_METADATA_OUTPUT_IDENTITIES_UNCHANGED',
            scope='Experimental 80-state filepath to decode-input mapping under an explicit parameter filename/named-port contract. No decode input copies, controls, accepted seed, state continuity, numerical pass or device execution.',
            next='Review the explicit experimental assumptions and selected new prefill logits observation before preparing a separate decode trial; this plan does not execute feedback')
        _stable(snapshots)
        candidate = out / '.manifest.pending.json'
        if len(json.dumps(result, ensure_ascii=False, allow_nan=False).encode()) > _MAX_PLAN:
            raise ValueError('feedback plan exceeds the 4 MiB bound')
        save_json(candidate, result)
        _stable(snapshots)
        # Publish only the new owned metadata file, without replacing an
        # existing destination or linking any source/state payload.
        os.link(candidate, out / 'manifest.json', follow_symlinks=False)
        candidate.unlink()
    except (OSError, ValueError, TypeError, KeyError, IndexError, StopIteration, RecursionError) as error:
        result.update(status='BLOCKED', error=f'{type(error).__name__}: {error}')
        save_json(out / 'manifest.json', result)
    return result
