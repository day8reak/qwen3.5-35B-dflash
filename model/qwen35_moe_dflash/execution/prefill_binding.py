"""Bind one experimental padded prefill to the autonomous serial verifier.

The historical prefill supplies graph/ABI evidence only. Every new request
generates its own original-table inputs and zero initial states on board.
Padding controls are a declared experiment, never a numerical acceptance.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path, PurePosixPath
import re

from qwen35_moe_dflash.execution.original_embedding_lookup import _current
from qwen35_moe_dflash.common.file_io import _read_small
from qwen35_moe_dflash.target.prefill_decode_feedback import prepare_prefill_decode_feedback, _STATUS
from qwen35_moe_dflash.target.prefill_smoke_plan import _bound_probe, _tool_binding, _runtime_keys, _nd_view_key, _ge_key, _WIDTHS, _TOOL_TYPES


RECIPE = 'PREFILL64_PADDED_SHORT_PROMPT_EXPERIMENTAL'


def control_assumption(length):
    if type(length) is not int or not 1 <= length <= 63:
        raise ValueError('padded prefill requires a logical prompt of 1..63 tokens')
    return {'id': RECIPE, 'physical_sequence_length': 64, 'valid_prompt_length': length,
        'numerical_validation': 'NOT_RUN', 'state_valid_length_semantics': 'UNVERIFIED',
        'initial_states': 'ALL80_FP16_ZERO',
        'embedding_rows': 'ORIGINAL_INT8_TABLE_AND_SEPARATE_FP32_SCALE_FOR_VALID_PROMPT',
        'padding_embedding': 'ZERO_INT8_ROW_AND_ZERO_FP32_SCALE',
        'position_ids': list(range(64)), 'new_kv_cache_pos': list(range(64)),
        'output_pos': [length - 1], 'allQLen': [length], 'effective_length': [length],
        'conv_state_pos': [length] + [0] * 127,
        'attention_mask': 'VALID_QUERY_Q_ALLOWS_KEYS_0_THROUGH_Q; PAD_QUERY_ALLOWS_KEY0_ONLY; OTHERS_NEGATIVE_INFINITY',
        'padding_kv_policy': 'PHYSICAL_FUTURE_SLOTS_NOT_LOGICAL_HISTORY; MASK_UNTIL_OVERWRITTEN_BY_COMMITTED_DECODE',
        'feature_selection': 'FIRST_VALID_PROMPT_ROWS_FROM_ACTUAL_PREFILL_OUTPUT',
        'scope': 'Explicit controls for an execution experiment; private convolution/recurrent padding semantics remain unvalidated'}


def payload_inventory(compiled, compilation):
    """Retain exact compiler declarations; upload performs its own fresh check."""
    rows = []
    for row in compiled.get('generated_artifacts', []):
        relative = row.get('path', '')
        if (not isinstance(relative, str) or not relative.startswith('compile-work/generated/')
                or any(part in {'', '.', '..'} for part in relative.split('/'))
                or not re.fullmatch(r'[A-Za-z0-9_./-]+', relative)
                or type(row.get('bytes')) is not int or row['bytes'] <= 0
                or re.fullmatch(r'[0-9a-f]{64}', str(row.get('sha256'))) is None):
            raise ValueError('prefill compiler payload declaration is invalid')
        rows.append({'local': str(Path(compilation) / relative),
            'remote': str(PurePosixPath(relative).relative_to('compile-work/generated')),
            'bytes': row['bytes'], 'sha256': row['sha256'], 'record': deepcopy(row),
            'primary': row == compiled.get('omc')})
    if (not 1 <= len(rows) <= 8192 or sum(row['primary'] for row in rows) != 1
            or len({row['remote'] for row in rows}) != len(rows)):
        raise ValueError('prefill inventory needs exactly one primary OMC and unique paths')
    return rows


def _named_abi(contract, native):
    compiled_inputs = contract.get('boundaries', {}).get('inputs_in_serialized_op_order', [])
    by_name = {row.get('name'): row for row in compiled_inputs}
    if len(compiled_inputs) != 89 or len(by_name) != 89:
        raise ValueError('prefill requires all 89 unique compiled input names')
    result = {}
    for direction, count in (('inputs', 89), ('outputs', 82)):
        keys = _runtime_keys(native, direction)
        entries = native.get(direction, [])
        if len(entries) != count or len(keys) != count:
            raise ValueError('prefill native descriptor count differs')
        rows, names = [], set()
        for index, (entry, (dtype, shape)) in enumerate(zip(entries, keys)):
            name = entry.get('tensor_name')
            if (type(entry.get('index')) is not int or entry['index'] != index
                    or entry.get('tensor_name_status') != 'OBSERVED'
                    or not isinstance(name, str) or not name or name in names):
                raise ValueError('prefill requires complete distinct observed runtime names')
            names.add(name)
            if direction == 'inputs' and (name not in by_name
                    or _nd_view_key(_ge_key(by_name[name])) != _nd_view_key((dtype, shape))):
                raise ValueError('prefill runtime input name/view differs from its compiled boundary')
            row = {'index': index, 'name': name, 'runtime_tensor_name': name,
                'dtype': dtype, 'shape': list(shape), 'bytes': math.prod(shape) * _WIDTHS[dtype],
                'raw_dtype': entry['raw_dtype'], 'raw_format': entry['raw_format']}
            row['tool_input_type' if direction == 'inputs' else 'tool_output_type'] = _TOOL_TYPES[dtype]
            rows.append(row)
        result[direction] = rows
    if (_nd_view_key((result['outputs'][0]['dtype'], tuple(result['outputs'][0]['shape'])))
                != ('FLOAT32', (248320,))
            or _nd_view_key((result['outputs'][81]['dtype'], tuple(result['outputs'][81]['shape'])))
                != ('FLOAT16', (64, 16384))):
        raise ValueError('prefill complete logits/features must bind native slots 0/81')
    return result


def _mapping_signature(rows):
    return sorted((row['owner_key'], row['bytes'], row['prefill']['source_refdata'],
        row['prefill']['runtime_input_index'], row['prefill']['runtime_output_index'],
        row['prefill']['runtime_input_name'], row['prefill']['runtime_output_name'],
        row['decode']['source_refdata'], row['decode']['runtime_input_index'],
        row['decode']['runtime_output_index'], row['decode']['runtime_input_name'],
        row['decode']['runtime_output_name']) for row in rows)


def prepare_prefill_binding(directory, target_binding, source_torchair_pb2, prompt_length, output):
    """Recompute graph/ABI joins; never import old output bytes as new state."""
    root = Path(directory).resolve()
    raw, receipt = _read_small(root / 'manifest.json', 4 << 20)
    requested = json.loads(raw)
    if (requested.get('status') != _STATUS or requested.get('mapped_state_count') != 80
            or requested.get('artifact_kind') != 'qwen35-target-prefill-decode-feedback-plan'):
        raise ValueError('prefill integration requires a completed all80 feedback mapping plan')
    decode = target_binding['base_plan']
    if (Path(requested['decode']['job']).resolve() != Path(decode['compilation']).parent.resolve()
            or Path(requested['decode_runtime_probe']).resolve() != Path(decode['runtime_probe']).resolve()
            or requested['decode']['model']['sha256'] != decode['model_sha256']
            or requested['decode']['compilation_report_sha256'] != decode['compilation_report_sha256']):
        raise ValueError('prefill mapping targets a different decode model or runtime probe')
    # This host-only regeneration checks the actual graphs, original metadata
    # and retained slot witnesses again. A saved success label alone is not used.
    fresh = prepare_prefill_decode_feedback(requested['prefill']['job'], requested['decode']['job'],
        requested['prefill_execution'], requested['named_prefill_probe'], requested['decode_runtime_probe'],
        Path(output) / 'mapping', prefill_source_torchair_pb2=source_torchair_pb2,
        decode_source_torchair_pb2=source_torchair_pb2)
    if fresh.get('status') != _STATUS or fresh.get('mapped_state_count') != 80:
        raise ValueError('fresh prefill state mapping failed: ' + str(fresh.get('error', fresh.get('status'))))
    if _mapping_signature(fresh['state_feedback_bindings']) != _mapping_signature(requested['state_feedback_bindings']):
        raise ValueError('fresh prefill/decode state ownership differs from the requested mapping')
    compilation = Path(fresh['prefill']['job']) / 'compilation'
    bindings = []
    compiled, contract, probe, native, model = _bound_probe(compilation, Path(fresh['named_prefill_probe']), bindings)
    tool = _tool_binding(probe, bindings)
    if (any(probe[key] != decode[key] for key in ('runtime_version', 'firmware_id', 'ddk_id'))
            or any(tool.get(key) != decode['tool_binding'].get(key) for key in ('path', 'sha256'))
            or model['sha256'] != fresh['prefill']['model']['sha256']
            or probe['compilation_report_sha256'] != fresh['prefill']['compilation_report_sha256']):
        raise ValueError('prefill/decode runtime, tool or newly bound model identity differs')
    abi = _named_abi(contract, native)
    mapping = fresh['state_feedback_bindings']
    target_states = {row['source_refdata']: row for row in target_binding['state_mapping']}
    for row in mapping:
        old = row['decode']; current = target_states.get(old['source_refdata'], {})
        if any(current.get(key) != old.get(key) for key in ('runtime_input_index', 'runtime_output_index')):
            raise ValueError('cross-model state mapping differs from the active decode state map')
        for direction, key, name_key in (('inputs', 'runtime_input_index', 'runtime_input_name'),
                                        ('outputs', 'runtime_output_index', 'runtime_output_name')):
            desc = abi[direction][row['prefill'][key]]
            if desc['name'] != row['prefill'][name_key] or desc['bytes'] != row['bytes'] or desc['dtype'] != 'FLOAT16':
                raise ValueError('prefill state owner differs from the freshly named ABI')
    if len(target_states) != 80 or {row['decode']['source_refdata'] for row in mapping} != set(target_states):
        raise ValueError('cross-model mapping must cover every decode state exactly once')
    base = {'compilation': str(compilation), 'runtime_probe': fresh['named_prefill_probe'],
        'gear': 64, 'sequence_length': 64, 'profile': 'cpro-40block-prefill-64',
        'compilation_report_sha256': probe['compilation_report_sha256'], 'model_sha256': model['sha256'],
        'primary_model_binding': model, 'tool_binding': tool, **abi,
        **{key: probe[key] for key in ('runtime_version', 'firmware_id', 'ddk_id')}}
    pins = {}
    for record in [receipt, *bindings, *fresh['source_bindings']]:
        path, identity = _current(record['path'])
        if [identity[key] for key in ('device', 'inode', 'bytes', 'mtime_ns', 'ctime_ns')] != record['stat']:
            raise ValueError('prefill source changed during fresh binding: ' + str(path))
        pins[str(path)] = identity
    path, identity = _current(model['path'])
    if identity != model['identity']:
        raise ValueError('prefill primary changed after fresh model hash')
    pins[str(path)] = identity
    return {'model': model, 'compiled': compiled, 'payloads': payload_inventory(compiled, compilation),
        'base_plan': base, 'state_mapping': deepcopy(mapping), 'control_recipe': RECIPE,
        'experimental_control_assumption': control_assumption(prompt_length),
        'mapping_receipt': receipt, 'mapping_recheck': str(Path(output) / 'mapping/manifest.json'),
        'source_bindings': [{'path': path, 'identity': identity} for path, identity in sorted(pins.items())],
        'old_prefill_output_use': 'GRAPH_ABI_WITNESS_ONLY_NO_REUSE_AS_NEW_REQUEST_STATES',
        'model_compilation': 'NOT_RUN', 'device_execution': 'NOT_RUN', 'numerical_validation': 'NOT_RUN'}
