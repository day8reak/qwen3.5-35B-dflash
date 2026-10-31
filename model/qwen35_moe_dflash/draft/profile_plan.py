"""Bind an existing S64 real-feature fixture to the observed vendor tool CLI."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re

from qwen35_moe_dflash.draft.real_feature_draft import validate_real_feature_draft
from qwen35_moe_dflash.target.source_inventory import _read_small
from qwen35_moe_dflash.target.board_tool_inspect import _tool_path


KIND = 'qwen35-draft-operator-profile'
FLAGS = ('model', 'input', 'output_dir', 'modelmanager_itf', 'enable_item', 'times',
         'input_type', 'output_type', 'dev_perf', 'priority', 'async')


def bind_profile(prepared, inspection, *, refresh_output=None):
    root = Path(prepared).absolute()
    rebound = None
    if refresh_output is not None:
        from qwen35_moe_dflash.draft.profile_inputs import refresh_profile_inputs
        root, manifest, rebound = refresh_profile_inputs(root, refresh_output)
    else:
        manifest = validate_real_feature_draft(root)
    path = Path(inspection)
    if path.is_dir():
        path /= 'report.json'
    raw, identity = _read_small(path, 1 << 20)
    tool = json.loads(raw)
    help_row = tool.get('help', {})
    help_text = help_row.get('text', '')
    if (tool.get('status') != 'INSPECTED' or tool.get('artifact_kind') != 'qwen35-target-board-tool-inspection'
            or tool.get('tool_identity') != 'SHA256_MATCHED_BEFORE_AND_AFTER'
            or not re.fullmatch('[0-9a-f]{64}', str(tool.get('tool_sha256_before', '')))
            or tool.get('tool_sha256_before') != tool.get('tool_sha256_after')
            or tool.get('firmware', {}).get('text', '').strip() != manifest['firmware_id'].strip()
            or help_row.get('observation') != 'HELP_TEXT_OBSERVED' or help_row.get('truncated') is not False
            or help_row.get('utf8_sha256') != hashlib.sha256(help_text.encode()).hexdigest()
            or help_row.get('utf8_bytes') != len(help_text.encode())
            or any(not re.search(r'--' + flag + r'\b', help_text) for flag in FLAGS)
            or not re.search(r'V1\s*;\s*V2', help_text)
            or not re.search(r'1\s*:\s*Enable profiling', help_text, re.I)):
        raise ValueError('selected tool help/hash/firmware does not establish the V1 profiling CLI')
    inputs = []
    for index, row in enumerate(manifest['inputs']):
        if row['index'] != index:
            raise ValueError('ordered prepared inputs are required')
        _, record = _read_small(root / row['path'], 3 << 20)
        if any(record[k] != row[k] for k in ('bytes', 'sha256')):
            raise ValueError('prepared input changed before profiling')
        inputs.append(dict(record, index=index, dtype=row['dtype'], shape=row['shape']))
    _, prepared_record = _read_small(root / 'manifest.json', 4 << 20)
    return {'schema_version': 1, 'artifact_kind': KIND, 'prepared': prepared_record,
        'input_revalidation': rebound,
        'tool': {'path': _tool_path(tool['tool']), 'sha256': tool['tool_sha256_before'], 'inspection': identity},
        'model': manifest['model_source'], 'inputs': inputs, 'outputs': manifest['outputs'],
        'ddk_id': manifest['ddk_id'], 'firmware_id': manifest['firmware_id'],
        'source_runtime_version': manifest['runtime_version'],
        'input_scope': 'RETAINED_REAL_PREFILL64_FEATURES_NOT_EXACT_18_TOKEN_RESIDENT_CALL',
        'input_provenance': {'features': manifest['feature_provenance']['source'],
            'tokens': manifest['token_provenance'], 'mask': manifest['mask_provenance']},
        'model_compilation': 'NOT_RUN', 'numerical_validation': 'NOT_EVALUATED',
        'model_manager_interface': 'V1', 'times': 1, 'warmup': 0,
        'timing_scope': 'ONE_FRESH_VENDOR_TOOL_PROFILED_INFERENCE_NOT_RESIDENT_LATENCY_DISTRIBUTION',
        'placement': 'LITERAL_CSV_DEVICE_LABELS_ONLY',
        'control_contract': 'USER_OBSERVED_INTERNAL_TOOL_HELP_ENABLE_ITEM_1',
        'public_example': 'https://developer.huawei.com/consumer/jp/doc/hiai-Guides/examples-0000001053605660',
        'public_tool_scope': 'PUBLIC_DOC_EXCLUDES_HARMONYOS_NEXT; SELECTED_INTERNAL_TOOL_REQUIRES_ACTUAL_ARTIFACTS'}


def model_argv(plan, board, model):
    board = PurePosixPath(board)
    return [plan['tool']['path'], '--model=' + str(model),
        '--input=' + ','.join(str(board / f'input_{i}.bin') for i in range(len(plan['inputs']))),
        '--output_dir=' + str(board / 'outputs'), '--modelmanager_itf=V1', '--enable_item=1',
        '--times=1', '--async=false', '--dev_perf=3', '--priority=7',
        '--input_type=' + ','.join(_tool_dtype(r['dtype']) for r in plan['inputs']),
        '--output_type=' + ','.join(_tool_dtype(r['dtype']) for r in plan['outputs'])]


def _tool_dtype(dtype):
    return {'float16': 'FP16', 'int32': 'INT32', 'float32': 'FP32'}[dtype]


def check_model_help(text):
    if (any(not re.search(r'--'+flag+r'\b', text) for flag in FLAGS)
            or not re.search(r'V1\s*;\s*V2', text)
            or not re.search(r'1\s*:\s*Enable profiling', text, re.I)):
        raise ValueError('selected model tool help does not establish the V1 profiling CLI')


def bind_accuracy_profile(accuracy, model_tool):
    from qwen35_moe_dflash.common.hashing import sha256_file
    from qwen35_moe_dflash.execution.runtime_config import read_json
    from qwen35_moe_dflash.draft.incremental.runtime import validate_abi
    from qwen35_moe_dflash.draft.incremental.accuracy import KIND as ACCURACY_KIND
    root = Path(accuracy).resolve()
    result = read_json(root/'result.json', 4 << 20)
    if (result.get('artifact_kind') != ACCURACY_KIND or result.get('status') != 'PASSED'
            or result.get('numerical_validation') != 'PASSED'
            or result.get('device_execution') != 'EXECUTED_OUTPUTS_RETRIEVED'):
        raise ValueError('profiling requires a passed standalone incremental accuracy result')
    fixture = Path(result['fixture'])
    if sha256_file(fixture/'manifest.json') != result['fixture_sha256']:
        raise ValueError('accuracy fixture changed after numerical comparison')
    meta = read_json(fixture/'manifest.json', 4 << 20)
    validate_abi(meta)
    if meta['model'] != result['model']:
        raise ValueError('accuracy model and fixture differ')
    step = meta['cases'][meta['profile_case']]
    if step['proposals'] < 1:
        raise ValueError('profile case must contain active proposals')
    inputs = []
    for spec in meta['inputs']:
        row = next(r for r in meta['input_files'] if r['case']==step['index'] and r['index']==spec['index'])
        relative = Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('profile input must be a fixture member')
        _, observed = _read_small(fixture/relative, 3 << 20)
        if any(observed[k] != row[k] for k in ('bytes','sha256')):
            raise ValueError('profile input changed since accuracy execution')
        inputs.append(dict(observed,index=spec['index'],dtype=spec['dtype'],shape=spec['shape']))
    return dict(schema_version=1,artifact_kind=KIND,
        tool={'path':_tool_path(model_tool),'sha256':None}, model=meta['model'],
        inputs=inputs,outputs=meta['outputs'],ddk_id=meta['ddk_id'],
        firmware_id=result['runtime']['firmware_id'],source_runtime_version=result['runtime']['runtime_version'],
        input_scope='PASSED_INCREMENTAL_ACCURACY_FIXTURE_WITH_REFERENCE_KV',
        input_provenance={'accuracy':str(root),'case':step},
        model_compilation='NOT_RUN',numerical_validation='PASSED_STANDALONE_REFERENCE_CASES',
        model_manager_interface='V1',times=1,warmup=0,
        timing_scope='ONE_FRESH_VENDOR_TOOL_PROFILED_INFERENCE_NOT_RESIDENT_LATENCY_DISTRIBUTION')
