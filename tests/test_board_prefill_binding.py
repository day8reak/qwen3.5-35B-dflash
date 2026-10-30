"""Cross-model preparation uses fresh named ABI evidence, never old state bytes."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from qwen35_moe_dflash import board_prefill_binding as binding
from qwen35_moe_dflash import board_serial_dflash as board
from qwen35_moe_dflash.target_compile_failure import _read_small
from test_target_prefill_smoke_plan import case, rewrite_native, ENUMS, save


@pytest.fixture
def pair(case, tmp_path, monkeypatch):
    monkeypatch.setenv('AI_RUN_DIR', str(tmp_path))
    inputs = case['contract']['boundaries']['inputs_in_serialized_op_order']
    for desc, original in zip(case['native']['inputs'], reversed(inputs)):
        desc.update(tensor_name=original['name'], tensor_name_status='OBSERVED')
    outputs = case['contract']['boundaries']['outputs']
    ordered = [outputs[0], *reversed(outputs[1:81]), outputs[81]]
    case['native']['outputs'] = [dict(index=index, shape=row['shape'],
        raw_dtype=ENUMS['FLOAT32' if index == 0 else 'FLOAT16'], raw_format=211,
        tensor_name=f'prefill-runtime-output-{index}', tensor_name_status='OBSERVED')
        for index, row in enumerate(ordered)]
    rewrite_native(case)
    decode_job = tmp_path / 'decode-job'; decode_probe = tmp_path / 'decode-probe'
    decode_states, mapping = [], []
    for i in range(80):
        ref = f'RefData_{i}'
        native_input = next(row for row in case['native']['inputs'] if row['tensor_name'] == ref)
        native_output = case['native']['outputs'][80 - i]
        decode_ref = f'RefData_{(i + 13) % 80}'
        target = dict(source_refdata=decode_ref, runtime_input_index=(i + 13) % 80,
            runtime_output_index=(i + 17) % 80 + 1, runtime_input_name=decode_ref,
            runtime_output_name=f'decode-output-{i}')
        decode_states.append(target)
        size = 2
        for extent in native_output['shape']: size *= extent
        mapping.append(dict(owner_key=f'fixture-owner-{i}', bytes=size,
            prefill=dict(source_refdata=ref, runtime_input_index=native_input['index'],
                runtime_output_index=native_output['index'], runtime_input_name=ref,
                runtime_output_name=native_output['tensor_name']), decode=deepcopy(target)))
    report = dict(artifact_kind='qwen35-target-prefill-decode-feedback-plan', status=binding._STATUS,
        mapped_state_count=80, state_feedback_bindings=mapping, source_bindings=[],
        prefill=dict(job=str(case['job']), model=dict(sha256=case['compiled']['omc']['sha256']),
            compilation_report_sha256=case['outer']['compilation_report_sha256']),
        decode=dict(job=str(decode_job), model=dict(sha256='d' * 64), compilation_report_sha256='e' * 64),
        prefill_execution=str(tmp_path / 'old-execution'), named_prefill_probe=str(case['probe']),
        decode_runtime_probe=str(decode_probe))
    path = tmp_path / 'feedback/manifest.json'; save(path, report)
    old_states = tmp_path / 'old-execution/outputs'
    old_states.mkdir(parents=True)
    sentinel = old_states / 'must-not-copy-state.bin'; sentinel.write_bytes(b'old unrelated state bytes')
    calls = []
    def refresh(*args, **kwargs):
        calls.append((args, kwargs))
        return deepcopy(report)
    monkeypatch.setattr(binding, 'prepare_prefill_decode_feedback', refresh)
    base = dict(compilation=str(decode_job / 'compilation'), runtime_probe=str(decode_probe),
        model_sha256='d' * 64, compilation_report_sha256='e' * 64,
        runtime_version=case['outer']['runtime_version'], firmware_id=case['outer']['firmware_id'],
        ddk_id=case['outer']['ddk_id'], tool_binding=dict(
            path='/data/local/tmp/tools_sysdbg/model_run_tool_internal', sha256='a' * 64))
    return dict(case=case, root=tmp_path, report=report, receipt=path,
        target=dict(base_plan=base, state_mapping=decode_states), calls=calls, sentinel=sentinel)


def prepare(pair):
    return binding.prepare_prefill_binding(pair['receipt'].parent, pair['target'],
        pair['root'] / 'schema.py', 8, pair['root'] / 'new-preparation')


def test_reordered_named_prefill_mapping_and_original_models_without_old_state_copy(pair, monkeypatch):
    old = pair['sentinel'].read_bytes()
    original = Path.open
    def guarded(path, *args, **kwargs):
        assert path != pair['sentinel'], 'old states cannot seed a new board request'
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)
    result = prepare(pair)
    assert len(pair['calls']) == 1
    assert result['base_plan']['gear'] == 64
    assert result['base_plan']['inputs'][0]['name'] == 'arg18_1'
    row = result['state_mapping'][0]
    assert row['prefill']['source_refdata'] == 'RefData_0'
    assert row['decode']['source_refdata'] == 'RefData_13'
    assert row['prefill']['runtime_output_index'] == 80
    assert result['model']['path'] == str(pair['case']['model'])
    assert result['control_recipe'] == binding.RECIPE
    assert result['experimental_control_assumption']['conv_state_pos'] == [8] + [0] * 127
    assert result['experimental_control_assumption']['output_pos'] == [7]
    assert result['experimental_control_assumption']['new_kv_cache_pos'] == list(range(64))
    assert result['old_prefill_output_use'].endswith('NO_REUSE_AS_NEW_REQUEST_STATES')
    assert not (pair['root'] / 'new-preparation/inputs').exists()
    monkeypatch.setattr(Path, 'open', original)
    assert pair['sentinel'].read_bytes() == old


@pytest.mark.parametrize('fault', ['different_decode', 'partial', 'changed_mapping',
    'different_active_map', 'unnamed_input', 'renamed_input', 'wrong_extent', 'changed_model'])
def test_rejects_stale_or_ambiguous_cross_variant_evidence(pair, fault):
    if fault == 'different_decode':
        pair['target']['base_plan']['model_sha256'] = 'c' * 64
    elif fault == 'partial':
        pair['report']['mapped_state_count'] = 79
    elif fault == 'changed_mapping':
        pair['report']['state_feedback_bindings'][0]['prefill']['runtime_output_index'] = 1
    elif fault == 'different_active_map':
        pair['target']['state_mapping'][0]['runtime_output_index'] = 1
    elif fault in {'unnamed_input', 'renamed_input', 'wrong_extent'}:
        desc = pair['case']['native']['inputs'][0]
        if fault == 'unnamed_input': desc.pop('tensor_name_status')
        elif fault == 'renamed_input': desc['tensor_name'] = 'wrong_source_input'
        else: desc['shape'] = [2]
        rewrite_native(pair['case'])
    else:
        pair['case']['model'].write_bytes(b'changed-primary')
    with pytest.raises((ValueError, KeyError)):
        prepare(pair)


def test_complete_prefill_wire_contract_is_optional_and_pins_explicit_controls(pair):
    prefill = prepare(pair)
    rec = dict(bytes=1, sha256='a' * 64)
    manifest = dict(target_model=rec, draft=dict(model_source=rec, abi=rec),
        tables=dict(weight=rec, scale=rec), prompt_token_ids=list(range(8)),
        max_new_tokens=8, max_proposals_per_round=2, minimum_draft_rounds=2, timeout_seconds=120,
        target_binding=dict(base_plan=dict(runtime_version='v', ddk_id='d', firmware_id='f', inputs=[], outputs=[]),
            state_mapping=[]), prefill=prefill)
    paths = {key: '/board/' + key for key in ('target_model', 'prefill_model', 'draft_model', 'draft_abi',
        'target_tool', 'draft_runner', 'weight_table', 'scale_table')}
    paths.update(target_tool_record=rec, draft_runner_record=rec,
        target_library_path='/system/lib64', draft_library_path='/board/lib')
    wire = board.make_config(manifest, paths)
    assert b'prefill_model\t/board/prefill_model\t' in wire
    assert wire.count(b'\nprefill_input\t') == 89 and wire.count(b'\nprefill_output\t') == 82
    row = prefill['state_mapping'][0]
    exact = f"prefill_state\tRefData_13\tRefData_0\t{row['prefill']['runtime_input_index']}\t80\t{row['bytes']}\n".encode()
    assert exact in wire
    prefill['experimental_control_assumption']['conv_state_pos'][0] = 64
    with pytest.raises(ValueError, match='controls differ'):
        board.make_config(manifest, paths)


@pytest.mark.parametrize('length', [0, 64, True, 8.0])
def test_no_silent_full64_prefix_plus_anchor_or_implicit_logical_length(length):
    with pytest.raises(ValueError): binding.control_assumption(length)
