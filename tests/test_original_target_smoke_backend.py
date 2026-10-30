"""Real extent/raw-row host fixtures exercise original serial target evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from qwen35_moe_dflash import original_target_smoke_backend as backend
from qwen35_moe_dflash import original_target_board as board
from qwen35_moe_dflash.incremental_smoke import (SmokeContract, run_incremental_smoke,
    feature_sha256, prefix_sha256)
from qwen35_moe_dflash.original_embedding_lookup import OriginalEmbeddingLookup
from qwen35_moe_dflash.target_decode_smoke_plan import prepare_decode_smoke_plan
from qwen35_moe_dflash.target_compile_failure import _read_small
from test_target_decode_smoke_execution import case as decode_case


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value))
    return _read_small(path, 8 << 20)[1]


def record(path):
    return {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def case(tmp_path, monkeypatch, request):
    value = decode_case.__wrapped__(tmp_path, monkeypatch)
    large_report = getattr(request, 'param', False)
    if large_report:
        from test_target_prefill_smoke_plan import binding, rewrite_native
        path = value['compilation'] / 'report.json'
        raw = json.dumps(value['compiled']).encode()
        path.write_bytes(raw + b' ' * (10_306_827 - len(raw)))
        report_sha = record(path)['sha256']
        outer = json.loads((value['job'] / 'result.json').read_text())
        outer['compilation_report_sha256'] = report_sha; save(value['job'] / 'result.json', outer)
        value['contract']['compilation_report_sha256'] = report_sha
        save(value['job'] / 'target-contract/report.json', value['contract'])
        value['outer']['compilation_report_sha256'] = report_sha
        value['outer']['source_bindings'] = [binding(path) for path in (value['job'] / 'result.json',
            value['compilation'] / 'report.json', value['job'] / 'compile-profile.json',
            value['job'] / 'target-contract/report.json', value['help_report'])]
        rewrite_native(value)
    base = prepare_decode_smoke_plan(value['compilation'], value['probe'], value['source'])
    old = tmp_path / 'old-unrelated-decode'; base_id = save(old / 'plan.json', base)
    mapping = []
    for row in base['inputs']:
        if not row['name'].startswith('RefData_'): continue
        state_id = int(row['name'].split('_')[1])
        output = next(item for item in base['outputs'] if item['runtime_tensor_name'] == f'output_{state_id+1}_fixture')
        mapping.append({'source_refdata': row['name'], 'runtime_input_index': row['index'],
            'runtime_input_name': row['runtime_tensor_name'], 'runtime_output_index': output['index'],
            'runtime_output_name': output['runtime_tensor_name'], 'status': 'SAME_MODEL_STATE_FILE_BOUND',
            'output_file': {'path': '/never-use-old-state-history', 'sha256': 'f' * 64}})
    lineage_dir = tmp_path / 'lineage'
    lineage_id = save(lineage_dir / 'report.json', {'fixture': 'validated same-model mapping only'})
    current = {'status': 'ALL80_SAME_MODEL_STATE_BINDINGS_OBSERVED', 'execution': str(old),
        'source_bindings': [base_id], 'saved_report_identity': lineage_id,
        'decode': {'job': str(value['job']), 'model_sha256': base['model_sha256'],
            'state_file_mapping': {'states': mapping}}}
    monkeypatch.setattr(backend, 'validate_target_decode_state_lineage', lambda path: deepcopy(current))
    if not large_report:
        monkeypatch.setattr(backend, 'validate_decode_smoke_plan', lambda plan: deepcopy(plan))
    weight, scale = tmp_path / 'original-embedding-table', tmp_path / 'original-scale-table'
    with weight.open('wb') as stream:
        stream.truncate(248320 * 2048)
        for token in range(40): stream.seek(token * 2048); stream.write(bytes([token]) * 2048)
    with scale.open('wb') as stream:
        stream.truncate(248320 * 4)
        for token in range(40): stream.seek(token * 4); stream.write(struct.pack('<f', (token + 1) / 1000))
    lookup = OriginalEmbeddingLookup(weight, scale)
    value.update(base=base, mapping=mapping, lineage=lineage_dir, lookup=lookup, current=current,
        output=tmp_path / 'serial-target', calls=[], fail_at=None, mutate_output=False)

    def execute(compilation, contract, output, **options):
        prepared = options['runtime_smoke_backend_prepared']
        plan = backend.validate_smoke_backend_step(prepared)
        assert Path(compilation) == value['compilation'] and options['target_features'] is True and options['gear'] == 1
        assert 'runtime_decode_probe' not in options and 'runtime_continuation_prepared' not in options
        assert len(plan['inputs']) == len({row['source_filename'] for row in plan['inputs']}) == 89
        if value['fail_at'] == len(value['calls']):
            failed = {'status': 'FAILED', 'stage': 'retrieve-outputs', 'remote_exitcode': 0,
                'error': 'fixture HDC disconnected while retrieving output10'}
            save(Path(output) / 'result.json', failed)
            return failed
        raw = json.loads((Path(prepared) / 'manifest.json').read_text())
        token = raw['token_id']; position = len(raw['prefix_before'])
        by_name = {row['name']: row for row in plan['inputs']}
        assert Path(by_name['arg1_1']['source']).read_bytes() == bytes([token]) * 2048
        assert Path(by_name['arg0_1']['source']).read_bytes() == struct.pack('<f', (token + 1) / 1000)
        assert Path(by_name['arg107_1']['source']).read_bytes() == struct.pack('<i', position)
        assert Path(by_name['arg2_1']['source']).read_bytes() == struct.pack('<i', position)
        assert Path(by_name['arg113_1']['source']).read_bytes() == bytes(4 * (position + 1)) + struct.pack('<f', float('-inf')) * (2047 - position)
        root = Path(output); (root / 'outputs').mkdir(parents=True)
        remote = '/data/local/tmp/fixture/serial-' + str(len(value['calls']))
        retrieved, lines = {}, []
        for row in plan['outputs']:
            relative = f'outputs/unrelated_{row["index"] * 17 % 83:03d}.bin'
            path = root / relative
            with path.open('wb') as stream:
                stream.truncate(row['bytes'])
                if row['index'] == plan['logits_runtime_output_index']:
                    stream.seek((token + 1) * 4); stream.write(struct.pack('<f', 9))
                elif row['index'] == plan['feature_runtime_output_index']:
                    stream.write(np.full(16384, token, dtype='<f2').tobytes())
                else:
                    stream.write(struct.pack('<e', (token + row['index']) % 100))
            retrieved[relative] = record(path)
            lines.append(f'output[{row["index"]}] :{remote}/{relative}')
        for name, raw in (('tool.stdout', ('\n'.join(reversed(lines))+'\n').encode()), ('tool.stderr', b''),
                          ('tool.exitcode', b'0\n'), ('tool.pid', b'1730\n')):
            (root / name).write_bytes(raw); retrieved[name] = record(root / name)
        observed = {'status': 'EXECUTED_UNVALIDATED', 'device_execution': 'EXECUTED_OUTPUTS_RETRIEVED',
            'remote_exitcode': 0, 'tool_process_id': 1730, 'retrieval_errors': [],
            'lock_state': 'NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT', 'source_preservation': backend.SOURCE_PRESERVATION,
            'board_attempt': remote, 'retrieved': retrieved,
            'inputs': [dict(row, host_source=row['source'], source=remote+'/source-inputs/'+row['source_filename']) for row in plan['inputs']]}
        save(root / 'result.json', observed)
        value['calls'].append({'plan': deepcopy(plan), 'observed': observed, 'token': token, 'root': root})
        return observed
    monkeypatch.setattr(board, 'execute_target_tool', execute)
    value['execute'] = execute
    return value


def make(case, **options):
    return backend.OriginalTargetSmokeBackend(case['job'], case['probe'], case['lineage'], case['lookup'],
        case['output'], evidence_kind='host-fake-hiai', **options)


def test_constructor_and_preflight_are_host_only_and_do_not_use_old_decode_state_history(case):
    target = make(case)
    preflight = target.preflight([3, 4, 3])
    assert not case['calls'] and target.native_call_count == 0
    assert preflight['prompt_length'] == preflight['planned_prompt_native_calls'] == 3
    assert preflight['embedding_lookup']['selected_table_read_bytes'] == 2 * 2052
    assert preflight['embedding_lookup']['initialization']['full_table_read_bytes'] == 0
    assert target.binding['old_execution_state_values'] == 'DISCARDED_NOT_USED_AS_PROMPT_HISTORY'
    assert len(target.states) == 80 and all(set(Path(row['output_file']['path']).read_bytes()) == {0} for row in target.states)
    assert target.runtime_version == case['base']['runtime_version'] and target.firmware_id == case['base']['firmware_id']
    assert target.execution_summary()['validated_native_calls'] is False


def test_serial_prompt_and_advance_bind_all_actual_calls80_states_raw_rows_and_complete_feature_prefix(case):
    target = make(case)
    prompt = list(range(3, 11))
    prefill = target.prefill(prompt)
    assert prefill.next_token_id == 11 and prefill.prefix_length == 8
    assert target.native_call_count == len(case['calls']) == 8
    np.testing.assert_array_equal(prefill.features[:, :, 0], [prompt])
    assert prefill.evidence.feature_output_sha256 == feature_sha256(prefill.features)
    combined = json.loads((case['output'] / 'prefill-receipt.json').read_text())
    assert combined['native_call_count'] == len(combined['native_calls']) == 8
    assert combined['feature_binding'] == 'CONCATENATION_OF_ALL_ACTUAL_NATIVE_FEATURE_ROWS'
    assert prefill.evidence.receipt_sha256 == hashlib.sha256((case['output'] / 'prefill-receipt.json').read_bytes()).hexdigest()
    old_state = prefill.state_sha256
    advanced = target.advance(11)
    assert advanced.prefix_length == 9 and advanced.next_token_id == 12
    assert advanced.evidence.input_state_sha256 == old_state
    assert advanced.evidence.prefix_sha256 == prefix_sha256([*prompt, 11])
    assert advanced.evidence.feature_output_sha256 == feature_sha256(advanced.features[:, -1:])
    np.testing.assert_array_equal(advanced.features[:, :, 0], [[*prompt, 11]])
    for previous, current in zip(case['calls'], case['calls'][1:]):
        chain = json.loads((current['root'].parent / 'state-chain.json').read_text())
        assert chain['feedback_input_count'] == 80 and chain['bound_child_output_count'] == 82
        previous_chain = json.loads((previous['root'].parent / 'state-chain.json').read_text())
        old = {row['source_refdata']: row['child_output']['sha256'] for row in previous_chain['states']}
        assert all(row['parent_output']['sha256'] == old[row['source_refdata']] for row in chain['states'])
    assert target.execution_summary()['native_calls'] == 9 and target.execution_summary()['validated_native_calls']
    assert len(target.execution_summary()['receipt_paths']) == 9


def test_real_fullsize_target_adapter_integrates_existing_serial_controller_without_advancing_rejected_id(case):
    from test_incremental_smoke import Draft
    contract = SmokeContract.for_host_test(target_layers=40, draft_layers=6, vocab_size=248320,
        feature_width=16384, draft_width=15, sequence_capacity=64, feature_layer_ids=(1,6,11,16,22,27,32,37))
    target = make(case); draft = Draft(contract, mismatch_at=0)
    result = run_incremental_smoke(target, draft, [3], contract=contract, max_new_tokens=3,
        max_proposals_per_round=2, minimum_draft_rounds=2, eos_token_ids=[], target_next_anchor=True)
    assert result['generated_token_ids'] == [4, 5, 6]
    assert [row['token'] for row in case['calls']] == [3, 4, 5, 6]
    assert target.execution_summary()['native_calls'] == 4
    assert result['numerical_validation'] == 'NOT_EVALUATED' and result['device_execution'] == 'UNVERIFIED'


def test_receive_failure_stops_target_request_without_automatic_model_retry_or_state_commit(case):
    target = make(case); case['fail_at'] = 0
    with pytest.raises(RuntimeError, match='retrieve-outputs.*fixture HDC disconnected.*board_result='):
        target.prefill([3])
    assert target.prefix == [] and target.native_call_count == 0 and target.failed
    with pytest.raises(RuntimeError, match='no automatic model retry'):
        target.prefill([3])
    assert target.execution_summary()['validated_native_calls'] is False
    assert Path(target.execution_summary()['retained_board_result']).is_file()
    assert target.execution_summary()['failed_step'].endswith('step_0000')


@pytest.mark.parametrize('fault', ['control', 'prepared_state', 'parent_state', 'plan', 'binding'])
def test_prepared_step_validator_rejects_any_changed_control_state_or_exact_proof(case, monkeypatch, fault):
    target = make(case)
    class PreparedStop(Exception): pass
    def stop(*args, **options):
        prepared = Path(options['runtime_smoke_backend_prepared'])
        if fault == 'control':
            (prepared / 'inputs/arg1238_1.bin').write_bytes(struct.pack('<i', 1))
        elif fault == 'prepared_state':
            with (prepared / 'inputs/RefData_0.bin').open('r+b') as stream: stream.write(b'\x01')
        elif fault == 'parent_state':
            with Path(target.states[0]['output_file']['path']).open('r+b') as stream: stream.write(b'\x01')
        elif fault == 'plan':
            plan = json.loads((prepared / 'plan.json').read_text()); plan['model_sha256'] = 'f' * 64; save(prepared / 'plan.json', plan)
        else:
            binding = json.loads((case['output'] / 'backend-binding.json').read_text()); binding['target_layers'] = 1
            save(case['output'] / 'backend-binding.json', binding)
        with pytest.raises(ValueError): backend.validate_smoke_backend_step(prepared)
        raise PreparedStop()
    monkeypatch.setattr(board, 'execute_target_tool', stop)
    with pytest.raises(PreparedStop): target.prefill([3])
    assert target.prefix == [] and target.failed


def test_incompatible_state_mapping_fails_before_any_native_call(case):
    case['current']['decode']['state_file_mapping']['states'][0]['runtime_input_name'] = 'RefData_wrong'
    with pytest.raises(ValueError, match='mapping'):
        make(case)
    assert not case['calls']


def test_child_state_mutation_after_receipt_write_never_commits_state_or_reports_validated_calls(case, monkeypatch):
    target = make(case); original = backend.save_json
    def changed(path, value):
        original(path, value)
        if path.name == 'receipt.json':
            chain = json.loads((path.parent / 'state-chain.json').read_text())
            selected = path.parent / 'board' / chain['states'][0]['child_output']['path']
            with selected.open('r+b') as stream: stream.write(b'x')
    monkeypatch.setattr(backend, 'save_json', changed)
    with pytest.raises(ValueError, match='changed'):
        target.prefill([3])
    assert target.prefix == [] and target.failed and not target.execution_summary()['validated_native_calls']


def test_step_plan_reports_only_its_new_controls_and_compiled_weight_scope(case):
    case['base']['control_observations'] = [{'file': 'position_ids.bin', 'values': [256]}]
    case['base']['warnings'] = ['stale old-control warning']
    target = make(case)
    target.prefill([3])
    plan = case['calls'][0]['plan']
    assert plan['control_observations']['absolute_position'] == 0
    assert plan['control_observations']['output_pos'] == 0
    assert plan['compiled_weight_payload_access'] == 'NONE'
    assert plan['embedding_table_payload_access'] == 'SELECTED_ORIGINAL_ROWS_ONLY'
    assert 'stale' not in repr(plan['warnings'])


@pytest.mark.parametrize('case', [True], indirect=True)
def test_actual10306827_byte_compilation_report_uses_existing32mib_planner_not_backend8mib_receipt_reader(case):
    report = case['compilation'] / 'report.json'
    before = record(report)
    assert before['bytes'] == 10_306_827
    target = make(case)
    assert target.preflight([3, 4])['planned_prompt_native_calls'] == 2
    assert report.read_bytes().rstrip().endswith(b'}') and record(report) == before
    assert any(row['path'] == str(report) and row['identity']['bytes'] == 10_306_827
        for row in target.binding['active_source_identities'])
    assert not case['calls']


def test_noninitial_step_cannot_drop_the_preceding_actual_native_receipt(case, monkeypatch):
    target = make(case)
    target.prefill([3])
    def refuse(*args, **options):
        prepared = Path(options['runtime_smoke_backend_prepared'])
        manifest = json.loads((prepared / 'manifest.json').read_text())
        manifest['parent_native_receipt'] = None
        save(prepared / 'manifest.json', manifest)
        return case['execute'](*args, **options)
    monkeypatch.setattr(board, 'execute_target_tool', refuse)
    with pytest.raises(ValueError, match='preceding actual native call receipt'):
        target.advance(4)
    assert target.prefix == [3] and target.native_call_count == 1 and target.failed
