"""Old prefill files gain a later same-model name witness, never a bridge pass."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from qwen35_moe_dflash import target_prefill_decode_bridge as bridge
from qwen35_moe_dflash.target_prefill_smoke_plan import prepare_prefill_smoke_plan
from test_target_prefill_smoke_plan import case as source_case, rewrite_native
from test_target_prefill_output_inventory import case as output_case, record, save


@pytest.fixture
def case(tmp_path, monkeypatch):
    source_root = tmp_path / 'source-case'; source_root.mkdir()
    source = source_case.__wrapped__(source_root)
    proof = prepare_prefill_smoke_plan(source['compilation'], source['probe'], source['source'])
    value = output_case.__wrapped__(tmp_path, monkeypatch)
    attempt = Path(value['report']['board_attempt'])
    inputs = [dict(row, host_source=row['source'], source=str(attempt / 'source-inputs' / row['source_filename'])) for row in proof['inputs']]
    plan = dict(proof, inputs=inputs)
    value['plan'] = plan
    value['report'].update(runtime_prefill_probe=proof['runtime_probe'], inputs=inputs)
    value['report']['model_payloads'][0].update(sha256=proof['model_sha256'], bytes=proof['primary_model_binding']['bytes'])
    for desc, relative in zip(proof['outputs'], value['paths']):
        path = value['board'] / relative
        with path.open('wb') as stream: stream.truncate(desc['bytes'])
        value['report']['retrieved'][relative] = record(path)
    save(value['board'] / 'plan.json', plan)
    value['report']['plan_sha256'] = record(value['board'] / 'plan.json')['sha256']
    save(value['board'] / 'runtime-prefill-proof.json', proof); save(value['execution'] / 'plan.json', proof)
    save(value['board'] / 'input-evidence.json', {'inputs': inputs}); save(value['board'] / 'result.json', value['report'])
    outer = dict(value['report'], artifact_kind='qwen35-target-prefill-smoke-job', mode='APPLY',
        board_result=str(value['board'] / 'result.json'), plan=str(value['execution'] / 'plan.json'))
    save(value['execution'] / 'result.json', outer)
    model = value['report']['model_payloads'][0]['path']
    save(value['board'] / 'command.json', {'argv': [proof['tool_binding']['path'], '--model=' + model,
        '--modelmanager_itf=V2', '--direct_from_file=true', '--times=1', '--weight_dir=' + str(Path(model).parent),
        '--output_dir=' + str(attempt / 'outputs'), '--output_type=' + proof['output_type']]})
    named = tmp_path / 'new-named-probe'; shutil.copytree(source['probe'], named)
    source['probe'] = named
    original_inputs = source['contract']['boundaries']['inputs_in_serialized_op_order']
    for row, original in zip(source['native']['inputs'], reversed(original_inputs)):
        row.update(tensor_name=original['name'], tensor_name_status='OBSERVED')
    for row in source['native']['outputs']:
        row.update(tensor_name='literal_output_' + str(row['index']), tensor_name_status='OBSERVED')
    source['outer']['runtime_tensor_names'] = 'OBSERVED_ALL'
    rewrite_native(source)
    return dict(value, source=source, named=named, out=tmp_path / 'bridge-diagnosis')


def inspect(case):
    return bridge.inspect_prefill_decode_bridge(case['execution'], case['named'], case['out'])


def test_new_same_model_names_bind_old86_files_and89_executed_inputs_without_rewriting_or_preparing_feedback(case, monkeypatch):
    watched = [path for path in case['execution'].rglob('*') if path.is_file()]
    before = {path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_ctime_ns) for path in watched}
    opened = Path.open
    def no_models(path, *args, **kwargs):
        assert path.suffix not in {'.weight', '.omc', '.air'}, 'metadata bridge cannot load model or weight payloads'
        return opened(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', no_models)
    result = inspect(case)
    assert result['status'] == 'PREFILL_NAMED_SLOTS_AND_RETAINED_FILES_BOUND_BRIDGE_GAPS_REMAIN', result
    assert result['feedback_prepared'] is False and result['state_continuity'] == 'NOT_VALIDATED'
    assert len(result['retained_outputs']) == 82 and len(result['original_source_inputs']) == 13
    witness = result['named_probe_witness']
    assert witness['status'] == 'POST_HOC_SAME_MODEL_ORDERED_DESCRIPTOR_NAMES_BOUND'
    assert len(witness['inputs']) == 89 and len(witness['outputs']) == 82
    assert len(result['required_next_evidence']) == 4
    assert result['executed_input_byte_binding'].startswith('ALL89')
    active = {row['path']: row['identity'] for row in result['active_source_identities']}
    assert str(case['board'] / 'tool.stdout') in active and str(case['execution'] / 'plan.json') in active
    assert len(active[str(case['board'] / 'tool.stdout')]) == 9
    assert not (case['out'] / 'inputs').exists()
    assert before == {path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_ctime_ns) for path in watched}


@pytest.mark.parametrize('fault', ['runtime', 'model', 'ordered_shape', 'duplicate_name', 'missing_name'])
def test_later_names_are_rejected_unless_same_identity_and_exact_ordered_raw_abi(case, fault):
    source = case['source']
    if fault == 'runtime':
        source['outer']['firmware_id'] = source['native']['firmware_id'] = 'different-firmware'
        command = json.loads((case['named'] / 'model-io-command.json').read_text())
        command['argv'][command['argv'].index('--firmware-id') + 1] = 'different-firmware'; save(case['named'] / 'model-io-command.json', command)
    elif fault == 'model': source['outer']['model_sha256'] = source['native']['model_sha256'] = 'f' * 64
    elif fault == 'ordered_shape': source['native']['outputs'][0]['shape'] = [1, 16384, 64]
    elif fault == 'duplicate_name': source['native']['outputs'][0]['tensor_name'] = source['native']['outputs'][1]['tensor_name']
    else: source['native']['inputs'][0].pop('tensor_name_status')
    rewrite_native(source)
    result = inspect(case)
    assert result['status'] == 'FAILED' and result['feedback_prepared'] is False, result


@pytest.mark.parametrize('fault', ['input_hash', 'source_bytes', 'output_bytes', 'log_mapping'])
def test_real_old_input_bytes_all_outputs_and_explicit_old_stdout_are_still_required(case, fault):
    if fault == 'input_hash':
        path = case['board'] / 'input-evidence.json'; value = json.loads(path.read_text()); value['inputs'][2]['sha256'] = 'f' * 64; save(path, value)
    elif fault == 'source_bytes':
        path = case['source']['source'] / 'inputs_embeds.bin'
        with path.open('r+b') as stream: stream.write(b'x')
    elif fault == 'output_bytes':
        with (case['board'] / case['paths'][3]).open('r+b') as stream: stream.write(b'x')
    else:
        path = case['board'] / 'tool.stdout'; text = path.read_text(); path.write_text(text + '\n' + text.splitlines()[2] + '\n')
        # Re-sign only the stdout transport record: duplicate slots must still fail.
        case['report']['retrieved']['tool.stdout'] = record(path)
        save(case['board'] / 'result.json', case['report'])
        outer = json.loads((case['execution'] / 'result.json').read_text()); outer['retrieved'] = case['report']['retrieved']; save(case['execution'] / 'result.json', outer)
    result = inspect(case)
    assert result['status'] == 'FAILED' and result['feedback_prepared'] is False, result


def test_bridge_cli_prints_only_diagnostic_bindings_and_missing_evidence(case, capsys):
    import runpy
    module = runpy.run_path(str(Path(__file__).parents[1] / 'targets/cpro/scripts/inspect_target_prefill_decode_bridge.py'))
    assert module['main'](['--execution', str(case['execution']), '--named-runtime-probe', str(case['named']), '--output', str(case['out'])]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary['feedback_prepared'] is False and summary['input_count'] == 89 and summary['output_count'] == 82
    assert summary['named_probe_witness'] == 'POST_HOC_SAME_MODEL_ORDERED_DESCRIPTOR_NAMES_BOUND'
    assert len(summary['required_next_evidence']) == 4


@pytest.mark.parametrize('kind', ['metadata', 'retained_payload'])
def test_active_ctime_only_change_after_current_snapshot_stays_a_failure(case, monkeypatch, kind):
    import os
    if kind == 'metadata':
        original, calls = bridge._probe, 0
        def changed(*args, **kwargs):
            nonlocal calls
            result = original(*args, **kwargs); calls += 1
            if calls == 2:
                path = case['execution'] / 'plan.json'
                os.chmod(path, path.stat().st_mode & 0o7777)
            return result
        monkeypatch.setattr(bridge, '_probe', changed)
    else:
        original = bridge._hash_retrieved
        def changed(path, *args, **kwargs):
            result = original(path, *args, **kwargs)
            os.chmod(path, path.stat().st_mode & 0o7777)
            return result
        monkeypatch.setattr(bridge, '_hash_retrieved', changed)
    result = inspect(case)
    assert result['status'] == 'FAILED' and 'changed' in result['error']
