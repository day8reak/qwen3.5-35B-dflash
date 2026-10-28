"""Small CPU replay artifact fixtures; fabricated native records are not board evidence."""
import copy
import json

import numpy as np
import pytest

from qwen35_moe_dflash import draft_reference_replay as replay
from qwen35_moe_dflash import draft_reference_trace as trace
from qwen35_moe_dflash.hashing import sha256_file
from test_draft_board import bundle, record, write_json
from test_draft_review import retained, _snapshot, _store_execution, _rebind_output


def _bind(case):
    write_json(case['prepared']/'prepared.json', case['plan'])
    case['reference']['prepared_manifest_sha256'] = sha256_file(case['prepared']/'prepared.json')
    write_json(case['fixture']/'report.json', case['reference'])
    case['manifest']['fixture_report_sha256'] = sha256_file(case['fixture']/'report.json')
    write_json(case['bundle']/'manifest.json', case['manifest'])
    case['execution']['bundle_manifest_sha256'] = sha256_file(case['bundle']/'manifest.json')
    _store_execution(case)
    jobfile = case['job']/'result.json'
    job = json.loads(jobfile.read_text())
    for name, directory, filename in (
            ('prepared',case['prepared'],'prepared.json'),
            ('fixture',case['fixture'],'report.json'),
            ('bundle',case['bundle'],'manifest.json')):
        job['components'][name] = {'directory':str(directory),'manifest':record(directory/filename,directory)}
    write_json(jobfile,job)


@pytest.fixture
def replay_case(retained,tmp_path,monkeypatch):
    case = retained
    prepared = case['job']/'prepared'; prepared.mkdir()
    norm = prepared/'norm.bin'; np.ones(2048,dtype='<f2').tofile(norm)
    case.update(prepared=prepared,fixture=tmp_path/'fixture',norm=norm)
    case['plan'] = {'artifact_kind':'qwen35-draft-ge-prepared','status':'PREPARED','tiny':False,
        'sequence_length':8,'draft_config':case['manifest']['draft_config'],
        'weights':{'draft.norm.weight':record(norm,prepared,dtype='float16',shape=[2048])}}
    case['reference'] = json.loads((case['fixture']/'report.json').read_text())
    case['reference']['reference'] = 'PYTORCH_FP16_PRIMITIVES'
    _bind(case)
    case['graph'] = object()
    case['calls'] = []
    case['outputs'] = {name:np.fromfile(case['bundle']/row['path'],dtype=dtype).reshape(row['shape'])
        for name,dtype,row in zip(('draft_top1','draft_hidden'),('<i4','<f2'),case['manifest']['expected'])}
    case['candidates'] = {mode:{key:value.copy() for key,value in case['outputs'].items()}
                          for mode in ('legacy-fp16','fp32-stats')}
    case['traces'] = {mode:{'validation':{'status':'PASSED','finite':True,'nondegenerate':True,'errors':[]},
                            'torch_version':'test-torch','first_collapse':None,'rms_nodes':[]}
                      for mode in case['candidates']}
    def validated(root):
        assert root == prepared
        case['calls'].append('validate')
        return prepared,copy.deepcopy(case['plan']),object(),{'draft.norm.weight':norm}
    def load(config,paths):
        case['calls'].append('load')
        assert paths == {'draft.norm.weight':norm}
        return case['graph']
    def run(graph,inputs,*,rms_mode,progress):
        assert graph is case['graph']
        case['calls'].append(rms_mode)
        for values,row in zip(inputs,case['manifest']['inputs']):
            assert values.tobytes() == (case['bundle']/row['path']).read_bytes()
        return case['candidates'][rms_mode],case['traces'][rms_mode]
    monkeypatch.setattr(replay,'_validated_prepared',validated)
    monkeypatch.setattr(replay,'_load_graph',load)
    monkeypatch.setattr(trace,'replay_graph',run)
    return case


def _run(case,**kwargs):
    return replay.replay_reference(case['job'],case['job']/'reference-replay',**kwargs)


def _source_snapshot(case):
    return {name:digest for name,digest in _snapshot(case['job']).items()
            if not name.startswith('reference-replay/')}


def test_replay_loads_once_exact_inputs_and_keeps_original_evidence(replay_case):
    case = replay_case
    before = _source_snapshot(case)
    messages = []
    result = _run(case,progress=messages.append)
    assert case['calls'] == ['validate','load','legacy-fp16','fp32-stats']
    assert result['status'] == 'DIAGNOSTIC_COMPLETE'
    assert result['legacy_reproduced'] is True
    assert result['numerical_validation'] == 'TOP1_MATCHED'
    assert result['new_device_execution'] == result['new_compilation'] == 'NOT_RUN'
    assert result['provenance']['torch_version'] == 'test-torch'
    names = {row['path'] for row in result['provenance']['source_files']}
    assert {'draft_reference.py','draft_reference_trace.py','model.py','ops.py','graphs.py'} <= names
    assert result['sources']['model_sha256'] == case['manifest']['model']['sha256']
    assert result['sources']['inputs'] == case['manifest']['inputs']
    assert 'not a complete upstream BF16 oracle' in result['scope']
    candidate = result['modes']['fp32-stats']
    assert candidate['candidate_valid'] is True and candidate['token_mismatches'] == 0
    assert candidate['hidden_diagnostic']['relative_l2'] == 0
    root = case['job']/'reference-replay'
    assert {path.name for path in root.iterdir()} == replay._FILES
    assert all(path.stat().st_size < 2 << 20 for path in root.iterdir())
    assert json.loads((root/'report.json').read_text()) == result
    assert any('Top1 mismatches=0' in message for message in messages)
    assert _source_snapshot(case) == before
    # A retry only replaces known small diagnostic files in the same folder.
    assert _run(case) == result
    assert _source_snapshot(case) == before


def test_legacy_exact_gate_required_before_candidate_top1_match(replay_case):
    replay_case['candidates']['legacy-fp16']['draft_top1'][0,0] += 1
    result = _run(replay_case)
    assert result['legacy_reproduced'] is False
    assert result['numerical_validation'] == 'NOT_ESTABLISHED'
    assert result['modes']['fp32-stats']['token_mismatches'] == 0


def test_collapsed_legacy_golden_reproduces_and_candidate_stays_separate(replay_case):
    case = replay_case
    for index,key in enumerate(('expected_top1','expected_hidden')):
        row = case['reference'][key]
        values = np.zeros_like(case['outputs'][row['name']])
        fixture_file = case['fixture']/row['path']; values.tofile(fixture_file)
        row['sha256'] = sha256_file(fixture_file)
        bundled = case['manifest']['expected'][index]
        bundle_file = case['bundle']/bundled['path']; bundle_file.chmod(0o600); values.tofile(bundle_file)
        bundled['sha256'] = sha256_file(bundle_file)
        case['candidates']['legacy-fp16'][row['name']] = values
    _bind(case)
    case['traces']['legacy-fp16'].update(
        validation={'status':'FAILED','finite':True,'nondegenerate':False,'errors':['all-zero hidden']},
        first_collapse={'name':'draft.hidden_norm','cause_candidate':'FP16 square overflow'})
    before = _source_snapshot(case)
    messages = []
    result = _run(case,progress=messages.append)
    assert result['legacy_reproduced'] is True
    assert result['numerical_validation'] == 'TOP1_MATCHED'
    assert result['modes']['legacy-fp16']['hidden_summary']['nonzero_count'] == 0
    assert result['modes']['fp32-stats']['hidden_diagnostic']['relative_l2'] == 0
    assert any('draft.hidden_norm' in message for message in messages)
    assert _source_snapshot(case) == before


@pytest.mark.parametrize('fault',['invalid-trace','all-zero','nonfinite','invalid-token','top1-difference','board-nonfinite'])
def test_candidate_must_be_valid_and_match_board_top1(replay_case,fault):
    case = replay_case
    candidate = case['candidates']['fp32-stats']
    if fault == 'invalid-trace':
        case['traces']['fp32-stats']['validation']['status'] = 'FAILED'
    elif fault == 'all-zero':
        candidate['draft_hidden'].fill(0)
    elif fault == 'nonfinite':
        candidate['draft_hidden'][0,0,0] = np.inf
    elif fault == 'invalid-token':
        candidate['draft_top1'][0,0] = case['manifest']['vocab_size']
    elif fault == 'top1-difference':
        candidate['draft_top1'][0,0] += 1
    else:
        path = case['board']/'results/output_0001.bin'
        values = case['outputs']['draft_hidden'].copy(); values[0,0,0] = np.nan; values.tofile(path)
        _rebind_output(case,1)
    result = _run(case)
    assert result['legacy_reproduced'] is True
    assert result['numerical_validation'] == 'FAILED'
    json.dumps(result,allow_nan=False)
    assert result['modes']['fp32-stats']['candidate_valid'] is (fault == 'top1-difference')


@pytest.mark.parametrize('fault',['fixture-binding','input-description','board-tampered','prepared-weight'])
def test_source_mismatch_fails_before_graph_or_output(replay_case,fault):
    case = replay_case
    if fault == 'fixture-binding':
        case['reference']['prepared_manifest_sha256'] = '0'*64
        write_json(case['fixture']/'report.json',case['reference'])
    elif fault == 'input-description':
        case['reference']['inputs'][0]['name'] = 'wrong'
        _bind(case)
    elif fault == 'board-tampered':
        (case['board']/'results/output_0000.bin').write_bytes(b'changed')
    else:
        case['norm'].write_bytes(b'changed')
    before = _source_snapshot(case)
    with pytest.raises(ValueError):
        _run(case)
    assert case['calls'] == []
    assert not (case['job']/'reference-replay').exists()
    assert _source_snapshot(case) == before


def test_weight_change_during_validation_cannot_become_identity_baseline(replay_case,monkeypatch):
    case = replay_case
    validate = replay._validated_prepared
    def changed_after_hash(root):
        result = validate(root)
        case['norm'].write_bytes(b'changed after hash check')
        return result
    monkeypatch.setattr(replay,'_validated_prepared',changed_after_hash)
    with pytest.raises(ValueError,match='changed during validation'):
        _run(case)
    assert case['calls'] == ['validate']
    assert json.loads((case['job']/'reference-replay/report.json').read_text())['status'] == 'FAILED'


def test_input_change_after_file_hash_cannot_be_loaded_as_verified_bytes(replay_case,monkeypatch):
    case = replay_case
    checked = replay._file
    path = case['bundle']/case['manifest']['inputs'][0]['path']
    def changed_after_hash(root,row):
        result = checked(root,row)
        if result == path:
            path.chmod(0o600)
            np.zeros((1,8,16384),dtype='<f2').tofile(path)
        return result
    monkeypatch.setattr(replay,'_file',changed_after_hash)
    with pytest.raises(ValueError,match='tensor changed after verification'):
        _run(case)
    assert case['calls'] == []
    assert not (case['job']/'reference-replay').exists()


@pytest.mark.parametrize('fault',['foreign-file','directory','symlink','unowned'])
def test_replay_never_removes_unowned_output_content(replay_case,fault):
    case = replay_case
    root = case['job']/'reference-replay'
    if fault != 'unowned':
        _run(case)
    else:
        root.mkdir()
    if fault in ('foreign-file','unowned'):
        (root/'keep-me').write_text('unrelated')
    elif fault == 'directory':
        (root/'foreign-dir').mkdir()
    else:
        (root/'report.json').unlink(); (root/'report.json').symlink_to(case['job']/'result.json')
    calls = list(case['calls'])
    with pytest.raises(ValueError):
        _run(case)
    assert case['calls'] == calls
    assert (root/('foreign-dir' if fault == 'directory' else 'report.json' if fault == 'symlink' else 'keep-me')).exists()


@pytest.mark.parametrize('fault',['exception','mutation','shape','dtype','source-change','version','nan-trace'])
def test_failed_replay_is_recorded_without_promoting_or_changing_sources(replay_case,monkeypatch,fault):
    case = replay_case
    original = trace.replay_graph
    def run(graph,inputs,*,rms_mode,progress):
        result,log = original(graph,inputs,rms_mode=rms_mode,progress=progress)
        if rms_mode == 'fp32-stats':
            if fault == 'exception':
                raise RuntimeError('injected CPU failure')
            if fault == 'mutation':
                inputs[0].fill(0)
            elif fault == 'shape':
                result['draft_hidden'] = result['draft_hidden'].reshape(-1)
            elif fault == 'dtype':
                result['draft_top1'] = result['draft_top1'].astype(np.int64)
            elif fault == 'source-change':
                case['norm'].write_bytes(b'changed')
            elif fault == 'version':
                log['torch_version'] = 'other-version'
            elif fault == 'nan-trace':
                log['bad'] = float('nan')
        return result,log
    monkeypatch.setattr(trace,'replay_graph',run)
    before = _source_snapshot(case)
    with pytest.raises((ValueError,RuntimeError)):
        _run(case)
    result = json.loads((case['job']/'reference-replay/report.json').read_text())
    assert result['status'] == 'FAILED' and result['numerical_validation'] == 'NOT_ESTABLISHED'
    assert result['error']
    if fault != 'source-change':
        assert _source_snapshot(case) == before
