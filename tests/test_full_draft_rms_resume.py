"""Precision contract resume gates; mocked pipeline stages are not device evidence."""
import copy
import json

import pytest

from qwen35_moe_dflash import draft_reference
from qwen35_moe_dflash.hashing import sha256_file
from test_full_draft_cli import job, _make_legacy_air, _stage_payloads, _assert_payloads_unchanged


def _json(path):
    return json.loads(path.read_text())


def _write(path,value):
    path.write_text(json.dumps(value))


def _legacy_fixture(root, *, contract=None):
    path = root/'fixture/report.json'
    reference = _json(path)
    reference['reference'] = ('PYTORCH_FP16_RMS_FP32' if contract == 'pytorch-fp16-rms-fp32-v2'
                              else 'PYTORCH_FP16_PRIMITIVES')
    if contract is None:
        reference.pop('reference_contract')
    else:
        reference['reference_contract'] = contract
    _write(path,reference)
    receipt_path = root/'completed-fixture.json'
    receipt = _json(receipt_path)
    receipt['manifest'].update(bytes=path.stat().st_size,sha256=sha256_file(path))
    _write(receipt_path,receipt)
    result_path = root/'result.json'
    result = _json(result_path)
    result['components']['fixture'] = receipt
    _write(result_path,result)
    return result


@pytest.fixture
def precision_job(job,monkeypatch):
    # Tiny byte strings exercise identity plumbing only. Full graph/array ABI
    # correctness is covered separately by reference and board tests.
    reference = draft_reference.prepare_draft_reference
    behavior = {'change':None}
    def generate(prepared,folder,**kwargs):
        report = reference(prepared,folder,**kwargs)
        inputs = []
        for index,(name,dtype) in enumerate((('target_features','float16'),('input_ids','int32'),
                                            ('attention_mask','int32'))):
            path = folder/(name+'.bin')
            data = bytes([index+1])*8
            if behavior['change'] == 'bytes' and index == 0:
                data = b'changed!'
            path.write_bytes(data)
            inputs.append({'index':index,'name':name,'path':path.name,'dtype':dtype,
                           'shape':[1,4 if dtype=='float16' else 2],
                           'bytes':len(data),'sha256':sha256_file(path)})
        if behavior['change'] == 'shape':
            inputs[0]['shape'] = [2,2]
        elif behavior['change'] == 'dtype':
            inputs[0]['dtype'] = 'int32'
        elif behavior['change'] == 'order':
            inputs.reverse()
        elif behavior['change'] == 'missing':
            inputs.pop()
        elif behavior['change'] == 'path-only':
            for row in inputs:
                previous = folder/row['path']
                row['path'] = 'new-'+row['path']
                previous.rename(folder/row['path'])
        report.update(inputs=inputs,reference=draft_reference.DRAFT_REFERENCE_KIND)
        _write(folder/'report.json',report)
        return report
    monkeypatch.setattr(draft_reference,'prepare_draft_reference',generate)
    return (*job,behavior)


@pytest.mark.parametrize('old_reference,old_export',[
    (None,'cpro-fileconstant-relative-int-fc-conv2d-mask-gather-v4'),
    ('pytorch-fp16-rms-fp32-v2','cpro-fileconstant-relative-int-fc-conv2d-mask-gather-rms-fp32-v5'),
    (None,'cpro-fileconstant-relative-int-fc-conv2d-mask-gather-rms-fp32-v5'),
    ('pytorch-fp16-rms-fp32-v2','cpro-fileconstant-relative-int-fc-conv2d-mask-gather-v4'),
])
def test_obsolete_precision_contracts_rebuild_downstream_in_fixed_workspace(
        precision_job,tmp_path,old_reference,old_export):
    main,fresh,common,calls,_,_ = precision_job
    root = tmp_path/'full-draft'
    assert main(fresh) == 0
    _legacy_fixture(root,contract=old_reference)
    old = _make_legacy_air(root,contract=old_export)
    prepared = _stage_payloads(old,('prepared',))
    (root/'board/stale.log').write_text('old numerical result')
    calls.clear()
    assert main(common) == 0
    assert calls == ['fixture','air','compilation','bundle','board']
    result = _json(root/'result.json')
    assert result['components']['prepared'] == old['components']['prepared']
    assert result['reference_input_identity'] == _json(root/'fixture/report.json')['inputs']
    assert _json(root/'fixture/report.json')['reference_contract'] == draft_reference.DRAFT_REFERENCE_CONTRACT
    assert [row['reason'] for row in result['resume_invalidations']] == [
        'OBSOLETE_DRAFT_REFERENCE_CONTRACT','OBSOLETE_DRAFT_EXPORT_CONTRACT']
    assert set(result['resume_invalidations'][0]['components']) == {'fixture','bundle'}
    assert set(result['resume_invalidations'][1]['components']) == {'air','compilation'}
    assert not (root/'board/stale.log').exists()
    _assert_payloads_unchanged(prepared)
    assert [path.name for path in tmp_path.glob('full-draft*')] == ['full-draft']


@pytest.mark.parametrize('old_contract',[None,'pytorch-fp16-rms-v1','pytorch-fp16-rms-fp32-v2'])
def test_reference_only_contract_change_reuses_current_air_and_compilation(precision_job,tmp_path,old_contract):
    main,fresh,_,calls,_,_ = precision_job
    root = tmp_path/'full-draft'
    assert main(fresh) == 0
    old = _legacy_fixture(root,contract=old_contract)
    preserved = _stage_payloads(old,('prepared','air','compilation'))
    calls.clear()
    # GE/DDK flags are unnecessary: only CPU reference, bundle and board rerun.
    assert main([]) == 0
    assert calls == ['fixture','bundle','board']
    result = _json(root/'result.json')
    assert all(result['components'][name] == old['components'][name] for name in ('prepared','air','compilation'))
    invalidation, = result['resume_invalidations']
    assert invalidation['reason'] == 'OBSOLETE_DRAFT_REFERENCE_CONTRACT'
    assert invalidation['preserved'] == ['prepared','air','compilation']
    assert invalidation['required_reference_contract'] == draft_reference.DRAFT_REFERENCE_CONTRACT
    _assert_payloads_unchanged(preserved)
    calls.clear()
    assert main([]) == 0
    assert calls == ['board']
    assert [path.name for path in tmp_path.glob('full-draft*')] == ['full-draft']


def test_intermediate_rms_only_air_is_rebuilt_without_replacing_current_reference(precision_job,tmp_path):
    main,fresh,common,calls,_,_ = precision_job
    root = tmp_path/'full-draft'
    assert main(fresh) == 0
    old = _make_legacy_air(root,contract='cpro-fileconstant-relative-int-fc-conv2d-mask-gather-rms-fp32-v5')
    kept = _stage_payloads(old,('prepared','fixture'))
    calls.clear()
    assert main(common) == 0
    assert calls == ['air','compilation','bundle','board']
    result = _json(root/'result.json')
    assert all(result['components'][stage] == old['components'][stage] for stage in ('prepared','fixture'))
    invalidation, = result['resume_invalidations']
    assert invalidation['reason'] == 'OBSOLETE_DRAFT_EXPORT_CONTRACT'
    assert set(invalidation['components']) == {'air','compilation','bundle'}
    _assert_payloads_unchanged(kept)


@pytest.mark.parametrize('changed',['bytes','shape','dtype','order','missing'])
def test_changed_regenerated_inputs_stop_before_bundle_board_and_stay_guarded_on_retry(precision_job,tmp_path,changed):
    main,fresh,_,calls,_,behavior = precision_job
    root = tmp_path/'full-draft'
    assert main(fresh) == 0
    old = _legacy_fixture(root)
    preserved = _stage_payloads(old,('prepared','air','compilation'))
    original_inputs = copy.deepcopy(_json(root/'fixture/report.json')['inputs'])
    behavior['change'] = changed
    for _ in range(2):
        calls.clear()
        assert main([]) == 1
        assert calls == ['fixture']
        result = _json(root/'result.json')
        assert result['status'] == 'FAILED' and result['stage'] == 'fixture'
        assert 'rebuilt reference input bytes differ' in result['error']
        assert result['reference_input_identity'] == original_inputs
        assert set(result['components']) == {'prepared','air','compilation'}
        assert not (root/'completed-fixture.json').exists()
        assert not (root/'completed-bundle.json').exists()
        assert not (root/'bundle').exists() and not (root/'board').exists()
        _assert_payloads_unchanged(preserved)
    behavior['change'] = None
    calls.clear()
    assert main([]) == 0
    assert calls == ['fixture','bundle','board']
    assert _json(root/'result.json')['reference_input_identity'] == original_inputs
    _assert_payloads_unchanged(preserved)
    assert [path.name for path in tmp_path.glob('full-draft*')] == ['full-draft']


def test_rebuilt_input_paths_can_change_without_changing_tensor_identity(precision_job,tmp_path):
    main,fresh,_,calls,_,behavior = precision_job
    root = tmp_path/'full-draft'
    assert main(fresh) == 0
    _legacy_fixture(root)
    original = _json(root/'fixture/report.json')['inputs']
    behavior['change'] = 'path-only'
    calls.clear()
    assert main([]) == 0
    assert calls == ['fixture','bundle','board']
    rebuilt = _json(root/'fixture/report.json')['inputs']
    assert [row['path'] for row in original] != [row['path'] for row in rebuilt]
    assert [row['sha256'] for row in original] == [row['sha256'] for row in rebuilt]


def test_bad_legacy_reference_receipt_is_rejected_before_workspace_reset(precision_job,tmp_path):
    main,fresh,_,calls,_,_ = precision_job
    root = tmp_path/'full-draft'
    assert main(fresh) == 0
    _legacy_fixture(root)
    path = root/'fixture/report.json'
    path.write_text(path.read_text()+'\n')
    before = {path:path.read_bytes() for path in root.rglob('*') if path.is_file()}
    calls.clear()
    assert main([]) == 1
    assert calls == []
    assert all(path.read_bytes() == data for path,data in before.items())
