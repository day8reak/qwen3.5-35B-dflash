"""Standalone draft bundling and evidence gates; never device accuracy evidence."""
import json
from pathlib import Path

import numpy as np
import pytest

from qwen35_moe_dflash import draft_board as board
from qwen35_moe_dflash.hashing import sha256_file
from qwen35_moe_dflash.resources import repository_root


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def record(path,root,**extra):
    return {**extra,'path':str(path.relative_to(root)),'bytes':path.stat().st_size,'sha256':sha256_file(path)}


@pytest.fixture
def bundle(tmp_path,monkeypatch):
    monkeypatch.setenv('AI_RUN_DIR',str(tmp_path))
    compile_dir=tmp_path/'compilation'
    model=compile_dir/'compile-work/generated/model.omc'
    model.parent.mkdir(parents=True)
    model.write_bytes(b'draft-standalone-fixture')
    weights=model.with_name('weights.bin');weights.write_bytes(b'kept-external-weight')
    write_json(compile_dir/'report.json',{'artifact_kind':'qwen35-existing-air-compilation','status':'COMPILED',
        'model_compilation':'PASSED','ddk_id':'fixture','platform':'kirinx90','omc':record(model,compile_dir),
        'generated_artifacts':[record(model,compile_dir),record(weights,compile_dir)]})
    cfg=json.loads((repository_root()/'specs/draft-config.json').read_text())
    fixture=tmp_path/'fixture';fixture.mkdir()
    descriptors,_=board._io(8)
    arrays=[np.full((1,8,16384),.25,dtype='<f2'),np.arange(8,dtype='<i4').reshape(1,8),
            np.asarray([[1,1,1,1,1,1,0,0]],dtype='<i4')]
    records=[]
    for d,v in zip(descriptors,arrays):
        path=fixture/(d['name']+'.bin');v.tofile(path);records.append(record(path,fixture,**d))
    _,outputs=board._io(8)
    top=fixture/'top.bin';np.arange(10,25,dtype='<i4').reshape(1,15).tofile(top)
    hidden=fixture/'hidden.bin';np.full((1,15,2048),.5,dtype='<f2').tofile(hidden)
    write_json(fixture/'report.json',{'artifact_kind':'qwen35-draft-reference','status':'REFERENCE_COMPLETE',
       'sequence_length':8,'draft_config':cfg,'inputs':records,
       'expected_top1':record(top,fixture,**outputs[0]),'expected_hidden':record(hidden,fixture,**outputs[1])})
    output=tmp_path/'bundle'
    manifest=board.prepare_draft_bundle(compile_dir,fixture,output,config=cfg,ddk_id='fixture')
    return output,manifest


def native_result(bundle,result_root,*,evidence='host-fake-hiai'):
    root,m=bundle
    result_root.mkdir()
    outputs=[]
    for r in m['expected']:
        path=result_root/f"output_{r['index']:04d}.bin"
        path.write_bytes((root/r['path']).read_bytes())
        outputs.append(record(path,result_root,**{k:r[k] for k in ('index','name','shape','dtype')}))
    result={'schema_version':1,'scope':'standalone-draft-step','status':'EXECUTED',
       'model_sha256':m['model']['sha256'],'draft_abi_sha256':m['draft_abi']['sha256'],
       'step_plan_sha256':m['step_plan']['sha256'],'ddk_id':'fixture','target_calls':0,'draft_calls':1,
       'runtime_version':'fake','firmware_id':'fake','evidence_kind':evidence,
       'inputs':m['inputs'],'outputs':outputs,'io_diagnostic':{'enabled':True,'process_called':True,
          'inputs':[{'index':r['index'],'readback_matches_source':True,'readback_sha256':r['sha256']} for r in m['inputs']],
          'outputs':[{'index':i,'checked_after_process':True,'object_same':True,'buffer_same':True} for i in range(2)]}}
    write_json(result_root/'result.json',result)
    return result


def test_bundle_copies_all_compiler_payloads_and_strict_draft_wire(bundle):
    root,m=bundle
    assert len(m['model_payloads'])==2
    assert (root/'model/weights.bin').read_bytes()==b'kept-external-weight'
    assert (root/'step.tsv').read_text().startswith('QWEN35_DRAFT_STEP\t1\n')
    manifest,uploads=board._validate_bundle(root)
    assert manifest==m
    assert any(r['remote']=='bundle/model/weights.bin' for r in uploads)
    assert all('target.omc' not in r['remote'] for r in uploads)


def _nested_compiler_payloads(bundle):
    root, manifest = bundle
    compiler = root.parent/'compilation'
    generated = compiler/'compile-work/generated'
    model_dir = generated/'model'
    model_dir.mkdir()
    (generated/'model.omc').rename(model_dir/'model.omc')
    (generated/'weights.bin').rename(model_dir/'SubGraph_0.weight')
    extra = model_dir/'partitions/SubGraph_1.weight'
    extra.parent.mkdir()
    extra.write_bytes(b'second-subgraph-weight')
    paths = [model_dir/'model.omc', model_dir/'SubGraph_0.weight', extra]
    report = json.loads((compiler/'report.json').read_text())
    report.update(omc=record(paths[0],compiler), generated_artifacts=[record(path,compiler) for path in paths],
                  external_data_candidates=[record(path,compiler) for path in paths[1:]])
    write_json(compiler/'report.json',report)
    return compiler, report, manifest


def test_nested_omc_sidecars_keep_exact_relative_layout_through_upload(bundle,tmp_path):
    compiler, report, original = _nested_compiler_payloads(bundle)
    root = tmp_path/'nested-bundle'
    manifest = board.prepare_draft_bundle(compiler,tmp_path/'fixture',root,
        config=original['draft_config'],ddk_id='fixture')
    assert manifest['model']['path']=='model/model/model.omc'
    assert 'model\tmodel/model/model.omc\t' in (root/'step.tsv').read_text()
    _, uploads = board._validate_bundle(root)
    remote = {row['remote']: row for row in uploads}
    for source in report['generated_artifacts']:
        suffix = Path(source['path']).relative_to('compile-work/generated')
        destination = root/'model'/suffix
        assert destination.read_bytes()==(compiler/source['path']).read_bytes()
        assert remote['bundle/model/'+suffix.as_posix()]['sha256']==source['sha256']
    assert not (root/'model.omc').exists()
    assert not (root/'model/SubGraph_0.weight').exists()


@pytest.mark.parametrize('fault',['missing-sidecar','duplicate-path','outside-generated','wrong-primary-hash'])
def test_invalid_compiler_layout_fails_before_bundle_creation(bundle,tmp_path,fault):
    compiler, report, original = _nested_compiler_payloads(bundle)
    if fault=='missing-sidecar':
        report['generated_artifacts'].pop()
    elif fault=='duplicate-path':
        report['generated_artifacts'].append(report['generated_artifacts'][1])
    elif fault=='outside-generated':
        report['generated_artifacts'][1]['path']='compile-work/elsewhere.weight'
    else:
        report['omc']['sha256']='0'*64
    write_json(compiler/'report.json',report)
    destination = tmp_path/'invalid-bundle'
    with pytest.raises(ValueError):
        board.prepare_draft_bundle(compiler,tmp_path/'fixture',destination,
            config=original['draft_config'],ddk_id='fixture')
    assert not destination.exists()


def test_host_fake_requires_explicit_test_opt_in(bundle,tmp_path):
    result=tmp_path/'native';native_result(bundle,result)
    with pytest.raises(ValueError,match='device evidence'):
        board.compare_draft_result(bundle[0],result)
    report=board.compare_draft_result(bundle[0],result,allow_host_fake=True)
    assert report['status']=='EXECUTION_COMPLETE' and report['token_mismatches']==0
    assert report['hidden_diagnostic']['max_abs_error']==0
    assert report['hidden_diagnostic']['acceptance_threshold']=='NOT_DEFINED'


@pytest.mark.parametrize('change',['model','readback','input','calls','output','nonfinite'])
def test_wrong_native_identity_or_buffers_never_pass(bundle,tmp_path,change):
    folder=tmp_path/'native';result=native_result(bundle,folder)
    if change=='model': result['model_sha256']='0'*64
    elif change=='readback': result['io_diagnostic']['inputs'][0]['readback_matches_source']=False
    elif change=='input': result['inputs'][0]={**result['inputs'][0],'sha256':'0'*64}
    elif change=='calls': result['draft_calls']=2
    elif change=='output': result['outputs'][0]['dtype']='float32'
    else:
        path=folder/'output_0001.bin';data=np.fromfile(path,dtype='<f2');data[0]=np.nan;data.tofile(path)
        result['outputs'][1].update(bytes=path.stat().st_size,sha256=sha256_file(path))
    write_json(folder/'result.json',result)
    if change=='nonfinite':
        report=board.compare_draft_result(bundle[0],folder,allow_host_fake=True)
        assert report['status']=='FAILED' and report['numerical_validation']=='FAILED'
    else:
        with pytest.raises(ValueError):
            board.compare_draft_result(bundle[0],folder,allow_host_fake=True)


def test_top1_mismatch_is_failed_not_shape_only_success(bundle,tmp_path):
    folder=tmp_path/'native';result=native_result(bundle,folder)
    path=folder/'output_0000.bin';np.zeros((1,15),dtype='<i4').tofile(path)
    result['outputs'][0].update(sha256=sha256_file(path))
    write_json(folder/'result.json',result)
    report=board.compare_draft_result(bundle[0],folder,allow_host_fake=True)
    assert report['status']=='FAILED' and report['token_mismatches']==15
    assert report['device_execution']=='PASSED'


def test_bundle_mutation_rejected_before_any_hdc(bundle,tmp_path,monkeypatch):
    root,m=bundle
    (root/'input_0000.bin').chmod(0o644)
    (root/'input_0000.bin').write_bytes(b'changed')
    def bad_hdc(*args,**kwargs):
        pytest.fail('must reject local mutation before HDC')
    monkeypatch.setattr(board,'Hdc',bad_hdc)
    report=board.execute_draft_bundle(root,tmp_path/'execution')
    assert report['status']=='FAILED' and report['stage']=='preflight'
    assert report['hdc_invocations']==0 and 'SHA256' in report['error']


def test_no_reference_is_execution_only(bundle,tmp_path):
    root,m=bundle
    m['expected']=[None,None];write_json(root/'manifest.json',m)
    # No corresponding native bytes need change; construct from expected fixtures.
    original=json.loads((root.parent/'fixture/report.json').read_text())
    m_for_native={**m,'expected':[]}
    for key in ('expected_top1','expected_hidden'):
        row=original[key]
        path=root/(key+'.bin')
        m_for_native['expected'].append({**row,**record(path,root)})
    result=tmp_path/'native';native_result((root,m_for_native),result)
    report=board.compare_draft_result(root,result,allow_host_fake=True)
    assert report['status']=='EXECUTION_COMPLETE' and report['numerical_validation']=='NOT_EVALUATED'


def test_export_and_compiler_weight_snapshots_must_match(bundle,tmp_path):
    root,m=bundle
    compiler=tmp_path/'compilation'; fixture=tmp_path/'fixture'; exported=tmp_path/'exported'
    exported.mkdir()
    air=exported/'draft.air';air.write_bytes(b'full-draft-air-fixture')
    weight=exported/'w00001.bin';weight.write_bytes(b'source-weight-fixture')
    rows=[record(air,exported),record(weight,exported)]
    export={'artifact_kind':'qwen35-cpro-draft-air','status':'EXPORTED','prepared_manifest_sha256':'a'*64,
      'graphs':[{'name':'draft','air':'draft.air','outputs':['draft_top1','draft_hidden'],'files':rows}]}
    write_json(exported/'manifest.json',export)
    reference=json.loads((fixture/'report.json').read_text());reference['prepared_manifest_sha256']='a'*64
    write_json(fixture/'report.json',reference)
    report=json.loads((compiler/'report.json').read_text());report['input_snapshots']=[]
    for row in rows:
        path=compiler/'compile-work/generated'/row['path'];path.write_bytes((exported/row['path']).read_bytes())
        report['input_snapshots'].append(record(path,compiler,source_relative=row['path'],role='air' if row['path']=='draft.air' else 'neighbor_data_candidate'))
    write_json(compiler/'report.json',report)
    good=board.prepare_draft_bundle(compiler,fixture,tmp_path/'bound',config=m['draft_config'],ddk_id='fixture',export_dir=exported)
    assert good['source_identity']=='MATCHED_EXPORT_AND_REFERENCE'
    report['input_snapshots'][1]['sha256']='0'*64
    write_json(compiler/'report.json',report)
    with pytest.raises(ValueError,match='compiler snapshot differs'):
        board.prepare_draft_bundle(compiler,fixture,tmp_path/'bad-bound',config=m['draft_config'],ddk_id='fixture',export_dir=exported)
    assert not (tmp_path/'bad-bound').exists()


def test_failed_runtime_retains_native_failure_stage(bundle,tmp_path):
    root,m=bundle
    folder=tmp_path/'native';result=native_result(bundle,folder)
    result.update(status='RUNTIME_FAILED',draft_calls=0,failure_stage='PROCESS',reason='fixture Process failed',outputs=[])
    write_json(folder/'result.json',result)
    report=board.compare_draft_result(root,folder,allow_host_fake=True)
    assert report['device_execution']=='FAILED' and report['failure_stage']=='PROCESS'
    assert report['numerical_validation']=='NOT_EVALUATED'
