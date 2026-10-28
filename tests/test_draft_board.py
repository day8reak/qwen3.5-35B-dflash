"""Standalone draft bundling and evidence gates; never device accuracy evidence."""
import json
import hashlib
from pathlib import Path
import shlex

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


def test_single_embedded_omc_bundle_needs_no_external_weight_payload(bundle, tmp_path):
    _, original = bundle
    compiler = tmp_path/'compilation'
    report = json.loads((compiler/'report.json').read_text())
    report.update(generated_artifacts=[report['omc']], external_data_candidates=[],
                  requested_options={'save_weights_as_external_data':'false'})
    write_json(compiler/'report.json', report)
    root = tmp_path/'embedded-bundle'
    manifest = board.prepare_draft_bundle(compiler, tmp_path/'fixture', root,
        config=original['draft_config'], ddk_id='fixture')
    assert manifest['model_payloads'] == [manifest['model']]
    assert manifest['model']['sha256'] == report['omc']['sha256']
    _, uploads = board._validate_bundle(root)
    model_uploads = [row for row in uploads if row['remote'].startswith('bundle/model/')]
    assert len(model_uploads) == 1
    assert model_uploads[0]['remote'] == 'bundle/model/model.omc'
    assert not (root/'model/weights.bin').exists()


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


def execute_with_fake_hdc(bundle,tmp_path,monkeypatch,*,failure_stage=None,fault=None,model_load_api='legacy-buffer'):
    """Exercise orchestration and real hash-checked receive with a transport double.

    Native device markers below are fabricated test data, never board evidence.
    """
    root, manifest=bundle
    manifest['source_identity']='MATCHED_EXPORT_AND_REFERENCE'
    write_json(root/'manifest.json',manifest)
    native_dir=tmp_path/'fake-native-source'
    native=native_result((root,manifest),native_dir,evidence='device-hiai')
    native.update(runtime_version='test-runtime',firmware_id='test-firmware',process_id=4321)
    if model_load_api=='file-v2':
        native.update(model_load_api=model_load_api,runtime_stage='MODEL_MANAGER_INIT' if failure_stage else 'COMPLETE')
    rc=1 if failure_stage else 0
    if failure_stage:
        native.update(status='RUNTIME_FAILED',draft_calls=0,failure_stage=failure_stage,
                      reason='HIAI '+failure_stage+' fixture failure',outputs=[],
                      expected_runtime_version='test-runtime')
    if fault=='runtime':
        native['expected_runtime_version' if failure_stage else 'runtime_version']='wrong-runtime'
    elif fault=='firmware':
        native['firmware_id']='wrong-firmware'
    elif fault=='model':
        native['model_sha256']='0'*64
    elif fault=='plan':
        native['step_plan_sha256']='0'*64
    elif fault=='output_hash':
        native['outputs'][0]['sha256']='0'*64
    elif fault=='loader_mismatch':
        native['model_load_api']='legacy-buffer' if model_load_api=='file-v2' else 'file-v2'
    elif fault=='loader_missing':
        native.pop('model_load_api',None)
    elif fault=='top1':
        path=native_dir/'output_0000.bin'
        np.zeros((1,15),dtype='<i4').tofile(path)
        native['outputs'][0].update(sha256=sha256_file(path))
    write_json(native_dir/'result.json',native)
    files={'draft.stdout':b'','draft.stderr':b'native fixture stderr\n',
           'draft.exitcode':str(rc).encode()+b'\n',
           'results/result.json':(native_dir/'result.json').read_bytes()}
    if not failure_stage:
        for index in range(2):
            name=f'output_{index:04d}.bin'
            files['results/'+name]=(native_dir/name).read_bytes()
    if fault=='missing_stderr':
        files.pop('draft.stderr')
    elif fault=='missing_output':
        files.pop('results/output_0001.bin')
    elif fault=='missing_native':
        files.pop('results/result.json')
    elif fault=='exit_mismatch':
        files['draft.exitcode']=b'7\n'
    elif fault=='invalid_exit':
        files['draft.exitcode']=b'\xff\n'
    elif fault=='native_exit_conflict':
        rc=0 if failure_stage else 1
        files['draft.exitcode']=str(rc).encode()+b'\n'
    requested=[]
    class FakeHdc:
        def __init__(self,executable,sn,bip,out,timeout):
            self.count=0; self.timeout=timeout; self.root=None; self.files={}
        def shell(self,command,label,**kwargs):
            self.count+=1
            if label=='create-root':
                self.root=Path(shlex.split(command)[1])
            elif label=='runtime-probe':
                self.files.update({'runtime.json':json.dumps({'npu_supported':True,'runtime_version':'test-runtime'}).encode(),
                                   'runtime.stderr':b'', 'runtime.exitcode':b'0\n'})
            elif label=='execute':
                self.files.update(files)
                assert '--timeout-ms 1250' in command
                assert '--model-load-api '+model_load_api in command
                if fault=='hdc_disconnect':
                    raise RuntimeError('HDC fixture connection lost after remote execution')
                return rc,''
            elif label=='hash-result':
                tokens=shlex.split(command)
                path=Path(tokens[tokens.index('-f')+1].rstrip(';'))
                name=path.relative_to(self.root).as_posix()
                requested.append(name)
                if name not in self.files:
                    return 0,'__MISSING__'
                return 0,hashlib.sha256(self.files[name]).hexdigest()+'  '+str(path)
            return 0,''
        def receive(self,remote,local,**kwargs):
            self.count+=1
            name=Path(remote).relative_to(self.root).as_posix()
            payload=self.files[name]
            if ((fault=='delivery_hash' and name=='results/result.json')
                    or (fault=='stderr_hash' and name=='draft.stderr')):
                payload+=b'tampered'
            Path(local).write_bytes(payload)
    def runner(out,timeout):
        path=out/'fake-runner';path.write_bytes(b'not-a-device-binary');return path
    for key in ('CPRO_DDK_ID','HIAI_RUNTIME_VERSION'):
        monkeypatch.delenv(key,raising=False)
    for key,value in {'SN':'fake-serial','BIP':'127.0.0.1:1234','FIRMWARE_ID':'test-firmware',
                      'BOARD_HIAI_LIB_DIR':'/fake-existing-board-libraries'}.items():
        monkeypatch.setenv(key,value)
    monkeypatch.setattr(board,'Hdc',FakeHdc)
    monkeypatch.setattr(board,'_build_runner',runner)
    monkeypatch.setattr(board,'_upload',lambda *a,**k: None)
    from qwen35_moe_dflash import draft_hilog
    captures=[]
    def capture(hdc,remote,out,*,receive,recorded_pid=None):
        captures.append(recorded_pid)
        assert receive is board._receive
        if fault=='hilog_failure':
            raise RuntimeError('fixture log capture failed')
        return {'status':'CAPTURED','recorded_pid':recorded_pid}
    monkeypatch.setattr(draft_hilog,'capture_draft_hilog',capture)
    output=tmp_path/'fake-hdc-execution'
    result=board.execute_draft_bundle(root,output,timeout_seconds=1.25,model_load_api=model_load_api)
    assert json.loads((output/'result.json').read_text())==result
    assert len(captures)==(1 if result['status']=='FAILED' else 0)
    return result, requested


@pytest.mark.parametrize('failure_stage',['LOAD_OR_SETUP','INPUT_BINDING','PROCESS'])
def test_fake_hdc_native_failure_preserves_cause_without_success_outputs(bundle,tmp_path,monkeypatch,failure_stage):
    result,requested=execute_with_fake_hdc(bundle,tmp_path,monkeypatch,failure_stage=failure_stage)
    assert result['status']=='FAILED' and result['device_execution']=='FAILED'
    assert result['native_status']=='RUNTIME_FAILED' and result['failure_stage']==failure_stage
    assert result['error']=='HIAI '+failure_stage+' fixture failure'
    assert result['numerical_validation']=='NOT_EVALUATED'
    assert result['remote_exitcode']==1
    assert result['native_process_id']==4321 and result['hilog']['recorded_pid']==4321
    assert 'transport_error' not in result and 'comparison_error' not in result
    assert not any(name.startswith('results/output_') for name in requested)


def test_fake_hdc_success_checks_outputs_before_promoting_execution(bundle,tmp_path,monkeypatch):
    result,requested=execute_with_fake_hdc(bundle,tmp_path,monkeypatch)
    assert result['status']=='EXECUTION_COMPLETE' and result['device_execution']=='PASSED'
    assert result['numerical_validation']=='TOP1_MATCHED' and result['native_status']=='EXECUTED'
    assert {'results/output_0000.bin','results/output_0001.bin'} <= set(requested)
    assert not any(key in result for key in ('error','comparison_error','transport_error'))


def test_fake_hdc_top1_failure_is_reported_separately_from_transport(bundle,tmp_path,monkeypatch):
    result,_=execute_with_fake_hdc(bundle,tmp_path,monkeypatch,fault='top1')
    assert result['status']=='FAILED' and result['device_execution']=='PASSED'
    assert result['numerical_validation']=='FAILED' and 'Top1 differs' in result['error']
    assert 'transport_error' not in result and 'comparison_error' not in result


@pytest.mark.parametrize('failure_stage',[None,'PROCESS'])
@pytest.mark.parametrize('fault',['runtime','firmware','model','plan'])
def test_fake_hdc_unbound_native_records_never_promote_status_or_cause(bundle,tmp_path,monkeypatch,failure_stage,fault):
    result,_=execute_with_fake_hdc(bundle,tmp_path,monkeypatch,failure_stage=failure_stage,fault=fault)
    assert result['status']=='FAILED' and result['device_execution']=='UNVERIFIED'
    assert result['numerical_validation']=='NOT_EVALUATED' and result['comparison_error']
    assert 'native_status' not in result and 'failure_stage' not in result and 'error' not in result
    assert result['hilog']['recorded_pid'] is None


@pytest.mark.parametrize('fault',['hdc_disconnect','missing_stderr','stderr_hash','exit_mismatch','invalid_exit','native_exit_conflict'])
@pytest.mark.parametrize('failure_stage',[None,'LOAD_OR_SETUP'])
def test_fake_hdc_transport_failure_is_separate_and_keeps_verified_native_cause(bundle,tmp_path,monkeypatch,fault,failure_stage):
    result,_=execute_with_fake_hdc(bundle,tmp_path,monkeypatch,failure_stage=failure_stage,fault=fault)
    assert result['status']=='FAILED' and result['transport_error']
    assert result['numerical_validation']=='NOT_EVALUATED'
    if failure_stage:
        assert result['device_execution']=='FAILED' and result['failure_stage']==failure_stage
        assert result['error']=='HIAI '+failure_stage+' fixture failure'
    else:
        assert result['device_execution']=='UNVERIFIED'
    assert 'comparison_error' not in result
    if fault=='stderr_hash':
        assert result['retrieval_errors'][0]['path']=='draft.stderr'


@pytest.mark.parametrize('fault',['missing_native','missing_output','delivery_hash','output_hash'])
def test_fake_hdc_missing_or_corrupt_evidence_never_passes(bundle,tmp_path,monkeypatch,fault):
    result,_=execute_with_fake_hdc(bundle,tmp_path,monkeypatch,fault=fault)
    assert result['status']=='FAILED' and result['device_execution']=='UNVERIFIED'
    assert result['numerical_validation']=='NOT_EVALUATED' and result['comparison_error']
    if fault!='output_hash':
        assert result['transport_error']
    if fault=='delivery_hash':
        assert result['retrieval_errors'][0]['path']=='results/result.json'
        assert 'SHA256 mismatch' in result['retrieval_errors'][0]['error']


def test_diagnostic_log_failure_does_not_mask_native_load_failure(bundle,tmp_path,monkeypatch):
    result,_=execute_with_fake_hdc(bundle,tmp_path,monkeypatch,failure_stage='LOAD_OR_SETUP',fault='hilog_failure')
    assert result['status']=='FAILED' and result['device_execution']=='FAILED'
    assert result['error']=='HIAI LOAD_OR_SETUP fixture failure'
    assert result['hilog']['status']=='FAILED' and 'fixture log capture failed' in result['hilog']['error']


@pytest.mark.parametrize('failure_stage',[None,'LOAD_OR_SETUP','PROCESS'])
def test_file_v2_runs_the_same_bundle_once_and_preserves_native_stage(bundle,tmp_path,monkeypatch,failure_stage):
    result,requested=execute_with_fake_hdc(bundle,tmp_path,monkeypatch,
        failure_stage=failure_stage,model_load_api='file-v2')
    assert result['model_load_api']=='file-v2'
    assert result['runtime_stage']==('MODEL_MANAGER_INIT' if failure_stage else 'COMPLETE')
    assert result['execution_timeout']['native_process_timeout_ms'] is None
    assert result['execution_timeout']['host_hdc_seconds']==1.25
    assert 'native_timeout_ms' not in result
    assert result['status']==('FAILED' if failure_stage else 'EXECUTION_COMPLETE')
    assert result['numerical_validation']==('NOT_EVALUATED' if failure_stage else 'TOP1_MATCHED')
    if failure_stage:
        assert result['hilog']['recorded_pid']==4321
        assert not any(name.startswith('results/output_') for name in requested)


@pytest.mark.parametrize('fault',['loader_mismatch','loader_missing'])
@pytest.mark.parametrize('failure_stage',[None,'LOAD_OR_SETUP'])
def test_v2_requires_bound_route_even_when_model_and_outputs_match(bundle,tmp_path,monkeypatch,fault,failure_stage):
    result,_=execute_with_fake_hdc(bundle,tmp_path,monkeypatch,
        failure_stage=failure_stage,model_load_api='file-v2',fault=fault)
    assert result['status']=='FAILED' and result['device_execution']=='UNVERIFIED'
    assert 'load API differs' in result['comparison_error']
    assert 'runtime_stage' not in result and 'native_status' not in result


def test_invalid_loader_is_rejected_before_creating_output(bundle,tmp_path,monkeypatch):
    def unexpected(*args,**kwargs):
        pytest.fail('invalid load API must fail before HDC')
    monkeypatch.setattr(board,'Hdc',unexpected)
    output=tmp_path/'bad-loader'
    with pytest.raises(ValueError,match='model_load_api'):
        board.execute_draft_bundle(bundle[0],output,model_load_api='buffer-then-path')
    assert not output.exists()
