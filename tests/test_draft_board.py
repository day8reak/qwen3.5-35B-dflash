"""Standalone draft bundling and evidence gates; never device accuracy evidence."""
import json
import hashlib
from pathlib import Path
import shlex
import sys
import types

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
          'outputs':[{'index':i,'checked_after_process':True,'object_same':True,'buffer_same':True} for i in range(len(m['outputs']))]}}
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


@pytest.fixture
def trace_bundle(bundle, tmp_path):
    _, normal = bundle
    fixture = tmp_path/'fixture'
    report = json.loads((fixture/'report.json').read_text())
    spec = [{'name':'context_fc', 'shape':[1,2], 'dtype':'float16', 'offset':0, 'elements':2},
            {'name':'scores_fp32', 'shape':[1,2], 'dtype':'float32', 'offset':2, 'elements':2}]
    values = fixture/'trace.bin'
    np.asarray([.5, 1., -np.inf, 0.], dtype='<f4').reshape(1,1,4).tofile(values)
    descriptor = board._io(8,spec)[1][1]
    report.update(trace_spec=spec, expected_trace=record(values,fixture,**descriptor))
    write_json(fixture/'report.json',report)
    root = tmp_path/'trace-bundle'
    manifest = board.prepare_draft_bundle(tmp_path/'compilation', fixture, root,
        config=normal['draft_config'], ddk_id='fixture', trace_spec=spec)
    return root, manifest


def test_trace_bundle_has_separate_explicit_wire_and_abi(trace_bundle):
    root, manifest = trace_bundle
    assert manifest['artifact_kind'] == 'qwen35-standalone-draft-trace-bundle'
    assert (root/'step.tsv').read_text().startswith('QWEN35_DRAFT_TRACE_STEP\t1\n')
    abi = json.loads((root/'draft-abi.json').read_text())
    assert abi['artifact_kind'] == 'qwen35-standalone-draft-trace-abi'
    assert board._validate_bundle(root)[0] == manifest


def test_trace_comparison_matches_signed_infinities_and_reports_differences(trace_bundle, tmp_path):
    root, manifest = trace_bundle
    result = tmp_path/'trace-result'
    native_result(trace_bundle, result)
    report = board.compare_draft_result(root,result,allow_host_fake=True)
    assert report['numerical_validation'] == 'TOP1_MATCHED'
    diagnostic = report['trace_diagnostic']
    assert diagnostic['first_nonexact'] is None
    assert diagnostic['nodes'][1]['matching_infinity_count'] == 1
    values = np.fromfile(result/'output_0001.bin',dtype='<f4')
    values[1] = 2.
    values.tofile(result/'output_0001.bin')
    native = json.loads((result/'result.json').read_text())
    native['outputs'][1].update(record(result/'output_0001.bin',result))
    write_json(result/'result.json',native)
    report = board.compare_draft_result(root,result,allow_host_fake=True)
    assert report['trace_diagnostic']['first_nonexact'] == 'context_fc'
    assert report['trace_diagnostic']['nodes'][0]['relative_l2'] > 0
    assert report['trace_diagnostic']['nodes'][0]['acceptance_threshold'] == 'NOT_DEFINED'


def test_trace_top1_mismatch_still_fails(trace_bundle,tmp_path):
    root, _ = trace_bundle
    out = tmp_path/'trace-top1-result'
    native_result(trace_bundle,out)
    path = out/'output_0000.bin'
    np.zeros(15,dtype='<i4').tofile(path)
    native = json.loads((out/'result.json').read_text())
    native['outputs'][0].update(record(path,out))
    write_json(out/'result.json',native)
    report = board.compare_draft_result(root,out,allow_host_fake=True)
    assert report['status'] == 'FAILED' and report['token_mismatches'] == 15


@pytest.mark.parametrize('change', ['gap','duplicate','size','unknown-dtype','oversize'])
def test_trace_spec_rejects_malformed_layout(change):
    spec = [{'name':'x','shape':[2],'dtype':'float16','offset':0,'elements':2}]
    if change == 'gap': spec[0]['offset'] = 1
    elif change == 'duplicate': spec += [{**spec[0], 'offset':2}]
    elif change == 'size': spec[0]['elements'] = 3
    elif change == 'unknown-dtype': spec[0]['dtype'] = 'float64'
    else: spec[0].update(shape=[2**25],elements=2**25)
    with pytest.raises(ValueError): board._trace_spec(spec)


def test_production_bundle_cannot_smuggle_trace_layout(bundle):
    root, manifest = bundle
    manifest['trace_spec'] = [{'name':'x','shape':[1],'dtype':'float16','offset':0,'elements':1}]
    write_json(root/'manifest.json',manifest)
    with pytest.raises(ValueError,match='separate trace bundle'):
        board._validate_bundle(root)


@pytest.fixture
def direct_bundle(bundle, tmp_path):
    _, normal = bundle
    fixture = tmp_path/'fixture'
    reference = json.loads((fixture/'report.json').read_text())
    spec = [
        {'name':'draft_hidden','source':'draft_hidden','shape':[1,15,2048],'dtype':'float16','output_index':1},
        {'name':'context_fc','source':'v41','shape':[1,2,2],'dtype':'float16','output_index':2},
        {'name':'context_fc_alias','source':'v41','shape':[1,2,2],'dtype':'float16','output_index':2},
        {'name':'scores','source':'layer_0_scores','shape':[1,1,2,2],'dtype':'float32','output_index':3},
        {'name':'positions','source':'positions','shape':[1,2],'dtype':'int32','output_index':4}]
    outputs = board._io(8, direct_spec=spec)[1]
    expected = [reference['expected_top1'],reference['expected_hidden']]
    arrays = [np.arange(4,dtype='<f2').reshape(1,2,2),
              np.asarray([.5,-np.inf,0.,1.],dtype='<f4').reshape(1,1,2,2),
              np.asarray([[4,5]],dtype='<i4')]
    for descriptor,array in zip(outputs[2:],arrays):
        path=fixture/f"raw_{descriptor['index']}.bin";array.tofile(path)
        expected.append(record(path,fixture,**descriptor))
    reference.update(direct_spec=spec,expected_direct=expected)
    write_json(fixture/'report.json',reference)
    root=tmp_path/'direct-bundle'
    manifest=board.prepare_draft_bundle(tmp_path/'compilation',fixture,root,
        config=normal['draft_config'],ddk_id='fixture',direct_spec=spec)
    return root,manifest


def test_direct_bundle_keeps_raw_dtypes_and_shared_alias_output(direct_bundle):
    root,manifest=direct_bundle
    assert manifest['artifact_kind']=='qwen35-standalone-draft-direct-bundle'
    assert len(manifest['outputs'])==5 and len(manifest['direct_spec'])==5
    assert [row['dtype'] for row in manifest['outputs']]==['int32','float16','float16','float32','int32']
    assert manifest['outputs'][2]['name']=='v41'
    assert (root/'step.tsv').read_text().startswith('QWEN35_DRAFT_DIRECT_STEP\t1\n')
    assert json.loads((root/'draft-abi.json').read_text())['artifact_kind']=='qwen35-standalone-draft-direct-abi'
    assert board._validate_bundle(root)[0]==manifest


def test_direct_comparison_uses_native_typed_outputs_and_original_gates(direct_bundle,tmp_path):
    root,_=direct_bundle
    out=tmp_path/'direct-native'
    native=native_result(direct_bundle,out)
    result=board.compare_draft_result(root,out,allow_host_fake=True)
    assert result['status']=='EXECUTION_COMPLETE' and result['numerical_validation']=='TOP1_MATCHED'
    nodes=result['direct_diagnostic']['nodes']
    assert all(node['exact_match'] for node in nodes)
    assert nodes[1]['output_index']==nodes[2]['output_index']==2
    assert nodes[3]['matching_infinity_count']==1
    path=out/'output_0000.bin';np.zeros(15,dtype='<i4').tofile(path)
    native['outputs'][0].update(record(path,out));write_json(out/'result.json',native)
    result=board.compare_draft_result(root,out,allow_host_fake=True)
    assert result['status']=='FAILED' and result['token_mismatches']==15


@pytest.mark.parametrize('damage',['gap','alias','same-source','base','dtype','rank','size','count'])
def test_direct_spec_rejects_ambiguous_or_unbounded_outputs(damage):
    spec=[{'name':'x','source':'v1','shape':[1,2],'dtype':'float16','output_index':2}]
    if damage=='gap': spec[0]['output_index']=3
    elif damage=='alias': spec.append({**spec[0],'name':'alias','source':'v2'})
    elif damage=='same-source': spec.append({**spec[0],'name':'alias','output_index':3})
    elif damage=='base': spec[0].update(output_index=1,source='draft_hidden')
    elif damage=='dtype': spec[0]['dtype']='bool'
    elif damage=='rank': spec[0]['shape']=[1]*5
    elif damage=='size': spec[0]['shape']=[2**26]
    else: spec[0]['output_index']=16
    with pytest.raises(ValueError): board._direct_outputs(spec)


def test_production_and_packed_bundles_cannot_smuggle_direct_spec(bundle,trace_bundle):
    for root,manifest in (bundle,trace_bundle):
        manifest['direct_spec']=[];write_json(root/'manifest.json',manifest)
        with pytest.raises(ValueError,match='separate direct bundle'):
            board._validate_bundle(root)


def test_direct_requires_explicit_spec_and_cannot_mix_packed_mode(direct_bundle):
    root,manifest=direct_bundle
    manifest['direct_spec']=None;write_json(root/'manifest.json',manifest)
    with pytest.raises(ValueError,match='explicit probe specification'):
        board._validate_bundle(root)
    with pytest.raises(ValueError,match='mutually exclusive'):
        board._io(8,[{'name':'x','shape':[1],'dtype':'float32','offset':0,'elements':1}],[])


def test_direct_export_spec_must_bind_the_same_raw_outputs(direct_bundle,tmp_path):
    _,manifest=direct_bundle
    exported=tmp_path/'direct-air'
    write_json(exported/'manifest.json',{'artifact_kind':'qwen35-cpro-draft-air','status':'EXPORTED',
        'direct_spec':[], 'graphs':[{'name':'draft','outputs':[row['name'] for row in manifest['outputs']]}]})
    out=tmp_path/'wrong-direct-spec'
    with pytest.raises(ValueError,match='direct AIR specification differs'):
        board.prepare_draft_bundle(tmp_path/'compilation',tmp_path/'fixture',out,
            config=manifest['draft_config'],ddk_id='fixture',export_dir=exported,direct_spec=manifest['direct_spec'])
    assert not out.exists()


@pytest.fixture
def promoted_direct_source(direct_bundle,tmp_path):
    """Mock inspection metadata tests binding only, never compiler accuracy."""
    _,normal=direct_bundle
    declared=json.loads(json.dumps(normal['direct_spec']))
    observed=json.loads(json.dumps(declared))
    for row in observed:
        if row['output_index']==2: row['dtype']='float32'
    compiler=tmp_path/'compilation'
    compiled=json.loads((compiler/'report.json').read_text())
    ddk=tmp_path/'ddk'
    schema=ddk/'lib/schema.so';schema.parent.mkdir(parents=True);schema.write_bytes(b'mock selected schema')
    compiled['ddk_root']=str(ddk)
    write_json(compiler/'report.json',compiled)
    binding={'artifact_kind':'qwen35-draft-direct-output-abi-inspection','status':'SOURCE_DTYPE_CHANGED',
        'declared_spec':declared,'observed_spec':observed,
        'changes':[{'kind':'DIAGNOSTIC_OUTPUT_PROMOTED_FP16_TO_FP32','output_index':2,'source':'v41',
                    'semantic_names':['context_fc','context_fc_alias'],'declared_dtype':'float16',
                    'observed_dtype':'float32','shape':[1,2,2]}],
        'omc':compiled['omc'], 'schema':{'library':str(schema),'library_sha256':sha256_file(schema)},
        'source_binding':'ORDERED_COMPILED_OUTPUTS_CALLER_MUST_BIND_AIR_AND_DECLARED_SPEC'}
    fixture=tmp_path/'fixture'
    reference=json.loads((fixture/'report.json').read_text())
    row=reference['expected_direct'][2]
    array=np.fromfile(fixture/row['path'],dtype='<f2').astype('<f4')
    path=fixture/'promoted_context.bin';array.tofile(path)
    reference['expected_direct'][2]=record(path,fixture,**board._io(8,direct_spec=observed)[1][2])
    reference.update(direct_spec=observed,prepared_manifest_sha256='a'*64)
    write_json(fixture/'report.json',reference)
    exported=tmp_path/'bound-direct-air';exported.mkdir()
    air=exported/'draft.air';air.write_bytes(b'mock direct AIR')
    air_row=record(air,exported)
    staged=compiler/'compile-work/input/draft.air';staged.parent.mkdir(parents=True);staged.write_bytes(air.read_bytes())
    compiled['input_snapshots']=[record(staged,compiler,source_relative='draft.air',role='air')]
    write_json(compiler/'report.json',compiled)
    write_json(exported/'manifest.json',{'artifact_kind':'qwen35-cpro-draft-air','status':'EXPORTED',
        'direct_spec':declared,'prepared_manifest_sha256':'a'*64,'graphs':[{'name':'draft','air':'draft.air',
        'outputs':[row['name'] for row in normal['outputs']],'files':[air_row]}]})
    return compiler,fixture,exported,normal['draft_config'],declared,observed,binding


def test_promoted_direct_output_binding_preserves_declared_precision_and_reads_actual_wire(promoted_direct_source,tmp_path):
    compiler,fixture,exported,config,declared,observed,binding=promoted_direct_source
    root=tmp_path/'promoted-direct-bundle'
    manifest=board.prepare_draft_bundle(compiler,fixture,root,config=config,ddk_id='fixture',
        export_dir=exported,direct_spec=observed,direct_output_binding=binding)
    assert manifest['source_output_precision']=='SOURCE_DTYPE_CHANGED'
    assert manifest['outputs'][2]['dtype']=='float32'
    assert manifest['direct_output_binding']['declared_spec']==declared
    assert board._validate_bundle(root)[0]==manifest
    out=tmp_path/'promoted-native';native_result((root,manifest),out)
    result=board.compare_draft_result(root,out,allow_host_fake=True)
    assert result['device_execution']=='PASSED' and result['token_mismatches']==0
    assert result['status']=='FAILED' and result['numerical_validation']=='FAILED'
    assert result['source_output_precision']=='SOURCE_DTYPE_CHANGED'
    assert all(node['exact_match'] for node in result['direct_diagnostic']['nodes'])


def test_unchanged_direct_output_binding_is_retained_in_the_bundle(promoted_direct_source,tmp_path):
    compiler,fixture,exported,config,_,observed,binding=promoted_direct_source
    binding.update(status='PASSED',declared_spec=observed,changes=[])
    export=json.loads((exported/'manifest.json').read_text());export['direct_spec']=observed
    write_json(exported/'manifest.json',export)
    root=tmp_path/'bound-unchanged-direct'
    manifest=board.prepare_draft_bundle(compiler,fixture,root,config=config,ddk_id='fixture',
        export_dir=exported,direct_spec=observed,direct_output_binding=binding)
    assert manifest['source_output_precision']=='SOURCE_DTYPES_PRESERVED'
    assert board._validate_bundle(root)[0]['direct_output_binding']==binding


@pytest.mark.parametrize('damage',['omc','bytes','status','changes','source','shape','schema','kind'])
def test_direct_output_binding_rejects_unbound_or_unsupported_changes(promoted_direct_source,tmp_path,damage):
    compiler,fixture,exported,config,declared,observed,binding=promoted_direct_source
    if damage=='omc': binding['omc']={**binding['omc'],'sha256':'0'*64}
    elif damage=='bytes': binding['omc']={**binding['omc'],'bytes':binding['omc']['bytes']+1}
    elif damage=='status': binding['status']='PASSED'
    elif damage=='changes': binding['changes']=[]
    elif damage=='source': binding['declared_spec'][1]['source']='wrong'
    elif damage=='shape': binding['declared_spec'][1]['shape']=[1,4]
    elif damage=='schema': binding['schema']['library_sha256']='0'*64
    else: binding['artifact_kind']='unverified'
    with pytest.raises(ValueError):
        board.prepare_draft_bundle(compiler,fixture,tmp_path/('bad-binding-'+damage),config=config,ddk_id='fixture',
            export_dir=exported,direct_spec=observed,direct_output_binding=binding)


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


def _stream_source(bundle, tmp_path):
    """Small immutable compiler inputs; a mocked converter is not OMC evidence."""
    _, manifest = bundle
    compiler = tmp_path/'compilation'
    generated = compiler/'compile-work/generated'
    (generated/'weights.bin').rename(generated/'SubGraph_0.weight')
    ddk = tmp_path/'selected-ddk'
    omg = ddk/'tools/tools_omg/master/omg'
    omg.parent.mkdir(parents=True)
    omg.write_bytes(b'selected-compiler-fixture')
    schema = omg.parent/'lib64/libai_fmk_graph_proto_host.so'
    schema.parent.mkdir()
    schema.write_bytes(b'selected-schema-fixture')
    report = json.loads((compiler/'report.json').read_text())
    sidecar = record(generated/'SubGraph_0.weight', compiler)
    report.update(ddk_root=str(ddk), omg_sha256=sha256_file(omg),
                  requested_options={'save_weights_as_external_data':'true'},
                  generated_artifacts=[report['omc'],sidecar], external_data_candidates=[sidecar])
    write_json(compiler/'report.json',report)
    return compiler, report, manifest['draft_config'], schema


def _mock_stream_converter(monkeypatch, callback):
    module = types.ModuleType('qwen35_moe_dflash.omc_embed_weights')
    module.embed_external_weights = callback
    monkeypatch.setitem(sys.modules,module.__name__,module)


def test_streamed_bundle_keeps_source_provenance_and_uploads_only_new_model(bundle,tmp_path,monkeypatch):
    compiler, compiled, config, schema = _stream_source(bundle,tmp_path)
    source_records = compiled['generated_artifacts']
    source_hashes = {r['path']:sha256_file(compiler/r['path']) for r in source_records}
    calls = []
    def convert(source,output,selected_schema,*,expected_source_sha256,sidecars,**kwargs):
        calls.append((source,output,selected_schema))
        assert selected_schema == schema
        assert expected_source_sha256 == compiled['omc']['sha256']
        assert len(sidecars) == 1
        assert sidecars[0]['path'] == str(compiler/source_records[1]['path'])
        assert sidecars[0]['sha256'] == source_records[1]['sha256']
        output.write_bytes(b'fixture-repacked-model'+Path(sidecars[0]['path']).read_bytes())
        return {'status':'EMBEDDED','source_sha256':expected_source_sha256,
                'sidecars':sidecars,'output':{'bytes':output.stat().st_size,'sha256':sha256_file(output)}}
    _mock_stream_converter(monkeypatch,convert)
    root = tmp_path/'streamed-bundle'
    manifest = board.prepare_draft_bundle(compiler,tmp_path/'fixture',root,config=config,
        ddk_id='fixture',weight_storage='embedded-stream')
    assert len(calls) == 1
    assert manifest['weight_storage'] == 'embedded-stream'
    assert manifest['compiler_report_sha256'] == sha256_file(compiler/'report.json')
    assert manifest['model_payloads'] == [manifest['model']]
    assert manifest['model']['path'] == 'model/model.omc'
    assert manifest['model']['sha256'] != compiled['omc']['sha256']
    assert not list(root.rglob('*.weight'))
    assert record(root/'weight-packing.json',root) == manifest['weight_packing']
    validated, uploads = board._validate_bundle(root)
    assert validated == manifest
    assert [r['remote'] for r in uploads if r['remote'].startswith('bundle/model/')] == ['bundle/model/model.omc']
    assert all(sha256_file(compiler/path) == digest for path,digest in source_hashes.items())
    assert json.loads((compiler/'report.json').read_text())['requested_options']['save_weights_as_external_data'] == 'true'
    packing = json.loads((root/'weight-packing.json').read_text())
    packing['output']['sha256'] = '0'*64
    write_json(root/'weight-packing.json',packing)
    with pytest.raises(ValueError,match='SHA256/size mismatch'):
        board._validate_bundle(root)


@pytest.mark.parametrize('fault',['compiler-changed','foreign-schema','embedded-compiler','sidecar-changed'])
def test_streamed_bundle_rejects_wrong_source_before_converter_or_manifest(bundle,tmp_path,monkeypatch,fault):
    compiler, report, config, schema = _stream_source(bundle,tmp_path)
    if fault == 'compiler-changed':
        report['omg_sha256'] = '0'*64
    elif fault == 'foreign-schema':
        schema = tmp_path/'other-schema.so';schema.write_bytes(b'unrelated-schema')
    elif fault == 'embedded-compiler':
        report['requested_options']['save_weights_as_external_data'] = 'false'
    else:
        (compiler/report['external_data_candidates'][0]['path']).write_bytes(b'changed-weight')
    write_json(compiler/'report.json',report)
    def unexpected(*args,**kwargs):
        pytest.fail('invalid source identity must fail before conversion')
    _mock_stream_converter(monkeypatch,unexpected)
    root = tmp_path/'invalid-stream'
    with pytest.raises(ValueError):
        board.prepare_draft_bundle(compiler,tmp_path/'fixture',root,config=config,
            ddk_id='fixture',weight_storage='embedded-stream',schema_library=schema)
    assert not (root/'manifest.json').exists()


def test_stream_conversion_failure_never_publishes_prepared_bundle(bundle,tmp_path,monkeypatch):
    compiler, report, config, schema = _stream_source(bundle,tmp_path)
    def fail(source,output,selected_schema,**kwargs):
        assert selected_schema == schema
        output.write_bytes(b'partial-write')
        raise ValueError('fixture unsupported selected schema or OMC layout')
    _mock_stream_converter(monkeypatch,fail)
    root = tmp_path/'failed-stream'
    with pytest.raises(ValueError,match='unsupported selected schema'):
        board.prepare_draft_bundle(compiler,tmp_path/'fixture',root,config=config,
            ddk_id='fixture',weight_storage='embedded-stream')
    assert not (root/'manifest.json').exists() and not (root/'step.tsv').exists()
    assert sha256_file(compiler/report['omc']['path']) == report['omc']['sha256']


@pytest.mark.parametrize('fault',['status','hash','bytes'])
def test_streamed_bundle_rejects_converter_receipt_mismatch_before_publishing(bundle,tmp_path,monkeypatch,fault):
    compiler, _, config, _ = _stream_source(bundle,tmp_path)
    def convert(source,output,schema,**kwargs):
        output.write_bytes(b'fixture-repacked-model')
        receipt = {'status':'EMBEDDED','output':{'bytes':output.stat().st_size,'sha256':sha256_file(output)}}
        if fault == 'status':
            receipt['status'] = 'FAILED'
        elif fault == 'hash':
            receipt['output']['sha256'] = '0'*64
        else:
            receipt['output']['bytes'] += 1
        return receipt
    _mock_stream_converter(monkeypatch,convert)
    root = tmp_path/'mismatched-stream'
    with pytest.raises(ValueError):
        board.prepare_draft_bundle(compiler,tmp_path/'fixture',root,config=config,
            ddk_id='fixture',weight_storage='embedded-stream')
    assert not (root/'manifest.json').exists() and not (root/'step.tsv').exists()


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


def _set_hidden_comparison(bundle,folder,expected,actual):
    root,manifest=bundle
    native=native_result(bundle,folder)
    shape=tuple(manifest['outputs'][1]['shape'])
    for values,path in ((expected,root/manifest['expected'][1]['path']),
                        (actual,folder/native['outputs'][1]['path'])):
        path.chmod(0o644)
        np.broadcast_to(np.asarray(values,dtype='<f2'),shape).astype('<f2').tofile(path)
    reference=root/manifest['expected'][1]['path']
    manifest['expected'][1].update(bytes=reference.stat().st_size,sha256=sha256_file(reference))
    output=folder/native['outputs'][1]['path']
    native['outputs'][1].update(bytes=output.stat().st_size,sha256=sha256_file(output))
    write_json(root/'manifest.json',manifest)
    write_json(folder/'result.json',native)
    return native


@pytest.mark.parametrize('actual',[0,.5])
def test_zero_hidden_reference_has_explicit_undefined_ratio_and_serializable_metrics(bundle,tmp_path,actual):
    folder=tmp_path/'native-zero-reference'
    _set_hidden_comparison(bundle,folder,0,actual)
    with np.errstate(all='raise'):
        report=board.compare_draft_result(bundle[0],folder,allow_host_fake=True)
    hidden=report['hidden_diagnostic']
    count=15*2048
    assert hidden['relative_l2'] is None
    assert hidden['relative_l2_status']=='RELATIVE_L2_UNDEFINED_ZERO_REFERENCE'
    assert report['reference_quality']=='DEGENERATE_ZERO_HIDDEN'
    assert hidden['status']==('MISMATCH_ZERO_REFERENCE' if actual else 'MATCHED_ZERO_REFERENCE')
    assert hidden['max_abs_error']==actual
    assert hidden['error_l2']==pytest.approx(actual*np.sqrt(count))
    assert hidden['expected']['l2_norm']==0 and hidden['expected']['l2_norm_status']=='ZERO'
    assert hidden['expected']['zero_count']==count and hidden['expected']['nonzero_count']==0
    assert hidden['actual']['nonzero_count']==(count if actual else 0)
    assert hidden['actual']['finite_min']==actual and hidden['actual']['finite_max']==actual
    assert hidden['exact_match']==(actual==0)
    assert hidden['acceptance_threshold']=='NOT_DEFINED'
    assert report['device_execution']=='PASSED' and report['numerical_validation']=='TOP1_MATCHED'
    assert 'full numerical accuracy' in report['accuracy_scope']
    json.dumps(report,allow_nan=False)


def test_small_nonzero_fp16_reference_keeps_real_finite_ratio(bundle,tmp_path):
    folder=tmp_path/'native-small-reference'
    expected=float(np.nextafter(np.float16(0),np.float16(1)))
    actual=float(np.finfo(np.float16).max)
    _set_hidden_comparison(bundle,folder,expected,actual)
    with np.errstate(all='raise'):
        report=board.compare_draft_result(bundle[0],folder,allow_host_fake=True)
    hidden=report['hidden_diagnostic']
    assert hidden['relative_l2_status']=='DEFINED'
    assert hidden['relative_l2']==pytest.approx((actual-expected)/expected)
    assert hidden['expected']['l2_norm']>0
    assert hidden['actual']['l2_norm']==pytest.approx(actual*np.sqrt(15*2048))
    json.dumps(report,allow_nan=False)


@pytest.mark.parametrize('owner',['expected','actual','both'])
def test_nonfinite_hidden_values_return_failed_json_with_counts_not_exception(bundle,tmp_path,owner):
    folder=tmp_path/('native-nonfinite-'+owner)
    invalid=np.full((1,15,2048),.5,dtype='<f2')
    invalid.flat[:3]=[np.nan,np.inf,-np.inf]
    _set_hidden_comparison(bundle,folder,invalid if owner in {'expected','both'} else .5,
                           invalid if owner in {'actual','both'} else .5)
    with np.errstate(all='raise'):
        report=board.compare_draft_result(bundle[0],folder,allow_host_fake=True)
    assert report['status']=='FAILED' and report['numerical_validation']=='FAILED'
    assert report['device_execution']=='PASSED' and report['token_mismatches']==0
    hidden=report['hidden_diagnostic']
    assert hidden['status']=='NONFINITE_VALUES' and hidden['relative_l2'] is None
    assert hidden['relative_l2_status']=='NONFINITE_VALUES'
    assert hidden['max_abs_error'] is None and hidden['error_l2'] is None
    for name in ('expected','actual'):
        if owner in {name,'both'}:
            summary=hidden[name]
            assert summary['nonfinite_count']==3 and summary['finite_count']==15*2048-3
            assert summary['nan_count']==summary['positive_infinity_count']==summary['negative_infinity_count']==1
            assert summary['l2_norm'] is None and summary['l2_norm_status']=='NONFINITE_VALUES'
            assert summary['finite_min']==summary['finite_max']==.5
    json.dumps(report,allow_nan=False)


def test_all_nonfinite_hidden_has_null_bounds_and_never_looks_zero(bundle,tmp_path):
    folder=tmp_path/'native-all-nonfinite'
    _set_hidden_comparison(bundle,folder,np.nan,np.inf)
    report=board.compare_draft_result(bundle[0],folder,allow_host_fake=True)
    assert report['status']=='FAILED'
    for name in ('expected','actual'):
        summary=report['hidden_diagnostic'][name]
        assert summary['finite_count']==summary['zero_count']==summary['nonzero_count']==0
        assert summary['finite_min'] is None and summary['finite_max'] is None
        assert summary['l2_norm'] is None
    json.dumps(report,allow_nan=False)


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
        for index in range(len(manifest['outputs'])):
            name=f'output_{index:04d}.bin'
            files['results/'+name]=(native_dir/name).read_bytes()
    if fault=='missing_stderr':
        files.pop('draft.stderr')
    elif fault=='missing_output':
        files.pop('results/output_0001.bin')
    elif fault=='missing_extra_output':
        files.pop('results/output_0002.bin')
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


def test_fake_hdc_direct_collects_every_raw_output(direct_bundle,tmp_path,monkeypatch):
    result,requested=execute_with_fake_hdc(direct_bundle,tmp_path,monkeypatch)
    assert result['device_execution']=='PASSED' and result['status']=='EXECUTION_COMPLETE'
    assert {f'results/output_{index:04d}.bin' for index in range(5)} <= set(requested)


def test_fake_hdc_direct_missing_extra_output_never_promotes_execution(direct_bundle,tmp_path,monkeypatch):
    result,_=execute_with_fake_hdc(direct_bundle,tmp_path,monkeypatch,fault='missing_extra_output')
    assert result['status']=='FAILED' and result['device_execution']=='UNVERIFIED'
    assert result.get('transport_error') and result.get('comparison_error')


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
