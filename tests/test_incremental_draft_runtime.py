"""Host transport/cache tests only; never an incremental device/math result."""
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess

import pytest

from qwen35_moe_dflash.incremental_draft_runtime import VARIANT, validate_abi, bind_compiled
from qwen35_moe_dflash.board_serial_transport import validate_retrieved_tokens, _full, _incremental_counts
from test_board_serial_integration import (mixed_native_programs, make_case, enable_resident_mixed,
    enable_compact_memory, execute, digest, RUNTIME)


def abi(k=15, reverse=False):
    ins = [('target_features','float16',[1,16,16384])]
    ins += [(n,'int32',[1]) for n in ('start_position','valid_rows','anchor','proposal_count')]
    outs = [('draft_top1','int32',[1,k]),('draft_hidden','float16',[1,k,2048])]
    for layer in range(6):
        for kind in ('k','v'):
            ins.append((f'past_{kind}_{layer}','float16',[1,8,192,128]))
            outs.append((f'present_{kind}_{layer}','float16',[1,8,192,128]))
    if reverse: ins.reverse()
    def rows(specs):
        return [dict(index=i,name=n,dtype=dt,shape=shape,bytes=math.prod(shape)*(2 if dt=='float16' else 4))
                for i,(n,dt,shape) in enumerate(specs)]
    return dict(context_rows=16,cache_capacity=192,max_draft_tokens=k,inputs=rows(ins),outputs=rows(outs))


def enable_incremental(case,k,reverse,prompt_length=18):
    spec=abi(k,reverse)
    path=case['files']['draft_abi']
    path.write_text(json.dumps(dict(schema_version=1,variant=VARIANT,
        model_sha256=case['manifest']['draft']['model_source']['sha256'],**spec)))
    case['manifest']['draft'].update(variant=VARIANT,**spec)
    case['manifest']['draft']['abi'].update(bytes=path.stat().st_size,sha256=digest(path),identity=_full(path))
    case['manifest']['max_proposals_per_round']=min(k,2)
    case['manifest']['prompt_token_ids']=list(range(1,prompt_length+1))
    lines=[]
    for line in case['config'].read_text().splitlines():
        row=line.split('\t')
        if row[0]=='draft_abi':row[2:]=[digest(path),str(path.stat().st_size)]
        if row[0]=='prompt':row[1]=','.join(map(str,case['manifest']['prompt_token_ids']))
        if row[0]=='max_proposals':row[1]=str(min(k,2))
        lines.append('\t'.join(row))
    case['config'].write_text('\n'.join(lines)+f'\ndraft_variant\t{VARIANT}\n')


@pytest.mark.parametrize('k,reverse,reject',[(15,False,False),(7,True,False),(3,False,True),(1,True,True)])
def test_actual_incremental_full_loop(tmp_path,mixed_native_programs,monkeypatch,k,reverse,reject):
    case=make_case(tmp_path/'case',mixed_native_programs,dump=False)
    enable_resident_mixed(case);enable_compact_memory(case);enable_incremental(case,k,reverse)
    monkeypatch.setenv('FAKE_HIAI_INCREMENTAL_K',str(k))
    if reverse:monkeypatch.setenv('FAKE_HIAI_INCREMENTAL_REVERSE','1')
    if reject:monkeypatch.setenv('FAKE_HIAI_INCREMENTAL_REJECT','1')
    trace=case['root']/'trace'
    result=execute(case,native_trace=trace)
    assert result.returncode==0,(result.stdout,result.stderr)
    loop=validate_retrieved_tokens(case['manifest'],case['output'],_expected_evidence_kind='host-fake-hiai',
                                   _expected_config_sha256=digest(case['config']))
    assert loop['generated_tokens']==list(range(19,27))
    kv=loop['draft_kv']
    assert kv['catchup_calls']==1
    assert kv['proposal_calls']==loop['completed_rounds']
    assert kv['cache_updates']==loop['counts']['draft_native_calls']==kv['proposal_calls']+1
    assert loop['resident_lifecycle']['draft_loads']==1
    calls=[line for line in trace.read_text().splitlines() if line.startswith('incremental:')]
    assert calls[:2]==['incremental:0:16:0',f'incremental:16:2:{k}']
    assert not (case['output']/'draft').exists()
    # Receipt mutation cannot hide a lost KV bank update/cursor.
    bad=deepcopy(loop);bad['draft_kv']['cache_updates']-=1
    with pytest.raises(ValueError,match='accounting'):
        _incremental_counts(case['manifest'],bad,bad['counts'],bad['full_prefix'],success=True)


def test_invalid_incremental_result_retains_prior_chunk(tmp_path,mixed_native_programs,monkeypatch):
    case=make_case(tmp_path/'failed',mixed_native_programs,dump=False)
    enable_resident_mixed(case);enable_compact_memory(case);enable_incremental(case,15,False)
    monkeypatch.setenv('FAKE_HIAI_INCREMENTAL_K','15')
    result=execute(case,native_fault='incremental-invalid-token')
    assert result.returncode!=0
    loop=validate_retrieved_tokens(case['manifest'],case['output'],_expected_evidence_kind='host-fake-hiai',
        _expected_config_sha256=digest(case['config']),_remote_exitcode=1)
    assert loop['draft_kv']['cached_context_rows']==16
    assert loop['draft_kv']['cache_updates']==1 and loop['counts']['draft_native_calls']==2
    assert loop['generated_tokens']==[19]


@pytest.mark.parametrize('mutation',['duplicate','index','dtype','shape','output-order','capacity'])
def test_strict_native_abi(mutation):
    value=abi(7,True);validate_abi(value)
    if mutation=='duplicate':value['inputs'][1]['name']=value['inputs'][0]['name']
    if mutation=='index':value['inputs'][0]['index']=1
    if mutation=='dtype':value['inputs'][0]['dtype']='float32'
    if mutation=='shape':value['inputs'][0]['shape']=[1,192,8,128]
    if mutation=='output-order':value['outputs'][2],value['outputs'][3]=value['outputs'][3],value['outputs'][2]
    if mutation=='capacity':value['cache_capacity']=8
    with pytest.raises(ValueError):validate_abi(value)


def compiled_fixture(root):
    root.mkdir();(root/'model').mkdir();(root/'model/model.omc').write_bytes(b'HOST TEST NOT A MODEL')
    def record(name):
        p=root/name;return dict(path=name,bytes=p.stat().st_size,sha256=digest(p))
    source=dict(artifact_kind='qwen35-incremental-draft-compiled',status='DRAFT_INCREMENTAL_OMC_COMPILED_ABI_BOUND',
        compiled_abi='PASSED',compiled_precision='PASSED',weight_storage='embedded-stream',ddk_id='test',**abi())
    binding=dict(status='PASSED',omc={'sha256':'a'*64},**abi())
    (root/'compiled-abi.json').write_text(json.dumps(binding))
    packing=dict(status='EMBEDDED',source={'sha256':'a'*64},output=record('model/model.omc'))
    (root/'weight-packing.json').write_text(json.dumps(packing))
    source.update(model=record('model/model.omc'),compiled_abi_file=record('compiled-abi.json'),weight_packing=record('weight-packing.json'))
    (root/'manifest.json').write_text(json.dumps(source));return source


def test_bind_packed_abi_and_gear(tmp_path,monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR",str(tmp_path))
    root=tmp_path/'compiled';compiled_fixture(root)
    bound,pins=bind_compiled(root,tmp_path,ddk_id='test',max_draft_tokens=15,max_proposals=2,total_tokens=64)
    assert bound['variant']==VARIANT and len(bound['inputs'])==17 and len(bound['outputs'])==14
    assert len(pins)==4
    with pytest.raises(ValueError,match='gear'):
        bind_compiled(root,tmp_path,ddk_id='test',max_draft_tokens=7,max_proposals=2,total_tokens=64)
    (root/'model/model.omc').write_bytes(b'CHANGED MODEL')
    with pytest.raises(ValueError,match='model changed'):
        bind_compiled(root,tmp_path,ddk_id='test',max_draft_tokens=15,max_proposals=2,total_tokens=64)


def test_native_cache_boundary_and_atomic_banks(tmp_path):
    binary=tmp_path/'cache-test'
    built=subprocess.run(['g++','-std=c++17','-O2','-I'+str(RUNTIME/'include'),
        str(RUNTIME/'tests/test_incremental_cache.cpp'),'-o',str(binary)],capture_output=True,text=True)
    assert built.returncode==0,built.stderr
    subprocess.run([str(binary)],check=True)


def test_three_context_catchups_do_not_count_as_proposal_rounds(tmp_path,mixed_native_programs,monkeypatch):
    case=make_case(tmp_path/'long-prompt',mixed_native_programs,dump=False)
    enable_resident_mixed(case);enable_compact_memory(case);enable_incremental(case,7,False,prompt_length=49)
    monkeypatch.setenv('FAKE_HIAI_INCREMENTAL_K','7')
    monkeypatch.setenv('FAKE_HIAI_INCREMENTAL_REJECT','1')
    result=execute(case)
    assert result.returncode==0,(result.stdout,result.stderr)
    loop=validate_retrieved_tokens(case['manifest'],case['output'],_expected_evidence_kind='host-fake-hiai',
        _expected_config_sha256=digest(case['config']))
    assert loop['draft_kv']['catchup_calls']==3
    assert loop['draft_kv']['proposal_calls']==loop['completed_rounds']==7
    assert loop['counts']['draft_native_calls']==10
    assert len(loop['generated_tokens'])==8
