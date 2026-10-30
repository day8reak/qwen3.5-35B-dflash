"""Actual board-local C++ loop with host-only process doubles; no NPU claims."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import time
import struct
import subprocess
import textwrap

import pytest

ROOT = Path(__file__).resolve().parents[4]
RUNTIME = ROOT / 'targets/cpro/runtime'
V = 248320


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


@pytest.fixture(scope='session')
def binaries(tmp_path_factory):
    directory = tmp_path_factory.mktemp('native-build')
    if os.environ.get('QWEN35_BOARD_SERIAL_TEST_FAKE'):
        return {True: Path(os.environ['QWEN35_BOARD_SERIAL_TEST_FAKE']),False: Path(os.environ['QWEN35_BOARD_SERIAL_TEST_PRODUCTION'])}
    result = {}
    for fake in (True, False):
        binary = directory / ('board-fake' if fake else 'board-production')
        subprocess.run(['g++', '-O2', '-std=c++17', '-Wall', '-Wextra', '-Wpedantic',
            *(['-DQWEN35_BOARD_SERIAL_HOST_FAKE=1'] if fake else []),
            '-I', str(RUNTIME / 'include'), str(RUNTIME / 'src/board_serial_main.cpp'),
            str(RUNTIME / 'src/sha256.cpp'), '-o', str(binary)], check=True,
            capture_output=True, text=True)
        result[fake] = binary
    return result


@pytest.fixture(scope='session')
def tables(tmp_path_factory):
    directory = tmp_path_factory.mktemp('tables')
    weight, scale = directory / 'weight.bin', directory / 'scale.bin'
    with weight.open('wb') as file:
        file.truncate(V * 2048)
    # First four bytes identify each selected token without changing INT8 layout.
    with weight.open('r+b') as file:
        for token in range(256):
            file.seek(token * 2048)
            file.write(struct.pack('<I', token))
    scale.write_bytes(struct.pack('<f', .25) * V)
    return [(weight, sha(weight)), (scale, sha(scale))]


CHILD = r'''#!/usr/bin/env python3
import hashlib,json,os,struct,sys
from pathlib import Path
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
mode=os.environ.get('BOARD_SERIAL_CASE','accept')
args=sys.argv[1:]
if '--draft-step' not in args:
    a=dict(s[2:].split('=',1) for s in args)
    paths=list(map(Path,a['input'].split(',')))
    assert len(paths)==89 and a['modelmanager_itf']=='V2' and a['direct_from_file']=='true'
    p=struct.unpack('<i',paths[2].read_bytes())[0]
    if os.environ.get('BOARD_SERIAL_SCRATCH_TRACE'):
        steps=sorted(Path.cwd().parent.glob('step_*'))
        assert len(steps)==(1 if p==0 else 2), (p,steps)
        if p:
            parent=Path.cwd().parent/f'step_{p-1:04d}'
            assert not (parent/'inputs').exists() and len(list((parent/'outputs').iterdir()))==82
        with open(os.environ['BOARD_SERIAL_SCRATCH_TRACE'],'a') as log:
            log.write(json.dumps(dict(position=p,target_steps=len(steps)))+'\n')
    token=struct.unpack('<I',paths[0].read_bytes()[:4])[0]
    assert struct.unpack('<i',paths[83].read_bytes())[0]==p
    assert paths[84].read_bytes()==struct.pack('<i',0)
    assert paths[86].read_bytes()==struct.pack('<q',1)
    assert paths[88].read_bytes()==struct.pack('<h',1)
    assert paths[87].read_bytes()==struct.pack('<128i',1,*([0]*127))
    assert paths[1].read_bytes()==bytes((p+1)*4)+struct.pack('<f',float('-inf'))*(2047-p)
    assert paths[85].read_bytes()==struct.pack('<f',.25)
    for i in range(80):
        raw=paths[3+i].read_bytes()
        assert raw==struct.pack('<H',p), (p,i,raw)
    if mode=='target-fail' and p==3: sys.exit(7)
    if mode=='past-receipt-change' and p==3:
        old=Path.cwd().parent/'step_0000/receipt.json';old.write_text(old.read_text()+' ')
    if mode=='target-input-change': paths[0].write_bytes(bytes(2048))
    if mode=='target-input-same-bytes': paths[0].write_bytes(paths[0].read_bytes())
    out=Path(a['output_dir'])
    for i in range(82):
        if mode=='missing-output' and i==10: continue
        file=out/f'retrieved_{81-i:03d}'
        if i==0:
            raw=bytearray(V*4);raw[(token+1)*4:(token+2)*4]=struct.pack('<f',2.)
            if mode=='nan-logits':raw[:4]=struct.pack('<f',float('nan'))
        elif i==81: raw=struct.pack('<H',0x3800)*16384
        else: raw=struct.pack('<H',p+1)
        file.write_bytes(raw)
        if mode=='bad-extent' and i==10:file.write_bytes(b'x')
        print(f'output[{i}]      :{file}')
    if mode=='duplicate-log':print(f'output[0] :{out / "retrieved_081"}')
    print('Execute result :Success')
else:
    a=dict(zip(args[::2],args[1::2]));plan=Path(a['--draft-step']);base=plan.parent
    if os.environ.get('BOARD_SERIAL_SCRATCH_TRACE'):
        assert len(list(Path.cwd().parent.glob('call-*')))==1
    lines=[r.split('\t') for r in plan.read_text().splitlines()]
    assert lines[0]==['QWEN35_DRAFT_DIRECT_STEP','1']
    assert (base/lines[1][1]).stat().st_ino==Path(os.environ['DRAFT_SOURCE']).stat().st_ino
    ins=[];arrays=[]
    for r in lines[4:7]:
        _,index,name,dtype,shape,path,digest=r;file=base/path;raw=file.read_bytes();assert h(file)==digest
        ins.append(dict(index=int(index),name=name,dtype=dtype,shape=list(map(int,shape.split(','))),bytes=len(raw),path=path,sha256=digest))
        arrays.append(raw)
    ids=struct.unpack('<64i',arrays[1]);mask=struct.unpack('<64i',arrays[2]);length=sum(mask)
    assert mask==tuple([1]*length+[0]*(64-length)) and arrays[0][length*32768:]==bytes((64-length)*32768)
    assert arrays[0][:length*32768]==struct.pack('<H',0x3800)*(length*16384)
    if mode=='draft-fail':sys.exit(8)
    proposals=list(range(ids[length-1]+1,ids[length-1]+16))
    if mode=='reject':proposals=[200]*15
    if mode=='mixed':proposals[1]=200
    if mode=='bad-token':proposals[0]=248320
    dest=Path(a['--output']);dest.mkdir()
    (dest/'output_0000.bin').write_bytes(struct.pack('<15i',*proposals))
    (dest/'output_0001.bin').write_bytes(bytes(61440))
    outputs=[]
    for i,(name,dtype,shape,size) in enumerate([('draft_top1','int32',[1,15],60),('draft_hidden','float16',[1,15,2048],61440)]):
        path=f'output_{i:04d}.bin';outputs.append(dict(index=i,name=name,dtype=dtype,shape=shape,bytes=size,path=path,sha256=h(dest/path)))
    result=dict(schema_version=1,state_policy='prefix-recompute',status='EXECUTED',scope='standalone-draft-step',evidence_kind='host-fake-hiai',process_id=os.getpid(),target_calls=0,draft_calls=1,
        model_sha256=lines[1][2],draft_abi_sha256=lines[2][2],step_plan_sha256=h(plan),runtime_version=a['--expected-runtime-version'],
        ddk_id=a['--ddk-id'],firmware_id=a['--firmware-id'],model_load_api=a['--model-load-api'],sequence_length=64,inputs=ins,outputs=outputs,
        io_diagnostic=dict(enabled=True,process_called=True,output_fill_byte=0,input_readback_point='AFTER_ALL_INPUT_COPIES_AND_OUTPUT_FILL_BEFORE_PROCESS',outputs=[dict(index=i,checked_after_process=True,object_same=True,buffer_same=True) for i in range(2)],inputs=[dict(index=i,logical_bytes=len(raw),readback_sha256=hashlib.sha256(raw).hexdigest(),readback_matches_source=True) for i,raw in enumerate(arrays)]))
    if mode=='bad-receipt':result['step_plan_sha256']='0'*64
    if mode=='wrong-native-kind':result['evidence_kind']='device-hiai'
    if mode=='bad-readback':result['io_diagnostic']['inputs'][0]['readback_sha256']='0'*64
    if mode=='bad-shape':result['outputs'][0]['shape']=[15]
    if mode=='bad-output-identity':result['io_diagnostic']['outputs'][0]['buffer_same']=False
    if mode=='bad-readback-point':result['io_diagnostic']['input_readback_point']='BEFORE_COPIES'
    if mode=='bad-native-schema':result['schema_version']=True
    if mode=='bad-output-hash':result['outputs'][0]['sha256']='0'*64
    if mode=='draft-input-same-bytes':
        source=base/ins[0]['path'];source.write_bytes(source.read_bytes())
    (dest/'result.json').write_text(json.dumps(result))
'''.replace('V*4', '248320*4')


@pytest.fixture
def case(tmp_path, tables):
    target, draft, abi, tool = [tmp_path / s for s in ('target.omc', 'draft.omc', 'draft-abi.json', 'child.py')]
    target.write_bytes(b'original-target-fixture'); draft.write_bytes(b'full-six-layer-fixture')
    abi.write_text('{"artifact_kind":"qwen35-standalone-draft-direct-abi"}')
    tool.write_text(CHILD);tool.chmod(0o755)
    rows = ['QWEN35_BOARD_SERIAL\t1']
    for key,file,digest in [('target_model',target,sha(target)),('draft_model',draft,sha(draft)),('draft_abi',abi,sha(abi)),
            ('target_tool',tool,sha(tool)),('draft_runner',tool,sha(tool)),('weight_table',tables[0][0],tables[0][1]),('scale_table',tables[1][0],tables[1][1])]:
        rows.append(f'{key}\t{file}\t{digest}\t{file.stat().st_size}')
    strings=dict(target_library_path=str(tmp_path),draft_library_path=str(tmp_path),runtime_version='fixture-runtime',ddk_id='fixture-ddk',firmware_id='fixture-firmware',
        prompt=','.join(map(str,range(1,9))),max_new_tokens='8',max_proposals='2',min_draft_rounds='2',timeout_ms='300000')
    rows += [f'{k}\t{v}' for k,v in strings.items()]
    desc=[('arg1_1','INT8',2048),('arg113_1','FP32',8192),('arg107_1','INT32',4)]
    desc += [(f'RefData_{i}','FP16',2) for i in range(80)]
    desc += [('arg2_1','INT32',4),('arg1238_1','INT32',4),('arg0_1','FP32',4),('arg114_1','INT64',8),('arg14_1','INT32',512),('arg18_1','INT16',2)]
    rows += [f'target_input\t{i}\t{name}\t{dtype}\t{size}' for i,(name,dtype,size) in enumerate(desc)]
    rows += [f'target_output\t{i}\toutput_{i}\t'+('FP32\t993280' if i==0 else 'FP16\t32768' if i==81 else 'FP16\t2') for i in range(82)]
    rows += [f'state\tRefData_{i}\t{i+3}\t{i+1}\t2' for i in range(80)]
    config=tmp_path/'config.tsv';config.write_text('\n'.join(rows)+'\n')
    return dict(config=config,out=tmp_path/'results',draft=draft,root=tmp_path,rows=rows)


def execute(binaries, case, mode='accept', fake=True):
    return subprocess.run([str(binaries[fake]),'--config',str(case['config']),'--output',str(case['out'])],capture_output=True,text=True,
        env=dict(os.environ,BOARD_SERIAL_CASE=mode,DRAFT_SOURCE=str(case['draft'])),timeout=90)


def report(case):
    return json.loads((case['out']/'loop.json').read_text())


@pytest.mark.parametrize('mode', ['accept','reject','mixed'])
def test_complete_serial_loop(binaries,case,mode):
    run=execute(binaries,case,mode);assert run.returncode==0,run.stderr
    result=report(case)
    assert result['evidence_kind']=='host-fake-hiai' and result['device_execution']=='HOST_FAKE_EXECUTED'
    assert result['status']=='BOARD_SERIAL_DFLASH_EXECUTED_UNVALIDATED'
    assert result['generated_tokens']==list(range(9,17))
    assert result['full_prefix']==list(range(1,17))
    assert result['counts']['target_native_calls']==16 and result['counts']['draft_native_calls']>=2
    assert len(result['commits'])==8 and result['commits'][-1]['target_step_index']==15
    timing=result['supervisor_timing']
    assert timing['scope']=='MONOTONIC_WALL_CLOCK_NOT_NPU_INFERENCE'
    assert timing['target_child_processes_timed']==16
    assert timing['draft_child_processes_timed']==result['counts']['draft_native_calls']
    assert all(type(timing[key]) is int and timing[key]>=0 for key in ('startup_source_hash_ms','target_child_process_wall_ms','draft_child_process_wall_ms'))
    assert run.stdout.count('startup-hash-begin')==7 and run.stdout.count('startup-hash-end')==7
    assert run.stdout.count('child-start')==16+result['counts']['draft_native_calls']
    assert run.stdout.count('child-end')==16+result['counts']['draft_native_calls']
    assert result['performance']=='NOT_EVALUATED'
    if mode=='accept':assert result['counts']['bonuses']>0 and result['counts']['corrections']==0
    if mode=='reject':assert result['counts']['accepted']==0 and result['counts']['corrections']==7
    if mode=='mixed':assert result['counts']['accepted']>0 and result['counts']['corrections']>0
    assert len(result['final_state_files'])==80 and len(result['final_target_outputs'])==82
    assert result['final_features']['bytes']==16*32768
    for n,step in enumerate(result['target_steps']):
        receipt=json.loads((case['out']/step['receipt']['path']).read_text())
        assert len(receipt['inputs'])==89 and len(receipt['outputs'])==82 and len(receipt['state_feedback'])==80
        for row in receipt['state_feedback']:
            assert row['input']['sha256']==(hashlib.sha256(b'\0\0').hexdigest() if n==0 else row['parent_output']['sha256'])
    lines=(case['out']/'retrieval-manifest.tsv').read_text().splitlines()
    assert lines[0]=='QWEN35_BOARD_SERIAL_RETRIEVAL\t1'
    for line in lines[1:]:
        kind,path,digest,size=line.split('\t');file=case['out']/path
        assert kind=='file' and file.stat().st_size==int(size) and sha(file)==digest
    assert 'target/step_0000/outputs/retrieved_000' not in '\n'.join(lines)
    assert all(item['path'] in '\n'.join(lines) for item in result['final_target_outputs'])


def assert_compact_inventory(case):
    lines=(case['out']/'retrieval-manifest.tsv').read_text().splitlines()
    assert lines[0]=='QWEN35_BOARD_SERIAL_RETRIEVAL\t1'
    rows=[line.split('\t') for line in lines[1:]]
    assert {row[1] for row in rows}=={'loop.json','progress.json'} and len(rows)==2
    for kind,path,digest,size in rows:
        file=case['out']/path
        assert kind=='file' and file.stat().st_size==int(size) and sha(file)==digest


@pytest.mark.parametrize('mode',['accept','reject','mixed'])
def test_compact_same_tokens_counts_and_bounded_scratch(binaries,case,mode,monkeypatch):
    # Legacy missing dump keeps schema1. A second run explicitly asks for the
    # compact wire, against exactly the same model bytes and input configuration.
    debug=execute(binaries,case,mode);assert debug.returncode==0,debug.stderr
    old=report(case);assert old['schema_version']==1
    case['out']=case['root']/'compact-results'
    case['config'].write_text('\n'.join(case['rows']+['dump\t0'])+'\n')
    trace=case['root']/'scratch-trace.jsonl';monkeypatch.setenv('BOARD_SERIAL_SCRATCH_TRACE',str(trace))
    source_before=case['draft'].stat();source_hash=sha(case['draft'])
    run=execute(binaries,case,mode);assert run.returncode==0,run.stderr
    result=report(case);progress=json.loads((case['out']/'progress.json').read_text())
    assert result['schema_version']==2 and result['dump'] is False and result['retrieval_mode']=='FINAL_TOKENS_ONLY'
    for key in ['generated_tokens','full_prefix','prompt','counts']:
        assert result[key]==old[key]
    assert result['completed_rounds']==result['counts']['draft_native_calls']
    for key in ['target_steps','draft_steps','rounds','commits','final_features','final_state_files','final_target_outputs','generated_token_ids']:
        assert key not in result
    assert_compact_inventory(case)
    assert {p.relative_to(case['out']).as_posix() for p in case['out'].rglob('*') if p.is_file()}=={'loop.json','progress.json','retrieval-manifest.tsv'}
    life=result['child_lifecycle'];total=sum(result['counts'][key] for key in ['target_native_calls','draft_native_calls'])
    assert life['launched_child_count']==life['completed_child_count']==len(life['children'])==total
    assert life['active_child_count']==0 and life['child_status']=='EXITED' and life['last_child_exitcode']==0
    assert len({row['pid'] for row in life['children']})==total and all(row['exitcode']==0 for row in life['children'])
    for key in ['launched_child_count','completed_child_count','active_child_count','last_child_pid','last_child_exitcode','child_status']:
        assert progress[key]==life[key]
    assert len(trace.read_text().splitlines())==16
    source_after=case['draft'].stat()
    assert source_after.st_ino==source_before.st_ino and source_after.st_nlink==source_before.st_nlink
    assert source_after.st_mtime_ns==source_before.st_mtime_ns and sha(case['draft'])==source_hash
    acceptance=result['acceptance'];assert acceptance['accepted']==result['counts']['accepted']
    assert acceptance['considered']==result['counts']['considered']
    assert acceptance['rate']==pytest.approx(acceptance['accepted']/acceptance['considered'])
    phases=result['phase_latency'];assert phases['timing_scope']=='CHILD_WALL_INCLUDES_LOAD_RUN_UNLOAD'
    assert phases['inference_only_status']=='NOT_PROVIDED'
    assert phases['prompt_ingest']['mechanism']=='DECODE1_SERIAL_PROMPT_INGESTION'
    assert phases['prompt_ingest']['calls']==8 and phases['decode']['calls']==8
    assert phases['draft']['calls']==result['counts']['draft_native_calls']
    for key in ['prompt_ingest','decode','draft']:
        assert phases[key]['mean_ms']==pytest.approx(phases[key]['total_ms']/phases[key]['calls'])
    assert phases['prompt_ingest']['total_ms']+phases['decode']['total_ms']==result['supervisor_timing']['target_child_process_wall_ms']


@pytest.mark.parametrize('mode',['target-fail','missing-output','target-input-same-bytes','draft-fail','bad-output-hash'])
def test_compact_failure_retains_current_scratch_only_small_envelope(binaries,case,mode):
    case['config'].write_text('\n'.join(case['rows']+['dump\t0'])+'\n')
    run=execute(binaries,case,mode);assert run.returncode==1
    result=report(case);assert result['schema_version']==2 and result['status']=='FAILED'
    assert_compact_inventory(case)
    assert len(list((case['out']/'target').glob('step_*'))) in (1,2)
    assert list(case['out'].rglob('input_*.bin'))  # failed work was not swept
    assert not (case['out']/'final-features.bin').exists()
    assert not list(case['out'].rglob('receipt.json'))
    assert all(name not in result for name in ['target_steps','draft_steps','rounds','commits','final_state_files'])
    life=result['child_lifecycle']
    assert life['active_child_count']==0 and life['completed_child_count']==life['launched_child_count']
    assert life['child_status']=='EXITED' and life['last_child_pid']==result['failed_child_pid']
    if mode=='target-fail':
        assert (case['out']/'target/step_0002/outputs').is_dir()
        assert (case['out']/'target/step_0003/inputs').is_dir()


@pytest.mark.parametrize('rows',[['dump\t2'],['dump\tfalse'],['dump\t0','dump\t1']])
def test_dump_option_strict_singleton(binaries,case,rows):
    case['config'].write_text('\n'.join(case['rows']+rows)+'\n')
    run=execute(binaries,case);assert run.returncode==1
    assert 'dump must be one explicit0/1 singleton' in report(case)['error']


def test_explicit_dump_one_retains_full_schema1(binaries,case):
    case['config'].write_text('\n'.join(case['rows']+['dump\t1'])+'\n')
    run=execute(binaries,case);assert run.returncode==0,run.stderr
    result=report(case)
    assert result['schema_version']==1 and result['generated_tokens']==list(range(9,17))
    assert len(result['target_steps'])==16 and len(result['final_state_files'])==80
    assert (case['out']/'final-features.bin').is_file()


@pytest.mark.parametrize('mode,error', [
    ('target-fail','target child failed'),('draft-fail','draft child failed'),
    ('past-receipt-change','pinned file changed'),
    ('missing-output','complete explicit'),('duplicate-log','duplicate logged'),('bad-extent','extent mismatch'),
    ('target-input-change','pinned file changed'),('target-input-same-bytes','pinned file changed'),
    ('draft-input-same-bytes','pinned file changed'),('bad-output-hash','SHA256 mismatch'),('nan-logits','nonfinite target logits'),
    ('bad-token','draft token outside'),('bad-receipt','model/ABI/environment'),
    ('wrong-native-kind','execution identity'),('bad-readback','actual copied input'),('bad-shape','output ABI'),
    ('bad-output-identity','output buffer identity'),('bad-readback-point','readback point/fill'),('bad-native-schema','JSON integer')])
def test_failure_stops_without_retry(binaries,case,mode,error):
    run=execute(binaries,case,mode);assert run.returncode==1
    result=report(case);assert result['status']=='FAILED' and error in result['error'], (run.stderr,result)
    assert result['performance']=='NOT_EVALUATED' and result['numerical_validation']=='NOT_RUN'
    assert result['counts']['target_native_calls']<=9
    assert result['counts']['draft_native_calls']<=1
    assert len(list((case['out']/'target').glob('step_*')))==result['counts']['target_native_calls']
    if mode=='past-receipt-change':
        # A changed prior receipt cannot be relabelled as a new valid inventory.
        assert not (case['out']/'retrieval-manifest.tsv').exists()
        assert 'failure evidence incomplete' in run.stderr
    else:
        assert (case['out']/'retrieval-manifest.tsv').exists()


def test_production_refuses_fake_native_receipt(binaries,case):
    run=execute(binaries,case,fake=False)
    assert run.returncode==1 and 'execution identity' in report(case)['error']


@pytest.mark.parametrize('change', ['duplicate','unsafe','state-duplicate','unknown','capacity','negative','bool','table-size','min-rounds'])
def test_config_rejected_before_children(binaries,case,change):
    rows=case['rows'][:]
    if change=='duplicate':rows.append('max_new_tokens\t8')
    elif change=='unsafe':rows=[r.replace(str(case['root']),str(case['root'])+'/../'+case['root'].name) if r.startswith('target_tool\t') else r for r in rows]
    elif change=='state-duplicate':rows[-1]=rows[-2]
    elif change=='unknown':rows.append('surprise\t1')
    elif change=='capacity':rows=[r.replace('max_new_tokens\t8','max_new_tokens\t64') for r in rows]
    elif change=='negative':rows=[r.replace('max_proposals\t2','max_proposals\t-1') for r in rows]
    elif change=='bool':rows=[r.replace('max_proposals\t2','max_proposals\ttrue') for r in rows]
    elif change=='min-rounds':rows=[r.replace('min_draft_rounds\t2','min_draft_rounds\t8') for r in rows]
    elif change=='table-size':rows=[r.rsplit('\t',1)[0]+'\t1' if r.startswith('weight_table\t') else r for r in rows]
    case['config'].write_text('\n'.join(rows)+'\n')
    run=execute(binaries,case);assert run.returncode==1 and report(case)['stage']=='validate-config-and-source-hashes'
    assert not (case['out']/'target').exists()


def test_existing_output_never_reused(binaries,case):
    case['out'].mkdir();(case['out']/'saved').write_text('retain')
    run=execute(binaries,case)
    assert run.returncode==1 and (case['out']/'saved').read_text()=='retain'


def test_hangup_after_start_does_not_stop_board_loop(binaries,case):
    env=dict(os.environ,BOARD_SERIAL_CASE='accept',DRAFT_SOURCE=str(case['draft']))
    with subprocess.Popen([str(binaries[True]),'--config',str(case['config']),'--output',str(case['out'])],
            stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,env=env,text=True) as proc:
        deadline=time.monotonic()+30
        while not (case['out']/'progress.json').exists():
            assert proc.poll() is None and time.monotonic()<deadline
            time.sleep(.01)
        early=json.loads((case['out']/'progress.json').read_text())
        assert early['supervisor_pid']==proc.pid
        os.kill(proc.pid,signal.SIGHUP)
        _,stderr=proc.communicate(timeout=90)
    assert proc.returncode==0,stderr
    result=report(case)
    assert result['counts']['target_native_calls']==16 and len(result['generated_tokens'])==8
    assert result['evidence_kind']=='host-fake-hiai'


@pytest.mark.parametrize('dump',[False,True])
def test_pid_publication_failure_keeps_current_child_uncertain(case,tmp_path,dump):
    # Force the actual parent open(tool.pid) to fail after fork. The real
    # production source, rather than a mock Run(), must publish uncertainty.
    shim=tmp_path/'open_failure.cpp'
    shim.write_text(r'''
#include <cstdarg>
#include <cstring>
#include <cerrno>
#include <fcntl.h>
#include <sys/types.h>
extern "C" int __real_open(const char*,int,...);
extern "C" int __wrap_open(const char* path,int flags,...) {
  mode_t mode=0;if(flags&O_CREAT){va_list args;va_start(args,flags);mode=va_arg(args,int);va_end(args);}
  const char* suffix="/tool.pid";auto n=std::strlen(path),m=std::strlen(suffix);
  if(n>=m&&std::strcmp(path+n-m,suffix)==0&&(flags&O_CREAT)){errno=ENOSPC;return -1;}
  return __real_open(path,flags,mode);
}
''')
    binary=tmp_path/'pid-write-failure'
    case['config'].write_text('\n'.join(case['rows']+[f'dump\t{int(dump)}'])+'\n')
    subprocess.run(['g++','-O2','-std=c++17','-DQWEN35_BOARD_SERIAL_HOST_FAKE=1',
        '-I',str(RUNTIME/'include'),str(RUNTIME/'src/board_serial_main.cpp'),
        str(RUNTIME/'src/sha256.cpp'),str(shim),'-Wl,--wrap=open','-o',str(binary)],check=True,capture_output=True,text=True)
    run=execute({True:binary},case)
    assert run.returncode==1
    failed=report(case);progress=json.loads((case['out']/'progress.json').read_text())
    assert failed['status']=='FAILED' and 'tool.pid' in failed['error']
    assert type(failed['failed_child_pid']) is int and failed['failed_child_pid']>0
    assert failed['failed_child_exitcode'] is None
    assert progress['last_child_pid']==failed['failed_child_pid'] and progress['last_child_exitcode'] is None
    assert progress['child_status']=='RUNNING'
    assert not (case['out']/'target/step_0000/tool.pid').exists()
    assert not (case['out']/'target/step_0000/tool.exitcode').exists()
    assert failed['counts']['target_native_calls']==0 and failed['generated_tokens']==[]
    if not dump:
        assert_compact_inventory(case)
        assert len(list((case['out']/'target/step_0000/inputs').glob('input_*.bin')))==89
        life=failed['child_lifecycle']
        assert life['active_child_count']==1 and life['launched_child_count']==1 and life['completed_child_count']==0
        assert life['children']==[dict(pid=failed['failed_child_pid'],exitcode=None)]
