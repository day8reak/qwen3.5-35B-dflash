"""Actual standalone CLI against host HIAI double: transport, never DFlash math."""
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

runner=Path(sys.argv[1]).resolve()
base=Path(sys.argv[2]);base.mkdir(parents=True,exist_ok=True)
root=Path(tempfile.mkdtemp(prefix='draft-',dir=base))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(root/'model.omc').write_bytes(b'draft-standalone-fixture')
(root/'draft-abi.json').write_text('{}')
features=root/'features.bin';features.write_bytes(struct.pack('<H',0x3800)*(8*16384))
ids=root/'ids.bin';ids.write_bytes(struct.pack('<8i',*range(8)))
mask=root/'mask.bin';mask.write_bytes(struct.pack('<8i',1,1,1,1,1,1,0,0))

def plan(model_relative='model.omc'):
    rows=['QWEN35_DRAFT_STEP\t1','model\t'+model_relative+'\t'+sha(root/model_relative),
          'draft_abi\tdraft-abi.json\t'+sha(root/'draft-abi.json'),'tokens\t8\t0']
    for i,(name,dtype,shape,file) in enumerate([
       ('target_features','float16','1,8,16384',features),('input_ids','int32','1,8',ids),
       ('attention_mask','int32','1,8',mask)]):
        rows.append(f'input\t{i}\t{name}\t{dtype}\t{shape}\t{file.name}\t{sha(file)}')
    rows+=['output\t0\tdraft_top1\tint32\t1,15','output\t1\tdraft_hidden\tfloat16\t1,15,2048']
    (root/'step.tsv').write_text('\n'.join(rows)+'\n')

args=[str(runner),'--draft-step',str(root/'step.tsv'),'--expected-runtime-version','FAKE-HIAI-X90',
      '--ddk-id','fake','--firmware-id','fake']
plan()
for failure in ('','load','run'):
    out=root/('result-'+(failure or 'success'));trace=root/(out.name+'.trace')
    with subprocess.Popen(args+['--output',str(out)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
        env=dict(os.environ,FAKE_HIAI_FAILURE=failure,FAKE_HIAI_TRACE=str(trace))) as proc:
        stdout,stderr=proc.communicate()
    assert 'LEAK' not in stderr,stderr
    result=json.loads((out/'result.json').read_text())
    assert type(result['process_id']) is int and result['process_id']==proc.pid,result
    assert result['scope']=='standalone-draft-step' and result['target_calls']==0
    assert result['evidence_kind']=='host-fake-hiai'
    events=trace.read_text().splitlines()
    assert events.count('load')==1,events
    if failure:
        assert proc.returncode==1 and result['status']=='RUNTIME_FAILED',(stderr,result)
    else:
        assert proc.returncode==0,(stderr,result)
        assert result['draft_calls']==1 and result['status']=='EXECUTED'
        assert events==['load','process','unload'],events
        assert all(x['readback_matches_source'] for x in result['io_diagnostic']['inputs'])
        assert struct.unpack('<15i',(out/'output_0000.bin').read_bytes())==tuple(range(10,25))
        assert len((out/'output_0001.bin').read_bytes())==15*2048*2

# Exercise the real plan reader and mmap loader with a nested OMC from an
# unrelated working directory. External-weight interpretation belongs to the
# proprietary runtime and is deliberately not simulated by this host double.
nested=root/'model/model.omc'
nested.parent.mkdir()
nested.write_bytes((root/'model.omc').read_bytes())
unrelated=root/'unrelated-cwd';unrelated.mkdir()
plan('model/model.omc')
out=root/'nested-result'
proc=subprocess.run(args+['--output',str(out)],capture_output=True,text=True,cwd=unrelated,
    env=dict(os.environ,FAKE_HIAI_FAILURE=''))
assert proc.returncode==0,proc.stderr
result=json.loads((out/'result.json').read_text())
assert result['model_sha256']==sha(nested) and result['draft_calls']==1 and result['target_calls']==0
plan()

for label,data,path in [
    ('hole',struct.pack('<8i',1,0,1,0,0,0,0,0),mask),
    ('empty',struct.pack('<8i',*[0]*8),mask),
    ('bad-token',struct.pack('<8i',248320,*range(1,8)),ids),
    ('nan',struct.pack('<H',0x7e00)+features.read_bytes()[2:],features),
]:
    original=path.read_bytes();path.write_bytes(data);plan()
    trace=root/(label+'.trace')
    proc=subprocess.run(args+['--output',str(root/label)],capture_output=True,text=True,
        env=dict(os.environ,FAKE_HIAI_FAILURE='',FAKE_HIAI_TRACE=str(trace)))
    assert proc.returncode==1 and not trace.exists(),(label,proc.stderr)
    path.write_bytes(original)
plan()
# Hash tampering and a target/fused mode must not bypass the draft contract.
ids.write_bytes(b'changed')
proc=subprocess.run(args+['--output',str(root/'tampered')],capture_output=True,text=True)
assert proc.returncode==1 and 'size' in proc.stderr
print('Standalone full draft native transport/lifecycle/validation tests passed (host fake only)')

# Packed FP32 traces are a separate diagnostic wire contract. The real CLI
# must accept its bound ABI and reject production-header and shape bypasses.
ids.write_bytes(struct.pack('<8i',*range(8)))
(root/'draft-abi.json').write_text(json.dumps({'artifact_kind':'qwen35-standalone-draft-trace-abi'}))
plan()
trace_wire = (root/'step.tsv').read_text().replace('QWEN35_DRAFT_STEP\t1','QWEN35_DRAFT_TRACE_STEP\t1').replace(
    'draft_hidden\tfloat16\t1,15,2048','draft_trace_values\tfloat32\t1,1,4')
(root/'step.tsv').write_text(trace_wire)
out = root/'trace-success'
proc = subprocess.run(args+['--output',str(out)],capture_output=True,text=True,
    env=dict(os.environ,FAKE_HIAI_FAILURE='',FAKE_HIAI_DRAFT_TRACE_ELEMENTS='4'))
assert proc.returncode == 0,proc.stderr
assert struct.unpack('<4f',(out/'output_0001.bin').read_bytes()) == (.5,)*4
for label, wire in [
    ('production-header',trace_wire.replace('QWEN35_DRAFT_TRACE_STEP','QWEN35_DRAFT_STEP')),
    ('oversize',trace_wire.replace('float32\t1,1,4','float32\t1,1,16777217')),
    ('wrong-dtype',trace_wire.replace('draft_trace_values\tfloat32','draft_trace_values\tfloat16')),
]:
    (root/'step.tsv').write_text(wire)
    proc = subprocess.run(args+['--output',str(root/('trace-'+label))],capture_output=True,text=True,
        env=dict(os.environ,FAKE_HIAI_FAILURE='',FAKE_HIAI_DRAFT_TRACE_ELEMENTS='4'))
    assert proc.returncode == 1,(label,proc.stderr)
(root/'draft-abi.json').write_text('{}')
plan()
wrong_abi_wire = (root/'step.tsv').read_text().replace('QWEN35_DRAFT_STEP\t1','QWEN35_DRAFT_TRACE_STEP\t1').replace(
    'draft_hidden\tfloat16\t1,15,2048','draft_trace_values\tfloat32\t1,1,4')
(root/'step.tsv').write_text(wrong_abi_wire)
proc = subprocess.run(args+['--output',str(root/'trace-wrong-abi')],capture_output=True,text=True)
assert proc.returncode == 1 and 'explicit ABI kind' in proc.stderr,proc.stderr
print('Draft trace explicit ABI, packed FP32 transport and production contract isolation passed (host fake only)')

# Raw typed probes have a distinct direct-output ABI. The native adapter keeps
# all outputs separate and accepts no new layout/padding exception for them.
(root/'draft-abi.json').write_text(json.dumps({'artifact_kind':'qwen35-standalone-draft-direct-abi'}))
plan()
direct_wire=(root/'step.tsv').read_text().replace('QWEN35_DRAFT_STEP\t1','QWEN35_DRAFT_DIRECT_STEP\t1')
direct_wire+='output\t2\tv41\tfloat16\t1,2\noutput\t3\tlayer_0_scores\tfloat32\t1,1,2,2\noutput\t4\tpositions\tint32\t1,2\n'
(root/'step.tsv').write_text(direct_wire)
direct_env=dict(os.environ,FAKE_HIAI_FAILURE='',FAKE_HIAI_DRAFT_DIRECT_OUTPUTS='float16:1,2;float32:1,1,2,2;int32:1,2')
out=root/'direct-success'
proc=subprocess.run(args+['--output',str(out)],capture_output=True,text=True,env=direct_env)
assert proc.returncode==0,proc.stderr
native=json.loads((out/'result.json').read_text())
assert len(native['outputs'])==5 and len(native['io_diagnostic']['outputs'])==5
assert struct.unpack('<2e',(out/'output_0002.bin').read_bytes())==(.5,.5)
assert struct.unpack('<4f',(out/'output_0003.bin').read_bytes())==(.5,)*4
assert struct.unpack('<2i',(out/'output_0004.bin').read_bytes())==(7,7)
for label,wire in [
    ('production-header',direct_wire.replace('QWEN35_DRAFT_DIRECT_STEP','QWEN35_DRAFT_STEP')),
    ('packed-header',direct_wire.replace('QWEN35_DRAFT_DIRECT_STEP','QWEN35_DRAFT_TRACE_STEP')),
    ('wrong-base',direct_wire.replace('draft_hidden\tfloat16','draft_hidden\tfloat32')),
    ('wrong-extra-dtype',direct_wire.replace('v41\tfloat16','v41\tint64')),
    ('rank-five',direct_wire.replace('v41\tfloat16\t1,2','v41\tfloat16\t1,1,1,1,2')),
    ('oversize',direct_wire.replace('v41\tfloat16\t1,2','v41\tfloat16\t33554433')),
    ('too-many',direct_wire+''.join(f'output\t{i}\tprobe_{i}\tfloat16\t1\n' for i in range(5,17))),
]:
    (root/'step.tsv').write_text(wire)
    proc=subprocess.run(args+['--output',str(root/('direct-'+label))],capture_output=True,text=True,env=direct_env)
    assert proc.returncode==1,(label,proc.stderr)
(root/'draft-abi.json').write_text('{}')
plan()
wrong_direct=(root/'step.tsv').read_text().replace('QWEN35_DRAFT_STEP\t1','QWEN35_DRAFT_DIRECT_STEP\t1')
(root/'step.tsv').write_text(wrong_direct)
proc=subprocess.run(args+['--output',str(root/'direct-wrong-abi')],capture_output=True,text=True)
assert proc.returncode==1 and 'explicit ABI kind' in proc.stderr,proc.stderr
print('Raw direct draft outputs preserve typed buffers and bounded separate ABI (host fake only)')

# The production host bundle embeds the complete output-ABI inspection. Both
# records have artifact_kind members, so a document-wide regex is incorrect.
# Exercise a realistic binding rather than a minimal top-level-only ABI.
inputs=[{'index':i,'name':name,'dtype':dtype,'shape':shape}
        for i,(name,dtype,shape) in enumerate([
            ('target_features','float16',[1,8,16384]),('input_ids','int32',[1,8]),
            ('attention_mask','int32',[1,8])])]
outputs=[{'index':i,'name':name,'dtype':dtype,'shape':shape}
         for i,(name,dtype,shape) in enumerate([
             ('draft_top1','int32',[1,15]),('draft_hidden','float16',[1,15,2048]),
             ('v41','float16',[1,2]),('layer_0_scores','float32',[1,1,2,2]),
             ('positions','int32',[1,2])])]
direct_spec=[{'name':name,'source':row['name'],'shape':row['shape'],
              'dtype':row['dtype'],'output_index':row['index']}
             for name,row in zip(('draft_top1','draft_hidden','context_fc','layer_0_scores_fp32','positions'),outputs)]
source_omc={'path':'model.omc','bytes':(root/'model.omc').stat().st_size,'sha256':sha(root/'model.omc')}
binding={'schema_version':1,'artifact_kind':'qwen35-draft-direct-output-abi-inspection','status':'PASSED',
         'declared_spec':direct_spec,'observed_spec':direct_spec,'changes':[],'omc':source_omc,
         'schema':{'library':'/selected/ddk/libai_fmk_graph_proto_host.so','library_sha256':'b'*64},
         'source_binding':'ORDERED_COMPILED_OUTPUTS_CALLER_MUST_BIND_AIR_AND_DECLARED_SPEC',
         'physical_outputs':outputs,'physical_output_bytes':sum(
             (2 if row['dtype']=='float16' else 4)*math.prod(row['shape']) for row in outputs)}
complete_abi={'schema_version':1,'artifact_kind':'qwen35-standalone-draft-direct-abi',
              'inputs':inputs,'outputs':outputs,'draft_config':{'num_hidden_layers':6,'hidden_size':2048},
              'sequence_length':8,'state_policy':'prefix-recompute','direct_spec':direct_spec,
              'source_output_precision':'SOURCE_DTYPES_PRESERVED','source_compiled_omc':source_omc,
              'direct_output_binding':binding}

def diagnostic_wire():
    plan()
    wire=(root/'step.tsv').read_text().replace('QWEN35_DRAFT_STEP\t1','QWEN35_DRAFT_DIRECT_STEP\t1')
    return wire+'output\t2\tv41\tfloat16\t1,2\noutput\t3\tlayer_0_scores\tfloat32\t1,1,2,2\noutput\t4\tpositions\tint32\t1,2\n'

complete_text=json.dumps(complete_abi,indent=2)
for label,text in [
    ('complete-nested-binding',complete_text),
    ('escaped-top-kind',complete_text.replace('"artifact_kind"','"\\u0061rtifact_kind"',1)),
    ('kind-like-string',json.dumps({**complete_abi,'description':'"artifact_kind":"not the root"; braces {} [] and escaped \\ paths'})),
]:
    (root/'draft-abi.json').write_text(text)
    (root/'step.tsv').write_text(diagnostic_wire())
    out=root/label
    proc=subprocess.run(args+['--output',str(out)],capture_output=True,text=True,env=direct_env)
    assert proc.returncode==0,(label,proc.stderr)
    assert len(json.loads((out/'result.json').read_text())['outputs'])==5

for label,text,reason in [
    ('nested-correct-root-wrong',json.dumps({'artifact_kind':'wrong','nested':complete_abi}),'explicit ABI kind'),
    ('only-nested-kind',json.dumps({'nested':complete_abi}),'explicit ABI kind'),
    ('duplicate-root-kind','{"artifact_kind":"qwen35-standalone-draft-direct-abi","artifact_kind":"wrong"}','duplicate object member'),
    ('escaped-duplicate-root-kind','{"artifact_kind":"qwen35-standalone-draft-direct-abi","\\u0061rtifact_kind":"qwen35-standalone-draft-direct-abi"}','duplicate object member'),
    ('non-string-root-kind','{"artifact_kind":true}','artifact_kind must be a string'),
    ('non-object-root',json.dumps([complete_abi]),'top-level value must be an object'),
    ('trailing-object',complete_text+'{}','trailing data'),
    ('trailing-comma','{"artifact_kind":"qwen35-standalone-draft-direct-abi",}','object key must be a string'),
    ('bad-number','{"artifact_kind":"qwen35-standalone-draft-direct-abi","v":01}','unexpected token'),
    ('bad-string-escape','{"artifact_kind":"qwen35-standalone-draft-direct-abi","v":"\\q"}','invalid string escape'),
    ('unpaired-surrogate','{"artifact_kind":"qwen35-standalone-draft-direct-abi","v":"\\ud800"}','unexpected token'),
    ('too-deep','{"artifact_kind":"qwen35-standalone-draft-direct-abi","v":'+'['*33+'0'+']'*33+'}','nesting/token budget'),
]:
    (root/'draft-abi.json').write_text(text)
    (root/'step.tsv').write_text(diagnostic_wire())
    trace=root/(label+'.trace')
    proc=subprocess.run(args+['--output',str(root/label)],capture_output=True,text=True,
        env=dict(direct_env,FAKE_HIAI_TRACE=str(trace)))
    assert proc.returncode==1 and reason in proc.stderr,(label,proc.stderr)
    assert not trace.exists(),(label,'invalid ABI reached model Load')
print('Nested direct binding and bounded top-level JSON ABI-kind validation passed (host fake only)')
