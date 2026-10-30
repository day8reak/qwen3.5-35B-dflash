"""Resident supervisor scheduling with an explicit host-only Session double.

The real V2 adapter/session lifecycle is tested separately; these cases exercise
the same C++ supervisor, original prefill child, private state files and receipts.
"""
import json
import os
from pathlib import Path
import subprocess

import pytest

from test_board_serial import RUNTIME, case, tables, sha, with_prefill, report


STUB = r'''
#include "qwen35_dflash/resident_session.hpp"
#include "qwen35_dflash/sha256.hpp"
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
namespace qwen35::dflash {
using B=std::vector<std::uint8_t>;
static std::string Mode(){const char* s=std::getenv("BOARD_RESIDENT_FAULT");return s?s:"";}
static void Check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
static unsigned Word(const B& b){unsigned v=0;for(unsigned i=0;i<4;++i)v|=unsigned(b.at(i))<<(8*i);return v;}
struct ResidentSession::Impl {ResidentSessionEvidence e;ResidentSessionConfig c;bool loaded=false;};
ResidentSession::ResidentSession():impl_(new Impl){}
ResidentSession::~ResidentSession()=default;
const ResidentSessionEvidence& ResidentSession::evidence()const noexcept{return impl_->e;}
void ResidentSession::Close(){auto& e=impl_->e;if(e.closed)return;e.closed=true;
  if(e.target_loads)++e.target_close_calls;if(e.draft_loads)++e.draft_close_calls;
  if(Mode()=="close-fail"){e.poisoned=true;e.status="FAILED";e.cleanup_error="injected close failure";throw std::runtime_error(e.cleanup_error);}
  if(!e.poisoned){e.status="CLOSED";e.stage="CLOSED";}}
void ResidentSession::Load(const ResidentSessionConfig& c){auto& e=impl_->e;Check(!impl_->loaded,"second load forbidden");impl_->loaded=true;impl_->c=c;
  const char* proof=std::getenv("BOARD_RESIDENT_PREFILL_EXIT");Check(proof&&std::ifstream(proof).good(),"resident load occurred before prefill exit");
  try{e.status="LOADING";e.stage="TARGET_LOAD";++e.target_load_attempts;if(Mode()=="target-load-fail")throw std::runtime_error("target load failure");
    const std::map<std::string,int> types={{"FLOAT16",2},{"FLOAT32",1},{"INT8",4},{"INT16",5},{"INT32",3},{"INT64",7}};
    for(const auto* rows:{&c.target_inputs,&c.target_outputs})for(const auto& r:*rows)Check(r.descriptor.raw_dtype==types.at(r.dtype)&&r.descriptor.raw_format==2,"wrong exact runtime enum");
    ++e.target_loads;e.stage="DRAFT_LOAD";++e.draft_load_attempts;if(Mode()=="draft-load-fail")throw std::runtime_error("draft load failure");
    ++e.draft_loads;e.status="READY";e.stage="READY";
  }catch(...){e.poisoned=true;e.status="FAILED";Close();throw;}}
static HiaiIoDiagnostic Diag(const std::vector<B>& inputs,std::size_t count){HiaiIoDiagnostic d;d.output_fill_byte=0;d.process_called=true;
  for(std::size_t i=0;i<inputs.size();++i){HiaiInputIoDiagnostic r;r.index=i;r.logical_bytes=r.allocated_bytes=inputs[i].size();r.readback_matches_source=true;r.readback_sha256=Sha256(std::string_view(reinterpret_cast<const char*>(inputs[i].data()),inputs[i].size()));d.inputs.push_back(r);}
  for(std::size_t i=0;i<count;++i){HiaiOutputIoDiagnostic r;r.index=i;r.checked_after_process=r.object_same=r.buffer_same=true;d.outputs.push_back(r);}return d;}
ResidentRunResult ResidentSession::RunTarget(const std::vector<B>& inputs){auto& e=impl_->e;Check(!e.closed&&!e.poisoned&&e.draft_loads==1,"target session not live");++e.target_run_attempts;e.stage="TARGET_RUN";
  try{if(Mode()=="target-run-fail")throw std::runtime_error("target run failure");Check(inputs.size()==89,"missing target input");
    std::map<std::string,B> byname;for(std::size_t i=0;i<inputs.size();++i){Check(inputs[i].size()==impl_->c.target_inputs[i].bytes,"target extent mismatch");byname.emplace(impl_->c.target_inputs[i].descriptor.tensor_name,inputs[i]);}
    auto position=Word(byname.at("arg107_1"));Check(position>=8&&position<64&&Word(byname.at("arg2_1"))==position,"absolute position mismatch");
    Check(Word(byname.at("arg1238_1"))==0&&Word(byname.at("arg114_1"))==1&&byname.at("arg18_1")[0]==1,"local query mismatch");
    for(unsigned i=0;i<80;++i){const auto& b=byname.at("RefData_"+std::to_string(i));Check(b.size()==2&&unsigned(b[0])+(unsigned(b[1])<<8)==position+i*256,"wrong cross-owner state bytes");}
    const auto token=Word(byname.at("arg1_1"));ResidentRunResult r;for(const auto& t:impl_->c.target_outputs)r.outputs.emplace_back(t.bytes,0);
    float score=2;std::memcpy(r.outputs[0].data()+4*(token+1),&score,4);for(unsigned i=1;i<81;++i){auto value=position+1+(i-1)*256;r.outputs[i][0]=value&255;r.outputs[i][1]=value>>8;}
    for(std::size_t i=0;i<r.outputs[81].size();i+=2)r.outputs[81][i+1]=0x38;r.io_diagnostic=Diag(inputs,82);if(Mode()=="bad-readback")r.io_diagnostic.inputs[0].readback_matches_source=false;
    if(Mode()=="bad-output-count")r.outputs.pop_back();if(Mode()=="source-change"){std::ofstream f(impl_->c.target_model,std::ios::app);f<<'x';}
    ++e.target_runs;e.stage="READY";return r;
  }catch(...){e.poisoned=true;e.status="FAILED";Close();throw;}}
ResidentRunResult ResidentSession::RunDraft(const std::vector<B>& inputs){auto& e=impl_->e;Check(!e.closed&&!e.poisoned,"draft session not live");++e.draft_run_attempts;e.stage="DRAFT_RUN";
  try{if(Mode()=="draft-run-fail")throw std::runtime_error("draft run failure");Check(inputs.size()==3&&inputs[0].size()==2097152&&inputs[1].size()==256&&inputs[2].size()==256,"draft exact input ABI mismatch");
    unsigned valid=0;while(valid<64&&inputs[2][valid*4]==1)++valid;Check(valid>=9,"draft excludes eager anchor");
    for(unsigned i=0;i<64;++i){Check(Word(B(inputs[2].begin()+i*4,inputs[2].begin()+i*4+4))==(i<valid?1u:0u),"mask padding mismatch");Check(Word(B(inputs[1].begin()+i*4,inputs[1].begin()+i*4+4))==(i<valid?i+1:0),"full ordered prefix mismatch");}
    for(std::size_t i=0;i<inputs[0].size();i+=2)Check(inputs[0][i]==0&&inputs[0][i+1]==(i<valid*32768?0x38:0),"feature prefix/zero padding mismatch");
    ResidentRunResult r;r.outputs={B(60,0),B(61440,0)};const char* raw=std::getenv("BOARD_SERIAL_CASE");std::string mode=raw?raw:"accept";
    for(unsigned i=0;i<15;++i){auto token=valid+1+i+(mode=="reject"||(mode=="mixed"&&e.draft_runs%2)?1:0);for(unsigned j=0;j<4;++j)r.outputs[0][4*i+j]=token>>(8*j);}
    r.io_diagnostic=Diag(inputs,2);++e.draft_runs;e.stage="READY";return r;
  }catch(...){e.poisoned=true;e.status="FAILED";Close();throw;}}
}
'''


@pytest.fixture(scope='session')
def resident_binary(tmp_path_factory):
    root = tmp_path_factory.mktemp('resident-build')
    source = root / 'session-double.cpp'; source.write_text(STUB)
    binary = root / 'resident-host-fake'
    subprocess.run(['g++', '-O1', '-std=c++17', '-Wall', '-Wextra', '-Wpedantic',
        '-DQWEN35_BOARD_SERIAL_HOST_FAKE=1', '-DQWEN35_RESIDENT_V2=1',
        '-I', str(RUNTIME / 'include'), str(RUNTIME / 'src/board_serial_main.cpp'),
        str(RUNTIME / 'src/sha256.cpp'), str(source), '-o', str(binary)],
        capture_output=True, text=True, check=True)
    return binary


def enable_resident(case, dump=True):
    rows = with_prefill(case, dump)
    abi = ['QWEN35_RESIDENT_ABI\t1']
    types = {'FP16': ('FLOAT16', 2, 2), 'FP32': ('FLOAT32', 1, 4), 'INT8': ('INT8', 4, 1),
        'INT16': ('INT16', 5, 2), 'INT32': ('INT32', 3, 4), 'INT64': ('INT64', 7, 8)}
    for row in rows:
        fields = row.split('\t')
        if fields[0] in {'target_input', 'target_output'}:
            role, index, name, tooltype, size = fields
            dtype, raw, width = types[tooltype]
            abi.append('\t'.join((role, index, name, dtype, str(raw), '2', str(int(size) // width), size)))
    path = case['root'] / 'resident-abi.tsv'; path.write_text('\n'.join(abi) + '\n')
    rows += ['execution_mode\tresident-v2', f'resident_abi\t{path}\t{sha(path)}\t{path.stat().st_size}']
    case['config'].write_text('\n'.join(rows) + '\n')
    return path


def run(resident_binary, case, mode='accept', fault=''):
    return subprocess.run([str(resident_binary), '--config', str(case['config']), '--output', str(case['out'])],
        capture_output=True, text=True, timeout=90,
        env=dict(os.environ, BOARD_SERIAL_CASE=mode, BOARD_RESIDENT_FAULT=fault,
            BOARD_RESIDENT_PREFILL_EXIT=str(case['out'] / 'prefill/step_0000/tool.exitcode')))


@pytest.mark.parametrize('dump', [False, True])
@pytest.mark.parametrize('mode', ['accept', 'reject', 'mixed'])
def test_prefill_then_two_resident_managers_all_commits(resident_binary, case, dump, mode):
    enable_resident(case, dump); process = run(resident_binary, case, mode)
    assert process.returncode == 0, process.stderr
    value = report(case); life = value['resident_lifecycle']
    assert value['schema_version'] == 3 and value['execution_mode'] == 'resident-v2'
    assert value['evidence_kind'] == 'host-fake-hiai'
    assert value['generated_tokens'] == list(range(9, 17)) and value['full_prefix'] == list(range(1, 17))
    assert life['target_loads'] == life['draft_loads'] == 1 and life['closed'] and not life['poisoned']
    assert life['target_close_calls'] == life['draft_close_calls'] == 1
    # The host double leaves the SDK-status sentinel untouched. Preserve its
    # signed value rather than misreporting an unobserved status as success.
    assert type(life['target_lifecycle']['weights_deinit_status']) is int
    assert life['target_lifecycle']['weights_deinit_status'] == -1
    assert life['target_runs'] == value['counts']['target_native_calls'] == 8
    assert life['draft_runs'] == value['counts']['draft_native_calls'] >= 2
    assert value['child_lifecycle']['completed_child_count'] == 1
    assert len(value['child_lifecycle']['children']) == 1
    assert value['supervisor_timing']['target_child_processes_timed'] == value['supervisor_timing']['draft_child_processes_timed'] == 0
    assert value['phase_latency']['timing_scope'] == 'RESIDENT_RUN_WALL_EXCLUDES_MODEL_LOAD_AND_CLOSE'
    assert [row['phase'] for row in value['resident_memory']['snapshots']] == ['before-load', 'after-both-load', 'after-close']
    assert process.stdout.count('startup-hash-begin role=target_model ') == 1
    assert process.stdout.count('startup-hash-begin role=draft_model ') == 1
    assert case['draft'].stat().st_nlink == 1
    if dump:
        calls = []
        for key in ('target_steps', 'draft_steps'):
            for event in value[key]:
                receipt = json.loads((case['out'] / event['receipt']['path']).read_text())
                assert not {'pid', 'exitcode'} & receipt.keys()
                assert receipt['session_id'] == life['session_id'] and receipt['process_id'] == value['process_id']
                calls.append(receipt['call_index'])
        assert sorted(calls) == list(range(8 + life['draft_runs']))
        assert len(value['final_state_files']) == 80 and len(value['final_target_outputs']) == 82
    else:
        assert not list(case['out'].rglob('*.bin'))
        assert not {'target_steps', 'draft_steps', 'commits', 'rounds'} & value.keys()


@pytest.mark.parametrize('fault', ['target-load-fail', 'draft-load-fail', 'target-run-fail',
    'draft-run-fail', 'close-fail', 'bad-readback', 'bad-output-count', 'source-change'])
def test_resident_failure_closes_and_preserves_current_scratch_without_retry(resident_binary, case, fault):
    enable_resident(case, False); process = run(resident_binary, case, fault=fault)
    assert process.returncode == 1
    value = report(case); life = value['resident_lifecycle']
    assert value['status'] == 'FAILED' and life['closed']
    assert life['target_load_attempts'] == 1 and life['draft_load_attempts'] <= 1
    assert life['target_run_attempts'] <= 8 and life['draft_run_attempts'] <= 8
    if fault == 'target-run-fail': assert life['target_run_attempts'] == 1
    if fault == 'draft-run-fail': assert life['draft_run_attempts'] == 1
    assert life['target_close_calls'] <= 1 and life['draft_close_calls'] <= 1
    assert value['child_lifecycle']['completed_child_count'] == 1
    assert value['child_lifecycle']['active_child_count'] == 0
    assert list(case['out'].glob('*/step_*/outputs/*'))


def test_resident_descriptor_and_mode_fail_closed_before_any_child(resident_binary, case):
    abi = enable_resident(case)
    raw = abi.read_text().replace('INT8\t4\t2\t2048', 'INT8\t4\t2\t2049', 1)
    abi.write_text(raw)
    config = case['config'].read_text().splitlines()
    config = [f'resident_abi\t{abi}\t{sha(abi)}\t{abi.stat().st_size}' if row.startswith('resident_abi\t') else row for row in config]
    case['config'].write_text('\n'.join(config) + '\n')
    process = run(resident_binary, case)
    assert process.returncode == 1 and 'descriptor differs' in process.stderr
    assert not list(case['out'].rglob('tool.pid'))


def test_resident_binary_never_falls_back_without_explicit_mode(resident_binary, case):
    enable_resident(case)
    case['config'].write_text(case['config'].read_text().replace('execution_mode\tresident-v2\n', ''))
    process = run(resident_binary, case)
    assert process.returncode == 1 and 'requires explicit resident-v2' in process.stderr
    assert not list(case['out'].rglob('tool.pid'))
