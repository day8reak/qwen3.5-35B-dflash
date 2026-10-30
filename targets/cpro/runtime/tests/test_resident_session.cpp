#include "qwen35_dflash/resident_session.hpp"
#include <tensor/nd_tensor_buffer.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>

using namespace qwen35::dflash;
namespace fs=std::filesystem;
using Payloads=std::vector<std::vector<std::uint8_t>>;
namespace {
void Require(bool value,const std::string& message){if(!value)throw std::runtime_error(message);}
std::string Throws(const std::function<void()>& call){try{call();}catch(const HiaiRuntimeError& e){return e.stage();}throw std::runtime_error("expected HiaiRuntimeError");}
void Failure(const std::string& value){if(value.empty())unsetenv("FAKE_HIAI_FAILURE");else setenv("FAKE_HIAI_FAILURE",value.c_str(),1);}
std::vector<std::string> Trace(const fs::path& path){std::ifstream in(path);std::vector<std::string> result;for(std::string s;std::getline(in,s);)result.push_back(s);return result;}
std::size_t Count(const std::vector<std::string>& rows,const std::string& key){return std::count(rows.begin(),rows.end(),key);}
std::vector<HiaiBoundRuntimeTensor> Contract(bool input){
  const auto enums=HiaiModelIoEnums();std::vector<HiaiBoundRuntimeTensor> result;
  const auto nd=static_cast<int>(hiai::Format::ND),physical=static_cast<int>(hiai::Format::NCHW);
  auto add=[&](std::vector<std::int64_t> shape,std::string dtype,int format){
    HiaiBoundRuntimeTensor row;row.descriptor.index=result.size();row.descriptor.shape=std::move(shape);
    row.descriptor.raw_dtype=enums.dtypes.at(dtype);row.descriptor.raw_format=format;
    row.descriptor.tensor_name=std::string(input?"runtime_input_":"runtime_output_")+std::to_string(result.size());
    row.descriptor.tensor_name_status="OBSERVED";row.dtype=std::move(dtype);
    row.bytes=row.dtype=="INT8"?1:row.dtype=="INT64"?8:(row.dtype=="FLOAT16"||row.dtype=="INT16")?2:4;
    for(const auto dim:row.descriptor.shape)row.bytes*=static_cast<std::size_t>(dim);
    result.push_back(std::move(row));};
  if(input){add({1,1,1,2048},"INT8",nd);add({1,1,1,2048},"FLOAT32",nd);add({1,1},"INT32",nd);}
  else add({1,1,1,248320},"FLOAT32",nd);
  for(std::size_t i=0;i<80;++i){const auto kind=i%8;add(kind==0||kind==2||kind==4?std::vector<std::int64_t>{1,8192,4}:
    kind==1||kind==3||kind==5?std::vector<std::int64_t>{1,32,128,128}:std::vector<std::int64_t>{64,32,32,16},"FLOAT16",physical);}
  if(input){add({1},"INT32",nd);add({1},"INT32",nd);add({1,1,1},"FLOAT32",nd);add({1},"INT64",nd);add({128},"INT32",nd);add({1},"INT16",nd);}
  else add({1,1,1,16384},"FLOAT16",nd);
  return result;
}
Payloads TargetInputs(const ResidentSessionConfig& c){Payloads result;for(const auto& row:c.target_inputs)result.emplace_back(row.bytes,0);return result;}
Payloads DraftInputs(){
  Payloads result{std::vector<std::uint8_t>(64*16384*2),std::vector<std::uint8_t>(256),std::vector<std::uint8_t>(256)};
  for(std::size_t i=0;i<64*16384;++i)result[0][i*2+1]=0x38;
  for(std::size_t i=0;i<64;++i){result[1][i*4]=static_cast<std::uint8_t>(i);result[2][i*4]=i<6?1:0;}
  return result;
}
std::int32_t Top(const ResidentRunResult& r){std::int32_t top=0;std::memcpy(&top,r.outputs.at(0).data(),4);return top;}
void Diagnostic(const ResidentRunResult& r,std::size_t count){
  Require(r.io_diagnostic.process_called&&r.io_diagnostic.inputs.size()==count,"missing per-call input diagnostic");
  for(const auto& d:r.io_diagnostic.inputs)Require(d.readback_matches_source,"input readback differs");
  for(const auto& d:r.io_diagnostic.outputs)Require(d.checked_after_process&&d.object_same&&d.buffer_same,"output identity differs");
}
}
int main(int argc,char** argv){try{
  Require(argc==2,"expected output directory");const auto root=fs::absolute(argv[1]);fs::create_directories(root/"weights");
  std::ofstream(root/"target.omc")<<"original-target-fixture";
  std::ofstream(root/"draft.omc")<<"draft-resident64-fixture";
  ResidentSessionConfig config{root/"target.omc",root/"weights",root/"draft.omc","FAKE-HIAI-X90",Contract(true),Contract(false),300000};
  auto trace=[&](const std::string& name){const auto p=root/(name+".trace");std::ofstream(p).close();setenv("FAKE_HIAI_TRACE",p.c_str(),1);return p;};
  Failure("");const auto repeated=trace("alternating");
  {
    ResidentSession session;Throws([&]{session.RunDraft(DraftInputs());});session.Load(config);
    Require(session.evidence().target_loads==1&&session.evidence().draft_loads==1,"both V2 models must load once");
    Throws([&]{session.Load(config);});
    auto inputs=TargetInputs(config);auto first=session.RunTarget(inputs);Diagnostic(first,89);
    Require(first.outputs.size()==82&&first.outputs[1][0]==1,"target state first run differs");
    auto first_draft=session.RunDraft(DraftInputs());Diagnostic(first_draft,3);Require(Top(first_draft)==10,"first draft run differs");
    // Output copies survive both subsequent model runs and a caller mutation.
    // Mutating a returned feature cannot overwrite either persistent model.
    first.outputs[81][0]=77;
    for(std::size_t round=2;round<=3;++round){
      for(std::size_t i=0;i<80;++i)inputs[i+3][0]=static_cast<std::uint8_t>(round-1);
      auto target=session.RunTarget(inputs);Require(target.outputs[1][0]==round,"latest target input was not copied");
      auto draft=session.RunDraft(DraftInputs());Require(Top(draft)==static_cast<std::int32_t>(round+9),"draft manager was recreated or stale output used");
      Require(first.outputs[1][0]==1&&first.outputs[81][0]==77&&Top(first_draft)==10,"returned output bytes alias live executor storage");
    }
    Require(session.evidence().target_runs==3&&session.evidence().draft_runs==3,"alternating run counters differ");
    session.Close();session.Close();const auto& e=session.evidence();
    Require(e.closed&&!e.poisoned&&e.target_close_calls==1&&e.draft_close_calls==1,"close must be idempotent for both managers");
    Require(e.target_lifecycle.manager_deinit_called&&e.target_lifecycle.weights_deinit_status==0,"target weights cleanup missing");
    Require(e.target_lifecycle.weights_init_status==0&&e.target_lifecycle.manager_init_status==0&&
      e.target_lifecycle.failure_stage.empty()&&e.target_lifecycle.cleanup_error.empty(),"successful lifecycle statuses differ");
    Throws([&]{session.RunTarget(inputs);});Throws([&]{session.Load(config);});
  }
  auto events=Trace(repeated);
  Require(Count(events,"v2-restore-file")==2&&Count(events,"v2-init")==2&&Count(events,"v2-run")==6,"load/run lifecycle was not resident");
  Require(Count(events,"v2-create-tensor")==176,"persistent tensors reallocated during runs");
  Require(Count(events,"v2-set-name:qwen35_resident_target")==1&&Count(events,"v2-set-name:qwen35_resident_draft")==1,"distinct V2 names not bound");
  Require(Count(events,"v2-deinit")==2&&Count(events,"v2-deinit-weights")==1,"model cleanup counts differ");
  for(const auto* forbidden:{"load","process","unload"})Require(Count(events,forbidden)==0,"legacy model API used by resident session");
  const auto first_run=std::find(events.begin(),events.end(),"v2-run");
  Require(std::count(events.begin(),first_run,"v2-init")==2,"first Run occurred before both models initialized");
  const auto first_close=std::find(events.begin(),events.end(),"v2-deinit");
  Require(std::count(events.begin(),first_close,"v2-run")==6,"manager unloaded between alternating runs");

  for(const auto& fault:std::vector<std::string>{"original-init-weights","original-init-weights-throw","original-init-weights-unknown",
      "original-init","original-init-cleanup-failure","original-init-deinit-throw",
      "resident-draft-init","v2-shape","resident-name","resident-cross-alias"}){
    Failure(fault);const auto path=trace(fault);ResidentSession session;
    const auto stage=Throws([&]{session.Load(config);});const auto& e=session.evidence();
    Require(e.closed&&e.poisoned&&e.status=="FAILED"&&stage==e.stage,"load failure stage lost or session not poisoned");
    if(fault=="resident-draft-init")Require(e.target_loads==1&&e.draft_loads==0&&stage.find("DRAFT_LOAD/MODEL_MANAGER_INIT")!=std::string::npos,"partial load failure not attributable");
    if(fault=="resident-cross-alias")Require(stage.find("LIVE_STORAGE")!=std::string::npos,"zero-content cross-manager alias accepted");
    if(fault.rfind("original-init-weights",0)==0){
      const auto& t=e.target_lifecycle;
      Require(e.target_loads==0&&e.draft_load_attempts==0&&t.weights_init_attempted&&!t.weights_initialized&&
        !t.manager_init_attempted&&!t.manager_init_status.has_value()&&!t.manager_deinit_called&&!t.weights_deinit_called&&t.closed,
        "InitWeights constructor failure lost its actual attempt/cleanup snapshot");
      Require(t.failure_stage=="EXPERIMENTAL_INIT_WEIGHTS_BEFORE_INIT"&&stage=="TARGET_LOAD/"+t.failure_stage,
        "first InitWeights failure stage was overwritten by cleanup");
      if(fault=="original-init-weights")Require(t.weights_init_status==72,"returned InitWeights status lost");
      else Require(!t.weights_init_status.has_value(),"throwing InitWeights invented a return status");
    }
    if(fault=="original-init"||fault=="original-init-cleanup-failure"||fault=="original-init-deinit-throw"){
      const auto& t=e.target_lifecycle;
      Require(e.target_loads==0&&e.draft_load_attempts==0&&t.weights_init_attempted&&t.weights_initialized&&t.weights_init_status==0&&
        t.manager_init_attempted&&!t.manager_initialized&&t.manager_init_status==32&&t.manager_deinit_called&&t.weights_deinit_called&&t.closed,
        "failed Init partial ownership and cleanup were not copied out of constructor");
      Require(t.failure_stage=="MODEL_MANAGER_INIT"&&stage=="TARGET_LOAD/MODEL_MANAGER_INIT","cleanup hid the first Init failure");
      Require(t.weights_deinit_status==(fault=="original-init-cleanup-failure"?73:0),"partial cleanup return status lost");
      Require(t.cleanup_error.empty()==(fault=="original-init"),"partial cleanup error lost or fabricated");
      Require(e.cleanup_error.empty()==t.cleanup_error.empty(),"session hid partial-constructor cleanup error");
    }
    Failure("");Throws([&]{session.Load(config);});Throws([&]{session.RunDraft(DraftInputs());});
    events=Trace(path);Require(Count(events,"v2-run")==0,"load failure launched a run");
    if(e.target_loads)Require(Count(events,"v2-deinit-weights")==1,"partial load leaked target weights");
    if(fault.rfind("original-init-weights",0)==0)Require(Count(events,"v2-init-weights")==1&&Count(events,"v2-init")==0&&Count(events,"v2-deinit-weights")==0,"failed weights init retried or cleaned unowned scope");
    if(fault=="original-init"||fault=="original-init-cleanup-failure"||fault=="original-init-deinit-throw")
      Require(Count(events,"v2-init")==1&&Count(events,"v2-deinit")==1&&Count(events,"v2-deinit-weights")==1,"partial Init cleanup retried or omitted");
  }
  for(const auto& fault:std::vector<std::string>{"original-run","original-run-unknown-exception","v2-run","v2-replace-output","v2-short-buffer"}){
    Failure("");const auto path=trace(fault);ResidentSession session;session.Load(config);Failure(fault);
    const bool draft=fault=="v2-run"||fault=="v2-replace-output";
    Throws([&]{if(draft)session.RunDraft(DraftInputs());else session.RunTarget(TargetInputs(config));});
    const auto& e=session.evidence();Require(e.poisoned&&e.closed&&e.target_runs==0&&e.draft_runs==0,"failed call promoted result or remained live");
    Failure("");Throws([&]{session.RunTarget(TargetInputs(config));});Throws([&]{session.RunDraft(DraftInputs());});session.Close();
    events=Trace(path);Require(Count(events,"v2-run")==static_cast<std::size_t>(fault=="v2-short-buffer"?0:1),"failed run retried");
    Require(Count(events,"v2-deinit")==2&&Count(events,"v2-deinit-weights")==1,"run failure leaked managers");
  }
  Failure("");const auto invalid=trace("incomplete-input");{
    ResidentSession session;session.Load(config);auto short_input=DraftInputs();short_input.pop_back();Throws([&]{session.RunDraft(short_input);});
    Require(session.evidence().closed&&session.evidence().draft_runs==0,"incomplete input accepted");
  }Require(Count(Trace(invalid),"v2-run")==0,"incomplete input called HIAI Run");
  Failure("");const auto close=trace("cleanup-failure");{
    ResidentSession session;session.Load(config);Failure("original-deinit-weights");
    Require(Throws([&]{session.Close();})=="SESSION_CLEANUP","cleanup failure lost");
    const auto& e=session.evidence();Require(e.poisoned&&e.closed&&e.target_lifecycle.weights_deinit_status==73,"cleanup Status not retained");
    session.Close();
  }Require(Count(Trace(close),"v2-deinit-weights")==1,"failed cleanup retried");
  Failure("");const auto draft_close=trace("draft-cleanup-failure");{
    ResidentSession session;session.Load(config);Failure("resident-draft-deinit");
    Require(Throws([&]{session.Close();})=="SESSION_CLEANUP","throwing draft cleanup not reported");
    const auto& e=session.evidence();Require(e.poisoned&&e.closed&&e.stage=="SESSION_CLEANUP"&&e.status=="FAILED"&&
      e.cleanup_error.find("draft:")!=std::string::npos,"draft cleanup error/status lost");
    Require(e.target_lifecycle.weights_deinit_status==0&&e.target_close_calls==1&&e.draft_close_calls==1,"draft failure skipped target cleanup");
    session.Close();Throws([&]{session.RunDraft(DraftInputs());});
  }Require(Count(Trace(draft_close),"v2-deinit")==2&&Count(Trace(draft_close),"v2-deinit-weights")==1,"throwing cleanup retried or leaked peer");
  Failure("");unsetenv("FAKE_HIAI_TRACE");
  std::cout<<"PASS resident session: two distinct file-v2 managers, alternating3+3 Run, persistent176 buffers, owned output copies and18 failure cases; host fake only\n";
  return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
