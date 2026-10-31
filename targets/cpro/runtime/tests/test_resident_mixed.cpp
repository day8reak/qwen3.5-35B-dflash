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
  ResidentSessionConfig config{root/"target.omc",root/"weights",root/"draft.omc","FAKE-HIAI-X90",Contract(true),Contract(false),300000,"target-v2-draft-v1"};
  setenv("FAKE_HIAI_V1_RESIDENT_TRACE","1",1);unsetenv("FAKE_HIAI_ALLOW_DUAL_MODEL_SOURCE");
  auto trace=[&](const std::string& name){const auto p=root/(name+".trace");std::ofstream(p).close();setenv("FAKE_HIAI_TRACE",p.c_str(),1);return p;};
  Failure("");const auto repeated=trace("mixed-alternating");
  setenv("FAKE_HIAI_DRAFT_LOAD_DELAY","1",1);
  {
    ResidentSession session("target-v2-draft-v1");
    Require(session.evidence().target_lifecycle.weight_cleanup_policy=="MANAGER_DEINIT_ONLY_NO_SEPARATE_WEIGHT_UNLOAD","preload evidence used historical cleanup policy");
    session.Load(config);Throws([&]{session.Load(config);});
    const auto& phases=session.evidence().draft_v1_lifecycle.load_phases;
    const std::vector<std::string> expected={"CLIENT_INIT","RUNTIME_QUERY_AND_DESCRIPTION","MODEL_MMAP","SET_MODEL_BUFFER","MODEL_LOAD","IO_DESCRIPTORS","TENSOR_SETUP"};
    Require(phases.size()==expected.size(),"draft load stages missing");
    for(std::size_t i=0;i<phases.size();++i)Require(phases[i].phase==expected[i]&&phases[i].completed,"draft load stage order or completion differs");
    Require(phases[4].wall_us>=20000,"deliberate SDK load delay not attributed to MODEL_LOAD");
    unsetenv("FAKE_HIAI_DRAFT_LOAD_DELAY");
    auto inputs=TargetInputs(config);auto first=session.RunTarget(inputs);Diagnostic(first,89);
    auto first_draft=session.RunDraft(DraftInputs());Diagnostic(first_draft,3);Require(Top(first_draft)==10,"V1 draft result differs");
    first.outputs[81][0]=77;
    for(std::size_t round=2;round<=3;++round){
      for(std::size_t i=0;i<80;++i)inputs[i+3][0]=static_cast<std::uint8_t>(round-1);
      auto target=session.RunTarget(inputs);Require(target.outputs[1][0]==round,"V2 target did not consume latest state");
      auto draft=session.RunDraft(DraftInputs());Require(Top(draft)==static_cast<int>(round+9),"draft client did not remain resident");
      Require(first.outputs[1][0]==1&&first.outputs[81][0]==77&&Top(first_draft)==10,"mixed output copies alias live storage");
    }
    const auto before=Trace(repeated);Require(Count(before,"unload")==0&&Count(before,"v2-deinit")==0,"models closed between alternating calls");
    session.Close();session.Close();const auto& e=session.evidence();const auto& t=e.target_lifecycle;
    Require(e.closed&&!e.poisoned&&e.target_loads==1&&e.draft_loads==1&&e.target_runs==3&&e.draft_runs==3,"mixed lifecycle counters differ");
    Require(e.target_close_calls==1&&e.draft_close_calls==1&&t.manager_deinit_called&&t.manager_deinit_returned&&!t.weights_deinit_called&&t.weights_deinit_status==-1,"mixed manager-only cleanup evidence differs");
    Require(e.draft_v1_lifecycle.loaded&&e.draft_v1_lifecycle.unload_called&&e.draft_v1_lifecycle.unload_status==0&&e.draft_v1_lifecycle.successful_processes==3,"V1 draft lifecycle missing");
    Throws([&]{session.RunTarget(inputs);});
  }
  auto events=Trace(repeated);
  Require(Count(events,"v2-restore-file")==1&&Count(events,"v2-init")==1&&Count(events,"v2-init-weights")==1&&Count(events,"load")==1,"mixed loads must select V2 target and V1 draft exactly once");
  Require(Count(events,"v2-run")==3&&Count(events,"process")==3,"mixed Run/Process dispatch differs");
  Require(Count(events,"v2-create-tensor")==171&&Count(events,"v1-create-tensor")==5,"mixed buffers not allocated once");
  Require(Count(events,"v2-deinit")==1&&Count(events,"v2-deinit-weights")==0&&Count(events,"unload")==1,"mixed cleanup unexpectedly called separate weights unload or retried");
  Require(std::find(events.begin(),events.end(),"unload")<std::find(events.begin(),events.end(),"v2-deinit"),"draft must close while target is still owned");
  for(const auto& fault:std::vector<std::string>{"original-init","original-init-weights","original-init-weights-throw","original-init-weights-unknown","v1-draft-load","mixed-cross-alias"}){
    Failure(fault);const auto path=trace(fault);ResidentSession session("target-v2-draft-v1");
    Throws([&]{session.Load(config);});const auto& e=session.evidence();
    Require(e.closed&&e.poisoned&&e.status=="FAILED"&&e.target_runs==0&&e.draft_runs==0,"failed mixed load promoted runnable session");
    if(fault=="original-init")Require(e.target_lifecycle.manager_init_status==32&&!e.target_lifecycle.weights_init_attempted,"manager failure snapshot differs");
    if(fault.rfind("original-init-weights",0)==0)Require(e.target_lifecycle.manager_init_status==0&&e.target_lifecycle.weights_init_attempted&&!e.target_lifecycle.weights_initialized,"weights failure snapshot lost");
    if(fault=="mixed-cross-alias")Require(e.stage.find("LIVE_STORAGE")!=std::string::npos,"cross-API overlapping allocations accepted");
    if(fault=="v1-draft-load"){
      const auto& phases=e.draft_v1_lifecycle.load_phases;
      Require(phases.size()==5&&phases.back().phase=="MODEL_LOAD"&&!phases.back().completed,"failed load timing lost or marked complete");
    }
    Failure("");Throws([&]{session.Load(config);});Throws([&]{session.RunDraft(DraftInputs());});session.Close();
    events=Trace(path);Require(Count(events,"v2-run")==0&&Count(events,"process")==0&&Count(events,"v2-deinit-weights")==0&&Count(events,"v2-deinit")==1,"failed load retried or used separate weights cleanup");
  }
  for(const auto& fault:std::vector<std::string>{"original-run","original-run-unknown-exception","v1-draft-process","v1-process-unknown"}){
    Failure("");const auto path=trace(fault);ResidentSession session("target-v2-draft-v1");session.Load(config);Failure(fault);
    Throws([&]{if(fault.rfind("original-",0)==0)session.RunTarget(TargetInputs(config));else session.RunDraft(DraftInputs());});
    const auto& e=session.evidence();Require(e.closed&&e.poisoned&&e.target_runs==0&&e.draft_runs==0,"failed mixed call stayed reusable");
    Failure("");Throws([&]{session.RunDraft(DraftInputs());});session.Close();
    events=Trace(path);Require(Count(events,"v2-run")+Count(events,"process")==1&&Count(events,"v2-deinit")==1&&Count(events,"unload")==1&&Count(events,"v2-deinit-weights")==0,"failed mixed call retried or skipped peer cleanup");
  }
  for(const auto& fault:std::vector<std::string>{"original-deinit","v1-draft-unload"}){
    Failure("");const auto path=trace(fault);ResidentSession session("target-v2-draft-v1");session.Load(config);Failure(fault);
    Require(Throws([&]{session.Close();})=="SESSION_CLEANUP","cleanup failure not reported");const auto& e=session.evidence();
    Require(e.closed&&e.poisoned&&!e.cleanup_error.empty()&&e.target_lifecycle.manager_deinit_called,"cleanup failure lost");
    Require(e.target_lifecycle.manager_deinit_returned==(fault!="original-deinit"),"void DeInit return was fabricated");
    session.Close();events=Trace(path);Require(Count(events,"v2-deinit")==1&&Count(events,"unload")==1&&Count(events,"v2-deinit-weights")==0,"failed cleanup retried");
  }
  Failure("");config.diagnostics=false;
  {
    const auto path=trace("compact-memory");ResidentSession session("target-v2-draft-v1");session.Load(config);
    auto inputs=TargetInputs(config);
    for(std::size_t step=0;step<3;++step){
      auto target=session.RunTarget(inputs);
      Require(target.outputs.size()==82&&target.io_diagnostic.inputs.empty(),"compact target emitted diagnostics");
      for(std::size_t i=0;i<80;++i){Require(target.outputs[i+1][0]==step+1,"compact state feedback differs");inputs[i+3]=std::move(target.outputs[i+1]);}
      auto draft=session.RunDraft(DraftInputs());
      Require(draft.outputs.size()==1&&draft.io_diagnostic.inputs.empty()&&Top(draft)==static_cast<int>(10+step),"compact draft top1/hidden-readback policy differs");
    }
    Require(session.evidence().target_api_us>0&&session.evidence().draft_api_us>0,"API timing missing");
    Failure("original-run");Throws([&]{session.RunTarget(inputs);});
    Require(session.evidence().closed&&session.evidence().poisoned&&session.evidence().target_runs==3,"compact failure committed/reused state");
    Failure("");session.Close();
    const auto events=Trace(path);Require(Count(events,"v2-init")==1&&Count(events,"load")==1&&Count(events,"v2-deinit")==1&&Count(events,"unload")==1,"compact mode reloaded or skipped close");
  }
  Failure("");unsetenv("FAKE_HIAI_TRACE");unsetenv("FAKE_HIAI_V1_RESIDENT_TRACE");
  std::cout<<"PASS mixed session: V2 target + V1 draft, 3+3 alternating calls, 176 persistent buffers, output ownership, 12 failure cases; host fake only\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
