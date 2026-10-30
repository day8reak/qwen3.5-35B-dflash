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
  Require(argc==2,"expected output directory");const auto root=fs::absolute(argv[1]);fs::create_directories(root);
  std::ofstream(root/"target.omc")<<"original-target-fixture";std::ofstream(root/"draft.omc")<<"draft-resident64-fixture";
  ResidentSessionConfig config{root/"target.omc",root,root/"draft.omc","FAKE-HIAI-X90",Contract(true),Contract(false),300000,"legacy-buffer"};
  auto trace=[&](const std::string& name){const auto p=root/(name+".trace");std::ofstream(p).close();setenv("FAKE_HIAI_TRACE",p.c_str(),1);return p;};
  setenv("FAKE_HIAI_V1_RESIDENT_TRACE","1",1);Failure("");const auto repeated=trace("v1-alternating");
  {
    ResidentSession session("legacy-buffer");Require(session.evidence().model_load_api=="legacy-buffer"&&session.evidence().target_loads==0,"preload API is not explicit");
    session.Load(config);Throws([&]{session.Load(config);});auto inputs=TargetInputs(config);
    ResidentRunResult first,firstdraft;
    for(std::size_t round=1;round<=3;++round){
      for(std::size_t ref=0;ref<80;++ref)inputs[3+ref][0]=round-1;
      auto result=session.RunTarget(inputs);Diagnostic(result,89);Require(result.outputs[1][0]==round,"state update did not use latest input");
      auto draft=session.RunDraft(DraftInputs());Diagnostic(draft,3);Require(Top(draft)==static_cast<int>(9+round),"draft client was reloaded");
      if(round==1){first=std::move(result);firstdraft=std::move(draft);first.outputs[81][0]=77;}
    }
    Require(first.outputs[1][0]==1&&first.outputs[81][0]==77&&Top(firstdraft)==10,"returned outputs alias live model buffers");
    session.Close();session.Close();const auto& e=session.evidence();
    Require(e.closed&&!e.poisoned&&e.target_loads==1&&e.draft_loads==1&&e.target_runs==3&&e.draft_runs==3,"V1 session counts differ");
    for(const auto* life:{&e.target_v1_lifecycle,&e.draft_v1_lifecycle}){
      Require(life->client_init_status==0&&life->load_status==0&&life->unload_status==0&&life->closed&&life->unload_called,"raw V1 lifecycle status lost");
      Require(life->process_attempts==3&&life->successful_processes==3,"V1 Process lifecycle counters differ");
      Require(life->runtime_tensor_names=="NOT_QUERIED_V1_SAVED_NAMED_ABI_ONLY"&&life->runtime_dtypes=="DECLARED_BOUND_ABI_NOT_QUERIED","V1 introspection fabricated");
    }
    Require(e.target_v1_lifecycle.observed_inputs.size()==89&&e.target_v1_lifecycle.observed_outputs.size()==82&&e.draft_v1_lifecycle.observed_inputs.size()==3,"actual V1 dimensions not retained");
    Require(e.target_v1_lifecycle.observed_inputs[0].bytes==2048&&e.target_v1_lifecycle.observed_inputs[86].bytes==8&&e.target_v1_lifecycle.observed_inputs[88].bytes==2,"V1 typed INT8/INT64/INT16 byte widths wrong");
    Throws([&]{session.RunTarget(inputs);});
  }
  auto events=Trace(repeated);
  Require(Count(events,"load")==2&&Count(events,"process")==6&&Count(events,"unload")==2&&Count(events,"v1-create-tensor")==176,"V1 models/tensors did not remain resident");
  for(const auto& event:events)Require(event.rfind("v2-",0)!=0,"V1 path called V2 SDK");
  Require(Count(events,"v1-load:qwen35_resident_target")==1&&Count(events,"v1-load:qwen35_resident_draft")==1,"separate V1 client model names missing");
  for(const auto& fault:std::vector<std::string>{"init","v1-target-load","v1-draft-load","original-shape","original-format","original-count","descriptor-exclusive","model-buffer-null","short-buffer","gdn-input-alias","v1-cross-alias"}){
    Failure(fault);auto path=trace(fault);ResidentSession session("legacy-buffer");Throws([&]{session.Load(config);});const auto& e=session.evidence();
    Require(e.closed&&e.poisoned&&e.target_runs==0&&e.draft_runs==0,"failed V1 construction accepted or left live");
    if(fault=="init")Require(e.target_v1_lifecycle.client_init_attempted&&e.target_v1_lifecycle.client_init_status==1&&!e.target_v1_lifecycle.load_attempted&&!e.target_v1_lifecycle.unload_called,"failed Init snapshot fabricated Load or Unload");
    if(fault=="v1-target-load")Require(e.target_v1_lifecycle.load_attempted&&e.target_v1_lifecycle.load_status==1&&e.target_v1_lifecycle.unload_called&&e.target_v1_lifecycle.failure_stage=="LEGACY_LOAD","target failed-constructor status lost");
    if(fault=="v1-draft-load")Require(e.target_loads==1&&e.draft_v1_lifecycle.load_attempted&&e.draft_v1_lifecycle.load_status==1&&e.draft_v1_lifecycle.unload_called,"draft partial Load lost");
    Failure("");Throws([&]{session.Load(config);});session.Close();events=Trace(path);Require(Count(events,"process")==0,"failed load called Process");
    for(const auto& event:events)Require(event.rfind("v2-",0)!=0,"failed V1 load fell back to V2");
  }
  for(const auto& fault:std::vector<std::string>{"v1-process","v1-process-unknown","v1-replace-output","v1-draft-process"}){
    Failure("");auto path=trace(fault);ResidentSession session("legacy-buffer");session.Load(config);Failure(fault);
    Throws([&]{if(fault=="v1-draft-process")session.RunDraft(DraftInputs());else session.RunTarget(TargetInputs(config));});
    const auto& e=session.evidence();Require(e.closed&&e.poisoned&&e.target_runs==0&&e.draft_runs==0,"failed Process result promoted");
    Failure("");Throws([&]{session.RunTarget(TargetInputs(config));});Throws([&]{session.RunDraft(DraftInputs());});session.Close();
    events=Trace(path);Require(Count(events,"process")==1&&Count(events,"unload")==2,"failed Process retried or skipped peer cleanup");
  }
  for(const auto& fault:std::vector<std::string>{"v1-unload","v1-unload-throw","v1-unload-unknown","v1-draft-unload"}){
    Failure("");auto path=trace(fault);ResidentSession session("legacy-buffer");session.Load(config);auto result=session.RunTarget(TargetInputs(config));Failure(fault);
    Require(Throws([&]{session.Close();})=="SESSION_CLEANUP","Unload failure not reported");const auto& e=session.evidence();
    Require(e.target_runs==1&&e.closed&&e.poisoned&&!e.cleanup_error.empty()&&e.target_v1_lifecycle.unload_called&&e.draft_v1_lifecycle.unload_called,"Unload failure erased run or skipped peer");
    if(fault=="v1-unload")Require(e.target_v1_lifecycle.unload_status==81&&e.draft_v1_lifecycle.unload_status==81,"returned Unload status lost");
    if(fault=="v1-unload-throw"||fault=="v1-unload-unknown")Require(!e.target_v1_lifecycle.unload_status&&!e.draft_v1_lifecycle.unload_status,"throw fabricated Unload status");
    session.Close();Failure("");Require(Count(Trace(path),"unload")==2,"failed Unload retried");
  }
  // A client close is local: its peer must remain valid and Process again.
  Failure("");const auto peers=trace("peer-unload-isolation");{
    HiaiOriginalTargetV1Executor first(root/"target.omc",root,"FAKE-HIAI-X90",config.target_inputs,config.target_outputs);
    HiaiOriginalTargetV1Executor second(root/"target.omc",root,"FAKE-HIAI-X90",config.target_inputs,config.target_outputs);
    first.Close();auto inputs=TargetInputs(config);for(std::size_t i=0;i<inputs.size();++i)second.SetInput(i,inputs[i]);second.Execute();
    Require(second.CopyOutput(1)[0]==1,"UnLoadModel closed another client");second.Close();
  }Require(Count(Trace(peers),"load")==2&&Count(Trace(peers),"unload")==2,"peer lifetime counts wrong");
  // Validate input bytes before any new Process; unsupported source contracts fail closed.
  Failure("");{
    auto bad=config;bad.target_inputs[0].bytes=4096;ResidentSession session("legacy-buffer");Throws([&]{session.Load(bad);});Require(session.evidence().target_v1_lifecycle.process_attempts==0,"bad typed ABI executed");
  }
  unsetenv("FAKE_HIAI_TRACE");unsetenv("FAKE_HIAI_V1_RESIDENT_TRACE");
  std::cout<<"PASS actual V1 adapter/session: two clients, Load2 Process6 Unload2, persistent176 tensors, exact89/82 and21 failure/ownership cases; fake SDK only\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
