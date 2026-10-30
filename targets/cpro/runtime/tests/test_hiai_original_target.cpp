#include "qwen35_dflash/hiai_executor.hpp"
#include <tensor/nd_tensor_buffer.h>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

using namespace qwen35::dflash;
namespace fs = std::filesystem;
namespace {
void Require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
std::string Throws(const std::function<void()>& call) {
  try { call(); }
  catch (const HiaiRuntimeError& error) { return error.stage(); }
  throw std::runtime_error("expected HiaiRuntimeError");
}
void Failure(const char* value) {
  if (*value) setenv("FAKE_HIAI_FAILURE",value,1);
  else unsetenv("FAKE_HIAI_FAILURE");
}
std::vector<HiaiBoundRuntimeTensor> Contract(bool input) {
  const auto enums=HiaiModelIoEnums();
  std::vector<HiaiBoundRuntimeTensor> rows;
  auto add=[&](std::vector<std::int64_t> dims,const std::string& dtype,int format) {
    HiaiBoundRuntimeTensor row;
    row.descriptor.index=rows.size();
    row.descriptor.shape=std::move(dims);
    row.descriptor.raw_dtype=enums.dtypes.at(dtype);
    row.descriptor.raw_format=format;
    row.descriptor.tensor_name=std::string(input?"runtime_input_":"runtime_output_")+std::to_string(rows.size());
    row.descriptor.tensor_name_status="OBSERVED";
    row.dtype=dtype;
    row.bytes=dtype=="INT8"?1:dtype=="INT64"?8:(dtype=="FLOAT16" || dtype=="INT16")?2:4;
    for(auto dim:row.descriptor.shape) row.bytes*=static_cast<std::size_t>(dim);
    rows.push_back(std::move(row));
  };
  const auto nd=static_cast<int>(hiai::Format::ND), physical=static_cast<int>(hiai::Format::NCHW);
  if(input) { add({1,1,1,2048},"INT8",nd); add({1,1,1,2048},"FLOAT32",nd); add({1,1},"INT32",nd); }
  else add({1,1,1,248320},"FLOAT32",nd);
  for(std::size_t i=0;i<80;++i) {
    const auto kind=i%8;
    add(kind==0 || kind==2 || kind==4 ? std::vector<std::int64_t>{1,8192,4} :
        kind==1 || kind==3 || kind==5 ? std::vector<std::int64_t>{1,32,128,128} :
                                     std::vector<std::int64_t>{64,32,32,16},"FLOAT16",physical);
  }
  if(input) {
    add({1},"INT32",nd); add({1},"INT32",nd); add({1,1,1},"FLOAT32",nd);
    add({1},"INT64",nd); add({128},"INT32",nd); add({1},"INT16",nd);
  } else add({1,1,1,16384},"FLOAT16",nd);
  return rows;
}
std::vector<std::string> Trace(const fs::path& path) {
  std::ifstream file(path); std::vector<std::string> events;
  for(std::string line;std::getline(file,line);) events.push_back(line);
  return events;
}
std::size_t Count(const std::vector<std::string>& events,const std::string& value) {
  return static_cast<std::size_t>(std::count(events.begin(),events.end(),value));
}
void Order(const std::vector<std::string>& events,const std::string& a,const std::string& b) {
  const auto left=std::find(events.begin(),events.end(),a),right=std::find(events.begin(),events.end(),b);
  Require(left!=events.end() && right!=events.end() && left<right,"lifecycle order differs: "+a+" before "+b);
}
void Supply(HiaiOriginalTargetExecutor& executor,const std::vector<HiaiBoundRuntimeTensor>& specs) {
  for(std::size_t i=0;i<specs.size();++i)
    executor.SetInput(i,std::vector<std::uint8_t>(specs[i].bytes,static_cast<std::uint8_t>(i+1)));
}
}
int main(int argc,char** argv) {
  try {
    Require(argc==2,"expected output directory");
    const auto root=fs::absolute(argv[1]); fs::create_directories(root);
    const auto model=root/"model.omc",weights=root/"weights";
    fs::create_directories(weights); std::ofstream(model)<<"original-target-fixture";
    const auto in=Contract(true),out=Contract(false);
    auto trace_case=[&](const std::string& name) {
      const auto path=root/(name+".trace"); std::ofstream(path).close();
      setenv("FAKE_HIAI_TRACE",path.c_str(),1); return path;
    };
    Failure("");
    const auto repeated=trace_case("repeated");
    {
      HiaiOriginalTargetExecutor executor(model,weights,"FAKE-HIAI-X90",in,out);
      Require(executor.observed_inputs().size()==89 && executor.observed_outputs().size()==82,"observed89/82 missing");
      Require(executor.observed_inputs()[0].dtype=="INT8" && executor.observed_inputs()[86].dtype=="INT64","typed original inputs missing");
      Require(executor.observed_inputs()[3].descriptor.raw_format==static_cast<int>(hiai::Format::NCHW),"physical format rewritten");
      Throws([&]{executor.CopyOutput(0);});
      std::vector<std::vector<std::uint8_t>> states;
      for(std::size_t i=0;i<80;++i) states.emplace_back(in[i+3].bytes,static_cast<std::uint8_t>(i+4));
      for(std::size_t round=1;round<=3;++round) {
        for(std::size_t i=0;i<89;++i) {
          if(i>=3 && i<83) executor.SetInput(i,states[i-3]);
          else executor.SetInput(i,std::vector<std::uint8_t>(in[i].bytes,static_cast<std::uint8_t>(round)));
        }
        executor.Execute();
        for(std::size_t i=0;i<80;++i) {
          auto value=executor.CopyOutput(i+1);
          Require(value.size()==states[i].size() && value[0]==i+4+round,"state feedback content/extent differs");
          Require(std::equal(value.begin()+1,value.end(),states[i].begin()+1),"state tail modified unexpectedly");
          states[i]=std::move(value);
        }
        const auto features=executor.CopyOutput(81);
        Require(std::all_of(features.begin(),features.end(),[&](auto byte){return byte==round;}),"latest features differ");
        Require(executor.io_diagnostic().process_called && executor.io_diagnostic().inputs.size()==89,"Run/input diagnostic missing");
        for(const auto& d:executor.io_diagnostic().inputs) Require(d.readback_matches_source,"input readback failed");
        for(const auto& d:executor.io_diagnostic().outputs) Require(d.checked_after_process && d.object_same && d.buffer_same,"output identity diagnostic failed");
      }
      Require(executor.lifecycle().run_attempts==3 && executor.lifecycle().successful_runs==3,"repeated Run counts differ");
      // Last step outputs become unavailable as soon as any new input is bound.
      executor.SetInput(0,std::vector<std::uint8_t>(in[0].bytes,0));
      Throws([&]{executor.CopyOutput(81);}); Throws([&]{executor.Execute();});
      executor.Close(); executor.Close();
      Require(executor.lifecycle().closed && executor.lifecycle().weights_deinit_status==0,"explicit cleanup missing");
      Require(executor.lifecycle().weight_init_policy=="INIT_BEFORE_WEIGHTS_NO_FALLBACK"&&
              executor.lifecycle().weight_cleanup_policy=="DEINIT_WEIGHTS_BEFORE_DEINIT","explicit lifecycle policies missing");
      Throws([&]{executor.SetInput(0,std::vector<std::uint8_t>(in[0].bytes));});
    }
    auto events=Trace(repeated);
    for(const auto* one:{"v2-restore-file","v2-create-manager","v2-init-weights","v2-init","v2-deinit","v2-deinit-weights","v2-manager-destroy","v2-built-model-destroy"})
      Require(Count(events,one)==1,std::string("expected exactly one lifecycle event: ")+one);
    Require(Count(events,"v2-run")==3,"expected exactly3 resident Run calls");
    Order(events,"v2-restore-file","v2-init"); Order(events,"v2-init","v2-init-weights");
    Order(events,"v2-deinit-weights","v2-deinit"); Order(events,"v2-deinit","v2-manager-destroy");
    Order(events,"v2-manager-destroy","v2-built-model-destroy");

    const std::vector<std::pair<std::string,std::string>> construction_faults={
      {"original-init-weights","INIT_WEIGHTS_AFTER_INIT"},
      {"original-init","MODEL_MANAGER_INIT"},{"original-name","TENSOR_DESCRIPTORS"},
      {"original-dtype","TENSOR_DESCRIPTORS"},{"original-format","TENSOR_DESCRIPTORS"},
      {"original-shape","TENSOR_DESCRIPTORS"},{"original-count","TENSOR_DESCRIPTORS"},
      {"v2-allocation","INPUT_TENSOR_ALLOCATION"},{"v2-short-buffer","INPUT_TENSOR_ALLOCATION"},
      // Constructor storage is all zero. Byte/hash comparison alone cannot
      // expose these aliases; exact live buffer-range checks must reject them.
      {"original-input-alias","INPUT_TENSOR_ALLOCATION"},
      {"original-input-partial-alias","INPUT_TENSOR_ALLOCATION"},
      {"original-output-input-alias","OUTPUT_TENSOR_ALLOCATION"},
      {"original-output-alias","OUTPUT_TENSOR_ALLOCATION"}};
    for(const auto& fault:construction_faults) {
      Failure(fault.first.c_str()); const auto trace=trace_case(fault.first);
      Require(Throws([&]{HiaiOriginalTargetExecutor executor(model,weights,"FAKE-HIAI-X90",in,out);})==fault.second,"construction failure stage differs: "+fault.first);
      events=Trace(trace); Require(Count(events,"v2-init")==1 && Count(events,"v2-run")==0,"constructor retried or ran model");
      if(fault.first=="original-init-weights" || fault.first=="original-init") {
        Require(Count(events,"v2-init-weights")==static_cast<std::size_t>(fault.first=="original-init-weights") &&
                Count(events,"v2-deinit")==1 && Count(events,"v2-deinit-weights")==0,"failed initialization cleaned unowned weights or leaked manager");
      } else {
        Require(Count(events,"v2-init-weights")==1,"weights initialization omitted");
        Require(Count(events,"v2-deinit")==1 && Count(events,"v2-deinit-weights")==1,"successful weights scope not released exactly once");
        Order(events,"v2-deinit-weights","v2-deinit");
      }
    }
    for(const auto* fault:{"original-run","original-replace-output","original-run-unknown-exception"}) {
      Failure(fault); const auto trace=trace_case(fault);
      {
        HiaiOriginalTargetExecutor executor(model,weights,"FAKE-HIAI-X90",in,out);
        Supply(executor,in); Throws([&]{executor.Execute();});
        Require(executor.lifecycle().successful_runs==0,"failed Run promoted success");
        Failure(""); // Clearing the injected fault must not unpoison/retry it.
        Throws([&]{executor.SetInput(0,std::vector<std::uint8_t>(in[0].bytes));});
        Throws([&]{executor.Execute();}); Throws([&]{executor.CopyOutput(0);});
      }
      events=Trace(trace);
      Require(Count(events,"v2-run")==1,"failed executor retried");
      Require(Count(events,"v2-deinit")==1 && Count(events,"v2-deinit-weights")==1,"Run failure leaked lifecycle");
    }
    Failure("");
    const auto late=trace_case("late-output-mutation");
    {
      HiaiOriginalTargetExecutor executor(model,weights,"FAKE-HIAI-X90",in,out);
      Supply(executor,in); executor.Execute();
      Failure("v2-short-buffer"); Throws([&]{executor.CopyOutput(1);});
      Failure("");
      Throws([&]{executor.CopyOutput(1);});
      Throws([&]{executor.SetInput(0,std::vector<std::uint8_t>(in[0].bytes));});
    }
    Require(Count(Trace(late),"v2-run")==1,"late output mutation caused retry");
    for(const auto* fault:{"original-deinit-weights","original-deinit-weights-throw","original-deinit-weights-unknown","original-both-cleanup-failures"}) {
      Failure(""); const auto cleanup=trace_case(fault);
      HiaiOriginalTargetExecutor executor(model,weights,"FAKE-HIAI-X90",in,out);
      Failure(fault);
      Require(Throws([&]{executor.Close();})=="CLEANUP_FAILED","cleanup error not reported");
      const auto returned=std::string(fault)=="original-deinit-weights"||std::string(fault)=="original-both-cleanup-failures";
      Require(executor.lifecycle().weights_deinit_status==(returned?73:-1),"cleanup Status lost or fabricated");
      Require(executor.lifecycle().manager_deinit_called && executor.lifecycle().closed,"weights failure skipped manager cleanup");
      if(std::string(fault)=="original-both-cleanup-failures")
        Require(executor.lifecycle().cleanup_error.find("DeinitWeights failed: 73; injected target DeInit exception")==0,"first cleanup error overwritten");
      executor.Close();
      const auto cleanup_events=Trace(cleanup);
      Require(Count(cleanup_events,"v2-deinit-weights")==1&&Count(cleanup_events,"v2-deinit")==1,"failed cleanup retried or skipped manager");
      Order(cleanup_events,"v2-deinit-weights","v2-deinit");
    }
    Failure("");
    for(const auto* fault:{"declared-raw-dtype","declared-extent","duplicate-name"}) {
      auto bad=in;
      if(std::string(fault)=="declared-raw-dtype") ++bad[0].descriptor.raw_dtype;
      if(std::string(fault)=="declared-extent") ++bad[0].bytes;
      if(std::string(fault)=="duplicate-name") bad[1].descriptor.tensor_name=bad[0].descriptor.tensor_name;
      const auto trace=trace_case(fault);
      Require(Throws([&]{HiaiOriginalTargetExecutor executor(model,weights,"FAKE-HIAI-X90",bad,out);})=="VALIDATE_ORIGINAL_TARGET_CONTRACT","invalid declaration passed contract gate");
      Require(Count(Trace(trace),"v2-restore-file")==0,"invalid contract loaded model");
    }
    unsetenv("FAKE_HIAI_TRACE");
    std::cout<<"PASS original target adapter: repeated3 Run, full89/82 extents, exact cleanup and24 failure cases; host fake only\n";
    return 0;
  } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
