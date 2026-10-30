#include "qwen35_dflash/resident_session.hpp"
#include <algorithm>
#include <chrono>
#include <exception>
#include <limits>

namespace qwen35::dflash {
namespace {
using Clock = std::chrono::steady_clock;
std::uint64_t Milliseconds(Clock::time_point start) {
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now()-start).count());
}
const std::vector<HiaiTensorSpec> kDraftInputs = {
  {"target_features", "float16", {1,64,16384}},
  {"input_ids", "int32", {1,64}}, {"attention_mask", "int32", {1,64}}};
const std::vector<HiaiTensorSpec> kDraftOutputs = {
  {"draft_top1", "int32", {1,15}}, {"draft_hidden", "float16", {1,15,2048}}};
std::string ErrorStage(const std::exception& error,const std::string& stage) {
  const auto* precise=dynamic_cast<const HiaiRuntimeError*>(&error);
  return precise ? stage+"/"+precise->stage() : stage;
}
}
struct ResidentSession::Impl {
  std::unique_ptr<HiaiOriginalTargetExecutor> target;
  std::unique_ptr<HiaiStatefulExecutor> draft;
  ResidentSessionEvidence record;
  std::vector<std::size_t> input_sizes,output_sizes;
  std::vector<HiaiTensorStorage> storage;
  bool load_attempted=false;

  std::vector<HiaiTensorStorage> CurrentStorage() const {
    auto result=target->tensor_storage();
    const auto other=draft->tensor_storage();
    result.insert(result.end(),other.begin(),other.end());
    return result;
  }
  void VerifyStorage(bool first=false) {
    const auto current=CurrentStorage();
    if(current.size()!=89+82+3+2 || (!first && current.size()!=storage.size()))
      throw HiaiRuntimeError("LIVE_STORAGE","resident tensor count changed");
    for(std::size_t i=0;i<current.size();++i) {
      const auto& row=current[i];
      const auto begin=reinterpret_cast<std::uintptr_t>(row.data);
      if(!row.object || !row.data || !row.bytes || begin>UINTPTR_MAX-row.bytes)
        throw HiaiRuntimeError("LIVE_STORAGE","resident tensor extent/address is invalid");
      for(std::size_t j=0;j<i;++j) {
        const auto& old=current[j];const auto other=reinterpret_cast<std::uintptr_t>(old.data);
        if(row.object==old.object || (begin<other+old.bytes && other<begin+row.bytes))
          throw HiaiRuntimeError("LIVE_STORAGE","resident model tensor allocations overlap");
      }
      if(!first && (row.object!=storage[i].object || row.data!=storage[i].data || row.bytes!=storage[i].bytes))
        throw HiaiRuntimeError("LIVE_STORAGE","resident tensor storage changed during session");
    }
    if(first)storage=current;
  }
  void RequireLive() const {
    if(record.closed || record.poisoned || !target || !draft)
      throw HiaiRuntimeError(record.stage,"resident session is closed, failed or not loaded");
  }
  void Cleanup(bool report_error) {
    if(record.closed)return;
    const auto start=Clock::now();
    record.closed=true;
    // Keep target alive until the draft manager/tensors finish shutdown.
    if(draft){
      ++record.draft_close_calls;
      try {draft->Close();}
      catch(const std::exception& error){record.cleanup_error="draft: "+std::string(error.what());}
      catch(...){record.cleanup_error="draft: unknown cleanup exception";}
      draft.reset();
    }
    if(target) {
      ++record.target_close_calls;
      try {target->Close();}
      catch(const std::exception& error){record.cleanup_error+=" target: "+std::string(error.what());}
      catch(...){record.cleanup_error+=" target: unknown cleanup exception";}
      record.target_lifecycle=target->lifecycle();target.reset();
    }
    storage.clear();record.close_ms+=Milliseconds(start);
    if(!record.cleanup_error.empty()) {
      record.poisoned=true;record.status="FAILED";
      if(report_error){record.stage="SESSION_CLEANUP";throw HiaiRuntimeError(record.stage,record.cleanup_error);}
    }
  }
  [[noreturn]] void Fail(const std::string& stage,const std::string& message) {
    record.stage=stage;record.poisoned=true;record.status="FAILED";
    Cleanup(false);
    throw HiaiRuntimeError(stage,message);
  }
  ResidentRunResult Run(const std::vector<std::vector<std::uint8_t>>& inputs,bool is_target) {
    RequireLive();
    record.stage=is_target?"TARGET_RUN":"DRAFT_RUN";
    const auto begin=Clock::now();
    auto& attempts=is_target?record.target_run_attempts:record.draft_run_attempts;
    auto& total=is_target?record.target_run_ms:record.draft_run_ms;
    ++attempts;
    try {
      const auto count=is_target?input_sizes.size():kDraftInputs.size();
      if(inputs.size()!=count)throw std::invalid_argument("resident call must supply every ordered input");
      // Validate the complete list before modifying any live tensor.
      for(std::size_t i=0;i<count;++i)
        if(inputs[i].size()!=(is_target?input_sizes[i]:HiaiTensorBytes(kDraftInputs[i])))
          throw std::invalid_argument("resident input differs from exact bound byte extent");
      VerifyStorage();
      for(std::size_t i=0;i<count;++i) {
        if(is_target)target->SetInput(i,inputs[i]);else draft->SetInput(i,inputs[i]);
      }
      VerifyStorage();
      if(is_target)target->Execute();else draft->Execute();
      VerifyStorage();
      ResidentRunResult result;
      const auto outputs=is_target?output_sizes.size():kDraftOutputs.size();
      for(std::size_t i=0;i<outputs;++i) {
        auto bytes=is_target?target->CopyOutput(i):draft->CopyOutput(i);
        if(bytes.size()!=(is_target?output_sizes[i]:HiaiTensorBytes(kDraftOutputs[i])))
          throw std::runtime_error("resident copied output extent changed");
        result.outputs.push_back(std::move(bytes));
      }
      VerifyStorage();
      result.io_diagnostic=is_target?target->io_diagnostic():draft->io_diagnostic();
      if(!result.io_diagnostic.process_called || result.io_diagnostic.inputs.size()!=count ||
          result.io_diagnostic.outputs.size()!=outputs)
        throw std::runtime_error("resident copied IO diagnostic is incomplete");
      for(const auto& d:result.io_diagnostic.inputs)
        if(!d.readback_matches_source)throw std::runtime_error("resident input readback differs");
      for(const auto& d:result.io_diagnostic.outputs)
        if(!d.checked_after_process || !d.object_same || !d.buffer_same)
          throw std::runtime_error("resident output storage identity differs");
      result.wall_ms=Milliseconds(begin);total+=result.wall_ms;
      if(is_target){++record.target_runs;record.target_lifecycle=target->lifecycle();}
      else ++record.draft_runs;
      record.stage="READY";return result;
    } catch(const std::exception& error) {
      total+=Milliseconds(begin);Fail(ErrorStage(error,record.stage),error.what());
    } catch(...) {total+=Milliseconds(begin);Fail(record.stage,"unknown resident Run exception");}
  }
};

ResidentSession::ResidentSession():impl_(std::make_unique<Impl>()) {}
ResidentSession::~ResidentSession(){impl_->Cleanup(false);}
void ResidentSession::Load(const ResidentSessionConfig& config) {
  if(impl_->load_attempted || impl_->record.closed || impl_->record.poisoned)
    throw HiaiRuntimeError(impl_->record.stage,"resident session Load is allowed exactly once");
  impl_->load_attempted=true;auto& r=impl_->record;r.status="LOADING";r.stage="VALIDATE_SESSION";
  try {
    if(config.target_inputs.size()!=89 || config.target_outputs.size()!=82 || !config.timeout_ms ||
        !config.target_model.is_absolute() || !config.draft_model.is_absolute() || config.runtime_version.empty())
      throw std::invalid_argument("resident session requires exact target89/82 and absolute model paths");
    for(const auto& row:config.target_inputs)impl_->input_sizes.push_back(row.bytes);
    for(const auto& row:config.target_outputs)impl_->output_sizes.push_back(row.bytes);
    r.stage="TARGET_LOAD";++r.target_load_attempts;auto start=Clock::now();
    try {impl_->target=std::make_unique<HiaiOriginalTargetExecutor>(config.target_model,
        config.target_weight_directory,config.runtime_version,config.target_inputs,config.target_outputs,r.target_model_name);}
    catch(...){r.target_load_ms+=Milliseconds(start);throw;}
    r.target_load_ms+=Milliseconds(start);++r.target_loads;r.target_lifecycle=impl_->target->lifecycle();
    r.stage="DRAFT_LOAD";++r.draft_load_attempts;start=Clock::now();
    try {impl_->draft=std::make_unique<HiaiStatefulExecutor>(config.draft_model,config.runtime_version,
        kDraftInputs,kDraftOutputs,config.timeout_ms,false,0,true,"file-v2",r.draft_model_name);}
    catch(...){r.draft_load_ms+=Milliseconds(start);throw;}
    r.draft_load_ms+=Milliseconds(start);++r.draft_loads;
    r.stage="VALIDATE_LIVE_STORAGE";impl_->VerifyStorage(true);
    r.status="READY";r.stage="READY";
  } catch(const std::exception& error){impl_->Fail(ErrorStage(error,r.stage),error.what());}
  catch(...){impl_->Fail(r.stage,"unknown resident Load exception");}
}
ResidentRunResult ResidentSession::RunTarget(const std::vector<std::vector<std::uint8_t>>& inputs){return impl_->Run(inputs,true);}
ResidentRunResult ResidentSession::RunDraft(const std::vector<std::vector<std::uint8_t>>& inputs){return impl_->Run(inputs,false);}
void ResidentSession::Close(){impl_->Cleanup(true);if(!impl_->record.poisoned){impl_->record.stage="CLOSED";impl_->record.status="CLOSED";}}
const ResidentSessionEvidence& ResidentSession::evidence() const noexcept{return impl_->record;}
}  // namespace qwen35::dflash
