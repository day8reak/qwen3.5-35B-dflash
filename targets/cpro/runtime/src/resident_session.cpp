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
  std::unique_ptr<HiaiOriginalTargetV1Executor> target_v1;
  std::unique_ptr<HiaiStatefulExecutor> draft;
  ResidentSessionEvidence record;
  std::vector<std::size_t> input_sizes,output_sizes;
  std::vector<HiaiTensorStorage> storage;
  bool load_attempted=false, diagnostics=true, incremental=false;
  IncrementalDraftAbi incremental_abi;
  bool kv_pending=false;
  std::uint32_t kv_start=0,kv_valid=0;
  std::vector<HiaiTensorSpec> draft_inputs=kDraftInputs,draft_outputs=kDraftOutputs;

  std::vector<HiaiTensorStorage> CurrentStorage() const {
    auto result=target_v1?target_v1->tensor_storage():target->tensor_storage();
    const auto other=draft->tensor_storage();
    result.insert(result.end(),other.begin(),other.end());
    return result;
  }
  void VerifyStorage(bool first=false) {
    const auto current=CurrentStorage();
    if(current.size()!=input_sizes.size()+output_sizes.size()+draft_inputs.size()+draft_outputs.size() || (!first && current.size()!=storage.size()))
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
    if(record.closed || record.poisoned || (!target&&!target_v1) || !draft)
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
      if(record.model_load_api!="file-v2")record.draft_v1_lifecycle=draft->legacy_lifecycle();
      draft.reset();
    }
    if(target_v1) {
      ++record.target_close_calls;
      try {target_v1->Close();}
      catch(const std::exception& error){record.cleanup_error+=" target: "+std::string(error.what());}
      catch(...){record.cleanup_error+=" target: unknown cleanup exception";}
      record.target_v1_lifecycle=target_v1->lifecycle();target_v1.reset();
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
  ResidentRunResult Run(const std::vector<std::vector<std::uint8_t>>& inputs,bool is_target,
                        const IncrementalDraftStep* step=nullptr) {
    RequireLive();
    record.stage=is_target?"TARGET_RUN":"DRAFT_RUN";
    const auto begin=Clock::now();
    auto& attempts=is_target?record.target_run_attempts:record.draft_run_attempts;
    auto& total=is_target?record.target_run_ms:record.draft_run_ms;
    ++attempts;
    try {
      const auto count=is_target?input_sizes.size():draft_inputs.size();
      if(step&&(is_target||!incremental))throw std::invalid_argument("incremental step requires its compiled Draft ABI");
      if(!is_target&&incremental) {
        if(!step||diagnostics||kv_pending||step->start!=record.draft_kv_rows||
           step->start>incremental_abi.capacity||step->valid>incremental_abi.context_rows||
           step->valid>incremental_abi.capacity-step->start||step->anchor>=248320||
           (step->proposals!=0&&step->proposals!=incremental_abi.candidates))
          throw std::invalid_argument("incremental Draft step/cursor/pending commit differs");
      }
      if(inputs.size()!=count)throw std::invalid_argument("resident call must supply every ordered input");
      // Validate the complete list before modifying any live tensor.
      for(std::size_t i=0;i<count;++i) {
        const bool retained=step&&draft_inputs[i].name.rfind("past_",0)==0;
        const auto expected=is_target?input_sizes[i]:(retained?0:HiaiTensorBytes(draft_inputs[i]));
        if(inputs[i].size()!=expected)
          throw std::invalid_argument("resident input differs from exact bound byte extent");
      }
      if(diagnostics)VerifyStorage();
      for(std::size_t i=0;i<count;++i) {
        if(is_target){if(target_v1)target_v1->SetInput(i,inputs[i]);else target->SetInput(i,inputs[i]);}
        else if(step&&draft_inputs[i].name.rfind("past_",0)==0)draft->ReuseInput(i);
        else draft->SetInput(i,inputs[i]);
      }
      if(diagnostics)VerifyStorage();
      if(is_target){if(target_v1)target_v1->Execute();else target->Execute();}else draft->Execute();
      if(diagnostics)VerifyStorage();
      ResidentRunResult result;
      const auto outputs=is_target?output_sizes.size():(diagnostics?draft_outputs.size():1);
      for(std::size_t i=0;i<outputs;++i) {
        auto bytes=is_target?(target_v1?target_v1->CopyOutput(i):target->CopyOutput(i)):draft->CopyOutput(i);
        if(bytes.size()!=(is_target?output_sizes[i]:HiaiTensorBytes(draft_outputs[i])))
          throw std::runtime_error("resident copied output extent changed");
        result.outputs.push_back(std::move(bytes));
      }
      if(diagnostics)VerifyStorage();
      if(diagnostics) {
        result.io_diagnostic=is_target?(target_v1?target_v1->io_diagnostic():target->io_diagnostic()):draft->io_diagnostic();
        if(!result.io_diagnostic.process_called || result.io_diagnostic.inputs.size()!=count ||
            result.io_diagnostic.outputs.size()!=outputs)
          throw std::runtime_error("resident copied IO diagnostic is incomplete");
        for(const auto& d:result.io_diagnostic.inputs)
          if(!d.readback_matches_source)throw std::runtime_error("resident input readback differs");
        for(const auto& d:result.io_diagnostic.outputs)
          if(!d.checked_after_process || !d.object_same || !d.buffer_same)
            throw std::runtime_error("resident output storage identity differs");
      }
      result.api_us=is_target?(target?target->last_api_us():0):draft->last_api_us();
      (is_target?record.target_api_us:record.draft_api_us)+=result.api_us;
      result.wall_ms=Milliseconds(begin);total+=result.wall_ms;
      if(is_target){++record.target_runs;if(target_v1)record.target_v1_lifecycle=target_v1->lifecycle();else record.target_lifecycle=target->lifecycle();}
      else {++record.draft_runs;if(record.model_load_api!="file-v2")record.draft_v1_lifecycle=draft->legacy_lifecycle();}
      if(step){kv_pending=true;kv_start=step->start;kv_valid=step->valid;}
      record.stage="READY";return result;
    } catch(const std::exception& error) {
      total+=Milliseconds(begin);Fail(ErrorStage(error,record.stage),error.what());
    } catch(...) {total+=Milliseconds(begin);Fail(record.stage,"unknown resident Run exception");}
  }
  void CommitIncremental() {
    RequireLive();record.stage="DRAFT_CACHE_COMMIT";
    const auto begin=Clock::now();
    try {
      if(!incremental||!kv_pending)throw std::logic_error("no incremental Draft result to commit");
      std::vector<HiaiInputSlice> slices;
      if(kv_valid)for(std::size_t i=0;i<draft_inputs.size();++i) {
        const auto& name=draft_inputs[i].name;
        if(name.rfind("past_",0)!=0)continue;
        const auto output="present_"+name.substr(5);
        const auto found=std::find_if(draft_outputs.begin(),draft_outputs.end(),
          [&](const auto& spec){return spec.name==output;});
        if(found==draft_outputs.end())throw std::logic_error("missing incremental output cache");
        slices.push_back({i,static_cast<std::size_t>(found-draft_outputs.begin()),
          std::size_t(kv_start)*256,std::size_t(kv_valid)*256,std::size_t(incremental_abi.capacity)*256,8});
      }
      const auto bytes=draft->CommitInputSlices(slices);
      record.draft_kv_commit_bytes+=bytes;record.draft_kv_rows+=kv_valid;++record.draft_kv_commits;
      record.draft_kv_commit_us+=std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-begin).count();
      kv_pending=false;record.stage="READY";
    }catch(const std::exception& error){Fail(ErrorStage(error,record.stage),error.what());}
    catch(...){Fail(record.stage,"unknown incremental cache commit exception");}
  }
};

ResidentSession::ResidentSession(const std::string& api):impl_(std::make_unique<Impl>()) {
  if(api!="file-v2"&&api!="legacy-v1-mixed"&&api!="target-v2-draft-v1")throw std::invalid_argument("resident model API must be an explicit supported per-model policy");
  impl_->record.model_load_api=api;
  if(api=="target-v2-draft-v1")impl_->record.target_lifecycle.weight_cleanup_policy="MANAGER_DEINIT_ONLY_NO_SEPARATE_WEIGHT_UNLOAD";
  impl_->record.target_v1_lifecycle.model_name=impl_->record.target_model_name;
  impl_->record.target_v1_lifecycle.model_load_api="legacy-file";
  impl_->record.draft_v1_lifecycle.model_name=impl_->record.draft_model_name;
}
ResidentSession::~ResidentSession(){impl_->Cleanup(false);}
void ResidentSession::Load(const ResidentSessionConfig& config) {
  if(impl_->load_attempted || impl_->record.closed || impl_->record.poisoned)
    throw HiaiRuntimeError(impl_->record.stage,"resident session Load is allowed exactly once");
  impl_->load_attempted=true;auto& r=impl_->record;r.status="LOADING";r.stage="VALIDATE_SESSION";
  try {
    impl_->diagnostics=config.diagnostics;
    if(config.incremental_draft.enabled()) {
      config.incremental_draft.Validate();
      if(config.model_load_api!="target-v2-draft-v1")throw std::invalid_argument("incremental Draft requires mixed residency");
      if(config.diagnostics)throw std::invalid_argument("incremental Draft requires compact residency");
      impl_->incremental=true;impl_->draft_inputs=config.incremental_draft.inputs;impl_->draft_outputs=config.incremental_draft.outputs;
      impl_->incremental_abi=config.incremental_draft;
    }
    if(config.draft_preread && r.model_load_api!="target-v2-draft-v1")
      throw std::invalid_argument("draft preread requires mixed runtime");
    if(!config.diagnostics && r.model_load_api!="target-v2-draft-v1")throw std::invalid_argument("compact memory requires mixed runtime");
    if(config.model_load_api!=r.model_load_api)throw std::invalid_argument("resident configured API differs from session policy");
    if(config.target_inputs.size()!=89 || config.target_outputs.size()!=82 || !config.timeout_ms ||
        !config.target_model.is_absolute() || !config.draft_model.is_absolute() || config.runtime_version.empty())
      throw std::invalid_argument("resident session requires exact target89/82 and absolute model paths");
    for(const auto& row:config.target_inputs)impl_->input_sizes.push_back(row.bytes);
    for(const auto& row:config.target_outputs)impl_->output_sizes.push_back(row.bytes);
    r.stage="TARGET_LOAD";++r.target_load_attempts;auto start=Clock::now();
    try {
      if(r.model_load_api=="legacy-v1-mixed")impl_->target_v1=std::make_unique<HiaiOriginalTargetV1Executor>(config.target_model,
        config.target_weight_directory,config.runtime_version,config.target_inputs,config.target_outputs,config.timeout_ms,r.target_model_name,&r.target_v1_lifecycle);
      else impl_->target=std::make_unique<HiaiOriginalTargetExecutor>(config.target_model,
        config.target_weight_directory,config.runtime_version,config.target_inputs,config.target_outputs,r.target_model_name,&r.target_lifecycle,
        r.model_load_api=="target-v2-draft-v1"?HiaiOriginalTargetCleanupPolicy::ManagerDeinitOnly:HiaiOriginalTargetCleanupPolicy::ExplicitWeightsBeforeManager,config.diagnostics);
    } catch(...){r.target_load_ms+=Milliseconds(start);
      const auto& error=r.model_load_api=="legacy-v1-mixed"?r.target_v1_lifecycle.cleanup_error:r.target_lifecycle.cleanup_error;
      if(!error.empty())r.cleanup_error="target construction: "+error;
      throw;}
    r.target_load_ms+=Milliseconds(start);++r.target_loads;
    if(impl_->target_v1)r.target_v1_lifecycle=impl_->target_v1->lifecycle();else r.target_lifecycle=impl_->target->lifecycle();
    r.stage="DRAFT_LOAD";++r.draft_load_attempts;start=Clock::now();
    try {impl_->draft=std::make_unique<HiaiStatefulExecutor>(config.draft_model,config.runtime_version,
        impl_->draft_inputs,impl_->draft_outputs,config.timeout_ms,false,impl_->incremental?-1:0,true,r.model_load_api=="file-v2"?"file-v2":"legacy-buffer",r.draft_model_name,&r.draft_v1_lifecycle,
        impl_->target_v1?impl_->target_v1->tensor_storage():impl_->target->tensor_storage(),config.diagnostics,config.draft_preread);}
    catch(...){r.draft_load_ms+=Milliseconds(start);if(!r.draft_v1_lifecycle.cleanup_error.empty())r.cleanup_error+=" draft construction: "+r.draft_v1_lifecycle.cleanup_error;throw;}
    r.draft_load_ms+=Milliseconds(start);++r.draft_loads;
    if(r.model_load_api!="file-v2")r.draft_v1_lifecycle=impl_->draft->legacy_lifecycle();
    r.stage="VALIDATE_LIVE_STORAGE";impl_->VerifyStorage(true);
    r.status="READY";r.stage="READY";
  } catch(const std::exception& error){impl_->Fail(ErrorStage(error,r.stage),error.what());}
  catch(...){impl_->Fail(r.stage,"unknown resident Load exception");}
}
ResidentRunResult ResidentSession::RunTarget(const std::vector<std::vector<std::uint8_t>>& inputs){return impl_->Run(inputs,true);}
ResidentRunResult ResidentSession::RunDraft(const std::vector<std::vector<std::uint8_t>>& inputs){return impl_->Run(inputs,false);}
ResidentRunResult ResidentSession::RunIncrementalDraft(const IncrementalDraftStep& step) {
  using Bytes=std::vector<std::uint8_t>;
  auto scalar=[](std::uint32_t v){Bytes bytes(4);for(unsigned b=0;b<4;++b)bytes[b]=(v>>(8*b))&255;return bytes;};
  std::map<std::string,Bytes> controls={{"target_features",step.features},{"start_position",scalar(step.start)},
    {"valid_rows",scalar(step.valid)},{"anchor",scalar(step.anchor)},{"proposal_count",scalar(step.proposals)}};
  std::vector<Bytes> inputs;
  for(const auto& spec:impl_->draft_inputs)
    inputs.push_back(spec.name.rfind("past_",0)==0?Bytes{}:std::move(controls.at(spec.name)));
  return impl_->Run(inputs,false,&step);
}
void ResidentSession::CommitIncrementalDraft(){impl_->CommitIncremental();}
void ResidentSession::Close(){impl_->Cleanup(true);if(!impl_->record.poisoned){impl_->record.stage="CLOSED";impl_->record.status="CLOSED";}}
const ResidentSessionEvidence& ResidentSession::evidence() const noexcept{return impl_->record;}
}  // namespace qwen35::dflash
