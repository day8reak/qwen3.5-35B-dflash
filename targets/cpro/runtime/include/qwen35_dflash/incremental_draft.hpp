#pragma once
#include "qwen35_dflash/hiai_executor.hpp"
#include <algorithm>
#include <map>
#include <stdexcept>

namespace qwen35::dflash {
// Only committed context enters these banks. Anchor/MASK KV is temporary in
// the graph, so rejecting proposals never needs to roll back these banks.
struct IncrementalDraftAbi {
  std::uint32_t context_rows=0, capacity=0, candidates=0;
  std::vector<HiaiTensorSpec> inputs, outputs;
  bool enabled() const { return context_rows!=0; }
  std::size_t bank_bytes() const { return std::size_t(8)*capacity*128*2; }
  void Validate() const {
    if((context_rows!=16&&context_rows!=64)||capacity<context_rows||capacity>4096||
       (candidates!=1&&candidates!=3&&candidates!=7&&candidates!=15))
      throw std::invalid_argument("unsupported incremental Draft geometry");
    std::vector<HiaiTensorSpec> expected_in={
      {"target_features","float16",{1,context_rows,16384}},
      {"start_position","int32",{1}},{"valid_rows","int32",{1}},
      {"anchor","int32",{1}},{"proposal_count","int32",{1}}};
    std::vector<HiaiTensorSpec> expected_out={
      {"draft_top1","int32",{1,candidates}},
      {"draft_hidden","float16",{1,candidates,2048}}};
    for(unsigned layer=0;layer<6;++layer)for(const auto* kind:{"k_","v_"}) {
      expected_in.push_back({"past_"+std::string(kind)+std::to_string(layer),"float16",{1,8,capacity,128}});
      expected_out.push_back({"present_"+std::string(kind)+std::to_string(layer),"float16",{1,8,capacity,128}});
    }
    auto check=[](const auto& actual,const auto& expected,bool ordered) {
      if(actual.size()!=expected.size())throw std::invalid_argument("incremental Draft ABI count differs");
      std::map<std::string,HiaiTensorSpec> remaining;
      for(const auto& spec:expected)remaining.emplace(spec.name,spec);
      for(std::size_t i=0;i<actual.size();++i) {
        const auto& a=actual[i];auto found=remaining.find(a.name);
        if(found==remaining.end()||found->second.dtype!=a.dtype||found->second.shape!=a.shape||
           (ordered&&a.name!=expected[i].name))throw std::invalid_argument("incremental Draft ABI slot differs: "+a.name);
        remaining.erase(found);
      }
    };
    check(inputs,expected_in,false);check(outputs,expected_out,true);
  }
};

struct IncrementalDraftStep {
  std::vector<std::uint8_t> features;
  std::uint32_t start=0, valid=0, anchor=0, proposals=0;
};

// Scheduling only: the resident executor owns all twelve persistent KV inputs.
class IncrementalDraftCache {
 public:
  using Bytes=std::vector<std::uint8_t>;
  explicit IncrementalDraftCache(IncrementalDraftAbi abi):abi_(std::move(abi)) {
    abi_.Validate();
  }
  IncrementalDraftStep Prepare(const Bytes& features,std::size_t committed,std::uint32_t anchor) {
    if(pending_||!committed||committed-1>abi_.capacity||cursor_>committed-1||
       features.size()!=committed*32768||anchor>=248320)
      throw std::invalid_argument("incremental Draft committed prefix/cache mismatch");
    valid_=std::min<std::size_t>(abi_.context_rows,committed-1-cursor_);
    final_=cursor_+valid_==committed-1;
    Bytes context(std::size_t(abi_.context_rows)*32768,0);
    std::copy_n(features.begin()+cursor_*32768,valid_*32768,context.begin());
    pending_=true;
    return {std::move(context),static_cast<std::uint32_t>(cursor_),static_cast<std::uint32_t>(valid_),
            anchor,final_?abi_.candidates:0};
  }
  // Validate tokens first, commit the SDK cache slices, then advance the cursor.
  // A failed call or malformed token never changes either persistent state.
  template<class Commit>
  std::vector<std::uint32_t> Complete(const Bytes& top1,Commit commit) {
    if(!pending_||top1.size()!=std::size_t(abi_.candidates)*4)
      throw std::invalid_argument("incremental Draft result extent/pending call differs");
    std::vector<std::uint32_t> tokens;
    if(final_)for(unsigned i=0;i<abi_.candidates;++i) {
      std::uint32_t token=0;for(unsigned b=0;b<4;++b)token|=std::uint32_t(top1[i*4+b])<<(8*b);
      if(token>=248320)throw std::invalid_argument("incremental Draft token outside vocabulary");
      tokens.push_back(token);
    }
    commit();
    cursor_+=valid_;++calls_;if(final_)++rounds_;else ++catchup_;
    pending_=false;return tokens;
  }
  std::size_t cursor()const{return cursor_;}
  std::size_t calls()const{return calls_;}
  std::size_t rounds()const{return rounds_;}
  std::size_t catchup_calls()const{return catchup_;}
 private:
  IncrementalDraftAbi abi_;
  std::size_t cursor_=0,valid_=0,calls_=0,rounds_=0,catchup_=0;
  bool pending_=false,final_=false;
};
} // namespace qwen35::dflash
