// Host cache transport unit test; no HIAI implementation or numerical claims.
#include "qwen35_dflash/incremental_draft.hpp"
#include <cassert>
#include <functional>
using namespace qwen35::dflash;
using B=std::vector<std::uint8_t>;
void Bad(const std::function<void()>& f){bool failed=false;try{f();}catch(const std::exception&){failed=true;}assert(failed);}
IncrementalDraftAbi Abi(unsigned k=15){
  IncrementalDraftAbi a;a.context_rows=16;a.capacity=32;a.candidates=k;
  a.inputs={{"target_features","float16",{1,16,16384}},{"start_position","int32",{1}},
    {"valid_rows","int32",{1}},{"anchor","int32",{1}},{"proposal_count","int32",{1}}};
  a.outputs={{"draft_top1","int32",{1,k}},{"draft_hidden","float16",{1,k,2048}}};
  for(unsigned i=0;i<12;++i){auto n=std::string(i%2?"v_":"k_")+std::to_string(i/2);
    a.inputs.push_back({"past_"+n,"float16",{1,8,32,128}});a.outputs.push_back({"present_"+n,"float16",{1,8,32,128}});}
  return a;
}
std::vector<B> Output(const IncrementalDraftAbi& a,unsigned fill){
  std::vector<B> out={B(a.candidates*4,0),B(a.candidates*4096,0)};
  for(unsigned i=0;i<12;++i)out.push_back(B(a.bank_bytes(),fill+i));return out;
}
int main(){
  auto a=Abi();a.Validate();auto broken=a;broken.inputs[1].name="anchor";Bad([&]{broken.Validate();});
  broken=a;std::swap(broken.outputs[2],broken.outputs[3]);Bad([&]{broken.Validate();});
  IncrementalDraftCache cache(a);B features(33*32768,0);
  auto in=cache.Prepare(features,33,9);assert(in[1][0]==0&&in[2][0]==16&&in[4][0]==0);
  for(unsigned i=5;i<17;++i)assert(in[i]==B(a.bank_bytes(),0));
  assert(cache.Complete(Output(a,10)).empty());assert(cache.cursor()==16);
  in=cache.Prepare(features,33,9);assert(in[1][0]==16&&in[2][0]==16&&in[4][0]==15);
  for(unsigned i=0;i<12;++i)assert(in[5+i]==B(a.bank_bytes(),10+i));
  auto wrong=Output(a,20);wrong.back().pop_back();Bad([&]{cache.Complete(wrong);});assert(cache.cursor()==16&&cache.calls()==1);
  auto result=cache.Complete(Output(a,20));assert(result.size()==15&&cache.cursor()==32&&cache.calls()==2);
  // A zero-delta proposal is allowed at capacity; every prior bank must survive.
  in=cache.Prepare(features,33,9);assert(in[2][0]==0&&in[4][0]==15);
  for(unsigned i=0;i<12;++i)assert(in[5+i]==B(a.bank_bytes(),20+i));
  assert(in[0]==B(16*32768,0));cache.Complete(Output(a,20));assert(cache.cursor()==32);
  Bad([&]{cache.Prepare(B(34*32768),34,9);});Bad([&]{cache.Prepare(B(2*32768),2,9);});
  // Cache transport itself crosses the old S64 limit; target request controls
  // remain bounded separately by the board smoke admission policy.
  auto long_abi=Abi(7);long_abi.capacity=192;
  for(auto& spec:long_abi.inputs)if(spec.name.rfind("past_",0)==0)spec.shape[2]=192;
  for(auto& spec:long_abi.outputs)if(spec.name.rfind("present_",0)==0)spec.shape[2]=192;
  IncrementalDraftCache long_cache(long_abi);B history(96*32768,0);
  unsigned calls=0;do{long_cache.Prepare(history,96,12);result=long_cache.Complete(Output(long_abi,30));++calls;}while(result.empty());
  assert(calls==6&&long_cache.cursor()==95&&long_cache.catchup_calls()==5);
  for(unsigned k:{1,3,7,15}) {
    auto gear=Abi(k);IncrementalDraftCache empty(gear);
    auto input=empty.Prepare(B(32768),1,7);assert(input[2][0]==0&&input[4][0]==k);
    auto output=Output(gear,0);output[0][0]=255;output[0][1]=255;output[0][2]=255;
    Bad([&]{empty.Complete(output);});assert(empty.cursor()==0&&empty.calls()==0);
  }
}
