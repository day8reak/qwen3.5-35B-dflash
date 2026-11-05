// Explicit standalone numerical diagnostic: one V1 Load, two KV sequences.
#include "qwen35_dflash/incremental_draft.hpp"
#include "qwen35_dflash/sha256.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <limits>

namespace fs=std::filesystem;
using namespace qwen35::dflash;
using Bytes=std::vector<std::uint8_t>;
struct Step { unsigned index,start,valid,anchor,proposals; };
struct Plan { IncrementalDraftAbi abi; std::vector<Step> steps; bool block_embeddings=false; };

unsigned Number(const std::string& value) {
  if(value.empty()||value.size()>10||value.find_first_not_of("0123456789")!=std::string::npos)
    throw std::invalid_argument("invalid plan integer");
  const auto n=std::stoull(value);
  if(n>std::numeric_limits<unsigned>::max())throw std::invalid_argument("plan integer overflow");
  return static_cast<unsigned>(n);
}
Plan ReadPlan(const fs::path& path) {
  if(!fs::is_regular_file(path)||fs::file_size(path)>65536)
    throw std::invalid_argument("missing/oversized accuracy plan");
  std::ifstream file(path);std::string line;
  std::getline(file,line);
  Plan p;bool geometry=false;
  if(line=="QWEN35_INCREMENTAL_ACCURACY 2")p.block_embeddings=true;
  else if(line!="QWEN35_INCREMENTAL_ACCURACY 1")throw std::invalid_argument("invalid accuracy header");
  while(std::getline(file,line)) {
    std::istringstream input(line);std::vector<std::string> f;std::string v;
    while(input>>v)f.push_back(v);
    if(f.empty())throw std::invalid_argument("empty plan row");
    if(f[0]=="geometry"&&f.size()==4&&!geometry&&p.abi.inputs.empty()) {
      p.abi.context_rows=Number(f[1]);p.abi.capacity=Number(f[2]);p.abi.candidates=Number(f[3]);geometry=true;
    } else if((f[0]=="input"||f[0]=="output")&&f.size()==5&&geometry&&p.steps.empty()) {
      auto& rows=f[0]=="input"?p.abi.inputs:p.abi.outputs;
      if(Number(f[1])!=rows.size())throw std::invalid_argument("noncontiguous tensor slots");
      HiaiTensorSpec s{f[2],f[3],{}};std::istringstream dimensions(f[4]);
      while(std::getline(dimensions,v,','))s.shape.push_back(Number(v));
      rows.push_back(std::move(s));
    } else if(f[0]=="case"&&f.size()==6&&geometry) {
      p.abi.Validate(p.block_embeddings);
      Step s{Number(f[1]),Number(f[2]),Number(f[3]),Number(f[4]),Number(f[5])};
      unsigned cursor=p.steps.empty()?0:p.steps.back().start+p.steps.back().valid;
      if(s.index!=p.steps.size()||s.start!=cursor||s.valid>p.abi.context_rows||
          s.start>p.abi.capacity||s.valid>p.abi.capacity-s.start||s.anchor>=248320||
          s.proposals>p.abi.candidates||p.steps.size()>=64)
        throw std::invalid_argument("invalid accuracy controls");
      p.steps.push_back(s);
    } else throw std::invalid_argument("unknown accuracy plan row");
  }
  p.abi.Validate(p.block_embeddings);
  if(p.steps.empty()||p.abi.capacity>512)throw std::invalid_argument("accuracy needs cases and capacity<=512");
  return p;
}
Bytes Read(const fs::path& path,std::size_t bytes) {
  if(fs::is_symlink(path)||!fs::is_regular_file(path)||fs::file_size(path)!=bytes)
    throw std::runtime_error("input extent differs: "+path.string());
  Bytes value(bytes);std::ifstream in(path,std::ios::binary);
  if(!in.read(reinterpret_cast<char*>(value.data()),bytes))throw std::runtime_error("input read failed");
  return value;
}
std::uint32_t Scalar(const Bytes& b) {
  if(b.size()!=4)throw std::invalid_argument("invalid scalar extent");
  return std::uint32_t(b[0])|(std::uint32_t(b[1])<<8)|(std::uint32_t(b[2])<<16)|(std::uint32_t(b[3])<<24);
}

// This tail never commits outputs to inputs. Each rewrite/reuse pair consumes
// the same retained case, after the original two validation sequences finish.
void ReproduceKv(HiaiStatefulExecutor& executor,const Plan& p,const fs::path& fixture,
                 const fs::path& out,unsigned index,const char* evidence) {
  std::ostringstream calls;std::string stage="load-input",checks="[]";
  unsigned completed=0;bool failed=false;
  auto check=[&](const std::vector<Bytes>& inputs) {
    std::ostringstream rows;rows<<'[';bool matches=true;
    for(std::size_t i=0;i<inputs.size();++i) {
      const auto got=executor.CopyInput(i);const bool same=got==inputs[i];matches&=same;
      if(i)rows<<',';
      rows<<"{\"index\":"<<i<<",\"bytes\":"<<got.size()<<",\"matches\":"<<(same?"true":"false")
          <<",\"sha256\":\""<<Sha256(std::string_view(reinterpret_cast<const char*>(got.data()),got.size()))<<"\"}";
    }
    rows<<']';checks=rows.str();
    if(!matches)throw std::runtime_error("KV reproduction CPU input readback differs from retained bytes");
    return checks;
  };
  try {
    std::vector<Bytes> inputs;
    for(std::size_t i=0;i<p.abi.inputs.size();++i)
      inputs.push_back(Read(fixture/"inputs"/std::to_string(index)/(std::to_string(i)+".bin"),HiaiTensorBytes(p.abi.inputs[i])));
    for(unsigned call=0;call<20;++call) {
      const bool rewrite=call%2==0;stage=rewrite?"rewrite":"reuse";checks="[]";
      for(std::size_t i=0;i<inputs.size();++i) {
        if(rewrite)executor.SetInput(i,inputs[i]);else executor.ReuseInput(i);
      }
      stage="input-before";const auto before=check(inputs);
      stage="execute";executor.Execute();
      stage="input-after";const auto after=check(inputs);
      stage="snapshot";
      std::ofstream snapshot(out/"kv-repeat-pending.bin",std::ios::binary);
      for(std::size_t i=0;i<p.abi.outputs.size();++i) {
        const auto bytes=executor.CopyOutput(i);
        snapshot.write(reinterpret_cast<const char*>(bytes.data()),bytes.size());
      }
      snapshot.close();if(!snapshot)throw std::runtime_error("KV reproduction snapshot write failed");
      fs::rename(out/"kv-repeat-pending.bin",out/("kv-repeat-"+std::to_string(call)+".bin"));
      if(completed)calls<<',';
      calls<<"{\"call\":"<<call<<",\"strategy\":\""<<(rewrite?"rewrite":"reuse")
           <<"\",\"api_us\":"<<executor.last_api_us()<<",\"input_before\":"<<before
           <<",\"input_after\":"<<after<<'}';
      ++completed;
      std::cout<<"draft-kv-reproduction: case="<<index<<" call="<<call<<" strategy="<<(rewrite?"rewrite":"reuse")<<std::endl;
    }
  } catch(const std::exception& error) {
    failed=true;std::cerr<<"KV reproduction tail failed at "<<stage<<": "<<error.what()<<'\n';
    fs::remove(out/"kv-repeat-pending.bin");
  }
  std::ofstream report(out/"kv-reproduction.json");
  report<<"{\"version\":1,\"evidence_kind\":\""<<evidence<<"\",\"case\":"<<index
        <<",\"original_calls\":"<<2*p.steps.size()<<",\"planned_calls\":20,\"commit_outputs\":false"
        <<",\"status\":\""<<(failed?"FAILED":"EXECUTED")<<"\",\"calls\":["<<calls.str()<<']';
  if(failed)report<<",\"failure\":{\"call\":"<<completed<<",\"stage\":\""<<stage<<"\",\"input_checks\":"<<checks<<'}';
  report<<"}\n";report.close();if(!report)throw std::runtime_error("KV reproduction report write failed");
}
int main(int argc,char** argv) {
  try {
    if(argc==2&&std::string(argv[1])=="--help") {
      std::cout<<"qwen35_draft_accuracy MODEL RUNTIME_VERSION FIXTURE_DIRECTORY OUTPUT_DIRECTORY [zero|untouched [KV_REPRODUCTION_CASE]]\n";return 0;
    }
    if(argc<5||argc>7)throw std::invalid_argument("expected model, runtime, fixture, output, optional initialization and reproduction case");
    const std::string output_initialization=argc>=6?argv[5]:"zero";
    if(output_initialization!="zero"&&output_initialization!="untouched")
      throw std::invalid_argument("output initialization must be zero or untouched");
    const fs::path fixture=argv[3],out=argv[4];
    const auto p=ReadPlan(fixture/"plan.txt");
    const unsigned reproduction=argc==7?Number(argv[6]):0;
    if(argc==7&&(output_initialization!="untouched"||reproduction>=p.steps.size()||!p.steps[reproduction].valid))
      throw std::invalid_argument("KV reproduction requires untouched outputs and an appending case");
    if(fs::exists(out))throw std::invalid_argument("accuracy output already exists");
    fs::create_directories(out);
    HiaiStatefulExecutor executor(argv[1],argv[2],p.abi.inputs,p.abi.outputs,
        300000,false,-1,true,"legacy-buffer","qwen35_draft_accuracy",nullptr,{},false,true,
        output_initialization=="zero");
    std::ofstream result(out/"result.json");
#ifdef QWEN35_FAKE_HIAI
    const char* evidence="host-fake-hiai";
#else
    const char* evidence="device-hiai";
#endif
    result<<"{\"evidence_kind\":\""<<evidence<<"\",\"output_initialization\":\""
          <<output_initialization<<"\",\"calls\":[";
    bool first=true;
    for(const std::string mode:{"isolated","feedback"})for(const auto& s:p.steps) {
      std::map<std::string,unsigned> control{{"start_position",s.start},{"valid_rows",s.valid},
          {"anchor",s.anchor},{"proposal_count",s.proposals}};
      std::vector<std::size_t> slots;
      for(unsigned layer=0;layer<6;++layer)for(const char* kind:{"k_","v_"}) {
        const std::string name="past_"+std::string(kind)+std::to_string(layer);
        auto pos=std::find_if(p.abi.inputs.begin(),p.abi.inputs.end(),[&](const auto& r){return r.name==name;});
        slots.push_back(std::distance(p.abi.inputs.begin(),pos));
      }
      for(std::size_t i=0;i<p.abi.inputs.size();++i) {
        const auto& spec=p.abi.inputs[i];
        if(mode=="feedback"&&s.index>0&&spec.name.rfind("past_",0)==0) {executor.ReuseInput(i);continue;}
        auto bytes=Read(fixture/"inputs"/std::to_string(s.index)/(std::to_string(i)+".bin"),HiaiTensorBytes(spec));
        if(control.count(spec.name)&&Scalar(bytes)!=control.at(spec.name))
          throw std::invalid_argument("input control differs from plan");
        executor.SetInput(i,bytes);
      }
      std::vector<Bytes> before;
      for(auto i:slots)before.push_back(executor.CopyInput(i));
      executor.Execute();
      std::vector<Bytes> values;
      std::ofstream snapshot(out/(mode+"-"+std::to_string(s.index)+".bin"),std::ios::binary);
      for(std::size_t i=0;i<p.abi.outputs.size();++i) {
        values.push_back(executor.CopyOutput(i));
        snapshot.write(reinterpret_cast<const char*>(values.back().data()),values.back().size());
      }
      snapshot.close();if(!snapshot)throw std::runtime_error("accuracy snapshot write failed");
      std::vector<HiaiInputSlice> slices;
      for(std::size_t j=0;j<12;++j)if(s.valid)
        slices.push_back({slots[j],j+2,std::size_t(s.start)*256,std::size_t(s.valid)*256,
                          std::size_t(p.abi.capacity)*256,8});
      const auto copied=executor.CommitInputSlices(slices);
      for(std::size_t j=0;j<12;++j) {
        for(unsigned head=0;head<8;++head) {
          const auto offset=(std::size_t(head)*p.abi.capacity+s.start)*256;
          std::copy_n(values[j+2].begin()+offset,std::size_t(s.valid)*256,before[j].begin()+offset);
        }
        if(executor.CopyInput(slots[j])!=before[j])
          throw std::runtime_error("native delta commit changed bytes outside the selected rows");
      }
      if(!first)result<<',';
      first=false;
      result<<"{\"mode\":\""<<mode<<"\",\"index\":"<<s.index
          <<",\"api_us\":"<<executor.last_api_us()<<",\"commit_bytes\":"<<copied
          <<",\"commit_readback_matches\":true}";
      std::cout<<"draft-accuracy: "<<mode<<" case="<<s.index<<" api_us="<<executor.last_api_us()<<std::endl;
    }
    result<<"],\"status\":\"EXECUTED\",\"numerical_validation\":\"HOST_COMPARISON_REQUIRED\"}\n";
    result.close();if(!result)throw std::runtime_error("accuracy result write failed");
    if(argc==7)ReproduceKv(executor,p,fixture,out,reproduction,evidence);
    executor.Close();
    return 0;
  } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
