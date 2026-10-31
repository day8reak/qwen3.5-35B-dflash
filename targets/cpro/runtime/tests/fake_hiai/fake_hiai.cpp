#include "HiAiModelManagerService.h"
#include <model_manager/model_manager.h>
#include <filesystem>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <chrono>
#include <thread>
namespace {
int tensors=0, clients=0;
std::uint16_t* target_features=nullptr;
void* diagnostic_query_buffer=nullptr;
void* mixed_target_storage=nullptr;
std::string Failure() { const auto* p=std::getenv("FAKE_HIAI_FAILURE"); return p?p:""; }
bool AllowDualModelSource() {
  // Synthetic scheduling fixtures only. The selected device has confirmed
  // mutually exclusive setters; this override is not SDK compatibility proof.
  const auto* value=std::getenv("FAKE_HIAI_ALLOW_DUAL_MODEL_SOURCE");
  return value && std::string(value)=="1" && Failure()!="descriptor-exclusive";
}
std::uint32_t FeatureWidth() {
  return Failure()=="large-features"?1025:Failure()=="capture-limit"?1024:32;
}
std::string Layout() { const auto* p=std::getenv("FAKE_HIAI_LAYOUT"); return p?p:""; }
std::vector<std::pair<std::string, std::vector<std::uint32_t>>> DirectOutputs() {
  std::vector<std::pair<std::string, std::vector<std::uint32_t>>> result;
  const auto* value = std::getenv("FAKE_HIAI_DRAFT_DIRECT_OUTPUTS");
  if (!value) return result;
  std::stringstream stream(value);
  for (std::string entry; std::getline(stream, entry, ';');) {
    const auto colon = entry.find(':');
    if (colon == std::string::npos) throw std::runtime_error("invalid direct test dimensions");
    std::stringstream dimensions(entry.substr(colon + 1));
    std::vector<std::uint32_t> shape;
    for (std::string dimension; std::getline(dimensions, dimension, ',');)
      shape.push_back(static_cast<std::uint32_t>(std::stoul(dimension)));
    result.push_back({entry.substr(0, colon), shape});
  }
  return result;
}
void Trace(const char* event) {
  if (const auto* path=std::getenv("FAKE_HIAI_TRACE")) std::ofstream(path, std::ios::app) << event << '\n';
}
bool ModelBytesValid(const void* buffer, std::uint32_t bytes, const std::string& name) {
  if(name=="qwen35_resident_target"||name=="qwen35_resident_prefill"||name=="qwen35_resident_draft") {
    const std::vector<std::string> markers=name=="qwen35_resident_target"?std::vector<std::string>{"original-target-fixture","original-resident-target-fixture"}:
      name=="qwen35_resident_prefill"?std::vector<std::string>{"original-resident-prefill64-fixture"}:
      std::vector<std::string>{"draft-resident64-fixture","draft-resident-loop64-fixture"};
    for(const auto& m:markers)if(buffer&&bytes==m.size()&&!std::memcmp(buffer,m.data(),bytes))return true;
    return false;
  }
  if (name=="qwen35_standalone_draft") {
    const char* expected="draft-standalone-fixture";
    return buffer && bytes==std::strlen(expected) && std::memcmp(buffer,expected,bytes)==0;
  }
  if (name=="qwen35_fused_operator") {
    for (const char* expected : {"fused-gdn-s8-fixture", "fused-gdn-s1-fixture",
                                "fused-conv-s8-fixture", "fused-conv-s1-fixture", "fused-rms-s8-fixture",
                                "fused-conv4-s8-fixture", "fused-conv4-s1-fixture",
                                "fused-cache-s8-fixture", "fused-cache-s1-fixture",
                                "fused-attention-s8-fixture", "fused-attention-s1-fixture"})
      if (buffer && bytes==std::strlen(expected) && std::memcmp(buffer,expected,bytes)==0) return true;
    return false;
  }
  if (name=="qwen35_stateful_target") {
    for (const char* expected : {"stateful-prefill-fixture", "stateful-decode-fixture"})
      if (buffer && bytes==std::strlen(expected) && std::memcmp(buffer,expected,bytes)==0) return true;
    return false;
  }
  const char* expected=name=="qwen35_target"?"target-fixture":
                       name=="qwen35_draft"?"draft-fixture":nullptr;
  return expected && buffer && bytes==std::strlen(expected) &&
         std::memcmp(buffer, expected, bytes)==0;
}
void RequireModelBytes(const void* buffer, std::uint32_t bytes, const std::string& name) {
  // Keep only the original pointer: late reads expose premature unmap/free,
  // including release after Load, before UnLoadModel or before client teardown.
  if (buffer && !ModelBytesValid(buffer, bytes, name)) {
    std::cerr << "invalid or prematurely released model buffer\n";
    std::abort();
  }
}
struct CheckLeaks {
  ~CheckLeaks() { if (tensors || clients) { std::cerr << "LEAK\n"; std::abort(); } }
} check_leaks;
}
namespace hiai {
namespace {std::vector<NDTensorDesc> OriginalDescriptors(bool input,bool resident_loop);}
AiTensor::AiTensor() : ordinal_(++tensors) {}
AiTensor::~AiTensor() { --tensors; }
AIStatus AiTensor::Init(const TensorDimension* dim, HIAI_DataType type) {
  if(std::getenv("FAKE_HIAI_V1_RESIDENT_TRACE"))Trace("v1-create-tensor");
  if (const auto* path=std::getenv("FAKE_HIAI_INIT_TRACE")) {
    std::ofstream trace(path, std::ios::app);
    trace << "format=" << static_cast<int>(dim->GetFormat()) << ";shape=";
    for (auto d:dim->GetDims()) trace << d << ',';
    trace << '\n';
  }
  if(Layout()=="padded" && dim->GetDims().size()!=4) return AI_FAILED;
  if (Failure()=="allocation" && tensors==5) return AI_FAILED;
  if (const auto* path=std::getenv("FAKE_HIAI_DTYPE_TRACE"))
    std::ofstream(path,std::ios::app) << static_cast<int>(type) << '\n';
  if (type!=HIAI_DATATYPE_INT8 && type!=HIAI_DATATYPE_INT32 && type!=HIAI_DATATYPE_INT16 && type!=HIAI_DATATYPE_INT64 &&
      type!=HIAI_DATATYPE_FLOAT16 && type!=HIAI_DATATYPE_FLOAT32) return AI_FAILED;
  if (dim->GetDims().size()==3 && type!=HIAI_DATATYPE_FLOAT16 && type!=HIAI_DATATYPE_FLOAT32) return AI_FAILED;
  std::size_t n=1; for(auto d:dim->GetDims()) n*=d;
  bytes_=n*(type==HIAI_DATATYPE_INT8?1:type==HIAI_DATATYPE_INT64?8:type==HIAI_DATATYPE_FLOAT16 || type==HIAI_DATATYPE_INT16?2:4);
  data_.resize((bytes_+3)/4); return AI_SUCCESS;
}
void* AiTensor::GetBuffer() const {
  auto* data = const_cast<std::int32_t*>(data_.data());
  if (ordinal_==1) diagnostic_query_buffer=data;
  if (Failure()=="v1-cross-alias" && ordinal_==172) return diagnostic_query_buffer;
  if (Failure()=="mixed-cross-alias" && ordinal_==172) return mixed_target_storage;
  if ((Failure()=="gdn-input-alias" && ordinal_==2) ||
      (Failure()=="gdn-output-alias" && ordinal_==8)) return diagnostic_query_buffer;
  // Init validation and zeroing read this pointer four times; SetInput reads
  // it twice for memcpy. Corrupt only the subsequent diagnostic readback.
  if (Failure()=="gdn-readback-corrupt" && ++buffer_reads_==7 && bytes_)
    reinterpret_cast<std::uint8_t*>(data)[0] ^= 1;
  return data;
}
void AiTensor::ReplaceBufferForTest() {
  auto replacement=data_;
  data_.swap(replacement);
}
std::uint32_t AiTensor::GetSize() const {
  return Failure()=="short-buffer"?1:Failure()=="oversized-buffer"?65537:bytes_;
}
AiModelDescription::AiModelDescription(const std::string& n,int freq,int framework,int model,int device):name(n) {
  if(freq!=3 || framework!=0 || model!=1 || device!=0) throw std::runtime_error("wrong HIAI model mode");
}
AIStatus AiModelDescription::SetModelPath(const std::string& p) {
  path=p;
  if (!AllowDualModelSource()) { model_buffer_=nullptr; model_bytes_=0; }
  if(Failure()=="file-path-loss")path.clear();
  if(Failure()=="file-buffer-present") {static const char marker=0;model_buffer_=&marker;model_bytes_=1;}
  return Failure()=="path"?1:0;
}
AIStatus AiModelDescription::SetModelBuffer(const void* data, std::uint32_t bytes) {
  if(Failure()=="model-buffer") return AI_FAILED;
  if(name=="qwen35_model_description_probe") {
    if(!data || !bytes) return AI_FAILED;
  } else if(!ModelBytesValid(data, bytes, name)) return AI_FAILED;
  if(!AllowDualModelSource()) path.clear();
  model_buffer_=data; model_bytes_=bytes; return AI_SUCCESS;
}
void* AiModelDescription::GetModelBuffer() const {
  return Failure()=="model-buffer-null"?nullptr:const_cast<void*>(model_buffer_);
}
std::uint32_t AiModelDescription::GetModelNetSize() const {
  return Failure()=="model-buffer-size"?model_bytes_+1:model_bytes_;
}
void AiContext::AddPara(const std::string& key,const std::string& value) {
  if(key!="model_name") throw std::runtime_error("context missing model_name"); model=value;
}
AiModelMngerClient::AiModelMngerClient(){ ++clients; }
AiModelMngerClient::~AiModelMngerClient(){
  RequireModelBytes(model_buffer_, model_bytes_, name_);
  if(loaded_) { std::cerr<<"LEAK loaded model\n"; std::abort(); } --clients;
}
AIStatus AiModelMngerClient::Init(std::shared_ptr<AiModelManagerClientListener> listener) {
  if(listener) throw std::runtime_error("must initialize synchronously"); return Failure()=="init"?1:0;
}
char* AiModelMngerClient::GetVersion(){ static char s[]="FAKE-HIAI-X90"; return Failure()=="version"?nullptr:s; }
bool AiModelMngerClient::CheckNPUSupport(){ return Failure()!="no-npu"; }
AIStatus AiModelMngerClient::CheckModelCompatibility(AiModelDescription&,bool&){
  Trace("compatibility-query");
  throw std::runtime_error("standalone compatibility query unavailable for path-based OMC fixture");
}
AIStatus AiModelMngerClient::Load(std::vector<std::shared_ptr<AiModelDescription>>& d){
  Trace("load");
  if(load_called_)throw std::runtime_error("same V1 client loaded more than once");
  load_called_=true;
  if(d.size()!=1)return AI_FAILED;
  const bool external=d[0]->name=="qwen35_resident_target"||d[0]->name=="qwen35_resident_prefill";
  const bool file_only=!d[0]->path.empty()&&!d[0]->GetModelBuffer()&&d[0]->GetModelNetSize()==0;
  if(file_only) {
    // Explicit unvalidated FILE_ONLY fixture branch, not proof that the
    // selected device resolves external OMC sidecars. Only known tiny markers
    // are read, and the fake SDK owns their bytes through Process/Unload.
    if(!external||!std::filesystem::is_regular_file(d[0]->path)||std::filesystem::file_size(d[0]->path)>4096)return AI_FAILED;
    std::ifstream input(d[0]->path,std::ios::binary);
    file_model_bytes_.assign(std::istreambuf_iterator<char>(input),std::istreambuf_iterator<char>());
    if(!input.eof()&&input.fail())return AI_FAILED;
    if(!ModelBytesValid(file_model_bytes_.data(),static_cast<std::uint32_t>(file_model_bytes_.size()),d[0]->name))return AI_FAILED;
    model_buffer_=file_model_bytes_.data();model_bytes_=static_cast<std::uint32_t>(file_model_bytes_.size());
  } else {
    if(external||!d[0]->path.empty()||!ModelBytesValid(d[0]->GetModelBuffer(),d[0]->GetModelNetSize(),d[0]->name))return AI_FAILED;
    model_buffer_=d[0]->GetModelBuffer();model_bytes_=d[0]->GetModelNetSize();
  }
  name_=d[0]->name;
  if(name_=="qwen35_resident_draft"&&std::getenv("FAKE_HIAI_DRAFT_LOAD_DELAY"))
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  if(name_.rfind("qwen35_resident_",0)==0)Trace(("v1-load:"+name_).c_str());
  if(name_.rfind("qwen35_resident_",0)==0)Trace(("v1-model-source:"+name_+(file_only?":file-only":":buffer-only")).c_str());
  loaded_=true; // partial target or draft load failure must still unload.
  return Failure()=="load" || Failure()=="incompatible" ||
         (Failure()=="draft-load" && name_=="qwen35_draft") ||
         (Failure()=="v1-target-load"&&name_=="qwen35_resident_target") ||
         (Failure()=="v1-draft-load"&&name_=="qwen35_resident_draft")?1:0;
}
AIStatus AiModelMngerClient::UnLoadModel(){
  RequireModelBytes(model_buffer_, model_bytes_, name_);
  Trace("unload"); loaded_=false;
  if(name_.rfind("qwen35_resident_",0)==0) {
    Trace(("v1-unload:"+name_).c_str());
    if(Failure()=="v1-unload-throw")throw std::runtime_error("injected V1 UnLoadModel exception");
    if(Failure()=="v1-unload-unknown")throw 19;
    if(Failure()=="v1-unload"||(Failure()=="v1-draft-unload"&&name_=="qwen35_resident_draft"))return 81;
  }
  return 0;
}
AIStatus AiModelMngerClient::GetModelIOTensorDim(const std::string& n,std::vector<TensorDimension>& in,std::vector<TensorDimension>& out){
  if(n!=name_ || !loaded_ || Failure()=="io-dims") return 1;
  TensorDimension ids({1,32}), features({1,32,FeatureWidth()});
  if(n=="qwen35_resident_target"||n=="qwen35_resident_prefill") {
    const std::string marker(static_cast<const char*>(model_buffer_),model_bytes_);
    const bool loop=marker!="original-target-fixture",prefill=n=="qwen35_resident_prefill";
    for(bool input:{true,false}) {
      auto rows=OriginalDescriptors(input,loop);
      if(prefill) {
        if(input) {rows[0].dims={1,1,64,2048};rows[1].dims={1,1,64,2048};rows[2].dims={1,64};rows[83].dims={64};rows[85].dims={1,64,1};}
        else rows[81].dims={1,1,64,16384};
      }
      for(const auto& row:rows){std::vector<std::uint32_t> dims;for(auto d:row.dims)dims.push_back(d);(input?in:out).emplace_back(dims,row.format);}
    }
  }
  else if(n=="qwen35_resident_draft") {
    in={TensorDimension({1,64,16384}),TensorDimension({1,64}),TensorDimension({1,64})};
    out={TensorDimension({1,15}),TensorDimension({1,15,2048})};
    if(const auto* gear=std::getenv("FAKE_HIAI_INCREMENTAL_K")) {
      const auto k=static_cast<std::uint32_t>(std::stoul(gear));
      in={TensorDimension({1,16,16384}),TensorDimension({1}),TensorDimension({1}),TensorDimension({1}),TensorDimension({1})};
      out={TensorDimension({1,k}),TensorDimension({1,k,2048})};
      for(unsigned i=0;i<12;++i){in.emplace_back(std::vector<std::uint32_t>{1,8,192,128});out.emplace_back(std::vector<std::uint32_t>{1,8,192,128});}
      if(std::getenv("FAKE_HIAI_INCREMENTAL_REVERSE"))std::reverse(in.begin(),in.end());
    }
  }
  else if(n=="qwen35_standalone_draft") {
    in={TensorDimension({1,8,16384}),TensorDimension({1,8}),TensorDimension({1,8})};
    out={TensorDimension({1,15}),TensorDimension({1,15,2048})};
    if (const auto* elements = std::getenv("FAKE_HIAI_DRAFT_TRACE_ELEMENTS"))
      out[1] = TensorDimension({1,1,static_cast<std::uint32_t>(std::stoul(elements))});
    for (const auto& spec : DirectOutputs()) out.emplace_back(spec.second);
  }
  else if(n=="qwen35_stateful_target") {
    const bool decode = std::string(static_cast<const char*>(model_buffer_),model_bytes_)=="stateful-decode-fixture";
    const std::uint32_t s=decode?1:3;
    in={TensorDimension({1,s})};
    if(decode) {
      in.emplace_back(std::vector<std::uint32_t>{1,8,3});
      in.emplace_back(std::vector<std::uint32_t>{1,2,2,2});
      in.emplace_back(std::vector<std::uint32_t>{1,1,3,2});
      in.emplace_back(std::vector<std::uint32_t>{1,1,3,2});
    }
    out={TensorDimension({1,s}),TensorDimension({1,s,4}),TensorDimension({1,8,3}),
         TensorDimension({1,2,2,2}),TensorDimension({1,1,decode?4U:3U,2}),
         TensorDimension({1,1,decode?4U:3U,2})};
  }
  else if(n=="qwen35_fused_operator") {
    const std::string marker(static_cast<const char*>(model_buffer_),model_bytes_);
    const std::uint32_t s=marker.find("-s1-")!=std::string::npos?1:8;
    if(marker.find("fused-gdn-")==0) {
      in={TensorDimension({1,s,32,128}),TensorDimension({1,s,32,128}),TensorDimension({1,s,32,128}),
          TensorDimension({1,s,32}),TensorDimension({1,s,32}),TensorDimension({1,32,128,128}),
          TensorDimension({1})};
      out={TensorDimension({1,s,32,128}),TensorDimension({1,32,128,128})};
    } else if(marker.find("fused-conv-")==0 || marker.find("fused-conv4-")==0) {
      const std::uint32_t width=marker.find("fused-conv4-")==0?4:3;
      in={TensorDimension({1,s,8192}),TensorDimension({1,8192,width}),TensorDimension({128})};
      out={TensorDimension({1,8192,4}),TensorDimension({1,8192,s+3})};
    } else if(marker.find("fused-cache-")==0) {
      in={TensorDimension({64,32,32,16}),TensorDimension({s,32,16}),TensorDimension({1}),TensorDimension(std::vector<std::uint32_t>{})};
      out={TensorDimension({64,32,32,16})};
    } else if(marker.find("fused-attention-")==0) {
      in={TensorDimension({1,256,s,16}),TensorDimension({64,32,32,16}),TensorDimension({64,32,32,16}),
          TensorDimension({1,1,s,2048}),TensorDimension({1}),TensorDimension({1}),
          TensorDimension({1,64}),TensorDimension({1})};
      out={TensorDimension({1,256,s,16})};
    } else {
      in={TensorDimension({1,8,2048}),TensorDimension({2048})};
      out={TensorDimension({1,8,2048}),TensorDimension({1,8,1})};
    }
  }
  else if(n=="qwen35_target") { in={ids,ids}; out={ids,features}; }
  else { in={features,ids,ids}; out={TensorDimension({1,3})}; }
  if(Layout()=="padded" || Layout()=="padded-mixed") {
    for(auto* dims : {&in, &out}) {
      for(auto& dim : *dims) {
        auto shape=dim.GetDims();
        if(Layout()=="padded-mixed" && shape.size()==3) continue;
        shape.resize(4,1); dim=TensorDimension(shape);
      }
    }
  }
  if(Layout()=="leading-feature" || Layout()=="mixed-feature") {
    for(auto* dims : {&in, &out}) {
      for(auto& dim : *dims) {
        auto shape=dim.GetDims();
        if(shape.size()==3 && (Layout()=="leading-feature" || n=="qwen35_target"))
          shape.insert(shape.begin(),1);
        else if(Layout()=="mixed-feature") shape.resize(4,1);
        dim=TensorDimension(shape);
      }
    }
  }
  if(n=="qwen35_fused_operator" && in.size()==4 && Layout()=="scalar-nonunit")
    in[3]=TensorDimension({1,1,1,2});
  if(n=="qwen35_target") {
    if(Layout()=="leading-one") in[0]=TensorDimension({1,1,32,1});
    if(Layout()=="middle-one") out[1]=TensorDimension({1,32,1,32});
    if(Layout()=="rank-five") in[1]=TensorDimension({1,32,1,1,1});
    if(Layout()=="rank-three-token") in[1]=TensorDimension({1,32,1});
    if(Layout()=="non-one-tail") in[1]=TensorDimension({1,32,1,2});
    if(Layout()=="zero-tail") in[1]=TensorDimension({1,32,1,0});
    if(Layout()=="non-nd") in[1]=TensorDimension({1,32,1,1},Format::NCHW);
    if(Layout()=="non-nd-leading-feature") out[1]=TensorDimension({1,1,32,32},Format::NCHW);
    if(Layout()=="permuted") in[1]=TensorDimension({32,1,1,1});
  } else {
    if(Layout()=="draft-width-tail") out[0]=TensorDimension({1,3,2,1});
    if(Layout()=="draft-feature-width") in[0]=TensorDimension({1,32,16,2});
  }
  if(Failure()=="shape") in[0]=TensorDimension({2,3});
  if(Failure()=="format") in[0]=TensorDimension({1,32},Format::NCHW);
  if(Failure()=="count") out.clear();
  return 0;
}
AIStatus AiModelMngerClient::Process(AiContext& c,std::vector<std::shared_ptr<AiTensor>>& in,std::vector<std::shared_ptr<AiTensor>>& out,std::uint32_t timeout,std::int32_t& stamp){
  Trace("process");
  RequireModelBytes(model_buffer_, model_bytes_, name_);
  if(c.model!=name_ || !timeout || !loaded_ || Failure()=="run") return 1;
  if(name_.rfind("qwen35_resident_",0)==0) {
    Trace(("v1-process:"+name_).c_str());
    if(Failure()=="v1-process-unknown")throw 17;
    if(Failure()=="v1-process"||(Failure()=="v1-draft-process"&&name_=="qwen35_resident_draft"))return 82;
    const std::string marker(static_cast<const char*>(model_buffer_),model_bytes_);
    if(name_=="qwen35_resident_draft"&&std::getenv("FAKE_HIAI_INCREMENTAL_K")) {
      // Byte-transport/cache sentinel fixture, NOT the Draft attention math.
      if(in.size()!=17||out.size()!=14)return 101;
      auto tensor=[&](unsigned i){return in[std::getenv("FAKE_HIAI_INCREMENTAL_REVERSE")?16-i:i];};
      auto scalar=[&](unsigned i){return *static_cast<const std::uint32_t*>(tensor(i)->GetBuffer());};
      const auto start=scalar(1),valid=scalar(2),anchor=scalar(3),count=scalar(4);
      const auto k=static_cast<unsigned>(std::stoul(std::getenv("FAKE_HIAI_INCREMENTAL_K")));
      if(start+valid>192||valid>16||(count!=0&&count!=k))return 102;
      const auto* features=static_cast<const std::uint16_t*>(tensor(0)->GetBuffer());
      for(unsigned row=0;row<16;++row)for(unsigned col=0;col<16384;++col)
        if(features[row*16384+col]!=(row<valid?0x3800:0))return 103;
      for(unsigned bank=0;bank<12;++bank) {
        const auto* old=static_cast<const std::uint16_t*>(tensor(5+bank)->GetBuffer());
        auto* next=static_cast<std::uint16_t*>(out[2+bank]->GetBuffer());
        for(unsigned head=0;head<8;++head)for(unsigned row=0;row<192;++row)for(unsigned col=0;col<128;++col) {
          const auto idx=(head*192+row)*128+col;
          const auto sentinel=std::uint16_t(0x3000+bank*64+row);
          if(old[idx]!=(row<start?sentinel:0))return 104;
          next[idx]=row<start+valid?sentinel:0;
        }
      }
      const bool reject=std::getenv("FAKE_HIAI_INCREMENTAL_REJECT")!=nullptr;
      for(unsigned i=0;i<k;++i)static_cast<std::uint32_t*>(out[0]->GetBuffer())[i]=anchor+1+i+(reject?1:0);
      std::memset(out[1]->GetBuffer(),0,out[1]->GetSize());
      Trace(("incremental:"+std::to_string(start)+":"+std::to_string(valid)+":"+std::to_string(count)).c_str());
      if(Failure()=="incremental-invalid-token")static_cast<std::uint32_t*>(out[0]->GetBuffer())[0]=248320;
    } else if(name_=="qwen35_resident_draft") {
      if(in.size()!=3||out.size()!=2)return 83;
      const auto* ids=static_cast<const std::int32_t*>(in[1]->GetBuffer());
      const auto* mask=static_cast<const std::int32_t*>(in[2]->GetBuffer());
      const auto* features=static_cast<const std::uint16_t*>(in[0]->GetBuffer());
      const bool loop=marker=="draft-resident-loop64-fixture";std::size_t valid=0;
      while(valid<64&&mask[valid]==1)++valid;if(!valid)return 84;
      std::vector<std::int32_t> text_prompt;
      if(const auto* value=std::getenv("FAKE_HIAI_TEXT_PROMPT_IDS")) {
        std::istringstream words(value);std::string word;
        while(std::getline(words,word,','))text_prompt.push_back(std::stoi(word));
      }
      for(std::size_t i=0;i<64;++i) {
        const auto expected=loop?(i<valid?static_cast<std::int32_t>(i+1):0):static_cast<std::int32_t>(i);
        const auto text_expected=i<text_prompt.size()?text_prompt[i]:static_cast<std::int32_t>(i-text_prompt.size()+1);
        if(ids[i]!=(!text_prompt.empty()&&loop?(i<valid?text_expected:0):expected)||mask[i]!=(i<valid?1:0))return 85;
        if(!std::all_of(features+i*16384,features+(i+1)*16384,[&](auto x){return x==(loop&&i>=valid?0:0x3800);}))return 86;
      }
      for(int i=0;i<15;++i)static_cast<std::int32_t*>(out[0]->GetBuffer())[i]=loop?ids[valid-1]+1+i:10+static_cast<int>(runs_)+i;
      std::fill_n(static_cast<std::uint16_t*>(out[1]->GetBuffer()),15*2048,0x3800);
    } else {
      if(in.size()!=89||out.size()!=82)return 87;
      const bool prefill=name_=="qwen35_resident_prefill",loop=marker!="original-target-fixture";
      const auto position=prefill?*static_cast<const std::int16_t*>(in[88]->GetBuffer()):*static_cast<const std::int32_t*>(in[2]->GetBuffer());
      if(prefill&&(position<1||position>64))return 88;
      for(std::size_t ref=0;ref<80;++ref) {
        const auto idx=loop?1+ref*37%80:1+ref;
        if(in[ref+3]->GetSize()!=out[idx]->GetSize())return 89;
        std::memcpy(out[idx]->GetBuffer(),in[ref+3]->GetBuffer(),in[ref+3]->GetSize());
        if(loop) {
          const auto* old=static_cast<const std::int32_t*>(in[ref+3]->GetBuffer());
          if(!prefill&&(old[0]!=static_cast<std::int32_t>(ref)||old[1]!=position))return 90;
          auto* next=static_cast<std::int32_t*>(out[idx]->GetBuffer());next[0]=ref;next[1]=prefill?position:position+1;
        } else ++static_cast<std::uint8_t*>(out[idx]->GetBuffer())[0];
      }
      std::memset(out[0]->GetBuffer(),0,out[0]->GetSize());
      auto token=loop?static_cast<const std::uint8_t*>(in[0]->GetBuffer())[(prefill?position-1:0)*2048]+1:static_cast<int>(runs_+1);
      static_cast<float*>(out[0]->GetBuffer())[token]=10;
      std::fill_n(static_cast<std::uint16_t*>(out[81]->GetBuffer()),out[81]->GetSize()/2,0x3800);
    }
    if(Failure()=="v1-replace-output")out[0]->ReplaceBufferForTest();
    ++runs_;stamp=static_cast<std::int32_t>(runs_);return 0;
  }
  if(name_=="qwen35_standalone_draft") {
    // Transport fixture only; never a numerical DFlash implementation.
    const auto direct = DirectOutputs();
    if(in.size()!=3 || out.size()!=2+direct.size()) return AI_FAILED;
    auto* top=static_cast<std::int32_t*>(out[0]->GetBuffer());
    for(std::int32_t i=0;i<15;++i) top[i]=i+10;
    if (std::getenv("FAKE_HIAI_DRAFT_TRACE_ELEMENTS"))
      std::fill_n(static_cast<float*>(out[1]->GetBuffer()),out[1]->GetSize()/sizeof(float),.5f);
    else std::fill_n(static_cast<std::uint16_t*>(out[1]->GetBuffer()),15*2048,0x3800);
    for (std::size_t i=0; i<direct.size(); ++i) {
      auto& tensor = out[i+2];
      if (direct[i].first == "float16")
        std::fill_n(static_cast<std::uint16_t*>(tensor->GetBuffer()),tensor->GetSize()/2,0x3800);
      else if (direct[i].first == "float32")
        std::fill_n(static_cast<float*>(tensor->GetBuffer()),tensor->GetSize()/4,.5f);
      else if (direct[i].first == "int32")
        std::fill_n(static_cast<std::int32_t*>(tensor->GetBuffer()),tensor->GetSize()/4,7);
      else return AI_FAILED;
    }
    stamp=1; return AI_SUCCESS;
  }
  if(name_=="qwen35_fused_operator") {
    // Deliberately a lifecycle/byte transport double, never a fused-op kernel.
    // Distinct finite raw patterns let tests detect truncation and wrong dtypes.
    const std::string marker(static_cast<const char*>(model_buffer_),model_bytes_);
    if(marker.find("fused-gdn-")==0) {
      const std::int16_t s=marker.find("-s1-")!=std::string::npos?1:8;
      if(in.size()!=7 || out.size()!=2) return AI_FAILED;
      const auto effective=*static_cast<const std::int16_t*>(in[6]->GetBuffer());
      if(effective<1 || effective>s) return AI_FAILED;
      if(Failure()=="gdn-no-write") { stamp=1; return AI_SUCCESS; }
      if(Failure()=="gdn-partial-write" || Failure()=="gdn-zero-write") {
        for(auto& tensor:out)
          std::memset(tensor->GetBuffer(),0,Failure()=="gdn-partial-write"?64:tensor->GetSize());
        stamp=1; return AI_SUCCESS;
      }
      if(Failure()=="gdn-replace-output") {
        auto replacement=std::make_shared<AiTensor>();
        TensorDimension dim({1,static_cast<std::uint32_t>(s),32,128});
        if(replacement->Init(&dim,HIAI_DATATYPE_FLOAT16)!=AI_SUCCESS) return AI_FAILED;
        out[0]=replacement;
      }
      if(Failure()=="gdn-replace-buffer") out[0]->ReplaceBufferForTest();
      std::fill_n(static_cast<std::uint16_t*>(out[0]->GetBuffer()),s*32*128,0x3800);
      if(out[1]->GetSize()==32*128*128*2)
        std::fill_n(static_cast<std::uint16_t*>(out[1]->GetBuffer()),32*128*128,0x3400);
      else if(out[1]->GetSize()==32*128*128*4)
        std::fill_n(static_cast<float*>(out[1]->GetBuffer()),32*128*128,0.25F);
      else return AI_FAILED;
    } else if(marker.find("fused-conv-")==0 || marker.find("fused-conv4-")==0) {
      if(in.size()!=3 || out.size()!=2) return AI_FAILED;
      for(auto& tensor:out) std::fill_n(static_cast<std::uint16_t*>(tensor->GetBuffer()),tensor->GetSize()/2,0x3400);
    } else if(marker.find("fused-cache-")==0) {
      if(in.size()!=4 || out.size()!=1 || in[3]->GetSize()!=4) return AI_FAILED;
      std::fill_n(static_cast<std::uint16_t*>(out[0]->GetBuffer()),out[0]->GetSize()/2,0x3400);
    } else if(marker.find("fused-attention-")==0) {
      if(in.size()!=8 || out.size()!=1 || in[4]->GetSize()!=8 || in[5]->GetSize()!=8 || in[7]->GetSize()!=8) return AI_FAILED;
      std::fill_n(static_cast<std::uint16_t*>(out[0]->GetBuffer()),out[0]->GetSize()/2,0x3800);
    } else {
      if(in.size()!=2 || out.size()!=2) return AI_FAILED;
      std::fill_n(static_cast<std::uint16_t*>(out[0]->GetBuffer()),8*2048,0x3c00);
      std::fill_n(static_cast<float*>(out[1]->GetBuffer()),8,0.5F);
    }
    stamp=1;
    return AI_SUCCESS;
  }
  if(name_=="qwen35_stateful_target") {
    const bool decode = std::string(static_cast<const char*>(model_buffer_),model_bytes_)=="stateful-decode-fixture";
    const std::size_t s=decode?1:3;
    if(in.size()!=(decode?5U:1U) || out.size()!=6) return AI_FAILED;
    auto* ids=static_cast<std::int32_t*>(in[0]->GetBuffer());
    auto* top=static_cast<std::int32_t*>(out[0]->GetBuffer());
    for(std::size_t i=0;i<s;++i) top[i]=(ids[i]+1)%64;
    auto* features=static_cast<std::uint16_t*>(out[1]->GetBuffer());
    for(std::size_t i=0;i<s*4;++i) features[i]=static_cast<std::uint16_t>(0x3800+i);
    auto* history=static_cast<std::uint16_t*>(out[2]->GetBuffer());
    auto* recurrent=static_cast<float*>(out[3]->GetBuffer());
    for(std::size_t i=0;i<24;++i) {
      const auto old=decode?static_cast<const std::uint16_t*>(in[1]->GetBuffer())[i]:0;
      if(decode && old!=0x3c00+i) return AI_FAILED;
      history[i]=static_cast<std::uint16_t>(decode?old+1:0x3c00+i);
    }
    for(std::size_t i=0;i<8;++i) {
      const auto old=decode?static_cast<const float*>(in[2]->GetBuffer())[i]:0;
      if(decode && old!=static_cast<float>(i+1)) return AI_FAILED;
      recurrent[i]=decode?old+1:static_cast<float>(i+1);
    }
    for(std::size_t index=4;index<6;++index) {
      auto* next=static_cast<std::uint16_t*>(out[index]->GetBuffer());
      const std::uint16_t base=index==4?0x4000:0x4200;
      for(std::size_t i=0;i<(decode?8U:6U);++i) {
        if(decode && i<6 && static_cast<const std::uint16_t*>(in[index-1]->GetBuffer())[i]!=base+i)
          return AI_FAILED;
        next[i]=static_cast<std::uint16_t>(base+i);
      }
    }
    stamp=1;
    return AI_SUCCESS;
  }
  const bool target=name_=="qwen35_target";
  auto* ids=static_cast<std::int32_t*>(in[target?0:1]->GetBuffer());
  auto* mask=static_cast<std::int32_t*>(in[target?1:2]->GetBuffer());
  auto* top=static_cast<std::int32_t*>(out[0]->GetBuffer());
  int length=0; for(int i=0;i<32;++i) length+=mask[i];
  if(!length) return 1;
  if(target) {
    for(int i=0;i<32;++i) top[i]=(ids[i]+1)%64;
    auto* f=static_cast<std::uint16_t*>(out[1]->GetBuffer()); std::fill_n(f,32*FeatureWidth(),0);
    f[0]=1234;
    f[1]=0x8000; // Negative zero and a NaN payload must survive raw capture.
    f[2]=0x7e01;
    f[32*FeatureWidth()-1]=4321;
    target_features=f;
  } else {
    const auto* f=static_cast<std::uint16_t*>(in[0]->GetBuffer());
    if(f[0]!=1234 || f[32*FeatureWidth()-1]!=4321) return 1;
    if(Failure()=="overwrite-target-on-draft") std::fill_n(target_features,32*FeatureWidth(),0);
    for(int i=0;i<3;++i) top[i]=(ids[length-1]+i+1)%64;
    if(Failure()=="corrupt-draft") top[1]=63;
  }
  if(Failure()=="token") top[0]=64;
  stamp=1; return 0;
}
}

// V2 transport/lifetime double. It does not model proprietary external-weight
// restoration or prove that a device supports this route.
namespace hiai {
namespace {
std::weak_ptr<INDTensorBuffer> original_alias_input;
std::weak_ptr<INDTensorBuffer> original_alias_output;
std::weak_ptr<INDTensorBuffer> resident_alias_storage;
class FakeNDTensor final : public INDTensorBuffer {
 public:
  explicit FakeNDTensor(NDTensorDesc desc) : desc_(std::move(desc)) {
    ++tensors;
    std::size_t size=desc_.dataType==DataType::INT8?1:desc_.dataType==DataType::INT64?8:
        (desc_.dataType==DataType::FLOAT16 || desc_.dataType==DataType::INT16)?2:4;
    for (auto d:desc_.dims) size*=static_cast<std::size_t>(d);
    data_.resize(size);
  }
  ~FakeNDTensor() override { --tensors; }
  void* GetData() override { return data_.data(); }
  std::size_t GetSize() const override { return Failure()=="v2-short-buffer"?1:data_.size(); }
  const NDTensorDesc& GetTensorDesc() const override { return desc_; }
 private:
  NDTensorDesc desc_;
  std::vector<std::uint8_t> data_;
};
class FakeOffsetNDTensor final : public INDTensorBuffer {
 public:
  FakeOffsetNDTensor(NDTensorDesc desc,std::shared_ptr<INDTensorBuffer> owner)
      : desc_(std::move(desc)),owner_(std::move(owner)) { ++tensors; }
  ~FakeOffsetNDTensor() override { --tensors; }
  void* GetData() override { return static_cast<std::uint8_t*>(owner_->GetData())+16; }
  std::size_t GetSize() const override { return owner_->GetSize(); }
  const NDTensorDesc& GetTensorDesc() const override { return desc_; }
 private:
  NDTensorDesc desc_;
  std::shared_ptr<INDTensorBuffer> owner_;
};
NDTensorDesc DraftDesc(std::vector<std::int32_t> dims, DataType type) {
  if (Layout()=="padded") dims.resize(4,1);
  else if (Layout()=="leading-feature" && dims.size()==3) dims.insert(dims.begin(),1);
  return {dims,type,Format::ND};
}
void ModelIoNames(std::vector<NDTensorDesc>& descriptors, const std::string& direction) {
#ifndef QWEN35_FAKE_NO_TENSOR_NAME
  const auto* mode = std::getenv("FAKE_HIAI_TENSOR_NAMES");
  if (!mode || !*mode) return;
  for (std::size_t index = 0; index < descriptors.size(); ++index) {
    if (std::string(mode) == "partial" && index != 1) continue;
    // Escaping and duplicate values are deliberately not interpreted as graph
    // identities. They are independent raw descriptor field observations.
    descriptors[index].tensorName = direction + "\"\\\n\t" + std::string(1, '\0') + "same";
  }
  if (std::string(mode) == "limit") descriptors.front().tensorName = std::string(4096, 'x');
  if (std::string(mode) == "over-limit") descriptors.front().tensorName = std::string(4097, 'x');
#else
  (void)descriptors;
  (void)direction;
#endif
}
NDTensorDesc OriginalStateDesc(std::size_t index) {
  const auto kind=index%8;
  // A declared non-ND format exercises exact descriptor preservation.
  return {kind==0 || kind==2 || kind==4 ? std::vector<std::int32_t>{1,8192,4} :
          kind==1 || kind==3 || kind==5 ? std::vector<std::int32_t>{1,32,128,128} :
                                       std::vector<std::int32_t>{64,32,32,16},
          DataType::FLOAT16,Format::NCHW};
}
std::vector<NDTensorDesc> OriginalDescriptors(bool input,bool resident_loop=false) {
  std::vector<NDTensorDesc> rows;
  if (input) rows={{{1,1,1,2048},DataType::INT8,Format::ND},
                   {{1,1,1,2048},DataType::FLOAT32,Format::ND},
                   {{1,1},DataType::INT32,Format::ND}};
  else rows={{{1,1,1,248320},DataType::FLOAT32,Format::ND}};
  for (std::size_t i=0;i<80;++i) rows.push_back(OriginalStateDesc(i));
  if (input) {
    rows.push_back({{1},DataType::INT32,Format::ND});
    rows.push_back({{1},DataType::INT32,Format::ND});
    rows.push_back({{1,1,1},DataType::FLOAT32,Format::ND});
    rows.push_back({{1},DataType::INT64,Format::ND});
    rows.push_back({{128},DataType::INT32,Format::ND});
    rows.push_back({{1},DataType::INT16,Format::ND});
  } else rows.push_back({{1,1,1,16384},DataType::FLOAT16,Format::ND});
#ifndef QWEN35_FAKE_NO_TENSOR_NAME
  for (std::size_t i=0;i<rows.size();++i)
    rows[i].tensorName=std::string(input?"runtime_input_":"runtime_output_")+std::to_string(i);
  if(resident_loop) {
    for(auto& row:rows)row.format=Format::ND;
    if(input) {
      const std::vector<std::string> controls={"arg1_1","arg113_1","arg107_1"};
      for(std::size_t i=0;i<3;++i)rows[i].tensorName=controls[i];
      for(std::size_t ref=0;ref<80;++ref)rows[ref+3].tensorName="RefData_"+std::to_string(ref);
      const std::vector<std::string> tail={"arg2_1","arg1238_1","arg0_1","arg114_1","arg14_1","arg18_1"};
      for(std::size_t i=0;i<tail.size();++i)rows[83+i].tensorName=tail[i];
    } else {
      const auto ordered=rows;
      for(std::size_t ref=0;ref<80;++ref)rows[1+ref*37%80]=ordered[ref+1];
      for(std::size_t i=0;i<rows.size();++i)rows[i].tensorName="native-output-"+std::to_string(i);
    }
  }
#endif
  if (Failure()=="original-count" && !input) rows.pop_back();
  if (Failure()=="original-dtype" && input) rows[0].dataType=DataType::INT16;
  if (Failure()=="original-shape" && input) rows[0].dims={1,1,2,1024};
  if (Failure()=="original-format" && input) rows[0].format=Format::NCHW;
#ifndef QWEN35_FAKE_NO_TENSOR_NAME
  if (Failure()=="original-name" && input) rows[0].tensorName="wrong_runtime_name";
#endif
  return rows;
}
class FakeBuiltModel final : public IBuiltModel {
 public:
  bool restored=false,original=false,draft64=false,resident_loop=false,draft_loop=false,load_probe=false,prefill=false;
  std::string name="graph_1";
  std::string GetName() const override{return name;}
  void SetName(const std::string& value) override {
    Trace(("v2-set-name:"+value).c_str());
    if(Failure()!="resident-name")name=value;
  }
  ~FakeBuiltModel() override { Trace("v2-built-model-destroy"); }
  Status RestoreFromFile(const char* path) override {
    Trace("v2-restore-file");
    if (const auto* trace=std::getenv("FAKE_HIAI_PATH_TRACE")) std::ofstream(trace) << path;
    std::ifstream file(path,std::ios::binary);
    std::string data((std::istreambuf_iterator<char>(file)),std::istreambuf_iterator<char>());
    prefill=data=="original-resident-prefill64-fixture";
    resident_loop=data=="original-resident-target-fixture" || prefill;
    load_probe=data=="original-target-load-probe-fixture";
    original=data=="original-target-fixture" || resident_loop || load_probe;
    draft_loop=data=="draft-resident-loop64-fixture";
    draft64=data=="draft-resident64-fixture" || draft_loop;
    if (!std::filesystem::path(path).is_absolute() || (!original && !draft64 && data!="draft-standalone-fixture")) return 2;
    restored=Failure()!="v2-restore";
    return restored?0:31;
  }
  std::vector<NDTensorDesc> GetInputTensorDescs() const override {
    if (original) {auto rows=OriginalDescriptors(true,resident_loop);if(prefill){rows[0].dims={1,1,64,2048};rows[1].dims={1,1,64,2048};rows[2].dims={1,64};rows[83].dims={64};rows[85].dims={1,64,1};}return rows;}
    const std::int32_t seq=draft64?64:8;
    auto input=std::vector<NDTensorDesc>{DraftDesc({1,seq,16384},DataType::FLOAT16),
        DraftDesc({1,seq},DataType::INT32),DraftDesc({1,seq},DataType::INT32)};
    if (Failure()=="v2-dtype") input[0].dataType=DataType::FLOAT32;
    if (Failure()=="v2-format") input[0].format=Format::NCHW;
    if (Failure()=="v2-shape") input[0].dims={1,4,32768};
    if (Failure()=="v2-zero-dim") input[0].dims={1,0,16384};
    ModelIoNames(input, "input");
    return input;
  }
  std::vector<NDTensorDesc> GetOutputTensorDescs() const override {
    if (original) {auto rows=OriginalDescriptors(false,resident_loop);if(prefill)rows[81].dims={1,1,64,16384};return rows;}
    if (Failure()=="v2-count") return {};
    auto output = std::vector<NDTensorDesc>{DraftDesc({1,15},DataType::INT32),DraftDesc({1,15,2048},DataType::FLOAT16)};
    ModelIoNames(output, "output");
    return output;
  }
};
class FakeModelManager final : public IModelManager {
 public:
  FakeModelManager() { ++clients; }
  ~FakeModelManager() override {
    if (initialized_ || weights_) { std::cerr << "LEAK v2 model manager/weights\n"; std::abort(); }
    Trace("v2-manager-destroy"); --clients;
  }
  Status InitWeights(const std::string& directory) override {
    Trace("v2-init-weights");
    if (weights_ || !initialized_ || !std::filesystem::path(directory).is_absolute() ||
        !std::filesystem::is_directory(directory)) return 71;
    if (Failure()=="original-init-weights" || (Failure()=="v2-target-init-weights"&&name_=="qwen35_resident_target")) return 72;
    if (Failure()=="original-init-weights-throw") throw std::runtime_error("injected InitWeights exception");
    if (Failure()=="original-init-weights-unknown") throw 123;
    weights_=true;
    return 0;
  }
  Status DeinitWeights() override {
    Trace("v2-deinit-weights");
    if (!weights_ || !initialized_) { std::cerr << "invalid weights cleanup order\n"; std::abort(); }
    weights_=false;
    if (Failure()=="original-deinit-weights-throw") throw std::runtime_error("injected DeinitWeights exception");
    if (Failure()=="original-deinit-weights-unknown") throw 123;
    return Failure()=="original-deinit-weights" || Failure()=="original-both-cleanup-failures"?73:0;
  }
  Status Init(const ModelInitOptions& options,const std::shared_ptr<IBuiltModel>& model,
              const std::shared_ptr<IModelManagerListener>& listener) override {
    Trace("v2-init");
    auto concrete=std::dynamic_pointer_cast<FakeBuiltModel>(model);
    if (listener || options.perfMode!=PerfMode::HIGH || !concrete || !concrete->restored) return 4;
    if (concrete->original && weights_) { model_=model; return 74; }
    model_=model; initialized_=true; original_=concrete->original; draft64_=concrete->draft64;
    load_probe_=concrete->load_probe;prefill_=concrete->prefill;name_=concrete->name;Trace(("v2-init:"+name_).c_str());
    resident_loop_=concrete->resident_loop;draft_loop_=concrete->draft_loop;
    return Failure()=="v2-init" || Failure()=="original-init" ||
      Failure()=="original-init-deinit-throw" || (prefill_&&Failure()=="v2-prefill-init") || (draft64_ && Failure()=="resident-draft-init")?32:0;
  }
  Status Run(const std::vector<std::shared_ptr<INDTensorBuffer>>& inputs,
             std::vector<std::shared_ptr<INDTensorBuffer>>& outputs) override {
    Trace("v2-run");Trace(("v2-run:"+name_).c_str());
    if (original_) {
      if (!initialized_ || !weights_ || model_.expired() || inputs.size()!=89 || outputs.size()!=82) return 75;
      if (Failure()=="original-run" || (prefill_&&Failure()=="v2-prefill-run")) return 76;
      if (Failure()=="original-run-unknown-exception") throw 123;
      if (Failure()=="original-replace-output") outputs[1]=CreateNDTensorBuffer(outputs[1]->GetTensorDesc());
      if(resident_loop_) {
        std::int32_t position=0;
        if(prefill_){std::int16_t valid=0;std::memcpy(&valid,inputs[88]->GetData(),2);position=valid;if(position<1||position>64)return 80;}
        else std::memcpy(&position,inputs[2]->GetData(),4);
        const auto token=static_cast<const std::uint8_t*>(inputs[0]->GetData())[(prefill_?position-1:0)*2048];
        for(std::size_t ref=0;ref<80;++ref) {
          const auto& in=inputs[ref+3];const auto& out=outputs[1+ref*37%80];
          std::int32_t actual_ref=0,actual_position=0;
          std::memcpy(&actual_ref,in->GetData(),4);std::memcpy(&actual_position,static_cast<const std::uint8_t*>(in->GetData())+4,4);
          if((!prefill_&&(actual_ref!=static_cast<std::int32_t>(ref)||actual_position!=position))||in->GetSize()!=out->GetSize())return 79;
          std::memcpy(out->GetData(),in->GetData(),in->GetSize());const auto next=prefill_?position:position+1;
          const auto state_ref=static_cast<std::int32_t>(ref);std::memcpy(out->GetData(),&state_ref,4);
          std::memcpy(static_cast<std::uint8_t*>(out->GetData())+4,&next,4);
        }
        std::memset(outputs[0]->GetData(),0,outputs[0]->GetSize());static_cast<float*>(outputs[0]->GetData())[token+1]=2.0f;
        std::fill_n(static_cast<std::uint16_t*>(outputs[81]->GetData()),outputs[81]->GetSize()/2,0x3800);++runs_;return 0;
      }
      ++runs_;
      for (std::size_t i=0;i<80;++i) {
        const auto& in=inputs[i+3]; const auto& out=outputs[i+1];
        if (in->GetSize()!=out->GetSize()) return 77;
        std::memcpy(out->GetData(),in->GetData(),in->GetSize());
        ++static_cast<std::uint8_t*>(out->GetData())[0];
      }
      std::memset(outputs[0]->GetData(),0,outputs[0]->GetSize());
      static_cast<float*>(outputs[0]->GetData())[runs_]=static_cast<float>(runs_);
      std::memset(outputs[81]->GetData(),static_cast<int>(runs_),outputs[81]->GetSize());
      return 0;
    }
    if (!initialized_ || model_.expired() || inputs.size()!=3 || outputs.size()!=2) return 5;
    if (Failure()=="v2-run") return 33;
    // Validate all three supplied input payloads before producing test patterns.
    const std::size_t seq=draft64_?64:8;
    if (inputs[0]->GetSize()!=seq*16384*2 || inputs[1]->GetSize()!=seq*4 || inputs[2]->GetSize()!=seq*4) return 6;
    auto* features=static_cast<const std::uint16_t*>(inputs[0]->GetData());
    auto* ids=static_cast<const std::int32_t*>(inputs[1]->GetData());
    auto* mask=static_cast<const std::int32_t*>(inputs[2]->GetData());
    if(draft_loop_) {
      std::size_t valid=0;while(valid<seq&&mask[valid]==1)++valid;
      if(!valid)return 7;
      for(std::size_t i=0;i<seq;++i) {
        if(mask[i]!=(i<valid?1:0)||ids[i]!=(i<valid?static_cast<std::int32_t>(i+1):0))return 8;
        if(!std::all_of(features+i*16384,features+(i+1)*16384,[&](auto x){return x==(i<valid?0x3800:0);}))return 7;
      }
      auto* top=static_cast<std::int32_t*>(outputs[0]->GetData());
      for(std::int32_t i=0;i<15;++i)top[i]=ids[valid-1]+1+i;
      std::fill_n(static_cast<std::uint16_t*>(outputs[1]->GetData()),15*2048,0x3800);++runs_;return 0;
    }
    if (!std::all_of(features,features+seq*16384,[](auto x){return x==0x3800;})) return 7;
    for (std::size_t i=0;i<seq;++i) if (ids[i]!=static_cast<std::int32_t>(i) || mask[i]!=(i<6?1:0)) return 8;
    if (Failure()=="v2-replace-output") outputs[0]=CreateNDTensorBuffer(outputs[0]->GetTensorDesc());
    auto* top=static_cast<std::int32_t*>(outputs[0]->GetData());
    for (std::int32_t i=0;i<15;++i) top[i]=i+10+(draft64_?static_cast<std::int32_t>(runs_):0);
    ++runs_;
    std::fill_n(static_cast<std::uint16_t*>(outputs[1]->GetData()),15*2048,0x3800);
    return 0;
  }
  void DeInit() override {
    Trace("v2-deinit");
    if (model_.expired()) { std::cerr << "premature v2 built model release\n"; std::abort(); }
    // Host fixture models manager-owned teardown for the explicit mixed candidate.
    // This is not evidence of private-device external-weight release behavior.
    initialized_=false;weights_=false;Trace(("v2-deinit:"+name_).c_str());
    if(prefill_&&Failure()=="v2-prefill-deinit")throw std::runtime_error("injected prefill DeInit exception");
    if(original_&&(Failure()=="original-deinit"||(Failure()=="v2-target-deinit"&&name_=="qwen35_resident_target")))throw std::runtime_error("injected target DeInit exception");
    if(original_&&(Failure()=="original-init-deinit-throw"||Failure()=="original-both-cleanup-failures"))throw std::runtime_error("injected target DeInit exception");
    if(draft64_&&Failure()=="resident-draft-deinit")throw std::runtime_error("injected draft DeInit exception");
  }
 private:
  bool initialized_=false,weights_=false,original_=false,draft64_=false,resident_loop_=false,draft_loop_=false,load_probe_=false,prefill_=false;
  std::string name_;
  std::size_t runs_=0;
  std::weak_ptr<IBuiltModel> model_;
};
}
std::shared_ptr<INDTensorBuffer> CreateNDTensorBuffer(const NDTensorDesc& desc) {
  Trace("v2-create-tensor");
  if (Failure()=="v2-allocation" && tensors==3) return nullptr;
#ifndef QWEN35_FAKE_NO_TENSOR_NAME
  if (Failure()=="resident-cross-alias" && desc.dims==std::vector<std::int32_t>{1,64,16384})
    return resident_alias_storage.lock();
  if (Failure()=="original-input-partial-alias" && desc.tensorName=="runtime_input_5")
    return std::make_shared<FakeOffsetNDTensor>(desc,original_alias_input.lock());
  if (Failure()=="original-output-alias" && desc.tensorName=="runtime_output_3")
    return original_alias_output.lock();
  if ((Failure()=="original-input-alias" && desc.tensorName=="runtime_input_5") ||
      (Failure()=="original-output-input-alias" && desc.tensorName=="runtime_output_1"))
    return original_alias_input.lock();
#endif
  auto tensor=std::make_shared<FakeNDTensor>(desc);
#ifndef QWEN35_FAKE_NO_TENSOR_NAME
  if (desc.tensorName=="runtime_input_0") mixed_target_storage=tensor->GetData();
  if (desc.tensorName=="runtime_input_3") original_alias_input=tensor;
  if (desc.tensorName=="runtime_output_1") original_alias_output=tensor;
  if (desc.tensorName=="runtime_input_9") resident_alias_storage=tensor;
#endif
  return tensor;
}
std::shared_ptr<IBuiltModel> CreateBuiltModel() {
  Trace("v2-create-built-model");
  if (Failure()=="v2-create-model") return nullptr;
  return std::make_shared<FakeBuiltModel>();
}
std::shared_ptr<IModelManager> CreateModelManager() {
  Trace("v2-create-manager");
  if (Failure()=="v2-create-manager") return nullptr;
  return std::make_shared<FakeModelManager>();
}
}
