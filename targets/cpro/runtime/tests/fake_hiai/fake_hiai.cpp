#include "HiAiModelManagerService.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
namespace {
int tensors=0, clients=0;
std::uint16_t* target_features=nullptr;
std::string Failure() { const auto* p=std::getenv("FAKE_HIAI_FAILURE"); return p?p:""; }
std::uint32_t FeatureWidth() {
  return Failure()=="large-features"?1025:Failure()=="capture-limit"?1024:32;
}
std::string Layout() { const auto* p=std::getenv("FAKE_HIAI_LAYOUT"); return p?p:""; }
void Trace(const char* event) {
  if (const auto* path=std::getenv("FAKE_HIAI_TRACE")) std::ofstream(path, std::ios::app) << event << '\n';
}
bool ModelBytesValid(const void* buffer, std::uint32_t bytes, const std::string& name) {
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
AiTensor::AiTensor() { ++tensors; }
AiTensor::~AiTensor() { --tensors; }
AIStatus AiTensor::Init(const TensorDimension* dim, HIAI_DataType type) {
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
  if (type!=HIAI_DATATYPE_INT32 && type!=HIAI_DATATYPE_FLOAT16 && type!=HIAI_DATATYPE_FLOAT32) return AI_FAILED;
  if (dim->GetDims().size()==3 && type!=HIAI_DATATYPE_FLOAT16) return AI_FAILED;
  std::size_t n=1; for(auto d:dim->GetDims()) n*=d;
  bytes_=n*(type==HIAI_DATATYPE_FLOAT16?2:4);
  data_.resize((bytes_+3)/4); return AI_SUCCESS;
}
void* AiTensor::GetBuffer() const { return const_cast<std::int32_t*>(data_.data()); }
std::uint32_t AiTensor::GetSize() const {
  return Failure()=="short-buffer"?1:Failure()=="oversized-buffer"?65537:bytes_;
}
AiModelDescription::AiModelDescription(const std::string& n,int freq,int framework,int model,int device):name(n) {
  if(freq!=3 || framework!=0 || model!=1 || device!=0) throw std::runtime_error("wrong HIAI model mode");
}
AIStatus AiModelDescription::SetModelPath(const std::string& p) { path=p; return Failure()=="path"?1:0; }
AIStatus AiModelDescription::SetModelBuffer(const void* data, std::uint32_t bytes) {
  if(Failure()=="model-buffer") return AI_FAILED;
  if(!ModelBytesValid(data, bytes, name)) return AI_FAILED;
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
  if(d.size()!=1 || d[0]->path.empty() ||
     !ModelBytesValid(d[0]->GetModelBuffer(), d[0]->GetModelNetSize(), d[0]->name)) return AI_FAILED;
  name_=d[0]->name;
  model_buffer_=d[0]->GetModelBuffer(); model_bytes_=d[0]->GetModelNetSize();
  loaded_=true; // partial target or draft load failure must still unload.
  return Failure()=="load" || Failure()=="incompatible" ||
         (Failure()=="draft-load" && name_=="qwen35_draft")?1:0;
}
AIStatus AiModelMngerClient::UnLoadModel(){
  RequireModelBytes(model_buffer_, model_bytes_, name_);
  Trace("unload"); loaded_=false; return 0;
}
AIStatus AiModelMngerClient::GetModelIOTensorDim(const std::string& n,std::vector<TensorDimension>& in,std::vector<TensorDimension>& out){
  if(n!=name_ || !loaded_ || Failure()=="io-dims") return 1;
  TensorDimension ids({1,32}), features({1,32,FeatureWidth()});
  if(n=="qwen35_stateful_target") {
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
