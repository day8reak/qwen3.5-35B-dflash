#include "HiAiModelManagerService.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
namespace {
int tensors=0, clients=0;
std::string Failure() { const auto* p=std::getenv("FAKE_HIAI_FAILURE"); return p?p:""; }
struct CheckLeaks {
  ~CheckLeaks() { if (tensors || clients) { std::cerr << "LEAK\n"; std::abort(); } }
} check_leaks;
}
namespace hiai {
AiTensor::AiTensor() { ++tensors; }
AiTensor::~AiTensor() { --tensors; }
AIStatus AiTensor::Init(const TensorDimension* dim, HIAI_DataType type) {
  if (Failure()=="allocation" && tensors==5) return AI_FAILED;
  if (type!=HIAI_DATATYPE_INT32 && type!=HIAI_DATATYPE_FLOAT16) return AI_FAILED;
  if (dim->GetDims().size()==3 && type!=HIAI_DATATYPE_FLOAT16) return AI_FAILED;
  std::size_t n=1; for(auto d:dim->GetDims()) n*=d;
  bytes_=n*(type==HIAI_DATATYPE_FLOAT16?2:4);
  data_.resize((bytes_+3)/4); return AI_SUCCESS;
}
void* AiTensor::GetBuffer() const { return const_cast<std::int32_t*>(data_.data()); }
std::uint32_t AiTensor::GetSize() const { return Failure()=="short-buffer"?1:bytes_; }
AiModelDescription::AiModelDescription(const std::string& n,int freq,int framework,int model,int device):name(n) {
  if(freq!=3 || framework!=0 || model!=1 || device!=0) throw std::runtime_error("wrong HIAI model mode");
}
AIStatus AiModelDescription::SetModelPath(const std::string& p) { path=p; return Failure()=="path"?1:0; }
void AiContext::AddPara(const std::string& key,const std::string& value) {
  if(key!="model_name") throw std::runtime_error("context missing model_name"); model=value;
}
AiModelMngerClient::AiModelMngerClient(){ ++clients; }
AiModelMngerClient::~AiModelMngerClient(){ if(loaded_) { std::cerr<<"LEAK loaded model\n"; std::abort(); } --clients; }
AIStatus AiModelMngerClient::Init(std::shared_ptr<AiModelManagerClientListener> listener) {
  if(listener) throw std::runtime_error("must initialize synchronously"); return Failure()=="init"?1:0;
}
char* AiModelMngerClient::GetVersion(){ static char s[]="FAKE-HIAI-X90"; return Failure()=="version"?nullptr:s; }
bool AiModelMngerClient::CheckNPUSupport(){ return Failure()!="no-npu"; }
AIStatus AiModelMngerClient::CheckModelCompatibility(AiModelDescription&,bool& yes){ yes=Failure()!="incompatible"; return Failure()=="compat-error"?1:0; }
AIStatus AiModelMngerClient::Load(std::vector<std::shared_ptr<AiModelDescription>>& d){
  if(d.size()!=1 || d[0]->path.empty()) return 1;
  name_=d[0]->name; loaded_=true; // partial load failure must still unload.
  return Failure()=="load"?1:0;
}
AIStatus AiModelMngerClient::UnLoadModel(){ loaded_=false; return 0; }
AIStatus AiModelMngerClient::GetModelIOTensorDim(const std::string& n,std::vector<TensorDimension>& in,std::vector<TensorDimension>& out){
  if(n!=name_ || !loaded_) return 1;
  TensorDimension ids({1,32}), features({1,32,32});
  if(n=="qwen35_target") { in={ids,ids}; out={ids,features}; }
  else { in={features,ids,ids}; out={TensorDimension({1,3})}; }
  if(Failure()=="shape") in[0]=TensorDimension({2,3});
  if(Failure()=="format") in[0]=TensorDimension({1,32},Format::NCHW);
  if(Failure()=="count") out.clear();
  return 0;
}
AIStatus AiModelMngerClient::Process(AiContext& c,std::vector<std::shared_ptr<AiTensor>>& in,std::vector<std::shared_ptr<AiTensor>>& out,std::uint32_t timeout,std::int32_t& stamp){
  if(c.model!=name_ || !timeout || !loaded_ || Failure()=="run") return 1;
  const bool target=name_=="qwen35_target";
  auto* ids=static_cast<std::int32_t*>(in[target?0:1]->GetBuffer());
  auto* mask=static_cast<std::int32_t*>(in[target?1:2]->GetBuffer());
  auto* top=static_cast<std::int32_t*>(out[0]->GetBuffer());
  int length=0; for(int i=0;i<32;++i) length+=mask[i];
  if(!length) return 1;
  if(target) {
    for(int i=0;i<32;++i) top[i]=(ids[i]+1)%64;
    auto* f=static_cast<std::uint16_t*>(out[1]->GetBuffer()); std::fill_n(f,32*32,0);
    f[0]=1234;
    f[32*32-1]=4321;
  } else {
    const auto* f=static_cast<std::uint16_t*>(in[0]->GetBuffer());
    if(f[0]!=1234 || f[32*32-1]!=4321) return 1;
    for(int i=0;i<3;++i) top[i]=(ids[length-1]+i+1)%64;
    if(Failure()=="corrupt-draft") top[1]=63;
  }
  if(Failure()=="token") top[0]=64;
  stamp=1; return 0;
}
}
