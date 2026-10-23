// Probe transport test double. It copies bytes; it does not simulate NPU math.
#include "HiAiModelManagerService.h"
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
namespace {
std::string Failure() { const auto* p = std::getenv("FAKE_HIAI_FAILURE"); return p ? p : ""; }
void Trace(const char* event) {
  if (const auto* p = std::getenv("FAKE_HIAI_TRACE")) std::ofstream(p, std::ios::app) << event << '\n';
}
bool Valid(const void* data, std::uint32_t bytes) {
  return data && bytes == 13 && std::memcmp(data, "probe-fixture", 13) == 0;
}
}
namespace hiai {
AiTensor::AiTensor() = default;
AiTensor::~AiTensor() = default;
AIStatus AiTensor::Init(const TensorDimension* dim, HIAI_DataType type) {
  if (Failure() == "allocation") return 1;
  bytes_ = type == HIAI_DATATYPE_FLOAT16 ? 2 : 4;
  for (auto n : dim->GetDims()) bytes_ *= n;
  data_.resize((bytes_ + 3) / 4);
  return 0;
}
void* AiTensor::GetBuffer() const { return const_cast<std::int32_t*>(data_.data()); }
std::uint32_t AiTensor::GetSize() const { return Failure() == "short-buffer" ? 1 : bytes_; }
AiModelDescription::AiModelDescription(const std::string& n, int f, int w, int m, int d) : name(n) {
  if (f != 3 || w != 0 || m != 1 || d != 0) throw std::runtime_error("wrong model mode");
}
AIStatus AiModelDescription::SetModelPath(const std::string& p) { path = p; return 0; }
AIStatus AiModelDescription::SetModelBuffer(const void* p, std::uint32_t n) {
  if (!Valid(p, n) || Failure() == "model-buffer") return 1;
  model_buffer_ = p; model_bytes_ = n; return 0;
}
void* AiModelDescription::GetModelBuffer() const { return const_cast<void*>(model_buffer_); }
std::uint32_t AiModelDescription::GetModelNetSize() const { return model_bytes_; }
void AiContext::AddPara(const std::string& key, const std::string& value) {
  if (key != "model_name") throw std::runtime_error("bad context key");
  model = value;
}
AiModelMngerClient::AiModelMngerClient() = default;
AiModelMngerClient::~AiModelMngerClient() {
  if (loaded_ || (model_buffer_ && !Valid(model_buffer_, model_bytes_))) std::abort();
  Trace("destroy");
}
AIStatus AiModelMngerClient::Init(std::shared_ptr<AiModelManagerClientListener> listener) {
  if (listener) throw std::runtime_error("probe must execute synchronously");
  return Failure() == "init" ? 1 : 0;
}
char* AiModelMngerClient::GetVersion() { static char version[] = "FAKE-PROBE"; return version; }
AIStatus AiModelMngerClient::Load(std::vector<std::shared_ptr<AiModelDescription>>& models) {
  Trace("load");
  if (models.size() != 1 || models[0]->path.empty() || !Valid(models[0]->GetModelBuffer(), models[0]->GetModelNetSize())) return 1;
  model_buffer_ = models[0]->GetModelBuffer(); model_bytes_ = models[0]->GetModelNetSize();
  name_ = models[0]->name; loaded_ = true;
  return Failure() == "load" ? 1 : 0;
}
AIStatus AiModelMngerClient::GetModelIOTensorDim(const std::string& name, std::vector<TensorDimension>& in,
                                              std::vector<TensorDimension>& out) {
  if (name != name_ || !loaded_) return 1;
  in = {TensorDimension({1, 4})}; out = in;
  if (Failure() == "shape") in = {TensorDimension({4})};
  if (Failure() == "format") in = {TensorDimension({1, 4}, Format::NCHW)};
  if (Failure() == "count") out.clear();
  return 0;
}
AIStatus AiModelMngerClient::Process(AiContext& context, std::vector<std::shared_ptr<AiTensor>>& in,
                                    std::vector<std::shared_ptr<AiTensor>>& out, std::uint32_t timeout,
                                    std::int32_t& stamp) {
  Trace("process");
  if (!Valid(model_buffer_, model_bytes_) || context.model != name_ || timeout != 30000 || !loaded_) std::abort();
  if (Failure() == "process") return 1;
  if (in.size() != 1 || out.size() != 1 || in[0]->GetSize() != out[0]->GetSize()) return 1;
  std::memcpy(out[0]->GetBuffer(), in[0]->GetBuffer(), in[0]->GetSize()); stamp = 1;
  return 0;
}
AIStatus AiModelMngerClient::UnLoadModel() {
  if (model_buffer_ && !Valid(model_buffer_, model_bytes_)) std::abort();
  Trace("unload"); loaded_ = false; return Failure() == "unload" ? 1 : 0;
}
}
