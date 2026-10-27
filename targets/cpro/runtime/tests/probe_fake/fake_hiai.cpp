// Probe transport test double. It copies bytes; it does not simulate NPU math.
#include "HiAiModelManagerService.h"
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
namespace {
std::vector<const hiai::TensorDimension*> runtime_dimensions;
std::size_t initialized_dimensions = 0, input_count = 0, output_count = 0;
std::map<const hiai::AiTensor*, std::uint32_t> logical_sizes;
std::string Failure() { const auto* p = std::getenv("FAKE_HIAI_FAILURE"); return p ? p : ""; }
void Trace(const char* event) {
  if (const auto* p = std::getenv("FAKE_HIAI_TRACE")) std::ofstream(p, std::ios::app) << event << '\n';
}
bool Valid(const void* data, std::uint32_t bytes) {
  return data && bytes == 13 && std::memcmp(data, "probe-fixture", 13) == 0;
}
std::vector<hiai::TensorDimension> Dimensions(const char* shape_key, const char* format_key) {
  const auto* value = std::getenv(shape_key);
  const auto* format_value = std::getenv(format_key);
  const auto format = format_value && std::string(format_value) == "0" ? hiai::Format::NCHW : hiai::Format::ND;
  std::istringstream tensors(value && *value ? value : "1,4");
  std::vector<hiai::TensorDimension> result;
  std::string tensor;
  while (std::getline(tensors, tensor, ';')) {
    std::istringstream dimensions(tensor);
    std::vector<std::uint32_t> shape;
    std::string field;
    while (std::getline(dimensions, field, ',')) shape.push_back(static_cast<std::uint32_t>(std::stoul(field)));
    result.emplace_back(shape, format);
  }
  return result;
}
}
namespace hiai {
AiTensor::AiTensor() = default;
AiTensor::~AiTensor() { logical_sizes.erase(this); }
AIStatus AiTensor::Init(const TensorDimension* dim, HIAI_DataType type) {
  if (initialized_dimensions >= runtime_dimensions.size() || runtime_dimensions[initialized_dimensions++] != dim)
    throw std::runtime_error("AiTensor::Init must receive the actual runtime descriptor");
  if (const auto* path = std::getenv("FAKE_HIAI_INIT_TRACE")) {
    std::ofstream trace(path, std::ios::app);
    trace << "format=" << static_cast<int>(dim->GetFormat()) << ";dtype=" << static_cast<int>(type) << ";shape=";
    const auto shape = dim->GetDims();
    for (std::size_t i = 0; i < shape.size(); ++i) trace << (i ? "," : "") << shape[i];
    trace << '\n';
  }
  if (Failure() == "allocation" || (Failure() == "allocation-second" && initialized_dimensions == 2)) return 1;
  bytes_ = type == HIAI_DATATYPE_FLOAT16 ? 2 : 4;
  for (auto n : dim->GetDims()) bytes_ *= n;
  logical_sizes[this] = bytes_;
  if (const auto* extra = std::getenv("FAKE_HIAI_EXTRA_BUFFER_BYTES")) {
    const auto count = std::stoul(extra);
    if (count > 1024) throw std::runtime_error("fake extra allocation exceeds test bound");
    bytes_ += static_cast<std::uint32_t>(count);
  }
  data_.resize((bytes_ + 3) / 4);
  // A known test sentinel makes extra allocation observable without assuming
  // that production allocation slack is initialized or is a row stride.
  std::memset(data_.data(), 0xa5, data_.size() * sizeof(data_[0]));
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
  in = Dimensions("FAKE_HIAI_INPUT_DIMS", "FAKE_HIAI_INPUT_FORMAT");
  out = Dimensions("FAKE_HIAI_OUTPUT_DIMS", "FAKE_HIAI_OUTPUT_FORMAT");
  if (Failure() == "shape") in = {TensorDimension({4})};
  if (Failure() == "format") in = {TensorDimension({1, 4}, Format::NCHW)};
  if (Failure() == "count") out.clear();
  runtime_dimensions.clear(); initialized_dimensions = 0;
  input_count = in.size(); output_count = out.size();
  for (const auto& dim : in) runtime_dimensions.push_back(&dim);
  for (const auto& dim : out) runtime_dimensions.push_back(&dim);
  return 0;
}
AIStatus AiModelMngerClient::Process(AiContext& context, std::vector<std::shared_ptr<AiTensor>>& in,
                                    std::vector<std::shared_ptr<AiTensor>>& out, std::uint32_t timeout,
                                    std::int32_t& stamp) {
  Trace("process");
  if (!Valid(model_buffer_, model_bytes_) || context.model != name_ || timeout != 30000 || !loaded_) std::abort();
  if (Failure() == "process") return 1;
  if (initialized_dimensions != runtime_dimensions.size() || in.size() != input_count || out.size() != output_count
      || in.empty()) return 1;
  for (const auto& tensor : in) {
    const auto* bytes = static_cast<const unsigned char*>(tensor->GetBuffer());
    for (auto i = logical_sizes.at(tensor.get()); i < tensor->GetSize(); ++i)
      if (bytes[i] != 0xa5) throw std::runtime_error("probe changed input allocation slack");
  }
  std::vector<std::size_t> sources(out.size(), 0);
  if (const auto* mapping = std::getenv("FAKE_HIAI_OUTPUT_INPUTS"); mapping && *mapping) {
    std::istringstream fields(mapping);
    std::string field;
    sources.clear();
    while (std::getline(fields, field, ',')) sources.push_back(std::stoul(field));
  }
  if (sources.size() != out.size()) return 1;
  for (std::size_t i = 0; i < out.size(); ++i) {
    if (sources[i] >= in.size()) return 1;
    const auto bytes = logical_sizes.at(out[i].get());
    if (bytes != logical_sizes.at(in[sources[i]].get())) return 1;
    std::memcpy(out[i]->GetBuffer(), in[sources[i]]->GetBuffer(), bytes);
  }
  stamp = 1;
  return 0;
}
AIStatus AiModelMngerClient::UnLoadModel() {
  if (model_buffer_ && !Valid(model_buffer_, model_bytes_)) std::abort();
  Trace("unload"); loaded_ = false; return Failure() == "unload" ? 1 : 0;
}
}
