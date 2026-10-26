#include "qwen35_dflash/hiai_executor.hpp"
#include <HiAiModelManagerService.h>
#include <algorithm>
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace qwen35::dflash {
namespace {
void Check(hiai::AIStatus status, const char* operation) {
  if (status != hiai::AI_SUCCESS)
    throw std::runtime_error(std::string("HIAI ") + operation + " failed: " + std::to_string(status));
}
std::string Version(hiai::AiModelMngerClient& client) {
  const auto* version = client.GetVersion();
  if (!version || !*version) throw std::runtime_error("HIAI GetVersion returned empty");
  return version;
}
std::string Shape(const std::vector<std::uint32_t>& dims) {
  std::ostringstream out;
  out << '[';
  for (std::size_t i = 0; i < dims.size(); ++i) {
    if (i) out << ',';
    out << dims[i];
  }
  return out.str() + ']';
}
bool IsTrailingOnePadding(const std::vector<std::uint32_t>& declared,
                          const std::vector<std::uint32_t>& actual) {
  // Same bounded adaptation as the board-validated operator probe. Preserve
  // every declared axis; equal element counts alone do not establish layout.
  if (declared.empty() || declared.size() > 3 || actual.size() != 4) return false;
  for (std::size_t i = 0; i < actual.size(); ++i)
    if (actual[i] != (i < declared.size() ? declared[i] : 1)) return false;
  return true;
}
struct Tensor {
  std::shared_ptr<hiai::AiTensor> value;
  std::size_t bytes;
  void* Data() const {
    if (!value || !value->GetBuffer() || value->GetSize() < bytes)
      throw std::runtime_error("HIAI tensor buffer is smaller than ABI v2");
    return value->GetBuffer();
  }
};
// The HCL runtime uses the descriptor's bytes even when a model path is set.
// File-backed read-only storage avoids an additional whole-model heap copy.
class ModelBuffer {
 public:
  ModelBuffer() = default;
  ModelBuffer(const ModelBuffer&) = delete;
  ModelBuffer& operator=(const ModelBuffer&) = delete;
  ~ModelBuffer() {
    if (data_ != MAP_FAILED) static_cast<void>(munmap(data_, bytes_));
    if (fd_ >= 0) static_cast<void>(close(fd_));
  }
  void Open(const std::filesystem::path& path) {
    if (fd_ >= 0) throw std::logic_error("model buffer already open");
    fd_ = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) throw std::runtime_error(std::string("open OMC: ") + std::strerror(errno));
    struct stat info {};
    if (fstat(fd_, &info) != 0)
      throw std::runtime_error(std::string("stat OMC: ") + std::strerror(errno));
    if (!S_ISREG(info.st_mode) || info.st_size <= 0)
      throw std::runtime_error("OMC model buffer requires a nonempty regular file");
    if (static_cast<std::uint64_t>(info.st_size) > UINT32_MAX)
      throw std::runtime_error("OMC exceeds SetModelBuffer uint32 size limit (UINT32_MAX bytes); "
                               "a validated large-model loading route is required");
    bytes_ = static_cast<std::uint32_t>(info.st_size);
    data_ = mmap(nullptr, bytes_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (data_ == MAP_FAILED)
      throw std::runtime_error(std::string("mmap OMC: ") + std::strerror(errno));
  }
  const void* Data() const { return data_ == MAP_FAILED ? nullptr : data_; }
  std::uint32_t Size() const { return bytes_; }
 private:
  int fd_ = -1;
  void* data_ = MAP_FAILED;
  std::uint32_t bytes_ = 0;
};
struct Model {
  // Reverse destruction order keeps mapped bytes and descriptor alive through
  // UnLoadModel AND the client destructor, including partial-load failures.
  ModelBuffer model_buffer;
  std::shared_ptr<hiai::AiModelDescription> description;
  hiai::AiModelMngerClient client;
  bool initialized = false;
  std::string name, version;
  std::vector<hiai::TensorDimension> input_dims, output_dims;
  std::vector<Tensor> inputs, outputs;
  std::vector<std::shared_ptr<hiai::AiTensor>> input_ptrs, output_ptrs;
  std::uint32_t timeout = 0;
  Model() = default;
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;
  ~Model() {
    // Process is synchronous: unload before releasing persistent tensor buffers.
    if (initialized) static_cast<void>(client.UnLoadModel());
  }
  void Load(const std::filesystem::path& path, const std::string& model_name,
            const std::string& expected_version, std::uint32_t timeout_ms,
            std::size_t input_count, std::size_t output_count) {
    name = model_name; timeout = timeout_ms;
    Check(client.Init(nullptr), "Init(sync)"); initialized = true;
    version = Version(client);
    if (version != expected_version)
      throw std::runtime_error("HIAI version differs from --expected-runtime-version");
    if (!client.CheckNPUSupport()) throw std::runtime_error("HIAI NPU unavailable");
    description = std::make_shared<hiai::AiModelDescription>(
        name, hiai::AiModelDescription_Frequency_HIGH, hiai::HIAI_FRAMEWORK_NONE,
        hiai::HIAI_MODELTYPE_OFFLINE, hiai::AiModelDescription_DeviceType_NPU);
    auto check_load = [&](hiai::AIStatus status, const char* operation) {
      if (status != hiai::AI_SUCCESS)
        throw std::runtime_error(std::string("HIAI ") + operation + " failed: " +
            std::to_string(status) + " [model=" + name + ", path=" + path.string() +
            ", runtime=" + version + "]");
    };
    check_load(description->SetModelPath(path.string()), "SetModelPath");
    try {
      model_buffer.Open(path);
    } catch (const std::exception& error) {
      throw std::runtime_error(std::string(error.what()) + " [model=" + name +
                               ", path=" + path.string() + ", runtime=" + version + "]");
    }
    check_load(description->SetModelBuffer(model_buffer.Data(), model_buffer.Size()), "SetModelBuffer");
    if (!description->GetModelBuffer() || description->GetModelNetSize() != model_buffer.Size())
      throw std::runtime_error("HIAI model buffer descriptor is empty or truncated [model=" + name +
                               ", path=" + path.string() + ", runtime=" + version + "]");
    // Load still decides compatibility; no bypass or model type change.
    std::vector<std::shared_ptr<hiai::AiModelDescription>> descriptions{description};
    check_load(client.Load(descriptions), "Load");
    check_load(client.GetModelIOTensorDim(name, input_dims, output_dims), "GetModelIOTensorDim");
    for (bool input : {true, false}) {
      const auto& dims = input ? input_dims : output_dims;
      for (std::size_t i = 0; i < dims.size(); ++i)
        std::cerr << "HIAI " << name << ' ' << (input ? "input" : "output")
                  << '[' << i << "]: shape=" << Shape(dims[i].GetDims())
                  << ", format=" << static_cast<int>(dims[i].GetFormat())
                  << ", dtype=unavailable\n";
    }
    if (input_dims.size() != input_count || output_dims.size() != output_count)
      throw std::runtime_error("HIAI OMC tensor count differs from ABI v2");
  }
  void Add(bool input, std::size_t index, const std::vector<std::uint32_t>& shape,
           hiai::HIAI_DataType dtype) {
    const auto& dim = (input ? input_dims : output_dims).at(index);
    const auto actual = dim.GetDims();
    const std::string context = name + ' ' + (input ? "input[" : "output[") +
        std::to_string(index) + "]: declared_shape=" + Shape(shape) +
        ", actual_shape=" + Shape(actual) + ", actual_format=" +
        std::to_string(static_cast<int>(dim.GetFormat()));
    const bool padded = IsTrailingOnePadding(shape, actual);
    if (dim.GetFormat() != hiai::Format::ND || (actual != shape && !padded))
      throw std::runtime_error("HIAI OMC tensor shape/format differs from ND ABI v2: " + context);
    if (padded)
      std::cerr << "HIAI accepted ND trailing-one padding for " << context
                << "; using runtime descriptor\n";
    if (dtype != hiai::HIAI_DATATYPE_FLOAT16 && dtype != hiai::HIAI_DATATYPE_INT32)
      throw std::invalid_argument("ABI v2 allows only FP16 features and INT32 tokens/masks");
    std::uint64_t bytes = dtype == hiai::HIAI_DATATYPE_FLOAT16 ? 2 : 4;
    for (auto n : shape) {
      if (!n || bytes > UINT32_MAX / n)
        throw std::runtime_error("HIAI tensor exceeds uint32 byte-size limit");
      bytes *= n;
    }
    Tensor t{std::make_shared<hiai::AiTensor>(), static_cast<std::size_t>(bytes)};
    // Legacy HIAI metadata exposes dimensions only. Dtypes are locked by the
    // AIR/OMC manifest and model SHA256; do not claim dtype introspection here.
    Check(t.value->Init(&dim, dtype), "AiTensor::Init");
    static_cast<void>(t.Data());
    (input ? input_ptrs : output_ptrs).push_back(t.value);
    (input ? inputs : outputs).push_back(std::move(t));
  }
  void Run() {
    hiai::AiContext context;
    context.AddPara("model_name", name);
    std::int32_t stamp = 0;
    Check(client.Process(context, input_ptrs, output_ptrs, timeout, stamp), "Process(sync)");
  }
};
}  // namespace

HiaiProbe ProbeHiai() {
  hiai::AiModelMngerClient client;
  Check(client.Init(nullptr), "Init(sync)");
  return {Version(client), client.CheckNPUSupport()};
}
struct HiaiExecutor::Impl {
  Model target, draft;
  std::size_t sequence = 0, width = 0, vocab = 0;
  GraphOutputs outputs;
};
HiaiExecutor::HiaiExecutor(const std::filesystem::path& target,
                           const std::filesystem::path& draft,
                           const std::string& expected_version,
                           std::size_t feature_width, std::size_t vocabulary_size,
                           std::uint32_t timeout_ms)
    : impl_(std::make_unique<Impl>()) {
  if (!feature_width || feature_width > INT32_MAX || !timeout_ms ||
      !vocabulary_size || vocabulary_size > INT32_MAX || expected_version.empty())
    throw std::invalid_argument("positive int32 feature width/vocabulary and runtime version required");
  impl_->vocab = vocabulary_size;
  impl_->target.Load(target, "qwen35_target", expected_version, timeout_ms, 2, 2);
  impl_->draft.Load(draft, "qwen35_draft", expected_version, timeout_ms, 3, 1);
  const auto actual_shape = impl_->target.input_dims[0].GetDims();
  if (actual_shape.size() < 2 || actual_shape[0] != 1 || actual_shape[1] < 2 || actual_shape[1] > INT32_MAX)
    throw std::runtime_error("target IDs must have shape [1,S], S>1");
  const std::vector<std::uint32_t> shape{1, actual_shape[1]};
  impl_->sequence = shape[1];
  const std::vector<std::uint32_t> features{1, shape[1], static_cast<std::uint32_t>(feature_width)};
  impl_->target.Add(true, 0, shape, hiai::HIAI_DATATYPE_INT32);
  impl_->target.Add(true, 1, shape, hiai::HIAI_DATATYPE_INT32);
  impl_->target.Add(false, 0, shape, hiai::HIAI_DATATYPE_INT32);
  impl_->target.Add(false, 1, features, hiai::HIAI_DATATYPE_FLOAT16);
  impl_->draft.Add(true, 0, features, hiai::HIAI_DATATYPE_FLOAT16);
  impl_->draft.Add(true, 1, shape, hiai::HIAI_DATATYPE_INT32);
  impl_->draft.Add(true, 2, shape, hiai::HIAI_DATATYPE_INT32);
  const auto actual_draft_shape = impl_->draft.output_dims[0].GetDims();
  if (actual_draft_shape.size() < 2 || actual_draft_shape[0] != 1 || !actual_draft_shape[1] || actual_draft_shape[1] > INT32_MAX)
    throw std::runtime_error("draft output must have shape [1,K]");
  const std::vector<std::uint32_t> draft_shape{1, actual_draft_shape[1]};
  impl_->draft.Add(false, 0, draft_shape, hiai::HIAI_DATATYPE_INT32);
  impl_->width = draft_shape[1];
  impl_->outputs.target_top1.resize(impl_->sequence);
  impl_->outputs.draft_top1.resize(impl_->width);
}
HiaiExecutor::~HiaiExecutor() = default;
std::size_t HiaiExecutor::sequence_length() const noexcept { return impl_->sequence; }
std::size_t HiaiExecutor::draft_width() const noexcept { return impl_->width; }
const std::string& HiaiExecutor::runtime_version() const noexcept { return impl_->target.version; }
const GraphOutputs& HiaiExecutor::Execute(const std::vector<std::int64_t>& prefix,
                                         std::int64_t pad, bool produce_draft) {
  if (prefix.empty() || prefix.size() > impl_->sequence)
    throw std::invalid_argument("prefix outside sequence gear");
  auto valid = [this](std::int64_t t) { return t >= 0 && static_cast<std::size_t>(t) < impl_->vocab; };
  if (!valid(pad) || !std::all_of(prefix.begin(), prefix.end(), valid))
    throw std::invalid_argument("input token outside vocabulary");
  auto* ids = static_cast<std::int32_t*>(impl_->target.inputs[0].Data());
  auto* mask = static_cast<std::int32_t*>(impl_->target.inputs[1].Data());
  std::fill_n(ids, impl_->sequence, static_cast<std::int32_t>(pad));
  std::fill_n(mask, impl_->sequence, 0);
  std::copy(prefix.begin(), prefix.end(), ids);
  std::fill_n(mask, prefix.size(), 1);
  impl_->target.Run();
  const auto* top1 = static_cast<const std::int32_t*>(impl_->target.outputs[0].Data());
  std::copy_n(top1, impl_->sequence, impl_->outputs.target_top1.begin());
  std::fill(impl_->outputs.draft_top1.begin(), impl_->outputs.draft_top1.end(), 0);
  if (produce_draft) {
    std::memcpy(impl_->draft.inputs[0].Data(), impl_->target.outputs[1].Data(), impl_->target.outputs[1].bytes);
    std::memcpy(impl_->draft.inputs[1].Data(), ids, impl_->target.inputs[0].bytes);
    std::memcpy(impl_->draft.inputs[2].Data(), mask, impl_->target.inputs[1].bytes);
    impl_->draft.Run();
    const auto* proposed = static_cast<const std::int32_t*>(impl_->draft.outputs[0].Data());
    std::copy_n(proposed, impl_->width, impl_->outputs.draft_top1.begin());
  }
  if (!std::all_of(impl_->outputs.target_top1.begin(), impl_->outputs.target_top1.end(), valid) ||
      !std::all_of(impl_->outputs.draft_top1.begin(), impl_->outputs.draft_top1.end(), valid))
    throw std::runtime_error("HIAI produced a token outside vocabulary");
  return impl_->outputs;
}
}  // namespace qwen35::dflash
