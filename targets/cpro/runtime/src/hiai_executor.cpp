#include "qwen35_dflash/hiai_executor.hpp"
#include "qwen35_dflash/sha256.hpp"
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
constexpr std::size_t kFeatureCaptureLimit = 65536;
constexpr std::size_t kStatefulTensorLimit = 512ULL << 20;
constexpr std::uint64_t kStatefulTotalLimit = 2ULL << 30;
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
bool IsLeadingOnePadding(const std::vector<std::uint32_t>& declared,
                         const std::vector<std::uint32_t>& actual) {
  // Match the probe's rank3 ND [a,b,c] -> [1,a,b,c] adaptation. Rank2
  // token IO still follows its original exact/trailing-one ABI rules.
  if (declared.size() != 3 || actual.size() != 4 || actual[0] != 1) return false;
  for (std::size_t i = 0; i < declared.size(); ++i)
    if (actual[i + 1] != declared[i]) return false;
  return true;
}
struct Tensor {
  std::shared_ptr<hiai::AiTensor> value;
  std::size_t bytes;
  void* Data() const {
    if (!value || !value->GetBuffer() || value->GetSize() < bytes)
      throw std::runtime_error("HIAI tensor buffer is smaller than the declared tensor ABI");
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
            std::size_t input_count, std::size_t output_count, const char* abi = "ABI v2") {
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
      throw std::runtime_error(std::string("HIAI OMC tensor count differs from ") + abi);
  }
  void Add(bool input, std::size_t index, const std::vector<std::uint32_t>& shape,
           hiai::HIAI_DataType dtype, bool stateful = false) {
    const auto& dim = (input ? input_dims : output_dims).at(index);
    const auto actual = dim.GetDims();
    const std::string context = name + ' ' + (input ? "input[" : "output[") +
        std::to_string(index) + "]: declared_shape=" + Shape(shape) +
        ", actual_shape=" + Shape(actual) + ", actual_format=" +
        std::to_string(static_cast<int>(dim.GetFormat()));
    const bool trailing = IsTrailingOnePadding(shape, actual);
    const bool leading = IsLeadingOnePadding(shape, actual);
    const bool scalar = shape.empty() && !actual.empty() && actual.size() <= 4 &&
                        std::all_of(actual.begin(), actual.end(), [](auto dim) { return dim == 1; });
    const bool padded = trailing || leading || scalar;
    if (dim.GetFormat() != hiai::Format::ND || (actual != shape && !padded))
      throw std::runtime_error(std::string("HIAI OMC tensor shape/format differs from ") +
          (stateful ? "stateful ND ABI: " : "ND ABI v2: ") + context);
    if (padded)
      std::cerr << "HIAI accepted ND " << (scalar ? "scalar-one" : trailing ? "trailing-one" : "leading-one")
                << " padding for " << context
                << "; using runtime descriptor\n";
    if (dtype != hiai::HIAI_DATATYPE_FLOAT16 && dtype != hiai::HIAI_DATATYPE_INT32 &&
        !(stateful && (dtype == hiai::HIAI_DATATYPE_FLOAT32 || dtype == hiai::HIAI_DATATYPE_INT16 ||
                       dtype == hiai::HIAI_DATATYPE_INT64)))
      throw std::invalid_argument("ABI v2 allows only FP16 features and INT32 tokens/masks");
    std::uint64_t bytes = dtype == hiai::HIAI_DATATYPE_INT64 ? 8 :
                          dtype == hiai::HIAI_DATATYPE_FLOAT16 || dtype == hiai::HIAI_DATATYPE_INT16 ? 2 : 4;
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

std::size_t HiaiTensorBytes(const HiaiTensorSpec& spec) {
  if (spec.name.empty() || spec.shape.size() > 8 ||
      (spec.dtype != "int32" && spec.dtype != "int16" && spec.dtype != "int64" &&
       spec.dtype != "float16" && spec.dtype != "float32"))
    throw std::invalid_argument("ordered tensor requires a name, rank 0..8 and int16/int32/int64/float16/float32 dtype");
  std::size_t bytes = spec.dtype == "int64" ? 8 : spec.dtype == "float16" || spec.dtype == "int16" ? 2 : 4;
  for (auto dim : spec.shape) {
    if (!dim || bytes > kStatefulTensorLimit / dim)
      throw std::invalid_argument("stateful tensor exceeds 512 MiB or has a zero dimension");
    bytes *= dim;
  }
  return bytes;
}

struct HiaiStatefulExecutor::Impl {
  Model target;
  std::vector<bool> ready;
  bool outputs_available = false;
  HiaiIoDiagnostic diagnostic;
  std::vector<const void*> output_buffers_before;
  std::vector<std::string> input_source_hashes;

  bool ObserveDiagnosticOutputs() {
    bool unchanged = target.output_ptrs.size() == target.outputs.size();
    for (std::size_t i = 0; i < target.outputs.size(); ++i) {
      auto& record = diagnostic.outputs.at(i);
      record.checked_after_process = true;
      const auto current = i < target.output_ptrs.size() ? target.output_ptrs[i] : nullptr;
      const auto* buffer = current ? current->GetBuffer() : nullptr;
      record.allocated_bytes_after = current ? current->GetSize() : 0;
      record.object_same = current == target.outputs[i].value;
      record.buffer_same = buffer && buffer == output_buffers_before.at(i);
      record.fill_count_available = buffer && record.allocated_bytes_after >= record.logical_bytes;
      if (record.fill_count_available) {
        const auto* data = static_cast<const std::uint8_t*>(buffer);
        record.fill_byte_count = static_cast<std::size_t>(std::count(
            data, data + record.logical_bytes, static_cast<std::uint8_t>(diagnostic.output_fill_byte)));
      }
      unchanged = unchanged && record.object_same && record.buffer_same && record.fill_count_available;
    }
    return unchanged;
  }
};
HiaiStatefulExecutor::HiaiStatefulExecutor(
    const std::filesystem::path& target, const std::string& expected_version,
    const std::vector<HiaiTensorSpec>& inputs, const std::vector<HiaiTensorSpec>& outputs,
    std::uint32_t timeout_ms, bool fused_operator_probe,
    int diagnostic_output_fill_byte, bool standalone_draft) : impl_(std::make_unique<Impl>()) {
  if (inputs.empty() || inputs.size() > 1024 || outputs.empty() || outputs.size() > 1024 ||
      expected_version.empty() || !timeout_ms)
    throw std::invalid_argument("stateful target requires bounded ordered IO, runtime identity and timeout");
  if (diagnostic_output_fill_byte < -1 || diagnostic_output_fill_byte > 255 ||
      (diagnostic_output_fill_byte >= 0 && !fused_operator_probe && !standalone_draft))
    throw std::invalid_argument("output fill diagnostic requires a probe/draft step and a byte in 0..255");
  impl_->diagnostic.output_fill_byte = diagnostic_output_fill_byte;
  const std::uint16_t endian = 1;
  if (*reinterpret_cast<const std::uint8_t*>(&endian) != 1)
    throw std::invalid_argument("stateful target payloads require little-endian buffers");
  std::uint64_t total = 0;
  for (const auto* specs : {&inputs, &outputs}) {
    for (const auto& spec : *specs) {
      if (!fused_operator_probe && (spec.shape.empty() || spec.dtype == "int64"))
        throw std::invalid_argument("scalar/INT64 IO is restricted to the explicit fused-operator contract");
      total += HiaiTensorBytes(spec);
      if (total > kStatefulTotalLimit)
        throw std::invalid_argument("stateful target IO exceeds 2 GiB");
    }
  }
  impl_->target.Load(target, standalone_draft ? "qwen35_standalone_draft" :
                     fused_operator_probe ? "qwen35_fused_operator" : "qwen35_stateful_target",
                     expected_version, timeout_ms, inputs.size(), outputs.size(),
                     standalone_draft ? "standalone full draft ABI" :
                     fused_operator_probe ? "documented fused operator ABI" : "stateful target ABI");
  std::uint64_t allocated = 0;
  for (bool input : {true, false}) {
    const auto& specs = input ? inputs : outputs;
    for (std::size_t i = 0; i < specs.size(); ++i) {
      const auto& spec = specs[i];
      const auto dtype = spec.dtype == "float16" ? hiai::HIAI_DATATYPE_FLOAT16 :
                         spec.dtype == "float32" ? hiai::HIAI_DATATYPE_FLOAT32 :
                         spec.dtype == "int64" ? hiai::HIAI_DATATYPE_INT64 :
                         spec.dtype == "int16" ? hiai::HIAI_DATATYPE_INT16 : hiai::HIAI_DATATYPE_INT32;
      impl_->target.Add(input, i, spec.shape, dtype, true);
      const auto& tensor = (input ? impl_->target.inputs : impl_->target.outputs).back();
      allocated += tensor.value->GetSize();
      if (tensor.value->GetSize() > kStatefulTensorLimit || allocated > kStatefulTotalLimit)
        throw std::runtime_error("stateful runtime tensor buffers exceed bounded IO allocation");
      // Extra runtime allocation is not interpreted as physical row padding.
      std::memset(tensor.Data(), 0, tensor.bytes);
      if (diagnostic_output_fill_byte >= 0) {
        if (input) {
          HiaiInputIoDiagnostic record;
          record.index = i; record.logical_bytes = tensor.bytes; record.allocated_bytes = tensor.value->GetSize();
          impl_->diagnostic.inputs.push_back(record);
        } else {
          HiaiOutputIoDiagnostic record;
          record.index = i; record.logical_bytes = tensor.bytes;
          impl_->diagnostic.outputs.push_back(record);
        }
      }
    }
  }
  impl_->ready.assign(inputs.size(), false);
  if (diagnostic_output_fill_byte >= 0) impl_->input_source_hashes.resize(inputs.size());
}
HiaiStatefulExecutor::~HiaiStatefulExecutor() = default;
void HiaiStatefulExecutor::SetInput(std::size_t index, const std::vector<std::uint8_t>& bytes) {
  impl_->outputs_available = false;
  const auto& tensor = impl_->target.inputs.at(index);
  impl_->ready.at(index) = false;
  if (bytes.size() != tensor.bytes)
    throw std::invalid_argument("stateful input byte count differs from declared shape/dtype");
  std::memcpy(tensor.Data(), bytes.data(), bytes.size());
  if (impl_->diagnostic.output_fill_byte >= 0) {
    impl_->input_source_hashes.at(index) = Sha256(std::string_view(
        reinterpret_cast<const char*>(bytes.data()), bytes.size()));
    impl_->diagnostic.inputs.at(index).readback_sha256.clear();
    impl_->diagnostic.inputs.at(index).readback_matches_source = false;
  }
  impl_->ready[index] = true;
}
void HiaiStatefulExecutor::Execute() {
  impl_->outputs_available = false;
  if (!std::all_of(impl_->ready.begin(), impl_->ready.end(), [](bool ready) { return ready; }))
    throw std::logic_error("all explicit stateful inputs must be supplied for each step");
  std::fill(impl_->ready.begin(), impl_->ready.end(), false);
  if (impl_->diagnostic.output_fill_byte >= 0) {
    impl_->diagnostic.process_called = false;
    impl_->output_buffers_before.clear();
    for (std::size_t i = 0; i < impl_->target.outputs.size(); ++i) {
      const auto& tensor = impl_->target.outputs[i];
      auto& record = impl_->diagnostic.outputs[i];
      record.allocated_bytes_before = tensor.value->GetSize();
      record.checked_after_process = record.object_same = record.buffer_same = false;
      record.fill_count_available = false; record.fill_byte_count = 0;
      void* data = tensor.Data();
      impl_->output_buffers_before.push_back(data);
      std::memset(data, impl_->diagnostic.output_fill_byte, tensor.bytes);
    }
    // Read every input only after all copies AND output fills. Immediate
    // per-SetInput readback would miss later writes through aliased buffers.
    bool inputs_match = true;
    for (std::size_t i = 0; i < impl_->target.inputs.size(); ++i) {
      const auto& tensor = impl_->target.inputs[i];
      auto& record = impl_->diagnostic.inputs[i];
      record.allocated_bytes = tensor.value->GetSize();
      const auto* readback = static_cast<const char*>(tensor.Data());
      record.readback_sha256 = Sha256(std::string_view(readback, tensor.bytes));
      record.readback_matches_source = record.readback_sha256 == impl_->input_source_hashes[i];
      inputs_match = inputs_match && record.readback_matches_source;
    }
    if (!inputs_match)
      throw std::runtime_error("diagnostic input tensor readback differs from supplied bytes before Process");
    impl_->diagnostic.process_called = true;
    try {
      impl_->target.Run();
    } catch (...) {
      static_cast<void>(impl_->ObserveDiagnosticOutputs());
      throw;
    }
    if (!impl_->ObserveDiagnosticOutputs())
      throw std::runtime_error("diagnostic output object, buffer or size changed during Process; "
                               "refusing to read the original output tensor");
  } else {
    impl_->target.Run();
  }
  impl_->outputs_available = true;
}
std::vector<std::uint8_t> HiaiStatefulExecutor::CopyOutput(std::size_t index) const {
  if (!impl_->outputs_available)
    throw std::logic_error("stateful outputs unavailable before successful Process");
  const auto& tensor = impl_->target.outputs.at(index);
  const auto* data = static_cast<const std::uint8_t*>(tensor.Data());
  return {data, data + tensor.bytes};
}
const std::string& HiaiStatefulExecutor::runtime_version() const noexcept {
  return impl_->target.version;
}
const HiaiIoDiagnostic& HiaiStatefulExecutor::io_diagnostic() const noexcept {
  return impl_->diagnostic;
}

struct HiaiExecutor::Impl {
  Model target, draft;
  std::size_t sequence = 0, width = 0, vocab = 0;
  GraphOutputs outputs;
  bool capture_target_features = false, features_available = false;
  std::vector<std::uint8_t> target_features;
};
HiaiExecutor::HiaiExecutor(const std::filesystem::path& target,
                           const std::filesystem::path& draft,
                           const std::string& expected_version,
                           std::size_t feature_width, std::size_t vocabulary_size,
                           std::uint32_t timeout_ms, bool capture_target_features)
    : impl_(std::make_unique<Impl>()) {
  if (!feature_width || feature_width > INT32_MAX || !timeout_ms ||
      !vocabulary_size || vocabulary_size > INT32_MAX || expected_version.empty())
    throw std::invalid_argument("positive int32 feature width/vocabulary and runtime version required");
  const std::uint16_t endian_marker = 1;
  if (capture_target_features &&
      *reinterpret_cast<const std::uint8_t*>(&endian_marker) != 1)
    throw std::invalid_argument("target feature capture requires little-endian FP16 buffers");
  impl_->vocab = vocabulary_size;
  impl_->capture_target_features = capture_target_features;
  impl_->target.Load(target, "qwen35_target", expected_version, timeout_ms, 2, 2);
  impl_->draft.Load(draft, "qwen35_draft", expected_version, timeout_ms, 3, 1);
  const auto actual_shape = impl_->target.input_dims[0].GetDims();
  if (actual_shape.size() < 2 || actual_shape[0] != 1 || actual_shape[1] < 2 || actual_shape[1] > INT32_MAX)
    throw std::runtime_error("target IDs must have shape [1,S], S>1");
  const std::vector<std::uint32_t> shape{1, actual_shape[1]};
  impl_->sequence = shape[1];
  const std::vector<std::uint32_t> features{1, shape[1], static_cast<std::uint32_t>(feature_width)};
  if (capture_target_features &&
      feature_width > kFeatureCaptureLimit / 2 / impl_->sequence)
    throw std::invalid_argument("target feature capture exceeds 65536-byte limit");
  impl_->target.Add(true, 0, shape, hiai::HIAI_DATATYPE_INT32);
  impl_->target.Add(true, 1, shape, hiai::HIAI_DATATYPE_INT32);
  impl_->target.Add(false, 0, shape, hiai::HIAI_DATATYPE_INT32);
  impl_->target.Add(false, 1, features, hiai::HIAI_DATATYPE_FLOAT16);
  if (capture_target_features &&
      impl_->target.outputs[1].value->GetSize() > kFeatureCaptureLimit)
    throw std::invalid_argument("target feature buffer exceeds 65536-byte capture limit");
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
const std::vector<std::uint8_t>& HiaiExecutor::target_features() const {
  if (!impl_->capture_target_features || !impl_->features_available)
    throw std::logic_error("target feature capture unavailable before successful execution");
  return impl_->target_features;
}
const GraphOutputs& HiaiExecutor::Execute(const std::vector<std::int64_t>& prefix,
                                         std::int64_t pad, bool produce_draft) {
  impl_->features_available = false;
  impl_->target_features.clear();
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
  if (impl_->capture_target_features) {
    const auto& features = impl_->target.outputs[1];
    if (features.value->GetSize() > kFeatureCaptureLimit)
      throw std::runtime_error("target feature buffer exceeds 65536-byte capture limit");
    const auto* bytes = static_cast<const std::uint8_t*>(features.Data());
    impl_->target_features.assign(bytes, bytes + features.bytes);
  }
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
  impl_->features_available = impl_->capture_target_features;
  return impl_->outputs;
}
}  // namespace qwen35::dflash
