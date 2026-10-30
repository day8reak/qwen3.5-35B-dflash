#include "qwen35_dflash/hiai_executor.hpp"
#include "qwen35_dflash/sha256.hpp"
#include <HiAiModelManagerService.h>
#include <model/built_model.h>
#include <model_manager/model_manager.h>
#include <tensor/nd_tensor_buffer.h>
#include <algorithm>
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <limits>
#include <set>
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
  std::shared_ptr<hiai::INDTensorBuffer> nd_value;
  std::size_t Size() const { return nd_value ? nd_value->GetSize() : value ? value->GetSize() : 0; }
  void* Buffer() const { return nd_value ? nd_value->GetData() : value ? value->GetBuffer() : nullptr; }
  const void* Object() const { return nd_value ? static_cast<const void*>(nd_value.get()) : value.get(); }
  void* Data() const {
    if (!Object() || !Buffer() || Size() < bytes)
      throw std::runtime_error("HIAI tensor buffer is smaller than the declared tensor ABI");
    return Buffer();
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
  bool file_v2 = false, v2_init_attempted = false;
  std::string stage = "LEGACY_LOAD";
  std::shared_ptr<hiai::IBuiltModel> built_model;
  std::shared_ptr<hiai::IModelManager> manager;
  std::vector<hiai::NDTensorDesc> nd_input_dims, nd_output_dims;
  std::vector<std::shared_ptr<hiai::INDTensorBuffer>> nd_input_ptrs, nd_output_ptrs;
  void Stage(const char* next) {
    stage = next;
    std::cerr << "HIAI file-v2 stage=" << stage << " [model=" << name << "]" << std::endl;
  }
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
    if (manager && v2_init_attempted) manager->DeInit();
    // Release manager while its restored model and tensor storage are alive.
    manager.reset();
    built_model.reset();
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
  void LoadFile(const std::filesystem::path& path, const std::string& expected_version,
                std::size_t input_count, std::size_t output_count, bool descriptor_only = false) {
    file_v2 = true;
    name = descriptor_only ? "qwen35_model_io_probe" : "qwen35_standalone_draft";
    try {
      Stage("RUNTIME_IDENTITY");
      const auto probe = ProbeHiai();
      version = probe.version;
      if (version != expected_version)
        throw std::runtime_error("HIAI version differs from --expected-runtime-version");
      if (!probe.npu_supported) throw std::runtime_error("HIAI NPU unavailable");
      Stage("CREATE_BUILT_MODEL");
      built_model = hiai::CreateBuiltModel();
      if (!built_model) throw std::runtime_error("CreateBuiltModel returned null");
      Stage("RESTORE_FROM_FILE");
      // Keep the full file path and adjacent external-weight files available to
      // the runtime. Do not round-trip through a buffer or legacy descriptor.
      const auto restored = built_model->RestoreFromFile(path.c_str());
      if (restored != 0) throw std::runtime_error("RestoreFromFile failed: " + std::to_string(restored));
      if (descriptor_only) {
        // Explicit restore-only experiment: query the restored object's public
        // metadata before creating or initializing a manager. External-weight
        // setup is not requested or inferred. Failure has no fallback route.
        Stage("TENSOR_DESCRIPTORS");
        nd_input_dims = built_model->GetInputTensorDescs();
        nd_output_dims = built_model->GetOutputTensorDescs();
        if (nd_input_dims.empty() || nd_output_dims.empty()
            || nd_input_dims.size() > 4096 || nd_output_dims.size() > 4096)
          throw std::runtime_error("model IO probe requires 1..4096 ordered descriptors per direction");
        return;
      }
      Stage("CREATE_MODEL_MANAGER");
      manager = hiai::CreateModelManager();
      if (!manager) throw std::runtime_error("CreateModelManager returned null");
      hiai::ModelInitOptions options;
      options.perfMode = hiai::PerfMode::HIGH;
      Stage("MODEL_MANAGER_INIT");
      v2_init_attempted = true;
      const auto initialized_status = manager->Init(options, built_model, nullptr);
      if (initialized_status != 0)
        throw std::runtime_error("IModelManager::Init failed: " + std::to_string(initialized_status));
      Stage("TENSOR_DESCRIPTORS");
      nd_input_dims = built_model->GetInputTensorDescs();
      nd_output_dims = built_model->GetOutputTensorDescs();
      if (nd_input_dims.size() != input_count || nd_output_dims.size() != output_count)
        throw std::runtime_error("HIAI V2 tensor count differs from standalone full draft ABI");
    } catch (const std::exception& error) {
      throw HiaiRuntimeError(stage, std::string(error.what()) + " [path=" + path.string() +
                             ", runtime=" + version + ", api=file-v2]");
    }
  }
  void AddFileTensor(bool input, std::size_t index, const HiaiTensorSpec& spec) {
    try {
      Stage(input ? "INPUT_TENSOR_ALLOCATION" : "OUTPUT_TENSOR_ALLOCATION");
      const auto& desc = (input ? nd_input_dims : nd_output_dims).at(index);
      std::vector<std::uint32_t> actual;
      for (const auto dim : desc.dims) {
        if (dim <= 0) throw std::runtime_error("V2 tensor has a nonpositive runtime dimension");
        actual.push_back(static_cast<std::uint32_t>(dim));
      }
      const auto dtype = spec.dtype == "float16" ? hiai::DataType::FLOAT16 :
                         spec.dtype == "float32" ? hiai::DataType::FLOAT32 :
                         spec.dtype == "int64" ? hiai::DataType::INT64 :
                         spec.dtype == "int16" ? hiai::DataType::INT16 : hiai::DataType::INT32;
      const auto context = name + ' ' + (input ? "input[" : "output[") +
          std::to_string(index) + "]: declared_shape=" + Shape(spec.shape) +
          ", actual_shape=" + Shape(actual) + ", actual_format=" +
          std::to_string(static_cast<int>(desc.format)) + ", actual_dtype=" +
          std::to_string(static_cast<int>(desc.dataType));
      std::cerr << "HIAI file-v2 " << context << std::endl;
      if (desc.format != hiai::Format::ND ||
          (actual != spec.shape && !IsTrailingOnePadding(spec.shape, actual) &&
           !IsLeadingOnePadding(spec.shape, actual)))
        throw std::runtime_error("HIAI V2 tensor shape/format differs from full draft ND ABI: " + context);
      if (desc.dataType != dtype)
        throw std::runtime_error("HIAI V2 tensor dtype differs from full draft ABI: " + context);
      auto buffer = hiai::CreateNDTensorBuffer(desc);
      if (!buffer) throw std::runtime_error("CreateNDTensorBuffer returned null: " + context);
      Tensor tensor{nullptr, HiaiTensorBytes(spec), buffer};
      static_cast<void>(tensor.Data());
      (input ? nd_input_ptrs : nd_output_ptrs).push_back(buffer);
      (input ? inputs : outputs).push_back(std::move(tensor));
    } catch (const std::exception& error) {
      throw HiaiRuntimeError(stage, error.what());
    }
  }
  Tensor CurrentOutput(std::size_t index) const {
    if (file_v2)
      return {nullptr, outputs.at(index).bytes, index < nd_output_ptrs.size() ? nd_output_ptrs[index] : nullptr};
    return {index < output_ptrs.size() ? output_ptrs[index] : nullptr, outputs.at(index).bytes, nullptr};
  }
  std::size_t OutputCount() const { return file_v2 ? nd_output_ptrs.size() : output_ptrs.size(); }
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
    Tensor t{std::make_shared<hiai::AiTensor>(), static_cast<std::size_t>(bytes), nullptr};
    // Legacy HIAI metadata exposes dimensions only. Dtypes are locked by the
    // AIR/OMC manifest and model SHA256; do not claim dtype introspection here.
    Check(t.value->Init(&dim, dtype), "AiTensor::Init");
    static_cast<void>(t.Data());
    (input ? input_ptrs : output_ptrs).push_back(t.value);
    (input ? inputs : outputs).push_back(std::move(t));
  }
  void Run() {
    if (file_v2) {
      Stage("MODEL_MANAGER_RUN");
      const auto status = manager->Run(nd_input_ptrs, nd_output_ptrs);
      if (status != 0)
        throw HiaiRuntimeError(stage, "IModelManager::Run failed: " + std::to_string(status));
      Stage("OUTPUT_VALIDATION");
      return;
    }
    stage = "LEGACY_PROCESS";
    hiai::AiContext context;
    context.AddPara("model_name", name);
    std::int32_t stamp = 0;
    Check(client.Process(context, input_ptrs, output_ptrs, timeout, stamp), "Process(sync)");
    stage = "OUTPUT_VALIDATION";
  }
};
}  // namespace

HiaiProbe ProbeHiai() {
  hiai::AiModelMngerClient client;
  Check(client.Init(nullptr), "Init(sync)");
  return {Version(client), client.CheckNPUSupport()};
}

HiaiDescriptionProbe ProbeHiaiModelDescription(const std::filesystem::path& model) {
  constexpr std::uintmax_t limit = 64ULL << 20;
  if (!std::filesystem::is_regular_file(model) || std::filesystem::file_size(model) > limit)
    throw std::invalid_argument("descriptor probe requires a regular main OMC of at most 64 MiB");
  ModelBuffer buffer;
  buffer.Open(model);
  if (buffer.Size() > limit)
    throw std::invalid_argument("descriptor probe main OMC exceeds 64 MiB");
  HiaiDescriptionProbe result;
  result.model_path = std::filesystem::absolute(model).string();
  result.model_bytes = buffer.Size();
  result.model_sha256 = Sha256(std::string_view(static_cast<const char*>(buffer.Data()), buffer.Size()));
  result.runtime = ProbeHiai();
  for (const bool path_first : {true, false}) {
    // A fresh description for each order isolates setter interactions. Keep
    // mapped bytes alive until both descriptions have been destroyed.
    hiai::AiModelDescription description("qwen35_model_description_probe",
        hiai::AiModelDescription_Frequency_HIGH, hiai::HIAI_FRAMEWORK_NONE,
        hiai::HIAI_MODELTYPE_OFFLINE, hiai::AiModelDescription_DeviceType_NPU);
    HiaiDescriptionOrder observation;
    observation.order = path_first ? "path-then-buffer" : "buffer-then-path";
    for (const bool set_path : {path_first, !path_first}) {
      const auto status = set_path ? description.SetModelPath(result.model_path)
          : description.SetModelBuffer(buffer.Data(), buffer.Size());
      const auto* actual_buffer = description.GetModelBuffer();
      observation.steps.push_back({set_path ? "SetModelPath" : "SetModelBuffer",
          description.GetModelPath(), static_cast<std::int64_t>(status),
          actual_buffer != nullptr, actual_buffer == buffer.Data(), description.GetModelNetSize()});
    }
    result.observations.push_back(std::move(observation));
  }
  return result;
}

HiaiModelIoProbe ProbeHiaiModelIo(const std::filesystem::path& model,
                                const std::string& expected_runtime_version) {
  const auto path = std::filesystem::absolute(model);
  if (expected_runtime_version.empty() || std::filesystem::is_symlink(path)
      || !std::filesystem::is_regular_file(path) || std::filesystem::file_size(path) == 0)
    throw std::invalid_argument("model IO probe requires a nonempty regular main OMC and runtime identity");
  struct stat before {};
  if (stat(path.c_str(), &before) != 0)
    throw std::runtime_error("cannot stat model IO probe OMC");
  HiaiModelIoProbe result;
  result.model_path = path.string();
  result.model_bytes = static_cast<std::uint64_t>(before.st_size);
  result.model_sha256 = Sha256File(path);
  result.tensor_name_field_available = detail::OptionalTensorName<hiai::NDTensorDesc>::available;
  {
    Model loaded;
    loaded.LoadFile(path, expected_runtime_version, 0, 0, true);
    result.runtime = {loaded.version, true};
    auto capture = [](const std::vector<hiai::NDTensorDesc>& descriptors) {
      std::vector<HiaiModelIoTensor> observations;
      for (std::size_t index = 0; index < descriptors.size(); ++index) {
        const auto& descriptor = descriptors[index];
        if (descriptor.dims.size() > 32)
          throw HiaiRuntimeError("TENSOR_DESCRIPTORS", "runtime descriptor exceeds model IO probe rank limit32");
        HiaiModelIoTensor item;
        item.index = index;
        item.shape.assign(descriptor.dims.begin(), descriptor.dims.end());
        item.raw_dtype = static_cast<int>(descriptor.dataType);
        item.raw_format = static_cast<int>(descriptor.format);
        item.tensor_name = detail::OptionalTensorName<hiai::NDTensorDesc>::Read(descriptor);
        if (item.tensor_name.size() > 4096)
          throw HiaiRuntimeError("TENSOR_DESCRIPTORS", "runtime tensor name exceeds model IO probe byte limit4096");
        if (detail::OptionalTensorName<hiai::NDTensorDesc>::available)
          item.tensor_name_status = item.tensor_name.empty() ? "EMPTY" : "OBSERVED";
        observations.push_back(std::move(item));
      }
      return observations;
    };
    result.inputs = capture(loaded.nd_input_dims);
    result.outputs = capture(loaded.nd_output_dims);
  }
  struct stat after {};
  if (stat(path.c_str(), &after) != 0 || before.st_dev != after.st_dev || before.st_ino != after.st_ino
      || before.st_size != after.st_size || before.st_mtime != after.st_mtime
      || before.st_ctime != after.st_ctime || before.st_mode != after.st_mode
      || Sha256File(path) != result.model_sha256)
    throw HiaiRuntimeError("MODEL_PRESERVATION", "main OMC changed during restore-only model IO observation");
  return result;
}

HiaiModelIoEnumTable HiaiModelIoEnums() {
  HiaiModelIoEnumTable result = {{{"FLOAT16", static_cast<int>(hiai::DataType::FLOAT16)},
           {"FLOAT32", static_cast<int>(hiai::DataType::FLOAT32)},
           {"INT32", static_cast<int>(hiai::DataType::INT32)},
           {"INT16", static_cast<int>(hiai::DataType::INT16)},
           {"INT64", static_cast<int>(hiai::DataType::INT64)}},
          {{"ND", static_cast<int>(hiai::Format::ND)}}};
  detail::OptionalInt8Enum<hiai::DataType>::Append(result.dtypes);
  return result;
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
    bool unchanged = target.OutputCount() == target.outputs.size();
    for (std::size_t i = 0; i < target.outputs.size(); ++i) {
      auto& record = diagnostic.outputs.at(i);
      record.checked_after_process = true;
      const auto current = target.CurrentOutput(i);
      const auto* buffer = current.Buffer();
      record.allocated_bytes_after = current.Size();
      record.object_same = current.Object() == target.outputs[i].Object();
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
    int diagnostic_output_fill_byte, bool standalone_draft,
    const std::string& model_load_api) : impl_(std::make_unique<Impl>()) {
  if (model_load_api != "legacy-buffer" && model_load_api != "file-v2")
    throw std::invalid_argument("model load API must be legacy-buffer or file-v2");
  if (model_load_api == "file-v2" && !standalone_draft)
    throw std::invalid_argument("file-v2 requires the standalone draft ABI");
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
  if (model_load_api == "file-v2")
    impl_->target.LoadFile(target, expected_version, inputs.size(), outputs.size());
  else impl_->target.Load(target, standalone_draft ? "qwen35_standalone_draft" :
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
      if (model_load_api == "file-v2") impl_->target.AddFileTensor(input, i, spec);
      else impl_->target.Add(input, i, spec.shape, dtype, true);
      const auto& tensor = (input ? impl_->target.inputs : impl_->target.outputs).back();
      const auto allocated_bytes = tensor.Size();
      if (allocated_bytes > kStatefulTensorLimit || allocated_bytes > kStatefulTotalLimit - allocated) {
        if (impl_->target.file_v2)
          throw HiaiRuntimeError(impl_->target.stage, "V2 runtime tensor buffers exceed bounded IO allocation");
        throw std::runtime_error("stateful runtime tensor buffers exceed bounded IO allocation");
      }
      allocated += allocated_bytes;
      // Extra runtime allocation is not interpreted as physical row padding.
      std::memset(tensor.Data(), 0, tensor.bytes);
      if (diagnostic_output_fill_byte >= 0) {
        if (input) {
          HiaiInputIoDiagnostic record;
          record.index = i; record.logical_bytes = tensor.bytes; record.allocated_bytes = tensor.Size();
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
  if (impl_->target.file_v2 && impl_->target.stage != "INPUT_BINDING") impl_->target.Stage("INPUT_BINDING");
  else impl_->target.stage = "INPUT_BINDING";
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
      record.allocated_bytes_before = tensor.Size();
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
      record.allocated_bytes = tensor.Size();
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
  if (impl_->target.file_v2) impl_->target.Stage("EXECUTED");
  else impl_->target.stage = "EXECUTED";
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
const std::string& HiaiStatefulExecutor::runtime_stage() const noexcept {
  return impl_->target.stage;
}
const HiaiIoDiagnostic& HiaiStatefulExecutor::io_diagnostic() const noexcept {
  return impl_->diagnostic;
}

struct HiaiOriginalTargetExecutor::Impl {
  std::shared_ptr<hiai::IBuiltModel> built_model;
  std::shared_ptr<hiai::IModelManager> manager;
  std::vector<std::shared_ptr<hiai::INDTensorBuffer>> input_ptrs, output_ptrs;
  std::vector<Tensor> inputs, outputs;
  std::vector<HiaiBoundRuntimeTensor> observed_inputs, observed_outputs;
  std::vector<bool> ready;
  std::vector<std::string> input_hashes;
  std::vector<const void*> output_buffers;
  std::string version, stage = "VALIDATE_ORIGINAL_TARGET_CONTRACT";
  HiaiIoDiagnostic diagnostic;
  HiaiOriginalTargetLifecycle lifecycle;
  bool outputs_available = false, poisoned = false;

  void Stage(const char* value) {
    stage = value;
    std::cerr << "HIAI original-target-v2 stage=" << stage << std::endl;
  }
  void RequireLive() const {
    if (lifecycle.closed || poisoned || !lifecycle.manager_initialized)
      throw HiaiRuntimeError(stage, "original target executor is closed, failed or not initialized");
  }
  void VerifyDisjointBuffers() const {
    struct Region { const void* object; std::uintptr_t begin, end; };
    std::vector<Region> regions;
    for (const auto* tensors : {&inputs, &outputs}) {
      for (const auto& tensor : *tensors) {
        const auto start = reinterpret_cast<std::uintptr_t>(tensor.Data());
        if (tensor.Size() != tensor.bytes || start > UINTPTR_MAX - tensor.bytes)
          throw std::runtime_error("original target allocation extent/address is invalid");
        const Region current{tensor.Object(), start, start + tensor.bytes};
        for (const auto& previous : regions)
          if (current.object == previous.object ||
              (current.begin < previous.end && previous.begin < current.end))
            throw std::runtime_error("original target live tensor buffers overlap");
        regions.push_back(current);
      }
    }
  }
  bool ObserveOutputs() {
    bool ok = output_ptrs.size() == outputs.size();
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      auto& d = diagnostic.outputs.at(i);
      d.checked_after_process = true;
      const auto p = i < output_ptrs.size() ? output_ptrs[i] : nullptr;
      d.allocated_bytes_after = p ? p->GetSize() : 0;
      d.object_same = p && p.get() == outputs[i].nd_value.get();
      const auto* buffer = p ? p->GetData() : nullptr;
      d.buffer_same = buffer && buffer == output_buffers.at(i);
      d.fill_count_available = false;
      ok = ok && d.object_same && d.buffer_same && d.allocated_bytes_after == outputs[i].bytes;
    }
    return ok;
  }
  void Cleanup(bool report_error) {
    if (lifecycle.closed) return;
    lifecycle.closed = true;
    outputs_available = false;
    std::string cleanup_error;
    if (manager && lifecycle.manager_init_attempted) {
      Stage("MODEL_MANAGER_DEINIT");
      lifecycle.manager_deinit_called = true;
      try { manager->DeInit(); }
      catch (const std::exception& e) { cleanup_error = e.what(); }
      catch (...) { cleanup_error = "unknown DeInit exception"; }
    }
    if (manager && lifecycle.weights_initialized) {
      Stage("EXTERNAL_WEIGHTS_DEINIT");
      lifecycle.weights_deinit_called = true;
      try {
        lifecycle.weights_deinit_status = static_cast<std::int64_t>(manager->DeinitWeights());
        if (lifecycle.weights_deinit_status != 0)
          cleanup_error += " DeinitWeights failed: " + std::to_string(lifecycle.weights_deinit_status);
      } catch (const std::exception& e) { cleanup_error += e.what(); }
      catch (...) { cleanup_error += " unknown DeinitWeights exception"; }
    }
    // Keep restored model and all tensors alive until BOTH cleanup APIs and the
    // manager destructor have finished, including construction/Run failures.
    manager.reset();
    built_model.reset();
    input_ptrs.clear(); output_ptrs.clear(); inputs.clear(); outputs.clear();
    Stage(cleanup_error.empty() ? "CLOSED" : "CLEANUP_FAILED");
    if (!cleanup_error.empty()) {
      if (report_error) throw HiaiRuntimeError(stage, cleanup_error);
      std::cerr << "HIAI original-target-v2 cleanup error=" << cleanup_error << std::endl;
    }
  }
  ~Impl() { Cleanup(false); }
};

HiaiOriginalTargetExecutor::HiaiOriginalTargetExecutor(
    const std::filesystem::path& model, const std::filesystem::path& weight_directory,
    const std::string& expected_version, const std::vector<HiaiBoundRuntimeTensor>& inputs,
    const std::vector<HiaiBoundRuntimeTensor>& outputs) : impl_(std::make_unique<Impl>()) {
  try {
    if (!model.is_absolute() || !std::filesystem::is_regular_file(model) ||
        !weight_directory.is_absolute() || !std::filesystem::is_directory(weight_directory) ||
        expected_version.empty() || inputs.size() != 89 || outputs.size() != 82)
      throw std::invalid_argument("original target requires an absolute OMC/weight directory, runtime version and89/82 bound descriptors");
    if (!detail::OptionalTensorName<hiai::NDTensorDesc>::available)
      throw std::invalid_argument("selected SDK does not expose original target runtime tensor names");
    const std::uint16_t endian = 1;
    if (*reinterpret_cast<const std::uint8_t*>(&endian) != 1)
      throw std::invalid_argument("original target requires little-endian input buffers");
    const auto enums = HiaiModelIoEnums();
    std::uint64_t total = 0;
    for (const auto* specs : {&inputs, &outputs}) {
      std::set<std::string> names;
      for (std::size_t i = 0; i < specs->size(); ++i) {
        const auto& spec = specs->at(i);
        const auto& d = spec.descriptor;
        const auto dtype = enums.dtypes.find(spec.dtype);
        if (d.index != i || d.tensor_name_status != "OBSERVED" || d.tensor_name.empty() ||
            d.tensor_name.size() > 4096 || d.tensor_name.find('\0') != std::string::npos ||
            !names.insert(d.tensor_name).second || d.shape.size() > 8 ||
            dtype == enums.dtypes.end() || dtype->second != d.raw_dtype ||
            !spec.bytes || spec.bytes > kStatefulTensorLimit)
          throw std::invalid_argument("invalid or ambiguous original target named dtype/extent contract");
        std::size_t bytes = spec.dtype == "INT8" ? 1 : spec.dtype == "INT64" ? 8 :
                            (spec.dtype == "INT16" || spec.dtype == "FLOAT16") ? 2 : 4;
        for (const auto dim : d.shape) {
          if (dim <= 0 || dim > INT32_MAX || bytes > kStatefulTensorLimit / static_cast<std::size_t>(dim))
            throw std::invalid_argument("original target descriptor has invalid/beyond-budget dimensions");
          bytes *= static_cast<std::size_t>(dim);
        }
        if (bytes != spec.bytes || total > kStatefulTotalLimit - spec.bytes)
          throw std::invalid_argument("original target descriptor byte extent or total IO budget differs");
        total += spec.bytes;
      }
    }
    impl_->Stage("RUNTIME_IDENTITY");
    const auto runtime = ProbeHiai();
    impl_->version = runtime.version;
    if (runtime.version != expected_version || !runtime.npu_supported)
      throw std::runtime_error("original target runtime version/NPU support differs");
    impl_->Stage("CREATE_BUILT_MODEL");
    impl_->built_model = hiai::CreateBuiltModel();
    if (!impl_->built_model) throw std::runtime_error("CreateBuiltModel returned null");
    impl_->Stage("RESTORE_FROM_FILE");
    auto status = impl_->built_model->RestoreFromFile(model.c_str());
    if (status != 0) throw std::runtime_error("RestoreFromFile failed: " + std::to_string(status));
    impl_->Stage("CREATE_MODEL_MANAGER");
    impl_->manager = hiai::CreateModelManager();
    if (!impl_->manager) throw std::runtime_error("CreateModelManager returned null");
    // The selected private headers declare these calls. They do not validate
    // ordering or external-weight loading on the device. Exactly one explicit
    // experiment is permitted here; no automatic alternate order/load path.
    impl_->Stage("EXPERIMENTAL_INIT_WEIGHTS_BEFORE_INIT");
    impl_->lifecycle.weights_init_attempted = true;
    status = impl_->manager->InitWeights(weight_directory.string());
    if (status != 0) throw std::runtime_error("InitWeights failed: " + std::to_string(status));
    impl_->lifecycle.weights_initialized = true;
    impl_->Stage("MODEL_MANAGER_INIT");
    hiai::ModelInitOptions options;
    options.perfMode = hiai::PerfMode::HIGH;
    impl_->lifecycle.manager_init_attempted = true;
    status = impl_->manager->Init(options, impl_->built_model, nullptr);
    if (status != 0) throw std::runtime_error("IModelManager::Init failed: " + std::to_string(status));
    impl_->lifecycle.manager_initialized = true;
    impl_->Stage("TENSOR_DESCRIPTORS");
    const auto in_desc = impl_->built_model->GetInputTensorDescs();
    const auto out_desc = impl_->built_model->GetOutputTensorDescs();
    if (in_desc.size() != inputs.size() || out_desc.size() != outputs.size())
      throw std::runtime_error("original target runtime descriptor count differs from89/82");
    // Verify EVERY descriptor before any tensor allocation. Pass the exact
    // returned descriptor to the allocator, including its physical format.
    for (const bool input : {true, false}) {
      const auto& descs = input ? in_desc : out_desc;
      const auto& specs = input ? inputs : outputs;
      auto& observed = input ? impl_->observed_inputs : impl_->observed_outputs;
      for (std::size_t i = 0; i < specs.size(); ++i) {
        const auto& actual = descs[i];
        const auto& expected = specs[i].descriptor;
        HiaiBoundRuntimeTensor row = specs[i];
        row.descriptor.shape.assign(actual.dims.begin(), actual.dims.end());
        row.descriptor.raw_dtype = static_cast<int>(actual.dataType);
        row.descriptor.raw_format = static_cast<int>(actual.format);
        row.descriptor.tensor_name = detail::OptionalTensorName<hiai::NDTensorDesc>::Read(actual);
        if (row.descriptor.shape != expected.shape || row.descriptor.raw_dtype != expected.raw_dtype ||
            row.descriptor.raw_format != expected.raw_format || row.descriptor.tensor_name != expected.tensor_name)
          throw std::runtime_error("original target " + std::string(input ? "input[" : "output[") +
                                   std::to_string(i) + "] runtime name/dtype/format/dimensions differ");
        observed.push_back(std::move(row));
      }
    }
    impl_->diagnostic.output_fill_byte = 0;
    for (const bool input : {true, false}) {
      impl_->Stage(input ? "INPUT_TENSOR_ALLOCATION" : "OUTPUT_TENSOR_ALLOCATION");
      const auto& descs = input ? in_desc : out_desc;
      const auto& specs = input ? inputs : outputs;
      auto& tensors = input ? impl_->inputs : impl_->outputs;
      auto& ptrs = input ? impl_->input_ptrs : impl_->output_ptrs;
      for (std::size_t i = 0; i < specs.size(); ++i) {
        auto ptr = hiai::CreateNDTensorBuffer(descs[i]);
        if (!ptr || !ptr->GetData() || ptr->GetSize() != specs[i].bytes)
          throw std::runtime_error("original target tensor allocation differs from exact bound byte extent");
        Tensor tensor{nullptr, specs[i].bytes, ptr};
        tensors.push_back(tensor); ptrs.push_back(std::move(ptr));
        // Hash equality cannot detect aliasing between all-zero states. Reject
        // overlapping objects/storage before even the initialization write.
        impl_->VerifyDisjointBuffers();
        std::memset(tensor.Data(), 0, tensor.bytes);
        if (input) {
          HiaiInputIoDiagnostic row;
          row.index = i; row.logical_bytes = row.allocated_bytes = tensor.bytes;
          impl_->diagnostic.inputs.push_back(row);
        } else {
          HiaiOutputIoDiagnostic row;
          row.index = i; row.logical_bytes = tensor.bytes;
          impl_->diagnostic.outputs.push_back(row);
        }
      }
    }
    impl_->ready.assign(inputs.size(), false);
    impl_->input_hashes.resize(inputs.size());
    impl_->Stage("READY");
  } catch (const HiaiRuntimeError&) { throw; }
  catch (const std::exception& error) { throw HiaiRuntimeError(impl_->stage, error.what()); }
}
HiaiOriginalTargetExecutor::~HiaiOriginalTargetExecutor() = default;
void HiaiOriginalTargetExecutor::SetInput(std::size_t index, const std::vector<std::uint8_t>& bytes) {
  impl_->RequireLive();
  impl_->Stage("INPUT_BINDING");
  impl_->outputs_available = false;
  auto& tensor = impl_->inputs.at(index);
  impl_->ready.at(index) = false;
  if (bytes.size() != tensor.bytes || tensor.Size() != tensor.bytes)
    throw HiaiRuntimeError(impl_->stage, "original target input byte extent differs");
  std::memcpy(tensor.Data(), bytes.data(), bytes.size());
  impl_->input_hashes[index] = Sha256(std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size()));
  impl_->diagnostic.inputs[index].readback_sha256.clear();
  impl_->diagnostic.inputs[index].readback_matches_source = false;
  impl_->ready[index] = true;
}
void HiaiOriginalTargetExecutor::Execute() {
  impl_->RequireLive();
  impl_->outputs_available = false;
  impl_->diagnostic.process_called = false;
  if (!std::all_of(impl_->ready.begin(), impl_->ready.end(), [](bool value) { return value; }))
    throw HiaiRuntimeError(impl_->stage, "all89 original target inputs must be supplied for every Run");
  std::fill(impl_->ready.begin(), impl_->ready.end(), false);
  try {
    impl_->Stage("INPUT_READBACK_AND_OUTPUT_BINDING");
    impl_->VerifyDisjointBuffers();
    impl_->output_buffers.clear();
    for (std::size_t i = 0; i < impl_->outputs.size(); ++i) {
      auto& tensor = impl_->outputs[i];
      auto& row = impl_->diagnostic.outputs[i];
      row.checked_after_process = row.object_same = row.buffer_same = false;
      row.allocated_bytes_before = tensor.Size();
      if (tensor.Size() != tensor.bytes || impl_->output_ptrs.at(i).get() != tensor.nd_value.get())
        throw std::runtime_error("original target output allocation changed before Run");
      impl_->output_buffers.push_back(tensor.Data());
      std::memset(tensor.Data(), 0, tensor.bytes);
    }
    for (std::size_t i = 0; i < impl_->inputs.size(); ++i) {
      const auto& tensor = impl_->inputs[i];
      auto& row = impl_->diagnostic.inputs[i];
      row.allocated_bytes = tensor.Size();
      if (tensor.Size() != tensor.bytes || impl_->input_ptrs.at(i).get() != tensor.nd_value.get())
        throw std::runtime_error("original target input allocation changed before Run");
      row.readback_sha256 = Sha256(std::string_view(static_cast<const char*>(tensor.Data()), tensor.bytes));
      row.readback_matches_source = row.readback_sha256 == impl_->input_hashes[i];
      if (!row.readback_matches_source) throw std::runtime_error("original target input readback differs before Run");
    }
    impl_->Stage("MODEL_MANAGER_RUN");
    impl_->diagnostic.process_called = true;
    ++impl_->lifecycle.run_attempts;
    const auto status = impl_->manager->Run(impl_->input_ptrs, impl_->output_ptrs);
    impl_->Stage("OUTPUT_VALIDATION");
    const bool objects_unchanged = impl_->ObserveOutputs();
    if (status != 0) {
      impl_->Stage("MODEL_MANAGER_RUN");
      throw std::runtime_error("original target Run failed: " + std::to_string(status));
    }
    if (!objects_unchanged) throw std::runtime_error("original target output object/buffer/extent changed during Run");
    impl_->VerifyDisjointBuffers();
    ++impl_->lifecycle.successful_runs;
    impl_->outputs_available = true;
    impl_->Stage("EXECUTED");
  } catch (const std::exception& error) {
    impl_->poisoned = true;
    throw HiaiRuntimeError(impl_->stage, error.what());
  } catch (...) {
    impl_->poisoned = true;
    throw HiaiRuntimeError(impl_->stage, "unknown exception during original target Run; executor cannot be reused");
  }
}
std::vector<std::uint8_t> HiaiOriginalTargetExecutor::CopyOutput(std::size_t index) const {
  impl_->RequireLive();
  if (!impl_->outputs_available)
    throw HiaiRuntimeError(impl_->stage, "original target outputs unavailable before successful Run or after new input binding");
  try { impl_->VerifyDisjointBuffers(); }
  catch (const std::exception& error) {
    impl_->poisoned = true;
    impl_->outputs_available = false;
    throw HiaiRuntimeError(impl_->stage, error.what());
  }
  const auto& tensor = impl_->outputs.at(index);
  if (impl_->output_ptrs.at(index).get() != tensor.nd_value.get() || tensor.Size() != tensor.bytes ||
      tensor.Buffer() != impl_->output_buffers.at(index)) {
    impl_->poisoned = true;
    impl_->outputs_available = false;
    throw HiaiRuntimeError(impl_->stage, "original target output allocation changed after Run");
  }
  const auto* data = static_cast<const std::uint8_t*>(tensor.Data());
  return {data, data + tensor.bytes};
}
void HiaiOriginalTargetExecutor::Close() { impl_->Cleanup(true); }
const std::string& HiaiOriginalTargetExecutor::runtime_version() const noexcept { return impl_->version; }
const std::string& HiaiOriginalTargetExecutor::runtime_stage() const noexcept { return impl_->stage; }
const HiaiIoDiagnostic& HiaiOriginalTargetExecutor::io_diagnostic() const noexcept { return impl_->diagnostic; }
const HiaiOriginalTargetLifecycle& HiaiOriginalTargetExecutor::lifecycle() const noexcept { return impl_->lifecycle; }
const std::vector<HiaiBoundRuntimeTensor>& HiaiOriginalTargetExecutor::observed_inputs() const noexcept { return impl_->observed_inputs; }
const std::vector<HiaiBoundRuntimeTensor>& HiaiOriginalTargetExecutor::observed_outputs() const noexcept { return impl_->observed_outputs; }

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
      impl_->target.outputs[1].Size() > kFeatureCaptureLimit)
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
    if (features.Size() > kFeatureCaptureLimit)
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
