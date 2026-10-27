#include <HiAiModelManagerService.h>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace fs = std::filesystem;
constexpr std::uint64_t kModelLimit = 64ULL << 20;
constexpr std::uint64_t kTensorLimit = 32ULL << 20;
constexpr std::uint64_t kTotalTensorLimit = 256ULL << 20;
std::string stage = "arguments", runtime_version;

void Require(bool value, const std::string& message) {
  if (!value) throw std::runtime_error(message);
}
void Check(hiai::AIStatus status, const char* operation) {
  Require(status == hiai::AI_SUCCESS,
          std::string(operation) + " failed: " + std::to_string(status));
}
std::string Json(const std::string& value) {
  std::ostringstream out;
  out << '"';
  constexpr char hex[] = "0123456789abcdef";
  for (unsigned char c : value) {
    if (c == '\\' || c == '"') out << '\\' << c;
    else if (c < 32) out << "\\u00" << hex[c >> 4] << hex[c & 15];
    else out << c;
  }
  return out.str() + '"';
}
std::vector<std::string> Split(const std::string& line, char separator) {
  std::vector<std::string> parts;
  std::size_t begin = 0;
  for (;;) {
    const auto end = line.find(separator, begin);
    parts.push_back(line.substr(begin, end - begin));
    if (end == std::string::npos) return parts;
    begin = end + 1;
  }
}
std::vector<char> Read(const fs::path& path, std::uint64_t limit,
                       std::uint64_t expected = 0) {
  Require(fs::is_regular_file(path), "not a regular file: " + path.string());
  const auto bytes = fs::file_size(path);
  Require(bytes > 0 && bytes <= limit, "file outside size limit: " + path.string());
  Require(!expected || bytes == expected, "file byte count differs from tensor: " + path.string());
  std::ifstream stream(path, std::ios::binary);
  Require(bool(stream), "cannot open file: " + path.string());
  std::vector<char> result(static_cast<std::size_t>(bytes));
  stream.read(result.data(), static_cast<std::streamsize>(bytes));
  Require(stream.gcount() == static_cast<std::streamsize>(bytes) && !stream.bad(),
          "short read: " + path.string());
  Require(stream.peek() == std::char_traits<char>::eof(), "file changed while reading: " + path.string());
  return result;
}
struct Spec {
  bool input;
  hiai::HIAI_DataType dtype;
  std::vector<std::uint32_t> shape;
  std::uint64_t bytes;
  fs::path file;
};
std::string Shape(const std::vector<std::uint32_t>& dims) {
  std::ostringstream out;
  out << '[';
  for (std::size_t i = 0; i < dims.size(); ++i) {
    if (i) out << ',';
    out << dims[i];
  }
  return out.str() + ']';
}
const char* DtypeName(hiai::HIAI_DataType dtype) {
  switch (dtype) {
    case hiai::HIAI_DATATYPE_FLOAT16: return "float16";
    case hiai::HIAI_DATATYPE_FLOAT32: return "float32";
    case hiai::HIAI_DATATYPE_INT32: return "int32";
    default: return "unknown";
  }
}
struct TensorEvidence {
  bool input;
  std::size_t index;
  bool declared = false, runtime = false, allocated = false;
  std::vector<std::uint32_t> logical_shape, runtime_shape;
  std::string logical_dtype;
  std::uint64_t logical_bytes = 0, allocation_bytes = 0;
  int runtime_format = 0;
};
// Retain bounded metadata across Probe destruction so failures report the same
// observed contracts as successful runs. No buffer contents or addresses here.
std::vector<TensorEvidence> io_tensors;
TensorEvidence& TensorRecord(bool input, std::size_t index) {
  for (auto& record : io_tensors)
    if (record.input == input && record.index == index) return record;
  Require(io_tensors.size() < 256, "runtime IO metadata exceeds tensor-count limit");
  TensorEvidence record{};
  record.input = input;
  record.index = index;
  io_tensors.push_back(std::move(record));
  return io_tensors.back();
}
void RecordDeclared(const std::vector<Spec>& specs) {
  std::size_t input_index = 0, output_index = 0;
  for (const auto& spec : specs) {
    auto& record = TensorRecord(spec.input, spec.input ? input_index++ : output_index++);
    record.declared = true;
    record.logical_shape = spec.shape;
    record.logical_dtype = DtypeName(spec.dtype);
    record.logical_bytes = spec.bytes;
  }
}
void RecordRuntime(bool input, const std::vector<hiai::TensorDimension>& dims) {
  for (std::size_t i = 0; i < dims.size(); ++i) {
    auto& record = TensorRecord(input, i);
    record.runtime = true;
    record.runtime_shape = dims[i].GetDims();
    record.runtime_format = static_cast<int>(dims[i].GetFormat());
  }
}
std::string IoEvidence() {
  std::ostringstream out;
  out << '[';
  bool first = true;
  for (const auto& record : io_tensors) {
    if (!first) out << ',';
    first = false;
    out << "{\"direction\":" << Json(record.input ? "input" : "output")
        << ",\"index\":" << record.index
        << ",\"logical_shape\":" << (record.declared ? Shape(record.logical_shape) : "null")
        << ",\"logical_dtype\":" << (record.declared ? Json(record.logical_dtype) : "null")
        << ",\"logical_bytes\":" << (record.declared ? std::to_string(record.logical_bytes) : "null")
        << ",\"runtime_shape\":" << (record.runtime ? Shape(record.runtime_shape) : "null")
        << ",\"runtime_format\":" << (record.runtime ? std::to_string(record.runtime_format) : "null")
        << ",\"runtime_dtype\":\"unavailable\""
        << ",\"allocation_bytes\":" << (record.allocated ? std::to_string(record.allocation_bytes) : "null")
        << '}';
  }
  return out.str() + ']';
}
void DumpDimensions(const char* direction, const std::vector<hiai::TensorDimension>& dims) {
  for (std::size_t i = 0; i < dims.size(); ++i)
    std::cerr << "qwen35_hiai_op_probe: model IO " << direction << '[' << i
              << "]: shape=" << Shape(dims[i].GetDims())
              << ", format=" << static_cast<int>(dims[i].GetFormat())
              << ", dtype=unavailable\n";
}
std::string DimensionMismatch(const Spec& spec, std::size_t index,
                              const hiai::TensorDimension& dim) {
  std::ostringstream out;
  out << "OMC tensor shape/format differs from IO TSV: "
      << (spec.input ? "input" : "output") << '[' << index << ']'
      << "; declared_dtype=" << DtypeName(spec.dtype) << '(' << static_cast<int>(spec.dtype) << ')'
      << ", declared_shape=" << Shape(spec.shape)
      << ", declared_format=ND(" << static_cast<int>(hiai::Format::ND) << ')'
      << "; actual_shape=" << Shape(dim.GetDims())
      << ", actual_format=" << static_cast<int>(dim.GetFormat())
      << ", actual_dtype=unavailable";
  return out.str();
}
bool IsTrailingOnePadding(const std::vector<std::uint32_t>& declared,
                          const std::vector<std::uint32_t>& actual) {
  // This runtime reports some ND rank1..3 IO as rank4 by appending ones.
  // Preserve every original axis; equal element counts alone are insufficient.
  if (declared.empty() || declared.size() > 3 || actual.size() != 4) return false;
  for (std::size_t i = 0; i < actual.size(); ++i)
    if (actual[i] != (i < declared.size() ? declared[i] : 1)) return false;
  return true;
}
bool IsLeadingOnePadding(const std::vector<std::uint32_t>& declared,
                         const std::vector<std::uint32_t>& actual) {
  // Some rank3 ND IO is reported as [1,a,b,c]. Keep all original axes
  // in order; do not generalize this to squeezing singleton dimensions.
  if (declared.size() != 3 || actual.size() != 4 || actual[0] != 1) return false;
  for (std::size_t i = 0; i < declared.size(); ++i)
    if (actual[i + 1] != declared[i]) return false;
  return true;
}
std::vector<Spec> ReadSpecs(const fs::path& path) {
  const auto text = Read(path, 64 << 10);
  std::istringstream lines(std::string(text.begin(), text.end()));
  const auto parent = fs::canonical(path).parent_path();
  std::string line;
  std::vector<Spec> result;
  std::set<std::string> output_names;
  std::size_t inputs = 0, outputs = 0;
  std::uint64_t total = 0;
  while (std::getline(lines, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const auto fields = Split(line, '\t');
    Require(fields.size() == 4, "IO TSV requires four tab-separated fields per line");
    Spec spec {};
    Require(fields[0] == "input" || fields[0] == "output", "invalid IO direction");
    spec.input = fields[0] == "input";
    if (fields[1] == "float16") {
      spec.dtype = hiai::HIAI_DATATYPE_FLOAT16; spec.bytes = 2;
    } else if (fields[1] == "float32") {
      spec.dtype = hiai::HIAI_DATATYPE_FLOAT32; spec.bytes = 4;
    } else if (fields[1] == "int32") {
      spec.dtype = hiai::HIAI_DATATYPE_INT32; spec.bytes = 4;
    } else throw std::runtime_error("IO dtype must be float16, float32 or int32");
    for (const auto& field : Split(fields[2], ',')) {
      Require(!field.empty() && field.find_first_not_of("0123456789") == std::string::npos,
              "shape dimensions must be positive decimal integers");
      const auto dim = std::stoull(field);
      Require(dim > 0 && dim <= UINT32_MAX && spec.bytes <= kTensorLimit / dim,
              "tensor exceeds 32 MiB or has invalid shape");
      spec.shape.push_back(static_cast<std::uint32_t>(dim));
      spec.bytes *= dim;
    }
    Require(spec.shape.size() <= 8, "tensor rank exceeds probe limit of eight");
    Require(!fields[3].empty(), "tensor filename must not be empty");
    const fs::path relative(fields[3]);
    Require(!relative.is_absolute(), "tensor filenames must be relative");
    for (const auto& component : relative)
      Require(component != "..", "tensor filename must not traverse parent directories");
    if (spec.input) {
      ++inputs;
      spec.file = fs::canonical(parent / relative);
      const auto confined = spec.file.lexically_relative(parent);
      Require(!confined.empty() && *confined.begin() != "..", "input file escapes IO directory");
    } else {
      ++outputs;
      Require(relative == relative.filename() && relative != "." && relative != "..",
              "output filename must be a basename");
      Require(output_names.insert(fields[3]).second, "duplicate output filename");
      spec.file = relative;
    }
    total += spec.bytes;
    Require(total <= kTotalTensorLimit && result.size() < 128,
            "IO specification exceeds total memory or tensor-count limit");
    result.push_back(std::move(spec));
  }
  Require(inputs > 0 && outputs > 0, "at least one input and output are required");
  return result;
}

struct Probe {
  // Destruction is reversed: the client must die before model/IO storage.
  std::vector<char> model_bytes;
  std::shared_ptr<hiai::AiModelDescription> description;
  std::vector<std::shared_ptr<hiai::AiTensor>> inputs, outputs;
  hiai::AiModelMngerClient client;
  bool initialized = false;
  const std::string name = "qwen35_operator_probe";
  ~Probe() {
    if (initialized) static_cast<void>(client.UnLoadModel());
  }
  void Run(const fs::path& model, const std::vector<Spec>& specs, const fs::path& output) {
    stage = "model-buffer";
    model_bytes = Read(model, kModelLimit);
    stage = "init";
    Check(client.Init(nullptr), "Init(sync)");
    initialized = true;
    const auto* version = client.GetVersion();
    Require(version && *version, "GetVersion returned empty");
    runtime_version = version;
    description = std::make_shared<hiai::AiModelDescription>(
        name, hiai::AiModelDescription_Frequency_HIGH, hiai::HIAI_FRAMEWORK_NONE,
        hiai::HIAI_MODELTYPE_OFFLINE, hiai::AiModelDescription_DeviceType_NPU);
    stage = "load";
    Check(description->SetModelPath(fs::absolute(model).string()), "SetModelPath");
    Check(description->SetModelBuffer(model_bytes.data(), static_cast<std::uint32_t>(model_bytes.size())),
          "SetModelBuffer");
    Require(description->GetModelBuffer() && description->GetModelNetSize() == model_bytes.size(),
            "model buffer descriptor is empty or truncated");
    std::vector<std::shared_ptr<hiai::AiModelDescription>> descriptions{description};
    Check(client.Load(descriptions), "Load");
    stage = "io";
    std::vector<hiai::TensorDimension> input_dims, output_dims;
    Check(client.GetModelIOTensorDim(name, input_dims, output_dims), "GetModelIOTensorDim");
    RecordRuntime(true, input_dims);
    RecordRuntime(false, output_dims);
    // Keep all actual descriptors even when the first TSV comparison fails.
    // This legacy API does not expose the model's IO dtypes.
    std::cerr << "qwen35_hiai_op_probe: model IO: inputs=" << input_dims.size()
              << ", outputs=" << output_dims.size() << '\n';
    DumpDimensions("input", input_dims);
    DumpDimensions("output", output_dims);
    std::size_t input_index = 0, output_index = 0;
    for (const auto& spec : specs) {
      const auto& dims = spec.input ? input_dims : output_dims;
      auto& index = spec.input ? input_index : output_index;
      Require(index < dims.size(), "IO TSV has more tensors than OMC");
      const auto& dim = dims[index];
      const auto actual_shape = dim.GetDims();
      const bool trailing = IsTrailingOnePadding(spec.shape, actual_shape);
      const bool leading = IsLeadingOnePadding(spec.shape, actual_shape);
      const bool padded = trailing || leading;
      Require(dim.GetFormat() == hiai::Format::ND && (actual_shape == spec.shape || padded),
              DimensionMismatch(spec, index, dim));
      if (padded)
        std::cerr << "qwen35_hiai_op_probe: accepted ND "
                  << (trailing ? "trailing-one" : "leading-one") << " padding for "
                  << (spec.input ? "input" : "output") << '[' << index << ']'
                  << ": declared_shape=" << Shape(spec.shape)
                  << ", actual_shape=" << Shape(actual_shape)
                  << "; using runtime descriptor\n";
      ++index;
      auto tensor = std::make_shared<hiai::AiTensor>();
      // The legacy API exposes shape/format, not dtype introspection. The
      // host manifest binds dtype declarations to the model and input hashes.
      Check(tensor->Init(&dim, spec.dtype), "AiTensor::Init");
      auto& evidence = TensorRecord(spec.input, index - 1);
      evidence.allocation_bytes = tensor->GetSize();
      evidence.allocated = true;
      Require(tensor->GetBuffer() && tensor->GetSize() >= spec.bytes,
              "tensor allocation smaller than declared IO bytes");
      if (spec.input) {
        const auto bytes = Read(spec.file, kTensorLimit, spec.bytes);
        std::memcpy(tensor->GetBuffer(), bytes.data(), bytes.size());
        inputs.push_back(std::move(tensor));
      } else outputs.push_back(std::move(tensor));
    }
    Require(input_index == input_dims.size() && output_index == output_dims.size(),
            "IO TSV tensor count differs from OMC");
    stage = "process";
    hiai::AiContext context;
    context.AddPara("model_name", name);
    std::int32_t stamp = 0;
    Check(client.Process(context, inputs, outputs, 30000, stamp), "Process(sync)");
    stage = "output";
    Require(fs::create_directory(output), "output directory must not already exist");
    output_index = 0;
    for (const auto& spec : specs) {
      if (spec.input) continue;
      const auto& tensor = outputs.at(output_index++);
      Require(tensor->GetBuffer() && tensor->GetSize() >= spec.bytes, "output buffer became invalid");
      std::ofstream file(output / spec.file, std::ios::binary | std::ios::out);
      Require(bool(file), "cannot create output tensor");
      file.write(static_cast<const char*>(tensor->GetBuffer()), static_cast<std::streamsize>(spec.bytes));
      file.close();
      Require(bool(file) && fs::file_size(output / spec.file) == spec.bytes, "output tensor write failed");
    }
    stage = "unload";
    Check(client.UnLoadModel(), "UnLoadModel");
    initialized = false;
  }
};
}  // namespace

int main(int argc, char** argv) {
  try {
    fs::path model, io, output;
    std::set<std::string> seen;
    for (int i = 1; i < argc; i += 2) {
      const std::string key(argv[i]);
      Require(i + 1 < argc && seen.insert(key).second, "arguments require unique key/value pairs");
      if (key == "--model") model = argv[i + 1];
      else if (key == "--io") io = argv[i + 1];
      else if (key == "--output") output = argv[i + 1];
      else throw std::runtime_error("unknown argument: " + key);
    }
    Require(!model.empty() && !io.empty() && !output.empty(),
            "usage: qwen35_hiai_op_probe --model MODEL --io IO_TSV --output NEW_DIRECTORY");
    Require(!fs::exists(output), "output directory must not already exist");
    stage = "io-spec";
    const auto specs = ReadSpecs(io);
    RecordDeclared(specs);
    {
      Probe probe;
      probe.Run(model, specs, output);
    }
    std::cout << "{\"status\":\"EXECUTED\",\"runtime_version\":" << Json(runtime_version)
              << ",\"placement\":\"UNVERIFIED\",\"io_tensors\":" << IoEvidence() << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "qwen35_hiai_op_probe: stage=" << stage << ": " << error.what() << '\n';
    std::cout << "{\"status\":\"FAILED\",\"stage\":" << Json(stage)
              << ",\"runtime_version\":" << Json(runtime_version)
              << ",\"error\":" << Json(error.what())
              << ",\"io_tensors\":" << IoEvidence() << "}\n";
    return 1;
  }
}
