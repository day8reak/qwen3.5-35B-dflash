#pragma once
// Test double, not a redistributed SDK. Production compiles against real DDK.
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
namespace hiai {
using AIStatus = std::int32_t;
constexpr AIStatus AI_SUCCESS=0, AI_FAILED=1;
enum HIAI_DataType { HIAI_DATATYPE_FLOAT32=1, HIAI_DATATYPE_FLOAT16=2,
                     HIAI_DATATYPE_INT32=3, HIAI_DATATYPE_INT8=4, HIAI_DATATYPE_INT16=5, HIAI_DATATYPE_INT64=7 };
enum class Format { NCHW=0, NHWC=1, ND=2 };
constexpr int AiModelDescription_Frequency_HIGH=3, HIAI_FRAMEWORK_NONE=0,
              HIAI_MODELTYPE_OFFLINE=1, AiModelDescription_DeviceType_NPU=0;
class TensorDimension {
 public:
  explicit TensorDimension(std::vector<std::uint32_t> dims, Format format=Format::ND)
    : dims_(std::move(dims)), format_(format) {}
  std::vector<std::uint32_t> GetDims() const { return dims_; }
  Format GetFormat() const { return format_; }
 private:
  std::vector<std::uint32_t> dims_;
  Format format_;
};
class AiTensor {
 public:
  AiTensor();
  ~AiTensor();
  AIStatus Init(const TensorDimension*, HIAI_DataType);
  void* GetBuffer() const;
  std::uint32_t GetSize() const;
  void ReplaceBufferForTest();
 private:
  std::vector<std::int32_t> data_;
  std::uint32_t bytes_=0;
  mutable unsigned buffer_reads_=0;
  int ordinal_=0;
};
class AiModelDescription {
 public:
  AiModelDescription(const std::string&, int, int, int, int);
  AIStatus SetModelPath(const std::string& path);
  const std::string& GetModelPath() const { return path; }
  AIStatus SetModelBuffer(const void*, std::uint32_t);
  void* GetModelBuffer() const;
  std::uint32_t GetModelNetSize() const;
  std::string name, path;
 private:
  const void* model_buffer_=nullptr;
  std::uint32_t model_bytes_=0;
};
class AiContext {
 public:
  void AddPara(const std::string& key, const std::string& value);
  std::string model;
};
class AiModelManagerClientListener {};
class AiModelMngerClient {
 public:
  AiModelMngerClient();
  ~AiModelMngerClient();
  AIStatus Init(std::shared_ptr<AiModelManagerClientListener>);
  char* GetVersion();
  bool CheckNPUSupport();
  AIStatus CheckModelCompatibility(AiModelDescription&, bool&);
  AIStatus Load(std::vector<std::shared_ptr<AiModelDescription>>&);
  AIStatus GetModelIOTensorDim(const std::string&, std::vector<TensorDimension>&, std::vector<TensorDimension>&);
  AIStatus Process(AiContext&, std::vector<std::shared_ptr<AiTensor>>&,
                   std::vector<std::shared_ptr<AiTensor>>&, std::uint32_t, std::int32_t&);
  AIStatus UnLoadModel();
 private:
  bool loaded_=false,load_called_=false;
  std::size_t runs_=0;
  std::string name_;
  const void* model_buffer_=nullptr;
  std::uint32_t model_bytes_=0;
  std::vector<std::uint8_t> file_model_bytes_;
};
}
