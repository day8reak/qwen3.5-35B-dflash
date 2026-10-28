#pragma once
// Minimal host test double; production includes the selected SDK headers.
#include <HiAiModelManagerService.h>
#include <cstddef>
namespace hiai {
enum class DataType { UINT8=0, FLOAT32, FLOAT16, INT32, INT8, INT16, BOOL, INT64, UINT32, DOUBLE, RESERVED=255 };
struct NDTensorDesc {
  std::vector<std::int32_t> dims;
  DataType dataType=DataType::FLOAT32;
  Format format=Format::NCHW;
};
class INDTensorBuffer {
 public:
  virtual ~INDTensorBuffer()=default;
  virtual void* GetData()=0;
  virtual std::size_t GetSize() const=0;
  virtual const NDTensorDesc& GetTensorDesc() const=0;
};
std::shared_ptr<INDTensorBuffer> CreateNDTensorBuffer(const NDTensorDesc&);
}
