#pragma once
// Minimal host test double; not an SDK ABI declaration.
#include <tensor/nd_tensor_buffer.h>
namespace hiai {
using Status=std::uint32_t;
class IBuiltModel {
 public:
  virtual ~IBuiltModel()=default;
  virtual Status RestoreFromFile(const char*)=0;
  virtual std::string GetName() const=0;
  virtual void SetName(const std::string&)=0;
  virtual std::vector<NDTensorDesc> GetInputTensorDescs() const=0;
  virtual std::vector<NDTensorDesc> GetOutputTensorDescs() const=0;
};
std::shared_ptr<IBuiltModel> CreateBuiltModel();
}
