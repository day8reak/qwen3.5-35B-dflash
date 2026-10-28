#pragma once
// Minimal host test double; not an SDK ABI declaration.
#include <model/built_model.h>
namespace hiai {
enum class PerfMode { UNSET=0, LOW=1, MIDDLE, HIGH, EXTREME_HIGH, HIGH_COMPUTE_UNIT=103 };
struct ModelInitOptions { PerfMode perfMode=PerfMode::MIDDLE; };
class IModelManagerListener { public: virtual ~IModelManagerListener()=default; };
class IModelManager {
 public:
  virtual ~IModelManager()=default;
  virtual Status Init(const ModelInitOptions&, const std::shared_ptr<IBuiltModel>&,
                      const std::shared_ptr<IModelManagerListener>&)=0;
  virtual Status Run(const std::vector<std::shared_ptr<INDTensorBuffer>>&,
                     std::vector<std::shared_ptr<INDTensorBuffer>>&)=0;
  virtual void DeInit()=0;
};
std::shared_ptr<IModelManager> CreateModelManager();
}
