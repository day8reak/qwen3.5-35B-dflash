#pragma once
#include "qwen35_dflash/generation.hpp"
#include <cstdint>
#include <filesystem>
#include <memory>
#include <map>
#include <string>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace qwen35::dflash {
struct HiaiProbe {
  std::string version;
  bool npu_supported;
};
HiaiProbe ProbeHiai();

struct HiaiDescriptionSnapshot {
  std::string operation, model_path;
  std::int64_t setter_status = 0;
  bool buffer_present = false, buffer_matches_input = false;
  std::uint32_t model_bytes = 0;
};
struct HiaiDescriptionOrder {
  std::string order;
  std::vector<HiaiDescriptionSnapshot> steps;
};
struct HiaiDescriptionProbe {
  HiaiProbe runtime;
  std::string model_path, model_sha256;
  std::uint32_t model_bytes = 0;
  std::vector<HiaiDescriptionOrder> observations;
};
// Observes descriptor setters only. Never loads or executes the model.
HiaiDescriptionProbe ProbeHiaiModelDescription(const std::filesystem::path& model);

struct HiaiModelIoTensor {
  std::size_t index = 0;
  std::vector<std::int64_t> shape;
  int raw_dtype = 0, raw_format = 0;
  std::string tensor_name, tensor_name_status = "FIELD_NOT_AVAILABLE";
};
struct HiaiModelIoEnumTable {
  std::map<std::string, int> dtypes, formats;
};
namespace detail {
// Read only the descriptor's actual public name field when this SDK exposes it.
// An absent/empty name never establishes a semantic input or state identity.
template <typename Descriptor, typename = void>
struct OptionalTensorName {
  static constexpr bool available = false;
  static std::string Read(const Descriptor&) { return {}; }
};
template <typename Descriptor>
struct OptionalTensorName<Descriptor,
    std::void_t<decltype(std::string(std::declval<const Descriptor&>().tensorName))>> {
  static constexpr bool available = true;
  static std::string Read(const Descriptor& value) { return std::string(value.tensorName); }
};
template <typename Enum, typename = void>
struct OptionalInt8Enum {
  static constexpr bool available = false;
  static void Append(std::map<std::string, int>&) {}
};
template <typename Enum>
struct OptionalInt8Enum<Enum, std::void_t<decltype(Enum::INT8)>> {
  static constexpr bool available = true;
  static void Append(std::map<std::string, int>& values) {
    values.emplace("INT8", static_cast<int>(Enum::INT8));
  }
};
}  // namespace detail
// Existing adapter constants plus INT8 only when that selected enum member
// exists at compile time. Unknown raw IDs remain uninterpreted; no ID is guessed.
HiaiModelIoEnumTable HiaiModelIoEnums();
struct HiaiModelIoProbe {
  HiaiProbe runtime;
  std::string model_path, model_sha256;
  std::uint64_t model_bytes = 0;
  bool tensor_name_field_available = false;
  std::vector<HiaiModelIoTensor> inputs, outputs;
};
// Restores file-v2 metadata, observes ordered descriptors and releases it.
// Never creates/initializes a file-v2 IModelManager, requests tensor buffers or calls
// Process/Run. This bounded experiment does not establish external-weight
// initialization or equal-shaped input semantics. Names are raw SDK observations
// only when the selected descriptor exposes a nonempty tensorName member.
HiaiModelIoProbe ProbeHiaiModelIo(const std::filesystem::path& model,
                                const std::string& expected_runtime_version);

// The explicit-state target has a separate, ordered ABI. Dtype comes from the
// hash-bound graph contract; the legacy HIAI descriptor exposes shape/format.
struct HiaiTensorSpec {
  std::string name, dtype;
  std::vector<std::uint32_t> shape;
};
std::size_t HiaiTensorBytes(const HiaiTensorSpec& spec);

struct HiaiInputIoDiagnostic {
  std::size_t index = 0, logical_bytes = 0, allocated_bytes = 0;
  std::string readback_sha256;
  bool readback_matches_source = false;
};
struct HiaiOutputIoDiagnostic {
  std::size_t index = 0, logical_bytes = 0, allocated_bytes_before = 0, allocated_bytes_after = 0;
  bool checked_after_process = false, object_same = false, buffer_same = false;
  bool fill_count_available = false;
  std::size_t fill_byte_count = 0;
};
struct HiaiIoDiagnostic {
  int output_fill_byte = -1;
  bool process_called = false;
  std::vector<HiaiInputIoDiagnostic> inputs;
  std::vector<HiaiOutputIoDiagnostic> outputs;
};

// Carries the precise failure even when executor construction has not completed.
class HiaiRuntimeError final : public std::runtime_error {
 public:
  HiaiRuntimeError(std::string stage, const std::string& message)
      : std::runtime_error(message), stage_(std::move(stage)) {}
  const std::string& stage() const noexcept { return stage_; }
 private:
  std::string stage_;
};

// One immutable graph, synchronous execution, caller-owned ordered tensor flow.
// The caller enforces the stateful target, standalone draft or fused-op ABI.
class HiaiStatefulExecutor final {
 public:
  HiaiStatefulExecutor(const std::filesystem::path& target,
                      const std::string& expected_runtime_version,
                      const std::vector<HiaiTensorSpec>& inputs,
                      const std::vector<HiaiTensorSpec>& outputs,
                      std::uint32_t timeout_ms = 300000,
                      bool fused_operator_probe = false,
                      int diagnostic_output_fill_byte = -1,
                      bool standalone_draft = false,
                      const std::string& model_load_api = "legacy-buffer");
  ~HiaiStatefulExecutor();
  void SetInput(std::size_t index, const std::vector<std::uint8_t>& bytes);
  void Execute();
  std::vector<std::uint8_t> CopyOutput(std::size_t index) const;
  const std::string& runtime_version() const noexcept;
  const HiaiIoDiagnostic& io_diagnostic() const noexcept;
  const std::string& runtime_stage() const noexcept;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Synchronous HIAI, two immutable offline models, persistent tensors.
class HiaiExecutor final : public GraphExecutor {
 public:
  HiaiExecutor(const std::filesystem::path& target,
               const std::filesystem::path& draft,
               const std::string& expected_runtime_version,
               std::size_t feature_width, std::size_t vocabulary_size,
               std::uint32_t timeout_ms = 300000,
               bool capture_target_features = false);
  ~HiaiExecutor() override;
  std::size_t sequence_length() const noexcept override;
  std::size_t draft_width() const noexcept override;
  const GraphOutputs& Execute(const std::vector<std::int64_t>& prefix,
                              std::int64_t pad_token_id,
                              bool produce_draft = false) override;
  const std::string& runtime_version() const noexcept;
  // Exact ABI FP16 bytes, copied before the draft can run. Available only when
  // capture was enabled and the latest Execute completed successfully.
  const std::vector<std::uint8_t>& target_features() const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace qwen35::dflash
