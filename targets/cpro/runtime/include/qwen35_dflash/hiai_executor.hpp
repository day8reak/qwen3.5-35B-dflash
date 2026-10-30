#pragma once
#include "qwen35_dflash/generation.hpp"
#include <cstdint>
#include <filesystem>
#include <memory>
#include <map>
#include <optional>
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

// Process-local ownership observations only. Never serialized as a device
// identity or used to infer a state slot. ResidentSession rejects overlap and
// requires these live allocations to remain stable across alternating runs.
struct HiaiTensorStorage {
  const void* object = nullptr;
  const void* data = nullptr;
  std::size_t bytes = 0;
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

// V1 gives ordered dimensions/format, not SDK names or dtype introspection.
// Flags are historical; closed means shutdown was attempted exactly once.
struct HiaiLegacyRuntimeTensor {
  std::size_t index=0;
  std::vector<std::int64_t> shape;
  std::int64_t raw_format=0;
  std::size_t bytes=0;
};
struct HiaiLegacyLifecycle {
  std::string model_load_api = "legacy-buffer";
  std::string lifecycle_policy = "legacy-v1-load-process-unload-v1";
  std::string model_name;
  std::string runtime_tensor_names = "NOT_QUERIED_V1_SAVED_NAMED_ABI_ONLY";
  std::string runtime_dtypes = "DECLARED_BOUND_ABI_NOT_QUERIED";
  std::string external_weight_resolution = "MODEL_PATH_PARENT_SIDECARS_UNVALIDATED_ON_DEVICE";
  bool client_init_attempted=false, client_initialized=false, load_attempted=false, loaded=false;
  bool unload_called=false, closed=false;
  std::optional<std::int64_t> client_init_status, load_status, unload_status;
  std::uint64_t process_attempts=0, successful_processes=0;
  std::string failure_stage, cleanup_error;
  std::vector<HiaiLegacyRuntimeTensor> observed_inputs, observed_outputs;
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
                      const std::string& model_load_api = "legacy-buffer",
                      const std::string& resident_model_name = "",
                      HiaiLegacyLifecycle* constructor_failure_snapshot = nullptr,
                      const std::vector<HiaiTensorStorage>& disjoint_from = {});
  ~HiaiStatefulExecutor();
  void SetInput(std::size_t index, const std::vector<std::uint8_t>& bytes);
  void Execute();
  std::vector<std::uint8_t> CopyOutput(std::size_t index) const;
  // Explicit API-specific shutdown; idempotent even on SDK failure.
  // Destructor cleanup is best-effort and never retries this call.
  void Close();
  const std::string& runtime_version() const noexcept;
  const HiaiIoDiagnostic& io_diagnostic() const noexcept;
  const std::string& runtime_stage() const noexcept;
  const HiaiLegacyLifecycle& legacy_lifecycle() const noexcept;
  std::vector<HiaiTensorStorage> tensor_storage() const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Exact ordered descriptors captured from the original target's named V2 probe.
// These are runtime names and dimensions, not source RefData names or draft ND
// shapes. The caller separately binds source state owners to these runtime slots.
struct HiaiBoundRuntimeTensor {
  HiaiModelIoTensor descriptor;
  std::string dtype;
  std::size_t bytes = 0;
};
// Exact89/82 original target ABI, V1 synchronous Load/Process/UnLoadModel.
// A separate instance can serve static64 prefill once. Sidecar loading from the
// preserved main-model path is an explicit, not device-validated assumption.
class HiaiOriginalTargetV1Executor final {
 public:
  HiaiOriginalTargetV1Executor(const std::filesystem::path& model,
      const std::filesystem::path& weight_directory, const std::string& expected_runtime_version,
      const std::vector<HiaiBoundRuntimeTensor>& inputs,
      const std::vector<HiaiBoundRuntimeTensor>& outputs, std::uint32_t timeout_ms=300000,
      const std::string& model_name="qwen35_resident_target",
      HiaiLegacyLifecycle* constructor_failure_snapshot=nullptr);
  ~HiaiOriginalTargetV1Executor();
  void SetInput(std::size_t index,const std::vector<std::uint8_t>& bytes);
  void Execute();
  std::vector<std::uint8_t> CopyOutput(std::size_t index) const;
  void Close();
  const HiaiIoDiagnostic& io_diagnostic() const noexcept;
  const HiaiLegacyLifecycle& lifecycle() const noexcept;
  std::vector<HiaiTensorStorage> tensor_storage() const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

struct HiaiOriginalTargetLifecycle {
  std::string weight_init_policy = "INIT_BEFORE_WEIGHTS_NO_FALLBACK";
  std::string weight_cleanup_policy = "DEINIT_WEIGHTS_BEFORE_DEINIT";
  std::string ordering_validation = "NOT_VALIDATED_ON_DEVICE";
  // Init flags retain historical attempts/successes after Close. They do not
  // mean resources are live: closed/deinit fields describe shutdown separately.
  bool weights_init_attempted = false, weights_initialized = false;
  bool manager_init_attempted = false, manager_initialized = false;
  bool manager_deinit_called = false, weights_deinit_called = false, closed = false;
  // No value means the SDK has not returned from this call. In particular a
  // real signed -1 status is distinct from an exception or an uncalled API.
  std::optional<std::int64_t> weights_init_status, manager_init_status;
  std::string failure_stage, cleanup_error;
  std::int64_t weights_deinit_status = -1;
  std::uint64_t run_attempts = 0, successful_runs = 0;
};
struct HiaiOriginalTargetLoadProbe {
  bool passed = false;
  std::string runtime_stage, error, runtime_version;
  HiaiOriginalTargetLifecycle lifecycle;
  std::uint64_t elapsed_ms = 0;
};

// Original 40-layer target only: one file-v2 model, one explicit external-weight
// initialization and persistent exact-descriptor tensors. Repeated Run is local
// to this object's lifetime. The selected headers declare InitWeights but do
// not establish its ordering; this experimental order never retries/falls back.
// The session caller must preserve/hash-bind model and external-weight sources
// for the full object lifetime; this adapter does not attest their provenance.
// This adapter is not a resident-worker protocol or a device validation claim.
class HiaiOriginalTargetExecutor final {
 public:
  // A separate, single-order load/close experiment. It never constructs
  // tensors or calls Run. The complete lifecycle remains to be device-tested.
  static HiaiOriginalTargetLoadProbe ProbeInitBeforeWeights(
      const std::filesystem::path& model, const std::filesystem::path& weight_directory,
      const std::string& expected_runtime_version);
  HiaiOriginalTargetExecutor(const std::filesystem::path& model,
                            const std::filesystem::path& weight_directory,
                            const std::string& expected_runtime_version,
                            const std::vector<HiaiBoundRuntimeTensor>& inputs,
                            const std::vector<HiaiBoundRuntimeTensor>& outputs,
                            const std::string& resident_model_name = "",
                            HiaiOriginalTargetLifecycle* constructor_failure_snapshot = nullptr);
  ~HiaiOriginalTargetExecutor();
  void SetInput(std::size_t index, const std::vector<std::uint8_t>& bytes);
  void Execute();
  std::vector<std::uint8_t> CopyOutput(std::size_t index) const;
  // Explicit shutdown makes DeinitWeights' Status inspectable. Idempotent;
  // failures are reported, never retried by the destructor.
  void Close();
  const std::string& runtime_version() const noexcept;
  const std::string& runtime_stage() const noexcept;
  const HiaiIoDiagnostic& io_diagnostic() const noexcept;
  const HiaiOriginalTargetLifecycle& lifecycle() const noexcept;
  const std::vector<HiaiBoundRuntimeTensor>& observed_inputs() const noexcept;
  const std::vector<HiaiBoundRuntimeTensor>& observed_outputs() const noexcept;
  std::vector<HiaiTensorStorage> tensor_storage() const;
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
