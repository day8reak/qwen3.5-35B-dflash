#pragma once
#include "qwen35_dflash/generation.hpp"
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace qwen35::dflash {
struct HiaiProbe {
  std::string version;
  bool npu_supported;
};
HiaiProbe ProbeHiai();

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

// One immutable graph, synchronous Process, caller-owned ordered tensor flow.
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
                      bool standalone_draft = false);
  ~HiaiStatefulExecutor();
  void SetInput(std::size_t index, const std::vector<std::uint8_t>& bytes);
  void Execute();
  std::vector<std::uint8_t> CopyOutput(std::size_t index) const;
  const std::string& runtime_version() const noexcept;
  const HiaiIoDiagnostic& io_diagnostic() const noexcept;
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
