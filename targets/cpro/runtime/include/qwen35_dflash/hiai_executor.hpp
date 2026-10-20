#pragma once
#include "qwen35_dflash/generation.hpp"
#include <filesystem>
#include <memory>

namespace qwen35::dflash {
struct HiaiProbe {
  std::string version;
  bool npu_supported;
};
HiaiProbe ProbeHiai();

// Synchronous HIAI, two immutable offline models, persistent tensors.
class HiaiExecutor final : public GraphExecutor {
 public:
  HiaiExecutor(const std::filesystem::path& target,
               const std::filesystem::path& draft,
               const std::string& expected_runtime_version,
               std::size_t feature_width, std::size_t vocabulary_size,
               std::uint32_t timeout_ms = 300000);
  ~HiaiExecutor() override;
  std::size_t sequence_length() const noexcept override;
  std::size_t draft_width() const noexcept override;
  const GraphOutputs& Execute(const std::vector<std::int64_t>& prefix,
                              std::int64_t pad_token_id,
                              bool produce_draft = false) override;
  const std::string& runtime_version() const noexcept;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace qwen35::dflash
