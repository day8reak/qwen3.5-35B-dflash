#pragma once
#include "qwen35_dflash/hiai_executor.hpp"
#include "qwen35_dflash/incremental_draft.hpp"

namespace qwen35::dflash {
struct ResidentSessionConfig {
  std::filesystem::path target_model, target_weight_directory, draft_model;
  std::string runtime_version;
  std::vector<HiaiBoundRuntimeTensor> target_inputs, target_outputs;
  std::uint32_t timeout_ms = 300000;
  std::string model_load_api = "file-v2";
  IncrementalDraftAbi incremental_draft;
  bool diagnostics = true;
  bool draft_preread = false;
};
struct ResidentRunResult {
  std::vector<std::vector<std::uint8_t>> outputs;
  HiaiIoDiagnostic io_diagnostic;
  std::uint64_t wall_ms = 0, api_us = 0;
};
struct ResidentSessionEvidence {
  std::string stage = "NOT_LOADED", status = "NOT_LOADED", model_load_api = "file-v2";
  std::string target_model_name = "qwen35_resident_target", draft_model_name = "qwen35_resident_draft";
  std::string cleanup_error;
  std::uint64_t target_load_attempts = 0, target_loads = 0, draft_load_attempts = 0, draft_loads = 0;
  std::uint64_t target_run_attempts = 0, target_runs = 0, draft_run_attempts = 0, draft_runs = 0;
  std::uint64_t target_close_calls = 0, draft_close_calls = 0;
  std::uint64_t target_load_ms = 0, draft_load_ms = 0, target_run_ms = 0, draft_run_ms = 0, close_ms = 0;
  std::uint64_t target_api_us = 0, draft_api_us = 0;
  bool closed = false, poisoned = false;
  HiaiOriginalTargetLifecycle target_lifecycle;
  HiaiLegacyLifecycle target_v1_lifecycle, draft_v1_lifecycle;
};

// One process owns two immutable models through one explicit policy. V1 uses
// file-only target and buffer-only embedded draft; V2 is a separate policy.
// Mixed uses a V2 external-weight target and V1 buffer-only embedded draft.
// Clients/managers retain their persistent tensor storage.
// The caller supplies hash-bound sources/descriptors, preserves source lifetimes,
// and advances only committed target states. This class never falls back to a
// different API, recompiles a model or retries a failed load/run.
class ResidentSession final {
 public:
  explicit ResidentSession(const std::string& model_load_api="file-v2");
  ~ResidentSession();
  ResidentSession(const ResidentSession&) = delete;
  ResidentSession& operator=(const ResidentSession&) = delete;
  void Load(const ResidentSessionConfig& config);
  ResidentRunResult RunTarget(const std::vector<std::vector<std::uint8_t>>& inputs);
  ResidentRunResult RunDraft(const std::vector<std::vector<std::uint8_t>>& inputs);
  void Close();
  const ResidentSessionEvidence& evidence() const noexcept;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace qwen35::dflash
