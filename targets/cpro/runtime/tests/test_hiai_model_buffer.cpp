#include "qwen35_dflash/hiai_executor.hpp"
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace fs = std::filesystem;

namespace {
void Require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void ExpectRejected(const fs::path& target, const fs::path& draft,
                    const fs::path& trace, const std::string& category) {
  Require(setenv("FAKE_HIAI_TRACE", trace.c_str(), 1) == 0, "set trace path failed");
  std::string error;
  try {
    qwen35::dflash::HiaiExecutor executor(target, draft, "FAKE-HIAI-X90", 32, 64, 1000);
  } catch (const std::exception& failure) {
    error = failure.what();
  }
  Require(!error.empty(), "invalid OMC file was accepted: " + target.string());
  Require(error.find(category) != std::string::npos,
          "wrong rejection category for " + target.string() + ": " + error);
  Require(error.find(target.string()) != std::string::npos,
          "rejection omitted model path: " + error);
  std::ifstream events(trace);
  for (std::string event; std::getline(events, event); ) {
    Require(event != "load" && event != "process",
            "invalid OMC reached HIAI Load/Process: " + event);
  }
}
}  // namespace

int main(int argc, char** argv) {
  try {
    Require(argc == 2, "expected test output directory");
    const fs::path root = fs::path(argv[1]) / ("case-" + std::to_string(getpid()));
    Require(fs::create_directories(root), "test output directory must be new");
    Require(setenv("FAKE_HIAI_FAILURE", "", 1) == 0, "clear fake fault failed");
    const auto draft = root / "draft.omc";
    std::ofstream(draft, std::ios::binary) << "draft-fixture";

    const auto empty = root / "empty.omc";
    std::ofstream(empty, std::ios::binary).close();
    ExpectRejected(empty, draft, root / "empty.trace", "empty");
    ExpectRejected(root / "missing.omc", draft, root / "missing.trace", "open");

    // A sparse file exercises the actual adapter's size check without allocating
    // or hashing 4 GiB. Reaching Load is a failure, even if the fake rejects bytes.
    const auto oversized = root / "oversized.omc";
    std::ofstream(oversized, std::ios::binary).close();
    const auto bytes = static_cast<std::uintmax_t>(UINT32_MAX) + 1;
    fs::resize_file(oversized, bytes);
    Require(fs::file_size(oversized) == bytes, "sparse test file size mismatch");
    ExpectRejected(oversized, draft, root / "oversized.trace", "uint32");

    std::cout << "HIAI model buffer: empty, missing and uint32 overflow rejected before Load\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "HIAI model buffer test: " << error.what() << '\n';
    return 1;
  }
}
