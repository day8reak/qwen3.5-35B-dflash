#include "qwen35_dflash/hiai_executor.hpp"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace {
void Require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void Unavailable(const qwen35::dflash::HiaiExecutor& executor) {
  bool rejected = false;
  try { static_cast<void>(executor.target_features()); }
  catch (const std::logic_error&) { rejected = true; }
  Require(rejected, "uncaptured or stale feature buffer was exposed");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    Require(argc == 2, "expected test output directory");
    const auto root = std::filesystem::path(argv[1]) / ("case-" + std::to_string(getpid()));
    Require(std::filesystem::create_directories(root), "test output directory must be new");
    const auto target = root / "target.omc", draft = root / "draft.omc";
    std::ofstream(target, std::ios::binary) << "target-fixture";
    std::ofstream(draft, std::ios::binary) << "draft-fixture";
    Require(setenv("FAKE_HIAI_FAILURE", "", 1) == 0, "clear fake fault failed");
    {
      qwen35::dflash::HiaiExecutor disabled(target, draft, "FAKE-HIAI-X90", 32, 64);
      Unavailable(disabled);
      disabled.Execute({1, 2}, 0, true);
      Unavailable(disabled);
    }
    {
      qwen35::dflash::HiaiExecutor captured(target, draft, "FAKE-HIAI-X90", 32, 64, 1000, true);
      Unavailable(captured);
      captured.Execute({1, 2}, 0);
      const auto& bytes = captured.target_features();
      Require(bytes.size() == 2048 && bytes[0] == 0xd2 && bytes[1] == 0x04 &&
              bytes[2] == 0 && bytes[3] == 0x80 && bytes[4] == 1 && bytes[5] == 0x7e &&
              bytes[2046] == 0xe1 && bytes[2047] == 0x10,
              "captured FP16 bits differ from target output");
      Require(setenv("FAKE_HIAI_FAILURE", "run", 1) == 0, "set Process fault failed");
      bool failed = false;
      try { captured.Execute({1, 2}, 0); }
      catch (const std::runtime_error&) { failed = true; }
      Require(failed, "Process fault did not fail execution");
      Unavailable(captured);
      Require(setenv("FAKE_HIAI_FAILURE", "", 1) == 0, "clear fake fault failed");
      captured.Execute({1, 2}, 0);
      Require(captured.target_features().size() == 2048, "capture did not recover");
      failed = false;
      try { captured.Execute({}, 0); }
      catch (const std::invalid_argument&) { failed = true; }
      Require(failed, "invalid prefix did not fail execution");
      Unavailable(captured);
    }
    std::cout << "HIAI feature capture: unavailable before success, exact bytes and no stale snapshots\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "HIAI feature capture test: " << error.what() << '\n';
    return 1;
  }
}
