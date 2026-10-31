// Bounded diagnostic runner: one Load, warmup, repeated synchronous Process.
#include "qwen35_dflash/hiai_executor.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

namespace fs = std::filesystem;
using namespace qwen35::dflash;
using Bytes = std::vector<std::uint8_t>;

unsigned Number(const std::string& text) {
  if (text.empty() || text.size() > 9 || text.find_first_not_of("0123456789") != std::string::npos)
    throw std::invalid_argument("invalid probe integer");
  return static_cast<unsigned>(std::stoul(text));
}
struct Plan {
  unsigned warmup = 0, repetitions = 0;
  std::vector<HiaiTensorSpec> inputs, outputs;
};
Plan ReadPlan(const fs::path& path) {
  if (!fs::is_regular_file(path) || fs::file_size(path) > 65536)
    throw std::invalid_argument("missing or oversized probe plan");
  std::ifstream stream(path); std::string line;
  std::getline(stream, line);
  if (line != "QWEN35_TENSOR_PROBE 1") throw std::invalid_argument("invalid probe header");
  Plan plan; bool iterations = false; std::size_t total = 0;
  while (std::getline(stream, line)) {
    std::istringstream row(line); std::vector<std::string> fields; std::string token;
    while (row >> token) fields.push_back(token);
    if (fields.size() == 3 && fields[0] == "iterations" && !iterations && plan.inputs.empty()) {
      plan.warmup = Number(fields[1]); plan.repetitions = Number(fields[2]); iterations = true;
      if (plan.warmup > 10 || !plan.repetitions || plan.repetitions > 20)
        throw std::invalid_argument("invalid probe iteration count");
    } else if (fields.size() == 5 && iterations && (fields[0] == "input" || fields[0] == "output")) {
      auto& specs = fields[0] == "input" ? plan.inputs : plan.outputs;
      if (Number(fields[1]) != specs.size() || specs.size() >= 8 || fields[3] != "float32")
        throw std::invalid_argument("probe requires contiguous bounded FP32 slots");
      HiaiTensorSpec spec{fields[2], fields[3], {}};
      std::istringstream dimensions(fields[4]);
      while (std::getline(dimensions, token, ',')) {
        const auto n = Number(token);
        if (!n || n > 65536 || spec.shape.size() == 4) throw std::invalid_argument("invalid probe shape");
        spec.shape.push_back(n);
      }
      if (spec.shape.empty()) throw std::invalid_argument("probe shape is empty");
      const auto bytes = HiaiTensorBytes(spec);
      if (bytes > (16U << 20) || total > (16U << 20) - bytes)
        throw std::invalid_argument("probe IO exceeds 16 MiB");
      total += bytes; specs.push_back(spec);
    } else throw std::invalid_argument("unknown probe row");
  }
  if (!iterations || plan.inputs.empty() || plan.outputs.empty()) throw std::invalid_argument("incomplete probe plan");
  return plan;
}
Bytes Read(const fs::path& path, std::size_t size) {
  if (fs::is_symlink(path) || !fs::is_regular_file(path) || fs::file_size(path) != size)
    throw std::invalid_argument("probe input byte extent differs");
  Bytes bytes(size); std::ifstream stream(path, std::ios::binary);
  if (!stream.read(reinterpret_cast<char*>(bytes.data()), size)) throw std::runtime_error("probe input read failed");
  return bytes;
}
int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--help") {
      std::cout << "qwen35_tensor_probe MODEL RUNTIME_VERSION FIXTURE_DIRECTORY NEW_OUTPUT_DIRECTORY\n";
      return 0;
    }
    if (argc != 5) throw std::invalid_argument("expected model, runtime, fixture and output");
    const fs::path fixture = argv[3], output = argv[4];
    const auto plan = ReadPlan(fixture / "plan.txt");
    if (fs::exists(output)) throw std::invalid_argument("probe output already exists");
    std::vector<Bytes> inputs;
    for (std::size_t i = 0; i < plan.inputs.size(); ++i)
      inputs.push_back(Read(fixture / ("input_" + std::to_string(i) + ".bin"), HiaiTensorBytes(plan.inputs[i])));
    HiaiStatefulExecutor executor(argv[1], argv[2], plan.inputs, plan.outputs,
        30000, true, -1, false, "legacy-buffer", "qwen35_tensor_probe", nullptr, {}, false);
    fs::create_directories(output);
    for (std::size_t i = 0; i < inputs.size(); ++i) executor.SetInput(i, inputs[i]);
    std::ofstream result(output / "result.json");
#ifdef QWEN35_FAKE_HIAI
    const char* evidence = "host-fake-hiai";
#else
    const char* evidence = "device-hiai";
#endif
    result << "{\"evidence_kind\":\"" << evidence << "\",\"warmup\":" << plan.warmup
           << ",\"repetitions\":" << plan.repetitions << ",\"api_us\":[";
    for (unsigned i = 0; i < plan.warmup + plan.repetitions; ++i) {
      if (i) for (std::size_t j = 0; j < inputs.size(); ++j) executor.ReuseInput(j);
      executor.Execute();
      const auto us = executor.last_api_us();
      if (i < plan.warmup) continue;
      const auto index = i - plan.warmup;
      if (index) result << ',';
      result << us;
      // Every retained call has a snapshot; file IO is outside the API timer.
      std::ofstream snapshot(output / (std::to_string(index) + ".bin"), std::ios::binary);
      for (std::size_t j = 0; j < plan.outputs.size(); ++j) {
        const auto bytes = executor.CopyOutput(j);
        if (bytes.size() != HiaiTensorBytes(plan.outputs[j])) throw std::runtime_error("output extent changed");
        snapshot.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
      }
      snapshot.close();
      if (!snapshot) throw std::runtime_error("probe snapshot write failed");
    }
    executor.Close();
    result << "],\"status\":\"EXECUTED\",\"timing_scope\":\"SYNCHRONOUS_API_WALL_NOT_NPU_ONLY\"}\n";
    result.close();
    if (!result) throw std::runtime_error("probe result write failed");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
