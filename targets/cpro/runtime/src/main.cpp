#include "qwen35_dflash/hiai_executor.hpp"
#include "qwen35_dflash/sha256.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
using namespace qwen35::dflash;
#ifdef QWEN35_FAKE_HIAI
constexpr const char* kEvidenceKind = "host-fake-hiai";
#else
constexpr const char* kEvidenceKind = "device-hiai";
#endif
std::string JsonEscape(const std::string& value) {
  std::ostringstream output;
  for (const unsigned char item : value) {
    switch (item) {
      case '"':
        output << "\\\"";
        break;
      case '\\':
        output << "\\\\";
        break;
      case '\b':
        output << "\\b";
        break;
      case '\f':
        output << "\\f";
        break;
      case '\n':
        output << "\\n";
        break;
      case '\r':
        output << "\\r";
        break;
      case '\t':
        output << "\\t";
        break;
      default:
        if (item < 0x20U) {
          output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<int>(item) << std::dec;
        } else {
          output << static_cast<char>(item);
        }
    }
  }
  return output.str();
}

void WriteTokenIds(
    std::ostream& output,
    const std::vector<std::int64_t>& values) {
  output << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) {
      output << ',';
    }
    output << values[index];
  }
  output << ']';
}

void WriteDoubles(std::ostream& output, const std::vector<double>& values) {
  output << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) {
      output << ',';
    }
    output << values[index];
  }
  output << ']';
}

void WriteDistribution(std::ostream& output, const Distribution& value) {
  output << "{\"count\":" << value.count << ",\"min\":" << value.min
         << ",\"max\":" << value.max << ",\"mean\":" << value.mean
         << ",\"median\":" << value.median << ",\"p90\":" << value.p90
         << ",\"population_stdev\":" << value.population_stdev << '}';
}

void WriteMeasurement(
    std::ostream& output,
    const GenerationMeasurement& value,
    std::size_t repetition) {
  output << "{\"repetition\":" << repetition
         << ",\"generated_token_ids\":";
  WriteTokenIds(output, value.generated_token_ids);
  output << ",\"stop_reason\":\"" << JsonEscape(value.stop_reason) << "\""
         << ",\"counters\":{\"graph_calls\":"
         << value.counters.graph_calls << ",\"drafted_tokens\":"
         << value.counters.drafted_tokens
         << ",\"accepted_draft_tokens\":"
         << value.counters.accepted_draft_tokens
         << ",\"rejected_draft_tokens\":"
         << value.counters.rejected_draft_tokens
         << ",\"decode_iterations\":"
         << value.counters.decode_iterations << "},\"latency_ms\":{"
         << "\"prefill\":" << value.prefill_ms
         << ",\"decode\":" << value.decode_ms
         << ",\"model_total\":" << value.model_total_ms
         << "},\"decode_iteration_ms\":";
  WriteDoubles(output, value.decode_iteration_ms);
  output << '}';
}

void WriteBenchmark(std::ostream& output, const BenchmarkResult& value) {
  output << "{\"status\":\"PASS\",\"generation_mode\":\""
         << qwen35::dflash::ModeName(value.mode) << "\",\"warmup\":"
         << value.warmup << ",\"repetitions\":" << value.repetitions
         << ",\"stable_generated_token_ids\":";
  WriteTokenIds(output, value.stable_generated_token_ids);
  output << ",\"stable_stop_reason\":\""
         << JsonEscape(value.stable_stop_reason) << "\",\"latency_ms\":{";
  output << "\"prefill\":";
  WriteDistribution(output, value.prefill_ms);
  output << ",\"decode\":";
  WriteDistribution(output, value.decode_ms);
  output << ",\"model_total\":";
  WriteDistribution(output, value.model_total_ms);
  output << "},\"totals\":{\"graph_calls\":" << value.total_graph_calls
         << ",\"drafted_tokens\":" << value.total_drafted_tokens
         << ",\"accepted_draft_tokens\":"
         << value.total_accepted_draft_tokens
         << ",\"rejected_draft_tokens\":"
         << value.total_rejected_draft_tokens
         << "},\"acceptance_rate\":" << value.acceptance_rate
         << ",\"generated_tokens_per_second\":"
         << value.generated_tokens_per_second << ",\"measurements\":[";
  for (std::size_t index = 0; index < value.measurements.size(); ++index) {
    if (index != 0) {
      output << ',';
    }
    WriteMeasurement(output, value.measurements[index], index);
  }
  output << "]}";
}

void AtomicWrite(
    const std::filesystem::path& path,
    const std::string& payload) {
  const std::filesystem::path absolute = std::filesystem::absolute(path);
  if (std::filesystem::exists(absolute)) {
    throw std::runtime_error("refusing to overwrite output: " + absolute.string());
  }
  std::filesystem::create_directories(absolute.parent_path());
  const std::filesystem::path temporary = absolute.string() + ".tmp";
  if (std::filesystem::exists(temporary)) {
    throw std::runtime_error(
        "temporary output already exists: " + temporary.string());
  }
  {
    std::ofstream stream(temporary, std::ios::binary);
    if (!stream) {
      throw std::runtime_error("cannot create output: " + temporary.string());
    }
    stream << payload << '\n';
    stream.flush();
    if (!stream) {
      throw std::runtime_error("failed while writing output: " + temporary.string());
    }
  }
  std::filesystem::rename(temporary, absolute);
}


std::vector<std::int64_t> Tokens(const std::string& csv) {
  std::vector<std::int64_t> result;
  if (csv.empty()) return result;
  std::istringstream input(csv);
  std::string part;
  while (std::getline(input, part, ',')) {
    std::size_t n = 0;
    const auto value = std::stoll(part, &n);
    if (n != part.size() || value < 0) throw std::invalid_argument("invalid token CSV");
    result.push_back(value);
  }
  if (csv.back() == ',') throw std::invalid_argument("trailing token separator");
  return result;
}
std::size_t Size(const std::string& text, bool allow_zero = false) {
  std::size_t n = 0;
  auto value = std::stoll(text, &n);
  if (n != text.size() || value < (allow_zero ? 0 : 1))
    throw std::invalid_argument("invalid nonnegative integer: " + text);
  return static_cast<std::size_t>(value);
}
void Help() {
  std::cout << "qwen35_dflash_hiai --probe\n"
            << "qwen35_dflash_hiai --target TARGET.omc --target-sha256 HEX "
            << "--draft DRAFT.omc --draft-sha256 HEX --expected-runtime-version VERSION "
            << "--ddk-id VERSION --firmware-id VERSION "
            << "--prompt-token-ids CSV --output REPORT.json "
            << "[--mode generate|benchmark --generation-mode dflash|ordinary --max-new-tokens 32 --max-draft-tokens 15 --pad-token-id 248044 "
            << "--eos-token-ids 248044,248046 --feature-width 16384 --vocab-size 248320]\n";
}
}  // namespace
int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--help") { Help(); return 0; }
    if (argc == 2 && std::string(argv[1]) == "--probe") {
      const auto probe = qwen35::dflash::ProbeHiai();
      std::cout << "{\"runtime_version\":\"" << JsonEscape(probe.version)
                << "\",\"npu_supported\":" << (probe.npu_supported ? "true" : "false") << "}\n";
      return probe.npu_supported ? 0 : 1;
    }
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; i += 2) {
      if (i + 1 >= argc || std::string(argv[i]).rfind("--", 0) != 0 ||
          !args.emplace(argv[i], argv[i+1]).second)
        throw std::invalid_argument("expected unique --name value arguments");
    }
    auto take = [&args](const std::string& name, const std::string& fallback = "") {
      auto it = args.find(name);
      if (it == args.end()) {
        if (fallback.empty()) throw std::invalid_argument("missing " + name);
        return fallback;
      }
      std::string value = it->second; args.erase(it); return value;
    };
    const auto target = take("--target"), draft = take("--draft");
    const auto target_hash = take("--target-sha256"), draft_hash = take("--draft-sha256");
    const auto output = take("--output");
    if (std::filesystem::exists(output)) throw std::runtime_error("output already exists");
    const auto runtime_version = take("--expected-runtime-version");
    const auto timeout = Size(take("--timeout-ms", "300000"));
    if (timeout > UINT32_MAX) throw std::invalid_argument("timeout exceeds uint32");
    const auto ddk = take("--ddk-id"), firmware = take("--firmware-id");
    if (ddk.empty() || firmware.empty()) throw std::invalid_argument("DDK and firmware identities required");
    const auto mode = take("--mode", "generate");
    const auto generation_mode = take("--generation-mode", "dflash");
    if (mode != "generate" && mode != "benchmark") throw std::invalid_argument("unknown --mode");
    if (generation_mode != "ordinary" && generation_mode != "dflash")
      throw std::invalid_argument("unknown --generation-mode");
    const auto prompt = Tokens(take("--prompt-token-ids"));
    const auto feature_width = Size(take("--feature-width", "16384"));
    const auto vocab = Size(take("--vocab-size", "248320"));
    qwen35::dflash::GenerationOptions options;
    options.pad_token_id = static_cast<std::int64_t>(Size(take("--pad-token-id", "248044"), true));
    options.eos_token_ids = Tokens(take("--eos-token-ids", "248044,248046"));
    options.max_new_tokens = Size(take("--max-new-tokens", "32"));
    options.max_draft_tokens = Size(take("--max-draft-tokens", "15"));
    if (!args.empty()) throw std::invalid_argument("unknown option " + args.begin()->first);
    if (Sha256File(target) != target_hash || Sha256File(draft) != draft_hash)
      throw std::runtime_error("OMC SHA256 mismatch");
    for (auto token : options.eos_token_ids)
      if (static_cast<std::size_t>(token) >= vocab) throw std::invalid_argument("EOS outside vocabulary");
    qwen35::dflash::HiaiExecutor executor(target, draft, runtime_version, feature_width, vocab,
                                         static_cast<std::uint32_t>(timeout));
    std::ostringstream report;
    report << std::setprecision(17)
           << "{\"schema_version\":2,\"abi_version\":2,\"feature_dtype\":\"torch.float16\",\"status\":\"PASS\",\"scope\":\""
           << (mode == "benchmark" ? "paired-greedy-protocol" : "single-generation") << "\","
           << "\"evidence_kind\":\"" << kEvidenceKind << "\","
           << "\"target\":\"cpro-x90\",\"runtime\":\"HarmonyOS HIAI synchronous Process\","
           << "\"runtime_version\":\"" << JsonEscape(executor.runtime_version()) << "\""
           << ",\"ddk_id\":\"" << JsonEscape(ddk) << "\",\"firmware_id\":\"" << JsonEscape(firmware)
           << "\",\"target_sha256\":\"" << target_hash << "\",\"draft_sha256\":\"" << draft_hash
           << "\",\"cpu_fallback_requested\":false,\"state_policy\":\"recompute\","
           << "\"timing_scope\":\"synchronous model loop; excludes load/tokenization\","
           << "\"sequence_length\":" << executor.sequence_length()
           << ",\"max_new_tokens\":" << options.max_new_tokens
           << ",\"pad_token_id\":" << options.pad_token_id
           << ",\"prompt_token_ids\":";
    WriteTokenIds(report, prompt);
    report << ",\"eos_token_ids\":"; WriteTokenIds(report, options.eos_token_ids);
    if (mode == "benchmark") {
      const auto result = qwen35::dflash::BenchmarkPair(executor, prompt, options, 3, 10);
      report << ",\"ordinary\":"; WriteBenchmark(report, result.ordinary);
      report << ",\"dflash\":"; WriteBenchmark(report, result.dflash);
      report << ",\"token_id_mismatches\":" << result.token_id_mismatches
             << ",\"eos_mismatches\":" << result.eos_mismatches;
    } else {
      const auto selected = generation_mode == "ordinary" ? GenerationMode::kOrdinary : GenerationMode::kDFlash;
      const auto result = GenerateOnce(executor, prompt, selected, options);
      report << ",\"generation_mode\":\"" << ModeName(selected) << "\",\"generation\":";
      WriteMeasurement(report, result, 0);
    }
    report << '}';
    AtomicWrite(output, report.str());
    std::cout << "PASS: " << mode << "; report=" << output << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "qwen35_dflash_hiai: " << error.what() << '\n'; return 1;
  }
}
