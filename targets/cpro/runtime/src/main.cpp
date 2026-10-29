#include "qwen35_dflash/hiai_executor.hpp"
#include "qwen35_dflash/sha256.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
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

void WriteHex(std::ostream& output, const std::vector<std::uint8_t>& bytes) {
  constexpr char digits[] = "0123456789abcdef";
  output << '"';
  for (const auto byte : bytes) output << digits[byte >> 4] << digits[byte & 15];
  output << '"';
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
std::uint64_t StepInteger(const std::string& text, std::uint64_t maximum, bool zero = false) {
  if (text.empty() || text.size() > 20)
    throw std::invalid_argument("invalid stateful plan integer");
  std::uint64_t value = 0;
  for (char c : text) {
    if (c < '0' || c > '9' || value > maximum / 10 ||
        (value == maximum / 10 && static_cast<unsigned>(c - '0') > maximum % 10))
      throw std::invalid_argument("stateful plan integer outside bounds");
    value = value * 10 + static_cast<unsigned>(c - '0');
  }
  if (!zero && !value) throw std::invalid_argument("stateful plan integer must be positive");
  return value;
}
std::vector<std::string> Split(const std::string& text, char separator) {
  std::vector<std::string> values;
  std::size_t start = 0;
  for (;;) {
    const auto end = text.find(separator, start);
    values.push_back(text.substr(start, end == std::string::npos ? end : end - start));
    if (end == std::string::npos) return values;
    start = end + 1;
  }
}
void NoSymlinks(const std::filesystem::path& path) {
  std::filesystem::path current;
  for (const auto& part : std::filesystem::absolute(path).lexically_normal()) {
    current /= part;
    if (std::filesystem::is_symlink(std::filesystem::symlink_status(current)))
      throw std::invalid_argument("stateful paths cannot contain symlinks: " + current.string());
  }
}
std::uintmax_t StepFileSize(const std::filesystem::path& path, std::uintmax_t maximum) {
  NoSymlinks(path);
  if (!std::filesystem::is_regular_file(path))
    throw std::invalid_argument("stateful payload must be a regular file: " + path.string());
  const auto bytes = std::filesystem::file_size(path);
  if (!bytes || bytes > maximum)
    throw std::invalid_argument("stateful payload is empty or exceeds its size limit: " + path.string());
  return bytes;
}
std::filesystem::path StepPayload(const std::filesystem::path& root, const std::string& value) {
  const std::filesystem::path relative(value);
  if (value.empty() || value.size() > 1024 || relative.is_absolute() ||
      !std::regex_match(value, std::regex("[A-Za-z0-9_./-]+")))
    throw std::invalid_argument("stateful payload requires a bounded relative path");
  for (const auto& part : relative)
    if (part == "." || part == ".." || part.empty())
      throw std::invalid_argument("stateful payload path traversal is forbidden");
  const auto result = root / relative;
  NoSymlinks(result);
  return result;
}
void StepHash(const std::string& value) {
  if (!std::regex_match(value, std::regex("[0-9a-f]{64}")))
    throw std::invalid_argument("stateful payload requires lowercase SHA256");
}
std::vector<std::uint8_t> StepRead(const std::filesystem::path& path, std::uintmax_t maximum) {
  const auto size = StepFileSize(path, maximum);
  std::vector<std::uint8_t> data(static_cast<std::size_t>(size));
  std::ifstream input(path, std::ios::binary);
  input.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
  if (!input || input.peek() != std::char_traits<char>::eof())
    throw std::runtime_error("stateful payload changed size or could not be read");
  return data;
}
std::string BytesHash(const std::vector<std::uint8_t>& bytes) {
  return Sha256(std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size()));
}
HiaiTensorSpec StepTensor(const std::vector<std::string>& fields, bool raw_source_name = false) {
  const auto pattern = raw_source_name ? "[A-Za-z_][A-Za-z0-9_./-]{0,255}" : "[A-Za-z_][A-Za-z0-9_]{0,127}";
  if (!std::regex_match(fields[2], std::regex(pattern)))
    throw std::invalid_argument("invalid stateful tensor name");
  HiaiTensorSpec result{fields[2], fields[3], {}};
  if (!fields[4].empty())
    for (const auto& dim : Split(fields[4], ','))
      result.shape.push_back(static_cast<std::uint32_t>(StepInteger(dim, UINT32_MAX)));
  static_cast<void>(HiaiTensorBytes(result));
  return result;
}
struct StepInput {
  HiaiTensorSpec spec;
  std::filesystem::path file;
  std::string path, sha256;
};
struct StepPlan {
  std::filesystem::path model;
  std::string model_hash, abi_hash, plan_hash, operator_type;
  std::uint32_t sequence = 0, past = 0;
  bool draft_trace = false;
  bool draft_direct = false;
  std::vector<StepInput> inputs;
  std::vector<HiaiTensorSpec> outputs;
};
void CheckFusedPlan(StepPlan& plan) {
  if (plan.inputs.empty())
    throw std::invalid_argument("fused operator requires its documented ordered IO contract");
  std::uint32_t s = 0;
  if (plan.operator_type == "CacheUpdate") {
    if (plan.inputs.size() != 4 || plan.inputs[1].spec.shape.size() != 3)
      throw std::invalid_argument("CacheUpdate requires its four-port contract");
    s = plan.inputs[1].spec.shape[0];
  } else if (plan.operator_type == "AdnFusedInferAttention") {
    if (plan.inputs[0].spec.shape.size() != 4)
      throw std::invalid_argument("AdnFusedInferAttention requires its NZ query contract");
    s = plan.inputs[0].spec.shape[2];
  } else {
    if (plan.inputs[0].spec.shape.size() < 2)
      throw std::invalid_argument("fused operator requires its documented ordered IO contract");
    s = plan.inputs[0].spec.shape[1];
  }
  if (s != 1 && s != 8)
    throw std::invalid_argument("fused operator probe supports sequence lengths 1 or 8 only");
  std::vector<HiaiTensorSpec> inputs, outputs;
  if (plan.operator_type == "ChunkGatedDeltaRule") {
    if (plan.outputs.size() != 2)
      throw std::invalid_argument("ChunkGatedDeltaRule output count differs from its two-port contract");
    if (plan.inputs.size() != 7 ||
        (plan.inputs[5].spec.dtype != "float16" && plan.inputs[5].spec.dtype != "float32"))
      throw std::invalid_argument("ChunkGatedDeltaRule requires seven inputs and explicit FP16/FP32 state");
    if (plan.outputs[1].dtype != "float16" && plan.outputs[1].dtype != "float32")
      throw std::invalid_argument("ChunkGatedDeltaRule final state contract requires explicit FP16/FP32 dtype");
    inputs = {{"query", "float16", {1,s,32,128}}, {"key", "float16", {1,s,32,128}},
              {"value", "float16", {1,s,32,128}}, {"g", "float16", {1,s,32}},
              {"beta", "float16", {1,s,32}},
              {"initial_state", plan.inputs[5].spec.dtype, {1,32,128,128}},
              {"effective_length", "int16", {1}}};
    outputs = {{"core_attn", "float16", {1,s,32,128}},
               {"last_recurrent_state", plan.outputs[1].dtype, {1,32,128,128}}};
  } else if (plan.operator_type == "LinearAttentionConvstateUpdate") {
    if (plan.inputs.size() != 3 || plan.inputs[1].spec.shape.size() != 3 ||
        (plan.inputs[1].spec.shape[2] != 3 && plan.inputs[1].spec.shape[2] != 4))
      throw std::invalid_argument("LinearAttentionConvstateUpdate requires its explicit width-3/4 probe contract");
    inputs = {{"x1", "float16", {1,s,8192}}, {"x2", "float16", {1,8192,plan.inputs[1].spec.shape[2]}},
              {"x3", "int32", {128}}};
    outputs = {{"y1", "float16", {1,8192,4}}, {"y2", "float16", {1,8192,s+3}}};
  } else if (plan.operator_type == "AdnRmsNorm") {
    if (s != 8) throw std::invalid_argument("AdnRmsNorm probe requires sequence length 8");
    inputs = {{"self", "float16", {1,s,2048}}, {"gamma", "float16", {2048}}};
    outputs = {{"y", "float16", {1,s,2048}}, {"rstd", "float32", {1,s,1}}};
  } else if (plan.operator_type == "CacheUpdate") {
    inputs = {{"x", "float16", {64,32,32,16}}, {"updates", "float16", {s,32,16}},
              {"targetBlock", "int32", {1}}, {"offsetInBlock", "int32", {}}};
    outputs = {{"x", "float16", {64,32,32,16}}};
  } else if (plan.operator_type == "AdnFusedInferAttention") {
    inputs = {{"query", "float16", {1,256,s,16}}, {"key0", "float16", {64,32,32,16}},
              {"value0", "float16", {64,32,32,16}}, {"atten_mask", "float16", {1,1,s,2048}},
              {"actual_seq_lengths_q", "int64", {1}}, {"actual_seq_lengths_kv", "int64", {1}},
              {"block_table", "int32", {1,64}}, {"all_seq_lengths_q", "int64", {1}}};
    outputs = {{"attention_out", "float16", {1,256,s,16}}};
  } else throw std::invalid_argument("unknown documented fused operator");
  if (inputs.size() != plan.inputs.size() || outputs.size() != plan.outputs.size())
    throw std::invalid_argument("fused operator IO count differs from documented contract");
  for (bool input : {true, false}) {
    const auto& expected = input ? inputs : outputs;
    for (std::size_t i = 0; i < expected.size(); ++i) {
      const auto& actual = input ? plan.inputs[i].spec : plan.outputs[i];
      if (actual.name != expected[i].name || actual.dtype != expected[i].dtype ||
          actual.shape != expected[i].shape)
        throw std::invalid_argument("fused operator ordered name/shape/dtype differs from documented contract");
    }
  }
  plan.sequence = s;
}
void CheckDraftPlan(const StepPlan& plan) {
  const auto s = plan.sequence;
  const std::vector<HiaiTensorSpec> inputs{{"target_features", "float16", {1,s,16384}},
      {"input_ids", "int32", {1,s}}, {"attention_mask", "int32", {1,s}}};
  std::vector<HiaiTensorSpec> outputs{{"draft_top1", "int32", {1,15}},
      {"draft_hidden", "float16", {1,15,2048}}};
  if (plan.draft_trace) {
    if (plan.outputs.size() != 2 || plan.outputs[1].shape.size() != 3 ||
        plan.outputs[1].shape[0] != 1 || plan.outputs[1].shape[1] != 1 ||
        plan.outputs[1].shape[2] == 0 || HiaiTensorBytes(plan.outputs[1]) > 64ULL * 1024 * 1024)
      throw std::invalid_argument("draft trace requires a positive bounded packed output");
    outputs[1] = {"draft_trace_values", "float32", plan.outputs[1].shape};
  }
  if (plan.draft_direct) {
    if (plan.outputs.size() < 2 || plan.outputs.size() > 16)
      throw std::invalid_argument("direct draft requires two through sixteen outputs");
    std::uint64_t bytes = 0;
    for (const auto& spec : plan.outputs) {
      if (spec.shape.empty() || spec.shape.size() > 4 ||
          (spec.dtype != "float16" && spec.dtype != "float32" && spec.dtype != "int32"))
        throw std::invalid_argument("direct draft output requires a supported raw dtype and rank1..4");
      bytes += HiaiTensorBytes(spec);
      if (bytes > 64ULL * 1024 * 1024)
        throw std::invalid_argument("direct draft outputs exceed 64 MiB");
    }
    outputs.insert(outputs.end(), plan.outputs.begin() + 2, plan.outputs.end());
  }
  if (plan.past || plan.inputs.size() != inputs.size() || plan.outputs.size() != outputs.size())
    throw std::invalid_argument("standalone draft requires three inputs, two outputs and prefix recomputation");
  for (bool input : {true, false}) {
    const auto& expected = input ? inputs : outputs;
    for (std::size_t i = 0; i < expected.size(); ++i) {
      const auto& actual = input ? plan.inputs[i].spec : plan.outputs[i];
      if (actual.name != expected[i].name || actual.dtype != expected[i].dtype || actual.shape != expected[i].shape)
        throw std::invalid_argument("standalone draft differs from the locked full draft IO contract");
    }
  }
}
void CheckDraftInputs(const std::vector<std::vector<std::uint8_t>>& values, std::uint32_t sequence) {
  bool padding = false, nonempty = false;
  for (std::size_t i = 0; i < sequence; ++i) {
    std::int32_t token, mask;
    std::memcpy(&token, values[1].data() + 4*i, 4);
    std::memcpy(&mask, values[2].data() + 4*i, 4);
    if (token < 0 || token >= 248320) throw std::invalid_argument("draft token ID outside full vocabulary");
    if ((mask != 0 && mask != 1) || (padding && mask))
      throw std::invalid_argument("draft attention mask must be a right-padded 0/1 prefix");
    nonempty = nonempty || mask;
    padding = padding || !mask;
  }
  if (!nonempty) throw std::invalid_argument("draft prefix must contain at least the anchor token");
  for (std::size_t i = 0; i < values[0].size(); i += 2) {
    const auto bits = static_cast<std::uint16_t>(values[0][i]) |
                      (static_cast<std::uint16_t>(values[0][i+1]) << 8);
    if ((bits & 0x7c00) == 0x7c00) throw std::invalid_argument("draft features must be finite FP16");
  }
}
StepPlan ReadStepPlan(const std::filesystem::path& path, bool fused = false, bool draft = false) {
  const auto payload = StepRead(path, 1ULL << 20);
  const std::string text(payload.begin(), payload.end());
  for (unsigned char c : text)
    if ((c < 32 && c != '\t' && c != '\n') || c > 126)
      throw std::invalid_argument("stateful step plan must contain ASCII fields with LF line endings");
  auto rows = Split(text, '\n');
  if (!rows.empty() && rows.back().empty()) rows.pop_back();
  const bool draft_trace = draft && !rows.empty() && rows[0] == "QWEN35_DRAFT_TRACE_STEP\t1";
  const bool draft_direct = draft && !rows.empty() && rows[0] == "QWEN35_DRAFT_DIRECT_STEP\t1";
  const std::string header = draft_direct ? "QWEN35_DRAFT_DIRECT_STEP\t1" :
                             draft_trace ? "QWEN35_DRAFT_TRACE_STEP\t1" : draft ? "QWEN35_DRAFT_STEP\t1" :
                             fused ? "QWEN35_FUSED_OP_STEP\t1" : "QWEN35_STATEFUL_STEP\t1";
  if (rows.size() < 7 || rows[0] != header)
    throw std::invalid_argument("step plan header must be " + header);
  const auto root = std::filesystem::absolute(path).lexically_normal().parent_path();
  StepPlan plan;
  plan.draft_trace = draft_trace;
  plan.draft_direct = draft_direct;
  plan.plan_hash = BytesHash(payload);
  for (std::size_t i = 1; i <= 2; ++i) {
    const auto fields = Split(rows[i], '\t');
    if (fields.size() != 3 || fields[0] != (i == 1 ? "model" : draft ? "draft_abi" : fused ? "operator_abi" : "state_abi"))
      throw std::invalid_argument("step plan requires ordered model and ABI records");
    StepHash(fields[2]);
    const auto file = StepPayload(root, fields[1]);
    StepFileSize(file, i == 1 ? UINT32_MAX : 1ULL << 20);
    if (Sha256File(file) != fields[2]) throw std::runtime_error("stateful model/state ABI SHA256 mismatch");
    if (i == 1) { plan.model = file; plan.model_hash = fields[2]; }
    else {
      plan.abi_hash = fields[2];
      if (draft_trace || draft_direct) {
        // The host validates the complete trace layout. On the board the
        // separate wire header must also bind an explicitly typed trace ABI;
        // it cannot relax the production draft wire contract.
        const auto abi_bytes = StepRead(file, 1ULL << 20);
        const std::string abi(abi_bytes.begin(), abi_bytes.end());
        const std::regex kind("\\\"artifact_kind\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"");
        const auto begin = std::sregex_iterator(abi.begin(), abi.end(), kind);
        const auto end = std::sregex_iterator();
        const std::string expected = draft_direct ? "qwen35-standalone-draft-direct-abi" :
                                                   "qwen35-standalone-draft-trace-abi";
        if (std::distance(begin, end) != 1 || (*begin)[1] != expected)
          throw std::invalid_argument("draft diagnostic wire requires its explicit ABI kind");
      }
    }
  }
  const auto control = Split(rows[3], '\t');
  if (fused) {
    if (control.size() != 2 || control[0] != "operator")
      throw std::invalid_argument("fused operator plan requires one operator record");
    plan.operator_type = control[1];
  } else {
    if (control.size() != 3 || control[0] != "tokens")
      throw std::invalid_argument("stateful plan requires sequence/past token counts");
    plan.sequence = static_cast<std::uint32_t>(StepInteger(control[1], INT32_MAX));
    plan.past = static_cast<std::uint32_t>(StepInteger(control[2], INT32_MAX, true));
    if (static_cast<std::uint64_t>(plan.sequence) + plan.past > INT32_MAX)
      throw std::invalid_argument("stateful next past length exceeds int32");
  }
  std::set<std::string> input_names, output_names;
  std::uint64_t total = 0;
  for (std::size_t i = 4; i < rows.size(); ++i) {
    const auto fields = Split(rows[i], '\t');
    const bool input = fields[0] == "input";
    if ((input && (fields.size() != 7 || !plan.outputs.empty())) ||
        (!input && (fields.size() != 5 || fields[0] != "output")))
      throw std::invalid_argument("invalid stateful tensor record or input/output ordering");
    const auto index = StepInteger(fields[1], 1023, true);
    if (index != (input ? plan.inputs.size() : plan.outputs.size()))
      throw std::invalid_argument("stateful tensor indices must be contiguous and ordered");
    const auto spec = StepTensor(fields, draft_direct && !input);
    total += HiaiTensorBytes(spec);
    if (total > (2ULL << 30)) throw std::invalid_argument("stateful target IO exceeds 2 GiB");
    if (!(input ? input_names : output_names).insert(spec.name).second)
      throw std::invalid_argument("duplicate stateful tensor name");
    if (input) {
      StepHash(fields[6]);
      const auto file = StepPayload(root, fields[5]);
      if (StepFileSize(file, 512ULL << 20) != HiaiTensorBytes(spec))
        throw std::invalid_argument("stateful input file size differs from tensor shape/dtype");
      plan.inputs.push_back({spec, file, fields[5], fields[6]});
    } else plan.outputs.push_back(spec);
  }
  if (fused) { CheckFusedPlan(plan); return plan; }
  if (draft) { CheckDraftPlan(plan); return plan; }
  const std::vector<std::uint32_t> ids_shape{1, plan.sequence};
  if (plan.inputs.empty() || plan.outputs.size() < 4 ||
      plan.inputs[0].spec.name != "input_ids" || plan.inputs[0].spec.dtype != "int32" ||
      plan.inputs[0].spec.shape != ids_shape || plan.outputs[0].name != "target_top1" ||
      plan.outputs[0].dtype != "int32" || plan.outputs[0].shape != ids_shape ||
      plan.outputs[1].name != "target_features" || plan.outputs[1].dtype != "float16" ||
      plan.outputs[1].shape.size() != 3 || plan.outputs[1].shape[0] != 1 ||
      plan.outputs[1].shape[1] != plan.sequence)
    throw std::invalid_argument("step plan is not the explicit-state Qwen target IO contract");
  if ((!plan.past && plan.inputs.size() != 1) ||
      (plan.past && plan.inputs.size() + 1 != plan.outputs.size()))
    throw std::invalid_argument("prefill must initialize state in graph; decode must supply every state input");
  std::map<std::string, std::set<std::string>> layers;
  for (std::size_t i = 2; i < plan.outputs.size(); ++i) {
    const auto& out = plan.outputs[i];
    std::smatch match;
    if (!std::regex_match(out.name, match, std::regex("layer_([0-9]+)_(conv_history|recurrent|key|value)_out")))
      throw std::invalid_argument("invalid target state output name");
    const std::string layer = match[1], role = match[2];
    layers[layer].insert(role);
    const auto rank = role == "conv_history" ? 3U : 4U;
    if (out.shape.size() != rank || out.shape[0] != 1 ||
        (out.dtype != "float16" && !(role == "recurrent" && out.dtype == "float32")))
      throw std::invalid_argument("invalid target state output shape/dtype");
    if ((role == "key" || role == "value") && out.shape[2] != plan.past + plan.sequence)
      throw std::invalid_argument("KV output length differs from next past length");
    if (plan.past) {
      const auto& in = plan.inputs[i - 1].spec;
      auto expected_shape = out.shape;
      if (role == "key" || role == "value") expected_shape[2] = plan.past;
      if (in.name != out.name.substr(0, out.name.size() - 4) || in.dtype != out.dtype ||
          in.shape != expected_shape)
        throw std::invalid_argument("state input/output order, shape or dtype differs");
    }
  }
  for (const auto& entry : layers)
    if (entry.second != std::set<std::string>{"conv_history", "recurrent"} &&
        entry.second != std::set<std::string>{"key", "value"})
      throw std::invalid_argument("each target layer must expose its complete state pair");
  return plan;
}
void StepTensorJson(std::ostream& out, std::size_t index, const HiaiTensorSpec& spec,
                    const std::string& path, const std::string& hash) {
  out << "{\"index\":" << index << ",\"name\":\"" << spec.name << "\",\"dtype\":\""
      << spec.dtype << "\",\"shape\":[";
  for (std::size_t i = 0; i < spec.shape.size(); ++i) { if (i) out << ','; out << spec.shape[i]; }
  out << "],\"bytes\":" << HiaiTensorBytes(spec) << ",\"path\":\"" << JsonEscape(path)
      << "\",\"sha256\":\"" << hash << "\"}";
}
void WriteIoDiagnostic(std::ostream& out, int fill, const HiaiStatefulExecutor* executor) {
  if (fill < 0) return;
  const HiaiIoDiagnostic empty;
  const auto& diagnostic = executor ? executor->io_diagnostic() : empty;
  out << ",\"io_diagnostic\":{\"enabled\":true,\"output_fill_byte\":" << fill
      << ",\"process_called\":" << (diagnostic.process_called ? "true" : "false")
      << ",\"input_readback_point\":\"AFTER_ALL_INPUT_COPIES_AND_OUTPUT_FILL_BEFORE_PROCESS\",\"inputs\":[";
  for (std::size_t i = 0; i < diagnostic.inputs.size(); ++i) {
    if (i) out << ',';
    const auto& input = diagnostic.inputs[i];
    out << "{\"index\":" << input.index << ",\"logical_bytes\":" << input.logical_bytes
        << ",\"allocated_bytes\":" << input.allocated_bytes
        << ",\"readback_sha256\":\"" << input.readback_sha256 << "\",\"readback_matches_source\":"
        << (input.readback_matches_source ? "true" : "false") << '}';
  }
  out << "],\"outputs\":[";
  for (std::size_t i = 0; i < diagnostic.outputs.size(); ++i) {
    if (i) out << ',';
    const auto& output = diagnostic.outputs[i];
    out << "{\"index\":" << output.index << ",\"logical_bytes\":" << output.logical_bytes
        << ",\"allocated_bytes_before\":" << output.allocated_bytes_before
        << ",\"allocated_bytes_after\":" << output.allocated_bytes_after
        << ",\"checked_after_process\":" << (output.checked_after_process ? "true" : "false")
        << ",\"object_same\":" << (output.object_same ? "true" : "false")
        << ",\"buffer_same\":" << (output.buffer_same ? "true" : "false")
        << ",\"fill_count_available\":" << (output.fill_count_available ? "true" : "false")
        << ",\"fill_byte_count\":" << output.fill_byte_count << '}';
  }
  out << "]}";
}
int RunTensorStep(const std::string& path, const std::string& output,
                    const std::string& version, const std::string& ddk,
                    const std::string& firmware, std::uint32_t timeout, bool fused = false,
                    int diagnostic_output_fill_byte = -1, bool draft = false,
                    const std::string& model_load_api = "legacy-buffer") {
  const auto process_id = static_cast<std::uint64_t>(::getpid());
  const auto destination = std::filesystem::absolute(output).lexically_normal();
  NoSymlinks(destination);
  if (std::filesystem::exists(destination)) throw std::runtime_error("stateful output already exists");
  if (!std::filesystem::is_directory(destination.parent_path()))
    throw std::invalid_argument("stateful output parent directory must already exist");
  const auto plan = ReadStepPlan(path, fused, draft);
  if (diagnostic_output_fill_byte >= 0 && !draft && (!fused || plan.operator_type != "ChunkGatedDeltaRule"))
    throw std::invalid_argument("--output-fill-byte is restricted to fused ChunkGatedDeltaRule probes");
  std::vector<HiaiTensorSpec> input_specs;
  std::vector<std::vector<std::uint8_t>> input_bytes;
  for (const auto& input : plan.inputs) input_specs.push_back(input.spec);
  // Reject input tampering before any device load, for either execution scope.
  for (std::size_t i = 0; i < plan.inputs.size(); ++i) {
    const auto& input = plan.inputs[i];
    const auto bytes = StepRead(input.file, HiaiTensorBytes(input.spec));
    if (BytesHash(bytes) != input.sha256) throw std::runtime_error("stateful input SHA256 mismatch");
    if (fused && input.spec.name == "effective_length") {
      const std::uint32_t length = static_cast<std::uint32_t>(bytes[0]) |
                                   (static_cast<std::uint32_t>(bytes[1]) << 8);
      if (!length || length > plan.sequence)
        throw std::invalid_argument("fused GDN probe effective_length must be within 1..sequence length");
    }
    input_bytes.push_back(bytes);
  }
  if (draft) CheckDraftInputs(input_bytes, plan.sequence);
  std::unique_ptr<HiaiStatefulExecutor> executor;
  std::string failure_stage = "LOAD_OR_SETUP";
  try {
    executor = std::make_unique<HiaiStatefulExecutor>(plan.model, version, input_specs,
                                                     plan.outputs, timeout, fused, diagnostic_output_fill_byte, draft, model_load_api);
    failure_stage = "INPUT_BINDING";
    for (std::size_t i = 0; i < input_bytes.size(); ++i) executor->SetInput(i, input_bytes[i]);
    failure_stage = "PROCESS";
    executor->Execute();
  } catch (const std::exception& error) {
    if (diagnostic_output_fill_byte >= 0 && executor && !executor->io_diagnostic().process_called)
      failure_stage = "INPUT_BINDING";
    if ((fused || draft) && std::filesystem::create_directory(destination)) {
      const auto* runtime_error = dynamic_cast<const HiaiRuntimeError*>(&error);
      const auto runtime_stage = runtime_error ? runtime_error->stage() :
          executor ? executor->runtime_stage() : failure_stage;
      std::ostringstream failed;
      failed << "{\"schema_version\":1,\"status\":\"RUNTIME_FAILED\","
             << "\"process_id\":" << process_id << ','
             << "\"scope\":\"" << (draft ? "standalone-draft-step" : "fused-operator-probe")
             << "\",\"evidence_kind\":\"" << kEvidenceKind
             << "\",\"operator_type\":\"" << plan.operator_type
             << "\",\"model_load_api\":\"" << model_load_api
             << "\",\"runtime_stage\":\"" << JsonEscape(runtime_stage)
             << "\",\"failure_stage\":\"" << failure_stage << "\",\"reason\":\""
             << JsonEscape(error.what()) << "\",\"expected_runtime_version\":\"" << JsonEscape(version)
             << "\",\"ddk_id\":\"" << JsonEscape(ddk) << "\",\"firmware_id\":\"" << JsonEscape(firmware)
             << "\",\"step_plan_sha256\":\"" << plan.plan_hash
             << "\",\"" << (draft ? "draft_abi_sha256" : "operator_abi_sha256") << "\":\"" << plan.abi_hash
             << "\",\"model_sha256\":\"" << plan.model_hash
             << "\",\"target_calls\":0,\"draft_calls\":0,\"accuracy\":\"NOT_EVALUATED\","
             << "\"performance\":\"NOT_EVALUATED\",\"runtime_placement\":\"UNVERIFIED\",\"outputs\":[]";
      WriteIoDiagnostic(failed, diagnostic_output_fill_byte, executor.get());
      failed << '}';
      AtomicWrite(destination / "result.json", failed.str());
    }
    throw;
  }
  // A failed write leaves no success result.json; existing output is never replaced.
  if (!std::filesystem::create_directory(destination))
    throw std::runtime_error("cannot create new stateful output directory");
  std::ostringstream report;
  report << "{\"schema_version\":1,\"status\":\"EXECUTED\",\"process_id\":" << process_id
         << ",\"scope\":\""
         << (draft ? "standalone-draft-step" : fused ? "fused-operator-probe" : "stateful-target-step") << "\","
         << "\"evidence_kind\":\"" << kEvidenceKind << "\",\"runtime_version\":\""
         << JsonEscape(executor->runtime_version()) << "\",\"ddk_id\":\"" << JsonEscape(ddk)
         << "\",\"firmware_id\":\"" << JsonEscape(firmware) << "\",\"step_plan_sha256\":\""
         << plan.plan_hash << "\",\"" << (draft ? "draft_abi_sha256" : fused ? "operator_abi_sha256" : "state_abi_sha256")
         << "\":\"" << plan.abi_hash << "\",\"model_sha256\":\"" << plan.model_hash
         << "\",\"model_load_api\":\"" << model_load_api
         << "\",\"runtime_stage\":\"" << JsonEscape(executor->runtime_stage())
         << "\",\"sequence_length\":" << plan.sequence;
  if (draft)
    report << ",\"state_policy\":\"prefix-recompute\",\"target_calls\":0,\"draft_calls\":1,";
  else if (fused)
    report << ",\"operator_type\":\"" << plan.operator_type
           << "\",\"operator_calls\":1,\"target_calls\":0,\"draft_calls\":0,";
  else
    report << ",\"past_length\":" << plan.past << ",\"next_past_length\":" << plan.past + plan.sequence
           << ",\"state_policy\":\"explicit\",\"target_calls\":1,\"draft_calls\":0,";
  report << "\"accuracy\":\"NOT_EVALUATED\",\"performance\":\"NOT_EVALUATED\","
         << "\"runtime_placement\":\"UNVERIFIED\",\"dtype_source\":\"hash-bound-plan\",\"inputs\":[";
  for (std::size_t i = 0; i < plan.inputs.size(); ++i) {
    if (i) report << ',';
    const auto& input = plan.inputs[i];
    StepTensorJson(report, i, input.spec, input.path, input.sha256);
  }
  report << "],\"outputs\":[";
  for (std::size_t i = 0; i < plan.outputs.size(); ++i) {
    if (i) report << ',';
    std::ostringstream name;
    name << "output_" << std::setfill('0') << std::setw(4) << i << ".bin";
    const auto bytes = executor->CopyOutput(i);
    std::ofstream file(destination / name.str(), std::ios::binary);
    file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    file.flush();
    if (!file) throw std::runtime_error("could not write stateful output tensor");
    StepTensorJson(report, i, plan.outputs[i], name.str(), BytesHash(bytes));
  }
  report << ']';
  WriteIoDiagnostic(report, diagnostic_output_fill_byte, executor.get());
  report << '}';
  AtomicWrite(destination / "result.json", report.str());
  std::cout << "EXECUTED: " << (draft ? "standalone draft step" : fused ? "fused operator probe" : "stateful target step")
            << "; report=" << (destination / "result.json") << '\n';
  return 0;
}
void Help() {
  std::cout << "qwen35_dflash_hiai --probe\n"
            << "qwen35_dflash_hiai --probe-model-description MAIN.omc (at most 64 MiB; no Load/Process)\n"
            << "qwen35_dflash_hiai --target TARGET.omc --target-sha256 HEX "
            << "--draft DRAFT.omc --draft-sha256 HEX --expected-runtime-version VERSION "
            << "--ddk-id VERSION --firmware-id VERSION "
            << "--prompt-token-ids CSV --output REPORT.json "
            << "[--mode generate|benchmark|smoke --generation-mode dflash|ordinary --max-new-tokens 32 --max-draft-tokens 15 --pad-token-id 248044 "
            << "--eos-token-ids 248044,248046 --feature-width 16384 --vocab-size 248320]\n";
  std::cout << "Smoke diagnostics only: [--capture-target-features 0|1] (at most 65536 bytes)\n";
  std::cout << "qwen35_dflash_hiai --stateful-step STEP.tsv --output NEW_DIRECTORY "
            << "--expected-runtime-version VERSION --ddk-id ID --firmware-id ID [--timeout-ms 300000]\n";
  std::cout << "qwen35_dflash_hiai --fused-op-step STEP.tsv --output NEW_DIRECTORY "
            << "--expected-runtime-version VERSION --ddk-id ID --firmware-id ID [--timeout-ms 300000]\n";
  std::cout << "GDN probe I/O diagnostics only: [--output-fill-byte 0..255]\n";
  std::cout << "qwen35_dflash_hiai --draft-step STEP.tsv --output NEW_DIRECTORY "
            << "--expected-runtime-version VERSION --ddk-id ID --firmware-id ID [--timeout-ms 300000] "
            << "[--model-load-api legacy-buffer|file-v2]\n"
            << "file-v2 uses synchronous IModelManager::Run with no native timeout; bound the process externally.\n";
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
    if (argc == 3 && std::string(argv[1]) == "--probe-model-description") {
      const auto probe = ProbeHiaiModelDescription(argv[2]);
      std::cout << "{\"schema_version\":1,\"artifact_kind\":\"qwen35-hiai-model-description-probe\","
                << "\"status\":\"INSPECTED\",\"evidence_kind\":\"" << kEvidenceKind
                << "\",\"model_path\":\"" << JsonEscape(probe.model_path)
                << "\",\"model_sha256\":\"" << probe.model_sha256
                << "\",\"model_bytes\":" << probe.model_bytes
                << ",\"runtime_version\":\"" << JsonEscape(probe.runtime.version)
                << "\",\"npu_supported\":" << (probe.runtime.npu_supported ? "true" : "false")
                << ",\"process_id\":" << static_cast<long long>(getpid())
                << ",\"load_called\":false,\"process_called\":false,\"observations\":[";
      for (std::size_t i = 0; i < probe.observations.size(); ++i) {
        if (i) std::cout << ',';
        const auto& observation = probe.observations[i];
        std::cout << "{\"order\":\"" << JsonEscape(observation.order) << "\",\"steps\":[";
        for (std::size_t j = 0; j < observation.steps.size(); ++j) {
          if (j) std::cout << ',';
          const auto& step = observation.steps[j];
          std::cout << "{\"operation\":\"" << JsonEscape(step.operation)
                    << "\",\"setter_status\":" << step.setter_status
                    << ",\"model_path\":\"" << JsonEscape(step.model_path)
                    << "\",\"buffer_present\":" << (step.buffer_present ? "true" : "false")
                    << ",\"buffer_matches_input\":" << (step.buffer_matches_input ? "true" : "false")
                    << ",\"model_bytes\":" << step.model_bytes << '}';
        }
        std::cout << "]}";
      }
      std::cout << "]}\n";
      return 0;
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
    if (args.count("--stateful-step") || args.count("--fused-op-step") || args.count("--draft-step")) {
      const bool fused = args.count("--fused-op-step");
      const bool draft_step = args.count("--draft-step");
      const auto plan = take(draft_step ? "--draft-step" : fused ? "--fused-op-step" : "--stateful-step"), output = take("--output");
      const auto version = take("--expected-runtime-version");
      const auto ddk = take("--ddk-id"), firmware = take("--firmware-id");
      const auto timeout = StepInteger(take("--timeout-ms", "300000"), UINT32_MAX);
      std::string model_load_api = "legacy-buffer";
      if (args.count("--model-load-api")) {
        if (!draft_step) throw std::invalid_argument("--model-load-api requires --draft-step");
        model_load_api = take("--model-load-api");
        if (model_load_api != "legacy-buffer" && model_load_api != "file-v2")
          throw std::invalid_argument("--model-load-api must be legacy-buffer or file-v2");
      }
      int output_fill_byte = draft_step ? 0 : -1;
      if (args.count("--output-fill-byte")) {
        if (!fused) throw std::invalid_argument("--output-fill-byte requires --fused-op-step");
        output_fill_byte = static_cast<int>(StepInteger(take("--output-fill-byte"), 255, true));
      }
      if (plan.empty() || output.empty() || version.empty() || ddk.empty() || firmware.empty())
        throw std::invalid_argument("stateful plan, output and runtime/toolchain identities are required");
      if (!args.empty()) throw std::invalid_argument(std::string("unknown ") +
          (fused ? "fused operator" : "stateful") + " option " + args.begin()->first);
      return RunTensorStep(plan, output, version, ddk, firmware, static_cast<std::uint32_t>(timeout), fused,
                           output_fill_byte, draft_step, model_load_api);
    }
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
    const bool capture_option_present = args.count("--capture-target-features") != 0;
    const auto capture_option = take("--capture-target-features", "0");
    if (capture_option != "0" && capture_option != "1")
      throw std::invalid_argument("--capture-target-features must be 0 or 1");
    if (capture_option_present && mode != "smoke")
      throw std::invalid_argument("--capture-target-features requires --mode smoke");
    const bool capture_target_features = capture_option == "1";
    const auto generation_mode = take("--generation-mode", "dflash");
    if (mode != "generate" && mode != "benchmark" && mode != "smoke")
      throw std::invalid_argument("unknown --mode");
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
                                         static_cast<std::uint32_t>(timeout), capture_target_features);
    std::ostringstream report;
    report << std::setprecision(17)
           << "{\"schema_version\":2,\"abi_version\":2,\"feature_dtype\":\"torch.float16\",\"status\":\""
           << (mode == "smoke" ? "EXECUTION_COMPLETE" : "PASS") << "\",\"scope\":\""
           << (mode == "smoke" ? "target-draft-execution" :
               mode == "benchmark" ? "paired-greedy-protocol" : "single-generation") << "\","
           << "\"evidence_kind\":\"" << kEvidenceKind << "\","
           << "\"target\":\"cpro-x90\",\"runtime\":\"HarmonyOS HIAI synchronous Process\","
           << "\"runtime_version\":\"" << JsonEscape(executor.runtime_version()) << "\""
           << ",\"ddk_id\":\"" << JsonEscape(ddk) << "\",\"firmware_id\":\"" << JsonEscape(firmware)
           << "\",\"target_sha256\":\"" << target_hash << "\",\"draft_sha256\":\"" << draft_hash
           << "\",\"cpu_fallback_requested\":false,\"state_policy\":\"recompute\","
           << "\"sequence_length\":" << executor.sequence_length()
           << ",\"pad_token_id\":" << options.pad_token_id
           << ",\"prompt_token_ids\":";
    WriteTokenIds(report, prompt);
    report << ",\"eos_token_ids\":"; WriteTokenIds(report, options.eos_token_ids);
    if (mode == "smoke") {
      // Exercise both graphs independently of generation/EOS scheduling. This
      // validates execution only; these raw top-1 arrays are not generated text.
      const auto& result = executor.Execute(prompt, options.pad_token_id, true);
      report << ",\"mode\":\"smoke\",\"accuracy\":\"NOT_EVALUATED\","
             << "\"performance\":\"NOT_EVALUATED\",\"device_execution\":\"PASS\","
             << "\"execution\":{\"target_calls\":1,\"draft_calls\":1,\"target_top1\":";
      WriteTokenIds(report, result.target_top1);
      report << ",\"draft_top1\":";
      WriteTokenIds(report, result.draft_top1);
      if (capture_target_features) {
        const auto& features = executor.target_features();
        report << ",\"target_features\":{\"dtype\":\"float16\",\"encoding\":\"hex-le\",\"shape\":[1,"
               << executor.sequence_length() << ',' << feature_width << "],\"bytes\":"
               << features.size() << ",\"data\":";
        WriteHex(report, features);
        report << '}';
      }
      report << '}';
    } else {
      report << ",\"timing_scope\":\"synchronous model loop; excludes load/tokenization\","
             << "\"max_new_tokens\":" << options.max_new_tokens;
    }
    if (mode == "benchmark") {
      const auto result = qwen35::dflash::BenchmarkPair(executor, prompt, options, 3, 10);
      report << ",\"ordinary\":"; WriteBenchmark(report, result.ordinary);
      report << ",\"dflash\":"; WriteBenchmark(report, result.dflash);
      report << ",\"token_id_mismatches\":" << result.token_id_mismatches
             << ",\"eos_mismatches\":" << result.eos_mismatches;
    } else if (mode == "generate") {
      const auto selected = generation_mode == "ordinary" ? GenerationMode::kOrdinary : GenerationMode::kDFlash;
      const auto result = GenerateOnce(executor, prompt, selected, options);
      report << ",\"generation_mode\":\"" << ModeName(selected) << "\",\"generation\":";
      WriteMeasurement(report, result, 0);
    }
    report << '}';
    AtomicWrite(output, report.str());
    std::cout << (mode == "smoke" ? "EXECUTION_COMPLETE: " : "PASS: ")
              << mode << "; report=" << output << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "qwen35_dflash_hiai: " << error.what() << '\n'; return 1;
  }
}
