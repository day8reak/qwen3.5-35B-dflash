// Host-only AIR serialization smoke. This defines the IR ports of existing
// Data/Const/Add operations; it does not provide or execute an NPU kernel.
// Build against the real CANN graph libraries, not lib64/stub or devlib stubs.
#include <graph/graph.h>
#include <graph/operator_reg.h>
#include <graph/tensor.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ge {
REG_OP(Data)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .ATTR(index, Int, 0)
    .OP_END_FACTORY_REG(Data)

REG_OP(Const)
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .ATTR(value, Tensor, Tensor())
    .OP_END_FACTORY_REG(Const)

REG_OP(Add)
    .INPUT(x1, TensorType({DT_FLOAT16}))
    .INPUT(x2, TensorType({DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .OP_END_FACTORY_REG(Add)
}  // namespace ge

namespace {
constexpr std::array<std::uint16_t, 4> kBiasBits{0x3c00, 0x4000, 0x4200, 0x4400};

void Check(ge::graphStatus status, const char* action) {
  if (status != ge::GRAPH_SUCCESS) {
    throw std::runtime_error(std::string(action) + " failed, graph status=" +
                             std::to_string(status));
  }
}

void CheckDesc(const ge::TensorDesc& desc, const char* name) {
  if (desc.GetDataType() != ge::DT_FLOAT16 || desc.GetFormat() != ge::FORMAT_ND ||
      desc.GetShape().GetDims() != std::vector<std::int64_t>{1, 4}) {
    throw std::runtime_error(std::string(name) + " must be FP16 ND [1,4]");
  }
}

ge::Operator Find(ge::Graph& graph, const char* name, const char* expected_type) {
  ge::Operator op;
  Check(graph.FindOpByName(name, op), name);
  ge::AscendString actual_type;
  Check(op.GetOpType(actual_type), "GetOpType");
  if (actual_type.GetString() == nullptr ||
      std::strcmp(actual_type.GetString(), expected_type) != 0) {
    throw std::runtime_error(std::string(name) + " has an unexpected op type");
  }
  return op;
}

void VerifyGraph(ge::Graph& graph) {
  if (!graph.IsValid()) {
    throw std::runtime_error("graph is invalid");
  }
  auto x = Find(graph, "x", "Data");
  auto bias = Find(graph, "bias", "Const");
  auto y = Find(graph, "y", "Add");
  CheckDesc(x.GetOutputDesc(0), "x output");
  CheckDesc(bias.GetOutputDesc(0), "bias output");
  CheckDesc(y.GetInputDesc(0), "y first input");
  CheckDesc(y.GetInputDesc(1), "y second input");
  CheckDesc(y.GetOutputDesc(0), "y output");
  std::int64_t input_index = -1;
  Check(x.GetAttr("index", input_index), "Data index");
  if (input_index != 0) {
    throw std::runtime_error("Data input index must be zero");
  }
  ge::Tensor value;
  Check(bias.GetAttr("value", value), "Const value");
  CheckDesc(value.GetTensorDesc(), "bias value");
  if (value.GetSize() != sizeof(kBiasBits) || value.GetData() == nullptr ||
      std::memcmp(value.GetData(), kBiasBits.data(), sizeof(kBiasBits)) != 0) {
    throw std::runtime_error("bias FP16 bytes did not survive serialization");
  }
}

int Export(const char* filename) {
  namespace fs = std::filesystem;
  const fs::path path(filename);
  if (path.empty() || path.extension() != ".air") {
    throw std::runtime_error("output path must end in .air");
  }
  // symlink_status also detects a dangling symlink, which must not be replaced.
  std::error_code ec;
  const auto status = fs::symlink_status(path, ec);
  if (ec && ec != std::errc::no_such_file_or_directory) {
    throw std::runtime_error("cannot inspect output path: " + ec.message());
  }
  if (status.type() != fs::file_type::not_found) {
    throw std::runtime_error("refusing to overwrite an existing output path");
  }
  // CANN on the supported Linux x86_64 host and X90 both use little endian.
  const std::uint16_t endian_check = 1;
  if (*reinterpret_cast<const std::uint8_t*>(&endian_check) != 1) {
    throw std::runtime_error("this FP16 smoke requires a little-endian host");
  }

  ge::TensorDesc desc(ge::Shape({1, 4}), ge::FORMAT_ND, ge::DT_FLOAT16);
  ge::op::Data x("x");
  x.set_attr_index(0);
  Check(x.update_input_desc_x(desc), "Data input descriptor");
  Check(x.update_output_desc_y(desc), "Data output descriptor");

  ge::Tensor value(desc);
  Check(value.SetData(reinterpret_cast<const std::uint8_t*>(kBiasBits.data()),
                      sizeof(kBiasBits)), "Const tensor data");
  ge::op::Const bias("bias");
  bias.set_attr_value(value);
  Check(bias.update_output_desc_y(desc), "Const output descriptor");

  ge::op::Add y("y");
  y.set_input_x1(x).set_input_x2(bias);
  Check(y.update_input_desc_x1(desc), "Add first input descriptor");
  Check(y.update_input_desc_x2(desc), "Add second input descriptor");
  Check(y.update_output_desc_y(desc), "Add output descriptor");

  ge::Graph graph("qwen35_fp16_add_smoke");
  graph.SetInputs(std::vector<ge::Operator>{x});
  graph.SetOutputs(std::vector<ge::Operator>{y});
  VerifyGraph(graph);
  Check(graph.SaveToFile(filename), "SaveToFile");
  if (!fs::is_regular_file(path) || fs::file_size(path) == 0) {
    throw std::runtime_error("SaveToFile did not produce a nonempty regular file");
  }
  ge::Graph loaded("roundtrip");
  Check(loaded.LoadFromFile(filename), "LoadFromFile");
  VerifyGraph(loaded);
  std::cout << "AIR serialized and reloaded: " << filename << '\n'
            << "Data x + Const [1,2,3,4] -> Add y; FP16 ND [1,4].\n"
            << "Board test: x=[0,1,2,3], expected y=[1,3,5,7].\n"
            << "Serialization only; OMG conversion and board execution remain.\n";
  return 0;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " /path/to/new-output.air\n";
    return 2;
  }
  try {
    return Export(argv[1]);
  } catch (const std::exception& error) {
    std::cerr << "AIR smoke failed: " << error.what() << '\n';
    return 1;
  }
}
