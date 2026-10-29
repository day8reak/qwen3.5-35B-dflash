"""Emit a standalone GE feature-branch template, never load an original target.

The caller may transplant only the eight Cast nodes and one ConcatD node after
schema-bound validation. This generator neither rewrites original protobufs nor
claims feature provenance, numerical correctness, compilation or device success.
"""
from __future__ import annotations


FEATURE_PREFIX = "__dflash_features"
INPUT_NAMES = tuple(f"{FEATURE_PREFIX}/input_{index}" for index in range(8))
CAST_NAMES = tuple(f"{FEATURE_PREFIX}/cast_{index}" for index in range(8))
CONCAT_NAME = f"{FEATURE_PREFIX}/concat"


def emit_feature_branch_cpp(sequence_length: int) -> str:
    """Generate an 8Data to 8Cast(FP16) to ConcatD(axis=2) to 1NetOutput template.

    Main takes exactly branch.air and native.json. Output AIR is created
    exclusively; no input AIR, checkpoint or sidecar path is accepted.
    sequence_length=-1 retains an explicitly dynamic descriptor.
    """
    if type(sequence_length) is not int or sequence_length not in {-1, *range(1, 65)}:
        raise ValueError("feature branch sequence length must be -1 or an integer in 1..64")
    return _CPP.replace("@SEQUENCE_LENGTH@", str(sequence_length))


_CPP = r"""#include <cstdint>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>
#include "graph/graph.h"
#include "graph/operator.h"
#include "graph/operator_factory.h"
#include "graph/operator_reg.h"
#include "graph/tensor.h"
#include "graph/types.h"

// Reuse the repository's host Data IR registration for Graph::SetInputs.
// This declares ports only; it neither installs nor implements a kernel.
namespace ge {
REG_OP(Data)
    .INPUT(x, TensorType({DT_FLOAT16, DT_INT32, DT_INT64, DT_BOOL, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_INT32, DT_INT64, DT_BOOL, DT_FLOAT}))
    .ATTR(index, Int, 0)
    .OP_END_FACTORY_REG(Data)
}

namespace {
constexpr int64_t kSequence = @SEQUENCE_LENGTH@;
constexpr const char *kPrefix = "__dflash_features/";
constexpr const char *kConcat = "__dflash_features/concat";
int64_t save_status = -1;
int64_t reload_status = -1;
void Require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
void Check(ge::graphStatus status, const char *message) {
  if (status != ge::GRAPH_SUCCESS)
    throw std::runtime_error(std::string(message) + " status=" + std::to_string(status));
}
std::string Name(const char *kind, size_t index) {
  return std::string(kPrefix) + kind + "_" + std::to_string(index);
}
class TemplateOp : public ge::Operator {
 public:
  TemplateOp(const char *name, const char *type, bool data = false)
      : ge::Operator(name, type) {
    if (data) InputRegister("x");
    OutputRegister("y");
  }
  void RegisterCast() { InputRegister("x"); }
  void RegisterConcat() { DynamicInputRegister("x", uint32_t{8}, true); }
};
void VerifyDesc(const ge::TensorDesc &desc, const std::vector<int64_t> &shape, ge::DataType dtype) {
  Require(desc.GetShape().GetDims() == shape, "branch shape differs");
  Require(desc.GetDataType() == dtype, "branch dtype differs");
  Require(desc.GetFormat() == ge::FORMAT_ND, "branch format differs");
}
void VerifyParent(const ge::GNode &node, int32_t input, const std::string &want) {
  auto parent = node.GetInDataNodesAndPortIndexs(input);
  Require(parent.first != nullptr && parent.second == 0, "branch producer port differs");
  ge::AscendString name;
  Check(parent.first->GetName(name), "producer GetName");
  Require(name.GetString() != nullptr && std::string(name.GetString()) == want,
          "branch producer name differs");
}
void Verify(ge::Graph &graph) {
  const std::vector<int64_t> input_shape{1, kSequence, 2048};
  const std::vector<int64_t> feature_shape{1, kSequence, 16384};
  auto nodes = graph.GetDirectNode();
  Require(nodes.size() == 18, "template must contain exactly 18 nodes");
  size_t netoutputs = 0;
  for (const auto &node : nodes) {
    ge::AscendString type;
    Check(node.GetType(type), "node GetType");
    if (type.GetString() != nullptr && std::string(type.GetString()) == "NetOutput") {
      ++netoutputs;
      Require(node.GetInputsSize() == 1, "template must have one output");
      ge::TensorDesc desc;
      Check(node.GetInputDesc(0, desc), "NetOutput GetInputDesc");
      VerifyDesc(desc, feature_shape, ge::DT_FLOAT16);
      VerifyParent(node, 0, kConcat);
    }
  }
  Require(netoutputs == 1, "template must have one NetOutput");
  for (size_t index = 0; index < 8; ++index) {
    auto data = graph.FindNodeByName(ge::AscendString(Name("input", index).c_str()));
    auto cast = graph.FindNodeByName(ge::AscendString(Name("cast", index).c_str()));
    Require(data != nullptr && cast != nullptr, "template Data/Cast not found");
    ge::TensorDesc desc;
    Check(data->GetOutputDesc(0, desc), "Data GetOutputDesc");
    VerifyDesc(desc, input_shape, ge::DT_FLOAT);
    Check(cast->GetInputDesc(0, desc), "Cast GetInputDesc");
    VerifyDesc(desc, input_shape, ge::DT_FLOAT);
    Check(cast->GetOutputDesc(0, desc), "Cast GetOutputDesc");
    VerifyDesc(desc, input_shape, ge::DT_FLOAT16);
    VerifyParent(*cast, 0, Name("input", index));
  }
  auto concat = graph.FindNodeByName(ge::AscendString(kConcat));
  Require(concat != nullptr && concat->GetInputsSize() == 8, "ConcatD input count differs");
  ge::TensorDesc desc;
  Check(concat->GetOutputDesc(0, desc), "ConcatD GetOutputDesc");
  VerifyDesc(desc, feature_shape, ge::DT_FLOAT16);
  for (int32_t index = 0; index < 8; ++index)
    VerifyParent(*concat, index, Name("cast", static_cast<size_t>(index)));
}
void Result(const char *path, const char *status) {
  std::ofstream stream(path);
  stream << "{\"status\":\"" << status << "\",\"sequence_length\":" << kSequence
         << ",\"save_status\":" << save_status << ",\"reload_status\":" << reload_status
         << ",\"data_count\":8,\"cast_count\":8,\"concat_count\":1,"
            "\"model_execution\":\"NOT_RUN\",\"scope\":\"Standalone feature branch metadata only\"}\n";
}
}  // namespace

int main(int argc, char **argv) {
  if (argc != 3) return 10;
  bool native_owned = false;
  try {
    int fd = open(argv[1], O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
    Require(fd >= 0, "output AIR already exists or cannot be reserved");
    close(fd);
    fd = open(argv[2], O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
    Require(fd >= 0, "native receipt already exists or cannot be reserved");
    close(fd);
    native_owned = true;
    const std::vector<int64_t> input_shape{1, kSequence, 2048};
    const ge::TensorDesc fp32(ge::Shape(input_shape), ge::FORMAT_ND, ge::DT_FLOAT);
    const ge::TensorDesc fp16(ge::Shape(input_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
    const ge::TensorDesc features(ge::Shape({1, kSequence, 16384}), ge::FORMAT_ND, ge::DT_FLOAT16);
    std::vector<ge::Operator> inputs;
    std::vector<TemplateOp> casts;
    for (size_t index = 0; index < 8; ++index) {
      const auto input_name = Name("input", index);
      TemplateOp data(input_name.c_str(), "Data", true);
      data.SetAttr("index", static_cast<int64_t>(index));
      Check(data.UpdateInputDesc("x", fp32), "Data UpdateInputDesc");
      Check(data.UpdateOutputDesc("y", fp32), "Data UpdateOutputDesc");
      inputs.push_back(data);
      const auto cast_name = Name("cast", index);
      TemplateOp cast(cast_name.c_str(), "Cast");
      cast.RegisterCast();
      cast.SetInput("x", data, uint32_t{0});
      cast.SetAttr("dst_type", static_cast<int64_t>(ge::DT_FLOAT16));
      Check(cast.UpdateInputDesc("x", fp32), "Cast UpdateInputDesc");
      Check(cast.UpdateOutputDesc("y", fp16), "Cast UpdateOutputDesc");
      casts.push_back(cast);
    }
    TemplateOp concat(kConcat, "ConcatD");
    concat.RegisterConcat();
    concat.SetAttr("concat_dim", int64_t{2});
    concat.SetAttr("N", int64_t{8});
    for (uint32_t index = 0; index < 8; ++index) {
      concat.SetInput("x", index, casts[index], "y");
      Check(concat.UpdateDynamicInputDesc("x", index, fp16), "ConcatD UpdateDynamicInputDesc");
    }
    Check(concat.UpdateOutputDesc("y", features), "ConcatD UpdateOutputDesc");
    ge::Graph graph("__dflash_features_template");
    graph.SetInputs(inputs);
    Require(graph.IsValid(), "template SetInputs failed");
    graph.SetOutputs(std::vector<std::pair<ge::Operator, std::vector<size_t>>>{{concat, {0}}});
    Verify(graph);
    save_status = static_cast<int64_t>(graph.SaveToFile(argv[1]));
    Check(static_cast<ge::graphStatus>(save_status), "template SaveToFile");
    ge::Graph reloaded("verify_feature_branch");
    reload_status = static_cast<int64_t>(reloaded.LoadFromFile(argv[1]));
    Check(static_cast<ge::graphStatus>(reload_status), "template LoadFromFile");
    Verify(reloaded);
    Result(argv[2], "BRANCH_TEMPLATE_EXPORTED");
    std::cout << "BRANCH_TEMPLATE_EXPORTED: no original target loaded or model executed" << std::endl;
    return 0;
  } catch (const std::exception &error) {
    if (native_owned) Result(argv[2], "FAILED");
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
"""

