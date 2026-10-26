"""Emit C++ which constructs static GE operators and serializes real AIR.

Operator schemas are taken from Ascend/torchair generated GE IR definitions at
e1118e8b0fc1b95190ab53b732c0872989550e20. No TorchAir package is imported.
The public schemas are a construction baseline; internal X90 OMG support must
be established by conversion, never inferred from successful host compilation.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .ge_ir import DTYPES, Value
from .ge_type_contracts import GE_DTYPE_IDS

SCHEMA_SOURCE = "https://github.com/Ascend/torchair/blob/e1118e8b0fc1b95190ab53b732c0872989550e20/python/torchair/_ge_concrete_graph/auto_generated_ge_raw_ops.py"
_DT = {"float16": "ge::DT_FLOAT16", "int32": "ge::DT_INT32", "int64": "ge::DT_INT64", "bool": "ge::DT_BOOL"}


def _q(text):
    return json.dumps(str(text), ensure_ascii=True)


def _ints(values):
    return "std::vector<int64_t>{" + ",".join(str(int(v)) for v in values) + "}"


def _desc(value):
    return f"Desc({_ints(value.shape)}, {_DT[value.dtype]})"


def _validate_cast_selection(node):
    """Reject source descriptor contradictions before lowering/shared outputs.

    This checks source-IR contracts, not compiler-created CastT nodes or target
    support. TopK must preserve declared descriptors when sharing its GE op.
    """
    kind, output, attrs = node.kind, node.output, node.attrs
    if kind not in {"cast", "argmax", "topk_values", "topk_indices"}:
        return
    if len(node.inputs) != 1:
        raise ValueError(f"{output.name}: {kind} requires exactly one input")
    source = node.inputs[0]
    if kind == "cast":
        if output.shape != source.shape:
            raise ValueError(f"{output.name}: Cast must preserve its input shape")
        return
    if source.dtype == "bool":
        raise ValueError(f"{output.name}: {kind} does not accept bool data")
    rank = len(source.shape)
    axis = attrs.get("axis", -1)
    if (isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer))
            or not -rank <= axis < rank):
        raise ValueError(f"{output.name}: {kind} axis must be an integer within input rank {rank}")
    axis = int(axis) % rank
    if kind == "argmax":
        if output.dtype not in {"int32", "int64"}:
            raise ValueError(f"{output.name}: ArgMax output dtype must be int32 or int64")
        if source.shape[axis] == 0:
            raise ValueError(f"{output.name}: ArgMax requires a nonempty reduction axis")
        keepdims = attrs.get("keepdims", False)
        if not isinstance(keepdims, (bool, np.bool_)):
            raise ValueError(f"{output.name}: ArgMax keepdims must be boolean")
        expected = source.shape[:axis] + ((1,) if keepdims else ()) + source.shape[axis + 1:]
    else:
        if axis != rank - 1:
            raise ValueError("GE TopKV2 lowering currently requires the last axis")
        k = attrs.get("k")
        if (isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer))
                or not 0 <= k <= min(source.shape[axis], (1 << 31) - 1)):
            raise ValueError(f"{output.name}: TopKV2 k must be an INT32 integer between zero and the axis size")
        expected_dtype = "int32" if kind == "topk_indices" else source.dtype
        if output.dtype != expected_dtype:
            raise ValueError(f"{output.name}: {kind} output dtype must be {expected_dtype}")
        expected = source.shape[:-1] + (int(k),)
    if output.shape != expected:
        raise ValueError(f"{output.name}: {kind} output shape {output.shape} differs from expected {expected}")


_HEADER = r'''// Generated host-only GE graph construction. Not an NPU implementation.
#include <graph/graph.h>
#include <graph/operator.h>
#include <graph/operator_factory.h>
#include <graph/operator_reg.h>
#include <graph/tensor.h>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

// GE Graph::SetInputs looks up a fresh prototype for every input type through
// OperatorFactory before accepting Data nodes. Registering ports on the recipe
// instances alone does not register that factory. This is a host IR definition
// of the existing Data operation; it does not install or implement a kernel.
namespace ge {
REG_OP(Data)
    .INPUT(x, TensorType({DT_FLOAT16, DT_INT32, DT_INT64, DT_BOOL}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_INT32, DT_INT64, DT_BOOL}))
    .ATTR(index, Int, 0)
    .OP_END_FACTORY_REG(Data)
}  // namespace ge

namespace {
void Require(bool ok, const std::string& why) { if (!ok) throw std::runtime_error(why); }
void Check(ge::graphStatus s, const char* action) {
  if (s != ge::GRAPH_SUCCESS) throw std::runtime_error(std::string(action) + " status=" + std::to_string(s));
}
void VerifyDataFactory() {
  std::cout << "Checking GE Data factory..." << std::endl;
  Require(ge::OperatorFactory::IsExistOp("Data"), "GE Data factory is not registered");
  auto prototype = ge::OperatorFactory::CreateOperator("q35_data_factory_probe", "Data");
  Require(!prototype.IsEmpty(), "GE Data factory returned an empty operator");
  ge::AscendString type; Check(prototype.GetOpType(type), "Data factory GetOpType");
  Require(type.GetString() != nullptr && std::string(type.GetString()) == "Data", "GE Data factory returned the wrong type");
  Require(prototype.GetInputsSize() == 1 && prototype.GetOutputsSize() == 1, "GE Data factory prototype must have one input and one output");
  // TryGetInputDesc checks InputIsSet, not whether an IR port is registered.
  // Factory prototypes have unset descriptors. Actual Data descriptors are
  // assigned during graph construction and checked by Verify afterward.
  prototype.BreakConnect();
}
ge::TensorDesc Desc(const std::vector<int64_t>& shape, ge::DataType dtype) {
  return ge::TensorDesc(ge::Shape(shape), ge::FORMAT_ND, dtype);
}
class Op : public ge::Operator {
 public:
  Op(const char* name, const char* type, std::initializer_list<const char*> inputs,
     std::initializer_list<const char*> outputs) : ge::Operator(name, type) {
    if (std::strcmp(type, "ConcatD") == 0) DynamicInputRegister("x", inputs.size());
    else for (auto port : inputs) InputRegister(port);
    if (std::strcmp(type, "Data") == 0) InputRegister("x");
    for (auto port : outputs) OutputRegister(port);
  }
};
struct Expected { std::string name; std::string type; std::vector<ge::TensorDesc> outputs; };
[[maybe_unused]] std::vector<uint8_t> Read(const char* path, size_t bytes) {
  if (!std::filesystem::is_regular_file(path) || std::filesystem::file_size(path) != bytes)
    throw std::runtime_error(std::string("weight size mismatch: ") + path);
  std::ifstream f(path, std::ios::binary);
  std::vector<uint8_t> data(bytes);
  if (!f.read(reinterpret_cast<char*>(data.data()), bytes)) throw std::runtime_error("weight read failed");
  return data;
}
[[maybe_unused]] void SetConst(Op& op, const ge::TensorDesc& desc, const std::vector<uint8_t>& bytes) {
  ge::Tensor value(desc);
  Check(value.SetData(bytes.data(), bytes.size()), "Const SetData");
  op.SetAttr("value", value);
}
void Verify(ge::Graph& graph, const std::vector<Expected>& expected) {
  if (!graph.IsValid()) throw std::runtime_error("invalid graph");
  std::map<std::string, ge::GNode> nodes;
  for (const auto& node : graph.GetAllNodes()) {
    ge::AscendString name;
    Check(node.GetName(name), "GetName");
    if (name.GetString() == nullptr) throw std::runtime_error("null node name");
    nodes.emplace(name.GetString(), node);
  }
  for (const auto& e : expected) {
    auto it = nodes.find(e.name);
    if (it == nodes.end()) throw std::runtime_error("missing compute node " + e.name);
    ge::AscendString type;
    Check(it->second.GetType(type), "GetType");
    if (type.GetString() == nullptr || type.GetString() != e.type) throw std::runtime_error("type mismatch " + e.name);
    for (size_t i = 0; i < e.outputs.size(); ++i) {
      ge::TensorDesc got;
      Check(it->second.GetOutputDesc(static_cast<int32_t>(i), got), "GetOutputDesc");
      const auto& want = e.outputs[i];
      if (got.GetShape().GetDims() != want.GetShape().GetDims() || got.GetFormat() != ge::FORMAT_ND ||
          got.GetDataType() != want.GetDataType()) throw std::runtime_error("descriptor mismatch " + e.name);
    }
  }
}
int Export(const char* filename) {
  namespace fs = std::filesystem;
  const fs::path path(filename);
  if (path.extension() != ".air") throw std::runtime_error("output must end in .air");
  std::error_code ec;
  const auto st = fs::symlink_status(path, ec);
  if (ec && ec != std::errc::no_such_file_or_directory) throw std::runtime_error(ec.message());
  if (st.type() != fs::file_type::not_found) throw std::runtime_error("refusing to overwrite output");
  const uint16_t endian = 1;
  if (*reinterpret_cast<const uint8_t*>(&endian) != 1) throw std::runtime_error("little-endian host required");
  VerifyDataFactory();
  std::vector<Expected> expected;
  std::vector<Op> ops;
'''

_FOOTER = r'''
  std::cout << "Checking constructed compute graph..." << std::endl;
  Verify(graph, expected);
  Check(graph.SaveToFile(filename), "SaveToFile");
  if (!fs::is_regular_file(path) || fs::file_size(path) == 0) throw std::runtime_error("empty AIR");
  ge::Graph loaded("roundtrip");
  Check(loaded.LoadFromFile(filename), "LoadFromFile");
  std::cout << "Checking reloaded compute graph..." << std::endl;
  Verify(loaded, expected);
  std::cout << "AIR_EXPORTED " << filename << std::endl;
  return 0;
}
}  // namespace
int main(int argc, char** argv) {
  if (argc != 2) { std::cerr << "Usage: " << argv[0] << " output.air\n"; return 2; }
  try { return Export(argv[1]); }
  catch (const std::exception& e) { std::cerr << "AIR export failed: " << e.what() << '\n'; return 1; }
}
'''


class _DiscardLines:
    def append(self, line):
        pass


def _typed_attr(literal):
    """Decode this module's closed C++ literal vocabulary for recipe emission."""
    enum = {_DT[dtype]: value for dtype, value in GE_DTYPE_IDS.items()}
    if literal in enum:
        return {"type": "dtype", "value": {v: k for k, v in _DT.items()}[literal]}
    if literal in {"true", "false"}:
        return {"type": "bool", "value": literal == "true"}
    if literal.startswith("int64_t{"):
        value = literal[len("int64_t{"):-1]
        return {"type": "int", "value": enum[value] if value in enum else int(value)}
    if literal.startswith("float{"):
        return {"type": "float", "value": float(literal[len("float{"):-1])}
    if literal.startswith("std::vector<int64_t>{"):
        value = literal[len("std::vector<int64_t>{"):-1]
        return {"type": "ints", "value": [] if not value else [int(x) for x in value.split(",")]}
    if literal.startswith("ge::AscendString("):
        return {"type": "string", "value": json.loads(literal[len("ge::AscendString("):-1])}
    raise ValueError(f"unsupported internal GE attribute literal {literal!r}")


def _opindex(reference):
    if not reference.startswith("ops[") or not reference.endswith("]"):
        raise ValueError("invalid internal operator reference")
    return int(reference[4:-1])


def _cpp_attr(attribute):
    """Render the same closed attribute vocabulary after a record rewrite."""
    kind, value = attribute["type"], attribute["value"]
    if kind == "dtype":
        return _DT[value]
    if kind == "bool":
        return "true" if value else "false"
    if kind == "int":
        return f"int64_t{{{value}}}"
    if kind == "float":
        return f"float{{{value!r}}}"
    if kind == "ints":
        return _ints(value)
    if kind == "string":
        return f"ge::AscendString({_q(value)})"
    raise ValueError(f"unsupported internal GE attribute kind {kind!r}")


def _inline_records(graph, records):
    """Rebuild inline C++ from the exact records also consumed by the recipe writer."""
    emitter = _Emitter(graph)
    bound = {}
    for index, record in enumerate(records):
        outputs = [(port["port"], Value(f"__render/{index}/{port['port']}",
                                       tuple(port["shape"]), port["dtype"])) for port in record["outputs"]]
        inputs = [(port["port"], bound[(port["src"], port["src_port"])]) for port in record["inputs"]]
        attrs = {key: _cpp_attr(value) for key, value in record["attrs"].items()}
        var = emitter.op(record["name"], record["type"], inputs, outputs, attrs)
        for port, value in outputs:
            bound[(index, port)] = value
            emitter.refs[value.name] = (var, port)
        if record["type"] == "Data":
            emitter.lines.append(f"  Check({var}.UpdateInputDesc(\"x\", {_desc(outputs[0][1])}), \"Data input descriptor\");")
        constant = record.get("constant")
        if constant is not None:
            if "data" in constant:
                payload = "std::vector<uint8_t>{" + ",".join(str(x) for x in constant["data"]) + "}"
            else:
                if constant.get("offset", 0) != 0:
                    raise ValueError("inline weight constants require zero offset")
                payload = f"Read({_q(str(Path(constant['path']).resolve()))}, {constant['size']}ULL)"
            emitter.lines.append(f"  SetConst({var}, {_desc(outputs[0][1])}, {payload});")
        if record["type"] == "FileConstant":
            path = str(Path(record["attrs"]["file_path"]["value"]).resolve())
            value = outputs[0][1]
            size = math.prod(value.shape) * DTYPES[value.dtype].itemsize
            emitter.lines.append(f"  if (!fs::is_regular_file({_q(path)}) || fs::file_size({_q(path)}) != {size}ULL) throw std::runtime_error(\"external weight size mismatch\");")
    return emitter


class _Emitter:
    def __init__(self, graph, *, recipe=False, fp16_reductions="native", fp16_row_mul="native"):
        if not isinstance(fp16_reductions, str) or fp16_reductions not in {"native", "matmul"}:
            raise ValueError("fp16_reductions must be native or matmul")
        if not isinstance(fp16_row_mul, str) or fp16_row_mul not in {"native", "compact"}:
            raise ValueError("fp16_row_mul must be native or compact")
        self.graph = graph
        self.fp16_reductions = fp16_reductions
        self.fp16_row_mul = fp16_row_mul
        self.row_mul_report = {"mode": fp16_row_mul, "graph": graph.name,
            "counts": {"total": 0, "replaced": 0, "retained": 0}, "replaced": [], "retained": []}
        self.reduction_report = {"mode": fp16_reductions, "graph": graph.name,
            "counts": {"total": 0, "replaced": 0, "retained": 0, "fp16_replaced": 0,
                       "integer_retained": 0, "scan_prefix_replaced": 0},
            "replaced": [], "retained": []}
        self.lines = _DiscardLines() if recipe else [_HEADER]
        self.refs = {}
        self.counter = 0
        self.records = []
        self.recipe = recipe
        self.inline_constants = {}
        self.pending_constants = {}
        self.constant_aliases = {}
        self.broadcast_cache = {}
        self.broadcast_bytes = 0
        self.max_broadcast_tensor_bytes = 1 << 20
        self.max_broadcast_total_bytes = 8 << 20

    def op(self, name, typ, inputs, outputs, attrs=None):
        for _, value in inputs:
            self.ensure_constant(value)
        def desc(value):
            return {"shape": list(value.shape), "dtype": value.dtype}
        self.records.append({
            "name": name, "type": typ,
            "inputs": [{"port": port, "src": _opindex(self.refs[value.name][0]),
                        "src_port": self.refs[value.name][1], **desc(value)} for port, value in inputs],
            "outputs": [{"port": port, **desc(value)} for port, value in outputs],
            "attrs": {k: _typed_attr(v) for k, v in (attrs or {}).items()}, "constant": None,
        })
        var = f"ops[{self.counter}]"
        self.counter += 1
        in_ports = ",".join(_q(port) for port, _ in inputs)
        out_ports = ",".join(_q(port) for port, _ in outputs)
        self.lines.append(f"  ops.push_back(Op({_q(name)}, {_q(typ)}, {{{in_ports}}}, {{{out_ports}}}));")
        for port, value in inputs:
            src, srcport = self.refs[value.name]
            self.lines.append(f"  {var}.SetInput({_q(port)}, {src}, {_q(srcport)});")
            self.lines.append(f"  Check({var}.UpdateInputDesc({_q(port)}, {_desc(value)}), \"input descriptor\");")
        for port, value in outputs:
            self.lines.append(f"  Check({var}.UpdateOutputDesc({_q(port)}, {_desc(value)}), \"output descriptor\");")
        for attr, literal in (attrs or {}).items():
            self.lines.append(f"  {var}.SetAttr({_q(attr)}, {literal});")
        descs = ",".join(_desc(value) for _, value in outputs)
        self.lines.append(f"  expected.push_back({{{_q(name)},{_q(typ)},{{{descs}}}}});")
        return var

    def ensure_constant(self, value):
        if value.name in self.refs:
            return
        if value.name in self.constant_aliases:
            canonical = self.constant_aliases[value.name]
            self.ensure_constant(canonical)
            self.refs[value.name] = self.refs[canonical.name]
        elif value.name in self.pending_constants:
            array = self.pending_constants.pop(value.name)
            self.materialize_constant(array, value)

    def alias_constant(self, value, canonical):
        if value.name != canonical.name:
            self.constant_aliases[value.name] = canonical
            if canonical.name in self.inline_constants:
                self.inline_constants[value.name] = self.inline_constants[canonical.name]

    def constant(self, data, dtype, name):
        array = np.asarray(data, dtype=DTYPES[dtype])
        value = Value(name, array.shape, dtype)
        self.pending_constants[name] = array
        if array.nbytes <= self.max_broadcast_tensor_bytes:
            self.inline_constants[name] = array
        return value

    def materialize_constant(self, array, value):
        var = self.op(value.name, "Const", [], [("y", value)])
        self.records[-1]["constant"] = {"data": array.tobytes()}
        if not self.recipe:
            literal = ",".join(str(x) for x in array.tobytes())
            self.lines.append(f"  SetConst({var}, {_desc(value)}, std::vector<uint8_t>{{{literal}}});")
        self.refs[value.name] = (var, "y")

    def helper(self, value, data, dtype="int64", suffix="parameter"):
        return self.constant(data, dtype, f"__ge/{value.name}/{suffix}")

    def checked_reduction_axes(self, output, source, axes):
        """Keep the ReduceSum/ReduceMean axes contract INT32 without narrowing overflow."""
        if (not isinstance(axes, (list, tuple, np.ndarray))
                or (isinstance(axes, np.ndarray) and axes.ndim != 1)):
            raise ValueError(f"{output.name}: reduction axes must be a one-dimensional integer sequence")
        checked, normalized = [], set()
        rank = len(source.shape)
        for axis in axes:
            if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)):
                raise ValueError(f"{output.name}: reduction axes must contain integers")
            axis = int(axis)
            if not -(1 << 31) <= axis < (1 << 31):
                raise ValueError(f"{output.name}: reduction axis must fit INT32")
            if not -rank <= axis < rank:
                raise ValueError(f"{output.name}: reduction axis {axis} is outside input rank {rank}")
            if axis % rank in normalized:
                raise ValueError(f"{output.name}: duplicate reduction axis {axis}")
            normalized.add(axis % rank)
            checked.append(axis)  # Preserve legal negative values and axis order.
        return checked

    def reduction_axes(self, output, source, axes):
        checked = self.checked_reduction_axes(output, source, axes)
        return self.helper(output, checked, dtype="int32", suffix="axes")

    def checked_reduction_contract(self, output, source, axes, keepdims):
        checked = self.checked_reduction_axes(output, source, axes)
        if not isinstance(keepdims, (bool, np.bool_)):
            raise ValueError(f"{output.name}: reduction keepdims must be boolean")
        if source.dtype != output.dtype:
            raise ValueError(f"{output.name}: reduction source/output dtype must match")
        reduced = {axis % len(source.shape) for axis in checked}
        expected_shape = tuple(1 if axis in reduced else dim for axis, dim in enumerate(source.shape)) if keepdims else tuple(
            dim for axis, dim in enumerate(source.shape) if axis not in reduced)
        if output.shape != expected_shape:
            raise ValueError(f"{output.name}: reduction output shape {output.shape} differs from expected {expected_shape}")
        return checked

    def emit_reduction(self, output, source, kind, axes, keepdims, *, axes_value=None, origin="graph"):
        """Apply the optional contraction policy to graph and scan reductions."""
        checked = self.checked_reduction_contract(output, source, axes, keepdims)
        keepdims = bool(keepdims)
        contract = {"name": output.name, "origin": origin, "kind": kind,
                    "source": {"name": source.name, "shape": list(source.shape), "dtype": source.dtype},
                    "output": {"name": output.name, "shape": list(output.shape), "dtype": output.dtype},
                    "axes": checked, "keepdims": keepdims}
        reason, plan = "mode_native", None
        if self.fp16_reductions == "matmul":
            if source.dtype != "float16" or output.dtype != "float16":
                reason = "non_fp16_source_or_output"
            elif len(checked) != 1:
                reason = "requires_single_axis"
            elif not source.shape or any(d <= 0 for d in source.shape):
                reason = "requires_positive_static_shape"
            else:
                from .ge_reduction_matmul import reduction_matmul_plan
                try:
                    plan = reduction_matmul_plan(source, kind=kind, axis=checked[0], keepdims=keepdims)
                except ValueError:
                    # Axes, dtype, shape and kind are already checked; a mean
                    # with an inexact reciprocal retains its native operation.
                    reason = "mean_requires_exact_fp16_reciprocal"
                else:
                    if any(d > (1 << 31) - 1 for d in (*source.shape, plan["M"], plan["K"])):
                        reason = "reshape_dimensions_exceed_int32"
                    else:
                        reason = None
        counts = self.reduction_report["counts"]
        counts["total"] += 1
        if reason is None:
            self._emit_reduction_matmul(output, source, plan)
            contract.update({key: plan[key] for key in ("axis", "K", "M", "coefficient")})
            self.reduction_report["replaced"].append(contract)
            counts["replaced"] += 1
            counts["fp16_replaced"] += 1
            counts["scan_prefix_replaced"] += int(origin == "scan_prefix")
        else:
            axes_value = axes_value if axes_value is not None else self.helper(output, checked, dtype="int32", suffix="axes")
            attrs = {"keep_dims": str(keepdims).lower()}
            if kind == "reduce_mean":
                attrs["noop_with_empty_axes"] = "true"
            var = self.op(output.name, "ReduceMean" if kind == "reduce_mean" else "ReduceSum",
                          [("x", source), ("axes", axes_value)], [("y", output)], attrs)
            self.refs[output.name] = (var, "y")
            self.reduction_report["retained"].append({**contract, "reason": reason})
            counts["retained"] += 1
            counts["integer_retained"] += int(source.dtype in {"int32", "int64"}
                                               and output.dtype == source.dtype)
        return output

    def _emit_reduction_matmul(self, output, source, plan):
        """Emit the probe-validated plan without mutating the original Graph."""
        prefix = f"__ge/{output.name}/reduction"

        def bind(value, kind, inputs, attrs):
            var = self.op(value.name, kind, inputs, [("y", value)], attrs)
            self.refs[value.name] = (var, "y")
            return value

        def reshape(value, result):
            shape = self.reshape_shape(result)
            return bind(result, "Reshape", [("x", value), ("shape", shape)],
                        {"axis": "int64_t{0}", "num_axes": "int64_t{-1}"})

        value = source
        perm = plan["permutation"]
        if perm != list(range(len(source.shape))):
            transposed = Value(prefix + "/transpose", tuple(source.shape[i] for i in perm), "float16")
            permutation = self.helper(transposed, perm, suffix="perm")
            value = bind(transposed, "Transpose", [("x", value), ("perm", permutation)], {})
        flat = Value(prefix + "/flat", (plan["M"], plan["K"]), "float16")
        value = reshape(value, flat)
        weight = self.constant(np.full((1, plan["K"]), plan["coefficient"], dtype=np.float16),
                               "float16", prefix + "/weight")
        contracted = Value(prefix + "/matmul", (plan["M"], 1), "float16")
        value = bind(contracted, "MatMul", [("x1", value), ("x2", weight)],
                     {"transpose_x1": "false", "transpose_x2": "true"})
        reshape(value, output)

    def emit_mul(self, output, left, right):
        """Compact only the two board-tested FP16 row-scale shape contracts."""
        def descriptor(value):
            return {"name": value.name, "shape": list(value.shape), "dtype": value.dtype}

        contract = {"name": output.name, "left": descriptor(left), "right": descriptor(right),
                    "output": descriptor(output)}
        reason = "mode_native"
        if self.fp16_row_mul == "compact":
            verified_shapes = {((1, 8, 2, 8), (1, 8, 2, 1)),
                               ((1, 2, 64, 8), (1, 2, 64, 1))}
            if left.dtype != "float16" or right.dtype != "float16":
                reason = "non_fp16"
            elif (left.shape, right.shape) not in verified_shapes:
                reason = "outside_verified_shapes"
            elif output.shape != left.shape or output.dtype != "float16":
                raise ValueError(f"{output.name}: row Mul output shape/dtype differs from the verified contract")
            else:
                reason = None
        if reason is None:
            rows, width = math.prod(left.shape[:-1]), left.shape[-1]
            prefix = f"__ge/{output.name}/row_mul"
            lhs = Value(prefix + "/left", (rows, width), "float16")
            rhs = Value(prefix + "/right", (rows, 1), "float16")
            product = Value(prefix + "/product", (rows, width), "float16")
            generated_names = [lhs.name, rhs.name, product.name,
                               *(f"__ge/{value.name}/shape" for value in (lhs, rhs, output))]
            if any(name in self.graph._values or name in self.refs or name in self.pending_constants
                   for name in generated_names):
                raise ValueError(f"{output.name}: row Mul generated name collision")

            def reshape(source, result):
                shape = self.reshape_shape(result)
                var = self.op(result.name, "Reshape", [("x", source), ("shape", shape)], [("y", result)],
                              {"axis": "int64_t{0}", "num_axes": "int64_t{-1}"})
                self.refs[result.name] = (var, "y")

            reshape(left, lhs)
            reshape(right, rhs)
            var = self.op(product.name, "Mul", [("x1", lhs), ("x2", rhs)], [("y", product)])
            self.refs[product.name] = (var, "y")
            reshape(product, output)
            contract.update(rows=rows, width=width, compact_shape=[rows, width])
            self.row_mul_report["replaced"].append(contract)
            self.row_mul_report["counts"]["replaced"] += 1
        else:
            var = self.op(output.name, "Mul", [("x1", left), ("x2", right)], [("y", output)])
            self.refs[output.name] = (var, "y")
            self.row_mul_report["retained"].append({**contract, "reason": reason})
            self.row_mul_report["counts"]["retained"] += 1
        self.row_mul_report["counts"]["total"] += 1

    def reshape_shape(self, output, dimensions=None):
        """Use checked INT32 dimensions for OMG's Reshape shape rewrites.

        Public GE Reshape accepts INT32 and INT64. Internal X90 converter
        traces changed the words "55 0" to "-1 0" for an INT64 dimension.
        INT32 makes -1 a signed sentinel without depending on a 64-bit rewrite.
        Tensor data and INT64 graph descriptor dimensions remain unchanged;
        the private converter implementation has not been inspected.
        """
        dimensions = tuple(output.shape if dimensions is None else dimensions)
        if dimensions != output.shape:
            raise ValueError(f"{output.name}: Reshape shape differs from output descriptor")
        if any(isinstance(d, (bool, np.bool_)) or not isinstance(d, (int, np.integer))
               or not 0 <= d <= (1 << 31) - 1 for d in dimensions):
            raise ValueError(f"{output.name}: Reshape dimensions must fit nonnegative INT32")
        return self.helper(output, dimensions, dtype="int32", suffix="shape")

    def broadcast(self, source, output, index):
        """Give Select concrete equal-shaped inputs before OMG lowers its IR."""
        if source.shape == output.shape:
            return source
        if np.broadcast_shapes(source.shape, output.shape) != output.shape:
            raise ValueError(f"{output.name}: cannot broadcast {source.shape} to {output.shape}")
        value = Value(f"__ge/{output.name}/broadcast_{index}", output.shape, source.dtype)
        constant = self.static_broadcast(source, value)
        if constant is not None:
            return constant
        shape = self.helper(value, value.shape, suffix="shape")
        var = self.op(value.name, "BroadcastTo", [("x", source), ("shape", shape)], [("y", value)])
        self.refs[value.name] = (var, "y")
        return value

    def static_broadcast(self, source, output):
        """Reuse small byte-exact Const expansions, avoiding CPUCL FP16 folding."""
        array = self.inline_constants.get(source.name)
        size = math.prod(output.shape) * DTYPES[source.dtype].itemsize
        if array is None or size > self.max_broadcast_tensor_bytes:
            return None
        key = (source.dtype, source.shape, array.tobytes(), output.shape)
        if key in self.broadcast_cache:
            return self.broadcast_cache[key]
        if self.broadcast_bytes + size > self.max_broadcast_total_bytes:
            return None
        # Copying elements preserves signed zeros and NaN payloads; no mask
        # arithmetic and no reading checkpoint/FileConstant tensors here.
        expanded = np.broadcast_to(array, output.shape).copy(order="C")
        value = self.constant(expanded, source.dtype, output.name)
        self.broadcast_cache[key] = value
        self.broadcast_bytes += size
        return value


def emit_cpp(graph, weight_paths=None, *, external_weights=False, recipe_dir=None,
             fold_constant_arithmetic=True, fp16_reductions="native", reduction_report=None,
             fp16_row_mul="native", row_mul_report=None):
    """Return a self-contained C++ AIR serializer.

    ``weight_paths`` maps canonical graph weight keys to existing raw contiguous
    little-endian files. External mode emits official FileConstant nodes; files
    must remain at the same absolute paths until OMG conversion has completed.
    Small constant arithmetic is evaluated with the original FP16 boundaries;
    bounded static Slice/Reshape operations copy their bytes before serialization.
    The complete static target RoPE chain folds atomically through its final
    cosine/sine views with the oracle's original FP16 boundaries. Standalone
    direct-constant K=1 MatMuls fold only without live node consumers; no
    partial constant prefix is introduced on a rejected region. The legacy
    fold_constant_arithmetic flag controls all of these folds.
    Descriptor-only tooling can disable folding when real weight payloads are
    unavailable; model export always enables it.
    ``fp16_reductions="matmul"`` replaces eligible FP16 single-axis reductions
    only during emission. The default native serialization is unchanged.
    An optional dictionary receives deterministic lowering evidence, not a
    claim about compiler partitioning or runtime numerical agreement.
    ``fp16_row_mul="compact"`` limits Mul layout changes to two tested shape
    pairs; all other Mul shapes and all Sub/Concat operations stay unchanged.
    """
    if reduction_report is not None and not isinstance(reduction_report, dict):
        raise TypeError("reduction_report must be a mutable dictionary")
    if row_mul_report is not None and not isinstance(row_mul_report, dict):
        raise TypeError("row_mul_report must be a mutable dictionary")
    graph.validate()
    if recipe_dir is None and len(graph.nodes) > 10_000:
        raise ValueError("large graphs require recipe_dir to bound generated C++ compiler memory")
    weight_paths = {} if weight_paths is None else weight_paths
    missing = set(graph.weights) - set(weight_paths)
    if missing:
        raise ValueError(f"missing weight paths: {sorted(missing)[:4]}")
    total = sum(math.prod(v.shape) * DTYPES[v.dtype].itemsize for v in graph.weights.values())
    if not external_weights and total >= 1_800_000_000:
        raise ValueError("large graphs require external_weights=True (AIR protobuf size limit)")
    e = _Emitter(graph, recipe=recipe_dir is not None, fp16_reductions=fp16_reductions,
                 fp16_row_mul=fp16_row_mul)
    live = {v.name for v in graph.outputs}
    for node in reversed(graph.nodes):
        if node.output.name in live:
            live.update(value.name for value in node.inputs)
    for node in graph.nodes:
        if node.output.name in live:
            _validate_cast_selection(node)
            if node.kind in {"reduce_mean", "reduce_sum"}:
                if len(node.inputs) != 1:
                    raise ValueError(f"{node.kind}: unexpected input count {len(node.inputs)}")
                e.checked_reduction_contract(node.output, node.inputs[0], node.attrs.get("axes"),
                                             node.attrs.get("keepdims", False))
    folded = {}
    if fold_constant_arithmetic:
        from .ge_constants import fold_constants
        folded = fold_constants(graph, weight_paths, live)
        # Recompute liveness: derived constants replace their entire static
        # producer chain, so do not keep unused raw parameter nodes in AIR.
        live = {v.name for v in graph.outputs}
        for node in reversed(graph.nodes):
            if node.output.name in live and node.output.name not in folded:
                live.update(value.name for value in node.inputs)
    for index, value in enumerate(graph.inputs):
        var = e.op(value.name, "Data", [], [("y", value)], {"index": f"int64_t{{{index}}}"})
        e.lines.append(f"  Check({var}.UpdateInputDesc(\"x\", {_desc(value)}), \"Data input descriptor\");")
        e.refs[value.name] = (var, "y")
    for name, array in graph.constants.items():
        if name in live:
            # A static broadcast may replace this source entirely. Materialize
            # only on first live use so Verify never expects disconnected nodes.
            e.pending_constants[name] = array
            if array.nbytes <= e.max_broadcast_tensor_bytes:
                e.inline_constants[name] = array
    for key, value in graph.weights.items():
        if value.name not in live:
            continue
        path = str(Path(weight_paths[key]).resolve())
        size = math.prod(value.shape) * DTYPES[value.dtype].itemsize
        if external_weights:
            var = e.op(value.name, "FileConstant", [], [("y", value)],
                       {"file_path": f"ge::AscendString({_q(path)})", "shape": _ints(value.shape),
                        "dtype": _DT[value.dtype], "file_id": 'ge::AscendString("")'})
            e.lines.append(f"  if (!fs::is_regular_file({_q(path)}) || fs::file_size({_q(path)}) != {size}ULL) throw std::runtime_error(\"external weight size mismatch\");")
        else:
            var = e.op(value.name, "Const", [], [("y", value)])
            e.records[-1]["constant"] = {"path": path, "offset": 0, "size": size}
            e.lines.append(f"  SetConst({var}, {_desc(value)}, Read({_q(path)}, {size}ULL));")
        e.refs[value.name] = (var, "y")

    binary = {"add": "Add", "sub": "Sub", "mul": "Mul", "div": "RealDiv",
              "maximum": "Maximum", "minimum": "Minimum", "eq": "Equal",
              "lt": "Less", "le": "LessEqual", "gt": "Greater", "ge": "GreaterEqual",
              "logical_and": "LogicalAnd", "logical_or": "LogicalOr"}
    unary = {"neg": "Neg", "sigmoid": "Sigmoid", "exp": "Exp", "log": "Log",
             "sqrt": "Sqrt", "rsqrt": "Rsqrt", "cos": "Cos", "sin": "Sin"}
    topk_cache = {}
    for node in graph.nodes:
        kind, v, a, ins = node.kind, node.output, node.attrs, list(node.inputs)
        if v.name not in live:
            continue
        if v.name in folded:
            e.constant(folded[v.name], v.dtype, v.name)
            continue
        if kind == "where" and (len(ins) != 3 or ins[0].dtype != "bool" or
                                 any(x.dtype != v.dtype for x in ins[1:])):
            raise ValueError(f"{v.name}: Select needs BOOL condition and matching branch dtypes")
        # Public GE BasicType excludes BOOL. These operations preserve boolean
        # values exactly through an integer carrier, with no floating casts.
        boolean_output = None
        if v.dtype == "bool" and kind in {"concat", "slice", "transpose", "where"}:
            boolean_output = v
            for i, source in enumerate(ins):
                if source.dtype != "bool" or (kind == "where" and i == 0):
                    continue
                carrier = Value(f"__ge/{v.name}/integer_input_{i}", source.shape, "int32")
                if source.name in e.inline_constants:
                    carrier = e.constant(e.inline_constants[source.name].astype(DTYPES["int32"]),
                                         "int32", carrier.name)
                else:
                    var = e.op(carrier.name, "Cast", [("x", source)], [("y", carrier)],
                               {"dst_type": "int64_t{ge::DT_INT32}"})
                    e.refs[carrier.name] = (var, "y")
                ins[i] = carrier
            v = Value(f"__ge/{v.name}/integer_output", v.shape, "int32")
        attrs, outport = {}, "y"
        outputs = [("y", v)]
        if kind == "mul":
            if len(ins) != 2:
                raise ValueError(f"{v.name}: Mul requires exactly two inputs")
            e.emit_mul(v, ins[0], ins[1])
            continue
        elif kind in binary:
            typ, ports = binary[kind], ["x1", "x2"]
        elif kind in unary:
            typ, ports = unary[kind], ["x"]
            if kind in {"exp", "log"}:
                attrs = {"base": "float{-1.0}", "scale": "float{1.0}", "shift": "float{0.0}"}
        elif kind in {"scalar_mul", "scalar_add"}:
            typ, ports = "Power", ["x"]
            scalar = float(a["value"])
            if not math.isfinite(scalar):
                raise ValueError("Power scalar attributes must be finite")
            attrs = {"power": "float{1.0}",
                     "scale": f"float{{{scalar if kind == 'scalar_mul' else 1.0}}}",
                     "shift": f"float{{{scalar if kind == 'scalar_add' else 0.0}}}"}
        elif kind == "silu":
            typ, ports, attrs = "Swish", ["x"], {"scale": "float{1.0}"}
        elif kind == "softplus":
            # Internal OMG lacks SoftplusV2 shape inference. Native Softplus
            # expresses this model's beta=1 case without intermediate rounding.
            # Above threshold 20, log1p(exp(x)) and x round to the same FP16
            # value. Do not silently generalize this mapping to other settings.
            if (float(a.get("beta", 1)) != 1.0 or float(a.get("threshold", 20)) != 20.0
                    or v.dtype != "float16" or ins[0].dtype != "float16"):
                raise NotImplementedError("native Softplus lowering requires FP16 beta=1 threshold=20")
            typ, ports = "Softplus", ["x"]
        elif kind in {"reshape", "broadcast_to", "transpose"}:
            if kind == "broadcast_to":
                if ins[0].dtype != v.dtype or tuple(a["shape"]) != v.shape:
                    raise ValueError("BroadcastTo descriptor differs from its source/shape")
                if np.broadcast_shapes(ins[0].shape, v.shape) != v.shape:
                    raise ValueError("BroadcastTo input cannot expand to output shape")
                constant = e.static_broadcast(ins[0], v)
                if constant is not None:
                    e.alias_constant(v, constant)
                    continue
            typ = {"reshape": "Reshape", "broadcast_to": "BroadcastTo", "transpose": "Transpose"}[kind]
            key = "perm" if kind == "transpose" else "shape"
            ins.append(e.reshape_shape(v, a[key]) if kind == "reshape"
                       else e.helper(v, a[key], suffix=key))
            ports = ["x", key]
            if kind == "reshape":
                attrs = {"axis": "int64_t{0}", "num_axes": "int64_t{-1}"}
        elif kind in {"reduce_mean", "reduce_sum"}:
            if len(ins) != 1:
                raise ValueError(f"{kind}: unexpected input count {len(ins)}")
            e.emit_reduction(v, ins[0], kind, a.get("axes"), a.get("keepdims", False))
            continue
        elif kind == "slice":
            typ, ports = "Slice", ["x", "offsets", "size"]
            ins.extend([e.helper(v, a["starts"], suffix="starts"), e.helper(v, a["sizes"], suffix="sizes")])
        elif kind == "concat":
            typ, ports = "ConcatD", [f"x{i}" for i in range(len(ins))]
            attrs = {"concat_dim": f"int64_t{{{a['axis']}}}", "N": f"int64_t{{{len(ins)}}}"}
        elif kind == "gather":
            typ, ports = "GatherV2", ["x", "indices", "axis"]
            ins.append(e.helper(v, a.get("axis", 0), dtype="int32", suffix="axis"))
            attrs = {"batch_dims": "int64_t{0}", "negative_index_support": "false", "is_preprocessed": "false"}
        elif kind == "gather_elements":
            # The X90 verifier rejects INT32 data; token IDs must never be
            # rounded through a floating carrier to satisfy that restriction.
            if ins[0].dtype != "float16" or v.dtype != ins[0].dtype:
                raise NotImplementedError("GE GatherElements requires FP16 data; use an exact integer Gather path")
            typ, ports = "GatherElements", ["x", "index"]
            attrs = {"dim": f"int64_t{{{a['axis']}}}"}
        elif kind == "softmax":
            typ, ports = "SoftmaxV2", ["x"]
            attrs = {"axes": _ints([a.get("axis", -1)]), "half_to_float": "false"}
        elif kind == "argmax":
            typ, ports = "ArgMaxV2", ["x", "dimension"]
            ins.append(e.helper(v, a.get("axis", -1), dtype="int32", suffix="axis"))
            attrs = {"dtype": _DT[v.dtype]}
            if a.get("keepdims", False):
                axis = a.get("axis", -1) % len(ins[0].shape)
                squeezed = Value(f"__ge/{v.name}/squeezed",
                                 ins[0].shape[:axis] + ins[0].shape[axis + 1:], v.dtype)
                var = e.op(squeezed.name, typ, list(zip(ports, ins)), [("y", squeezed)], attrs)
                e.refs[squeezed.name] = (var, "y")
                shape = e.reshape_shape(v)
                typ, ports, ins = "Reshape", ["x", "shape"], [squeezed, shape]
                attrs = {"axis": "int64_t{0}", "num_axes": "int64_t{-1}"}
        elif kind in {"topk_values", "topk_indices"}:
            axis = a.get("axis", -1) % len(ins[0].shape)
            if axis != len(ins[0].shape) - 1:
                raise ValueError("GE TopKV2 lowering currently requires the last axis")
            outport = "values" if kind == "topk_values" else "indices"
            if kind == "topk_indices" and v.dtype != "int32":
                raise ValueError("TopKV2 indices must be int32")
            cache_key = (ins[0].name, int(a["k"]), axis)
            if cache_key in topk_cache:
                e.refs[v.name] = (topk_cache[cache_key], outport)
                continue
            typ, ports = "TopKV2", ["x", "k"]
            ins.append(e.helper(v, a["k"], dtype="int32", suffix="k"))
            values = Value(v.name, v.shape, node.inputs[0].dtype)
            indices = Value(v.name, v.shape, "int32")
            outputs = [("values", values), ("indices", indices)]
            attrs = {"sorted": "true", "dim": "int64_t{-1}", "largest": "true"}
        elif kind == "where":
            # X90 OMG lowers SelectV2 to Select, whose verifier does not accept
            # the implicit scalar/rank broadcasting used by NumPy/PyTorch.
            # Broadcast condition and both branches, including boolean carriers,
            # without arithmetic masking (0 * -inf/NaN would change semantics).
            ins = [e.broadcast(x, v, i) for i, x in enumerate(ins)]
            typ, ports = "SelectV2", ["condition", "then", "else"]
            outputs, outport = [("result", v)], "result"
        elif kind == "cast":
            typ, ports = "Cast", ["x"]
            attrs = {"dst_type": f"int64_t{{{_DT[v.dtype]}}}"}
        elif kind == "cumsum":
            from .ge_scan import emit_cumsum
            emit_cumsum(e, v, ins[0], a["axis"], scope=a.get("scope"))
            continue
        elif kind == "matmul":
            if len(ins[0].shape) > 2 and len(ins[1].shape) == 2 and not a.get("transpose_a", False):
                # Dense projections need no mixed-rank broadcast matmul. Preserve
                # the exact FP16 matmul boundary by flattening only batch axes.
                flat = Value(f"__ge/{v.name}/flat_input",
                             (math.prod(ins[0].shape[:-1]), ins[0].shape[-1]), ins[0].dtype)
                shape = e.reshape_shape(flat)
                var = e.op(flat.name, "Reshape", [("x", ins[0]), ("shape", shape)], [("y", flat)],
                           {"axis": "int64_t{0}", "num_axes": "int64_t{-1}"})
                e.refs[flat.name] = (var, "y")
                projected = Value(f"__ge/{v.name}/flat_output", (flat.shape[0], v.shape[-1]), v.dtype)
                var = e.op(projected.name, "MatMul", [("x1", flat), ("x2", ins[1])], [("y", projected)],
                           {"transpose_x1": "false", "transpose_x2": str(bool(a.get("transpose_b", False))).lower()})
                e.refs[projected.name] = (var, "y")
                shape = e.reshape_shape(v)
                var = e.op(v.name, "Reshape", [("x", projected), ("shape", shape)], [("y", v)],
                           {"axis": "int64_t{0}", "num_axes": "int64_t{-1}"})
                e.refs[v.name] = (var, "y")
                continue
            if (max(len(x.shape) for x in ins) > 3
                    or (min(len(x.shape) for x in ins) > 2 and a.get("transpose_a", False))):
                from .ge_matmul import emit_rank3_matmul
                emit_rank3_matmul(e, v, ins[0], ins[1], a)
                continue
            batched = max(len(x.shape) for x in ins) > 2
            typ, ports = ("BatchMatMulV2" if batched else "MatMul"), ["x1", "x2"]
            attrs = {"adj_x1" if batched else "transpose_x1": str(bool(a.get("transpose_a", False))).lower(),
                     "adj_x2" if batched else "transpose_x2": str(bool(a.get("transpose_b", False))).lower()}
            if batched:
                attrs["offset_x"] = "int64_t{0}"
        else:
            raise NotImplementedError(f"GE lowering for {kind!r}")
        if len(ports) != len(ins):
            raise ValueError(f"{kind}: unexpected input count {len(ins)}")
        var = e.op(v.name, typ, list(zip(ports, ins)), outputs, attrs)
        if kind in {"topk_values", "topk_indices"}:
            topk_cache[cache_key] = var
        e.refs[v.name] = (var, outport)
        if boolean_output is not None:
            var = e.op(boolean_output.name, "Cast", [("x", v)], [("y", boolean_output)],
                       {"dst_type": "int64_t{ge::DT_BOOL}"})
            e.refs[boolean_output.name] = (var, "y")
    for output in graph.outputs:
        e.ensure_constant(output)
    input_indices = [_opindex(e.refs[v.name][0]) for v in graph.inputs]
    output_pairs = [(_opindex(e.refs[v.name][0]), 1 if e.refs[v.name][1] == "indices" else 0)
                    for v in graph.outputs]
    records = e.records
    if fp16_row_mul == "compact":
        from .ge_reshape_chains import compact_row_mul_reshape_chains
        records, input_indices, output_pairs, chains = compact_row_mul_reshape_chains(
            records, input_indices, output_pairs, [r["name"] for r in e.row_mul_report["replaced"]])
        e.row_mul_report["reshape_chains"] = chains
    # Inline and recipe serialization must enforce the same descriptor/edge
    # contracts, including when lowering creates or shares intermediate nodes.
    from .ge_recipe import _validate
    _validate(records, graph.name, input_indices, output_pairs)
    if reduction_report is not None:
        reduction_report.clear()
        reduction_report.update(e.reduction_report)
    if row_mul_report is not None:
        row_mul_report.clear()
        row_mul_report.update(e.row_mul_report)
    # A compact recipe keeps full-model compiler memory independent of node count.
    if recipe_dir is not None:
        from .ge_recipe import emit_recipe_cpp
        return emit_recipe_cpp(records, graph.name, input_indices, output_pairs, recipe_dir)
    if fp16_row_mul == "compact" and chains["rewrites"]:
        e = _inline_records(graph, records)
    # Explicit output indices preserve the ABI even for TopK's second output.
    e.lines.append(f"  ge::Graph graph({_q(graph.name)});")
    e.lines.append('  std::cout << "Constructing GE graph with SetInputs..." << std::endl;')
    e.lines.append("  graph.SetInputs(std::vector<ge::Operator>{" + ",".join(f"ops[{index}]" for index in input_indices) + "});")
    e.lines.append('  Require(graph.IsValid(), "GE Graph::SetInputs failed; inspect earlier GE errors in export.log");')
    output_ops = []
    for index, port in output_pairs:
        output_ops.append("{ops[" + str(index) + "],{" + str(port) + "}}")
    e.lines.append("  graph.SetOutputs(std::vector<std::pair<ge::Operator, std::vector<size_t>>>{" + ",".join(output_ops) + "});")
    e.lines.append(_FOOTER)
    return "\n".join(e.lines)
