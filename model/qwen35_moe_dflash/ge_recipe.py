"""Bounded C++ GE serializer driven by a binary recipe, independent of graph size.

The recipe is host construction input, never AIR. Version Q35GE001 uses little
endian integers, length-prefixed UTF-8 strings, typed descriptors/attributes,
backward input references, and explicit graph input/output indices. Constants
share a binary blob; checkpoint tensors reference their existing raw files.
Only the real GE runtime can write the AIR output.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import struct

_DT = {"float16": (1, 2), "int32": (2, 4), "int64": (3, 8), "bool": (4, 1)}
_AT = {"int": 1, "float": 2, "bool": 3, "ints": 4, "dtype": 5, "string": 6}
_MAX_OPS = 2_000_000
_MAX_STRING = 1_048_576


def _string(value):
    if not isinstance(value, str) or "\x00" in value:
        raise ValueError("recipe strings must be text without NUL")
    encoded = value.encode("utf-8")
    if len(encoded) > _MAX_STRING:
        raise ValueError("recipe string exceeds size limit")
    return struct.pack("<I", len(encoded)) + encoded


def _integer(value, *, minimum=-(1 << 63), maximum=(1 << 63)-1):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError("recipe integer is outside its allowed range")
    return value


def _descriptor(value):
    dtype = value["dtype"]
    if dtype not in _DT:
        raise ValueError(f"unsupported recipe dtype {dtype!r}")
    shape = tuple(value["shape"])
    if len(shape) > 16:
        raise ValueError("recipe tensor rank exceeds 16")
    for dimension in shape:
        _integer(dimension, minimum=0)
    size = math.prod(shape) * _DT[dtype][1]
    if size >= 1 << 63:
        raise ValueError("recipe tensor byte size overflows int64")
    return struct.pack("<BI", _DT[dtype][0], len(shape)) + b"".join(struct.pack("<q", d) for d in shape), size


def _attribute(value):
    kind, data = value["type"], value["value"]
    if kind not in _AT:
        raise ValueError(f"unsupported recipe attribute type {kind!r}")
    prefix = struct.pack("<B", _AT[kind])
    if kind == "int":
        return prefix + struct.pack("<q", _integer(data))
    if kind == "float":
        scalar = float(data)
        if not math.isfinite(scalar):
            raise ValueError("recipe floating attributes must be finite")
        try:
            encoded = struct.pack("<f", scalar)
        except OverflowError as error:
            raise ValueError("recipe floating attribute exceeds FP32") from error
        return prefix + encoded
    if kind == "bool":
        if not isinstance(data, bool):
            raise ValueError("recipe bool attribute must be boolean")
        return prefix + struct.pack("<B", data)
    if kind == "ints":
        if len(data) > 65536:
            raise ValueError("recipe integer list exceeds size limit")
        return prefix + struct.pack("<I", len(data)) + b"".join(struct.pack("<q", _integer(v)) for v in data)
    if kind == "dtype":
        if data not in _DT:
            raise ValueError("unsupported recipe dtype attribute")
        return prefix + struct.pack("<B", _DT[data][0])
    return prefix + _string(data)


def _validate(records, graph_name, input_indices, output_pairs):
    if not 0 < len(records) <= _MAX_OPS:
        raise ValueError("recipe operator count is outside bounds")
    if not graph_name or not input_indices or not output_pairs:
        raise ValueError("recipe requires graph name, inputs and outputs")
    _string(graph_name)
    names = set()
    for index, record in enumerate(records):
        name, typ = record["name"], record["type"]
        _string(name); _string(typ)
        if not name or not typ or name in names:
            raise ValueError("recipe operator names must be nonempty and unique")
        names.add(name)
        inputs, outputs, attrs = record["inputs"], record["outputs"], record.get("attrs", {})
        if len(inputs) > 65536 or not 0 < len(outputs) <= 65536 or len(attrs) > 1024:
            raise ValueError("recipe port/attribute count is outside bounds")
        for direction in (inputs, outputs):
            ports = set()
            for port in direction:
                _string(port["port"])
                if not port["port"] or port["port"] in ports:
                    raise ValueError("recipe port names must be nonempty and unique")
                ports.add(port["port"])
                _descriptor(port)
        for port in inputs:
            source = _integer(port["src"], minimum=0, maximum=index-1)
            candidates = [v for v in records[source]["outputs"] if v["port"] == port["src_port"]]
            if len(candidates) != 1 or _descriptor(candidates[0])[0] != _descriptor(port)[0]:
                raise ValueError("recipe source port or connected descriptor mismatch")
        for key, value in attrs.items():
            _string(key)
            if not key:
                raise ValueError("recipe attribute name is empty")
            _attribute(value)
        constant = record.get("constant")
        if typ == "Const":
            if len(outputs) != 1 or constant is None:
                raise ValueError("Const requires one output and explicit storage")
            size = _descriptor(outputs[0])[1]
            if "data" in constant:
                if len(constant["data"]) != size:
                    raise ValueError("Const byte count does not match descriptor")
            else:
                _string(str(Path(constant["path"]).resolve()))
                offset = _integer(constant.get("offset", 0), minimum=0)
                if offset != 0 or _integer(constant["size"], minimum=0) != size:
                    raise ValueError("weight Const requires exact file size and zero offset")
        elif constant is not None:
            raise ValueError("constant storage is only valid for Const operators")
        if typ == "FileConstant":
            required = {"file_path": "string", "shape": "ints", "dtype": "dtype"}
            if len(outputs) != 1 or any(attrs.get(k, {}).get("type") != v for k, v in required.items()):
                raise ValueError("FileConstant requires file_path, shape and dtype attributes")
            if attrs["shape"]["value"] != list(outputs[0]["shape"]) or attrs["dtype"]["value"] != outputs[0]["dtype"]:
                raise ValueError("FileConstant descriptor and metadata disagree")
        if typ == "ConcatD" and [v["port"] for v in inputs] != [f"x{i}" for i in range(len(inputs))]:
            raise ValueError("ConcatD requires contiguous dynamic x ports")
    if len(input_indices) > 65536 or len(output_pairs) > 65536 or len(set(input_indices)) != len(input_indices):
        raise ValueError("recipe graph boundary count is outside bounds")
    for order, index in enumerate(input_indices):
        _integer(index, minimum=0, maximum=len(records)-1)
        record = records[index]
        if record["type"] != "Data" or record["inputs"] or len(record["outputs"]) != 1 or record.get("attrs", {}).get("index") != {"type": "int", "value": order}:
            raise ValueError("recipe graph inputs must be ordered Data indices")
    for index, port in output_pairs:
        _integer(index, minimum=0, maximum=len(records)-1)
        _integer(port, minimum=0, maximum=len(records[index]["outputs"])-1)


def emit_recipe_cpp(records, graph_name, input_indices, output_pairs, recipe_dir):
    """Write an exclusive recipe/blob pair and return the bounded C++ reader.

    ``recipe_dir`` must be a new caller-owned output directory. A failed write
    leaves diagnostic partial inputs there and never overwrites earlier files.
    Large checkpoint tensors are read by the generated executable. Small
    derived FP16 parameter constants may already be evaluated by the emitter.
    """
    _validate(records, graph_name, input_indices, output_pairs)
    folder = Path(recipe_dir).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    recipe, blob = folder / "graph.recipe", folder / "constants.bin"
    with recipe.open("xb") as stream, blob.open("xb") as constants:
        stream.write(b"Q35GE001")
        stream.write(_string(graph_name))
        stream.write(struct.pack("<I", len(records)))
        for record in records:
            stream.write(_string(record["name"]) + _string(record["type"]))
            stream.write(struct.pack("<I", len(record["inputs"])))
            for port in record["inputs"]:
                stream.write(_string(port["port"]) + struct.pack("<I", port["src"]) + _string(port["src_port"]) + _descriptor(port)[0])
            stream.write(struct.pack("<I", len(record["outputs"])))
            for port in record["outputs"]:
                stream.write(_string(port["port"]) + _descriptor(port)[0])
            attrs = record.get("attrs", {})
            stream.write(struct.pack("<I", len(attrs)))
            for name, attr in attrs.items():
                stream.write(_string(name) + _attribute(attr))
            constant = record.get("constant")
            if constant is None:
                stream.write(b"\0")
            elif "data" in constant:
                data = constant["data"]
                stream.write(b"\1" + _string(str(blob)) + struct.pack("<QQ", constants.tell(), len(data)))
                constants.write(data)
            else:
                stream.write(b"\2" + _string(str(Path(constant["path"]).resolve())) + struct.pack("<QQ", constant.get("offset", 0), constant["size"]))
        stream.write(struct.pack("<I", len(input_indices)))
        for index in input_indices:
            stream.write(struct.pack("<I", index))
        stream.write(struct.pack("<I", len(output_pairs)))
        for index, port in output_pairs:
            stream.write(struct.pack("<II", index, port))
    return _CPP.replace("@RECIPE@", json.dumps(str(recipe), ensure_ascii=True))


_CPP = r'''// Host GE construction recipe reader. No device or GE session initialization.
#ifndef Q35_RECIPE_PARSE_ONLY
#include <graph/graph.h>
#include <graph/operator.h>
#include <graph/operator_factory.h>
#include <graph/operator_reg.h>
#include <graph/tensor.h>
#endif
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

#ifndef Q35_RECIPE_PARSE_ONLY
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
#endif

namespace {
namespace fs = std::filesystem;
constexpr const char* kRecipe = @RECIPE@;
void Require(bool ok, const std::string& why) { if (!ok) throw std::runtime_error(why); }
struct Desc { uint8_t dtype; std::vector<int64_t> shape; uint64_t bytes; };
struct Port { std::string name; Desc desc; uint32_t source = 0; std::string source_port; };
struct Attr {
  std::string name; uint8_t type = 0; int64_t integer = 0; float scalar = 0;
  bool boolean = false; uint8_t dtype = 0; std::string text; std::vector<int64_t> integers;
};
struct Record {
  std::string name, type; std::vector<Port> inputs, outputs; std::vector<Attr> attrs;
  uint8_t storage = 0; std::string path; uint64_t offset = 0, bytes = 0, checksum = 0;
};
struct Recipe {
  std::string name; std::vector<Record> records; std::vector<uint32_t> inputs;
  std::vector<std::pair<uint32_t, uint32_t>> outputs;
};
class Reader {
  std::ifstream stream_; uint64_t remaining_;
 public:
  explicit Reader(const char* path) : stream_(path, std::ios::binary), remaining_(fs::file_size(path)) {
    Require(stream_.good(), "cannot open recipe");
    Require(remaining_ <= 4ULL * 1024 * 1024 * 1024, "recipe exceeds 4 GiB limit");
    const uint16_t endian = 1;
    Require(*reinterpret_cast<const uint8_t*>(&endian) == 1, "little-endian host required");
  }
  void Read(void* data, uint64_t bytes) {
    Require(bytes <= remaining_, "truncated recipe");
    Require(bytes <= static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max()), "recipe read too large");
    if (bytes) Require(static_cast<bool>(stream_.read(static_cast<char*>(data), static_cast<std::streamsize>(bytes))), "recipe read failed");
    remaining_ -= bytes;
  }
  template<class T> T Number() { T result{}; Read(&result, sizeof(result)); return result; }
  uint32_t Count(uint32_t limit) { auto n = Number<uint32_t>(); Require(n <= limit, "recipe count exceeds bound"); return n; }
  std::string Text() {
    auto n = Count(1048576); Require(n <= remaining_, "truncated recipe string");
    std::string s(n, '\0'); Read(s.data(), n);
    Require(s.find('\0') == std::string::npos, "NUL in recipe string"); return s;
  }
  Desc Descriptor() {
    Desc d{}; d.dtype = Number<uint8_t>();
    Require(d.dtype >= 1 && d.dtype <= 4, "unsupported recipe dtype");
    d.bytes = d.dtype == 1 ? 2 : d.dtype == 2 ? 4 : d.dtype == 3 ? 8 : 1;
    auto rank = Count(16);
    for (uint32_t i = 0; i < rank; ++i) {
      auto size = Number<int64_t>(); Require(size >= 0, "negative static dimension");
      Require(size == 0 || d.bytes <= static_cast<uint64_t>(INT64_MAX) / static_cast<uint64_t>(size), "tensor size overflow");
      d.bytes *= static_cast<uint64_t>(size); d.shape.push_back(size);
    }
    return d;
  }
  Attr Attribute() {
    Attr a; a.name = Text(); Require(!a.name.empty(), "empty attribute name"); a.type = Number<uint8_t>();
    switch (a.type) {
      case 1: a.integer = Number<int64_t>(); break;
      case 2: a.scalar = Number<float>(); Require(std::isfinite(a.scalar), "nonfinite float attribute"); break;
      case 3: { auto b = Number<uint8_t>(); Require(b <= 1, "invalid boolean attribute"); a.boolean = b; break; }
      case 4: { auto n = Count(65536); for (uint32_t i = 0; i < n; ++i) a.integers.push_back(Number<int64_t>()); break; }
      case 5: a.dtype = Number<uint8_t>(); Require(a.dtype >= 1 && a.dtype <= 4, "invalid dtype attribute"); break;
      case 6: a.text = Text(); break;
      default: throw std::runtime_error("invalid attribute tag");
    }
    return a;
  }
  void End() { Require(remaining_ == 0, "trailing recipe bytes"); }
};
bool Same(const Desc& a, const Desc& b) { return a.dtype == b.dtype && a.shape == b.shape; }
const Attr& FindAttr(const Record& op, const std::string& name, uint8_t type) {
  for (const auto& a : op.attrs) if (a.name == name) { Require(a.type == type, "attribute type mismatch " + name); return a; }
  throw std::runtime_error("missing attribute " + name);
}
void FileRegion(const Record& op) {
  Require(fs::is_regular_file(op.path), "constant file missing: " + op.path);
  uint64_t size = fs::file_size(op.path);
  Require(op.offset <= size && op.bytes <= size - op.offset, "constant file region is truncated");
  if (op.storage == 2) Require(op.offset == 0 && op.bytes == size, "weight file size mismatch");
}
Recipe Load(const char* path) {
  Reader in(path); char magic[8]; in.Read(magic, 8);
  Require(std::memcmp(magic, "Q35GE001", 8) == 0, "unknown recipe format");
  Recipe recipe; recipe.name = in.Text(); Require(!recipe.name.empty(), "empty graph name");
  auto count = in.Count(2000000); Require(count > 0, "empty graph");
  std::set<std::string> names;
  for (uint32_t index = 0; index < count; ++index) {
    Record op; op.name = in.Text(); op.type = in.Text();
    Require(!op.name.empty() && !op.type.empty() && names.insert(op.name).second, "duplicate or empty operator name");
    auto n = in.Count(65536); std::set<std::string> ports;
    for (uint32_t i = 0; i < n; ++i) {
      Port p; p.name = in.Text(); p.source = in.Number<uint32_t>(); p.source_port = in.Text(); p.desc = in.Descriptor();
      Require(!p.name.empty() && ports.insert(p.name).second, "duplicate or empty input port");
      Require(p.source < index, "recipe source must refer to an earlier operator");
      const auto& source = recipe.records[p.source];
      auto found = std::find_if(source.outputs.begin(), source.outputs.end(), [&](const Port& s) { return s.name == p.source_port; });
      Require(found != source.outputs.end() && Same(found->desc, p.desc), "source port or connected descriptor mismatch");
      op.inputs.push_back(std::move(p));
    }
    n = in.Count(65536); Require(n > 0, "operator has no outputs"); ports.clear();
    for (uint32_t i = 0; i < n; ++i) {
      Port p; p.name = in.Text(); p.desc = in.Descriptor();
      Require(!p.name.empty() && ports.insert(p.name).second, "duplicate or empty output port"); op.outputs.push_back(std::move(p));
    }
    n = in.Count(1024); ports.clear();
    for (uint32_t i = 0; i < n; ++i) { auto a = in.Attribute(); Require(ports.insert(a.name).second, "duplicate attribute"); op.attrs.push_back(std::move(a)); }
    op.storage = in.Number<uint8_t>(); Require(op.storage <= 2, "invalid constant storage tag");
    if (op.storage) {
      op.path = in.Text(); op.offset = in.Number<uint64_t>(); op.bytes = in.Number<uint64_t>();
      Require(op.type == "Const" && op.outputs.size() == 1 && op.bytes == op.outputs[0].desc.bytes, "Const storage/descriptor mismatch");
      FileRegion(op);
    } else Require(op.type != "Const", "Const has no storage");
    if (op.type == "FileConstant") {
      Require(op.outputs.size() == 1, "FileConstant must have one output");
      const auto& filename = FindAttr(op, "file_path", 6);
      Require(FindAttr(op, "shape", 4).integers == op.outputs[0].desc.shape && FindAttr(op, "dtype", 5).dtype == op.outputs[0].desc.dtype, "FileConstant metadata mismatch");
      Require(fs::is_regular_file(filename.text) && fs::file_size(filename.text) == op.outputs[0].desc.bytes, "external weight file size mismatch");
    }
    if (op.type == "ConcatD") for (size_t i = 0; i < op.inputs.size(); ++i) Require(op.inputs[i].name == "x" + std::to_string(i), "ConcatD dynamic ports not contiguous");
    recipe.records.push_back(std::move(op));
  }
  auto n = in.Count(65536); Require(n > 0, "graph has no inputs"); std::set<uint32_t> seen;
  for (uint32_t i = 0; i < n; ++i) {
    auto index = in.Number<uint32_t>(); Require(index < count && seen.insert(index).second, "invalid or duplicate graph input");
    const auto& op = recipe.records[index];
    Require(op.type == "Data" && op.inputs.empty() && op.outputs.size() == 1 && FindAttr(op, "index", 1).integer == i, "Data boundary index mismatch");
    recipe.inputs.push_back(index);
  }
  n = in.Count(65536); Require(n > 0, "graph has no outputs");
  for (uint32_t i = 0; i < n; ++i) {
    auto index = in.Number<uint32_t>(); auto port = in.Number<uint32_t>();
    Require(index < count && port < recipe.records[index].outputs.size(), "invalid graph output index"); recipe.outputs.emplace_back(index, port);
  }
  in.End(); return recipe;
}

#ifndef Q35_RECIPE_PARSE_ONLY
void Check(ge::graphStatus status, const char* why) {
  if (status != ge::GRAPH_SUCCESS) throw std::runtime_error(std::string(why) + " status=" + std::to_string(status));
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
ge::DataType DType(uint8_t code) {
  switch (code) { case 1: return ge::DT_FLOAT16; case 2: return ge::DT_INT32; case 3: return ge::DT_INT64; case 4: return ge::DT_BOOL; }
  throw std::runtime_error("unknown dtype");
}
ge::TensorDesc TensorDesc(const Desc& d) { return ge::TensorDesc(ge::Shape(d.shape), ge::FORMAT_ND, DType(d.dtype)); }
class Op : public ge::Operator {
 public:
  explicit Op(const Record& r) : ge::Operator(r.name.c_str(), r.type.c_str()) {
    if (r.type == "ConcatD") DynamicInputRegister("x", r.inputs.size());
    else for (const auto& p : r.inputs) InputRegister(p.name.c_str());
    if (r.type == "Data") InputRegister("x");
    for (const auto& p : r.outputs) OutputRegister(p.name.c_str());
  }
};
void SetAttribute(Op& op, const Attr& a) {
  switch (a.type) {
    case 1: op.SetAttr(a.name.c_str(), a.integer); break;
    case 2: op.SetAttr(a.name.c_str(), a.scalar); break;
    case 3: op.SetAttr(a.name.c_str(), a.boolean); break;
    case 4: op.SetAttr(a.name.c_str(), a.integers); break;
    case 5: op.SetAttr(a.name.c_str(), DType(a.dtype)); break;
    case 6: op.SetAttr(a.name.c_str(), ge::AscendString(a.text.c_str())); break;
    default: throw std::runtime_error("unsupported attribute");
  }
}
uint64_t Hash(const uint8_t* bytes, size_t size) {
  uint64_t h = 14695981039346656037ULL;
  for (size_t i = 0; i < size; ++i) { h ^= bytes[i]; h *= 1099511628211ULL; } return h;
}
void SetConst(Op& op, Record& r) {
  FileRegion(r);
  Require(r.bytes <= static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max()) && r.offset <= static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max()), "constant stream limit exceeded");
  std::ifstream file(r.path, std::ios::binary); file.seekg(static_cast<std::streamoff>(r.offset));
  std::vector<uint8_t> bytes(static_cast<size_t>(r.bytes));
  if (r.bytes) Require(static_cast<bool>(file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(r.bytes))), "constant read failed");
  r.checksum = Hash(bytes.data(), bytes.size());
  ge::Tensor value(TensorDesc(r.outputs[0].desc)); Check(value.SetData(bytes.data(), bytes.size()), "Const SetData"); op.SetAttr("value", value);
}
void VerifyDesc(const ge::TensorDesc& got, const Desc& want, const std::string& name) {
  Require(got.GetShape().GetDims() == want.shape && got.GetFormat() == ge::FORMAT_ND && got.GetDataType() == DType(want.dtype), "descriptor mismatch " + name);
}
void VerifyAttribute(const ge::GNode& node, const Attr& a) {
  ge::AscendString name(a.name.c_str());
  switch (a.type) {
    case 1: { int64_t v; Check(node.GetAttr(name, v), "GetAttr int"); Require(v == a.integer, "int attribute mismatch"); break; }
    case 2: { float v; Check(node.GetAttr(name, v), "GetAttr float"); Require(v == a.scalar, "float attribute mismatch"); break; }
    case 3: { bool v; Check(node.GetAttr(name, v), "GetAttr bool"); Require(v == a.boolean, "bool attribute mismatch"); break; }
    case 4: { std::vector<int64_t> v; Check(node.GetAttr(name, v), "GetAttr ints"); Require(v == a.integers, "int-list attribute mismatch"); break; }
    case 5: { ge::DataType v; Check(node.GetAttr(name, v), "GetAttr dtype"); Require(v == DType(a.dtype), "dtype attribute mismatch"); break; }
    case 6: { ge::AscendString v; Check(node.GetAttr(name, v), "GetAttr string"); Require(v.GetString() != nullptr && v.GetString() == a.text, "string attribute mismatch"); break; }
  }
}
void Verify(ge::Graph& graph, const Recipe& recipe) {
  Require(graph.IsValid(), "invalid GE graph"); std::map<std::string, ge::GNode> nodes;
  for (const auto& node : graph.GetAllNodes()) {
    ge::AscendString name; Check(node.GetName(name), "GetName"); Require(name.GetString() != nullptr, "null compute node name");
    Require(nodes.emplace(name.GetString(), node).second, "duplicate compute node name");
  }
  for (const auto& r : recipe.records) {
    auto it = nodes.find(r.name); Require(it != nodes.end(), "missing compute node " + r.name);
    auto node = it->second; ge::AscendString type; Check(node.GetType(type), "GetType");
    Require(type.GetString() != nullptr && type.GetString() == r.type, "type mismatch " + r.name);
    Require(node.GetOutputsSize() == r.outputs.size() && node.GetInputsSize() == (r.type == "Data" ? 1 : r.inputs.size()), "port count mismatch " + r.name);
    for (size_t i = 0; i < r.outputs.size(); ++i) {
      int32_t index = -1; Check(node.GetOutputIndexByName(ge::AscendString(r.outputs[i].name.c_str()), index), "GetOutputIndexByName");
      Require(index == static_cast<int32_t>(i), "output port order mismatch " + r.name);
      ge::TensorDesc d; Check(node.GetOutputDesc(i, d), "GetOutputDesc"); VerifyDesc(d, r.outputs[i].desc, r.name);
    }
    for (size_t i = 0; i < r.inputs.size(); ++i) {
      const auto& p = r.inputs[i]; int32_t index = -1;
      Check(node.GetInputIndexByName(ge::AscendString(p.name.c_str()), index), "GetInputIndexByName");
      Require(index == static_cast<int32_t>(i), "input port order mismatch " + r.name);
      ge::TensorDesc d; Check(node.GetInputDesc(i, d), "GetInputDesc"); VerifyDesc(d, p.desc, r.name);
      auto source = node.GetInDataNodesAndPortIndexs(i); Require(source.first != nullptr, "missing input edge " + r.name);
      ge::AscendString source_name; Check(source.first->GetName(source_name), "source GetName");
      const auto& expected = recipe.records[p.source];
      Require(source_name.GetString() != nullptr && source_name.GetString() == expected.name, "input edge source mismatch " + r.name);
      auto found = std::find_if(expected.outputs.begin(), expected.outputs.end(), [&](const Port& v) { return v.name == p.source_port; });
      Require(source.second == std::distance(expected.outputs.begin(), found), "input edge source port mismatch " + r.name);
    }
    if (r.type == "Data") {
      int32_t index = -1; Check(node.GetInputIndexByName(ge::AscendString("x"), index), "Data GetInputIndexByName");
      Require(index == 0, "Data x port index mismatch");
      ge::TensorDesc d; Check(node.GetInputDesc(0, d), "Data GetInputDesc"); VerifyDesc(d, r.outputs[0].desc, r.name);
    }
    for (const auto& a : r.attrs) VerifyAttribute(node, a);
    if (r.storage) {
      ge::Tensor value; Check(node.GetAttr(ge::AscendString("value"), value), "GetAttr Const"); VerifyDesc(value.GetTensorDesc(), r.outputs[0].desc, r.name);
      Require(value.GetSize() == r.bytes && (r.bytes == 0 || value.GetData() != nullptr), "Const payload size mismatch");
      Require(Hash(value.GetData(), value.GetSize()) == r.checksum, "Const payload checksum mismatch");
    }
  }
}
void Absent(const fs::path& path) {
  std::error_code ec; auto st = fs::symlink_status(path, ec);
  Require(!ec || ec == std::errc::no_such_file_or_directory, "cannot inspect output path");
  Require(st.type() == fs::file_type::not_found, "refusing to overwrite existing output");
}
int Export(const char* recipe_path, const char* filename) {
  fs::path output(filename), temporary(std::string(filename) + ".partial");
  Require(output.extension() == ".air", "output must end in .air"); Absent(output); Absent(temporary);
  VerifyDataFactory();
  auto recipe = Load(recipe_path); std::vector<Op> ops; ops.reserve(recipe.records.size());
  for (auto& r : recipe.records) {
    ops.emplace_back(r); auto& op = ops.back();
    for (const auto& p : r.inputs) { op.SetInput(p.name.c_str(), ops[p.source], p.source_port.c_str()); Check(op.UpdateInputDesc(p.name.c_str(), TensorDesc(p.desc)), "UpdateInputDesc"); }
    for (const auto& p : r.outputs) Check(op.UpdateOutputDesc(p.name.c_str(), TensorDesc(p.desc)), "UpdateOutputDesc");
    if (r.type == "Data") Check(op.UpdateInputDesc("x", TensorDesc(r.outputs[0].desc)), "Data UpdateInputDesc");
    for (const auto& a : r.attrs) SetAttribute(op, a);
    if (r.storage) SetConst(op, r);
  }
  ge::Graph graph(recipe.name.c_str()); std::vector<ge::Operator> inputs;
  for (auto i : recipe.inputs) inputs.push_back(ops[i]);
  std::cout << "Constructing GE graph with SetInputs..." << std::endl;
  graph.SetInputs(inputs);
  Require(graph.IsValid(), "GE Graph::SetInputs failed; inspect earlier GE errors in export.log");
  std::vector<std::pair<ge::Operator, std::vector<size_t>>> outputs;
  for (auto p : recipe.outputs) outputs.push_back({ops[p.first], {p.second}});
  graph.SetOutputs(outputs);
  std::cout << "Checking constructed compute graph..." << std::endl; Verify(graph, recipe);
  int fd = ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
  Require(fd >= 0, "cannot reserve new partial AIR output"); ::close(fd);
  Check(graph.SaveToFile(temporary.c_str()), "SaveToFile");
  Require(fs::is_regular_file(temporary) && fs::file_size(temporary) > 0, "GE produced empty AIR");
  ge::Graph loaded("roundtrip"); Check(loaded.LoadFromFile(temporary.c_str()), "LoadFromFile");
  std::cout << "Checking reloaded compute graph..." << std::endl; Verify(loaded, recipe);
  Require(::link(temporary.c_str(), output.c_str()) == 0, "cannot publish AIR without overwriting");
  fs::remove(temporary); std::cout << "AIR_EXPORTED " << filename << std::endl; return 0;
}
#endif
}  // namespace
int main(int argc, char** argv) {
  try {
#ifdef Q35_RECIPE_PARSE_ONLY
    if (argc != 2) throw std::runtime_error("Usage: recipe-check recipe.bin");
    auto recipe = Load(argv[1]); std::cout << "RECIPE_VALID " << recipe.records.size() << '\n'; return 0;
#else
    if (argc != 2 && argc != 3) throw std::runtime_error("Usage: serializer output.air [recipe.bin]");
    return Export(argc == 3 ? argv[2] : kRecipe, argv[1]);
#endif
  } catch (const std::exception& e) { std::cerr << "GE recipe failed: " << e.what() << '\n'; return 1; }
}
'''
