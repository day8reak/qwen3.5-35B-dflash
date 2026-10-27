"""Bounded C++ GE serializer driven by a binary recipe, independent of graph size.

The recipe is host construction input, never AIR. Version Q35GF001 uses little
endian integers, length-prefixed UTF-8 strings, typed descriptors/attributes,
backward input references, and explicit graph input/output indices. Constants
share a binary blob; checkpoint tensors reference their existing raw files.
Only the real GE runtime can write the AIR output. The bounded metadata reader
validates recipe structure without opening referenced tensor files.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import stat
import struct

from ..ge_type_contracts import validate_record_types as _legacy_validate_record_types
from ..ge_recipe import _CPP as _LEGACY_CPP, _string, _integer
from ..ge_recipe_read import _Reader as _LegacyReader
from .ir import ATTENTION_INPUT_SCHEMA, ATTENTION_CONNECTED_PORTS, attention_registration_cpp

_DT = {"float16": (1, 2), "int32": (2, 4), "int64": (3, 8), "bool": (4, 1),
       "float32": (5, 4), "int16": (6, 2)}
_AT = {"int": 1, "float": 2, "bool": 3, "ints": 4, "dtype": 5, "string": 6}
_MAX_OPS = 2_000_000


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


def validate_record_types(record):
    # Extend only public Cast's enum. Other primitive contracts remain shared;
    # custom schemas are validated by bindings before graph construction.
    if record["type"] == "AdnFusedInferAttention":
        if (tuple(port["port"] for port in record["inputs"]) != ATTENTION_CONNECTED_PORTS
                or [port["port"] for port in record["outputs"]] != ["attention_out"]):
            raise ValueError("documented attention probe requires key/value dynamic lists of one and fixed optional-slot wiring")
    elif record["type"] == "Cast" and any(p["dtype"] in {"float32", "int16"}
                                        for p in record["outputs"]):
        inputs, outputs = record["inputs"], record["outputs"]
        if (len(inputs) != 1 or inputs[0]["port"] != "x" or len(outputs) != 1
                or outputs[0]["port"] != "y" or inputs[0]["shape"] != outputs[0]["shape"]
                or record.get("attrs", {}).get("dst_type") !=
                {"type": "int", "value": {"float32": 0, "int16": 6}[outputs[0]["dtype"]]}):
            raise ValueError("Cast descriptor/attribute contract mismatch")
    else:
        _legacy_validate_record_types(record)


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
        validate_record_types(record)
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
        stream.write(b"Q35GF001")
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


# The shared bounded reader is extended at explicit versioned seams. Assertions
# prevent a future legacy reader edit from silently losing an extension. No
# module globals are modified, and old recipes retain their original magic.
def _attention_verifier_cpp():
    # The recipe stores connected tensors only. Optional registered GE slots
    # still occupy indices, so their positions cannot be inferred from that list.
    names = [port + "0" if kind == "dynamic" else port for port, kind in ATTENTION_INPUT_SCHEMA]
    return r'''
const std::vector<std::string>& AttentionInputNames() {
  static const std::vector<std::string> names = {@NAMES@}; return names;
}
int32_t ExpectedInputIndex(const Record& r, const Port& port, size_t compact) {
  if (r.type != "AdnFusedInferAttention") return static_cast<int32_t>(compact);
  const auto& names = AttentionInputNames();
  auto found = std::find(names.begin(), names.end(), port.name);
  Require(found != names.end(), "unknown attention input " + port.name);
  return static_cast<int32_t>(std::distance(names.begin(), found));
}
void VerifyAttentionInputs(const ge::GNode& node, const Record& r) {
  if (r.type != "AdnFusedInferAttention") return;
  const auto& names = AttentionInputNames();
  for (size_t i = 0; i < names.size(); ++i) {
    int32_t index = -1;
    const bool connected = std::any_of(r.inputs.begin(), r.inputs.end(), [&](const Port& p) { return p.name == names[i]; });
    // GE can omit descriptors for unconnected optional ports. Connected names
    // are always checked by Verify; inspect absent ports when GE exposes them.
    if (node.GetInputIndexByName(ge::AscendString(names[i].c_str()), index) != ge::GRAPH_SUCCESS) {
      Require(!connected, "missing attention schema port " + r.name + "/" + names[i]);
      continue;
    }
    Require(index == static_cast<int32_t>(i), "attention schema port order mismatch " + r.name);
    if (!connected) Require(node.GetInDataNodesAndPortIndexs(index).first == nullptr,
                            "unexpected optional attention edge " + r.name + "/" + names[i]);
  }
}
'''.replace("@NAMES@", ", ".join(json.dumps(name) for name in names))


def _extended_cpp():
    source = _LEGACY_CPP
    replacements = (
        ("Q35GE001", "Q35GF001", 1),
        ("DT_FLOAT16, DT_INT32, DT_INT64, DT_BOOL",
         "DT_FLOAT16, DT_INT32, DT_INT64, DT_BOOL, DT_FLOAT, DT_INT16", 2),
        ("d.dtype <= 4", "d.dtype <= 6", 1),
        ("a.dtype <= 4", "a.dtype <= 6", 1),
        ("d.dtype == 1 ? 2 : d.dtype == 2 ? 4 : d.dtype == 3 ? 8 : 1",
         "d.dtype == 1 || d.dtype == 6 ? 2 : d.dtype == 2 || d.dtype == 5 ? 4 : d.dtype == 3 ? 8 : 1", 1),
        ("case 4: return ge::DT_BOOL;",
         "case 4: return ge::DT_BOOL; case 5: return ge::DT_FLOAT; case 6: return ge::DT_INT16;", 1),
        ('    if (r.type == "ConcatD")',
         '    if (r.type == "AdnFusedInferAttention") {\n' + attention_registration_cpp()
         + '\n    } else if (r.type == "ConcatD")', 1),
        ('void Verify(ge::Graph& graph, const Recipe& recipe) {',
         _attention_verifier_cpp() + '\nvoid Verify(ge::Graph& graph, const Recipe& recipe) {', 1),
        ('node.GetInputsSize() == (r.type == "Data" ? 1 : r.inputs.size())',
         '(node.GetInputsSize() == (r.type == "Data" ? 1 : r.inputs.size()) || '
         '(r.type == "AdnFusedInferAttention" && node.GetInputsSize() == AttentionInputNames().size()))', 1),
        ('    for (size_t i = 0; i < r.outputs.size(); ++i) {\n      int32_t index = -1;',
         '    VerifyAttentionInputs(node, r);\n    for (size_t i = 0; i < r.outputs.size(); ++i) {\n      int32_t index = -1;', 1),
        ('Require(index == static_cast<int32_t>(i), "input port order mismatch " + r.name);',
         'Require(index == ExpectedInputIndex(r, p, i), "input port order mismatch " + r.name);', 1),
        ('node.GetInputDesc(i, d)', 'node.GetInputDesc(index, d)', 1),
        ('node.GetInDataNodesAndPortIndexs(i)', 'node.GetInDataNodesAndPortIndexs(index)', 1),
    )
    for old, new, expected in replacements:
        if source.count(old) != expected:
            raise RuntimeError("legacy recipe reader changed; review stateful GE reader extension")
        source = source.replace(old, new)
    return source


_CPP = _extended_cpp()


_MAX_BYTES = 256 << 20
_READ_DTYPES = {1: ("float16", 2), 2: ("int32", 4), 3: ("int64", 8), 4: ("bool", 1),
           5: ("float32", 4), 6: ("int16", 2)}


class _Reader(_LegacyReader):
    def dtype(self):
        code = self.number("B")
        if code not in _READ_DTYPES:
            raise ValueError("unsupported stateful GE recipe dtype")
        return _READ_DTYPES[code]


def _byte_size(descriptor):
    width = next(width for dtype, width in _READ_DTYPES.values() if dtype == descriptor["dtype"])
    return math.prod(descriptor["shape"]) * width


def _read(reader):
    if reader.take(8) != b"Q35GF001":
        raise ValueError("unknown recipe format")
    graph_name = reader.text(nonempty=True)
    count = reader.count(_MAX_OPS, "operator", minimum=1)
    records, names, source_ports = [], set(), []
    for index in range(count):
        name, op_type = reader.text(nonempty=True), reader.text(nonempty=True)
        if name in names:
            raise ValueError("duplicate recipe operator name")
        names.add(name)
        record = {"name": name, "type": op_type, "inputs": [], "outputs": [], "attrs": {}}
        for direction in ("inputs", "outputs"):
            port_count = reader.count(65536, direction, minimum=direction == "outputs")
            ports = {}
            for _ in range(port_count):
                port_name = reader.text(nonempty=True)
                if port_name in ports:
                    raise ValueError(f"duplicate recipe {direction} port")
                port = {"port": port_name}
                if direction == "inputs":
                    port.update(src=reader.number("I"), src_port=reader.text(nonempty=True))
                port.update(reader.descriptor())
                if direction == "inputs":
                    if port["src"] >= index:
                        raise ValueError("recipe source must refer to an earlier operator")
                    source = source_ports[port["src"]].get(port["src_port"])
                    if source is None or any(source[key] != port[key] for key in ("shape", "dtype")):
                        raise ValueError("recipe source port or connected descriptor mismatch")
                ports[port_name] = port
                record[direction].append(port)
            if direction == "outputs":
                source_ports.append(ports)
        for _ in range(reader.count(1024, "attribute")):
            attr_name = reader.text(nonempty=True)
            if attr_name in record["attrs"]:
                raise ValueError("duplicate recipe attribute name")
            record["attrs"][attr_name] = reader.attribute()
        storage = reader.number("B")
        if storage not in (0, 1, 2):
            raise ValueError("invalid recipe constant storage tag")
        if storage:
            constant = {"kind": storage, "path": reader.text(nonempty=True),
                        "offset": reader.number("Q"), "size": reader.number("Q")}
            if op_type != "Const" or len(record["outputs"]) != 1:
                raise ValueError("constant storage requires Const with one output")
            if constant["size"] != _byte_size(record["outputs"][0]):
                raise ValueError("Const storage/descriptor size mismatch")
            if constant["offset"] + constant["size"] >= 1 << 64 or (storage == 2 and constant["offset"] != 0):
                raise ValueError("invalid recipe constant storage offset")
            record["constant"] = constant
        elif op_type == "Const":
            raise ValueError("Const has no storage")
        if op_type == "FileConstant":
            attrs = record["attrs"]
            if (len(record["outputs"]) != 1
                    or attrs.get("file_path", {}).get("type") != "string"
                    or attrs.get("shape") != {"type": "ints", "value": record["outputs"][0]["shape"]}
                    or attrs.get("dtype") != {"type": "dtype", "value": record["outputs"][0]["dtype"]}):
                raise ValueError("FileConstant descriptor and metadata disagree")
        if op_type == "ConcatD" and [port["port"] for port in record["inputs"]] != [f"x{i}" for i in range(len(record["inputs"]))]:
            raise ValueError("ConcatD dynamic ports must be contiguous")
        if op_type == "AdnFusedInferAttention":
            validate_record_types(record)
        records.append(record)
    inputs, seen_inputs = [], set()
    for order in range(reader.count(65536, "graph input", minimum=1, item_bytes=4)):
        index = reader.number("I")
        if index >= count or index in seen_inputs:
            raise ValueError("invalid or duplicate recipe graph input")
        record = records[index]
        if (record["type"] != "Data" or record["inputs"] or len(record["outputs"]) != 1
                or record["attrs"].get("index") != {"type": "int", "value": order}):
            raise ValueError("recipe Data boundary index mismatch")
        inputs.append(index)
        seen_inputs.add(index)
    outputs = []
    for _ in range(reader.count(65536, "graph output", minimum=1, item_bytes=8)):
        index, port = reader.number("I"), reader.number("I")
        if index >= count or port >= len(records[index]["outputs"]):
            raise ValueError("invalid recipe graph output index")
        outputs.append([index, port])
    if reader.remaining or reader.stream.read(1):
        raise ValueError("trailing recipe bytes")
    return {"graph_name": graph_name, "records": records, "inputs": inputs, "outputs": outputs}


def read_recipe(path):
    """Read structural metadata, with a 256 MiB limit and no tensor file access.

    The graph and its source edges are validated before being returned. This does
    not verify constant contents, GE optimization results, or device placement.
    """
    path = Path(path)
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("recipe must be a regular file")
    if info.st_size > _MAX_BYTES:
        raise ValueError("recipe exceeds metadata reader size limit")
    with path.open("rb") as stream:
        return _read(_Reader(stream, info.st_size))
