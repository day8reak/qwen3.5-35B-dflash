"""Read bounded Q35GE001/002 metadata using only the standard library.

Descriptors and edges describe the exported recipe, not the optimized OMG graph.
Constant storage references are returned as metadata; their paths are never opened.
"""
from __future__ import annotations

import math
from pathlib import Path
import stat
import struct

from qwen35_moe_dflash.graph.type_contracts import GE_DTYPE_IDS


_MAX_BYTES = 256 << 20
_MAX_OPS = 2_000_000
_MAX_STRING = 1 << 20
_DTYPES = {1: ("float16", 2), 2: ("int32", 4), 3: ("int64", 8), 4: ("bool", 1), 5: ("float32", 4)}
_FORMATS = {0: "ND", 1: "NCHW"}


class _Reader:
    def __init__(self, stream, size):
        self.stream, self.remaining = stream, size
        self.with_layout = False

    def take(self, size):
        if size > self.remaining:
            raise ValueError("truncated recipe")
        value = self.stream.read(size)
        if len(value) != size:
            raise ValueError("truncated recipe")
        self.remaining -= size
        return value

    def number(self, fmt):
        return struct.unpack("<" + fmt, self.take(struct.calcsize("<" + fmt)))[0]

    def count(self, limit, label, *, minimum=0, item_bytes=1):
        count = self.number("I")
        if not minimum <= count <= limit:
            raise ValueError(f"recipe {label} count is outside bounds")
        if count > self.remaining // item_bytes:
            raise ValueError(f"truncated recipe {label}")
        return count

    def text(self, *, nonempty=False):
        size = self.count(_MAX_STRING, "string")
        try:
            text = self.take(size).decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("recipe string is not UTF-8") from error
        if "\0" in text or (nonempty and not text):
            raise ValueError("NUL or empty recipe name")
        return text

    def dtype(self):
        code = self.number("B")
        if code not in _DTYPES:
            raise ValueError("unsupported recipe dtype")
        return _DTYPES[code]

    def descriptor(self):
        dtype, width = self.dtype()
        result = {"dtype": dtype}
        if self.with_layout:
            layout = self.number("B")
            if layout not in _FORMATS:
                raise ValueError("unsupported recipe descriptor format")
            result["format"] = _FORMATS[layout]
        rank = self.count(16, "rank", item_bytes=8)
        if result.get("format") == "NCHW" and rank != 4:
            raise ValueError("NCHW recipe descriptors require rank 4")
        shape = [self.number("q") for _ in range(rank)]
        if any(dimension < 0 for dimension in shape):
            raise ValueError("negative recipe dimension")
        if math.prod(shape) * width >= 1 << 63:
            raise ValueError("recipe tensor byte size overflows int64")
        result["shape"] = shape
        return result

    def attribute(self):
        tag = self.number("B")
        if tag == 1:
            kind, value = "int", self.number("q")
        elif tag == 2:
            kind, value = "float", self.number("f")
            if not math.isfinite(value):
                raise ValueError("nonfinite recipe float attribute")
        elif tag == 3:
            value = self.number("B")
            if value > 1:
                raise ValueError("invalid recipe boolean attribute")
            kind, value = "bool", bool(value)
        elif tag == 4:
            count = self.count(65536, "integer attribute", item_bytes=8)
            kind, value = "ints", [self.number("q") for _ in range(count)]
        elif tag == 5:
            kind, value = "dtype", self.dtype()[0]
        elif tag == 6:
            kind, value = "string", self.text()
        else:
            raise ValueError("unsupported recipe attribute tag")
        return {"type": kind, "value": value}


def _byte_size(descriptor):
    width = next(width for dtype, width in _DTYPES.values() if dtype == descriptor["dtype"])
    return math.prod(descriptor["shape"]) * width


def _read(reader):
    magic = reader.take(8)
    if magic not in (b"Q35GE001", b"Q35GE002"):
        raise ValueError("unknown recipe format")
    reader.with_layout = magic == b"Q35GE002"
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
                    if (source is None or any(source[key] != port[key] for key in ("shape", "dtype"))
                            or source.get("format", "ND") != port.get("format", "ND")):
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
            dtype = record["outputs"][0]["dtype"]
            if (len(record["outputs"]) != 1
                    or attrs.get("file_path", {}).get("type") != "string"
                    or attrs.get("shape") != {"type": "ints", "value": record["outputs"][0]["shape"]}
                    or attrs.get("dtype") not in ({"type": "dtype", "value": dtype},
                                                 {"type": "int", "value": GE_DTYPE_IDS[dtype]})):
                raise ValueError("FileConstant descriptor and metadata disagree")
        if op_type == "ConcatD" and [port["port"] for port in record["inputs"]] != [f"x{i}" for i in range(len(record["inputs"]))]:
            raise ValueError("ConcatD dynamic ports must be contiguous")
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
