"""Read named static OMG inputs from AIR, without importing Torch or loading weights."""
from __future__ import annotations

from collections import Counter
import mmap
import os
from pathlib import Path
import re

from .air_schema import load_torchair_schema
from .omc_wire import Schema


_INPUT_NAME = re.compile(r"[A-Za-z_][A-Za-z_0-9./-]*\Z")
_GRAPH_FIELDS = {
    "name": None,
    "op": {"name": None, "type": None,
           "output_desc": {"dtype": None, "layout": None, "shape": {"dim": None}}},
}


def _root_graph(schema, raw):
    candidates, errors = [], []
    for kind, fields in (("ModelDef", {"graph": _GRAPH_FIELDS}), ("GraphDef", _GRAPH_FIELDS)):
        try:
            decoded = schema.decode(raw, kind, fields=fields)
            graphs = decoded.get("graph", []) if kind == "ModelDef" else [decoded]
            if len(graphs) != 1:
                raise ValueError("expected exactly one root graph")
            graph = graphs[0]
            nodes = graph.get("op", [])
            names = [node.get("name") for node in nodes]
            if not names or any(not name for name in names) or len(set(names)) != len(names):
                raise ValueError("root nodes must have unique, nonempty names")
            candidates.append((kind, graph))
        except (ValueError, TypeError, KeyError, RecursionError) as exc:
            errors.append(f"{kind}: {str(exc)[:240]}")
    if len(candidates) != 1:
        raise ValueError("cannot identify a single raw AIR root graph; " + "; ".join(errors)
                         + ". If the exporter schema differs, supply --torchair-pb2.")
    return candidates[0]


def inspect_input_shape(air, *, torchair_pb2=None):
    """Return a static named shape argument and its source descriptors.

    Root Data/RefData order is preserved for readability; it is NOT asserted to
    be compiled runtime slot order. Constants and nested graph inputs are excluded.
    Missing/scalar/unknown/dynamic shapes fail rather than being guessed as [1].
    """
    path = Path(air).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Expected a regular AIR file: {path}")
    if torchair_pb2:
        schema = load_torchair_schema(torchair_pb2)
    else:
        descriptor = Path(__file__).resolve().parents[3] / "specs/torchair-ge-ir.pb"
        schema = Schema(descriptor.read_bytes(), expected_file="ge_ir.proto", expected_package="ge.proto")
    with path.open("rb") as stream:
        if not os.fstat(stream.fileno()).st_size:
            raise ValueError(f"AIR file is empty: {path}")
        # Inline constants can be much larger than the ABI. Map the file so
        # skipping an OpDef.attr does not read or copy its tensor contents.
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as raw:
            kind, graph = _root_graph(schema, raw)
    rows, problems = [], []
    for node in graph["op"]:
        if node.get("type") not in {"Data", "RefData"}:
            continue
        name = node["name"]
        if not _INPUT_NAME.fullmatch(name):
            problems.append(f"{name!r}: name cannot be represented safely in --input_shape")
            continue
        descriptors = node.get("output_desc", [])
        if len(descriptors) != 1:
            problems.append(f"{name}: expected one output descriptor, found {len(descriptors)}")
            continue
        desc = descriptors[0]
        dims = desc.get("shape", {}).get("dim", [])
        if not dims:
            problems.append(f"{name}: empty/missing shape; scalar versus unmaterialized shape is unknown")
            continue
        if any(type(dim) is not int or dim <= 0 for dim in dims):
            problems.append(f"{name}: {dims}; static compilation requires positive dimensions, "
                            "dynamic axes need an explicit --input_shape/--dynamic_dims plan")
            continue
        rows.append(dict(name=name, type=node["type"], shape=dims,
                         dtype_raw=desc.get("dtype"), layout=desc.get("layout")))
    if problems:
        raise ValueError("cannot generate static --input_shape:\n  " + "\n  ".join(problems))
    if not rows:
        raise ValueError("AIR has no root Data/RefData inputs")
    value = ";".join(row["name"] + ":" + ",".join(map(str, row["shape"])) for row in rows)
    return dict(air=str(path), serialization=kind, graph=graph.get("name"),
                input_count=len(rows), input_types=dict(Counter(row["type"] for row in rows)),
                inputs=rows, input_shape=value,
                scope="SOURCE_AIR_ROOT_INPUTS; NOT_COMPILED_RUNTIME_SLOT_ORDER")
