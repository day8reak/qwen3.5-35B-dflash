"""Small, static graph vocabulary shared by the direct GE target and draft.

This is a construction IR, not an AIR file format. Only the GE C++ serializer
produces AIR. No torch, torch_npu, TorchAir, or device initialization is used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any

import numpy as np

DTYPES = {"float16": np.dtype("<f2"), "int32": np.dtype("<i4"),
          "int64": np.dtype("<i8"), "bool": np.dtype("?")}


@dataclass(frozen=True)
class Value:
    name: str
    shape: tuple[int, ...]
    dtype: str
    format: str = "ND"


@dataclass
class Node:
    kind: str
    inputs: tuple[Value, ...]
    output: Value
    attrs: dict[str, Any] = field(default_factory=dict)


class Graph:
    def __init__(self, name: str):
        self.name = name
        self.inputs: list[Value] = []
        self.weights: dict[str, Value] = {}
        self.constants: dict[str, np.ndarray] = {}
        self.nodes: list[Node] = []
        self.outputs: list[Value] = []
        self._values: dict[str, Value] = {}
        self._counter = 0

    def _value(self, name, shape, dtype):
        if dtype not in DTYPES:
            raise ValueError(f"unsupported dtype {dtype!r}; floating operations must be FP16")
        shape = tuple(int(x) for x in shape)
        if any(x < 0 for x in shape):
            raise ValueError("direct GE graph requires static nonnegative shapes")
        if name is None:
            while True:
                name = f"v{self._counter}"
                self._counter += 1
                if name not in self._values:
                    break
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_./-]*", name):
            raise ValueError(f"invalid graph value name {name!r}")
        if name in self._values:
            raise ValueError(f"duplicate graph value {name!r}")
        value = Value(name, shape, dtype)
        self._values[name] = value
        return value

    def input(self, name, shape, dtype="float16"):
        value = self._value(name, shape, dtype)
        self.inputs.append(value)
        return value

    def const(self, data, dtype="float16", name=None):
        array = np.asarray(data, dtype=DTYPES[dtype]).copy()
        value = self._value(name, array.shape, dtype)
        self.constants[value.name] = array
        return value

    def weight(self, key, shape, dtype="float16"):
        if key in self.weights:
            old = self.weights[key]
            if old.shape != tuple(shape) or old.dtype != dtype:
                raise ValueError(f"inconsistent reused weight {key}")
            return old
        value = self._value(key, shape, dtype)
        self.weights[key] = value
        return value

    def op(self, kind, *inputs, shape, dtype="float16", attrs=None, name=None):
        if not inputs or any(not isinstance(v, Value) or self._values.get(v.name) is not v for v in inputs):
            raise ValueError("operation inputs must be values owned by this graph")
        value = self._value(name, shape, dtype)
        self.nodes.append(Node(kind, tuple(inputs), value, dict(attrs or {})))
        return value

    def validate(self):
        if not self.inputs or not self.outputs:
            raise ValueError("graph needs inputs and outputs")
        if len({v.name for v in self.outputs}) != len(self.outputs):
            raise ValueError("duplicate graph outputs")
        for value in self.outputs:
            if self._values.get(value.name) is not value:
                raise ValueError("graph output is not owned by this graph")
        return self

    def manifest(self):
        def desc(v):
            return {"name": v.name, "shape": list(v.shape), "dtype": v.dtype}
        return {"name": self.name, "inputs": [desc(v) for v in self.inputs],
                "outputs": [desc(v) for v in self.outputs],
                "weights": {k: {**desc(v), "bytes": math.prod(v.shape) * DTYPES[v.dtype].itemsize}
                            for k, v in self.weights.items()},
                "node_count": len(self.nodes),
                "operations": sorted({n.kind for n in self.nodes})}
