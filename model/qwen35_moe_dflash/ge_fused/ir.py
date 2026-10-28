"""Independent stateful GE construction IR; no Torch or device dependency.

The legacy primitive vocabulary is retained. Custom nodes have explicit named
ports and may return multiple tensors; their schemas come from caller-supplied
bindings, never from guessed operator defaults. The direct graph vocabulary
supports FP32 intermediates; INT16 lengths remain a stateful extension.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import re

import numpy as np

from ..ge_ir import DTYPES as _LEGACY_DTYPES, Graph as _Graph, Value

DTYPES = {**_LEGACY_DTYPES, "float32": np.dtype("<f4"), "int16": np.dtype("<i2"),
          "int8": np.dtype("i1")}
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_./-]*")

# Document §4.6: these are GE schema slots, not the compact Data boundary.
# key/value are dynamic lists with one tensor each in this diagnostic profile.
ATTENTION_INPUT_SCHEMA = (
    ("query", "required"), ("key", "dynamic"), ("value", "dynamic"),
    ("pse_shift", "optional"), ("atten_mask", "optional"),
    ("actual_seq_lengths_q", "optional"), ("actual_seq_lengths_kv", "optional"),
    ("dequant_scale1", "optional"), ("quant_scale1", "optional"),
    ("dequant_scale2", "optional"), ("quant_scale2", "optional"),
    ("quant_offset2", "optional"), ("antiquant_scale", "optional"),
    ("antiquant_offset", "optional"), ("block_table", "optional"),
    ("kv_padding_size", "optional"), ("all_seq_lengths_q", "optional"),
    ("actual_seq_lengths_q_back", "optional"),
)
ATTENTION_CONNECTED_PORTS = ("query", "key0", "value0", "atten_mask", "actual_seq_lengths_q",
                             "actual_seq_lengths_kv", "block_table", "all_seq_lengths_q")


def attention_registration_cpp():
    """Register real dynamic/optional GE ports without compressing absent slots."""
    calls = {"required": "InputRegister", "optional": "OptionalInputRegister", "dynamic": "DynamicInputRegister"}
    return "\n".join(f'      {calls[kind]}("{port}"' + (", 1" if kind == "dynamic" else "") + ");"
                     for port, kind in ATTENTION_INPUT_SCHEMA)


@dataclass
class CustomNode:
    name: str
    op_type: str
    input_ports: tuple[str, ...]
    inputs: tuple[Value, ...]
    output_ports: tuple[str, ...]
    outputs: tuple[Value, ...]
    attrs: dict
    kind: str = "custom"

    @property
    def output(self):
        """Primary output for legacy primitive helpers; liveness uses all ports."""
        return self.outputs[0]


def node_outputs(node):
    return node.outputs if isinstance(node, CustomNode) else (node.output,)


class Graph(_Graph):
    def __init__(self, name):
        super().__init__(name)
        self._custom_names = set()

    def _value(self, name, shape, dtype):
        if dtype not in DTYPES:
            raise ValueError(f"unsupported stateful GE dtype {dtype!r}")
        shape = tuple(shape)
        if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer))
               for x in shape):
            raise ValueError("GE dimensions must be static integers")
        shape = tuple(int(x) for x in shape)
        if any(x < 0 for x in shape):
            raise ValueError("GE dimensions must be nonnegative")
        if name is None:
            while True:
                name = f"v{self._counter}"
                self._counter += 1
                if name not in self._values and name not in self._custom_names:
                    break
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ValueError(f"invalid graph value name {name!r}")
        if name in self._values or name in self._custom_names:
            raise ValueError(f"duplicate graph value {name!r}")
        value = Value(name, shape, dtype)
        self._values[name] = value
        return value

    def const(self, data, dtype="float16", name=None):
        if dtype not in DTYPES:
            raise ValueError(f"unsupported stateful GE dtype {dtype!r}")
        array = np.asarray(data, dtype=DTYPES[dtype]).copy()
        value = self._value(name, array.shape, dtype)
        self.constants[value.name] = array
        return value

    def custom(self, op_type, inputs, outputs, attrs=None, name=None):
        """Insert an opaque, explicitly bound GE op and return its named tensors.

        ``outputs`` maps each actual output port to ``(shape, dtype)``. Attribute
        values use the recipe's closed ``{type, value}`` vocabulary. This checks
        graph wiring and encoding, not installation or kernel availability.
        """
        from .recipe import _attribute

        if not isinstance(op_type, str) or not _NAME.fullmatch(op_type):
            raise ValueError("custom GE op type must be a nonempty identifier")
        if not isinstance(inputs, dict) or not inputs or not isinstance(outputs, dict) or not outputs:
            raise ValueError("custom op needs named input and output mappings")
        for ports in (inputs, outputs):
            if any(not isinstance(port, str) or not _NAME.fullmatch(port) for port in ports):
                raise ValueError("custom port names must be nonempty identifiers")
        if any(not isinstance(v, Value) or self._values.get(v.name) is not v for v in inputs.values()):
            raise ValueError("custom inputs must be values owned by this graph")
        attrs = {} if attrs is None else deepcopy(attrs)
        if not isinstance(attrs, dict):
            raise ValueError("custom attributes must be a typed mapping")
        for key, attr in attrs.items():
            if not isinstance(key, str) or not key or "\0" in key:
                raise ValueError("invalid custom attribute name")
            if not isinstance(attr, dict) or set(attr) != {"type", "value"}:
                raise ValueError("custom attributes require explicit type and value")
            _attribute(attr)
        if name is None:
            while True:
                name = f"custom_{self._counter}"
                self._counter += 1
                if name not in self._values and name not in self._custom_names:
                    break
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ValueError("invalid custom operator name")
        if name in self._values or name in self._custom_names:
            raise ValueError(f"duplicate graph operator {name!r}")
        # Stage only this node's outputs. Copying the full existing registry for
        # every custom node makes a full quantized model quadratic to construct.
        # Check global collisions explicitly before merging the staged values.
        staged = Graph(self.name)
        staged._custom_names = {name}
        result = {}
        for port, descriptor in outputs.items():
            if not isinstance(descriptor, (tuple, list)) or len(descriptor) != 2:
                raise ValueError("custom output descriptor must be (shape, dtype)")
            output_name = f"{name}/{port}"
            if output_name in self._values or output_name in self._custom_names:
                raise ValueError(f"duplicate graph value {output_name!r}")
            result[port] = staged._value(output_name, *descriptor)
        self._values.update({v.name: v for v in result.values()})
        self._custom_names.add(name)
        self.nodes.append(CustomNode(name, op_type, tuple(inputs), tuple(inputs.values()),
                                     tuple(result), tuple(result.values()), attrs))
        return result

    def validate(self):
        super().validate()
        available = {v.name for v in self.inputs} | set(self.constants) | {v.name for v in self.weights.values()}
        for node in self.nodes:
            if any(v.name not in available or self._values.get(v.name) is not v for v in node.inputs):
                raise ValueError("graph input edge is foreign or not topologically ordered")
            for value in node_outputs(node):
                if value.name in available or self._values.get(value.name) is not value:
                    raise ValueError("graph output edge is duplicate or foreign")
                available.add(value.name)
        if any(v.name not in available for v in self.outputs):
            raise ValueError("graph output has no producer")
        return self

    def manifest(self):
        def desc(v):
            return {"name": v.name, "shape": list(v.shape), "dtype": v.dtype}
        result = {"name": self.name, "inputs": [desc(v) for v in self.inputs],
                  "outputs": [desc(v) for v in self.outputs],
                  "weights": {k: {**desc(v), "bytes": math.prod(v.shape) * DTYPES[v.dtype].itemsize}
                              for k, v in self.weights.items()},
                  "node_count": len(self.nodes), "operations": sorted({n.kind for n in self.nodes}),
                  "custom_ops": [{"name": n.name, "type": n.op_type,
                                  "inputs": {p: desc(v) for p, v in zip(n.input_ports, n.inputs)},
                                  "outputs": {p: desc(v) for p, v in zip(n.output_ports, n.outputs)},
                                  "attrs": deepcopy(n.attrs)} for n in self.nodes if isinstance(n, CustomNode)]}
        if hasattr(self, "state_abi"):
            result["state_abi"] = deepcopy(self.state_abi)
        if hasattr(self, "output_contract"):
            result["output_contract"] = deepcopy(self.output_contract)
        if hasattr(self, "quantization_contract"):
            result["quantization_contract"] = deepcopy(self.quantization_contract)
        return result
