"""CPU semantic oracle for the neutral IR; never claims GE/device execution."""
from __future__ import annotations

import numpy as np

from .ge_ir import DTYPES


def evaluate(graph, inputs, weights=None):
    """Evaluate and round at every FP16 IR boundary; return ordered outputs.

    NumPy uses host float32 intermediates for elementary functions/reductions.
    The result is a semantic oracle, not a promise of device bitwise parity.
    """
    graph.validate()
    weights = {} if weights is None else weights
    values = {}

    def put(value, array):
        array = np.asarray(array, dtype=DTYPES[value.dtype])
        if tuple(array.shape) != value.shape:
            raise ValueError(f"{value.name}: expected {value.shape}, got {array.shape}")
        values[value.name] = array

    for value in graph.inputs:
        array = np.asarray(inputs[value.name])
        if array.dtype != DTYPES[value.dtype]:
            raise ValueError(f"{value.name}: input dtype must be {value.dtype}")
        put(value, array)
    for name, array in graph.constants.items():
        values[name] = array
    for name, value in graph.weights.items():
        array = np.asarray(weights[name])
        if array.dtype != DTYPES[value.dtype]:
            raise ValueError(f"{name}: weight dtype must be {value.dtype}")
        put(value, array)

    binary = {"add": np.add, "sub": np.subtract, "mul": np.multiply,
              "div": np.divide, "eq": np.equal, "lt": np.less, "le": np.less_equal,
              "gt": np.greater, "ge": np.greater_equal, "logical_and": np.logical_and,
              "logical_or": np.logical_or, "maximum": np.maximum, "minimum": np.minimum}
    unary = {"neg": np.negative, "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
             "cos": np.cos, "sin": np.sin, "rsqrt": lambda x: 1 / np.sqrt(x),
             "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
             "silu": lambda x: x / (1 + np.exp(-x))}
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for node in graph.nodes:
            kind, attrs = node.kind, node.attrs
            raw = [values[v.name] for v in node.inputs]
            args = [x.astype(np.float32) if x.dtype == np.float16 else x for x in raw]
            if kind in binary:
                result = binary[kind](*args)
            elif kind in unary:
                result = unary[kind](args[0])
            elif kind == "scalar_mul":
                result = args[0] * np.float32(attrs["value"])
            elif kind == "scalar_add":
                result = args[0] + np.float32(attrs["value"])
            elif kind == "softplus":
                beta, threshold = attrs.get("beta", 1), attrs.get("threshold", 20)
                bx = args[0] * beta
                result = np.where(bx > threshold, args[0], np.logaddexp(0, bx) / beta)
            elif kind == "reshape":
                result = raw[0].reshape(attrs["shape"])
            elif kind == "transpose":
                result = np.transpose(raw[0], attrs["perm"])
            elif kind in {"reduce_mean", "reduce_sum"}:
                fn = np.mean if kind == "reduce_mean" else np.sum
                result = fn(args[0], axis=tuple(attrs["axes"]), keepdims=attrs.get("keepdims", False))
            elif kind == "slice":
                if len(attrs["starts"]) != raw[0].ndim or len(attrs["sizes"]) != raw[0].ndim:
                    raise ValueError("slice starts/sizes must cover every axis")
                result = raw[0][tuple(slice(s, None if n == -1 else s + n)
                                     for s, n in zip(attrs["starts"], attrs["sizes"]))]
            elif kind == "concat":
                result = np.concatenate(raw, axis=attrs["axis"])
            elif kind == "gather":
                result = np.take(raw[0], raw[1], axis=attrs.get("axis", 0))
            elif kind == "gather_elements":
                result = np.take_along_axis(raw[0], raw[1], axis=attrs["axis"])
            elif kind == "softmax":
                x = args[0] - np.max(args[0], axis=attrs.get("axis", -1), keepdims=True)
                x = np.exp(x)
                result = x / np.sum(x, axis=attrs.get("axis", -1), keepdims=True)
            elif kind == "argmax":
                result = np.argmax(raw[0], axis=attrs.get("axis", -1))
                if attrs.get("keepdims", False):
                    result = np.expand_dims(result, attrs.get("axis", -1))
            elif kind in {"topk_values", "topk_indices"}:
                axis, k = attrs.get("axis", -1), attrs["k"]
                indices = np.argsort(-args[0], axis=axis, kind="stable")
                indices = np.take(indices, np.arange(k), axis=axis)
                result = indices if kind == "topk_indices" else np.take_along_axis(raw[0], indices, axis=axis)
            elif kind == "where":
                result = np.where(raw[0], raw[1], raw[2])
            elif kind == "cast":
                result = raw[0]
            elif kind == "cumsum":
                result = np.cumsum(args[0], axis=attrs["axis"])
            elif kind == "matmul":
                a, b = args
                if attrs.get("transpose_a", False):
                    a = np.swapaxes(a, -1, -2)
                if attrs.get("transpose_b", False):
                    b = np.swapaxes(b, -1, -2)
                result = np.matmul(a, b)
            elif kind == "broadcast_to":
                result = np.broadcast_to(raw[0], attrs["shape"])
            else:
                raise NotImplementedError(f"CPU IR operation {kind!r}")
            put(node.output, result)
    return [values[v.name] for v in graph.outputs]
