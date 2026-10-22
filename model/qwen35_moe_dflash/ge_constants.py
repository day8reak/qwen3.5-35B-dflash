"""Bounded host folding of small, static FP16 arithmetic for direct GE.

Only the four primitive unary arithmetic operations below are eligible. Large
checkpoint tensors and expert slices/reshapes stay on the external-weight path.
Host arithmetic uses FP32 intermediates, followed by the original FP16 boundary
at every node, just like the neutral IR oracle. This does not assert bitwise
parity with a device elementary-function implementation.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np


_ALLOWED = frozenset({"scalar_add", "scalar_mul", "exp", "neg"})
_FP16 = np.dtype("<f2")


def fold_constants(graph, weight_paths, live, *, max_tensor_bytes=65536,
                   max_total_bytes=8 * 1024 * 1024):
    """Return small static node outputs without changing graph or weight files.

    The total budget counts each cached raw weight read once and every returned
    folded output. Existing graph constants require neither a read nor a new
    retained copy. Tensor sizes and budgets are checked before accessing paths;
    skipped nodes remain available to the ordinary GE lowering.
    """
    for name, limit in (("max_tensor_bytes", max_tensor_bytes),
                        ("max_total_bytes", max_total_bytes)):
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    folded, weights = {}, {}
    retained = 0
    for node in graph.nodes:
        value = node.output
        if value.name not in live or node.kind not in _ALLOWED:
            continue
        if len(node.inputs) != 1:
            raise ValueError(f"{value.name}: constant unary arithmetic requires one input")
        source = node.inputs[0]
        if value.dtype != "float16" or source.dtype != "float16":
            continue
        if value.shape != source.shape:
            raise ValueError(f"{value.name}: constant unary arithmetic must preserve shape")
        scalar = None
        if node.kind in {"scalar_add", "scalar_mul"}:
            try:
                scalar = float(node.attrs["value"])
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"{value.name}: scalar attribute must be finite FP32") from error
            if not math.isfinite(scalar) or abs(scalar) > float(np.finfo(np.float32).max):
                raise ValueError(f"{value.name}: scalar attribute must be finite FP32")
            scalar = np.float32(scalar)
        size = math.prod(value.shape) * _FP16.itemsize
        if size > max_tensor_bytes or retained + size > max_total_bytes:
            continue
        read_size = 0
        if source.name in folded:
            array = folded[source.name]
        elif source.name in graph.constants:
            array = graph.constants[source.name]
        elif source.name in weights:
            array = weights[source.name]
        elif source.name in graph.weights:
            # Metadata suffices for this decision: do not even resolve or stat
            # a large tensor's file, and do not read a weight we cannot fold.
            read_size = math.prod(source.shape) * _FP16.itemsize
            if read_size > max_tensor_bytes or retained + read_size + size > max_total_bytes:
                continue
            with Path(weight_paths[source.name]).open("rb") as stream:
                payload = stream.read(read_size + 1)
            if len(payload) != read_size:
                raise ValueError(f"{source.name}: raw FP16 weight byte count must be {read_size}")
            array = np.frombuffer(payload, dtype=_FP16).reshape(source.shape)
            weights[source.name] = array
        else:
            continue
        if array.dtype != _FP16 or array.shape != source.shape:
            raise ValueError(f"{source.name}: static FP16 tensor descriptor mismatch")
        args = array.astype(np.float32)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            if node.kind == "scalar_add":
                result = args + scalar
            elif node.kind == "scalar_mul":
                result = args * scalar
            elif node.kind == "exp":
                result = np.exp(args)
            else:
                result = np.negative(args)
            folded[value.name] = np.asarray(result, dtype=_FP16)
        retained += read_size + size
    return folded
