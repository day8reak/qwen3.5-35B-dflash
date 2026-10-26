"""Bounded host folding of small, static FP16 arithmetic for direct GE.

The primitive unary arithmetic, byte-preserving views, and bounded finite
K=1 matmuls between direct graph constants below are eligible.
Large checkpoint tensors and their expert slices stay on the external-weight path.
Host arithmetic uses FP32 intermediates, followed by the original FP16 boundary
at every node, just like the neutral IR oracle. This does not assert bitwise
parity with a device elementary-function implementation.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np


_ARITHMETIC = frozenset({"scalar_add", "scalar_mul", "exp", "neg"})
_VIEWS = frozenset({"slice", "reshape"})
_ALLOWED = _ARITHMETIC | _VIEWS
_FP16 = np.dtype("<f2")


def _view_index(node, source):
    """Validate fully static views using metadata before accessing payloads."""
    value, attrs = node.output, node.attrs
    if node.kind == "reshape":
        shape = tuple(attrs.get("shape", ()))
        if (any(isinstance(dim, bool) or not isinstance(dim, (int, np.integer)) or dim < 0
                for dim in shape) or shape != value.shape
                or math.prod(shape) != math.prod(source.shape)):
            raise ValueError(f"{value.name}: static reshape shape/element count mismatch")
        return None
    starts, sizes = attrs.get("starts", ()), attrs.get("sizes", ())
    if len(starts) != len(source.shape) or len(sizes) != len(source.shape):
        raise ValueError(f"{value.name}: static slice starts/sizes must cover every axis")
    if any(isinstance(item, bool) or not isinstance(item, (int, np.integer))
           for item in (*starts, *sizes)):
        raise ValueError(f"{value.name}: static slice indices must be integers")
    resolved = tuple(dim - start if size == -1 else size
                     for dim, start, size in zip(source.shape, starts, sizes))
    if (any(start < 0 or size < 0 or start + size > dim
            for dim, start, size in zip(source.shape, starts, resolved))
            or resolved != value.shape):
        raise ValueError(f"{value.name}: static slice bounds/output shape mismatch")
    return tuple(slice(int(start), int(start + size)) for start, size in zip(starts, resolved))


def _constant_k1_matmul(graph, node, *, max_tensor_bytes, remaining_bytes):
    """Fold only small, direct constant products, with the IR matmul boundary.

    Neither checkpoint weights nor previously folded weight expressions are
    eligible. K=1 avoids reassociating a reduction, but multiplying the arrays
    directly is still unsuitable: matmul and multiply can differ on signed
    zeros. Use the oracle's FP32 matmul and single FP16 output conversion.
    """
    value = node.output
    if (value.dtype != "float16" or any(source.dtype != "float16"
            or source.name not in graph.constants for source in node.inputs)):
        return None
    if len(node.inputs) != 2:
        raise ValueError(f"{value.name}: constant matmul requires two inputs")
    # Do not reinterpret attributes outside the neutral IR's ordinary matmul.
    if set(node.attrs) - {"transpose_a", "transpose_b"}:
        return None
    transpose = tuple(node.attrs.get(key, False) for key in ("transpose_a", "transpose_b"))
    if any(type(flag) is not bool for flag in transpose):
        raise ValueError(f"{value.name}: constant matmul transpose attributes must be booleans")
    left, right = node.inputs
    if len(left.shape) < 2 or len(right.shape) < 2:
        return None
    for tensor in (left, right, value):
        if any(isinstance(dim, (bool, np.bool_)) or not isinstance(dim, (int, np.integer))
               or dim <= 0 for dim in tensor.shape):
            raise ValueError(f"{value.name}: constant matmul requires positive static dimensions")
    m, ka = left.shape[-2:][::-1] if transpose[0] else left.shape[-2:]
    kb, n = right.shape[-2:][::-1] if transpose[1] else right.shape[-2:]
    if ka != 1 and kb != 1:
        return None
    if ka != kb:
        raise ValueError(f"{value.name}: constant matmul contracted dimensions differ")
    try:
        batch = np.broadcast_shapes(left.shape[:-2], right.shape[:-2])
    except ValueError as error:
        raise ValueError(f"{value.name}: constant matmul batch shapes are incompatible") from error
    expected_shape = batch + (m, n)
    if value.shape != expected_shape:
        raise ValueError(f"{value.name}: constant matmul output shape mismatch")
    sizes = tuple(math.prod(tensor.shape) * _FP16.itemsize for tensor in (left, right, value))
    if any(size > max_tensor_bytes for size in sizes) or sizes[2] > remaining_bytes:
        return None
    # Only now inspect existing in-memory payloads. This path never resolves,
    # stats, or reads an external weight path, even for a very small weight.
    arrays = [graph.constants[source.name] for source in (left, right)]
    for source, array in zip((left, right), arrays):
        if not isinstance(array, np.ndarray) or array.dtype != _FP16 or array.shape != source.shape:
            raise ValueError(f"{source.name}: static FP16 tensor descriptor mismatch")
        if not np.isfinite(array).all():
            return None
    args = [array.astype(np.float32) for array in arrays]
    args = [np.swapaxes(array, -1, -2) if flag else array for array, flag in zip(args, transpose)]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result = np.asarray(np.matmul(*args), dtype=_FP16)
    if not np.isfinite(result).all():
        return None
    return result


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
        if value.name not in live:
            continue
        if node.kind == "matmul":
            result = _constant_k1_matmul(graph, node, max_tensor_bytes=max_tensor_bytes,
                                         remaining_bytes=max_total_bytes - retained)
            if result is not None:
                folded[value.name] = result
                retained += result.nbytes
            continue
        if node.kind not in _ALLOWED:
            continue
        if len(node.inputs) != 1:
            raise ValueError(f"{value.name}: constant unary arithmetic requires one input")
        source = node.inputs[0]
        if value.dtype != "float16" or source.dtype != "float16":
            continue
        if node.kind in _ARITHMETIC and value.shape != source.shape:
            raise ValueError(f"{value.name}: constant unary arithmetic must preserve shape")
        view_index = _view_index(node, source) if node.kind in _VIEWS else None
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
        source_size = math.prod(source.shape) * _FP16.itemsize
        if (size > max_tensor_bytes or source_size > max_tensor_bytes
                or retained + size > max_total_bytes):
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
            read_size = source_size
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
        if node.kind in _VIEWS:
            # Copy bytes directly: do not pass NaNs or signed zeros through
            # floating arithmetic, and do not retain an uncharged large view.
            result = array.reshape(value.shape) if node.kind == "reshape" else array[view_index]
            folded[value.name] = result.copy(order="C")
            retained += read_size + size
            continue
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
