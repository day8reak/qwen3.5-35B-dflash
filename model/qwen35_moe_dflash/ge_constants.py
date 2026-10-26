"""Bounded host folding of small, static FP16 arithmetic for direct GE.

The primitive unary arithmetic, byte-preserving views, complete static RoPE
regions, and standalone K=1 constant matmul outputs below are eligible.
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


def _constant_k1_plan(graph, node, *, max_tensor_bytes, remaining_bytes):
    """Check a direct constant product without accessing either payload."""
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
    return transpose


def _constant_k1_matmul(graph, node, *, max_tensor_bytes, remaining_bytes):
    """Evaluate an approved seed with the oracle's FP32 matmul/FP16 boundary.

    Neither checkpoint weights nor previously folded weight expressions are
    eligible. K=1 avoids reassociating a reduction, but multiplying the arrays
    directly is still unsuitable: matmul and multiply can differ on signed
    zeros. Callers decide whether the whole consumer region is safe to fold.
    """
    transpose = _constant_k1_plan(graph, node, max_tensor_bytes=max_tensor_bytes,
                                  remaining_bytes=remaining_bytes)
    if transpose is None:
        return None
    left, right = node.inputs
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


def _constant_rope_region(graph, root, consumers, dynamic, *, max_tensor_bytes, remaining_bytes):
    """Stage the complete seven-node target RoPE region, or return nothing.

    Internal values cannot escape the matched region. Terminal values may be
    public outputs or feed consumers that transitively depend on graph inputs;
    static fanout is rejected, even if it depends on checkpoint weights. All
    metadata and seven-output budgets are checked before either seed payload.
    """
    if _constant_k1_plan(graph, root, max_tensor_bytes=max_tensor_bytes,
                         remaining_bytes=remaining_bytes) is None:
        return None
    shape = root.output.shape
    if len(shape) != 3 or shape[0] != 1:
        return None
    _, width, sequence = shape
    public = {value.name for value in graph.outputs}

    def one_user(node, kind):
        users = consumers.get(node.output.name, ())
        if len(users) != 1 or users[0].kind != kind:
            return None
        return users[0]

    def exact_unary(node, source, output_shape):
        return (node is not None and node.inputs == (source.output,)
                and node.output.dtype == "float16" and node.output.shape == output_shape)

    transposed = one_user(root, "transpose")
    if not exact_unary(transposed, root, (1, sequence, width)):
        return None
    perm = transposed.attrs.get("perm")
    if (set(transposed.attrs) != {"perm"} or not isinstance(perm, (list, tuple))
            or any(isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)) for axis in perm)
            or tuple(perm) != (0, 2, 1)):
        return None
    angles = one_user(transposed, "concat")
    angles_shape = (1, sequence, 2 * width)
    if (angles is None or angles.inputs != (transposed.output, transposed.output)
            or angles.output.dtype != "float16" or angles.output.shape != angles_shape
            or set(angles.attrs) != {"axis"}):
        return None
    axis = angles.attrs["axis"]
    if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)) or axis not in (-1, 2):
        return None
    branches = consumers.get(angles.output.name, ())
    if len(branches) != 2 or {node.kind for node in branches} != {"cos", "sin"}:
        return None
    branches = sorted(branches, key=lambda node: node.kind)
    endpoints, region = [], [root, transposed, angles]
    for branch in branches:
        if not exact_unary(branch, angles, angles_shape) or branch.attrs:
            return None
        end = one_user(branch, "reshape")
        end_shape = (1, 1, sequence, 2 * width)
        if not exact_unary(end, branch, end_shape) or set(end.attrs) != {"shape"}:
            return None
        declared_shape = end.attrs["shape"]
        if (not isinstance(declared_shape, (list, tuple))
                or any(isinstance(dim, (bool, np.bool_)) or not isinstance(dim, (int, np.integer))
                       for dim in declared_shape) or tuple(declared_shape) != end_shape):
            return None
        if any(user.output.name not in dynamic for user in consumers.get(end.output.name, ())):
            return None
        endpoints.append(end)
        region.extend((branch, end))
    if any(node.output.name in public for node in region if node not in endpoints):
        return None
    # Exact shape equality alone accepts True == 1; validate metadata types too.
    for node in region:
        if any(isinstance(dim, (bool, np.bool_)) or not isinstance(dim, (int, np.integer))
               or dim <= 0 for dim in node.output.shape):
            return None
    sizes = [math.prod(node.output.shape) * _FP16.itemsize for node in region]
    if any(size > max_tensor_bytes for size in sizes) or sum(sizes) > remaining_bytes:
        return None

    first = _constant_k1_matmul(graph, root, max_tensor_bytes=max_tensor_bytes,
                                remaining_bytes=remaining_bytes)
    if first is None:
        return None
    staged = {root.output.name: first}
    staged[transposed.output.name] = np.transpose(first, (0, 2, 1)).copy(order="C")
    view = staged[transposed.output.name]
    staged[angles.output.name] = np.concatenate((view, view), axis=2)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for branch, end in zip(branches, endpoints):
            function = np.cos if branch.kind == "cos" else np.sin
            result = np.asarray(function(staged[angles.output.name].astype(np.float32)), dtype=_FP16)
            if not np.isfinite(result).all():
                return None
            staged[branch.output.name] = result
            staged[end.output.name] = result.reshape(end.output.shape).copy(order="C")
    return staged


def fold_constants(graph, weight_paths, live, *, max_tensor_bytes=65536,
                   max_total_bytes=8 * 1024 * 1024):
    """Return small static node outputs without changing graph or weight files.

    The total budget counts each cached raw weight read once and every returned
    folded output. Existing graph constants require neither a read nor a new
    retained copy. Tensor sizes and budgets are checked before accessing paths;
    skipped nodes remain available to the ordinary GE lowering. Constant K=1
    products with live node consumers require a complete atomic RoPE region;
    only a product with no live node consumers may fold on its own.
    """
    for name, limit in (("max_tensor_bytes", max_tensor_bytes),
                        ("max_total_bytes", max_total_bytes)):
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    folded, weights = {}, {}
    retained = 0
    consumers, dynamic = {}, {value.name for value in graph.inputs}
    for node in graph.nodes:
        if any(value.name in dynamic for value in node.inputs):
            dynamic.add(node.output.name)
        if node.output.name in live:
            for name in {value.name for value in node.inputs}:
                consumers.setdefault(name, []).append(node)
    for node in graph.nodes:
        value = node.output
        if value.name not in live or value.name in folded:
            continue
        if node.kind == "matmul":
            if consumers.get(value.name):
                region = _constant_rope_region(graph, node, consumers, dynamic,
                                               max_tensor_bytes=max_tensor_bytes,
                                               remaining_bytes=max_total_bytes - retained)
                if region is not None:
                    folded.update(region)
                    retained += sum(array.nbytes for array in region.values())
            else:
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
