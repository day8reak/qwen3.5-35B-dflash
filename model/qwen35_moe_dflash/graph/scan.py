"""Static inclusive-scan lowerings for the direct GE model.

The internal OMG frontend lacks Cumsum inference. Integer scans use associative
modular additions. FP16 lowering is limited to the target GDN nonpositive decay
scope: independent prefix reductions keep one FP16 result boundary per prefix.
Reduction accumulation order is backend-dependent; this is not bitwise scan
parity and must pass the existing FP16 accuracy gates on the target device.
"""
from __future__ import annotations

import numpy as np

from qwen35_moe_dflash.graph.ir import DTYPES, Value


MAX_FP16_SCAN_LENGTH = 256


def _output(e, value, kind, inputs, attrs=None):
    var = e.op(value.name, kind, inputs, [("y", value)], attrs)
    e.refs[value.name] = (var, "y")
    return value


def _slice(e, source, output, axis, size, suffix):
    shape = list(source.shape)
    shape[axis] = size
    value = Value(f"__ge/{output.name}/{suffix}", tuple(shape), source.dtype)
    starts = e.helper(value, [0] * len(shape), suffix="starts")
    sizes = e.helper(value, shape, suffix="sizes")
    return _output(e, value, "Slice", [("x", source), ("offsets", starts), ("size", sizes)])


def _concat(e, output, inputs, axis):
    # Scan assembly must use the same rank3 layout handling as graph Concat;
    # otherwise the prefix reductions can be correct but assembled on a wrong
    # physical axis by the rank3 backend path.
    return e.emit_concat(output, inputs, axis)


def emit_cumsum(e, output: Value, source: Value, axis: int, *, scope=None):
    """Bind ``output`` to a static scan using known basic GE operators.

    ``gdn_nonpositive`` is a model-owned contract, set only for the decay
    ``-exp(A_log) * softplus(a + dt_bias)`` and its zero padding. It is not a
    generic FP16 scan opt-in: signed cancellation can exceed FP16 tolerances
    when a reduction tree replaces sequential accumulation. No intermediate
    FP32 tensors or stepwise FP16 running sums are introduced in the graph.
    """
    if output.shape != source.shape or output.dtype != source.dtype:
        raise ValueError(f"{output.name}: cumsum source/output shape and dtype must match")
    rank = len(source.shape)
    if not rank or any(dim <= 0 for dim in source.shape):
        raise ValueError(f"{output.name}: cumsum requires nonempty static dimensions")
    if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)) or not -rank <= axis < rank:
        raise ValueError(f"{output.name}: cumsum axis is out of range or not an integer")
    axis = int(axis) % rank
    length = source.shape[axis]
    if source.dtype == "float16":
        if scope != "gdn_nonpositive":
            raise NotImplementedError("FP16 cumsum lowering requires the model GDN nonpositive decay scope")
        if length > MAX_FP16_SCAN_LENGTH:
            raise NotImplementedError(f"GDN FP16 cumsum length exceeds {MAX_FP16_SCAN_LENGTH}")
        shape = list(source.shape)
        shape[axis] = 1
        axes = e.reduction_axes(output, source, [axis])
        parts = []
        for size in range(1, length + 1):
            prefix = _slice(e, source, output, axis, size, f"prefix_{size}")
            part = Value(f"__ge/{output.name}/sum_{size}", tuple(shape), source.dtype)
            parts.append(e.emit_reduction(part, prefix, "reduce_sum", [axis], True,
                                          axes_value=axes, origin="scan_prefix"))
        if length == 1:
            e.refs[output.name] = e.refs[parts[0].name]
        else:
            _concat(e, output, parts, axis)
        return
    if source.dtype not in {"int32", "int64"}:
        raise NotImplementedError(f"cumsum lowering does not support {source.dtype}")
    previous = source
    offset = 1
    while offset < length:
        # Hillis-Steele reads the previous complete stage for both operands.
        # Signed integer arithmetic wraps in the destination width, so the
        # reassociation is exact even when the mathematical sum overflows.
        shape = list(source.shape)
        shape[axis] = offset
        zeros = e.constant(np.zeros(shape, dtype=DTYPES[source.dtype]), source.dtype,
                           f"__ge/{output.name}/zero_{offset}")
        tail = _slice(e, previous, output, axis, length - offset, f"shift_source_{offset}")
        shifted = Value(f"__ge/{output.name}/shifted_{offset}", source.shape, source.dtype)
        _concat(e, shifted, [zeros, tail], axis)
        stage = output if offset * 2 >= length else Value(
            f"__ge/{output.name}/scan_{offset}", source.shape, source.dtype)
        previous = _output(e, stage, "Add", [("x1", previous), ("x2", shifted)])
        offset *= 2
    if length == 1:
        e.ensure_constant(source)
        e.refs[output.name] = e.refs[source.name]
