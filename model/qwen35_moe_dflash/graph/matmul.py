"""Normalize matching static batch prefixes without changing matrix arithmetic."""
import math

from qwen35_moe_dflash.graph.ir import Value


def _reshape(emitter, source, output):
    shape = emitter.reshape_shape(output)
    var = emitter.op(output.name, "Reshape", [("x", source), ("shape", shape)], [("y", output)],
                     {"axis": "int64_t{0}", "num_axes": "int64_t{-1}"})
    emitter.refs[output.name] = (var, "y")


def emit_rank3_matmul(emitter, output, left, right, attrs):
    """Use one 3D BMM for a higher-rank pair with identical batch prefixes.

    X90 rejected the GDN 5D BatchMatMul in the user's compatibility check.
    Merging only matching batch axes preserves row-major pairing and every
    M/K/N dimension; it neither replicates batches nor unrolls thousands of
    convolution matrices. Kernel support for the normalized form still needs
    real OMG validation. General broadcast prefixes cannot be flattened this
    way and are rejected explicitly.
    """
    if min(len(left.shape), len(right.shape)) < 3 or left.shape[:-2] != right.shape[:-2]:
        raise NotImplementedError(f"{output.name}: rank3 BatchMatMul lowering requires equal batch prefixes")
    if any(d <= 0 for d in left.shape + right.shape):
        raise ValueError(f"{output.name}: BatchMatMul dimensions must be positive")
    if left.dtype != right.dtype or output.dtype != left.dtype:
        raise ValueError(f"{output.name}: BatchMatMul input/output dtype mismatch")
    transpose_a = bool(attrs.get("transpose_a", False))
    transpose_b = bool(attrs.get("transpose_b", False))
    m, ka = left.shape[-2:][::-1] if transpose_a else left.shape[-2:]
    kb, n = right.shape[-2:][::-1] if transpose_b else right.shape[-2:]
    if ka != kb:
        raise ValueError(f"{output.name}: BatchMatMul contracted K dimensions differ")
    if output.shape != left.shape[:-2] + (m, n):
        raise ValueError(f"{output.name}: BatchMatMul output shape does not match inputs")

    batches = math.prod(left.shape[:-2])
    flat_left = Value(f"__ge/{output.name}/batch_left", (batches,) + left.shape[-2:], left.dtype)
    flat_right = Value(f"__ge/{output.name}/batch_right", (batches,) + right.shape[-2:], right.dtype)
    flat_output = Value(f"__ge/{output.name}/batch_output", (batches, m, n), output.dtype)
    if flat_left.shape != left.shape:
        _reshape(emitter, left, flat_left)
    else:
        flat_left = left
    if flat_right.shape != right.shape:
        _reshape(emitter, right, flat_right)
    else:
        flat_right = right
    if transpose_a:
        # The available blue AscendC matmul support checker rejects a left
        # transpose attribute. An explicit permutation preserves element bits
        # and keeps the native BMM's left operand in its normal orientation.
        transposed = Value(f"__ge/{output.name}/batch_left_transposed", (batches, m, ka), left.dtype)
        perm = emitter.helper(transposed, (0, 2, 1), suffix="perm")
        var = emitter.op(transposed.name, "Transpose", [("x", flat_left), ("perm", perm)],
                         [("y", transposed)])
        emitter.refs[transposed.name] = (var, "y")
        flat_left = transposed
    if flat_output.shape == output.shape:
        flat_output = output
    var = emitter.op(flat_output.name, "BatchMatMulV2", [("x1", flat_left), ("x2", flat_right)],
                     [("y", flat_output)], {"adj_x1": "false",
                                           "adj_x2": str(transpose_b).lower(), "offset_x": "int64_t{0}"})
    emitter.refs[flat_output.name] = (var, "y")
    if flat_output.name != output.name:
        _reshape(emitter, flat_output, output)
