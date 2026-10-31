"""A narrowly scoped Conv2D expression of the complete draft feature projection.

The selected OMG rejected its MatMul K=16384. A valid 4x4 convolution expresses
the same dot product using the same FP16 input/weight bytes and one FP16 result.
This is a graph compatibility candidate, not evidence of target Conv2D support
or bitwise agreement between different reduction implementations.
"""
from __future__ import annotations

import math

from qwen35_moe_dflash.graph.ir import Value


# Public GE schema only; selected private-DDK support must be checked separately.
CONV2D_SCHEMA_SOURCE = (
    "https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/"
    "programug/graphdevg/atlasag_25_0015.html"
)


def draft_fc_conv_plan(output, left, right, attrs):
    """Return a plan only for the locked full draft's initial feature projection."""
    if (left.name != "target_features" or right.name != "draft.fc.weight"
            or left.dtype != "float16" or right.dtype != "float16" or output.dtype != "float16"
            or getattr(left, "format", "ND") != "ND" or getattr(right, "format", "ND") != "ND"
            or getattr(output, "format", "ND") != "ND"
            or len(left.shape) != 3 or left.shape[0] != 1 or left.shape[1] < 2
            or left.shape[2] != 16384 or right.shape != (2048, 16384)
            or output.shape != left.shape[:-1] + (2048,)
            or set(attrs) - {"transpose_a", "transpose_b"}
            or attrs.get("transpose_a", False) is not False or attrs.get("transpose_b") is not True):
        return None
    rows = math.prod(left.shape[:-1])
    return {"source_op": "MatMul", "replacement_op": "Conv2D",
            "source_node": output.name, "source_weight": right.name,
            "M": rows, "K": 16384, "N": 2048,
            "input_view": [rows, 1024, 4, 4], "filter_view": [2048, 1024, 4, 4],
            "output_view": [rows, 2048, 1, 1], "result_shape": list(output.shape),
            "data_format": "NCHW", "strides": [1, 1, 1, 1], "pads": [0, 0, 0, 0],
            "dilations": [1, 1, 1, 1], "groups": 1, "bias": False,
            "operand_storage": "UNCHANGED_FP16_BYTES", "arithmetic_output_dtype": "float16",
            "intermediate_arithmetic_boundaries": 0,
            "mathematical_equivalence": "FULL_DOT_PRODUCT",
            "bitwise_equivalence": "NOT_ESTABLISHED", "target_validation": "NOT_RUN",
            "schema_source": CONV2D_SCHEMA_SOURCE}


def _reshape(emitter, source, output):
    shape = emitter.reshape_shape(output)
    var = emitter.op(output.name, "Reshape", [("x", source), ("shape", shape)], [("y", output)],
                     {"axis": "int64_t{0}", "num_axes": "int64_t{-1}"})
    emitter.refs[output.name] = (var, "y")


def emit_draft_fc_conv2d(emitter, output, left, right, attrs):
    """Emit a single complete projection, or leave all other graphs untouched."""
    plan = draft_fc_conv_plan(output, left, right, attrs)
    if (plan is None or emitter.graph.name != "draft"
            or emitter.graph.weights.get("draft.fc.weight") is not right
            or not any(value is left for value in emitter.graph.inputs)):
        return None
    prefix = f"__ge/{output.name}/full_dot_conv"
    image = Value(prefix + "/input", tuple(plan["input_view"]), "float16", format="NCHW")
    kernel = Value(prefix + "/filter", tuple(plan["filter_view"]), "float16", format="NCHW")
    convolved = Value(prefix + "/output", tuple(plan["output_view"]), "float16", format="NCHW")
    _reshape(emitter, left, image)
    _reshape(emitter, right, kernel)
    var = emitter.op(convolved.name, "Conv2D", [("x", image), ("filter", kernel)], [("y", convolved)],
                     {"strides": "std::vector<int64_t>{1,1,1,1}",
                      "pads": "std::vector<int64_t>{0,0,0,0}",
                      "dilations": "std::vector<int64_t>{1,1,1,1}",
                      "groups": "int64_t{1}", "data_format": 'ge::AscendString("NCHW")',
                      "offset_x": "int64_t{0}"})
    emitter.refs[convolved.name] = (var, "y")
    _reshape(emitter, convolved, output)
    return plan
