"""Emit exact finite FP16 residual sums through an explicit FP32 Add boundary.

This is a placement candidate, not a device correctness result. With IEEE
round-to-nearest-even, FP32(a)+FP32(b) rounded once to FP16 has the same result
as an FP16 sum of finite FP16 operands, including signed zeros and overflow.
Nearby operands have an exact sum representable in FP32. When their binary
exponents differ by more than 13, the smaller operand is too far below an FP16
midpoint to change rounding, even after FP32 rounding. Subnormals remain exact
integer multiples of 2**-24. NaN payload preservation is not asserted.

The source IR and original FP16 output names stay unchanged. Selected compiler
CPUCL ownership and runtime arithmetic must still pass the normal gates.
"""
from __future__ import annotations

from .ge_ir import Value
from .ge_type_contracts import GE_DTYPE_IDS


POLICIES = {"native", "cpu-fp32-add"}
EXPORT_SUFFIX = "-residual-cpu-fp32-add-v1"


def residual_selection(graph, policy):
    """Select both residual Adds of every complete draft layer by connectivity."""
    if policy not in POLICIES:
        raise ValueError("draft residual policy must be native or cpu-fp32-add")
    if policy == "native":
        return {}
    if graph.name != "draft":
        raise ValueError("CPU residual lowering is restricted to the draft")
    from .draft_ge_trace import _trace_points
    points, layers = _trace_points(graph)
    by_name = {node.output.name: node for node in graph.nodes}
    selected = {}
    for layer in range(layers):
        for suffix in ("post_attention", "output"):
            name = points[f"layer_{layer}_{suffix}"].name
            node = by_name[name]
            if (node.kind != "add" or len(node.inputs) != 2 or node.output.dtype != "float16"
                    or any(value.dtype != "float16" or value.shape != node.output.shape for value in node.inputs)):
                raise ValueError("residual lowering requires same-shape FP16 operands/result")
            prefix = f"__draft_residual/{name}"
            if any(value.startswith(prefix + "/") for value in graph._values):
                raise ValueError("draft contains a reserved residual lowering name")
            selected[name] = {"layer": layer, "role": suffix}
    if len(selected) != layers * 2:
        raise ValueError("draft residual selection does not cover two unique sums per layer")
    return selected


def emit_residual_add(emitter, node, selected):
    """Retain original operands and final FP16 Value; emit three private values."""
    output = node.output
    promoted = []
    for index, source in enumerate(node.inputs):
        value = Value(f"__draft_residual/{output.name}/operand_{index}", source.shape, "float32", source.format)
        var = emitter.op(value.name, "Cast", [("x", source)], [("y", value)],
                         {"dst_type": f"int64_t{{{GE_DTYPE_IDS['float32']}}}"})
        emitter.refs[value.name] = (var, "y")
        promoted.append(value)
    total = Value(f"__draft_residual/{output.name}/sum", output.shape, "float32", output.format)
    var = emitter.op(total.name, "Add", [("x1", promoted[0]), ("x2", promoted[1])], [("y", total)])
    emitter.refs[total.name] = (var, "y")
    var = emitter.op(output.name, "Cast", [("x", total)], [("y", output)],
                     {"dst_type": f"int64_t{{{GE_DTYPE_IDS['float16']}}}"})
    emitter.refs[output.name] = (var, "y")
    return {**selected, "name": output.name, "inputs": [value.name for value in node.inputs],
            "shape": list(output.shape), "sum_node": total.name,
            "boundary": "FP16_OPERANDS_CAST_FP32_ADD_CAST_FP16_ORIGINAL_OUTPUT",
            "finite_rn_even_equivalence": "EXACT", "nan_payload_equivalence": "NOT_ASSERTED"}
