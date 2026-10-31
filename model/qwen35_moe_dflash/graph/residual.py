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

import re

from qwen35_moe_dflash.graph.ir import Value
from qwen35_moe_dflash.graph.type_contracts import GE_DTYPE_IDS


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
    # Resolve just the residual chain. Cached drafts have separate context
    # and block RoPE/K norm paths, unlike the full-prefix diagnostic tracer.
    def one(kind, *, weight=None, inputs=None, source=None):
        found = [node.output for node in graph.nodes if node.kind == kind
                 and (weight is None or any(v.name == weight for v in node.inputs))
                 and (inputs is None or node.inputs == tuple(inputs))
                 and (source is None or source in node.inputs)]
        if len(found) != 1:
            raise ValueError('residual chain requires one unambiguous ' + kind + ' producer')
        return found[0]

    layer_ids = sorted({int(match.group(1)) for key in graph.weights
                        if (match := re.fullmatch(r'draft\.layers\.(\d+)\.input_layernorm\.weight', key))})
    if not layer_ids or layer_ids != list(range(len(layer_ids))):
        raise ValueError('residual chain requires contiguous complete draft layers')
    layers = len(layer_ids)
    embedding = one('gather', weight='target.embed_tokens.weight')
    hidden = one('scalar_mul', source=embedding)
    points = {}
    for layer in layer_ids:
        prefix = f'draft.layers.{layer}.'
        attention = one('matmul', weight=prefix + 'self_attn.o_proj.weight')
        hidden = one('add', inputs=(hidden, attention))
        points[f'layer_{layer}_post_attention'] = hidden
        mlp = one('matmul', weight=prefix + 'mlp.down_proj.weight')
        hidden = one('add', inputs=(hidden, mlp))
        points[f'layer_{layer}_output'] = hidden
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
