"""Expose original draft values as bounded native-dtype diagnostic outputs.

This lane has no observer Cast, Reshape, Concat or assumed copy operation.
Making a value a graph output asks the runtime to return that tensor directly;
it is not evidence that compiler partitioning or production outputs stay equal.
Device consistency and terminal-equivalence checks remain separate gates.
"""
from __future__ import annotations

from copy import deepcopy
import math

from qwen35_moe_dflash.draft.ge_trace import _trace_points
from qwen35_moe_dflash.graph.ir import DTYPES, Graph


MAX_DIRECT_OUTPUTS = 16
MAX_DIRECT_BYTES = 64 << 20
DIRECT_TRACE_CONTRACT = "cpro-draft-original-value-direct-outputs-v1"
DEFAULT_DIRECT_NAMES = (
    "context_fc", "projected_context", "layer_0_input", "layer_0_input_norm",
    "layer_0_qk_fp16", "layer_0_scores_fp32", "layer_0_probability_fp32",
    "layer_0_pv_fp16", "layer_0_attention", "layer_0_post_attention",
    "layer_0_post_attention_norm", "layer_0_mlp_down", "layer_0_output",
    "layer_1_input", "layer_1_input_norm",
)


def instrument_draft_direct_outputs(graph, *, names=None, max_outputs=MAX_DIRECT_OUTPUTS):
    """Return ``(copied_graph, semantic_spec)`` with unchanged original nodes.

    Top1 and pre-head hidden are always output indices zero and one. Additional
    semantic probes that refer to one original Value share an output index;
    this does not manufacture independent snapshots of an aliased tensor.
    The spec retains both semantic names and the exact original source name.
    """
    if not isinstance(graph, Graph) or graph.name != "draft":
        raise ValueError("direct outputs require the original draft construction graph")
    graph.validate()
    if (not graph.outputs or graph.outputs[0].name != "draft_top1"
            or any(value.name not in {"draft_top1", "draft_hidden"} for value in graph.outputs)
            or any(value.name.startswith("__draft_trace/") or value.name == "draft_trace_values"
                   for value in graph._values.values())):
        raise ValueError("direct outputs require an uninstrumented draft with original Top1")
    if type(max_outputs) is not int or not 2 <= max_outputs <= MAX_DIRECT_OUTPUTS:
        raise ValueError("direct output count budget must be in 2..16")
    points, _ = _trace_points(graph)
    requested = DEFAULT_DIRECT_NAMES if names is None else names
    if isinstance(requested, (str, bytes)):
        raise ValueError("direct probe names must be an ordered sequence")
    requested = tuple(requested)
    if (not requested or len(requested) > 256
            or any(not isinstance(name, str) or name not in points for name in requested)
            or len(set(requested)) != len(requested)):
        raise ValueError("direct probe selection is empty, duplicated or unsupported")
    selected = tuple(dict.fromkeys(("draft_top1", "draft_hidden", *requested)))
    source_indices, outputs, spec, total_bytes = {}, [], [], 0
    for name in selected:
        value = points[name]
        if (value.dtype not in {"float16", "float32", "int32"} or not value.shape
                or any(type(dim) is not int or dim < 1 for dim in value.shape)):
            raise ValueError("native direct probe dtype/shape is unsupported: " + name)
        if value.name not in source_indices:
            if len(outputs) >= max_outputs:
                raise ValueError("direct probes exceed the 16-output budget")
            size = math.prod(value.shape) * DTYPES[value.dtype].itemsize
            if total_bytes + size > MAX_DIRECT_BYTES:
                raise ValueError("direct probes exceed the 64 MiB byte budget")
            source_indices[value.name] = len(outputs)
            outputs.append(value)
            total_bytes += size
        spec.append({"name": name, "source": value.name, "shape": list(value.shape),
                     "dtype": value.dtype, "output_index": source_indices[value.name]})
    result = deepcopy(graph)
    result.outputs = [result._values[value.name] for value in outputs]
    result.validate()
    return result, spec
