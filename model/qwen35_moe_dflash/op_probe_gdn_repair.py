"""Single source-bound validation batch for the observed GDN failures."""
from __future__ import annotations

import copy
import hashlib
import math

from .op_probe_gdn_accuracy import cases_from_prepared as core_cases
from .op_probe_gdn_comprehensive import cases_from_prepared as comprehensive_cases


SUITE = "gdn-repair"


def _native_norm_names(case):
    """Select only Q/K L2 sums and the core output RMS mean by graph edges."""
    graph = case["graph"]
    by_label = dict(zip(case["output_labels"], graph.outputs))
    producers = {node.output.name: node for node in graph.nodes}

    def node(value, kind):
        result = producers.get(value.name)
        if result is None or result.kind != kind:
            raise ValueError(f"unsupported native norm anchor {value.name}: expected {kind}")
        return result

    names = []
    for label in ("normalized_query", "normalized_key"):
        product = node(by_label[label], "mul")
        gains = [producers[v.name] for v in product.inputs
                 if v.name in producers and producers[v.name].kind == "rsqrt"]
        if len(gains) != 1:
            raise ValueError("GDN normalization requires exactly one Rsqrt gain")
        epsilon = node(gains[0].inputs[0], "scalar_add")
        total = node(epsilon.inputs[0], "reduce_sum")
        square = node(total.inputs[0], "mul")
        if (square.inputs[0] != square.inputs[1] or total.output.shape != (1, 8, 2, 1)
                or total.attrs != {"axes": [3], "keepdims": True} or epsilon.attrs != {"value": 1e-6}):
            raise ValueError("GDN L2 native override does not match the frozen FP16 boundary")
        names.append(total.output.name)

    core = by_label["chunk_core"]
    squares = [n for n in graph.nodes if n.kind == "mul" and n.inputs == (core, core)]
    if len(squares) != 1:
        raise ValueError("GDN output RMS square anchor is ambiguous")
    means = [n for n in graph.nodes if n.kind == "reduce_mean" and n.inputs == (squares[0].output,)]
    if (len(means) != 1 or means[0].attrs != {"axes": [3], "keepdims": True}
            or means[0].output.dtype != "float16" or means[0].output.shape != (1, 8, 2, 1)):
        raise ValueError("GDN output RMS native override does not match the frozen FP16 boundary")
    names.append(means[0].output.name)
    if len(set(names)) != 3:
        raise ValueError("GDN norm native overrides must name three distinct reductions")
    return names


def cases_from_prepared(prepared_dir, reference_path, omc_dir):
    from .op_probe_gdn_precision import build_cases

    batch, provenance = comprehensive_cases(prepared_dir, reference_path, omc_dir)
    if provenance["policies"] != {"fp16_reductions": "matmul", "fp16_row_mul": "compact", "gdn_indexed_copy": "gather"}:
        raise ValueError("GDN repair requires the observed MatMul/compact/gather source policies")
    originals, original_provenance = core_cases(prepared_dir, reference_path, omc_dir, core_stages=True)
    if provenance["bindings"] != original_provenance["bindings"]:
        raise ValueError("GDN source bindings changed while assembling repair validation")
    precision, precision_metadata = build_cases(originals, original_provenance)
    batch.extend(precision)
    full = originals[0]
    native_names = _native_norm_names(full)
    for policy in ("matmul", "native_norms"):
        for capture in ("final", "stages"):
            item = copy.deepcopy(full)
            item["name"] = item["graph"].name = f"gr_layer0_{policy}_{capture}"
            if capture == "final":
                for key in ("expected", "tolerances", "output_labels"):
                    item[key] = item[key][:1]
                item["graph"].outputs = item["graph"].outputs[:1]
            if policy == "native_norms":
                item["emit_options"]["fp16_native_reductions"] = list(native_names)
            item["description"] = (
                "Complete same-weight first layer with exact rank3 concat carriers; "
                + ("three original Q/K/output-normalization reductions retained native; all other reduction policies retained. "
                   if policy == "native_norms" else "original MatMul reduction policy retained. ")
                + ("Only final features exported." if capture == "final" else "Original core stages also exported."))
            item["graph"].validate()
            batch.append(item)

    names = [item["name"] for item in batch]
    if len(set(names)) != len(names):
        raise ValueError("duplicate GDN repair case name")
    for item in batch:
        item["probe_suite"] = SUITE
        tensors = item["graph"].inputs + item["graph"].outputs
        if len(tensors) > 32 or any(len(v.shape) > 4 or math.prod(v.shape)*2 > 32768 for v in tensors):
            raise ValueError("GDN repair probe exceeds bounded IO")
    provenance["suite"] = SUITE
    provenance["case_names"] = names
    provenance["limits"]["cases"] = len(batch)
    provenance["precision_controls"] = precision_metadata
    provenance["native_norm_reduction_names"] = native_names
    provenance["candidate_emit_options"] = {item["name"]: copy.deepcopy(item["emit_options"]) for item in batch}
    provenance["stage_maps"] = {item["name"]: {
        label: {"name": value.name, "shape": list(value.shape), "dtype": value.dtype, "output_index": i}
        for i, (label, value) in enumerate(zip(item["output_labels"], item["graph"].outputs))} for item in batch}
    provenance["fixture_inputs"] = {item["name"]: {name: {
        "shape": list(array.shape), "dtype": str(array.dtype), "bytes": array.nbytes,
        "sha256": hashlib.sha256(array.astype(array.dtype.newbyteorder('<'), copy=False).tobytes()).hexdigest()}
        for name, array in item["inputs"].items()} for item in batch}
    provenance["limitations"].append(
        "Exact layout rewrite and native reduction alternatives are device-unverified candidates. "
        "Native norms retain all FP16 boundaries, do not imply that final BMM subnormal losses are repaired, "
        "and are not selected by the production exporter. Original precision and accuracy gates are unchanged.")
    return batch, provenance
