"""One bounded batch covering same-weight GDN arithmetic and layout cut points.

These are diagnostic models. Runtime inputs are frozen host values or explicit
controls, never claimed to be intermediate tensors captured from the board.
The original target/draft graphs, weights, policies and tolerances are unchanged.
"""
from __future__ import annotations

import copy
import hashlib
import math

import numpy as np

from .ge_eval import evaluate
from .ge_ir import Graph
from .ge_target import _Builder
from .op_probe_cases import EXACT_TOLERANCE
from .op_probe_gdn_accuracy import _case, _observe_outputs, cases_from_prepared as source_cases


SUITE = "gdn-comprehensive"
_CONV_WEIGHT = "target.layers.0.linear_attn.conv1d.weight"


def _array(value):
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype("float16") or not np.isfinite(value).all():
        raise ValueError("comprehensive GDN fixtures require finite FP16 arrays without conversion")
    return np.ascontiguousarray(value)


def _input(graph, feeds, name, array):
    feeds[name] = _array(array)
    return graph.input(name, feeds[name].shape)


def _tagged(shape, *, positive=False, offset=0):
    # The small layout fixtures have distinct dyadic payloads, not a repeated
    # row. Larger matrix controls repeat at prime977, not a tensor-axis width.
    index = np.arange(math.prod(shape), dtype=np.int64)
    value = ((index + offset) % 977 + 1) / 1024
    if not positive:
        value *= np.where((index // 3 + offset) % 2, -1, 1)
    return value.astype(np.float16).reshape(shape)


def _normal_rows(offset):
    # Every square is normal, while varied row norms expose a misplaced gain.
    result = np.empty((16, 8), dtype=np.float16)
    for row in range(16):
        for column in range(8):
            magnitude = (column + 1) / 8 * (1, 1.25, 1.5, 2)[(row + offset) % 4]
            result[row, (column + row + offset) % 8] = -magnitude if (row + column + offset) % 2 else magnitude
    return result.reshape(1, 8, 2, 8)


def _make(graph, feeds, outputs, description, policies, *, weights=None, exact=False, fixture="actual"):
    # Direct Data outputs and observations that can be folded to a standalone
    # Const are avoidable diagnostic structures. Keep observations on runtime
    # operations; this does not assert why an older board model failed to load.
    operation_outputs = {node.output.name for node in graph.nodes}
    runtime_values = {value.name for value in graph.inputs}
    for node in graph.nodes:
        if any(value.name in runtime_values for value in node.inputs):
            runtime_values.add(node.output.name)
    for label, value in outputs.items():
        if value.name not in operation_outputs or value.name not in runtime_values:
            raise ValueError(f"diagnostic output requires a runtime operation: {graph.name}/{label}")
    labels, stage_map = _observe_outputs(graph, outputs)
    weights = {} if weights is None else weights
    expected = evaluate(graph, feeds, weights)
    case = _case(graph, feeds, weights, expected, labels, description,
                 {key: policies[key] for key in ("fp16_reductions", "fp16_row_mul")}, suite=SUITE)
    exact_labels = set(labels) if exact is True else set(exact or ())
    for index, label in enumerate(labels):
        if label in exact_labels:
            case["tolerances"][index] = dict(EXACT_TOLERANCE)
    case["input_domain"] = {"fixture": fixture, "source": "frozen same-weight host stage values" if fixture == "actual"
                            else "explicit deterministic control; not original model input"}
    return case, stage_map


def _conv_layout(builder, projected):
    transposed = builder.permute(projected, (0, 2, 1))
    padded = builder.cat([builder.constant(np.zeros((1, 48, 3), np.float16)), transposed], -1)
    windows = builder.cat([builder.view(builder.slice(padded, -1, tap, 8), (48, 8, 1))
                           for tap in range(4)], -1)
    return {"transposed": transposed, "padded": padded, "windows": windows}


def _pad(builder, value):
    permuted = builder.permute(value, (0, 2, 1, 3) if len(value.shape) == 4 else (0, 2, 1))
    shape = list(permuted.shape); shape[2] = 56
    padded = builder.cat([permuted, builder.constant(np.zeros(shape, np.float16))], 2)
    return permuted, padded


def _padded_arrays(value, beta, decay):
    return (np.pad(value.transpose(0, 2, 1, 3), ((0, 0), (0, 0), (0, 56), (0, 0))),
            np.pad(beta.transpose(0, 2, 1), ((0, 0), (0, 0), (0, 56))).reshape(1, 2, 64, 1),
            np.pad(decay.transpose(0, 2, 1), ((0, 0), (0, 0), (0, 56))).reshape(2, 1, 64))


def _extract(source, cut_values, output_values, name, feeds):
    """Copy reachable existing arithmetic, replacing only declared cut Values."""
    graph = Graph(name)
    producers = {node.output.name: node for node in source.nodes}
    mapped = {}
    for label, value in cut_values.items():
        if source._values.get(value.name) is not value:
            raise ValueError("core cut is not owned by its source graph")
        mapped[value.name] = _input(graph, {}, label, feeds[label])
    def visit(value):
        if value.name in mapped:
            return mapped[value.name]
        if value.name in source.constants:
            result = graph.const(source.constants[value.name], dtype=value.dtype, name=value.name)
        elif value.name in producers:
            node = producers[value.name]
            result = graph.op(node.kind, *(visit(item) for item in node.inputs), shape=value.shape,
                              dtype=value.dtype, attrs=copy.deepcopy(node.attrs), name=value.name)
        else:
            raise ValueError(f"core cut leaves an unresolved input/weight: {value.name}")
        mapped[value.name] = result
        return result
    return graph, {label: visit(value) for label, value in output_values.items()}


def cases_from_prepared(prepared_dir, reference_path, omc_dir):
    """Validate existing source bundles, then return exactly26 bounded cases."""
    original, provenance = source_cases(prepared_dir, reference_path, omc_dir, core_stages=True)
    full, core = original
    values = dict(zip(full["output_labels"], full["expected"]))
    core_values = dict(zip(core["output_labels"], core["expected"]))
    policies = provenance["policies"]
    cases, stage_maps = [], {}
    def add(*args, **kwargs):
        case, stages = _make(*args, policies=policies, **kwargs)
        if case["name"] in stage_maps:
            raise ValueError("duplicate comprehensive probe name")
        cases.append(case); stage_maps[case["name"]] = stages

    actual = core["inputs"]
    tagged = {name: _tagged(array.shape, positive=name == "beta", offset=index*17)
              for index, (name, array) in enumerate(actual.items())}
    tagged["decay"] = -np.abs(tagged["decay"])
    for fixture, inputs in (("actual", actual), ("tagged", tagged)):
        graph, feeds = Graph("gc_io_" + fixture), {}
        outputs = {name: graph.op("neg", _input(graph, feeds, name, array), shape=array.shape)
                   for name, array in inputs.items()}
        conv = values["convolved_qkv"] if fixture == "actual" else _tagged((48, 8, 1), offset=31)
        outputs["conv_shape"] = graph.op("neg", _input(graph, feeds, "conv_shape", conv), shape=conv.shape)
        add(graph, feeds, outputs, "Exact sign reversal at all five core IO shapes and narrow convolution output shape; "
            "exposes byte-order/axis/head loss independently of reductions and BMM.", exact=True, fixture=fixture)

    for fixture in ("actual", "tagged"):
        projected = values["projected_qkv"] if fixture == "actual" else _tagged((1, 8, 48), offset=13)
        graph, feeds = Graph("gc_conv_layout_" + fixture), {}
        stages = _conv_layout(_Builder(graph), _input(graph, feeds, "projected", projected))
        add(graph, feeds, stages, "Exact convolution transpose, left zero padding and four-tap window assembly.",
            exact=True, fixture=fixture)
        windows = cases[-1]["expected"][-1]
        for mode in ("const", "dynamic"):
            graph, feeds = Graph(f"gc_conv_bmm_{fixture}_{mode}"), {}
            builder = _Builder(graph)
            lhs = _input(graph, feeds, "windows", windows)
            weights = {}
            kernel = full["weights"][_CONV_WEIGHT]
            if mode == "const":
                rhs = graph.weight(_CONV_WEIGHT, kernel.shape); weights[_CONV_WEIGHT] = kernel
            else:
                rhs = _input(graph, feeds, "kernel", kernel)
            rhs = builder.view(rhs, (48, 4, 1))
            result = builder.op("matmul", lhs, rhs, shape=(48, 8, 1), transpose_a=False, transpose_b=False)
            produced = _conv_layout(builder, _input(graph, feeds, "projected", projected))["windows"]
            produced_result = builder.op("matmul", produced, rhs, shape=(48, 8, 1),
                                          transpose_a=False, transpose_b=False)
            # The const kernel's view folds to a Const. Its frozen payload is
            # already hash-bound; do not expose that standalone Const as an IO.
            outputs = {} if mode == "const" else {"kernel_view": rhs}
            outputs.update({"produced_windows": produced,
                            "convolved": result, "produced_convolved": produced_result,
                            "activation_view": builder.view(result, (1, 48, 8))})
            outputs["activated"] = builder.permute(builder.op("silu", outputs["activation_view"]), (0, 2, 1))
            add(graph, feeds, outputs, "Actual convolution kernel and independently frozen windows, "
                f"with {mode} right operand; compare graph-produced windows with runtime windows, then view/SiLU/transpose.",
                weights=weights, exact={"produced_windows"} | ({"kernel_view"} if mode == "dynamic" else set()),
                fixture=fixture)

    for kind in ("query", "key"):
        for fixture in ("actual", "normal"):
            graph, feeds = Graph(f"gc_l2_{kind}_{fixture}"), {}
            builder = _Builder(graph)
            array = actual[kind] if fixture == "actual" else _normal_rows(kind == "key")
            x = _input(graph, feeds, "x", array)
            square = builder.op("mul", x, x)
            total = builder.reduce("reduce_sum", square, -1)
            epsilon = builder.op("scalar_add", total, value=1e-6)
            gain = builder.op("rsqrt", epsilon)
            result = builder.op("mul", x, gain)
            dense_x = _input(graph, feeds, "dense_x", array.reshape(16, 8))
            dense_square = builder.op("mul", dense_x, dense_x)
            dense_sum = builder.reduce("reduce_sum", dense_square, -1)
            dense_epsilon = builder.op("scalar_add", dense_sum, value=1e-6)
            dense_gain = builder.op("rsqrt", dense_epsilon)
            dense_result = builder.op("mul", dense_x, dense_gain)
            add(graph, feeds, {"square": square, "sum": total, "epsilon": epsilon, "gain": gain, "result": result,
                "dense_square": dense_square, "dense_sum": dense_sum, "dense_epsilon": dense_epsilon,
                "dense_gain": dense_gain, "dense_result": dense_result},
                "Original FP16 L2 boundaries with an independent rank2 input/control branch; actual tiny inputs or varied "
                "normal-domain rows whose every square is normal. Epsilon and emitter policies unchanged.", fixture=fixture)

    for fixture, arrays in (("actual", actual), ("tagged", tagged)):
        graph, feeds = Graph("gc_pad_layout_" + fixture), {}
        builder, outputs = _Builder(graph), {}
        for kind in ("value", "beta", "decay"):
            transposed, padded = _pad(builder, _input(graph, feeds, kind, arrays[kind]))
            outputs[kind + "_transposed"] = transposed
            outputs[kind + "_padded"] = padded
            shape = {"value": (2, 1, 64, 8), "beta": (1, 2, 64, 1), "decay": (2, 1, 64)}[kind]
            outputs[kind + "_view"] = builder.view(padded, shape)
        add(graph, feeds, outputs, "Exact value/beta/decay permutation, right zero padding to64 and carrier reshape; "
            "both heads have distinct nonzero prefixes.", exact=True, fixture=fixture)
        padded_value, padded_beta, _ = _padded_arrays(arrays["value"], arrays["beta"], arrays["decay"])
        for mode in ("chain", "padded"):
            graph, feeds = Graph(f"gc_weighted_mul_{fixture}_{mode}"), {}
            builder = _Builder(graph)
            if mode == "chain":
                _, lhs = _pad(builder, _input(graph, feeds, "value", arrays["value"]))
                _, rhs = _pad(builder, _input(graph, feeds, "beta", arrays["beta"]))
                rhs = builder.view(rhs, (1, 2, 64, 1))
            else:
                lhs = _input(graph, feeds, "value_padded", padded_value)
                rhs = _input(graph, feeds, "beta_padded", padded_beta)
            result = builder.op("mul", lhs, rhs)
            # The rank2 branch consumes independent already-padded runtime
            # operands, so an upstream layout error cannot contaminate it.
            dense_lhs = _input(graph, feeds, "dense_value", padded_value.reshape(128, 8))
            dense_rhs = _input(graph, feeds, "dense_beta", padded_beta.reshape(128, 1))
            dense = builder.op("mul", dense_lhs, dense_rhs)
            if mode == "padded":
                # Sign reversal is observable and exact for finite FP16. The
                # actual Mul still consumes the original unmodified inputs.
                carriers = {"value_carrier_neg": builder.op("neg", lhs),
                            "beta_carrier_neg": builder.op("neg", rhs)}
            else:
                carriers = {"value_carrier": lhs, "beta_carrier": rhs}
            add(graph, feeds, {**carriers, "weighted": result, "dense_weighted": dense},
                "One FP16 weighted-value Mul with original carrier/layout policy and independent dense rank2 control. "
                "Padded runtime carriers are observed through exact Neg branches; Mul inputs are unchanged.",
                exact=set(carriers), fixture=fixture)

    for fixture in ("actual", "moderate"):
        decay = actual["decay"] if fixture == "actual" else -_tagged((1, 8, 2), positive=True, offset=127)
        carrier = np.pad(decay.transpose(0, 2, 1), ((0, 0), (0, 0), (0, 56))).reshape(2, 1, 64)
        for mode in ("chain", "padded"):
            graph, feeds = Graph(f"gc_scan_{fixture}_{mode}"), {}
            builder = _Builder(graph)
            if mode == "chain":
                _, padded = _pad(builder, _input(graph, feeds, "decay", decay))
                value = builder.view(padded, (2, 1, 64))
            else:
                value = _input(graph, feeds, "decay_padded", carrier)
            prefix = builder.op("cumsum", value, axis=-1, scope="gdn_nonpositive")
            observation = {"carrier_neg": builder.op("neg", value)} if mode == "padded" else {"carrier": value}
            add(graph, feeds, {**observation, "prefix": prefix},
                "Original nonpositive FP16 prefix-scan lowering, including56 zero-padded positions per head; "
                "compare prelayout and independently padded runtime inputs. Padded carrier observation uses exact Neg "
                "without changing the scan input.", exact=set(observation), fixture=fixture)

    for fixture in ("actual", "tagged"):
        graph, feeds = Graph("gc_final_bmm_" + fixture), {}
        scores = core_values["chunk_0_scores"] if fixture == "actual" else _tagged((1, 2, 64, 64), positive=True, offset=41) / np.float16(8)
        value = core_values["chunk_0_new_value"] if fixture == "actual" else _tagged((1, 2, 64, 8), positive=True, offset=73)
        lhs, rhs = [_input(graph, feeds, name, array) for name, array in (("scores", scores), ("new_value", value))]
        result = graph.op("matmul", lhs, rhs, shape=(1, 2, 64, 8), attrs={"transpose_a": False, "transpose_b": False})
        add(graph, feeds, {"intra": result}, "Final scores×new_value BMM with exact source operands or dense positive controls.", fixture=fixture)

    source = core["graph"]
    def stage(label):
        return source.outputs[core["output_labels"].index(label)]
    cuts = {"normalized_query": stage("normalized_query"), "normalized_key": stage("normalized_key"),
            **{value.name: value for value in source.inputs if value.name in {"value", "decay", "beta"}}}
    feeds = {name: core_values[name] if name.startswith("normalized_") else actual[name] for name in cuts}
    outputs = {label: stage(label) for label in ("scaled_query", "weighted_value", "decay_prefix", "decay_mask",
               "triangular_transform", "transformed_value", "chunk_0_scores", "chunk_0_new_value", "chunk_0_intra",
               "chunk_0_padded_output", "cropped_core", "chunk_core")}
    graph, outputs = _extract(source, cuts, outputs, "gc_core_normalized_cut", feeds)
    add(graph, feeds, outputs, "Exact production core graph after replacing only Q/K L2 outputs with their host FP16 values; "
        "all reachable downstream arithmetic and original copy policy remain unchanged.")

    producer = {node.output.name: node for node in source.nodes}
    padded = stage("chunk_0_padded_output")
    padded_node = producer[padded.name]
    intra = stage("chunk_0_intra")
    if padded_node.kind != "add" or intra not in padded_node.inputs:
        raise ValueError("unsupported final chunk addition topology")
    inter = next(value for value in padded_node.inputs if value != intra)
    snapshot = copy.copy(source); snapshot.outputs = [inter]
    inter_value = evaluate(snapshot, actual)[0]
    cuts = {"scores": stage("chunk_0_scores"), "new_value": stage("chunk_0_new_value"), "inter": inter}
    feeds = {"scores": core_values["chunk_0_scores"], "new_value": core_values["chunk_0_new_value"], "inter": inter_value}
    graph, outputs = _extract(source, cuts, {label: stage(label) for label in
        ("chunk_0_intra", "chunk_0_padded_output", "cropped_core", "chunk_core")}, "gc_core_final_cut", feeds)
    add(graph, feeds, outputs, "Exact final BMM, inter-chunk addition, crop and transpose with independent host operands; "
        "the source first-chunk inter contribution is zero, not removed from the graph.")

    for item in cases[-2:]:
        for label, value in zip(item["output_labels"], item["expected"]):
            if value.tobytes() != core_values[label].tobytes():
                raise ValueError(f"cut graph differs from original host core at {item['name']}/{label}")

    if len(cases) != 26:
        raise ValueError("comprehensive suite must contain exactly26 cases")
    for item in cases:
        tensors = item["graph"].inputs + item["graph"].outputs
        if len(tensors) > 32 or any(len(value.shape) > 4 or math.prod(value.shape)*2 > 32768 for value in tensors):
            raise ValueError("comprehensive probe exceeds its bounded IO contract")
    result = copy.deepcopy(provenance)
    result["suite"] = SUITE
    result["source_stage_maps"] = result.pop("stage_maps")
    result["stage_maps"] = stage_maps
    result["fixture_inputs"] = {item["name"]: {name: {"shape": list(array.shape), "dtype": "float16",
        "bytes": array.nbytes, "sha256": hashlib.sha256(array.astype("<f2", copy=False).tobytes()).hexdigest()}
        for name, array in item["inputs"].items()} for item in cases}
    result["case_names"] = [item["name"] for item in cases]
    result["limits"] = {"cases": 26, "sequence_length": 8, "layer": 0, "max_io_tensors": 32,
                         "max_tensor_bytes": 32768, "max_rank": 4, "selected_weight_tensors": 19}
    result["limitations"] = [
        "This batch changes diagnostic graph boundaries and cannot establish original-OMC internal values or runtime placement.",
        "Actual fixtures use same-weight host FP16 tensors; tagged/normal/moderate controls deliberately change only diagnostic inputs.",
        "Core cuts retain reachable original operations but replace declared intermediate Values with frozen host runtime inputs.",
        "Original arithmetic thresholds remain unchanged; exact copy controls use zero tolerance and small arithmetic signals require magnitude inspection.",
        "Standalone constant outputs are omitted; direct runtime-input observations use exact Neg branches with explicitly negated labels/goldens. This does not confirm any earlier Load-failure cause.",
        "Native dtype/physical-layout introspection remains limited by the selected HIAI API.",
        "Prepared/reference/OMC and compiler identities are retained from validated source; no original OMC is modified."]
    return cases, result
