"""Three same-weight tiny layer-0 diagnostics; never changes the original OMC."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from .artifacts import checked_file
from .config import Qwen35DFlashConfig
from .ge_eval import evaluate
from .ge_export import build_pair, configurations, weight_identity, weight_shapes
from .ge_ir import Graph
from .ge_target import _Builder, _config, _linear_attention, _moe
from .hashing import sha256_file
from .op_probe_cases import FP16_TOLERANCE


def _read(path):
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def _same(actual, expected):
    # Unlike Python equality, this distinguishes a bool from an integer in a
    # configuration or ABI shape as well as preserving recorded float types.
    return json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)


def _fields(document, expected, label):
    if any(key not in document or not _same(document[key], value) for key, value in expected.items()):
        raise ValueError(f"{label} differs from the supported tiny FP16 configuration")


def _payload(root, record):
    if (not isinstance(record, dict) or not isinstance(record.get("path"), str)
            or type(record.get("bytes")) is not int or record["bytes"] <= 0
            or not isinstance(record.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None):
        raise ValueError("invalid hash-locked payload record")
    return checked_file(root, record)


def _tokens(value, label):
    if (not isinstance(value, list) or not value
            or any(type(token) is not int or not 0 <= token < 64 for token in value)):
        raise ValueError(f"{label} must contain nonempty integer token IDs within the tiny vocabulary")
    return value


def _layer0(config, draft, dispatch, copy_mode):
    """Reuse production helpers and expose semantic boundaries by graph edges."""
    graph = Graph("gdn_layer0_stages")
    builder = _Builder(graph, gdn_indexed_copy=copy_mode)
    c = _config(config, Qwen35DFlashConfig.from_dict(draft), 8)
    ids = graph.input("input_ids", (1, 8), "int32")
    embed = builder.op("gather", graph.weight("target.embed_tokens.weight", (64, 16)),
                       ids, shape=(1, 8, 16), axis=0)
    prefix = "target.layers.0"
    normed = builder.rms(embed, prefix + ".input_layernorm.weight", c["rms_norm_eps"])
    branch = _linear_attention(builder, normed, prefix + ".linear_attn", c)
    residual = builder.op("add", embed, branch)
    postnorm = builder.rms(residual, prefix + ".post_attention_layernorm.weight", c["rms_norm_eps"])
    final = builder.op("add", residual, _moe(builder, postnorm, prefix + ".mlp", c, dispatch))
    producers = {node.output.name: node for node in graph.nodes}

    def producer(value, kind):
        node = producers.get(value.name)
        if node is None or node.kind != kind:
            raise ValueError(f"unsupported layer-0 stage topology: expected {kind} at {value.name}")
        return node

    def unique(nodes, label):
        if len(nodes) != 1:
            raise ValueError(f"layer-0 stage anchor is not unique: {label}")
        return nodes[0]

    out_weight = graph.weights[prefix + ".linear_attn.out_proj.weight"]
    output_projection = unique([node for node in graph.nodes
                                if node.kind == "matmul" and len(node.inputs) == 2
                                and node.inputs[1] == out_weight], "out_proj.weight")
    if output_projection.output != branch:
        raise ValueError("layer-0 attention branch differs from out_proj anchor")
    view = producer(output_projection.inputs[0], "reshape")
    gated = producer(view.inputs[0], "mul")
    gamma_weight = graph.weights[prefix + ".linear_attn.norm.weight"]
    gamma = unique([node for node in graph.nodes
                    if node.kind == "mul" and gamma_weight in node.inputs], "norm.weight")
    if len(gamma.inputs) != 2 or gamma.output not in gated.inputs:
        raise ValueError("layer-0 norm/gating stage edge differs from production")
    normalized_value = next(value for value in gamma.inputs if value != gamma_weight)
    normalized = producer(normalized_value, "mul")
    gain = unique([producers[value.name] for value in normalized.inputs
                   if value.name in producers and producers[value.name].kind == "rsqrt"], "norm Rsqrt")
    if len(gain.inputs) != 1 or len(normalized.inputs) != 2:
        raise ValueError("unsupported layer-0 normalization arity")
    core = next(value for value in normalized.inputs if value != gain.output)
    norm_input = gain.inputs[0]
    epsilon = producer(norm_input, "scalar_add")
    mean = producer(epsilon.inputs[0], "reduce_mean")
    square = producer(mean.inputs[0], "mul")
    if (square.inputs != (core, core) or epsilon.attrs != {"value": c["rms_norm_eps"]}
            or mean.attrs != {"axes": [3], "keepdims": True}
            or core.shape != (1, 8, 2, 8) or norm_input.shape != (1, 8, 2, 1)
            or gain.output.shape != norm_input.shape or gated.output.shape != core.shape
            or branch.shape != (1, 8, 16) or final.shape != branch.shape):
        raise ValueError("layer-0 normalization shape/arithmetic differs from supported production graph")
    stages = {"layer0_final": final, "input_norm": normed, "chunk_core": core, "norm_input": norm_input,
              "norm_gain": gain.output, "gated_core": gated.output, "attention_branch": branch}
    if any(value.dtype != "float16" for value in stages.values()):
        raise ValueError("layer-0 diagnostic stages must preserve FP16")
    graph.outputs = list(stages.values())
    graph.validate()
    return graph, stages


def _case(graph, inputs, weights, expected, labels, description, emit_options):
    graph.validate()
    if set(weights) != set(graph.weights):
        raise ValueError(f"{graph.name}: diagnostic weights differ from graph weights")
    if len(graph.outputs) != len(expected) or len(labels) != len(expected):
        raise ValueError(f"{graph.name}: diagnostic output count differs")
    expected = [np.ascontiguousarray(value) for value in expected]
    if any(not np.isfinite(value).all() for value in expected):
        raise ValueError(f"{graph.name}: diagnostic reference contains nonfinite values")
    return {"name": graph.name, "graph": graph,
            "inputs": {key: np.ascontiguousarray(value) for key, value in inputs.items()},
            "weights": dict(weights), "expected": expected,
            "tolerances": [dict(FP16_TOLERANCE) for _ in expected],
            "description": description, "output_labels": list(labels),
            "emit_options": dict(emit_options), "probe_suite": "gdn-accuracy", "probe_family": "gdn"}


def cases_from_prepared(prepared_dir, reference_path, omc_dir):
    """Return exactly three bounded cases and source provenance, without tools.

    Whole-pair records and payloads establish source identity; only the 19
    first-layer weights become case arrays. New probe OMCs may optimize
    differently from the supplied original pair and cannot establish its
    internal execution values or a GDN root cause by themselves.
    """
    prepared, omc = [Path(path).expanduser().resolve() for path in (prepared_dir, omc_dir)]
    reference_path = Path(reference_path).expanduser().resolve()
    prepared_path, manifest_path = prepared / "prepared.json", omc / "manifest.json"
    (plan, prepared_sha), (reference, reference_sha), (manifest, manifest_sha) = [
        _read(path) for path in (prepared_path, reference_path, manifest_path)]
    target_config, draft_config = configurations(tiny=True)
    _fields(plan, {"artifact_kind": "qwen35-ge-prepared", "status": "PREPARED", "tiny": True,
                   "precision": "float16", "quantization": None, "sequence_length": 8,
                   "target_config": target_config, "draft_config": draft_config}, "prepared pair")
    dispatch = plan.get("expert_dispatch")
    if not isinstance(dispatch, str) or dispatch not in {"static", "selected"}:
        raise ValueError("prepared expert_dispatch must be static or selected")
    source = plan.get("source")
    if (not isinstance(source, dict) or source.get("kind") != "ge-tiny-numpy-random"
            or source.get("not_upstream_weights") is not True):
        raise ValueError("prepared tiny source identity is missing")
    records = plan.get("weights")
    if not isinstance(records, dict) or not records:
        raise ValueError("prepared weight records must be nonempty")
    try:
        payload_identity = weight_identity(records)
    except (KeyError, TypeError) as error:
        raise ValueError("invalid prepared weight records") from error
    if source.get("fp16_payload_sha256") != payload_identity:
        raise ValueError("prepared source identity differs from weight records")
    _fields(reference, {"status": "PASS", "evidence_kind": "framework-reference", "source": source,
                       "precision": "torch.float16", "quantization": None, "sequence_length": 8,
                       "prepared_manifest_sha256": prepared_sha}, "reference")
    prompt = _tokens(reference.get("prompt_token_ids"), "reference prompt")
    limit = reference.get("max_new_tokens")
    if (len(prompt) > 8 or type(limit) is not int or limit < 1 or len(prompt) + limit - 1 > 8):
        raise ValueError("reference prompt/generation limit exceeds the supported sequence gear")
    modes = [reference.get(mode) for mode in ("ordinary", "dflash")]
    if any(not isinstance(mode, dict) for mode in modes):
        raise ValueError("reference must contain ordinary and dflash results")
    generated = [_tokens(mode.get("token_ids"), "reference generated tokens") for mode in modes]
    if (generated[0] != generated[1] or not isinstance(modes[0].get("stop_reason"), str)
            or not modes[0]["stop_reason"] or modes[0]["stop_reason"] != modes[1].get("stop_reason")):
        raise ValueError("reference ordinary/dflash results disagree")
    _fields(manifest, {"schema_version": 2, "artifact_kind": "qwen35-cpro-omc-pair", "status": "COMPILED",
                      "source": source, "precision": "torch.float16", "quantization": None,
                      "abi_version": 2, "feature_dtype": "torch.float16", "sequence_length": 8,
                      "feature_width": 32, "vocab_size": 64, "draft_width": 3, "pad_token_id": 0,
                      "eos_token_ids": [], "expert_dispatch": dispatch, "state_policy": "recompute",
                      "platform": "kirinx90", "runtime": "hiai"}, "original OMC pair")
    if (not isinstance(manifest.get("ddk_id"), str) or not manifest["ddk_id"].strip()
            or not isinstance(manifest.get("omg_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", manifest["omg_sha256"]) is None):
        raise ValueError("original OMC pair is missing DDK/OMG identity")
    policies = {}
    for key, allowed in (("fp16_reductions", {"native", "matmul"}),
                         ("fp16_row_mul", {"native", "compact"}),
                         ("gdn_indexed_copy", {"native", "gather"})):
        value = manifest.get(key)
        if not isinstance(value, str) or value not in allowed:
            raise ValueError(f"original OMC requires an explicit supported {key}")
        policies[key] = value
    original_graphs = manifest.get("graphs")
    if (not isinstance(original_graphs, list) or len(original_graphs) != 2
            or any(not isinstance(graph, dict) for graph in original_graphs)
            or [graph.get("name") for graph in original_graphs] != ["target", "draft"]):
        raise ValueError("original OMC requires target and draft graphs in order")
    graph_records = []
    for graph in original_graphs:
        expected = ({"inputs": ["input_ids", "attention_mask"], "outputs": ["target_top1", "target_features"],
                     "input_shapes": [[1, 8], [1, 8]], "input_dtypes": ["torch.int32", "torch.int32"],
                     "output_shapes": [[1, 8], [1, 8, 32]], "output_dtypes": ["torch.int32", "torch.float16"]}
                    if graph["name"] == "target" else
                    {"inputs": ["target_features", "input_ids", "attention_mask"], "outputs": ["draft_top1"],
                     "input_shapes": [[1, 8, 32], [1, 8], [1, 8]],
                     "input_dtypes": ["torch.float16", "torch.int32", "torch.int32"],
                     "output_shapes": [[1, 3]], "output_dtypes": ["torch.int32"]})
        _fields(graph, expected, graph["name"] + " ABI")
        for summary, policy in (("reduction_lowering", "fp16_reductions"),
                                ("row_mul_lowering", "fp16_row_mul"),
                                ("gdn_copy_lowering", "gdn_indexed_copy")):
            if summary in graph and (not isinstance(graph[summary], dict)
                                     or graph[summary].get("mode") != policies[policy]):
                raise ValueError(f"{graph['name']}: lowering summary disagrees with {policy}")
        record = {"path": graph.get("omc"), "bytes": graph.get("bytes"), "sha256": graph.get("sha256")}
        _payload(omc, record)
        payloads = graph.get("payloads", [record])
        if (not isinstance(payloads, list) or not payloads
                or not any(isinstance(item, dict) and all(item.get(key) == value for key, value in record.items())
                           for item in payloads)):
            raise ValueError("original OMC is absent from its payload list")
        for item in payloads:
            _payload(omc, item)
        graph_records.append({"name": graph["name"], "omc": record, "payloads": copy.deepcopy(payloads)})

    whole_pair = build_pair(target_config, draft_config, 8, expert_dispatch=dispatch,
                            gdn_indexed_copy=policies["gdn_indexed_copy"])
    shapes = weight_shapes(whole_pair)
    if set(records) != set(shapes):
        raise ValueError("prepared weights do not cover the supported whole pair exactly")
    paths = {}
    for key, shape in shapes.items():
        record = records[key]
        _fields(record, {"shape": list(shape), "dtype": "float16", "bytes": math.prod(shape) * 2},
                "prepared weight " + key)
        paths[key] = _payload(prepared, record)
    graph, stages = _layer0(target_config, draft_config, dispatch, policies["gdn_indexed_copy"])
    if len(graph.weights) != 19:
        raise ValueError("supported layer-0 graph must use exactly 19 original weight tensors")
    weights = {key: np.fromfile(paths[key], dtype="<f2").reshape(value.shape)
               for key, value in graph.weights.items()}
    if any(hashlib.sha256(value.tobytes()).hexdigest() != records[key]["sha256"]
           for key, value in weights.items()):
        raise ValueError("prepared layer-0 weight bytes changed during loading")
    if any(not np.isfinite(value).all() for value in weights.values()):
        raise ValueError("nonfinite prepared layer-0 weights")
    ids = np.zeros((1, 8), dtype=np.int32)
    ids[0, :len(prompt)] = prompt
    inputs = {"input_ids": ids}
    expected = evaluate(graph, inputs, weights)
    emit_options = {key: policies[key] for key in ("fp16_reductions", "fp16_row_mul")}
    final_graph = copy.deepcopy(graph)
    final_graph.name = "gdn_layer0_final"
    final_graph.outputs = [final_graph.outputs[0]]
    cases = [_case(final_graph, inputs, weights, [expected[0]], ["layer0_final"],
                   "Original first-layer embedding, GDN and MoE residuals with only the final block output; "
                   "control for optimization changes introduced by extra diagnostic outputs.", emit_options),
             _case(graph, inputs, weights, expected, stages,
                   "The same first layer with final output and semantic GDN normalization boundaries exposed.",
                   emit_options)]
    rsqrt = Graph("gdn_layer0_rsqrt")
    actual_input = rsqrt.input("actual_norm_input", stages["norm_input"].shape)
    control_input = rsqrt.input("normal_control", actual_input.shape)
    rsqrt.outputs = [rsqrt.op("rsqrt", value, shape=value.shape, name=name)
                     for value, name in ((actual_input, "actual_norm_gain"), (control_input, "control_norm_gain"))]
    rsqrt_inputs = {"actual_norm_input": np.ascontiguousarray(expected[list(stages).index("norm_input")]),
                   "normal_control": np.resize(np.array([0.25, 1, 4, 16], dtype=np.float16), actual_input.shape)}
    cases.append(_case(rsqrt, rsqrt_inputs, {}, evaluate(rsqrt, rsqrt_inputs),
                       ["actual_norm_gain", "control_norm_gain"],
                       "Rsqrt on the actual same-weight host norm input, with a same-shaped positive-normal runtime-input control; "
                       "this does not claim the input is an internal tensor captured from the original OMC.", emit_options))
    provenance = {"schema_version": 1, "source": copy.deepcopy(source),
                  "bindings": {"prepared_manifest_sha256": prepared_sha,
                               "reference_sha256": reference_sha,
                               "omc_manifest_sha256": manifest_sha},
                  "original_graphs": graph_records, "ddk_id": manifest["ddk_id"],
                  "omg_sha256": manifest["omg_sha256"], "policies": policies,
                  "expert_dispatch": dispatch, "sequence_length": 8, "prompt_token_ids": list(prompt),
                  "reference_first_token": generated[0][0], "weights_verified": len(records),
                  "selected_weights": {key: copy.deepcopy(records[key]) for key in sorted(weights)},
                  "stage_map": {label: {"name": value.name, "shape": list(value.shape), "dtype": value.dtype}
                                for label, value in stages.items()},
                  "limits": {"cases": 3, "sequence_length": 8, "layer": 0, "selected_weight_tensors": 19},
                  "limitations": [
                      "New probe models may change compiler optimization/partitioning; compare final-only and staged outputs to the original layer-0 feature capture.",
                      "Prepared/reference/OMC payload identities are checked; AIR is not read and a complete prepared-to-AIR document chain is not asserted.",
                      "The Rsqrt input is evaluated by the host neutral IR, not captured from inside the original OMC.",
                      "FP16 probe tolerances are unchanged; the 0.002 absolute tolerance can hide zeroing of a tiny chunk core, so raw magnitudes and error metrics remain necessary.",
                      "Batch-one GDN retains padded rows in the production helper; only the prefix positions are prompt tokens.",
                      "This preparation neither executes a device nor identifies a runtime backend or establishes a GDN root cause."]}
    if any(sha256_file(path) != digest for path, digest in (
            (prepared_path, prepared_sha), (reference_path, reference_sha), (manifest_path, manifest_sha))):
        raise ValueError("prepared/reference/OMC metadata changed during diagnostic preparation")
    return cases, provenance
