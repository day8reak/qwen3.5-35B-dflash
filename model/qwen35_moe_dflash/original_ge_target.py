"""Review the original Cpro source-to-GE contract without creating a graph.

The functional quant_target frontend is not the original 89-input RefData ABI.
Only explicit original-source receipts and actual AIR metadata are considered
here. A tap is a structural candidate, never proof of source equivalence.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


POLICY = "original-cpro-ge-binding-review-v1"
LAYER_IDS = (1, 6, 11, 16, 22, 27, 32, 37)
# Exact receipt of the user-supplied texts already audited locally. Only the
# five Python hashes bind the computation audit. YAML retains provenance but
# resource-path/format changes do not prove a different computation revision.
SOURCE_SHA256 = {
    "config/qwen3_5_moe.yaml": "0e894f64c4bffe253e716b2eed005c56105fc16c9b6bb13e949108d4b1c0ee64",
    "export_glm_model_qwen3_5_moe.py": "b72b05d7b07cdc096eadfed92317b38b88810fd24c33c3bdf66c625c888cb935",
    "models/export_model_wrapper_qwen3_5_moe.py": "fbd152783ac93d1fb0e525a24eb1d73db3ab2fbaa100cd7806d8eda890991e97",
    "models/modeling_qwen3_5_moe_hiai_cpro.py": "e994b0e194efee0ade63478647f0b11acd8a8bc40ba9b188a001a12f7d20651f",
    "models/modeling_qwen3_5_moe_hiai_nz.py": "f7ea28355ffa2fa2d92b2e1d171041ce72366c9437ecd68554797e961b0d6739",
    "utils.py": "67eac2adba648d0a8f6f35e31bffc7745ca7fc9480eaefe6603a8b7c5dc2c6c9",
}
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _identity(path):
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def read_json(path, max_bytes=16 << 20):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > max_bytes:
        raise ValueError("metadata must be a bounded regular non-symlink file")
    before = _identity(path)
    with path.open("rb") as stream:
        raw = stream.read(max_bytes + 1)
    if len(raw) > max_bytes or _identity(path) != before:
        raise ValueError("metadata grew beyond its read bound or changed")
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError("duplicate metadata key")
            result[key] = value
        return result
    def constant(value):
        raise ValueError("nonfinite metadata is unsupported")
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    if not isinstance(value, dict):
        raise ValueError("metadata root must be an object")
    return value, hashlib.sha256(raw).hexdigest()


def source_receipt(source_root):
    """Hash only the declared six small code/config files; no model import."""
    root = Path(source_root).resolve()
    rows = []
    for name in SOURCE_SHA256:
        path = root / name
        if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= 1 << 20:
            raise ValueError("missing, symlinked or excessive original source file: " + name)
        before = _identity(path)
        with path.open("rb") as stream:
            raw = stream.read((1 << 20) + 1)
        if len(raw) > 1 << 20 or _identity(path) != before:
            raise ValueError("source changed during bounded read")
        rows.append({"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    return {"source": "READ_ONLY_SIX_SOURCE_FILES", "files": rows}


def audit_source(receipt):
    if receipt.get("artifact_kind") == "qwen35-original-target-feature-source":
        sources = receipt.get("source_files")
        if not isinstance(sources, dict):
            raise ValueError("derived source receipt requires its original source_files")
        rows = [dict(value, path=name) for name, value in sources.items() if isinstance(value, dict)]
        receipt_selection = "ORIGINAL_SOURCE_FILES_ONLY; GENERATED_FEATURE_SOURCE_IGNORED"
    else:
        rows = receipt.get("files")
        receipt_selection = "ORIGINAL_SIX_FILE_RECEIPT"
    if not isinstance(rows, list) or len(rows) != 6:
        raise ValueError("original source audit requires exactly six file receipts")
    actual = {}
    for row in rows:
        name = row.get("path") if isinstance(row, dict) else None
        digest = row.get("sha256", "") if isinstance(row, dict) else ""
        if (name not in SOURCE_SHA256 or name in actual or not isinstance(digest, str) or not _SHA.fullmatch(digest)
                or type(row.get("bytes")) is not int or not 0 < row["bytes"] <= 1 << 20):
            raise ValueError("invalid or duplicate original source receipt")
        actual[name] = digest
    changed = sorted(name for name, digest in actual.items() if name.endswith(".py") and digest != SOURCE_SHA256[name])
    return {"status": "MATCHED_AUDITED_SOURCE_RECEIPTS" if not changed else "UNSUPPORTED_SOURCE_REVISION",
            "files": rows, "different_files": changed,
            "receipt_selection": receipt_selection,
            "configuration": {"path": "config/qwen3_5_moe.yaml",
                "sha256": actual["config/qwen3_5_moe.yaml"],
                "matches_audited_text": actual["config/qwen3_5_moe.yaml"] == SOURCE_SHA256["config/qwen3_5_moe.yaml"],
                "role": "PROVENANCE_ONLY; resource paths/format may differ; no configuration semantics accepted from the hash",
                "geometry": "40 layers/vocabulary/dimensions/expert extent still require actual AIR and model configuration evidence"},
            "scope": "Receipt identity only unless --source-root rehashes actual source; no source-to-AIR revision proof"}


def required_operators():
    """Source calls and exact custom converters; builtins need their own proof."""
    return [
        {"role": "full_attention", "source": "modeling_qwen3_5_moe_hiai_cpro.py:333-371; exporter:107-149",
         "source_call": "torch_npu.adn_fused_infer_attention", "exact_custom_ge_type": "AdnFusedInferAttention", "count": 10,
         "contract": "NZ/paged BNSD; block_size=32; K/V [64,32,32,16]; FP16 query/mask/cache; allQLen -> GE all_seq_lengths_q, not pse_shift"},
        {"role": "cache_update", "source": "modeling_cpro:283-296; exporter:37-66", "source_call": "torch_npu.npu_cache_update_",
         "exact_custom_ge_type": "CacheUpdate", "count": 20,
         "contract": "inputs x,updates,targetBlock,offsetInBlock; original FP16 blocked cache; preserve functionalize copy-back and original position handling"},
        {"role": "conv_state", "source": "modeling_cpro:644-652; exporter:185-218",
         "source_call": "torch.ops.my_ops.linear_attention_convstate_update", "exact_custom_ge_type": "LinearAttentionConvstateUpdate", "count": 30,
         "contract": "x1 projected [1,T,8192], x2 FP16 [1,8192,4], x3 INT32[128]; two outputs; original conv_state.copy_"},
        {"role": "gdn", "source": "modeling_cpro:667-681; exporter:160-182", "source_call": "torch_npu.npu_chunk_gated_delta_rule",
         "exact_custom_ge_type": "ChunkGatedDeltaRule", "count": 30,
         "contract": "Q/K/V [1,T,32,128]; source g expression is FP32 from a.float(), beta is sigmoid of projected b; preserve actual g/beta port precision and any exported casts; FP16 initial recurrent state; INT16 effective_length; final FP32 state cast FP16 before copy_; chunk64/1; internal QK L2 enabled"},
        {"role": "dynamic_quant", "source": "modeling_cpro:38-57,783-815", "source_call": "torch_npu.npu_dynamic_quant",
         "ge_family": "DynamicQuant", "selected_ge_type": None,
         "contract": "original FP32/FP16 activation boundaries -> INT8 plus FP32 per-token scale; do not import the functional frontend's FP16-only assumption"},
        {"role": "quant_matmul", "source": "modeling_cpro:49-53", "source_call": "torch_npu.npu_quant_matmul",
         "ge_family": "QuantBatchMatMul", "selected_ge_type": None,
         "contract": "original A8W4 I4ZN INT32 weights, group256 scales, pertoken_scale, FP16 result; preserve original weight bytes and layout"},
        {"role": "moe_router", "source": "modeling_cpro:979-983", "source_call": "torch_npu.npu_moe_gating_top_k_softmax",
         "ge_family": "MoeGatingTopKSoftmax", "selected_ge_type": None, "logical_count": 40,
         "contract": "all original experts and original configured top_k; optional renormalization and dtype cast retained"},
        {"role": "moe_init", "source": "modeling_cpro:933-951", "source_call": "torch_npu.npu_moe_init_routing_v2",
         "ge_family": "MoeInitRoutingV2", "selected_ge_type": None, "logical_count": 40,
         "contract": "quant_mode=-1; active_expert_range=[0,num_experts]; expert_tokens_num_type=1; tokens_per_expert cast INT64"},
        {"role": "moe_gmm", "source": "modeling_cpro:790-815", "source_call": "torch_npu.npu_grouped_matmul",
         "ge_family": "GroupedMatMul", "selected_ge_type": None, "logical_count": 80,
         "contract": "W4A8 full-expert GMM twice/layer; INT64 vendor qparam scale; group_list_type=1, split_item=3, group_type=0, act_type=0; FP16 output; intermediate SiLU*up and requant retained"},
        {"role": "moe_finalize", "source": "modeling_cpro:955-963", "source_call": "torch_npu.npu_moe_finalize_routing",
         "ge_family": "MoeFinalizeRouting", "selected_ge_type": None, "logical_count": 40,
         "contract": "original expanded_src_to_dst_row, topk scales cast to expert output dtype, drop_pad_mode=2; shared expert and sigmoid gate also retained"},
        {"role": "norm", "source": "wrapper:160-162; modeling_cpro:116-118,129-135", "source_call": "torch_npu.npu_rms_norm",
         "ge_family": "Cpro builtin RMS; not automatically AdnRmsNorm", "selected_ge_type": None,
         "contract": "ordinary source load stores 1+HF, forward passes stored_weight+1; preserve both source statements; gated RMS computes in FP32 with stored gamma then casts back; exact builtin converter required",
         "static_source_vs_functional": {"original_ordinary_gamma": "HF+2 when both audited source statements execute",
             "original_gated_gamma": "HF+1 from the audited load, passed directly in gated forward",
             "existing_functional_ordinary_gamma": "raw HF+1", "existing_functional_gated_gamma": "raw HF",
             "verdict": "STATIC_ARITHMETIC_DIFFERENCE; no automatic fix; actual original-machine export/AIR is final evidence"}},
    ]


def _port(edge):
    if not isinstance(edge, str) or ":" not in edge:
        raise ValueError("missing explicit producer port")
    name, index = edge.rsplit(":", 1)
    if not name or not re.fullmatch(r"0|[1-9][0-9]*", index):
        raise ValueError("invalid producer port")
    return name, int(index)


def resolve_taps(graph, gamma_template=None):
    """Resolve only a unique direct FileConstant -> RmsNorm gamma candidate.

    Add/Cast/fused paths require extra, value-bound evidence and are rejected.
    Even a direct path remains a candidate: the source gamma +1 operation must
    be reconciled before claiming a faithful source-to-GE binding.
    """
    result = {"status": "UNRESOLVED", "taps": [], "semantic_validation": "NOT_ESTABLISHED"}
    if (not isinstance(gamma_template, str) or gamma_template.count("{layer}") != 1
            or "{" in gamma_template.replace("{layer}", "") or "}" in gamma_template.replace("{layer}", "")):
        result["reason"] = "EXPLICIT_EXACT_GAMMA_PATH_TEMPLATE_REQUIRED"
        return result
    nodes, constants = graph.get("operations"), graph.get("file_constants", {})
    if (not isinstance(nodes, list) or not 1 <= len(nodes) <= 32768 or constants.get("shown") != constants.get("count")):
        result["reason"] = "FULL_OPERATION_AND_FILECONSTANT_METADATA_REQUIRED"
        return result
    by_name = {}
    for node in nodes:
        name = node.get("name")
        if not isinstance(name, str) or name in by_name:
            raise ValueError("missing or duplicate graph operation name")
        by_name[name] = node
    consumers = {}
    for node in nodes:
        for index, edge in enumerate(node.get("input", [])):
            if isinstance(edge, str) and edge.endswith(":-1"):
                continue
            key = _port(edge)
            consumers.setdefault(key, []).append((node, index))
    for layer in LAYER_IDS:
        path = gamma_template.format(layer=layer + 1)
        row = {"post_decoder_layer_index": layer, "next_input_norm_layer_index": layer + 1,
               "exact_gamma_file_path": path, "status": "UNRESOLVED"}
        result["taps"].append(row)
        matches = [c for c in constants.get("examples", []) if c.get("file_path") == path]
        if len(matches) != 1 or matches[0].get("name") not in by_name:
            row["reason"] = "EXACT_FILECONSTANT_PATH_MISSING_OR_AMBIGUOUS"
            continue
        gamma = by_name[matches[0]["name"]]
        if (gamma.get("type") != "FileConstant" or len(gamma.get("output_desc", [])) != 1
                or matches[0].get("outputs") != gamma["output_desc"]):
            row["reason"] = "FILECONSTANT_NODE_OR_PORT_AMBIGUOUS"
            continue
        gamma_desc = gamma["output_desc"][0]
        gamma_dtypes = gamma_desc.get("schema_dtype_names", [])
        if (gamma_desc.get("shape") != [2048] or not gamma_dtypes
                or not set(gamma_dtypes) <= {"DT_FLOAT", "DT_FLOAT32", "DT_FLOAT16"}):
            row["reason"] = "GAMMA_SHAPE_OR_DTYPE_UNVERIFIED"
            continue
        uses = consumers.get((gamma["name"], 0), [])
        if len(uses) != 1:
            row["reason"] = "GAMMA_CONSUMER_MISSING_OR_AMBIGUOUS"
            continue
        norm, input_index = uses[0]
        if norm.get("type") != "RmsNorm" or input_index != 1:
            row.update(reason="GAMMA_TRANSFORM_OR_NORM_CONVERTER_UNVERIFIED", actual_consumer=norm["name"],
                       actual_consumer_type=norm.get("type"), actual_input_port=input_index)
            continue
        ins, descs = norm.get("input", []), norm.get("input_desc", [])
        if len(ins) != 2 or len(descs) != 2 or descs[1] != gamma["output_desc"][0]:
            row["reason"] = "NORM_INTERFACE_DESCRIPTORS_DIFFER"
            continue
        producer_name, producer_port = _port(ins[0])
        producer = by_name.get(producer_name)
        if producer is None or producer is norm or producer_port >= len(producer.get("output_desc", [])):
            row["reason"] = "NORM_DATA_PRODUCER_UNRESOLVED"
            continue
        desc = producer["output_desc"][producer_port]
        shape, dtype = desc.get("shape"), desc.get("schema_dtype_names", [])
        if (desc != descs[0] or not isinstance(shape, list) or len(shape) != 3 or shape[0] != 1
                or shape[2] != 2048 or type(shape[1]) is not int or not (shape[1] == -1 or 1 <= shape[1] <= 2048)
                or not dtype or not set(dtype) <= {"DT_FLOAT", "DT_FLOAT32", "DT_FLOAT16"}):
            row["reason"] = "NORM_DATA_PORT_SHAPE_OR_PRECISION_UNVERIFIED"
            continue
        row.update(status="STRUCTURAL_CANDIDATE_NOT_SOURCE_EQUIVALENCE", gamma_reference=gamma["name"] + ":0",
                   norm=norm["name"], producer_reference=ins[0], descriptor=desc,
                   remaining="Verify source stored_gamma+1 and next-layer identity against actual original converter; then add only feature-branch FP16 snapshot")
    if all(row["status"] == "STRUCTURAL_CANDIDATE_NOT_SOURCE_EQUIVALENCE" for row in result["taps"]):
        if len({row["producer_reference"] for row in result["taps"]}) == 8:
            result["status"] = "EIGHT_STRUCTURAL_CANDIDATES_SOURCE_BINDING_PENDING"
        else:
            result["reason"] = "TAP_PRODUCER_REUSED_ACROSS_LAYERS"
    return result


def prepare_contract(receipt, air_report=None, *, gamma_template=None, air=None):
    source = audit_source(receipt)
    operators = required_operators()
    result = {"schema_version": 1, "artifact_kind": "qwen35-original-ge-target-contract", "policy": POLICY,
              "status": "PENDING_GRAPH_BINDING", "source_audit": source, "required_operators": operators,
              "target": {"layers": 40, "linear_layers": 30, "full_attention_layers": 10,
                  "hidden": 2048, "vocabulary": 248320, "experts": "ALL_ORIGINAL_EXPERTS; config/table extent binding pending"},
              "abi": {"inputs": 89, "refdata": 80, "original_outputs": 81, "feature_outputs": 82,
                  "features": {"layer_indices": list(LAYER_IDS), "dtype": "float16", "shape": [1, "T", 16384]},
                  "embedding": "external INT8 table rows times FP32 per-token scale once; no dynamic requantization or synthetic embedding substitution",
                  "precision": "Preserve every actual source FP32/FP16 boundary; feature-only snapshot Cast is explicit",
                  "state": "Original blocked K/V and FP16 conv/recurrent RefData; preserve all 80 input/output slots and positional control logic"},
              "weight_policy": "KEEP_EVERY_ORIGINAL_FILECONSTANT_REFERENCE_AND_PAYLOAD_BYTE; NOT_OPENED_NOT_HASHED",
              "frontend_routes": {"existing_recipe_emitter": "Reusable mechanics in ge_recipe.emit_recipe_cpp; not a faithful original target graph or original RefData ABI",
                  "original_air_ge_api_append": "Conditional Graph.LoadFromFile -> explicit verified producer ports -> feature-only FP16 snapshot/ConcatD -> original ordered outputs+feature -> SaveToFile; writer implementation awaits verified source/graph bindings",
                  "functional_quant_target": "NOT_ORIGINAL_ABI_NOT_USED_AS_EQUIVALENCE_EVIDENCE"},
              "tap_resolution": {"status": "UNRESOLVED", "reason": "ACTUAL_FULL_AIR_GRAPH_METADATA_MISSING"},
              "pending": ["ACTUAL_ORIGINAL_GRAPH_BINDING", "SELECTED_BUILTIN_CONVERTER_FULL_PORT_SCHEMAS_AND_DTYPES",
                          "EXACT_8_POST_DECODER_PRODUCER_PROVENANCE", "LOSSLESS_GE_IMPORT_APPEND_EXPORT_ROUNDTRIP",
                          "ORIGINAL_FILECONSTANT_STORAGE_AVAILABILITY", "GEAR64_TO_GEAR1_STATE_CONTINUITY"],
              "air_generation": "NOT_RUN_NO_GRAPH_WRITER_OR_RANDOM_WEIGHTS", "model_compilation": "NOT_RUN", "device_execution": "NOT_RUN"}
    if source["status"] != "MATCHED_AUDITED_SOURCE_RECEIPTS":
        result["status"] = "PENDING_SOURCE_BINDING"
        result["pending"].insert(0, "REVIEW_UNSUPPORTED_SOURCE_REVISION")
    if air_report is not None:
        if air_report.get("artifact_kind") != "qwen35-original-air-inspection":
            raise ValueError("requires a report from inspect_original_target_air.py")
        identity = air_report.get("air", {})
        if not _SHA.fullmatch(str(identity.get("sha256", ""))) or air_report.get("source_preservation") != "UNCHANGED_BY_STAT":
            raise ValueError("AIR report lacks unchanged source/hash evidence")
        result["air_identity"] = identity
        air = identity.get("path") or air
        candidates = [row for row in air_report.get("candidates", []) if isinstance(row.get("graph"), dict)
                      and row.get("format") in {"RAW_GRAPHDEF", "RAW_MODELDEF"}]
        if len(candidates) == 1:
            graph = candidates[0]["graph"]
            result["graph_observation"] = {"format": candidates[0]["format"], "input_count": graph.get("input_count"),
                "output_count": graph.get("output_count"), "original_boundary_check": graph.get("original_boundary_check"),
                "root_op_counts": graph.get("root_op_counts"), "source_to_air_revision": "UNVERIFIED"}
            result["tap_resolution"] = resolve_taps(graph, gamma_template)
            counts = graph.get("root_op_counts", {})
            for op in operators:
                typ = op.get("exact_custom_ge_type")
                op["graph_binding"] = ({"status": "COUNT_OBSERVED_PORTS_UNVERIFIED" if counts.get(typ, 0) == op["count"] else "COUNT_DIFFERENT",
                    "observed_count": counts.get(typ, 0), "expected_count": op["count"]} if typ else
                    {"status": "PENDING_SELECTED_BUILTIN_CONVERTER; family name is not a port binding"})
        else:
            result["graph_observation"] = {"status": "NO_UNAMBIGUOUS_RAW_ORIGINAL_GRAPH", "candidate_count": len(candidates),
                                           "container_graphs": "NOT_TREATED_AS_EDITABLE_RAW_AIR"}
            if not any(isinstance(row.get("graph"), dict) for row in air_report.get("candidates", [])):
                if result["status"] != "PENDING_SOURCE_BINDING":
                    result["status"] = "PENDING_AIR_SCHEMA"
                result["pending"].insert(0, "MATCH_ORIGINAL_TORCHAIR_GE_SERIALIZED_DESCRIPTOR_WITH_RAW_AIR")
                result["schema_observation"] = {"selected_schema": air_report.get("schema"),
                    "format_attempts": [{"format": row.get("format"), "status": row.get("status"),
                                         "error": row.get("error")} for row in air_report.get("candidates", [])],
                    "scope": "HIAI schema incompatibility is not proof of malformed AIR; strict decoder remains unchanged"}
    result["next_command"] = {"script": "targets/cpro/scripts/inspect_original_target_air.py",
        "arguments": ["--air", str(air or "ORIGINAL_TARGET_AIR"), "--full"],
        "stdout_destination": "$AI_RUN_DIR/original-target-air-full.json",
        "purpose": "Collect the actual bounded graph metadata without opening weights; rerun this contract audit with --air-report and an explicit exact --gamma-template"}
    if "schema_observation" in result:
        result["next_command"].update(status="REQUIRES_MATCHING_TORCHAIR_GE_DESCRIPTOR_INPUT",
            arguments=["--find-torchair-schema"],
            followup_arguments=["--air", str(air or "ORIGINAL_TARGET_AIR"), "--full",
                                "--torchair-pb2", "ACTUAL_ORIGINAL_TORCHAIR_GE_PB2"],
            purpose="Locate the original TorchAir GE pb2 without importing its package, then explicitly select its AST descriptor; repeating the selected HIAI schema cannot close this gate")
    return result
