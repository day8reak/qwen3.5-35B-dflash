"""Offline diagnostics for the original tiny target feature ABI, not accuracy acceptance."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .artifacts import checked_file, output_path, save_json
from .ge_eval import evaluate
from .ge_export import build_pair, configurations, weight_identity, weight_shapes
from .hashing import sha256_file
from .hdc_run import _smoke_checks


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _difference(expected, actual):
    expected, actual = expected.astype(np.float64), actual.astype(np.float64)
    finite = np.isfinite(expected) & np.isfinite(actual)
    result = {"elements": int(expected.size), "finite_pairs": int(finite.sum()),
              "source_nonfinite": int((~np.isfinite(expected)).sum()),
              "board_nonfinite": int((~np.isfinite(actual)).sum()),
              "numerically_equal": bool(finite.all() and np.array_equal(expected, actual))}
    # Do not turn nonfinite failures into finite-pair summary statistics.
    if not finite.all():
        return result
    delta = actual - expected
    location = np.unravel_index(np.argmax(np.abs(delta)), delta.shape)
    norm = float(np.linalg.norm(expected))
    result.update(max_abs=float(np.max(np.abs(delta))),
                  rms=float(np.sqrt(np.mean(delta * delta))),
                  relative_l2=float(np.linalg.norm(delta) / norm) if norm else None,
                  max_abs_index=list(map(int, location)),
                  source_at_max=float(expected[location]), board_at_max=float(actual[location]))
    return result


def compare_target_features(prepared_dir, omc_dir, board_dir, reference_path, output_dir):
    """Compare hash-bound captured bytes against the same-weight neutral IR.

    Restricted to the existing tiny configuration and at most 16 tokens. This
    does not recompile models, change the golden reference or run a device.
    """
    prepared, omc, board = [Path(p).expanduser().resolve() for p in (prepared_dir, omc_dir, board_dir)]
    reference_path = Path(reference_path).expanduser().resolve()
    out = output_path(output_dir)
    if out.exists():
        raise FileExistsError(f"diagnostic output already exists: {out}")
    plan = _read(prepared / "prepared.json")
    target_config, draft_config = configurations(tiny=True)
    if (plan.get("artifact_kind") != "qwen35-ge-prepared" or plan.get("status") != "PREPARED"
            or plan.get("tiny") is not True or plan.get("precision") != "float16"
            or plan.get("quantization") is not None
            or plan.get("target_config") != target_config or plan.get("draft_config") != draft_config):
        raise ValueError("feature diagnostics require the supported unquantized FP16 tiny prepared configuration")
    sequence = plan.get("sequence_length")
    if type(sequence) is not int or not 2 <= sequence <= 16:
        raise ValueError("feature diagnostics support tiny sequence lengths 2 through 16")
    source = plan.get("source", {})
    if source.get("kind") != "ge-tiny-numpy-random" or source.get("not_upstream_weights") is not True:
        raise ValueError("prepared tiny source identity is missing")
    records = plan.get("weights", {})
    if not records or source.get("fp16_payload_sha256") != weight_identity(records):
        raise ValueError("prepared source identity differs from weight records")

    manifest, run = _read(omc / "manifest.json"), _read(board / "result.json")
    reference, report = _read(reference_path), _read(board / "benchmark.json")
    if (manifest.get("artifact_kind") != "qwen35-cpro-omc-pair" or manifest.get("status") != "COMPILED"
            or manifest.get("source") != source or manifest.get("precision") != "torch.float16"
            or manifest.get("quantization") is not None or manifest.get("abi_version") != 2
            or manifest.get("feature_dtype") != "torch.float16"
            or manifest.get("sequence_length") != sequence or manifest.get("feature_width") != 32
            or manifest.get("vocab_size") != 64 or manifest.get("draft_width") != 3
            or manifest.get("pad_token_id") != 0 or manifest.get("eos_token_ids") != []
            or manifest.get("expert_dispatch") != plan.get("expert_dispatch")):
        raise ValueError("OMC identity/configuration differs from prepared tiny pair")
    graphs = manifest.get("graphs", [])
    if len(graphs) != 2 or {g["name"] for g in graphs} != {"target", "draft"}:
        raise ValueError("OMC must contain exactly target and draft")
    for graph in graphs:
        checked_file(omc, {"path": graph["omc"], "bytes": graph["bytes"], "sha256": graph["sha256"]})
        expected_abi = ({"inputs": ["input_ids", "attention_mask"],
                         "outputs": ["target_top1", "target_features"],
                         "input_shapes": [[1, sequence], [1, sequence]],
                         "input_dtypes": ["torch.int32", "torch.int32"],
                         "output_shapes": [[1, sequence], [1, sequence, 32]],
                         "output_dtypes": ["torch.int32", "torch.float16"]}
                        if graph["name"] == "target" else
                        {"inputs": ["target_features", "input_ids", "attention_mask"], "outputs": ["draft_top1"],
                         "input_shapes": [[1, sequence, 32], [1, sequence], [1, sequence]],
                         "input_dtypes": ["torch.float16", "torch.int32", "torch.int32"],
                         "output_shapes": [[1, 3]], "output_dtypes": ["torch.int32"]})
        if any(graph.get(key) != value for key, value in expected_abi.items()):
            raise ValueError(f"{graph['name']}: OMC ABI differs from supported tiny graph")
    bindings = {"manifest_sha256": sha256_file(omc / "manifest.json"),
                "reference_sha256": sha256_file(reference_path),
                "benchmark_sha256": sha256_file(board / "benchmark.json")}
    if (run.get("status") != "EXECUTION_COMPLETE" or run.get("device_execution") != "PASS"
            or run.get("capture_target_features") is not True or run.get("source") != source
            or run.get("precision") != manifest["precision"] or run.get("quantization") is not None
            or run.get("ddk_id") != manifest["ddk_id"]
            or any(run.get(key) != value for key, value in bindings.items())):
        raise ValueError("board run is not a completed, hash-bound target feature capture")
    if (reference.get("status") != "PASS" or reference.get("source") != source
            or reference.get("sequence_length") != sequence or reference.get("precision") != "torch.float16"
            or reference.get("quantization") is not None
            or reference.get("prepared_manifest_sha256") != sha256_file(prepared / "prepared.json")):
        raise ValueError("reference identity differs from prepared tiny pair")
    prompt = reference.get("prompt_token_ids")
    if (not isinstance(prompt, list) or not 1 <= len(prompt) <= sequence
            or any(type(token) is not int or not 0 <= token < 64 for token in prompt)):
        raise ValueError("invalid reference prompt for tiny target")
    reference_tokens = reference.get("ordinary", {}).get("token_ids")
    if (not isinstance(reference_tokens, list) or not reference_tokens
            or any(type(token) is not int or not 0 <= token < 64 for token in reference_tokens)):
        raise ValueError("reference must contain nonempty valid ordinary token IDs")
    runtime, firmware = run.get("runtime_version"), run.get("firmware_id")
    if not isinstance(runtime, str) or not runtime or not isinstance(firmware, str) or not firmware:
        raise ValueError("captured report is missing runtime/firmware identity")
    checks = _smoke_checks(report, manifest, reference, runtime, firmware, capture_target_features=True)
    if not all(checks.values()):
        raise ValueError("invalid captured smoke report: " + ", ".join(k for k, v in checks.items() if not v))

    copy_mode = manifest.get("gdn_indexed_copy", "native")
    graphs = build_pair(target_config, draft_config, sequence,
                        expert_dispatch=plan["expert_dispatch"], gdn_indexed_copy=copy_mode)
    shapes = weight_shapes(graphs)
    if records.keys() != shapes.keys():
        raise ValueError("prepared weights do not cover the tiny graph exactly")
    weights = {}
    for key, shape in shapes.items():
        record = records[key]
        if (record.get("shape") != list(shape) or record.get("dtype") != "float16"
                or record.get("bytes") != math.prod(shape) * 2):
            raise ValueError(f"invalid prepared FP16 weight descriptor: {key}")
        weights[key] = np.fromfile(checked_file(prepared, record), dtype="<f2").reshape(shape)
        if not np.isfinite(weights[key]).all():
            raise ValueError(f"nonfinite prepared weight: {key}")
    target = graphs[0]
    # S <= 16 has exactly one vocabulary-head chunk; append its input without
    # changing the two outputs of the original compiled target.
    heads = [node for node in target.nodes if node.kind == "argmax"]
    if len(heads) != 1 or heads[0].inputs[0].shape != (1, sequence, 64):
        raise ValueError("unexpected tiny target vocabulary head")
    target.outputs.append(heads[0].inputs[0])
    ids, mask = np.zeros((1, sequence), np.int32), np.zeros((1, sequence), np.int32)
    ids[0, :len(prompt)], mask[0, :len(prompt)] = prompt, 1
    source_top1, source_features, logits = evaluate(target, {"input_ids": ids, "attention_mask": mask}, weights)
    capture = report["execution"]["target_features"]
    raw = bytes.fromhex(capture["data"])
    actual = np.frombuffer(raw, dtype="<f2").reshape(capture["shape"])
    valid = len(prompt)
    source_tokens = source_top1[0, :valid].tolist()
    board_tokens = report["execution"]["target_top1"][:valid]
    layers = []
    width = target_config["hidden_size"]
    for slot, layer in enumerate(draft_config["dflash_config"]["target_layer_ids"]):
        expected = source_features[0, :valid, slot * width:(slot + 1) * width]
        observed = actual[0, :valid, slot * width:(slot + 1) * width]
        layers.append({"layer": layer, "feature_columns": [slot * width, (slot + 1) * width],
                       "valid_tokens": _difference(expected, observed),
                       "per_token": [{"prompt_index": i, **_difference(expected[i], observed[i])}
                                     for i in range(valid)]})
    row = logits[0, valid - 1].astype(np.float64)
    head = {"source_nonfinite": int((~np.isfinite(row)).sum())}
    if np.isfinite(row).all():
        order = np.argsort(-row, kind="stable")[:5]
        head.update(source_top5=[{"token": int(i), "logit": float(row[i])} for i in order],
                    source_top1_margin=float(row[order[0]] - row[order[1]]),
                    source_logit_of_board_token=float(row[board_tokens[-1]]))
    reference_first = reference_tokens[0]
    result = {"schema_version": 1, "status": "DIAGNOSTIC_COMPLETE", "accuracy": "NOT_EVALUATED",
              "scope": "Original OMC target feature output vs same-weight host neutral IR; no acceptance threshold or OMC arithmetic emulation.",
              "source": source, "bindings": {**bindings, "prepared_manifest_sha256": sha256_file(prepared / "prepared.json"),
                                             "board_result_sha256": sha256_file(board / "result.json")},
              "binding_scope": "Prepared/reference/OMC share the checked payload identity and tiny configuration; board report is bound to OMC/manifest/reference hashes. AIR is not read, so this is not a complete prepared-to-AIR document hash chain.",
              "host_code_sha256": {name: sha256_file(Path(__file__).parent / name) for name in
                                   ("ge_target_features.py", "ge_target.py", "ge_draft.py", "ge_ir.py",
                                    "ge_eval.py", "ge_export.py", "config.py")},
              "recorded_runner_sha256": run.get("runner_sha256"),
              "numpy": np.__version__, "prompt_token_ids": prompt,
              "sequence_length": sequence, "capture_shape": capture["shape"],
              "runtime_version": runtime, "firmware_id": firmware,
              "omc_lowerings": {key: manifest.get(key, "native") for key in
                                ("fp16_reductions", "fp16_row_mul", "gdn_indexed_copy")},
              "source_valid_top1": source_tokens, "board_valid_top1": board_tokens,
              "first_valid_top1_mismatch": next((i for i, (a, b) in enumerate(zip(source_tokens, board_tokens)) if a != b), None),
              "first_generated_token": {"prompt_index": valid - 1, "reference": reference_first,
                                        "source": source_tokens[-1], "board": board_tokens[-1],
                                        "source_matches_reference": source_tokens[-1] == reference_first},
              "source_head": head, "layers": layers,
              "limitations": ["Only valid prompt positions are compared; padded positions are excluded.",
                              "Each captured boundary follows a complete target block including GDN, normalization and MoE; it does not isolate GDN.",
                              "The neutral IR rounds at FP16 boundaries; compiler rewrites and device kernels may differ.",
                              "Numerical equality at these boundaries does not validate later layers, the vocabulary head, draft or full generation."]}
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / "report.json", result)
    lines = ["Original target feature diagnostics", "Status: DIAGNOSTIC_COMPLETE",
             "First generated token: " + json.dumps(result["first_generated_token"]),
             "Valid source top1: " + str(source_tokens), "Valid board top1: " + str(board_tokens)]
    for layer in layers:
        lines.append(f"Layer {layer['layer']} valid features: " + json.dumps(layer["valid_tokens"]))
        for row in layer["per_token"]:
            lines.append("  " + json.dumps(row))
    lines.extend(["Source head: " + json.dumps(head), result["scope"], *result["limitations"]])
    (out / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
