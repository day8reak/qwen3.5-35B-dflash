"""Host entry points; generated data always belongs to AI_RUN_DIR."""
import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from .artifacts import checked_file, output_path, save_json
from .resources import load_resources, resource_path
from .hashing import sha256_file
from .diagnostics import format_error, probe_reference_import


def _generation_difference(expected_tokens, expected_stop, actual_tokens, actual_stop):
    """Bounded diagnostic detail; this does not define an acceptance gate."""
    expected_valid = isinstance(expected_tokens, list)
    actual_valid = isinstance(actual_tokens, list)
    mismatch = None
    if expected_valid and actual_valid:
        for index in range(max(len(expected_tokens), len(actual_tokens))):
            expected_present = index < len(expected_tokens)
            actual_present = index < len(actual_tokens)
            expected = expected_tokens[index] if expected_present else None
            actual = actual_tokens[index] if actual_present else None
            if not expected_present or not actual_present or expected != actual:
                mismatch = {"generated_index": index, "expected_token": expected,
                            "actual_token": actual, "expected_present": expected_present,
                            "actual_present": actual_present}
                break
    return {"tokens_equal": expected_valid and actual_valid and expected_tokens == actual_tokens,
            "stop_equal": isinstance(expected_stop, str) and isinstance(actual_stop, str)
                          and expected_stop == actual_stop,
            "expected_length": len(expected_tokens) if expected_valid else None,
            "actual_length": len(actual_tokens) if actual_valid else None,
            "expected_stop": expected_stop, "actual_stop": actual_stop,
            "first_token_mismatch": mismatch}


def _comparison_diagnostics(board, reference):
    """Separate repetition consistency, cross-mode agreement and reference accuracy."""
    expected = reference["ordinary"]
    modes = {}
    for mode in ("ordinary", "dflash"):
        data = board[mode]
        rows = data["measurements"]
        first = rows[0] if rows and isinstance(rows[0], dict) else {}
        reference_differences, repetition_differences, stable_differences = [], [], []
        for row in rows:
            row = row if isinstance(row, dict) else {}
            args = (row.get("generated_token_ids"), row.get("stop_reason"))
            reference_differences.append(_generation_difference(
                expected["token_ids"], expected["stop_reason"], *args))
            repetition_differences.append(_generation_difference(
                first.get("generated_token_ids"), first.get("stop_reason"), *args))
            stable_differences.append(_generation_difference(
                data["stable_generated_token_ids"], data["stable_stop_reason"], *args))

        def first_failure(differences):
            return next(({"measurement_index": index, **detail}
                         for index, detail in enumerate(differences)
                         if not (detail["tokens_equal"] and detail["stop_equal"])), None)

        modes[mode] = {
            "measurement_count": len(rows), "has_ten_measurements": len(rows) == 10,
            "repeatable_tokens": bool(rows) and all(d["tokens_equal"] for d in repetition_differences),
            "repeatable_stop": bool(rows) and all(d["stop_equal"] for d in repetition_differences),
            "measurements_match_reported_stable": bool(rows) and first_failure(stable_differences) is None,
            "stable_vs_reference": _generation_difference(
                expected["token_ids"], expected["stop_reason"],
                data["stable_generated_token_ids"], data["stable_stop_reason"]),
            "first_measurement_reference_mismatch": first_failure(reference_differences),
            "first_measurement_repeatability_mismatch": first_failure(repetition_differences),
        }
    ordinary, dflash = board["ordinary"], board["dflash"]
    return {
        "scope": "Diagnostic only; existing acceptance checks remain unchanged.",
        "ten_stable_runs_meaning": "Requires ten measurements per mode, each equal to the reference tokens and stop reason; false does not by itself mean run-to-run variation.",
        "ordinary_vs_dflash": _generation_difference(
            ordinary["stable_generated_token_ids"], ordinary["stable_stop_reason"],
            dflash["stable_generated_token_ids"], dflash["stable_stop_reason"]),
        "modes": modes,
    }


def parser():
    p = argparse.ArgumentParser(description="Qwen3.5-35B-A3B DFlash: AIR -> OMG -> OMC -> HIAI")
    p.add_argument("--resources", help="Local resource JSON; default QWEN35_RESOURCES or project manifest")
    sub = p.add_subparsers(dest="command", required=True)
    full = sub.add_parser("build-full-pytorch", help="construct all 40 target and 6 draft layers as meta PyTorch/FX graphs; no checkpoint or AIR")
    full.add_argument("--output", required=True, help="new structure artifact directory below AI_RUN_DIR")
    full.add_argument("--sequence-length", type=int, default=128)
    for name in ["preflight", "audit", "demo", "reference", "export-air", "export-tiny-air", "compile-omc", "prompt", "decode", "compare", "board-command", "quant-plan", "quantize"]:
        cmd = sub.add_parser(name); cmd.add_argument("--output", required=True)
        if name in ("reference", "export-air", "export-tiny-air", "demo", "quant-plan", "quantize"):
            cmd.add_argument("--sequence-length", type=int, default=128)
        if name in ("quant-plan", "quantize"):
            cmd.add_argument("--group-size", type=int, choices=[128, 256], default=128)
        if name == "quant-plan":
            cmd.add_argument("--available-gib", type=float, default=18)
        if name == "quantize":
            cmd.add_argument("--calibration", required=True, help='JSON {"token_ids": [[...], ...]}')
            cmd.add_argument("--device", default="cpu")
        if name in ("reference", "demo"):
            cmd.add_argument("--precision", choices=["fp16", "upstream-bf16"], default="fp16")
        if name == "reference":
            cmd.add_argument("--quantized-checkpoint", help="DOPT simulation bundle for a quantized reference")
        if name == "export-air":
            choice = cmd.add_mutually_exclusive_group(required=True)
            choice.add_argument("--quantized-checkpoint", help="DOPT calibration bundle")
            choice.add_argument("--float-baseline", action="store_true", help="FP16 conversion diagnostic; not a 32GB deployment")
        if name in ("reference", "demo"):
            cmd.add_argument("--prompt-token-ids", default="1,2,3")
            cmd.add_argument("--max-new-tokens", type=int, default=8)
            cmd.add_argument("--device", default="cpu")
        if name == "compile-omc":
            cmd.add_argument("--air", required=True)
            cmd.add_argument("--timeout-seconds", type=int, default=3600)
            cmd.add_argument("--reject-cpu-fallback", action="store_true",
                             help="reject explicit NPU partition-limit rollback to CPU")
        if name == "prompt":
            cmd.add_argument("--text", required=True)
            cmd.add_argument("--enable-thinking", action="store_true")
        if name in ("decode", "compare"):
            cmd.add_argument("--report", required=True)
        if name == "compare":
            cmd.add_argument("--reference", required=True)
            cmd.add_argument("--omc", required=True)
        if name == "board-command":
            cmd.add_argument("--omc", required=True); cmd.add_argument("--prompt", required=True)
            cmd.add_argument("--board-model-dir", required=True)
            cmd.add_argument("--board-output", required=True)
            cmd.add_argument("--runner", default="./qwen35_dflash_hiai")
            cmd.add_argument("--runtime-version", required=True)
            cmd.add_argument("--firmware-id", required=True)
            cmd.add_argument("--max-new-tokens", type=int, default=32)
            cmd.add_argument("--mode", choices=["generate", "benchmark", "smoke"], default="generate")
            cmd.add_argument("--capture-target-features", action="store_true",
                             help="smoke only: capture at most 65536 bytes of target FP16 features for offline diagnosis")
    return p


def execute(args):
    output = output_path(args.output)
    if output.exists(): raise FileExistsError(output)
    if args.command == "build-full-pytorch":
        from .export import build_full_pytorch
        build_full_pytorch(output, sequence_length=args.sequence_length)
        return
    if args.command == "quant-plan":
        from .quantization import meta_graphs, make_plan, memory_budget, dopt_config
        target, draft = meta_graphs()
        plan = make_plan(target, draft, group_size=args.group_size)
        plan["memory"] = memory_budget(plan, available_gib=args.available_gib,
                                      sequence_length=args.sequence_length, feature_width=target.config.feature_size)
        plan["dopt_config"] = dopt_config(plan)
        save_json(output, plan); return
    if args.command == "preflight":
        resources, path = load_resources(args.resources)
        packages = {}
        for name in ["torch", "transformers", "torchair", "torch_npu", "safetensors"]:
            try: packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError: packages[name] = None
        result = {"status": "INVENTORY", "resources": str(path), "packages": packages,
                  "torch_import_ready": False, "cuda_available": None,
                  "reference_backend": "cpu", "reference_import_ready": False,
                  "reference_import_error": None,
                  "air_export_ready": False, "dopt_calibration_ready": False,
                  "dopt_air_ready": False, "device_validation": "PENDING",
                  "air_export_validation": "PENDING"}
        try:
            import torch
            result["torch_import_ready"] = True
        except Exception as error:
            result["torch_error"] = format_error(error)
        result.update(probe_reference_import())
        if result["torch_import_ready"]:
            try:
                result["cuda_available"] = torch.cuda.is_available()
            except Exception as error:
                result["cuda_error"] = format_error(error)
            try:
                from .export import require_torchair
                require_torchair()
                # Dependency import only; no GE graph or device was validated.
                result["air_export_ready"] = True
                result["air_dependency_error"] = None
            except Exception as error:
                result["air_dependency_error"] = format_error(error)
            try:
                from .quantization import load_dopt, require_dopt_air
                from .resources import dopt_resource
                dopt_root = dopt_resource(resources)
                _, result["dopt_identity"] = load_dopt(dopt_root)
                result["dopt_calibration_ready"] = True
                require_dopt_air(dopt_root)
                result["dopt_air_ready"] = True
            except Exception as error:
                result["dopt_error"] = format_error(error)
        else:
            result["air_dependency_error"] = "Skipped: PyTorch import failed; see torch_error"
            result["dopt_error"] = "Skipped: PyTorch import failed; see torch_error"
        for key in ["target_weights", "draft_weights", "ddk_root", "ohos_native_sdk", "hiai_include_dir"]:
            try: result[key] = str(resource_path(resources, key, directory=True))
            except (ValueError, FileNotFoundError) as error: result[key] = {"missing": str(error)}
        save_json(output, result); return
    if args.command in ("demo", "reference", "export-tiny-air", "export-air", "quantize"):
        import torch
    if args.command in ("demo", "export-tiny-air"):
        if args.command == "export-tiny-air":
            from .export import require_torchair, export_air
            torchair = require_torchair()
        from reference.tiny import create
        torch.set_num_threads(1)
        target, draft = create()
        identity = {"kind": "tiny-random-real-upstream-moe-gdn", "seed": 17}
        pad, eos = 0, ()
        if args.command == "export-tiny-air":
            export_air(target, draft, output, args.sequence_length, torchair_module=torchair, source_identity=identity)
            return
    elif args.command in ("audit", "reference", "export-air", "prompt", "decode", "quantize"):
        resources, _ = load_resources(args.resources)
        target_path = resource_path(resources, "target_weights", directory=True)
        if args.command in ("prompt", "decode"):
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(target_path, local_files_only=True, trust_remote_code=False)
            if args.command == "prompt":
                ids = tokenizer.apply_chat_template([{"role": "user", "content": args.text}], tokenize=True,
                        add_generation_prompt=True, enable_thinking=args.enable_thinking)
                result = {"text": args.text, "prompt_token_ids": ids, "prompt_token_ids_csv": ",".join(map(str, ids)),
                          "enable_thinking": args.enable_thinking}
                result["tokenizer_files"] = [{"file": p.name, "sha256": sha256_file(p)}
                    for p in sorted(target_path.iterdir()) if p.name.startswith(("tokenizer", "vocab", "merges", "chat_template")) and p.is_file()]
            else:
                report = json.loads(Path(args.report).read_text())
                ids = (report["generation"]["generated_token_ids"] if "generation" in report else
                       report["dflash"].get("stable_generated_token_ids", report["dflash"].get("token_ids")))
                result = {"token_ids": ids, "text": tokenizer.decode(ids, skip_special_tokens=False),
                          "report_sha256": sha256_file(Path(args.report))}
            save_json(output, result); return
        draft_path = resource_path(resources, "draft_weights", directory=True)
        if args.command == "export-air":
            from .export import require_torchair
            torchair = require_torchair()  # before checkpoint hashing/loading
        dopt_root = None
        if args.command == "quantize" or getattr(args, "quantized_checkpoint", None):
            from .quantization import load_dopt
            from .resources import dopt_resource
            dopt_root = dopt_resource(resources)
            load_dopt(dopt_root)
            if args.command == "export-air":
                from .quantization import require_dopt_air
                require_dopt_air(dopt_root)
        from .checkpoints import audit_checkpoints
        identity = audit_checkpoints(target_path, draft_path)
        if args.command == "audit":
            save_json(output, identity)
            if identity["status"] != "PASS": raise ValueError(identity["errors"])
            return
        if identity["status"] != "PASS": raise ValueError(identity["errors"])
        from .graphs import load_graphs
        dtype = torch.bfloat16 if getattr(args, "precision", "fp16") == "upstream-bf16" else torch.float16
        target, draft = load_graphs(target_path, draft_path, device=getattr(args, "device", "cpu"), dtype=dtype)
        pad, eos = 248044, (248044, 248046)
        if args.command == "quantize":
            from .quantization import calibrate
            prompts = json.loads(Path(args.calibration).read_text())["token_ids"]
            calibrate(target, draft, prompts, output, sequence_length=args.sequence_length,
                      dopt_root=dopt_root, source_identity=identity, group_size=args.group_size)
            return
        if args.command == "export-air":
            from .export import export_air
            quantization = None
            if args.quantized_checkpoint:
                from .quantization import restore_for_air
                target, draft, quantization = restore_for_air(
                    target, draft, args.quantized_checkpoint, dopt_root, source_identity=identity)
            export_air(target, draft, output, args.sequence_length, torchair_module=torchair,
                       source_identity=identity, quantization=quantization)
            return
    elif args.command == "compile-omc":
        from .compile import compile_air
        resources, _ = load_resources(args.resources)
        ddk_id = os.path.expandvars(resources.get("ddk_id", ""))
        if not ddk_id or "$" in ddk_id: raise ValueError("declare ddk_id in resource JSON or CPRO_DDK_ID")
        compile_air(args.air, output, resource_path(resources, "ddk_root", directory=True),
                    ddk_id=ddk_id, timeout_seconds=args.timeout_seconds,
                    reject_cpu_fallback=args.reject_cpu_fallback)
        return
    elif args.command == "board-command":
        import shlex
        if args.capture_target_features and args.mode != "smoke":
            raise ValueError("--capture-target-features requires --mode smoke")
        root = Path(args.omc).resolve(); m = json.loads((root / "manifest.json").read_text())
        if m.get("artifact_kind") != "qwen35-cpro-omc-pair" or m.get("status") != "COMPILED":
            raise ValueError("expected a successful OMC bundle")
        if m.get("abi_version") != 2 or m.get("feature_dtype") != "torch.float16":
            raise ValueError("runner requires ABI v2 with FP16 features; regenerate old FP32 bundles")
        if args.capture_target_features:
            if any(type(m.get(key)) is not int or m[key] <= 0
                   for key in ("sequence_length", "feature_width")):
                raise ValueError("target feature capture requires positive integer sequence_length and feature_width")
            if 2 * m["sequence_length"] * m["feature_width"] > 65536:
                raise ValueError("target feature capture exceeds 65536-byte limit")
        graphs = {g["name"]: g for g in m["graphs"]}
        for g in graphs.values(): checked_file(root, {**g, "path": g["omc"]})
        prompt = json.loads(Path(args.prompt).read_text())["prompt_token_ids"]
        if not prompt or len(prompt) > m["sequence_length"]:
            raise ValueError("prompt exceeds OMC sequence gear or is empty")
        if args.mode != "smoke" and len(prompt)+args.max_new_tokens-1 > m["sequence_length"]:
            raise ValueError("prompt plus requested generation exceeds OMC sequence gear")
        argv = [args.runner, "--mode", args.mode, "--pad-token-id", str(m["pad_token_id"]),
                "--eos-token-ids", ",".join(map(str,m["eos_token_ids"])), "--target", str(Path(args.board_model_dir)/graphs["target"]["omc"]),
                "--target-sha256", graphs["target"]["sha256"],
                "--draft", str(Path(args.board_model_dir)/graphs["draft"]["omc"]),
                "--draft-sha256", graphs["draft"]["sha256"], "--expected-runtime-version", args.runtime_version,
                "--ddk-id", m["ddk_id"], "--firmware-id", args.firmware_id,
                "--prompt-token-ids", ",".join(map(str,prompt)), "--feature-width", str(m["feature_width"]),
                "--vocab-size", str(m["vocab_size"]), "--max-new-tokens", str(args.max_new_tokens),
                "--max-draft-tokens", str(m["draft_width"]), "--output", args.board_output]
        if args.capture_target_features:
            argv += ["--capture-target-features", "1"]
        save_json(output, {"argv": argv, "shell": shlex.join(argv),
                           "manifest_sha256": sha256_file(root/"manifest.json"),
                           "prompt_sha256": sha256_file(Path(args.prompt))}); return
    elif args.command == "compare":
        board = json.loads(Path(args.report).read_text()); reference = json.loads(Path(args.reference).read_text())
        bundle = Path(args.omc).resolve()
        manifest = json.loads((bundle / "manifest.json").read_text())
        for g in manifest["graphs"]: checked_file(bundle, {**g, "path": g["omc"]})
        graphs = {g["name"]: g for g in manifest["graphs"]}
        checks = {"reference_passed": reference.get("status") == "PASS",
                  "board_passed": board.get("status") == "PASS",
                  "compiled_bundle": manifest.get("status") == "COMPILED" and manifest.get("artifact_kind") == "qwen35-cpro-omc-pair",
                  "same_sequence": board.get("sequence_length") == reference.get("sequence_length") == manifest.get("sequence_length"),
                  "same_weights": reference.get("source") is not None and reference.get("source") == manifest.get("source"),
                  "same_precision": reference.get("precision") == manifest.get("precision") == "torch.float16",
                  "same_quantization": reference.get("quantization") == manifest.get("quantization"),
                  "same_abi": board.get("abi_version") == manifest.get("abi_version") == 2,
                  "fp16_features": board.get("feature_dtype") == manifest.get("feature_dtype") == "torch.float16",
                  "same_omcs": all(board.get(k+"_sha256") == graphs[k]["sha256"] for k in ["target", "draft"]),
                  "same_limit": board.get("max_new_tokens") == reference.get("max_new_tokens"),
                  "real_hiai": board.get("evidence_kind") == "device-hiai",
                  "prompt": board.get("prompt_token_ids") == reference.get("prompt_token_ids"),
                  "tokens": board["ordinary"]["stable_generated_token_ids"] == reference["ordinary"]["token_ids"],
                  "dflash_tokens": board["dflash"]["stable_generated_token_ids"] == reference["ordinary"]["token_ids"],
                  "stop": board["ordinary"]["stable_stop_reason"] == reference["ordinary"]["stop_reason"],
                  "dflash_stop": board["dflash"]["stable_stop_reason"] == reference["ordinary"]["stop_reason"],
                  "ten_stable_runs": all(len(board[mode]["measurements"]) == 10 and all(
                      r["generated_token_ids"] == reference["ordinary"]["token_ids"] and
                      r["stop_reason"] == reference["ordinary"]["stop_reason"]
                      for r in board[mode]["measurements"]) for mode in ["ordinary", "dflash"])}
        result = {"scope": "token-and-stop-reference-comparison", "checks": checks,
                  "status": "PASS" if all(checks.values()) else "FAIL",
                  "diagnostics": _comparison_diagnostics(board, reference),
                  "reference_sha256": sha256_file(Path(args.reference)), "board_sha256": sha256_file(Path(args.report)),
                  "source": manifest.get("source")}
        save_json(output, result)
        if result["status"] != "PASS":
            diagnostics = result["diagnostics"]
            agreement = diagnostics["ordinary_vs_dflash"]
            summary = [f"board_modes(tokens/stop)={agreement['tokens_equal']}/{agreement['stop_equal']}"]
            for mode, detail in diagnostics["modes"].items():
                first = detail["stable_vs_reference"]["first_token_mismatch"]
                summary.append(f"{mode}: count={detail['measurement_count']}, "
                    f"repeatable(tokens/stop)={detail['repeatable_tokens']}/{detail['repeatable_stop']}, "
                    f"stable/reference_first_token_mismatch={json.dumps(first, separators=(',', ':'))}")
            raise ValueError(f"comparison failed: {checks}; {'; '.join(summary)}; diagnostic details: {output}")
        return
    else:
        raise ValueError(args.command)
    prompt = [int(s) for s in args.prompt_token_ids.split(",")]
    quantization = None
    if getattr(args, "quantized_checkpoint", None):
        if args.precision != "fp16":
            raise ValueError("quantized reference requires the FP16 non-quantization policy")
        from .quantization import restore_for_reference
        target, draft, quantization = restore_for_reference(
            target, draft, args.quantized_checkpoint, dopt_root, source_identity=identity)
    if args.precision == "fp16":
        from .precision import prepare_fp16_pair
        target, draft, precision_audit = prepare_fp16_pair(
            target, draft, args.sequence_length, quantization=quantization is not None)
    else:
        target.to(torch.bfloat16); draft.to(torch.bfloat16)
        precision_audit = None
    from .runtime import TorchExecutor, generate
    executor = TorchExecutor(target, draft, args.sequence_length, pad)
    ordinary = generate(executor, prompt, ordinary=True, max_new_tokens=args.max_new_tokens, eos_token_ids=eos)
    dflash = generate(executor, prompt, max_new_tokens=args.max_new_tokens, eos_token_ids=eos)
    match = (ordinary.token_ids, ordinary.stop_reason) == (dflash.token_ids, dflash.stop_reason)
    save_json(output, {"status": "PASS" if match else "FAIL", "evidence_kind": "framework-reference",
                       "source": identity, "prompt_token_ids": prompt, "device": str(executor.device),
                       "precision": "torch.float16" if args.precision == "fp16" else "upstream-bf16",
                       "quantization": quantization, "precision_audit": precision_audit,
                       "sequence_length": args.sequence_length, "max_new_tokens": args.max_new_tokens,
                       "ordinary": ordinary.to_dict(), "dflash": dflash.to_dict()})
    if not match: raise RuntimeError("ordinary and DFlash reference tokens differ")


def main():
    args = parser().parse_args()
    try:
        if args.command in ("demo", "reference"):
            from .backends import configure_reference_backend
            configure_reference_backend(args.device)
        elif args.command == "build-full-pytorch":
            from .backends import configure_reference_backend
            configure_reference_backend("cpu")
        execute(args)
    except Exception as error:
        print(f"qwen35: {format_error(error)}", file=sys.stderr)
        return 1
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
