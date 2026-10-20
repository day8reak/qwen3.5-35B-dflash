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
from .weights import sha256_file


def parser():
    p = argparse.ArgumentParser(description="Qwen3.5-35B-A3B DFlash: AIR -> OMG -> OMC -> HIAI")
    p.add_argument("--resources", help="Local resource JSON; default QWEN35_RESOURCES or project manifest")
    sub = p.add_subparsers(dest="command", required=True)
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
            cmd.add_argument("--mode", choices=["generate", "benchmark"], default="generate")
    return p


def execute(args):
    from .runtime import TorchExecutor, generate
    import torch
    output = output_path(args.output)
    if output.exists(): raise FileExistsError(output)
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
        try:
            from .export import require_torchair
            require_torchair()
            air_ready, air_error = True, None
        except (Exception,) as error:
            air_ready, air_error = False, str(error)
        result = {"status": "INVENTORY", "resources": str(path), "packages": packages,
                  "air_export_ready": air_ready, "air_dependency_error": air_error,
                  "cuda_available": torch.cuda.is_available(), "device_validation": "PENDING"}
        try:
            from .quantization import load_dopt, require_dopt_air
            from .resources import dopt_resource
            dopt_root = dopt_resource(resources)
            _, result["dopt_identity"] = load_dopt(dopt_root)
            result["dopt_calibration_ready"] = True
            require_dopt_air(dopt_root)
            result["dopt_air_ready"] = True
        except Exception as error:
            result["dopt_air_ready"] = False
            result["dopt_error"] = f"{type(error).__name__}: {error}"
        for key in ["target_weights", "draft_weights", "ddk_root", "ohos_native_sdk", "hiai_include_dir"]:
            try: result[key] = str(resource_path(resources, key, directory=True))
            except (ValueError, FileNotFoundError) as error: result[key] = {"missing": str(error)}
        save_json(output, result); return
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
                    ddk_id=ddk_id, timeout_seconds=args.timeout_seconds)
        return
    elif args.command == "board-command":
        import shlex
        root = Path(args.omc).resolve(); m = json.loads((root / "manifest.json").read_text())
        if m.get("artifact_kind") != "qwen35-cpro-omc-pair" or m.get("status") != "COMPILED":
            raise ValueError("expected a successful OMC bundle")
        if m.get("abi_version") != 2 or m.get("feature_dtype") != "torch.float16":
            raise ValueError("runner requires ABI v2 with FP16 features; regenerate old FP32 bundles")
        graphs = {g["name"]: g for g in m["graphs"]}
        for g in graphs.values(): checked_file(root, {**g, "path": g["omc"]})
        prompt = json.loads(Path(args.prompt).read_text())["prompt_token_ids"]
        if not prompt or len(prompt)+args.max_new_tokens-1 > m["sequence_length"]:
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
                  "reference_sha256": sha256_file(Path(args.reference)), "board_sha256": sha256_file(Path(args.report)),
                  "source": manifest.get("source")}
        save_json(output, result)
        if result["status"] != "PASS": raise ValueError(f"comparison failed: {checks}")
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
        execute(args)
    except Exception as error:
        print(f"qwen35: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
