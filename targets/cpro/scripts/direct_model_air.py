#!/usr/bin/env python3
"""Direct GE target/draft or explicit-state target -> AIR -> OMG."""
import argparse
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "model"))
sys.path.insert(0, str(REPO))


def _add_stateful_commands(sub):
    step = sub.add_parser("prepare-stateful-step", help="stage one explicit-state HIAI call and verified previous state")
    step.add_argument("--omc", required=True)
    step.add_argument("--output", required=True)
    step.add_argument("--token-ids", required=True, help="comma-separated valid integer IDs; exactly the graph sequence length")
    step.add_argument("--previous-step", help="previous prepared step directory; required for nonzero past length")
    step.add_argument("--previous-result", help="previous native result directory containing result.json")
    step.add_argument("--allow-host-fake", action="store_true", help="permit host fake-runner evidence for plumbing tests only")
    binding = sub.add_parser("bindings-from-document", help="transcribe user-documented GE ports; no SDK or device check")
    binding.add_argument("--output", required=True, help="new binding JSON below AI_RUN_DIR; a provenance sidecar is also written")
    binding.add_argument("--target", required=True, help="declared intended target; not a compatibility assertion")
    binding.add_argument("--ddk-id", required=True, help="declared intended DDK identity; no installation check")
    binding.add_argument("--gdn-initial-state-dtype", choices=("float16", "float32"), required=True,
                         help="document reports Cpro FP16 caller and FP32 interface/other variants; choose explicitly")
    for name in ("plan-stateful", "export-stateful"):
        p = sub.add_parser(name, help=("generate checked explicit-state C++/recipe without GE" if name == "plan-stateful" else
                                      "serialize explicit-state target AIR with a matching GE installation"))
        p.add_argument("--prepared", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--sequence-length", type=int, help="contiguous valid tokens; defaults to prepared length")
        p.add_argument("--past-length", type=int, default=0, help="static incoming cache length; zero starts fresh")
        p.add_argument("--fused-ops", default="none", help="none or comma-separated gdn,conv_state,rms")
        p.add_argument("--bindings", help="actual GE port/dtype/attribute binding JSON; required for selected fused ops")
        p.add_argument("--recurrent-dtype", choices=("float16", "float32"), default="float16")
        p.add_argument("--gdn-state-casts", choices=("reject", "allow"), default="reject",
                       help="allow explicitly adapts recurrent state dtype to/from declared fused GDN ports")
        p.add_argument("--expert-dispatch", choices=("static", "selected"), help="defaults to prepared policy")
        p.add_argument("--fp16-reductions", choices=("native", "matmul"), default="native")
        p.add_argument("--fp16-row-mul", choices=("native", "compact"), default="native")
        p.add_argument("--gdn-indexed-copy", choices=("native", "gather"), default="native")
        if name == "export-stateful":
            p.add_argument("--ge-root", required=True)
            p.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
            p.add_argument("--cxx11-abi", choices=(0, 1), type=int, default=0)
            p.add_argument("--timeout-seconds", type=int, default=3600)


def _run_stateful(args):
    from qwen35_moe_dflash.ge_fused.export import export_target, plan_target, selected_fusion

    options = dict(sequence_length=args.sequence_length, past_length=args.past_length,
                   fused_ops=selected_fusion(args.fused_ops), bindings_path=args.bindings,
                   recurrent_dtype=args.recurrent_dtype, expert_dispatch=args.expert_dispatch,
                   gdn_state_casts=args.gdn_state_casts,
                   fp16_reductions=args.fp16_reductions, fp16_row_mul=args.fp16_row_mul,
                   gdn_indexed_copy=args.gdn_indexed_copy)
    if args.command == "plan-stateful":
        return plan_target(args.prepared, args.output, **options)
    return export_target(args.prepared, args.output, args.ge_root, cxx=args.cxx,
                         cxx11_abi=args.cxx11_abi, timeout_seconds=args.timeout_seconds, **options)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "prepare"):
        p = sub.add_parser(name)
        choice = p.add_mutually_exclusive_group(required=True)
        choice.add_argument("--tiny", action="store_true")
        choice.add_argument("--full", action="store_true")
        p.add_argument("--sequence-length", type=int, default=8)
        p.add_argument("--expert-dispatch", choices=("static", "selected"), default="static",
                       help="selected: gather only routed expert weights for per-token BatchMatMul; board validation required")
        p.add_argument("--output", required=True)
        if name == "prepare":
            p.add_argument("--resources")
            p.add_argument("--target-weights")
            p.add_argument("--draft-weights")
            p.add_argument("--float-baseline", action="store_true",
                           help="full-size FP16 conversion diagnostic, not a 32GB deployment")
            p.add_argument("--external-weights", action="store_true",
                           help="also test FileConstant for tiny; mandatory automatically for full")
    p = sub.add_parser("export")
    p.add_argument("--prepared", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--ge-root")
    p.add_argument("--resources")
    p.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    p.add_argument("--cxx11-abi", choices=(0, 1), type=int, default=0)
    p.add_argument("--timeout-seconds", type=int, default=3600)
    p.add_argument("--fp16-reductions", choices=("native", "matmul"), default="native",
                   help="matmul: opt-in single-axis FP16 reduction lowering; retain integer reductions and original prepared weights")
    p.add_argument("--fp16-row-mul", choices=("native", "compact"), default="native",
                   help="compact: opt-in reshape for the two board-tested FP16 row-Mul shapes; retain original weights and graph IO")
    p.add_argument("--gdn-indexed-copy", choices=("native", "gather"), default="native",
                   help="gather: opt-in GDN mask/outer-Sub copies for the tested FP16 [2,1,64,64] shape; retain other shapes")
    for name in ("compile", "compile-stateful"):
        p = sub.add_parser(name)
        p.add_argument("--air", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--resources")
        p.add_argument("--ddk-root")
        p.add_argument("--ddk-id")
        p.add_argument("--reject-cpu-fallback", action="store_true",
                       help="reject explicit NPU partition-limit rollback to CPU; retain generated OMC/logs for diagnosis")
        p.add_argument("--timeout-seconds", type=int, default=3600)
    p = sub.add_parser("inspect-compile", help="read existing compiler support and construction metadata; no rebuild or board access")
    p.add_argument("--air", required=True)
    p.add_argument("--omc", required=True)
    p.add_argument("--graph", choices=("target", "draft"), default="target")
    p.add_argument("--output", required=True)
    p = sub.add_parser("inspect-omc", help="inspect existing target/draft OMC with the recorded OMG; no rebuild or board access")
    p.add_argument("--omc", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--ddk-root", required=True)
    p.add_argument("--ddk-id", required=True)
    p.add_argument("--timeout-seconds", type=int, default=300)
    p.add_argument("--max-json-bytes", type=int, default=64 << 20)
    p = sub.add_parser("inspect-omc-cpu", help="read CPU subgraphs using existing OMC/inspection JSON and the selected DDK schema; no compiler or board access")
    p.add_argument("--omc", required=True)
    p.add_argument("--inspection", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--ddk-root", required=True)
    p.add_argument("--ddk-id", required=True)
    p.add_argument("--max-partition-bytes", type=int, default=64 << 20)
    p = sub.add_parser("compare-target-features", help="compare captured original tiny OMC features to the prepared-weight host graph; no compilation or board access")
    p.add_argument("--prepared", required=True)
    p.add_argument("--omc", required=True)
    p.add_argument("--board-run", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("reference-tiny")
    p.add_argument("--prepared", required=True)
    p.add_argument("--prompt-token-ids", default="1,2,3")
    p.add_argument("--max-new-tokens", type=int, default=4)
    p.add_argument("--output", required=True)
    _add_stateful_commands(sub)
    args = parser.parse_args()
    try:
        from qwen35_moe_dflash.resources import load_resources, resource_path
        if args.command == "bindings-from-document":
            from qwen35_moe_dflash.ge_fused.bindings import write_document_bindings
            result = write_document_bindings(args.output, target=args.target, ddk_id=args.ddk_id,
                                             gdn_initial_state_dtype=args.gdn_initial_state_dtype)
            print(json.dumps(result, ensure_ascii=False))
            return 0
        elif args.command in {"plan-stateful", "export-stateful"}:
            result = _run_stateful(args)
            args.output = str(Path(args.output).resolve())
        elif args.command == "prepare-stateful-step":
            from qwen35_moe_dflash.ge_fused.export import prepare_stateful_step
            result = prepare_stateful_step(args.omc, args.output, [int(token) for token in args.token_ids.split(",")],
                                           previous_step=args.previous_step, previous_result=args.previous_result,
                                           allow_host_fake=args.allow_host_fake)
        elif args.command == "plan":
            from qwen35_moe_dflash.ge_export import plan_pair
            result = plan_pair(args.output, args.sequence_length, tiny=args.tiny,
                               expert_dispatch=args.expert_dispatch)
        elif args.command == "prepare":
            from qwen35_moe_dflash.ge_export import prepare_pair
            target, draft = args.target_weights, args.draft_weights
            if args.full:
                if not args.float_baseline:
                    raise ValueError("full direct GE needs --float-baseline; packed quantized GE is pending")
                if not target or not draft:
                    resources, _ = load_resources(args.resources)
                    target = target or resource_path(resources, "target_weights", directory=True)
                    draft = draft or resource_path(resources, "draft_weights", directory=True)
            result = prepare_pair(args.output, args.sequence_length, tiny=args.tiny,
                                  target_path=target, draft_path=draft,
                                  float_baseline=args.float_baseline, external_weights=args.external_weights,
                                  expert_dispatch=args.expert_dispatch)
        elif args.command == "export":
            from qwen35_moe_dflash.ge_export import export_pair
            ge = args.ge_root
            if not ge and args.resources:
                resources, _ = load_resources(args.resources)
                ge = resource_path(resources, "ge_root", directory=True)
            ge = ge or os.environ.get("CANN_ROOT") or os.environ.get("ASCEND_HOME_PATH")
            if not ge:
                resources, _ = load_resources(args.resources)
                ge = resource_path(resources, "ge_root", directory=True)
            result = export_pair(args.prepared, args.output, ge, cxx=args.cxx,
                                 cxx11_abi=args.cxx11_abi, timeout_seconds=args.timeout_seconds,
                                 fp16_reductions=args.fp16_reductions, fp16_row_mul=args.fp16_row_mul,
                                 gdn_indexed_copy=args.gdn_indexed_copy)
        elif args.command in {"compile", "compile-stateful"}:
            from qwen35_moe_dflash.compile import compile_air, compile_stateful_air
            resources = {}
            if not args.ddk_root or not args.ddk_id:
                resources, _ = load_resources(args.resources)
            ddk = args.ddk_root or resource_path(resources, "ddk_root", directory=True)
            identity = args.ddk_id or os.path.expandvars(resources.get("ddk_id", ""))
            if not identity or "$" in identity:
                raise ValueError("declare --ddk-id or CPRO_DDK_ID")
            compile_fn = compile_stateful_air if args.command == "compile-stateful" else compile_air
            result = compile_fn(args.air, args.output, ddk, ddk_id=identity, timeout_seconds=args.timeout_seconds,
                                reject_cpu_fallback=args.reject_cpu_fallback)
        elif args.command == "inspect-compile":
            from qwen35_moe_dflash.compile_inspect import inspect_compile
            result = inspect_compile(args.air, args.omc, args.output, graph=args.graph)
            print((Path(args.output) / "report.txt").read_text())
        elif args.command == "inspect-omc":
            from qwen35_moe_dflash.omc_inspect import inspect_omc
            result = inspect_omc(args.omc, args.output, args.ddk_root, ddk_id=args.ddk_id,
                                 timeout_seconds=args.timeout_seconds, max_json_bytes=args.max_json_bytes)
            print((Path(args.output) / "report.txt").read_text())
        elif args.command == "inspect-omc-cpu":
            from qwen35_moe_dflash.omc_cpu_inspect import inspect_omc_cpu
            result = inspect_omc_cpu(args.omc, args.inspection, args.output, args.ddk_root,
                                     ddk_id=args.ddk_id, max_partition_bytes=args.max_partition_bytes)
            print((Path(args.output) / "report.txt").read_text())
        elif args.command == "compare-target-features":
            from qwen35_moe_dflash.ge_target_features import compare_target_features
            result = compare_target_features(args.prepared, args.omc, args.board_run, args.reference, args.output)
            print((Path(args.output) / "report.txt").read_text())
        else:
            from qwen35_moe_dflash.ge_reference import reference_tiny
            result = reference_tiny(args.prepared, args.output,
                                    prompt=tuple(int(t) for t in args.prompt_token_ids.split(",")),
                                    max_new_tokens=args.max_new_tokens)
    except Exception as error:
        print(f"direct-model-air: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": result["status"], "output": args.output}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
