#!/usr/bin/env python3
"""Direct GE target/draft -> AIR -> OMG, retaining the PyTorch export route."""
import argparse
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "model"))
sys.path.insert(0, str(REPO))


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
    p = sub.add_parser("compile")
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
    p = sub.add_parser("reference-tiny")
    p.add_argument("--prepared", required=True)
    p.add_argument("--prompt-token-ids", default="1,2,3")
    p.add_argument("--max-new-tokens", type=int, default=4)
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        from qwen35_moe_dflash.resources import load_resources, resource_path
        if args.command == "plan":
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
        elif args.command == "compile":
            from qwen35_moe_dflash.compile import compile_air
            resources = {}
            if not args.ddk_root or not args.ddk_id:
                resources, _ = load_resources(args.resources)
            ddk = args.ddk_root or resource_path(resources, "ddk_root", directory=True)
            identity = args.ddk_id or os.path.expandvars(resources.get("ddk_id", ""))
            if not identity or "$" in identity:
                raise ValueError("declare --ddk-id or CPRO_DDK_ID")
            result = compile_air(args.air, args.output, ddk, ddk_id=identity, timeout_seconds=args.timeout_seconds,
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
