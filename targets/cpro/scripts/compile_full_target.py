#!/usr/bin/env python3
"""Compile the original target AIR using an explicitly selected Cpro profile."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import uuid

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "model"))


def _print_result(result, *, result_path=None, log_paths=()):
    """Show the tool's diagnostics without repeating the entire input-shape argv.

    The on-disk result retains the full exception. This is only a bounded
    console presentation, and does not infer the cause of a compiler failure.
    """
    from qwen35_moe_dflash.tool_logs import log_excerpt

    console = dict(result)
    if result.get("status") == "FAILED":
        if "returncode" in result:
            console["error"] = f"OMG exited with status {result['returncode']}"
        elif "timeout_seconds" in result:
            console["error"] = f"OMG timed out after {result['timeout_seconds']} seconds"
        elif len(str(result.get("error", ""))) > 512:
            console["error"] = str(result["error"])[:512] + " [full exception in result file]"
    if result_path is not None:
        console["result_json"] = str(result_path)
    print(json.dumps(console, ensure_ascii=False), flush=True)
    if result.get("status") == "FAILED":
        for path in log_paths:
            excerpt = log_excerpt(path)
            if excerpt:
                print(f"full-target: compiler diagnostics from {path}", flush=True)
                print(excerpt, flush=True)
                break


def main():
    from qwen35_moe_dflash.existing_air_profile import (
        EXISTING_AIR_PROFILE_NAMES, get_existing_air_profile,
    )
    from qwen35_moe_dflash.existing_air import INPUT_LAYOUTS, compile_existing_air

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--air", help="original exported AIR; never modified; inferred from --resume if omitted")
    parser.add_argument("--input-layout", choices=INPUT_LAYOUTS, default="export-directory",
                        help="relative weight paths: default retains original export folder prefix; "
                             "air-directory stages data directly beside AIR")
    parser.add_argument("--profile", choices=EXISTING_AIR_PROFILE_NAMES,
                        default="cpro-40block-64-1", help="user-provided input names and dynamic shapes")
    parser.add_argument("--ddk-root", default=os.environ.get("CPRO_DDK_ROOT") or os.environ.get("DDK_PATH"))
    parser.add_argument("--ddk-id", default=os.environ.get("CPRO_DDK_ID"))
    parser.add_argument("--tbe-diagnosis", help="reuse an existing check-tbe-platform --diagnose directory")
    parser.add_argument("--omg", help="optional actual ELF binary in selected DDK; default tools/tools_omg/master/omg")
    parser.add_argument("--output", help="new job directory under AI_RUN_DIR")
    parser.add_argument("--resume", help="reuse a completed original-target job after source/DDK/artifact checks; never recompile failed/interrupted jobs")
    parser.add_argument("--retry-static-prefill-from", help="new static64 attempt using a failed dual-gear COPY job or its first audited static64 successor")
    parser.add_argument("--tmpdir-probe", help="completed source-directory/IPC probe; required for an audited static64 successor retry")
    parser.add_argument("--stop-receipt", help="completed exact-job stop receipt; required for an audited static64 successor retry")
    parser.add_argument("--inspect-only", action="store_true", help="requires --resume; inspect existing compilation without model compilation or board execution")
    parser.add_argument("--hiai-include", default=os.environ.get("HIAI_INCLUDE_DIR"),
                        help="selected HIAI header root; retain dynamic-gear/RefData/load declarations locally")
    parser.add_argument("--source-root", help="original export/runtime source root; retain bounded input/state construction excerpts locally")
    parser.add_argument("--run-board", action="store_true", help="after actual contract checks, run the user's existing HIAI V2 model_run_tool_internal once and retrieve outputs")
    parser.add_argument("--input-dir", help="local directory containing the original chunksize_N_data binary inputs; no synthesized inputs")
    parser.add_argument("--board-input-dir", help="alternative existing board directory containing the original inputs")
    parser.add_argument("--sequence-length", choices=(64, 1), type=int, default=64, help="one original target gear; 1 alone does not prove state continuity")
    parser.add_argument("--target-features", action="store_true",
                        help="explicitly require 82 outputs for board execution: original logits/80 states plus FP16 [1,gear,16384]; boundary exposure alone does not validate feature provenance")
    parser.add_argument("--board-tool", default=os.environ.get("CPRO_MODEL_RUN_TOOL", "/data/local/tmp/tools_sysdbg/model_run_tool_internal"))
    parser.add_argument("--board-model-root", default=os.environ.get("CPRO_TARGET_BOARD_ROOT", "/data/local/tmp/qwen3_5_moe"))
    parser.add_argument("--timeout-seconds", type=int, default=7200, help="OMG compilation timeout")
    parser.add_argument("--skip-inspection", action="store_true", help="skip post-compile OMC-to-JSON inspection")
    args = parser.parse_args()
    if not args.air and not args.resume and not args.retry_static_prefill_from:
        parser.error("--air is required for a new compilation")
    if args.retry_static_prefill_from and (args.resume or args.inspect_only):
        parser.error("--retry-static-prefill-from creates a new attempt and cannot use --resume/--inspect-only")
    if args.retry_static_prefill_from and (args.profile != "cpro-40block-prefill-64" or args.input_layout != "air-directory"):
        parser.error("--retry-static-prefill-from requires --profile cpro-40block-prefill-64 and --input-layout air-directory")
    if (args.tmpdir_probe or args.stop_receipt) and not args.retry_static_prefill_from:
        parser.error("--tmpdir-probe/--stop-receipt require --retry-static-prefill-from")
    if bool(args.tmpdir_probe) != bool(args.stop_receipt):
        parser.error("--tmpdir-probe and --stop-receipt must be supplied together")
    if args.resume and args.output:
        parser.error("--resume and --output cannot be combined")
    if args.inspect_only and (not args.resume or args.skip_inspection):
        parser.error("--inspect-only requires --resume and cannot be combined with --skip-inspection")
    if args.run_board and (args.inspect_only or args.skip_inspection or bool(args.input_dir) == bool(args.board_input_dir)):
        parser.error("--run-board requires exactly one of --input-dir/--board-input-dir and actual contract inspection")

    from qwen35_moe_dflash.artifacts import output_path, save_json
    from qwen35_moe_dflash.full_target import (
        active_target_jobs, read_metadata, target_source_lock, validate_compilation,
    )

    root = None
    owned_root = False
    continuation = None
    retry_preflight = None
    guards = ExitStack()
    result = {"schema_version": 1, "artifact_kind": "qwen35-original-target-job",
              "status": "PREPARING", "device_execution": "NOT_RUN",
              "numerical_validation": "NOT_RUN", "device_ready": False}
    try:
        run = os.environ.get("AI_RUN_DIR")
        if not run or not args.ddk_root or not args.ddk_id:
            raise ValueError("AI_RUN_DIR, CPRO_DDK_ROOT/DDK_PATH and CPRO_DDK_ID are required; source the DDK and project env.sh")
        if args.timeout_seconds <= 0:
            raise ValueError("--timeout-seconds must be positive")
        if args.resume:
            root = output_path(args.resume)
            prior = read_metadata(root / "result.json")
            args.air = args.air or prior.get("air")
        if args.retry_static_prefill_from:
            predecessor = Path(args.retry_static_prefill_from).resolve()
            previous = read_metadata(predecessor / "result.json")
            args.air = args.air or previous.get("air")
            previous_profile_path = predecessor / "compile-profile.json"
            previous_profile = read_metadata(previous_profile_path) if previous_profile_path.is_file() else {}
            successor = previous_profile.get("name") == "cpro-40block-prefill-64" or bool(previous.get("retained_failed_job"))
            if successor:
                if not args.tmpdir_probe or not args.stop_receipt:
                    raise ValueError("an audited static64 successor retry requires --tmpdir-probe and --stop-receipt")
                from qwen35_moe_dflash.target_retry_gate import validate_static_retry_gate
                retry_preflight = validate_static_retry_gate(predecessor,
                    tmpdir_probe=args.tmpdir_probe, stop_receipt=args.stop_receipt)
                if retry_preflight.get("status") != "PREFLIGHT_PASSED":
                    raise ValueError("static64 retry preflight did not pass")
                print("full-target: final failed-job audit, process exit and directory/IPC probe verified", flush=True)
            elif args.tmpdir_probe or args.stop_receipt:
                raise ValueError("directory/stop retry gates apply only to an audited static64 predecessor")
        if not args.air:
            raise ValueError("completed target job does not declare its original AIR")
        air = Path(args.air).resolve()
        if not air.is_file():
            raise ValueError(f"AIR not found: {air}")
        profile = get_existing_air_profile(args.profile)
        guards.enter_context(target_source_lock(air))
        active = active_target_jobs(air)
        if active:
            raise RuntimeError("original target compiler entry is already running: " + json.dumps(active))
        if args.resume:
            print("full-target: verifying retained source, snapshot and external-output SHA256; reading only, no weight copies", flush=True)
            compiled = validate_compilation(root, air=air, ddk_root=args.ddk_root,
                ddk_id=args.ddk_id, profile=profile, input_layout=args.input_layout,
                omg_path=args.omg, tbe_diagnosis=args.tbe_diagnosis, target_features=args.target_features)
            result = dict(prior, resumed_from=str(root), model_compilation="PASSED",
                          compiler_invocation="NOT_RUN_REUSED_COMPLETE_COMPILE",
                          reuse_validation="SOURCE_SNAPSHOT_GENERATED_COMPILER_AND_LOG_SHA256",
                          device_execution="NOT_RUN", numerical_validation="NOT_RUN", device_ready=False)
            for key in ("interface_error", "interface_report", "target_contract_error", "target_contract_report",
                        "runtime_sources_error", "board_status", "board_result", "feature_output_evidence"):
                result.pop(key, None)
            result.pop("board_error", None)
            result.pop("delivery_status", None)
            continuation = root / "continuations" / uuid.uuid4().hex[:12]
            continuation.mkdir(parents=True)
            print(f"full-target: reusing completed compilation at {root}; no model compilation or weight copies", flush=True)
        else:
            root = output_path(args.output or Path(run) / ("full-target-compile-" + uuid.uuid4().hex[:12]))
            root.mkdir(parents=True, exist_ok=False)
            owned_root = True
            save_json(root / "compile-profile.json", profile)
        evidence_root = continuation or root
        result.update(output=str(root), air=str(air), profile="compile-profile.json",
                      input_layout=args.input_layout,
                      target_features_requested=args.target_features,
                      compilation="compilation/report.json")
        if args.retry_static_prefill_from:
            result["retained_failed_job"] = str(Path(args.retry_static_prefill_from).resolve())
        if retry_preflight is not None:
            save_json(evidence_root / "retry-preflight.json", retry_preflight)
            result["retry_preflight"] = {"status": retry_preflight["status"],
                "report": str(evidence_root / "retry-preflight.json"),
                "scope": "Final failed-job/process/probe gates only; no model execution or accuracy pass"}
        if args.hiai_include or args.source_root:
            from qwen35_moe_dflash.original_target_contract import capture_runtime_sources
            try:
                sources = capture_runtime_sources(evidence_root / "runtime-sources", include=args.hiai_include,
                                                  source_root=args.source_root)
                result.update(runtime_sources=sources["status"],
                              runtime_sources_report=str(evidence_root / "runtime-sources/report.txt"))
                print("full-target: original input/state and selected HIAI declarations: " + result["runtime_sources_report"], flush=True)
            except (OSError, ValueError) as error:
                result.update(runtime_sources="UNAVAILABLE", runtime_sources_error=f"{type(error).__name__}: {error}")
        print(f"full-target: output {root}", flush=True)
        print(f"full-target: original AIR; profile={args.profile}; external weights; kirinx90", flush=True)

        def progress(row):
            if (row["stage"] != "input-snapshot" or "completed" not in row
                    or row["completed"] in (1, row.get("total")) or row["completed"] % 25 == 0):
                print("full-target: " + row["message"], flush=True)

        if not args.resume:
            if retry_preflight is not None:
                from qwen35_moe_dflash.target_export_reuse import _absolute, _read_small
                for identity in retry_preflight.get("metadata_bindings", []):
                    if _read_small(_absolute(identity["path"]), 16 << 20)[1] != identity:
                        raise ValueError("static64 retry preflight metadata changed before compilation")
            compiled = compile_existing_air(
                air, root / "compilation", args.ddk_root, ddk_id=args.ddk_id,
                tbe_diagnosis=args.tbe_diagnosis, timeout_seconds=args.timeout_seconds,
                omg_path=args.omg, input_shape=profile["input_shape"], dynamic_dims=profile["dynamic_dims"],
                limit_partition_size=profile["limit_partition_size"],
                save_weights_as_external_data=profile["save_weights_as_external_data"], progress=progress,
                input_layout=args.input_layout,
                **({"reuse_failed_job": args.retry_static_prefill_from} if args.retry_static_prefill_from else {}),
            )
        result["status"] = compiled["status"]
        result["model_compilation"] = compiled["model_compilation"]
        if "retained_input_reuse" in compiled:
            result["input_snapshot_reuse"] = {
                "status": compiled["retained_input_reuse"].get("status"),
                "explicit_weight_copy_bytes": compiled["retained_input_reuse"].get("explicit_weight_copy_bytes"),
                "report": str(root / "compilation/retained-input-reuse.json"),
                "scope": "Freshly verified shared readonly snapshot inodes; independent compiler outputs and audits"}
        if (root / "compilation/report.json").is_file():
            from qwen35_moe_dflash.hashing import sha256_file
            result["compilation_report_sha256"] = sha256_file(root / "compilation/report.json")
        result["compile_log"] = str(root / "compilation/compile.log")
        if compiled["status"] == "COMPILED":
            result["omc"] = str(root / "compilation" / compiled["omc"]["path"])
            print("full-target: OMC compiled; retained generated external files", flush=True)
            if not args.skip_inspection:
                print("full-target: inspecting compiled graph interface (no board execution)", flush=True)
                try:
                    from qwen35_moe_dflash.existing_omc_inspect import inspect_existing_omc
                    inspected = inspect_existing_omc(root / "compilation", evidence_root / "interface", timeout_seconds=120)
                    result["interface_inspection"] = inspected["status"]
                    result["interface_report"] = str(evidence_root / "interface/report.txt")
                except Exception as error:
                    result["interface_inspection"] = "UNAVAILABLE"
                    result["interface_error"] = f"{type(error).__name__}: {error}"
                try:
                    from qwen35_moe_dflash.original_target_contract import boundary_report, inspect_target_contract
                    contract = inspect_target_contract(root / "compilation", evidence_root / "target-contract", profile)
                    result["target_contract_inspection"] = contract["status"]
                    result["target_contract_report"] = str(evidence_root / "target-contract/report.json")
                    print(boundary_report(contract), end="", flush=True)
                except Exception as error:
                    result.update(target_contract_inspection="UNAVAILABLE", target_contract_error=f"{type(error).__name__}: {error}")
            else:
                result["interface_inspection"] = "NOT_RUN"
                result["target_contract_inspection"] = "NOT_RUN"
            result["next"] = "Confirm original graph I/O, external weights, dynamic gear and RefData state contract before board execution"
            if args.run_board:
                from qwen35_moe_dflash.original_target_board import execute_target_tool
                board = execute_target_tool(root / "compilation", evidence_root / "target-contract/report.json",
                    evidence_root / "board", profile=profile, gear=args.sequence_length,
                    input_dir=args.board_input_dir, host_input_dir=args.input_dir, tool=args.board_tool,
                    board_model_root=args.board_model_root, timeout_seconds=args.timeout_seconds,
                    target_features=args.target_features)
                result.update(board_status=board["status"], board_result=str(evidence_root / "board/result.json"),
                              device_execution=board["device_execution"], numerical_validation=board["numerical_validation"],
                              delivery_status="TARGET_EXECUTION_FAILED" if board["status"] == "FAILED" else "TARGET_EXECUTED_ACCURACY_UNVERIFIED",
                              next="Review actual retrieved logits/state output mapping; establish same-input reference and 64-prefill to 1-decode state handoff before accuracy/performance claims")
                if "feature_output_evidence" in board:
                    result["feature_output_evidence"] = board["feature_output_evidence"]
                if board["status"] == "FAILED":
                    from qwen35_moe_dflash.tool_logs import log_excerpt
                    result["board_error"] = board.get("error", board.get("evidence_error", "target execution failed"))
                    for name in ("tool.stderr", "tool.stdout"):
                        if (evidence_root / "board" / name).is_file():
                            excerpt = log_excerpt(evidence_root / "board" / name)
                            if excerpt:
                                print("full-target: board diagnostics from " + name + "\n" + excerpt, flush=True)
        else:
            result["error"] = compiled.get("error", "compilation failed")
            result["stage"] = compiled.get("stage", "compilation")
            for field in ("returncode", "timeout_seconds"):
                if field in compiled:
                    result[field] = compiled[field]
            result["next"] = "Inspect compilation/report.txt and retained compiler diagnostics; completed source snapshots are preserved"
        result_path = evidence_root / "result.json"
        save_json(result_path, result)
        _print_result(result, result_path=result_path,
                      log_paths=(root / "compilation/compile.log", root / "compilation/help.log"))
        return 0 if result["status"] == "COMPILED" and result.get("board_status") != "FAILED" else 1
    except Exception as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        failure_path = (continuation / "failure.json" if continuation else root / "failure.json" if owned_root else None)
        if failure_path is not None:
            save_json(failure_path, result)
        _print_result(result, result_path=failure_path)
        return 1
    finally:
        guards.close()


if __name__ == "__main__":
    raise SystemExit(main())
