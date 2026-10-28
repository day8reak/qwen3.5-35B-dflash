#!/usr/bin/env python3
"""Prepare, compare and execute the complete six-layer draft independently of target."""
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

# Match the FP16 tiny target/draft compiler path. Generic original-AIR
# compilation deliberately has no dtype/layout override, since it may be W4.
DRAFT_COMPILE_OPTIONS = {
    "use_origin_format": "true", "weight_data_type": "FP16",
    "limit_partition_size": "false", "save_weights_as_external_data": "true",
}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-weights", help="original target folder; only embedding and LM head tensors are read")
    p.add_argument("--draft-weights", help="downloaded DFlash folder containing config.json and safetensors")
    p.add_argument("--sequence-length", type=int, help="fixed context capacity; default 64")
    p.add_argument("--valid-length", type=int, help="valid fixture prefix, default min(6, capacity)")
    p.add_argument("--resume", help="previous draft job; import completed stages into full-draft, or resume it in place")
    p.add_argument("--output", help="job directory below AI_RUN_DIR; default full-draft automatically resumes")
    p.add_argument("--ge-root", default=os.environ.get("CANN_ROOT"))
    p.add_argument("--ddk-root", default=os.environ.get("CPRO_DDK_ROOT") or os.environ.get("DDK_PATH"))
    p.add_argument("--ddk-id", default=os.environ.get("CPRO_DDK_ID"))
    p.add_argument("--tbe-diagnosis", help="reuse completed TBE diagnosis; no new diagnosis matrix")
    p.add_argument("--weight-storage", choices=("external", "embedded"),
                   help="OMC weight storage; fresh jobs default external, resume inherits the recorded compilation")
    p.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    p.add_argument("--cxx11-abi", type=int, choices=(0, 1), default=0)
    p.add_argument("--timeout-seconds", type=int, default=3600)
    p.add_argument("--schema-library", help="inspection only: selected DDK protobuf library, read without loading it")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true", help="prepare full weights and PyTorch fixture; no GE/DDK/board")
    mode.add_argument("--compile-only", action="store_true", help="prepare reference, export AIR and compile OMC; no board")
    mode.add_argument("--inspect-only", action="store_true", help="read --resume job's retained compiler evidence; no export, compilation or board")
    return p


def _record(folder, filename):
    from qwen35_moe_dflash.hashing import sha256_file
    path = folder / filename
    return {"directory": str(folder.resolve()), "manifest": {
        "path": filename, "bytes": path.stat().st_size, "sha256": sha256_file(path)}}


def _read_component(record):
    from qwen35_moe_dflash.artifacts import checked_file
    folder = Path(record["directory"]).resolve()
    path = checked_file(folder, record["manifest"])
    return folder, json.loads(path.read_text())


def _verify_air(folder, manifest):
    from qwen35_moe_dflash.artifacts import checked_file
    if manifest.get("status") != "EXPORTED" or [g["name"] for g in manifest.get("graphs", [])] != ["draft"]:
        raise ValueError("expected an exported draft-only AIR manifest")
    for row in manifest["graphs"][0]["files"]:
        checked_file(folder, row)


def _retained_output_candidate(previous, compile_options):
    """Identify the old post-OMG path error; recovery still verifies every file."""
    from qwen35_moe_dflash.retained_air import LEGACY_OUTPUT_ERROR
    folder = previous / "compilation"
    path = folder / "report.json"
    if not path.is_file():
        return None
    report = json.loads(path.read_text())
    if (report.get("artifact_kind") == "qwen35-existing-air-compilation"
            and report.get("status") == "FAILED" and report.get("stage") == "output-check"
            and report.get("error") == LEGACY_OUTPUT_ERROR
            and report.get("requested_options") == compile_options):
        return folder
    return None


def _external_weight_candidates(compiled):
    """Recognize observed payload names, not every generated diagnostic file."""
    weights = []
    for row in compiled.get("external_data_candidates", []):
        path = Path(row["path"])
        if "dump_graph" in path.parts:
            continue
        if (path.suffix.lower() in {".weight", ".weights"}
                or path.name.lower().endswith(".omc.data") or "model.weights" in path.parts):
            if type(row.get("bytes")) is not int or row["bytes"] < 0:
                raise ValueError("invalid external-weight candidate size: " + row["path"])
            if row["bytes"]:
                weights.append(row)
    return weights


def _weight_path(explicit, key):
    from qwen35_moe_dflash.resources import load_resources, resource_path
    if explicit:
        return resource_path({key: explicit}, key, directory=True)
    resources, _ = load_resources()
    try:
        return resource_path(resources, key, directory=True)
    except (ValueError, FileNotFoundError) as error:
        option = "--draft-weights" if key == "draft_weights" else "--target-weights"
        raise ValueError(f"{key}: set {option} to the downloaded model folder containing config.json and safetensors. {error}") from error


def _print_board_failure(result, hilog=None):
    """Show retained native failures without another device invocation."""
    from qwen35_moe_dflash.tool_logs import log_excerpt
    folder = Path(result["board"])
    print(f"full-draft: Board failure evidence: {folder / 'result.json'}", flush=True)
    if result.get("failure_stage"):
        print(f"full-draft: Native failure stage: {result['failure_stage']}", flush=True)
    for name in ("error", "transport_error", "comparison_error"):
        if result.get(name):
            print(f"full-draft: {name}: {str(result[name])[:4096]}", flush=True)
    names = (("runner-build.log",) if result.get("board_stage") == "build-runner"
             else ("runtime.stderr", "draft.stderr", "draft.stdout"))
    for name in names:
        path = folder / name
        try:
            if path.is_file() and path.stat().st_size:
                print(f"full-draft: {path}\n{log_excerpt(path)}", flush=True)
        except OSError as error:
            print(f"full-draft: Could not read {path}: {error}", flush=True)
    if hilog:
        print(f"full-draft: HIAI system log: {folder / 'draft-hilog.txt'}; status={hilog.get('status')}", flush=True)
        print("full-draft: System-log candidates may include other sessions; inspect PID and context.", flush=True)
        lines = hilog.get("summary_lines", [])
        if not lines:
            summary = hilog.get("summary", {})
            pid = summary.get("recorded_pid", hilog.get("process_identity", {}).get("recorded_pid"))
            print(f"full-draft: HiLog summary: status={summary.get('status', 'EMPTY')}; recorded_pid={pid}", flush=True)
            for key in ("reason", "snapshot_bytes", "max_bytes"):
                if key in summary:
                    print(f"full-draft: HiLog summary {key}: {str(summary[key])[:1000]}", flush=True)
            scope = summary.get("scope", hilog.get("scope"))
            if scope:
                print(f"full-draft: HiLog summary scope: {str(scope)[:1000]}", flush=True)
        for line in lines:
            print(str(line)[:1000], flush=True)


def main(argv=None):
    with ExitStack() as stack:
        return _main(parser().parse_args(argv), stack)


def _main(args, stack):
    from qwen35_moe_dflash.artifacts import checked_file, output_path, save_json

    result = {"schema_version": 1, "artifact_kind": "qwen35-full-draft-job",
              "status": "PREPARING", "stage": "preflight", "components": {},
              "target_execution": "NOT_RUN", "device_execution": "NOT_RUN",
              "numerical_validation": "NOT_RUN", "runtime_placement": "UNVERIFIED",
              "compile_options": dict(DRAFT_COMPILE_OPTIONS),
              "scope": "Complete draft with fixed synthetic target features; not language quality, acceptance or speedup evidence."}
    root = None
    retained_compilation = None
    board_hilog = None
    workspace = None
    weight_storage = args.weight_storage or "external"
    compile_options = {**DRAFT_COMPILE_OPTIONS,
                       "save_weights_as_external_data": "true" if weight_storage == "external" else "false"}
    result.update(weight_storage=weight_storage, compile_options=compile_options)

    def progress(row):
        message = row["message"]
        if row.get("stage") == "input-snapshot" and message == "Snapshotting input":
            index, total = row.get("completed", 0), row.get("total", 0)
            if index not in (0, total - 1) and index % 10:
                return
            message = (f"Snapshotting input {index + 1}/{total}: "
                       f"{row.get('source_relative', '')} ({row.get('bytes', 0)} bytes)")
        print("full-draft: " + message, flush=True)

    def stage(name, message):
        result["stage"] = name
        progress({"message": message})

    def save_job(name, value):
        if workspace is not None:
            workspace.save(name, value)
        else:
            save_json(root / name, value)

    def component(name, filename, build):
        if name in result["components"]:
            stage(name, f"Reusing completed {name}; verifying its manifest")
            return _read_component(result["components"][name])
        folder = root / name
        build(folder)
        record = _record(folder, filename)
        _, payload = _read_component(record)
        if payload.get("status") == "FAILED":
            raise RuntimeError(payload.get("error", f"{name} failed"))
        result["components"][name] = record
        # A completed receipt survives interruption; in-place retries remove
        # obsolete receipts before replacing their stage directories.
        save_job(f"completed-{name}.json", record)
        return _read_component(record)

    try:
        if args.timeout_seconds <= 0:
            raise ValueError("--timeout-seconds must be positive")
        run = os.environ.get("AI_RUN_DIR")
        if not run:
            raise ValueError("AI_RUN_DIR is required; source the project env.sh")
        if args.inspect_only:
            if not args.resume:
                raise ValueError("--inspect-only requires --resume pointing to the retained full-draft job")
            if any(value is not None for value in (args.target_weights, args.draft_weights,
                                                   args.sequence_length, args.valid_length, args.weight_storage)):
                raise ValueError("inspection reads the retained job; do not select new weights, inputs or weight storage")
            from qwen35_moe_dflash.draft_compile_inspect import inspect_draft_compile
            inspection_dir = output_path(args.output or Path(run) / ("full-draft-inspect-" + uuid.uuid4().hex[:12]))
            inspected = inspect_draft_compile(args.resume, inspection_dir, schema_library=args.schema_library)
            print(json.dumps({"status": inspected["status"], "output": str(inspection_dir),
                              "comparison": str(inspection_dir / "report.txt")}, ensure_ascii=False), flush=True)
            print((inspection_dir / "report.txt").read_text(), flush=True)
            return 0
        if args.schema_library:
            raise ValueError("--schema-library is only used with --inspect-only")
        requested_output = Path(args.output) if args.output else Path(run) / "full-draft"
        if requested_output.is_symlink():
            raise ValueError("draft workspace output must not be a symlink")
        destination = output_path(requested_output)
        pending = None
        previous = Path(args.resume).resolve() if args.resume else None
        if not args.output or destination == Path(run).resolve() / "full-draft" or destination.exists():
            from qwen35_moe_dflash.draft_workspace import DraftWorkspace
            workspace = stack.enter_context(DraftWorkspace(destination))
            pending = workspace.load_pending()
            if pending is not None:
                previous = destination
            elif (destination / "result.json").is_file() or (destination / "job.json").is_file():
                if previous is not None and previous != destination:
                    raise ValueError("full-draft already contains a job; omit --resume to continue it, "
                                     "or choose a new --output for another job")
                previous = destination
        settings = {"sequence_length": 64 if args.sequence_length is None else args.sequence_length, "valid_length": args.valid_length,
                    "seed": 17}
        if previous is not None:
            prior_result = previous / "result.json"
            old = pending if pending is not None else json.loads(
                (prior_result if prior_result.is_file() else previous / "job.json").read_text())
            if old.get("artifact_kind") != result["artifact_kind"]:
                raise ValueError("--resume requires a previous full-draft job")
            settings = old["settings"]
            for key in ("sequence_length", "valid_length"):
                value = getattr(args, key)
                if value is not None and value != settings[key]:
                    raise ValueError(f"resume {key} differs; use a new job for changed inputs")
            selected_paths = old.get("weight_paths", {})
            for owner in ("target", "draft"):
                selected = getattr(args, owner + "_weights")
                if selected and (workspace is None or previous != destination
                        or selected_paths.get(owner) != str(Path(selected).expanduser().resolve())):
                    raise ValueError("resume reuses existing weight identity; use a new --output to change weights")
            if selected_paths:
                result["weight_paths"] = selected_paths
            if old.get("imported_from"):
                result["imported_from"] = old["imported_from"]
            result["components"] = dict(old["components"])
            for name in ("prepared", "fixture", "air", "compilation", "bundle"):
                receipt = previous / f"completed-{name}.json"
                if pending is None and receipt.is_file():
                    record = json.loads(receipt.read_text())
                    _read_component(record)
                    result["components"][name] = record
            # Use the verified compilation receipt before the parent job's
            # summary, including when an export-contract change rebuilds AIR.
            inherited_options = old.get("compile_options", {})
            if "compilation" in result["components"]:
                _, inherited_compile = _read_component(result["components"]["compilation"])
                inherited_options = inherited_compile.get("requested_options", {})
            if args.weight_storage is None:
                stored = inherited_options.get("save_weights_as_external_data")
                if stored not in (None, "true", "false"):
                    raise ValueError("resume compilation has an invalid weight-storage option")
                weight_storage = "embedded" if stored == "false" else "external"
                compile_options = {**DRAFT_COMPILE_OPTIONS,
                    "save_weights_as_external_data": "true" if weight_storage == "external" else "false"}
                result.update(weight_storage=weight_storage, compile_options=compile_options)
            result["resumed_from"] = str(previous)
            if "air" in result["components"]:
                from qwen35_moe_dflash.draft_standalone import CURRENT_DRAFT_EXPORT_CONTRACT
                # Verify the immutable old receipt before deciding to rebuild.
                # The prepared weights and CPU reference remain valid: this
                # contract changes serialization, not the draft computation.
                _, prior_air = _read_component(result["components"]["air"])
                if prior_air.get("export_contract") != CURRENT_DRAFT_EXPORT_CONTRACT:
                    obsolete = {name: result["components"].pop(name)
                                for name in ("air", "compilation", "bundle")
                                if name in result["components"]}
                    result["resume_invalidations"] = [{
                        "reason": "OBSOLETE_DRAFT_EXPORT_CONTRACT",
                        "previous_export_contract": prior_air.get("export_contract"),
                        "required_export_contract": CURRENT_DRAFT_EXPORT_CONTRACT,
                        "components": obsolete,
                        "preserved": [name for name in ("prepared", "fixture") if name in result["components"]],
                        "scope": ("Replace obsolete stages in the fixed workspace." if workspace else
                                  "Re-export and recompile in the new job; previous evidence is unchanged.")}]
                    stage("resume", "Previous AIR export contract needs updating; reusing weights/reference and rebuilding AIR/OMC")
            if "compilation" in result["components"]:
                _, prior_compile = _read_component(result["components"]["compilation"])
                options = prior_compile.get("requested_options", {})
                if any(options.get(key) != value for key, value in compile_options.items()):
                    obsolete = {name: result["components"].pop(name)
                                for name in ("compilation", "bundle") if name in result["components"]}
                    result.setdefault("resume_invalidations", []).append({
                        "reason": "DRAFT_COMPILE_OPTIONS_CHANGED", "previous_options": options,
                        "required_options": dict(compile_options), "components": obsolete,
                        "preserved": [name for name in ("prepared", "fixture", "air") if name in result["components"]],
                        "scope": ("Replace compilation and bundle in the fixed workspace." if workspace else
                                  "Recompile existing AIR in the new job; previous evidence is unchanged.")})
                    stage("resume", f"Updating draft compilation to FP16 with {weight_storage} OMC weights; reusing weights/reference/AIR")
            if (workspace is None and not args.prepare_only and "air" in result["components"]
                    and "compilation" not in result["components"] and not result.get("resume_invalidations")):
                retained_compilation = _retained_output_candidate(previous, compile_options)
        if settings["sequence_length"] < 2:
            raise ValueError("--sequence-length must be at least 2")
        if settings["valid_length"] is None:
            settings["valid_length"] = min(6, settings["sequence_length"])
        if not 1 <= settings["valid_length"] <= settings["sequence_length"]:
            raise ValueError("--valid-length must be within the context capacity")
        result["settings"] = settings

        from qwen35_moe_dflash.draft_standalone import prepare_draft, export_draft
        from qwen35_moe_dflash.draft_reference import prepare_draft_reference

        if "prepared" not in result["components"]:
            paths = result.get("weight_paths", {})
            target = _weight_path(args.target_weights or paths.get("target"), "target_weights")
            draft = _weight_path(args.draft_weights or paths.get("draft"), "draft_weights")
            result["weight_paths"] = {"target": str(target), "draft": str(draft)}
        else:
            target = draft = None
        if not args.prepare_only:
            if "air" not in result["components"] and not args.ge_root:
                raise ValueError("CANN_ROOT or --ge-root is required for AIR export")
            if ("compilation" not in result["components"] and retained_compilation is None
                    and (not args.ddk_root or not args.ddk_id)):
                raise ValueError("CPRO_DDK_ROOT/DDK_PATH and CPRO_DDK_ID are required for OMG")

        # Validate the retained AIR before removing any old results. Importing
        # keeps manifests byte-for-byte, including their historical provenance.
        if workspace is not None:
            if "air" in result["components"]:
                prior_air_dir, prior_air = _read_component(result["components"]["air"])
                _verify_air(prior_air_dir, prior_air)
            if previous is not None and previous != destination:
                result["imported_from"] = str(previous)
            result = workspace.adopt(result)
            workspace.reset(result)
            root = destination
            result["workspace_mode"] = "in-place"
            result["output"] = str(root)
            save_job("job.json", result)
        else:
            root = destination
            root.mkdir(parents=True, exist_ok=False)
            result["output"] = str(root)
            save_job("job.json", result)
        stage("preflight", f"output {root}; full six-layer draft, block 16, full vocabulary; OMC weights={weight_storage}")

        stage("prepared", "Preparing only draft weights and the shared embedding/LM head")
        prepared_dir, prepared = component("prepared", "prepared.json", lambda folder: prepare_draft(
            folder, settings["sequence_length"], target_path=target, draft_path=draft, progress=progress))
        if prepared.get("tiny") or prepared.get("draft_config", {}).get("num_hidden_layers") != 6:
            raise ValueError("full-draft entry requires the full six-layer checkpoint")
        stage("fixture", "Running standalone PyTorch draft on fixed inputs")
        fixture_dir, fixture = component("fixture", "report.json", lambda folder: prepare_draft_reference(
            prepared_dir, folder, seed=settings["seed"], valid_length=settings["valid_length"]))
        if (fixture.get("status") != "REFERENCE_COMPLETE" or not fixture.get("expected_top1")
                or not fixture.get("expected_hidden")):
            raise RuntimeError(fixture.get("error", "complete draft reference outputs are required"))
        result["reference"] = fixture.get("status")
        if args.prepare_only:
            result.update(status="PREPARED", stage="complete")
        else:
            stage("air", "Exporting only the full draft AIR")
            air_dir, exported = component("air", "manifest.json", lambda folder: export_draft(
                prepared_dir, folder, args.ge_root, cxx=args.cxx, cxx11_abi=args.cxx11_abi,
                timeout_seconds=args.timeout_seconds, progress=progress))
            _verify_air(air_dir, exported)
            stage("compilation", f"Verifying retained OMG output with {weight_storage} OMC weights; no recompilation"
                  if retained_compilation is not None else
                  f"Compiling draft AIR with {weight_storage} OMC weights: use_origin_format=true, weight_data_type=FP16, "
                  f"save_weights_as_external_data={compile_options['save_weights_as_external_data']}")
            from qwen35_moe_dflash.existing_air import compile_existing_air
            def compile_draft(folder):
                if retained_compilation is not None:
                    from qwen35_moe_dflash.retained_air import recover_retained_air
                    # Any failed recovery check is terminal; do not silently run OMG.
                    compiled = recover_retained_air(
                        retained_compilation, folder, export_dir=air_dir,
                        expected_options=compile_options, ddk_id=args.ddk_id,
                        ddk_root=args.ddk_root, progress=progress)
                    result["compilation_recovery"] = compiled["recovery"]
                else:
                    compiled = compile_existing_air(
                        air_dir / exported["graphs"][0]["air"], folder, args.ddk_root,
                        ddk_id=args.ddk_id, tbe_diagnosis=args.tbe_diagnosis,
                        timeout_seconds=args.timeout_seconds, input_layout="air-directory",
                        limit_partition_size=False, save_weights_as_external_data=(weight_storage == "external"),
                        use_origin_format=True, weight_data_type="FP16", progress=progress)
                _verify_air(air_dir, exported)
                if compiled.get("status") != "COMPILED":
                    result["compile_log"] = str(folder / "compile.log")
                    raise RuntimeError(compiled.get("error", "draft compilation failed"))
            compilation_dir, compiled = component("compilation", "report.json", compile_draft)
            if compiled.get("status") != "COMPILED":
                raise ValueError("resume compilation is not complete")
            if any(compiled.get("requested_options", {}).get(key) != value
                   for key, value in compile_options.items()):
                raise ValueError("draft compilation does not match the required FP16 compiler options")
            model_bytes = compiled["omc"].get("bytes")
            if type(model_bytes) is not int or not 0 < model_bytes <= (1 << 32) - 1:
                raise ValueError("compiled OMC exceeds the native SetModelBuffer UINT32_MAX size limit or has invalid size")
            model_path = checked_file(compilation_dir, compiled["omc"])
            if model_path.stat().st_size > (1 << 32) - 1:
                raise ValueError("compiled OMC exceeds the native SetModelBuffer UINT32_MAX size limit")
            observed_weights = _external_weight_candidates(compiled)
            result["weight_storage_validation"] = {
                "requested": weight_storage, "omc_bytes": model_bytes,
                "observed_weight_candidates": observed_weights,
                "scope": "Recorded compiler option and known weight filenames only; OMC dependency contents are not decoded."}
            if weight_storage == "embedded" and observed_weights:
                raise ValueError("embedded OMC compilation retained nonempty external weight payloads: "
                                 + ", ".join(row["path"] for row in observed_weights[:8]))
            if args.ddk_id and compiled["ddk_id"] != args.ddk_id:
                raise ValueError("selected DDK differs from the reused compilation")
            result["model_compilation"] = "PASSED"
            if args.compile_only:
                result.update(status="COMPILED", stage="complete")
            else:
                from qwen35_moe_dflash.draft_board import prepare_draft_bundle, execute_draft_bundle
                stage("bundle", "Preparing hash-bound draft inputs and compiled payloads")
                bundle_dir, _ = component("bundle", "manifest.json", lambda folder: prepare_draft_bundle(
                    compilation_dir, fixture_dir, folder, config=prepared["draft_config"], ddk_id=compiled["ddk_id"], export_dir=air_dir))
                stage("board", "Running the complete draft on the board")
                board = execute_draft_bundle(bundle_dir, root / "board", timeout_seconds=args.timeout_seconds)
                result["board"] = str(root / "board")
                result["status"] = board["status"]
                result["device_execution"] = board.get("device_execution", "UNVERIFIED")
                result["numerical_validation"] = board.get("numerical_validation", "UNVERIFIED")
                result["board_stage"] = board.get("stage", "UNVERIFIED")
                board_hilog = board.get("hilog")
                if board_hilog:
                    result["hilog_status"] = board_hilog.get("status", "UNVERIFIED")
                for key in ("failure_stage", "native_status", "error", "comparison_error",
                            "transport_error", "remote_exitcode", "retrieval_errors", "native_process_id"):
                    if key in board:
                        result[key] = board[key]
                if board["status"] == "FAILED" and not result.get("error"):
                    result["error"] = (board.get("transport_error") or board.get("comparison_error")
                                       or f"Board execution failed; inspect {root / 'board/result.json'}")
                result["stage"] = "complete" if board["status"] != "FAILED" else "board"
    except Exception as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
    if root is not None and root.is_dir() and result.get("output") == str(root):
        save_job("result.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if result.get("compile_log"):
        from qwen35_moe_dflash.tool_logs import log_excerpt
        print(log_excerpt(Path(result["compile_log"])), flush=True)
    if result.get("board") and result["status"] == "FAILED":
        _print_board_failure(result, board_hilog)
    return 1 if result["status"] == "FAILED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
