#!/usr/bin/env python3
"""Reuse both compiled GDN probes with three output buffer prefills; no OMG compilation."""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_fused_probes as board

from qwen35_moe_dflash.artifacts import output_path, save_json
from qwen35_moe_dflash.compile import _stateful_json
from qwen35_moe_dflash.ge_fused.export import _step_file
from qwen35_moe_dflash.ge_fused.gdn_diagnostics import (
    GDN_CASES, FILLS, collect_compile_evidence, collect_sources, compare_fills, validate_io,
)
from qwen35_moe_dflash.hashing import sha256_file
from qwen35_moe_dflash.hdc_run import _positive, _text


def _resume_report(path):
    root = Path(path).expanduser().resolve()
    report = _stateful_json(root / "report.json")
    if (not isinstance(report, dict) or type(report.get("schema_version")) is not int
            or report["schema_version"] != 1 or report.get("fills") != list(FILLS)
            or report.get("model_compilation") != "NOT_RUN" or report.get("status") not in {"FAILED", "DIAGNOSIS_COMPLETE"}
            or not isinstance(report.get("runs"), list) or len(report["runs"]) > 6
            or any(not isinstance(row, dict) for row in report["runs"])
            or not isinstance(report.get("probe_run"), str)
            or not Path(report.get("probe_run", "")).is_absolute()):
        raise ValueError("--resume requires an existing GDN diagnostic report")
    return root, report


def _reuse_execution(row, path, names, fill, out, runner_cache):
    """Recheck one successful retained run; never reuse an uncertain Process."""
    previous = Path(row["output"])
    if not previous.is_absolute() or previous.is_symlink():
        raise ValueError("resumed execution output must be an absolute regular directory")
    execution = _stateful_json(previous / "result.json")
    if any(row.get(key) != value for key, value in execution.items()):
        raise ValueError("resumed execution differs from its saved result.json")
    if (execution.get("status") != "EXECUTION_COMPLETE" or execution.get("probe_run") != str(path)
            or execution.get("selected_cases") != names or execution.get("output_fill_byte") != fill
            or execution.get("probe_manifest_sha256") != sha256_file(path / "manifest.json")):
        raise ValueError("resumed execution does not match selected GDN model/input/fill contract")
    if os.environ.get("SN") != execution.get("serial"):
        raise ValueError("SN differs from the retained diagnostic device")
    for relative, digest in execution.get("retrieved_sha256", {}).items():
        file = (previous / relative).resolve()
        if previous.resolve() not in file.parents or not file.is_file() or sha256_file(file) != digest:
            raise ValueError("resumed retrieved file SHA256 mismatch: " + relative)
    for name in names:
        relative = f"results/{name}/result.json"
        if relative not in execution.get("retrieved_sha256", {}):
            raise ValueError("resumed native result lacks its retrieved SHA256")
        native = _stateful_json(previous / relative)
        if (native.get("runtime_version") != execution.get("runtime_version")
                or native.get("firmware_id") != execution.get("firmware_id")):
            raise ValueError("resumed native runtime/firmware differs from its run")
    build = Path(execution["runner_build"])
    if not build.is_absolute():
        raise ValueError("resumed runner build must be absolute")
    runner = build / "build/qwen35_dflash_hiai"
    linked = _stateful_json(build / "result.json")
    digest = sha256_file(runner)
    if (linked.get("status") != "LINKED" or linked.get("binary_sha256") != digest
            or linked.get("command_sha256") != sha256_file(build / "command.json")):
        raise ValueError("resumed runner differs from its recorded build")
    upload = _stateful_json(previous / "upload-plan.json")
    runners = [item for item in upload.get("files", []) if item.get("remote") == "runner"]
    if len(runners) != 1 or runners[0].get("sha256") != digest or runners[0].get("bytes") != runner.stat().st_size:
        raise ValueError("resumed runner differs from the one uploaded for the completed trial")
    if runner_cache and runner_cache["sha256"] != digest:
        raise ValueError("completed trials used different runner binaries")
    runner_cache.update(runner=runner, sha256=digest, native_build=build)
    comparison = out / f"rechecked-{fill:02x}-{'-'.join(names)}"
    checked = board.compare_fused_probes(path, previous / "results", comparison, case_names=names)
    if any(case.get("result_validation") != "PASSED" or case.get("process") != "PASSED" for case in checked["cases"]):
        raise ValueError("resumed result failed source/input/output revalidation")
    return execution, checked


def run(args):
    storage = Path(_text(os.environ.get("AI_RUN_DIR"), "AI_RUN_DIR"))
    out = output_path(args.output or storage / ("gdn-diagnostic-" + uuid.uuid4().hex[:12]))
    resume_root, resumed = _resume_report(args.resume) if getattr(args, "resume", None) else (None, None)
    root = Path(_text(resumed["probe_run"] if resumed else args.probe_run,
                      "--probe-run / FUSED_PROBE_RUN")).expanduser().resolve()
    if root == out or root in out.parents or out in root.parents:
        raise ValueError("diagnostic output must be separate from the original batch")
    if resume_root and (out == resume_root or resume_root in out.parents or out in resume_root.parents):
        raise ValueError("resumed diagnostic must use a new separate output directory")
    ddk_root = os.environ.get("CPRO_DDK_ROOT") or os.environ.get("DDK_PATH")
    if ddk_root and (out == Path(ddk_root).resolve() or Path(ddk_root).resolve() in out.parents):
        raise ValueError("diagnostic output must be outside the immutable DDK")
    out.mkdir(parents=True, exist_ok=False)
    result = {"schema_version": 1, "status": "FAILED", "output": str(out), "probe_run": str(root),
              "model_compilation": "NOT_RUN", "numerical_acceptance": "NOT_DEFINED",
              "placement": "UNVERIFIED", "fills": list(FILLS), "runs": [], "cases": [], "completed_trials": [],
              "scope": "Same GDN OMC/input bytes. Host buffer readback and prefill are observable; device buffer coherence/kernel writes are not directly observed."}
    try:
        groups, compiled = board._probe_groups(root, out)
        if not set(GDN_CASES) <= compiled:
            raise ValueError("both gdn_prefill and gdn_decode must already be compiled; no compilation is performed")
        selected = [(path, manifest, [case for case in manifest["cases"]
                                     if case["name"] in GDN_CASES and case["status"] == "COMPILED"])
                    for path, manifest in groups]
        selected = [(path, manifest, cases) for path, manifest, cases in selected if cases]
        result["probe_manifests"] = {str(path): sha256_file(path / "manifest.json") for path, _, _ in selected}
        reusable = {}
        if resumed:
            if resumed.get("probe_manifests") != result["probe_manifests"]:
                raise ValueError("probe manifests changed since the prior diagnosis")
            result["resumed_from"] = {"path": str(resume_root), "sha256": sha256_file(resume_root / "report.json")}
            for row in resumed["runs"]:
                if row.get("status") != "EXECUTION_COMPLETE":
                    continue
                key = (row.get("fill"), row.get("probe_run"))
                if key in reusable or key[0] not in FILLS or key[1] not in result["probe_manifests"]:
                    raise ValueError("duplicate or unknown completed diagnostic trial")
                reusable[key] = row
        print("gdn-diagnostic: copying installed GDN source evidence", flush=True)
        source = collect_sources(ddk_root, out / "installed")
        result["installed_source"] = {"status": source["status"], "report": str(out / "installed/report.txt")}
        result["compiler_evidence"] = []
        for index, (path, _, cases) in enumerate(selected):
            evidence = out / f"compiler-{index}"
            collect_compile_evidence(path, [case["name"] for case in cases], evidence)
            result["compiler_evidence"].append(str(evidence / "report.json"))
        runner_cache, gathered, diagnostics, native_identity = {}, {}, {}, None
        for fill in FILLS:
            print(f"gdn-diagnostic: output fill 0x{fill:02x}; same OMC and inputs", flush=True)
            for group_index, (path, manifest, cases) in enumerate(selected):
                names = [case["name"] for case in cases]
                old = reusable.get((fill, str(path)))
                comparison = None
                if old is not None:
                    print(f"gdn-diagnostic: rechecking completed fill=0x{fill:02x} ({', '.join(names)}); no board execution", flush=True)
                    execution, comparison = _reuse_execution(old, path, names, fill, out, runner_cache)
                else:
                    execution = board.execute(SimpleNamespace(probe_run=str(path), output=str(out / f"fill-{fill:02x}-{group_index}"),
                        timeout_seconds=args.timeout_seconds, selected_cases=names, output_fill_byte=fill), runner_cache=runner_cache)
                result["runs"].append({"fill": fill, "cases": names, "reused": old is not None, **execution})
                with (out / "progress.jsonl").open("a") as progress:
                    progress.write(json.dumps({"fill": fill, "cases": names, "status": execution["status"],
                                               "output": execution["output"]}) + "\n")
                if execution["status"] != "EXECUTION_COMPLETE":
                    result["failed_execution"] = {
                        "fill": fill, "cases": names, "stage": execution.get("stage", "unknown"),
                        "result": str(Path(execution["output"]) / "result.json"),
                        **{key: execution[key] for key in ("error", "execution_error", "comparison_error", "retrieval_errors", "identity_errors")
                           if execution.get(key)},
                    }
                    result["native_failures"] = []
                    for name in names:
                        relative = f"results/{name}/result.json"
                        native_path = Path(execution["output"]) / relative
                        digest = execution.get("retrieved_sha256", {}).get(relative)
                        if digest and native_path.is_file() and sha256_file(native_path) == digest:
                            failed = _stateful_json(native_path)
                            result["native_failures"].append({"case": name, "status": failed.get("status"),
                                "failure_stage": failed.get("failure_stage"), "reason": failed.get("reason"),
                                "io_diagnostic": failed.get("io_diagnostic"), "path": str(native_path), "sha256": digest})
                    detail = "; ".join(dict.fromkeys(str(execution[key]) for key in
                        ("error", "execution_error", "comparison_error") if execution.get(key)))
                    raise RuntimeError(f"GDN fill=0x{fill:02x} stage={execution.get('stage', 'unknown')}: "
                        f"{detail or 'execution/transport verification failed'}; "
                        f"evidence={result['failed_execution']['result']}")
                identity = tuple(execution[key] for key in ("serial", "ddk_id", "runtime_version", "firmware_id"))
                if native_identity is not None and identity != native_identity:
                    raise ValueError("device/DDK/runtime/firmware changed between diagnostic runs")
                native_identity = identity
                if comparison is None:
                    comparison = _stateful_json(Path(execution["output"]) / "comparison/report.json")
                compared = {case["name"]: case for case in comparison["cases"]}
                for case in cases:
                    name = case["name"]
                    step = _stateful_json(_step_file(path, case["bundle_manifest"]))
                    result_root = Path(execution["output"]) / "results" / name
                    native = _stateful_json(result_root / "result.json")
                    gathered.setdefault(name, {})[fill] = validate_io(native, step, result_root, fill)
                    diagnostics.setdefault(name, []).append({"fill": fill, "io": native["io_diagnostic"],
                        "numerical": compared[name].get("diagnostics", []), "result": str(result_root / "result.json"),
                        "result_sha256": sha256_file(result_root / "result.json")})
                    result["completed_trials"].append({"case": name, **diagnostics[name][-1]})
        for path, manifest, cases in selected:
            if sha256_file(path / "manifest.json") != result["probe_manifests"][str(path)]:
                raise ValueError("source manifest changed during diagnostic")
            for case in cases:
                result["cases"].append({"name": case["name"], "host_input_readback": "MATCHED",
                    "outputs": compare_fills(gathered[case["name"]], case["outputs"]),
                    "runs": diagnostics[case["name"]]})
        result["status"] = "DIAGNOSIS_COMPLETE"
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    save_json(out / "report.json", result)
    lines = ["GDN output buffer diagnosis (existing OMC; no model compilation)",
             "status=" + result["status"], result["scope"], ""]
    if "error" in result:
        lines.append(result["error"])
    if result["status"] != "DIAGNOSIS_COMPLETE" and result["completed_trials"]:
        lines.append("Validated completed trials (three-fill comparison INCOMPLETE):")
        for trial in result["completed_trials"]:
            lines.append(f"  {trial['case']} fill=0x{trial['fill']:02x}: input readback MATCHED; "
                         + json.dumps({"outputs": trial["io"]["outputs"], "numerical": trial["numerical"]}))
    for failed in result.get("native_failures", []):
        lines.append("  Native report: " + json.dumps(failed))
    for case in result["cases"]:
        lines.append(f"{case['name']}: host input readback SHA256 MATCHED for every run")
        for item in case["outputs"]:
            lines.append("  " + json.dumps(item))
        for run_record in case["runs"]:
            lines.append(f"  fill=0x{run_record['fill']:02x}: " + json.dumps(run_record["numerical"]))
    lines += ["", "Prefill-dependent bytes suggest untouched/stale output; this does not locate the fault in kernel, runtime or transfer.",
              "Zero output for all fills does not prove kernel execution; runtime may clear buffers. Identical output does not establish accuracy.",
              "Full input/output evidence: report.json and fill-*/results/*/result.json.",
              "Installed GDN source excerpts: installed/report.txt (full bounded copies in installed/sources/)."]
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--probe-run", default=os.environ.get("FUSED_PROBE_RUN"), help="existing compiled probe directory or fused batch")
    source.add_argument("--resume", help="prior GDN diagnostic directory; revalidate completed trials and run only the missing fills")
    parser.add_argument("--output", help="new directory below AI_RUN_DIR; automatically allocated by default")
    parser.add_argument("--timeout-seconds", type=_positive, default=3600)
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except (OSError, ValueError) as error:
        print(f"gdn-diagnostic: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": result["status"], "output": result["output"]}))
    print((Path(result["output"]) / "report.txt").read_text(), end="")
    return 0 if result["status"] == "DIAGNOSIS_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
