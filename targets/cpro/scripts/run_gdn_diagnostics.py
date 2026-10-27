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


def run(args):
    storage = Path(_text(os.environ.get("AI_RUN_DIR"), "AI_RUN_DIR"))
    out = output_path(args.output or storage / ("gdn-diagnostic-" + uuid.uuid4().hex[:12]))
    root = Path(_text(args.probe_run, "--probe-run / FUSED_PROBE_RUN")).expanduser().resolve()
    if root == out or root in out.parents or out in root.parents:
        raise ValueError("diagnostic output must be separate from the original batch")
    ddk_root = os.environ.get("CPRO_DDK_ROOT") or os.environ.get("DDK_PATH")
    if ddk_root and (out == Path(ddk_root).resolve() or Path(ddk_root).resolve() in out.parents):
        raise ValueError("diagnostic output must be outside the immutable DDK")
    out.mkdir(parents=True, exist_ok=False)
    result = {"schema_version": 1, "status": "FAILED", "output": str(out), "probe_run": str(root),
              "model_compilation": "NOT_RUN", "numerical_acceptance": "NOT_DEFINED",
              "placement": "UNVERIFIED", "fills": list(FILLS), "runs": [], "cases": [],
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
                execution = board.execute(SimpleNamespace(probe_run=str(path), output=str(out / f"fill-{fill:02x}-{group_index}"),
                    timeout_seconds=args.timeout_seconds, selected_cases=names, output_fill_byte=fill), runner_cache=runner_cache)
                result["runs"].append({"fill": fill, "cases": names, **execution})
                with (out / "progress.jsonl").open("a") as progress:
                    progress.write(json.dumps({"fill": fill, "cases": names, "status": execution["status"],
                                               "output": execution["output"]}) + "\n")
                if execution["status"] != "EXECUTION_COMPLETE":
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
                    raise RuntimeError("GDN execution/transport failed; see retained run result and native io_diagnostic")
                identity = tuple(execution[key] for key in ("serial", "ddk_id", "runtime_version", "firmware_id"))
                if native_identity is not None and identity != native_identity:
                    raise ValueError("device/DDK/runtime/firmware changed between diagnostic runs")
                native_identity = identity
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
    parser.add_argument("--probe-run", default=os.environ.get("FUSED_PROBE_RUN"), help="existing compiled probe directory or fused batch")
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
