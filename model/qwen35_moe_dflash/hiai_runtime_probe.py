"""Compare or replay isolated HIAI deployments using existing probe artifacts.

Models are never recompiled. The recorded runner is reused unless explicitly
refreshed or replaying all cases; backend selection and NPU placement remain
unverified after success.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile

from .artifacts import output_path, save_json
from .elf_dynamic import dynamic_info
from .hashing import sha256_file
from .hiai_deployment import DDK_HCL_EXCLUDED as EXCLUDED, candidate_blockers
from .op_probe_hdc import run_board
from .op_probe_report import write_report


CASE_NAMES = ("matmul_m8_b_dynamic", "add_rank4", "add_rank5")
ARMS = ("ddk_baseline", "candidate_without_ddk_hcl")
_DTYPES = {"float16": 2, "float32": 4, "int32": 4}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--probe-run", required=True, help="existing operator probe directory with compiled models")
    result.add_argument("--native-from", help="existing probe supplying its verified native build and exact libraries for a compile-only batch; requires --replay-all --refresh-runner")
    result.add_argument("--output", required=True, help="new comparison directory below AI_RUN_DIR")
    result.add_argument("--keep-board", action="store_true", help="retain only these comparison deployments")
    result.add_argument("--refresh-runner", action="store_true",
                        help="build only the diagnostic runner from recorded SDK/header/link inputs")
    result.add_argument("--replay-all", action="store_true",
                        help="replay every compiled case in the candidate deployment only; automatically refresh the runner once, never compile models")
    result.add_argument("--timeout-seconds", type=int, default=300)
    return result


def _digest(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("expected a lowercase SHA256 digest in the existing probe records")
    return value


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_+.-]+", value) or value in (".", ".."):
        raise ValueError(f"unsafe artifact basename: {value!r}")
    return value


class Snapshot:
    """Bind source paths and contents; detect mutation before and after each arm."""
    def __init__(self):
        self.files = {}

    def add(self, path, *, expected=None, size=None, role="artifact"):
        path = Path(path).absolute()
        if not path.is_file():
            raise ValueError(f"required source file is missing: {path}")
        record = {"path": str(path), "resolved_path": str(path.resolve()),
                  "sha256": sha256_file(path), "bytes": path.stat().st_size, "role": role}
        if expected is not None and record["sha256"] != _digest(expected):
            raise ValueError(f"source SHA256 mismatch: {path}")
        if size is not None and (type(size) is not int or record["bytes"] != size):
            raise ValueError(f"source size mismatch: {path}")
        previous = self.files.get(str(path))
        if previous and any(previous[k] != record[k] for k in ("resolved_path", "sha256", "bytes")):
            raise ValueError(f"source changed while validating: {path}")
        self.files.setdefault(str(path), record)
        return path

    def verify(self):
        for record in self.files.values():
            path = Path(record["path"])
            if (not path.is_file() or str(path.resolve()) != record["resolved_path"]
                    or path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]):
                raise ValueError(f"snapshotted source changed: {path}")


def _json(path, snapshot):
    snapshot.add(path, role="source-metadata")
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _case_file(folder, item, snapshot):
    if not isinstance(item, dict) or not isinstance(item.get("path"), str):
        raise ValueError("case payload record requires path, bytes and SHA256")
    if type(item.get("bytes")) is not int or item["bytes"] < 0:
        raise ValueError("case payload requires a nonnegative integer byte size")
    relative = Path(item["path"])
    path = folder / relative
    if (relative.is_absolute() or ".." in relative.parts or path.is_symlink()
            or folder not in path.resolve().parents):
        raise ValueError(f"case payload escapes its original directory: {path}")
    return snapshot.add(path, expected=_digest(item["sha256"]), size=item["bytes"], role="case-payload")


def _tensor_bytes(tensor):
    shape = tensor.get("shape")
    if (tensor.get("dtype") not in _DTYPES or not isinstance(shape, list) or not shape
            or any(type(value) is not int or value <= 0 for value in shape)):
        raise ValueError("unsupported or invalid tensor contract in existing probe")
    size = math.prod(shape) * _DTYPES[tensor["dtype"]]
    if size > 32 << 20:
        raise ValueError("probe tensor exceeds the existing 32 MiB comparison limit")
    return size


def _validate_case(record, source, snapshot):
    name = record["name"]
    folder = source / "cases" / name
    if Path(record.get("directory", "")).resolve() != folder or not folder.is_dir():
        raise ValueError(f"case directory does not match --probe-run: {name}")
    compilation = record.get("compile", {})
    if compilation.get("status") != "COMPILED":
        raise ValueError(f"required case was not compiled: {name}")
    model = _case_file(folder, compilation["model"], snapshot)
    if (model != folder / "model.omc" or Path(record["model"]).resolve() != model
            or not 0 < model.stat().st_size <= 64 << 20):
        raise ValueError(f"invalid compiled probe model: {name}")
    io = folder / "io.tsv"
    if Path(record["io"]).resolve() != io:
        raise ValueError(f"IO path does not match case: {name}")
    snapshot.add(io, expected=_digest(record["io_sha256"]), role="case-io")
    snapshot.add(folder / "graph.cpp", expected=_digest(record["source_sha256"]), role="case-source")
    construction = record.get("construction_files")
    if not isinstance(construction, list) or not construction:
        raise ValueError(f"case requires original construction hashes: {name}")
    for item in construction:
        _case_file(folder, item, snapshot)
    inputs, expected_rows = [], []
    for tensor in record["input_tensors"]:
        path = _case_file(folder, tensor, snapshot)
        _name(path.name)
        if path.parent != folder or tensor["bytes"] != _tensor_bytes(tensor):
            raise ValueError(f"input tensor byte contract mismatch: {name}")
        inputs.append(str(path))
        expected_rows.append(["input", tensor["dtype"], ",".join(map(str, tensor["shape"])), path.name])
    if record["inputs"] != inputs or not inputs:
        raise ValueError(f"input list differs from hashed tensor contract: {name}")
    outputs = record.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ValueError(f"case needs reference outputs: {name}")
    for tensor in outputs:
        reference = _case_file(folder, tensor["reference"], snapshot)
        if (reference.parent != folder or tensor["expected"] != reference.name
                or tensor["reference"]["bytes"] != _tensor_bytes(tensor)):
            raise ValueError(f"reference tensor contract mismatch: {name}")
        output_name = _name(tensor["path"])
        for field in ("rtol", "atol"):
            value = tensor.get(field)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid numerical tolerance in original case: {name}")
        expected_rows.append(["output", tensor["dtype"], ",".join(map(str, tensor["shape"])), output_name])
    rows = [line.split("\t") for line in io.read_text().splitlines() if line.strip()]
    if rows != expected_rows:
        raise ValueError(f"IO TSV differs from hashed tensor metadata: {name}")


def _validate_native(source, manifest, snapshot):
    """Validate one original native inventory; never merge runtime sources."""
    command = _json(source / "native/command.json", snapshot)
    result = _json(source / "native/result.json", snapshot)
    if (result.get("status") != "LINKED"
            or _digest(result.get("command_sha256")) != sha256_file(source / "native/command.json")):
        raise ValueError("native command SHA256 does not match its LINKED result")
    if manifest.get("native") != {"build": command, "result": result}:
        raise ValueError("manifest native records differ from the recorded build")
    runner = snapshot.add(source / "native/build/qwen35_hiai_op_probe",
                          expected=_digest(result.get("binary_sha256")), role="native-runner")
    # Historical source hashes bind the old binary's provenance. A newer checkout
    # is allowed; do not silently rebuild it or compare it with that old source.
    for entry in command.get("sources", []):
        _digest(entry["sha256"])
        if not Path(entry["path"]).is_absolute():
            raise ValueError("historical native source paths must be absolute")
    recorded = command.get("library_discovery", {}).get("libraries")
    if not isinstance(recorded, list) or not recorded:
        raise ValueError("native command needs its exact library_discovery inventory")
    libraries, names = [], set()
    for entry in recorded:
        name = _name(entry["name"])
        path = Path(entry["path"])
        if not path.is_absolute() or path.name != name or name in names:
            raise ValueError("recorded library paths/basenames must be absolute and unique")
        names.add(name)
        libraries.append(snapshot.add(path, expected=_digest(entry["sha256"]), role="runtime-library"))
    linked = command.get("libraries")
    if not isinstance(linked, list) or not linked:
        raise ValueError("native command has no recorded link inputs")
    inventory = {entry["name"]: entry for entry in recorded}
    for entry in linked:
        if not Path(entry["path"]).is_absolute():
            raise ValueError("recorded link-input path must be absolute")
        path = snapshot.add(entry["path"], expected=_digest(entry["sha256"]), role="link-input")
        deployed = inventory.get(path.name)
        if not deployed or deployed["sha256"] != entry["sha256"]:
            raise ValueError("recorded link input is absent from the original runtime deployment")
    return command, result, runner, libraries


def _ddk_identity(manifest):
    """Require explicit compiler identity when joining independently made runs."""
    tools = manifest.get("tools")
    if not isinstance(tools, dict):
        raise ValueError("native donor reuse requires both probes' recorded DDK identity")
    root, version = tools.get("ddk_root"), tools.get("ddk_id")
    if (not isinstance(root, str) or not Path(root).is_absolute()
            or any(ord(c) < 32 or ord(c) == 127 for c in root)
            or not isinstance(version, str) or not version.strip()
            or version.lower() in {"unknown", "none"}):
        raise ValueError("native donor reuse requires an absolute DDK root and explicit DDK ID")
    return {"ddk_root": str(Path(root).resolve()), "ddk_id": version,
            "omg_sha256": _digest(tools.get("omg_sha256"))}


def validate_source(source, *, replay_all=False, native_from=None):
    """Validate original model/native records without rediscovery or vendor tools.

    An explicit donor can supply native records only for a genuine compile-only
    run. It cannot replace an original native inventory, including broken ones.
    """
    source = Path(source).expanduser().resolve(strict=True)
    snapshot = Snapshot()
    manifest = _json(source / "manifest.json", snapshot)
    if manifest.get("artifact_kind") != "qwen35-cpro-op-probe":
        raise ValueError("--probe-run is not an operator probe manifest")
    native_source, native_manifest, ddk_identity = source, manifest, None
    if native_from is not None:
        if not replay_all:
            raise ValueError("--native-from requires --replay-all")
        if ("native" in manifest or (source / "native").exists() or (source / "native").is_symlink()):
            raise ValueError("--native-from cannot replace existing or incomplete native records")
        if manifest.get("status") != "COMPILE_COMPLETE" or manifest.get("board_requested") is not False:
            raise ValueError("--native-from requires an original COMPILE_COMPLETE compile-only probe")
        native_source = Path(native_from).expanduser().resolve(strict=True)
        native_manifest = _json(native_source / "manifest.json", snapshot)
        if native_manifest.get("artifact_kind") != "qwen35-cpro-op-probe":
            raise ValueError("--native-from is not an operator probe manifest")
        ddk_identity = _ddk_identity(manifest)
        if ddk_identity != _ddk_identity(native_manifest):
            raise ValueError("compile-only probe and native donor have different recorded DDK identities")
    command, result, runner, libraries = _validate_native(native_source, native_manifest, snapshot)
    if native_from is not None:
        ddk_root = Path(ddk_identity["ddk_root"])
        for entry in command["libraries"]:
            if ddk_root not in Path(entry["path"]).resolve().parents:
                raise ValueError("native donor link input is outside the matched compiler DDK: " + entry["path"])
    all_cases = manifest.get("cases")
    if not isinstance(all_cases, list):
        raise ValueError("existing probe must contain a case list")
    by_name = {}
    for record in all_cases:
        if not isinstance(record, dict):
            raise ValueError("each original probe case must be a JSON object")
        name = record.get("name")
        if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", name)
                or name in {".", "..", "calls", "_evidence", "deployment.json", "failure.json",
                            "result.json", "payloads.sha256", ".qwen35-hdc-owner"}):
            raise ValueError(f"unsafe original probe case name: {name!r}")
        if name in by_name:
            raise ValueError(f"duplicate original probe case: {name}")
        if not isinstance(record.get("compile", {}), dict):
            raise ValueError(f"invalid original compile metadata: {name}")
        if record.get("compile", {}).get("status", "NOT_RUN") not in {"COMPILED", "FAILED", "NOT_RUN"}:
            raise ValueError(f"invalid original compile status: {name}")
        by_name[name] = record
    skipped = []
    if replay_all:
        records = [record for record in all_cases if record.get("compile", {}).get("status") == "COMPILED"]
        skipped = [record for record in all_cases if record.get("compile", {}).get("status") != "COMPILED"]
        if not records:
            raise ValueError("--replay-all requires at least one existing COMPILED case")
    else:
        missing = set(CASE_NAMES) - set(by_name)
        if missing:
            raise ValueError("missing required compiled cases: " + ", ".join(sorted(missing)))
        records = [by_name[name] for name in CASE_NAMES]
    for record in records:
        _validate_case(record, source, snapshot)
    dependencies = [{"path": str(path), "name": path.name, "sha256": sha256_file(path),
                     "kind": "runner" if path == runner else "library", **dynamic_info(path)}
                    for path in [runner, *libraries]]
    snapshot.verify()
    return {"source": source, "manifest": manifest, "command": command, "result": result,
            "native_source": native_source, "native_manifest": native_manifest,
            "native_from": native_from is not None, "ddk_identity": ddk_identity,
            "runner": runner, "libraries": libraries, "records": records, "skipped": skipped,
            "snapshot": snapshot, "dependencies": dependencies}


def _freeze(validated, root):
    """Copy only previously verified files; each arm uses the same local bytes."""
    snapshot, frozen = validated["snapshot"], Snapshot()
    snapshot.verify()

    def copy_file(source, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        source_record = snapshot.files[str(Path(source).absolute())]
        frozen.add(destination, expected=source_record["sha256"], size=source_record["bytes"], role="frozen-payload")
        return destination

    runner = copy_file(validated["runner"], root / "reused/bin/qwen35_hiai_op_probe")
    libraries = [copy_file(path, root / "reused/lib" / path.name) for path in validated["libraries"]]
    records = copy.deepcopy(validated["records"])
    for record in records:
        original = Path(record["directory"])
        folder = root / "reused/cases" / record["name"]
        for item in snapshot.files.values():
            path = Path(item["path"])
            if original in path.parents:
                copy_file(path, folder / path.relative_to(original))
        record.update(original_directory=str(original), directory=str(folder),
                      model=str(folder / "model.omc"), io=str(folder / "io.tsv"),
                      inputs=[str(folder / Path(path).name) for path in record["inputs"]])
    copy_file(validated["source"] / "manifest.json", root / "source-records/manifest.json")
    native_records = root / ("native-source-records" if validated["native_from"] else "source-records")
    for name in ("manifest.json", "native/command.json", "native/result.json"):
        if name != "manifest.json" or validated["native_from"]:
            copy_file(validated["native_source"] / name, native_records / name)
    snapshot.verify()
    return runner, libraries, records, frozen


def _combined_report(root, comparison):
    save_json(root / "runtime-comparison.json", comparison)
    reuse = ("Replays existing compiled cases in the candidate deployment only; refreshes the diagnostic runner once. No model compilation."
             if comparison["replay_all"] else
             "Reuses existing models, inputs and references; builds only the diagnostic runner once for both arms."
             if comparison["refresh_runner"] else
             "Reuses existing models, inputs, references and native binary; no compilation or build.")
    lines = ["HIAI runtime deployment comparison", "", "Status: " + comparison["status"], reuse,
             "Backend selection and NPU execution remain UNVERIFIED.",
             f"Cases: {comparison['counts']['compiled']} compiled, {comparison['counts']['skipped']} skipped without retry.", "",
             "arm | case | compile | compiler partition (not runtime) | load | run | numerical | placement"]
    if comparison.get("native_from"):
        lines[4:4] = ["Native build/library donor: " + comparison["native_from"],
                      "The compiled-model source and native donor are separately hash-bound; recorded DDK identities match."]
    for arm in comparison["arms"]:
        lines.append(f"{arm['name']}: {arm['status']}")
        for reason in arm.get("blocked_reasons", []):
            lines.append("  blocked: " + reason)
        if arm.get("error"):
            lines.append("  error: " + arm["error"])
        for case in arm.get("cases", []):
            partition = case.get("compiler_partition")
            partition_text = (",".join(f"{key}={partition['counts'][key]}" for key in ("NPU", "CPU", "GPU", "ISP"))
                              + f" [{partition['scope']}]" if partition else "UNAVAILABLE")
            lines.append(" | ".join((arm["name"], case["name"], case["compile"], partition_text, case["load"], case["run"],
                                     case["numerical"], case["placement"])))
            if case.get("retry") == "NOT_RETRIED":
                lines.append("  not retried: original case was not COMPILED; no model compilation requested.")
            if case.get("error"):
                lines.append("  runtime: " + str(case["error"]))
    lines.extend(["", *comparison["limitations"]])
    with (root / "runtime-comparison.txt").open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    # Explicit allowlist: nested arm archives are made by write_report, which
    # excludes models, libraries, tensors and the shared raw HiLog snapshot.
    members = [root / name for name in ("runtime-comparison.json", "runtime-comparison.txt", "source-snapshot.json")]
    if comparison.get("native_from"):
        for directory in ("source-records", "native-source-records"):
            members += [root / directory / name for name in ("manifest.json", "native/command.json", "native/result.json")]
    members += [root / "arms" / arm["name"] / "results.tar.gz" for arm in comparison["arms"]]
    if comparison["refresh_runner"]:
        members += [root / "refreshed-native" / name for name in ("command.json", "build-command.json", "result.json")]
    with tarfile.open(root / "results.tar.gz", "x:gz") as archive:
        for path in members:
            if path.is_file() and not path.is_symlink():
                archive.add(path, arcname=str(path.relative_to(root)), recursive=False)


def _partition_summary(compilation):
    """Keep historical main/unknown partitions separate from execution evidence."""
    evidence = compilation.get("evidence") or {}
    if not isinstance(evidence, dict):
        return None
    def scoped_events(key, scopes):
        values = evidence.get(key, [])
        return [event for event in values if isinstance(event, dict) and event.get("scope") in scopes] if isinstance(values, list) else []
    events = (scoped_events("main_partition_events", {"main"}) or
              scoped_events("partition_events", {"main"}) or scoped_events("partition_events", {"unknown"}))
    if not events:
        return None
    event = events[-1]
    counts = event.get("counts", {})
    if not isinstance(counts, dict) or any(type(counts.get(key)) is not int or counts[key] < 0 for key in ("NPU", "CPU", "GPU", "ISP")):
        return None
    return {"counts": {key: counts[key] for key in ("NPU", "CPU", "GPU", "ISP")},
            "scope": event.get("scope", "unknown"), "line": event.get("line")}


def _case_summary(record, runtime=None, numerical="NOT_RUN"):
    compilation = record.get("compile", {})
    runtime = runtime or {}
    status = compilation.get("status", "NOT_RUN")
    return {"name": record["name"], "compile": status, "compiler_partition": _partition_summary(compilation),
            "retry": "NOT_RETRIED" if status != "COMPILED" else None,
            "load": runtime.get("load", "NOT_RUN"), "run": runtime.get("run", "NOT_RUN"),
            "numerical": numerical, "placement": "UNVERIFIED",
            "runtime_version": runtime.get("runtime_version"), "error": runtime.get("error"),
            "diagnostics": runtime.get("diagnostics", [])}


def run(args):
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive")
    if args.native_from is not None and not (args.replay_all and args.refresh_runner):
        raise ValueError("--native-from requires both --replay-all and --refresh-runner")
    root = output_path(args.output)
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"comparison output already exists: {root}")
    validated = validate_source(args.probe_run, replay_all=args.replay_all, native_from=args.native_from)
    source = validated["source"]
    for protected in {source, validated["native_source"]}:
        if root == protected or protected in root.parents or root in protected.parents:
            raise ValueError("comparison output must be separate from the original model and native probe directories")
    root.mkdir(parents=True)
    try:
        runner, libraries, records, frozen = _freeze(validated, root)
        runner_record = {"mode": "REUSED_NATIVE", "path": str(runner), "sha256": sha256_file(runner),
                         "historical_binary": {"path": str(validated["runner"]), "sha256": sha256_file(validated["runner"])}}
        dependencies = validated["dependencies"]
        refresh_runner = args.refresh_runner or args.replay_all
        if refresh_runner:
            from .hiai_probe_build import build_probe
            runner, runner_record = build_probe(root, validated, libraries, frozen, args.timeout_seconds)
            dependencies = [{"path": str(runner), "name": runner.name, "sha256": runner_record["sha256"],
                             "kind": "runner", **runner_record["dependencies"]},
                            *[entry for entry in dependencies if entry["kind"] == "library"]]
    except Exception as error:
        save_json(root / "failure.json", {"status": "PREFLIGHT_FAILED", "board_started": False,
                  "error": f"{type(error).__name__}: {error}"})
        raise
    save_json(root / "source-snapshot.json", {
        "schema_version": 1, "probe_run": str(source), "files": list(validated["snapshot"].files.values()),
        "source_manifest_sha256": sha256_file(source / "manifest.json"),
        "native_source_probe_run": str(validated["native_source"]),
        "native_source_manifest_sha256": sha256_file(validated["native_source"] / "manifest.json"),
        "native_source_origin": "explicit-donor" if validated["native_from"] else "original-probe",
        "matched_ddk_identity": validated["ddk_identity"],
        "frozen_files": list(frozen.files.values()), "historical_native_sources": validated["command"].get("sources", []),
        "dependencies": dependencies, "historical_dependencies": validated["dependencies"], "runner": runner_record,
        "scope": "Hashes bind reused artifacts and, when explicitly requested, the refreshed native build; backend selection is unverified."})
    blockers = candidate_blockers(dependencies)
    frozen_by_name = {record["name"]: record for record in records}
    report_records = [frozen_by_name.get(record["name"], copy.deepcopy(record))
                      for record in validated["manifest"]["cases"]] if args.replay_all else records
    comparison = {"schema_version": 1, "artifact_kind": "qwen35-hiai-runtime-comparison",
        "created_utc": datetime.now(timezone.utc).isoformat(), "probe_run": str(source),
        "source_manifest_sha256": sha256_file(source / "manifest.json"), "status": "RUNNING",
        "native_from": str(validated["native_source"]) if validated["native_from"] else None,
        "native_source_manifest_sha256": sha256_file(validated["native_source"] / "manifest.json"),
        "matched_ddk_identity": validated["ddk_identity"],
        "refresh_runner": refresh_runner, "replay_all": args.replay_all, "runner": runner_record,
        "counts": {"cases": len(report_records), "compiled": len(records), "skipped": len(validated["skipped"])},
        "selected_cases": [record["name"] for record in records],
        "skipped_cases": [record["name"] for record in validated["skipped"]], "excluded_library_names": sorted(EXCLUDED),
        "backend_selection": "UNVERIFIED", "runtime_placement": "UNVERIFIED", "arms": [],
        "limitations": [
            "The candidate changes only its isolated uploaded library set and adds /vendor/lib64/chipsetsdk to its process search path.",
            "Passing Load, Process or numerical comparison does not identify the loaded backend or prove NPU execution.",
            "DT_NEEDED checks cover the inspected files, not dlopen dependencies, firmware ABI compatibility or system namespaces.",
            "Historical compiler partitions do not establish runtime execution placement.",
            "Existing compiled cases are replayed without retrying model compilation; full-model accuracy and performance are not validated."
            if args.replay_all else "Only these three previous case contracts are compared; full-model accuracy and performance are not validated."]}
    for name in (ARMS[1:] if args.replay_all else ARMS):
        comparison["arms"].append({"name": name, "status": "NOT_RUN", "backend_selection": "UNVERIFIED",
                                    "placement": "UNVERIFIED", "cases": [_case_summary(record) for record in report_records]})
    failed = False
    try:
        for arm in comparison["arms"]:
            if arm["name"] == ARMS[1] and blockers:
                arm.update(status="BLOCKED", blocked_reasons=blockers)
                continue
            validated["snapshot"].verify()
            frozen.verify()
            arm_root = root / "arms" / arm["name"]
            arm_root.mkdir(parents=True)
            candidate = arm["name"] == ARMS[1]
            selected = [path for path in libraries if not candidate or path.name not in EXCLUDED]
            search_paths = ("/vendor/lib64/chipsetsdk",) if candidate else ()
            arm.update(libraries=[{"name": path.name, "sha256": sha256_file(path)} for path in selected],
                       excluded_library_names=sorted(EXCLUDED) if candidate else [],
                       library_search_paths=list(search_paths))
            manifest = {"schema_version": 1, "artifact_kind": "qwen35-cpro-op-probe", "status": "RUNNING",
                        "cases": report_records, "comparison_arm": arm["name"], "source_probe_run": str(source),
                        "skipped_cases": comparison["skipped_cases"], "replay_all": args.replay_all}
            board = None
            try:
                print(f"runtime-probe: {arm['name']}: running {len(records)} existing cases", flush=True)
                board = run_board(arm_root, runner, selected, records,
                                  hdc_executable=os.environ.get("HDC", "hdc"), timeout=args.timeout_seconds,
                                  keep_board=args.keep_board, library_search_paths=search_paths)
                validated["snapshot"].verify()
                frozen.verify()
                manifest["status"] = "COLLECTED"
                arm["status"] = "COMPLETE"
            except Exception as error:
                failed = True
                arm.update(status="INFRASTRUCTURE_FAILED", error=f"{type(error).__name__}: {error}")
                manifest.update(status="INFRASTRUCTURE_FAILED", error=arm["error"])
            finally:
                save_json(arm_root / "manifest.json", manifest)
                report = write_report(arm_root, manifest, board)
                arm["cases"] = [_case_summary(row, row["runtime"], row["numerical"]["status"]) for row in report["cases"]]
                arm["board_transport_status"] = report["board_transport_status"]
                arm["board_transport"] = report["board_transport"]
                arm["report"] = str((arm_root / "report.json").relative_to(root))
            if failed:
                break
    except Exception as error:
        failed = True
        comparison["error"] = f"{type(error).__name__}: {error}"
    finally:
        comparison["status"] = ("INFRASTRUCTURE_FAILED" if failed else
                                "CANDIDATE_BLOCKED" if blockers else "REPLAY_COMPLETE" if args.replay_all else "COMPARISON_COMPLETE")
        _combined_report(root, comparison)
        print(f"Runtime comparison: {root / 'runtime-comparison.txt'}", flush=True)
        print(f"Share results: {root / 'results.tar.gz'}", flush=True)
    return 1 if failed or blockers else 0


def main(argv=None):
    try:
        return run(parser().parse_args(argv))
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(f"probe-hdc-runtime: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
