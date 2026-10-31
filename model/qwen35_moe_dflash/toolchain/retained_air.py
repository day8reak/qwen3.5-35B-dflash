"""Recover only a legacy output-path check failure without running OMG again."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path, PurePosixPath

from qwen35_moe_dflash.common.artifacts import output_path, save_json
from qwen35_moe_dflash.toolchain.compile import verify_tbe_compatibility
from qwen35_moe_dflash.toolchain.existing_air import _generated_inventory, _snapshot, _stat, select_primary_omc
from qwen35_moe_dflash.common.hashing import sha256_file


LEGACY_OUTPUT_ERROR = "RuntimeError: OMG succeeded without a nonempty regular model.omc"


def _relative(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("invalid retained artifact path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError("invalid retained artifact path")
    return path


def _regular(root, relative):
    path = Path(root) / _relative(relative)
    for part in (path, *path.parents):
        if part == Path(root).parent:
            break
        if part.is_symlink():
            raise ValueError("retained artifact contains a symlink: " + str(path))
    if not path.is_file():
        raise ValueError("retained artifact is not a regular file: " + str(path))
    return path


def _checked(root, row):
    if not isinstance(row, dict):
        raise ValueError("invalid retained artifact record")
    path = _regular(root, row.get("path"))
    if (type(row.get("bytes")) is not int or row["bytes"] < 0
            or path.stat().st_size != row["bytes"] or sha256_file(path) != row.get("sha256")):
        raise ValueError("retained artifact SHA256/size mismatch: " + str(path))
    return path


def _rows(rows, label):
    if not isinstance(rows, list) or len(rows) > 8192:
        raise ValueError("invalid retained " + label + " inventory")
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("invalid retained " + label + " record")
        key = str(_relative(row.get("path")))
        if key in result:
            raise ValueError("duplicate retained artifact: " + key)
        result[key] = row
    return result


def _read(root, relative):
    path = _regular(root, relative)
    return json.loads(path.read_text())


def recover_retained_air(compilation_dir, output_dir, *, export_dir, expected_options,
                         ddk_id=None, platform="kirinx90", ddk_root=None, progress=None):
    """Snapshot validated compiler outputs after one known legacy path failure.

    Old versions did not record a successful subprocess return code. Their exact
    post-subprocess failure stage, together with the hash-bound terminal success
    log, establishes a normal ``check=True`` return. That inference is recorded
    separately from an explicitly observed return code. No compiler, export,
    subprocess, or board operation is invoked here.
    """
    source = Path(compilation_dir).resolve()
    exported = Path(export_dir).resolve()
    root = output_path(output_dir)
    if root == source or source in root.parents or root in source.parents:
        raise ValueError("recovery output must be separate from retained compilation")
    if root.exists():
        raise FileExistsError("recovery output already exists")
    report = _read(source, "report.json")
    report_hash = sha256_file(source / "report.json")
    help_log = report.get("help_check", {}).get("log")
    if help_log is not None:
        help_path = _regular(source, help_log)
        if sha256_file(help_path) != report["help_check"].get("sha256"):
            raise ValueError("retained help log SHA256 mismatch")
    if (report.get("artifact_kind") != "qwen35-existing-air-compilation"
            or report.get("status") != "FAILED" or report.get("stage") != "output-check"
            or report.get("error") != LEGACY_OUTPUT_ERROR
            or report.get("model_compilation") != "FAILED"):
        raise ValueError("retained compilation is not the legacy output-path check failure")
    if "returncode" in report and (type(report["returncode"]) is not int or report["returncode"] != 0):
        raise ValueError("retained compiler exit was not successful")
    if (platform != "kirinx90" or report.get("platform") != platform
            or not isinstance(report.get("ddk_id"), str) or not report["ddk_id"].strip()
            or (ddk_id is not None and report["ddk_id"] != ddk_id)
            or (ddk_root is not None and Path(report.get("ddk_root", "")).resolve() != Path(ddk_root).resolve())
            or not isinstance(expected_options, dict) or report.get("requested_options") != expected_options):
        raise ValueError("retained compiler DDK/platform/options mismatch")
    if (report.get("audit_error") or report.get("artifact_inventory_error")
            or report.get("skipped_nonregular_artifacts")
            or report.get("source_preservation", {}).get("status") != "UNCHANGED_BY_STAT"
            or report["source_preservation"].get("changed")):
        raise ValueError("retained compilation has failed audit/inventory/source preservation")
    log = _regular(source, "compile.log")
    if sha256_file(log) != report.get("compile_log_sha256"):
        raise ValueError("retained compile log SHA256 mismatch")
    with log.open("rb") as stream:
        stream.seek(max(0, log.stat().st_size - (1 << 20)))
        terminal = stream.read().decode(errors="replace").lower()
    if "omg generate offline model success" not in terminal:
        raise ValueError("retained compiler log has no terminal OMG success")
    compatibility = report.get("tbe_compatibility")
    if compatibility is not None:
        audit = compatibility.get("audit", {})
        metadata = compatibility.get("metadata")
        if (compatibility.get("compiler_invocation") != "ATTEMPTED"
                or audit.get("status") not in {"APPLIED", "NOT_OBSERVED"}
                or not isinstance(metadata, dict)):
            raise ValueError("retained TBE compatibility audit is not successful")
        audit_path = Path(metadata.get("audit_log", ""))
        if not audit_path.is_absolute() or source not in audit_path.parents:
            raise ValueError("retained TBE audit escapes compilation")
        if verify_tbe_compatibility(metadata) != audit:
            raise ValueError("retained TBE compatibility audit changed")

    manifest = _read(exported, "manifest.json")
    export_hash = sha256_file(exported / "manifest.json")
    graphs = manifest.get("graphs", [])
    if (manifest.get("artifact_kind") != "qwen35-cpro-draft-air"
            or manifest.get("status") != "EXPORTED" or len(graphs) != 1
            or graphs[0].get("name") != "draft"):
        raise ValueError("retained recovery requires the matching standalone draft export")
    graph = graphs[0]
    files = _rows(graph.get("files"), "export")
    snapshots = _rows(report.get("input_snapshots"), "input")
    by_source = {}
    for row in snapshots.values():
        relative = str(_relative(row.get("source_relative")))
        if relative in by_source:
            raise ValueError("duplicate retained input source")
        by_source[relative] = row
        if (row["path"] != "compile-work/generated/" + relative
                or Path(row.get("source", "")).resolve() != exported / relative
                or _stat(exported / relative) != row.get("source_stat")
                or _stat(_regular(source, row["path"])) != row.get("snapshot_stat")):
            raise ValueError("retained source or input snapshot changed")
        _checked(source, row)
    if set(files) != set(by_source):
        raise ValueError("retained input inventory differs from draft export")
    for relative, row in files.items():
        _checked(exported, row)
        if any(by_source[relative].get(key) != row[key] for key in ("bytes", "sha256")):
            raise ValueError("retained input differs from draft export")
    air = files.get(graph.get("air"))
    if (air is None or [row["source_relative"] for row in snapshots.values() if row.get("role") == "air"] != [graph["air"]]
            or report.get("air_source") != str(exported / graph["air"])
            or any(report.get("air", {}).get(key) != air[key] for key in ("bytes", "sha256"))):
        raise ValueError("retained AIR identity differs from draft export")
    command = _read(source, "compile-command.json")
    command_hash = sha256_file(source / "compile-command.json")
    expected_command = [report.get("omg_path"), "--framework=1",
                        "--model=" + str(source / by_source[graph["air"]]["path"]),
                        "--output=model", "--target=omc", "--platform=" + platform]
    expected_command.extend(f"--{key}={value}" for key, value in expected_options.items())
    argv = command.get("argv", [])
    if (len(argv) != len(expected_command) or argv[:6] != expected_command[:6]
            or set(argv[6:]) != set(expected_command[6:])
            or command.get("cwd") != str(source / "compile-work/generated")):
        raise ValueError("retained compiler command differs from required profile")

    generated, context, skipped = _generated_inventory(source, report["input_snapshots"])
    if skipped:
        raise ValueError("retained compiler files include nonregular artifacts")
    for actual, key in ((generated, "generated_artifacts"), (context, "compiler_work_artifacts")):
        old, now = _rows(report.get(key), key), _rows(actual, key)
        if set(old) != set(now) or any(any(old[name].get(k) != now[name][k]
                                           for k in ("bytes", "sha256")) for name in old):
            raise ValueError("retained compiler artifact inventory changed: " + key)
    primary = select_primary_omc(source)
    if primary.relative_to(source).as_posix() not in {row["path"] for row in generated}:
        raise ValueError("retained primary OMC is missing from artifact inventory")
    all_rows = [*snapshots.values(), *generated, *context]
    _rows(all_rows, "combined")
    root.mkdir(parents=True)
    copy_methods = {}
    for index, row in enumerate(all_rows):
        if progress is not None and (index == 0 or index % 10 == 0 or index + 1 == len(all_rows)):
            progress({"stage": "retained-recovery", "message": f"Snapshotting retained compiler evidence {index + 1}/{len(all_rows)}: {row['path']}"})
        copied = _snapshot(source / row["path"], root / row["path"])
        copy_methods[row["path"]] = copied["snapshot_method"]
        if any(copied[key] != row[key] for key in ("bytes", "sha256")):
            raise ValueError("retained compiler artifact changed during recovery")
    for name, expected_hash in (("compile.log", report["compile_log_sha256"]),
                                ("compile-command.json", command_hash),
                                ("report.json", report_hash)):
        target = root / ("recovery/original-report.json" if name == "report.json" else name)
        if _snapshot(source / name, target)["sha256"] != expected_hash:
            raise ValueError("retained compiler evidence changed during recovery")
    if help_log is not None and _snapshot(source / help_log, root / help_log)["sha256"] != report["help_check"]["sha256"]:
        raise ValueError("retained help log changed during recovery")
    if (sha256_file(source / "report.json") != report_hash
            or sha256_file(exported / "manifest.json") != export_hash
            or any(_stat(Path(row["source"])) != row["source_stat"] for row in snapshots.values())):
        raise ValueError("retained compiler/export/source evidence changed during recovery")
    recovered = deepcopy(report)
    recovered.pop("error", None)
    recovered.update(status="COMPILED", stage="retained-output-recovery", model_compilation="PASSED",
                     output=str(root), comparison=str(root / "report.txt"), returncode=0,
                     primary_omc_layout=primary.relative_to(source / "compile-work/generated").as_posix(),
                     returncode_evidence="OBSERVED" if "returncode" in report else "INFERRED_FROM_LEGACY_POSTPROCESS_STAGE_AND_SUCCESS_LOG",
                     recovery={"source": str(source), "source_report_sha256": report_hash,
                               "original_error": report["error"], "compiler_execution": "NOT_RUN",
                               "export_manifest_sha256": export_hash})
    for row in recovered["input_snapshots"]:
        row["snapshot_stat"] = _stat(root / row["path"])
        row["snapshot_method"] = copy_methods[row["path"]]
        row["snapshot_source"] = str(source / row["path"])
    generated, context, skipped = _generated_inventory(root, recovered["input_snapshots"])
    recovered.update(generated_artifacts=generated, compiler_work_artifacts=context,
                     external_data_candidates=[row for row in generated if row["role"] != "primary_omc"],
                     omc=next(row for row in generated if row["role"] == "primary_omc"),
                     compiler_diagnostics={"status": "RETAINED_ORIGINAL", "root": str(source),
                                           "record": report.get("compiler_diagnostics")})
    if compatibility is not None:
        recovered["tbe_compatibility"]["compile"] = "PASSED"
    save_json(root / "input-inventory.json", {"files": recovered["input_snapshots"],
                                              "external_weights": recovered.get("external_weights")})
    save_json(root / "report.json", recovered)
    save_json(root / "manifest.json", recovered)
    (root / "report.txt").write_text("Retained OMG output recovery\nstatus=COMPILED\n"
        "No compiler or board execution. Original failure evidence is unchanged.\n"
        f"Source: {source}\nPrimary OMC: {recovered['omc']['path']}\n"
        f"Compiler return: {recovered['returncode_evidence']}\n")
    return recovered
