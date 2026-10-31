"""An explicit decode1 input view over one completed prefill64 job.

Only the small transferred AIR is copied. Each weight reference is an owned
file symlink to a hash-bound, regular read-only compiler input snapshot. No
source link count, permissions or receipt is changed. This is host input
identity evidence, not independent storage or a model/device validation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import stat

from qwen35_moe_dflash.toolchain.existing_air_profile import get_existing_air_profile
from qwen35_moe_dflash.common.file_io import MAX_AIR_BYTES, MAX_METADATA_BYTES, _relative, _safe
from qwen35_moe_dflash.target.source_inventory import _inventory, _read_small, _unique_object, _reject_constant
from qwen35_moe_dflash.target.snapshot_retry import _identity, _five, _hash_file, _open_directory


REFERENCE_METHOD = "READONLY_SOURCE_REFERENCE_SYMLINK"
_KIND = "qwen35-decode-input-reference-view"
_GENERATED = Path("compile-work/generated")
_ORIGIN_SCHEMA = "a7cd334beff74bfeb637b1886329a70122842667a257d84ba111c70991695679"
_ORIGIN_DATA_DTYPES = {"arg0_1": 1, "arg1_1": 3, "arg2_1": 7, "arg14_1": 7,
    "arg18_1": 5, "arg107_1": 7, "arg113_1": 1, "arg114_1": 8, "arg1238_1": 7}


def _json(path, bindings):
    raw, identity = _read_small(Path(path), MAX_METADATA_BYTES)
    value = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError("decode input metadata must be an object")
    bindings.append(identity)
    return value, identity


def _small(path, bindings, maximum=MAX_AIR_BYTES):
    _, identity = _read_small(Path(path), maximum)
    bindings.append(identity)
    return identity


def _options(name):
    profile = get_existing_air_profile(name)
    return {"input_shape": profile["input_shape"], "limit_partition_size": "false",
            "save_weights_as_external_data": "true"}


def _same_content(expected, observed):
    return all(expected.get(key) == observed[key] for key in ("bytes", "sha256"))


def _transfer(air, inventory, export_receipt, bindings):
    """Read only the three explicit small files, never origin payload paths."""
    receipt, receipt_id = _json(export_receipt, bindings)
    raw, inventory_id = _read_small(inventory, MAX_METADATA_BYTES)
    bindings.append(inventory_id)
    manifest, rows, prefix = _inventory(raw, sequence_length=1)
    air_id = _small(air, bindings)
    if (receipt.get("artifact_kind") != "qwen35-canonical-target-export"
            or receipt.get("status") != "EXPORT_READY"
            or not _same_content(receipt.get("air", {}), air_id)
            or not _same_content(manifest["air"], air_id)
            or receipt.get("inventory", {}).get("sha256") != inventory_id["sha256"]
            or ("bytes" in receipt.get("inventory", {})
                and receipt["inventory"]["bytes"] != inventory_id["bytes"])):
        raise ValueError("decode1 transfer AIR/inventory is not bound to a completed canonical export")
    schema = receipt.get("schema", {})
    normalization = receipt.get("normalization", {})
    if (normalization.get("status") != "OUTPUT_ORDER_CANONICALIZED"
            or type(normalization.get("sequence_length")) is not int
            or normalization["sequence_length"] != 1
            or normalization.get("patched_sha256") != air_id["sha256"]
            or any(schema.get(key) != manifest["schema"].get(key)
                   or normalization.get("schema", {}).get(key) != schema.get(key)
                   for key in ("descriptor_sha256", "schema_package", "schema_file"))):
        raise ValueError("decode1 normalization/schema receipt differs from the transferred files")
    observed = normalization.get("input_validation", {})
    inputs = observed.get("inputs")
    profile = get_existing_air_profile("cpro-40block-decode-1")
    expected = {name: [int(dim) for dim in shape.split(",")] for name, shape in
                (item.split(":", 1) for item in profile["input_shape"].split(";"))}
    if (observed.get("status") != "MATCHED_TYPED_INPUT_GEOMETRIES"
            or not isinstance(inputs, list) or len(inputs) != 89
            or any(not isinstance(row, dict) for row in inputs)
            or {row.get("name") for row in inputs} != set(expected)):
        raise ValueError("decode1 canonical receipt requires exact typed 89-input profile names")
    for row in inputs:
        name, shape = row.get("name"), row.get("shape")
        if (row.get("type") != ("RefData" if name.startswith("RefData_") else "Data")
                or not isinstance(shape, list) or any(type(dim) is not int for dim in shape)
                or shape != expected[name] or row.get("layout") != "ND"
                or type(row.get("raw_dtype")) is not int or row["raw_dtype"] < 0):
            raise ValueError("decode1 canonical typed input geometry/layout differs: " + str(name))
        # Numeric enum values are schema-specific. This hash is the exact
        # captured 3378-byte origin schema, not the different installed DDK
        # schema. Other schemas retain their declared raw values as evidence.
        if schema["descriptor_sha256"] == _ORIGIN_SCHEMA:
            dtype = 2 if name.startswith("RefData_") else _ORIGIN_DATA_DTYPES[name]
            if row["raw_dtype"] != dtype:
                raise ValueError("decode1 known origin-schema input dtype differs: " + name)
    return manifest, rows, prefix, {"air": air_id, "inventory": inventory_id,
        "export_receipt": receipt_id, "schema_descriptor_sha256": schema["descriptor_sha256"],
        "profile": profile["name"], "status": "MATCHED_EXACT_DATA_NAMES_AND_DECLARED_GEOMETRIES",
        "data_dtype_validation": ("MATCHED_KNOWN_ORIGIN_SCHEMA_RAW_DTYPES"
            if schema["descriptor_sha256"] == _ORIGIN_SCHEMA else "DATA_DTYPE_SEMANTICS_UNVERIFIED"),
        "scope": "Receipt-declared typed geometries only; no ordinary control identity, source-to-AIR or runtime-slot proof"}


def _bound_source(job, *, ddk, ddk_id, omg, omg_sha256, bindings):
    """Bind a fixed successful prefill job, not a recursively borrowed view."""
    compilation = job / "compilation"
    outer, outer_id = _json(job / "result.json", bindings)
    report, report_id = _json(compilation / "report.json", bindings)
    profile, _ = _json(job / "compile-profile.json", bindings)
    inventory, _ = _json(compilation / "input-inventory.json", bindings)
    command, _ = _json(compilation / "compile-command.json", bindings)
    old_air, _ = _safe(report.get("air_source"))
    if (outer.get("artifact_kind") != "qwen35-original-target-job"
            or outer.get("status") != "COMPILED" or outer.get("model_compilation") != "PASSED"
            or outer.get("output") != str(job) or outer.get("air") != str(old_air)
            or outer.get("compilation") != "compilation/report.json"
            or outer.get("compilation_report_sha256") != report_id["sha256"]
            or outer.get("input_layout") != "air-directory"
            or report.get("artifact_kind") != "qwen35-existing-air-compilation"
            or report.get("status") != "COMPILED" or report.get("model_compilation") != "PASSED"
            or type(report.get("returncode")) is not int or report["returncode"] != 0
            or report.get("output") != str(compilation)
            or profile != get_existing_air_profile("cpro-40block-prefill-64")
            or report.get("requested_options") != _options("cpro-40block-prefill-64")
            or report.get("ddk_root") != str(ddk) or report.get("ddk_id") != ddk_id
            or report.get("omg_path") != str(omg) or report.get("omg_sha256") != omg_sha256
            or report.get("platform") != "kirinx90" or ddk not in omg.parents
            or report.get("decode_input_view") is not None):
        raise ValueError("decode borrowing requires this exact completed static prefill64 job/DDK")
    if (report.get("source_preservation", {}).get("status") != "UNCHANGED_BY_STAT"
            or report["source_preservation"].get("changed") != []
            or any(report.get(key) for key in ("audit_error", "artifact_inventory_error",
                    "retained_input_audit_error", "skipped_nonregular_artifacts"))):
        raise ValueError("completed prefill source/audit evidence did not pass")
    log_id = _small(compilation / "compile.log", bindings, 64 << 20)
    if report.get("compile_log_sha256") != log_id["sha256"] or report.get("compile_log") != "compile.log":
        raise ValueError("completed prefill compiler log SHA256 differs from its successful report")
    retained = report.get("retained_input_reuse")
    if retained is not None:
        saved, _ = _json(compilation / "retained-input-reuse.json", bindings)
        if (saved != retained or retained.get("status") != "PASSED"
                or retained.get("artifact_kind") != "qwen35-static-prefill-snapshot-reuse"
                or retained.get("output") != str(compilation)
                or retained.get("audit_errors") != []
                or retained.get("explicit_weight_copy_bytes") != 0):
            raise ValueError("completed prefill retained-input audit is not exact and passed")
    compatibility = report.get("tbe_compatibility")
    if compatibility is not None:
        saved, _ = _json(compilation / "tbe-compatibility-result.json", bindings)
        if (saved != compatibility or compatibility.get("compiler_invocation") != "ATTEMPTED"
                or compatibility.get("compile") != "PASSED"
                or compatibility.get("audit", {}).get("status") not in {"APPLIED", "NOT_OBSERVED"}):
            raise ValueError("completed prefill compatibility audit did not pass")
    snapshots = report.get("input_snapshots")
    if (not isinstance(snapshots, list) or not 2 <= len(snapshots) <= 4096
            or inventory.get("files") != snapshots):
        raise ValueError("completed prefill requires exact complete input snapshot inventories")
    layout = report.get("input_layout", {})
    if (layout.get("mode") != "air-directory" or layout.get("sidecar_prefix") != "."
            or layout.get("compiler_cwd") != _GENERATED.as_posix()
            or layout.get("air_path") != (_GENERATED / old_air.name).as_posix()):
        raise ValueError("completed prefill input layout differs from air-directory")
    argv = [str(omg), "--framework=1", "--model=" + str(compilation / layout["air_path"]),
            "--output=model", "--target=omc", "--platform=kirinx90",
            *("--" + key + "=" + value for key, value in _options("cpro-40block-prefill-64").items())]
    actual = command.get("argv")
    if (not isinstance(actual, list) or actual[:6] != argv[:6] or len(actual) != len(argv)
            or sorted(actual[6:]) != sorted(argv[6:])
            or command.get("cwd") != str(compilation / _GENERATED)
            or command.get("weight_dtype_override") is not None):
        raise ValueError("completed prefill compiler argv/cwd is not the frozen static invocation")
    by_name, protected, seen_paths, seen_sources, inodes, airs = {}, {}, set(), set(), set(), []
    attempts = retained.get("snapshot_attempts", []) if retained else []
    observations = {row.get("destination"): row for row in attempts if isinstance(row, dict)}
    for row in snapshots:
        if not isinstance(row, dict) or row.get("snapshot_method") not in {
                "COPY", "REFLINK", "SHARED_READONLY_SNAPSHOT_INODE"}:
            raise ValueError("prefill inputs must be original regular snapshots, never borrowed views")
        source, _ = _safe(row.get("source"))
        snapshot, _ = _safe(_relative(compilation, row.get("path")))
        source_id, snapshot_id = _identity(source), _identity(snapshot)
        if (row["path"] in seen_paths or str(source) in seen_sources
                or row.get("role") not in {"air", "neighbor_data_candidate"}
                or _five(source_id) != row.get("source_stat")
                or _five(snapshot_id) != row.get("snapshot_stat")
                or type(row.get("bytes")) is not int or row["bytes"] != snapshot_id["bytes"]
                or row["bytes"] != source_id["bytes"] or snapshot_id["mode"] & 0o222
                or (snapshot_id["dev"], snapshot_id["ino"]) in inodes
                or (snapshot_id["dev"], snapshot_id["ino"]) == (source_id["dev"], source_id["ino"])
                or row.get("source_relative") != source.relative_to(old_air.parent).as_posix()
                or row["path"] != (_GENERATED / row["source_relative"]).as_posix()):
            raise ValueError("prefill snapshot identity/role/layout changed or is writable")
        seen_paths.add(row["path"]); seen_sources.add(str(source))
        inodes.add((snapshot_id["dev"], snapshot_id["ino"]))
        protected[str(source)] = source_id
        protected[str(snapshot)] = snapshot_id
        if row["snapshot_method"] == "SHARED_READONLY_SNAPSHOT_INODE":
            observation = observations.get(str(snapshot), {})
            if (observation.get("status") != "SHARED_READONLY_SNAPSHOT_INODE"
                    or observation.get("after") != snapshot_id):
                raise ValueError("prefill shared snapshot lacks its full passed identity audit")
        if row["role"] == "air":
            airs.append(row)
            if source != old_air or row["path"] != layout["air_path"]:
                raise ValueError("prefill AIR source/layout differs")
            for path in (source, snapshot):
                identity = _small(path, bindings)
                if not _same_content(row, identity):
                    raise ValueError("prefill AIR content identity differs")
        else:
            name = snapshot.name
            if name in by_name or len(Path(row["source_relative"]).parts) != 2:
                raise ValueError("prefill weights require unique direct basenames")
            by_name[name] = {"snapshot": snapshot, "row": row, "identity": snapshot_id}
    if len(airs) != 1 or report.get("air") != {
            "path": str(old_air), "bytes": airs[0]["bytes"], "sha256": airs[0]["sha256"]}:
        raise ValueError("prefill requires one uniquely bound original AIR")
    generated = report.get("generated_artifacts", [])
    output_bytes = (sum(row["bytes"] for row in generated) if isinstance(generated, list) and generated
        and all(isinstance(row, dict) and type(row.get("bytes")) is int and row["bytes"] >= 0
                for row in generated) else None)
    return by_name, protected, {"source_result_sha256": outer_id["sha256"],
        "source_compilation_report_sha256": report_id["sha256"], "source_air": str(old_air),
        "prior_compiler_generated_bytes": output_bytes}


def _alias_identity(path):
    _safe(path.parent, directory=True)
    info = path.lstat()
    if not stat.S_ISLNK(info.st_mode):
        raise ValueError("decode reference must remain an owned individual file symlink")
    return {"dev": info.st_dev, "ino": info.st_ino, "bytes": info.st_size,
            "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns,
            "mode": info.st_mode, "uid": info.st_uid, "gid": info.st_gid, "nlink": info.st_nlink}


def _metadata_unchanged(bindings):
    for identity in bindings:
        _, current = _read_small(Path(identity["path"]),
            (64 << 20) if identity["path"].endswith("/compile.log") else
            MAX_AIR_BYTES if identity["path"].endswith(".air") else MAX_METADATA_BYTES)
        if current != identity:
            raise ValueError("decode bound small evidence changed: " + identity["path"])


class DecodeInputView:
    def __init__(self, *, output, air, rows, evidence):
        self.output, self.air, self.rows, self.evidence = output, air, rows, evidence
        self.sources = [Path(key) for key in rows]
        self.attempted = set()

    def source_relative(self, source):
        return Path(self.rows[str(source)]["source_relative"])

    def snapshot(self, source, destination, normal_snapshot):
        source, destination = Path(source), Path(destination)
        key = str(source)
        if key not in self.rows or key in self.attempted:
            raise ValueError("unknown or repeated decode input snapshot source")
        row = self.rows[key]
        if destination != self.output / row["path"]:
            raise ValueError("decode input destination differs from its exact reference path")
        self.attempted.add(key)
        if _identity(source) != row["identity"]:
            raise ValueError("decode source identity changed before snapshot")
        if source == self.air:
            result = normal_snapshot(source, destination)
            after = _identity(destination)
            if (not _same_content(row, result) or result.get("source_stat") != _five(row["identity"])
                    or result.get("snapshot_stat") != _five(after) or after["mode"] & 0o222
                    or after["nlink"] != 1
                    or (after["dev"], after["ino"]) == (row["identity"]["dev"], row["identity"]["ino"])):
                raise ValueError("decode AIR snapshot must be independent, read-only and byte identical")
            observation = dict(row, source=key, status="INDEPENDENT_AIR_SNAPSHOT", after=after)
        else:
            digest, before = _hash_file(source)
            if before != row["identity"] or digest != row["sha256"]:
                raise ValueError("prefill snapshot SHA256 differs from decode1 transfer inventory")
            destination.parent.mkdir(parents=True, exist_ok=True)
            _safe(destination.parent, directory=True)
            parent = _open_directory(destination.parent)
            try:
                os.symlink(key, destination.name, dir_fd=parent)
            finally:
                os.close(parent)
            alias = _alias_identity(destination)
            if (os.readlink(destination) != key or _identity(source) != before
                    or destination.stat().st_ino != before["ino"]
                    or destination.stat().st_dev != before["dev"]):
                raise ValueError("decode symlink creation changed its source or reference")
            reference = {"target": key, "target_identity": before, "alias_identity": alias}
            result = {"source": key, "source_stat": _five(before), "bytes": row["bytes"],
                "sha256": digest, "snapshot_stat": _five(before), "snapshot_method": REFERENCE_METHOD,
                "reference": reference}
            observation = dict(row, source=key, status=REFERENCE_METHOD, reference=reference)
        self.evidence["snapshot_attempts"].append(observation)
        return result

    def audit(self):
        errors, checks = [], []
        try:
            _metadata_unchanged(self.evidence["metadata_bindings"])
        except (OSError, ValueError) as error:
            errors.append(str(error))
        for path, expected in self.evidence["protected_files"].items():
            try:
                if _identity(path) != expected:
                    raise ValueError("protected prefill/source inode metadata changed: " + path)
            except (OSError, ValueError) as error:
                errors.append(str(error))
        observations = {row["source"]: row for row in self.evidence["snapshot_attempts"]}
        for key, row in self.rows.items():
            try:
                observation = observations.get(key)
                if observation is None:
                    raise ValueError("decode snapshot did not complete: " + key)
                _verify_snapshot(self.output, observation)
                checks.append({"source": key, "status": "PASSED", "sha256": row["sha256"],
                               "payload_verification": "FRESH_SHA256_MATCHED_AFTER_COMPILE"})
            except (OSError, ValueError) as error:
                errors.append(str(error))
                checks.append({"source": key, "status": "FAILED", "error": str(error)})
        try:
            _verify_prefix(self.output, self.evidence)
        except (OSError, ValueError) as error:
            errors.append(str(error))
        self.evidence.update(status="FAILED" if errors else "PASSED", audit_errors=errors,
                             audit_checks=checks, completed_inputs=len(observations))
        return self.evidence


def _verify_snapshot(output, observation):
    destination = _relative(output, observation["path"])
    source = Path(observation["source"])
    if observation["status"] == "INDEPENDENT_AIR_SNAPSHOT":
        digest, current = _hash_file(destination)
        if (current != observation["after"] or digest != observation["sha256"]
                or _identity(source) != observation["identity"]):
            raise ValueError("decode independent AIR/source changed")
    elif observation["status"] == REFERENCE_METHOD:
        reference = observation["reference"]
        if (reference["target"] != str(source) or reference["target_identity"] != observation["identity"]
                or _alias_identity(destination) != reference["alias_identity"]
                or os.readlink(destination) != str(source)):
            raise ValueError("decode weight reference alias/target identity changed")
        digest, current = _hash_file(source)
        if current != observation["identity"] or digest != observation["sha256"]:
            raise ValueError("decode referenced prefill snapshot content/metadata changed")
    else:
        raise ValueError("unknown decode snapshot observation")


def _verify_prefix(output, evidence):
    folder = output / _GENERATED / evidence["source_relative_prefix"]
    _safe(folder, directory=True)
    expected = {Path(row["path"]).name for row in evidence["snapshot_attempts"]
                if row["status"] == REFERENCE_METHOD}
    if {path.name for path in folder.iterdir()} != expected:
        raise ValueError("decode reference directory has missing or additional aliases")


def prepare_decode_input_view(source_job, *, air, inventory, export_receipt, output,
                              ddk_root, ddk_id, omg_path, omg_sha256, options, input_layout):
    """Validate metadata before OMG help; create no aliases until snapshot()."""
    job, _ = _safe(source_job, directory=True)
    air, _ = _safe(air)
    inventory, _ = _safe(inventory)
    export_receipt, _ = _safe(export_receipt)
    output, _ = _safe(output, directory=True)
    ddk, _ = _safe(ddk_root, directory=True)
    omg, _ = _safe(omg_path)
    if (input_layout != "air-directory" or options != _options("cpro-40block-decode-1")
            or air.name != "export.air"
            or any(output == root or root in output.parents or output in root.parents
                   for root in (job, air.parent, ddk))):
        raise ValueError("decode view requires a separate air-directory static decode1 compilation")
    bindings = []
    manifest, weights, prefix, transfer = _transfer(air, inventory, export_receipt, bindings)
    old, protected, source_binding = _bound_source(job, ddk=ddk, ddk_id=ddk_id,
        omg=omg, omg_sha256=omg_sha256, bindings=bindings)
    if set(old) != {row["basename"] for row in weights}:
        raise ValueError("decode1 inventory weight basenames differ from completed prefill snapshots")
    digest, tool_id = _hash_file(omg)
    if digest != omg_sha256 or not os.access(omg, os.X_OK):
        raise ValueError("decode selected OMG content identity changed")
    protected[str(omg)] = tool_id
    air_identity = _identity(air)
    protected[str(air)] = air_identity
    rows = {str(air): {"source_relative": "export.air", "path": (_GENERATED / "export.air").as_posix(),
        "role": "air", "bytes": manifest["air"]["bytes"], "sha256": manifest["air"]["sha256"],
        "identity": air_identity}}
    for row in weights:
        previous = old[row["basename"]]
        if any(previous["row"].get(key) != row[key] for key in ("bytes", "sha256")):
            raise ValueError("decode1 inventory weight content declaration differs from prefill snapshot")
        source = previous["snapshot"]
        rows[str(source)] = {"source_relative": row["relative_path"],
            "path": (_GENERATED / row["relative_path"]).as_posix(), "role": "neighbor_data_candidate",
            "bytes": row["bytes"], "sha256": row["sha256"], "identity": previous["identity"]}
    _metadata_unchanged(bindings)
    disk = os.statvfs(output)
    free_bytes = disk.f_bavail * disk.f_frsize
    estimate = source_binding["prior_compiler_generated_bytes"]
    evidence = {"schema_version": 1, "artifact_kind": _KIND, "status": "PREPARED",
        "source_job": str(job), "output": str(output), "decode_air": str(air),
        "decode_inventory": str(inventory), "decode_export_receipt": str(export_receipt),
        "source_relative_prefix": prefix, "sequence_length": 1, "input_count": len(rows),
        "reference_method": REFERENCE_METHOD, "ownership": "READONLY_REFERENCE_VIEW",
        "explicit_weight_copy_bytes": 0, "physical_weight_allocation": "NO_PAYLOAD_ALLOCATION_REQUESTED",
        "transfer": transfer, "metadata_bindings": bindings, "protected_files": protected,
        "source_binding": source_binding, "tool_binding": {"path": str(omg), "sha256": digest},
        "ddk_root": str(ddk), "ddk_id": ddk_id, "options": dict(options), "snapshot_attempts": [],
        "disk_budget": {"free_bytes_observed_before_snapshot": free_bytes,
            "independent_air_copy_bytes": manifest["air"]["bytes"], "explicit_weight_copy_bytes": 0,
            "prior_compiler_generated_bytes": estimate,
            "free_exceeds_prior_output_and_air": (None if estimate is None else
                free_bytes > estimate + manifest["air"]["bytes"]),
            "scope": "Observed shared filesystem space and prior compiler-output declaration only. Decode output/temp size is unknown; this is not a reservation or guaranteed space pass."},
        "device_execution": "NOT_RUN", "numerical_validation": "NOT_RUN",
        "scope": "One completed regular prefill64 input snapshot set to static decode1 only. Individual owned symlinks preserve old inode metadata and receipts. Fresh source hashes before/after compilation; no independent snapshot, output-space guarantee, runtime ABI or model accuracy proof."}
    return DecodeInputView(output=output, air=air, rows=rows, evidence=evidence)


def validate_decode_input_view(compilation, report):
    """Return only exact audited alias -> regular target paths for continuation."""
    root, _ = _safe(compilation, directory=True)
    saved, _ = _json(root / "decode-input-view.json", [])
    if (saved != report.get("decode_input_view") or saved.get("artifact_kind") != _KIND
            or saved.get("status") != "PASSED" or saved.get("audit_errors") != []
            or saved.get("reference_method") != REFERENCE_METHOD or saved.get("sequence_length") != 1
            or saved.get("output") != str(root) or saved.get("explicit_weight_copy_bytes") != 0
            or report.get("requested_options") != _options("cpro-40block-decode-1")):
        raise ValueError("completed decode input reference audit is not exact and passed")
    plan = prepare_decode_input_view(saved["source_job"], air=saved["decode_air"],
        inventory=saved["decode_inventory"], export_receipt=saved["decode_export_receipt"], output=root,
        ddk_root=report["ddk_root"], ddk_id=report["ddk_id"], omg_path=report["omg_path"],
        omg_sha256=report["omg_sha256"], options=report["requested_options"],
        input_layout=report.get("input_layout", {}).get("mode"))
    initial, _ = _json(root / "decode-input-view-plan.json", [])
    immutable_initial = {key: value for key, value in initial.items() if key != "disk_budget"}
    current_initial = {key: value for key, value in plan.evidence.items() if key != "disk_budget"}
    if immutable_initial != current_initial or initial.get("disk_budget") != saved.get("disk_budget"):
        raise ValueError("decode immutable pre-compiler input-view plan changed")
    for key in ("metadata_bindings", "protected_files", "source_binding", "transfer", "tool_binding",
                "input_count", "source_relative_prefix", "ownership", "options", "ddk_root", "ddk_id"):
        if saved.get(key) != plan.evidence[key]:
            raise ValueError("completed decode reference binding changed: " + key)
    snapshots, observations = report.get("input_snapshots"), saved.get("snapshot_attempts")
    if (not isinstance(snapshots, list) or not isinstance(observations, list)
            or len(snapshots) != len(plan.rows) or len(observations) != len(plan.rows)
            or saved.get("completed_inputs") != len(plan.rows)
            or len(saved.get("audit_checks", [])) != len(plan.rows)
            or any(row.get("status") != "PASSED" for row in saved["audit_checks"])):
        raise ValueError("completed decode reference audit/snapshot coverage differs")
    by_source = {row.get("source"): row for row in observations if isinstance(row, dict)}
    actual = {row.get("source"): row for row in snapshots if isinstance(row, dict)}
    if set(by_source) != set(plan.rows) or set(actual) != set(plan.rows):
        raise ValueError("completed decode source set differs from bound transfer/source")
    aliases = {}
    for source, expected in plan.rows.items():
        row, observation = actual[source], by_source[source]
        if any(observation.get(key) != value for key, value in expected.items()):
            raise ValueError("completed decode observation differs from original source binding")
        if any(row.get(key) != expected[key] for key in ("bytes", "sha256", "path", "role", "source_relative")):
            raise ValueError("completed decode snapshot path/content/role differs")
        if row.get("source_stat") != _five(expected["identity"]):
            raise ValueError("completed decode input source stat differs")
        _verify_snapshot(root, observation)
        if source != str(plan.air):
            if (row.get("snapshot_method") != REFERENCE_METHOD or row.get("reference") != observation["reference"]
                    or row.get("snapshot_stat") != _five(expected["identity"])):
                raise ValueError("completed decode reference snapshot receipt differs")
            aliases[row["path"]] = Path(source)
        elif (row.get("snapshot_method") not in {"COPY", "REFLINK"}
              or row.get("snapshot_stat") != _five(observation["after"])):
            raise ValueError("completed decode AIR snapshot receipt differs")
    _verify_prefix(root, saved)
    return aliases
