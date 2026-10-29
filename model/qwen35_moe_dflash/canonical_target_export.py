"""Canonicalize only target output ports, then inventory existing payloads.

The original AIR and sidecars are immutable inputs. Only a small new AIR and
metadata are written. A completed inventory may be reused after identity/stat
checks; failed inventory metadata is retained before a retry.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import stat

from .air_schema import load_torchair_schema
from .artifacts import output_path, save_json
from .original_ge_target import read_json
from .target_export_inventory import (_checked_path, _hash_file, _read_air, _select_schema,
                                      inventory_target_export)
from .target_output_order import reorder_target_output_bytes
from .target_output_order_validation import validate_target_output_order


KIND = "qwen35-canonical-target-export"
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _archive_file(path, attempts, *, expected_sha=None, remove=False):
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha is not None and digest != expected_sha:
        raise ValueError("retained metadata changed before archive")
    attempts.mkdir(exist_ok=True)
    target = attempts / (path.name + "." + digest + ".json")
    if target.exists():
        if target.is_symlink() or target.read_bytes() != raw:
            raise ValueError("retained metadata archive differs")
    else:
        with target.open("xb") as stream:
            stream.write(raw)
    if remove:
        if path.read_bytes() != raw:
            raise ValueError("retained metadata changed during archive")
        path.unlink()


def _owned_output(out, original, normalized_sha, schema_sha, resume):
    if not out.exists():
        out.mkdir(parents=True, exist_ok=False)
        return None
    if not resume or out.is_symlink() or not out.is_dir():
        raise ValueError("output already exists; --resume requires this task's unchanged evidence")
    prior, prior_sha = read_json(out / "result.json", 16 << 20)
    if (prior.get("artifact_kind") != KIND or prior.get("original_air") != original
            or prior.get("schema", {}).get("descriptor_sha256") != schema_sha
            or prior.get("model_compilation") != "NOT_RUN" or prior.get("device_execution") != "NOT_RUN"):
        raise ValueError("output is not this source-bound metadata-only task")
    allowed = {"export.air", "result.json", "normalization.json", "validation.json", "inventory", "attempts"}
    if any(path.name not in allowed or path.is_symlink() for path in out.iterdir()):
        raise ValueError("output contains unknown files or symlinks; retained unchanged")
    if not (out / "export.air").is_file():
        raise ValueError("retained task has no complete canonical AIR")
    _, _, raw = _read_air(out / "export.air")
    if hashlib.sha256(raw).hexdigest() != normalized_sha:
        raise ValueError("retained canonical AIR differs from the freshly verified permutation")
    inventory = out / "inventory"
    if inventory.exists():
        if (inventory.is_symlink() or not inventory.is_dir()
                or {path.name for path in inventory.iterdir()} != {"inventory.json"}):
            raise ValueError("retained inventory contains unknown entries")
        manifest_path = inventory / "inventory.json"
        _, digest = read_json(manifest_path, 16 << 20)
        if (prior.get("inventory", {}).get("sha256") != digest
                or prior.get("inventory", {}).get("path") != str(manifest_path)):
            raise ValueError("retained inventory is not owned by this task")
    metadata = []
    for name, key in (("normalization.json", "normalization"), ("validation.json", "validation")):
        value, digest = read_json(out / name, 16 << 20)
        if value != prior.get(key):
            raise ValueError("retained metadata is not bound by this task: " + name)
        metadata.append((out / name, digest))
    for path, digest in [(out / "result.json", prior_sha), *metadata]:
        _archive_file(path, out / "attempts", expected_sha=digest, remove=True)
    return prior


def _cached_inventory(folder, prior, normalized_sha, schema_sha, reference_root, need_hash):
    if not folder.exists():
        return None
    if folder.is_symlink() or not folder.is_dir() or {path.name for path in folder.iterdir()} != {"inventory.json"}:
        raise ValueError("inventory directory contains unknown files or symlinks")
    manifest_path = folder / "inventory.json"
    manifest, digest = read_json(manifest_path, 16 << 20)
    if (manifest.get("artifact_kind") != "qwen35-target-export-inventory"
            or manifest.get("air", {}).get("sha256") != normalized_sha
            or manifest.get("schema", {}).get("descriptor_sha256") != schema_sha):
        raise ValueError("retained inventory does not bind this canonical AIR/schema")
    if (prior.get("inventory", {}).get("sha256") != digest
            or prior.get("inventory", {}).get("path") != str(manifest_path)):
        raise ValueError("retained inventory digest is not bound by this task")
    if manifest.get("status") == "INVENTORIED":
        if manifest.get("reference_root") != str(reference_root):
            raise ValueError("complete inventory identity or reference root changed")
        files = manifest.get("files", [])
        if not isinstance(files, list) or not files:
            raise ValueError("complete inventory has no referenced files")
        for row in files:
            relative = Path(row.get("relative_path", ""))
            if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 2:
                raise ValueError("invalid retained payload reference")
            _, stamp = _checked_path(reference_root / relative)
            if list(stamp) != row.get("stat"):
                raise ValueError("referenced payload metadata changed since the complete inventory")
            if need_hash and manifest.get("weight_identity") == "SHA256_COMPLETE" and not _SHA.fullmatch(str(row.get("sha256", ""))):
                raise ValueError("complete payload hash missing")
        if not need_hash or manifest.get("weight_identity") == "SHA256_COMPLETE":
            return manifest
    elif manifest.get("status") != "BLOCKED" or manifest.get("changed_sources"):
        raise ValueError("retained inventory is not a safely retryable metadata failure")
    attempts = folder.parent / "attempts"
    attempts.mkdir(exist_ok=True)
    archive = attempts / ("inventory-" + digest)
    if archive.exists():
        raise ValueError("failed inventory archive already exists; retained unchanged")
    folder.rename(archive)
    return None


def canonicalize_target_export(air, output, *, torchair_pb2=None, discover_schema=False,
                               sequence_length=64, reference_root=None, hash_weights=False,
                               resume=False, progress=None):
    requested = Path(output).expanduser()
    if ".." in requested.parts:
        raise ValueError("output parent traversal is unsupported")
    current = Path(os.path.abspath(requested))
    for ancestor in [*reversed(current.parents), current]:
        try:
            info = ancestor.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("output symlink is unsupported")
    out = output_path(output)
    result = {"schema_version": 1, "artifact_kind": KIND, "status": "BLOCKED", "output": str(out),
              "model_compilation": "NOT_RUN", "device_execution": "NOT_RUN", "numerical_validation": "NOT_RUN",
              "weight_copies": "NONE", "scope": "Output-port permutation and source-payload identity only; no model accuracy or state-continuity pass"}
    source, stamp, owned, schema_record, code_identities = None, None, False, None, {}
    try:
        source, stamp, raw = _read_air(air)
        original = {"path": str(source), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        result["original_air"] = original
        schema = _select_schema(torchair_pb2, discover_schema)
        result["schema"] = schema.evidence
        schema_record = dict(schema.evidence)
        for name in ("canonical_target_export.py", "target_output_order.py", "target_output_order_validation.py",
                     "target_export_inventory.py", "air_schema.py", "omc_wire.py"):
            path = Path(__file__).with_name(name)
            code_identities[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        result["code_identities"] = code_identities
        normalized, receipt = reorder_target_output_bytes(raw, schema, sequence_length=sequence_length)
        validation = validate_target_output_order(raw, normalized, schema, sequence_length=sequence_length)
        result["normalization"] = receipt
        result["validation"] = validation
        if validation.get("status") != "PASS_METADATA_ORDER_ONLY":
            raise ValueError("independent output-order preservation validation failed")
        normalized_sha = hashlib.sha256(normalized).hexdigest()
        prior = _owned_output(out, original, normalized_sha, schema.evidence["descriptor_sha256"], resume)
        owned = True
        destination = out / "export.air"
        if prior is None:
            partial = out / ".export.air.partial"
            with partial.open("xb") as stream:
                stream.write(normalized)
            os.link(partial, destination)
            partial.unlink()
        result["air"] = {"path": str(destination), "bytes": len(normalized), "sha256": normalized_sha}
        save_json(out / "normalization.json", receipt)
        save_json(out / "validation.json", validation)
        base, _ = _checked_path(reference_root or source.parent.parent, directory=True)
        cached = _cached_inventory(out / "inventory", prior or {}, normalized_sha,
                                   schema.evidence["descriptor_sha256"], base, hash_weights)
        manifest = cached or inventory_target_export(destination, out / "inventory",
            torchair_pb2=schema.evidence["pb2"], hash_weights=hash_weights,
            reference_root=base, sequence_length=sequence_length, progress=progress)
        manifest_path = out / "inventory/inventory.json"
        _, manifest_sha = read_json(manifest_path, 16 << 20)
        result["inventory"] = {"path": str(manifest_path), "sha256": manifest_sha,
                               "status": manifest.get("status"), "reused": cached is not None}
        if manifest.get("status") != "INVENTORIED":
            raise ValueError("canonical AIR inventory blocked: " + str(manifest.get("error")))
        result.update(status="EXPORT_READY", weight_identity=manifest["weight_identity"],
                      unique_reference_count=manifest["unique_reference_count"],
                      total_weight_bytes=manifest["total_weight_bytes"],
                      feature_output_index=81, source_relative_prefix=manifest["source_relative_prefix"])
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        if source is not None and "original_air" in result:
            try:
                same = (_checked_path(source)[1] == stamp and _hash_file(source, stamp) == result["original_air"]["sha256"])
            except (OSError, ValueError):
                same = False
            result["original_preservation"] = "UNCHANGED_STAT_AND_SHA256" if same else "CHANGED"
            if not same:
                result.update(status="BLOCKED", error="original AIR changed during output normalization")
        if schema_record is not None:
            try:
                current = load_torchair_schema(schema_record["pb2"]).evidence
                same = all(current.get(key) == schema_record.get(key) for key in
                           ("pb2_sha256", "descriptor_sha256", "schema_file", "schema_package"))
            except (OSError, ValueError):
                same = False
            result["schema_preservation"] = "UNCHANGED_PB2_AND_DESCRIPTOR_SHA256" if same else "CHANGED"
            if not same:
                result.update(status="BLOCKED", error="selected schema source changed during normalization")
        changed = []
        for name, digest in code_identities.items():
            try:
                same = hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
            except OSError:
                same = False
            if not same:
                changed.append(name)
        if changed:
            result.update(status="BLOCKED", error="normalization code changed during execution", changed_code=changed)
        if owned:
            try:
                save_json(out / "result.json", result)
            except OSError as error:
                result.update(status="BLOCKED", error=f"unable to publish new task receipt: {type(error).__name__}: {error}")
    return result
