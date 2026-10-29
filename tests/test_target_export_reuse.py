"""Small host-only files validate reuse; no real model payload or compiler run."""
import errno
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from qwen35_moe_dflash import target_export_reuse as reuse
from test_target_export_inventory import source


@pytest.fixture
def case(tmp_path, monkeypatch):
    run, old, incoming = (tmp_path / name for name in ("run", "old-55-export", "incoming"))
    run.mkdir()
    old.mkdir()
    incoming.mkdir()
    monkeypatch.setenv("AI_RUN_DIR", str(run))
    air = incoming / "new64.air"
    air.write_bytes(b"HOST FAKE AIR; no model execution")
    prefix = "glm_export_air_cpro_40block_64_quant"
    files = []
    for index, payload in enumerate((b"first exact sidecar", b"second exact sidecar")):
        name = f"model_parameter_{index}"
        (old / name).write_bytes(payload)
        reference = "./" + prefix + "/" + name
        files.append({"reference": reference, "references": [reference, reference[2:]],
                      "relative_path": reference[2:], "basename": name,
                      "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
                      "nodes": [f"FileConstant_{index}"]})
    manifest = {"artifact_kind": "qwen35-target-export-inventory", "status": "INVENTORIED",
                "weight_identity": "SHA256_COMPLETE", "source_relative_prefix": prefix,
                "air": {"path": "/origin/private/not-followed/export.air", "bytes": air.stat().st_size,
                        "sha256": hashlib.sha256(air.read_bytes()).hexdigest()},
                "files": files, "schema": {"descriptor_sha256": "a" * 64},
                "declared_abi": {"status": "MATCHED_INTERFACE_METADATA", "input_count": 89,
                    "output_count": 82, "feature_output_index": 81, "feature_shape": [1, 64, 16384],
                    "feature_dtype": "DT_FLOAT16"}}
    inventory = incoming / "inventory.json"
    inventory.write_text(json.dumps(manifest))
    return {"run": run, "old": old, "air": air, "inventory": inventory,
            "manifest": manifest, "prefix": prefix, "output": run / "target-pytorch-import"}


def save(case):
    case["inventory"].write_text(json.dumps(case["manifest"]))


def stage(case, *, apply=False):
    return reuse.stage_target_export_reuse(case["inventory"], case["air"], case["old"], case["output"], apply=apply)


def core(path):
    info = path.stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_mode, info.st_uid, info.st_gid)


def test_default_plan_reads_exact_basename_bytes_without_creating_any_output(case, monkeypatch):
    before = {p.name: (core(p), p.stat().st_ctime_ns, p.stat().st_nlink) for p in case["old"].iterdir()}
    monkeypatch.setattr(reuse.os, "link", lambda *a, **k: pytest.fail("plan cannot create hardlinks"))
    result = stage(case)
    assert result["status"] == "PLAN_READY" and result["mode"] == "PLAN_ONLY"
    assert not case["output"].exists()
    assert result["transfer_required"] == []
    assert all(row["status"] == "MATCHED" for row in result["files"])
    assert result["source_bytes"] == sum(row["bytes"] for row in case["manifest"]["files"])
    assert result["replicated_weight_data_bytes"] == 0
    assert result["explicit_sidecar_copy_bytes"] == 0
    assert result["physical_weight_allocation"].startswith("NOT_INFERRED")
    assert result["required_compile_input_layout"] == "air-directory"
    assert result["disk"]["free_bytes"] > 0 and result["disk"]["later_compiler_output_bytes"] == "UNKNOWN"
    assert result["compiler_budget"].startswith("PENDING")
    assert {p.name: (core(p), p.stat().st_ctime_ns, p.stat().st_nlink) for p in case["old"].iterdir()} == before


def test_apply_links_all_sidecars_rehashes_final_bytes_and_never_changes_permissions(case, monkeypatch):
    sources = {row["basename"]: case["old"] / row["basename"] for row in case["manifest"]["files"]}
    for path in sources.values():
        path.chmod(0o440)
    before = {name: (core(path), path.stat().st_nlink, path.stat().st_ctime_ns) for name, path in sources.items()}
    monkeypatch.setattr(reuse.os, "chmod", lambda *a, **k: pytest.fail("hardlinks must never be chmodded"))
    monkeypatch.setattr(Path, "chmod", lambda *a, **k: pytest.fail("hardlinks must never be chmodded"))
    result = stage(case, apply=True)
    assert result["status"] == "STAGED_HARDLINKS"
    assert result["weight_payload_validation"] == "INITIAL_AND_FINAL_STREAMED_SHA256_MATCHED"
    assert result["replicated_weight_data_bytes"] == 0
    assert result["disk"]["free_bytes_after"] > 0
    assert (case["output"] / "export.air").read_bytes() == case["air"].read_bytes()
    assert (case["output"] / "inventory.json").read_bytes() == case["inventory"].read_bytes()
    for name, source in sources.items():
        destination = case["output"] / case["prefix"] / name
        assert core(source) == before[name][0] == core(destination)
        assert source.stat().st_nlink == before[name][1] + 1
        assert source.stat().st_ctime_ns >= before[name][2]
        assert source.read_bytes() == destination.read_bytes()
    receipt = case["output"] / "reuse-receipt.json"
    assert hashlib.sha256(receipt.read_bytes()).hexdigest() == result["receipt"]["sha256"]
    saved = json.loads(receipt.read_text())
    assert saved["status"] == "STAGED_HARDLINKS"
    assert saved["inventory"]["sha256"] == hashlib.sha256(case["inventory"].read_bytes()).hexdigest()
    assert all(row["final_sha256"] == row["sha256"] and row["own_hardlink_count"] == 1 for row in saved["files"])
    assert result["model_compilation"] == result["device_execution"] == "NOT_RUN"


@pytest.mark.parametrize("kind", ["missing", "size", "hash", "symlink"])
@pytest.mark.parametrize("apply", [False, True])
def test_incomplete_sources_require_transfer_without_copying_or_creating_output(case, kind, apply):
    source = case["old"] / case["manifest"]["files"][0]["basename"]
    if kind == "missing":
        source.unlink()
    elif kind == "size":
        source.write_bytes(b"short")
    elif kind == "hash":
        source.write_bytes(b"X" * source.stat().st_size)
    else:
        source.unlink()
        source.symlink_to(case["air"])
    result = stage(case, apply=apply)
    assert result["status"] == "TRANSFER_REQUIRED"
    assert len(result["transfer_required"]) == 1
    assert not case["output"].exists()
    assert result["transfer_required"][0]["status"] == {
        "missing": "MISSING", "size": "SIZE_MISMATCH", "hash": "HASH_MISMATCH", "symlink": "UNSAFE_OR_CHANGED_SOURCE"}[kind]


@pytest.mark.parametrize("kind", ["absolute", "escape", "nested", "basename", "alias_escape", "alias_other_file", "duplicate_basename", "prefixes", "claimed_prefix", "relative_path", "weight_hash", "stat_only", "abi_count", "abi_dtype", "schema", "reserved_prefix"])
def test_invalid_manifest_cannot_choose_other_paths_or_bypass_original_interface(case, kind):
    manifest = case["manifest"]
    row = manifest["files"][0]
    if kind == "absolute":
        row["reference"] = "/origin/private/state"
    elif kind == "escape":
        row["reference"] = "./../state"
    elif kind == "nested":
        row["reference"] = "./folder/subfolder/weight"
    elif kind == "basename":
        row["basename"] = "../state"
    elif kind == "alias_escape":
        row["references"].append("../state")
    elif kind == "alias_other_file":
        row["references"].append("./" + case["prefix"] + "/different")
    elif kind == "duplicate_basename":
        manifest["files"].append(dict(row))
    elif kind == "prefixes":
        manifest["files"][1]["reference"] = "./other/weight"
    elif kind == "claimed_prefix":
        manifest["source_relative_prefix"] = "other"
    elif kind == "relative_path":
        row["relative_path"] = "wrong/weight"
    elif kind == "weight_hash":
        row["sha256"] = None
    elif kind == "stat_only":
        manifest["weight_identity"] = "STAT_ONLY"
    elif kind == "abi_count":
        manifest["declared_abi"]["output_count"] = 81
    elif kind == "abi_dtype":
        manifest["declared_abi"]["feature_dtype"] = "DT_FLOAT"
    elif kind == "schema":
        manifest["schema"] = {}
    else:
        row["reference"] = "./inventory.json/weight"
    save(case)
    result = stage(case, apply=True)
    assert result["status"] == "BLOCKED" and not case["output"].exists()


def test_inventory_absolute_source_paths_are_only_metadata_and_never_followed(case, monkeypatch):
    outside = case["run"].parent / "private-state.bin"
    outside.write_bytes(b"must not be read")
    case["manifest"]["files"][0]["path"] = str(outside)
    case["manifest"]["files"][0]["source"] = str(outside)
    save(case)
    old_open = os.open

    def guarded(name, *args, **kwargs):
        assert str(name) != str(outside) and str(name) != outside.name
        return old_open(name, *args, **kwargs)

    monkeypatch.setattr(reuse.os, "open", guarded)
    assert stage(case)["status"] == "PLAN_READY"


@pytest.mark.parametrize("which", ["inventory", "air", "existing", "output_parent"])
def test_symlink_in_any_directory_component_is_rejected(case, which):
    link = case["run"].parent / "directory-link"
    if which == "existing":
        link.symlink_to(case["old"], target_is_directory=True)
        case["old"] = link
    elif which == "output_parent":
        link.symlink_to(case["run"], target_is_directory=True)
        case["output"] = link / "new"
        os.environ["AI_RUN_DIR"] = str(link)
    else:
        link.symlink_to(case[which].parent, target_is_directory=True)
        case[which] = link / case[which].name
    assert stage(case, apply=True)["status"] == "BLOCKED"
    assert not (case["run"] / "new").exists()


def test_air_identity_is_exact_and_cannot_be_taken_from_inventory_origin_path(case):
    case["air"].write_bytes(b"X" * case["air"].stat().st_size)
    result = stage(case, apply=True)
    assert result["status"] == "BLOCKED" and "AIR file" in result["error"]
    assert not case["output"].exists()


def test_existing_partial_output_is_preserved_without_even_rehashing_sources(case, monkeypatch):
    case["output"].mkdir()
    retained = case["output"] / "keep-me"
    retained.write_bytes(b"retained partial evidence")
    monkeypatch.setattr(reuse, "_read_small", lambda *a, **k: pytest.fail("existing output must reject before reading inputs"))
    result = stage(case, apply=True)
    assert result["status"] == "BLOCKED" and "preserve" in result["error"]
    assert retained.read_bytes() == b"retained partial evidence"


@pytest.mark.parametrize("kind", ["low_space", "cross_filesystem"])
def test_space_or_filesystem_mismatch_has_no_weight_copy_fallback(case, kind, monkeypatch):
    original = reuse._destination_parent

    def changed(output):
        result = original(output)
        if kind == "low_space":
            result["free_bytes"] = 1
        else:
            result["device"] += 1
        return result

    monkeypatch.setattr(reuse, "_destination_parent", changed)
    monkeypatch.setattr(reuse.os, "link", lambda *a, **k: pytest.fail("checks must fail before hardlinking"))
    result = stage(case, apply=True)
    assert result["status"] == "BLOCKED" and not case["output"].exists()
    assert result["replicated_weight_data_bytes"] == 0


def test_hardlink_failure_preserves_partial_evidence_and_does_not_copy_or_unlink(case, monkeypatch):
    original = os.link
    calls = []

    def failing(*args, **kwargs):
        calls.append(args)
        if len(calls) == 2:
            raise OSError(errno.EXDEV, "simulated same-FS link failure; no copy fallback")
        return original(*args, **kwargs)

    monkeypatch.setattr(reuse.os, "link", failing)
    result = stage(case, apply=True)
    assert result["status"] == "APPLY_FAILED_PARTIAL_PRESERVED"
    assert result["partial_output_preserved"]
    staged = list((case["output"] / case["prefix"]).iterdir())
    assert len(staged) == 1 and staged[0].stat().st_ino == (case["old"] / staged[0].name).stat().st_ino
    assert (case["output"] / "failed-reuse-receipt.json").is_file()
    assert stage(case, apply=True)["status"] == "BLOCKED"


def test_source_mutated_during_plan_never_becomes_a_reuse_pass(case, monkeypatch):
    original = reuse._hash_file
    target = case["old"] / case["manifest"]["files"][0]["basename"]
    changed = False

    def mutate(folder, name, **kwargs):
        nonlocal changed
        result = original(folder, name, **kwargs)
        if name == target.name and not changed:
            changed = True
            before = target.stat()
            target.write_bytes(b"X" * target.stat().st_size)
            # Some CI filesystems batch timestamps within one clock tick.
            os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
        return result

    monkeypatch.setattr(reuse, "_hash_file", mutate)
    report = stage(case, apply=True)
    assert report["status"] == "TRANSFER_REQUIRED" and report["files"][0]["status"] == "SOURCE_CHANGED"
    assert not case["output"].exists()


def test_source_payload_changed_after_linking_fails_final_streamed_hash(case, monkeypatch):
    original = os.link
    target = case["old"] / case["manifest"]["files"][0]["basename"]
    calls = 0

    def mutate(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == 2:
            target.write_bytes(b"X" * target.stat().st_size)
        return result

    monkeypatch.setattr(reuse.os, "link", mutate)
    report = stage(case, apply=True)
    assert report["status"] == "APPLY_FAILED_PARTIAL_PRESERVED"
    assert "changed" in report["error"] or "differs" in report["error"]


def test_multiple_source_names_with_shared_inode_have_exact_own_nlink_increment(case):
    first, second = (case["old"] / row["basename"] for row in case["manifest"]["files"])
    second.unlink()
    os.link(first, second)
    row = case["manifest"]["files"][1]
    row.update(bytes=first.stat().st_size, sha256=hashlib.sha256(first.read_bytes()).hexdigest())
    save(case)
    before = first.stat().st_nlink
    report = stage(case, apply=True)
    assert report["status"] == "STAGED_HARDLINKS"
    assert first.stat().st_nlink == before + 2
    assert all(row["own_hardlink_count"] == 2 for row in report["files"])


def test_duplicate_json_keys_and_output_escape_are_blocked(case):
    case["inventory"].write_text('{"status":"INVENTORIED","status":"INVENTORIED"}')
    assert stage(case)["status"] == "BLOCKED"
    save(case)
    case["output"] = case["run"].parent / "outside-run"
    assert stage(case, apply=True)["status"] == "BLOCKED"


def test_nonfinite_json_constant_is_rejected_before_staging(case):
    case["manifest"]["untrusted_extra"] = float("nan")
    save(case)
    assert stage(case, apply=True)["status"] == "BLOCKED"
    assert not case["output"].exists()


def test_replaced_staged_directory_cannot_hide_a_symlink_in_successful_receipt(case, monkeypatch):
    original = reuse._hash_file
    first = case["manifest"]["files"][0]["basename"]
    counts = 0

    def replace(folder, name, **kwargs):
        nonlocal counts
        result = original(folder, name, **kwargs)
        if name == first:
            counts += 1
            if counts == 2:
                directory = case["output"] / case["prefix"]
                directory.rename(case["output"] / "retained-moved-links")
                directory.symlink_to(case["old"], target_is_directory=True)
        return result

    monkeypatch.setattr(reuse, "_hash_file", replace)
    result = stage(case, apply=True)
    assert result["status"] == "APPLY_FAILED_PARTIAL_PRESERVED"
    assert result["partial_output_preserved"]


def test_cli_default_is_read_only_and_json_reports_compiler_budget_pending(case):
    script = Path(__file__).resolve().parents[1] / "targets/cpro/scripts/stage_target_export_reuse.py"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, str(script), "--inventory", str(case["inventory"]),
        "--air", str(case["air"]), "--existing-export", str(case["old"]), "--output", str(case["output"])],
        capture_output=True, text=True, env=env)
    assert result.returncode == 0 and result.stderr == ""
    report = json.loads(result.stdout)
    assert report["status"] == "PLAN_READY" and not case["output"].exists()
    assert report["compiler_budget"].startswith("PENDING")
    assert report["file_count"] == 2 and report["file_status_counts"] == {"MATCHED": 2}
    assert "files" not in report and report["transfer_required_count"] == 0


def test_cli_transfer_summary_is_bounded_to_eight_rows(case):
    for index in range(12):
        name = f"missing_{index}"
        reference = "./" + case["prefix"] + "/" + name
        case["manifest"]["files"].append({"reference": reference, "basename": name,
            "relative_path": reference[2:], "references": [reference], "bytes": 1,
            "sha256": hashlib.sha256(b"x").hexdigest()})
    save(case)
    script = Path(__file__).resolve().parents[1] / "targets/cpro/scripts/stage_target_export_reuse.py"
    result = subprocess.run([sys.executable, str(script), "--inventory", str(case["inventory"]),
        "--air", str(case["air"]), "--existing-export", str(case["old"]), "--output", str(case["output"])],
        capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report["transfer_required_count"] == 12 and len(report["transfer_required"]) == 8
    assert report["file_count"] == 14 and "files" not in report


def test_real_inventory_api_feeds_reuse_without_field_translation(source, tmp_path):
    from qwen35_moe_dflash.target_export_inventory import inventory_target_export

    manifest = inventory_target_export(source["air"], source["out"],
                                       torchair_pb2=source["schema"], hash_weights=True)
    assert manifest["status"] == "INVENTORIED" and manifest["weight_identity"] == "SHA256_COMPLETE"
    old = tmp_path / "old-export"
    old.mkdir()
    for name, path in source["weights"].items():
        (old / name).write_bytes(path.read_bytes())
    result = reuse.stage_target_export_reuse(source["out"] / "inventory.json", source["air"], old,
                                             tmp_path / "run/reused", apply=True)
    assert result["status"] == "STAGED_HARDLINKS"
    assert result["source_bytes"] == manifest["total_weight_bytes"]
