"""Three-small-file CLI checks must precede any sidecar inspection/staging."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

from qwen35_moe_dflash import target_export_reuse as reuse


@pytest.fixture
def cli():
    script = Path(__file__).resolve().parents[1] / "targets/cpro/scripts/stage_target_export_reuse.py"
    spec = importlib.util.spec_from_file_location("stage_target_export_reuse_under_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def transfer(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path / "run"))
    air, inventory, receipt = (tmp_path / name for name in ("export.air", "inventory.json", "result.json"))
    air.write_bytes(b"small host fixture AIR; no model")
    schema = {"descriptor_sha256": "a" * 64, "schema_package": "ge.proto", "schema_file": "ge_ir.proto",
              "selection": "EXPLICIT_TORCHAIR_PB2"}
    manifest = {"artifact_kind": "qwen35-target-export-inventory", "status": "INVENTORIED",
                "weight_identity": "SHA256_COMPLETE", "schema": schema,
                "air": {"path": "/origin/export.air", "bytes": air.stat().st_size,
                        "sha256": hashlib.sha256(air.read_bytes()).hexdigest()}}
    inventory.write_text(json.dumps(manifest))
    names = [("arg1_1", [1, 64, 2048]), ("arg113_1", [1, 1, 64, 2048]), ("arg107_1", [1, 64]),
             ("arg2_1", [64]), ("arg1238_1", [1]), ("arg0_1", [1, 64, 1]),
             ("arg114_1", [1]), ("arg14_1", [128]), ("arg18_1", [1])]
    result = {"artifact_kind": "qwen35-canonical-target-export", "status": "EXPORT_READY",
        "air": dict(manifest["air"], path="/another/origin/export.air"),
        # Real canonical results do not currently contain inventory.bytes.
        "inventory": {"path": "/origin/inventory/inventory.json", "sha256": hashlib.sha256(inventory.read_bytes()).hexdigest()},
        "schema": dict(schema, selection="UNIQUE_DISCOVERED_DESCRIPTOR_HASH_SMALLEST_LITERAL_SOURCE"),
        "normalization": {"status": "OUTPUT_ORDER_CANONICALIZED", "sequence_length": 64,
            "patched_sha256": manifest["air"]["sha256"], "schema": dict(schema),
            "input_validation": {"status": "MATCHED_TYPED_INPUT_GEOMETRIES", "inputs": [
                {"name": name, "type": "Data", "shape": shape} for name, shape in names]}}}
    receipt.write_text(json.dumps(result))
    weights = tmp_path / "old55"
    weights.mkdir()
    (weights / "large-weight-must-not-open").write_bytes(b"not read by preflight")
    return {"air": air, "inventory": inventory, "receipt": receipt, "result": result,
            "manifest": manifest, "weights": weights, "output": tmp_path / "run/staged"}


def save(transfer, *, rebind_inventory=False):
    transfer["inventory"].write_text(json.dumps(transfer["manifest"]))
    if rebind_inventory:
        transfer["result"]["inventory"]["sha256"] = hashlib.sha256(transfer["inventory"].read_bytes()).hexdigest()
    transfer["receipt"].write_text(json.dumps(transfer["result"]))


def argv(transfer, *, receipt=True):
    result = ["--inventory", str(transfer["inventory"]), "--air", str(transfer["air"]),
              "--existing-export", str(transfer["weights"]), "--output", str(transfer["output"])]
    return result + ["--export-receipt", str(transfer["receipt"])] if receipt else result


def staging_report(transfer):
    return {"status": "PLAN_READY", "air": {"sha256": hashlib.sha256(transfer["air"].read_bytes()).hexdigest()},
        "inventory": {"sha256": hashlib.sha256(transfer["inventory"].read_bytes()).hexdigest()},
        "files": [], "transfer_required": [], "compiler_budget": "PENDING"}


def guard_weight_reads(monkeypatch):
    original = os.open
    opened = []
    def checked(path, *args, **kwargs):
        assert "large-weight-must-not-open" not in str(path)
        opened.append(str(path))
        return original(path, *args, **kwargs)
    monkeypatch.setattr(reuse.os, "open", checked)
    return opened


def test_bound_three_file_receipt_matches_names_without_cross_machine_path_or_selection_equality(cli, transfer, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(reuse, "stage_target_export_reuse", lambda *a, **k: calls.append(a) or staging_report(transfer))
    guard_weight_reads(monkeypatch)
    assert cli.main(argv(transfer)) == 0 and len(calls) == 1
    report = json.loads(capsys.readouterr().out)
    check = report["export_profile_check"]
    assert check["status"] == "MATCHED_EXACT_DATA_NAMES_AND_DECLARED_GEOMETRIES"
    assert check["export_receipt"]["sha256"] == hashlib.sha256(transfer["receipt"].read_bytes()).hexdigest()
    assert "control identity" in check["scope"] and "unverified" in check["scope"]


def test_old_cli_without_optional_third_file_remains_compatible(cli, transfer, monkeypatch, capsys):
    calls = []
    transfer["receipt"].unlink()
    monkeypatch.setattr(reuse, "stage_target_export_reuse", lambda *a, **k: calls.append(a) or {"status": "PLAN_READY"})
    assert cli.main(argv(transfer, receipt=False)) == 0 and len(calls) == 1
    assert "export_profile_check" not in json.loads(capsys.readouterr().out)


@pytest.mark.parametrize("kind", ["unknown_name", "duplicate_Data", "missing_Data", "fixed_extent", "dynamic_wrong", "float_extent", "bool_extent", "no_input_metadata", "wrong_receipt_kind", "wrong_status", "AIR_tamper", "inventory_tamper", "inventory_size", "schema_changed", "normalization_schema", "wrong_patch_sha", "unhashed_inventory"])
def test_bad_preflight_never_calls_staging_or_opens_weights(cli, transfer, monkeypatch, capsys, kind):
    result = transfer["result"]
    rows = result["normalization"]["input_validation"]["inputs"]
    if kind == "unknown_name": rows[4]["name"] = "arg9999_1"
    elif kind == "duplicate_Data": rows.append(deepcopy(rows[0]))
    elif kind == "missing_Data": rows.pop()
    elif kind == "fixed_extent": rows[0]["shape"][-1] = 1024
    elif kind == "dynamic_wrong": rows[0]["shape"][1] = 55
    elif kind == "float_extent": rows[0]["shape"][1] = 64.0
    elif kind == "bool_extent": rows[4]["shape"][0] = True
    elif kind == "no_input_metadata": result["normalization"].pop("input_validation")
    elif kind == "wrong_receipt_kind": result["artifact_kind"] = "unrelated-receipt"
    elif kind == "wrong_status": result["status"] = "BLOCKED"
    elif kind == "inventory_size": result["inventory"]["bytes"] = 1
    elif kind == "schema_changed": transfer["manifest"]["schema"]["descriptor_sha256"] = "b" * 64
    elif kind == "normalization_schema": result["normalization"]["schema"]["descriptor_sha256"] = "b" * 64
    elif kind == "wrong_patch_sha": result["normalization"]["patched_sha256"] = "b" * 64
    elif kind == "unhashed_inventory": transfer["manifest"]["weight_identity"] = "STAT_ONLY"
    save(transfer, rebind_inventory=kind in {"schema_changed", "unhashed_inventory"})
    if kind == "AIR_tamper": transfer["air"].write_bytes(b"X" * transfer["air"].stat().st_size)
    if kind == "inventory_tamper": transfer["inventory"].write_bytes(transfer["inventory"].read_bytes() + b"\n")
    monkeypatch.setattr(reuse, "stage_target_export_reuse", lambda *a, **k: pytest.fail("preflight must block before staging"))
    guard_weight_reads(monkeypatch)
    assert cli.main(argv(transfer)) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "BLOCKED" and report["stage"] == "export-profile-preflight"
    if kind in {"unknown_name", "duplicate_Data", "missing_Data", "fixed_extent", "dynamic_wrong", "float_extent", "bool_extent"}:
        assert "PROFILE_BINDING_REQUIRED" in report["error"]
    assert not transfer["output"].exists()


def test_profile_dynamic_axes_accept_only_declared_64_or_actual_dynamic_minus_one(cli, transfer, monkeypatch, capsys):
    rows = transfer["result"]["normalization"]["input_validation"]["inputs"]
    for index, axis in ((0, 1), (1, 2), (2, 1), (3, 0), (5, 1)):
        rows[index]["shape"][axis] = -1
    save(transfer)
    monkeypatch.setattr(reuse, "stage_target_export_reuse", lambda *a, **k: staging_report(transfer))
    assert cli.main(argv(transfer)) == 0
    assert json.loads(capsys.readouterr().out)["export_profile_check"]["status"].startswith("MATCHED")


@pytest.mark.parametrize("kind", ["receipt_changed", "stage_AIR_changed", "stage_inventory_changed"])
def test_post_stage_binding_failure_preserves_stage_receipt_and_links(cli, transfer, monkeypatch, capsys, kind):
    def staging(*args, **kwargs):
        report = staging_report(transfer)
        transfer["output"].mkdir(parents=True)
        (transfer["output"] / "reuse-receipt.json").write_bytes(b"valid retained stage evidence")
        (transfer["output"] / "retained-link").write_bytes(b"retained generated content")
        report.update(status="STAGED_HARDLINKS", receipt={"path": str(transfer["output"] / "reuse-receipt.json")})
        if kind == "receipt_changed":
            transfer["receipt"].write_bytes(transfer["receipt"].read_bytes() + b"\n")
        elif kind == "stage_AIR_changed": report["air"]["sha256"] = "c" * 64
        else: report["inventory"]["sha256"] = "c" * 64
        return report
    monkeypatch.setattr(reuse, "stage_target_export_reuse", staging)
    guard_weight_reads(monkeypatch)
    assert cli.main(argv(transfer) + ["--apply"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["stage"] == "export-profile-postcheck"
    assert report["stage_result_status"] == "STAGED_HARDLINKS"
    assert report["export_profile_check"]["status"] == "BINDING_CHANGED_AFTER_STAGE"
    assert (transfer["output"] / "reuse-receipt.json").read_bytes() == b"valid retained stage evidence"
    assert (transfer["output"] / "retained-link").read_bytes() == b"retained generated content"
