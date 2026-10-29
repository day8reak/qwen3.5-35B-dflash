"""Transferred decode metadata is inspectable without touching shared payloads."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest

from qwen35_moe_dflash.existing_air_profile import get_existing_air_profile
from qwen35_moe_dflash import target_export_reuse as reuse


@pytest.fixture
def cli():
    path = Path(__file__).resolve().parents[1] / "targets/cpro/scripts/stage_target_export_reuse.py"
    spec = importlib.util.spec_from_file_location("target_transfer_inspection_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def incoming(tmp_path, monkeypatch):
    """Small caller-owned metadata represents one export, never a model run."""
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path / "unused-run"))
    paths = {name: tmp_path / filename for name, filename in (
        ("air", "export.air"), ("inventory", "inventory.json"), ("receipt", "result.json"))}
    paths["air"].write_bytes(b"host-only AIR bytes; no private graph or model")
    origin = tmp_path / "origin-weights-must-not-open"
    origin.mkdir()
    files = []
    for number in range(2):
        name = f"large-weight-must-not-open-{number}"
        raw = f"small synthetic payload {number}".encode()
        (origin / name).write_bytes(raw)
        reference = f"./glm_export_air_cpro_40block_1_quant/{name}"
        files.append({"reference": reference, "references": [reference],
            "relative_path": reference[2:], "basename": name, "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(), "path": str(origin / name)})
    schema = {"descriptor_sha256": "a" * 64, "schema_package": "ge.proto", "schema_file": "ge_ir.proto"}
    manifest = {"artifact_kind": "qwen35-target-export-inventory", "status": "INVENTORIED",
        "weight_identity": "SHA256_COMPLETE", "source_relative_prefix": "glm_export_air_cpro_40block_1_quant",
        "air": {"path": "/unfollowed-origin/export.air", "bytes": paths["air"].stat().st_size,
                "sha256": hashlib.sha256(paths["air"].read_bytes()).hexdigest()},
        "schema": schema, "files": files, "unique_reference_count": len(files),
        "total_weight_bytes": sum(row["bytes"] for row in files),
        "declared_abi": {"status": "MATCHED_INTERFACE_METADATA", "input_count": 89,
            "output_count": 82, "feature_output_index": 81, "feature_shape": [1, 1, 16384],
            "feature_dtype": "DT_FLOAT16"}}
    rows = []
    for index, entry in enumerate(get_existing_air_profile("cpro-40block-decode-1")["input_shape"].split(";")):
        name, shape = entry.split(":", 1)
        rows.append({"name": name, "type": "RefData" if name.startswith("RefData_") else "Data",
            "shape": list(map(int, shape.split(","))), "raw_dtype": 2, "layout": "ND",
            "serialized_index": index})
    result = {"artifact_kind": "qwen35-canonical-target-export", "status": "EXPORT_READY",
        "air": deepcopy(manifest["air"]), "inventory": {"path": "/unfollowed-origin/inventory.json"},
        "schema": deepcopy(schema), "normalization": {"status": "OUTPUT_ORDER_CANONICALIZED",
            "sequence_length": 1, "patched_sha256": manifest["air"]["sha256"], "schema": deepcopy(schema),
            "input_validation": {"status": "MATCHED_TYPED_INPUT_GEOMETRIES", "input_count": 89,
                "input_type_counts": {"Data": 9, "RefData": 80}, "inputs": rows}}}
    case = dict(paths, manifest=manifest, result=result, origin=origin)
    save(case)
    return case


def save(case):
    case["inventory"].write_text(json.dumps(case["manifest"]))
    case["result"]["inventory"]["sha256"] = hashlib.sha256(case["inventory"].read_bytes()).hexdigest()
    case["receipt"].write_text(json.dumps(case["result"]))


def arguments(case, sequence=1):
    args = ["--inspect-transfer", "--air", str(case["air"]), "--inventory", str(case["inventory"]),
            "--export-receipt", str(case["receipt"])]
    return args if sequence is None else args + ["--sequence-length", str(sequence)]


def metadata_only(monkeypatch):
    """No writes, payload reads, staging, or external process is permitted."""
    old_open = os.open
    opened = []

    def checked_open(path, flags, *args, **kwargs):
        assert "must-not-open" not in str(path), "inspection must not access payload roots"
        assert not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        opened.append(str(path))
        return old_open(path, flags, *args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("transferred-metadata inspection must remain read-only and process-free")

    monkeypatch.setattr(os, "open", checked_open)
    for name in ("link", "chmod", "mkdir"):
        monkeypatch.setattr(os, name, forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    monkeypatch.setattr(Path, "chmod", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(reuse, "stage_target_export_reuse", forbidden)
    return opened


def invoke(cli, argv, capsys):
    try:
        code = cli.main(argv)
    except SystemExit as error:
        code = error.code
    captured = capsys.readouterr()
    report = json.loads(captured.out) if captured.out.strip() else None
    return code, report


@pytest.mark.parametrize("sequence", [1, 64])
def test_metadata_only_inspection_binds_both_static_export_gears(cli, incoming, monkeypatch, capsys, sequence):
    if sequence == 64:
        incoming["manifest"]["declared_abi"]["feature_shape"][1] = sequence
        incoming["result"]["normalization"]["sequence_length"] = sequence
        expected = get_existing_air_profile("cpro-40block-prefill-64")["input_shape"]
        shapes = {name: list(map(int, shape.split(",")))
                  for name, shape in (entry.split(":", 1) for entry in expected.split(";"))}
        for row in incoming["result"]["normalization"]["input_validation"]["inputs"]:
            row["shape"] = shapes[row["name"]]
        save(incoming)
    before = {path: (path.read_bytes(), path.stat()) for path in incoming["origin"].iterdir()}
    opened = metadata_only(monkeypatch)
    code, report = invoke(cli, arguments(incoming, sequence), capsys)
    assert code == 0 and report["status"] == "TRANSFER_INSPECTED"
    assert report["artifact_kind"] == "qwen35-target-export-transfer-inspection"
    assert report["mode"] == "METADATA_ONLY" and report["sequence_length"] == sequence
    assert report["weight_payload_access"] == "NONE"
    assert report["model_compilation"] == report["device_execution"] == "NOT_RUN"
    assert report["declared_abi"]["feature_shape"] == [1, sequence, 16384]
    assert report["unique_reference_count"] == 2
    assert report["total_weight_bytes"] == incoming["manifest"]["total_weight_bytes"]
    check = report["export_profile_check"]
    assert check["status"] == "MATCHED_EXACT_DATA_NAMES_AND_DECLARED_GEOMETRIES"
    assert len(check["observed_data_inputs"]) == 9
    assert check["air_sha256"] == incoming["manifest"]["air"]["sha256"]
    assert check["inventory_sha256"] == hashlib.sha256(incoming["inventory"].read_bytes()).hexdigest()
    assert {path: (path.read_bytes(), path.stat()) for path in incoming["origin"].iterdir()} == before
    assert any("export.air" in path for path in opened)
    assert not (incoming["air"].parent / "unused-run").exists()


@pytest.mark.parametrize("kind", ["name", "fixed_shape", "count", "duplicate", "dynamic_extent"])
def test_actual_data_rows_survive_candidate_profile_differences(cli, incoming, monkeypatch, capsys, kind):
    rows = incoming["result"]["normalization"]["input_validation"]["inputs"]
    chosen = next(row for row in rows if row["name"] == "arg1_1")
    if kind == "name":
        chosen["name"] = "arg999_1"
    elif kind == "fixed_shape":
        chosen["shape"][-1] = 4096
    elif kind == "count":
        rows.remove(next(row for row in rows if row["name"] == "arg18_1"))
    elif kind == "duplicate":
        rows.append(deepcopy(chosen))
    else:
        chosen["shape"][1] = -1
    save(incoming)
    metadata_only(monkeypatch)
    code, report = invoke(cli, arguments(incoming), capsys)
    assert code == 0 and report["status"] == "TRANSFER_INSPECTED"
    check = report["export_profile_check"]
    assert check["status"] == "PROFILE_BINDING_REQUIRED"
    assert check["profile_differences"] and chosen in check["observed_data_inputs"]
    assert len(check["observed_data_inputs"]) == sum(row["type"] == "Data" for row in rows)
    assert report["weight_payload_access"] == "NONE"


@pytest.mark.parametrize("item", ["receipt", "inventory", "air"])
def test_small_metadata_changed_between_reads_blocks_without_repair(cli, incoming, monkeypatch, capsys, item):
    original = reuse._read_small
    changed = False
    payload = incoming[item].read_bytes() + b"\n"

    def mutate_after_first_read(path, maximum):
        nonlocal changed
        result = original(path, maximum)
        if Path(path) == incoming[item] and not changed:
            incoming[item].write_bytes(payload)
            changed = True
        return result

    metadata_only(monkeypatch)
    monkeypatch.setattr(reuse, "_read_small", mutate_after_first_read)
    code, report = invoke(cli, arguments(incoming), capsys)
    assert changed and code != 0 and report["status"] == "BLOCKED"
    assert "changed" in report["error"]
    assert incoming[item].read_bytes() == payload


@pytest.mark.parametrize("kind", ["receipt_sequence", "abi_sequence", "abi_float", "abi_bool",
    "float_data_extent", "bool_data_extent", "malformed_data_shape", "missing_inputs", "wrong_receipt",
    "wrong_inventory_status", "unhashed_inventory", "schema", "normalization_schema", "patch_sha",
    "air_bytes", "inventory_sha", "invalid_reference", "invalid_payload_sha"])
def test_invalid_transfer_never_reaches_payloads_or_mutations(cli, incoming, monkeypatch, capsys, kind):
    manifest, result = incoming["manifest"], incoming["result"]
    normalization = result["normalization"]
    data = next(row for row in normalization["input_validation"]["inputs"] if row["type"] == "Data")
    if kind == "receipt_sequence": normalization["sequence_length"] = 64
    elif kind == "abi_sequence": manifest["declared_abi"]["feature_shape"][1] = 64
    elif kind == "abi_float": manifest["declared_abi"]["feature_shape"][1] = 1.0
    elif kind == "abi_bool": manifest["declared_abi"]["feature_shape"][1] = True
    elif kind == "float_data_extent": data["shape"][1] = 1.0
    elif kind == "bool_data_extent": data["shape"][1] = True
    elif kind == "malformed_data_shape": data["shape"] = "[1,1,2048]"
    elif kind == "missing_inputs": normalization["input_validation"].pop("inputs")
    elif kind == "wrong_receipt": result["artifact_kind"] = "not-an-export-receipt"
    elif kind == "wrong_inventory_status": manifest["status"] = "BLOCKED"
    elif kind == "unhashed_inventory": manifest["weight_identity"] = "STAT_ONLY"
    elif kind == "schema": manifest["schema"]["descriptor_sha256"] = "b" * 64
    elif kind == "normalization_schema": normalization["schema"]["descriptor_sha256"] = "b" * 64
    elif kind == "patch_sha": normalization["patched_sha256"] = "b" * 64
    elif kind == "air_bytes": manifest["air"]["bytes"] += 1
    elif kind == "invalid_reference": manifest["files"][0]["reference"] = "../../private"
    elif kind == "invalid_payload_sha": manifest["files"][0]["sha256"] = "not-a-sha"
    save(incoming)
    if kind == "inventory_sha":
        incoming["inventory"].write_bytes(incoming["inventory"].read_bytes() + b"\n")
    metadata_only(monkeypatch)
    code, report = invoke(cli, arguments(incoming), capsys)
    assert code != 0 and report["status"] == "BLOCKED"


@pytest.mark.parametrize("extra", [["--apply"], ["--existing-export", "/payload-must-not-open"],
    ["--output", "/output-must-not-open"]])
def test_inspection_rejects_all_staging_arguments(cli, incoming, monkeypatch, capsys, extra):
    metadata_only(monkeypatch)
    code, report = invoke(cli, arguments(incoming) + extra, capsys)
    assert code != 0


def test_inspection_requires_the_bound_export_receipt(cli, incoming, monkeypatch, capsys):
    args = arguments(incoming)
    offset = args.index("--export-receipt")
    del args[offset:offset + 2]
    metadata_only(monkeypatch)
    code, report = invoke(cli, args, capsys)
    assert code != 0


def test_decode_inspection_requires_explicit_one_token_selection(cli, incoming, monkeypatch, capsys):
    metadata_only(monkeypatch)
    code, report = invoke(cli, arguments(incoming, None), capsys)
    assert code != 0 and report["status"] == "BLOCKED"


def test_decode_staging_stays_rejected_before_payload_access(cli, incoming, monkeypatch, capsys):
    args = arguments(incoming)
    args.remove("--inspect-transfer")
    args += ["--existing-export", str(incoming["origin"]), "--output", str(incoming["origin"] / "new")]
    metadata_only(monkeypatch)
    code, report = invoke(cli, args, capsys)
    assert code != 0
