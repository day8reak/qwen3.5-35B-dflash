"""Selected-row host fixtures: no checkpoint, vendor import or model execution."""
import hashlib
import json
from pathlib import Path
import struct

import pytest
import yaml

from qwen35_moe_dflash import original_embedding_probe as probe
from qwen35_moe_dflash.target_prefill_output_inventory import inspect_target_prefill_outputs
from qwen35_moe_dflash.target_prefill_smoke_plan import _source
from test_target_prefill_output_inventory import case as retained_case, refresh


def save(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def embedding_case(retained_case, tmp_path):
    case = retained_case
    source = Path(case["plan"]["input_dir"])
    source.mkdir()
    ids = [7, 11, 17, 27, 42, 79, 101, 220] * 8
    (source / "input_ids.bin").write_bytes(struct.pack("<64i", *ids))
    resources = tmp_path / "original-resources"
    resources.mkdir()
    weights, factors = resources / "original-int8.bin", resources / "original-scale.bin"
    rows, scales = {}, {}
    # Sparse fixed-extent tables avoid writing/copying a real508MB payload.
    for path, size in ((weights, probe._VOCAB * probe._HIDDEN), (factors, probe._VOCAB * 4)):
        with path.open("wb") as stream:
            stream.truncate(size)
    for token in sorted(set(ids) | {0}):
        rows[token] = bytes((token + item) % 256 for item in range(probe._HIDDEN))
        scales[token] = struct.pack("<f", .001 * (token + 1))
        with weights.open("r+b") as stream:
            stream.seek(token * probe._HIDDEN); stream.write(rows[token])
        with factors.open("r+b") as stream:
            stream.seek(token * 4); stream.write(scales[token])
    (source / "inputs_embeds.bin").write_bytes(b"".join(rows[token] for token in ids))
    (source / "embed_scale.bin").write_bytes(b"".join(scales[token] for token in ids))
    records = [_source(source / filename, width) for filename, width in
        (("inputs_embeds.bin", 64 * probe._HIDDEN), ("embed_scale.bin", 64 * 4))]
    case["plan"]["source_inputs"] = records
    case["plan"]["inputs"] = [{} for _ in range(89)]
    for index, semantic, name, dtype, size in ((0, "inputs_embeds", "arg1_1", "INT8", 64 * probe._HIDDEN),
            (85, "embed_scale", "arg0_1", "FLOAT32", 64 * 4)):
        record = next(row for row in records if Path(row["path"]).name == semantic + ".bin")
        case["plan"]["inputs"][index] = {"index": index, "semantic": semantic,
            "compiled_candidates": [name], "dtype": dtype, "bytes": size,
            "source": record["path"], "sha256": record["sha256"]}
    refresh(case)
    config = resources / "original.yaml"
    config.write_text(yaml.safe_dump({"embedding_weight_path": weights.name, "embedding_scale_path": factors.name}))
    inspection = tmp_path / "bound-output-inspection"
    report = inspect_target_prefill_outputs(case["execution"], inspection, input_dir=source)
    assert report["status"] == "OUTPUT_FILES_BOUND_TO_LOGGED_SLOTS", report
    return {"case": case, "source": source, "weights": weights, "factors": factors,
            "ids": ids, "rows": rows, "scales": scales, "config": config,
            "inspection": inspection, "report": report, "out": tmp_path / "embedding-inspection"}


def run(case):
    return probe.inspect_original_embedding_rows(case["inspection"], case["config"], case["out"])


def stamp(path):
    row = path.stat()
    return (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns, row.st_ctime_ns, row.st_nlink)


def test_exact_original_rows_bound_without_full_table_read_or_mutation(embedding_case, monkeypatch):
    case = embedding_case
    sources = [case[key] for key in ("weights", "factors", "config")]
    sources += list(case["source"].iterdir())
    before = {path: stamp(path) for path in sources}
    pread = probe.os.pread
    reads = []
    def bounded(fd, size, offset):
        reads.append((size, offset))
        assert size in {2048, 4}
        return pread(fd, size, offset)
    monkeypatch.setattr(probe.os, "pread", bounded)
    result = run(case)
    assert result["status"] == "ORIGINAL_EMBEDDING_ROWS_BOUND_NEXT_TOKEN_PREPARED", result
    assert result["selected_next_token_id"] == 0
    assert result["prefix_comparison"]["status"] == "BYTE_IDENTICAL"
    assert result["prefix_comparison"]["positions"] == 64
    assert len(reads) == 18 and sum(size for size, _ in reads) == 9 * 2052
    assert result["observed_payload_reads"] == {"bound_logits_bytes": 993280,
        "original_prefix_input_bytes": 131584, "table_bytes": 18468}
    assert (case["out"] / "next-token/inputs_embeds.bin").read_bytes() == case["rows"][0]
    assert (case["out"] / "next-token/embed_scale.bin").read_bytes() == case["scales"][0]
    assert (case["out"] / "next-token/input_ids.bin").read_bytes() == struct.pack("<i", 0)
    assert sum(row["bytes"] for row in result["next_token_inputs"]) == 2056
    assert result["full_table_hash"] == "NOT_PERFORMED"
    assert result["runtime_control_recipe"] == result["state_feedback"] == "UNVERIFIED"
    assert result["source_preservation"] == "UNCHANGED_METADATA_PREFIX_INPUTS_LOGITS_AND_SELECTED_TABLE_IDENTITIES"
    assert all(stamp(path) == info for path, info in before.items())
    assert result["device_execution"] == result["model_compilation"] == "NOT_RUN"


@pytest.mark.parametrize("field", ["weights", "factors"])
def test_missing_original_table_has_explicit_receipt_without_next_inputs(embedding_case, field):
    case = embedding_case
    path = case[field]
    path.unlink()
    result = run(case)
    assert result["status"] == "ORIGINAL_EMBEDDING_TABLES_UNAVAILABLE", result
    assert result["missing_paths"] == [str(path)]
    assert result["next_token_inputs"] == [] and not (case["out"] / "next-token").exists()
    assert json.loads((case["out"] / "report.json").read_text())["status"] == result["status"]


@pytest.mark.parametrize("field", ["weights", "factors"])
def test_changed_prefix_table_rows_do_not_prepare_unbound_next_token(embedding_case, field):
    case = embedding_case
    width = probe._HIDDEN if field == "weights" else 4
    with case[field].open("r+b") as stream:
        stream.seek(case["ids"][0] * width)
        stream.write(b"different"[:width] if field == "weights" else struct.pack("<f", .9))
    result = run(case)
    assert result["status"] == "ORIGINAL_EMBEDDING_ROWS_DIFFER", result
    assert result["prefix_comparison"]["mismatches"]
    assert result["next_token_inputs"] == [] and not (case["out"] / "next-token").exists()


@pytest.mark.parametrize("fault", ["ids", "prefix", "logits", "plan", "winner", "mapping", "wrong_size", "symlink", "yaml", "nan_scale"])
def test_changed_or_invalid_source_evidence_fails_before_preparing_next_inputs(embedding_case, fault):
    case = embedding_case
    if fault == "ids": (case["source"] / "input_ids.bin").write_bytes(struct.pack("<64i", *([1] * 64)))
    elif fault == "prefix": (case["source"] / "inputs_embeds.bin").write_bytes(bytes(64 * probe._HIDDEN))
    elif fault == "logits":
        path = case["case"]["board"] / case["case"]["paths"][0]
        with path.open("r+b") as stream: stream.write(struct.pack("<f", 1))
    elif fault == "plan":
        case["case"]["plan"]["input_dir"] = str(case["source"] / "other")
        refresh(case["case"])
    elif fault in {"winner", "mapping"}:
        row = case["report"]
        if fault == "winner": row["bound_tensors"]["logits"]["statistics"]["argmax"]["token_id"] = 1
        else: row["logged_slot_mapping"]["entries"][0]["path"] = "outputs/not_this_file"
        save(case["inspection"] / "report.json", row)
    elif fault == "wrong_size":
        with case["weights"].open("r+b") as stream: stream.truncate(2048)
    elif fault == "symlink":
        other = case["weights"].with_name("other.bin")
        case["weights"].rename(other); case["weights"].symlink_to(other)
    elif fault == "yaml": case["config"].write_text("[unclosed")
    else:
        with case["factors"].open("r+b") as stream: stream.write(struct.pack("<f", float("nan")))
    result = run(case)
    assert result["status"] == "FAILED", result
    assert result.get("error")
    assert result["next_token_inputs"] == [] and not (case["out"] / "next-token").exists()


def test_table_changes_during_selected_read_are_detected(embedding_case, monkeypatch):
    case = embedding_case
    original = probe.os.pread
    changed = False
    def racing(fd, size, offset):
        nonlocal changed
        result = original(fd, size, offset)
        if not changed:
            with case["weights"].open("r+b") as stream:
                stream.seek(probe._HIDDEN * 2); stream.write(b"x")
            changed = True
        return result
    monkeypatch.setattr(probe.os, "pread", racing)
    result = run(case)
    assert result["status"] == "FAILED" and "changed during" in result["error"], result
    assert result["next_token_inputs"] == []
