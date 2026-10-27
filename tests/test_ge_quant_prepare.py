import hashlib
import json
from pathlib import Path
import runpy
import struct
import sys

import numpy as np
import pytest

from qwen35_moe_dflash import ge_quant_plan, ge_quant_prepare as prepare
from qwen35_moe_dflash.ge_quant_plan import target_tensor_quantization
from qwen35_moe_dflash.ge_weights import SafeTensorFiles


@pytest.fixture
def weight_case(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    specs = [target_tensor_quantization(key, shape) for key, shape in (
        ("target.embed_tokens.weight", (3, 256)),
        ("target.layers.0.mlp.experts.gate_up_proj", (2, 4, 256)),
        ("target.lm_head.weight", (4, 256)),
        ("target.norm.weight", (256,)),
    )]
    specs.sort(key=lambda x: x["key"])
    header, chunks, offset = {}, [], 0
    for i, spec in enumerate(specs):
        values = (np.arange(spec["parameters"], dtype=np.float32) % 15 - 7) / (i + 1)
        raw = (values.view("<u4") >> 16).astype("<u2").tobytes()
        header[spec["source_tensor"]] = {"dtype": "BF16", "shape": spec["shape"],
                                          "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
        chunks.append(raw)
    encoded = json.dumps(header).encode()
    shard = checkpoint / "model.safetensors"
    shard.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(chunks))
    (checkpoint / "config.json").write_text('{"test":true}')
    source_hash = hashlib.sha256((checkpoint / "config.json").read_bytes()).hexdigest()
    repo = tmp_path / "repo"
    (repo / "specs").mkdir(parents=True)
    (repo / "specs/upstream.lock.json").write_text(json.dumps([{
        "repo": "fixture", "revision": "fixture", "files": {"config.json": {"sha256": source_hash}}}]))
    monkeypatch.setattr(prepare, "repository_root", lambda: repo)
    plan = {"schema_version": 1, "status": "PORTABLE_QUANTIZATION_PLAN", "tensors": specs,
            "summary": {"tensor_count": len(specs),
                        "payload_bytes": sum(x["weight_bytes"] + x["scale_bytes"] for x in specs)}}
    monkeypatch.setattr(ge_quant_plan, "target_quantization_plan", lambda: plan)
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    return checkpoint, plan, shard


def _files(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_full_plan_preparation_records_all_tensors_and_preserves_checkpoint(weight_case, tmp_path):
    checkpoint, plan, _ = weight_case
    original = _files(checkpoint)
    result = prepare.prepare_target_weights(checkpoint, tmp_path / "prepared")
    assert result["status"] == "WEIGHTS_PREPARED_LAYOUT_PENDING"
    assert result["device_ready"] is False and result["omg_compile"] == "NOT_RUN"
    assert len(result["tensors"]) == 4
    reader = SafeTensorFiles(checkpoint)
    for spec, entry in zip(plan["tensors"], result["tensors"]):
        record = json.loads((tmp_path / "prepared" / entry["path"]).read_text())
        assert record["spec"] == spec
        payload = record["converted"]
        assert payload["source"]["sha256"] == prepare._source_hash(reader, spec)
        for field in ("weights", "scales"):
            if field in payload:
                assert not Path(payload[field]["path"]).is_absolute()
                data = (tmp_path / "prepared" / payload[field]["path"]).read_bytes()
                assert hashlib.sha256(data).hexdigest() == payload[field]["sha256"]
        if spec["kind"] == "float16":
            data = (tmp_path / "prepared" / payload["weights"]["path"]).read_bytes()
            # Norm is the original HF offset, not folded again during packing.
            assert np.frombuffer(data, dtype="<f2")[0] == -1.75
    assert _files(checkpoint) == original


def test_resume_verifies_and_skips_completed_conversion_keeps_old_files(weight_case, tmp_path, monkeypatch):
    checkpoint, _, _ = weight_case
    writer = prepare.write_int4

    def fail(*args, **kwargs):
        Path(args[3]).write_bytes(b"partial output")
        raise RuntimeError("interrupted conversion")

    monkeypatch.setattr(prepare, "write_int4", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        prepare.prepare_target_weights(checkpoint, tmp_path / "old")
    before = _files(tmp_path / "old")
    assert json.loads(before["failure.json"])["completed_tensors"] == 1
    monkeypatch.setattr(prepare, "write_int4", writer)
    monkeypatch.setattr(prepare, "write_embedding_int8", lambda *a, **k: pytest.fail("completed tensor re-quantized"))
    events = []
    result = prepare.prepare_target_weights(checkpoint, tmp_path / "new", resume=tmp_path / "old", progress=events.append)
    assert result["status"] == "WEIGHTS_PREPARED_LAYOUT_PENDING"
    assert [e["reused"] for e in events] == [True, False, False, False]
    assert _files(tmp_path / "old") == before


@pytest.mark.parametrize("tamper", ["source", "cache", "record", "plan"])
def test_resume_rejects_changed_source_payload_or_contract(weight_case, tmp_path, tamper):
    checkpoint, _, shard = weight_case
    prepare.prepare_target_weights(checkpoint, tmp_path / "old")
    if tamper == "source":
        reader = SafeTensorFiles(checkpoint)
        spec = json.loads((tmp_path / "old/tensors/00000.json").read_text())["spec"]
        _, offset, _, _ = reader.tensor(spec["source_tensor"], spec["shape"])
        with shard.open("r+b") as stream:
            stream.seek(offset)
            stream.write(b"\0\0")
    elif tamper == "cache":
        (tmp_path / "old/weights/00000.weight.bin").write_bytes(b"corrupt")
    elif tamper == "record":
        record = tmp_path / "old/tensors/00000.json"
        data = json.loads(record.read_text())
        data["converted"]["device_pack"] = "I4ZN"
        record.write_text(json.dumps(data))
    else:
        path = tmp_path / "old/plan.json"
        data = json.loads(path.read_text())
        data["contract"]["arithmetic"]["statistics"] = "float16"
        path.write_text(json.dumps(data))
    before = _files(tmp_path / "old")
    with pytest.raises(ValueError):
        prepare.prepare_target_weights(checkpoint, tmp_path / "new", resume=tmp_path / "old")
    assert not (tmp_path / "new/manifest.json").exists()
    assert _files(tmp_path / "old") == before


def test_original_checkpoint_metadata_mismatch_rejects_before_output(weight_case, tmp_path):
    checkpoint, _, _ = weight_case
    (checkpoint / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="locked revision"):
        prepare.prepare_target_weights(checkpoint, tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()


def test_checkpoint_change_during_preparation_cannot_produce_success(weight_case, tmp_path, monkeypatch):
    checkpoint, _, shard = weight_case
    writer = prepare.write_fp16

    def change(*args, **kwargs):
        result = writer(*args, **kwargs)
        with shard.open("ab") as stream:
            stream.write(b"change after conversion")
        return result

    monkeypatch.setattr(prepare, "write_fp16", change)
    with pytest.raises(ValueError, match="checkpoint changed"):
        prepare.prepare_target_weights(checkpoint, tmp_path / "prepared")
    assert not (tmp_path / "prepared/manifest.json").exists()


def _run_cli(monkeypatch, *args):
    script = Path(__file__).resolve().parents[1] / "targets/cpro/scripts/prepare_full_target.py"
    monkeypatch.setattr(sys, "argv", [str(script), *map(str, args)])
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(script), run_name="__main__")
    return result.value.code


def test_single_entry_prepares_and_resumes_without_draft_or_board(weight_case, tmp_path, monkeypatch):
    checkpoint, _, _ = weight_case
    ddk = tmp_path / "ddk"
    ddk.mkdir()
    assert _run_cli(monkeypatch, "--target-weights", checkpoint, "--ddk-root", ddk,
                    "--output", tmp_path / "job") == 0
    report = json.loads((tmp_path / "job/report.json").read_text())
    assert report["status"] == "WEIGHTS_PREPARED_LAYOUT_PENDING"
    assert report["model_compilation"] == report["device_execution"] == "NOT_RUN"
    monkeypatch.delenv("QWEN35_TARGET_WEIGHTS", raising=False)
    assert _run_cli(monkeypatch, "--ddk-root", ddk, "--resume", tmp_path / "job",
                    "--output", tmp_path / "job-resumed") == 0
    records = sorted((tmp_path / "job-resumed/weights/tensors").glob("*.json"))
    assert all(json.loads(path.read_text())["reused_from"] for path in records)


def test_cli_existing_output_does_not_write_failure_into_old_job(weight_case, tmp_path, monkeypatch):
    checkpoint, _, _ = weight_case
    ddk = tmp_path / "ddk"
    ddk.mkdir()
    old = tmp_path / "old"
    old.mkdir()
    (old / "existing.json").write_text("original")
    before = _files(old)
    assert _run_cli(monkeypatch, "--target-weights", checkpoint, "--ddk-root", ddk,
                    "--output", old) == 1
    assert _files(old) == before


def test_cli_inspect_only_does_not_read_checkpoint_and_failed_gdn_does_not_block_weights(weight_case, tmp_path, monkeypatch):
    checkpoint, _, _ = weight_case
    ddk = tmp_path / "ddk"
    ddk.mkdir()
    assert _run_cli(monkeypatch, "--target-weights", tmp_path / "missing", "--ddk-root", ddk,
                    "--output", tmp_path / "inspect", "--inspect-only") == 0
    assert not (tmp_path / "inspect/weights").exists()
    from qwen35_moe_dflash.ge_fused import gdn_diagnostics
    monkeypatch.setattr(gdn_diagnostics, "summarize_retained_diagnosis",
                        lambda *a, **k: (_ for _ in ()).throw(ValueError("old evidence unavailable")))
    assert _run_cli(monkeypatch, "--target-weights", checkpoint, "--ddk-root", ddk,
                    "--output", tmp_path / "with-gdn", "--gdn-diagnostic", tmp_path / "missing") == 0
    report = json.loads((tmp_path / "with-gdn/report.json").read_text())
    assert "old evidence unavailable" in report["gdn_summary_error"]
    assert report["status"] == "WEIGHTS_PREPARED_LAYOUT_PENDING"
