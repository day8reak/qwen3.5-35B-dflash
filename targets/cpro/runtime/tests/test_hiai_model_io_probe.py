"""Restore-only ordered-descriptor observations through host HIAI test doubles.

The doubles are not selected vendor headers and do not prove device support,
external-weight restoration, semantic input identities or model correctness.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

runner = Path(sys.argv[1]).resolve()
base = Path(sys.argv[2]); base.mkdir(parents=True, exist_ok=True)
root = Path(tempfile.mkdtemp(prefix="model-io-", dir=base)).resolve()
model = root / 'model "quoted"\\name.omc'
model.write_bytes(b"draft-standalone-fixture")
model_hash = hashlib.sha256(model.read_bytes()).hexdigest()


def command(path=model, digest=model_hash, version="FAKE-HIAI-X90"):
    return [str(runner), "--model-io-only", str(path), "--model-sha256", digest,
            "--descriptor-stage", "restore-only",
            "--expected-runtime-version", version, "--ddk-id", 'fake "ddk"',
            "--firmware-id", 'fake\\firmware']


def execute(label, failure="", *, args=None, output=None, layout=""):
    destination = output or root / (label + ".json")
    trace = root / (label + ".trace")
    proc = subprocess.run((args or command()) + ["--output", str(destination)],
        capture_output=True, text=True, cwd=root,
        env=dict(os.environ, FAKE_HIAI_FAILURE=failure, FAKE_HIAI_LAYOUT=layout,
                 FAKE_HIAI_TRACE=str(trace)))
    events = trace.read_text().splitlines() if trace.exists() else []
    assert not any(event in events for event in ("process", "load", "unload", "v2-run", "v2-create-tensor",
                                               "v2-create-manager", "v2-init", "v2-deinit")), events
    assert "LEAK" not in proc.stderr, proc.stderr
    return proc, destination, events


# Allocation/inference failures are never reached, because this route calls
# neither CreateModelManager, Init, CreateNDTensorBuffer nor IModelManager::Run.
for failure, layout in [("", ""), ("", "padded"), ("", "leading-feature"),
                        ("v2-run", ""), ("v2-allocation", ""), ("v2-short-buffer", ""),
                        ("v2-create-manager", ""), ("v2-init", "")]:
    label = (failure or "success") + ("-" + layout if layout else "")
    proc, output, events = execute(label, failure, layout=layout)
    assert proc.returncode == 0, proc.stderr
    report = json.loads(output.read_text())
    assert report == json.loads(proc.stdout)
    assert report["artifact_kind"] == "qwen35-hiai-model-io-probe"
    assert report["evidence_kind"] == "host-fake-hiai"
    assert report["status"] == "IO_DESCRIPTORS_OBSERVED" and report["restore"] == "PASSED"
    assert report["descriptor_stage"] == "restore-only" and report["manager_init"] == "NOT_RUN"
    assert report["restore_init"] == "NOT_RUN_METADATA_ONLY"
    assert report["model_load_scope"] == "RESTORE_METADATA_ONLY"
    assert report["external_weight_initialization"] == "NOT_REQUESTED"
    assert report["model_path"] == str(model) and report["model_sha256"] == model_hash
    assert report["model_bytes"] == model.stat().st_size
    assert report["model_load_api"] == "file-v2" and report["runtime_version"] == "FAKE-HIAI-X90"
    assert report["runtime_input_order"] == "ORDERED_DESCRIPTOR_INDICES_ONLY"
    assert report["semantic_slot_identity"] == "UNVERIFIED"
    assert report["runtime_tensor_names"] == "NOT_EXPOSED_BY_SELECTED_ADAPTER"
    assert report["dtype_enum"] == "hiai::DataType" and report["format_enum"] == "hiai::Format"
    assert report["dtype_enum_values"] == {"FLOAT16": 2, "FLOAT32": 1, "INT32": 3, "INT16": 5, "INT64": 7, "INT8": 4}
    assert report["format_enum_values"] == {"ND": 2}
    assert report["tensor_allocations_requested"] == 0
    assert report["process_called"] is False and report["run_called"] is False
    assert report["numerical_validation"] == "NOT_RUN"
    assert report["input_count"] == 3 and report["output_count"] == 2
    assert [row["index"] for row in report["inputs"]] == [0, 1, 2]
    assert [row["raw_dtype"] for row in report["inputs"]] == [2, 3, 3]
    assert [row["raw_dtype"] for row in report["outputs"]] == [3, 2]
    expected = [1, 8, 16384]
    if layout == "padded": expected += [1]
    if layout == "leading-feature": expected = [1] + expected
    assert report["inputs"][0]["shape"] == expected
    assert events.index("v2-restore-file") < events.index("v2-built-model-destroy")


# Descriptor observation preserves actual values, including an unknown-to-the-
# draft shape, alternate format/dtype and unresolved zero dimension. It does
# not silently coerce them to the frozen ABI or allocate based on their size.
for failure, field, expected in [
        ("v2-dtype", "raw_dtype", 1), ("v2-format", "raw_format", 0),
        ("v2-shape", "shape", [1, 4, 32768]), ("v2-zero-dim", "shape", [1, 0, 16384])]:
    proc, output, _ = execute(failure, failure)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(output.read_text())["inputs"][0][field] == expected


for failure, stage in [
        ("no-npu", "RUNTIME_IDENTITY"), ("v2-create-model", "CREATE_BUILT_MODEL"),
        ("v2-restore", "RESTORE_FROM_FILE"), ("v2-count", "TENSOR_DESCRIPTORS")]:
    proc, output, events = execute(failure, failure)
    assert proc.returncode == 1, proc.stderr
    report = json.loads(output.read_text())
    assert report["status"] == "MODEL_IO_FAILED" and report["runtime_stage"] == stage
    assert report["dtype_enum_values"]["FLOAT16"] == 2 and report["format_enum_values"] == {"ND": 2}
    assert report["inputs"] == report["outputs"] == []
    assert report["manager_init"] == "NOT_RUN" and report["restore_init"] == "NOT_RUN_METADATA_ONLY"
    assert report["restore"] == ("PASSED" if stage == "TENSOR_DESCRIPTORS" else "FAILED_OR_NOT_REACHED")
    assert report["process_called"] is False and report["run_called"] is False

proc, output, events = execute("wrong-version", args=command(version="unmatched-runtime"))
assert proc.returncode == 1 and json.loads(output.read_text())["runtime_stage"] == "RUNTIME_IDENTITY"
assert "v2-restore-file" not in events

empty = root / "empty.omc"; empty.touch()
symlink = root / "model-link.omc"; symlink.symlink_to(model)
for label, args in [
        ("bad-hash", command(digest="0" * 64)), ("invalid-hash", command(digest="bad")),
        ("empty", command(path=empty)), ("missing", command(path=root / "missing.omc")),
        ("directory", command(path=root)), ("symlink", command(path=symlink)),
        ("unrelated", command() + ["--draft-step", "never-read"]),
        ("duplicate", command() + ["--model-sha256", model_hash]),
        ("stage-init", ["init" if value == "restore-only" else value for value in command()]),
        ("missing-stage", command()[:5] + command()[7:])]:
    proc, output, events = execute(label, args=args)
    assert proc.returncode == 1 and not output.exists() and not events, (label, proc.stderr, events)

existing = root / "existing.json"; existing.write_text("retained")
proc, _, events = execute("existing", output=existing)
assert proc.returncode == 1 and existing.read_text() == "retained" and not events
assert hashlib.sha256(model.read_bytes()).hexdigest() == model_hash
print("HIAI restore-only model IO probe: raw ordered descriptors, no manager Init/tensor allocation/inference, lifecycle faults and original CLI guards passed (host fake only)")
