"""V2 full draft transport, fault boundaries and ownership; no device/math claim."""
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

runner = Path(sys.argv[1]).resolve()
base = Path(sys.argv[2]); base.mkdir(parents=True, exist_ok=True)
root = Path(tempfile.mkdtemp(prefix="draft-v2-", dir=base)).resolve()
model = root / "model" / "model.omc"; model.parent.mkdir()
model.write_bytes(b"draft-standalone-fixture")
(root / "draft-abi.json").write_text("{}")
inputs = [
    ("target_features", "float16", "1,8,16384", struct.pack("<H", 0x3800) * (8 * 16384)),
    ("input_ids", "int32", "1,8", struct.pack("<8i", *range(8))),
    ("attention_mask", "int32", "1,8", struct.pack("<8i", 1, 1, 1, 1, 1, 1, 0, 0)),
]
sha = lambda data: hashlib.sha256(data).hexdigest()
rows = ["QWEN35_DRAFT_STEP\t1", "model\tmodel/model.omc\t" + sha(model.read_bytes()),
        "draft_abi\tdraft-abi.json\t" + sha(b"{}"), "tokens\t8\t0"]
for index, (name, dtype, shape, data) in enumerate(inputs):
    (root / (name + ".bin")).write_bytes(data)
    rows.append(f"input\t{index}\t{name}\t{dtype}\t{shape}\t{name}.bin\t{sha(data)}")
rows += ["output\t0\tdraft_top1\tint32\t1,15", "output\t1\tdraft_hidden\tfloat16\t1,15,2048"]
(root / "step.tsv").write_text("\n".join(rows) + "\n")
elsewhere = root / "elsewhere"; elsewhere.mkdir()
args = [str(runner), "--draft-step", str(root / "step.tsv"), "--model-load-api", "file-v2",
        "--expected-runtime-version", "FAKE-HIAI-X90", "--ddk-id", "fake", "--firmware-id", "fake"]

failures = {
    "v2-create-model": "CREATE_BUILT_MODEL", "v2-restore": "RESTORE_FROM_FILE",
    "v2-create-manager": "CREATE_MODEL_MANAGER", "v2-init": "MODEL_MANAGER_INIT",
    "v2-count": "TENSOR_DESCRIPTORS", "v2-dtype": "INPUT_TENSOR_ALLOCATION",
    "v2-format": "INPUT_TENSOR_ALLOCATION", "v2-shape": "INPUT_TENSOR_ALLOCATION",
    "v2-zero-dim": "INPUT_TENSOR_ALLOCATION", "v2-short-buffer": "INPUT_TENSOR_ALLOCATION",
    "v2-allocation": "OUTPUT_TENSOR_ALLOCATION", "v2-run": "MODEL_MANAGER_RUN",
}
for failure, layout in [("", ""), ("", "padded"), ("", "leading-feature"),
                        *[(x, "") for x in failures], ("v2-replace-output", "")]:
    label = (failure or "success") + ("-" + layout if layout else "")
    output = root / label
    trace = root / (label + ".trace")
    path_trace = root / (label + ".path")
    proc = subprocess.run(args + ["--output", str(output)], cwd=elsewhere, capture_output=True, text=True,
        env=dict(os.environ, FAKE_HIAI_FAILURE=failure, FAKE_HIAI_LAYOUT=layout,
                 FAKE_HIAI_TRACE=str(trace), FAKE_HIAI_PATH_TRACE=str(path_trace)))
    assert "LEAK" not in proc.stderr, proc.stderr
    result = json.loads((output / "result.json").read_text())
    assert result["model_load_api"] == "file-v2", result
    assert result["evidence_kind"] == "host-fake-hiai" and result["target_calls"] == 0
    events = trace.read_text().splitlines()
    assert "load" not in events and "process" not in events and "unload" not in events, events
    if failure:
        assert proc.returncode == 1 and result["status"] == "RUNTIME_FAILED", (proc.stderr, result)
        assert not result["outputs"] and result["draft_calls"] == 0
        if failure in failures:
            assert result["runtime_stage"] == failures[failure], (failure, result)
        if failure not in {"v2-run", "v2-replace-output"}:
            assert "v2-run" not in events and not result["io_diagnostic"]["process_called"], (failure, events)
        else:
            assert events.count("v2-run") == 1
            assert result["io_diagnostic"]["process_called"]
        if failure == "v2-replace-output":
            assert result["runtime_stage"] == "OUTPUT_VALIDATION"
            assert not result["io_diagnostic"]["outputs"][0]["object_same"]
            assert "refusing to read" in result["reason"]
    else:
        assert proc.returncode == 0 and result["status"] == "EXECUTED", (proc.stderr, result)
        assert result["draft_calls"] == 1 and events.count("v2-run") == 1
        assert result["runtime_stage"] == "EXECUTED"
        assert all(x["readback_matches_source"] for x in result["io_diagnostic"]["inputs"])
        assert all(x["object_same"] and x["buffer_same"] for x in result["io_diagnostic"]["outputs"])
        assert struct.unpack("<15i", (output / "output_0000.bin").read_bytes()) == tuple(range(10, 25))
        assert (output / "output_0001.bin").read_bytes() == struct.pack("<H", 0x3800) * (15 * 2048)
    if "v2-init" in events:
        assert events.index("v2-init") < events.index("v2-deinit") < events.index("v2-manager-destroy")
        assert events.index("v2-manager-destroy") < events.index("v2-built-model-destroy")
    if "v2-restore-file" in events:
        assert path_trace.read_text() == str(model)

# A wrong route or non-draft mode must be rejected before any runtime factory.
for label, override in [("bad-api", ["--draft-step", "--model-load-api", "unknown"]),
                        ("wrong-scope", ["--stateful-step", "--model-load-api", "file-v2"])]:
    changed = args[:]
    changed[1] = override[0]
    changed[4] = override[2]
    trace = root / (label + ".trace")
    proc = subprocess.run(changed + ["--output", str(root / label)], capture_output=True, text=True,
        env=dict(os.environ, FAKE_HIAI_TRACE=str(trace)))
    assert proc.returncode == 1 and not trace.exists(), proc.stderr
print("File V2 standalone draft full-run, descriptor, lifetime and failure tests passed (host fake only)")
