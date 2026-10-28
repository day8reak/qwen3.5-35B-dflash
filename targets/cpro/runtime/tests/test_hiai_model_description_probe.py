"""Observe real CLI setter order without invoking fake Load or Process."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

runner = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2])
root.mkdir(parents=True, exist_ok=True)
root = Path(tempfile.mkdtemp(prefix="case-", dir=root)).resolve()
model = root / 'draft "quoted"\\name\n.omc'
model.write_bytes(b"descriptor-only-fixture")


def execute(label, failure="", path=model, extra=()):
    trace = root / (label + ".trace")
    command = [str(runner), "--probe-model-description", str(path), *extra]
    result = subprocess.run(command, capture_output=True, text=True,
        env=dict(os.environ, FAKE_HIAI_FAILURE=failure, FAKE_HIAI_TRACE=str(trace)))
    events = trace.read_text().splitlines() if trace.exists() else []
    assert not events, (label, events, result.stderr)
    assert "LEAK" not in result.stderr, result.stderr
    return result


for failure in ("", "descriptor-exclusive", "model-buffer", "path", "load", "no-npu"):
    result = execute(failure or "preserve", failure)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["schema_version"] == 1
    assert report["artifact_kind"] == "qwen35-hiai-model-description-probe"
    assert report["status"] == "INSPECTED"
    assert report["evidence_kind"] == "host-fake-hiai"
    assert report["runtime_version"] == "FAKE-HIAI-X90"
    assert report["model_path"] == str(model)
    assert report["model_bytes"] == model.stat().st_size
    assert report["model_sha256"] == hashlib.sha256(model.read_bytes()).hexdigest()
    assert report["process_id"] > 0
    assert report["npu_supported"] is (failure != "no-npu")
    assert report["load_called"] is False and report["process_called"] is False
    orders = report["observations"]
    assert [row["order"] for row in orders] == ["path-then-buffer", "buffer-then-path"]
    first, second = (row["steps"] for row in orders)
    assert [row["operation"] for row in first] == ["SetModelPath", "SetModelBuffer"]
    assert [row["operation"] for row in second] == ["SetModelBuffer", "SetModelPath"]
    # The first snapshot in each independent order begins from a fresh object.
    assert first[0]["model_path"] == str(model)
    assert first[0]["model_bytes"] == 0 and first[0]["buffer_present"] is False
    assert second[0]["model_path"] == ""
    for row in first + second:
        failed = (failure == "path" and row["operation"] == "SetModelPath"
                  or failure == "model-buffer" and row["operation"] == "SetModelBuffer")
        assert row["setter_status"] == int(failed), (failure, row)
    if failure == "descriptor-exclusive":
        assert first[1]["model_path"] == ""
        assert first[1]["buffer_present"] is True and first[1]["buffer_matches_input"] is True
        assert first[1]["model_bytes"] == model.stat().st_size
        assert second[1]["model_path"] == str(model)
        assert second[1]["buffer_present"] is False and second[1]["buffer_matches_input"] is False
        assert second[1]["model_bytes"] == 0
    elif failure != "model-buffer":
        for row in (first[1], second[0], second[1]):
            assert row["buffer_present"] is True and row["buffer_matches_input"] is True
            assert row["model_bytes"] == model.stat().st_size
        assert first[1]["model_path"] == second[1]["model_path"] == str(model)
    else:
        assert all(not row["buffer_present"] and row["model_bytes"] == 0 for row in first + second)

empty = root / "empty.omc"
empty.touch()
large = root / "oversized.omc"
with large.open("wb") as handle:
    handle.truncate((64 << 20) + 1)
for label, path, message in (("empty", empty, "nonempty"),
                            ("oversized", large, "64 MiB"),
                            ("missing", root / "missing.omc", "regular"),
                            ("directory", root, "regular")):
    result = execute(label, path=path)
    assert result.returncode == 1 and message in result.stderr, (label, result.stderr)
    assert not result.stdout
result = execute("extra-option", extra=("--target", str(model)))
assert result.returncode == 1 and not result.stdout
print("HIAI descriptor probe: setter order, fresh descriptors, failure statuses, escaped paths, hashes, bounds, no Load/Process checked")
