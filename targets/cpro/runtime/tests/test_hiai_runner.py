"""Execute the actual CLI/adapter against a deliberately fallible HIAI double."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
runner = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]); root.mkdir(parents=True, exist_ok=True)
root = Path(tempfile.mkdtemp(prefix="case-", dir=root))
target, draft = root / "target.omc", root / "draft.omc"
target.write_bytes(b"target-fixture"); draft.write_bytes(b"draft-fixture")
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
base = [str(runner), "--mode", "benchmark", "--target", str(target), "--target-sha256", sha(target),
        "--draft", str(draft), "--draft-sha256", sha(draft),
        "--expected-runtime-version", "FAKE-HIAI-X90", "--ddk-id", "fake", "--firmware-id", "fake",
        "--feature-width", "32", "--vocab-size", "64", "--prompt-token-ids", "1,2",
        "--pad-token-id", "0", "--eos-token-ids", "10", "--max-new-tokens", "12"]
probe = subprocess.run([str(runner), "--probe"], capture_output=True, text=True, check=True)
assert json.loads(probe.stdout) == {"runtime_version": "FAKE-HIAI-X90", "npu_supported": True}
for failure in ["", "corrupt-draft", "init", "version", "no-npu", "incompatible", "compat-query-error", "load",
                "path", "io-dims", "shape", "format", "count", "allocation", "short-buffer", "run", "token"]:
    report = root / (failure or "success")
    trace_path = root / ((failure or "success") + ".trace")
    result = subprocess.run(base + ["--output", str(report)],
        env=dict(os.environ, FAKE_HIAI_FAILURE=failure, FAKE_HIAI_TRACE=str(trace_path)),
        capture_output=True, text=True)
    assert "LEAK" not in result.stderr, result.stderr
    trace = trace_path.read_text().splitlines() if trace_path.exists() else []
    # The fixture's standalone query always throws. Path-based loading must
    # work without querying it; Load remains the authoritative failure gate.
    assert "compatibility-query" not in trace, (failure, trace, result.stderr)
    if failure in ("", "corrupt-draft", "compat-query-error"):
        assert result.returncode == 0, result.stderr
        assert trace.count("load") == trace.count("unload") == 2
        assert "process" in trace
        data = json.loads(report.read_text())
        assert data["evidence_kind"] == "host-fake-hiai"
        assert data["token_id_mismatches"] == data["eos_mismatches"] == 0
        assert data["ordinary"]["stable_generated_token_ids"] == list(range(3, 11))
        assert len(data["ordinary"]["measurements"]) == len(data["dflash"]["measurements"]) == 10
        assert data["ordinary"]["warmup"] == data["dflash"]["warmup"] == 3
    else:
        assert result.returncode == 1, (failure, result.returncode, result.stderr)
        assert not report.exists()
        if failure in ("load", "incompatible", "io-dims"):
            assert ("GetModelIOTensorDim" if failure == "io-dims" else "Load") in result.stderr
            assert trace == ["load", "unload"], (failure, trace)
        if failure == "path":
            assert "SetModelPath" in result.stderr
            assert trace == ["unload"], (failure, trace)
        if failure in ("path", "load", "incompatible", "io-dims"):
            for context in ("qwen35_target", str(target), "FAKE-HIAI-X90"):
                assert context in result.stderr, (failure, context, result.stderr)
for flag, value, message in [("--target-sha256", "0"*64, "SHA256"),
                             ("--expected-runtime-version", "wrong", "version")]:
    bad = base[:]; bad[bad.index(flag)+1] = value
    result = subprocess.run(bad + ["--output", str(root / "bad")], capture_output=True, text=True)
    assert result.returncode == 1 and message in result.stderr
print("HIAI production executor: parity, rejection, hashes and lifecycle fault checks passed")

for mode in ["ordinary", "dflash"]:
    once = base[:]; once[once.index("--mode") + 1] = "generate"
    report = root / ("once-"+mode)
    subprocess.run(once + ["--generation-mode", mode, "--output", str(report)], check=True)
    data = json.loads(report.read_text())
    assert data["scope"] == "single-generation" and "ordinary" not in data
    assert data["generation"]["generated_token_ids"] == list(range(3,11))
