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
                "path", "model-buffer", "model-buffer-null", "model-buffer-size", "draft-load", "io-dims", "shape", "format", "count", "allocation", "short-buffer", "run", "token"]:
    report = root / (failure or "success")
    trace_path = root / ((failure or "success") + ".trace")
    result = subprocess.run(base + ["--output", str(report)],
        env=dict(os.environ, FAKE_HIAI_FAILURE=failure, FAKE_HIAI_TRACE=str(trace_path)),
        capture_output=True, text=True)
    assert "LEAK" not in result.stderr, result.stderr
    trace = trace_path.read_text().splitlines() if trace_path.exists() else []
    # The fixture requires exact OMC bytes and checks the raw pointer through
    # Process, UnLoadModel and client teardown. The standalone query throws;
    # explicit buffer loading must proceed without querying it.
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
        if failure in ("model-buffer", "model-buffer-null", "model-buffer-size"):
            expected = "SetModelBuffer" if failure == "model-buffer" else "model buffer descriptor"
            assert expected in result.stderr, (failure, result.stderr)
            assert trace == ["unload"], (failure, trace)
        if failure == "draft-load":
            assert "Load" in result.stderr and "qwen35_draft" in result.stderr
            assert str(draft) in result.stderr
            assert trace == ["load", "load", "unload", "unload"], trace
        if failure in ("path", "load", "incompatible", "io-dims", "model-buffer",
                       "model-buffer-null", "model-buffer-size"):
            for context in ("qwen35_target", str(target), "FAKE-HIAI-X90"):
                assert context in result.stderr, (failure, context, result.stderr)
for flag, value, message in [("--target-sha256", "0"*64, "SHA256"),
                             ("--expected-runtime-version", "wrong", "version")]:
    bad = base[:]; bad[bad.index(flag)+1] = value
    result = subprocess.run(bad + ["--output", str(root / "bad")], capture_output=True, text=True)
    assert result.returncode == 1 and message in result.stderr
empty = root / "empty.omc"
empty.write_bytes(b"")
empty_case = base[:]
empty_case[empty_case.index("--target")+1] = str(empty)
empty_case[empty_case.index("--target-sha256")+1] = sha(empty)
empty_trace = root / "empty.trace"
empty_result = subprocess.run(empty_case + ["--output", str(root / "empty-report")],
    env=dict(os.environ, FAKE_HIAI_FAILURE="", FAKE_HIAI_TRACE=str(empty_trace)),
    capture_output=True, text=True)
assert empty_result.returncode == 1 and "empty" in empty_result.stderr.lower(), empty_result.stderr
assert str(empty) in empty_result.stderr
assert not (root / "empty-report").exists()
assert "load" not in (empty_trace.read_text().splitlines() if empty_trace.exists() else [])
print("HIAI production executor: parity, rejection, hashes and lifecycle fault checks passed")

for mode in ["ordinary", "dflash"]:
    once = base[:]; once[once.index("--mode") + 1] = "generate"
    report = root / ("once-"+mode)
    subprocess.run(once + ["--generation-mode", mode, "--output", str(report)], check=True)
    data = json.loads(report.read_text())
    assert data["scope"] == "single-generation" and "ordinary" not in data
    assert data["generation"]["generated_token_ids"] == list(range(3,11))

# Smoke must invoke both graphs once, even when the first target token is EOS.
# Valid but wrong draft tokens must not turn this execution test into an
# accuracy/acceptance gate or an implicit generation benchmark.
for failure in ("", "corrupt-draft"):
    smoke = base[:]
    smoke[smoke.index("--mode") + 1] = "smoke"
    smoke[smoke.index("--eos-token-ids") + 1] = "3"
    smoke[smoke.index("--max-new-tokens") + 1] = "999"
    label = "smoke-" + (failure or "success")
    report, trace_path = root / label, root / (label + ".trace")
    result = subprocess.run(smoke + ["--output", str(report)],
        env=dict(os.environ, FAKE_HIAI_FAILURE=failure, FAKE_HIAI_TRACE=str(trace_path)),
        capture_output=True, text=True)
    assert result.returncode == 0 and "LEAK" not in result.stderr, result.stderr
    assert trace_path.read_text().splitlines() == ["load", "load", "process", "process", "unload", "unload"]
    data = json.loads(report.read_text())
    assert data["status"] == "EXECUTION_COMPLETE" and data["mode"] == "smoke"
    assert data["scope"] == "target-draft-execution"
    assert data["accuracy"] == data["performance"] == "NOT_EVALUATED"
    assert data["device_execution"] == "PASS" and data["evidence_kind"] == "host-fake-hiai"
    assert data["abi_version"] == 2 and data["feature_dtype"] == "torch.float16"
    assert data["target_sha256"] == sha(target) and data["draft_sha256"] == sha(draft)
    assert data["runtime_version"] == "FAKE-HIAI-X90"
    assert data["sequence_length"] == 32 and data["prompt_token_ids"] == [1, 2]
    assert data["execution"] == {"target_calls": 1, "draft_calls": 1,
        "target_top1": [2, 3] + [1] * 30,
        "draft_top1": [3, 63 if failure else 4, 5]}
    assert not {"generation", "ordinary", "dflash", "timing_scope", "max_new_tokens",
                "generated_token_ids", "speedup", "acceptance_rate"}.intersection(data)
    assert result.stdout.startswith("EXECUTION_COMPLETE: smoke;")

for number, (failure, flag, value, message) in enumerate([
    ("", "--target-sha256", "0" * 64, "SHA256"),
    ("", "--expected-runtime-version", "wrong", "version"),
    ("", "--prompt-token-ids", "", "prefix"),
    ("", "--prompt-token-ids", ",".join(["1"] * 33), "prefix"),
    ("", "--prompt-token-ids", "64", "vocabulary"),
    ("", "--pad-token-id", "64", "vocabulary"),
    ("shape", None, None, "shape"),
    ("format", None, None, "format"),
    ("draft-load", None, None, "Load"),
    ("run", None, None, "Process"),
    ("token", None, None, "vocabulary"),
]):
    smoke = base[:]; smoke[smoke.index("--mode") + 1] = "smoke"
    if flag is not None:
        smoke[smoke.index(flag) + 1] = value
    report = root / ("smoke-rejected-" + str(number))
    result = subprocess.run(smoke + ["--output", str(report)],
        env=dict(os.environ, FAKE_HIAI_FAILURE=failure), capture_output=True, text=True)
    assert result.returncode == 1 and message.lower() in result.stderr.lower(), (failure, result.stderr)
    assert not report.exists() and "LEAK" not in result.stderr, result.stderr
print("HIAI smoke: two calls including immediate EOS, identity/ABI faults, and no quality/performance claims passed")
