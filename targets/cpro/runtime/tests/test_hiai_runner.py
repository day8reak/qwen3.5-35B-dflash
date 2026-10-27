"""Execute the actual CLI/adapter against a deliberately fallible HIAI double."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import struct
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

# Capture is opt-in and preserves the exact target buffer before draft execution,
# including signed zero/NaN bits that a floating-point JSON conversion can lose.
for number, (failure, width, capture) in enumerate([
    ("", 32, "0"), ("", 32, "1"), ("overwrite-target-on-draft", 32, "1"),
    ("capture-limit", 1024, "1"), ("large-features", 1025, "0"),
]):
    smoke = base[:]
    smoke[smoke.index("--mode") + 1] = "smoke"
    smoke[smoke.index("--feature-width") + 1] = str(width)
    report = root / ("capture-" + str(number))
    result = subprocess.run(smoke + ["--capture-target-features", capture, "--output", str(report)],
        env=dict(os.environ, FAKE_HIAI_FAILURE=failure), capture_output=True, text=True)
    assert result.returncode == 0 and "LEAK" not in result.stderr, result.stderr
    execution = json.loads(report.read_text())["execution"]
    if capture == "0":
        assert "target_features" not in execution
    else:
        values = [1234, 0x8000, 0x7e01] + [0] * (32 * width - 4) + [4321]
        expected = struct.pack("<" + "H" * len(values), *values)
        assert execution["target_features"] == {
            "dtype": "float16", "encoding": "hex-le", "shape": [1, 32, width],
            "bytes": len(expected), "data": expected.hex(),
        }

for number, (mode, capture, failure, width, message) in enumerate([
    ("smoke", "2", "", 32, "must be 0 or 1"),
    ("smoke", "true", "", 32, "must be 0 or 1"),
    ("smoke", "01", "", 32, "must be 0 or 1"),
    ("benchmark", "1", "", 32, "requires --mode smoke"),
    ("generate", "0", "", 32, "requires --mode smoke"),
    ("smoke", "1", "large-features", 1025, "65536-byte"),
    ("smoke", "1", "oversized-buffer", 32, "65536-byte"),
    ("smoke", "1", "run", 32, "Process"),
]):
    rejected = base[:]
    rejected[rejected.index("--mode") + 1] = mode
    rejected[rejected.index("--feature-width") + 1] = str(width)
    report = root / ("capture-rejected-" + str(number))
    result = subprocess.run(rejected + ["--capture-target-features", capture, "--output", str(report)],
        env=dict(os.environ, FAKE_HIAI_FAILURE=failure), capture_output=True, text=True)
    assert result.returncode == 1 and message in result.stderr, result.stderr
    assert not report.exists() and "LEAK" not in result.stderr, result.stderr
print("HIAI feature capture: exact pre-draft FP16 bytes, opt-in, smoke-only and 65536-byte bound checked")


# The independent stateful target executes once and hands state to the host.
# Its fake models check the actual history/recurrent/KV bytes across two loads.
def stateful_bundle(label, previous=None):
    folder = root / ("stateful-" + label)
    folder.mkdir()
    sequence, past = (1, 3) if previous else (3, 0)
    model = folder / "target.omc"
    model.write_bytes(b"stateful-decode-fixture" if previous else b"stateful-prefill-fixture")
    abi = folder / "state-abi.json"
    abi.write_text(json.dumps({"schema_version": 1, "sequence_length": sequence, "past_length": past}))
    inputs = [("input_ids", "int32", [1, sequence])]
    outputs = [
        ("target_top1", "int32", [1, sequence]),
        ("target_features", "float16", [1, sequence, 4]),
        ("layer_0_conv_history_out", "float16", [1, 8, 3]),
        ("layer_0_recurrent_out", "float32", [1, 2, 2, 2]),
        ("layer_1_key_out", "float16", [1, 1, past + sequence, 2]),
        ("layer_1_value_out", "float16", [1, 1, past + sequence, 2]),
    ]
    payloads = [struct.pack("<" + "i" * sequence, *([4] if previous else [1, 2, 3]))]
    if previous:
        previous_report = json.loads((previous / "result.json").read_text())
        for item in previous_report["outputs"][2:]:
            inputs.append((item["name"].removesuffix("_out"), item["dtype"], item["shape"]))
            payloads.append((previous / item["path"]).read_bytes())
    rows = ["QWEN35_STATEFUL_STEP\t1", f"model\ttarget.omc\t{sha(model)}",
            f"state_abi\tstate-abi.json\t{sha(abi)}", f"tokens\t{sequence}\t{past}"]
    for i, ((name, dtype, shape), payload) in enumerate(zip(inputs, payloads)):
        file = folder / f"input_{i:04d}.bin"
        file.write_bytes(payload)
        rows.append("\t".join(["input", str(i), name, dtype, ",".join(map(str, shape)), file.name, sha(file)]))
    rows.extend("\t".join(["output", str(i), name, dtype, ",".join(map(str, shape))])
                for i, (name, dtype, shape) in enumerate(outputs))
    (folder / "step.tsv").write_text("\n".join(rows) + "\n")
    return folder, rows


def stateful_run(folder, label, failure="", layout="", extra=()):
    output, trace = folder / label, folder / (label + ".trace")
    result = subprocess.run([
        str(runner), "--stateful-step", str(folder / "step.tsv"), "--output", str(output),
        "--expected-runtime-version", "FAKE-HIAI-X90", "--ddk-id", "fake", "--firmware-id", "fake",
        *extra,
    ], env=dict(os.environ, FAKE_HIAI_FAILURE=failure, FAKE_HIAI_LAYOUT=layout,
                FAKE_HIAI_TRACE=str(trace), FAKE_HIAI_DTYPE_TRACE=str(folder / (label + ".dtypes"))),
        text=True, capture_output=True)
    assert "LEAK" not in result.stderr, result.stderr
    events = trace.read_text().splitlines() if trace.exists() else []
    return result, output, events


prefill, good_rows = stateful_bundle("prefill")
first, first_output, events = stateful_run(prefill, "result")
assert first.returncode == 0, first.stderr
assert events == ["load", "process", "unload"], events
for folder, output in [(prefill, first_output)]:
    report = json.loads((output / "result.json").read_text())
    assert report["status"] == "EXECUTED" and report["scope"] == "stateful-target-step"
    assert report["evidence_kind"] == "host-fake-hiai"
    assert report["step_plan_sha256"] == sha(folder / "step.tsv")
    assert report["state_abi_sha256"] == sha(folder / "state-abi.json")
    assert report["model_sha256"] == sha(folder / "target.omc")
    assert report["sequence_length"] == 3 and report["past_length"] == 0 and report["next_past_length"] == 3
    assert report["target_calls"] == 1 and report["draft_calls"] == 0
    assert report["accuracy"] == report["performance"] == "NOT_EVALUATED"
    assert report["dtype_source"] == "hash-bound-plan"
    assert len(report["inputs"]) == 1 and len(report["outputs"]) == 6
    for i, item in enumerate(report["outputs"]):
        assert item["index"] == i and item["path"] == f"output_{i:04d}.bin"
        file = output / item["path"]
        assert item["bytes"] == file.stat().st_size and item["sha256"] == sha(file)
    assert (output / "output_0000.bin").read_bytes() == struct.pack("<3i", 2, 3, 4)
    assert (output / "output_0003.bin").read_bytes() == struct.pack("<8f", *range(1, 9))
    assert "1" in (folder / "result.dtypes").read_text().splitlines(), "FP32 state was not initialized as FP32"

decode, decode_rows = stateful_bundle("decode", first_output)
second, second_output, events = stateful_run(decode, "result")
assert second.returncode == 0, second.stderr
assert events == ["load", "process", "unload"], events
report = json.loads((second_output / "result.json").read_text())
assert (report["sequence_length"], report["past_length"], report["next_past_length"]) == (1, 3, 4)
assert len(report["inputs"]) == 5 and len(report["outputs"]) == 6
assert (second_output / "output_0000.bin").read_bytes() == struct.pack("<i", 5)
assert (second_output / "output_0003.bin").read_bytes() == struct.pack("<8f", *range(2, 10))
for i in (4, 5):
    assert (second_output / f"output_{i:04d}.bin").read_bytes()[:12] == (first_output / f"output_{i:04d}.bin").read_bytes()
for item in report["inputs"]:
    assert item["sha256"] == sha(decode / item["path"])
for item in report["outputs"]:
    assert item["sha256"] == sha(second_output / item["path"])

for layout in ("padded", "leading-feature", "mixed-feature"):
    result, output, events = stateful_run(prefill, "layout-" + layout, layout=layout)
    assert result.returncode == 0, result.stderr
    assert events == ["load", "process", "unload"]
    assert (output / "output_0003.bin").read_bytes() == struct.pack("<8f", *range(1, 9))

for failure in ("init", "version", "no-npu", "path", "load", "model-buffer", "io-dims",
                "shape", "format", "count", "allocation", "short-buffer", "run"):
    result, output, events = stateful_run(prefill, "failure-" + failure, failure=failure)
    assert result.returncode == 1 and not (output / "result.json").exists(), (failure, result.stderr)
    assert events.count("process") <= 1 and "compatibility-query" not in events
    if failure != "init":
        assert events[-1] == "unload", (failure, events)

# Keep the known-good prefill intact while checking strict manifest parsing.
for label, change, message in [
    ("version", lambda rows: ["QWEN35_STATEFUL_STEP\t2", *rows[1:]], "header"),
    ("model-hash", lambda rows: [rows[0], rows[1].rsplit("\t", 1)[0] + "\t" + "0"*64, *rows[2:]], "SHA256"),
    ("abi-hash", lambda rows: [*rows[:2], rows[2].rsplit("\t", 1)[0] + "\t" + "0"*64, *rows[3:]], "SHA256"),
    ("input-hash", lambda rows: [*rows[:4], rows[4].rsplit("\t", 1)[0] + "\t" + "0"*64, *rows[5:]], "SHA256"),
    ("input-index", lambda rows: [*rows[:4], rows[4].replace("input\t0\t", "input\t1\t"), *rows[5:]], "indices"),
    ("input-type", lambda rows: [*rows[:4], rows[4].replace("\tint32\t", "\tfloat32\t"), *rows[5:]], "contract"),
    ("traversal", lambda rows: [*rows[:4], rows[4].replace("input_0000.bin", "../input_0000.bin"), *rows[5:]], "traversal"),
    ("absolute", lambda rows: [*rows[:4], rows[4].replace("input_0000.bin", "/input_0000.bin"), *rows[5:]], "relative"),
    ("unknown-dtype", lambda rows: [*rows[:4], rows[4].replace("\tint32\t", "\tint16\t"), *rows[5:]], "dtype"),
    ("zero-shape", lambda rows: [*rows[:4], rows[4].replace("\t1,3\t", "\t1,0\t"), *rows[5:]], "positive"),
    ("generic-op", lambda rows: [*rows[:5], rows[5].replace("target_top1", "generic_output"), *rows[6:]], "contract"),
    ("missing-state", lambda rows: rows[:-1], "complete state pair"),
    ("trailing-field", lambda rows: [*rows[:4], rows[4] + "\textra", *rows[5:]], "record"),
]:
    (prefill / "step.tsv").write_text("\n".join(change(good_rows)) + "\n")
    result, output, events = stateful_run(prefill, "reject-" + label)
    assert result.returncode == 1 and message in result.stderr, (label, result.stderr)
    assert not (output / "result.json").exists() and "process" not in events, (label, events)
(prefill / "step.tsv").write_text("\n".join(good_rows) + "\n")
result, output, events = stateful_run(prefill, "result")
assert result.returncode == 1 and "already exists" in result.stderr
assert sha(first_output / "output_0003.bin") == hashlib.sha256(struct.pack("<8f", *range(1, 9))).hexdigest()
result, output, events = stateful_run(prefill, "reject-old-cli", extra=("--draft", "unused.omc"))
assert result.returncode == 1 and "unknown stateful option" in result.stderr and not events
(prefill / "input-alias.bin").symlink_to(prefill / "input_0000.bin")
(prefill / "step.tsv").write_text("\n".join(good_rows).replace("input_0000.bin", "input-alias.bin") + "\n")
result, output, events = stateful_run(prefill, "reject-symlink")
assert result.returncode == 1 and "symlink" in result.stderr and not events
(prefill / "step.tsv").write_text("x" * ((1 << 20) + 1))
result, output, events = stateful_run(prefill, "reject-plan-size")
assert result.returncode == 1 and "size limit" in result.stderr and not events
(prefill / "step.tsv").write_text("\n".join(good_rows) + "\n")
(decode / "step.tsv").write_text("\n".join(decode_rows[:5] + decode_rows[6:]) + "\n")
result, output, events = stateful_run(decode, "reject-state-omission")
assert result.returncode == 1 and not events and not (output / "result.json").exists()
(decode / "step.tsv").write_text("\n".join(decode_rows) + "\n")
state = decode / "input_0002.bin"
state.write_bytes(struct.pack("<8f", *([0] * 8)))
changed = decode_rows[:]
changed[6] = changed[6].rsplit("\t", 1)[0] + "\t" + sha(state)
(decode / "step.tsv").write_text("\n".join(changed) + "\n")
result, output, events = stateful_run(decode, "reject-wrong-state")
assert result.returncode == 1 and "Process" in result.stderr
assert events == ["load", "process", "unload"] and not (output / "result.json").exists()
print("HIAI stateful target: single Process, FP16/FP32 cache handoff, hashes, ND padding, bounded plans and failures checked")
