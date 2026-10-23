"""Exercise bounded IO, transport and failure lifetimes; no device claims."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

runner, root = Path(sys.argv[1]), Path(sys.argv[2])
root.mkdir(parents=True, exist_ok=True)
root = Path(tempfile.mkdtemp(prefix="probe-", dir=root))
counter = 0


def case(*, dtype="float16", failure="", spec=None, data=None, model=b"probe-fixture",
         expected="EXECUTED", expected_stage=None, expected_error=None,
         existing=False, oversized=False):
    global counter
    counter += 1
    directory = root / str(counter)
    directory.mkdir()
    (directory / "model.omc").write_bytes(model)
    if oversized:
        with (directory / "model.omc").open("r+b") as stream:
            stream.truncate((64 << 20) + 1)
    data = bytes(range(8 if dtype == "float16" else 16)) if data is None else data
    (directory / "input.bin").write_bytes(data)
    (directory / "io.tsv").write_text(spec or f"input\t{dtype}\t1,4\tinput.bin\noutput\t{dtype}\t1,4\toutput.bin\n")
    output = directory / "output"
    if existing:
        output.mkdir()
        (output / "sentinel").write_text("preserve")
    trace = directory / "trace.txt"
    result = subprocess.run([str(runner), "--model", str(directory / "model.omc"),
                             "--io", str(directory / "io.tsv"), "--output", str(output)],
                            env={**os.environ, "FAKE_HIAI_FAILURE": failure, "FAKE_HIAI_TRACE": str(trace)},
                            text=True, capture_output=True)
    parsed = json.loads(result.stdout)
    assert parsed["status"] == expected, (parsed, result.stderr)
    assert result.returncode == (0 if expected == "EXECUTED" else 1), result.stderr
    if expected_stage:
        assert parsed["stage"] == expected_stage, parsed
        assert f"stage={expected_stage}:" in result.stderr, result.stderr
    if expected_error:
        assert parsed["error"] == expected_error, parsed
        assert expected_error in result.stderr, result.stderr
    if expected_stage == "io":
        # Descriptor checks must fail before submitting any inference work.
        assert trace.read_text().splitlines() == ["load", "unload", "destroy"]
    if expected == "EXECUTED":
        assert parsed["placement"] == "UNVERIFIED"
        assert (output / "output.bin").read_bytes() == data
        assert trace.read_text().splitlines() == ["load", "process", "unload", "destroy"]
    if failure in {"load", "process", "shape", "format", "count", "allocation", "short-buffer"}:
        assert trace.read_text().splitlines()[-2:] == ["unload", "destroy"]
    if existing:
        assert (output / "sentinel").read_text() == "preserve"
    return result


for dtype in ("float16", "float32", "int32"):
    result = case(dtype=dtype)
    assert "model IO: inputs=1, outputs=1\n" in result.stderr, result.stderr
    for direction in ("input", "output"):
        assert f"model IO {direction}[0]: shape=[1,4], format=2, dtype=unavailable\n" in result.stderr
for failure, stage in (("init", "init"), ("model-buffer", "load"), ("load", "load"),
                       ("process", "process"),
                       ("count", "io"), ("allocation", "io"), ("short-buffer", "io"),
                       ("unload", "unload")):
    case(failure=failure, expected="FAILED", expected_stage=stage)
# Fake enum values are deliberate: diagnostics must show the actual header values,
# and must not infer a model dtype from the TSV's declared dtype.
for dtype, dtype_value in (("float16", 2), ("float32", 1), ("int32", 3)):
    for failure, actual_shape, actual_format in (("shape", "[4]", 2), ("format", "[1,4]", 0)):
        result = case(
            dtype=dtype, failure=failure, expected="FAILED", expected_stage="io",
            expected_error=(
                "OMC tensor shape/format differs from IO TSV: input[0]"
                f"; declared_dtype={dtype}({dtype_value}), declared_shape=[1,4], declared_format=ND(2)"
                f"; actual_shape={actual_shape}, actual_format={actual_format}, actual_dtype=unavailable"
            ),
        )
        assert (f"model IO input[0]: shape={actual_shape}, format={actual_format}, dtype=unavailable\n"
                in result.stderr), result.stderr
        # Even an input[0] mismatch must preserve the output ABI for diagnosis.
        assert "model IO output[0]: shape=[1,4], format=2, dtype=unavailable\n" in result.stderr
result = case(
    spec="input\tfloat16\t1,4\tinput.bin\noutput\tfloat16\t4\toutput.bin\n",
    expected="FAILED", expected_stage="io",
    expected_error=(
        "OMC tensor shape/format differs from IO TSV: output[0]"
        "; declared_dtype=float16(2), declared_shape=[4], declared_format=ND(2)"
        "; actual_shape=[1,4], actual_format=2, actual_dtype=unavailable"
    ),
)
case(model=b"", expected="FAILED", expected_stage="model-buffer")
case(oversized=True, expected="FAILED", expected_stage="model-buffer")
case(data=b"short", expected="FAILED", expected_stage="io")
case(existing=True, expected="FAILED", expected_stage="arguments")
for invalid in (
    "input\tbool\t1,4\tinput.bin\noutput\tfloat16\t1,4\toutput.bin\n",
    "input\tfloat16\t0,4\tinput.bin\noutput\tfloat16\t1,4\toutput.bin\n",
    "input\tfloat16\t1,20000000\tinput.bin\noutput\tfloat16\t1,4\toutput.bin\n",
    "input\tfloat16\t1,4\t../input.bin\noutput\tfloat16\t1,4\toutput.bin\n",
    "input\tfloat16\t1,4\tinput.bin\noutput\tfloat16\t1,4\t../output.bin\n",
    "input\tfloat16\t1,4\tinput.bin\noutput\tfloat16\t1,4\toutput.bin\noutput\tfloat16\t1,4\toutput.bin\n",
):
    case(spec=invalid, expected="FAILED", expected_stage="io-spec")
print(f"PASS: {counter} probe transport/lifetime cases (fake HIAI)")
