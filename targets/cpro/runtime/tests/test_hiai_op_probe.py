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
         expected="EXECUTED", expected_stage=None, existing=False, oversized=False):
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
    case(dtype=dtype)
for failure, stage in (("init", "init"), ("model-buffer", "load"), ("load", "load"),
                       ("process", "process"), ("shape", "io"), ("format", "io"),
                       ("count", "io"), ("allocation", "io"), ("short-buffer", "io"),
                       ("unload", "unload")):
    case(failure=failure, expected="FAILED", expected_stage=stage)
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
