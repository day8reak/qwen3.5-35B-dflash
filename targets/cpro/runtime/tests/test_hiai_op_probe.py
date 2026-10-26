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
         existing=False, oversized=False, actual_inputs=None, actual_outputs=None,
         input_format=2, output_format=2, additional_inputs=None, expected_init_shapes=None):
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
    for name, contents in (additional_inputs or {}).items():
        (directory / name).write_bytes(contents)
    (directory / "io.tsv").write_text(spec or f"input\t{dtype}\t1,4\tinput.bin\noutput\t{dtype}\t1,4\toutput.bin\n")
    output = directory / "output"
    if existing:
        output.mkdir()
        (output / "sentinel").write_text("preserve")
    trace = directory / "trace.txt"
    init_trace = directory / "init-trace.txt"
    shape_text = lambda shapes: ";".join(",".join(map(str, shape)) for shape in shapes) if shapes is not None else ""
    result = subprocess.run([str(runner), "--model", str(directory / "model.omc"),
                             "--io", str(directory / "io.tsv"), "--output", str(output)],
                            env={**os.environ, "FAKE_HIAI_FAILURE": failure, "FAKE_HIAI_TRACE": str(trace),
                                 "FAKE_HIAI_INIT_TRACE": str(init_trace),
                                 "FAKE_HIAI_INPUT_DIMS": shape_text(actual_inputs),
                                 "FAKE_HIAI_OUTPUT_DIMS": shape_text(actual_outputs),
                                 "FAKE_HIAI_INPUT_FORMAT": str(input_format), "FAKE_HIAI_OUTPUT_FORMAT": str(output_format)},
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
    if expected_init_shapes is not None:
        dtype_value = {"float16": 2, "float32": 1, "int32": 3}[dtype]
        assert init_trace.read_text().splitlines() == [
            f"format=2;dtype={dtype_value};shape=" + ",".join(map(str, shape)) for shape in expected_init_shapes]
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

# Runtime ND descriptors may append trailing singleton axes up to exactly rank4.
# These tests check descriptor identity in fake Init as well as output bytes.
for declared in ([4], [1, 4], [1, 2, 2]):
    padded = declared + [1] * (4 - len(declared))
    dims = ",".join(map(str, declared))
    for direction in ("input", "output", "both"):
        actual_input = declared if direction == "output" else padded
        actual_output = declared if direction == "input" else padded
        result = case(spec=f"input\tfloat16\t{dims}\tinput.bin\noutput\tfloat16\t{dims}\toutput.bin\n",
                      actual_inputs=[actual_input], actual_outputs=[actual_output],
                      expected_init_shapes=[actual_input, actual_output])
        assert result.stderr.count("accepted ND trailing-one padding") == (2 if direction == "both" else 1)
        assert "using runtime descriptor" in result.stderr
for dtype in ("float32", "int32"):
    case(dtype=dtype, actual_inputs=[[1, 4, 1, 1]], actual_outputs=[[1, 4, 1, 1]],
         expected_init_shapes=[[1, 4, 1, 1], [1, 4, 1, 1]])

# Match the board-reported MatMul ABI: two inputs and one output. Fake HIAI
# still only copies input[0]; this validates ABI handling, not MatMul arithmetic.
matmul_shapes = [[8, 16, 1, 1], [16, 16, 1, 1], [8, 16, 1, 1]]
result = case(
    spec="input\tfloat16\t8,16\tinput.bin\ninput\tfloat16\t16,16\trhs.bin\noutput\tfloat16\t8,16\toutput.bin\n",
    data=bytes(range(256)), additional_inputs={"rhs.bin": bytes(range(256)) * 2},
    actual_inputs=matmul_shapes[:2], actual_outputs=matmul_shapes[2:], expected_init_shapes=matmul_shapes,
)
assert result.stderr.count("accepted ND trailing-one padding") == 3
assert "padding for input[1]: declared_shape=[16,16], actual_shape=[16,16,1,1]" in result.stderr

# The exact-shape path remains valid for every previously permitted rank.
for rank in range(1, 9):
    shape = [1] * (rank - 1) + [4]
    dims = ",".join(map(str, shape))
    result = case(spec=f"input\tfloat16\t{dims}\tinput.bin\noutput\tfloat16\t{dims}\toutput.bin\n",
                  actual_inputs=[shape], actual_outputs=[shape], expected_init_shapes=[shape, shape])
    assert "accepted ND trailing-one padding" not in result.stderr

# Equal element counts do not permit moved/inserted axes or arbitrary ranks.
for declared, incompatible in (
    ([2, 4], [2, 4, 1, 2]),       # non-singleton tail
    ([2, 4], [1, 8, 1, 1]),       # changed prefix, same numel
    ([2, 4], [1, 2, 4, 1]),       # leading singleton
    ([2, 4], [2, 1, 4, 1]),       # inserted middle singleton
    ([2, 4], [4, 2, 1, 1]),       # swapped axes
    ([2, 4], [8]),                # flattened, same numel
    ([2, 4], [2, 4, 1]),          # padded to rank3, not rank4
    ([2, 4], [2, 4, 1, 1, 1]),    # rank2 extended beyond rank4
    ([1, 2, 2, 2], [1, 2, 2, 2, 1]),  # rank4 must not be extended
):
    dims = ",".join(map(str, declared))
    for direction in ("input", "output"):
        result = case(spec=f"input\tfloat16\t{dims}\tinput.bin\noutput\tfloat16\t{dims}\toutput.bin\n",
                      data=bytes(range(16)),
                      actual_inputs=[incompatible if direction == "input" else declared],
                      actual_outputs=[incompatible if direction == "output" else declared],
                      expected="FAILED", expected_stage="io")
        assert f"IO TSV: {direction}[0];" in json.loads(result.stdout)["error"]
        assert "accepted ND trailing-one padding" not in result.stderr

for direction in ("input", "output"):
    result = case(actual_inputs=[[1, 4, 1, 1]], actual_outputs=[[1, 4, 1, 1]],
                  input_format=0 if direction == "input" else 2,
                  output_format=0 if direction == "output" else 2,
                  expected="FAILED", expected_stage="io")
    assert f"IO TSV: {direction}[0];" in json.loads(result.stdout)["error"]
    assert "actual_format=0" in json.loads(result.stdout)["error"]

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
