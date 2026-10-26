"""Probe orchestration contracts with isolated fake tool processes, not device tests."""
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from qwen35_moe_dflash import op_probe
from qwen35_moe_dflash.hashing import sha256_file
from qwen35_moe_dflash.op_probe_cases import cases


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def run_root(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    return tmp_path


def _no_tool(*args, **kwargs):
    raise AssertionError("prepare-only must not contact compiler/native/HDC tools")


def test_prepare_only_all_cases_serializes_locked_contracts(run_root, monkeypatch):
    monkeypatch.setattr(op_probe, "compiler_resources", _no_tool)
    monkeypatch.setattr(op_probe, "native_build", _no_tool)
    calls = []
    monkeypatch.setattr(op_probe, "write_report", lambda *args: calls.append(args))
    for key in ("CANN_ROOT", "CPRO_DDK_ROOT", "HIAI_BUILD_DIR", "SN", "BIP"):
        monkeypatch.delenv(key, raising=False)
    output = run_root / "prepared"
    assert op_probe.run(op_probe.parser().parse_args(["--prepare-only", "--output", str(output)])) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "PREPARED" and not manifest["board_requested"]
    assert len(manifest["cases"]) == 118
    assert len({record["name"] for record in manifest["cases"]}) == 118
    assert calls[0][0] == output and calls[0][2] is None
    for record in manifest["cases"]:
        folder = Path(record["directory"])
        assert record["compile"]["status"] == "NOT_RUN"
        assert record["source_sha256"] == sha256_file(folder / "graph.cpp")
        assert record["io_sha256"] == sha256_file(folder / "io.tsv")
        assert (folder / "recipe").is_dir()
        rows = [line.split("\t") for line in (folder / "io.tsv").read_text().splitlines()]
        assert len(rows) == len(record["input_tensors"]) + len(record["outputs"])
        for fields in rows:
            assert len(fields) == 4 and fields[1] in {"float16", "float32", "int32"}
            assert Path(fields[3]).name == fields[3]
        for tensor in record["input_tensors"]:
            path = folder / tensor["path"]
            assert tensor["sha256"] == sha256_file(path)
            assert path.stat().st_size == np.prod(tensor["shape"]) * np.dtype(tensor["dtype"]).itemsize
        for tensor in record["outputs"]:
            path = folder / tensor["expected"]
            assert tensor["reference"]["sha256"] == sha256_file(path)
            values = np.fromfile(path, dtype=tensor["dtype"])
            assert values.size == np.prod(tensor["shape"]) and np.isfinite(values).all()
            assert (tensor["rtol"], tensor["atol"]) in {(0., 0.), (.01, .002)}


def test_selected_case_validation_is_before_tools_or_files(run_root, monkeypatch, capsys):
    monkeypatch.setattr(op_probe, "compiler_resources", _no_tool)
    monkeypatch.setattr(op_probe, "write_report", lambda *args: None)
    output = run_root / "selected"
    args = ["--prepare-only", "--case", "silu", "--case", "matmul_m1_b_const", "--output", str(output)]
    assert op_probe.run(op_probe.parser().parse_args(args)) == 0
    assert {r["name"] for r in json.loads((output / "manifest.json").read_text())["cases"]} == {
        "silu", "matmul_m1_b_const"}
    missing = run_root / "missing"
    with pytest.raises(ValueError, match="unknown probe cases"):
        op_probe.run(op_probe.parser().parse_args(["--case", "not-a-case", "--output", str(missing)]))
    assert not missing.exists()
    assert op_probe.run(op_probe.parser().parse_args(["--list-cases", "--case", "silu"])) == 0
    assert "silu:" in capsys.readouterr().out



def test_remaining_cpu_suite_prepares_only_selected_families(run_root, monkeypatch, capsys):
    monkeypatch.setattr(op_probe, "compiler_resources", _no_tool)
    monkeypatch.setattr(op_probe, "native_build", _no_tool)
    # Exercise the real report writer: category and input-domain provenance
    # must survive preparation instead of existing only in Python fixtures.
    output = run_root / "remaining"
    args = op_probe.parser().parse_args([
        "--suite", "remaining-cpu", "--prepare-only", "--output", str(output)])
    assert op_probe.run(args) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    records = manifest["cases"]
    names = [row["name"] for row in records]
    assert manifest["selection"] == {
        "suite": "remaining-cpu", "requested_cases": [], "excluded_families": [], "selected_cases": names}
    assert names[:7] == [
        "gdn_mask_lower_slice_concat", "gdn_mask_lower_gather",
        "gdn_mask_strict_slice_concat", "gdn_mask_strict_gather",
        "gdn_mask_exp_chain_slice_concat", "gdn_mask_exp_chain_gather", "sub_outer_gather"]
    assert "matmul_m1_b_const" not in names and "reduce_sum_scan_matmul" not in names
    assert {row["probe_family"] for row in records} == {"gdn", "routing", "selection", "position"}
    report = json.loads((output / "report.json").read_text())
    assert report["probe_families"] == {"gdn": 7, "routing": 16, "selection": 15, "position": 12}
    assert sum(report["probe_families"].values()) == len(names) == 50
    assert report["selection"] == manifest["selection"]
    domains = [r for r in report["cases"] if r["name"] == "attention_mask_lut_fp16"]
    assert domains[0]["input_domain"] == {"attention_mask": {"allowed_values": [0, 1]}}
    assert op_probe.run(op_probe.parser().parse_args([
        "--suite", "remaining-cpu", "--list-cases"])) == 0
    listed = capsys.readouterr().out
    assert "sub_outer_gather:" in listed and "matmul_m1_b_const:" not in listed


def test_suite_excludes_already_reported_gdn_without_touching_tools(run_root, monkeypatch, capsys):
    monkeypatch.setattr(op_probe, "compiler_resources", _no_tool)
    assert op_probe.run(op_probe.parser().parse_args([
        "--suite", "remaining-cpu", "--exclude-family", "gdn", "--list-cases"])) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 43
    assert all(not line.startswith(("gdn_", "sub_outer_gather:")) for line in lines)
    assert any(line.startswith("moe_route_gather:") for line in lines)
    with pytest.raises(ValueError, match="requires --suite"):
        op_probe.run(op_probe.parser().parse_args(["--exclude-family", "gdn", "--list-cases"]))
    with pytest.raises(ValueError, match="no cases"):
        op_probe.run(op_probe.parser().parse_args([
            "--suite", "remaining-cpu", "--list-cases",
            *[arg for name in ("gdn", "routing", "selection", "position")
              for arg in ("--exclude-family", name)]]))


@pytest.mark.parametrize("options", [
    ["--suite", "remaining-cpu", "--case", "silu"],
    ["--suite", "does-not-exist"],
])
def test_suite_selector_rejects_ambiguous_or_unknown_selection(options):
    with pytest.raises(SystemExit) as exc:
        op_probe.parser().parse_args(options)
    assert exc.value.code == 2


def _compiler(monkeypatch, root, *, failure=None, missing_omc=False):
    ge, ddk = root / "ge", root / "ddk"
    (ge / "lib64").mkdir(parents=True)
    (ge / "lib64/libgraph.so").write_bytes(b"test GE")
    ddk.mkdir()
    omg = ddk / "omg"
    omg.write_bytes(b"test OMG")
    commands = []

    def logged(argv, *, folder, log, **kwargs):
        folder = Path(folder)
        commands.append((list(argv), folder))
        Path(log).write_text("mocked tool; no device execution\n")
        if argv[0] == "mock-c++":
            if failure == folder.name:
                raise subprocess.CalledProcessError(1, argv)
            Path(argv[-1]).write_bytes(b"mock exporter")
        elif Path(argv[0]).name == "exporter":
            Path(argv[1]).write_bytes(b"mock AIR")
        elif argv[0] == str(omg) and not missing_omc:
            (folder / "model.omc").write_bytes(b"mock OMC")

    @contextmanager
    def ipc(scratch, env):
        yield env, {"mock": True}

    monkeypatch.setattr(op_probe, "_logged", logged)
    monkeypatch.setattr(op_probe, "_omg_ipc_environment", ipc)
    monkeypatch.setattr(op_probe, "_environment_record", lambda *args: {})
    monkeypatch.setattr(op_probe, "_retain_compiler_diagnostics", lambda *args, **kwargs: {"mock": True})
    resources = (ge, ddk, "mock-c++", omg, {})
    monkeypatch.setattr(op_probe, "compiler_resources", lambda: resources)
    return resources, commands


def test_compile_failure_is_recorded_and_later_cases_continue(run_root, monkeypatch):
    first, second = "matmul_m1_b_const", "matmul_m1_b_dynamic"
    _, commands = _compiler(monkeypatch, run_root, failure=first)
    monkeypatch.setattr(op_probe, "native_build", _no_tool)
    monkeypatch.setattr(op_probe, "write_report", lambda *args: None)
    output = run_root / "compiled"
    op_probe.run(op_probe.parser().parse_args(["--compile-only", "--case", first, "--case", second,
                                               "--output", str(output)]))
    records = json.loads((output / "manifest.json").read_text())["cases"]
    assert records[0]["compile"]["status"] == "FAILED"
    assert records[0]["compile"]["stage"] == "build-exporter"
    assert records[1]["compile"]["status"] == "COMPILED"
    assert Path(records[1]["model"]).read_bytes() == b"mock OMC"
    assert sum(argv[0] == "mock-c++" for argv, _ in commands) == 2


def test_zero_exit_without_omc_is_failure(run_root, monkeypatch):
    resources, _ = _compiler(monkeypatch, run_root, missing_omc=True)
    record = op_probe.prepare_case(next(cases()), run_root / "probe")
    result = op_probe.compile_case(record, resources, 60)
    assert result["status"] == "FAILED" and result["stage"] == "omg"
    assert "OMG exited zero" in result["error"]
    assert not Path(record["model"]).exists()


def test_compile_only_all_fail_has_nonzero_exit(run_root, monkeypatch):
    first = "matmul_m1_b_const"
    _compiler(monkeypatch, run_root, failure=first)
    monkeypatch.setattr(op_probe, "native_build", _no_tool)
    monkeypatch.setattr(op_probe, "write_report", lambda *args: None)
    output = run_root / "all-failed"
    result = op_probe.run(op_probe.parser().parse_args([
        "--compile-only", "--case", first, "--output", str(output)]))
    manifest = json.loads((output / "manifest.json").read_text())
    assert result == 1
    assert manifest["status"] == "NO_COMPILED_CASES"
    assert manifest["cases"][0]["compile"]["status"] == "FAILED"


@pytest.mark.parametrize("changed", ["graph.cpp", "input_0.bin", "expected_0.bin", "io.tsv"])
def test_compile_refuses_changed_construction_before_tools(run_root, monkeypatch, changed):
    resources, commands = _compiler(monkeypatch, run_root)
    record = op_probe.prepare_case(next(cases()), run_root / "probe")
    path = Path(record["directory"]) / changed
    path.write_bytes(path.read_bytes() + b"changed")
    result = op_probe.compile_case(record, resources, 60)
    assert result["status"] == "FAILED" and result["stage"] == "build-exporter"
    assert not commands
    assert not Path(record["model"]).exists()


def _elf(path, marker):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = bytearray(64)
    data[:7] = b"\x7fELF\x02\x01\x01"
    data[16:18] = (3).to_bytes(2, "little")
    data[18:20] = (183).to_bytes(2, "little")
    path.write_bytes(data + marker)
    return path


def _native_records(root, monkeypatch):
    repository = root / "repository"
    source = repository / "targets/cpro/runtime"
    (source / "src").mkdir(parents=True)
    (source / "CMakeLists.txt").write_text("fake isolated probe CMake source")
    (source / "src/hiai_op_probe.cpp").write_text("fake isolated native probe source")
    monkeypatch.setattr(op_probe, "repository_root", lambda: repository)
    build = root / "previous"
    runner = _elf(build / "build/qwen35_dflash_hiai", b"previous runner")
    library = _elf(root / "lib/libhiai.so", b"recorded HIAI")
    _elf(root / "lib/libc++_shared.so", b"recorded CXX")
    include = root / "include"
    include.mkdir()
    header = include / "HiAiModelManagerService.h"
    header.write_text("fake public header")
    toolchain = root / "sdk/ohos.toolchain.cmake"
    toolchain.parent.mkdir()
    toolchain.write_text("fake toolchain")
    original = {
        "argv": ["untrusted-original-executable", "-S", "ignored-source", "-B", "ignored-build",
                 f"-DCMAKE_TOOLCHAIN_FILE={toolchain}", "-DOHOS_ARCH=arm64-v8a", "-DOHOS_PLATFORM=OHOS",
                 "-DOHOS_STL=c++_shared", "-DOHOS_COMPATIBLE_SDK_VERSION=21", f"-DHIAI_INCLUDE_DIR={include}",
                 "-DCMAKE_CXX_FLAGS=-DUNTRUSTED", "-DBUILD_HOST_TESTS=ON", "-DHIAI_LIBRARIES=/wrong.so"],
        "libraries": [{"path": str(library), "sha256": sha256_file(library)}],
        "header_sha256": sha256_file(header), "sdk_toolchain_sha256": sha256_file(toolchain),
    }
    _json(build / "command.json", original)
    _json(build / "result.json", {"status": "LINKED", "binary_sha256": sha256_file(runner),
                                  "command_sha256": sha256_file(build / "command.json")})
    monkeypatch.setenv("HIAI_BUILD_DIR", str(build))
    return original, library, header, toolchain


def test_native_build_reuses_bound_flags_and_only_probe_target(run_root, monkeypatch):
    original, library, _, toolchain = _native_records(run_root, monkeypatch)
    commands = []

    def logged(argv, *, folder, **kwargs):
        commands.append(list(argv))
        if argv[:2] == ["cmake", "--build"]:
            _elf(Path(argv[2]) / "qwen35_hiai_op_probe", b"new probe")

    monkeypatch.setattr(op_probe, "_logged", logged)
    output = run_root / "probe"
    output.mkdir()
    runner, libraries = op_probe.native_build(output, 60)
    configure, build = commands
    assert configure[0] == "cmake"
    assert f"-DCMAKE_TOOLCHAIN_FILE={toolchain}" in configure
    assert "-DOHOS_COMPATIBLE_SDK_VERSION=21" in configure
    assert f"-DHIAI_LIBRARIES={library}" in configure
    assert "-DBUILD_HOST_TESTS=OFF" in configure and "-DBUILD_HIAI_OBJECTS=OFF" in configure
    assert not any("UNTRUSTED" in arg or "ignored-" in arg or "/wrong.so" in arg for arg in configure)
    assert build == ["cmake", "--build", str(output / "native/build"), "--target", "qwen35_hiai_op_probe", "-j", "4"]
    command = json.loads((output / "native/command.json").read_text())
    result = json.loads((output / "native/result.json").read_text())
    assert command["libraries"] == original["libraries"]
    assert command["library_discovery"]["command_binding"] == "sha256-verified"
    assert result["binary_sha256"] == sha256_file(runner)
    assert result["command_sha256"] == sha256_file(output / "native/command.json")
    assert result["device_validation"] == "NOT_RUN"
    assert len(command["sources"]) == 2
    assert all(item["sha256"] == sha256_file(Path(item["path"])) for item in command["sources"])
    assert library in libraries


@pytest.mark.parametrize("changed", ["library", "header", "toolchain"])
def test_native_build_refuses_changed_linked_resources(run_root, monkeypatch, changed):
    _, library, header, toolchain = _native_records(run_root, monkeypatch)
    selected = {"library": library, "header": header, "toolchain": toolchain}[changed]
    selected.write_bytes(selected.read_bytes() + b"changed")
    monkeypatch.setattr(op_probe, "_logged", _no_tool)
    output = run_root / "probe"
    output.mkdir()
    with pytest.raises(ValueError, match="SHA256|sha256"):
        op_probe.native_build(output, 60)
    assert not (output / "native").exists()


def test_native_build_refuses_source_changed_during_build(run_root, monkeypatch):
    _native_records(run_root, monkeypatch)

    def logged(argv, *, folder, **kwargs):
        if argv[:2] == ["cmake", "--build"]:
            _elf(Path(argv[2]) / "qwen35_hiai_op_probe", b"new probe")
            source = op_probe.repository_root() / "targets/cpro/runtime/src/hiai_op_probe.cpp"
            source.write_text(source.read_text() + " changed during build")

    monkeypatch.setattr(op_probe, "_logged", logged)
    output = run_root / "probe"
    output.mkdir()
    with pytest.raises(ValueError, match="native source changed during build"):
        op_probe.native_build(output, 60)
    assert not (output / "native/result.json").exists()
