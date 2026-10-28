"""Isolated existing-AIR compilation; mock OMG only, never an actual device."""
import errno
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from qwen35_moe_dflash import existing_air as module


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    source = tmp_path / "source"
    source.mkdir()
    air = source / "export.air"
    air.write_bytes(b"SIMULATED UNINSPECTED AIR")
    (source / "weights").mkdir()
    (source / "weights/w.bin").write_bytes(b"SIDE DATA")
    (source / "target.omc").write_bytes(b"OLD MODEL")
    (source / "compile.log").write_text("OLD LOG")
    (source / "outside").mkdir()
    (source / "outside/ignored.bin").write_bytes(b"NOT SELECTED")
    ddk = tmp_path / "ddk"
    omg = ddk / "tools/tools_omg/master/omg"
    omg.parent.mkdir(parents=True)
    omg.write_text("#!/bin/sh\nexit 0\n")
    omg.chmod(0o755)
    return air, tmp_path / "result", ddk


def _tool(command, **kwargs):
    if "--help" in command:
        return _help(command, **kwargs)
    model = Path(next(value.split("=", 1)[1] for value in command if value.startswith("--model=")))
    output_name = next(value.split("=", 1)[1] for value in command if value.startswith("--output="))
    assert output_name == "model", "OMG must receive a bare basename, even below long/punctuated run roots"
    output = kwargs["cwd"] / output_name
    assert model.read_bytes() == b"SIMULATED UNINSPECTED AIR"
    assert (model.parent / "weights/w.bin").read_bytes() == b"SIDE DATA"
    assert kwargs["cwd"] == model.parent
    assert kwargs["env"]["SOC_VERSION"] == "kirinx90"
    assert "--platform=kirinx90" in command
    assert not any(value.startswith("--weight_data_type") for value in command)
    assert not any(value.startswith("--use_origin_format") for value in command)
    output.with_suffix(".omc").write_bytes(b"SIMULATED OMC")
    kwargs["stdout"].write("simulation only\n")


def _help(command, **kwargs):
    kwargs["stdout"].write("OMG usage: --input_shape --dynamic_dims --limit_partition_size --save_weights_as_external_data\n")
    return subprocess.CompletedProcess(command, 0)


def test_existing_air_compiles_without_original_writes_or_runner_abi(case, monkeypatch):
    air, output, ddk = case
    before = {str(p.relative_to(air.parent)): p.read_bytes() for p in air.parent.rglob("*") if p.is_file()}
    monkeypatch.setattr(subprocess, "run", _tool)
    result = module.compile_existing_air(air, output, ddk, ddk_id="MOCK")
    assert result["status"] == "COMPILED" and result["model_compilation"] == "PASSED"
    assert result["device_execution"] == "NOT_RUN" and result["device_ready"] is False
    assert set(result["source_contract"].values()) == {"UNVERIFIED"}
    assert result["legacy_hiai_abi_compatible"] == "UNVERIFIED"
    assert (output / result["omc"]["path"]).read_bytes() == b"SIMULATED OMC"
    assert {row["source_relative"] for row in result["input_snapshots"]} == {"export.air", "weights/w.bin"}
    assert not list((output / "compile-work").rglob("*.sh"))
    assert all(not path.is_symlink() for path in (output / "compile-work").rglob("*"))
    assert before == {str(p.relative_to(air.parent)): p.read_bytes() for p in air.parent.rglob("*") if p.is_file()}
    assert result["source_preservation"]["status"] == "UNCHANGED_BY_STAT"
    assert result["requested_options"] == {}
    assert json.loads((output / "compile-command.json").read_text())["weight_dtype_override"] is None


def test_output_basename_is_independent_of_punctuated_long_run_directory(case, monkeypatch):
    air, output, ddk = case
    output = output.parent / ("run.with spaces-and-punctuation_" * 5) / "result"
    monkeypatch.setattr(subprocess, "run", _tool)
    result = module.compile_existing_air(air, output, ddk, ddk_id="MOCK")
    assert result["status"] == "COMPILED"
    recorded = json.loads((output / "compile-command.json").read_text())
    assert "--output=model" in recorded["argv"]
    assert Path(recorded["cwd"]) == output / "compile-work/generated"
    assert Path(recorded["cwd"]) / "model.omc" == output / result["omc"]["path"]


def test_export_directory_resolves_extensionless_fileconstants_without_source_writes(case, monkeypatch):
    air, output, ddk = case
    source = air.parent.with_name("glm_export_air_cpro_40block_55_quant")
    air.parent.rename(source)
    air = source / air.name
    weight_name = "model_model_layers_0_input_layernorm_weight"
    reference = f"./{source.name}/{weight_name}"
    # This is a mock AIR payload, used only to exercise the actual file-open
    # behavior of the compiler's reported relative FileConstant reference.
    air.write_text(json.dumps({"file_constant": reference}))
    (source / weight_name).write_bytes(b"EXPORTED PHYSICAL WEIGHT")
    (source / "model.bin").write_bytes(b"SAFE NESTED SIDECAR")
    (source / ".hidden_weight").write_bytes(b"UNSELECTED HIDDEN FILE")
    (source / "weight_alias").symlink_to(source / weight_name)
    (source / "outside/extensionless_weight").write_bytes(b"UNSELECTED SUBTREE")
    before = {p: (p.read_bytes(), module._stat(p)) for p in source.rglob("*")
              if p.is_file() and not p.is_symlink()}

    def tool(command, **kwargs):
        if "--help" in command:
            return _help(command, **kwargs)
        model = Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("--model=")))
        cwd = kwargs["cwd"]
        assert model == cwd / "export.air"
        file_constant = json.loads(model.read_text())["file_constant"]
        assert file_constant == reference
        assert (cwd / file_constant).read_bytes() == b"EXPORTED PHYSICAL WEIGHT"
        assert (model.parent / file_constant).read_bytes() == b"EXPORTED PHYSICAL WEIGHT"
        assert (cwd / source.name / "weights/w.bin").read_bytes() == b"SIDE DATA"
        assert (cwd / source.name / "model.bin").read_bytes() == b"SAFE NESTED SIDECAR"
        assert not (cwd / weight_name).exists()
        assert not (cwd / source.name / "export.air").exists()
        assert "--output=model" in command
        (cwd / "model.omc").write_bytes(b"SIMULATED OMC")

    monkeypatch.setattr(subprocess, "run", tool)
    result = module.compile_existing_air(air, output, ddk, ddk_id="MOCK", input_layout="export-directory")
    assert result["status"] == "COMPILED"
    rows = {row["source_relative"]: row for row in result["input_snapshots"]}
    assert set(rows) == {"export.air", weight_name, "model.bin", "weights/w.bin"}
    assert rows["export.air"]["path"] == "compile-work/generated/export.air"
    assert rows[weight_name]["path"] == f"compile-work/generated/{source.name}/{weight_name}"
    assert {row["path"] for row in result["generated_artifacts"]} == {"compile-work/generated/model.omc"}
    skipped = {row["path"]: row["reason"] for row in result["external_weights"]["skipped"]}
    assert skipped["weight_alias"] == "SYMLINK_NOT_FOLLOWED"
    assert ".hidden_weight" in skipped and skipped["outside"] == "UNSELECTED_DIRECTORY"
    assert all(not p.is_symlink() for p in (output / "compile-work").rglob("*"))
    assert before == {p: (p.read_bytes(), module._stat(p)) for p in before}
    assert result["source_preservation"]["status"] == "UNCHANGED_BY_STAT"
    for row in rows.values():
        snapshot = output / row["path"]
        assert row["sha256"] == module.sha256_file(snapshot)
        assert module._stat(snapshot)[:2] != module._stat(Path(row["source"]))[:2]


def test_default_air_directory_includes_extensionless_sidecars_in_allowed_roots(case, monkeypatch):
    air, output, ddk = case
    (air.parent / "layer_weight").write_bytes(b"ROOT WEIGHT")
    (air.parent / "weights/layer_scale").write_bytes(b"NESTED SCALE")

    def tool(command, **kwargs):
        result = _tool(command, **kwargs)
        if "--help" not in command:
            assert (kwargs["cwd"] / "layer_weight").read_bytes() == b"ROOT WEIGHT"
            assert (kwargs["cwd"] / "weights/layer_scale").read_bytes() == b"NESTED SCALE"
        return result

    monkeypatch.setattr(subprocess, "run", tool)
    result = module.compile_existing_air(air, output, ddk, ddk_id="MOCK")
    assert result["status"] == "COMPILED"
    assert {row["source_relative"] for row in result["input_snapshots"]} == {
        "export.air", "weights/w.bin", "layer_weight", "weights/layer_scale"}


@pytest.mark.parametrize("folder_name", ["model", "model.weights"])
def test_export_directory_cannot_collide_with_compiler_output_root(case, monkeypatch, folder_name):
    air, output, ddk = case
    source = air.parent.with_name(folder_name)
    air.parent.rename(source)
    air = source / air.name

    def help_only(command, **kwargs):
        assert "--help" in command, "input layout collision must prevent model compilation"
        return _help(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", help_only)
    result = module.compile_existing_air(air, output, ddk, ddk_id="MOCK", input_layout="export-directory")
    assert result["status"] == "FAILED" and result["model_compilation"] == "NOT_RUN"
    assert "reserved compiler output basename" in result["error"]
    assert (source / "weights/w.bin").read_bytes() == b"SIDE DATA"


def test_unknown_input_layout_fails_before_output_or_tool(case, monkeypatch):
    air, output, ddk = case
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("invalid layout must not invoke tool"))
    with pytest.raises(ValueError, match="input_layout"):
        module.compile_existing_air(air, output, ddk, ddk_id="MOCK", input_layout="unknown")
    assert not output.exists()


def test_reserved_output_prefix_cannot_overwrite_staged_input(case, monkeypatch):
    air, output, ddk = case
    collision = air.parent / "model.bin"
    collision.write_bytes(b"ORIGINAL SIDECAR")
    def help_only(command, **kwargs):
        assert "--help" in command
        return _help(command, **kwargs)
    monkeypatch.setattr(subprocess, "run", help_only)
    result = module.compile_existing_air(air, output, ddk, ddk_id="MOCK")
    assert result["status"] == "FAILED" and result["model_compilation"] == "NOT_RUN"
    assert "reserved compiler output basename" in result["error"]
    assert collision.read_bytes() == b"ORIGINAL SIDECAR"


def test_snapshot_copy_fallback_rejects_insufficient_disk_before_copy(case, monkeypatch):
    air, output, ddk = case
    sources = [air, air.parent / "weights/w.bin"]
    before = {path: (path.read_bytes(), module._stat(path)) for path in sources}
    total_bytes = sum(len(value[0]) for value in before.values())
    progress = []

    def no_clone(*args):
        raise OSError(errno.EOPNOTSUPP, "reflink unavailable")

    def no_space(fd):
        # Selection accounting must be persisted before any physical copy.
        selection = json.loads((output / "input-selection.json").read_text())
        assert selection["file_count"] == 2 and selection["total_bytes"] == total_bytes
        return SimpleNamespace(f_bavail=0, f_frsize=4096)

    def help_only(command, **kwargs):
        assert command[-1] == "--help", "disk failure must prevent model compilation"
        return _help(command, **kwargs)

    monkeypatch.setattr(module.fcntl, "ioctl", no_clone)
    monkeypatch.setattr(module.os, "fstatvfs", no_space)
    monkeypatch.setattr(subprocess, "run", help_only)
    result = module.compile_existing_air(air, output, ddk, ddk_id="MOCK", progress=progress.append)
    assert result["status"] == "FAILED" and result["stage"] == "input-snapshot"
    assert result["model_compilation"] == "NOT_RUN" and result["input_snapshots"] == []
    assert "physical copy needs" in result["error"] and "Free disk space" in result["error"]
    assert (output / "compile-work/generated/export.air").read_bytes() == b""
    assert not (output / "compile-work/generated/weights/w.bin").exists()
    assert before == {path: (path.read_bytes(), module._stat(path)) for path in sources}
    assert result["input_selection"]["total_bytes"] == total_bytes
    assert any(row.get("total") == 2 and row.get("total_bytes") == total_bytes for row in progress)


def test_snapshot_copy_fallback_with_sufficient_disk_preserves_exact_bytes(case, monkeypatch):
    air, output, ddk = case
    sources = [air, air.parent / "weights/w.bin"]
    before = {path: (path.read_bytes(), module._stat(path)) for path in sources}
    checks = []

    def no_clone(*args):
        raise OSError(errno.EXDEV, "cross-filesystem reflink unavailable")

    def sufficient_space(fd):
        checks.append(fd)
        return SimpleNamespace(f_bavail=(1 << 30) // 4096, f_frsize=4096)

    monkeypatch.setattr(module.fcntl, "ioctl", no_clone)
    monkeypatch.setattr(module.os, "fstatvfs", sufficient_space)
    monkeypatch.setattr(subprocess, "run", _tool)
    result = module.compile_existing_air(air, output, ddk, ddk_id="MOCK")
    assert result["status"] == "COMPILED" and len(checks) == 2
    assert result["input_selection"]["file_count"] == 2
    assert result["input_selection"]["total_bytes"] == sum(len(value[0]) for value in before.values())
    for row in result["input_snapshots"]:
        assert row["snapshot_method"] == "COPY"
        assert (output / row["path"]).read_bytes() == before[Path(row["source"])][0]
        assert row["sha256"] == module.sha256_file(output / row["path"])
    assert before == {path: (path.read_bytes(), module._stat(path)) for path in sources}


@pytest.mark.parametrize("failure", ["compiler", "timeout", "empty"])
def test_failure_retains_logs_dumps_context_and_original(case, monkeypatch, failure):
    air, output, ddk = case
    def tool(command, **kwargs):
        if "--help" in command:
            return _help(command, **kwargs)
        scratch = kwargs["cwd"]
        (scratch / "dump_graph").mkdir()
        (scratch / "dump_graph/failed.irpb").write_bytes(b"FAILED GRAPH")
        kwargs["stdout"].write("Get format configs failed\n")
        if failure == "compiler":
            raise subprocess.CalledProcessError(1, command)
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 3)
    monkeypatch.setattr(subprocess, "run", tool)
    result = module.compile_existing_air(air, output, ddk, ddk_id="MOCK")
    assert result["status"] == "FAILED" and not (output / "manifest.json").exists()
    assert (output / "failure.json").is_file()
    assert (output / "diagnostics/failed.irpb").read_bytes() == b"FAILED GRAPH"
    assert (output / "compile-work/generated/export.air").read_bytes() == air.read_bytes()
    assert "Get format configs failed" in (output / "report.txt").read_text()


@pytest.mark.parametrize("audit,expected", [("APPLIED", "COMPILED"), ("NOT_OBSERVED", "COMPILED"),
                                            ("REJECTED", "FAILED"), ("MALFORMED", "FAILED")])
def test_reuses_tbe_diagnosis_and_fails_closed_on_invalid_audit(case, monkeypatch, audit, expected):
    air, output, ddk = case
    diagnosis = output.parent / "existing-diagnosis"
    calls = []
    def prepare(selected, retained, destination, *, inherited):
        calls.append((selected, retained))
        return inherited, {"audit_log": str(destination / "audit.jsonl")}
    monkeypatch.setattr(module, "prepare_tbe_compatibility", prepare)
    monkeypatch.setattr(module, "verify_tbe_compatibility", lambda metadata: {"status": audit})
    monkeypatch.setattr(subprocess, "run", _tool)
    result = module.compile_existing_air(air, output, ddk, ddk_id="MOCK", tbe_diagnosis=diagnosis)
    assert result["status"] == expected and calls == [(ddk, diagnosis)]
    assert result["tbe_compatibility"]["compile"] == "PASSED"
    assert result["tbe_compatibility"]["audit"]["status"] == ("AUDIT_FAILED" if audit == "MALFORMED" else audit)
    assert (output / "tbe-compatibility-result.json").is_file()
    assert (output / "manifest.json").exists() == (expected == "COMPILED")


def test_invalid_diagnosis_prevents_compiler_and_source_overlap_is_rejected(case, monkeypatch):
    air, output, ddk = case
    def prepare(*args, **kwargs):
        raise ValueError("diagnosis does not match selected DDK")
    monkeypatch.setattr(module, "prepare_tbe_compatibility", prepare)
    def help_only(command, **kwargs):
        assert command[-1] == "--help", "invalid diagnosis must prevent actual compilation"
        return _help(command, **kwargs)
    monkeypatch.setattr(subprocess, "run", help_only)
    result = module.compile_existing_air(air, output, ddk, ddk_id="MOCK", tbe_diagnosis="old")
    assert result["status"] == "FAILED"
    assert result["tbe_compatibility"]["compiler_invocation"] == "NOT_ATTEMPTED"
    assert result["tbe_compatibility"]["audit"]["status"] == "NOT_PREPARED"
    with pytest.raises(ValueError, match="outside"):
        module.compile_existing_air(air, air.parent / "new", ddk, ddk_id="MOCK")


def test_existing_output_and_symlink_sidecars_cannot_modify_source(case, monkeypatch):
    air, output, ddk = case
    (air.parent / "alias.bin").symlink_to(air.parent / "weights/w.bin")
    output.mkdir()
    (output / "keep").write_text("OLD")
    with pytest.raises(FileExistsError):
        module.compile_existing_air(air, output, ddk, ddk_id="MOCK")
    assert list(output.iterdir()) == [output / "keep"]
    monkeypatch.setattr(subprocess, "run", _tool)
    result = module.compile_existing_air(air, output.parent / "second", ddk, ddk_id="MOCK")
    assert {row["path"]: row["reason"] for row in result["external_weights"]["skipped"]}["alias.bin"] == "SYMLINK_NOT_FOLLOWED"


def test_explicit_elf_and_original_conversion_flags_are_passed_as_single_arguments(case, monkeypatch):
    air, output, ddk = case
    binary = ddk / "tools/tools_omg/omg-actual"
    binary.write_bytes(b"\x7fELF SIMULATED EXECUTABLE")
    binary.chmod(0o755)
    alias = ddk / "tools/tools_omg/omg-selected"
    alias.symlink_to(binary)
    shape = "tokens:1,-1;mask:1,-1;state:1,32,128,128"
    gears = "64,64;1,1"
    calls, progress = [], []
    def tool(command, **kwargs):
        calls.append(command)
        assert command[0] == str(binary)
        if "--help" not in command:
            assert command[-4:] == [f"--input_shape={shape}", f"--dynamic_dims={gears}",
                                    "--limit_partition_size=false", "--save_weights_as_external_data=true"]
        return _tool(command, **kwargs)
    monkeypatch.setattr(subprocess, "run", tool)
    result = module.compile_existing_air(air, output, ddk, ddk_id="MOCK", omg_path=alias,
        input_shape=shape, dynamic_dims=gears, limit_partition_size=False,
        save_weights_as_external_data=True, progress=progress.append)
    assert result["status"] == "COMPILED"
    assert len(calls) == 2 and calls[0] == [str(binary), "--help"]
    assert result["omg_path"] == str(binary)
    assert result["help_check"]["missing"] == []
    assert {row["stage"] for row in progress} >= {"omg-help", "input-snapshot", "omg"}
    assert (output / "help.log").is_file() and (output / "help-command.json").is_file()


@pytest.mark.parametrize("use_origin_format", [True, False])
def test_explicit_fp16_profile_reaches_compiler_and_retained_metadata(case, monkeypatch, use_origin_format):
    air, output, ddk = case
    before = {path: path.read_bytes() for path in (air, air.parent / "weights/w.bin")}
    commands = []

    def tool(command, **kwargs):
        if "--help" in command:
            # Internal flags absent from public help are still passed unchanged.
            return _help(command, **kwargs)
        commands.append(list(command))
        assert command.count("--use_origin_format=" + str(use_origin_format).lower()) == 1
        assert command.count("--weight_data_type=FP16") == 1
        assert "--limit_partition_size=false" in command
        assert "--save_weights_as_external_data=true" in command
        model = Path(next(value.split("=", 1)[1] for value in command if value.startswith("--model=")))
        assert model.read_bytes() == before[air]
        assert model.parent == kwargs["cwd"]
        assert (kwargs["cwd"] / "weights/w.bin").read_bytes() == before[air.parent / "weights/w.bin"]
        (kwargs["cwd"] / "model.omc").write_bytes(b"SIMULATED FP16 PROFILE OMC")
        kwargs["stdout"].write("mock profile command accepted; no actual OMG invocation\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", tool)
    result = module.compile_existing_air(
        air, output, ddk, ddk_id="MOCK", use_origin_format=use_origin_format,
        weight_data_type="FP16", limit_partition_size=False, save_weights_as_external_data=True)
    assert result["status"] == "COMPILED" and len(commands) == 1
    assert result["requested_options"] == {
        "use_origin_format": str(use_origin_format).lower(), "weight_data_type": "FP16",
        "limit_partition_size": "false", "save_weights_as_external_data": "true",
    }
    assert set(result["help_check"]["missing"]) == {"use_origin_format", "weight_data_type"}
    recorded = json.loads((output / "compile-command.json").read_text())
    assert recorded["argv"] == commands[0]
    assert recorded["weight_dtype_override"] == "FP16"
    assert {path: path.read_bytes() for path in before} == before
    assert result["device_execution"] == "NOT_RUN" and result["device_ready"] is False


@pytest.mark.parametrize("options", [
    {"dynamic_dims": "64;1"}, {"input_shape": "x:1,-1\n--bad"},
    {"input_shape": "x:1,-1", "dynamic_dims": "64;$(touch bad)"},
    {"limit_partition_size": "false"}, {"save_weights_as_external_data": 1},
    {"use_origin_format": "true"}, {"use_origin_format": 1},
    {"weight_data_type": "FP32"}, {"weight_data_type": "fp16"},
    {"weight_data_type": True}, {"weight_data_type": ["FP16"]},
])
def test_bad_dynamic_or_boolean_options_fail_before_output_or_tool(case, monkeypatch, options):
    air, output, ddk = case
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("invalid options must not invoke tool"))
    with pytest.raises(ValueError):
        module.compile_existing_air(air, output, ddk, ddk_id="MOCK", **options)
    assert not output.exists()


def test_explicit_vendor_script_and_foreign_elf_are_rejected_before_execution(case, monkeypatch):
    air, output, ddk = case
    script = ddk / "tools/tools_omg/omg"
    script.write_text("#!/bin/sh\n# vendor wrapper\n")
    script.chmod(0o755)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("wrapper must not run"))
    with pytest.raises(ValueError, match="ELF.*master/omg"):
        module.compile_existing_air(air, output, ddk, ddk_id="MOCK", omg_path=script)
    foreign = output.parent / "foreign-omg"
    foreign.write_bytes(b"\x7fELF MOCK")
    foreign.chmod(0o755)
    with pytest.raises(ValueError, match="within the selected DDK"):
        module.compile_existing_air(air, output, ddk, ddk_id="MOCK", omg_path=foreign)
    assert not output.exists()


def test_unadvertised_internal_flags_are_preserved_for_actual_compiler_validation(case, monkeypatch):
    air, output, ddk = case
    commands, events = [], []
    def tool(command, **kwargs):
        commands.append(command)
        if command[-1] == "--help":
            kwargs["stdout"].write("Usage: --input_shape --d11ynamic_dims --limit_partition_size\n")
            return subprocess.CompletedProcess(command, 0)
        assert "--dynamic_dims=64;1" in command
        assert "--save_weights_as_external_data=true" in command
        assert "--input_shape=tokens:1,-1" in command
        assert (kwargs["cwd"] / "export.air").read_bytes() == air.read_bytes()
        kwargs["stdout"].write("Unknown option --dynamic_dims\n")
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(subprocess, "run", tool)
    result = module.compile_existing_air(air, output, ddk, ddk_id="MOCK",
        input_shape="tokens:1,-1", dynamic_dims="64;1", save_weights_as_external_data=True,
        progress=events.append)
    assert result["status"] == "FAILED" and result["stage"] == "omg"
    assert result["model_compilation"] == "FAILED" and result["input_snapshots"]
    assert len(commands) == 2 and result["returncode"] == 1
    check = json.loads((output / "help-check.json").read_text())
    assert check["status"] == "OPTIONS_NOT_ADVERTISED"
    assert check["missing"] == ["dynamic_dims", "save_weights_as_external_data"]
    assert any("compiler will validate" in event["message"] for event in events)
    assert (output / "compile-command.json").is_file()


def test_help_loader_error_is_not_option_evidence(case, monkeypatch):
    air, output, ddk = case
    def tool(command, **kwargs):
        assert command[-1] == "--help"
        kwargs["stdout"].write("error while loading shared libraries: libmissing.so: cannot open shared object file\n")
        return subprocess.CompletedProcess(command, 127)
    monkeypatch.setattr(subprocess, "run", tool)
    monkeypatch.setattr(module, "_snapshot", lambda *a: pytest.fail("loader error must stop before copying"))
    result = module.compile_existing_air(air, output, ddk, ddk_id="MOCK")
    assert result["status"] == "FAILED" and result["stage"] == "omg-help"
    assert result["help_check"]["status"] == "INVALID_HELP_RESPONSE"
    assert result["input_snapshots"] == []


@pytest.mark.parametrize("compile_fails", [False, True])
def test_external_data_and_auxiliary_outputs_are_retained_hashed_even_on_failure(case, monkeypatch, compile_fails):
    air, output, ddk = case
    def tool(command, **kwargs):
        if "--help" in command:
            return _help(command, **kwargs)
        _tool(command, **kwargs)
        generated = kwargs["cwd"]
        (generated / "model.weights").mkdir()
        (generated / "model.weights/chunk0.bin").write_bytes(b"EXTERNAL WEIGHTS")
        (generated / "model.omc.data").write_bytes(b"MORE EXTERNAL DATA")
        (kwargs["cwd"] / "compiler-extra.bin").write_bytes(b"RETAIN WORK TOO")
        if compile_fails:
            raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(subprocess, "run", tool)
    result = module.compile_existing_air(air, output, ddk, ddk_id="MOCK", save_weights_as_external_data=True)
    assert result["status"] == ("FAILED" if compile_fails else "COMPILED")
    assert {row["path"] for row in result["generated_artifacts"]} == {
        "compile-work/generated/model.omc", "compile-work/generated/model.omc.data",
        "compile-work/generated/model.weights/chunk0.bin", "compile-work/generated/compiler-extra.bin"}
    assert len(result["external_data_candidates"]) == 3
    assert any(row["path"] == "compile-work/generated/compiler-extra.bin" for row in result["generated_artifacts"])
    for row in result["generated_artifacts"] + result["compiler_work_artifacts"]:
        assert row["sha256"] == module.sha256_file(output / row["path"])
        assert row["stat"] == module._stat(output / row["path"])
    assert result["omc"]["role"] == "primary_omc"
