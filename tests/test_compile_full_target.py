"""The full-target entry invokes existing AIR compilation, never probe/weight rebuilds."""
import json
from pathlib import Path
import runpy
import sys

import pytest

from qwen35_moe_dflash import existing_air


@pytest.fixture
def cli(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("CPRO_DDK_ROOT", str(tmp_path / "ddk"))
    monkeypatch.setenv("CPRO_DDK_ID", "TEST-DDK")
    air = tmp_path / "original/export.air"
    air.parent.mkdir()
    air.write_bytes(b"original opaque AIR")
    output = tmp_path / "job"
    script = Path(__file__).resolve().parents[1] / "targets/cpro/scripts/compile_full_target.py"
    main = runpy.run_path(str(script))["main"]
    monkeypatch.setattr(sys, "argv", [str(script), "--air", str(air), "--output", str(output),
                                     "--tbe-diagnosis", str(tmp_path / "diagnosis"), "--skip-inspection"])
    return main, air, output


def test_default_cli_reuses_air_and_diagnosis_with_user_dynamic_contract(cli, monkeypatch):
    main, air, output = cli
    seen = []

    def compile_air(source, destination, ddk, **options):
        assert Path(source).read_bytes() == b"original opaque AIR"
        assert options["ddk_id"] == "TEST-DDK"
        assert options["dynamic_dims"] == "64,64,64,64,64;1,1,1,1,1"
        assert options["input_shape"].count("-1") == 5
        assert options["limit_partition_size"] is False
        assert options["save_weights_as_external_data"] is True
        assert options["input_layout"] == "export-directory"
        assert options["tbe_diagnosis"] == str(output.parent / "diagnosis")
        assert options["omg_path"] is None
        seen.append(source)
        return {"status": "COMPILED", "model_compilation": "PASSED", "omc": {"path": "generated/model.omc"}}

    monkeypatch.setattr(existing_air, "compile_existing_air", compile_air)
    assert main() == 0
    assert seen == [air]
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "COMPILED" and result["device_execution"] == "NOT_RUN"
    assert result["interface_inspection"] == "NOT_RUN" and result["device_ready"] is False
    assert air.read_bytes() == b"original opaque AIR"


def test_cli_can_select_air_directory_relative_weight_layout(cli, monkeypatch):
    main, air, _ = cli
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--input-layout", "air-directory"])
    seen = []

    def compile_air(source, destination, ddk, **options):
        seen.append((source, options["input_layout"]))
        return {"status": "COMPILED", "model_compilation": "PASSED", "omc": {"path": "generated/model.omc"}}

    monkeypatch.setattr(existing_air, "compile_existing_air", compile_air)
    assert main() == 0
    assert seen == [(air, "air-directory")]


def test_compile_failure_retained_and_no_inspection(cli, monkeypatch):
    main, _, output = cli
    monkeypatch.setattr(sys, "argv", sys.argv[:-1])
    monkeypatch.setattr(existing_air, "compile_existing_air", lambda *a, **k: {
        "status": "FAILED", "model_compilation": "FAILED", "error": "unsupported dynamic flag"})
    assert main() == 1
    result = json.loads((output / "result.json").read_text())
    assert result["error"] == "unsupported dynamic flag"
    assert "interface_inspection" not in result and not (output / "interface").exists()


def test_compiler_diagnostics_visible_without_repeating_full_argv(cli, monkeypatch, capsys):
    main, _, output = cli
    original_error = "CalledProcessError: ['omg', '--input_shape=" + "RefData_0:1,8192,4;" * 500 + "'] failed"

    def failed(source, destination, *args, **kwargs):
        destination.mkdir()
        (destination / "compile-command.json").write_text(original_error)
        (destination / "compile.log").write_text(
            'ERROR: fixture compiler root cause\n' + '[INFO] trailing cleanup\n' * 1000)
        return {"status": "FAILED", "model_compilation": "FAILED", "stage": "omg",
                "returncode": 1, "error": original_error}

    monkeypatch.setattr(existing_air, "compile_existing_air", failed)
    assert main() == 1
    stdout = capsys.readouterr().out
    summary = next(json.loads(line) for line in stdout.splitlines() if line.startswith('{'))
    assert summary["error"] == "OMG exited with status 1" and summary["stage"] == "omg"
    assert summary["result_json"] == str(output / "result.json")
    assert 'L1: ERROR: fixture compiler root cause' in stdout
    assert "RefData_0:1,8192,4" not in stdout and len(stdout) < 15000
    assert json.loads((output / "result.json").read_text())["error"] == original_error
    assert (output / "compilation/compile-command.json").read_text() == original_error


def test_help_failure_displays_help_log_when_compilation_did_not_start(cli, monkeypatch, capsys):
    main, _, output = cli

    def failed(source, destination, *args, **kwargs):
        destination.mkdir()
        (destination / "help.log").write_text('error: fixture missing tool library\n')
        return {"status": "FAILED", "model_compilation": "NOT_RUN", "stage": "omg-help",
                "error": "ValueError: help check failed"}

    monkeypatch.setattr(existing_air, "compile_existing_air", failed)
    assert main() == 1
    stdout = capsys.readouterr().out
    assert "fixture missing tool library" in stdout and "diagnostics from " in stdout
    assert str(output / "compilation/help.log") in stdout
    result = json.loads((output / "result.json").read_text())
    assert result["model_compilation"] == "NOT_RUN" and result["stage"] == "omg-help"


def test_timeout_summary_keeps_full_error_on_disk(cli, monkeypatch, capsys):
    main, _, output = cli
    error = "TimeoutExpired: --input_shape=" + "long-argument;" * 500
    monkeypatch.setattr(existing_air, "compile_existing_air", lambda *a, **k: {
        "status": "FAILED", "model_compilation": "FAILED", "stage": "omg",
        "timeout_seconds": 12, "error": error})
    assert main() == 1
    stdout = capsys.readouterr().out
    assert "OMG timed out after 12 seconds" in stdout and "long-argument" not in stdout
    assert json.loads((output / "result.json").read_text())["error"] == error


def test_existing_output_is_not_overwritten(cli, monkeypatch):
    main, _, output = cli
    output.mkdir()
    (output / "failure.json").write_text("retained")
    monkeypatch.setattr(existing_air, "compile_existing_air", lambda *a, **k: pytest.fail("must not compile"))
    assert main() == 1
    assert (output / "failure.json").read_text() == "retained"
    assert list(output.iterdir()) == [output / "failure.json"]


def test_bad_environment_does_not_start_work(cli, monkeypatch):
    main, _, output = cli
    monkeypatch.delenv("CPRO_DDK_ID")
    monkeypatch.setattr(existing_air, "compile_existing_air", lambda *a, **k: pytest.fail("must not compile"))
    assert main() == 1 and not output.exists()


@pytest.mark.parametrize("failure", ["import", "inspection"])
def test_optional_inspection_cannot_relabel_successful_compile(cli, monkeypatch, failure):
    main, _, output = cli
    monkeypatch.setattr(sys, "argv", sys.argv[:-1])
    monkeypatch.setattr(existing_air, "compile_existing_air", lambda *a, **k: {
        "status": "COMPILED", "model_compilation": "PASSED", "omc": {"path": "generated/model.omc"}})
    if failure == "import":
        monkeypatch.setitem(sys.modules, "qwen35_moe_dflash.existing_omc_inspect", None)
    else:
        from qwen35_moe_dflash import existing_omc_inspect
        def fail(*a, **k):
            raise ValueError("fixture conversion unavailable")
        monkeypatch.setattr(existing_omc_inspect, "inspect_existing_omc", fail)
    assert main() == 0
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "COMPILED"
    assert result["interface_inspection"] == "UNAVAILABLE"
    assert result["device_execution"] == "NOT_RUN"


@pytest.mark.parametrize("board_status", ["EXECUTED_UNVALIDATED", "FAILED"])
def test_original_tool_failure_or_missing_reference_never_relabels_model_compilation_or_accuracy(cli, monkeypatch, board_status):
    from qwen35_moe_dflash import original_target_board
    main, _, output = cli
    monkeypatch.setattr(sys, "argv", [*sys.argv[:-1], "--run-board", "--input-dir", str(output.parent / "original-inputs")])
    monkeypatch.setattr(existing_air, "compile_existing_air", lambda *a, **k: {
        "status": "COMPILED", "model_compilation": "PASSED", "omc": {"path": "generated/model.omc"}})
    seen = []
    def execute(compilation, contract, destination, **options):
        assert compilation == output / "compilation"
        assert contract == output / "target-contract/report.json"
        assert options["gear"] == 64 and options["host_input_dir"].endswith("original-inputs")
        seen.append(destination)
        return {"status": board_status, "device_execution": "FAILED" if board_status == "FAILED" else "EXECUTED_OUTPUTS_RETRIEVED",
                "numerical_validation": "NOT_RUN_NO_REFERENCE", "error": "simulated tool failure"}
    monkeypatch.setattr(original_target_board, "execute_target_tool", execute)
    assert main() == (1 if board_status == "FAILED" else 0)
    result = json.loads((output / "result.json").read_text())
    assert seen == [output / "board"]
    assert result["status"] == "COMPILED" and result["model_compilation"] == "PASSED"
    assert result["board_status"] == board_status and result["numerical_validation"] == "NOT_RUN_NO_REFERENCE"
    assert result["device_ready"] is False
