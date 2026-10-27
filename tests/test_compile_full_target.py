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


def test_compile_failure_retained_and_no_inspection(cli, monkeypatch):
    main, _, output = cli
    monkeypatch.setattr(sys, "argv", sys.argv[:-1])
    monkeypatch.setattr(existing_air, "compile_existing_air", lambda *a, **k: {
        "status": "FAILED", "model_compilation": "FAILED", "error": "unsupported dynamic flag"})
    assert main() == 1
    result = json.loads((output / "result.json").read_text())
    assert result["error"] == "unsupported dynamic flag"
    assert "interface_inspection" not in result and not (output / "interface").exists()


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
