"""Independent trial copies change one control word and never run a model."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest

from qwen35_moe_dflash import target_prefill_last_row_trial as trial
from qwen35_moe_dflash.target_prefill_smoke_plan import prepare_prefill_smoke_plan, validate_prefill_smoke_plan, _source, _identity
from test_target_prefill_smoke_plan import case as planner_case
import test_target_prefill_decode_bridge as bridge_fixture


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    source = planner_case.__wrapped__(tmp_path)
    root = source["source"]
    (root / "output_pos.bin").write_bytes(struct.pack("<i", 64))
    (root / "input_ids.bin").write_bytes(struct.pack("<64i", *range(64)))
    plan = prepare_prefill_smoke_plan(source["compilation"], source["probe"], root)
    execution = tmp_path / "retained-prefill"; execution.mkdir()
    (execution / "plan.json").write_text(json.dumps(plan))
    report = {"status": "PREFILL_NAMED_SLOTS_AND_RETAINED_FILES_BOUND_BRIDGE_GAPS_REMAIN", "input_count": 89, "output_count": 82,
        "executed_input_byte_binding": "ALL89_SAVED_PRIVATE_SLOT_SHA256_MATCHED_CURRENT13_ORIGINAL_SOURCES",
        "named_probe_witness": {"status": "POST_HOC_SAME_MODEL_ORDERED_DESCRIPTOR_NAMES_BOUND"},
        "original_source_inputs": plan["source_inputs"],
        "active_source_identities": [{"path": row["path"], "identity": row["identity"]} for row in plan["source_inputs"]]
            + [{"path": str(execution / "plan.json"), "identity": _identity((execution / "plan.json").stat())}]}
    calls = []
    def bridge(execution_arg, probe_arg, output):
        calls.append((execution_arg, probe_arg, output))
        Path(output).mkdir(parents=True)
        (Path(output) / "report.json").write_text(json.dumps(report))
        return deepcopy(report)
    monkeypatch.setattr(trial, "inspect_prefill_decode_bridge", bridge)
    return {**source, "execution": execution, "plan": plan, "report": report, "calls": calls,
        "out": tmp_path / "last-row-trial", "tmp": tmp_path}


def run(case):
    return trial.prepare_prefill_last_row_trial(case["execution"], case["probe"], case["out"])


def test_trial_creates14_independent_files_only_control_changes_and_new_planner_accepts(case, monkeypatch):
    old = {path: (path.read_bytes(), _identity(path.stat())) for directory in
        (case["source"], case["execution"]) for path in directory.rglob("*") if path.is_file()}
    opened = Path.open
    with monkeypatch.context() as guarded:
        def no_models(path, *args, **kwargs):
            assert path.suffix not in {".omc", ".air", ".weight"}
            return opened(path, *args, **kwargs)
        guarded.setattr(Path, "open", no_models)
        result = run(case)
    assert result["status"] == "TARGET_PREFILL_LAST_ROW_TRIAL_INPUTS_PREPARED", result
    assert result["target_input_count"] == 89 and result["derived_file_count"] == 14
    assert result["target_source_file_count"] == 13 and result["auxiliary_source_file_count"] == 1
    assert result["byte_identical_target_source_file_count"] == 12 and result["changed_target_source_file_count"] == 1
    assert result["control_replacement"]["old_values"] == [64] and result["control_replacement"]["new_values"] == [63]
    assert result["source_python_last_row_trial"]["old_index_out_of_range"] == "NOT_CLAIMED"
    assert result["numerical_validation"] == result["device_execution"] == result["model_compilation"] == "NOT_RUN"
    assert set(Path(result["input_dir"]).iterdir()) == {Path(row["path"]) for row in result["derived_inputs"]}
    for row in result["derived_inputs"]:
        path, source = Path(row["path"]), Path(row["source"]["path"])
        assert path.stat().st_nlink == 1 and path.stat().st_ino != source.stat().st_ino and not path.is_symlink()
        expected = struct.pack("<i", 63) if path.name == "output_pos.bin" else old[source][0]
        assert path.read_bytes() == expected
    assert all((path.read_bytes(), _identity(path.stat())) == observed for path, observed in old.items())
    fresh = prepare_prefill_smoke_plan(case["compilation"], case["probe"], result["input_dir"])
    assert validate_prefill_smoke_plan(fresh) == fresh
    assert fresh["model_sha256"] == case["plan"]["model_sha256"]
    assert len(fresh["source_inputs"]) == 13 and len(fresh["inputs"]) == 89
    old_sha = {tuple(row["compiled_candidates"]): row["sha256"] for row in case["plan"]["inputs"]}
    changed = [row for row in fresh["inputs"] if row["sha256"] != old_sha[tuple(row["compiled_candidates"])]]
    assert len(changed) == 1 and changed[0]["compiled_candidates"] == ["arg1238_1"]
    assert json.loads((case["out"] / "manifest.json").read_text()) == result


@pytest.mark.parametrize("fault", ["failed_bridge", "missing_names", "wrong_count", "unbound_inputs", "missing_sources"])
def test_incomplete_retained_evidence_never_creates_trial_inputs(case, fault):
    report = case["report"]
    if fault == "failed_bridge": report["status"] = "FAILED"
    elif fault == "missing_names": report["named_probe_witness"] = {}
    elif fault == "wrong_count": report["input_count"] = 90
    elif fault == "unbound_inputs": report["executed_input_byte_binding"] = "UNVERIFIED"
    else: report["original_source_inputs"] = report["original_source_inputs"][:-1]
    result = run(case)
    assert result["status"] == "BLOCKED" and not (case["out"] / "inputs").exists()


@pytest.mark.parametrize("fault", ["missing", "wrong_size", "symlink"])
def test_auxiliary_ids_are_required_exact256B_regular_and_never_synthesized(case, fault):
    path = case["source"] / "input_ids.bin"
    if fault == "wrong_size": path.write_bytes(bytes(255))
    else:
        path.unlink()
        if fault == "symlink":
            target = case["tmp"] / "other-ids.bin"; target.write_bytes(bytes(256)); path.symlink_to(target)
    result = run(case)
    assert result["status"] == "BLOCKED" and result["stage"] == "bind-auxiliary-original-ids"
    assert not (case["out"] / "inputs").exists()


def test_already63_is_not_misreported_as_a_changed_control_experiment(case):
    path = case["source"] / "output_pos.bin"; path.write_bytes(struct.pack("<i", 63))
    replacement = _source(path, 4)
    case["report"]["original_source_inputs"] = [replacement if Path(row["path"]).name == path.name else row
        for row in case["report"]["original_source_inputs"]]
    case["report"]["active_source_identities"] = [{"path": row["path"], "identity": replacement["identity"]}
        if row["path"] == str(path) else row for row in case["report"]["active_source_identities"]]
    result = run(case)
    assert result["status"] == "BLOCKED" and "already selects row 63" in result["error"]
    assert not (case["out"] / "inputs").exists()


def test_original_source_change_after_validation_is_refused(case, monkeypatch):
    copy = trial._copy
    changed = False
    def change(*args, **kwargs):
        nonlocal changed
        result = copy(*args, **kwargs)
        if not changed:
            path = case["source"] / "input_ids.bin"; os.chmod(path, path.stat().st_mode & 0o777)
            changed = True
        return result
    monkeypatch.setattr(trial, "_copy", change)
    result = run(case)
    assert result["status"] == "BLOCKED" and "changed" in result["error"]


def test_derived_input_change_before_manifest_cannot_be_reported_prepared(case, monkeypatch):
    copy = trial._copy
    def change(*args, **kwargs):
        result = copy(*args, **kwargs)
        path = Path(result["path"]); os.chmod(path, path.stat().st_mode & 0o777)
        return result
    monkeypatch.setattr(trial, "_copy", change)
    result = run(case)
    assert result["status"] == "BLOCKED" and "changed after its validated read" in result["error"]


@pytest.mark.parametrize("role", ["original", "derived"])
def test_change_after_metadata_write_is_caught_before_prepared_manifest_is_published(case, monkeypatch, role):
    save = trial.save_json
    def change(path, value):
        save(path, value)
        if Path(path).name == ".manifest.pending.json":
            changed = case["source"] / "input_ids.bin" if role == "original" else case["out"] / "inputs/output_pos.bin"
            os.chmod(changed, changed.stat().st_mode & 0o777)
    monkeypatch.setattr(trial, "save_json", change)
    result = run(case)
    assert result["status"] == "BLOCKED" and "changed within" in result["error"]
    assert json.loads((case["out"] / "manifest.json").read_text())["status"] == "BLOCKED"
    assert (case["out"] / ".manifest.pending.json").is_file()


def test_existing_output_and_output_inside_original_inputs_are_rejected_before_writes(case):
    case["out"].mkdir(); marker = case["out"] / "marker"; marker.write_text("preserve")
    with pytest.raises(FileExistsError): run(case)
    assert marker.read_text() == "preserve" and not case["calls"]
    nested = case["source"] / "not-an-owned-output"
    before = _identity(case["source"].stat())
    with pytest.raises(ValueError, match="separate"):
        trial.prepare_prefill_last_row_trial(case["execution"], case["probe"], nested)
    assert not nested.exists() and _identity(case["source"].stat()) == before


def test_real_saved_prefill_named_bridge_and_actual_inputs_feed_trial_without_model_reads(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    original_case = bridge_fixture.source_case
    def source_with64(path):
        result = original_case.__wrapped__(path)
        (result["source"] / "output_pos.bin").write_bytes(struct.pack("<i", 64))
        return result
    monkeypatch.setattr(bridge_fixture, "source_case", SimpleNamespace(__wrapped__=source_with64))
    actual = bridge_fixture.case.__wrapped__(tmp_path, monkeypatch)
    (actual["source"]["source"] / "input_ids.bin").write_bytes(struct.pack("<64i", *([220] * 64)))
    opened = Path.open
    def no_models(path, *args, **kwargs):
        assert path.suffix not in {".omc", ".air", ".weight"}
        return opened(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", no_models)
    result = trial.prepare_prefill_last_row_trial(actual["execution"], actual["named"], tmp_path / "real-bridge-trial")
    assert result["status"] == "TARGET_PREFILL_LAST_ROW_TRIAL_INPUTS_PREPARED", result
    assert result["original_input_ids"]["values"] == [220] * 64
    assert result["control_replacement"]["old_values"] == [64]
    assert len(result["derived_inputs"]) == 14 and len(result["source_bindings"]) > 100


def test_cli_prints_small_preparation_summary(case, capsys):
    script = Path(__file__).resolve().parents[1] / "targets/cpro/scripts/prepare_target_prefill_last_row_trial.py"
    spec = importlib.util.spec_from_file_location("prefill_last_row_trial_cli_under_test", script)
    cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    assert cli.main(["--execution", str(case["execution"]), "--named-runtime-probe", str(case["probe"]),
        "--output", str(case["out"])]) == 0
    text = capsys.readouterr().out
    value = json.loads(text)
    assert len(text.encode()) < 4096 and value["derived_file_count"] == 14
    assert value["manifest"] == str(case["out"] / "manifest.json")
    assert "source_bindings" not in value and value["device_execution"] == "NOT_RUN"
