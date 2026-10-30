"""Real-extent retained child files exercise byte-chain inspection without HDC."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct

import pytest

from qwen35_moe_dflash import original_target_board as board
from qwen35_moe_dflash import target_decode_continuation_run as runner
from qwen35_moe_dflash.existing_air import _stat
from test_original_target_primary_revalidation import ctime_only
from test_target_decode_continuation_board import case as board_case


def identity(path):
    return {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def case(tmp_path, monkeypatch):
    value = board_case.__wrapped__(tmp_path, monkeypatch)
    parents = tmp_path / "parent-state-outputs"
    parents.mkdir()
    bindings = []
    for row in value["plan"]["inputs"]:
        if not row["name"].startswith("RefData_"):
            continue
        state_id = int(row["name"].split("_")[1])
        output = next(item for item in value["plan"]["outputs"]
            if item["runtime_tensor_name"] == f"output_{state_id + 1}_fixture")
        path = parents / f"retained_parent_{state_id:03d}"
        with path.open("wb") as stream:
            stream.truncate(row["bytes"])
            stream.write(struct.pack("<I", state_id + 1))
        bindings.append({"source_refdata": row["name"], "runtime_input_index": row["index"],
            "runtime_output_index": output["index"], "output_file": {"path": str(path),
                "bytes": row["bytes"], "sha256": row["sha256"], "stat": _stat(path)}})
    value["plan"]["state_feedback_bindings"] = bindings
    value.update(board_root=tmp_path / "child-board", wrapper_out=tmp_path / "continuation-wrapper")
    return value


def emit(case, root=None):
    root = Path(root or case["board_root"])
    (root / "outputs").mkdir(parents=True)
    plan = case["plan"]
    remote = "/data/local/tmp/fixture-child/run-unique"
    retrieved, paths, lines = {}, {}, []
    for row in plan["outputs"]:
        # A permutation of filenames ensures only the log establishes slots.
        name = f"outputs/retained_{(row['index'] * 17 + 3) % 82:03d}.bin"
        path = root / name
        with path.open("wb") as stream:
            stream.truncate(row["bytes"])
            if row["index"] == plan["logits_runtime_output_index"]:
                stream.write(struct.pack("<4f", -3, 1, 8, 2))
            elif row["index"] == plan["feature_runtime_output_index"]:
                stream.write(struct.pack("<4e", -2, 0, 3, 1))
            else:
                stream.write(struct.pack("<e", row["index"] + 1))
        paths[row["index"]] = name
        retrieved[name] = identity(path)
        lines.append(f"output[{row['index']}] :{remote}/{name}")
    for name, raw in (("tool.stdout", ("\n".join(reversed(lines)) + "\n").encode()),
                      ("tool.stderr", b""), ("tool.exitcode", b"0\n"), ("tool.pid", b"5337\n")):
        (root / name).write_bytes(raw)
        retrieved[name] = identity(root / name)
    observed = {"artifact_kind": "qwen35-original-target-tool-execution", "status": "EXECUTED_UNVALIDATED",
        "device_execution": "EXECUTED_OUTPUTS_RETRIEVED", "remote_exitcode": 0, "tool_process_id": 5337,
        "retrieval_errors": [], "lock_state": "NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT",
        "source_preservation": "UNCHANGED_CONTINUATION_PROOF_AND_ALL89_PREPARED_INPUT_SHA256",
        "board_attempt": remote, "retrieved": retrieved,
        "inputs": [dict(row, host_source=row["source"], source=remote + "/source-inputs/" + row["source_filename"])
            for row in plan["inputs"]]}
    return root, observed, paths


def inspect(case, root, observed):
    return runner.inspect_continuation_outputs(case["plan"], root, observed)


def test_true_extent_child_files_bind_all80_parent_input_output_byte_chains_and_real_stats(case):
    root, observed, paths = emit(case)
    result = inspect(case, root, observed)
    assert result["status"] == "ALL80_PARENT_OUTPUT_BYTES_USED_BY_CHILD_INPUTS"
    assert result["feedback_input_count"] == result["changed_state_output_count"] == 80
    assert result["bound_child_output_count"] == 82
    assert result["numerical_validation"] == "NOT_RUN_NO_REFERENCE"
    assert result["state_continuity"] == "BYTE_CHAIN_OBSERVED_CONTROL_SEMANTICS_AND_NUMERICAL_CONTINUITY_NOT_VALIDATED"
    assert len({row["parent_output"]["sha256"] for row in result["states"]}) == 80
    for row in result["states"]:
        parent = next(item for item in case["plan"]["state_feedback_bindings"] if item["source_refdata"] == row["source_refdata"])
        assert row["parent_output"] == parent["output_file"]
        assert row["executed_input_sha256"] == row["parent_output"]["sha256"]
        assert row["child_output"]["path"] == paths[parent["runtime_output_index"]]
        assert row["child_output"]["sha256"] == identity(root / row["child_output"]["path"])["sha256"]
    logits = result["bound_tensors"]["logits"]["statistics"]
    feature = result["bound_tensors"]["feature"]["statistics"]
    assert logits["elements"] == logits["finite_count"] == 248320
    assert logits["finite_min"] == -3 and logits["finite_max"] == 8
    assert logits["argmax"]["token_id"] == 2 and logits["argmax"]["value"] == 8
    assert feature["elements"] == feature["finite_count"] == 16384
    assert feature["finite_min"] == -2 and feature["finite_max"] == 3


@pytest.mark.parametrize("field", ["index", "name", "runtime_tensor_name", "sha256", "host_source", "source_filename", "bytes", "shape"])
def test_every_executed_input_slot_must_match_its_prepared_name_type_extent_and_source(case, field):
    root, observed, _ = emit(case)
    row = observed["inputs"][20]
    row[field] = row[field] + 1 if field in {"index", "bytes"} else [7] if field == "shape" else "different"
    with pytest.raises(ValueError, match="executed child input"):
        inspect(case, root, observed)


@pytest.mark.parametrize("fault", ["parent_sha", "parent_bytes", "duplicate_owner", "duplicate_input", "duplicate_output", "feature_slot", "logits_slot"])
def test_feedback_requires80_individual_parent_bytes_and_only_the80_state_outputs(case, fault):
    root, observed, _ = emit(case)
    rows = case["plan"]["state_feedback_bindings"]
    if fault == "parent_sha": rows[0]["output_file"]["sha256"] = "f" * 64
    elif fault == "parent_bytes": rows[0]["output_file"]["bytes"] += 2
    elif fault == "duplicate_owner": rows[0]["source_refdata"] = rows[1]["source_refdata"]
    elif fault == "duplicate_input": rows[0]["runtime_input_index"] = rows[1]["runtime_input_index"]
    elif fault == "duplicate_output": rows[0]["runtime_output_index"] = rows[1]["runtime_output_index"]
    else: rows[0]["runtime_output_index"] = case["plan"]["feature_runtime_output_index" if fault == "feature_slot" else "logits_runtime_output_index"]
    with pytest.raises(ValueError):
        inspect(case, root, observed)


@pytest.mark.parametrize("fault", ["output_hash", "output_extent", "missing_output", "duplicate_log_slot", "outside_log_path", "exitcode", "pid"])
def test_child_files_and_explicit_log_mapping_are_required_in_addition_to_success_status(case, fault):
    root, observed, paths = emit(case)
    name = paths[case["plan"]["state_feedback_bindings"][0]["runtime_output_index"]]
    if fault == "output_hash":
        with (root / name).open("r+b") as stream: stream.write(b"changed")
    elif fault == "output_extent":
        (root / name).write_bytes(b"short")
        observed["retrieved"][name] = identity(root / name)
    elif fault == "missing_output": observed["retrieved"].pop(name)
    elif fault in {"duplicate_log_slot", "outside_log_path"}:
        path = root / "tool.stdout"
        text = path.read_text()
        text = text + text.splitlines()[0] + "\n" if fault == "duplicate_log_slot" else text.replace(
            observed["board_attempt"] + "/outputs/", "/data/local/tmp/unrelated-attempt/outputs/", 1)
        path.write_text(text)
        observed["retrieved"]["tool.stdout"] = identity(path)
    else:
        path = root / ("tool.exitcode" if fault == "exitcode" else "tool.pid")
        path.write_bytes(b"7\n")
        observed["retrieved"][path.name] = identity(path)
    with pytest.raises(ValueError):
        inspect(case, root, observed)


@pytest.mark.parametrize("payload", ["state", "log"])
def test_output_or_log_drift_after_its_fresh_hash_is_not_reconciled(case, monkeypatch, payload):
    root, observed, paths = emit(case)
    state_path = root / paths[case["plan"]["state_feedback_bindings"][0]["runtime_output_index"]]
    selected = state_path if payload == "state" else root / "tool.stdout"
    original, changed = runner._hash_retrieved, False
    def race(path, row, **options):
        nonlocal changed
        result = original(path, row, **options)
        if Path(path) == selected and not changed:
            changed = True
            ctime_only(selected)
        return result
    monkeypatch.setattr(runner, "_hash_retrieved", race)
    with pytest.raises(ValueError, match="changed during"):
        inspect(case, root, observed)


def test_plan_only_wrapper_preserves_proof_and_never_invokes_board(case, monkeypatch):
    monkeypatch.setattr(board, "execute_target_tool", lambda *a, **k: pytest.fail("plan-only cannot contact board"))
    result = runner.run_decode_continuation(case["prepared"], case["wrapper_out"])
    assert result["status"] == "DECODE_CONTINUATION_PLANNED" and result["mode"] == "PLAN_ONLY"
    assert result["device_execution"] == "NOT_RUN" and result["state_feedback"] == "NOT_RUN"
    assert result["state_continuity"] == "NOT_VALIDATED"
    assert json.loads((case["wrapper_out"] / "plan.json").read_text()) == case["plan"]


def fake_board(case, monkeypatch):
    calls = []
    def execute(compilation, contract, output, **options):
        calls.append((compilation, contract, output, options))
        root, observed, _ = emit(case, output)
        (root / "result.json").write_text(json.dumps(observed))
        return observed
    monkeypatch.setattr(board, "execute_target_tool", execute)
    return calls


def test_apply_wrapper_observes_byte_chain_and_numerics_without_accuracy_acceptance(case, monkeypatch):
    calls = fake_board(case, monkeypatch)
    result = runner.run_decode_continuation(case["prepared"], case["wrapper_out"], apply=True,
        board_model_root="/data/local/tmp/fixture-owned-cache", timeout_seconds=91, lock_wait_seconds=17)
    assert result["status"] == "DECODE_CONTINUATION_EXECUTED_UNVALIDATED", result
    assert result["state_feedback"] == "ALL80_PARENT_OUTPUT_BYTES_USED_BY_CHILD_INPUTS"
    assert result["feedback_input_count"] == 80 and result["bound_child_output_count"] == 82
    assert result["numerical_validation"] == "NOT_RUN_NO_REFERENCE"
    assert result["logits"]["argmax"]["token_id"] == 2
    assert result["feature"]["finite_count"] == 16384
    assert "NUMERICAL_CONTINUITY_NOT_VALIDATED" in result["state_continuity"]
    assert len(calls) == 1
    options = calls[0][3]
    assert options["runtime_continuation_prepared"] == case["prepared"]
    assert options["host_input_dir"] == case["plan"]["input_dir"] and options["gear"] == 1
    assert options["timeout_seconds"] == 91 and options["lock_wait_seconds"] == 17
    assert "runtime_decode_probe" not in options and "runtime_prefill_probe" not in options
    chain = json.loads((case["wrapper_out"] / "state-chain.json").read_text())
    assert chain["feedback_input_count"] == 80


def test_board_failure_is_retained_and_never_inspected_as_a_successful_chain(case, monkeypatch):
    monkeypatch.setattr(board, "execute_target_tool", lambda *a, **k: {
        "status": "FAILED", "device_execution": "NOT_RUN", "error": "fixture board cache rejected"})
    monkeypatch.setattr(runner, "inspect_continuation_outputs", lambda *a, **k: pytest.fail("failed board result cannot be inspected"))
    result = runner.run_decode_continuation(case["prepared"], case["wrapper_out"], apply=True)
    assert result["status"] == "FAILED" and "fixture board cache rejected" in result["error"]
    assert result["state_feedback"] == "NOT_RUN" and "state_chain" not in result


def test_changed_preparation_after_board_execution_cannot_be_promoted(case, monkeypatch):
    calls = fake_board(case, monkeypatch)
    case["validation"]["change_at"] = 2
    result = runner.run_decode_continuation(case["prepared"], case["wrapper_out"], apply=True)
    assert len(calls) == 1 and result["status"] == "FAILED", result
    assert "preparation changed" in result["error"]
    assert result["state_feedback"] == "NOT_RUN" and "state_chain" not in result
