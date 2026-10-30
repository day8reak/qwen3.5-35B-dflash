"""Retained decode outputs bind real named-plan fixtures; no board/model read."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import runpy
import stat
import struct
import time

import pytest

from qwen35_moe_dflash import target_decode_output_inventory as inventory
from qwen35_moe_dflash.target_compile_failure import _read_small
from qwen35_moe_dflash.target_decode_smoke_plan import prepare_decode_smoke_plan
from qwen35_moe_dflash.target_decode_trial_inputs import prepare_decode_trial_inputs
from test_target_decode_smoke_plan import case as source_case


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def record(path):
    return {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def refresh(case):
    proof = case["proof"]
    remote = Path(case["report"]["board_attempt"])
    plan = dict(proof, inputs=[dict(row, host_source=row["source"],
        source=str(remote / "source-inputs" / row["source_filename"])) for row in proof["inputs"]])
    save(case["board"] / "plan.json", plan)
    save(case["board"] / "runtime-decode-proof.json", proof)
    save(case["execution"] / "plan.json", proof)
    case["report"]["plan_sha256"] = record(case["board"] / "plan.json")["sha256"]
    save(case["board"] / "result.json", case["report"])
    outer = dict(case["report"], artifact_kind="qwen35-target-decode-smoke-job", mode="APPLY",
        board_result=str(case["board"] / "result.json"), plan=str(case["execution"] / "plan.json"), **case.get("trial", {}))
    save(case["execution"] / "result.json", outer)


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    sources = source_case.__wrapped__(tmp_path)
    proof = prepare_decode_smoke_plan(sources["compilation"], sources["probe"], sources["source"])
    execution = tmp_path / "decode-execution"
    board = execution / "board"
    (board / "outputs").mkdir(parents=True)
    remote, model = "/data/local/tmp/fixture-target/run-decode", "/data/local/tmp/fixture-target/payload/model/model.omc"
    report = {"artifact_kind": "qwen35-original-target-tool-execution", "status": "EXECUTED_UNVALIDATED",
        "device_execution": "EXECUTED_OUTPUTS_RETRIEVED", "remote_exitcode": 0, "tool_process_id": 29654,
        "execution_scope": "SINGLE_DECODE1_USER_SUPPLIED_INPUTS_NO_PREFILL_FEEDBACK", "retrieval_errors": [],
        "lock_state": "NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT",
        "source_preservation": "UNCHANGED_DECODE_PLAN_METADATA_AND_USER_INPUT_SHA256",
        "gear": 1, "runtime_decode_probe": proof["runtime_probe"], "board_attempt": remote,
        "model_payloads": [{"path": model, "sha256": proof["model_sha256"], "bytes": 1234}], "retrieved": {}}
    paths, lines = [], ["[WARNING] [debug]: from img skip realpath", "Inference succeeded"]
    for desc in proof["outputs"]:
        # Reverse-numbered filenames deliberately differ from runtime indices.
        name = f"outputs/retrieved_{81 - desc['index']:03d}"
        path = board / name
        with path.open("wb") as stream:
            stream.truncate(desc["bytes"])
        report["retrieved"][name] = record(path)
        paths.append(name)
        lines.append(f"output[{desc['index']}] :{remote}/{name}")
    for name, raw in (("tool.stdout", "\n".join(lines).encode()), ("tool.stderr", b""),
                      ("tool.exitcode", b"0\n"), ("tool.pid", b"29654\n")):
        (board / name).write_bytes(raw)
        report["retrieved"][name] = record(board / name)
    save(board / "command.json", {"argv": [proof["tool_binding"]["path"], "--model=" + model,
        "--input=" + ",".join(remote + "/" + row["file"] for row in proof["inputs"]),
        "--modelmanager_itf=V2", "--direct_from_file=true", "--times=1", "--weight_dir=" + str(Path(model).parent),
        "--output_dir=" + remote + "/outputs", "--input_type=" + proof["input_type"], "--output_type=" + proof["output_type"]]})
    value = {"sources": sources, "execution": execution, "board": board, "proof": proof,
             "report": report, "paths": paths, "out": tmp_path / "decode-inventory"}
    refresh(value)
    return value


def inspect(case, **options):
    return inventory.inspect_target_decode_outputs(case["execution"], case["out"],
        include_state_statistics=options.pop("include_state_statistics", False), **options)


def change_log(case, text):
    (case["board"] / "tool.stdout").write_text(text)
    case["report"]["retrieved"]["tool.stdout"] = record(case["board"] / "tool.stdout")
    refresh(case)


def test_decode_unique32768byte_feature_and_logits_follow_explicit_log_not_filenames(case):
    result = inspect(case)
    assert result["status"] == "DECODE_OUTPUT_FILES_BOUND_TO_LOGGED_SLOTS", result
    assert result["actual_retrieved_output_file_count"] == result["plan_output_count"] == 82
    assert result["bound_tensors"]["feature"]["runtime_index"] == 0
    assert result["bound_tensors"]["feature"]["path"] == "outputs/retrieved_081"
    assert result["bound_tensors"]["feature"]["bytes"] == 32768
    assert result["bound_tensors"]["feature"]["statistics"]["elements"] == 16384
    assert result["bound_tensors"]["logits"]["runtime_index"] == 81
    assert result["bound_tensors"]["logits"]["bytes"] == 993280
    assert result["state_layer_binding"] == "UNVERIFIED" and result["state_feedback"] == "NOT_PERFORMED"
    assert result["state_continuity"] == "NOT_VALIDATED"
    assert result["numerical_validation"] == "NOT_RUN_NO_REFERENCE"
    assert result["source_validation"] == "FRESH_DECODE_PLAN_METADATA_AND_USER_INPUT_SHA256_NO_MODEL_READ"
    assert result["model_payload_access"] == result["weight_payload_access"] == "NONE"
    assert result["trial_inputs"]["status"] == "NOT_RECORDED"
    assert result["raw_extent_counts"] == [
        {"bytes": 32768, "expected": 1, "observed": 1, "missing": 0, "excess": 0},
        {"bytes": 65536, "expected": 30, "observed": 30, "missing": 0, "excess": 0},
        {"bytes": 993280, "expected": 1, "observed": 1, "missing": 0, "excess": 0},
        {"bytes": 1048576, "expected": 30, "observed": 30, "missing": 0, "excess": 0},
        {"bytes": 2097152, "expected": 20, "observed": 20, "missing": 0, "excess": 0}]


def _change_only_primary_ctime(case):
    path = case["sources"]["model"]
    before = path.stat()
    os.chmod(path, before.st_mode ^ 0o100)
    os.chmod(path, before.st_mode)
    after = path.stat()
    assert (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) == (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    assert before.st_ctime_ns != after.st_ctime_ns
    return path


def test_saved_inventory_remains_untouched_when_primary_ctime_is_revalidated(case):
    saved = inspect(case)
    assert saved["status"] == "DECODE_OUTPUT_FILES_BOUND_TO_LOGGED_SLOTS"
    evidence = [case["out"] / "report.json", case["execution"] / "plan.json",
        case["board"] / "runtime-decode-proof.json", case["sources"]["compilation"] / "report.json"]
    before = {path: path.read_bytes() for path in evidence}
    model = _change_only_primary_ctime(case)
    observations = []
    assert inventory.validate_target_decode_output_inventory(case["out"], revalidation_observations=observations) == saved
    assert observations, "the new validation must retain its small-primary hash observation"
    assert all(path.read_bytes() == raw for path, raw in before.items())
    assert saved["model_payload_access"] == "NONE"
    assert any(str(model) in json.dumps(row) for row in observations)


def test_new_inventory_reports_small_primary_rehash_after_ctime_change(case):
    _change_only_primary_ctime(case)
    result = inspect(case)
    assert result["status"] == "DECODE_OUTPUT_FILES_BOUND_TO_LOGGED_SLOTS", result
    assert result["model_payload_access"] == "SMALL_PRIMARY_OMC_REHASHED_FOR_CTIME_ONLY"
    assert result["weight_payload_access"] == "NONE"
    assert result["source_validation"] == "FRESH_DECODE_PLAN_METADATA_AND_USER_INPUT_SHA256_WITH_CTIME_ONLY_PRIMARY_REHASH"
    assert result["primary_omc_revalidation_observations"]
    assert inventory.validate_target_decode_output_inventory(case["out"]) == result


def test_typed_all80state_nonfinite_summary_and_logit_ties_are_not_accuracy(case):
    state = next(row for row in case["proof"]["outputs"] if row["index"] not in {0, 81})
    for index, raw in ((state["index"], struct.pack("<4e", float("nan"), float("inf"), -float("inf"), -7)),
                       (0, struct.pack("<3e", float("nan"), float("inf"), -float("inf"))),
                       (81, struct.pack("<4f", 0, 7, 7, 2))):
        path = case["board"] / case["paths"][index]
        with path.open("r+b") as stream:
            stream.write(raw)
        case["report"]["retrieved"][case["paths"][index]] = record(path)
    refresh(case)
    result = inspect(case, include_state_statistics=True)
    assert result["status"] == "DECODE_OUTPUT_FILES_BOUND_TO_LOGGED_SLOTS", result
    stats = result["state_statistics"]
    assert stats["tensor_count"] == len(stats["entries"]) == 80
    assert stats["elements"] == (30 * 65536 + 30 * 1048576 + 20 * 2097152) // 2
    assert stats["nonfinite_count"] == 3
    assert stats["nan_count"] == stats["positive_infinity_count"] == stats["negative_infinity_count"] == 1
    assert stats["finite_min"] == -7 and stats["finite_max"] == 0
    assert result["bound_tensors"]["feature"]["statistics"]["nan_count"] == 1
    assert result["bound_tensors"]["logits"]["statistics"]["argmax"]["token_id"] == 1
    assert result["numerical_validation"] == "NOT_RUN_NO_REFERENCE"
    assert stats["state_layer_binding"] == "UNVERIFIED"


@pytest.mark.parametrize("fault", ["missing", "duplicate", "outside", "malformed", "extent", "extra"])
def test_unbound_or_extra_files_never_promote_numeric_filenames_or_extents(case, fault):
    text = (case["board"] / "tool.stdout").read_text()
    first = next(line for line in text.splitlines() if line.startswith("output[0]"))
    if fault == "missing": text = text.replace(first, "no slot statement")
    elif fault == "duplicate": text += "\n" + first
    elif fault == "outside": text = text.replace("run-decode/outputs/retrieved_081", "other-attempt/outputs/retrieved_081")
    elif fault == "malformed": text = text.replace("output[0]", "output[0x]")
    elif fault == "extent": text = text.replace(first, first.replace("retrieved_081", "retrieved_080"))
    else:
        extra = case["board"] / "outputs/extra-profile"
        extra.write_bytes(b"retained")
        case["report"]["retrieved"]["outputs/extra-profile"] = record(extra)
    change_log(case, text)
    result = inspect(case)
    assert result["status"] == "DECODE_OUTPUT_FILES_INSPECTED_SLOT_BINDING_UNVERIFIED", result
    assert result["bound_tensors"] == {} and result["logged_slot_mapping"]["status"] == "UNVERIFIED"
    assert result["actual_retrieved_output_file_count"] == (83 if fault == "extra" else 82)


@pytest.mark.parametrize("fault", ["hash", "exit", "scope", "preservation", "lock", "proof", "model", "format", "shape", "input_order", "source"])
def test_changed_source_or_execution_evidence_fails_without_model_or_device_read(case, fault, monkeypatch):
    if fault == "hash":
        with (case["board"] / case["paths"][0]).open("r+b") as stream: stream.write(b"x")
    elif fault == "source":
        (case["sources"]["source"] / "position_ids.bin").write_bytes(struct.pack("<i", 777))
    elif fault in {"format", "shape", "input_order"}:
        command = json.loads((case["board"] / "command.json").read_text())
        if fault == "input_order":
            pos = next(index for index, arg in enumerate(command["argv"]) if arg.startswith("--input="))
            command["argv"][pos] += ",/unexpected.bin"
        else: command["argv"].append("--output_format=ND" if fault == "format" else "--input_shape=arg:1")
        save(case["board"] / "command.json", command)
    elif fault == "proof":
        changed = deepcopy(case["proof"]); changed["gear"] = 64
        save(case["board"] / "runtime-decode-proof.json", changed)
    else:
        if fault == "exit": case["report"]["remote_exitcode"] = 255
        elif fault == "scope": case["report"]["execution_scope"] = "PREFILL_CONTINUATION"
        elif fault == "preservation": case["report"]["source_preservation"] = "UNKNOWN"
        elif fault == "lock": case["report"]["lock_state"] = "RETAINED_OR_UNCERTAIN"
        else: case["report"]["model_payloads"][0]["sha256"] = "f" * 64
        refresh(case)
    original = os.open
    def guard(path, *args, **kwargs):
        if str(path).endswith((".omc", ".weight")):
            raise AssertionError("output inspector opened model/weight payload")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(os, "open", guard)
    result = inspect(case)
    assert result["status"] == "FAILED" and result["error"], result


def attach_trial(case, tmp_path):
    (case["sources"]["source"] / "conv_state_pos.bin").unlink()
    root = tmp_path / "explicit-trial"
    trial = prepare_decode_trial_inputs(case["sources"]["source"], root, experimental_single_token_controls=True)
    case["proof"] = prepare_decode_smoke_plan(case["sources"]["compilation"], case["sources"]["probe"], trial["input_dir"])
    case["trial"] = {"trial_input_receipt": _read_small(root / "manifest.json", 8 << 20)[1],
        "experimental_input_assumption": trial["assumption"],
        "trial_input_preservation": "UNCHANGED_ORIGINALS_DERIVED_INPUTS_AND_EXPLICIT_ASSUMPTION"}
    refresh(case)
    return root


def test_trial_assumptions_and_originals_are_freshly_bound_and_cli_is_concise(case, tmp_path, capsys):
    root = attach_trial(case, tmp_path)
    script = Path(__file__).resolve().parents[1] / "targets/cpro/scripts/inspect_target_decode_outputs.py"
    main = runpy.run_path(str(script))["main"]
    assert main(["--execution", str(case["execution"]), "--output", str(case["out"]), "--skip-state-statistics"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["status"] == "DECODE_OUTPUT_FILES_BOUND_TO_LOGGED_SLOTS", shown
    assert "tool_logs" not in shown and "output_files" not in shown
    trial = shown["trial_inputs"]
    assert trial["source_file_count"] == 12 and trial["copied_input_count"] == 9 and trial["generated_control_count"] == 4
    assert trial["manifest"] == case["trial"]["trial_input_receipt"]
    assert len(trial["assumption"]["control_replacements"][0]["new_values"]) == 4
    assert trial["assumption"]["control_replacements"][0]["new_value_count"] == 128
    full = json.loads(Path(shown["report"]).read_text())
    assert len(full["trial_inputs"]["assumption"]["control_replacements"][0]["new_values"]) == 128
    assert len(full["logged_slot_mapping"]["entries"]) == 82
    assert not (case["sources"]["source"] / "conv_state_pos.bin").exists()
    assert (root / "inputs/conv_state_pos.bin").stat().st_size == 512


@pytest.mark.parametrize("fault", ["manifest", "original", "copied", "generated", "assumption", "missing_receipt"])
def test_changed_trial_provenance_cannot_be_lost_or_silently_accepted(case, tmp_path, fault):
    root = attach_trial(case, tmp_path)
    if fault == "manifest":
        with (root / "manifest.json").open("a") as stream: stream.write("\n")
    elif fault in {"original", "copied", "generated"}:
        path = ((case["sources"]["source"] / "position_ids.bin") if fault == "original" else
                (root / "inputs/position_ids.bin") if fault == "copied" else (root / "inputs/conv_state_pos.bin"))
        with path.open("r+b") as stream: stream.write(b"x")
    else:
        if fault == "assumption": case["trial"]["experimental_input_assumption"]["contract_status"] = "VALIDATED"
        else: case["trial"].pop("trial_input_receipt")
        refresh(case)
    result = inspect(case)
    assert result["status"] == "FAILED" and result["error"], result


def test_source_mutation_during_inventory_is_rejected(case, monkeypatch):
    original = inventory._hash_retrieved
    changed = False
    def mutate(path, row, **options):
        nonlocal changed
        result = original(path, row, **options)
        if not changed:
            changed = True
            (case["sources"]["source"] / "position_ids.bin").write_bytes(struct.pack("<i", 777))
        return result
    monkeypatch.setattr(inventory, "_hash_retrieved", mutate)
    assert inspect(case)["status"] == "FAILED"


def test_existing_output_directory_is_not_overwritten(case):
    case["out"].mkdir()
    marker = case["out"] / "keep"
    marker.write_text("retained")
    with pytest.raises(FileExistsError): inspect(case)
    assert marker.read_text() == "retained"


def test_public_readonly_validator_rebinds_trial_and_exact_output_bytes(case, tmp_path, monkeypatch):
    attach_trial(case, tmp_path)
    result = inspect(case)
    assert result["status"] == "DECODE_OUTPUT_FILES_BOUND_TO_LOGGED_SLOTS", result
    before = record(case["out"] / "report.json")
    original = os.open
    def guard(path, *args, **kwargs):
        if str(path).endswith((".omc", ".weight")):
            raise AssertionError("validator opened a model/weight payload")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(os, "open", guard)
    assert inventory.validate_target_decode_output_inventory(case["out"]) == result
    assert record(case["out"] / "report.json") == before


@pytest.mark.parametrize("fault", ["row_index", "mapping", "feature", "source", "bytes", "trial"])
def test_public_validator_rejects_rebound_inventory_or_source_changes(case, tmp_path, fault):
    root = attach_trial(case, tmp_path)
    report = inspect(case)
    assert report["status"] == "DECODE_OUTPUT_FILES_BOUND_TO_LOGGED_SLOTS", report
    if fault == "source":
        (case["sources"]["source"] / "position_ids.bin").write_bytes(struct.pack("<i", 888))
    elif fault == "bytes":
        with (case["board"] / case["paths"][0]).open("r+b") as stream: stream.write(b"x")
    elif fault == "trial":
        with (root / "manifest.json").open("a") as stream: stream.write("\n")
    else:
        if fault == "row_index": report["output_files"][0]["runtime_index"] = 99
        elif fault == "mapping": report["logged_slot_mapping"]["entries"][0]["path"] = report["output_files"][0]["path"]
        else: report["bound_tensors"]["feature"]["path"] = report["bound_tensors"]["logits"]["path"]
        save(case["out"] / "report.json", report)
    with pytest.raises((ValueError, OSError)):
        inventory.validate_target_decode_output_inventory(case["out"])


def chmod_ctime(path):
    before = path.stat()
    time.sleep(.02)
    mode = stat.S_IMODE(before.st_mode)
    path.chmod(mode ^ stat.S_IXUSR)
    path.chmod(mode)
    after = path.stat()
    assert (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) == (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    assert before.st_ctime_ns != after.st_ctime_ns


def metadata_paths(case, report):
    records = [*case["sources"]["outer"]["source_bindings"], *case["proof"]["metadata_bindings"],
               *report["evidence_bindings"]]
    return {Path(row["path"]) for row in records} | {case["out"] / "report.json"}


def test_all_historical_metadata_ctime_changes_replay_without_rewriting_receipts_or_payloads(case, tmp_path, monkeypatch):
    attach_trial(case, tmp_path)
    report = inspect(case)
    assert report["status"] == "DECODE_OUTPUT_FILES_BOUND_TO_LOGGED_SLOTS", report
    paths = metadata_paths(case, report)
    original_bytes = {path: path.read_bytes() for path in paths}
    for path in sorted(paths):
        chmod_ctime(path)
    from qwen35_moe_dflash import original_target_board
    monkeypatch.setattr(original_target_board, "Hdc", lambda *a: pytest.fail("metadata recovery must not contact a device"))
    original_open = os.open
    def no_models(path, *args, **options):
        if str(path).endswith((".omc", ".weight")):
            pytest.fail("metadata ctime replay must not open model or weight payloads")
        return original_open(path, *args, **options)
    monkeypatch.setattr(os, "open", no_models)
    observations = []
    assert inventory.validate_target_decode_output_inventory(case["out"],
        metadata_revalidation_observations=observations) == report
    assert observations and all(row["kind"] == "METADATA_CTIME_ONLY_REVALIDATED" for row in observations)
    assert all(row["status"] == "SHA256_VERIFIED" for row in observations)
    assert any(row["path"] == case["trial"]["trial_input_receipt"]["path"] for row in observations)
    assert all(path.read_bytes() == raw for path, raw in original_bytes.items())
    emitted = []
    newer = inventory.inspect_target_decode_outputs(case["execution"], tmp_path / "new-metadata-inventory",
        include_state_statistics=False, metadata_revalidation_observations=emitted)
    assert newer["status"] == "DECODE_OUTPUT_FILES_BOUND_TO_LOGGED_SLOTS", newer
    assert newer["metadata_revalidation_observations"] == emitted and emitted
    assert newer["metadata_identity_revalidation"] == "CTIME_ONLY_FRESH_METADATA_SHA256_OLD_RECEIPTS_UNCHANGED"
    assert newer["trial_inputs"]["manifest"] == report["trial_inputs"]["manifest"]
    assert newer["model_payload_access"] == newer["weight_payload_access"] == "NONE"
    assert all(path.read_bytes() == raw for path, raw in original_bytes.items())


@pytest.mark.parametrize("change", ["content", "mtime", "inode"])
def test_historical_metadata_recovery_rejects_other_changes(case, change):
    report = inspect(case)
    assert report["status"] == "DECODE_OUTPUT_FILES_BOUND_TO_LOGGED_SLOTS", report
    path = case["sources"]["help_report"]
    original = path.stat()
    raw = path.read_bytes()
    if change == "content":
        # Valid JSON with unchanged parsed values still violates recorded SHA.
        path.write_bytes(raw + b"\n")
    elif change == "mtime":
        os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns + 1))
    else:
        alternate = path.with_name("replacement-help.json")
        alternate.write_bytes(raw)
        os.utime(alternate, ns=(original.st_atime_ns, original.st_mtime_ns))
        alternate.replace(path)
    with pytest.raises((ValueError, OSError)):
        inventory.validate_target_decode_output_inventory(case["out"], metadata_revalidation_observations=[])


def test_original_input_ctime_remains_strict_under_metadata_and_derived_payload_recovery(case, tmp_path):
    attach_trial(case, tmp_path)
    report = inspect(case)
    assert report["status"] == "DECODE_OUTPUT_FILES_BOUND_TO_LOGGED_SLOTS", report
    chmod_ctime(case["sources"]["source"] / "position_ids.bin")
    with pytest.raises((ValueError, OSError)):
        inventory.validate_target_decode_output_inventory(case["out"], metadata_revalidation_observations=[],
            payload_revalidation_observations=[])


def test_all_approved_payloads_and_metadata_replay_after_input_modes_are_restored(case, tmp_path):
    trial = attach_trial(case, tmp_path)
    report = inspect(case)
    assert report["status"] == "DECODE_OUTPUT_FILES_BOUND_TO_LOGGED_SLOTS", report
    paths = metadata_paths(case, report)
    originals = {path: path.read_bytes() for path in paths}
    derived = [Path(row["path"]) for row in case["proof"]["source_inputs"]]
    assert len(derived) == 13
    for path in derived:
        path.chmod(0o777)
    with pytest.raises((ValueError, OSError)):
        inventory.validate_target_decode_output_inventory(case["out"], payload_revalidation_observations=[])
    for source in case["proof"]["source_inputs"]:
        Path(source["path"]).chmod(stat.S_IMODE(source["identity"]["mode"]))
    output_paths = [case["board"] / name for name in case["paths"]]
    log_paths = [case["board"] / name for name in inventory._LOGS]
    for path in [*sorted(paths), *derived, *output_paths, *log_paths]:
        chmod_ctime(path)
    metadata_observations, payload_observations = [], []
    current = inventory.validate_target_decode_output_inventory(case["out"],
        metadata_revalidation_observations=metadata_observations,
        payload_revalidation_observations=payload_observations)
    assert current == report
    assert metadata_observations and payload_observations
    assert all(row["kind"] == "PAYLOAD_CTIME_ONLY_REVALIDATED" for row in payload_observations)
    observed = {row["path"] for row in payload_observations}
    assert observed == {str(path) for path in [*derived, *output_paths, *log_paths]}
    assert all(path.read_bytes() == raw for path, raw in originals.items())
    emitted = []
    newer = inventory.inspect_target_decode_outputs(case["execution"], tmp_path / "payload-replayed-inventory",
        include_state_statistics=False, payload_revalidation_observations=emitted)
    assert newer["status"] == "DECODE_OUTPUT_FILES_BOUND_TO_LOGGED_SLOTS", newer
    assert newer["payload_revalidation_observations"] == emitted and emitted
    assert newer["payload_identity_revalidation"] == "CTIME_ONLY_FRESH_PAYLOAD_SHA256_OLD_RECEIPTS_UNCHANGED"
    assert newer["trial_inputs"]["manifest"] == report["trial_inputs"]["manifest"]
    assert all(path.read_bytes() == raw for path, raw in originals.items())


@pytest.mark.parametrize("payload", ["copy", "control", "output", "log"])
@pytest.mark.parametrize("change", ["content", "mtime", "inode"])
def test_approved_payload_ctime_policy_rejects_every_other_identity_or_content_change(case, tmp_path, payload, change):
    trial = attach_trial(case, tmp_path)
    report = inspect(case)
    assert report["status"] == "DECODE_OUTPUT_FILES_BOUND_TO_LOGGED_SLOTS", report
    selected = {"copy": trial / "inputs/inputs_embeds.bin", "control": trial / "inputs/conv_state_pos.bin",
                "output": case["board"] / case["paths"][0], "log": case["board"] / "tool.stdout"}[payload]
    before, raw = selected.stat(), selected.read_bytes()
    if change == "content":
        selected.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
        os.utime(selected, ns=(before.st_atime_ns, before.st_mtime_ns))
    elif change == "mtime":
        os.utime(selected, ns=(before.st_atime_ns, before.st_mtime_ns + 1))
    else:
        alternate = selected.with_name("temporary-replacement")
        alternate.write_bytes(raw)
        os.utime(alternate, ns=(before.st_atime_ns, before.st_mtime_ns))
        alternate.replace(selected)
    with pytest.raises((ValueError, OSError)):
        inventory.validate_target_decode_output_inventory(case["out"], payload_revalidation_observations=[])


def test_output_ctime_change_after_fresh_hash_stays_a_stability_failure(case, monkeypatch):
    assert inspect(case)["status"] == "DECODE_OUTPUT_FILES_BOUND_TO_LOGGED_SLOTS"
    original, changed = inventory._hash_retrieved, False
    def mutate(path, row, **options):
        nonlocal changed
        result = original(path, row, **options)
        if not changed:
            changed = True
            chmod_ctime(Path(path))
        return result
    monkeypatch.setattr(inventory, "_hash_retrieved", mutate)
    with pytest.raises(ValueError, match="changed during"):
        inventory.validate_target_decode_output_inventory(case["out"], payload_revalidation_observations=[])


def test_metadata_change_after_fresh_initial_read_stays_a_stability_failure(case, monkeypatch):
    original, changed = inventory._hash_retrieved, False
    def mutate(path, row, **options):
        nonlocal changed
        result = original(path, row, **options)
        if not changed:
            changed = True
            chmod_ctime(case["execution"] / "plan.json")
        return result
    monkeypatch.setattr(inventory, "_hash_retrieved", mutate)
    result = inspect(case, metadata_revalidation_observations=[])
    assert result["status"] == "FAILED" and "changed during" in result["error"], result


def test_metadata_collector_type_is_explicit_and_does_not_weaken_saved_inventory(case):
    report = inspect(case)
    with pytest.raises(ValueError, match="must be a list"):
        inventory.validate_target_decode_output_inventory(case["out"], metadata_revalidation_observations={})
    assert json.loads((case["out"] / "report.json").read_text()) == report


@pytest.mark.parametrize("mode", ["inspect", "validate"])
@pytest.mark.parametrize("change", ["input_between_passes", "metadata_between_passes", "input_after_final_trial"])
def test_historical_revalidation_never_accepts_source_drift_within_outer_inventory(case, tmp_path, monkeypatch, mode, change):
    trial = attach_trial(case, tmp_path)
    if mode == "validate":
        assert inspect(case)["status"] == "DECODE_OUTPUT_FILES_BOUND_TO_LOGGED_SLOTS"
    if change.endswith("between_passes"):
        original, changed = inventory._hash_retrieved, False
        selected = (trial / "inputs/inputs_embeds.bin" if change.startswith("input")
                    else case["sources"]["help_report"])
        def mutate(path, row, **options):
            nonlocal changed
            result = original(path, row, **options)
            if not changed:
                changed = True
                chmod_ctime(selected)
            return result
        monkeypatch.setattr(inventory, "_hash_retrieved", mutate)
    else:
        original, calls = inventory._trial, 0
        def mutate(*args, **options):
            nonlocal calls
            result = original(*args, **options)
            calls += 1
            if calls == 2:
                chmod_ctime(trial / "inputs/conv_state_pos.bin")
            return result
        monkeypatch.setattr(inventory, "_trial", mutate)
    if mode == "validate":
        with pytest.raises(ValueError, match="changed during"):
            inventory.validate_target_decode_output_inventory(case["out"],
                metadata_revalidation_observations=[], payload_revalidation_observations=[])
    else:
        result = inspect(case, metadata_revalidation_observations=[], payload_revalidation_observations=[])
        assert result["status"] == "FAILED" and "changed during" in result["error"], result
