"""Continuation transport tests use a validated-plan stub and simulated HDC.

The plan validator's graph/row lineage is tested separately. These tests only
exercise the board handoff, exact slot bytes, cache boundaries and lock policy.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import struct
import subprocess
import sys
import types

import pytest

from qwen35_moe_dflash import original_target_board as board
from qwen35_moe_dflash import target_decode_smoke_plan as planner
from qwen35_moe_dflash.existing_air import _stat
from qwen35_moe_dflash.existing_air_profile import get_existing_air_profile
from qwen35_moe_dflash.hashing import sha256_file
from qwen35_moe_dflash.target_prefill_smoke_plan import _identity
from test_target_decode_smoke_execution import DecodeFakeHdc, case as decode_case
from test_original_target_primary_revalidation import ctime_only


@pytest.fixture
def case(tmp_path, monkeypatch):
    value = decode_case.__wrapped__(tmp_path, monkeypatch)
    plan = planner.prepare_decode_smoke_plan(value["compilation"], value["probe"], value["source"])
    prepared = tmp_path / "prepared-continuation"
    inputs = prepared / "inputs"
    inputs.mkdir(parents=True)
    sources = []
    for row in plan["inputs"]:
        filename = f"slot_{row['index']:03d}_{row['name']}.bin"
        path = inputs / filename
        if row["name"].startswith("RefData_"):
            # Full declared extents, with a distinct marker for each state.
            with path.open("wb") as stream:
                stream.truncate(row["bytes"])
                stream.write(struct.pack("<I", int(row["name"].split("_")[1]) + 1))
        else:
            shutil.copyfile(row["source"], path)
        row.update(source=str(path), source_filename=filename, sha256=sha256_file(path))
        sources.append({"path": str(path), "bytes": row["bytes"], "sha256": row["sha256"],
            "identity": _identity(path.stat())})
    # Include a real sidecar declaration so cache-hit and SEND paths are both
    # covered. No vendor model or external weight payload is used in this test.
    weight = value["compilation"] / "compile-work/generated/model/SubGraph_0.weight"
    weight.write_bytes(b"fixture-only external weight")
    weight_row = {"path": weight.relative_to(value["compilation"]).as_posix(),
        "bytes": weight.stat().st_size, "sha256": sha256_file(weight), "stat": _stat(weight)}
    value["compiled"]["generated_artifacts"].append(weight_row)
    report_path = value["compilation"] / "report.json"
    report_path.write_text(json.dumps(value["compiled"]))
    report_sha = sha256_file(report_path)
    value["contract"]["compilation_report_sha256"] = report_sha
    (value["job"] / "target-contract/report.json").write_text(json.dumps(value["contract"]))
    plan.update(input_dir=str(inputs), source_inputs=sources, compilation_report_sha256=report_sha,
        execution_scope="SINGLE_SAME_MODEL_DECODE_CONTINUATION_WITH_80_BOUND_STATES")
    state = {"plan": plan, "calls": 0, "change_at": None}
    def validate(directory):
        assert Path(directory) == prepared
        state["calls"] += 1
        current = deepcopy(state["plan"])
        if state["change_at"] is not None and state["calls"] >= state["change_at"]:
            current["inputs"][0]["sha256"] = "f" * 64
        return current
    module = types.ModuleType("qwen35_moe_dflash.target_decode_continuation")
    module.validate_decode_continuation = validate
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(planner, "prepare_decode_smoke_plan", lambda *a, **k: pytest.fail("continuation cannot use the common-state smoke planner"))
    monkeypatch.setattr(planner, "validate_decode_smoke_plan", lambda *a, **k: pytest.fail("continuation cannot use the common-state smoke validator"))
    monkeypatch.setattr(board, "tool_plan", lambda *a, **k: pytest.fail("continuation cannot use legacy input ordering"))
    return dict(value, prepared=prepared, continuation_inputs=inputs, plan=plan, validation=state,
        weight=weight, weight_row=weight_row, out=tmp_path / "continuation-board")


def execute(case, **changes):
    options = dict(profile=get_existing_air_profile("cpro-40block-decode-1"), gear=1,
        host_input_dir=case["continuation_inputs"], target_features=True, lock_wait_seconds=0,
        runtime_continuation_prepared=case["prepared"])
    options.update(changes)
    return board.execute_target_tool(case["compilation"], case["job"] / "target-contract/report.json",
        case["out"], **options)


def cached_payloads(case):
    cache = PurePosixPath("/data/local/tmp/qwen3_5_moe") / ("codex-target-" + sha256_file(case["compilation"] / "report.json")[:20])
    for row in case["compiled"]["generated_artifacts"]:
        remote = cache / "payload" / PurePosixPath(row["path"]).relative_to("compile-work/generated")
        DecodeFakeHdc.remote[str(remote)] = (case["compilation"] / row["path"]).read_bytes()
    return cache


def labels():
    return [label for instance in DecodeFakeHdc.instances for _, label, _ in instance.calls]


def test_all80_distinct_state_files_reach_their_exact_named_runtime_slots(case):
    result = execute(case)
    assert result["status"] == "EXECUTED_UNVALIDATED", result
    assert result["execution_scope"] == case["plan"]["execution_scope"]
    assert result["state_continuity"] == "NOT_VALIDATED_SINGLE_TOOL_INVOCATION"
    assert result["numerical_validation"] == "NOT_RUN_NO_REFERENCE"
    assert result["source_preservation"] == "UNCHANGED_CONTINUATION_PROOF_AND_ALL89_PREPARED_INPUT_SHA256"
    assert result["state_input_layer_identity"] == "ALL80_SAME_MODEL_STATE_FILES_BOUND"
    assert result["runtime_continuation_prepared"] == str(case["prepared"])
    assert len(result["inputs"]) == len({row["source_filename"] for row in result["inputs"]}) == 89
    parent_states = {row["name"]: row for row in case["plan"]["inputs"] if row["name"].startswith("RefData_")}
    assert len(parent_states) == len({row["sha256"] for row in parent_states.values()}) == 80
    for row in result["inputs"]:
        slot = result["board_attempt"] + "/" + row["file"]
        raw = DecodeFakeHdc.remote[slot]
        assert len(raw) == row["bytes"]
        assert hashlib.sha256(raw).hexdigest() == row["sha256"]
        assert row["host_source"] == str(case["continuation_inputs"] / row["source_filename"])
        if row["name"].startswith("RefData_"):
            assert row["sha256"] == parent_states[row["name"]]["sha256"]
            assert struct.unpack("<I", raw[:4])[0] == int(row["name"].split("_")[1]) + 1
    sent = [local for local, _ in DecodeFakeHdc.sends if Path(local).parent == case["continuation_inputs"]]
    assert len(sent) == len(set(sent)) == 89
    assert json.loads((case["out"] / "runtime-continuation-proof.json").read_text()) == case["plan"]
    assert not (case["out"] / "runtime-decode-proof.json").exists()
    argv = json.loads((case["out"] / "command.json").read_text())["argv"]
    assert all(flag in argv for flag in ("--times=1", "--modelmanager_itf=V2", "--direct_from_file=true"))
    assert not any(arg.startswith(("--input_shape=", "--output_shape=", "--input_format=", "--output_format=")) for arg in argv)
    assert next(arg[8:] for arg in argv if arg.startswith("--input=")) == ",".join(
        result["board_attempt"] + "/" + row["file"] for row in result["inputs"])
    assert labels().count("execute-original-target") == 1


@pytest.mark.parametrize("local_weight", ["ctime_changed", "absent"])
def test_owned_cache_hit_does_not_validate_or_open_changed_local_weight(case, monkeypatch, local_weight):
    cached_payloads(case)
    if local_weight == "ctime_changed": ctime_only(case["weight"])
    else: case["weight"].rename(case["weight"].with_name("held-fixture-weight"))
    original = board._recorded_board_payload
    checked = []
    def primary_only(root, row, compiled, **options):
        checked.append(row["path"])
        assert not row["path"].endswith(".weight"), "cached weight must not reach a local payload validator"
        return original(root, row, compiled, **options)
    monkeypatch.setattr(board, "_recorded_board_payload", primary_only)
    original_open = os.open
    def no_weight(path, *args, **options):
        assert not str(path).endswith(".weight"), "cached weight bytes must not be opened"
        return original_open(path, *args, **options)
    monkeypatch.setattr(os, "open", no_weight)
    result = execute(case)
    assert result["status"] == "EXECUTED_UNVALIDATED", result
    assert result["board_disk_budget"]["missing_model_payload_bytes"] == 0
    assert checked == [case["compiled"]["omc"]["path"]]
    assert not any(local.endswith((".omc", ".weight")) for local, _ in DecodeFakeHdc.sends)
    assert all(row["upload"] == "OK" for row in result["model_payloads"])
    assert "SEND_REQUIRES_STRICT_LOCAL_VALIDATION" in result["model_cache_validation"]


def test_empty_declared_sidecar_cache_hit_is_hash_bound_without_local_access(case):
    case["weight"].write_bytes(b"")
    case["weight_row"].update(bytes=0, sha256=hashlib.sha256(b"").hexdigest(), stat=_stat(case["weight"]))
    path = case["compilation"] / "report.json"
    path.write_text(json.dumps(case["compiled"]))
    digest = sha256_file(path)
    case["contract"]["compilation_report_sha256"] = digest
    (case["job"] / "target-contract/report.json").write_text(json.dumps(case["contract"]))
    case["plan"]["compilation_report_sha256"] = digest
    cached_payloads(case)
    case["weight"].unlink()
    result = execute(case)
    assert result["status"] == "EXECUTED_UNVALIDATED", result
    assert result["board_disk_budget"]["missing_model_payload_bytes"] == 0
    row = next(row for row in result["model_payloads"] if row["path"].endswith(".weight"))
    assert row["bytes"] == 0 and row["sha256"] == hashlib.sha256(b"").hexdigest() and row["upload"] == "OK"
    assert not any(local.endswith((".omc", ".weight")) for local, _ in DecodeFakeHdc.sends)


@pytest.mark.parametrize("remote", ["missing", "corrupt"])
def test_cache_send_still_rejects_changed_local_weight_stat_before_input_upload(case, remote):
    cache = cached_payloads(case)
    remote_weight = str(cache / "payload/model/SubGraph_0.weight")
    if remote == "missing": DecodeFakeHdc.remote.pop(remote_weight)
    else: DecodeFakeHdc.remote[remote_weight] = b"corrupt remote payload"
    ctime_only(case["weight"])
    result = execute(case)
    assert result["status"] == "FAILED" and "changed since compilation" in result["error"], result
    assert "execute-original-target" not in labels()
    assert not any(local.endswith((".bin", ".weight")) for local, _ in DecodeFakeHdc.sends)
    assert result["lock_state"] == "NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT"


@pytest.mark.parametrize("at,executions", [(2, 0), (4, 0), (6, 1)])
def test_any_changed_prepared_proof_invalidates_the_current_attempt(case, at, executions):
    case["validation"]["change_at"] = at
    result = execute(case)
    assert result["status"] == "FAILED" and "continuation evidence changed" in str(result), result
    assert labels().count("execute-original-target") == executions
    if executions:
        assert result["source_preservation"] == "FAILED"


@pytest.mark.parametrize("fault", ["decode_conflict", "prefill_conflict", "host_root", "compilation", "report_sha", "duplicate_source", "source_path", "unsafe_filename"])
def test_continuation_proof_conflicts_are_rejected_before_hdc_or_payload_access(case, monkeypatch, fault):
    options = {}
    if fault == "decode_conflict": options["runtime_decode_probe"] = case["probe"]
    elif fault == "prefill_conflict": options["runtime_prefill_probe"] = case["probe"]
    elif fault == "host_root": options["host_input_dir"] = case["source"]
    elif fault == "compilation": case["plan"]["compilation"] += "-other"
    elif fault == "report_sha": case["plan"]["compilation_report_sha256"] = "f" * 64
    elif fault == "duplicate_source": case["plan"]["inputs"][1]["source_filename"] = case["plan"]["inputs"][0]["source_filename"]
    elif fault == "source_path": case["plan"]["inputs"][0]["source"] = str(case["source"] / "effective_length.bin")
    else: case["plan"]["inputs"][0]["source_filename"] = "../escape.bin"
    monkeypatch.setattr(board, "_recorded_board_payload", lambda *a, **k: pytest.fail("invalid proof must precede payload access"))
    result = execute(case, **options)
    assert result["status"] == "FAILED" and result["device_execution"] == "NOT_RUN", result
    assert DecodeFakeHdc.instances == [] and DecodeFakeHdc.sends == []


@pytest.mark.parametrize("fault", ["native", "execute_timeout", "lock_timeout", "release_timeout"])
def test_continuation_failures_preserve_pid_or_uncertain_locks_without_retry(case, monkeypatch, fault):
    if fault == "native": monkeypatch.setattr(DecodeFakeHdc, "status", 7)
    else:
        stage = {"execute_timeout": "execute-original-target", "lock_timeout": "acquire-global-model-io-lock",
            "release_timeout": "release-global-model-io-lock"}[fault]
        monkeypatch.setattr(DecodeFakeHdc, "timeout_label", stage)
    result = execute(case)
    assert result["status"] == "FAILED" and result["numerical_validation"] == "NOT_RUN_NO_REFERENCE", result
    assert labels().count("execute-original-target") == (0 if fault == "lock_timeout" else 1)
    if fault == "native":
        assert result["tool_process_id"] == 1337 and result["remote_exitcode"] == 7
        assert result["lock_state"] == "NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT"
    else:
        assert result["lock_state"] == "RETAINED_OR_UNCERTAIN"
        if fault != "release_timeout": assert "release-global-model-io-lock" not in labels()


@pytest.fixture
def shell_case(case, tmp_path):
    """Capture production shell, then relocate every path into an owned tmp."""
    result = execute(case)
    assert result["status"] == "EXECUTED_UNVALIDATED", result
    remote_cache = result["board_model_cache"]
    remote_root = str(PurePosixPath(remote_cache).parent)
    local_root = tmp_path / "local-shell-cache-root"
    local_cache = local_root / PurePosixPath(remote_cache).name
    commands = {}
    for command, label, _ in DecodeFakeHdc.instances[0].calls:
        if label in {"model-directories", "inspect-model-cache", "own-model-cache"}:
            relocated = command.replace(remote_cache, shlex.quote(str(local_cache))).replace(
                remote_root, shlex.quote(str(local_root)))
            assert remote_cache not in relocated and remote_root not in relocated
            commands[label] = relocated
    assert len(commands) == 3
    local_root.mkdir()
    return {"case": case, "cache": local_cache, "commands": commands,
        "manifest": (case["out"] / "model-files.tsv").read_text().replace(remote_cache, str(local_cache))}


def shell(value, label):
    return subprocess.run(["sh", "-c", value["commands"][label]], capture_output=True, text=True, timeout=10)


def populate_cache(value):
    case, cache = value["case"], value["cache"]
    for row in case["compiled"]["generated_artifacts"]:
        target = cache / "payload" / PurePosixPath(row["path"]).relative_to("compile-work/generated")
        shutil.copyfile(case["compilation"] / row["path"], target)
    (cache / "model-files.tsv").write_text(value["manifest"])


def test_actual_cache_shell_creates_plain_directories_and_checks_all_declared_sha(shell_case):
    value = shell_case
    value["cache"].mkdir()
    created = shell(value, "model-directories")
    assert created.returncode == 0, created.stderr
    assert (value["cache"] / "payload/model").is_dir()
    populate_cache(value)
    observed = shell(value, "inspect-model-cache")
    assert observed.returncode == 0, observed.stderr
    assert len(observed.stdout.splitlines()) == 2
    assert all(line.startswith("OK ") for line in observed.stdout.splitlines())


@pytest.mark.parametrize("level", ["payload", "payload/model"])
def test_actual_directory_shell_rejects_payload_parent_symlinks_without_touching_targets(shell_case, tmp_path, level):
    value = shell_case
    value["cache"].mkdir()
    target = tmp_path / "unrelated-target-directory"
    target.mkdir()
    marker = target / "sentinel"
    marker.write_bytes(b"must remain unchanged")
    link = value["cache"] / level
    link.parent.mkdir(exist_ok=True)
    link.symlink_to(target, target_is_directory=True)
    before = (marker.read_bytes(), marker.stat().st_mtime_ns, sorted(path.name for path in target.iterdir()))
    observed = shell(value, "model-directories")
    assert observed.returncode != 0
    assert link.is_symlink()
    assert (marker.read_bytes(), marker.stat().st_mtime_ns, sorted(path.name for path in target.iterdir())) == before


def test_actual_directory_shell_rejects_manifest_symlink_without_modifying_target(shell_case, tmp_path):
    value = shell_case
    value["cache"].mkdir()
    target = tmp_path / "unrelated-manifest-target"
    target.write_bytes(b"must remain unchanged\n")
    link = value["cache"] / "model-files.tsv"
    link.symlink_to(target)
    before = (target.read_bytes(), target.stat().st_mtime_ns)
    observed = shell(value, "model-directories")
    assert observed.returncode != 0
    assert link.is_symlink() and (target.read_bytes(), target.stat().st_mtime_ns) == before


def test_actual_cache_inspection_rejects_even_hash_matching_file_symlink(shell_case, tmp_path):
    value = shell_case
    value["cache"].mkdir()
    assert shell(value, "model-directories").returncode == 0
    populate_cache(value)
    target = tmp_path / "unrelated-weight-target"
    target.write_bytes(value["case"]["weight"].read_bytes())
    link = value["cache"] / "payload/model/SubGraph_0.weight"
    link.unlink()
    link.symlink_to(target)
    before = (target.read_bytes(), target.stat().st_mtime_ns)
    observed = shell(value, "inspect-model-cache")
    assert observed.returncode != 0
    assert "OK " + str(link) not in observed.stdout and "SEND " + str(link) not in observed.stdout
    assert link.is_symlink() and (target.read_bytes(), target.stat().st_mtime_ns) == before


def test_actual_owner_shell_rejects_symlink_marker_without_touching_target(shell_case, tmp_path):
    value = shell_case
    value["cache"].mkdir()
    target = tmp_path / "unrelated-owner-target"
    target.write_text(sha256_file(value["case"]["compilation"] / "report.json") + "\n")
    link = value["cache"] / ".compilation-sha256"
    link.symlink_to(target)
    before = (target.read_bytes(), target.stat().st_mtime_ns)
    observed = shell(value, "own-model-cache")
    assert observed.returncode != 0
    assert link.is_symlink() and (target.read_bytes(), target.stat().st_mtime_ns) == before


def test_actual_owner_shell_can_reopen_its_own_new_regular_marker(shell_case):
    value = shell_case
    first = shell(value, "own-model-cache")
    assert first.returncode == 0, first.stderr
    marker = value["cache"] / ".compilation-sha256"
    expected = sha256_file(value["case"]["compilation"] / "report.json") + "\n"
    assert marker.read_text() == expected
    second = shell(value, "own-model-cache")
    assert second.returncode == 0, second.stderr
    assert marker.read_text() == expected
