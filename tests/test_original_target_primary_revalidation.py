"""Board cache payload selection revalidates only a small primary OMC."""
from copy import deepcopy
import hashlib
import os
from pathlib import PurePosixPath
import stat
import time

import pytest

from qwen35_moe_dflash import original_target_board as board
from qwen35_moe_dflash import target_decode_smoke_plan as planner
from qwen35_moe_dflash.existing_air import _stat
from qwen35_moe_dflash.target_prefill_smoke_plan import _identity


@pytest.fixture
def case(tmp_path):
    root = tmp_path / "compilation"
    path = root / "compile-work/generated/model/model.omc"
    path.parent.mkdir(parents=True)
    raw = b"small primary OMC fixture"
    path.write_bytes(raw)
    row = {"path": str(path.relative_to(root)), "bytes": len(raw),
           "sha256": hashlib.sha256(raw).hexdigest(), "stat": _stat(path)}
    proof = {"primary_model_binding": {"path": str(path), "bytes": len(raw),
        "sha256": row["sha256"], "identity": _identity(path.stat())}}
    return {"root": root, "path": path, "row": row, "compiled": {"omc": row}, "proof": proof, "raw": raw}


def ctime_only(path):
    before = _stat(path)
    mode = stat.S_IMODE(path.stat().st_mode)
    # The test filesystem may coalesce timestamps within one scheduler tick.
    time.sleep(0.02)
    os.chmod(path, mode ^ stat.S_IXUSR)
    os.chmod(path, mode)
    assert _stat(path)[:4] == before[:4]
    assert _stat(path)[4] != before[4]


def bind(case, observations=None):
    return board._recorded_board_payload(case["root"], case["row"], case["compiled"],
        smoke_proof=case["proof"], primary_revalidations=observations)


def test_ctime_only_primary_cache_payload_is_freshly_verified_and_saved_proof_preserved(case, monkeypatch):
    saved_row, saved_proof = deepcopy(case["row"]), deepcopy(case["proof"])
    ctime_only(case["path"])
    observations = []
    monkeypatch.setattr(board, "_recorded_file", lambda *a: pytest.fail("ctime-only primary must use the bounded shared binder"))
    monkeypatch.setattr(board, "Hdc", lambda *a: pytest.fail("payload validation must not contact a device"))
    assert bind(case, observations) == case["path"]
    assert case["row"] == saved_row and case["proof"] == saved_proof
    assert len(observations) == 1
    assert observations[0]["path"] == str(case["path"])
    assert case["path"].read_bytes() == case["raw"]


def test_unchanged_primary_retains_original_fresh_hash_validator(case, monkeypatch):
    called, original = [], board._recorded_file
    def checked(root, row, key):
        called.append((root, row, key))
        return original(root, row, key)
    monkeypatch.setattr(board, "_recorded_file", checked)
    observations = []
    assert bind(case, observations) == case["path"]
    assert called == [(case["root"], case["row"], "stat")] and observations == []


@pytest.mark.parametrize("change", ["mode", "uid", "gid", "nlink"])
def test_saved_smoke_full_identity_changes_are_not_relabelled_as_ctime(case, change):
    ctime_only(case["path"])
    case["proof"]["primary_model_binding"]["identity"][change] += 1
    observations = []
    with pytest.raises(ValueError):
        bind(case, observations)
    assert observations == []


@pytest.mark.parametrize("change", ["mtime", "inode", "content"])
def test_primary_bytes_or_stable_file_identity_drift_remains_blocked(case, change):
    original = case["path"].stat()
    if change == "mtime":
        os.utime(case["path"], ns=(original.st_atime_ns, original.st_mtime_ns + 1))
    elif change == "inode":
        other = case["path"].with_name("replacement.omc")
        other.write_bytes(case["raw"])
        os.utime(other, ns=(original.st_atime_ns, original.st_mtime_ns))
        other.replace(case["path"])
    else:
        case["path"].write_bytes(bytes([88]) * len(case["raw"]))
        os.utime(case["path"], ns=(original.st_atime_ns, original.st_mtime_ns))
        ctime_only(case["path"])
    observations = []
    with pytest.raises(ValueError):
        bind(case, observations)
    if change == "content":
        assert len(observations) == 1 and observations[0]["path"] == str(case["path"])
        assert observations[0]["status"] == "FAILED"
    else:
        assert observations == []


@pytest.mark.parametrize("kind", ["weight", "other_small_primary", "large_primary"])
def test_nonprimary_and_large_payloads_keep_strict_recorded_stat_gate(case, monkeypatch, kind):
    path = case["path"].with_name("SubGraph_0.weight" if kind == "weight" else "unapproved-primary.omc")
    path.write_bytes(b"other payload")
    if kind == "large_primary":
        path = case["path"]
        with path.open("wb") as stream:
            stream.truncate((64 << 20) + 1)
    row = {"path": str(path.relative_to(case["root"])), "bytes": path.stat().st_size,
           "sha256": "1" * 64, "stat": _stat(path)}
    compiled = case["compiled"] if kind == "weight" else {"omc": row}
    ctime_only(path)
    monkeypatch.setattr(planner, "bind_primary_omc", lambda *a, **k: pytest.fail("only a declared small primary may use reconciliation"))
    monkeypatch.setattr("qwen35_moe_dflash.full_target.sha256_file", lambda *a: pytest.fail("changed generic payload must fail before any payload read"))
    observations = []
    with pytest.raises(ValueError, match="changed since compilation"):
        board._recorded_board_payload(case["root"], row, compiled, primary_revalidations=observations)
    assert observations == []


@pytest.mark.parametrize("field", ["path", "bytes", "sha256", "identity"])
def test_primary_smoke_binding_must_match_the_exact_selected_payload(case, field):
    ctime_only(case["path"])
    case["proof"]["primary_model_binding"][field] = None
    with pytest.raises(ValueError, match="saved smoke model binding"):
        bind(case, [])


def test_ctime_only_primary_without_a_smoke_proof_still_requires_recorded_content(case):
    ctime_only(case["path"])
    observations = []
    assert board._recorded_board_payload(case["root"], case["row"], case["compiled"],
        primary_revalidations=observations) == case["path"]
    assert len(observations) == 1


def test_failed_primary_hash_attempt_observation_is_retained_by_board(case, monkeypatch):
    ctime_only(case["path"])
    attempted = {"kind": "PRIMARY_OMC_CTIME_ONLY_REVALIDATED", "path": str(case["path"]), "status": "FAILED"}
    def fail(path, row, **options):
        options["revalidation_observations"].append(attempted)
        raise ValueError("primary changed during bounded hash")
    monkeypatch.setattr(planner, "bind_primary_omc", fail)
    observations = []
    with pytest.raises(ValueError, match="changed during bounded hash"):
        bind(case, observations)
    assert observations == [attempted]


def test_cached_decode_primary_revalidation_is_recorded_without_reupload_or_receipt_edit(tmp_path, monkeypatch):
    from test_target_decode_smoke_execution import DecodeFakeHdc, case as decode_case, execute
    case = decode_case.__wrapped__(tmp_path, monkeypatch)
    report_path = case["compilation"] / "report.json"
    report_bytes = report_path.read_bytes()
    primary = case["compilation"] / case["compiled"]["omc"]["path"]
    ctime_only(primary)
    cache = PurePosixPath("/data/local/tmp/qwen3_5_moe") / ("codex-target-" + hashlib.sha256(report_bytes).hexdigest()[:20])
    for row in case["compiled"]["generated_artifacts"]:
        relative = PurePosixPath(row["path"]).relative_to("compile-work/generated")
        DecodeFakeHdc.remote[str(cache / "payload" / relative)] = (case["compilation"] / row["path"]).read_bytes()
    result = execute(case)
    assert result["status"] == "EXECUTED_UNVALIDATED", result
    assert result["board_disk_budget"]["missing_model_payload_bytes"] == 0
    assert not any(local.endswith((".omc", ".weight")) for local, _ in DecodeFakeHdc.sends)
    assert len(result["primary_revalidations"]) == 1
    event = result["primary_revalidations"][0]
    assert event["path"] == str(primary)
    assert event["current_identity"] == _identity(primary.stat())
    assert event["recorded_stat"] == case["compiled"]["omc"]["stat"]
    assert event["sha256"] == case["compiled"]["omc"]["sha256"]
    assert report_path.read_bytes() == report_bytes
