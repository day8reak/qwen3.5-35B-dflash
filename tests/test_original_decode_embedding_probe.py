"""Sparse original-table fixtures; no checkpoint/vendor/device execution."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import struct

import pytest
import yaml

from qwen35_moe_dflash import original_decode_embedding_probe as probe
from test_target_decode_output_inventory import (case as retained_case, attach_trial, inspect,
    record, refresh, save)


@pytest.fixture
def embedding_case(retained_case, tmp_path):
    case = retained_case
    source = case["sources"]["source"]
    source_token, winner = 7, 101
    rows = {token: bytes((token + offset) % 256 for offset in range(probe._HIDDEN))
            for token in (source_token, winner)}
    scales = {token: struct.pack("<f", .001 * (token + 1)) for token in rows}
    (source / "input_ids.bin").write_bytes(struct.pack("<i", source_token))
    (source / "inputs_embeds.bin").write_bytes(rows[source_token])
    (source / "embed_scale.bin").write_bytes(scales[source_token])
    trial = attach_trial(case, tmp_path)
    logits = case["board"] / case["paths"][case["proof"]["logits_runtime_output_index"]]
    with logits.open("r+b") as stream:
        stream.seek(winner * 4)
        stream.write(struct.pack("<f", 3))
    relative = str(logits.relative_to(case["board"]))
    case["report"]["retrieved"][relative] = record(logits)
    refresh(case)
    resources = tmp_path / "original-resources"
    resources.mkdir()
    tables = (resources / "int8.bin", resources / "scale.bin")
    for path, width in zip(tables, (probe._HIDDEN, 4)):
        with path.open("wb") as stream:
            stream.truncate(probe._VOCAB * width)
        with path.open("r+b") as stream:
            for token in rows:
                stream.seek(token * width)
                stream.write(rows[token] if width == probe._HIDDEN else scales[token])
    config = resources / "original.yaml"
    config.write_text(yaml.safe_dump({"embedding_weight_path": tables[0].name,
                                     "embedding_scale_path": tables[1].name}))
    report = inspect(case)
    assert report["status"] == "DECODE_OUTPUT_FILES_BOUND_TO_LOGGED_SLOTS", report
    return {"case": case, "source": source, "trial": trial, "tables": tables, "rows": rows,
            "scales": scales, "source_token": source_token, "winner": winner, "config": config,
            "inspection": case["out"], "report": report, "out": tmp_path / "decode-row-inspection"}


def run(case, **options):
    return probe.inspect_original_decode_embeddings(case["inspection"], case["config"], case["out"], **options)


def stamp(path):
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns, value.st_nlink


def no_next(case, result):
    assert result["next_token_inputs"] == [], result
    assert not (case["out"] / "next-token").exists()


def test_exact_executed_decode_row_and_next_token_use_only_two_original_rows(embedding_case, monkeypatch):
    case = embedding_case
    watched = [*case["tables"], case["config"], case["trial"] / "manifest.json", *case["source"].iterdir()]
    before = {path: stamp(path) for path in watched}
    original_pread, original_open, original_fdopen = os.pread, os.open, os.fdopen
    table_fds, reads = {}, []
    def guarded_open(path, *args, **kwargs):
        assert not str(path).endswith((".omc", ".weight")), "no model/weight payload may be opened"
        fd = original_open(path, *args, **kwargs)
        if Path(path) in case["tables"]:
            table_fds[fd] = Path(path)
        return fd
    def bounded(fd, size, offset):
        assert fd in table_fds and size in {4, 2048}
        reads.append((table_fds[fd], size, offset))
        return original_pread(fd, size, offset)
    def no_table_stream(fd, *args, **kwargs):
        assert fd not in table_fds, "original tables must never be streamed or fully hashed"
        return original_fdopen(fd, *args, **kwargs)
    monkeypatch.setattr(os, "open", guarded_open)
    monkeypatch.setattr(os, "pread", bounded)
    monkeypatch.setattr(os, "fdopen", no_table_stream)
    result = run(case)
    assert result["status"] == probe._READY, result
    assert result["selected_source_token_id"] == case["source_token"]
    assert result["selected_next_token_id"] == case["winner"]
    assert result["source_row_comparison"] == {"positions": 1, "weight_bytes": 2048,
        "scale_bytes": 4, "status": "BYTE_IDENTICAL", "mismatches": []}
    assert len(reads) == 4 and sum(size for _, size, _ in reads) == 4104
    assert {(path, offset) for path, _, offset in reads} == {
        (path, token * width) for path, width in zip(case["tables"], (2048, 4))
        for token in (case["source_token"], case["winner"])}
    assert result["observed_payload_reads"] == {"bound_logits_bytes": 993280,
        "source_input_bytes": 4108, "table_bytes": 4104}
    assert result["original_input_ids"]["prior_trial_binding"] == "EXTRA_FILE_NOT_PREVIOUSLY_READ"
    for filename, raw in (("inputs_embeds.bin", case["rows"][case["winner"]]),
            ("embed_scale.bin", case["scales"][case["winner"]]),
            ("input_ids.bin", struct.pack("<i", case["winner"]))):
        assert (case["out"] / "next-token" / filename).read_bytes() == raw
    assert sum(row["bytes"] for row in result["next_token_inputs"]) == 2056
    assert all(stamp(path) == value for path, value in before.items())
    assert result["full_table_hash"] == "NOT_PERFORMED"
    assert result["logical_history"] == result["runtime_control_recipe"] == result["state_feedback"] == "UNVERIFIED"
    assert result["state_continuity"] == "NOT_VALIDATED" and "generation_ready" not in result
    assert result["model_compilation"] == result["device_execution"] == "NOT_RUN"


def test_same_source_and_winner_row_is_read_once(embedding_case, monkeypatch):
    case = embedding_case
    logits = case["case"]["board"] / case["case"]["paths"][case["case"]["proof"]["logits_runtime_output_index"]]
    with logits.open("r+b") as stream:
        stream.seek(case["winner"] * 4); stream.write(struct.pack("<f", 0))
        stream.seek(case["source_token"] * 4); stream.write(struct.pack("<f", 3))
    case["case"]["report"]["retrieved"][str(logits.relative_to(case["case"]["board"]))] = record(logits)
    refresh(case["case"])
    case["case"]["out"] = case["inspection"].with_name("same-token-inventory")
    case["inspection"] = case["case"]["out"]
    assert inspect(case["case"])["status"] == "DECODE_OUTPUT_FILES_BOUND_TO_LOGGED_SLOTS"
    original, calls = os.pread, []
    def count(fd, size, offset):
        calls.append((size, offset))
        return original(fd, size, offset)
    monkeypatch.setattr(os, "pread", count)
    result = run(case)
    assert result["status"] == probe._READY, result
    assert len(calls) == 2 and sum(size for size, _ in calls) == 2052
    assert len(result["selected_rows"]) == 1


@pytest.mark.parametrize("size", [0, 2, 8, 1024])
def test_unsupported_source_id_extent_is_stat_only_without_selecting_a_vector_element(embedding_case, monkeypatch, size):
    case = embedding_case
    ids = case["source"] / "input_ids.bin"
    ids.write_bytes(bytes(size))
    original_open = os.open
    def no_unbound_payload(path, *args, **kwargs):
        assert Path(path) not in {ids, *case["tables"]}, "unsupported IDs and original table rows must stay unopened"
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(os, "open", no_unbound_payload)
    result = run(case)
    assert result["status"] == "FAILED" and result["stage"] == "inspect-original-source-id-extent", result
    diagnosis = result["source_id_diagnosis"]
    assert diagnosis["path"] == str(ids) and diagnosis["bytes"] == size
    assert diagnosis["expected_bytes"] == diagnosis["maximum_read_bytes"] == 4
    assert diagnosis["candidate_int32_elements"] == (size // 4 if size % 4 == 0 else None)
    assert diagnosis["status"] == "UNSUPPORTED_SINGLE_DECODE_ID_EXTENT"
    assert diagnosis["payload_access"] == "NOT_READ" and diagnosis["token_selection"] == "NONE"
    assert str(ids) in result["error"] and f"observed_bytes={size}" in result["error"]
    assert "maximum_read_bytes=4" in result["error"]
    assert "selected_source_token_id" not in result and "selected_next_token_id" not in result
    no_next(case, result)


def test_extent_diagnosis_is_printed_by_cli_and_no_last_token_is_chosen(embedding_case, capsys):
    case = embedding_case
    (case["source"] / "input_ids.bin").write_bytes(struct.pack("<256i", *range(256)))
    script = Path(__file__).resolve().parents[1] / "targets/cpro/scripts/inspect_original_decode_embeddings.py"
    main = runpy.run_path(str(script))["main"]
    assert main(["--inspection", str(case["inspection"]), "--config-yaml", str(case["config"]), "--output", str(case["out"])]) == 1
    shown = json.loads(capsys.readouterr().out)
    assert shown["stage"] == "inspect-original-source-id-extent"
    assert shown["source_id_diagnosis"]["bytes"] == 1024
    assert shown["source_id_diagnosis"]["payload_access"] == "NOT_READ"
    assert "selected_source_token_id" not in shown and "selected_next_token_id" not in shown
    no_next(case, shown)


def set_candidates(case, ids):
    raw = struct.pack("<" + "i" * len(ids), *ids)
    (case["source"] / "input_ids.bin").write_bytes(raw)
    return raw


def test_explicit_256_id_set_matches_only_executed_bytes_at_interior_position(embedding_case, monkeypatch):
    case = embedding_case
    ids = list(range(256))
    ids[7], ids[143] = ids[143], ids[7]
    raw = set_candidates(case, ids)
    original, calls = os.pread, []
    def bounded(fd, size, offset):
        assert size in {2048, 4}
        calls.append((size, offset))
        return original(fd, size, offset)
    monkeypatch.setattr(os, "pread", bounded)
    result = run(case, match_source_id_candidates=True)
    assert result["status"] == probe._READY, result
    matching = result["source_candidate_matching"]
    assert matching["mode"] == "EXPLICIT_BOUNDED_CANDIDATES"
    assert matching["candidate_count"] == matching["unique_candidate_count"] == 256
    assert matching["matching_token_ids"] == [7]
    assert matching["matching_positions"] == [{"token_id": 7, "positions": [143]}]
    assert matching["uniqueness_scope"] == "EXPLICIT_SOURCE_ID_CANDIDATES_ONLY" and matching["position_index_base"] == 0
    assert result["selected_source_token_id"] == 7 and ids[0] != 7 and ids[-1] != 7
    assert result["original_input_ids"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["original_input_ids"]["shape"] == [1, 256]
    assert sum(size for size, _ in calls) == 256 * 2052
    assert result["read_budgets"]["table_payload_maximum_bytes"] == 257 * 2052
    assert result["logical_history"] == matching["logical_position_semantics"] == "UNVERIFIED"
    assert result["source_row_comparison"]["position_semantics"] == "CANDIDATE_OCCURRENCES_ONLY_NOT_LOGICAL_HISTORY"
    assert result["source_row_comparison"]["positions"] == 1
    assert result["source_row_comparison"]["matching_source_positions"] == [143]


def test_repeated_matching_positions_of_one_token_are_not_ambiguous(embedding_case):
    case = embedding_case
    set_candidates(case, [3, 7, 4, 7, 8])
    result = run(case, match_source_id_candidates=True)
    assert result["status"] == probe._READY, result
    matching = result["source_candidate_matching"]
    assert matching["candidate_count"] == 5 and matching["unique_candidate_count"] == 4
    assert matching["matching_token_ids"] == [7]
    assert matching["matching_positions"] == [{"token_id": 7, "positions": [1, 3]}]
    assert result["observed_payload_reads"]["table_bytes"] == 5 * 2052


@pytest.mark.parametrize("same_scale", [False, True])
def test_matching_requires_both_row_and_scale_and_rejects_distinct_identical_ids(embedding_case, same_scale):
    case = embedding_case
    set_candidates(case, [11, 7, 11])
    with case["tables"][0].open("r+b") as stream:
        stream.seek(11 * 2048); stream.write(case["rows"][7])
    with case["tables"][1].open("r+b") as stream:
        stream.seek(11 * 4); stream.write(case["scales"][7] if same_scale else struct.pack("<f", .9))
    result = run(case, match_source_id_candidates=True)
    matching = result["source_candidate_matching"]
    if same_scale:
        assert result["status"] == "ORIGINAL_DECODE_EMBEDDING_CANDIDATES_AMBIGUOUS", result
        assert matching["matching_token_ids"] == [7, 11]
        assert matching["status"] == "AMBIGUOUS_TOKEN_IDS"
        assert "selected_source_token_id" not in result
        assert result["source_id_diagnosis"]["token_selection"] == "NONE"
        no_next(case, result)
    else:
        assert result["status"] == probe._READY, result
        assert matching["matching_token_ids"] == [7]


def test_no_candidate_match_creates_no_next_row_or_guess(embedding_case):
    case = embedding_case
    set_candidates(case, [3, 4])
    result = run(case, match_source_id_candidates=True)
    assert result["status"] == "ORIGINAL_DECODE_EMBEDDING_CANDIDATES_NO_MATCH", result
    assert result["source_candidate_matching"]["matching_token_ids"] == []
    assert result["observed_payload_reads"]["table_bytes"] == 2 * 2052
    assert "selected_source_token_id" not in result
    no_next(case, result)


def test_full_1024_unique_candidates_and_extra_winner_obey_exact_read_cap(embedding_case, monkeypatch):
    case = embedding_case
    ids = list(range(1000, 2023))
    ids.insert(501, 7)
    set_candidates(case, ids)
    original, calls = os.pread, []
    def bounded(fd, size, offset):
        calls.append(size)
        assert sum(calls) <= 2103300
        return original(fd, size, offset)
    monkeypatch.setattr(os, "pread", bounded)
    result = run(case, match_source_id_candidates=True)
    assert result["status"] == probe._READY, result
    assert result["source_candidate_matching"]["unique_candidate_count"] == 1024
    assert result["source_candidate_matching"]["matching_positions"] == [{"token_id": 7, "positions": [501]}]
    assert result["read_budgets"]["table_payload_maximum_bytes"] == 2103300
    assert result["read_budgets"]["configured_table_payload_cap_bytes"] == 2103300
    assert sum(calls) == result["observed_payload_reads"]["table_bytes"] == 2103300


@pytest.mark.parametrize("size", [0, 6, 4100])
def test_candidate_flag_never_reads_oversize_or_non_int32_source_extent(embedding_case, monkeypatch, size):
    case = embedding_case
    ids = case["source"] / "input_ids.bin"
    ids.write_bytes(bytes(size))
    original = os.open
    def no_payload(path, *args, **kwargs):
        assert Path(path) not in {ids, *case["tables"]}
        return original(path, *args, **kwargs)
    monkeypatch.setattr(os, "open", no_payload)
    result = run(case, match_source_id_candidates=True)
    assert result["status"] == "FAILED" and result["stage"] == "inspect-original-source-id-extent", result
    assert result["source_id_diagnosis"]["maximum_read_bytes"] == 4096
    assert result["source_id_diagnosis"]["payload_access"] == "NOT_READ"
    assert "maximum_read_bytes=4096" in result["error"]
    no_next(case, result)


@pytest.mark.parametrize("token", [-1, probe._VOCAB])
def test_every_candidate_id_is_range_checked_before_any_table_access(embedding_case, monkeypatch, token):
    case = embedding_case
    set_candidates(case, [7, token])
    original = os.open
    def no_tables(path, *args, **kwargs):
        assert Path(path) not in case["tables"]
        return original(path, *args, **kwargs)
    monkeypatch.setattr(os, "open", no_tables)
    result = run(case, match_source_id_candidates=True)
    assert result["status"] == "FAILED" and "outside" in result["error"], result
    no_next(case, result)


@pytest.mark.parametrize("factor", [float("nan"), -.5])
def test_invalid_candidate_scale_is_rejected_without_next_inputs(embedding_case, factor):
    case = embedding_case
    set_candidates(case, [11, 7])
    with case["tables"][1].open("r+b") as stream:
        stream.seek(11 * 4); stream.write(struct.pack("<f", factor))
    result = run(case, match_source_id_candidates=True)
    assert result["status"] == "FAILED" and "finite and nonnegative" in result["error"], result
    no_next(case, result)


def test_candidate_source_hash_race_is_rejected_before_next_inputs(embedding_case, monkeypatch):
    case = embedding_case
    set_candidates(case, [3, 7, 4])
    original, changed = os.pread, False
    def race(fd, size, offset):
        nonlocal changed
        raw = original(fd, size, offset)
        if not changed:
            changed = True
            set_candidates(case, [3, 7, 5])
        return raw
    monkeypatch.setattr(os, "pread", race)
    result = run(case, match_source_id_candidates=True)
    assert result["status"] == "FAILED" and "changed" in result["error"], result
    no_next(case, result)


def test_cli_candidate_flag_is_explicit_and_prints_unique_match(embedding_case, capsys):
    case = embedding_case
    set_candidates(case, [3, 7, 4])
    script = Path(__file__).resolve().parents[1] / "targets/cpro/scripts/inspect_original_decode_embeddings.py"
    main = runpy.run_path(str(script))["main"]
    assert main(["--inspection", str(case["inspection"]), "--config-yaml", str(case["config"]),
        "--output", str(case["out"]), "--match-source-id-candidates"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["source_candidate_matching"]["matching_positions"] == [{"token_id": 7, "positions": [1]}]
    assert shown["selected_source_token_id"] == 7 and shown["status"] == probe._READY


@pytest.mark.parametrize("index", [0, 1])
def test_missing_configured_table_has_exact_paths_and_no_fallback(embedding_case, index):
    case = embedding_case
    case["tables"][index].unlink()
    result = run(case)
    assert result["status"] == "ORIGINAL_EMBEDDING_TABLES_UNAVAILABLE", result
    assert result["missing_paths"] == [str(case["tables"][index])]
    assert result["observed_payload_reads"]["table_bytes"] == 0
    no_next(case, result)


@pytest.mark.parametrize("field", ["input_ids", "weight", "scale"])
def test_source_id_or_original_row_mismatch_never_prepares_greedy_row(embedding_case, field):
    case = embedding_case
    if field == "input_ids":
        (case["source"] / "input_ids.bin").write_bytes(struct.pack("<i", case["winner"]))
    else:
        index, width = (0, 2048) if field == "weight" else (1, 4)
        with case["tables"][index].open("r+b") as stream:
            stream.seek(case["source_token"] * width)
            stream.write(b"different" if field == "weight" else struct.pack("<f", .9))
    result = run(case)
    assert result["status"] == "ORIGINAL_DECODE_EMBEDDING_ROWS_DIFFER", result
    assert result["source_row_comparison"]["mismatches"]
    no_next(case, result)


@pytest.mark.parametrize("fault", ["ids_missing", "ids_extent", "ids_range", "ids_link", "manifest", "source_payload",
    "trial_payload", "logits", "plan", "winner", "mapping", "wrong_table_extent", "table_symlink", "nan_scale"])
def test_provenance_and_hash_faults_fail_without_preparing_next_inputs(embedding_case, fault):
    case = embedding_case
    if fault == "ids_missing": (case["source"] / "input_ids.bin").unlink()
    elif fault == "ids_extent": (case["source"] / "input_ids.bin").write_bytes(b"xx")
    elif fault == "ids_range": (case["source"] / "input_ids.bin").write_bytes(struct.pack("<i", probe._VOCAB))
    elif fault == "ids_link":
        path = case["source"] / "input_ids.bin"
        other = case["source"] / "other_ids.bin"
        path.rename(other); path.symlink_to(other)
    elif fault == "manifest":
        with (case["trial"] / "manifest.json").open("a") as stream: stream.write("\n")
    elif fault in {"source_payload", "trial_payload"}:
        root = case["source"] if fault == "source_payload" else case["trial"] / "inputs"
        with (root / "inputs_embeds.bin").open("r+b") as stream: stream.write(b"x")
    elif fault == "logits":
        with Path(case["report"]["execution"]).joinpath("board", case["report"]["bound_tensors"]["logits"]["path"]).open("r+b") as stream:
            stream.write(struct.pack("<f", 99))
    elif fault == "plan":
        path = case["case"]["execution"] / "plan.json"
        plan = json.loads(path.read_text()); plan["input_dir"] = str(case["source"])
        save(path, plan)
    elif fault in {"winner", "mapping"}:
        report = case["report"]
        if fault == "winner": report["bound_tensors"]["logits"]["statistics"]["argmax"]["token_id"] = 3
        else: report["logged_slot_mapping"]["entries"][0]["path"] = "outputs/not_logged"
        save(case["inspection"] / "report.json", report)
    elif fault == "wrong_table_extent":
        with case["tables"][0].open("r+b") as stream: stream.truncate(2048)
    elif fault == "table_symlink":
        path = case["tables"][0]
        other = path.with_name("other.bin"); path.rename(other); path.symlink_to(other)
    else:
        with case["tables"][1].open("r+b") as stream:
            stream.seek(case["winner"] * 4); stream.write(struct.pack("<f", float("nan")))
    result = run(case)
    assert result["status"] == "FAILED" and result["error"], result
    no_next(case, result)


@pytest.mark.parametrize("changed", ["table", "source_ids", "metadata"])
def test_concurrent_source_or_table_changes_are_detected(embedding_case, monkeypatch, changed):
    case = embedding_case
    original, called = os.pread, False
    def race(fd, size, offset):
        nonlocal called
        raw = original(fd, size, offset)
        if not called:
            called = True
            if changed == "table":
                with case["tables"][0].open("r+b") as stream: stream.write(b"x")
            elif changed == "source_ids":
                (case["source"] / "input_ids.bin").write_bytes(struct.pack("<i", case["winner"]))
            else:
                with case["config"].open("a") as stream: stream.write("\n")
        return raw
    monkeypatch.setattr(os, "pread", race)
    result = run(case)
    assert result["status"] == "FAILED" and "changed" in result["error"], result
    no_next(case, result)


def test_plan_changed_after_inventory_revalidation_cannot_replace_execution_provenance(embedding_case, monkeypatch):
    case = embedding_case
    original = probe.validate_target_decode_output_inventory
    def replace_plan(path):
        result = original(path)
        plan_path = case["case"]["execution"] / "plan.json"
        with plan_path.open("a") as stream: stream.write("\n")
        return result
    monkeypatch.setattr(probe, "validate_target_decode_output_inventory", replace_plan)
    result = run(case)
    assert result["status"] == "FAILED" and "changed after" in result["error"], result
    no_next(case, result)


def test_truncated_selected_row_read_never_prepares_partial_next_inputs(embedding_case, monkeypatch):
    case = embedding_case
    original = os.pread
    def truncated(fd, size, offset):
        return original(fd, size, offset)[:-1]
    monkeypatch.setattr(os, "pread", truncated)
    result = run(case)
    assert result["status"] == "FAILED" and "truncated" in result["error"], result
    no_next(case, result)


def test_cli_outputs_compact_receipt_and_existing_outputs_are_preserved(embedding_case, capsys):
    case = embedding_case
    script = Path(__file__).resolve().parents[1] / "targets/cpro/scripts/inspect_original_decode_embeddings.py"
    main = runpy.run_path(str(script))["main"]
    args = ["--inspection", str(case["inspection"]), "--config-yaml", str(case["config"]), "--output", str(case["out"])]
    assert main(args) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["status"] == probe._READY and shown["selected_next_token_id"] == case["winner"]
    assert "source_bindings" not in shown and "experimental_input_assumption" not in shown
    saved = (case["out"] / "report.json").read_bytes()
    assert main(args) == 1
    assert (case["out"] / "report.json").read_bytes() == saved
    assert json.loads(capsys.readouterr().out)["status"] == "FAILED"
