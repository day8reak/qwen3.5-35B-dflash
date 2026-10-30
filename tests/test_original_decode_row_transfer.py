"""Portable row handoff uses sparse host tables and retained decode fixtures."""
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

import pytest
import yaml

from qwen35_moe_dflash import original_decode_row_transfer as transfer
from test_original_decode_embedding_probe import embedding_case, set_candidates, stamp
from test_target_decode_output_inventory import case as retained_case
from test_target_decode_output_inventory import inspect as inspect_outputs, record, refresh


@pytest.fixture
def handoff(embedding_case, tmp_path):
    case = embedding_case
    set_candidates(case, [3, 7, 4, 7, 8])
    resource = tmp_path / "original-resource-host"
    tables = (resource / "original-int8.bin", resource / "original-scale.bin")
    case["config"].write_text(yaml.safe_dump({"embedding_weight_path": str(tables[0]),
                                             "embedding_scale_path": str(tables[1])}))
    prepared = tmp_path / "row-transfer-prepared"
    result = transfer.prepare_rows(case["inspection"], case["config"], prepared)
    assert result["status"] == "ORIGINAL_DECODE_ROW_TRANSFER_PREPARED", result
    # The configured tables were absent during local preparation. Materialize
    # those same declared paths only for the separate resource-host stage.
    resource.mkdir()
    for original, target in zip(case["tables"], tables):
        original.rename(target)
    return {"case": case, "prepared": prepared, "manifest": result, "tables": tables,
            "request": prepared / "request.json", "resource": resource,
            "extracted": tmp_path / "resource-extraction", "imported": tmp_path / "row-transfer-import"}


def extract(case, request=None):
    return transfer.extract_rows(request or case["request"], *case["tables"], case["extracted"])


def consume(case, response=None, **options):
    return transfer.import_rows(case["prepared"], response or case["extracted"] / "response.json", case["imported"], **options)


def no_next(case, result):
    assert result["status"] == "FAILED" and result["error"], result
    assert result["next_token_inputs"] == [] and not (case["imported"] / "next-token").exists()


def save(path, value):
    path.write_bytes(transfer._encoded(value))


def test_complete_handoff_preserves_sources_reads_only_requested_rows_and_imports_exact_original_bytes(handoff, monkeypatch):
    case = handoff
    request = json.loads(case["request"].read_text())
    assert case["request"].stat().st_size < 64 << 10
    assert request["source_token_ids"]["bytes"] == 20 and request["executed_embedding"]["bytes"] == 2048
    assert request["executed_scale"]["bytes"] == 4
    script = case["prepared"] / "extract_rows.py"
    assert hashlib.sha256(script.read_bytes()).hexdigest() == case["manifest"]["extraction_script"]["sha256"]
    watched = [*case["tables"], case["request"], case["case"]["config"], *case["case"]["source"].iterdir()]
    before = {path: stamp(path) for path in watched}
    original_open, original_fdopen, original_pread = os.open, os.fdopen, os.pread
    table_fds, calls = {}, []
    with monkeypatch.context() as patch:
        def opened(path, *args, **kwargs):
            assert not str(path).endswith((".omc", ".weight"))
            fd = original_open(path, *args, **kwargs)
            if Path(path) in case["tables"]: table_fds[fd] = Path(path)
            return fd
        def streamed(fd, *args, **kwargs):
            assert fd not in table_fds, "original tables may only use bounded pread"
            return original_fdopen(fd, *args, **kwargs)
        def bounded(fd, size, offset):
            assert fd in table_fds and size in {2048, 4}
            calls.append((table_fds[fd], size, offset))
            return original_pread(fd, size, offset)
        patch.setattr(os, "open", opened); patch.setattr(os, "fdopen", streamed); patch.setattr(os, "pread", bounded)
        response = extract(case)
    assert response["status"] == transfer._EXTRACTED, response
    assert response["matching_token_ids"] == [7]
    assert response["matching_positions"] == [{"token_id": 7, "positions": [1, 3]}]
    assert sum(size for _, size, _ in calls) == 5 * 2052 == response["table_read_bytes"]
    assert response["extractor_sha256"] == case["manifest"]["extraction_script"]["sha256"]
    with monkeypatch.context() as patch:
        def no_tables(path, *args, **kwargs):
            assert Path(path) not in case["tables"], "local import must not open remote tables"
            assert not str(path).endswith((".omc", ".weight"))
            return original_open(path, *args, **kwargs)
        patch.setattr(os, "open", no_tables)
        result = consume(case)
    assert result["status"] == "ORIGINAL_DECODE_TRANSFERRED_ROWS_BOUND_NEXT_TOKEN_PREPARED", result
    assert result["selected_source_token_id"] == 7 and result["matching_source_positions"] == [1, 3]
    assert result["selected_next_token_id"] == 101
    assert sum(row["bytes"] for row in result["next_token_inputs"]) == 2056
    assert (case["imported"] / "next-token/inputs_embeds.bin").read_bytes() == case["case"]["rows"][101]
    assert (case["imported"] / "next-token/embed_scale.bin").read_bytes() == case["case"]["scales"][101]
    assert (case["imported"] / "next-token/input_ids.bin").read_bytes() == struct.pack("<i", 101)
    assert result["logical_history"] == result["runtime_control_recipe"] == result["state_feedback"] == "UNVERIFIED"
    assert "NO_FULL_TABLE_AUTHENTICATION" in result["table_authenticity"]
    assert all(stamp(path) == value for path, value in before.items())


def test_copied_extractor_runs_isolated_without_repo_site_packages_or_ai_run_dir(handoff):
    case = handoff
    env = dict(os.environ)
    env.pop("AI_RUN_DIR", None); env.pop("PYTHONPATH", None)
    done = subprocess.run([sys.executable, "-I", "-S", str(case["prepared"] / "extract_rows.py"), "extract",
        "--request", str(case["request"]), "--weight-table", str(case["tables"][0]),
        "--scale-table", str(case["tables"][1]), "--output", str(case["extracted"])],
        cwd=case["resource"], env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    assert json.loads(done.stdout)["status"] == transfer._EXTRACTED
    assert consume(case)["status"] == "ORIGINAL_DECODE_TRANSFERRED_ROWS_BOUND_NEXT_TOKEN_PREPARED"


@pytest.mark.parametrize("fault", ["request_sha", "binding", "source_bytes", "next_hash", "missing_row", "extra_row",
    "row_duplicate", "positions", "ambiguous_hashes", "table_path", "extractor", "next_dtype"])
def test_response_faults_never_promote_status_alone_or_create_next_inputs(handoff, fault):
    case = handoff
    response = extract(case)
    assert response["status"] == transfer._EXTRACTED
    if fault == "request_sha": response["request_sha256"] = "f" * 64
    elif fault == "binding": response["binding_digest"] = "f" * 64
    elif fault == "source_bytes": response["source_match"]["embedding"] = transfer._blob(bytes(2048), "INT8", [1, 1, 2048])
    elif fault == "next_hash": response["next_token"]["embedding"] = transfer._blob(bytes(2048), "INT8", [1, 1, 2048])
    elif fault == "missing_row": response["selected_rows"].pop()
    elif fault == "extra_row": response["selected_rows"].append(dict(response["selected_rows"][0], token_id=248000))
    elif fault == "row_duplicate": response["selected_rows"][1] = response["selected_rows"][0]
    elif fault == "positions": response["matching_positions"][0]["positions"] = [0]
    elif fault == "ambiguous_hashes":
        matched = next(row for row in response["selected_rows"] if row["token_id"] == 7)
        response["selected_rows"][0].update(weight_sha256=matched["weight_sha256"], scale_sha256=matched["scale_sha256"])
    elif fault == "table_path": response["tables"][0]["path"] = "/other/original-int8.bin"
    elif fault == "extractor": response["extractor_sha256"] = "f" * 64
    else: response["next_token"]["embedding"]["dtype"] = "FLOAT16"
    save(case["extracted"] / "response.json", response)
    no_next(case, consume(case))


@pytest.mark.parametrize("fault", ["source_ids", "executed_embedding", "logits", "request", "audit", "script"])
def test_local_sources_and_prepared_payload_identities_are_freshly_rebound_on_import(handoff, fault):
    case = handoff
    assert extract(case)["status"] == transfer._EXTRACTED
    if fault == "source_ids": set_candidates(case["case"], [3, 7, 4, 7, 9])
    elif fault == "executed_embedding":
        with (case["case"]["trial"] / "inputs/inputs_embeds.bin").open("r+b") as stream: stream.write(b"x")
    elif fault == "logits":
        info = case["case"]["report"]["bound_tensors"]["logits"]
        with (case["case"]["case"]["board"] / info["path"]).open("r+b") as stream: stream.write(b"x")
    else:
        path = {"request": case["request"], "audit": case["prepared"] / "source-audit/report.json",
                "script": case["prepared"] / "extract_rows.py"}[fault]
        with path.open("a") as stream: stream.write("\n")
    no_next(case, consume(case))


@pytest.mark.parametrize("ambiguity", [False, True])
def test_no_or_multiple_different_token_matches_return_no_next_bytes(handoff, ambiguity):
    case = handoff
    if ambiguity:
        for path, width, raw in zip(case["tables"], (2048, 4), (case["case"]["rows"][7], case["case"]["scales"][7])):
            with path.open("r+b") as stream: stream.seek(3 * width); stream.write(raw)
    else:
        with case["tables"][0].open("r+b") as stream: stream.seek(7 * 2048); stream.write(b"x")
    response = extract(case)
    assert response["status"] == ("ORIGINAL_DECODE_ROWS_AMBIGUOUS" if ambiguity else "ORIGINAL_DECODE_ROWS_NO_MATCH"), response
    assert response["source_match"] is None and response["next_token"] is None
    no_next(case, consume(case))


@pytest.mark.parametrize("fault", ["duplicate", "nan", "nonfinite_exponent", "base64", "range", "too_many_ids", "oversize"])
def test_portable_requests_are_strict_bounded_and_payload_hash_checked(handoff, fault):
    case = handoff
    request = json.loads(case["request"].read_text())
    path = case["resource"] / "invalid-request.json"
    if fault == "duplicate": path.write_bytes(b'{"schema_version":1,' + case["request"].read_bytes()[1:])
    elif fault == "nan": path.write_text('{"bad":NaN}')
    elif fault == "nonfinite_exponent": path.write_text('{"bad":1e9999}')
    elif fault == "oversize": path.write_bytes(b"x" * (65536 + 1))
    else:
        if fault == "base64": request["executed_embedding"]["base64"] = "!" * len(request["executed_embedding"]["base64"])
        elif fault == "range": request["source_token_ids"] = transfer._blob(struct.pack("<i", transfer._VOCAB), "INT32", [1, 1])
        else: request["source_token_ids"] = transfer._blob(bytes(4100), "INT32", [1, 1025])
        request["binding_digest"] = transfer._binding(request)
        save(path, request)
    response = extract(case, path)
    assert response["status"] == "FAILED" and response["error"], response
    assert response["table_read_bytes"] == 0 and response["next_token"] is None


@pytest.mark.parametrize("changed", ["table", "request"])
def test_resource_mutation_during_selected_row_reads_fails_preservation(handoff, monkeypatch, changed):
    case = handoff
    original, done = os.pread, False
    def race(fd, size, offset):
        nonlocal done
        raw = original(fd, size, offset)
        if not done:
            done = True
            path = case["tables"][0] if changed == "table" else case["request"]
            with path.open("r+b") as stream: stream.write(b"x")
        return raw
    monkeypatch.setattr(os, "pread", race)
    response = extract(case)
    assert response["status"] == "FAILED" and "changed" in response["error"], response
    assert response["next_token"] is None and response["source_match"] is None


def test_import_rechecks_local_sources_after_response_validation(handoff, monkeypatch):
    case = handoff
    assert extract(case)["status"] == transfer._EXTRACTED
    original = transfer._response
    def race(*args):
        result = original(*args)
        set_candidates(case["case"], [3, 7, 4, 7, 9])
        return result
    monkeypatch.setattr(transfer, "_response", race)
    no_next(case, consume(case))


def test_extract_enforces_full_candidate_plus_winner_read_cap_without_table_scan(handoff, monkeypatch):
    case = handoff
    request = json.loads(case["request"].read_text())
    ids = list(range(1000, 2023)); ids.insert(501, 7)
    request["source_token_ids"] = transfer._blob(struct.pack("<1024i", *ids), "INT32", [1, 1024])
    request["binding_digest"] = transfer._binding(request)
    path = case["resource"] / "max-request.json"; save(path, request)
    original, reads = os.pread, []
    def bounded(fd, size, offset):
        assert size in {4, 2048}
        reads.append(size)
        assert sum(reads) <= 2103300
        return original(fd, size, offset)
    monkeypatch.setattr(os, "pread", bounded)
    response = extract(case, path)
    assert response["status"] == transfer._EXTRACTED, response
    assert response["matching_positions"] == [{"token_id": 7, "positions": [501]}]
    assert response["table_read_bytes"] == sum(reads) == 2103300


def test_alternate_resource_table_path_cannot_satisfy_original_config_claim(handoff):
    case = handoff
    alternate = case["resource"] / "alternate-int8.bin"
    os.link(case["tables"][0], alternate)
    response = transfer.extract_rows(case["request"], alternate, case["tables"][1], case["extracted"])
    assert response["status"] == "FAILED" and "configuration" in response["error"]
    assert response["table_read_bytes"] == 0 and response["next_token"] is None


def locate(case):
    return transfer.locate_rows(case["request"], *case["tables"], case["extracted"].with_name("resource-location"))


def relocate_pair(case, tokens):
    # Remove the known context-token pair, then place exact original raw bytes
    # at IDs outside the request's context set. This fixture changes no request.
    with case["tables"][1].open("r+b") as stream:
        stream.seek(7 * 4); stream.write(struct.pack("<f", .5))
        for token in tokens:
            stream.seek(token * 4); stream.write(case["case"]["scales"][7])
    with case["tables"][0].open("r+b") as stream:
        for token in tokens:
            stream.seek(token * 2048); stream.write(case["case"]["rows"][7])


def test_location_diagnosis_can_find_exact_pair_outside_source_ids_without_changing_extract_or_import(handoff, monkeypatch):
    case = handoff
    relocate_pair(case, [5432])
    request_before = case["request"].read_bytes()
    assert extract(case)["status"] == "ORIGINAL_DECODE_ROWS_NO_MATCH"
    original, calls = os.pread, []
    def bounded(fd, size, offset):
        assert size in {65536, 10240, 4, 2044}
        calls.append((size, offset))
        return original(fd, size, offset)
    monkeypatch.setattr(os, "pread", bounded)
    result = locate(case)
    assert result["status"] == "ORIGINAL_DECODE_ROW_LOCATION_UNIQUE", result
    assert result["scale_scan_complete"] is result["prefix_scan_complete"] is True
    assert result["scale_scan_bytes"] == 993280
    assert result["scale_match_count"] == result["prefix_match_count"] == result["pair_match_count"] == 1
    assert result["matching_pair_token_ids"] == [5432]
    assert result["prefix_read_bytes"] == 4 and result["remainder_read_bytes"] == 2044
    assert result["table_read_bytes"] == 995328 == sum(size for size, _ in calls)
    assert result["row_components"][0]["present_in_request_source_ids"] is False
    assert result["row_components"][0]["embedding_matches"] is True
    assert result["executed_embedding_sha256"] == result["row_components"][0]["weight_sha256"]
    assert case["request"].read_bytes() == request_before
    assert result["diagnostic_only"] is True and result["import_eligibility"] == "NOT_AN_EXTRACTION_RESPONSE"
    assert "selected_source_token_id" not in result and "next_token" not in result
    assert not (case["extracted"].with_name("resource-location") / "response.json").exists()
    no_next(case, consume(case, case["extracted"].with_name("resource-location") / "diagnostic.json"))


def test_location_checks_more_than1024_equal_scales_with_sole_prefix_match_beyond1024(handoff):
    case = handoff
    with case["tables"][1].open("r+b") as stream:
        stream.write(case["case"]["scales"][7] * 4096)
    with case["tables"][0].open("r+b") as stream:
        stream.seek(7 * 2048); stream.write(bytes(2048))
        stream.seek(3000 * 2048); stream.write(case["case"]["rows"][7])
    result = locate(case)
    assert result["status"] == "ORIGINAL_DECODE_ROW_LOCATION_UNIQUE", result
    assert result["scale_match_count"] == 4096 and result["prefix_match_count"] == 1
    assert result["matching_pair_token_ids"] == [3000]
    assert result["prefix_read_bytes"] == 4096 * 4
    assert result["table_read_bytes"] == 993280 + 4096 * 4 + 2044
    assert result["pair_scan"] == "COMPLETE_EXACT_SCALE_PREFIX_AND_REMAINDER_SEARCH"


def test_location_budget_cap_covers_all_scale_and_prefix_candidates_before_reading_any_remainder(handoff, monkeypatch):
    case = handoff
    with case["tables"][1].open("r+b") as stream:
        stream.write(case["case"]["scales"][7] * transfer._VOCAB)
    with case["tables"][0].open("r+b") as stream:
        for token in range(100):
            stream.seek(token * 2048); stream.write(case["case"]["rows"][7][:4])
    original = os.pread
    def no_remainders(fd, size, offset):
        assert size != 2044
        return original(fd, size, offset)
    monkeypatch.setattr(os, "pread", no_remainders)
    result = locate(case)
    assert result["status"] == "ORIGINAL_DECODE_ROW_LOCATION_BUDGET_CAP", result
    assert result["scale_scan_complete"] is result["prefix_scan_complete"] is True
    assert result["scale_match_count"] == transfer._VOCAB and result["prefix_match_count"] == 100
    assert result["table_read_bytes"] == 1986560 and result["remainder_read_bytes"] == 0
    assert result["predicted_complete_table_read_bytes"] == 1986560 + 100 * 2044
    assert result["pair_match_count"] is None and result["matching_pair_token_ids"] == []
    assert result["pair_scan"] == "NOT_PERFORMED_TOTAL_PAYLOAD_BUDGET_EXCEEDED"
    assert result["uniqueness_scope"] == "NOT_ESTABLISHED"
    assert len(result["scale_matching_id_examples"]) == len(result["prefix_matching_id_examples"]) == 16


@pytest.mark.parametrize("tokens", [[], [5432, 6655]])
def test_location_complete_no_match_or_ambiguity_is_only_a_declared_table_representation_claim(handoff, tokens):
    case = handoff
    relocate_pair(case, tokens)
    result = locate(case)
    assert result["status"] == ("ORIGINAL_DECODE_ROW_LOCATION_AMBIGUOUS" if tokens else "ORIGINAL_DECODE_ROW_LOCATION_NO_MATCH"), result
    assert result["matching_pair_token_ids"] == tokens
    assert result["pair_match_count"] == len(tokens)
    assert result["scale_scan_complete"] is True
    assert result["logical_history"] == result["runtime_control_recipe"] == "UNVERIFIED"
    assert result["next_token_generation"] == "NOT_PERFORMED"


def test_location_reports_scale_and_prefix_match_but_rejects_full_int8_row_mismatch(handoff):
    case = handoff
    with case["tables"][0].open("r+b") as stream:
        stream.seek(7 * 2048 + 8); stream.write(b"\0")
    result = locate(case)
    assert result["status"] == "ORIGINAL_DECODE_ROW_LOCATION_NO_MATCH", result
    row, = result["row_components"]
    assert row["scale_matches"] is row["prefix_matches"] is True
    assert row["embedding_matches"] is row["pair_matches"] is False
    assert row["weight_sha256"] != result["executed_embedding_sha256"]


@pytest.mark.parametrize("changed", ["table", "request"])
def test_location_rejects_mutation_without_promoting_incomplete_search(handoff, monkeypatch, changed):
    case = handoff
    original, mutated = os.pread, False
    def race(fd, size, offset):
        nonlocal mutated
        raw = original(fd, size, offset)
        if not mutated:
            mutated = True
            path = case["tables"][0] if changed == "table" else case["request"]
            with path.open("r+b") as stream: stream.write(b"x")
        return raw
    monkeypatch.setattr(os, "pread", race)
    result = locate(case)
    assert result["status"] == "FAILED" and "changed" in result["error"], result
    assert result["uniqueness_scope"] == "NOT_ESTABLISHED_AFTER_ERROR"


def test_copied_locate_command_is_stdlib_only_and_diagnostic_even_when_unique(handoff):
    case = handoff
    env = dict(os.environ); env.pop("AI_RUN_DIR", None); env.pop("PYTHONPATH", None)
    output = case["extracted"].with_name("isolated-location")
    done = subprocess.run([sys.executable, "-I", "-S", str(case["prepared"] / "extract_rows.py"), "locate",
        "--request", str(case["request"]), "--weight-table", str(case["tables"][0]),
        "--scale-table", str(case["tables"][1]), "--output", str(output)],
        cwd=case["resource"], env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    shown = json.loads(done.stdout)
    assert shown["status"] == "ORIGINAL_DECODE_ROW_LOCATION_UNIQUE"
    assert shown["diagnostic_only"] is True and shown["import_eligibility"] == "NOT_AN_EXTRACTION_RESPONSE"
    assert (output / "diagnostic.json").is_file() and not (output / "response.json").exists()


@pytest.fixture
def query_handoff(embedding_case, tmp_path, request):
    case = embedding_case
    query_id = getattr(request, "param", 269)
    ids = [3, 4, 7, 8, 9, 10, 11] * 36 + [3, 4, 7, 8]
    context = set_candidates(case, ids)
    retained = case["case"]
    logits = retained["board"] / retained["paths"][retained["proof"]["logits_runtime_output_index"]]
    with logits.open("r+b") as stream:
        stream.seek(101 * 4); stream.write(struct.pack("<f", 0))
        stream.seek(0); stream.write(struct.pack("<f", 3))
    retained["report"]["retrieved"][str(logits.relative_to(retained["board"]))] = record(logits)
    refresh(retained)
    retained["out"] = tmp_path / "query-decode-inspection"
    assert inspect_outputs(retained)["status"] == "DECODE_OUTPUT_FILES_BOUND_TO_LOGGED_SLOTS"
    resource = tmp_path / "query-resource-host"
    tables = (resource / "original-int8.bin", resource / "original-scale.bin")
    case["config"].write_text(yaml.safe_dump({"embedding_weight_path": str(tables[0]), "embedding_scale_path": str(tables[1])}))
    prepared = tmp_path / "explicit-query-prepared"
    manifest = transfer.prepare_rows(retained["out"], case["config"], prepared, query_token_id=query_id)
    assert manifest["status"] == "ORIGINAL_DECODE_ROW_TRANSFER_PREPARED", manifest
    resource.mkdir()
    for original, target in zip(case["tables"], tables): original.rename(target)
    actual_id = 0 if query_id == 0 else 269
    next_row = case["rows"][7] if actual_id == 0 else bytes((13 + offset) % 256 for offset in range(2048))
    next_scale = case["scales"][7] if actual_id == 0 else struct.pack("<f", .002)
    with tables[0].open("r+b") as stream:
        stream.seek(0); stream.write(next_row)
        stream.seek(actual_id * 2048); stream.write(case["rows"][7])
    with tables[1].open("r+b") as stream:
        stream.seek(0); stream.write(next_scale)
        stream.seek(actual_id * 4); stream.write(case["scales"][7])
        stream.seek(7 * 4); stream.write(struct.pack("<f", .9))
    return {"case": case, "prepared": prepared, "manifest": manifest, "tables": tables,
        "request": prepared / "request.json", "resource": resource, "context": context, "query_id": query_id,
        "expected_next_row": next_row, "expected_next_scale": next_scale,
        "extracted": tmp_path / "explicit-query-extracted", "imported": tmp_path / "explicit-query-imported"}


@pytest.mark.parametrize("query_handoff", [0, 269], indirect=True)
def test_v2_explicit_query_keeps256_context_ids_and_reads_only_query_and_next_rows(query_handoff, monkeypatch):
    case = query_handoff
    request = json.loads(case["request"].read_text())
    assert request["schema_version"] == 2 and request["query_token_id"] == case["query_id"]
    assert request["binding_method"] == transfer._QUERY_METHOD
    assert request["source_token_ids"]["shape"] == [1, 256]
    assert request["source_token_ids"]["sha256"] == hashlib.sha256(case["context"]).hexdigest()
    assert request["source_token_ids"]["bytes"] == 1024
    assert case["manifest"]["source_id_vector_role"] == "UNCHANGED_CONTEXT_ONLY"
    assert case["manifest"]["context_id_count"] == 256 and case["manifest"]["effective_candidate_count"] == 1
    original, reads = os.pread, []
    def exact_rows(fd, size, offset):
        assert size in {2048, 4}
        reads.append((size, offset))
        return original(fd, size, offset)
    monkeypatch.setattr(os, "pread", exact_rows)
    response = extract(case)
    assert response["status"] == transfer._EXTRACTED, response
    assert response["schema_version"] == 2 and response["query_token_id"] == case["query_id"]
    assert response["context_id_count"] == 256 and response["context_unique_id_count"] == 7
    assert response["candidate_count"] == response["effective_candidate_count"] == response["unique_candidate_count"] == 1
    assert response["matching_positions"] == [] and response["source_match"]["positions"] == []
    assert response["uniqueness_scope"] == transfer._QUERY_METHOD
    assert response["table_read_bytes"] == sum(size for size, _ in reads) == (2052 if case["query_id"] == 0 else 4104)
    result = consume(case)
    assert result["status"] == "ORIGINAL_DECODE_TRANSFERRED_ROWS_BOUND_NEXT_TOKEN_PREPARED", result
    assert result["schema_version"] == 2 and result["selected_source_token_id"] == case["query_id"]
    assert result["matching_source_positions"] == [] and result["selected_next_token_id"] == 0
    assert result["uniqueness_scope"] == transfer._QUERY_METHOD
    assert result["logical_history"] == result["state_feedback"] == "UNVERIFIED"
    assert (case["imported"] / "next-token/inputs_embeds.bin").read_bytes() == case["expected_next_row"]
    assert (case["imported"] / "next-token/embed_scale.bin").read_bytes() == case["expected_next_scale"]
    assert sum(item["bytes"] for item in result["next_token_inputs"]) == 2056
    assert (case["case"]["source"] / "input_ids.bin").read_bytes() == case["context"]


@pytest.mark.parametrize("query_handoff", [268], indirect=True)
def test_v2_wrong_explicit_query_cannot_reuse_matching_context_or_prepare_next_bytes(query_handoff):
    case = query_handoff
    response = extract(case)
    assert response["status"] == "ORIGINAL_DECODE_ROWS_NO_MATCH", response
    assert response["matching_token_ids"] == [] and response["next_token"] is None
    assert response["table_read_bytes"] == 2052
    no_next(case, consume(case))


@pytest.mark.parametrize("fault", ["version1", "wrong_query", "bool_query", "context_count", "scope", "invented_position"])
def test_v2_response_cannot_downgrade_change_query_or_invent_context_positions(query_handoff, fault):
    case = query_handoff
    response = extract(case)
    assert response["status"] == transfer._EXTRACTED
    if fault == "version1": response["schema_version"] = 1
    elif fault == "wrong_query": response["query_token_id"] = 268
    elif fault == "bool_query": response["query_token_id"] = True
    elif fault == "context_count": response["context_id_count"] = 1
    elif fault == "scope": response["uniqueness_scope"] = "EXPLICIT_SOURCE_ID_CANDIDATES_ONLY"
    else: response["matching_positions"] = [{"token_id": 269, "positions": [255]}]
    save(case["extracted"] / "response.json", response)
    no_next(case, consume(case))


@pytest.mark.parametrize("fault", ["downgrade", "bool_query", "method", "query_hash"])
def test_v2_request_flags_and_query_are_sha_bound_and_version_checked(query_handoff, fault):
    case = query_handoff
    request = json.loads(case["request"].read_text())
    if fault == "downgrade": request["schema_version"] = 1
    elif fault == "bool_query": request["query_token_id"] = True
    elif fault == "method": request["binding_method"] = "USE_LAST_CONTEXT_ID"
    else: request["query_token_id"] = 268
    invalid = case["resource"] / "invalid-query.json"; save(invalid, request)
    response = extract(case, invalid)
    assert response["status"] == "FAILED" and response["table_read_bytes"] == 0, response
    assert response["next_token"] is None


def test_v2_import_keeps_context_hash_a_live_binding(query_handoff):
    case = query_handoff
    assert extract(case)["status"] == transfer._EXTRACTED
    raw = bytearray(case["context"]); raw[-4:] = struct.pack("<i", 12)
    (case["case"]["source"] / "input_ids.bin").write_bytes(raw)
    no_next(case, consume(case))


@pytest.mark.parametrize("query", [True, False, -1, 248320, 269.0])
def test_prepare_query_api_rejects_boolean_noninteger_and_out_of_range_flags(embedding_case, tmp_path, query):
    result = transfer.prepare_rows(embedding_case["inspection"], embedding_case["config"], tmp_path / "invalid-query", query_token_id=query)
    assert result["status"] == "FAILED" and "integer" in result["error"]
    assert not (tmp_path / "invalid-query/source-audit").exists()


def test_v1_request_does_not_silently_accept_query_flags(handoff):
    case = handoff
    request = json.loads(case["request"].read_text())
    request.update(query_token_id=269, binding_method=transfer._QUERY_METHOD)
    invalid = case["resource"] / "invalid-v1-query.json"; save(invalid, request)
    response = extract(case, invalid)
    assert response["status"] == "FAILED" and response["table_read_bytes"] == 0


def test_ctime_only_primary_revalidation_is_visible_in_new_query_prepare_and_import_without_rewriting_old_evidence(query_handoff, tmp_path):
    case = query_handoff
    retained = case["case"]["case"]
    primary = Path(retained["proof"]["primary_model_binding"]["path"])
    original_stat = primary.stat()
    original_sha = hashlib.sha256(primary.read_bytes()).hexdigest()
    old_audit = case["prepared"] / "source-audit/report.json"
    historical_paths = [old_audit, case["prepared"] / "manifest.json", case["request"],
        retained["execution"] / "plan.json", retained["execution"] / "result.json",
        retained["sources"]["compilation"] / "report.json", Path(case["manifest"]["inspection"]) / "report.json"]
    historical_bytes = {path: path.read_bytes() for path in historical_paths}
    assert json.loads(historical_bytes[old_audit])["model_payload_access"] == "NONE"
    # chmod of the existing exact file changes ctime without changing its
    # inode, extent, mtime, permissions or bytes. No receipt is rewritten.
    os.chmod(primary, original_stat.st_mode & 0o7777)
    current = primary.stat()
    assert (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns) == (
        original_stat.st_dev, original_stat.st_ino, original_stat.st_size, original_stat.st_mtime_ns)
    assert current.st_ctime_ns != original_stat.st_ctime_ns
    assert hashlib.sha256(primary.read_bytes()).hexdigest() == original_sha
    hidden = []
    for path in case["tables"]:
        held = path.with_suffix(".resource-only")
        path.rename(held); hidden.append((held, path))
    prepared = tmp_path / "ctime-query-prepared"
    try:
        manifest = transfer.prepare_rows(case["manifest"]["inspection"], case["case"]["config"], prepared, query_token_id=269)
    finally:
        for held, path in hidden: held.rename(path)
    assert manifest["status"] == "ORIGINAL_DECODE_ROW_TRANSFER_PREPARED", manifest
    assert manifest["model_payload_access"] == "SMALL_PRIMARY_OMC_REHASHED_FOR_CTIME_ONLY"
    assert manifest["primary_omc_revalidation_observations"]
    current_audit = json.loads((prepared / "source-audit/report.json").read_text())
    assert current_audit["model_payload_access"] == "SMALL_PRIMARY_OMC_REHASHED_FOR_CTIME_ONLY"
    assert current_audit["primary_omc_revalidation_observations"]
    case.update(prepared=prepared, request=prepared / "request.json", manifest=manifest)
    assert extract(case)["status"] == transfer._EXTRACTED
    result = consume(case)
    assert result["status"] == "ORIGINAL_DECODE_TRANSFERRED_ROWS_BOUND_NEXT_TOKEN_PREPARED", result
    assert result["model_payload_access"] == "SMALL_PRIMARY_OMC_REHASHED_FOR_CTIME_ONLY"
    assert result["primary_omc_revalidation_observations"]
    assert all(path.read_bytes() == raw for path, raw in historical_bytes.items())
    assert sum(row["bytes"] for row in result["next_token_inputs"]) == 2056


def test_historical_metadata_and_restored_payload_ctimes_rebind_end_to_end_without_changing_old_receipt_bytes(query_handoff, tmp_path):
    case = query_handoff
    retained = case["case"]["case"]
    inspection_path = Path(case["manifest"]["inspection"]) / "report.json"
    inspection = json.loads(inspection_path.read_text())
    audit_path = case["prepared"] / "source-audit/report.json"
    audit = json.loads(audit_path.read_text())
    identities = [*inspection["evidence_bindings"], *audit["source_bindings"],
        *retained["proof"]["metadata_bindings"], *retained["sources"]["outer"]["source_bindings"]]
    metadata_paths = {Path(row["path"]) for row in identities}
    metadata_paths |= {inspection_path, audit_path, case["prepared"] / "manifest.json", case["request"], case["case"]["config"]}
    assert all(path.suffix in {".json", ".yaml", ".yml", ".exitcode"} for path in metadata_paths)
    old_metadata = {path: path.read_bytes() for path in metadata_paths}
    originals = {path: stamp(path) for path in case["case"]["source"].iterdir()}
    for path in metadata_paths:
        before = path.stat()
        os.chmod(path, before.st_mode & 0o7777)
        assert stamp(path)[:4] == (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        assert path.read_bytes() == old_metadata[path]
    derived = list((case["case"]["trial"] / "inputs").iterdir())
    assert len(derived) == 13
    payload_paths = [*derived, *[retained["board"] / name for name in retained["report"]["retrieved"]]]
    old_payload_sha = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in payload_paths}
    for path in derived:
        before = path.stat()
        assert before.st_mode & 0o7777 == 0o644
        os.chmod(path, 0o777)
        os.chmod(path, 0o644)
        assert stamp(path)[:4] == (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    for path in payload_paths[len(derived):]:
        os.chmod(path, path.stat().st_mode & 0o7777)
    hidden = []
    for path in case["tables"]:
        held = path.with_suffix(".held-on-resource")
        path.rename(held); hidden.append((held, path))
    prepared = tmp_path / "all-ctime-query-prepared"
    try:
        manifest = transfer.prepare_rows(case["manifest"]["inspection"], case["case"]["config"], prepared, query_token_id=269)
    finally:
        for held, path in hidden: held.rename(path)
    assert manifest["status"] == "ORIGINAL_DECODE_ROW_TRANSFER_PREPARED", manifest
    assert manifest["metadata_revalidation_observations"] and manifest["payload_revalidation_observations"]
    assert manifest["model_payload_access"] == "NONE"
    current_audit = json.loads((prepared / "source-audit/report.json").read_text())
    assert current_audit["metadata_revalidation_observations"] and current_audit["payload_revalidation_observations"]
    assert current_audit["bound_logits"] == audit["bound_logits"]
    assert current_audit["bound_logits_current_read"]["stat"] != audit["bound_logits"]["stat"]
    assert current_audit["observed_payload_reads"]["bound_logits_bytes"] == 2 * 993280
    # A returned packet may also acquire historical metadata ctime drift. Its
    # JSON bytes stay exact; the copied Python extractor remains untouched.
    packet_paths = [prepared / "manifest.json", prepared / "request.json", prepared / "source-audit/report.json"]
    packet_bytes = {path: path.read_bytes() for path in packet_paths}
    for path in packet_paths: os.chmod(path, path.stat().st_mode & 0o7777)
    case.update(prepared=prepared, request=prepared / "request.json", manifest=manifest)
    assert extract(case)["status"] == transfer._EXTRACTED
    result = consume(case)
    assert result["status"] == "ORIGINAL_DECODE_TRANSFERRED_ROWS_BOUND_NEXT_TOKEN_PREPARED", result
    assert result["metadata_revalidation_observations"] and result["payload_revalidation_observations"]
    assert result["metadata_identity_revalidation"] == "CTIME_ONLY_FRESH_METADATA_SHA256_OLD_RECEIPTS_UNCHANGED"
    assert result["payload_identity_revalidation"] == "CTIME_ONLY_FRESH_PAYLOAD_SHA256_OLD_RECEIPTS_UNCHANGED"
    assert result["matching_source_positions"] == [] and result["selected_source_token_id"] == 269
    assert all(path.read_bytes() == raw for path, raw in {**old_metadata, **packet_bytes}.items())
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in old_payload_sha.items())
    assert all(stamp(path) == identity for path, identity in originals.items())


@pytest.mark.parametrize("changed", ["source_ids", "extractor_code"])
def test_metadata_ctime_rules_never_apply_to_original_context_ids_or_copied_python(query_handoff, changed):
    case = query_handoff
    assert extract(case)["status"] == transfer._EXTRACTED
    path = case["case"]["source"] / "input_ids.bin" if changed == "source_ids" else case["prepared"] / "extract_rows.py"
    raw = path.read_bytes()
    before = path.stat()
    time.sleep(.01)
    os.chmod(path, path.stat().st_mode & 0o7777)
    assert path.stat().st_ctime_ns != before.st_ctime_ns
    assert path.read_bytes() == raw
    no_next(case, consume(case))


def test_payload_ctime_rule_never_accepts_unrestored_derived_input_permissions(query_handoff):
    case = query_handoff
    assert extract(case)["status"] == transfer._EXTRACTED
    os.chmod(case["case"]["trial"] / "inputs/inputs_embeds.bin", 0o777)
    no_next(case, consume(case))


@pytest.mark.parametrize("which", ["request", "plan", "inspection"])
def test_active_import_metadata_ctime_change_is_refused_after_first_current_snapshot(query_handoff, monkeypatch, which):
    case = query_handoff
    assert extract(case)["status"] == transfer._EXTRACTED
    path = (case["request"] if which == "request" else
            case["case"]["case"]["execution"] / "plan.json" if which == "plan" else
            Path(case["manifest"]["inspection"]) / "report.json")
    original, body = transfer._response, path.read_bytes()
    def race(*args):
        result = original(*args)
        os.chmod(path, path.stat().st_mode & 0o7777)
        assert path.read_bytes() == body
        return result
    monkeypatch.setattr(transfer, "_response", race)
    result = consume(case)
    no_next(case, result)
    assert "changed" in result["error"]


def test_late_prepare_metadata_ctime_failure_never_reports_a_prepared_packet(query_handoff, tmp_path, monkeypatch):
    case = query_handoff
    original = transfer._assert_current_metadata
    def race(snapshot):
        packets = [Path(path) for path in snapshot if Path(path).name == "request.json"]
        if packets:
            path = packets[-1]
            before = path.stat()
            time.sleep(.01)
            os.chmod(path, path.stat().st_mode & 0o7777)
            assert path.stat().st_ctime_ns != before.st_ctime_ns
        return original(snapshot)
    monkeypatch.setattr(transfer, "_assert_current_metadata", race)
    hidden = []
    for path in case["tables"]:
        held = path.with_suffix(".held-for-late-prepare")
        path.rename(held); hidden.append((held, path))
    output = tmp_path / "late-prepare-race"
    try:
        result = transfer.prepare_rows(case["manifest"]["inspection"], case["case"]["config"], output, query_token_id=269)
    finally:
        for held, path in hidden: held.rename(path)
    assert result["status"] == "FAILED" and "changed" in result["error"], result
    assert json.loads((output / "manifest.json").read_text())["status"] == "FAILED"
    assert "next" not in result and not (output / "next-token").exists()


def test_explicit_copied_script_ctime_recovery_reuses_existing_response_after_chmod777_tree(query_handoff, tmp_path):
    case = query_handoff
    assert extract(case)["status"] == transfer._EXTRACTED
    preserved = {path: path.read_bytes() for path in case["prepared"].rglob("*") if path.is_file()}
    response = case["extracted"] / "response.json"
    preserved[response] = response.read_bytes()
    time.sleep(.01)
    for path in [case["prepared"], *case["prepared"].rglob("*")]: os.chmod(path, 0o777)
    failed = consume(case)
    no_next(case, failed)
    failed_report = (case["imported"] / "report.json").read_bytes()
    case["imported"] = tmp_path / "explicit-script-recovery-v4"
    result = consume(case, revalidate_extractor_metadata=True)
    assert result["status"] == "ORIGINAL_DECODE_TRANSFERRED_ROWS_BOUND_NEXT_TOKEN_PREPARED", result
    assert result["selected_source_token_id"] == 269 and result["selected_next_token_id"] == 0
    assert result["copied_extractor_revalidation_count"] == 1
    event, = result["copied_extractor_revalidation_observations"]
    assert event["kind"] == "COPIED_EXTRACTOR_CTIME_ONLY_REVALIDATED" and event["status"] == "FULL_SOURCE_SHA256_VERIFIED"
    assert event["recorded_identity"]["sha256"] == event["current_identity"]["sha256"] == case["manifest"]["extraction_script"]["sha256"]
    assert event["recorded_identity"]["stat"][:4] == event["current_identity"]["stat"][:4]
    assert event["recorded_identity"]["stat"][4] != event["current_identity"]["stat"][4]
    assert event["mode_action"] == "NONE" and "NOT_RECORDED" in event["historical_mode"]
    assert event["current_full_identity"]["mode"] & 0o7777 == 0o777
    assert (case["prepared"] / "extract_rows.py").stat().st_mode & 0o7777 == 0o777
    assert all(path.read_bytes() == raw for path, raw in preserved.items())
    assert (tmp_path / "explicit-query-imported/report.json").read_bytes() == failed_report
    assert sum(row["bytes"] for row in result["next_token_inputs"]) == 2056


@pytest.mark.parametrize("fault", ["bytes", "inode", "mtime", "recorded_path", "response_source_sha"])
def test_script_recovery_flag_never_accepts_source_change_or_nonctime_identity_drift(query_handoff, fault):
    case = query_handoff
    assert extract(case)["status"] == transfer._EXTRACTED
    script = case["prepared"] / "extract_rows.py"
    before, raw = script.stat(), script.read_bytes()
    if fault == "bytes":
        with script.open("r+b") as stream: stream.write(b"!")
        os.utime(script, ns=(before.st_atime_ns, before.st_mtime_ns))
    elif fault == "inode":
        other = script.with_name("different-inode.py"); other.write_bytes(raw)
        os.utime(other, ns=(before.st_atime_ns, before.st_mtime_ns)); other.replace(script)
    elif fault == "mtime": os.utime(script, ns=(before.st_atime_ns, before.st_mtime_ns + 1))
    elif fault == "recorded_path":
        path = case["prepared"] / "manifest.json"
        value = json.loads(path.read_text()); value["extraction_script"]["path"] = str(case["resource"] / "unrelated.py")
        save(path, value)
    else:
        path = case["extracted"] / "response.json"
        value = json.loads(path.read_text()); value["extractor_sha256"] = "f" * 64
        save(path, value)
    no_next(case, consume(case, revalidate_extractor_metadata=True))


def test_script_metadata_recovery_does_not_admit_new_drift_during_import(query_handoff, monkeypatch):
    case = query_handoff
    assert extract(case)["status"] == transfer._EXTRACTED
    script = case["prepared"] / "extract_rows.py"
    os.chmod(script, 0o777)
    original = transfer._response
    def race(*args):
        result = original(*args)
        time.sleep(.01)
        os.chmod(script, 0o644)
        return result
    monkeypatch.setattr(transfer, "_response", race)
    result = consume(case, revalidate_extractor_metadata=True)
    no_next(case, result)
    assert result["copied_extractor_revalidation_count"] == 1
    assert "changed" in result["error"]


def test_import_cli_exposes_explicit_copied_script_revalidation(query_handoff, capsys):
    case = query_handoff
    assert extract(case)["status"] == transfer._EXTRACTED
    time.sleep(.01)
    os.chmod(case["prepared"] / "extract_rows.py", 0o777)
    assert transfer.main(["import", "--prepared", str(case["prepared"]), "--response", str(case["extracted"] / "response.json"),
        "--output", str(case["imported"]), "--revalidate-extractor-metadata"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["copied_extractor_revalidation_count"] == 1
    assert shown["copied_extractor_identity_revalidation"] == "EXPLICIT_CTIME_ONLY_FULL_SOURCE_SHA256_OLD_RECEIPT_UNCHANGED"
