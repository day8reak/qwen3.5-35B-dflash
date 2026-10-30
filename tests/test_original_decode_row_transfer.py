"""Portable row handoff uses sparse host tables and retained decode fixtures."""
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import pytest
import yaml

from qwen35_moe_dflash import original_decode_row_transfer as transfer
from test_original_decode_embedding_probe import embedding_case, set_candidates, stamp
from test_target_decode_output_inventory import case as retained_case


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


def consume(case, response=None):
    return transfer.import_rows(case["prepared"], response or case["extracted"] / "response.json", case["imported"])


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
