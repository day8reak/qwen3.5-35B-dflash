import hashlib

import pytest

from qwen35_moe_dflash.draft_precision_log import inspect_precision_log


def _case(tmp_path, text):
    raw = text.encode()
    (tmp_path / "compile.log").write_bytes(raw)
    return {"compile_log": "compile.log", "compile_log_sha256": hashlib.sha256(raw).hexdigest()}


def test_exact_names_and_bound_context_do_not_claim_placement(tmp_path):
    report = _case(tmp_path, "begin\nname:v5810 rejected\nname:v581 CPUCL not support\nafter\n")
    result = inspect_precision_log(tmp_path, report, ["v581"])
    assert result["status"] == "INSPECTED" and result["matched_lines"] == 1
    assert [x["line"] for x in result["lines"] if x["exact_node_match"]] == [3]
    assert "do not prove" in result["scope"]
    assert result["source"]["sha256"] == report["compile_log_sha256"]


def test_log_digest_mismatch_discards_messages(tmp_path):
    report = _case(tmp_path, "v581\n")
    (tmp_path / "compile.log").write_text("v581 corrupt\n")
    result = inspect_precision_log(tmp_path, report, ["v581"])
    assert result["status"] == "NOT_AVAILABLE" and result["lines"] == []
    assert "SHA256" in result["reason"]


def test_log_line_budget_prioritizes_last_exact_hits(tmp_path):
    report = _case(tmp_path, "".join(f"context\nv581 message {i}\n" for i in range(100)))
    result = inspect_precision_log(tmp_path, report, ["v581"])
    assert result["matched_lines"] == 100 and result["truncated"]
    assert len(result["lines"]) <= 60
    assert result["lines"][-1]["text"] == "v581 message 99"


def test_support_context_survives_later_parameter_messages(tmp_path):
    report = _case(tmp_path, "v581 CPUCL dtype check\n" + "GetDataParam op=v581\n" * 100)
    result = inspect_precision_log(tmp_path, report, ["v581"])
    assert any(row["text"] == "v581 CPUCL dtype check" for row in result["lines"])
    assert len(result["lines"]) <= 60


@pytest.mark.parametrize("relative", ["../compile.log", "/tmp/compile.log"])
def test_log_rejects_paths_outside_compilation(tmp_path, relative):
    report = _case(tmp_path, "v581\n")
    report["compile_log"] = relative
    result = inspect_precision_log(tmp_path, report, ["v581"])
    assert result["status"] == "NOT_AVAILABLE" and not result["lines"]


def test_missing_receipt_and_no_matches_are_explicit(tmp_path):
    assert inspect_precision_log(tmp_path, {}, ["v581"])["status"] == "NOT_AVAILABLE"
    report = _case(tmp_path, "v5810\n")
    result = inspect_precision_log(tmp_path, report, ["v581"])
    assert result["status"] == "INSPECTED" and result["matched_lines"] == 0
    assert result["lines"] == []
