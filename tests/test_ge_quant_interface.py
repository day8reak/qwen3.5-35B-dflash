import hashlib
import json
from pathlib import Path

import pytest

from qwen35_moe_dflash import ge_quant_interface as interface


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_scans_only_selected_roots_and_keeps_ge_source_and_runtime_evidence_distinct(tmp_path):
    ddk = tmp_path / "ddk"
    ge = tmp_path / "ge"
    include = ddk / "ddk/ai_ddk_lib/include"
    source = _write(ddk / "ddk/ascendc/ops/impl/quant_matmul.py",
                    "raise AssertionError('must never import this source')\n"
                    "def quant_matmul(x, weight, scale):\n    pass\n")
    proto = _write(ge / "include/ops/public_proto.h",
                   "REG_OP(QuantBatchMatmul)\n.INPUT(x, TensorType({DT_INT8}))\n"
                   ".INPUT(weight, TensorType({DT_INT4}))\n.OUTPUT(y, TensorType({DT_FLOAT16}))\n")
    _write(ddk / "ddk/tbe/ops/general.h", "REG_OP(GroupedMatmul)\n.INPUT(x, DT_INT8)\n")
    _write(ddk / "tools/tools_ascendc/include/packing.h", "// I4ZN and QPARAM are interface text only.\n")
    manager = _write(include / "HiAiModelManagerService.h",
                     "class Manager {\n int Process(Tensors& in, Tensors& out);\n void Sync();\n};\n")
    tensor = _write(include / "compatible/AiTensor.h", "void FlushCache();\nvoid InvalidateCache();\nvoid *GetBuffer();\n")
    _write(ddk / "unselected/quant_other_profile.py", "def nd_to_i4zn(): pass\n")
    _write(ge / "lib64/quant_payload.py", "def trans_quant_param(): pass\n")
    output = tmp_path / "result"
    report = interface.inspect_quant_interfaces(ddk, output, ge_root=ge, hiai_include=include)
    assert report["status"] == "TEXT_EVIDENCE_ONLY"
    assert "no installation" in report["scope"]
    assert report["scan"]["copied_files"] == 6
    assert report["scan"]["limited"] is False
    found = {Path(row["path"]): row for row in report["files"]}
    assert found[source]["origins"] == ["selected_ddk"]
    assert found[proto]["origins"] == ["ge"]
    assert found[manager]["origins"] == ["selected_ddk", "hiai"]
    assert found[tensor]["evidence_type"] == "HIAI_RUNTIME_HEADER"
    assert any("Process" in line["text"] for line in found[manager]["excerpt"])
    assert any("InvalidateCache" in line["text"] for line in found[tensor]["excerpt"])
    assert any("INPUT" in line["text"] for line in found[proto]["excerpt"])
    assert all("unselected" not in row["path"] and "lib64" not in row["path"] for row in report["files"])
    for record in report["files"]:
        data = (output / record["copy"]).read_bytes()
        assert data == Path(record["path"]).read_bytes()
        assert record["bytes"] == len(data)
        assert record["sha256"] == hashlib.sha256(data).hexdigest()
    assert json.loads((output / "report.json").read_text()) == report
    assert "Missing text is not proof of no kernel" in (output / "report.txt").read_text()


@pytest.mark.parametrize("limit,value", [("MAX_COPIES", 2), ("MAX_ENTRIES", 2),
                                         ("MAX_TOTAL_BYTES", 64), ("MAX_FILE_BYTES", 16)])
def test_scan_budget_limits_are_recorded_and_never_exceeded(tmp_path, monkeypatch, limit, value):
    ddk = tmp_path / "ddk"
    for number in range(6):
        _write(ddk / f"ddk/ascendc/ops/quant_{number}.py", "#" * 31 + "\n")
    monkeypatch.setattr(interface, limit, value)
    report = interface.inspect_quant_interfaces(ddk, tmp_path / "result")
    assert report["scan"]["limited"] is True
    assert limit in report["scan"]["stop_reasons"]
    assert report["scan"]["entries"] <= interface.MAX_ENTRIES
    assert report["scan"]["bytes_read"] <= interface.MAX_TOTAL_BYTES
    assert report["scan"]["copied_files"] <= interface.MAX_COPIES
    assert all(row["bytes"] <= interface.MAX_FILE_BYTES for row in report["files"])
    if limit == "MAX_FILE_BYTES":
        assert report["scan"]["files_read"] == 0


def test_depth_limit_missing_roots_and_binary_rejection(tmp_path, monkeypatch):
    ddk = tmp_path / "ddk"
    _write(ddk / "ddk/ascendc/ops/deep/quant_hidden.py", "DynamicQuant\n")
    binary = ddk / "ddk/ascendc/ops/quant_binary.h"
    binary.write_bytes(b"\0QuantMatmul")
    _write(ddk / "ddk/ascendc/ops/normal_header.h", "// No matching interface here.\n")
    monkeypatch.setattr(interface, "MAX_DEPTH", 0)
    report = interface.inspect_quant_interfaces(ddk, tmp_path / "result")
    assert report["files"] == []
    assert "MAX_DEPTH" in report["scan"]["stop_reasons"]
    assert any(item["reason"] == "BINARY_CONTENT" for item in report["skipped"])
    assert sum(not item["exists"] for item in report["roots"]) == 3
    assert all(item["status"] == "MISSING" for item in report["runtime_headers"])


def test_symlinks_do_not_expand_scan_to_other_profiles(tmp_path):
    ddk = tmp_path / "ddk"
    outside = tmp_path / "other-profile"
    foreign = _write(outside / "quant.py", "def trans_quant_param(): pass\n")
    ops = ddk / "ddk/ascendc/ops"
    ops.mkdir(parents=True)
    (ops / "quant_link.py").symlink_to(foreign)
    (ops / "linked_dir").symlink_to(outside, target_is_directory=True)
    tbe = ddk / "ddk/tbe"
    tbe.mkdir(parents=True)
    (tbe / "ops").symlink_to(outside, target_is_directory=True)
    include = ddk / "ddk/ai_ddk_lib/include"
    include.mkdir(parents=True)
    (include / "HiAiModelManagerService.h").symlink_to(foreign)
    report = interface.inspect_quant_interfaces(ddk, tmp_path / "result")
    assert report["files"] == []
    assert report["scan"]["files_read"] == 0
    assert any(row["reason"] == "SYMLINK_ENTRY" for row in report["skipped"])
    assert any(not row["allowed"] for row in report["roots"])


def test_hiai_ancestor_symlink_cannot_escape_selected_ddk(tmp_path):
    ddk = tmp_path / "ddk"
    other = tmp_path / "other-profile"
    _write(other / "include/HiAiModelManagerService.h", "void Process();\n")
    (ddk / "ddk").mkdir(parents=True)
    (ddk / "ddk/ai_ddk_lib").symlink_to(other, target_is_directory=True)
    report = interface.inspect_quant_interfaces(ddk, tmp_path / "result")
    assert report["files"] == []
    assert report["scan"]["files_read"] == 0
    assert report["runtime_headers"][0]["status"] == "SYMLINK_OR_OUTSIDE_ROOT"


def test_outputs_must_be_new_and_outside_read_only_toolchains(tmp_path):
    ddk = tmp_path / "ddk"
    ddk.mkdir()
    with pytest.raises(ValueError, match="outside"):
        interface.inspect_quant_interfaces(ddk, ddk / "inspection")
    assert not (ddk / "inspection").exists()
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "report.json").write_text("keep")
    with pytest.raises(FileExistsError):
        interface.inspect_quant_interfaces(ddk, existing)
    assert (existing / "report.json").read_text() == "keep"


@pytest.mark.parametrize("op", ["QuantBatchMatmul", "dynamic_quant", "grouped_matmul", "quant_matmul"])
def test_large_proto_excerpt_and_short_report_prioritize_quant_matches(tmp_path, op):
    ddk = tmp_path / "ddk"
    prefix = "// copyright header\n" + "\n".join(
        f"REG_OP(Unrelated{index})\n.INPUT(other, DT_FLOAT)\nstruct Helper{index} {{}};"
        for index in range(40)) + "\n"
    target = (f"REG_OP({op})\n.INPUT(x, DT_INT8)\n.INPUT(weight, DT_INT4)\n"
              ".OUTPUT(y, DT_FLOAT16)\n.ATTR(group_size, Int, 256)\n")
    source = _write(ddk / "ddk/ascendc/ops/public_proto.h", prefix + target)
    report = interface.inspect_quant_interfaces(ddk, tmp_path / "result")
    record = next(row for row in report["files"] if row["path"] == str(source))
    excerpt = record["excerpt"]
    assert len(excerpt) <= 32
    assert op in record["text_matches"]
    assert any(row["matched"] and row["text"] == f"REG_OP({op})" for row in excerpt)
    assert any(row["text"] == ".ATTR(group_size, Int, 256)" for row in excerpt)
    assert min(row["line"] for row in excerpt) > 100
    text = (tmp_path / "result/report.txt").read_text()
    assert f"REG_OP({op})" in text
    assert ".INPUT(x, DT_INT8)" in text
    assert ".INPUT(weight, DT_INT4)" in text
    assert ".OUTPUT(y, DT_FLOAT16)" in text
    assert "copyright header" not in text
    assert "REG_OP(Unrelated" not in text
