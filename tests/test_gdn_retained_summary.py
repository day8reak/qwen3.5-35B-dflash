"""Read-only analysis of retained synthetic evidence; no device certification."""
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from qwen35_moe_dflash.ge_fused import gdn_diagnostics as diagnostic
from qwen35_moe_dflash.ge_fused.probes import build_fused_probes
from qwen35_moe_dflash.hashing import sha256_file
from test_ge_fused_probes import setup, _fake_results, _mock_tools


def _write(path, value):
    path.write_text(json.dumps(value))


def _snapshot(*roots):
    return {str(path): sha256_file(path) for root in roots for path in root.rglob("*") if path.is_file()}


@pytest.fixture
def retained(setup, tmp_path, monkeypatch):
    binding, ge, ddk = setup
    _mock_tools(monkeypatch)
    probe = tmp_path / "source-probes"
    manifest = build_fused_probes(probe, binding, cases="gdn_prefill,gdn_decode",
                                  ge_root=ge, ddk_root=ddk, ddk_id="test-ddk")
    assert manifest["status"] == "COMPILED"
    root = tmp_path / "completed-diagnosis"
    root.mkdir()
    source = ddk / "ddk/ascendc/ops/impl/custom/chunk_gated_delta_rule.cpp"
    source.parent.mkdir(parents=True)
    source.write_text("// SYNTHETIC retained source, not a vendor kernel\n"
                      "void CopyOut() {\n"
                      "  output.SetGlobalBuffer(output_pointer);\n"
                      "  Duplicate(local, 0, length);\n"
                      "  SetAtomicAdd<float>();\n"
                      "  DataCopy(output[offset], local, length);\n"
                      "  memset(output, 0, bytes);\n"
                      "}\n")
    captured = diagnostic.collect_sources(ddk, root / "installed")
    source.unlink()  # The original installation is intentionally unavailable.
    previous = tmp_path / "previous-diagnosis"
    previous.mkdir()
    cases = {name: {"name": name, "host_input_readback": "MATCHED", "runs": []}
             for name in diagnostic.GDN_CASES}
    for fill in diagnostic.FILLS:
        execution = (previous if fill == 0 else root) / f"fill-{fill:02x}-0"
        execution.mkdir()
        results = execution / "results"
        _fake_results(probe, results, manifest)
        for original in manifest["cases"]:
            folder = results / original["name"]
            native_path = folder / "result.json"
            native = json.loads(native_path.read_text())
            native["firmware_id"] = "FAKE TEST ONLY"
            io = {"enabled": True, "process_called": True, "output_fill_byte": fill,
                  "input_readback_point": "AFTER_ALL_INPUT_COPIES_AND_OUTPUT_FILL_BEFORE_PROCESS",
                  "inputs": [], "outputs": []}
            for item in native["inputs"]:
                io["inputs"].append({"index": item["index"], "logical_bytes": item["bytes"],
                                     "allocated_bytes": item["bytes"], "readback_matches_source": True,
                                     "readback_sha256": item["sha256"]})
            for item in native["outputs"]:
                dtype = np.dtype("<f2" if item["dtype"] == "float16" else "<f4")
                integer = np.dtype("<u2" if dtype.itemsize == 2 else "<u4")
                words = np.ones(item["shape"], dtype=dtype).view(integer)
                sentinel = int.from_bytes(bytes([fill]) * dtype.itemsize, "little")
                if item["name"] == "core_attn":
                    words[0, 0, 0, :] = sentinel  # One full valid row follows every fill.
                    words[0, 0, 1, 0] = 0xA5A5  # A coincidental single-fill match.
                    octets = words.view(np.uint8).reshape(*item["shape"], dtype.itemsize)
                    octets[0, 0, 2, 0, 0] = fill  # Only one byte follows; no full element.
                    if item["shape"][1] == 8:
                        words[:, 6:] = sentinel  # Padding must never inflate valid counts.
                else:
                    assert dtype.itemsize == 4
                    words[0, 3, 4, :] = sentinel
                    words[0, 0, 0, 0] = 0x5A5A5A5A
                path = folder / item["path"]
                raw = words.tobytes()
                path.write_bytes(raw)
                item["sha256"] = sha256_file(path)
                io["outputs"].append({"index": item["index"], "logical_bytes": len(raw),
                                      "allocated_bytes_before": len(raw), "allocated_bytes_after": len(raw),
                                      "checked_after_process": True, "object_same": True, "buffer_same": True,
                                      "fill_count_available": True, "fill_byte_count": raw.count(bytes([fill]))})
            native["io_diagnostic"] = io
            _write(native_path, native)
            cases[original["name"]]["runs"].append({"fill": fill, "result": str(native_path),
                                                    "result_sha256": sha256_file(native_path), "io": io,
                                                    "numerical": []})
    report = {"schema_version": 1, "status": "DIAGNOSIS_COMPLETE", "fills": list(diagnostic.FILLS),
              "model_compilation": "NOT_RUN", "probe_run": str(probe),
              "probe_manifests": {str(probe): sha256_file(probe / "manifest.json")},
              "installed_source": {"status": captured["status"], "report": str(root / "installed/report.txt")},
              "cases": list(cases.values())}
    _write(root / "report.json", report)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("summary must not run tools"))
    monkeypatch.setattr(diagnostic, "collect_sources", lambda *args, **kwargs: pytest.fail("must not scan DDK"))
    return {"root": root, "probe": probe, "previous": previous, "report": report,
            "manifest": manifest, "captured": captured}


def test_retained_summary_checks_old_resume_paths_and_complete_elements(retained, tmp_path):
    root, probe, previous = (retained[key] for key in ("root", "probe", "previous"))
    before = _snapshot(root, probe, previous)
    output = tmp_path / "summary"
    result = diagnostic.summarize_retained_diagnosis(root, output)
    assert result["status"] == "SUMMARIZED"
    assert result["model_compilation"] == result["device_execution"] == "NOT_RUN"
    assert result["numerical_acceptance"] == "NOT_DEFINED" and result["placement"] == "UNVERIFIED"
    assert result["diagnosis_report_sha256"] == sha256_file(root / "report.json")
    assert len((output / "report.txt").read_text().splitlines()) <= 24
    for case in result["cases"]:
        assert case["runs"][0]["result"].startswith(str(previous))
        core, state = case["outputs"]
        assert core["dtype"] == "float16" and core["element_bytes"] == 2
        assert core["complete_elements_following_prefill"] == 128
        assert core["bytes_following_prefill"] == 257  # Full row plus one partial byte.
        assert [row["valid_matching_elements"] for row in core["fills"]] == [128, 129, 128]
        assert [row["sentinel_word"] for row in core["fills"]] == ["0x0000", "0xa5a5", "0x5a5a"]
        assert core["following_prefill_rows"]["row_axis"] == "head"
        first = core["following_prefill_rows"]["groups"][0]
        assert first["token"] == 0 and first["row_element_counts"] == [128] + [0] * 31
        padding = 2 * 32 * 128 if case["name"] == "gdn_prefill" else 0
        assert all(row["padding_matching_elements"] == padding for row in core["fills"])
        assert len(core["following_prefill_rows"]["groups"]) == (6 if padding else 1)
        assert state["dtype"] == "float32" and state["element_bytes"] == 4
        assert state["complete_elements_following_prefill"] == 128
        assert [row["valid_matching_elements"] for row in state["fills"]] == [128, 128, 129]
        assert state["fills"][-1]["sentinel_word"] == "0x5a5a5a5a"
        assert state["following_prefill_rows"]["row_axis"] == "key_row"
        head = state["following_prefill_rows"]["groups"][3]
        assert head["head"] == 3 and head["row_element_counts"] == [0] * 4 + [128] + [0] * 123
    source = result["installed_source"]["files"][0]
    assert source["status"] == "VERIFIED_COPY"
    assert source["sha256"] == retained["captured"]["files"][0]["sha256"]
    contexts = {item["match_line"]: item for item in source["contexts"]}
    assert 5 in contexts and any("SetAtomicAdd" in line["text"] for line in contexts[5]["lines"])
    assert 7 in contexts and any("memset" in line["text"] for line in contexts[7]["lines"])
    assert _snapshot(root, probe, previous) == before


@pytest.mark.parametrize("tamper", ["manifest", "native", "output", "source-input", "bundled-input", "installed-copy"])
def test_retained_summary_rejects_changed_evidence_before_writing(retained, tmp_path, tamper):
    root, probe = retained["root"], retained["probe"]
    case = retained["manifest"]["cases"][0]
    native_path = Path(retained["report"]["cases"][0]["runs"][0]["result"])
    native = json.loads(native_path.read_text())
    step_path = probe / case["bundle_manifest"]["path"]
    step = json.loads(step_path.read_text())
    paths = {"manifest": probe / "manifest.json", "native": native_path,
             "output": native_path.parent / native["outputs"][0]["path"],
             "source-input": probe / case["inputs"][0]["path"],
             "bundled-input": step_path.parent / step["inputs"][0]["path"],
             "installed-copy": root / "installed" / retained["captured"]["files"][0]["path"]}
    with paths[tamper].open("ab") as stream:
        stream.write(b"changed")
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="SHA256|size mismatch"):
        diagnostic.summarize_retained_diagnosis(root, output)
    assert not output.exists()


@pytest.mark.parametrize("tamper", ["incomplete", "duplicate-fill", "wrong-result-source", "input-readback", "output-descriptor"])
def test_retained_summary_rejects_incomplete_or_inconsistent_reports(retained, tmp_path, tamper):
    root, report = retained["root"], retained["report"]
    if tamper == "incomplete":
        report["status"] = "FAILED"
    elif tamper == "duplicate-fill":
        report["cases"][0]["runs"][1]["fill"] = 0
    else:
        trial = report["cases"][0]["runs"][0]
        path = Path(trial["result"])
        native = json.loads(path.read_text())
        if tamper == "wrong-result-source":
            native["model_sha256"] = "0" * 64
        elif tamper == "input-readback":
            native["io_diagnostic"]["inputs"][0]["readback_sha256"] = "0" * 64
        else:
            native["outputs"][0]["dtype"] = "float32"
        _write(path, native)
        trial["result_sha256"] = sha256_file(path)
    _write(root / "report.json", report)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError):
        diagnostic.summarize_retained_diagnosis(root, output)
    assert not output.exists()


def test_retained_summary_cannot_write_into_any_retained_input(retained):
    for base in (retained["root"], retained["probe"],
                 Path(retained["report"]["cases"][0]["runs"][0]["result"]).parent):
        output = base / "forbidden-summary"
        with pytest.raises(ValueError, match="separate from every retained"):
            diagnostic.summarize_retained_diagnosis(retained["root"], output)
        assert not output.exists()


@pytest.mark.parametrize("has_critical", [True, False])
def test_source_summary_prioritizes_middle_atomic_and_prints_key_statements(retained, tmp_path, has_critical):
    root = retained["root"]
    inventory_path = root / "installed/report.json"
    inventory = json.loads(inventory_path.read_text())
    record = inventory["files"][0]
    source = root / "installed" / record["path"]
    middle = ["InitOutput(outGM, length, 0);", "SetAtomicAdd<float>();", "AtomicAdd(output, local);",
              "Duplicate(local, 0, length);", "memset(output, 0, bytes);"] if has_critical else []
    lines = [f"DataCopy(out[{i}], local, length);" for i in range(40)]
    lines += middle
    lines += [f"DataCopy(out[{i}], local, length);" for i in range(40, 80)]
    source.write_text("\n".join(lines) + "\n")
    record.update(bytes=source.stat().st_size, sha256=sha256_file(source), total_lines=len(lines))
    _write(inventory_path, inventory)
    output = tmp_path / "summary"
    result = diagnostic.summarize_retained_diagnosis(root, output)
    captured = result["installed_source"]["files"][0]
    contexts = captured["contexts"]
    assert captured["contexts_limited"] and len(contexts) == 24
    assert captured["critical_matched_lines"] == len(middle)
    assert not captured["critical_contexts_limited"]
    assert contexts[0]["match_line"] == 1 and contexts[-1]["match_line"] == len(lines)
    text = (output / "report.txt").read_text()
    if has_critical:
        critical = [context for context in contexts if context["critical"]]
        assert [context["match_line"] for context in critical] == list(range(41, 46))
        assert [context["match_text"] for context in critical] == middle
        for index, statement in enumerate(middle[:4], 41):
            assert f"L{index}: {statement}" in text
        assert sum(line.startswith("    L") for line in text.splitlines()) == 4
        assert "L45: memset" not in text  # Full selected context remains in JSON.
    else:
        assert not any(context["critical"] for context in contexts)
        assert "No explicit initialization/atomic match in this captured source." in text
    assert "Matches are clues, not proof" in text
