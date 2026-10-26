"""Compiler evidence retention using a fake OMG; no placement/device claims."""
import json
import os
from pathlib import Path
import sys

import pytest

from qwen35_moe_dflash.compile import (
    _compiled_payloads, _retain_compiler_diagnostics, compile_air,
)
from qwen35_moe_dflash.hashing import sha256_file


def _bundle(root):
    root.mkdir()
    keys = ("sequence_length", "draft_width", "feature_width", "vocab_size", "precision", "state_policy",
            "expert_dispatch", "source", "pad_token_id", "eos_token_ids", "precision_audit", "quantization")
    manifest = dict.fromkeys(keys)
    manifest.update(artifact_kind="qwen35-cpro-air-pair", status="EXPORTED", runtime="hiai",
                    abi_version=2, feature_dtype="torch.float16", export_root=str(root), graphs=[])
    for name in ("target", "draft"):
        air = root / f"{name}.air"
        air.write_bytes(b"TEST DOUBLE, NOT AIR")
        manifest["graphs"].append({"name": name, "air": air.name,
            "files": [{"path": air.name, "bytes": air.stat().st_size, "sha256": sha256_file(air)}],
            "inputs": ["ids"], "outputs": ["output"], "input_shapes": [[1, 8]],
            "input_dtypes": ["torch.int32"], "output_shapes": [[1, 8]], "output_dtypes": ["torch.int32"]})
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


@pytest.fixture
def fake_omg_case(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    binary = tmp_path / "ddk/tools/tools_omg/master/omg"
    binary.parent.mkdir(parents=True)
    binary.write_text(f"#!{sys.executable}\n" + '''import json, os, sys
from pathlib import Path
args = dict(arg.split('=', 1) for arg in sys.argv[1:])
name = args['--output']
dump = Path('dump_graph')
dump.mkdir()
(dump / '05_after_Partition.irpb').write_bytes(b'partition test fixture')
(dump / '00_cpucl_SubGraph_0.irpb').write_bytes(b'CPU subgraph test fixture')
(dump / '06_input_symlink.irpb').symlink_to(Path('input.air').resolve())
with (dump / '01_large.irpb').open('wb') as stream:
    stream.truncate((16 << 20) + 1)
Path('check_result.json').write_text(json.dumps({'note': 'pre-check candidates, not final placement'}))
Path('kernel_meta').mkdir()
Path('kernel_meta/private.bin').write_bytes(b'not a model sidecar')
print('subGraph fixture run by cl: CPUCL -- TEST DOUBLE')
if os.environ.get('FAKE_PARTITION_GRAPH', name) == name:
    print(os.environ.get('FAKE_PARTITION_LOG', ''))
if os.environ.get('FAKE_DIAGNOSTIC_OMG_FAILURE'):
    (dump / '02_fail.irpb').write_bytes(b'failure test fixture')
    print('[ERROR] primary test converter failure')
    sys.exit(7)
Path(name + '.omc').write_bytes(b'FAKE OMC')
sidecars = Path(name + '.sidecars')
sidecars.mkdir()
(sidecars / 'weight_0').write_bytes(b'preserved compiled sidecar')
''')
    binary.chmod(0o700)
    return _bundle(tmp_path / "air"), tmp_path / "ddk"


def test_success_retains_reports_and_partition_dumps_off_board_payloads(fake_omg_case, tmp_path):
    bundle, ddk = fake_omg_case
    output = tmp_path / "omc"
    result = compile_air(bundle, output, ddk, ddk_id="fake-retention")
    assert result["status"] == "COMPILED" and result["device_validation"] == "PENDING"
    assert not list(output.glob("omg-*"))
    for graph in result["graphs"]:
        folder = output / graph["name"]
        diagnostics = folder / "diagnostics"
        record = json.loads((diagnostics / "manifest.json").read_text())
        assert record["compile_outcome"] == "success"
        assert record["status"] == graph["compiler_diagnostics"]["status"] == "PARTIAL"
        assert "not final placement" in record["scope"]
        files = {row["name"]: row for row in record["files"]}
        for name in ("check_result.json", "05_after_Partition.irpb", "00_cpucl_SubGraph_0.irpb"):
            assert files[name]["status"] == "RETAINED"
            assert files[name]["sha256"] == sha256_file(diagnostics / name)
        assert files["01_large.irpb"]["status"] == "SKIPPED_SIZE_BUDGET"
        assert not (diagnostics / "01_large.irpb").exists()
        assert "06_input_symlink.irpb" not in files
        assert record["retained_bytes"] <= 16 << 20
        payloads = [row["path"] for row in graph["payloads"]]
        assert graph["omc"] in payloads
        assert f"{graph['name']}/{graph['name']}.sidecars/weight_0" in payloads
        assert not any("diagnostics" in Path(path).parts for path in payloads)
        assert not any("kernel_meta" in path or path.endswith(".air") for path in payloads)
        assert "subGraph fixture run by cl: CPUCL" in (folder / "omg.log").read_text()


def test_failure_retains_diagnostics_without_masking_primary_error(fake_omg_case, tmp_path, monkeypatch):
    bundle, ddk = fake_omg_case
    monkeypatch.setenv("FAKE_DIAGNOSTIC_OMG_FAILURE", "1")
    with pytest.raises(RuntimeError) as caught:
        compile_air(bundle, tmp_path / "omc", ddk, ddk_id="fake-retention")
    assert caught.value.__cause__.returncode == 7
    assert "primary test converter failure" in str(caught.value)
    folder = tmp_path / "omc/target/diagnostics"
    record = json.loads((folder / "manifest.json").read_text())
    assert record["compile_outcome"] == "failure"
    assert record["files"][0]["name"] == "02_fail.irpb"
    assert (folder / "check_result.json").exists()
    assert not list((tmp_path / "omc").glob("omg-*"))


def test_success_diagnostic_copy_error_does_not_misreport_compile_failure(fake_omg_case, tmp_path, monkeypatch):
    bundle, ddk = fake_omg_case

    def no_space(*args):
        raise OSError("fake diagnostic disk full")

    monkeypatch.setattr("qwen35_moe_dflash.compile.shutil.copyfile", no_space)
    result = compile_air(bundle, tmp_path / "omc", ddk, ddk_id="fake-retention")
    assert result["status"] == "COMPILED"
    for graph in result["graphs"]:
        assert graph["compiler_diagnostics"] == {"status": "FAILED", "error": "fake diagnostic disk full"}
        assert not any("diagnostics" in Path(row["path"]).parts for row in graph["payloads"])


def test_success_priority_preserves_partition_before_intermediate(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    scratch = tmp_path / "scratch"
    dump = scratch / "dump_graph"
    dump.mkdir(parents=True)
    (dump / "00_intermediate.irpb").write_bytes(b"early")
    (dump / "05_after_Partition.irpb").write_bytes(b"final")
    (scratch / "check_result.json").write_bytes(b"{}")
    _retain_compiler_diagnostics(scratch, tmp_path / "out", max_bytes=7)
    destination = tmp_path / "out/diagnostics"
    assert (destination / "check_result.json").read_bytes() == b"{}"
    assert (destination / "05_after_Partition.irpb").read_bytes() == b"final"
    assert not (destination / "00_intermediate.irpb").exists()


def test_report_and_dump_symlinks_are_not_collected(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    secret = tmp_path / "secret"
    secret.mkdir()
    (secret / "private.irpb").write_bytes(b"unrelated")
    (scratch / "dump_graph").symlink_to(secret, target_is_directory=True)
    (scratch / "check_result.json").symlink_to(secret / "private.irpb")
    result = _retain_compiler_diagnostics(scratch, tmp_path / "out")
    assert result["status"] == "NOT_EMITTED"
    assert not (tmp_path / "out").exists()


def test_diagnostics_excluded_before_payload_walk_but_other_symlinks_rejected(tmp_path):
    graph = tmp_path / "target"
    graph.mkdir()
    (graph / "target.omc").write_bytes(b"model")
    private = tmp_path / "private"
    private.mkdir()
    (private / "secret").write_bytes(b"not payload")
    (graph / "diagnostics").symlink_to(private, target_is_directory=True)
    assert [row["path"] for row in _compiled_payloads(graph, tmp_path)] == ["target/target.omc"]
    (graph / "unexpected-link").symlink_to(private / "secret")
    with pytest.raises(ValueError, match="symlink"):
        _compiled_payloads(graph, tmp_path)


_ROLLBACK = ('W/AI_FMK (158555): partitioner_strategy.cpp ToPartitionerList(345)::'
             '"Partition of type: NPUCL is more than 9," " will rollback to CPU"')


@pytest.mark.parametrize("graph", ["target", "draft"])
def test_explicit_rollback_rejected_after_retaining_compiler_output(fake_omg_case, tmp_path, monkeypatch, graph):
    bundle, ddk = fake_omg_case
    monkeypatch.setenv("FAKE_PARTITION_GRAPH", graph)
    monkeypatch.setenv("FAKE_PARTITION_LOG", "TCH iCompiler->Compile begin ...\n" + _ROLLBACK)
    output = tmp_path / "omc"
    with pytest.raises(RuntimeError, match="NPU partition limit 9 exceeded"):
        compile_air(bundle, output, ddk, ddk_id="fake-retention", reject_cpu_fallback=True)
    failure = json.loads((output / "failure.json").read_text())
    assert failure["status"] == "FAILED" and failure["stage"] == f"{graph}-placement-check"
    assert failure["returncode"] is None  # OMG succeeded; our requested placement check failed.
    assert failure["cpu_fallback_rejections"][0]["limit"] == 9
    assert failure["graphs"][-1]["name"] == graph
    assert not (output / "manifest.json").exists()
    assert (output / graph / f"{graph}.omc").read_bytes() == b"FAKE OMC"
    assert (output / graph / "diagnostics/05_after_Partition.irpb").exists()
    if graph == "target":
        assert not (output / "draft").exists()


def test_default_records_and_warns_of_rollback_without_claiming_npu(fake_omg_case, tmp_path, monkeypatch, capsys):
    bundle, ddk = fake_omg_case
    monkeypatch.setenv("FAKE_PARTITION_LOG", _ROLLBACK)
    output = tmp_path / "omc"
    result = compile_air(bundle, output, ddk, ddk_id="fake-retention")
    assert result["status"] == "COMPILED" and result["device_validation"] == "PENDING"
    assert "rollback to CPU" in capsys.readouterr().err
    for graph in result["graphs"]:
        evidence = graph["compiler_evidence"]
        assert evidence["rollback_events"][0]["scope"] == "unknown"
        assert evidence["log_sha256"] == sha256_file(output / graph["name"] / "omg.log")
        assert evidence["runtime_placement"] == "UNVERIFIED"


def test_constant_graph_cpu_fallback_does_not_block_main_npu_compile(fake_omg_case, tmp_path, monkeypatch):
    bundle, ddk = fake_omg_case
    monkeypatch.setenv("FAKE_PARTITION_LOG", "\n".join([
        "IR InferShape for Graph constNodeGraph Success", _ROLLBACK,
        "TCH BuildOrigin2IRGraph end ...", "start InferShape for graph[target] ...",
        "partition type NPU:1, CPU:0, GPU:0, ISP:0."]))
    result = compile_air(bundle, tmp_path / "omc", ddk, ddk_id="fake-retention", reject_cpu_fallback=True)
    assert result["status"] == "COMPILED"
    assert all(g["compiler_evidence"]["rollback_events"][0]["scope"] == "constant_folding"
               for g in result["graphs"])


def test_compile_module_stays_standard_library_only():
    import subprocess
    result = subprocess.run([sys.executable, "-S", "-c",
        "from qwen35_moe_dflash.compile import compile_air; "
        "import sys; assert 'numpy' not in sys.modules and 'torch' not in sys.modules"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_direct_compile_cli_forwards_rollback_check(fake_omg_case, tmp_path, monkeypatch):
    import subprocess
    bundle, ddk = fake_omg_case
    monkeypatch.setenv("FAKE_PARTITION_LOG", _ROLLBACK)
    script = Path(__file__).resolve().parents[1] / "targets/cpro/scripts/direct_model_air.py"
    output = tmp_path / "cli-omc"
    result = subprocess.run([sys.executable, str(script), "compile", "--air", str(bundle),
        "--ddk-root", str(ddk), "--ddk-id", "test", "--output", str(output),
        "--reject-cpu-fallback"], capture_output=True, text=True)
    assert result.returncode == 1 and "NPU partition limit 9 exceeded" in result.stderr
    assert json.loads((output / "failure.json").read_text())["stage"] == "target-placement-check"
