"""AIR orchestration fault tests; the dummy payload is not AIR device evidence."""
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import pytest
from reference.tiny import create
from qwen35_moe_dflash.export import export_air, require_torchair
from qwen35_moe_dflash.compile import compile_air, omg_environment


@pytest.fixture(autouse=True)
def isolate_graph_preparation(monkeypatch):
    # Real graph lowering is covered by test_precision.py. These tests exercise
    # payload identity, converter failures and cleanup, never model numerics.
    def prepare(t, d, length, **kwargs):
        t.half(); d.half()
        return t, d, {k: {"status": "PASS", "kind": "TEST_DOUBLE", "quantized_matmuls": []}
                      for k in ("target", "draft")}
    monkeypatch.setattr("qwen35_moe_dflash.export.prepare_fp16_pair", prepare)


@pytest.fixture
def air_case(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    calls = []
    def dynamo_export(*args, model, export_path, export_name, dynamic):
        calls.append((export_name, dynamic, [tuple(x.shape) for x in args]))
        folder = Path(export_path)
        assert Path.cwd() == folder
        (folder / f"{export_name}.air").write_bytes(b"TEST DOUBLE -- NOT AIR")
        (folder / "weight_0").write_bytes(b"external weight fixture")
    target, draft = create()
    root = tmp_path / "air"
    manifest = export_air(target, draft, root, 8,
        torchair_module=SimpleNamespace(dynamo_export=dynamo_export, __version__="FAKE"),
        source_identity={"kind": "tiny-test-double"})
    assert calls == [("target", False, [(1, 8), (1, 8)]),
                     ("draft", False, [(1, 8, 32), (1, 8), (1, 8)])]
    assert manifest["pipeline"] == ["pytorch", "air", "omg", "omc", "hiai"]
    ddk = tmp_path / "ddk"
    binary = ddk / "tools/tools_omg/master/omg"
    binary.parent.mkdir(parents=True)
    (ddk / "tools/platform/kirinx90").mkdir(parents=True)
    binary.write_text('''#!/usr/bin/env python3
import os, sys
from pathlib import Path
args=dict(a.split('=',1) for a in sys.argv[1:])
assert args['--framework']=='1'
assert args['--target']=='omc' and args['--platform']=='kirinx90'
assert args['--use_origin_format']=='true' and args['--weight_data_type']=='FP16'
assert args['--model']=='input.air' and '/' not in args['--output']
assert Path('weight_0').read_bytes()==b'external weight fixture'
assert os.environ['SOC_VERSION']=='kirinx90'
if os.environ.get('FAKE_OMG_REQUIRE_INTERNAL_ENV')=='yes':
    import internal_ddk_extension
    internal_ddk_extension.check()
Path('kernel_meta').mkdir()
if (os.environ.get('FAKE_OMG_FAIL')=='yes'
        and args['--output']==os.environ.get('FAKE_OMG_FAIL_GRAPH', 'target')):
    print('[ERROR] mock converter first cause')
    for _ in range(300):
        print('[DEBUG] ParseJsonFormatString:add error_code E68059 success')
    sys.exit(9)
Path(args['--output']+'.omc').write_bytes(b'FAKE OMC')
''')
    binary.chmod(0o700)
    return root, ddk


def test_air_omg_success_retains_external_hashes_and_cleans_scratch(air_case, tmp_path):
    root, ddk = air_case
    result = compile_air(root, tmp_path/"omc", ddk, ddk_id="fake")
    assert result["status"] == "COMPILED" and result["device_validation"] == "PENDING"
    assert [g["name"] for g in result["graphs"]] == ["target", "draft"]
    assert not list((tmp_path/"omc").glob("omg-*"))


def test_omg_child_keeps_sourced_ddk_extensions(air_case, tmp_path, monkeypatch):
    import os
    root, ddk = air_case
    extension = tmp_path / "internal-python"
    extension.mkdir()
    plugin_ops = ddk / "tools/platform/kirinx90/ops/impl"
    plugin_ops.mkdir(parents=True)
    (plugin_ops / "platform_marker.py").write_text("IDENTITY = 'selected-x90'\n")
    (extension / "platform_marker.py").write_text("IDENTITY = 'inherited-other'\n")
    simulator = str(ddk / "ddk/npu_simulator")
    simulator_config = str(ddk / "ddk/npu_simulator/model/model_v300/etc")
    (extension / "internal_ddk_extension.py").write_text(
        "import os, platform_marker\n"
        "def check():\n"
        "    assert platform_marker.IDENTITY == 'selected-x90'\n"
        "    paths = os.environ['LD_LIBRARY_PATH'].split(os.pathsep)\n"
        f"    assert paths[0] == {str(ddk / 'tools/tools_omg/master/lib64')!r}\n"
        f"    assert {simulator!r} in paths and {simulator_config!r} in paths\n")
    monkeypatch.setenv("PYTHONPATH", str(extension))
    monkeypatch.setenv("LD_LIBRARY_PATH", os.pathsep.join([simulator, simulator_config]))
    monkeypatch.setenv("FAKE_OMG_REQUIRE_INTERNAL_ENV", "yes")
    before = dict(os.environ)
    result = compile_air(root, tmp_path / "internal-omc", ddk, ddk_id="fake-internal")
    assert result["status"] == "COMPILED"
    assert dict(os.environ) == before


def test_omg_internal_layout_without_public_plugin(air_case, tmp_path, monkeypatch):
    """Run a converter subprocess using only the internal DDK's sourced paths."""
    import os
    root, ddk = air_case
    plugin = ddk / "tools/platform/kirinx90"
    plugin.rmdir()
    plugin.parent.rmdir()
    extension = tmp_path / "internal-python"
    extension.mkdir()
    simulator = str(ddk / "ddk/npu_simulator")
    simulator_config = str(ddk / "ddk/npu_simulator/model/model_v300/etc")
    kirin_config = str(tmp_path / "internal-platform-config")
    (extension / "internal_ddk_extension.py").write_text(
        "import os\n"
        "def check():\n"
        "    paths = os.environ['LD_LIBRARY_PATH'].split(os.pathsep)\n"
        f"    assert paths == {[str(ddk / 'tools/tools_omg/master/lib64'), simulator, simulator_config]!r}\n"
        f"    assert os.environ['PYTHONPATH'] == {str(extension)!r}\n"
        f"    assert os.environ['KIRIN_CONF_JSON_PATH'] == {kirin_config!r}\n")
    monkeypatch.setenv("PYTHONPATH", str(extension))
    monkeypatch.setenv("LD_LIBRARY_PATH", os.pathsep.join([simulator, simulator_config]))
    monkeypatch.setenv("KIRIN_CONF_JSON_PATH", kirin_config)
    monkeypatch.setenv("FAKE_OMG_REQUIRE_INTERNAL_ENV", "yes")
    before = dict(os.environ)
    result = compile_air(root, tmp_path / "internal-omc", ddk, ddk_id="fake-internal")
    assert result["status"] == "COMPILED"
    assert result["device_validation"] == "PENDING"
    assert dict(os.environ) == before
    assert not (ddk / "tools/platform").exists()


def test_omg_optional_platform_paths_prefer_existing_selected_ddk(air_case):
    import os
    _, ddk = air_case
    inherited = {"PATH": "/existing/bin", "LD_LIBRARY_PATH": "/existing/lib"}
    _, env = omg_environment(ddk, inherited)
    assert env["LD_LIBRARY_PATH"].split(os.pathsep) == [str(ddk / "tools/tools_omg/master/lib64"), "/existing/lib"]
    assert "PYTHONPATH" not in env
    assert "KIRIN_CONF_JSON_PATH" not in env

    plugin_lib = ddk / "tools/platform/kirinx90/lib64"
    plugin_ops = ddk / "tools/platform/kirinx90/ops/impl"
    config = ddk / "ddk/ascendcpp/conf"
    for path in (plugin_lib, plugin_ops, config): path.mkdir(parents=True)
    inherited.update(PYTHONPATH="/existing/python", KIRIN_CONF_JSON_PATH="/existing/conf")
    _, env = omg_environment(ddk, inherited)
    assert env["LD_LIBRARY_PATH"].split(os.pathsep) == [str(ddk / "tools/tools_omg/master/lib64"), str(plugin_lib), "/existing/lib"]
    assert env["PYTHONPATH"].split(os.pathsep) == [str(plugin_ops), "/existing/python"]
    assert env["KIRIN_CONF_JSON_PATH"] == str(config)
    assert inherited == {"PATH": "/existing/bin", "LD_LIBRARY_PATH": "/existing/lib",
                         "PYTHONPATH": "/existing/python", "KIRIN_CONF_JSON_PATH": "/existing/conf"}


def test_air_external_weight_tamper_fails_before_compiler(air_case, tmp_path):
    root, ddk = air_case
    (root/"target/weight_0").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA256"):
        compile_air(root, tmp_path/"omc", ddk, ddk_id="fake")
    assert not (tmp_path/"omc").exists()


@pytest.mark.parametrize("failed_graph", ["target", "draft"])
def test_omg_failure_leaves_no_success_manifest_and_cleans_scratch(
        air_case, tmp_path, monkeypatch, failed_graph):
    root, ddk = air_case
    monkeypatch.setenv("FAKE_OMG_FAIL", "yes")
    monkeypatch.setenv("FAKE_OMG_FAIL_GRAPH", failed_graph)
    with pytest.raises(RuntimeError, match="mock converter first cause") as caught:
        compile_air(root, tmp_path/"omc", ddk, ddk_id="fake")
    assert isinstance(caught.value.__cause__, subprocess.CalledProcessError)
    output = tmp_path / "omc"
    log = output / failed_graph / "omg.log"
    assert not (output / "manifest.json").exists()
    failure = json.loads((output / "failure.json").read_text())
    assert failure["status"] == "FAILED" and failure["failed_graph"] == failed_graph
    assert failure["stage"] == f"{failed_graph}-omg"
    assert failure["log_path"] == str(log) and failure["returncode"] == 9
    assert "mock converter first cause" in log.read_text()
    assert f"Full log: {log}" in str(caught.value)
    assert "First error messages:\nL1: [ERROR] mock converter first cause" in str(caught.value)
    assert len(str(caught.value)) < 16000
    assert not list(output.glob("omg-*"))
    assert [item["name"] for item in failure["graphs"]] == (["target"] if failed_graph == "draft" else [])
    if failed_graph == "draft":
        assert (output / "target/target.omc").read_bytes() == b"FAKE OMC"
    else:
        assert not (output / "draft").exists()


@pytest.mark.parametrize("failure_kind", ["timeout", "launch", "missing", "empty"])
@pytest.mark.parametrize("failed_graph", ["target", "draft"])
def test_omg_diagnostics_cover_timeout_launch_and_incomplete_output(
        air_case, tmp_path, monkeypatch, failure_kind, failed_graph):
    root, ddk = air_case
    timeout = 19
    expected_exception = {"timeout": subprocess.TimeoutExpired, "launch": OSError,
                          "missing": RuntimeError, "empty": RuntimeError}[failure_kind]

    def tool(command, **kwargs):
        name = next(arg.split("=", 1)[1] for arg in command if arg.startswith("--output="))
        scratch = Path(kwargs["cwd"])
        (scratch / "kernel_meta").mkdir()
        if name != failed_graph:
            (scratch / f"{name}.omc").write_bytes(b"FAKE OMC")
            return subprocess.CompletedProcess(command, 0)
        kwargs["stdout"].write("[ERROR] mock converter diagnostic before failure\n")
        kwargs["stdout"].write("[DEBUG] add error_code E68059 success\n" * 300)
        if failure_kind == "timeout":
            raise subprocess.TimeoutExpired(command, timeout)
        if failure_kind == "launch":
            raise OSError("mock executable could not start")
        if failure_kind == "empty":
            (scratch / f"{name}.omc").write_bytes(b"")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", tool)
    output = tmp_path / "omc-diagnostic"
    with pytest.raises(RuntimeError, match="mock converter diagnostic before failure") as caught:
        compile_air(root, output, ddk, ddk_id="fake", timeout_seconds=timeout)
    assert isinstance(caught.value.__cause__, expected_exception)
    report = json.loads((output / "failure.json").read_text())
    log = output / failed_graph / "omg.log"
    assert report["status"] == "FAILED" and report["failed_graph"] == failed_graph
    assert report["stage"] == f"{failed_graph}-" + (
        "output-check" if failure_kind in ("missing", "empty") else "omg")
    assert report["log_path"] == str(log) and report["returncode"] is None
    if failure_kind == "timeout":
        assert report["timeout_seconds"] == timeout
    assert "mock converter diagnostic before failure" in log.read_text()
    assert f"Full log: {log}" in str(caught.value)
    assert "First error messages:\nL1: [ERROR] mock converter diagnostic before failure" in str(caught.value)
    assert not (output / "manifest.json").exists()
    assert not list(output.glob("omg-*"))
    assert [item["name"] for item in report["graphs"]] == (["target"] if failed_graph == "draft" else [])
    if failed_graph == "draft":
        assert (output / "target/target.omc").read_bytes() == b"FAKE OMC"
    else:
        assert not (output / "draft").exists()


def test_air_missing_dependency_is_explicit(monkeypatch):
    import qwen35_moe_dflash.export as module
    def missing(name): raise ImportError(name)
    monkeypatch.setattr(module.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="TorchAir/torch_npu/CANN"):
        require_torchair()


def test_export_failure_retains_reason(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    def fail(*args, **kwargs): raise RuntimeError("unsupported GE op")
    target, draft = create()
    with pytest.raises(RuntimeError, match="unsupported GE op"):
        export_air(target, draft, tmp_path/"air", 8, torchair_module=SimpleNamespace(dynamo_export=fail))
    assert not (tmp_path/"air/manifest.json").exists()
    assert "unsupported GE op" in json.loads((tmp_path/"air/failure.json").read_text())["error"]


def test_reject_dense_fake_quant_before_export(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    def fail(*args, **kwargs): raise AssertionError("export must not start")
    target, draft = create()
    with pytest.raises(ValueError, match="native packed matmul"):
        export_air(target, draft, tmp_path/"air", 8,
                   quantization={"matrix_count": 1},
                   torchair_module=SimpleNamespace(dynamo_export=fail))
    assert not (tmp_path/"air/manifest.json").exists()


def test_old_abi_rejected_before_omg(air_case, tmp_path):
    root, ddk = air_case
    manifest = json.loads((root/"manifest.json").read_text())
    manifest["abi_version"] = 1
    (root/"manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="ABI v2"):
        compile_air(root, tmp_path/"omc", ddk, ddk_id="fake")


def test_tiny_board_command_keeps_token_62_and_matches_demo(air_case, tmp_path, monkeypatch):
    """Exercise the stop policy through demo -> AIR/OMC metadata -> board argv.

    Constant predictions isolate the EOS regression; this is host evidence only.
    """
    from qwen35_moe_dflash.cli import execute, parser
    from qwen35_moe_dflash.runtime import generate

    class Constant62:
        sequence_length, draft_width, vocab_size, device = 8, 3, 64, "cpu"

        def __init__(self, *args):
            pass

        def target(self, prefix):
            return [62] * self.sequence_length

        def draft(self):
            return [62] * self.draft_width

    monkeypatch.setattr("qwen35_moe_dflash.runtime.TorchExecutor", Constant62)
    monkeypatch.setattr("qwen35_moe_dflash.precision.prepare_fp16_pair",
                        lambda t, d, *args, **kwargs: (t, d, {"kind": "TEST_DOUBLE"}))
    reference_path = tmp_path / "tiny-reference.json"
    execute(parser().parse_args(["demo", "--sequence-length", "8", "--max-new-tokens", "4",
                                "--output", str(reference_path)]))
    reference = json.loads(reference_path.read_text())
    root, ddk = air_case
    compiled = tmp_path / "omc"
    compile_air(root, compiled, ddk, ddk_id="fake")
    prompt_path = tmp_path / "tiny-prompt.json"
    prompt_path.write_text(json.dumps({"prompt_token_ids": reference["prompt_token_ids"]}))
    command_path = tmp_path / "board-command.json"
    execute(parser().parse_args([
        "board-command", "--omc", str(compiled), "--prompt", str(prompt_path),
        "--board-model-dir", "/board/tiny", "--board-output", "/board/report.json",
        "--runtime-version", "fake", "--firmware-id", "fake", "--mode", "benchmark",
        "--max-new-tokens", "4", "--output", str(command_path)]))
    import shlex
    command = json.loads(command_path.read_text())
    argv = shlex.split(command["shell"])
    assert argv == command["argv"]  # Empty EOS must survive shell quoting.
    eos = [int(t) for t in argv[argv.index("--eos-token-ids") + 1].split(",") if t]
    for ordinary in (True, False):
        result = generate(Constant62(), reference["prompt_token_ids"], max_new_tokens=4,
                          eos_token_ids=eos, ordinary=ordinary)
        assert result.token_ids == reference["ordinary"]["token_ids"] == [62] * 4
        assert result.stop_reason == reference["ordinary"]["stop_reason"] == "length"
