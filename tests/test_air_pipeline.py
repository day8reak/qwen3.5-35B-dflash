"""AIR orchestration fault tests; the dummy payload is not AIR device evidence."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from reference.tiny import create
from qwen35_moe_dflash.export import export_air, require_torchair
from qwen35_moe_dflash.compile import compile_air


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
if os.environ.get('FAKE_OMG_FAIL')=='yes': sys.exit(9)
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


def test_air_external_weight_tamper_fails_before_compiler(air_case, tmp_path):
    root, ddk = air_case
    (root/"target/weight_0").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA256"):
        compile_air(root, tmp_path/"omc", ddk, ddk_id="fake")
    assert not (tmp_path/"omc").exists()


def test_omg_failure_leaves_no_success_manifest_and_cleans_scratch(air_case, tmp_path, monkeypatch):
    root, ddk = air_case
    monkeypatch.setenv("FAKE_OMG_FAIL", "yes")
    import subprocess
    with pytest.raises(subprocess.CalledProcessError):
        compile_air(root, tmp_path/"omc", ddk, ddk_id="fake")
    assert not (tmp_path/"omc/manifest.json").exists()
    assert json.loads((tmp_path/"omc/failure.json").read_text())["status"] == "FAILED"
    assert not list((tmp_path/"omc").glob("omg-*"))


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
