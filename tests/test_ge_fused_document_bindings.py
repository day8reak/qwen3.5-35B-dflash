"""Document-derived GE interfaces; no claim that a toolkit or kernel is installed."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from qwen35_moe_dflash.config import Qwen35DFlashConfig
from qwen35_moe_dflash.ge_export import configurations
from qwen35_moe_dflash.ge_fused.bindings import OperatorBindings
from qwen35_moe_dflash.ge_fused import bindings as module
from qwen35_moe_dflash.ge_fused.emitter import emit_cpp
from qwen35_moe_dflash.ge_fused.ir import Graph
from qwen35_moe_dflash.ge_fused.recipe import read_recipe
from qwen35_moe_dflash.ge_fused.target import build_target


REPO = Path(__file__).resolve().parents[1]
CLI = REPO / "targets/cpro/scripts/direct_model_air.py"


def make(dtype="float16"):
    return module.document_bindings(target="declared-test-target", ddk_id="declared-test-ddk",
                                    gdn_initial_state_dtype=dtype)


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_document_contract_exact_ports_dtypes_and_flag_encoding(dtype):
    doc = make(dtype)
    assert doc["source"]["sha256"] == hashlib.sha256((REPO / doc["source"]["path"]).read_bytes()).hexdigest()
    ops = doc["operators"]
    assert list(ops) == ["gdn", "conv_state", "rms"]
    gdn = ops["gdn"]
    assert gdn["op_type"] == "ChunkGatedDeltaRule"
    assert list(gdn["inputs"]) == ["query", "key", "value", "g", "beta", "initial_state", "effective_length"]
    assert [p["port"] for p in gdn["inputs"].values()] == list(gdn["inputs"])
    assert gdn["inputs"]["initial_state"]["dtype"] == dtype
    assert gdn["inputs"]["effective_length"] == {"port": "effective_length", "dtype": "int16", "shape": [1]}
    assert [p["port"] for p in gdn["outputs"].values()] == ["core_attn", "last_recurrent_state"]
    assert [p["dtype"] for p in gdn["outputs"].values()] == ["float16", "float32"]
    assert list(gdn["attrs"]) == ["chunk_size", "output_final_state", "use_qk_l2norm_in_kernel"]
    assert all(attr["type"] == "int" for attr in gdn["attrs"].values())
    conv = ops["conv_state"]
    assert [p["port"] for p in conv["inputs"].values()] == ["x1", "x2", "x3"]
    assert [p["port"] for p in conv["outputs"].values()] == ["y1", "y2"]
    assert conv["attrs"] == {} and conv["history_keep"] == "padded_tail"
    rms = ops["rms"]
    assert [p["port"] for p in rms["inputs"].values()] == ["self", "gamma"]
    assert [p["dtype"] for p in rms["outputs"].values()] == ["float16", "float32"]
    assert rms["attrs"] == {"epsilon": {"name": "epsilon", "type": "float"}}


def test_bound_gdn_serializes_named_final_state_and_int_flags(tmp_path):
    bindings = OperatorBindings(make("float32"))
    graph = Graph("documented_gdn")
    inputs = {name: graph.input(name, tuple(2 if dim is None else dim for dim in item["shape"]), item["dtype"])
              for name, item in bindings.manifest()["operators"]["gdn"]["inputs"].items()}
    outputs = bindings.call(graph, "gdn", inputs=inputs,
                            outputs={"core_attn": ((1, 2, 32, 128), "float16"),
                                     "final_state": ((1, 32, 128, 128), "float32")},
                            attrs={"chunk_size": 64, "output_final_state": True,
                                   "use_qk_l2norm_in_kernel": True})
    graph.outputs = [outputs["core_attn"], outputs["final_state"]]
    emit_cpp(graph, {}, recipe_dir=tmp_path / "recipe")
    recipe = read_recipe(tmp_path / "recipe/graph.recipe")
    op = next(node for node in recipe["records"] if node["type"] == "ChunkGatedDeltaRule")
    assert [port["port"] for port in op["outputs"]] == ["core_attn", "last_recurrent_state"]
    assert op["attrs"]["output_final_state"] == {"type": "int", "value": 1}
    assert op["attrs"]["use_qk_l2norm_in_kernel"] == {"type": "int", "value": 1}


@pytest.mark.parametrize("role", ["gdn", "conv_state"])
def test_documented_full_model_shapes_do_not_claim_tiny_support(role):
    config, draft = configurations(tiny=True)
    with pytest.raises(ValueError, match="shape"):
        build_target(config, Qwen35DFlashConfig.from_dict(draft), 3,
                     fused_ops=(role,), bindings=OperatorBindings(make("float32")),
                     recurrent_dtype="float32" if role == "gdn" else "float16")


def test_generator_requires_explicit_state_dtype_and_nonempty_identity():
    with pytest.raises(TypeError):
        module.document_bindings(target="target", ddk_id="ddk")
    with pytest.raises(ValueError, match="explicitly select"):
        make("int32")
    with pytest.raises(ValueError, match="target"):
        module.document_bindings(target="", ddk_id="ddk", gdn_initial_state_dtype="float16")


def test_changed_source_section_rejected_before_binding_write(tmp_path, monkeypatch):
    source = tmp_path / "changed.md"
    source.write_bytes(module._DOCUMENT_PATH.read_bytes().replace(b"last_recurrent_state", b"wrong_state"))
    monkeypatch.setattr(module, "_DOCUMENT_PATH", source)
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    output = tmp_path / "out/bindings.json"
    with pytest.raises(ValueError, match="section changed"):
        module.write_document_bindings(output, target="target", ddk_id="ddk", gdn_initial_state_dtype="float16")
    assert not output.exists()


def _cli(tmp_path, *arguments):
    env = dict(os.environ, AI_RUN_DIR=str(tmp_path), PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, str(CLI), "bindings-from-document", *map(str, arguments)],
                          cwd=tmp_path, env=env, text=True, capture_output=True)


def test_cli_retains_document_provenance_and_refuses_overwrite(tmp_path):
    output = tmp_path / "generated/bindings.json"
    args = ["--output", output, "--target", "test-target", "--ddk-id", "test-ddk",
            "--gdn-initial-state-dtype", "float16"]
    result = _cli(tmp_path, *args)
    assert result.returncode == 0, result.stderr
    status = json.loads(result.stdout)
    assert status["status"] == "BINDINGS_FROM_USER_DOCUMENT"
    assert status["source_platform"] == "Ascend310P3"
    assert status["kernel_installation"] == status["target_compatibility"] == "UNVERIFIED"
    assert status["device_validation"] == "NOT_RUN"
    metadata = json.loads(Path(status["metadata"]).read_text())
    assert metadata["binding_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    bound = OperatorBindings.from_file(output)
    assert bound.manifest()["operators"]["conv_state"]["history_keep"] == "padded_tail"
    before = output.read_bytes()
    again = _cli(tmp_path, *args)
    assert again.returncode != 0 and "already exists" in again.stderr
    assert output.read_bytes() == before


def test_cli_rejects_path_outside_run_and_missing_dtype(tmp_path):
    output = tmp_path.parent / (tmp_path.name + "-outside.json")
    args = ["--output", output, "--target", "test-target", "--ddk-id", "test-ddk"]
    missing = _cli(tmp_path, *args)
    assert missing.returncode != 0 and "--gdn-initial-state-dtype" in missing.stderr
    outside = _cli(tmp_path, *args, "--gdn-initial-state-dtype", "float32")
    assert outside.returncode != 0 and "below AI_RUN_DIR" in outside.stderr
    assert not output.exists()


def test_document_binding_generation_does_not_import_torch(tmp_path):
    source = """
import sys
class Block:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'torch_npu', 'torchair'}:
            raise RuntimeError('unexpected framework import: ' + fullname)
sys.meta_path.insert(0, Block())
from qwen35_moe_dflash.ge_fused.bindings import document_bindings
document_bindings(target='test',ddk_id='test',gdn_initial_state_dtype='float32')
"""
    env = dict(os.environ, PYTHONPATH=str(REPO / "model"), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", source], env=env, cwd=tmp_path,
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


_INSTALLED_STATE = """
GlobalTensor<half> isGlobal;
GlobalTensor<half> lsGlobal;
isGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(initial_state), stateElements);
lsGlobal.SetGlobalBuffer(
    reinterpret_cast<__gm__ half *>(last_recurrent_state), stateElements);
"""


def test_installed_state_override_pins_composite_source_without_changing_document(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    source = tmp_path / "installed/chunk_gated_delta_rule.cpp"
    source.parent.mkdir()
    source.write_text(_INSTALLED_STATE)
    before = source.read_bytes(), module._DOCUMENT_PATH.read_bytes()
    output = tmp_path / "generated/bindings.json"
    result = module.write_document_bindings(output, target="kirinx90", ddk_id="declared-ddk",
                                            gdn_initial_state_dtype="float16", gdn_state_source=source)
    binding = OperatorBindings.from_file(output).manifest()
    assert result["status"] == "BINDINGS_WITH_INSTALLED_GDN_STATE"
    assert result["gdn_final_state_dtype"] == "float16"
    assert result["kernel_installation"] == result["target_compatibility"] == "UNVERIFIED"
    assert result["device_validation"] == "NOT_RUN"
    assert binding["operators"]["gdn"]["outputs"]["final_state"]["dtype"] == "float16"
    assert binding["operators"]["conv_state"] == make()["operators"]["conv_state"]
    evidence_path = Path(binding["source"]["path"])
    assert binding["source"]["sha256"] == hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    evidence = json.loads(evidence_path.read_text())
    assert evidence["document"] == make()["source"]
    assert evidence["installed_gdn_state"]["sha256"] == hashlib.sha256(before[0]).hexdigest()
    assert set(evidence["installed_gdn_state"]["matches"]) == {"initial_state", "last_recurrent_state"}
    assert evidence["installed_gdn_state"]["verification_kind"] == "BOUNDED_SOURCE_PATTERN"
    assert evidence["compile_validation"] == evidence["device_validation"] == "NOT_RUN"
    assert source.read_bytes() == before[0] and module._DOCUMENT_PATH.read_bytes() == before[1]
    assert make()["operators"]["gdn"]["outputs"]["final_state"]["dtype"] == "float32"
    with pytest.raises(FileExistsError):
        module.write_document_bindings(output, target="kirinx90", ddk_id="declared-ddk",
                                       gdn_initial_state_dtype="float16", gdn_state_source=source)


@pytest.mark.parametrize("source_text", [
    "// " + _INSTALLED_STATE.replace("\n", "\n// "),
    "/*" + _INSTALLED_STATE + "*/",
    _INSTALLED_STATE.replace("last_recurrent_state", "some_other_output"),
    _INSTALLED_STATE.replace("__gm__ half *", "__gm__ float *"),
    _INSTALLED_STATE.replace("GlobalTensor<half> lsGlobal", "GlobalTensor<float> lsGlobal"),
    _INSTALLED_STATE + "\nGlobalTensor<float> other;\nother.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(last_recurrent_state));\n",
    _INSTALLED_STATE.replace("(initial_state)", "(initial_state + offset)"),
])
def test_installed_state_override_rejects_missing_or_contradictory_evidence(tmp_path, monkeypatch, source_text):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    source = tmp_path / "chunk_gated_delta_rule.cpp"
    source.write_text(source_text)
    output = tmp_path / "generated/bindings.json"
    with pytest.raises(ValueError, match="GDN"):
        module.write_document_bindings(output, target="kirinx90", ddk_id="declared-ddk",
                                       gdn_initial_state_dtype="float16", gdn_state_source=source)
    assert not output.exists() and not output.with_suffix(".sources.json").exists()


def test_installed_state_override_requires_matching_input_dtype(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    source = tmp_path / "chunk_gated_delta_rule.cpp"
    source.write_text(_INSTALLED_STATE)
    with pytest.raises(ValueError, match="requires.*float16"):
        module.write_document_bindings(tmp_path / "out.json", target="kirinx90", ddk_id="declared-ddk",
                                       gdn_initial_state_dtype="float32", gdn_state_source=source)
    assert not (tmp_path / "out.json").exists()
