"""Stateful GE construction/export contracts; real GE and devices are absent."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from qwen35_moe_dflash.artifacts import checked_file
from qwen35_moe_dflash.ge_export import prepare_pair, weight_identity
from qwen35_moe_dflash.ge_fused.export import export_target, plan_target, selected_fusion


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    path = tmp_path / "prepared"
    prepare_pair(path, tiny=True)
    return path


def _gdn_binding(path, recurrent_dtype="float16", final_dtype=None):
    inputs = {name: {"port": name, "dtype": "float16"}
              for name in ("query", "key", "value", "g", "beta", "initial_state")}
    inputs["effective_length"] = {"port": "effective_length", "dtype": "int16"}
    inputs["initial_state"]["dtype"] = recurrent_dtype
    doc = {"schema_version": 1, "target": "test-only", "ddk_id": "mock-schema",
           "source": {"path": "test_fixture.py", "sha256": "0" * 64},
           "operators": {"gdn": {"op_type": "TestOnlyGDN", "layout": "ND", "inputs": inputs,
                                  "outputs": {name: {"port": name, "dtype": (final_dtype or recurrent_dtype) if name == "final_state" else "float16"}
                                              for name in ("core_attn", "final_state")},
                                  "attrs": {name: {"name": name, "type": dtype} for name, dtype in
                                            (("chunk_size", "int"), ("output_final_state", "bool"),
                                             ("use_qk_l2norm_in_kernel", "bool"))}}}}
    path.write_text(json.dumps(doc))
    return path


@pytest.mark.parametrize("roles", ["all", "gdn,gdn", "none,gdn", "attention", "", ["gdn", "bad"]])
def test_selection_rejects_unknown_ambiguous_roles(roles):
    with pytest.raises(ValueError, match="fused_ops"):
        selected_fusion(roles)


def test_enabled_binding_required_before_prepared_or_output_read(tmp_path):
    with pytest.raises(ValueError, match="require --bindings"):
        plan_target(tmp_path / "missing", tmp_path / "output", fused_ops="gdn")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("sequence,past,roles,recurrent_dtype", [
    (3, 0, "none", "float16"), (1, 3, "none", "float16"),
    (2, 3, "gdn", "float16"), (1, 3, "gdn", "float32")])
def test_source_plan_contains_state_and_retains_inputs(prepared, tmp_path, sequence, past, roles, recurrent_dtype):
    original = (prepared / "prepared.json").read_bytes()
    binding = None if roles == "none" else _gdn_binding(tmp_path / "binding.json", recurrent_dtype)
    result = plan_target(prepared, tmp_path / "source", sequence_length=sequence, past_length=past,
                         fused_ops=roles, bindings_path=binding, fp16_reductions="matmul",
                         fp16_row_mul="compact", gdn_indexed_copy="gather", recurrent_dtype=recurrent_dtype)
    assert result["artifact_kind"] == "qwen35-stateful-ge-source"
    assert result["status"] == "SOURCE_READY" and result["air"] is None
    assert result["export_root"] == str(tmp_path / "source")
    assert result["device_validation"] == "PENDING" and not result["legacy_hiai_abi_compatible"]
    assert result["state_abi"]["next_past_length"] == sequence + past
    assert result["state_abi"]["input_order"][0] == "input_ids"
    assert len(result["state_abi"]["input_order"]) == (1 if past == 0 else 9)
    assert len(result["state_abi"]["output_order"]) == 10
    assert result["target_fp16_payload_sha256"] == weight_identity(result["weights"])
    assert all(key.startswith("target.") for key in result["weights"])
    for item in result["construction_files"] + list(result["weights"].values()):
        checked_file(tmp_path / "source", item)
    assert (prepared / "prepared.json").read_bytes() == original
    assert not list((tmp_path / "source").rglob("*.air"))
    if binding:
        checked_file(tmp_path / "source", result["bindings"])
        assert result["fused_ops"] == ["gdn"]
        assert len(result["graph"]["custom_ops"]) == 3
        assert {node["type"] for node in result["graph"]["custom_ops"]} == {"TestOnlyGDN"}
        for node in result["graph"]["custom_ops"]:
            assert node["outputs"]["final_state"]["dtype"] == recurrent_dtype
    else:
        assert result["bindings"] is None and result["graph"]["custom_ops"] == []
    with pytest.raises(FileExistsError):
        plan_target(prepared, tmp_path / "source")


def test_tampered_prepared_payload_rejected_before_output(prepared, tmp_path):
    manifest = json.loads((prepared / "prepared.json").read_text())
    weight = prepared / next(iter(manifest["weights"].values()))["path"]
    data = bytearray(weight.read_bytes())
    data[0] ^= 1
    weight.write_bytes(data)
    with pytest.raises(ValueError, match="SHA256"):
        plan_target(prepared, tmp_path / "source")
    assert not (tmp_path / "source").exists()


def test_plan_cannot_load_ge_or_quantized_payload(prepared, tmp_path):
    with pytest.raises(ValueError, match="does not accept"):
        plan_target(prepared, tmp_path / "source", ge_root="/unused")
    path = prepared / "prepared.json"
    manifest = json.loads(path.read_text())
    manifest["quantization"] = {"bits": 4}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unquantized FP16"):
        plan_target(prepared, tmp_path / "source")
    assert not (tmp_path / "source").exists()


def test_mixed_gdn_state_ports_require_explicit_cast_policy(prepared, tmp_path):
    binding = _gdn_binding(tmp_path / "binding.json", final_dtype="float32")
    with pytest.raises(ValueError, match="gdn_state_casts|state.*dtype|state.*cast"):
        plan_target(prepared, tmp_path / "rejected", sequence_length=1, past_length=3,
                    fused_ops="gdn", bindings_path=binding)
    assert not (tmp_path / "rejected").exists()
    result = plan_target(prepared, tmp_path / "source", sequence_length=1, past_length=3,
                         fused_ops="gdn", bindings_path=binding, gdn_state_casts="allow")
    assert result["gdn_state_casts"] == "allow" and result["recurrent_dtype"] == "float16"
    for node in result["graph"]["custom_ops"]:
        assert node["inputs"]["initial_state"]["dtype"] == "float16"
        assert node["outputs"]["final_state"]["dtype"] == "float32"
    recurrent = [item for item in result["state_abi"]["states"] if item["role"] == "recurrent"]
    assert len(recurrent) == 3
    assert {item["dtype"] for item in recurrent} == {"float16"}


def test_export_serializes_only_target_with_retained_tool_evidence(prepared, tmp_path, monkeypatch):
    ge = tmp_path / "ge"
    (ge / "include/graph").mkdir(parents=True)
    (ge / "lib64").mkdir()
    for name in ("graph.h", "operator.h"):
        (ge / "include/graph" / name).write_text("MOCK HEADER")
    (ge / "lib64/libgraph.so").write_bytes(b"MOCK LIBGRAPH")
    calls = []
    def tool(command, **kwargs):
        calls.append(command)
        if "-o" in command:
            Path(command[command.index("-o") + 1]).write_bytes(b"MOCK SERIALIZER")
        else:
            Path(command[1]).write_bytes(b"MOCK AIR")
        kwargs["stdout"].write("mock boundary, no GE used\n")
    monkeypatch.setattr(subprocess, "run", tool)
    result = export_target(prepared, tmp_path / "air", ge)
    assert len(calls) == 2
    assert result["artifact_kind"] == "qwen35-stateful-ge-air" and result["status"] == "EXPORTED"
    assert result["export_root"] == str(tmp_path / "air")
    assert result["device_validation"] == "PENDING" and not result["legacy_hiai_abi_compatible"]
    checked_file(tmp_path / "air", result["air"])
    assert (tmp_path / "air/target/build-command.json").is_file()
    assert not (tmp_path / "air/draft").exists()
    from qwen35_moe_dflash.compile import compile_air
    with pytest.raises(ValueError):
        compile_air(tmp_path / "air", tmp_path / "invalid-omc", tmp_path / "ddk", ddk_id="mock")


def test_cli_source_path_does_not_import_torch(prepared, tmp_path):
    repo = Path(__file__).resolve().parents[1]
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    for module in ("torch", "torch_npu", "torchair", "transformers"):
        (blocked / f"{module}.py").write_text("raise AssertionError('GE path imported framework')\n")
    env = dict(os.environ, AI_RUN_DIR=str(tmp_path), PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=os.pathsep.join((str(blocked), str(repo / "model"))))
    result = subprocess.run([sys.executable, str(repo / "targets/cpro/scripts/direct_model_air.py"),
                             "plan-stateful", "--prepared", str(prepared), "--sequence-length", "1",
                             "--past-length", "3", "--fused-ops", "none",
                             "--output", str(tmp_path / "source")],
                            env=env, cwd=tmp_path, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "SOURCE_READY"


def test_lowering_failure_retains_stage_without_success_manifest(prepared, tmp_path, monkeypatch):
    from qwen35_moe_dflash.ge_fused import emitter
    def fail(*args, **kwargs):
        raise ValueError("mock unsupported lowering")
    monkeypatch.setattr(emitter, "emit_cpp", fail)
    with pytest.raises(ValueError, match="unsupported lowering"):
        plan_target(prepared, tmp_path / "source")
    failure = json.loads((tmp_path / "source/failure.json").read_text())
    assert failure["stage"] == "target-lowering" and failure["status"] == "FAILED"
    assert not (tmp_path / "source/manifest.json").exists()


@pytest.fixture
def stateful_air(prepared, tmp_path, monkeypatch, request):
    """Mock only host tools; all construction/provenance checks remain real."""
    ge = tmp_path / "ge"
    (ge / "include/graph").mkdir(parents=True)
    (ge / "lib64").mkdir()
    for name in ("graph.h", "operator.h"):
        (ge / "include/graph" / name).write_text("MOCK HEADER")
    (ge / "lib64/libgraph.so").write_bytes(b"MOCK LIBGRAPH")
    def export_tool(command, **kwargs):
        path = command[command.index("-o") + 1] if "-o" in command else command[1]
        Path(path).write_bytes(b"MOCK HOST EXPORT")
        kwargs["stdout"].write("mock host export, no GE/device execution\n")
    monkeypatch.setattr(subprocess, "run", export_tool)
    options = getattr(request, "param", {})
    if options.get("external"):
        prepared_manifest = json.loads((prepared / "prepared.json").read_text())
        prepared_manifest["external_weights"] = True
        (prepared / "prepared.json").write_text(json.dumps(prepared_manifest))
    binding = _gdn_binding(tmp_path / "binding.json")
    if options.get("all_fused"):
        document = json.loads(binding.read_text())
        document["operators"]["rms"] = {
            "op_type": "TestOnlyRms", "layout": "ND",
            "inputs": {"x": {"port": "self", "dtype": "float16"}, "gamma": {"port": "gamma", "dtype": "float16"}},
            "outputs": {"y": {"port": "y", "dtype": "float16"}, "rstd": {"port": "rstd", "dtype": "float32"}},
            "attrs": {"epsilon": {"name": "epsilon", "type": "float"}}}
        document["operators"]["conv_state"] = {
            "op_type": "TestOnlyConvState", "layout": "ND",
            "inputs": {"projected": {"port": "x1", "dtype": "float16"},
                       "history": {"port": "x2", "dtype": "float16"},
                       "update_control": {"port": "x3", "dtype": "int32"}},
            "outputs": {"history": {"port": "y1", "dtype": "float16"}, "padded": {"port": "y2", "dtype": "float16"}},
            "attrs": {}, "history_output_width": "kernel", "history_keep": "padded_tail"}
        binding.write_text(json.dumps(document))
    roles = "conv_state,gdn,rms" if options.get("all_fused") else "gdn"
    if options.get("unfused"):
        roles, binding = "none", None
    # Both past cache inputs and fused multi-output port ordinals are exercised.
    manifest = export_target(prepared, tmp_path / "air", ge, sequence_length=1, past_length=options.get("past", 3),
                             fused_ops=roles, bindings_path=binding)
    ddk = tmp_path / "ddk"
    master = ddk / "tools/tools_omg/master"
    master.mkdir(parents=True)
    (master / "omg").write_bytes(b"MOCK OMG")
    (master / "omg").chmod(0o700)
    calls = []
    def compile_tool(command, **kwargs):
        calls.append((command, kwargs))
        assert Path(kwargs["cwd"]).is_relative_to(tmp_path)
        assert (Path(kwargs["cwd"]) / "input.air").read_bytes() == b"MOCK HOST EXPORT"
        (Path(kwargs["cwd"]) / "target.omc").write_bytes(b"MOCK OMC")
        (Path(kwargs["cwd"]) / "target.data").mkdir()
        (Path(kwargs["cwd"]) / "target.data/weights.bin").write_bytes(b"MOCK SIDECAR")
        kwargs["stdout"].write("mock compiler, no device execution\n")
    monkeypatch.setattr(subprocess, "run", compile_tool)
    return tmp_path / "air", manifest, ddk, calls


def test_compile_stateful_preserves_explicit_contract_and_payloads(stateful_air, tmp_path):
    from qwen35_moe_dflash.compile import compile_stateful_air
    air, original, ddk, calls = stateful_air
    result = compile_stateful_air(air, tmp_path / "omc", ddk, ddk_id="mock-schema")
    assert len(calls) == 1 and "--platform=kirinx90" in calls[0][0]
    assert result["artifact_kind"] == "qwen35-stateful-omc" and result["status"] == "COMPILED"
    assert result["runtime"] == "hiai" and result["state_policy"] == "explicit"
    assert result["legacy_hiai_abi_compatible"] is False
    assert result["device_validation"] == "PENDING" and result["runtime_placement"] == "UNVERIFIED"
    for field in ("source", "state_abi", "graph", "binding_contract", "target_fp16_payload_sha256"):
        assert result[field] == original[field]
    graph = result["graphs"][0]
    assert graph["inputs"] == original["graph"]["inputs"]
    assert graph["outputs"] == original["graph"]["outputs"]
    assert graph["state_abi"] == original["state_abi"]
    assert result["air_provenance"]["weights"] == original["weights"]
    assert result["air_source_root"] == str(air)
    assert "target/target.data/weights.bin" in {p["path"] for p in graph["payloads"]}
    for record in graph["payloads"]:
        checked_file(tmp_path / "omc", record)


@pytest.mark.parametrize("stateful_air", [{"external": True, "all_fused": True, "past": 0},
                                          {"unfused": True}], indirect=True)
def test_compile_stateful_supports_external_weights_all_fusion_and_unfused(stateful_air, tmp_path):
    from qwen35_moe_dflash.compile import compile_stateful_air
    air, manifest, ddk, calls = stateful_air
    result = compile_stateful_air(air, tmp_path / "omc", ddk, ddk_id="mock-schema")
    assert result["status"] == "COMPILED" and len(calls) == 1
    assert result["fused_ops"] == manifest["fused_ops"]


@pytest.mark.parametrize("change", ["source_ready", "moved", "air", "construction", "weight", "bindings",
                                     "abi", "graph", "weight_identity", "ddk"])
def test_compile_stateful_rejects_unbound_source_before_tool(stateful_air, tmp_path, change):
    from qwen35_moe_dflash.compile import compile_stateful_air
    air, manifest, ddk, calls = stateful_air
    identity = "mock-schema"
    if change == "source_ready":
        manifest.update(status="SOURCE_READY", artifact_kind="qwen35-stateful-ge-source", air=None)
    elif change == "moved":
        shutil.copytree(air, tmp_path / "moved")
        air = tmp_path / "moved"
    elif change in {"air", "construction", "weight", "bindings"}:
        record = {"air": manifest["air"], "construction": manifest["construction_files"][0],
                  "weight": next(iter(manifest["weights"].values())), "bindings": manifest["bindings"]}[change]
        path = air / record["path"]
        path.write_bytes(path.read_bytes() + b"tampered")
    elif change == "abi":
        manifest["state_abi"]["input_order"].reverse()
    elif change == "graph":
        manifest["graph"]["outputs"].reverse()
    elif change == "weight_identity":
        manifest["target_fp16_payload_sha256"] = "f" * 64
    elif change == "ddk":
        identity = "other-ddk"
    (air / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        compile_stateful_air(air, tmp_path / "omc", ddk, ddk_id=identity)
    assert calls == [] and not (tmp_path / "omc").exists()


def test_compile_stateful_rejects_mutation_by_compiler(stateful_air, tmp_path, monkeypatch):
    from qwen35_moe_dflash.compile import compile_stateful_air
    air, _, ddk, _ = stateful_air
    def mutate(command, **kwargs):
        (Path(kwargs["cwd"]) / "target.omc").write_bytes(b"MOCK OMC")
        (air / "target/target.air").write_bytes(b"changed during compile")
    monkeypatch.setattr(subprocess, "run", mutate)
    with pytest.raises(ValueError, match="mismatch"):
        compile_stateful_air(air, tmp_path / "omc", ddk, ddk_id="mock-schema")
    failure = json.loads((tmp_path / "omc/failure.json").read_text())
    assert failure["stage"] == "target-source-check"
    assert not (tmp_path / "omc/manifest.json").exists()


def test_compile_stateful_rejects_rehashed_recipe_output_reordering(stateful_air, tmp_path):
    from qwen35_moe_dflash.compile import compile_stateful_air
    from qwen35_moe_dflash.hashing import sha256_file
    air, manifest, ddk, calls = stateful_air
    record = next(item for item in manifest["construction_files"] if item["path"].endswith("graph.recipe"))
    path = air / record["path"]
    payload = bytearray(path.read_bytes())
    # The final recipe table holds ordered (operator index, output port) pairs.
    # Rehashing corrupted metadata must not bypass graph/ABI consistency checks.
    start = len(payload) - 8 * len(manifest["graph"]["outputs"])
    first, second = payload[start:start + 8], payload[start + 8:start + 16]
    payload[start:start + 16] = second + first
    path.write_bytes(payload)
    record["sha256"] = sha256_file(path)
    (air / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="recipe boundary order"):
        compile_stateful_air(air, tmp_path / "omc", ddk, ddk_id="mock-schema")
    assert calls == [] and not (tmp_path / "omc").exists()


@pytest.mark.parametrize("failure_kind", ["exit", "missing", "rollback"])
def test_compile_stateful_retains_failures_without_success(stateful_air, tmp_path, monkeypatch, failure_kind):
    from qwen35_moe_dflash.compile import compile_stateful_air
    air, _, ddk, _ = stateful_air
    def failing(command, **kwargs):
        scratch = Path(kwargs["cwd"])
        if failure_kind == "exit":
            (scratch / "check_result.json").write_text('{"mock": true}')
            kwargs["stdout"].write("E/OMG_TOOL mock rejected operator\n")
            raise subprocess.CalledProcessError(7, command)
        if failure_kind == "rollback":
            (scratch / "target.omc").write_bytes(b"MOCK CPU OMC")
            kwargs["stdout"].write('W/AI_FMK (1): "Partition of type: NPUCL is more than 9," " will rollback to CPU"\n')
    monkeypatch.setattr(subprocess, "run", failing)
    with pytest.raises(RuntimeError):
        compile_stateful_air(air, tmp_path / "omc", ddk, ddk_id="mock-schema", reject_cpu_fallback=True)
    failure = json.loads((tmp_path / "omc/failure.json").read_text())
    assert failure["status"] == "FAILED" and not (tmp_path / "omc/manifest.json").exists()
    if failure_kind == "exit":
        assert failure["returncode"] == 7
        assert (tmp_path / "omc/target/diagnostics/check_result.json").is_file()
    elif failure_kind == "rollback":
        assert failure["stage"] == "target-placement-check" and failure["cpu_fallback_rejections"]
        assert (tmp_path / "omc/target/target.omc").is_file()


def test_compile_stateful_cli_dispatch(stateful_air, tmp_path, monkeypatch):
    import runpy
    air, _, ddk, calls = stateful_air
    script = Path(__file__).resolve().parents[1] / "targets/cpro/scripts/direct_model_air.py"
    monkeypatch.setattr(sys, "argv", [str(script), "compile-stateful", "--air", str(air),
                                     "--ddk-root", str(ddk), "--ddk-id", "mock-schema",
                                     "--output", str(tmp_path / "omc"), "--reject-cpu-fallback"])
    with pytest.raises(SystemExit) as completed:
        runpy.run_path(str(script), run_name="__main__")
    assert completed.value.code == 0 and len(calls) == 1
    assert json.loads((tmp_path / "omc/manifest.json").read_text())["reject_cpu_fallback"] is True


@pytest.mark.parametrize("stateful_air", [{"past": 0}], indirect=True)
class TestPrepareStatefulStep:
    """Mock tool output; real hashes, ABI checks and continuation staging."""

    def compile(self, stateful_air, tmp_path):
        from qwen35_moe_dflash.compile import compile_stateful_air
        air, _, ddk, _ = stateful_air
        compile_stateful_air(air, tmp_path / "omc", ddk, ddk_id="mock-schema")
        return tmp_path / "omc"

    def decode(self, prepared, tmp_path, monkeypatch, *, past=1, policy="native"):
        from qwen35_moe_dflash.compile import compile_stateful_air
        def mock_tool(command, **kwargs):
            if "--target=omc" in command:
                scratch = Path(kwargs["cwd"])
                (scratch / "target.omc").write_bytes(b"MOCK DECODE OMC")
            else:
                path = command[command.index("-o") + 1] if "-o" in command else command[1]
                Path(path).write_bytes(b"MOCK DECODE EXPORT")
        with monkeypatch.context() as context:
            context.setattr(subprocess, "run", mock_tool)
            export_target(prepared, tmp_path / "decode-air", tmp_path / "ge", sequence_length=1,
                          past_length=past, fused_ops="gdn", bindings_path=tmp_path / "binding.json",
                          fp16_reductions=policy)
            compile_stateful_air(tmp_path / "decode-air", tmp_path / "decode-omc", tmp_path / "ddk", ddk_id="mock-schema")
        return tmp_path / "decode-omc"

    def result(self, path, step, *, evidence="device-hiai"):
        import math
        from qwen35_moe_dflash.hashing import sha256_file
        path.mkdir()
        outputs = []
        for index, desc in enumerate(step["outputs"]):
            item = path / f"output_{index:04d}.bin"
            item.write_bytes(bytes([index + 1]) * math.prod(desc["shape"]) * (2 if desc["dtype"] == "float16" else 4))
            outputs.append({"index": index, **desc, "path": item.name,
                            "bytes": item.stat().st_size, "sha256": sha256_file(item)})
        result = {"schema_version": 1, "status": "EXECUTED", "scope": "stateful-target-step",
                  "evidence_kind": evidence, "runtime_version": "TEST ONLY", "state_policy": "explicit",
                  "target_calls": 1, "draft_calls": 0, "ddk_id": step["identity"]["ddk_id"],
                  "step_plan_sha256": step["step_plan"]["sha256"], "state_abi_sha256": step["state_abi"]["sha256"],
                  "model_sha256": step["model"]["sha256"], "sequence_length": step["sequence_length"],
                  "past_length": step["past_length"], "next_past_length": step["next_past_length"],
                  "inputs": step["inputs"], "outputs": outputs}
        (path / "result.json").write_text(json.dumps(result))
        return result

    def test_prefill_wire_payloads_and_cli(self, stateful_air, tmp_path, monkeypatch):
        import runpy
        from qwen35_moe_dflash.ge_fused.export import _checked_step
        omc = self.compile(stateful_air, tmp_path)
        script = Path(__file__).resolve().parents[1] / "targets/cpro/scripts/direct_model_air.py"
        monkeypatch.setattr(sys, "argv", [str(script), "prepare-stateful-step", "--omc", str(omc),
                                         "--output", str(tmp_path / "step"), "--token-ids", "1"])
        with pytest.raises(SystemExit) as completed:
            runpy.run_path(str(script), run_name="__main__")
        assert completed.value.code == 0
        step, _, _ = _checked_step(tmp_path / "step")
        assert len(step["inputs"]) == 1 and step["past_length"] == 0
        assert (tmp_path / "step/inputs/input_0000.bin").read_bytes() == b"\1\0\0\0"
        assert (tmp_path / "step/model/target/target.data/weights.bin").read_bytes() == b"MOCK SIDECAR"
        lines = (tmp_path / "step/step.tsv").read_text().splitlines()
        assert lines[0] == "QWEN35_STATEFUL_STEP\t1" and lines[3] == "tokens\t1\t0"
        assert lines[4].startswith("input\t0\tinput_ids\tint32\t1,1\tinputs/input_0000.bin\t")
        assert lines[5] == "output\t0\ttarget_top1\tint32\t1,1"

    def test_continuation_copies_all_state_and_preserves_source(self, stateful_air, prepared, tmp_path, monkeypatch):
        from qwen35_moe_dflash.ge_fused.export import prepare_stateful_step, _checked_step
        omc = self.compile(stateful_air, tmp_path)
        step = prepare_stateful_step(omc, tmp_path / "step", [1])
        result = self.result(tmp_path / "result", step)
        decode = self.decode(prepared, tmp_path, monkeypatch)
        next_step = prepare_stateful_step(decode, tmp_path / "next-step", [2], tmp_path / "step", tmp_path / "result")
        assert next_step["past_length"] == 1 and next_step["next_past_length"] == 2
        assert len(next_step["inputs"]) == len(step["outputs"]) - 1
        for input_item, output_item in zip(next_step["inputs"][1:], result["outputs"][2:]):
            assert input_item["name"] == output_item["name"].removesuffix("_out")
            assert input_item["sha256"] == output_item["sha256"]
        _checked_step(tmp_path / "next-step")

    @pytest.mark.parametrize("tokens", [[], [1, 2], [True], [-1], [2 ** 31], [1.0]])
    def test_rejects_invalid_tokens(self, stateful_air, tmp_path, tokens):
        from qwen35_moe_dflash.ge_fused.export import prepare_stateful_step
        omc = self.compile(stateful_air, tmp_path)
        with pytest.raises(ValueError, match="token_ids"):
            prepare_stateful_step(omc, tmp_path / "step", tokens)
        assert not (tmp_path / "step").exists()

    @pytest.mark.parametrize("change", ["payload", "omc-contract", "air-manifest", "payload-symlink", "source-ready"])
    def test_rejects_changed_compiled_source(self, stateful_air, tmp_path, change):
        from qwen35_moe_dflash.ge_fused.export import prepare_stateful_step
        omc = self.compile(stateful_air, tmp_path)
        manifest = json.loads((omc / "manifest.json").read_text())
        if change == "payload":
            (omc / "target/target.omc").write_bytes(b"WRONG")
        elif change == "omc-contract":
            manifest["state_abi"]["states"][0]["output_shape"][-1] += 1
        elif change == "air-manifest":
            with (tmp_path / "air/manifest.json").open("a") as handle:
                handle.write(" ")
        elif change == "payload-symlink":
            payload = omc / "target/target.omc"
            payload.rename(omc / "saved.omc")
            payload.symlink_to(omc / "saved.omc")
        else:
            manifest["status"] = "SOURCE_READY"
        (omc / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(ValueError):
            prepare_stateful_step(omc, tmp_path / "step", [1])
        assert not (tmp_path / "step").exists()

    @pytest.mark.parametrize("change", ["status", "plan", "abi", "model", "past", "count", "dtype", "shape", "hash",
                                         "output-symlink", "nonstate-output", "input-hash", "step-tsv", "step-manifest", "host-fake",
                                         "ddk", "scope", "schema"])
    def test_rejects_unbound_previous_execution(self, stateful_air, prepared, tmp_path, monkeypatch, change):
        from qwen35_moe_dflash.ge_fused.export import prepare_stateful_step
        omc = self.compile(stateful_air, tmp_path)
        step = prepare_stateful_step(omc, tmp_path / "step", [1])
        result = self.result(tmp_path / "result", step)
        decode = self.decode(prepared, tmp_path, monkeypatch)
        if change == "status": result["status"] = "FAILED"
        elif change == "plan": result["step_plan_sha256"] = "0" * 64
        elif change == "abi": result["state_abi_sha256"] = "0" * 64
        elif change == "model": result["model_sha256"] = "0" * 64
        elif change == "past": result["next_past_length"] = 2
        elif change == "count": result["outputs"].pop()
        elif change == "dtype": result["outputs"][2]["dtype"] = "float32"
        elif change == "shape": result["outputs"][2]["shape"][-1] += 1
        elif change == "hash": result["outputs"][2]["sha256"] = "0" * 64
        elif change == "output-symlink":
            payload = tmp_path / "result" / result["outputs"][2]["path"]
            payload.rename(tmp_path / "result/saved.bin")
            payload.symlink_to(tmp_path / "result/saved.bin")
        elif change == "nonstate-output":
            (tmp_path / "result" / result["outputs"][0]["path"]).write_bytes(b"BAD!")
        elif change == "input-hash": result["inputs"][0]["sha256"] = "0" * 64
        elif change == "step-tsv":
            with (tmp_path / "step/step.tsv").open("a") as handle: handle.write("extra\n")
        elif change == "step-manifest":
            step["identity"]["ddk_id"] = "another-ddk"
            (tmp_path / "step/manifest.json").write_text(json.dumps(step))
        elif change == "host-fake": result["evidence_kind"] = "host-fake-hiai"
        elif change == "ddk": result["ddk_id"] = "another-ddk"
        elif change == "scope": result["scope"] = "legacy-pair"
        elif change == "schema": result["schema_version"] = 2
        (tmp_path / "result/result.json").write_text(json.dumps(result))
        with pytest.raises(ValueError):
            prepare_stateful_step(decode, tmp_path / "next-step", [2], tmp_path / "step", tmp_path / "result")
        assert not (tmp_path / "next-step").exists()

    def test_fake_requires_explicit_allow(self, stateful_air, prepared, tmp_path, monkeypatch):
        from qwen35_moe_dflash.ge_fused.export import prepare_stateful_step
        step = prepare_stateful_step(self.compile(stateful_air, tmp_path), tmp_path / "step", [1])
        self.result(tmp_path / "result", step, evidence="host-fake-hiai")
        decode = self.decode(prepared, tmp_path, monkeypatch)
        result = prepare_stateful_step(decode, tmp_path / "next-step", [2], tmp_path / "step", tmp_path / "result",
                                       allow_host_fake=True)
        assert result["continuation"]["evidence_kind"] == "host-fake-hiai"
        assert result["host_fake_ancestry"] is True

    def test_requires_complete_continuation_and_new_output(self, stateful_air, prepared, tmp_path, monkeypatch):
        from qwen35_moe_dflash.ge_fused.export import prepare_stateful_step
        omc = self.compile(stateful_air, tmp_path)
        step = prepare_stateful_step(omc, tmp_path / "step", [1])
        self.result(tmp_path / "result", step)
        with pytest.raises(FileExistsError):
            prepare_stateful_step(omc, tmp_path / "step", [2])
        with pytest.raises(ValueError, match="starts fresh"):
            prepare_stateful_step(omc, tmp_path / "invalid", [1], tmp_path / "step", tmp_path / "result")
        decode = self.decode(prepared, tmp_path, monkeypatch)
        for previous_step, previous_result in ((None, None), (tmp_path / "step", None), (None, tmp_path / "result")):
            with pytest.raises(ValueError, match="requires both"):
                prepare_stateful_step(decode, tmp_path / "invalid", [2], previous_step, previous_result)
        assert not (tmp_path / "invalid").exists()

    @pytest.mark.parametrize("past,policy", [(2, "native"), (1, "matmul")])
    def test_rejects_length_or_arithmetic_change(self, stateful_air, prepared, tmp_path, monkeypatch, past, policy):
        from qwen35_moe_dflash.ge_fused.export import prepare_stateful_step
        step = prepare_stateful_step(self.compile(stateful_air, tmp_path), tmp_path / "step", [1])
        self.result(tmp_path / "result", step)
        decode = self.decode(prepared, tmp_path, monkeypatch, past=past, policy=policy)
        with pytest.raises(ValueError):
            prepare_stateful_step(decode, tmp_path / "next-step", [2], tmp_path / "step", tmp_path / "result")
        assert not (tmp_path / "next-step").exists()
