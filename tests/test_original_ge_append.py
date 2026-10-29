"""Append coordinator evidence gates; native execution is explicit and bounded."""
import json
import os
from pathlib import Path

import pytest

from qwen35_moe_dflash import original_ge_append as append
from qwen35_moe_dflash.hashing import sha256_file
from qwen35_moe_dflash.omc_wire import _fields
from qwen35_moe_dflash.original_ge_target import SOURCE_SHA256
from test_draft_ir_inspect import _attr, _blob, _desc, _int
from test_original_air_inspect import _node, selected
from test_original_ge_taps import TEMPLATE, op, report_factory


def _padding(raw, *, first_path="./absent_weights/padding_0"):
    graph = next(bytes(value) for number, wire, value in _fields(raw, [2_000_000]) if number == 7)
    for index in range(1113):
        path = first_path if index == 0 else f"./absent_weights/padding_{index}"
        graph += _blob(6, _node(f"padding_{index}", "FileConstant", [2048], dtype=1)
                       + _attr("file_path", _blob(2, path)))
    return _blob(7, graph)


def _branch(sequence=55, *, half=7):
    nodes = []
    for index in range(8):
        data, cast = f"__dflash_features/input_{index}", f"__dflash_features/cast_{index}"
        nodes.append(_node(data, "Data", [1, sequence, 2048], dtype=1, index=index))
        nodes.append(op(cast, "Cast", [data + ":0"], [_desc(dtype=1, shape=[1, sequence, 2048])],
                        dtype=half, shape=[1, sequence, 2048]) + _attr("dst_type", _int(3, 1)))
    nodes.append(op("__dflash_features/concat", "ConcatD",
                    [f"__dflash_features/cast_{index}:0" for index in range(8)],
                    [_desc(dtype=half, shape=[1, sequence, 2048])] * 8,
                    dtype=half, shape=[1, sequence, 16384])
                 + _attr("concat_dim", _int(3, 2)) + _attr("N", _int(3, 8)))
    nodes.append(op("branch_net", "NetOutput", ["__dflash_features/concat:0"],
                    [_desc(dtype=half, shape=[1, sequence, 16384])], output_descs=[]))
    return _blob(7, _blob(1, "unit_branch") + b"".join(_blob(6, value) for value in nodes))


@pytest.fixture
def harness(tmp_path, monkeypatch, report_factory):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    report = report_factory()
    air = Path(report["air"]["path"])
    air.write_bytes(_padding(air.read_bytes()))
    schema = Path(report["schema"]["pb2"])
    receipt = tmp_path / "source.json"
    value = {"files": [{"path": name, "bytes": 100, "sha256": digest} for name, digest in SOURCE_SHA256.items()]}
    receipt.write_text(json.dumps(value))
    tool = tmp_path / "declared-tool"
    include, lib = tool / "include", tool / "lib64"
    for path in [include / "graph/graph.h", include / "graph/operator.h", include / "graph/types.h", lib / "libgraph.so", tool / "c++"]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"unit-tool-identity")
    calls = []
    hooks = {"branch": None, "roundtrip": None}
    def toolchain(*args):
        calls.append("toolchain")
        return tool, include, lib, str(tool / "c++")
    def logged(command, *, folder, env, log, timeout):
        log.write_text("unit native gate\n")
        if Path(command[0]).name == "c++":
            calls.append("build")
            Path(command[-1]).write_bytes(b"unit-native-binary")
        else:
            calls.append("branch")
            if hooks["branch"]:
                hooks["branch"]()
            Path(command[1]).write_bytes(_branch())
            Path(command[2]).write_text(json.dumps({"save_status": 0, "reload_status": 0, "model_execution": "NOT_RUN"}))
    def roundtrip(source, output, **options):
        calls.append("roundtrip")
        output = Path(output)
        output.mkdir()
        target = output / "roundtrip.air"
        target.write_bytes(Path(source).read_bytes())
        if hooks["roundtrip"]:
            hooks["roundtrip"](target)
        return {"status": "ROUNDTRIP_METADATA_PASSED", "source_preservation": "UNCHANGED_STAT_AND_SHA256",
                "roundtrip_air": {"path": str(target), "bytes": target.stat().st_size, "sha256": sha256_file(target)}}
    monkeypatch.setattr(append, "_toolchain", toolchain)
    monkeypatch.setattr(append, "_logged", logged)
    monkeypatch.setattr(append, "probe_original_air_roundtrip", roundtrip)
    def run(name="output", **options):
        return append.append_original_ge_features(air, tmp_path / name, torchair_pb2=schema,
            gamma_template=TEMPLATE, sequence_length=55, source_receipt_path=receipt, ge_root=tool, **options)
    return {"run": run, "air": air, "schema": schema, "receipt": receipt, "tool": tool,
            "tmp": tmp_path, "calls": calls, "hooks": hooks}


def test_complete_chain_reaches_publication_but_never_claims_compile_device_accuracy(harness, monkeypatch):
    payload = harness["air"].parent / "absent_weights/padding_0"
    payload.parent.mkdir()
    payload.write_bytes(b"PRIVATE_PAYLOAD_MUST_NOT_BE_READ_OR_COPIED")
    original_open = Path.open
    def guarded(path, *args, **kwargs):
        if Path(path).absolute() == payload.absolute():
            pytest.fail("metadata append must not open an external payload")
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    before = sha256_file(harness["air"])
    result = harness["run"]()
    assert result["status"] == "FEATURE_AIR_EXPORTED", result
    assert harness["calls"] == ["toolchain", "build", "branch", "roundtrip"]
    assert result["append_validation"]["file_constants"] == result["roundtrip_validation"]["file_constants"]
    assert result["append_validation"]["file_constants"]["count"] == 1153
    assert result["feature_output"] == {"index": 81, "dtype": "float16", "shape": [1, 55, 16384]}
    assert result["model_compilation"] == result["device_execution"] == result["numerical_validation"] == "NOT_RUN"
    assert sha256_file(harness["air"]) == before
    assert result["original_preservation"] == "UNCHANGED_STAT_AND_SHA256"
    assert Path(result["air"]["path"]).is_file()
    assert not list((harness["tmp"] / "output").rglob("padding_0"))
    retained = json.loads((harness["tmp"] / "output/result.json").read_text())
    assert retained["status"] == result["status"]


@pytest.mark.parametrize("gate", ["source", "plan", "native", "wire", "validation", "roundtrip", "post_roundtrip_validation"])
def test_failed_gate_blocks_every_dependent_stage_and_publication(harness, monkeypatch, gate):
    if gate == "source":
        value = json.loads(harness["receipt"].read_text())
        value["files"][-1]["sha256"] = "b" * 64
        harness["receipt"].write_text(json.dumps(value))
    elif gate == "plan":
        monkeypatch.setattr(append, "plan_original_ge_taps", lambda *a, **k: {"status": "PENDING_GRAPH_BINDING", "taps": []})
    elif gate == "native":
        def fail(*args, **kwargs):
            raise RuntimeError("native branch refused")
        monkeypatch.setattr(append, "_logged", fail)
    elif gate == "wire":
        def fail(*args, **kwargs):
            raise ValueError("wire refused original ambiguity")
        monkeypatch.setattr(append, "append_feature_branch_wire", fail)
    elif gate == "validation":
        monkeypatch.setattr(append, "validate_original_ge_append", lambda *a, **k: {"status": "VALIDATION_FAILED"})
    elif gate == "roundtrip":
        monkeypatch.setattr(append, "probe_original_air_roundtrip", lambda *a, **k: {"status": "FAILED"})
    elif gate == "post_roundtrip_validation":
        def mutate(target):
            target.write_bytes(target.read_bytes().replace(b"padding_1112", b"padding_9112"))
        harness["hooks"]["roundtrip"] = mutate
    result = harness["run"]()
    assert result["status"] in {"FAILED", "BLOCKED"}, result
    assert not (harness["tmp"] / "output/air/export.air").exists()
    if gate in {"source", "plan"}:
        assert harness["calls"] == []
    if gate in {"source", "plan", "native", "wire", "validation"}:
        assert "roundtrip" not in harness["calls"]


@pytest.mark.parametrize("mutation", ["original", "receipt", "toolchain", "schema"])
def test_mutation_during_native_cannot_publish_an_otherwise_successful_append(harness, mutation):
    target = {"original": harness["air"], "receipt": harness["receipt"],
              "toolchain": harness["tool"] / "lib64/libgraph.so", "schema": harness["schema"]}[mutation]
    # Change bytes only after all initial parse/audit checks and native branch
    # creation. Final source/tool identities must invalidate the whole result.
    def mutate(roundtrip):
        target.write_bytes(target.read_bytes() + b" \n")
    harness["hooks"]["roundtrip"] = mutate
    result = harness["run"]()
    assert result["status"] == "FAILED", result
    assert not (harness["tmp"] / "output/air/export.air").exists()


def test_unsupported_fp16_source_cannot_enter_fixed_fp32_native_template(harness, monkeypatch):
    original_plan = append.plan_original_ge_taps
    def unsupported_plan(*args, **options):
        result = original_plan(*args, **options)
        result["taps"][0]["descriptor"].update(raw_dtype=7, schema_dtype_names=["DT_FLOAT16"])
        return result
    monkeypatch.setattr(append, "plan_original_ge_taps", unsupported_plan)
    result = harness["run"]()
    assert result["status"] == "BLOCKED" and harness["calls"] == []
    assert "FP32" in result["error"]


@pytest.mark.parametrize("reference", ["../private-weight", "/private-weight"])
def test_absolute_or_traversing_external_reference_never_reaches_toolchain(harness, reference):
    raw = harness["air"].read_bytes()
    graph_raw = next(bytes(value) for number, wire, value in _fields(raw, [2_000_000]) if number == 7)
    graph_raw += _blob(6, _node("unsafe_constant", "FileConstant", [1], dtype=1) + _attr("file_path", _blob(2, reference)))
    harness["air"].write_bytes(_blob(7, graph_raw))
    result = harness["run"]()
    assert result["status"] == "BLOCKED" and harness["calls"] == []
    assert "cannot be isolated" in result["error"]


def test_declared_real_ge_full_boundary_and_1153_references_roundtrip_without_payloads(tmp_path, monkeypatch):
    """Host serialization integration only; all numerical bodies are synthetic."""
    ge_root, pb2 = os.environ.get("CANN_ROOT"), os.environ.get("TORCHAIR_SCHEMA_PB2")
    if not ge_root or not pb2:
        pytest.skip("requires explicitly declared CANN_ROOT and TORCHAIR_SCHEMA_PB2")
    import test_original_air_inspect as air_fixture
    import test_original_ge_taps as taps_fixture
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    # The fixture's deliberately nonstandard HALF enum is remapped explicitly
    # for the declared real GE schema. Native GE Cast attribute remains 1.
    original_desc = _desc
    def real_ge_desc(dtype=7, shape=(64, 16384)):
        return original_desc(dtype=2 if dtype == 7 else dtype, shape=shape)
    def real_ge_attr(name, value):
        if name == "shape":
            # Actual GE ListValue carries an explicit VT_LIST_INT tag. The
            # minimal selected-schema fixture intentionally has no such field.
            payload = next(bytes(item) for number, kind, item in _fields(value, [100]) if number == 1)
            value = _blob(1, payload + _int(20, 2))
        return _attr(name, value)
    monkeypatch.setattr(taps_fixture, "_desc", real_ge_desc)
    monkeypatch.setattr(taps_fixture, "_attr", real_ge_attr)
    monkeypatch.setattr(air_fixture, "_desc", real_ge_desc)
    air = tmp_path / "original-synthetic.air"
    air.write_bytes(_padding(taps_fixture.wire()))
    # Native serialization supplies its normal default metadata to this
    # hand-encoded fixture. Establish the GE-produced source fixture first;
    # every subsequent coordinator comparison must preserve it exactly.
    from qwen35_moe_dflash.original_air_roundtrip import probe_original_air_roundtrip
    seed = probe_original_air_roundtrip(air, tmp_path / "ge-fixture-seed", ge_root=ge_root, torchair_pb2=pb2)
    assert seed["native"]["load_status"] == seed["native"]["save_status"] == 0, seed
    assert seed["native"]["model_execution"] == "NOT_RUN"
    air.write_bytes(Path(seed["roundtrip_air"]["path"]).read_bytes())
    receipt = tmp_path / "original-source-receipt.json"
    receipt.write_text(json.dumps({"files": [{"path": name, "bytes": 100, "sha256": digest}
                                             for name, digest in SOURCE_SHA256.items()]}))
    result = append.append_original_ge_features(air, tmp_path / "actual-host-ge", torchair_pb2=pb2,
        gamma_template=TEMPLATE, sequence_length=55, source_receipt_path=receipt, ge_root=ge_root)
    assert result["status"] == "FEATURE_AIR_EXPORTED", result
    assert result["tap_plan"]["bound_input_norm_anchor_count"] == 40
    assert result["append_validation"]["original_preservation"]["inputs"] == 89
    assert result["append_validation"]["original_preservation"]["refdata"] == 80
    assert result["append_validation"]["file_constants"]["count"] == 1153
    assert result["roundtrip_validation"]["status"] == "METADATA_APPEND_VALIDATED"
    assert result["ge_roundtrip"]["status"] == "ROUNDTRIP_METADATA_PASSED"
    assert result["model_compilation"] == result["device_execution"] == result["numerical_validation"] == "NOT_RUN"
    assert not list(tmp_path.rglob("padding_0"))
