"""Structural state lineage fixtures; no vendor code, weights or model execute."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest

from qwen35_moe_dflash import target_state_lineage as audit
from qwen35_moe_dflash.existing_air_profile import get_existing_air_profile
from qwen35_moe_dflash.target_compile_failure import _stamp


ENUMS = {"dtype": {"values": {"DT_FLOAT16": 77, "DT_FLOAT": 78, "DT_INT8": 79,
                             "DT_INT16": 80, "DT_INT32": 81, "DT_INT64": 82}}}
RUNTIME_TYPES = {"FLOAT16": 2, "FLOAT32": 1, "INT8": 4, "INT16": 5, "INT32": 3, "INT64": 7}


def attr(key, value):
    return {"key": key, "value": value}


def desc(shape, dtype=77, *, fx=None, name=None):
    row = {"shape": {"dim": shape}, "dtype": dtype, "layout": "ND"}
    if fx is not None:
        row["attr"] = [attr("_fx_tensor_name", {"s": fx})]
    if name is not None:
        row["name"] = name
    return row


def named(node, inputs, outputs):
    node["attr"] = [attr("_input_name_key", {"list": {"s": list(inputs)}}),
        attr("_input_name_value", {"list": {"i": list(inputs.values())}}),
        attr("_output_name_key", {"list": {"s": list(outputs)}}),
        attr("_output_name_value", {"list": {"i": list(outputs.values())}})]
    return node


def graph(sequence=64):
    ordinary = [item.split(":")[0] for item in get_existing_air_profile("cpro-40block-prefill-64")["input_shape"].split(";")
                if not item.startswith("RefData_")]
    ops = [{"name": name, "type": "Data", "output_desc": [desc([1])]} for name in ordinary]
    outputs = []
    for index in range(80):
        position = index % 8
        shape = [1, 8192, 4] if position in (0, 2, 4) else [1, 32, 128, 128] if position in (1, 3, 5) else [64, 32, 32, 16]
        ref = f"RefData_{index}"
        ops.append({"name": ref, "type": "RefData", "output_desc": [desc(shape)]})
        kind = "LinearAttentionConvstateUpdate" if position in (0, 2, 4) else "ChunkGatedDeltaRule" if position in (1, 3, 5) else "CacheUpdate"
        state_in, state_out = audit._PAIRS[kind]
        # State input is deliberately at position1; no documented ordinal is
        # allowed to replace the graph's explicit serialized named mapping.
        node = named({"name": f"update_{index}", "type": kind, "input": [ordinary[0] + ":0", ref + ":0"],
                      "output_desc": [desc(shape, fx=f"source_tensor_{index}")]},
                     {"other_input": 0, state_in: 1}, {state_out: 0})
        ops.append(node)
        cast = {"name": f"state_cast_{index}", "type": "Cast", "input": [node["name"] + ":0"],
                "output_desc": [desc(shape, fx=f"source_tensor_{index}")]}
        ops.append(cast)
        outputs.append((cast["name"] + ":0", deepcopy(cast["output_desc"][0])))
    # Reverse source state order as well: RefData_i must not be inferred from
    # the source or compiled output's numeric ordinal.
    outputs.reverse()
    ops += [{"name": "logits", "type": "MatMul", "output_desc": [desc([1, 1, 248320], 78)]},
            {"name": "features", "type": "ConcatV2", "output_desc": [desc([1, sequence, 16384])]}]
    outputs = [("logits:0", deepcopy(ops[-2]["output_desc"][0])), *outputs,
               ("features:0", deepcopy(ops[-1]["output_desc"][0]))]
    ops.append({"name": "NetOutput", "type": "NetOutput", "input": [x[0] for x in outputs],
                "input_desc": [x[1] for x in outputs]})
    return {"graph": [{"name": "source", "op": ops}]}


def compiled_graph(source):
    _, _, _, source_outputs = audit._boundary(source, ENUMS)
    roots = [deepcopy(node) for node in source["graph"][0]["op"] if node["type"] in {"Data", "RefData"}]
    for node in roots:
        node["type"] = "Data"
    permutation = list(range(82))[17:] + list(range(17))
    descriptors = []
    for source_index in permutation:
        raw = deepcopy(source_outputs[source_index]["descriptor"])
        # This is a literal descriptor name, not a parsed output_N prefix.
        raw["name"] = f"literal-tensor-name/{source_index}"
        descriptors.append(raw)
    roots.append({"name": "opaque-task", "type": "GraphOp", "output_desc": descriptors})
    roots.append({"name": "NetOutput", "type": "NetOutput", "input": [f"opaque-task:{i}" for i in range(82)],
                  "input_desc": deepcopy(descriptors)})
    return {"graph": [{"name": "compiled", "op": roots}]}


def runtime_probe(compiled, *, names=True):
    _, nodes, _, outputs = audit._boundary(compiled, ENUMS)
    roots = list(reversed([node for node in nodes.values() if node["type"] == "Data"]))
    result = {"input_count": 89, "output_count": 82, "dtype_enum": "hiai::DataType", "format_enum": "hiai::Format",
              "dtype_enum_values": RUNTIME_TYPES, "format_enum_values": {"ND": 2}, "inputs": [], "outputs": []}
    for direction, values in (("inputs", [(node["name"], node["output_desc"][0]) for node in roots]),
                              ("outputs", [(row["descriptor"]["name"], row["descriptor"]) for row in reversed(outputs)])):
        for name, raw in values:
            kind, shape = audit._view(raw, ENUMS)
            row = {"index": len(result[direction]), "shape": [1, *shape], "raw_dtype": RUNTIME_TYPES[kind], "raw_format": 2}
            if names:
                row.update(tensor_name=name, tensor_name_status="OBSERVED")
            result[direction].append(row)
    return result


def test_actual_named_edges_override_every_source_compiled_and_runtime_ordinal():
    source = graph()
    lineage = audit.analyze_state_lineage(source, ENUMS)
    assert lineage["status"] == "ALL80_NAMED_PORT_STATE_LINEAGES_OBSERVED"
    first = lineage["states"][0]
    assert first["source_output_index"] == 1 and first["source_refdata"] == "RefData_79"
    assert first["updater"]["input_port_index"] == 1
    compiled = compiled_graph(source)
    joined = audit._join_compiled(lineage, compiled, ENUMS)
    assert joined["status"] == "ALL80_EXPLICIT_FX_TENSOR_IDENTITIES_MATCHED"
    assert joined["states"][0]["compiled_output_index"] != first["source_output_index"]
    runtime = audit._runtime_bindings(joined, runtime_probe(compiled), ENUMS)
    assert runtime["status"] == "ALL80_EXPLICIT_RUNTIME_STATE_BINDINGS_OBSERVED"
    row = runtime["states"][0]
    assert row["runtime_input_name"] == "RefData_79"
    assert row["runtime_output_index"] != row["compiled_output_index"]


def test_alternate_explicit_source_edges_are_supported_without_order_guessing():
    source = graph()
    for node in source["graph"][0]["op"]:
        if "input" not in node:
            continue
        edges = node.pop("input")
        node["src_name"] = [edge.rsplit(":", 1)[0] for edge in edges]
        node["src_index"] = [int(edge.rsplit(":", 1)[1]) for edge in edges]
    report = audit.analyze_state_lineage(source, ENUMS)
    assert report["status"] == "ALL80_NAMED_PORT_STATE_LINEAGES_OBSERVED"


def test_conflicting_dual_edge_representations_do_not_prove_state_owner():
    source = graph()
    update = next(node for node in source["graph"][0]["op"] if node["name"] == "update_0")
    update["src_name"] = [update["input"][0].rsplit(":", 1)[0], "RefData_2"]
    update["src_index"] = [0, 0]
    report = audit.analyze_state_lineage(source, ENUMS)
    assert report["status"] == "STATE_LINEAGE_PARTIAL"
    row = next(row for row in report["states"] if row["source_output_edge"] == "state_cast_0:0")
    assert "conflicting input" in row["reason"] and row["status"] == "UNVERIFIED"


@pytest.mark.parametrize("producer", ["features", "state_cast_0"])
def test_source_unmaterialized_producer_shape_preserves_named_edges_without_inference(producer):
    source = graph()
    node = next(node for node in source["graph"][0]["op"] if node["name"] == producer)
    node["output_desc"][0]["shape"] = {}
    report = audit.analyze_state_lineage(source, ENUMS)
    assert report["observed_state_owner_count"] == report["unique_state_owner_count"] == 80
    assert report["producer_shape_inference"].startswith("NOT_PERFORMED")
    assert node["output_desc"][0]["shape"] == {}
    if producer == "features":
        assert report["unmaterialized_output_producer_indices"] == [81]
        assert report["status"] == "ALL80_NAMED_PORT_STATE_LINEAGES_OBSERVED"
    else:
        assert report["fixed_state_output_producer_shape_count"] == 79
        assert report["status"] == "STATE_LINEAGE_PARTIAL"
        state = next(row for row in report["states"] if row["source_refdata"] == "RefData_0")
        assert state["producer_shape_status"] == "UNMATERIALIZED_OR_SCALAR_AMBIGUOUS"
        assert state["status"] == "NAMED_PORT_STATE_LINEAGE_OBSERVED"


@pytest.mark.parametrize("fault", ["shape", "dtype", "layout"])
def test_compiled_producers_require_actual_fixed_matching_descriptors(fault):
    source = graph()
    compiled = compiled_graph(source)
    producer = compiled["graph"][0]["op"][-2]["output_desc"][0]
    if fault == "shape": producer["shape"] = {}
    elif fault == "dtype": producer["dtype"] = 79
    else: producer["layout"] = "NCHW"
    with pytest.raises(ValueError, match="fixed shape|dtype differs|explicit ND"):
        audit._join_compiled(audit.analyze_state_lineage(source, ENUMS), compiled, ENUMS)


@pytest.mark.parametrize("fault", ["missing_map", "wrong_named_output", "wrong_owner", "cycle", "unsupported_carrier"])
def test_state_shapes_or_output_ordinals_cannot_resolve_missing_actual_edges(fault):
    source = graph()
    ops = source["graph"][0]["op"]
    update = next(node for node in ops if node["name"] == "update_0")
    cast = next(node for node in ops if node["name"] == "state_cast_0")
    if fault == "missing_map": update.pop("attr")
    elif fault == "wrong_named_output": update["attr"][2]["value"]["list"]["s"] = ["not_state"]
    elif fault == "wrong_owner": update["input"][1] = "RefData_2:0"
    elif fault == "cycle": cast["input"] = ["state_cast_0:0"]
    elif fault == "unsupported_carrier": cast["type"] = "Reshape"
    report = audit.analyze_state_lineage(source, ENUMS)
    assert report["status"] == "STATE_LINEAGE_PARTIAL"
    assert report["missing_state_owners"]
    if fault == "wrong_owner":
        assert report["duplicate_state_owners"] == ["RefData_2"]
    else:
        assert any(row["status"] == "UNVERIFIED" for row in report["states"])


@pytest.mark.parametrize("fault", ["missing_fx", "duplicate_fx", "conflicting_fx", "missing_names", "duplicate_names", "runtime_dtype"])
def test_declared_identity_or_runtime_name_gaps_are_not_filled_from_equal_shapes(fault):
    source = graph()
    compiled = compiled_graph(source)
    if fault in {"missing_fx", "duplicate_fx", "conflicting_fx"}:
        raw = compiled["graph"][0]["op"][-1]["input_desc"]
        body = compiled["graph"][0]["op"][-2]["output_desc"]
        if fault == "missing_fx":
            for row in [*raw, *body]: row.pop("attr", None)
        elif fault == "duplicate_fx":
            raw[1]["attr"] = deepcopy(raw[0].get("attr", []));body[1]["attr"] = deepcopy(raw[1]["attr"])
        else:
            raw[0]["attr"] = [attr("_fx_tensor_name", {"s": "different"})]
    lineage = audit.analyze_state_lineage(source, ENUMS)
    joined = audit._join_compiled(lineage, compiled, ENUMS)
    if fault in {"missing_fx", "duplicate_fx", "conflicting_fx"}:
        assert joined["matched_count"] < 80
    probe = runtime_probe(compiled, names=fault != "missing_names")
    if fault == "duplicate_names":
        probe["inputs"][0]["tensor_name"] = probe["inputs"][1]["tensor_name"]
    elif fault == "runtime_dtype":
        probe["outputs"][0]["raw_dtype"] = RUNTIME_TYPES["INT8"]
    runtime = audit._runtime_bindings(joined, probe, ENUMS)
    assert runtime["matched_count"] < 80


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def record(path, root, *, stat_key="stat"):
    return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), stat_key: _stamp(path.stat())}


@pytest.fixture
def jobs(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    ddk = tmp_path / "ddk"
    library = ddk / "tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"HOST FIXTURE ONLY STATIC SCHEMA")
    documents = {}

    class FakeSchema:
        evidence = {"library": str(library), "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                    "descriptor_sha256": "f" * 64}
        enums = {"fixture.DataType": ENUMS["dtype"]["values"]}
        _messages = {"hiai.proto.TensorDescriptor": {1: SimpleNamespace(kind=14, name="dtype", type_name="fixture.DataType")}}

        def decode(self, raw, kind):
            key = bytes(raw)
            if key not in documents or kind != "ModelDef": raise ValueError("unsupported fixture payload")
            return deepcopy(documents[key])

    monkeypatch.setattr(audit, "load_schema", lambda path: FakeSchema())
    result = {"tmp": tmp_path, "prefill_execution": tmp_path / "execution"}
    for label, sequence, profile in (("prefill", 64, "cpro-40block-prefill-64"), ("decode", 1, "cpro-40block-decode-1")):
        job = tmp_path / (label + "-job");comp = job / "compilation"
        air = comp / "compile-work/generated/export.air"
        air.parent.mkdir(parents=True)
        air.write_bytes((label + " SOURCE").encode())
        source, compiled = graph(sequence), compiled_graph(graph(sequence))
        documents[air.read_bytes()] = source
        model_payload = (label + " COMPILED").encode()
        documents[model_payload] = compiled
        task = b"opaque task not interpreted"
        content = struct.pack("<I", 2) + struct.pack("<III", 0, 0, len(model_payload)) + struct.pack("<III", 2, 0, len(task)) + model_payload + task
        header = bytearray(256);header[:4] = b"IMOD";struct.pack_into("<I", header, 0x4C, len(content))
        omc = comp / "compile-work/generated/model/model.omc";omc.parent.mkdir()
        omc.write_bytes(header + content)
        weight = comp / "compile-work/generated/never-open.weight";weight.write_bytes(b"DO NOT OPEN")
        snapshots = [dict(record(air, comp, stat_key="snapshot_stat"), role="air"),
                     {"path": weight.relative_to(comp).as_posix(), "bytes": weight.stat().st_size, "role": "neighbor_data_candidate", "sha256": "0" * 64}]
        report = {"artifact_kind": "qwen35-existing-air-compilation", "status": "COMPILED", "model_compilation": "PASSED",
            "returncode": 0, "platform": "kirinx90", "ddk_id": "fixture-ddk", "ddk_root": str(ddk),
            "source_preservation": {"status": "UNCHANGED_BY_STAT"}, "input_snapshots": snapshots,
            "omc": record(omc, comp), "generated_artifacts": [record(omc, comp)]}
        if label == "decode":
            previous = result["prefill_job"]
            borrowed = {"artifact_kind": "qwen35-decode-input-reference-view", "status": "PASSED",
                "source_job": str(previous), "reference_method": "READONLY_SOURCE_REFERENCE_SYMLINK",
                "explicit_weight_copy_bytes": 0, "audit_errors": [], "source_binding": {
                    "source_result_sha256": hashlib.sha256((previous / "result.json").read_bytes()).hexdigest(),
                    "source_compilation_report_sha256": hashlib.sha256((previous / "compilation/report.json").read_bytes()).hexdigest()}}
            report["decode_input_view"] = borrowed
            save(comp / "decode-input-view.json", borrowed)
        save(comp / "report.json", report);save(comp / "input-inventory.json", {"files": snapshots})
        sha = hashlib.sha256((comp / "report.json").read_bytes()).hexdigest()
        save(job / "result.json", {"artifact_kind": "qwen35-original-target-job", "status": "COMPILED", "output": str(job), "compilation_report_sha256": sha})
        save(job / "compile-profile.json", get_existing_air_profile(profile))
        probe = tmp_path / (label + "-probe")
        native = {**runtime_probe(compiled), "artifact_kind": "qwen35-hiai-model-io-probe", "evidence_kind": "device-hiai",
                  "status": "IO_DESCRIPTORS_OBSERVED", "restore": "PASSED", "descriptor_stage": "restore-only", "manager_init": "NOT_RUN",
                  "model_sha256": report["omc"]["sha256"], "model_bytes": report["omc"]["bytes"],
                  "model_path": "/data/local/tmp/owned/metadata/model.omc", "model_load_api": "file-v2",
                  "model_load_scope": "RESTORE_METADATA_ONLY", "restore_init": "NOT_RUN_METADATA_ONLY",
                  "external_weight_initialization": "NOT_REQUESTED", "process_called": False, "run_called": False,
                  "tensor_allocations_requested": 0, "npu_supported": True, "expected_runtime_version": "fixture-runtime",
                  "runtime_version": "fixture-runtime", "firmware_id": "fixture-firmware", "ddk_id": "fixture-ddk"}
        save(probe / "model-io.json", native)
        save(probe / "runtime.json", {"npu_supported": True, "runtime_version": "fixture-runtime"})
        for filename in ("runtime.exitcode", "model-io.exitcode"): (probe / filename).write_bytes(b"0\n")
        argv = ["/data/local/tmp/runner", "--model-io-only", native["model_path"], "--model-sha256", report["omc"]["sha256"],
            "--descriptor-stage", "restore-only", "--expected-runtime-version", "fixture-runtime", "--ddk-id", "fixture-ddk",
            "--firmware-id", "fixture-firmware", "--output", "/data/local/tmp/owned/model-io.json"]
        save(probe / "model-io-command.json", {"argv": argv, "cwd": str(Path(native["model_path"]).parent)})
        probe_result = {**native, "artifact_kind": "qwen35-target-model-io-probe", "status": "IO_METADATA_OBSERVED",
            "job": str(job), "profile": profile, "compilation_report_sha256": sha, "model_sha256": report["omc"]["sha256"],
            "model_bytes": report["omc"]["bytes"], "remote_exitcode": 0, "retrieval_errors": [],
            "lock_state": "NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT", "source_preservation": "UNCHANGED_METADATA_AND_SMALL_PRIMARY_SHA256; WEIGHTS_NOT_OPENED",
            "retrieved_sha256": {filename: hashlib.sha256((probe / filename).read_bytes()).hexdigest()
                for filename in ("model-io.json", "runtime.json", "runtime.exitcode", "model-io.exitcode")}}
        save(probe / "result.json", probe_result)
        result[label + "_job"], result[label + "_probe"] = job, probe
        result[label + "_report"], result[label + "_compiled"] = report, compiled
    save(result["prefill_execution"] / "result.json", {"status": "EXECUTED_UNVALIDATED", "remote_exitcode": 0, "device_execution": "EXECUTED_OUTPUTS_RETRIEVED"})
    prefill_probe = json.loads((result["prefill_probe"] / "result.json").read_text())
    save(result["prefill_execution"] / "board/runtime-prefill-proof.json", {
        "compilation_report_sha256": prefill_probe["compilation_report_sha256"], "model_sha256": prefill_probe["model_sha256"],
        "runtime_probe": str(result["prefill_probe"])})
    result["documents"] = documents
    return result


def run(jobs):
    return audit.inspect_target_state_lineage(jobs["prefill_job"], jobs["decode_job"], jobs["prefill_execution"],
        jobs["decode_probe"], jobs["tmp"] / "audit", prefill_runtime_probe=jobs["prefill_probe"])


def test_complete_readonly_audit_binds_graph_and_runtime_without_opening_weight(jobs, monkeypatch):
    original_open = Path.open
    def guarded(path, *args, **kwargs):
        assert path.suffix != ".weight", "weight payload access is outside this audit"
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    report = run(jobs)
    assert report["status"] == "ALL80_STRUCTURAL_STATE_BINDINGS_OBSERVED", report
    assert report["device_execution"] == report["state_feedback"] == "NOT_RUN"
    assert report["state_continuity"] == "NOT_VALIDATED"
    assert report["weight_payload_access"] == "NONE"
    assert report["cross_model_state_identity"] == "UNVERIFIED" and report["feedback_ready"] is False
    assert report["prefill"]["runtime_bindings"]["matched_count"] == 80
    assert report["decode"]["source_lineage"]["unique_state_owner_count"] == 80
    assert json.loads((jobs["tmp"] / "audit/report.json").read_text()) == report


def test_complete_per_model_edges_do_not_prove_same_source_layer_across_exports(jobs):
    source = jobs["documents"][b"decode SOURCE"]
    nodes = {node["name"]: node for node in source["graph"][0]["op"]}
    nodes["update_0"]["input"][1], nodes["update_2"]["input"][1] = "RefData_2:0", "RefData_0:0"
    report = run(jobs)
    assert report["status"] == "ALL80_STRUCTURAL_STATE_BINDINGS_OBSERVED", report
    assert report["prefill"]["runtime_bindings"]["matched_count"] == report["decode"]["runtime_bindings"]["matched_count"] == 80
    assert report["cross_model_state_identity"] == "UNVERIFIED"
    assert report["feedback_ready"] is False and report["state_feedback"] == "NOT_RUN"


@pytest.mark.parametrize("fault", ["air_bytes", "primary_bytes", "compile_binding", "native_hash", "other_model_probe", "runtime_pair"])
def test_changed_or_unbound_job_probe_metadata_fails_closed(jobs, fault):
    comp = jobs["prefill_job"] / "compilation"
    if fault == "air_bytes": (comp / jobs["prefill_report"]["input_snapshots"][0]["path"]).write_bytes(b"changed AIR")
    elif fault == "primary_bytes": (comp / jobs["prefill_report"]["omc"]["path"]).write_bytes(b"changed OMC")
    elif fault == "compile_binding":
        value = json.loads((jobs["prefill_job"] / "result.json").read_text());value["compilation_report_sha256"] = "0" * 64;save(jobs["prefill_job"] / "result.json", value)
    else:
        value = json.loads((jobs["decode_probe"] / "result.json").read_text())
        if fault == "native_hash": value["retrieved_sha256"]["model-io.json"] = "0" * 64
        elif fault == "other_model_probe": value["model_sha256"] = "0" * 64
        else: value["runtime_version"] = "different-runtime"
        save(jobs["decode_probe"] / "result.json", value)
    report = run(jobs)
    assert report["status"] == "FAILED" and report["error"]
    assert report["device_execution"] == "NOT_RUN"
