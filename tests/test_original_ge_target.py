"""Plan B reviews source receipts and real encoded graph metadata, never emits GE."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import runpy
import sys

import pytest

from qwen35_moe_dflash import original_ge_target as ge
from qwen35_moe_dflash.original_air_inspect import inspect_original_air
from test_draft_ir_inspect import _attr, _blob, _desc
from test_original_air_inspect import _graph, _node, selected


TEMPLATE = "./exact-export/model_model_layers_{layer}_input_layernorm_weight"


def receipt():
    return {"files": [{"path": name, "bytes": 100, "sha256": digest} for name, digest in ge.SOURCE_SHA256.items()]}


def feature_source_receipt():
    return {"artifact_kind": "qwen35-original-target-feature-source",
        "source_files": {row["path"]: dict(row, path="/original/" + row["path"]) for row in receipt()["files"]},
        "generated_files": {"source/models/export_model_wrapper_qwen3_5_moe.py": {"sha256": "b" * 64}}}


def _tapped_graph(mutation=None):
    graph = _graph(constants=0)
    for index, layer in enumerate(ge.LAYER_IDS):
        # Names deliberately have no layer semantics; only the explicit
        # FileConstant path and ordered producer ports can associate the tap.
        gamma, producer, norm = f"arbitrary_gamma_{index}", f"arbitrary_value_{index}", f"arbitrary_norm_{index}"
        path = TEMPLATE.format(layer=layer + 1)
        shape = [1, 55, 2048]
        gamma_shape = [2048]
        norm_type = "RmsNorm"
        if index == 0:
            if mutation == "wrong_path":
                path = "./other-export/model_model_layers_2_input_layernorm_weight"
            elif mutation == "gamma_shape":
                gamma_shape = [1024]
            elif mutation == "wrong_norm":
                norm_type = "AdnRmsNorm"
            elif mutation == "producer_shape":
                shape[-1] = 2047
        gamma_wire = (_node(gamma, "FileConstant", gamma_shape) + _attr("file_path", _blob(2, path)))
        producer_wire = _node(producer, "UnverifiedPostDecoder", shape, dtype=1)
        gamma_edge = gamma + ":0"
        extra = b""
        if index == 0 and mutation == "gamma_add":
            transform = "gamma_add_not_value_proven"
            extra = (_node(transform, "Add", gamma_shape) + _blob(5, gamma_edge))
            gamma_edge = transform + ":0"
        norm_wire = (_node(norm, norm_type, shape, dtype=1) +
                     _blob(5, producer + ":0") + _blob(5, gamma_edge) +
                     _blob(33, _desc(dtype=1, shape=shape)) + _blob(33, _desc(shape=gamma_shape)))
        if index == 0 and mutation == "norm_input_desc":
            norm_wire = norm_wire.replace(_blob(33, _desc(dtype=1, shape=shape)), _blob(33, _desc(shape=shape)))
        graph += _blob(6, gamma_wire) + _blob(6, producer_wire) + _blob(6, norm_wire)
        if extra:
            graph += _blob(6, extra)
        if index == 0 and mutation == "duplicate_path":
            graph += _blob(6, _node("same_path_other_fileconstant", "FileConstant", gamma_shape) + _attr("file_path", _blob(2, path)))
        if index == 0 and mutation == "extra_consumer":
            graph += _blob(6, _node("other_consumer", "Identity", gamma_shape) + _blob(5, gamma + ":0"))
    return graph


def report(tmp_path, selected, mutation=None):
    air = tmp_path / "source.air"
    air.write_bytes(_blob(7, _tapped_graph(mutation)))
    result = inspect_original_air(air, selected[0], full=True)
    assert result["status"] == "STRUCTURE_OBSERVED"
    return result


def test_missing_actual_graph_produces_concrete_pending_contract_without_using_functional_frontend():
    result = ge.prepare_contract(receipt(), air="/original/export.air")
    assert result["status"] == "PENDING_GRAPH_BINDING"
    assert result["source_audit"]["status"] == "MATCHED_AUDITED_SOURCE_RECEIPTS"
    assert result["abi"]["inputs"] == 89 and result["abi"]["refdata"] == 80
    assert result["abi"]["original_outputs"] == 81 and result["abi"]["feature_outputs"] == 82
    assert result["air_generation"] == "NOT_RUN_NO_GRAPH_WRITER_OR_RANDOM_WEIGHTS"
    assert "NOT_ORIGINAL_ABI" in result["frontend_routes"]["functional_quant_target"]
    operators = {row["role"]: row for row in result["required_operators"]}
    assert {name: operators[name]["count"] for name in ("full_attention", "cache_update", "conv_state", "gdn")} == {
        "full_attention": 10, "cache_update": 20, "conv_state": 30, "gdn": 30}
    assert operators["norm"]["source_call"] == "torch_npu.npu_rms_norm"
    assert "stored_weight+1" in operators["norm"]["contract"]
    assert result["next_command"]["arguments"] == ["--air", "/original/export.air", "--full"]


def test_only_original_six_source_files_bind_the_derived_source_receipt():
    value = feature_source_receipt()
    assert ge.audit_source(value)["status"] == "MATCHED_AUDITED_SOURCE_RECEIPTS"
    value["generated_files"] = {"all_feature_source": {"sha256": "c" * 64}}
    assert ge.audit_source(value)["status"] == "MATCHED_AUDITED_SOURCE_RECEIPTS"
    value["source_files"]["models/export_model_wrapper_qwen3_5_moe.py"]["sha256"] = "c" * 64
    result = ge.prepare_contract(value)
    assert result["status"] == "PENDING_SOURCE_BINDING"
    assert result["source_audit"]["different_files"] == ["models/export_model_wrapper_qwen3_5_moe.py"]


def test_yaml_path_or_format_change_keeps_provenance_without_changing_python_computation_binding():
    value = feature_source_receipt()
    config = value["source_files"]["config/qwen3_5_moe.yaml"]
    config.update(path="/new-machine/resolved-original.yaml", sha256="a" * 64, bytes=300)
    result = ge.prepare_contract(value)
    assert result["status"] == "PENDING_GRAPH_BINDING"
    audit = result["source_audit"]
    assert audit["status"] == "MATCHED_AUDITED_SOURCE_RECEIPTS" and audit["different_files"] == []
    assert audit["configuration"]["sha256"] == "a" * 64
    assert audit["configuration"]["matches_audited_text"] is False
    assert "PROVENANCE_ONLY" in audit["configuration"]["role"]
    assert "actual AIR" in audit["configuration"]["geometry"]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "malformed_hash"])
def test_incomplete_or_ambiguous_source_receipts_reject(mutation):
    value = receipt()
    if mutation == "missing":
        value["files"].pop()
    elif mutation == "duplicate":
        value["files"][-1] = deepcopy(value["files"][0])
    else:
        value["files"][0]["sha256"] = "not a sha256"
    with pytest.raises(ValueError):
        ge.prepare_contract(value)


def test_exact_fileconstant_consumer_edges_identify_only_structural_candidates(tmp_path, selected):
    value = report(tmp_path, selected)
    result = ge.prepare_contract(receipt(), value, gamma_template=TEMPLATE)
    assert result["status"] == "PENDING_GRAPH_BINDING"
    taps = result["tap_resolution"]
    assert taps["status"] == "EIGHT_STRUCTURAL_CANDIDATES_SOURCE_BINDING_PENDING"
    assert [row["producer_reference"] for row in taps["taps"]] == [f"arbitrary_value_{index}:0" for index in range(8)]
    assert [row["next_input_norm_layer_index"] for row in taps["taps"]] == [2, 7, 12, 17, 23, 28, 33, 38]
    assert all(row["descriptor"]["schema_dtype_names"] == ["DT_FLOAT"] for row in taps["taps"])
    assert taps["semantic_validation"] == "NOT_ESTABLISHED"
    assert "gamma+1" in taps["taps"][0]["remaining"]
    assert all(row["graph_binding"]["status"] == "COUNT_DIFFERENT" for row in result["required_operators"] if "count" in row)


@pytest.mark.parametrize("mutation,reason", [
    ("wrong_path", "EXACT_FILECONSTANT_PATH_MISSING_OR_AMBIGUOUS"),
    ("duplicate_path", "EXACT_FILECONSTANT_PATH_MISSING_OR_AMBIGUOUS"),
    ("gamma_shape", "GAMMA_SHAPE_OR_DTYPE_UNVERIFIED"),
    ("wrong_norm", "GAMMA_TRANSFORM_OR_NORM_CONVERTER_UNVERIFIED"),
    ("gamma_add", "GAMMA_TRANSFORM_OR_NORM_CONVERTER_UNVERIFIED"),
    ("producer_shape", "NORM_DATA_PORT_SHAPE_OR_PRECISION_UNVERIFIED"),
    ("norm_input_desc", "NORM_DATA_PORT_SHAPE_OR_PRECISION_UNVERIFIED"),
    ("extra_consumer", "GAMMA_CONSUMER_MISSING_OR_AMBIGUOUS"),
])
def test_tap_resolution_never_guesses_missing_paths_transforms_or_precision(tmp_path, selected, mutation, reason):
    value = report(tmp_path, selected, mutation)
    result = ge.prepare_contract(receipt(), value, gamma_template=TEMPLATE)["tap_resolution"]
    assert result["status"] == "UNRESOLVED"
    assert result["taps"][0]["reason"] == reason


def test_summary_report_or_no_template_remains_pending(tmp_path, selected):
    value = report(tmp_path, selected)
    graph = next(row["graph"] for row in value["candidates"] if "graph" in row)
    assert ge.resolve_taps(graph)["reason"] == "EXPLICIT_EXACT_GAMMA_PATH_TEMPLATE_REQUIRED"
    del graph["operations"]
    assert ge.resolve_taps(graph, TEMPLATE)["reason"] == "FULL_OPERATION_AND_FILECONSTANT_METADATA_REQUIRED"


def test_container_or_changed_evidence_is_not_a_source_editing_basis(tmp_path, selected):
    value = report(tmp_path, selected)
    altered = deepcopy(value)
    altered["source_preservation"] = "CHANGED"
    with pytest.raises(ValueError):
        ge.prepare_contract(receipt(), altered)
    candidate = next(row for row in value["candidates"] if "graph" in row)
    candidate["format"] = "IMOD_MODEL_PARTITION"
    result = ge.prepare_contract(receipt(), value, gamma_template=TEMPLATE)
    assert result["graph_observation"]["status"] == "NO_UNAMBIGUOUS_RAW_ORIGINAL_GRAPH"
    assert result["tap_resolution"]["status"] == "UNRESOLVED"


def test_actual_air_schema_mismatch_requires_original_torchair_descriptor_without_decoder_fallback():
    observed = {"artifact_kind": "qwen35-original-air-inspection", "status": "OPAQUE",
        "air": {"path": "/original/export.air", "bytes": 6361049,
                "sha256": "57934fc4177929a20c978eadbb121741602cefbfb212027e440fce67b08d32a6"},
        "source_preservation": "UNCHANGED_BY_STAT", "schema": {"status": "CANDIDATE_ONLY"},
        "candidates": [{"format": "RAW_GRAPHDEF", "status": "UNSUPPORTED", "error": "unknown protobuf field 7"},
                       {"format": "RAW_MODELDEF", "status": "UNSUPPORTED", "error": "wire type mismatch for AttrDef.list_list_float"}]}
    result = ge.prepare_contract(receipt(), observed)
    assert result["status"] == "PENDING_AIR_SCHEMA"
    assert "MATCH_ORIGINAL_TORCHAIR_GE_SERIALIZED_DESCRIPTOR_WITH_RAW_AIR" in result["pending"]
    assert result["schema_observation"]["format_attempts"] == observed["candidates"]
    assert result["next_command"]["status"] == "REQUIRES_MATCHING_TORCHAIR_GE_DESCRIPTOR_INPUT"
    assert result["air_generation"] == "NOT_RUN_NO_GRAPH_WRITER_OR_RANDOM_WEIGHTS"


def test_decoded_output_difference_is_an_abi_gap_and_keeps_full_graph(tmp_path, selected):
    air = tmp_path / "boundary.air"
    air.write_bytes(_blob(7, _graph(mutation="output_producer_desc")))
    observed = inspect_original_air(air, selected[0], full=True)
    assert observed["status"] == "DECODED_BOUNDARY_DIFFERENT"
    result = ge.prepare_contract(receipt(), observed)
    assert result["status"] == "PENDING_ORIGINAL_ABI_BINDING"
    assert result["graph_observation"]["output_descriptor_check"]["mismatch_count"] == 1
    assert "schema_observation" not in result
    assert result["air_generation"] == "NOT_RUN_NO_GRAPH_WRITER_OR_RANDOM_WEIGHTS"


def test_decoded_metadata_failure_never_restarts_schema_search(tmp_path, selected):
    air = tmp_path / "duplicate.air"
    air.write_bytes(_blob(7, _graph(mutation="duplicate_op")))
    observed = inspect_original_air(air, selected[0], full=True)
    assert observed["status"] == "DECODED_METADATA_UNSUPPORTED"
    observed["schema"].update(selection="EXPLICIT_TORCHAIR_PB2", pb2="/actual/ge_ir_by_protoc_3_19_pb2.py")
    result = ge.prepare_contract(receipt(), observed)
    assert result["status"] == "PENDING_GRAPH_METADATA"
    assert "schema_observation" not in result
    assert "RESOLVE_DECODED_GRAPH_METADATA_ERROR" in result["pending"]
    assert "--find-torchair-schema" not in result["next_command"]["arguments"]
    assert result["next_command"]["arguments"][-2:] == ["--torchair-pb2", observed["schema"]["pb2"]]


def test_cli_canonical_output_directory_contains_only_reviewable_json(tmp_path, monkeypatch, capsys):
    source = tmp_path / "source-receipt.json"
    source.write_text(json.dumps(feature_source_receipt()))
    output = tmp_path / "target-ge"
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    script = Path(__file__).resolve().parents[1] / "targets/cpro/scripts/prepare_original_ge_target.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--source-receipt", str(source), "--output", str(output)])
    assert runpy.run_path(str(script))["main"]() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "PENDING_GRAPH_BINDING"
    assert json.loads((output / "contract.json").read_text()) == result
    assert list(output.iterdir()) == [output / "contract.json"]
    assert result["metadata_receipts"]["source_receipt_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_json_duplicate_keys_cannot_replace_source_or_air_identity(tmp_path):
    path = tmp_path / "receipt.json"
    path.write_text('{"files":[],"files":[]}')
    with pytest.raises(ValueError, match="duplicate"):
        ge.read_json(path)
