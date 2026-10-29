"""Selected-schema physical output decoding; promotions remain diagnostic failures."""
from copy import deepcopy
import hashlib
import subprocess

import pytest

from qwen35_moe_dflash.draft_direct_abi import inspect_direct_output_abi
from qwen35_moe_dflash.omc_wire import Schema
from test_draft_ir_inspect import _blob, _desc, _int
from test_draft_precision_omc import _container
import test_draft_ir_inspect as schema_fixture


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("output ABI reader must not execute tools"))
    original = schema_fixture._schema()
    def enum(items):
        return _blob(5, _blob(1, "DataType") + b"".join(_blob(2, _blob(1, name) + _int(2, value)) for name, value in items))
    values = [("UNDEFINED", 0), ("FLOAT", 1), ("HALF", 7)]
    expanded = original.replace(enum(values), enum(values + [("INT32", 3)]))
    monkeypatch.setattr(schema_fixture, "_schema", lambda: expanded)
    library = tmp_path / "selected-schema.so"
    schema_fixture._library(library)
    spec = [
        {"name": "draft_top1", "source": "draft_top1", "shape": [1, 15], "dtype": "int32", "output_index": 0},
        {"name": "draft_hidden", "source": "draft_hidden", "shape": [1, 15, 2048], "dtype": "float16", "output_index": 1},
        {"name": "noise_embedding", "source": "v19", "shape": [1, 16, 2048], "dtype": "float16", "output_index": 2},
        {"name": "layer_0_input", "source": "v19", "shape": [1, 16, 2048], "dtype": "float16", "output_index": 2},
    ]
    return tmp_path / "model.omc", library, spec


def _write(case, *, dtypes=(3, 7, 7), shapes=None, missing_dtype=None, damage=None):
    path, _, spec = case
    shapes = shapes or [row["shape"] for row in spec[:3]]
    descriptors = []
    for index, (dtype, shape) in enumerate(zip(dtypes, shapes)):
        descriptors.append(_desc(dtype, shape) if index != missing_dtype else
            _blob(3, b"".join(_int(1, dim) for dim in shape)) + _blob(4, "ND"))
    producer_descs = list(descriptors)
    if damage == "producer_dtype":
        producer_descs[2] = _desc(1, shapes[2])
    producer = (_blob(1, "opaque_partition") + _blob(2, "GraphOp")
                + b"".join(_blob(34, desc) for desc in producer_descs))
    refs = ["opaque_partition:0", "opaque_partition:1", "opaque_partition:2"]
    if damage == "refs":
        refs = ["bad"]
    if damage == "missing_producer":
        refs[2] = "unknown_partition:0"
    if damage == "bad_port":
        refs[2] = "opaque_partition:8"
    if damage == "control":
        refs.insert(0, "control_only:-1")
    netout = (_blob(1, "ordered_model_outputs") + _blob(2, "NetOutput")
              + b"".join(_blob(5, ref) for ref in refs) + b"".join(_blob(33, desc) for desc in descriptors))
    if damage == "stale_metadata":
        netout += _blob(24, "pretend_source") + _int(25, 0)
    graph = _blob(1, "compiled") + _blob(6, producer) + _blob(6, netout)
    if damage == "two_netout":
        prefix = _blob(1, "ordered_model_outputs")
        graph += _blob(6, _blob(1, "other_model_outputs") + netout[len(prefix):])
    path.write_bytes(_container(_blob(7, graph), b"opaque NPU task bytes\x00\xff"))
    return path


def test_preserved_types_use_selected_schema_and_read_main_only(case, monkeypatch):
    path, library, spec = case
    _write(case)
    before = path.read_bytes(), library.read_bytes(), deepcopy(spec)
    original, calls = Schema.decode, []
    def decode(self, data, name):
        calls.append(name)
        return original(self, data, name)
    monkeypatch.setattr(Schema, "decode", decode)
    result = inspect_direct_output_abi(path, library, spec)
    assert result["status"] == "PASSED", result
    assert result["observed_spec"] == spec and result["declared_spec"] == spec
    assert result["changes"] == [] and result["ordered_output_count"] == 3
    assert result["omc"]["sha256"] == hashlib.sha256(before[0]).hexdigest()
    assert result["schema"]["library_sha256"] == hashlib.sha256(before[1]).hexdigest()
    assert calls == ["ModelDef"]
    assert (path.read_bytes(), library.read_bytes(), spec) == before
    assert "CALLER_MUST_BIND_AIR" in result["source_binding"]


def test_extra_half_to_float_is_reported_for_all_aliases_without_rewriting_declared(case):
    path, library, spec = case
    _write(case, dtypes=(3, 7, 1))
    before = deepcopy(spec)
    result = inspect_direct_output_abi(path, library, spec)
    assert result["status"] == "SOURCE_DTYPE_CHANGED", result
    assert result["source_output_precision"] == "SOURCE_DTYPE_CHANGED"
    assert result["declared_spec"] == spec == before
    assert [row["dtype"] for row in result["observed_spec"]] == ["int32", "float16", "float32", "float32"]
    assert result["changes"] == [{"kind": "DIAGNOSTIC_OUTPUT_PROMOTED_FP16_TO_FP32", "output_index": 2,
        "source": "v19", "semantic_names": ["noise_embedding", "layer_0_input"],
        "declared_dtype": "float16", "observed_dtype": "float32", "shape": [1, 16, 2048]}]
    assert result["physical_output_bytes"] == 15 * 4 + 15 * 2048 * 2 + 16 * 2048 * 4


@pytest.mark.parametrize("damage", ["top1", "hidden", "unknown", "missing_dtype", "shape", "count",
                                     "producer_dtype", "refs", "missing_producer", "bad_port", "two_netout"])
def test_forbidden_or_unbound_physical_outputs_do_not_return_an_observed_spec(case, damage):
    path, library, spec = case
    if damage == "top1":
        _write(case, dtypes=(1, 7, 7))
    elif damage == "hidden":
        _write(case, dtypes=(3, 1, 7))
    elif damage == "unknown":
        _write(case, dtypes=(3, 7, 99))
    elif damage == "missing_dtype":
        _write(case, missing_dtype=2)
    elif damage == "shape":
        _write(case, shapes=[[1, 15], [1, 15, 2048], [1, 2048, 16]])
    elif damage == "count":
        _write(case, dtypes=(3, 7), shapes=[[1, 15], [1, 15, 2048]])
    else:
        _write(case, damage=damage)
    result = inspect_direct_output_abi(path, library, spec)
    assert result["status"] == "FAILED", result
    assert result["observed_spec"] is None and result["changes"] == []
    assert result["reason"]


def test_extra_float_to_half_and_int_type_changes_remain_failures(case):
    path, library, spec = case
    for requested, actual in (("float32", 7), ("int32", 1)):
        for row in spec[2:]:
            row["dtype"] = requested
        _write(case, dtypes=(3, 7, actual))
        result = inspect_direct_output_abi(path, library, spec)
        assert result["status"] == "FAILED" and result["observed_spec"] is None


@pytest.mark.parametrize("damage", ["names", "alias_source", "alias_shape", "alias_dtype", "alias_index", "header", "noncontiguous"])
def test_malformed_declared_alias_contract_fails_before_opening_omc(case, damage):
    path, library, spec = case
    if damage == "names":
        spec[-1]["name"] = spec[-2]["name"]
    elif damage == "alias_source":
        spec[-1]["source"] = "different_value"
    elif damage == "alias_shape":
        spec[-1]["shape"] = [1, 8, 4096]
    elif damage == "alias_dtype":
        spec[-1]["dtype"] = "float32"
    elif damage == "alias_index":
        spec[-1]["output_index"] = 3
    elif damage == "header":
        spec[0]["dtype"] = "float32"
    else:
        spec[-1] = {**spec[-1], "source": "other", "output_index": 4}
    result = inspect_direct_output_abi(path, library, spec)
    assert result["status"] == "FAILED" and result["observed_spec"] is None
    assert "omc" not in result


def test_control_references_do_not_shift_ordered_output_indices(case):
    path, library, spec = case
    _write(case, damage="control")
    assert inspect_direct_output_abi(path, library, spec)["status"] == "PASSED"


def test_reader_does_not_use_stale_source_names_for_missing_explicit_edges(case):
    path, library, spec = case
    _write(case, damage="stale_metadata")
    raw = path.read_bytes().replace(b"opaque_partition:0", b"missing_valuexxx:0")
    path.write_bytes(raw)
    result = inspect_direct_output_abi(path, library, spec)
    assert result["status"] == "FAILED"
