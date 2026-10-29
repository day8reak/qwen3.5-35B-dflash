"""Reference reuse and native-dtype fixtures for bounded direct observations."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re

import numpy as np

from .artifacts import output_path, save_json
from .draft_board import _file, _record
from .hashing import sha256_file


def _metadata(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 << 20:
        raise ValueError("invalid direct observation metadata: " + str(path))
    return json.loads(path.read_text())


def _physical_spec(spec):
    """Validate a bounded semantic-to-physical map without fixed host test widths."""
    if not isinstance(spec, list) or not 2 <= len(spec) <= 256:
        raise ValueError("direct specification requires the two terminal outputs")
    physical, names, sources = {}, set(), {}
    for row in spec:
        if (not isinstance(row, dict) or set(row) != {"name", "source", "shape", "dtype", "output_index"}
                or not isinstance(row["name"], str) or row["name"] in names
                or not isinstance(row["source"], str) or not row["source"]
                or not isinstance(row["shape"], list) or not 1 <= len(row["shape"]) <= 4
                or any(type(dim) is not int or dim <= 0 for dim in row["shape"])
                or row["dtype"] not in {"float16", "float32", "int32"}
                or type(row["output_index"]) is not int or not 0 <= row["output_index"] < 16):
            raise ValueError("invalid direct semantic/physical descriptor")
        index = row["output_index"]
        descriptor = {key: row[key] for key in ("source", "shape", "dtype")}
        if index in physical and physical[index] != descriptor:
            raise ValueError("direct aliases disagree on their physical output descriptor")
        if row["source"] in sources and sources[row["source"]] != index:
            raise ValueError("one direct source must use one physical output")
        names.add(row["name"])
        physical[index], sources[row["source"]] = descriptor, index
    if set(physical) != set(range(len(physical))):
        raise ValueError("direct physical output indices must be contiguous")
    if (spec[0]["name"], spec[0]["source"], spec[0]["output_index"], spec[0]["dtype"]) != ("draft_top1", "draft_top1", 0, "int32"):
        raise ValueError("direct Top1 terminal contract changed")
    if (spec[1]["name"], spec[1]["source"], spec[1]["output_index"], spec[1]["dtype"]) != ("draft_hidden", "draft_hidden", 1, "float16"):
        raise ValueError("direct hidden terminal contract changed")
    if sum(math.prod(row["shape"]) * (2 if row["dtype"] == "float16" else 4) for row in physical.values()) > 64 << 20:
        raise ValueError("direct output bytes exceed 64 MiB")
    return [physical[index] for index in range(len(physical))]


def prepare_direct_reference(job, prepared, fixture, output, *, names, progress=None):
    """Reuse the verified CPU trace when available; never regenerate inputs."""
    from .draft_reference import DRAFT_REFERENCE_CONTRACT
    from .draft_layer_trace import trace_prepared_draft
    job, prepared, fixture, output = map(Path, (job, prepared, fixture, output))
    names = tuple(names)
    if not names or len(set(names)) != len(names) or len(names) > 256:
        raise ValueError("direct reference names must be nonempty and unique")
    prior = job / "layer-trace"
    receipt_path = prior / "completed-reference.json"
    if not receipt_path.exists():
        return trace_prepared_draft(prepared, fixture, output, names=names, progress=progress)
    owner = _metadata(prior / "owner.json")
    if (owner.get("source_job_sha256") != sha256_file(job / "result.json")
            or owner.get("code", {}).get("draft_layer_trace.py") != sha256_file(Path(__file__).with_name("draft_layer_trace.py"))
            or owner.get("code", {}).get("draft_reference.py") != sha256_file(Path(__file__).with_name("draft_reference.py"))):
        raise ValueError("retained CPU trace does not bind this source job/reference implementation")
    receipt = _metadata(receipt_path)
    if receipt.get("path") != "report.json":
        raise ValueError("retained CPU trace has an unexpected receipt filename")
    report_path = _file(prior / "reference", receipt)
    report = _metadata(report_path)
    original = _metadata(fixture / "report.json")
    if (report.get("status") != "REFERENCE_TRACE_COMPLETE"
            or report.get("reference_contract") != DRAFT_REFERENCE_CONTRACT
            or report.get("prepared_manifest_sha256") != sha256_file(prepared / "prepared.json")
            or report.get("fixture_report_sha256") != sha256_file(fixture / "report.json")
            or report.get("inputs") != original.get("inputs")):
        raise ValueError("retained CPU trace differs from the original prepared/fixture identities")
    reproduced = {row["name"]: row for row in report.get("golden_reproduction", [])}
    for name, key in (("draft_top1", "expected_top1"), ("draft_hidden", "expected_hidden")):
        if (reproduced.get(name, {}).get("status") != "EXACT_MATCH"
                or reproduced[name].get("sha256") != original[key]["sha256"]):
            raise ValueError("retained CPU trace did not reproduce the original golden: " + name)
    records = {row["name"]: row for row in report["tensors"]}
    if len(records) != len(report["tensors"]):
        raise ValueError("retained CPU trace has duplicate probe names")
    if any(name not in records for name in names):
        raise ValueError("retained CPU trace is missing a selected direct output")
    output = output_path(output)
    output.mkdir()
    copied = []
    from .existing_air import _snapshot
    for index, name in enumerate(names):
        row = records[name]
        source = _file(prior / "reference", row)
        destination = output / source.name
        try:
            os.link(source, destination)
        except OSError:
            _snapshot(source, destination)
        copied.append({**row, "index": index, **_record(destination, output)})
    result = {**report, "tensors": copied, "total_bytes": sum(row["bytes"] for row in copied),
              "reused_from": str(report_path), "reused_report_sha256": sha256_file(report_path)}
    save_json(output / "report.json", result)
    if progress:
        progress("Reused retained CPU probe values and byte-exact golden reproduction; no CPU model replay")
    return result


def prepare_direct_fixture(prepared, fixture, reference, reference_root, spec, output):
    """Construct ordered raw output goldens without a packed observer tensor."""
    from .existing_air import _snapshot
    prepared, fixture, reference_root, output = map(Path, (prepared, fixture, reference_root, output))
    _physical_spec(spec)
    original = _metadata(fixture / "report.json")
    if original["prepared_manifest_sha256"] != sha256_file(prepared / "prepared.json"):
        raise ValueError("direct fixture does not bind prepared weights")
    output = output_path(output)
    output.mkdir()
    result = {**original, "inputs": [], "direct_spec": spec, "diagnostic_only": True,
              "original_fixture_sha256": sha256_file(fixture / "report.json")}
    for row in original["inputs"]:
        source = _file(fixture, row)
        path = output / source.name
        _snapshot(source, path)
        result["inputs"].append({**row, **_record(path, output)})
    records = {row["name"]: row for row in reference["tensors"]}
    if len(records) != len(reference["tensors"]):
        raise ValueError("direct reference contains duplicate probes")
    expected, seen = [], {}
    for row in spec:
        record = records[row["name"]]
        if any(record[key] != row[key] for key in ("shape", "dtype")):
            raise ValueError("direct reference descriptor differs from GE output: " + row["name"])
        source = _file(reference_root, record)
        index = row["output_index"]
        if index in seen:
            if seen[index] != (record["bytes"], record["sha256"]):
                raise ValueError("same-value direct aliases differ in the CPU reference")
            continue
        if index != len(expected):
            raise ValueError("direct output reference order is not contiguous")
        path = output / ("expected_direct_%04d.bin" % index)
        _snapshot(source, path)
        expected.append({**record, "index": index, "name": row["source"], **_record(path, output)})
        seen[index] = (record["bytes"], record["sha256"])
    result["expected_direct"] = expected
    for row, key in zip(expected[:2], ("expected_top1", "expected_hidden")):
        if any(row.get(field) != original.get(key, {}).get(field) for field in ("shape", "dtype", "bytes", "sha256")):
            raise ValueError("direct terminal reference differs from the original golden")
    result["expected_top1"], result["expected_hidden"] = expected[:2]
    save_json(output / "report.json", result)
    return result


def prepare_observed_direct_fixture(source_fixture_dir, output_dir, binding):
    """Materialize explicitly bound wire dtypes while retaining source goldens.

    Only additional FP16 diagnostic outputs may be returned as FP32 by the
    compiler. Their expected FP16 values are cast to FP32 in this new fixture;
    no source fixture/golden, model arithmetic or numerical gate is overwritten.
    """
    from .existing_air import _snapshot
    source, output = Path(source_fixture_dir), Path(output_dir)
    original = _metadata(source / "report.json")
    if (original.get("artifact_kind") != "qwen35-draft-reference" or original.get("status") != "REFERENCE_COMPLETE"
            or not isinstance(binding, dict)
            or binding.get("artifact_kind") != "qwen35-draft-direct-output-abi-inspection"
            or binding.get("status") not in {"PASSED", "SOURCE_DTYPE_CHANGED"}
            or binding.get("declared_spec") != original.get("direct_spec")):
        raise ValueError("direct wire binding does not describe the complete source fixture")
    declared, observed = binding["declared_spec"], binding.get("observed_spec")
    declared_outputs, observed_outputs = _physical_spec(declared), _physical_spec(observed)
    if len(declared) != len(observed) or len(declared_outputs) != len(observed_outputs):
        raise ValueError("direct wire binding changed output/alias counts")
    for expected, actual in zip(declared, observed):
        if any(expected[key] != actual[key] for key in ("name", "source", "shape", "output_index")):
            raise ValueError("direct wire binding changed output source, order or shape")
        if expected["dtype"] != actual["dtype"] and not (
                expected["output_index"] >= 2 and expected["dtype"] == "float16" and actual["dtype"] == "float32"):
            raise ValueError("direct wire binding changed a fixed terminal or unsupported dtype")
    changes = []
    for index, (expected, actual) in enumerate(zip(declared_outputs, observed_outputs)):
        if expected["dtype"] != actual["dtype"]:
            changes.append({"kind": "DIAGNOSTIC_OUTPUT_PROMOTED_FP16_TO_FP32", "output_index": index,
                            "source": expected["source"], "semantic_names": [row["name"] for row in declared if row["output_index"] == index],
                            "declared_dtype": expected["dtype"], "observed_dtype": actual["dtype"], "shape": expected["shape"]})
    if (binding.get("changes") != changes or (binding["status"] == "PASSED") != (not changes)):
        raise ValueError("direct output changes do not match the observed wire descriptors")
    omc = binding.get("omc", {})
    if (type(omc.get("bytes")) is not int or not 0 < omc["bytes"] <= (1 << 32) - 1
            or not isinstance(omc.get("sha256"), str) or not re.fullmatch("[0-9a-f]{64}", omc["sha256"])):
        raise ValueError("direct wire binding has no well-formed compiled OMC identity")
    records = original.get("expected_direct", [])
    if len(records) != len(declared_outputs) or len(original.get("inputs", [])) != 3:
        raise ValueError("direct source fixture has incomplete expected outputs or inputs")
    # Validate all source extents before creating the replacement fixture.
    for index, (row, desc) in enumerate(zip(records, declared_outputs)):
        if row.get("index") != index or row.get("name") != desc["source"] or any(row.get(key) != desc[key] for key in ("shape", "dtype")):
            raise ValueError("source direct golden descriptor differs from its original declaration")
        if row.get("bytes") != math.prod(desc["shape"]) * (2 if desc["dtype"] == "float16" else 4):
            raise ValueError("source direct golden byte extent differs")
        _file(source, row)
    for row in original["inputs"]:
        _file(source, row)
    source_digest = sha256_file(source / "report.json")
    output = output_path(output)
    output.mkdir()
    result = {**original, "inputs": [], "direct_spec": observed, "declared_direct_spec": declared,
              "direct_output_binding": binding, "source_direct_fixture_sha256": source_digest,
              "wire_precision_scope": "Expected FP16 diagnostic values cast to observed FP32 wire dtype only; original precision equivalence is not established"}
    for row in original["inputs"]:
        destination = output / row["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        _snapshot(_file(source, row), destination)
        result["inputs"].append({**row, **_record(destination, output)})
    expected = []
    for index, (row, actual) in enumerate(zip(records, observed_outputs)):
        destination = output / ("expected_direct_%04d.bin" % index)
        source_path = _file(source, row)
        if row["dtype"] == actual["dtype"]:
            _snapshot(source_path, destination)
        else:
            np.fromfile(source_path, dtype="<f2").astype("<f4").tofile(destination)
        expected.append({**row, "dtype": actual["dtype"], "declared_dtype": row["dtype"],
                         "source_sha256": row["sha256"], **_record(destination, output)})
    if sha256_file(source / "report.json") != source_digest:
        raise ValueError("source direct fixture changed during wire adaptation")
    result["expected_direct"] = expected
    result["expected_top1"], result["expected_hidden"] = expected[:2]
    save_json(output / "report.json", result)
    return result


def read_direct_values(root, native, spec):
    """Read only retrieved hash-bound raw tensors, retaining native dtypes."""
    root = Path(root)
    physical = _physical_spec(spec)
    outputs = native["outputs"]
    if not isinstance(outputs, list) or len(outputs) != len(physical):
        raise ValueError("direct native output count differs from its specification")
    cached, values = {}, {}
    for row in spec:
        index = row["output_index"]
        record = outputs[index]
        if record.get("index") != index or any(record[key] != row[key] for key in ("dtype", "shape")) or record["name"] != row["source"]:
            raise ValueError("direct observation differs from its declared source output")
        if record.get("bytes") != math.prod(row["shape"]) * (2 if row["dtype"] == "float16" else 4):
            raise ValueError("direct observation byte count differs from its wire descriptor")
        if index not in cached:
            dtype = np.dtype({"float16": "<f2", "float32": "<f4", "int32": "<i4"}[row["dtype"]])
            cached[index] = np.fromfile(_file(root, record), dtype=dtype).reshape(row["shape"])
        values[row["name"]] = cached[index]
    return values
