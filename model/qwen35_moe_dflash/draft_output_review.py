"""Review retained direct observations and one small norm weight, without replay.

The packed multi-GB model and projection weights are deliberately not read.
Recorded bundle/native model identities are compared; this does not rehash the
executed model or prove where its output values originated inside the runtime.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .config import Qwen35DFlashConfig
from .draft_direct_job import _physical_spec
from .draft_trace_integrity import _Reads


def review_direct_output_math(job, *, allow_host_fake=False):
    from .draft_output_math import diagnose_output_relations

    root, read = Path(job).absolute(), _Reads()
    owner = read.json(root / "owner.json")
    if owner.get("artifact_kind") != "qwen35-draft-direct-trace-job-owner":
        raise ValueError("output review requires the retained direct owner")
    source_root = Path(owner.get("job", ""))
    if not source_root.is_absolute():
        raise ValueError("direct source job must have an absolute retained root")
    source_raw = read.bytes(source_root / "result.json")
    if hashlib.sha256(source_raw).hexdigest() != owner.get("source_job_sha256"):
        raise ValueError("direct owner does not bind the retained source job")
    source = json.loads(source_raw)
    component = source.get("components", {}).get("prepared", {})
    prepared_root = Path(component.get("directory", ""))
    if not prepared_root.is_absolute():
        raise ValueError("direct source is missing its prepared component")
    prepared_raw = read.bound(prepared_root, component["manifest"])
    prepared = json.loads(prepared_raw)
    if (prepared.get("artifact_kind") != "qwen35-draft-ge-prepared"
            or prepared.get("status") != "PREPARED"):
        raise ValueError("output review requires successful original preparation")
    cfg = Qwen35DFlashConfig.from_dict(prepared["draft_config"])

    def stage(name, filename):
        receipt = read.json(root / ("completed-" + name + ".json"))
        if receipt.get("path") != filename:
            raise ValueError("unexpected completed direct " + name + " receipt")
        raw = read.bound(root / name, receipt)
        return json.loads(raw), hashlib.sha256(raw).hexdigest()

    fixture, _ = stage("fixture", "report.json")
    observed, observed_sha = stage("observed-fixture", "report.json")
    bundle, bundle_sha = stage("bundle", "manifest.json")
    board, _ = stage("board", "result.json")
    if (fixture.get("prepared_manifest_sha256") != hashlib.sha256(prepared_raw).hexdigest()
            or fixture.get("direct_spec") != owner.get("trace_spec")
            or observed.get("declared_direct_spec") != owner.get("trace_spec")
            or observed.get("direct_spec") != bundle.get("direct_spec")
            or bundle.get("draft_config") != prepared["draft_config"]
            or bundle.get("fixture_report_sha256") != observed_sha
            or board.get("bundle_manifest_sha256") != bundle_sha
            or board.get("device_execution") != "PASSED"):
        raise ValueError("retained direct fixture, bundle or board identities differ")
    # Keep the original and physical fixtures independently bound. No golden
    # file is regenerated or changed when an extra diagnostic dtype is promoted.
    fixture_raw = read.bound(root / "fixture", read.json(root / "completed-fixture.json"))
    if observed.get("source_direct_fixture_sha256") != hashlib.sha256(fixture_raw).hexdigest():
        raise ValueError("physical fixture does not bind the original direct fixture")
    transport = board.get("retrieved_sha256", {})
    native_raw = read.bytes(root / "board/results/result.json")
    if hashlib.sha256(native_raw).hexdigest() != transport.get("results/result.json"):
        raise ValueError("native output review result differs from transport receipt")
    native = json.loads(native_raw)
    kind = native.get("evidence_kind")
    if kind != "device-hiai" and not (allow_host_fake and kind == "host-fake-hiai"):
        raise ValueError("retained device HIAI result is required")
    identities = {"scope": "standalone-draft-step", "status": "EXECUTED", "target_calls": 0,
                  "draft_calls": 1, "model_sha256": bundle["model"]["sha256"],
                  "draft_abi_sha256": bundle["draft_abi"]["sha256"],
                  "step_plan_sha256": bundle["step_plan"]["sha256"], "ddk_id": bundle["ddk_id"]}
    if any(native.get(key) != value for key, value in identities.items()):
        raise ValueError("native output review identity differs from the bundle")
    diagnostic = native.get("io_diagnostic", {})
    if diagnostic.get("process_called") is not True:
        raise ValueError("output review requires recorded Process execution")
    inputs, readbacks = native.get("inputs", []), diagnostic.get("inputs", [])
    if len(inputs) != 3 or len(readbacks) != 3 or len(bundle.get("inputs", [])) != 3:
        raise ValueError("output review requires three source-bound input readbacks")
    for actual, expected, back in zip(inputs, bundle["inputs"], readbacks):
        if (any(actual.get(k) != expected.get(k) for k in ("index", "name", "shape", "dtype", "bytes", "sha256"))
                or back.get("index") != expected["index"] or back.get("readback_matches_source") is not True
                or back.get("readback_sha256") != expected["sha256"]):
            raise ValueError("native output review input readback differs")
    spec = bundle["direct_spec"]
    physical = _physical_spec(spec)
    outputs, buffers = native.get("outputs", []), diagnostic.get("outputs", [])
    if len(outputs) != len(physical) or len(buffers) != len(physical):
        raise ValueError("output review physical port count differs")
    arrays = []
    for index, (desc, row, buffer) in enumerate(zip(physical, outputs, buffers)):
        if (row.get("index") != index or row.get("name") != desc["source"]
                or any(row.get(key) != desc[key] for key in ("dtype", "shape"))
                or buffer.get("index") != index
                or any(buffer.get(key) is not True for key in ("checked_after_process", "object_same", "buffer_same"))
                or transport.get("results/" + row["path"]) != row.get("sha256")):
            raise ValueError("output review wire descriptor or transport receipt differs")
        dtype = np.dtype({"float16": "<f2", "float32": "<f4", "int32": "<i4"}[desc["dtype"]])
        raw = read.bound(root / "board/results", row, limit=64 << 20)
        if len(raw) != int(np.prod(desc["shape"], dtype=object)) * dtype.itemsize:
            raise ValueError("output review byte extent differs from shape/dtype")
        arrays.append(np.frombuffer(raw, dtype=dtype).reshape(desc["shape"]))
    values = {row["name"]: arrays[row["output_index"]] for row in spec}
    key = "draft.layers.1.input_layernorm.weight"
    weight = prepared.get("weights", {}).get(key)
    if cfg.num_hidden_layers < 2 or not isinstance(weight, dict):
        raise ValueError("layer-one norm weight is unavailable")
    if weight.get("dtype") != "float16" or weight.get("shape") != [cfg.hidden_size]:
        raise ValueError("layer-one norm weight descriptor differs")
    gamma_raw = read.bound(prepared_root, weight, limit=64 << 10)
    gamma = np.frombuffer(gamma_raw, dtype="<f2")
    result = diagnose_output_relations(values, cfg, gamma)
    read.unchanged()
    result.update(artifact_kind="qwen35-draft-retained-output-math-review",
                  norm_weight={"key": key, "bytes": len(gamma_raw), "sha256": weight["sha256"]},
                  model_identity_scope="Recorded manifest/native identities agree; multi-GB model bytes not rehashed")
    return result
