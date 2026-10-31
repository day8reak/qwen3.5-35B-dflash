"""Prepare reusable full-target logical quantized weights, before hardware packing.

No model execution, NPU format conversion, or GE ABI is inferred here. Each
completed tensor is independently recorded so a stopped conversion can resume
into a new directory without rewriting or trusting partial old payloads.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil

from qwen35_moe_dflash.common.artifacts import checked_file, output_path, save_json
from qwen35_moe_dflash.graph.quant_weights import write_embedding_int8, write_fp16, write_int4
from qwen35_moe_dflash.graph.weights import SafeTensorFiles
from qwen35_moe_dflash.common.hashing import sha256_file
from qwen35_moe_dflash.common.resources import repository_root


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _locked_target(root):
    lock = json.loads((repository_root() / "specs/upstream.lock.json").read_text())[0]
    files = []
    for name, expected in lock["files"].items():
        if "sha256" not in expected:
            continue
        path = root / name
        if not path.is_file() or sha256_file(path) != expected["sha256"]:
            raise ValueError(f"target checkpoint {name} differs from locked revision {lock['revision']}")
        files.append({"path": name, "sha256": expected["sha256"], "bytes": path.stat().st_size})
    return {"repo": lock["repo"], "revision": lock["revision"], "metadata_files": files,
            "tensor_identity": "SHA256 of every consumed raw source tensor; not a full-shard hash"}


def _source_hash(reader, spec):
    path, offset, count, dtype = reader.tensor(spec["source_tensor"], spec["shape"])
    if dtype != "BF16":
        raise ValueError("target checkpoint must contain the locked BF16 source weights")
    digest = hashlib.sha256()
    remaining = count * 2
    with path.open("rb") as stream:
        stream.seek(offset)
        while remaining:
            data = stream.read(min(8 << 20, remaining))
            if not data:
                raise ValueError("source tensor was truncated during resume")
            digest.update(data)
            remaining -= len(data)
    return digest.hexdigest()


def _stat(path):
    s = path.stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def _link_or_copy(source, destination):
    try:
        os.link(source, destination)
    except OSError:
        with source.open("rb") as src, destination.open("xb") as dst:
            shutil.copyfileobj(src, dst, 8 << 20)


def _resume_tensor(old, record_path, spec, reader, destination, index):
    if record_path.is_symlink():
        raise ValueError("resume tensor record must be a regular file")
    record = json.loads(record_path.read_text())
    if (record.get("spec") != spec or record.get("status") != "TENSOR_PREPARED"
            or record.get("record_sha256") != _identity({k: v for k, v in record.items()
                                                       if k != "record_sha256"})):
        raise ValueError("resume tensor record does not match the planned tensor")
    converted = deepcopy(record["converted"])
    if converted["source"]["sha256"] != _source_hash(reader, spec):
        raise ValueError(f"resume checkpoint tensor changed: {spec['key']}")
    for field, suffix, expected in (("weights", "weight", spec["weight_bytes"]),
                                     ("scales", "scale", spec["scale_bytes"])):
        if not expected:
            if field in converted:
                raise ValueError("unexpected scale payload in retained FP16 tensor")
            continue
        item = converted[field]
        if item["bytes"] != expected:
            raise ValueError("resume payload size differs from plan")
        # The format stays explicit: an old logical cache is never adopted as
        # I4ZN/QPARAM or passed to the existing FP16 prepared-model reader.
        if converted.get("device_ready") is not False or converted.get("device_pack") != "NOT_PERFORMED":
            raise ValueError("resume requires portable logical weight records")
        source = checked_file(old, item)
        target = destination / "weights" / f"{index:05d}.{suffix}.bin"
        _link_or_copy(source, target)
        item["path"] = str(target.relative_to(destination))
    return converted


def prepare_target_weights(target_weights, output_dir, *, resume=None, progress=None):
    """Stream all target weights according to the fixed document W4A8 plan.

    Resume verifies both current raw checkpoint bytes and retained output bytes,
    then hardlinks/copies completed tensors. No file in the old run is edited.
    """
    from qwen35_moe_dflash.graph.quant_plan import target_quantization_plan

    checkpoint = Path(target_weights).resolve()
    source = _locked_target(checkpoint)
    plan = target_quantization_plan()
    reader = SafeTensorFiles(checkpoint)
    snapshots = {}
    for spec in plan["tensors"]:
        path, _, _, dtype = reader.tensor(spec["source_tensor"], spec["shape"])
        if dtype != "BF16":
            raise ValueError(f"{spec['key']}: expected original BF16 source")
        snapshots[path] = _stat(path)
    contract = {"artifact_kind": "qwen35-portable-quant-weights", "schema_version": 1,
                "source": source, "plan": plan,
                "arithmetic": {"source": "cast BF16 to FP16 before quantization",
                               "statistics": "FP32", "rounding": "ties-to-even",
                               "weight_group_size": 256, "zero_group_scale": 1.0,
                               "norm_policy": "Original HF norm weights; no extra folding"}}
    contract_sha = _identity(contract)
    old = Path(resume).resolve() if resume is not None else None
    if old is not None:
        retained = json.loads((old / "plan.json").read_text())
        if retained.get("contract_sha256") != contract_sha or retained.get("contract") != contract:
            raise ValueError("resume source/quantization plan differs from current preparation")
    root = output_path(output_dir)
    if old is not None and (root == old or old in root.parents or root in old.parents):
        raise ValueError("resume output must be separate from the retained run")
    root.mkdir(parents=True, exist_ok=False)
    (root / "weights").mkdir()
    (root / "tensors").mkdir()
    save_json(root / "plan.json", {"contract": contract, "contract_sha256": contract_sha,
                                  "checkpoint_root": str(checkpoint)})
    results = []
    stage = "prepare-weights"
    try:
        # No hidden full-FP16 intermediate, model download, or draft checkpoint.
        needed = sum(x["weight_bytes"] + x["scale_bytes"] for x in plan["tensors"])
        if old is None and shutil.disk_usage(root).free < needed + (64 << 20):
            raise OSError(f"logical target weight cache needs {needed} bytes plus 64 MiB free space")
        for index, spec in enumerate(plan["tensors"]):
            old_record = old / "tensors" / f"{index:05d}.json" if old else None
            reused = old_record is not None and old_record.exists()
            if reused:
                converted = _resume_tensor(old, old_record, spec, reader, root, index)
            else:
                weights = root / "weights" / f"{index:05d}.weight.bin"
                scales = root / "weights" / f"{index:05d}.scale.bin"
                args = (reader, spec["source_tensor"], spec["shape"], weights)
                if spec["kind"] in {"linear_int4", "expert_int4"}:
                    converted = write_int4(*args, scales)
                elif spec["kind"] == "embedding_int8":
                    converted = write_embedding_int8(*args, scales)
                elif spec["kind"] == "float16":
                    converted = write_fp16(*args)
                else:
                    raise ValueError("unsupported weight plan role")
                for field in ("weights", "scales"):
                    if field in converted:
                        converted[field]["path"] = str(Path(converted[field]["path"]).relative_to(root))
            if converted["weights"]["bytes"] != spec["weight_bytes"] or converted.get("scales", {}).get("bytes", 0) != spec["scale_bytes"]:
                raise ValueError("prepared tensor size differs from plan")
            record = {"status": "TENSOR_PREPARED", "spec": spec, "converted": converted,
                      "reused_from": str(old_record) if reused else None}
            record["record_sha256"] = _identity(record)
            record_path = root / "tensors" / f"{index:05d}.json"
            save_json(record_path, record)
            results.append({"key": spec["key"], "path": str(record_path.relative_to(root)),
                            "bytes": record_path.stat().st_size, "sha256": sha256_file(record_path)})
            if progress:
                progress({"completed": index + 1, "total": len(plan["tensors"]),
                          "tensor": spec["key"], "reused": reused})
        stage = "source-recheck"
        if any(_stat(path) != before for path, before in snapshots.items()) or _locked_target(checkpoint) != source:
            raise ValueError("checkpoint changed during weight preparation")
        result = {"schema_version": 1, "artifact_kind": "qwen35-portable-quant-weights",
                  "status": "WEIGHTS_PREPARED_LAYOUT_PENDING", "contract_sha256": contract_sha,
                  "checkpoint_root": str(checkpoint), "source": source,
                  "summary": plan["summary"], "tensors": results, "device_ready": False,
                  "device_pack": "NOT_PERFORMED", "graph_export": "NOT_RUN",
                  "omg_compile": "NOT_RUN", "device_execution": "NOT_RUN",
                  "pending": ["Verified internal-DDK I4ZN and QPARAM conversion",
                              "Quantized GE graph integration and target compilation",
                              "Complete model numerical validation"]}
        save_json(root / "manifest.json", result)
        return result
    except Exception as error:
        save_json(root / "failure.json", {"status": "FAILED", "stage": stage,
                                          "completed_tensors": len(results),
                                          "error": f"{type(error).__name__}: {error}",
                                          "resume": "Use --resume with this weights directory; partial tensors without records are not reused"})
        raise
