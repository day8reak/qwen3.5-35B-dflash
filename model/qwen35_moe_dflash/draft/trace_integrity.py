"""Read-only internal consistency checks for retained draft observations.

Equal observations of one original value and the independent native Top1 port
are exact invariants. Arithmetic relations are diagnostic unless they contain a
rounding-independent contradiction. These checks do not introduce a model
accuracy tolerance or turn a first non-exact observation into a root cause.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from qwen35_moe_dflash.config import Qwen35DFlashConfig


def _same_values(expected, actual):
    finite = np.isfinite(expected) & np.isfinite(actual)
    matching_inf = ((np.isposinf(expected) & np.isposinf(actual)) |
                    (np.isneginf(expected) & np.isneginf(actual)))
    with np.errstate(all="ignore"):
        difference = actual[finite].astype(np.float64) - expected[finite].astype(np.float64)
    error = float(np.linalg.norm(difference))
    norm = float(np.linalg.norm(expected[finite].astype(np.float64)))
    return {"exact_match": bool(np.array_equal(expected, actual)),
            "finite_pair_count": int(np.count_nonzero(finite)),
            "nonfinite_mismatch_count": int(np.count_nonzero(~(finite | matching_inf))),
            "max_abs_error": float(np.max(np.abs(difference))) if difference.size else None,
            "relative_l2": error / norm if norm and math.isfinite(error / norm) else None,
            "unexpected_nonfinite_count": int(np.count_nonzero(np.isfinite(expected) & ~np.isfinite(actual)))}


def evaluate_trace_integrity(values, config, *, native_top1=None, alias_deduplicated=False):
    """Evaluate observed NumPy tensors without reading or writing artifacts.

    Missing observations produce UNAVAILABLE relations, permitting a bounded
    direct-output subset. ``alias_deduplicated`` marks aliases supplied by the
    same direct graph output as non-independent observations.
    """
    cfg = config if isinstance(config, Qwen35DFlashConfig) else Qwen35DFlashConfig.from_dict(config)
    if (not isinstance(values, dict) or len(values) > 256
            or any(not isinstance(name, str) or not isinstance(array, np.ndarray)
                   or array.dtype not in (np.dtype("float16"), np.dtype("float32"), np.dtype("int32"), np.dtype("bool"))
                   for name, array in values.items())
            or sum(array.nbytes for array in values.values()) > 64 << 20):
        raise ValueError("integrity checks require bounded named FP16/FP32/INT32/BOOL NumPy observations")
    checks, contradictions = [], []

    def relation(name, names, compute, *, exact_invariant=False, independent=True):
        missing = [item for item in names if item not in values]
        row = {"name": name, "observations": list(names), "independent_observations": independent}
        checks.append(row)
        if missing:
            row.update(status="UNAVAILABLE", missing=missing)
            return
        expected, actual, hard = compute(*(values[item] for item in names))
        if expected.shape != actual.shape:
            raise ValueError("integrity relation shape mismatch: " + name)
        metrics = _same_values(expected, actual)
        row.update(metrics)
        hard = list(hard)
        if exact_invariant and expected.tobytes() != actual.tobytes():
            hard.append("SAME_SOURCE_BYTES_DIFFER")
        if metrics["unexpected_nonfinite_count"]:
            hard.append("FINITE_EXPECTATION_HAS_NONFINITE_OBSERVATION")
        if hard:
            row.update(status="CONTRADICTION", contradictions=list(dict.fromkeys(hard)))
            contradictions.append({"relation": name, "observations": list(names), "reasons": row["contradictions"]})
        else:
            row["status"] = "EXACT" if metrics["exact_match"] else "DIFFERENT_DIAGNOSTIC"

    aliases = [("noise_embedding", "layer_0_input")]
    aliases.extend((f"layer_{i}_output", f"layer_{i + 1}_input") for i in range(cfg.num_hidden_layers - 1))
    for left, right in aliases:
        relation("same_source:" + left + "=" + right, (left, right),
                 lambda a, b: (a, b, []), exact_invariant=True, independent=not alias_deduplicated)

    if native_top1 is not None:
        native_top1 = np.asarray(native_top1)
        if native_top1.dtype != np.int32 or native_top1.shape != (1, cfg.block_size - 1):
            raise ValueError("native Top1 integrity input differs from the draft ABI")
        row = {"name": "independent_native_top1", "observations": ["draft_top1", "native_output_0000"],
               "independent_observations": True}
        checks.append(row)
        if "draft_top1" not in values:
            row.update(status="UNAVAILABLE", missing=["draft_top1"])
        else:
            observed = values["draft_top1"]
            if observed.shape != native_top1.shape:
                raise ValueError("packed Top1 differs from its independent native shape")
            row.update(_same_values(native_top1.astype(np.float32), observed.astype(np.float32)))
            row["status"] = "EXACT" if row["exact_match"] else "CONTRADICTION"
            if not row["exact_match"]:
                row["contradictions"] = ["INDEPENDENT_TOP1_PORT_DIFFERS"]
                contradictions.append({"relation": row["name"], "observations": row["observations"], "reasons": row["contradictions"]})

    def residual(left, right, observed):
        with np.errstate(all="ignore"):
            expected = (left.astype(np.float16) + right.astype(np.float16)).astype(np.float16)
        # Zero and exact cancellation cannot acquire a finite nonzero result
        # through a small change in rounding mode or subnormal handling.
        zero = (expected == 0) & np.isfinite(left) & np.isfinite(right)
        hard = ["EXACT_ZERO_SUM_HAS_NONZERO_OBSERVATION"] if np.any(zero & np.isfinite(observed) & (observed != 0)) else []
        return expected.astype(np.float32), observed.astype(np.float32), hard

    def score(qk, visible, observed):
        mask = np.broadcast_to(visible.astype(bool), qk.shape)
        with np.errstate(all="ignore"):
            expected = qk.astype(np.float16).astype(np.float32) * np.float32(cfg.head_dim ** -.5)
            expected = np.where(mask, expected, np.float32(-np.inf))
        hard = []
        if np.any(~mask & ~np.isneginf(observed)):
            hard.append("EXPLICIT_MASK_NEGATIVE_INFINITY_DIFFERS")
        if np.any(mask & (qk == 0) & np.isfinite(observed) & (observed != 0)):
            hard.append("ZERO_QK_HAS_NONZERO_SCALED_SCORE")
        return expected, observed.astype(np.float32), hard

    for layer in range(cfg.num_hidden_layers):
        prefix = f"layer_{layer}_"
        relation(prefix + "post_attention_add", (prefix + "input", prefix + "attention", prefix + "post_attention"), residual)
        relation(prefix + "output_add", (prefix + "post_attention", prefix + "mlp_down", prefix + "output"), residual)
        relation(prefix + "masked_score", (prefix + "qk_fp16", prefix + "visible", prefix + "scores_fp32"), score)
    norms = [("context_fc", "projected_context")]
    norms.extend((f"layer_{i}_input", f"layer_{i}_input_norm") for i in range(cfg.num_hidden_layers))
    norms.extend((f"layer_{i}_post_attention", f"layer_{i}_post_attention_norm") for i in range(cfg.num_hidden_layers))
    for source, normalized in norms:
        row = {"name": "zero_norm:" + source, "observations": [source, normalized]}
        checks.append(row)
        if source not in values or normalized not in values or cfg.rms_norm_eps <= 0:
            row.update(status="UNAVAILABLE")
            continue
        x, y = values[source], values[normalized]
        if x.shape != y.shape:
            raise ValueError("normalization integrity shapes differ")
        zero_rows = np.all(x == 0, axis=-1)
        bad = zero_rows & np.any(np.isfinite(y) & (y != 0), axis=-1)
        row.update(zero_input_rows=int(np.count_nonzero(zero_rows)), contradictory_rows=int(np.count_nonzero(bad)),
                   status="CONTRADICTION" if np.any(bad) else "NO_ZERO_INPUT_CONTRADICTION")
        if np.any(bad):
            row["contradictions"] = ["ZERO_NORM_INPUT_HAS_FINITE_NONZERO_OUTPUT"]
            contradictions.append({"relation": row["name"], "observations": row["observations"], "reasons": row["contradictions"]})
    return {"schema_version": 1, "artifact_kind": "qwen35-draft-trace-integrity", "status": "FAILED" if contradictions else "PASSED",
            "trace_reliability": "INTERNALLY_CONTRADICTORY" if contradictions else "NO_CHECKED_CONTRADICTION_OBSERVED",
            "alias_deduplicated": bool(alias_deduplicated), "checks": checks, "contradictions": contradictions,
            "contradiction_count": len(contradictions), "unavailable_count": sum(row["status"] == "UNAVAILABLE" for row in checks),
            "arithmetic_acceptance_threshold": "NOT_DEFINED",
            "root_cause_attribution": "UNAVAILABLE_WHILE_TRACE_CONTRADICTORY" if contradictions else "NOT_ESTABLISHED",
            "scope": "Observation consistency only; no model accuracy pass. Arithmetic non-exactness alone is diagnostic, not a new tolerance gate."}


class _Reads:
    def __init__(self):
        self.identities = {}

    def bytes(self, path, limit=16 << 20):
        path = Path(path).absolute()
        if path.resolve() != path or any(parent.is_symlink() for parent in (path, *path.parents)):
            raise ValueError("integrity review input must be a real non-symlink path")
        before = path.stat()
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        if not path.is_file() or not 0 <= before.st_size <= limit:
            raise ValueError("integrity review input exceeds its bounded size")
        raw = path.read_bytes()
        self.identities[path] = identity
        if len(raw) != before.st_size:
            raise ValueError("integrity review input changed while reading")
        return raw

    def json(self, path):
        return json.loads(self.bytes(path))

    def bound(self, root, row, limit=16 << 20):
        relative = Path(row["path"])
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError("integrity review receipt escapes its artifact root")
        raw = self.bytes(root / relative, limit)
        if type(row.get("bytes")) is not int or len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row.get("sha256"):
            raise ValueError("integrity review SHA256/size mismatch: " + str(relative))
        return raw

    def unchanged(self):
        for path, expected in self.identities.items():
            info = path.stat()
            if expected != (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns):
                raise ValueError("integrity review evidence changed while reading")


def review_trace_integrity(job_dir, *, allow_host_fake=False):
    """Review the retained packed trace without rebuilding, HDC or file writes.

    The small external compiler OMC is rehashed by the precision inspector.
    The multi-GB packed model is deliberately not reread: its manifest identity
    must agree with the hash-bound native result and transport receipt.
    """
    from qwen35_moe_dflash.draft.board import _trace_spec, _wire
    from qwen35_moe_dflash.draft.trace_inspect import inspect_retained_trace_precision

    job = Path(job_dir).absolute()
    inspected = inspect_retained_trace_precision(job)
    if inspected["compiled_precision"].get("status") != "PASSED":
        raise ValueError("integrity review requires the retained compiled precision pass")
    root, read = job / "layer-trace", _Reads()
    owner = read.json(root / "owner.json")
    precision_raw = read.bytes(root / "precision-check.json")
    if hashlib.sha256(precision_raw).hexdigest() != inspected.get("precision_report_sha256"):
        raise ValueError("precision findings changed after source-bound inspection")
    source_raw = read.bytes(job / "result.json")
    if owner["source_job_sha256"] != hashlib.sha256(source_raw).hexdigest():
        raise ValueError("integrity owner differs from its retained source job")
    source = json.loads(source_raw)
    source_reports, source_hashes = {}, {}
    for name in ("prepared", "fixture"):
        component = source.get("components", {}).get(name)
        if not isinstance(component, dict) or not isinstance(component.get("directory"), str):
            raise ValueError("trace source job is missing its " + name + " receipt")
        folder = Path(component["directory"])
        if not folder.is_absolute():
            raise ValueError("trace source component path must be absolute")
        raw = read.bound(folder, component["manifest"])
        source_reports[name], source_hashes[name] = json.loads(raw), hashlib.sha256(raw).hexdigest()
    trace_job = read.json(root / "result.json")
    if trace_job.get("artifact_kind") != "qwen35-draft-layer-trace-job":
        raise ValueError("expected retained complete trace job")
    stages, hashes = {}, {}
    for stage, file in (("fixture", "report.json"), ("air", "manifest.json"), ("compilation", "report.json"),
                        ("bundle", "manifest.json"), ("board", "result.json")):
        receipt = read.json(root / ("completed-" + stage + ".json"))
        if receipt.get("path") != file:
            raise ValueError("unexpected completed trace " + stage + " receipt")
        raw = read.bound(root / stage, receipt)
        stages[stage], hashes[stage] = json.loads(raw), hashlib.sha256(raw).hexdigest()
    fixture, bundle, board = (stages[name] for name in ("fixture", "bundle", "board"))
    compiler_omc = stages["compilation"].get("omc", {})
    if (type(compiler_omc.get("bytes")) is not int or not 0 < compiler_omc["bytes"] <= 64 << 20
            or not isinstance(compiler_omc.get("sha256"), str) or not re.fullmatch("[0-9a-f]{64}", compiler_omc["sha256"])):
        raise ValueError("retained small compiler OMC identity is malformed")
    if any(stages["compilation"].get("omc", {}).get(key) != inspected.get("omc", {}).get(key)
           for key in ("bytes", "sha256")):
        raise ValueError("retained compilation changed after precision inspection")
    spec = bundle.get("trace_spec")
    elements = _trace_spec(spec)
    if (bundle.get("artifact_kind") != "qwen35-standalone-draft-trace-bundle" or bundle.get("status") != "PREPARED"
            or bundle.get("source_identity") != "MATCHED_EXPORT_AND_REFERENCE"
            or spec != owner.get("trace_spec") or spec != fixture.get("trace_spec")
            or bundle.get("fixture_report_sha256") != hashes["fixture"]
            or bundle.get("export_manifest_sha256") != hashes["air"]
            or bundle.get("compiler_report_sha256") != hashes["compilation"]
            or fixture.get("original_fixture_sha256") != source_hashes["fixture"]
            or fixture.get("prepared_manifest_sha256") != source_hashes["prepared"]
            or stages["air"].get("prepared_manifest_sha256") != source_hashes["prepared"]
            or stages["air"].get("trace_spec") != spec
            or fixture.get("status") != "REFERENCE_COMPLETE" or fixture.get("draft_config") != bundle.get("draft_config")
            or any(report.get("draft_config") != bundle["draft_config"] for report in source_reports.values())):
        raise ValueError("trace bundle/fixture/spec identity mismatch")
    cfg = Qwen35DFlashConfig.from_dict(bundle["draft_config"])
    model = bundle.get("model", {})
    if (type(model.get("bytes")) is not int or not 0 < model["bytes"] <= (1 << 32) - 1
            or not isinstance(model.get("sha256"), str) or not re.fullmatch("[0-9a-f]{64}", model["sha256"])):
        raise ValueError("recorded packed model identity is malformed")
    from qwen35_moe_dflash.draft.layer_trace import trace_names
    if {row["name"] for row in spec} != set(trace_names(cfg)):
        raise ValueError("retained packed trace must cover the complete frozen probe set")
    if (len(bundle.get("inputs", [])) != 3 or len(fixture.get("inputs", [])) != 3
            or len(source_reports["fixture"].get("inputs", [])) != 3):
        raise ValueError("trace fixture requires the three ordered source inputs")
    for source_input, expected, actual in zip(source_reports["fixture"]["inputs"], fixture["inputs"], bundle["inputs"]):
        if any(expected.get(key) != source_input.get(key) or actual.get(key) != expected.get(key)
               for key in ("index", "name", "dtype", "shape", "bytes", "sha256")):
            raise ValueError("trace inputs differ from the frozen production fixture")
        read.bound(root / "bundle", actual, 64 << 20)
    if (board.get("artifact_kind") != "qwen35-standalone-draft-execution" or board.get("device_execution") != "PASSED"
            or board.get("bundle_manifest_sha256") != hashes["bundle"] or board.get("remote_exitcode") != 0
            or type(board.get("remote_exitcode")) is not int or board.get("transport_error") or board.get("retrieval_errors")
            or board.get("model_load_api") != "legacy-buffer" or board.get("ddk_id") != bundle.get("ddk_id")):
        raise ValueError("trace execution/transport did not complete with this bundle")
    abi_raw = read.bound(root / "bundle", bundle["draft_abi"])
    abi = json.loads(abi_raw)
    if (abi.get("artifact_kind") != "qwen35-standalone-draft-trace-abi" or abi.get("trace_spec") != spec
            or abi.get("inputs") != [{key: row[key] for key in ("index", "name", "dtype", "shape")} for row in bundle["inputs"]]
            or abi.get("outputs") != bundle.get("outputs") or abi.get("draft_config") != bundle["draft_config"]):
        raise ValueError("trace ABI differs from its bundle")
    wire = read.bound(root / "bundle", bundle["step_plan"])
    if wire != _wire(bundle):
        raise ValueError("trace wire differs from its bound bundle")
    if bundle.get("weight_storage") == "embedded-stream":
        packing = json.loads(read.bound(root / "bundle", bundle["weight_packing"]))
        if (packing.get("status") != "EMBEDDED" or any(packing.get("output", {}).get(key) != bundle["model"].get(key)
                                                       for key in ("bytes", "sha256"))):
            raise ValueError("trace model identity differs from its packing receipt")
    required = ("draft.stdout", "draft.stderr", "draft.exitcode", "results/result.json",
                "results/output_0000.bin", "results/output_0001.bin", "runtime.json", "runtime.stderr", "runtime.exitcode")
    receipts = board.get("retrieved_sha256", {})
    raw_files = {}
    for name in required:
        digest = receipts.get(name)
        if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
            raise ValueError("missing trace retrieval identity: " + name)
        raw = read.bytes(root / "board" / name, 64 << 20 if name.endswith(".bin") else 16 << 20)
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("changed trace retrieval bytes: " + name)
        raw_files[name] = raw
    if any(raw_files[name].strip() != b"0" for name in ("draft.exitcode", "runtime.exitcode")):
        raise ValueError("trace retained exit contradicts successful execution")
    native = json.loads(raw_files["results/result.json"])
    identity = {"schema_version": 1, "scope": "standalone-draft-step", "status": "EXECUTED", "target_calls": 0, "draft_calls": 1,
                "model_sha256": bundle["model"]["sha256"], "draft_abi_sha256": bundle["draft_abi"]["sha256"],
                "step_plan_sha256": bundle["step_plan"]["sha256"], "ddk_id": bundle["ddk_id"]}
    if any(native.get(key) != value for key, value in identity.items()) or any(type(native.get(key)) is not int for key in ("schema_version", "target_calls", "draft_calls")):
        raise ValueError("native trace identity differs from its bundle")
    if native.get("evidence_kind") != "device-hiai" and not (allow_host_fake and native.get("evidence_kind") == "host-fake-hiai"):
        raise ValueError("native device evidence required; host fake is test-only")
    if native.get("model_load_api", "legacy-buffer") != "legacy-buffer":
        raise ValueError("native trace loader differs")
    for key in ("runtime_version", "firmware_id"):
        if not isinstance(board.get(key), str) or not board[key] or native.get(key) != board[key]:
            raise ValueError("native trace runtime/firmware differs from execution")
    probe = json.loads(raw_files["runtime.json"])
    if probe.get("npu_supported") is not True or probe.get("runtime_version") != board["runtime_version"]:
        raise ValueError("trace runtime capability differs from execution")
    if len(native.get("inputs", [])) != 3 or len(native.get("outputs", [])) != 2:
        raise ValueError("native trace IO counts differ")
    for actual, expected in zip(native["inputs"], bundle["inputs"]):
        if any(actual.get(key) != expected[key] for key in ("index", "name", "dtype", "shape", "bytes", "sha256")):
            raise ValueError("native trace input differs from the retained fixture")
    io = native.get("io_diagnostic", {})
    if io.get("enabled") is not True or io.get("process_called") is not True or len(io.get("inputs", [])) != 3:
        raise ValueError("native input readback is missing")
    for actual, expected in zip(io["inputs"], bundle["inputs"]):
        if actual.get("index") != expected["index"] or actual.get("readback_matches_source") is not True or actual.get("readback_sha256") != expected["sha256"]:
            raise ValueError("native input readback differs from the retained fixture")
    if len(io.get("outputs", [])) != 2 or any(row.get("index") != index or row.get("checked_after_process") is not True
           or row.get("object_same") is not True or row.get("buffer_same") is not True for index, row in enumerate(io["outputs"])):
        raise ValueError("native trace output ownership was not preserved")
    for actual, expected in zip(native["outputs"], bundle["outputs"]):
        if any(actual.get(key) != value for key, value in expected.items()):
            raise ValueError("native trace output ABI differs")
        raw = raw_files[f"results/output_{actual['index']:04d}.bin"]
        size = math.prod(expected["shape"]) * (4 if expected["dtype"] in {"float32", "int32"} else 2)
        if actual.get("bytes") != size or len(raw) != size or hashlib.sha256(raw).hexdigest() != actual.get("sha256"):
            raise ValueError("native trace output bytes differ from their receipt")
    packed = np.frombuffer(raw_files["results/output_0001.bin"], dtype="<f4")
    if packed.size != elements:
        raise ValueError("packed trace size differs from its source specification")
    values = {row["name"]: packed[row["offset"]:row["offset"] + row["elements"]].reshape(row["shape"]) for row in spec}
    top = np.frombuffer(raw_files["results/output_0000.bin"], dtype="<i4").reshape(1, cfg.block_size - 1)
    report = evaluate_trace_integrity(values, cfg, native_top1=top)
    golden = bundle.get("expected", [None])[0]
    if not isinstance(golden, dict) or any(golden.get(key) != value for key, value in bundle["outputs"][0].items()):
        raise ValueError("trace bundle has no bound native Top1 reference")
    if any(golden.get(key) != source_reports["fixture"].get("expected_top1", {}).get(key)
           or golden.get(key) != fixture.get("expected_top1", {}).get(key)
           for key in ("index", "name", "dtype", "shape", "bytes", "sha256")):
        raise ValueError("trace Top1 reference differs from the frozen production fixture")
    expected_top = np.frombuffer(read.bound(root / "bundle", golden), dtype="<i4")
    if expected_top.size != top.size:
        raise ValueError("trace native Top1 reference size differs")
    mismatches = int(np.count_nonzero(expected_top.reshape(top.shape) != top))
    report.update(device_execution="PASSED", evidence_kind=native["evidence_kind"],
                  numerical_validation="FAILED" if mismatches else "TOP1_MATCHED", token_mismatches=mismatches,
                  packed_output_sha256=native["outputs"][1]["sha256"], bundle_manifest_sha256=hashes["bundle"],
                  completed_board_sha256=hashes["board"],
                  model_identity_scope="Recorded manifest/native identities agree; multi-GB packed model bytes were not rehashed",
                  production_equivalence=trace_job.get("production_equivalence"),
                  production_equivalence_scope="Retained diagnostic job report value, not recomputed by this relation-only review")
    read.unchanged()
    return report
