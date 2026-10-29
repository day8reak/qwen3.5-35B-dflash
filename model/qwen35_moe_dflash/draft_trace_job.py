"""One resumable, isolated layer trace of the retained complete draft.

The production job, goldens and compiled payload are read only. Intermediate
outputs can change compiler partitioning; agreement with the production device
outputs is reported separately before a trace is used to explain that run.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil

import numpy as np

from .artifacts import output_path, save_json
from .draft_board import _file, _record, _hidden_comparison
from .hashing import sha256_file


KIND = "qwen35-draft-layer-trace-job"
STAGES = {"reference": "report.json", "fixture": "report.json", "air": "manifest.json",
          "compilation": "report.json", "bundle": "manifest.json", "board": "result.json"}


def _json(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 << 20:
        raise ValueError("invalid trace metadata: " + str(path))
    return json.loads(path.read_text())


def _save(root, name, value):
    """Replace only small trace-owned metadata, preserving real JSON numbers."""
    path = root / name
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError("trace metadata is not a regular file: " + name)
    temporary = root / ("." + name + ".tmp")
    if temporary.is_symlink():
        raise ValueError("trace temporary metadata is a symlink")
    data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    if len(data) > 16 << 20:
        raise ValueError("trace metadata exceeds limit")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _component(job, name):
    row = job["components"][name]
    directory = Path(row["directory"])
    path = _file(directory, row["manifest"])
    return directory, _json(path)


def _fixture(prepared, fixture, reference, trace, spec, output):
    """Bind diagnostic expectations to the unchanged production fixture."""
    from .existing_air import _snapshot
    output.mkdir()
    original = _json(fixture / "report.json")
    report = {**original, "inputs": [], "trace_spec": spec,
              "diagnostic_only": True, "original_fixture_sha256": sha256_file(fixture / "report.json")}
    for field in ("inputs", "expected_top1", "expected_hidden"):
        rows = original[field] if field == "inputs" else [original[field]]
        copied = []
        for row in rows:
            source = _file(fixture, row)
            destination = output / source.name
            _snapshot(source, destination)
            copied.append({**row, **_record(destination, output)})
        report[field] = copied if field == "inputs" else copied[0]
    if report["prepared_manifest_sha256"] != sha256_file(prepared / "prepared.json"):
        raise ValueError("trace fixture does not bind the current prepared weights")
    records = {row["name"]: row for row in reference["tensors"]}
    arrays = []
    for row in spec:
        record = records[row["name"]]
        if record["shape"] != row["shape"] or record["dtype"] != row["dtype"]:
            raise ValueError("reference/GE trace descriptor differs: " + row["name"])
        dtype = np.dtype({"float16": "<f2", "float32": "<f4", "int32": "<i4", "bool": "?"}[row["dtype"]])
        array = np.fromfile(_file(trace, record), dtype=dtype).astype("<f4").reshape(-1)
        if array.size != row["elements"]:
            raise ValueError("reference trace element count differs")
        arrays.append(array)
    packed = np.concatenate(arrays).reshape(1, 1, -1)
    path = output / "draft_trace_values.bin"
    packed.tofile(path)
    report["expected_trace"] = {"index": 1, "name": "draft_trace_values", "shape": list(packed.shape),
                                "dtype": "float32", **_record(path, output)}
    save_json(output / "report.json", report)
    return report


def production_equivalence(job_root, trace_root, spec):
    """Compare the diagnostic graph's terminal values with the original run."""
    original = _json(job_root / "board/results/result.json")
    diagnostic = _json(trace_root / "board/results/result.json")
    old_hidden = original["outputs"][1]
    new_trace = diagnostic["outputs"][1]
    row = next(row for row in spec if row["name"] == "draft_hidden")
    packed = np.fromfile(_file(trace_root / "board/results", new_trace), dtype="<f4")
    hidden = packed[row["offset"]:row["offset"] + row["elements"]].reshape(row["shape"])
    expected = np.fromfile(_file(job_root / "board/results", old_hidden), dtype="<f2").reshape(row["shape"])
    old_top = np.fromfile(_file(job_root / "board/results", original["outputs"][0]), dtype="<i4")
    new_top = np.fromfile(_file(trace_root / "board/results", diagnostic["outputs"][0]), dtype="<i4")
    metrics = _hidden_comparison(expected.astype(np.float64), hidden.astype(np.float64))
    mismatches = int(np.count_nonzero(old_top != new_top))
    return {"hidden": metrics, "token_mismatches": mismatches,
            "status": "EXACT" if metrics["exact_match"] and not mismatches else "DIFFERENT",
            "scope": "Trace outputs can change compilation. A DIFFERENT result cannot establish the original graph's first divergence."}


def _review_trace_board(bundle, board_root, execution):
    """Recheck complete transport/runtime evidence before comparing cached IO."""
    from .draft_board import compare_draft_result

    manifest = _json(bundle / "manifest.json")
    if execution.get("artifact_kind") != "qwen35-standalone-draft-execution":
        raise ValueError("diagnostic board evidence has an unexpected kind")
    if execution.get("retrieval_errors") or execution.get("transport_error"):
        raise ValueError("diagnostic board evidence is incomplete")
    if type(execution.get("remote_exitcode")) is not int or execution["remote_exitcode"] != 0:
        raise ValueError("diagnostic board process did not exit successfully")
    if execution.get("bundle_manifest_sha256") != sha256_file(bundle / "manifest.json"):
        raise ValueError("diagnostic board executed a different bundle")
    if execution.get("model_load_api") != "legacy-buffer":
        raise ValueError("diagnostic board loader differs from this trace invocation")
    required = ("draft.stdout", "draft.stderr", "draft.exitcode", "results/result.json",
                "results/output_0000.bin", "results/output_0001.bin",
                "runtime.json", "runtime.stderr", "runtime.exitcode")
    hashes = execution.get("retrieved_sha256")
    if not isinstance(hashes, dict) or len(hashes) > 64:
        raise ValueError("diagnostic retrieval hashes are missing or exceed budget")
    for name in required:
        if not isinstance(hashes.get(name), str) or not re.fullmatch("[0-9a-f]{64}", hashes[name]):
            raise ValueError("missing verified diagnostic retrieval hash: " + name)
    for name, digest in hashes.items():
        path = board_root / name
        _file(board_root, {"path": name, "bytes": path.stat().st_size, "sha256": digest})
    for name in ("draft.exitcode", "runtime.exitcode"):
        if (board_root / name).read_bytes().strip() != b"0":
            raise ValueError("diagnostic retained exit conflicts with successful execution: " + name)
    native = _json(board_root / "results/result.json")
    if native.get("status") != "EXECUTED":
        raise ValueError("diagnostic native execution did not complete")
    for name in ("runtime_version", "firmware_id"):
        value = execution.get(name)
        if not isinstance(value, str) or not value or native.get(name) != value:
            raise ValueError("diagnostic native " + name + " differs from execution evidence")
    probe = _json(board_root / "runtime.json")
    if probe.get("npu_supported") is not True or probe.get("runtime_version") != execution["runtime_version"]:
        raise ValueError("diagnostic runtime capability probe does not bind this execution")
    if execution.get("ddk_id") != manifest.get("ddk_id"):
        raise ValueError("diagnostic execution DDK differs from its bundle")
    return compare_draft_result(bundle, board_root / "results", expected_model_load_api="legacy-buffer")


def run_trace(job_dir, *, ge_root, ddk_root=None, ddk_id=None, tbe_diagnosis=None,
              schema_library=None, cxx="c++", cxx11_abi=0, timeout_seconds=3600,
              prepare_only=False, compile_only=False):
    from .draft_workspace import DraftWorkspace
    from .draft_review import review_draft_result
    from .draft_standalone import _graph, export_draft
    from .draft_ge_trace import instrument_draft_trace
    from .draft_layer_trace import trace_prepared_draft
    from .draft_compile_precision import collect_precision_plan, verify_compiled_precision
    from .draft_precision_policy import customization_bytes
    from .existing_air import compile_existing_air
    from .draft_board import prepare_draft_bundle, execute_draft_bundle

    job_root = output_path(job_dir)
    root = output_path(job_root / "layer-trace")
    notify = lambda message: print("draft-trace: " + message, flush=True)
    result = {"schema_version": 1, "artifact_kind": KIND, "status": "FAILED", "stage": "preflight",
              "output": str(root), "device_execution": "NOT_RUN", "numerical_validation": "NOT_RUN",
              "scope": "Intermediate diagnostics on the original fixture; no language-quality or speed claim."}
    with DraftWorkspace(job_root):
        notify("Verifying the retained execution and unchanged source receipts")
        review = review_draft_result(job_root)
        if review.get("device_execution") != "PASSED":
            raise ValueError("layer trace requires a verified completed production execution")
        source = _json(job_root / "result.json")
        if source.get("compiled_precision", {}).get("status") != "PASSED":
            raise ValueError("layer trace requires a production compiled-precision pass")
        prepared_dir, prepared = _component(source, "prepared")
        fixture_dir, _ = _component(source, "fixture")
        _, compiled_source = _component(source, "compilation")
        selected_ddk = Path(ddk_root or compiled_source["ddk_root"]).resolve()
        selected_id = ddk_id or compiled_source["ddk_id"]
        if str(selected_ddk) != compiled_source["ddk_root"] or selected_id != compiled_source["ddk_id"]:
            raise ValueError("trace must use the production compiler DDK identity")
        if not prepare_only and not ge_root:
            raise ValueError("CANN_ROOT or --ge-root is required for diagnostic AIR export")
        schema = Path(schema_library).resolve() if schema_library else selected_ddk / "tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so"
        if not schema.is_relative_to(selected_ddk):
            raise ValueError("trace schema must belong to the selected compiler DDK")
        _, _, graph = _graph(prepared["sequence_length"], tiny=prepared["tiny"], draft_config=prepared["draft_config"])
        _, spec = instrument_draft_trace(graph)
        names = [row["name"] for row in spec]
        code_files = ("draft_trace_job.py", "draft_layer_trace.py", "draft_ge_trace.py", "draft_standalone.py",
                      "draft_reference.py", "graphs.py", "model.py", "ops.py", "config.py",
                      "ge_draft.py", "ge_cpp.py", "ge_ir.py", "ge_type_contracts.py", "ge_recipe.py",
                      "ge_constants.py", "ge_reduction_matmul.py", "ge_scan.py", "ge_matmul.py",
                      "ge_dense_projection.py", "ge_reshape_chains.py", "existing_air.py", "compile.py",
                      "draft_board.py", "draft_compile_precision.py", "draft_precision_policy.py",
                      "omc_container.py", "omc_embed_weights.py", "omc_wire.py", "omc_cpu_inspect.py")
        identity = {"artifact_kind": KIND + "-owner", "job": str(job_root),
                    "source_job_sha256": sha256_file(job_root / "result.json"), "trace_spec": spec,
                    "code": {name: sha256_file(Path(__file__).parent / name) for name in code_files}}
        if root.exists():
            if root.is_symlink() or not root.is_dir() or _json(root / "owner.json") != identity:
                raise ValueError("layer-trace belongs to a different source/code identity; retained evidence was preserved")
        else:
            root.mkdir()
            save_json(root / "owner.json", identity)

        def stage(name, build):
            result["stage"] = name
            folder = root / name
            receipt_path = root / ("completed-" + name + ".json")
            if receipt_path.exists():
                receipt = _json(receipt_path)
                report = _json(_file(folder, receipt))
                notify("Reusing verified " + name)
                remember_board(name, folder, report)
                return folder, report
            started = root / ("started-" + name + ".json")
            marker = {"owner_sha256": sha256_file(root / "owner.json"), "stage": name}
            if folder.exists():
                if folder.is_symlink() or not started.exists() or _json(started) != marker:
                    raise ValueError("refusing to replace unowned trace stage: " + name)
                shutil.rmtree(folder)
            _save(root, started.name, marker)
            notify("Building " + name)
            report = build(folder)
            remember_board(name, folder, report)
            if report.get("status") == "FAILED" and not (name == "board" and report.get("device_execution") == "PASSED"):
                raise RuntimeError(report.get("error", name + " failed; inspect " + str(folder)))
            _save(root, receipt_path.name, _record(folder / STAGES[name], folder))
            return folder, report

        def remember_board(name, folder, report):
            if name != "board":
                return
            result["board"] = str(folder)
            result["board_evidence"] = {"result": str(folder / "result.json"),
                **{name: str(folder / name) for name in ("draft.stdout", "draft.stderr", "draft-hilog.txt")
                   if (folder / name).is_file()}}
            for field in ("device_execution", "numerical_validation", "native_status", "runtime_stage",
                          "remote_exitcode", "retrieval_errors", "transport_error", "hilog"):
                if field in report:
                    result[field] = report[field]

        try:
            trace_dir, reference = stage("reference", lambda folder: trace_prepared_draft(
                prepared_dir, fixture_dir, folder, names=names, progress=notify))
            fixture, fixture_report = stage("fixture", lambda folder: _fixture(
                prepared_dir, fixture_dir, reference, trace_dir, spec, folder))
            # Validate every tensor even when the trace reference was reused.
            for row in reference["tensors"]:
                _file(trace_dir, row)
            for row in [*fixture_report["inputs"], fixture_report["expected_top1"],
                        fixture_report["expected_hidden"], fixture_report["expected_trace"]]:
                _file(fixture, row)
            if prepare_only:
                result.update(status="PREPARED", stage="complete")
            else:
                air, exported = stage("air", lambda folder: export_draft(
                    prepared_dir, folder, ge_root, cxx=cxx, cxx11_abi=cxx11_abi,
                    timeout_seconds=timeout_seconds, trace_names=names,
                    progress=lambda row: notify(row["message"])))
                if exported.get("trace_spec") != spec:
                    raise ValueError("diagnostic AIR and reference trace specifications differ")
                for row in exported["graphs"][0]["files"] + exported["graphs"][0]["construction_files"]:
                    _file(air, row)
                plan = collect_precision_plan(air / "draft/recipe/graph.recipe")
                policy = customization_bytes(plan)
                policy_path = root / "cl-customization.json"
                if policy_path.exists():
                    if policy_path.is_symlink() or policy_path.read_bytes() != policy:
                        raise ValueError("trace precision configuration changed")
                else:
                    policy_path.write_bytes(policy)
                compilation, compiled = stage("compilation", lambda folder: compile_existing_air(
                    air / exported["graphs"][0]["air"], folder, selected_ddk, ddk_id=selected_id,
                    tbe_diagnosis=tbe_diagnosis, timeout_seconds=timeout_seconds,
                    input_layout="air-directory", use_origin_format=True, weight_data_type="FP16",
                    limit_partition_size=False, save_weights_as_external_data=True,
                    cl_customization=policy_path, progress=lambda row: notify(row["message"])))
                snapshot = compiled.get("cl_customization", {})
                if snapshot.get("sha256") != hashlib.sha256(policy).hexdigest():
                    raise ValueError("trace compilation precision configuration mismatch")
                _file(compilation, snapshot)
                omg = selected_ddk / "tools/tools_omg/master/omg"
                if sha256_file(omg) != compiled.get("omg_sha256"):
                    raise ValueError("trace compiler changed since compilation")
                precision = verify_compiled_precision(_file(compilation, compiled["omc"]), schema, plan)
                _save(root, "precision-check.json", precision)
                result["compiled_precision"] = {key: precision.get(key) for key in ("status", "required_nodes", "matched_nodes")}
                if precision.get("status") != "PASSED":
                    raise RuntimeError("diagnostic compiled precision did not pass; no bundle or board execution")
                if compile_only:
                    result.update(status="COMPILED", stage="complete")
                else:
                    bundle, bundled = stage("bundle", lambda folder: prepare_draft_bundle(
                        compilation, fixture, folder, config=prepared["draft_config"], ddk_id=selected_id,
                        export_dir=air, weight_storage="embedded-stream", schema_library=schema, trace_spec=spec))
                    for key, path in (("compiler_report_sha256", compilation / "report.json"),
                                      ("fixture_report_sha256", fixture / "report.json"),
                                      ("export_manifest_sha256", air / "manifest.json")):
                        if bundled.get(key) != sha256_file(path):
                            raise ValueError("diagnostic bundle source mismatch: " + key)
                    _, board = stage("board", lambda folder: execute_draft_bundle(
                        bundle, folder, timeout_seconds=timeout_seconds, model_load_api="legacy-buffer"))
                    # A reused board result must still pass full hash/transport validation.
                    comparison = _review_trace_board(bundle, root / "board", board)
                    _save(root / "board", "comparison.json", comparison)
                    result.update(status=comparison["status"], stage="complete", device_execution=comparison["device_execution"],
                                  numerical_validation=comparison["numerical_validation"],
                                  comparison=str(root / "board/comparison.json"),
                                  trace_diagnostic=comparison.get("trace_diagnostic"))
                    if board.get("device_execution") == "PASSED":
                        result["production_equivalence"] = production_equivalence(job_root, root, spec)
                    notify("Trace comparison: " + str(root / "board/comparison.json"))
            if sha256_file(job_root / "result.json") != identity["source_job_sha256"]:
                raise ValueError("production job changed during diagnostic execution")
        except Exception as error:
            result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        _save(root, "result.json", result)
        return result
