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
STAGES = {"reference": "report.json", "fixture": "report.json", "observed-fixture": "report.json", "air": "manifest.json",
          "compilation": "report.json", "bundle": "manifest.json", "board": "result.json"}
# These modules choose stages and evaluate already-bound evidence. They may be
# updated without changing an existing reference/AIR/OMC. All actual builders,
# reference arithmetic, lowering, packing and compile policy remain code-bound.
# Backward-compatible owners keep their original bytes so started-stage receipts
# continue to refer to the same owner SHA after a checker-only update.
_CONTROL_ONLY_CODE_FILES = ("draft_trace_job.py", "draft_compile_precision.py", "draft_precision_edges.py")
_LEGACY_OPTIONAL_CONTROL_CODE_FILES = frozenset({"draft_precision_edges.py"})
_ARTIFACT_CODE_FILES = (
    "draft_layer_trace.py", "draft_ge_trace.py", "draft_standalone.py",
    "draft_reference.py", "graphs.py", "model.py", "ops.py", "config.py",
    "ge_draft.py", "ge_cpp.py", "ge_ir.py", "ge_type_contracts.py", "ge_recipe.py",
    "ge_constants.py", "ge_reduction_matmul.py", "ge_scan.py", "ge_matmul.py",
    "ge_dense_projection.py", "ge_reshape_chains.py", "existing_air.py", "compile.py",
    "draft_board.py", "draft_precision_policy.py", "omc_container.py",
    "omc_embed_weights.py", "omc_wire.py", "omc_cpu_inspect.py")


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


def _runtime_source_identity():
    """Audit current runner sources without invalidating frozen model payloads.

    Native LINKED/build receipts bind the actual executed binary. These hashes
    describe the checkout used for this attempt; they never certify an older
    failed binary or allow a model/bundle builder change to bypass ownership.
    """
    from .resources import repository_root
    repository = repository_root()
    paths = ("targets/cpro/runtime/CMakeLists.txt", "targets/cpro/scripts/build_hiai.py",
             "targets/cpro/runtime/src/main.cpp", "targets/cpro/runtime/src/hiai_executor.cpp",
             "targets/cpro/runtime/src/generation.cpp", "targets/cpro/runtime/src/sha256.cpp",
             "targets/cpro/runtime/include/qwen35_dflash/hiai_executor.hpp",
             "targets/cpro/runtime/include/qwen35_dflash/generation.hpp",
             "targets/cpro/runtime/include/qwen35_dflash/sha256.hpp")
    return {"scope": "Current runner checkout source audit; native LINKED receipts bind the executed binary",
            "files": {name: sha256_file(repository / name) for name in paths}}


def _board_failure_evidence(folder, report, notify):
    """Print only bounded, hash-verified current-attempt native log tails."""
    metadata = {key: report[key] for key in ("native_status", "failure_stage", "runtime_stage", "remote_exitcode",
                "error", "comparison_error", "transport_error", "retrieval_errors") if key in report}
    notify("Board failure: " + json.dumps(metadata, ensure_ascii=False, allow_nan=False))
    logs = {}
    hashes = report.get("retrieved_sha256", {})
    for name in ("draft.stderr", "draft.stdout"):
        path = folder / name
        digest = hashes.get(name) if isinstance(hashes, dict) else None
        if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
            continue
        try:
            source = _file(folder, {"path": name, "bytes": path.stat().st_size, "sha256": digest})
            with source.open("rb") as stream:
                stream.seek(max(0, source.stat().st_size - 8192))
                tail = "\n".join(stream.read(8192).decode("utf-8", errors="replace").splitlines()[-40:])[-4096:]
            logs[name] = {"sha256": digest, "bytes": source.stat().st_size, "tail": tail}
            notify("Verified " + name + " tail:\n" + tail)
        except (OSError, ValueError) as error:
            logs[name] = {"error": str(error)}
            notify("Could not verify " + name + ": " + str(error))
    return {"metadata": metadata, "logs": logs,
            "scope": "Current attempt verified logs; missing native result does not hide the runner exception"}


def _archive_failed_board(root, folder, marker):
    """Retain a failed board attempt in place before rebuilding its runner."""
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError("failed board attempt must be a real directory")
    owner_sha = sha256_file(root / "owner.json")
    if marker != {"owner_sha256": owner_sha, "stage": "board"}:
        raise ValueError("failed board attempt does not match the workspace owner")
    archive_root = root / "board-attempts"
    archive_owner = {"artifact_kind": "qwen35-trace-board-attempts", "owner_sha256": owner_sha}
    if archive_root.exists():
        if archive_root.is_symlink() or not archive_root.is_dir() or _json(archive_root / "owner.json") != archive_owner:
            raise ValueError("board attempt archive has an unknown owner")
    else:
        archive_root.mkdir()
        save_json(archive_root / "owner.json", archive_owner)
    report_path = folder / "result.json"
    if report_path.is_symlink():
        raise ValueError("failed board result must not be a symlink")
    has_report = report_path.is_file()
    digest = (sha256_file(report_path) if has_report else
              hashlib.sha256(("incomplete:" + str(folder.stat().st_mtime_ns)).encode()).hexdigest())
    destination = archive_root / digest
    suffix = 1
    while destination.exists():
        suffix += 1
        if suffix > 1000:
            raise ValueError("board attempt archive exceeds retry bound")
        destination = archive_root / (digest + "-" + str(suffix))
    if (folder / ".trace-board-archive.json").exists():
        raise ValueError("failed board directory already contains an archive marker")
    folder.rename(destination)
    save_json(destination / ".trace-board-archive.json", {
        "artifact_kind": "qwen35-archived-trace-board-attempt", "original_directory": str(folder),
        "owner_sha256": owner_sha, "result_sha256": digest if has_report else None,
        "scope": "Directory renamed without changing original evidence bytes; retry uses a fresh board directory"})
    return destination


def _resume_owner(root, current):
    """Allow checker-only updates of fully retained build stages.

    No owner, receipt, or compiled file is migrated or rewritten. Cached payloads
    are rechecked by the normal stage path, and the current precision policy
    must still match the retained compiler customization before verification.
    """
    if root.is_symlink() or not root.is_dir():
        raise ValueError("layer-trace belongs to a different source/code identity; retained evidence was preserved")
    previous = _json(root / "owner.json")
    if previous == current:
        return []
    if ({key: value for key, value in previous.items() if key != "code"}
            != {key: value for key, value in current.items() if key != "code"}):
        raise ValueError("layer-trace belongs to a different source/code identity; retained evidence was preserved")
    old_code, new_code = previous.get("code"), current.get("code")
    if (not isinstance(old_code, dict) or not isinstance(new_code, dict)
            or set(old_code) - set(new_code)
            or (set(new_code) - set(old_code)) - _LEGACY_OPTIONAL_CONTROL_CODE_FILES
            or any(not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value)
                   for value in (*old_code.values(), *new_code.values()))):
        raise ValueError("layer-trace has a different code dependency set; retained evidence was preserved")
    changed = [name for name in sorted(new_code) if old_code.get(name) != new_code[name]]
    if set(changed) - set(_CONTROL_ONLY_CODE_FILES):
        raise ValueError("layer-trace has changed artifact-building code: " + ", ".join(changed)
                         + "; retained evidence was preserved")
    # The orchestration file also contains fixture assembly. Only accept its
    # update when that fixture and all downstream compiler inputs already exist;
    # a failed or missing lower stage must not run under different builder code.
    expected = {
        "reference": ("qwen35-draft-layer-reference-trace", "REFERENCE_TRACE_COMPLETE"),
        "fixture": ("qwen35-draft-reference", "REFERENCE_COMPLETE"),
        "air": ("qwen35-cpro-draft-air", "EXPORTED"),
        "compilation": ("qwen35-existing-air-compilation", "COMPILED"),
    }
    for name, (kind, status) in expected.items():
        receipt_path = root / ("completed-" + name + ".json")
        if not receipt_path.is_file() or receipt_path.is_symlink():
            raise ValueError("checker-only resume requires completed " + name + "; retained evidence was preserved")
        receipt = _json(receipt_path)
        if receipt.get("path") != STAGES[name]:
            raise ValueError("checker-only resume has an invalid " + name + " receipt")
        report = _json(_file(root / name, receipt))
        if report.get("artifact_kind") != kind or report.get("status") != status:
            raise ValueError("checker-only resume has an incomplete " + name + " stage")
    return [{"name": name, "previous_sha256": old_code.get(name), "current_sha256": new_code[name]}
            for name in changed]


def _precision_failures(precision, notify):
    """Expose bounded gate evidence without bypassing or reclassifying it."""
    from .draft_precision_edges import summarize_precision_failures
    if precision.get("reason"):
        notify("Precision check reason: " + str(precision["reason"])[:4000])
    failures = [row for row in precision.get("nodes", []) if row.get("status") != "MATCHED"]
    notify("Compiled precision: " + str(precision.get("status")) + "; matched "
           + str(precision.get("matched_nodes")) + "/" + str(precision.get("required_nodes")))
    if precision.get("cross_partition_edges"):
        notify("Verified FP32 CPU partition paths: " + str(precision["cross_partition_edges"]))
    for row in failures[:8]:
        notify("Precision mismatch: " + json.dumps(
            {"name": row.get("name"), "source_type": row.get("source_type", row.get("type")),
             "compiled_type": row.get("compiled_type"), "status": row.get("status"),
             "error_count": len(row.get("errors", [])), "errors": row.get("errors", [])[:1]},
            ensure_ascii=False, allow_nan=False)[:8000])
    if len(failures) > 8:
        notify("Additional unmatched nodes: " + str(len(failures) - 8))
    summary = summarize_precision_failures(precision)
    notify("Distinct precision causes: " + str(summary["unique_issue_count"]) + "; full findings: layer-trace/precision-check.json")
    for issue in summary["issues"][:24]:
        # Keep long lists in the report; the terminal needs the exact failing
        # port and both descriptors, plus counts of duplicate consequences.
        compact = {key: issue.get(key) for key in ("kind", "reason", "details", "occurrences")}
        compact["affected_nodes"] = issue.get("affected_nodes", [])[:4]
        notify("Precision cause: " + json.dumps(compact, ensure_ascii=False, allow_nan=False)[:8000])


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


def production_equivalence(job_root, trace_root, spec, *, direct=False):
    """Compare the diagnostic graph's terminal values with the original run."""
    original = _json(job_root / "board/results/result.json")
    diagnostic = _json(trace_root / "board/results/result.json")
    old_hidden = original["outputs"][1]
    new_trace = diagnostic["outputs"][1]
    row = next(row for row in spec if row["name"] == "draft_hidden")
    if direct:
        hidden = np.fromfile(_file(trace_root / "board/results", new_trace), dtype="<f2").reshape(row["shape"])
    else:
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
              prepare_only=False, compile_only=False, direct=False, residual_policy="native"):
    from .draft_workspace import DraftWorkspace
    from .draft_review import review_draft_result
    from .draft_standalone import _graph, export_draft
    from .draft_ge_trace import instrument_draft_trace
    from .draft_layer_trace import trace_prepared_draft
    from .draft_compile_precision import collect_precision_plan, verify_compiled_precision
    from .draft_precision_policy import customization_bytes
    from .existing_air import compile_existing_air
    from .draft_board import prepare_draft_bundle, execute_draft_bundle

    if residual_policy not in {"native", "cpu-fp32-add"}:
        raise ValueError("unsupported draft residual policy")
    candidate = residual_policy != "native"
    if candidate and not direct:
        raise ValueError("residual candidate requires direct original-value outputs")
    job_root = output_path(job_dir)
    root = output_path(job_root / ("residual-fix" if candidate else "direct-trace" if direct else "layer-trace"))
    prefix = "draft-residual: " if candidate else "draft-direct: " if direct else "draft-trace: "
    notify = lambda message: print(prefix + message, flush=True)
    kind = "qwen35-draft-direct-trace-job" if direct else KIND
    result = {"schema_version": 1, "artifact_kind": kind, "status": "FAILED", "stage": "preflight",
              "output": str(root), "device_execution": "NOT_RUN", "numerical_validation": "NOT_RUN",
              "scope": "Intermediate diagnostics on the original fixture; no language-quality or speed claim."}
    if candidate:
        result["residual_policy"] = residual_policy
    result["runtime_source_identity"] = _runtime_source_identity()
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
        if direct:
            from .draft_direct_trace import instrument_draft_direct_outputs
            # The repair candidate exercises the complete six-layer graph with
            # only its original terminal outputs. Extra observations previously
            # changed compilation and contradicted internal consumers.
            selection_names = ("draft_top1", "draft_hidden") if candidate else None
            _, spec = instrument_draft_direct_outputs(graph, names=selection_names)
        else:
            _, spec = instrument_draft_trace(graph)
        names = [row["name"] for row in spec]
        code_files = (*_ARTIFACT_CODE_FILES, *_CONTROL_ONLY_CODE_FILES,
                      *(("draft_direct_trace.py", "draft_direct_job.py", "draft_direct_abi.py") if direct else ()),
                      *(("ge_residual.py",) if candidate else ()))
        identity = {"artifact_kind": kind + "-owner", "job": str(job_root),
                    "source_job_sha256": sha256_file(job_root / "result.json"), "trace_spec": spec,
                    "code": {name: sha256_file(Path(__file__).parent / name) for name in code_files}}
        if candidate:
            identity["residual_policy"] = residual_policy
        if root.exists():
            control_updates = _resume_owner(root, identity)
            if control_updates:
                result["control_code_updates"] = control_updates
                notify("Reusing bound reference/AIR/compilation after checker-only code updates: "
                       + ", ".join(row["name"] for row in control_updates))
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
                if name == "board":
                    archived = _archive_failed_board(root, folder, marker)
                    result["previous_board_attempt"] = str(archived)
                    notify("Retained failed board evidence: " + str(archived))
                else:
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
            if report.get("status") == "FAILED":
                result["board_failure"] = _board_failure_evidence(folder, report, notify)
                excerpt = result["board_failure"]["logs"].get("draft.stderr", {}).get("tail")
                if excerpt:
                    result["native_error_excerpt"] = excerpt

        try:
            if direct:
                from .draft_direct_job import prepare_direct_reference, prepare_direct_fixture
                trace_dir, reference = stage("reference", lambda folder: prepare_direct_reference(
                    job_root, prepared_dir, fixture_dir, folder, names=names, progress=notify))
                fixture, fixture_report = stage("fixture", lambda folder: prepare_direct_fixture(
                    prepared_dir, fixture_dir, reference, trace_dir, spec, folder))
            else:
                trace_dir, reference = stage("reference", lambda folder: trace_prepared_draft(
                    prepared_dir, fixture_dir, folder, names=names, progress=notify))
                fixture, fixture_report = stage("fixture", lambda folder: _fixture(
                    prepared_dir, fixture_dir, reference, trace_dir, spec, folder))
            # Validate every tensor even when the trace reference was reused.
            for row in reference["tensors"]:
                _file(trace_dir, row)
            expectations = (fixture_report["expected_direct"] if direct else
                            [fixture_report["expected_top1"], fixture_report["expected_hidden"], fixture_report["expected_trace"]])
            for row in [*fixture_report["inputs"], *expectations]:
                _file(fixture, row)
            if prepare_only:
                result.update(status="PREPARED", stage="complete")
            else:
                selection = {"direct_names": names} if direct else {"trace_names": names}
                if candidate:
                    selection["residual_policy"] = residual_policy
                air, exported = stage("air", lambda folder: export_draft(
                    prepared_dir, folder, ge_root, cxx=cxx, cxx11_abi=cxx11_abi,
                    timeout_seconds=timeout_seconds, **selection,
                    progress=lambda row: notify(row["message"])))
                if exported.get("direct_spec" if direct else "trace_spec") != spec:
                    raise ValueError("diagnostic AIR and reference trace specifications differ")
                if candidate and exported.get("residual_policy") != residual_policy:
                    raise ValueError("candidate AIR does not bind the selected residual policy")
                for row in exported["graphs"][0]["files"] + exported["graphs"][0]["construction_files"]:
                    _file(air, row)
                plan = collect_precision_plan(air / "draft/recipe/graph.recipe")
                if candidate:
                    from .ge_residual import residual_selection
                    required_sums = {"__draft_residual/" + name + "/sum"
                                     for name in residual_selection(graph, residual_policy)}
                    observed_sums = {node["name"] for node in plan["required_nodes"]
                                     if node["name"].startswith("__draft_residual/") and node["type"] == "Add"}
                    if observed_sums != required_sums:
                        raise ValueError("candidate precision plan does not cover every residual sum")
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
                expected_options = {"use_origin_format": "true", "weight_data_type": "FP16",
                                    "limit_partition_size": "false", "save_weights_as_external_data": "true"}
                if any(compiled.get("requested_options", {}).get(key) != value
                       for key, value in expected_options.items()):
                    raise ValueError("trace compilation options differ from the bound draft profile")
                if compiled.get("ddk_root") != str(selected_ddk) or compiled.get("ddk_id") != selected_id:
                    raise ValueError("trace compilation DDK differs from the selected production compiler")
                snapshot = compiled.get("cl_customization", {})
                if snapshot.get("sha256") != hashlib.sha256(policy).hexdigest():
                    raise ValueError("trace compilation precision configuration mismatch")
                _file(compilation, snapshot)
                omg = selected_ddk / "tools/tools_omg/master/omg"
                if sha256_file(omg) != compiled.get("omg_sha256"):
                    raise ValueError("trace compiler changed since compilation")
                precision = verify_compiled_precision(_file(compilation, compiled["omc"]), schema, plan)
                _save(root, "precision-check.json", precision)
                result["compiled_precision"] = {key: precision.get(key) for key in ("status", "required_nodes", "matched_nodes", "cross_partition_edges")}
                if precision.get("status") != "PASSED":
                    _precision_failures(precision, notify)
                    raise RuntimeError("diagnostic compiled precision did not pass; no bundle or board execution")
                binding = None
                if direct:
                    from .draft_direct_abi import inspect_direct_output_abi
                    binding = inspect_direct_output_abi(_file(compilation, compiled["omc"]), schema, spec)
                    _save(root, "compiled-output-abi.json", binding)
                    result["source_output_precision"] = ("SOURCE_DTYPES_PRESERVED"
                        if binding.get("status") == "PASSED" else binding.get("status"))
                    result["compiled_output_abi"] = {"status": binding.get("status"),
                        "changes": binding.get("changes", []), "report": str(root / "compiled-output-abi.json")}
                    if binding.get("status") not in {"PASSED", "SOURCE_DTYPE_CHANGED"}:
                        raise RuntimeError("compiled direct output ABI could not be established; no board execution")
                if compile_only:
                    result.update(status="COMPILED", stage="complete")
                else:
                    observation = {"trace_spec": spec}
                    observed_spec = spec
                    if direct:
                        from .draft_direct_job import prepare_observed_direct_fixture
                        observed_spec = binding["observed_spec"]
                        fixture, physical_fixture = stage("observed-fixture", lambda folder: prepare_observed_direct_fixture(
                            root / "fixture", folder, binding))
                        if physical_fixture.get("direct_output_binding") != binding or physical_fixture.get("direct_spec") != observed_spec:
                            raise ValueError("retained physical fixture differs from the current compiled output ABI")
                        observation = {"direct_spec": observed_spec, "direct_output_binding": binding}
                        notify("Compiled direct output ABI: " + binding["status"]
                               + "; source dtype changes remain an explicit numerical failure")
                    bundle, bundled = stage("bundle", lambda folder: prepare_draft_bundle(
                        compilation, fixture, folder, config=prepared["draft_config"], ddk_id=selected_id,
                        export_dir=air, weight_storage="embedded-stream", schema_library=schema, **observation))
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
                                  trace_diagnostic=comparison.get("direct_diagnostic" if direct else "trace_diagnostic"))
                    if board.get("device_execution") == "PASSED":
                        result["production_equivalence"] = production_equivalence(job_root, root, spec, direct=direct)
                        from .draft_trace_integrity import evaluate_trace_integrity
                        native = _json(root / "board/results/result.json")
                        if direct:
                            from .draft_direct_job import read_direct_values
                            values = read_direct_values(root / "board/results", native, observed_spec)
                        else:
                            record = native["outputs"][1]
                            packed = np.fromfile(_file(root / "board/results", record), dtype="<f4")
                            values = {row["name"]: packed[row["offset"]:row["offset"] + row["elements"]].reshape(row["shape"])
                                      for row in spec}
                        native_top = np.fromfile(_file(root / "board/results", native["outputs"][0]), dtype="<i4")
                        result["trace_integrity"] = evaluate_trace_integrity(values, prepared["draft_config"],
                            native_top1=None if direct else native_top.reshape(values["draft_top1"].shape),
                            alias_deduplicated=direct)
                        if result["trace_integrity"].get("status") == "FAILED":
                            result.update(status="FAILED", diagnostic_validity="UNRELIABLE_OBSERVATIONS")
                    notify("Trace comparison: " + str(root / "board/comparison.json"))
            if sha256_file(job_root / "result.json") != identity["source_job_sha256"]:
                raise ValueError("production job changed during diagnostic execution")
        except Exception as error:
            result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        _save(root, "result.json", result)
        return result
