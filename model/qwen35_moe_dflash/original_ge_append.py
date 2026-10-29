"""Build a feature-only GE branch and append it to a source-bound original AIR.

No checkpoint or external FileConstant payload is copied, opened or rewritten.
The resulting AIR still needs weight staging, actual OMG and HIAI validation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath

from .air_schema import load_torchair_schema
from .artifacts import output_path, save_json
from .ge_export import _logged
from .ge_fused.export import _toolchain
from .hashing import sha256_file
from .original_air_inspect import inspect_original_air
from .original_air_roundtrip import _decode, _proof, _read_air, _stamp, _unavailable_payloads, probe_original_air_roundtrip
from .original_ge_target import audit_source, read_json, source_receipt
from .original_ge_append_cpp import emit_feature_branch_cpp
from .original_ge_append_validation import validate_original_ge_append
from .original_ge_append_wire import append_feature_branch_wire
from .original_ge_taps import plan_original_ge_taps


_OWN_FILES = {"branch.cpp", "branch", "branch.air", "native.json", "build.log", "branch.log", "appended.air"}


def _source(root, receipt_path):
    if bool(root) == bool(receipt_path):
        raise ValueError("provide exactly one original source root or source receipt")
    receipt, digest = (source_receipt(root), None) if root else read_json(receipt_path, 1 << 20)
    audit = audit_source(receipt)
    if audit["status"] != "MATCHED_AUDITED_SOURCE_RECEIPTS":
        raise ValueError("original source audit did not match")
    return receipt, audit, digest


def _identity(paths):
    return {str(Path(path).absolute()): {"sha256": sha256_file(Path(path)), "bytes": Path(path).stat().st_size,
                                      "resolved": str(Path(path).resolve())} for path in paths}


def append_original_ge_features(air, output, *, torchair_pb2, gamma_template, sequence_length,
                                source_root=None, source_receipt_path=None, ge_root=None,
                                cxx="c++", cxx11_abi=0, timeout_seconds=120):
    out = output_path(output)
    out.mkdir(parents=True, exist_ok=False)
    work = out / "work"
    work.mkdir()
    result = {"schema_version": 1, "artifact_kind": "qwen35-original-ge-feature-append",
              "status": "BLOCKED", "stage": "source", "output": str(out),
              "model_compilation": "NOT_RUN", "device_execution": "NOT_RUN", "numerical_validation": "NOT_RUN",
              "external_weights": "NOT_OPENED_NOT_COPIED_NOT_REWRITTEN",
              "scope": "Source-anchored original AIR plus auxiliary features only; no full-model accuracy, state continuity or speed pass"}
    original, original_record, source_bound, identities = None, None, None, {}
    receipt_stamp, receipt_sha, candidate_raw, candidate_record = None, None, None, None
    try:
        if source_receipt_path:
            receipt_stamp = _stamp(Path(source_receipt_path))
        receipt, source_audit, receipt_sha = _source(source_root, source_receipt_path)
        source_bound = receipt
        result["source_audit"] = source_audit
        result["source_receipt_sha256"] = receipt_sha
        save_json(out / "source-receipt.json", receipt)
        original, original_record = _read_air(air)
        result["original_air"] = original_record
        schema = load_torchair_schema(torchair_pb2)
        result["schema"] = schema.evidence
        kind, original_model, original_graph = _decode(schema, original)
        proof, bindings = _proof(kind, original_model, original_graph)
        result["original_metadata"] = proof
        if any(PurePosixPath(value).parts[0] in _OWN_FILES for value in bindings):
            raise ValueError("FileConstant path collides with feature-work files")
        _unavailable_payloads(work, bindings)
        inspection = inspect_original_air(air, full=True, torchair_pb2=torchair_pb2)
        if inspection.get("air", {}).get("sha256") != original_record["sha256"]:
            raise ValueError("source AIR changed between graph and metadata reads")
        save_json(out / "source-inspection.json", inspection)
        result["stage"] = "tap-provenance"
        plan = plan_original_ge_taps(inspection, source_audit, gamma_template, sequence_length=sequence_length)
        save_json(out / "tap-plan.json", plan)
        result["tap_plan"] = plan
        if plan.get("status") != "FEATURE_BRANCH_PLAN_REVIEW_REQUIRED":
            raise ValueError("actual original graph does not yet bind all eight feature taps")
        taps = plan.get("taps")
        if not isinstance(taps, list) or len(taps) != 8:
            raise ValueError("tap plan lacks eight explicit producer ports")
        if any(tap.get("descriptor", {}).get("layout") != "ND"
               or not tap.get("descriptor", {}).get("schema_dtype_names")
               or not set(tap["descriptor"]["schema_dtype_names"]) <= {"DT_FLOAT", "DT_FLOAT32"} for tap in taps):
            raise ValueError("this source-bound GE feature template requires eight actual FP32 ND hidden producers")
        if not ge_root:
            raise ValueError("declare matching --ge-root/CANN_ROOT used for host GE serialization")
        result["stage"] = "toolchain"
        ge, include, lib, compiler = _toolchain(ge_root, cxx, cxx11_abi, timeout_seconds)
        cpp = work / "branch.cpp"
        cpp.write_text(emit_feature_branch_cpp(sequence_length))
        direct_paths = [cpp, Path(compiler).resolve(), Path(torchair_pb2).resolve(),
                        include / "graph/graph.h", include / "graph/operator.h", include / "graph/types.h",
                        lib / "libgraph.so", Path(__file__)]
        for name in ("original_ge_append_cpp.py", "original_ge_append_wire.py", "original_ge_append_validation.py",
                     "original_ge_taps.py", "original_air_roundtrip.py", "original_air_inspect.py", "air_schema.py", "omc_wire.py"):
            direct_paths.append(Path(__file__).with_name(name))
        if (lib / "libgraph_base.so").is_file():
            direct_paths.append(lib / "libgraph_base.so")
        identities = _identity(direct_paths)
        result["toolchain"] = {"ge_root": str(ge), "cxx11_abi": cxx11_abi, "identities": identities,
                               "scope": "Direct sources/headers/libraries/compiler only; no transitive library closure claim"}
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, (str(lib), env.get("LD_LIBRARY_PATH", ""))))
        env["ASCEND_PROCESS_LOG_PATH"] = str(out / "runtime-log")
        build = [compiler, "-std=c++17", f"-D_GLIBCXX_USE_CXX11_ABI={cxx11_abi}", "-I", str(include), str(cpp),
                 "-L", str(lib), f"-Wl,-rpath,{lib}", "-lgraph"]
        if (lib / "libgraph_base.so").is_file():
            build.append("-lgraph_base")
        build.extend(["-o", str(work / "branch")])
        command = [str(work / "branch"), str(work / "branch.air"), str(work / "native.json")]
        result["commands"] = {"build": build, "branch": command, "cwd": str(work)}
        result["stage"] = "feature-branch"
        _logged(build, folder=work, env=env, log=work / "build.log", timeout=timeout_seconds)
        _unavailable_payloads(work, bindings)
        _logged(command, folder=work, env=env, log=work / "branch.log", timeout=timeout_seconds)
        native, _ = read_json(work / "native.json", 4096)
        result["branch_native"] = native
        if native.get("save_status") != 0 or native.get("reload_status") != 0:
            raise ValueError("GE branch template save/reload failed")
        branch, branch_record = _read_air(work / "branch.air")
        result["branch_air"] = branch_record
        result["stage"] = "append"
        patched, wire_receipt = append_feature_branch_wire(original, branch, schema, taps)
        result["wire_preservation"] = wire_receipt
        _, patched_model, _ = _decode(schema, patched)
        dtype_enum = schema.enums.get(schema.package + ".DataType", {})
        validation = validate_original_ge_append(original_model, patched_model, taps=taps,
                    dtype_enum=dtype_enum, sequence_length=sequence_length, cast_dst_type=1)
        save_json(out / "append-validation.json", validation)
        result["append_validation"] = validation
        if validation.get("status") != "METADATA_APPEND_VALIDATED":
            raise ValueError("original computation/ABI preservation check failed")
        patched_path = work / "appended.air"
        with patched_path.open("xb") as stream:
            stream.write(patched)
        result["stage"] = "ge-roundtrip"
        roundtrip = probe_original_air_roundtrip(patched_path, out / "roundtrip", ge_root=ge,
            torchair_pb2=torchair_pb2, cxx=cxx, cxx11_abi=cxx11_abi, timeout_seconds=timeout_seconds)
        result["ge_roundtrip"] = {key: roundtrip.get(key) for key in
            ("status", "native", "fidelity", "source_preservation", "error", "roundtrip_air")}
        if roundtrip.get("status") != "ROUNDTRIP_METADATA_PASSED":
            raise ValueError("whole appended AIR did not preserve metadata through actual GE Load/Save")
        final_raw, final_record = _read_air(roundtrip["roundtrip_air"]["path"])
        _, final_model, _ = _decode(schema, final_raw)
        final_validation = validate_original_ge_append(original_model, final_model, taps=taps,
                    dtype_enum=dtype_enum, sequence_length=sequence_length, cast_dst_type=1)
        save_json(out / "roundtrip-validation.json", final_validation)
        result["roundtrip_validation"] = final_validation
        if final_validation.get("status") != "METADATA_APPEND_VALIDATED":
            raise ValueError("GE roundtrip changed the original graph or feature branch")
        candidate_raw, candidate_record = final_raw, final_record
        result.update(status="FEATURE_APPEND_VERIFIED_PENDING_PUBLICATION", stage="complete",
                      feature_output={"index": 81, "dtype": "float16", "shape": [1, sequence_length, 16384]},
                      compilation_input="WEIGHT_REFERENCES_PRESERVED_BUT_PAYLOADS_NOT_STAGED",
                      source_to_air_revision="UNVERIFIED_ORIGINAL_EXPORT_RECEIPT_NOT_SUPPLIED")
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        if result["stage"] in {"feature-branch", "append", "ge-roundtrip"}:
            result["status"] = "FAILED"
    finally:
        if original_record is not None:
            try:
                same = (_stamp(Path(original_record["path"])) == tuple(original_record["stat"])
                        and sha256_file(Path(original_record["path"])) == original_record["sha256"])
            except OSError:
                same = False
            result["original_preservation"] = "UNCHANGED_STAT_AND_SHA256" if same else "CHANGED"
            if not same:
                result.update(status="FAILED", error="original AIR changed during append")
        if source_root and source_bound is not None:
            try:
                same = source_receipt(source_root) == source_bound
            except (OSError, ValueError):
                same = False
            result["source_preservation"] = "UNCHANGED_SIX_SOURCE_FILES" if same else "CHANGED"
            if not same:
                result.update(status="FAILED", error="original source changed during append")
        if source_receipt_path and source_bound is not None:
            try:
                current_receipt, current_sha = read_json(source_receipt_path, 1 << 20)
                same = (_stamp(Path(source_receipt_path)) == receipt_stamp and current_sha == receipt_sha
                        and current_receipt == source_bound)
            except (OSError, ValueError):
                same = False
            result["source_receipt_preservation"] = "UNCHANGED_STAT_AND_SHA256" if same else "CHANGED"
            if not same:
                result.update(status="FAILED", error="original source receipt changed during append")
        changed = []
        for path, identity in identities.items():
            try:
                same = str(Path(path).resolve()) == identity["resolved"] and sha256_file(Path(path)) == identity["sha256"]
            except OSError:
                same = False
            if not same:
                changed.append(path)
        if changed:
            result.update(status="FAILED", error="declared feature/toolchain identity changed", changed_identities=changed)
        if result["status"] == "FEATURE_APPEND_VERIFIED_PENDING_PUBLICATION":
            try:
                destination = out / "air/export.air"
                destination.parent.mkdir()
                partial = destination.with_name(".export.air.partial")
                with partial.open("xb") as stream:
                    stream.write(candidate_raw)
                if sha256_file(partial) != candidate_record["sha256"]:
                    raise ValueError("published candidate bytes differ from verified GE output")
                os.link(partial, destination)
                partial.unlink()
                result.update(status="FEATURE_AIR_EXPORTED", air={"path": str(destination),
                    "bytes": len(candidate_raw), "sha256": candidate_record["sha256"]})
            except Exception as error:
                result.update(status="FAILED", stage="publication", error=f"{type(error).__name__}: {error}")
        result["logs"] = {name: {"bytes": (work / name).stat().st_size, "sha256": sha256_file(work / name)}
                          for name in ("build.log", "branch.log") if (work / name).is_file()}
        save_json(out / "result.json", result)
    return result
