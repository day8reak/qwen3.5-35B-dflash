"""Build a feature-only GE branch and append it to a source-bound original AIR.

No checkpoint or external FileConstant payload is copied, opened or rewritten.
The resulting AIR still needs weight staging, actual OMG and HIAI validation.
"""
from __future__ import annotations

import ast
import json
import hashlib
import os
from pathlib import Path, PurePosixPath
import sys

from .air_schema import load_torchair_schema
from .artifacts import output_path, save_json
from .ge_export import _logged
from .ge_fused.export import _toolchain
from .hashing import sha256_file
from .original_air_inspect import inspect_original_air
from .original_air_roundtrip import _decode, _proof, _read_air, _stamp, _unavailable_payloads, probe_original_air_roundtrip
from .original_ge_target import SOURCE_SHA256, audit_source, read_json, source_receipt
from .original_ge_append_cpp import emit_feature_branch_cpp
from .original_ge_append_validation import validate_original_ge_append
from .original_ge_append_wire import append_feature_branch_wire
from .original_ge_taps import plan_original_ge_taps


_OWN_FILES = {"branch.cpp", "branch", "branch.air", "native.json", "build.log", "branch.log", "appended.air"}


# Fingerprints only from the already audited exact source receipts. No source
# text or runtime/weight data is embedded; AST matches never relax byte policy.
_SOURCE_AST_BASELINES = {
    "models/export_model_wrapper_qwen3_5_moe.py": {
        "bytes": 21326, "lines": 572, "ast_sha256": "5de5bf0dff2eb38a5c970acccf544aaf79e2676eaf7d4de1046b41944a92ab1b",
        "definitions": {
            "_load_text_config": "04d9803bb13feb320fa45810dc8f8a228d4b3b880f001d116037613079f8a073",
            "_load_hf_state_dict": "b0d0707a4adeefdcaef899394b65286c6097fa5e15818dc30b8d75e8aeb4c097",
            "load_npu_model": "bd7ca2c38dcc19e29f233e5903b6e0bd5d36a00e40511cb16a6ba2ceae1dfc81",
            "Qwen3_5MoeForCausalLMWrapper": "eaa4670b9eac497fd5efabec10a1d51f6e86edccd91bcb47c65fc20b2a8ff196",
            "Qwen3_5MoeForCausalLMWrapper.__init__": "01c0a00db02dcded9cbe67f052f894d691e8ac540b11692db2335283244f7162",
            "Qwen3_5MoeForCausalLMWrapper.forward": "769f3a5d7d1354f34151c24a91d45aefa73193f45385386244e1626ac7e87ffb",
            "Qwen3_5MoeForCausalLMCproWrapper": "79de2653e5ec1499bfbc95aed7c502f3133c9be8eca4ee3635c48dc7e82a2d8a",
            "Qwen3_5MoeForCausalLMCproWrapper.__init__": "bc41bce6d66c35c4c09098275781d0d0fe738ddaf0fedfd2160018b4548bfc65",
            "Qwen3_5MoeForCausalLMCproWrapper.forward": "5d9564cf30615d7714c9b17dd77374691867479f0fb1d3e2b60f12982a095911",
            "Qwen3_5MoeForCausalLMFirstWrapper": "56f91ac02e93095c5ff716ec3ddea7722f6a934b6e2115d1f1d8aef61d7ef089",
            "Qwen3_5MoeForCausalLMFirstWrapper.__init__": "2e9f343f7dc4f0db84a38647e43046269f12e80a63438b5be02f01efb4fbc5ac",
            "Qwen3_5MoeForCausalLMFirstWrapper.forward": "12c84ad30a0f03e887f151100c1b2c037a9ca41acbe0ac108634eded3aeee4b9",
            "Qwen3_5MoeForCausalLMSecondWrapper": "da90f1392e41378458b7caaa5cfb2f2c04247fb8294fc17a324384ab029477f7",
            "Qwen3_5MoeForCausalLMSecondWrapper.__init__": "2e6c1b76112f761cdc1c422cc08e101e5b73cdd72345edff465923cc0973b663",
            "Qwen3_5MoeForCausalLMSecondWrapper.forward": "390881b1b7f1a17f83a9ea2a5388001a3d6044742cf87603bed002bdd1f9068b",
        }},
    "utils.py": {
        "bytes": 26637, "lines": 631, "ast_sha256": "c797c9f8564dd284ea078009f3f088b5732bf7f26793b7aaee94dbc1ab079681",
        "definitions": {
            "nd_to_packed_zn": "89e2b7d03679bbf481868c66ed782cae76244942a51fb36e4cc22e29d5d60255",
            "nd_to_packed_i4zn": "b7397559ce5831d90f35cc85117182a349738ec1b1900a02f107debd0852e241",
            "nd_to_zn_int8": "53aab936be702ee3199a106364283295bb2d4e87fd497e3efdbfa6c86137fd00",
            "nd_to_blocked_zn_int8": "bf3531d13a5e1bd203f7fe471b9753ac6ae035e16c99593f5b4f5da87432c4d0",
            "convert_int4_to_qbm_zn": "d9d3f73bb804e2433247d75635a21a0b8e2700af60f790a3c4b5e3fb33e3a7cf",
            "pack_int4_to_int32_simple": "e60405c77f31cb74e5ba5a5219ff08fffc7664e705f68a76b7c25a4e5df6ccc0",
            "nd_to_i4zn": "76178de8b62b1b950102bfb0bd6927cbfd390d07517f53180a55df052adf90a4",
            "replace_module_by_names": "2a585ef6a98dff221308dcb6b6024aad909b2635f2b3edc519d121057d4ccf37",
            "replace_linear_to_QLinear": "5b7ec3d59f70b7c11b9f35795997808608f46d66dc225c615119d414691eaadf",
            "replace_linear_to_PseudoQuantLinear": "75f64da2beb8a1b4a964ba52683d87579bac9cdf7f4d9393d19f1a0a9e5a9ec7",
            "combined_experts_weight_quant_3d": "3766c0fc9cfd07408beae4a4683732d2b22ef9afb01e04195625f1f0c29b24a6",
            "combined_experts_weight_3d": "83f6b095381765a3c032cfc4101bae5f16f5952b5c9030d7bbfbec292780f11b",
            "load_router_weight": "8588c69d4a041d40d650da7cf6036788d183f4ea2870dc0838288503bdfa34ca",
            "load_rmsnorm_weight": "7f38f57e27a813fd06847a222e63e6002b4bc9f4104c2672022f07417c117e27",
            "quant_model": "9d7c96e6714acebc18c23c8023e455b117abe33e4bd5474942071fe4ef442b48",
            "combined_moe_weight": "31dd3ada6739443a5ec21b3828f435acbb3cc4df675786832cad23a1822f54e2",
            "quant_model_inplace": "a795cada9c77c7ecefc1583130d2f2f005c3764fafce54e03c0e752ce02f850a",
        }},
}


def _source_ast_diagnostic(name, raw):
    baseline = _SOURCE_AST_BASELINES[name]
    result = {"scope": "Diagnostic only; exact raw-byte SHA gate is unchanged, no source-to-AIR equivalence",
        "algorithm": "sha256(UTF8(ast.dump(ast.parse(source),include_attributes=False)))",
        "baseline_python": "3.10", "actual_python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "expected_ast_sha256": baseline["ast_sha256"], "expected_bytes": baseline["bytes"], "expected_lines": baseline["lines"],
        "actual_lines": len(raw.splitlines()), "bytes_delta": len(raw) - baseline["bytes"]}
    try:
        tree = ast.parse(raw)
        if sum(1 for _ in ast.walk(tree)) > 50000:
            raise ValueError("AST node count exceeds diagnostic bound")
        digest = lambda node: hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
        result["actual_ast_sha256"] = digest(tree)
        result["matches_expected_ast"] = result["actual_ast_sha256"] == baseline["ast_sha256"]
        definitions = {}
        def collect(nodes, prefix=""):
            for node in nodes:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    qualified = prefix + node.name
                    definitions.setdefault(qualified, []).append(node)
                    if isinstance(node, ast.ClassDef):
                        collect(node.body, qualified + ".")
        collect(tree.body)
        comparison, changed, excerpts = [], [], []
        for qualified in sorted(set(definitions) | set(baseline["definitions"])):
            nodes = definitions.get(qualified, [])
            node = nodes[0] if len(nodes) == 1 else None
            actual = digest(node) if node is not None else None
            expected = baseline["definitions"].get(qualified)
            comparison.append({"name": qualified, "expected_ast_sha256": expected, "actual_ast_sha256": actual,
                "matches_expected_ast": actual is not None and actual == expected, "occurrences": len(nodes),
                "line": getattr(node, "lineno", None), "end_line": getattr(node, "end_lineno", None)})
            if actual is None or actual != expected:
                changed.append(qualified)
                relevant = (name == "utils.py" or qualified in {"load_npu_model", "Qwen3_5MoeForCausalLMCproWrapper.__init__", "Qwen3_5MoeForCausalLMCproWrapper.forward"})
                if relevant and node is not None and not isinstance(node, ast.ClassDef) and len(excerpts) < 2:
                    text = ast.unparse(node)
                    excerpts.append({"name": qualified, "text": text if len(text) <= 1536 else text[:768] + "\n[... bounded AST excerpt ...]\n" + text[-768:],
                                     "truncated": len(text) > 1536, "original_ast_text_characters": len(text)})
        result.update(status="AST_MATCHED_DIAGNOSTIC_ONLY" if result["matches_expected_ast"] else "AST_DIFFERENT",
                      definitions=comparison, changed_definitions=changed, bounded_relevant_ast_excerpts=excerpts)
    except (SyntaxError, ValueError, TypeError, RecursionError) as error:
        result.update(status="AST_UNAVAILABLE", error=f"{type(error).__name__}: {error}"[:256])
    return result


def inspect_original_ge_source(*, source_root=None, source_receipt_path=None, source_ast=False):
    """Audit only six bounded source files, or their supplied receipt.

    Line-ending normalization is diagnostic evidence, never acceptance. This
    read-only entry point requires neither AIR nor a GE toolchain/run directory.
    """
    if bool(source_root) == bool(source_receipt_path):
        raise ValueError("provide exactly one original source root or source receipt")
    receipt, digest, diagnostics, errors = None, None, {}, []
    if source_root:
        root, rows = Path(source_root).expanduser().resolve(), []
        for name in SOURCE_SHA256:
            path = root / name
            try:
                if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= 1 << 20:
                    raise ValueError("missing, symlinked or excessive original source file")
                before = _stamp(path)
                with path.open("rb") as stream:
                    raw = stream.read((1 << 20) + 1)
                if len(raw) != before[2] or len(raw) > 1 << 20 or _stamp(path) != before:
                    raise ValueError("source changed during bounded read")
                rows.append({"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
                crlf, lf, cr = raw.count(b"\r\n"), raw.count(b"\n"), raw.count(b"\r")
                normalized = raw.replace(b"\r\n", b"\n")
                normalized_sha = hashlib.sha256(normalized).hexdigest()
                diagnostics[name] = {"status": "ACTUAL_BOUNDED_BYTES_READ", "crlf_pairs": crlf,
                    "bare_lf": lf - crlf, "bare_cr": cr - crlf,
                    "normalized_lf_sha256": normalized_sha,
                    "normalized_lf_matches_expected": normalized_sha == SOURCE_SHA256[name],
                    "scope": "CRLF-to-LF only; diagnostic candidate, never accepted by source policy"}
                if source_ast and name in _SOURCE_AST_BASELINES:
                    diagnostics[name]["python_ast"] = _source_ast_diagnostic(name, raw)
            except (OSError, ValueError) as error:
                errors.append({"path": name, "error": f"{type(error).__name__}: {error}"})
        receipt = {"source": "READ_ONLY_SIX_SOURCE_FILES", "files": rows}
    else:
        try:
            receipt, digest = read_json(source_receipt_path, 1 << 20)
        except (OSError, ValueError) as error:
            errors.append({"path": str(source_receipt_path), "error": f"{type(error).__name__}: {error}"})
    try:
        audit = audit_source(receipt or {}, source_root=source_root)
    except (ValueError, TypeError) as error:
        audit = {"status": "SOURCE_AUDIT_INPUT_UNAVAILABLE", "error": f"{type(error).__name__}: {error}"}
    rows = audit.get("files", (receipt or {}).get("files", []))
    comparison = []
    for name, expected in SOURCE_SHA256.items():
        matches = [row for row in rows if isinstance(row, dict) and row.get("path") == name] if isinstance(rows, list) else []
        actual = matches[0] if len(matches) == 1 else {}
        comparison.append({"path": name, "role": "COMPUTATION_SOURCE" if name.endswith(".py") else "CONFIGURATION_PROVENANCE_ONLY",
            "expected_sha256": expected, "actual_sha256": actual.get("sha256"), "actual_bytes": actual.get("bytes"),
            "matches_expected_bytes_sha256": actual.get("sha256") == expected,
            "receipt_occurrences": len(matches),
            "line_endings": diagnostics.get(name, {"status": "NOT_AVAILABLE_FROM_RECEIPT_OR_UNREADABLE_SOURCE"})})
        if audit.get("source_revision_evidence", {}).get("status") == "VERIFIED_ACTUAL_SIX_FILE_SOURCE_ROOT":
            accepted = {row["path"]: row for row in audit["source_revision_evidence"]["files"]}.get(name)
            if accepted:
                comparison[-1]["reviewed_documentation_revision_sha256"] = accepted["sha256"]
                comparison[-1]["matches_reviewed_documentation_revision"] = actual.get("sha256") == accepted["sha256"]
        if "python_ast" in comparison[-1]["line_endings"]:
            comparison[-1]["python_ast"] = comparison[-1]["line_endings"].pop("python_ast")
    audit["comparison"] = comparison
    audit["input_errors"] = errors
    if errors and audit["status"] == "MATCHED_AUDITED_SOURCE_RECEIPTS":
        audit["status"] = "SOURCE_AUDIT_INPUT_UNAVAILABLE"
    return {"schema_version": 1, "artifact_kind": "qwen35-original-ge-source-inspection",
        "status": "SOURCE_AUDIT_MATCHED" if audit["status"] == "MATCHED_AUDITED_SOURCE_RECEIPTS" else "BLOCKED",
        "source_audit": audit, "source_receipt": receipt, "source_receipt_sha256": digest,
        "actual_source_bytes_rehashed": bool(source_root), "model_compilation": "NOT_RUN", "device_execution": "NOT_RUN",
        "source_ast_requested": bool(source_ast),
        "external_weights": "NOT_OPENED", "scope": "Six named small source/config files only; exact Python SHA policy unchanged"}


def _identity(paths):
    return {str(Path(path).absolute()): {"sha256": sha256_file(Path(path)), "bytes": Path(path).stat().st_size,
                                      "resolved": str(Path(path).resolve())} for path in paths}


def _archive_source_failure(out):
    """Preserve one source-only failed attempt; never reuse graph/native work."""
    if out.is_symlink() or not out.is_dir():
        raise ValueError("retry requires a regular existing output directory")
    lock = out / ".source-retry.lock"
    with lock.open("x") as stream:
        stream.write("source-only evidence archive\n")
    try:
        allowed = {"result.json", "source-receipt.json", "source-audit.json", "work", "attempts", lock.name}
        if any(path.name not in allowed or path.is_symlink() for path in out.iterdir()):
            raise ValueError("retry refuses unknown files or symlinks in previous output")
        result, _ = read_json(out / "result.json", 1 << 20)
        advanced = {"original_air", "schema", "original_metadata", "tap_plan", "toolchain", "commands", "branch_native",
                    "branch_air", "wire_preservation", "append_validation", "ge_roundtrip", "roundtrip_validation", "air", "feature_output"}
        if (result.get("artifact_kind") != "qwen35-original-ge-feature-append"
                or result.get("status") != "BLOCKED" or result.get("stage") != "source"
                or result.get("model_compilation", "NOT_RUN") != "NOT_RUN"
                or result.get("device_execution", "NOT_RUN") != "NOT_RUN"
                or result.get("output") != str(out) or advanced & result.keys()):
            raise ValueError("retry is limited to a BLOCKED source-only attempt with no AIR, binding or native evidence")
        work = out / "work"
        if not work.is_dir() or any(work.iterdir()):
            raise ValueError("retry refuses nonempty or missing native work directory")
        files = [out / name for name in ("result.json", "source-receipt.json", "source-audit.json") if (out / name).exists()]
        identities, stamps = {}, {}
        for path in files:
            _, digest = read_json(path, 1 << 20)
            identities[path.name] = {"sha256": digest, "bytes": path.stat().st_size}
            stamps[path.name] = _stamp(path)
        attempts = out / "attempts"
        attempts.mkdir(exist_ok=True)
        for index in range(1, 10001):
            archive = attempts / f"source-{index:04d}"
            try:
                archive.mkdir()
                break
            except FileExistsError:
                continue
        else:
            raise ValueError("source attempt archive count exceeds its bound")
        save_json(archive / "archive.json", {"schema_version": 1, "artifact_kind": "qwen35-original-ge-source-attempt",
            "files": identities, "scope": "Preserved source-only blocked attempt; no graph/native evidence reused"})
        for path in files:
            if _stamp(path) != stamps[path.name] or sha256_file(path) != identities[path.name]["sha256"]:
                raise ValueError("previous source evidence changed before archiving")
            path.rename(archive / path.name)
        return str(archive)
    finally:
        lock.unlink()  # Only the lock created exclusively by this invocation.


def append_original_ge_features(air, output, *, torchair_pb2, gamma_template, sequence_length,
                                source_root=None, source_receipt_path=None, ge_root=None,
                                cxx="c++", cxx11_abi=0, timeout_seconds=120, retry=False):
    if type(retry) is not bool or Path(output).expanduser().is_symlink():
        raise ValueError("retry must be boolean and output cannot be a symlink")
    out = output_path(output)
    archive = _archive_source_failure(out) if out.exists() and retry else None
    out.mkdir(parents=True, exist_ok=bool(archive))
    work = out / "work"
    work.mkdir(exist_ok=bool(archive))
    result = {"schema_version": 1, "artifact_kind": "qwen35-original-ge-feature-append",
              "status": "BLOCKED", "stage": "source", "output": str(out),
              "model_compilation": "NOT_RUN", "device_execution": "NOT_RUN", "numerical_validation": "NOT_RUN",
              "external_weights": "NOT_OPENED_NOT_COPIED_NOT_REWRITTEN",
              "scope": "Source-anchored original AIR plus auxiliary features only; no full-model accuracy, state continuity or speed pass"}
    if archive:
        result["previous_source_attempt"] = archive
    original, original_record, source_bound, identities = None, None, None, {}
    receipt_stamp, receipt_sha, candidate_raw, candidate_record = None, None, None, None
    try:
        if source_receipt_path:
            try:
                receipt_stamp = _stamp(Path(source_receipt_path))
            except OSError:
                pass  # The structured read-only source audit records this gap.
        source_inspection = inspect_original_ge_source(source_root=source_root, source_receipt_path=source_receipt_path)
        receipt, source_audit, receipt_sha = (source_inspection["source_receipt"], source_inspection["source_audit"],
                                              source_inspection["source_receipt_sha256"])
        source_bound = receipt if "files" in source_audit else None
        result["source_audit"] = source_audit
        result["source_receipt_sha256"] = receipt_sha
        if receipt is not None:
            save_json(out / "source-receipt.json", receipt)
        save_json(out / "source-audit.json", source_inspection)
        if source_audit["status"] != "MATCHED_AUDITED_SOURCE_RECEIPTS":
            raise ValueError("original source audit did not match; inspect source_audit.comparison for exact byte differences")
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
        for name in ("original_ge_append_cpp.py", "original_ge_append_wire.py", "original_ge_append_validation.py", "original_ge_target.py",
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
