#!/usr/bin/env python3
"""Compile/run fused-op probes, or inspect an existing batch without compilation or HDC."""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys
import uuid

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "model"))

from qwen35_moe_dflash.artifacts import output_path, save_json
from qwen35_moe_dflash.compile import _stateful_json, check_tbe_platform
from qwen35_moe_dflash.elf_dynamic import dynamic_info
from qwen35_moe_dflash.ge_fused.bindings import OperatorBindings, write_document_bindings
from qwen35_moe_dflash.ge_fused.export import _step_file
from qwen35_moe_dflash.ge_fused.probes import (
    _CASES, _board_script, _validate_probe_manifest, _wire, build_fused_probes, compare_fused_probes, inspect_fused_probes,
)
from qwen35_moe_dflash.hashing import sha256_file
from qwen35_moe_dflash.hdc_run import Hdc, _file_record, _positive, _relative, _text, _upload
from qwen35_moe_dflash.hiai_deployment import (
    BOARD_LIBRARY_SEARCH_PATHS, DDK_HCL_EXCLUDED, candidate_blockers, discover_libraries,
)

_MODEL_CASES = tuple(case for case in _CASES if case[1] in {"gdn", "cache", "attention"})


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    source = result.add_mutually_exclusive_group()
    source.add_argument("--probe-run", default=os.environ.get("FUSED_PROBE_RUN"), help="existing compiled probe directory; defaults to FUSED_PROBE_RUN")
    source.add_argument("--fresh", action="store_true",
                        help="with --model-ops or --all-ops, regenerate bindings, TBE diagnosis and selected probes; ignore old run environment variables")
    result.add_argument("--output", help="new directory below AI_RUN_DIR; otherwise allocated automatically")
    result.add_argument("--all-ops", action="store_true",
                        help="reuse compiled cases, compile missing cases for all five custom operators, then run all compiled cases")
    result.add_argument("--model-ops", action="store_true",
                        help="run only GDN, CacheUpdate and Attention (six cases); Conv/RMS use the base graph path")
    result.add_argument("--inspect-only", action="store_true",
                        help="read existing probe/all-ops logs and selected-DDK registration text; no compilation or board execution")
    result.add_argument("--retry-cases", help="with --model-ops or --all-ops, compile only these missing cases; --probe-run may be an earlier batch")
    result.add_argument("--timeout-seconds", type=_positive, default=3600)
    return result


def _prepare(root, *, allow_empty=False, case_names=None):
    manifest = _stateful_json(root / "manifest.json")
    _validate_probe_manifest(manifest)
    if manifest.get("platform") != "kirinx90":
        raise ValueError("compiled probe platform must be kirinx90")
    bindings = OperatorBindings.from_file(_step_file(root, manifest["bindings"]))
    if bindings.manifest() != manifest["binding_contract"]:
        raise ValueError("probe bindings differ from the recorded contract")
    script = _step_file(root, manifest["run_script"])
    if script != root / "bundle/run-board.sh":
        raise ValueError("probe batch script must be bundle/run-board.sh")
    uploads = {"bundle/run-board.sh": _file_record(script, "bundle/run-board.sh")}
    if case_names is not None:
        if (not case_names or len(set(case_names)) != len(case_names)
                or not set(case_names) <= {case["name"] for case in manifest["cases"]}):
            raise ValueError("selected cases must be unique names from this probe manifest")
    compiled = []
    for case in manifest["cases"]:
        if case_names is not None and case["name"] not in case_names:
            continue
        if case["status"] != "COMPILED":
            continue
        if case.get("compile") != "PASSED" or case.get("ddk_id") != manifest["ddk_id"]:
            raise ValueError("compiled case lacks matching successful compilation identity")
        for record in case["files"]:
            _step_file(root, record)
        step_path = _step_file(root, case["bundle_manifest"])
        folder = root / "bundle" / case["name"]
        if step_path != folder / "manifest.json":
            raise ValueError("compiled bundle path differs from its case")
        step, abi = _stateful_json(step_path), _stateful_json(_step_file(root, case["operator_abi"]))
        if (step.get("artifact_kind") != "qwen35-fused-op-step" or step.get("status") != "PREPARED"
                or step.get("operator_type") != case["operator_type"]
                or abi.get("operator_type") != case["operator_type"]
                or abi.get("binding_sha256") != bindings.sha256 or step.get("outputs") != abi["outputs"]
                or _stateful_json(_step_file(folder, step["operator_abi"])) != abi):
            raise ValueError("compiled bundle operator ABI mismatch")
        source_model = _step_file(root, case["model"])
        if (step["model"]["sha256"] != sha256_file(source_model)
                or step["model"]["bytes"] != source_model.stat().st_size
                or step["model"] not in step["model_payloads"]):
            raise ValueError("bundle model does not match the compiled OMC")
        if len(step["inputs"]) != len(case["inputs"]):
            raise ValueError("bundle input count mismatch")
        for actual, expected in zip(step["inputs"], case["inputs"]):
            if any(actual.get(key) != expected[key] for key in ("index", "name", "dtype", "shape", "bytes", "sha256")):
                raise ValueError("bundle inputs differ from the compiled probe")
        if _step_file(folder, step["step_plan"]).read_bytes() != _wire(
                step["model"], step["operator_abi"], step["operator_type"], step["inputs"], step["outputs"]):
            raise ValueError("bundle plan differs from the operator ABI")
        records = [step["operator_abi"], step["step_plan"], *step["inputs"], *step["model_payloads"]]
        for path in [step_path, *[_step_file(folder, record) for record in records]]:
            remote = path.relative_to(root).as_posix()
            uploads[remote] = _file_record(path, remote)
        compiled.append(case)
    if not compiled and not allow_empty:
        raise ValueError("no successfully compiled fused probe cases; no model compilation is performed here")
    return manifest, compiled, list(uploads.values())


def _build_runner(out, timeout):
    command = [sys.executable, str(REPO / "targets/cpro/scripts/build_hiai.py"), "--output", str(out / "native")]
    save_json(out / "runner-build-command.json", {"argv": command})
    with (out / "runner-build.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=True)
    runner = out / "native/build/qwen35_dflash_hiai"
    record = _stateful_json(out / "native/result.json")
    if record.get("status") != "LINKED" or record.get("binary_sha256") != sha256_file(runner):
        raise ValueError("new runner does not match its LINKED build record")
    if record.get("command_sha256") != sha256_file(out / "native/command.json"):
        raise ValueError("runner link command hash mismatch")
    dynamic_info(runner)
    return runner


def _receive(hdc, board, out, name):
    _relative(name)
    remote = shlex.quote(str(board / name))
    _, text = hdc.shell(f"if test -f {remote}; then sha256sum {remote}; else printf '__MISSING__\\n'; fi", "hash-result", timeout=min(30, hdc.timeout))
    if text == "__MISSING__":
        return None
    if len(text.splitlines()) != 1 or not re.match(r"^[0-9a-f]{64} [ *]", text):
        raise ValueError("invalid remote SHA256 for " + name)
    local = out / name
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.exists():
        raise FileExistsError(local)
    temporary = local.with_name(local.name + ".receiving")
    hdc.receive(board / name, temporary, timeout=min(30, hdc.timeout))
    if sha256_file(temporary) != text[:64]:
        raise ValueError("retrieved SHA256 mismatch: " + name)
    temporary.rename(local)
    return text[:64]


def _tail(path):
    if path.is_file():
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 6000))
            text = stream.read().decode("utf-8", errors="replace").strip()
        if text:
            print(str(path) + ":\n" + "\n".join(text.splitlines()[-35:]), file=sys.stderr)


def execute(args, *, runner_cache=None):
    storage = Path(_text(os.environ.get("AI_RUN_DIR"), "AI_RUN_DIR"))
    out = output_path(args.output or storage / ("fused-board-" + uuid.uuid4().hex[:12]))
    out.mkdir(parents=True, exist_ok=False)
    result = {"status": "FAILED", "output": str(out), "stage": "preflight", "model_compilation": "NOT_RUN",
              "numerical_acceptance": "NOT_DEFINED", "runtime_placement": "UNVERIFIED", "board_cleanup": "RETAINED",
              "retrieved_sha256": {}, "retrieval_errors": []}
    hdc, board, cases, root = None, None, [], None
    execution_attempted = False

    def collect(names):
        for name in names:
            if name in result["retrieved_sha256"]:
                continue
            try:
                digest = _receive(hdc, board, out, name)
                if digest:
                    result["retrieved_sha256"][name] = digest
            except Exception as error:
                result["retrieval_errors"].append({"path": name, "error": f"{type(error).__name__}: {error}"})

    try:
        root = Path(_text(args.probe_run, "FUSED_PROBE_RUN / --probe-run")).expanduser().resolve()
        if out == root or root in out.parents:
            raise ValueError("board output must be separate from the compiled probe directory")
        selected = getattr(args, "selected_cases", None)
        manifest, cases, uploads = _prepare(root, case_names=selected)
        output_fill_byte = getattr(args, "output_fill_byte", None)
        if selected is not None or output_fill_byte is not None:
            # Keep the recorded script immutable; stage a selected script under
            # the same remote bundle path so relative per-case paths still work.
            script = out / "selected-run-board.sh"
            script.write_text(_board_script([case["name"] for case in cases], output_fill_byte=output_fill_byte))
            uploads = [entry for entry in uploads if entry["remote"] != "bundle/run-board.sh"]
            uploads.insert(0, _file_record(script, "bundle/run-board.sh"))
        if output_fill_byte is not None:
            result["output_fill_byte"] = output_fill_byte
        ddk = _text(manifest.get("ddk_id"), "compiled DDK identity")
        if os.environ.get("CPRO_DDK_ID") and os.environ["CPRO_DDK_ID"] != ddk:
            raise ValueError("CPRO_DDK_ID differs from the compiled probe manifest")
        sn, bip = _text(os.environ.get("SN"), "SN"), _text(os.environ.get("BIP"), "BIP")
        host, separator, port = bip.rpartition(":")
        if (sn.startswith("-") or bip.startswith("-") or not separator or not host
                or any(c.isspace() for c in host) or not port.isdigit() or not 1 <= int(port) <= 65535):
            raise ValueError("SN must be a device ID and BIP must be an HDC server IP:port")
        hdc = Hdc(_text(os.environ.get("HDC", "hdc"), "HDC"), sn, bip, out, args.timeout_seconds)
        board = PurePosixPath("/data/local/tmp/qwen35-fused-" + uuid.uuid4().hex)
        result.update(probe_run=str(root), probe_manifest_sha256=sha256_file(root / "manifest.json"),
                      selected_cases=[case["name"] for case in cases],
                      skipped_cases=[case["name"] for case in manifest["cases"] if case["name"] not in {item["name"] for item in cases}],
                      ddk_id=ddk, serial=sn, hdc_server=bip, board_root=str(board))
        print("fused-board: checking HDC and board library paths", flush=True)
        private = os.environ.get("BOARD_HIAI_LIB_DIR")
        paths = list(BOARD_LIBRARY_SEARCH_PATHS)
        if private:
            _text(private, "BOARD_HIAI_LIB_DIR")
            if not private.startswith("/") or ":" in private or ".." in PurePosixPath(private).parts:
                raise ValueError("BOARD_HIAI_LIB_DIR must be an absolute board directory without ':' or '..'")
            paths.insert(0, private)
        hdc.shell("command -v sha256sum >/dev/null && " + " && ".join(
            "test -d " + shlex.quote(path) + " && test -r " + shlex.quote(path) for path in paths), "preflight")
        result["stage"] = "build-runner"
        if runner_cache and "runner" in runner_cache:
            result["stage"] = "reuse-runner"
            print("fused-board: reusing this batch's runner and existing OMC files", flush=True)
            runner = runner_cache["runner"]
            if sha256_file(runner) != runner_cache["sha256"]:
                raise ValueError("runner changed during this batch")
            native_build = runner_cache["native_build"]
        else:
            print("fused-board: building current runner; reusing compiled OMC files", flush=True)
            runner = _build_runner(out, args.timeout_seconds)
            native_build = out / "native"
            if runner_cache is not None:
                runner_cache.update(runner=runner, sha256=sha256_file(runner), native_build=native_build)
        result["runner_build"] = str(native_build)
        uploads.append(_file_record(runner, "runner"))
        result["stage"] = "select-libraries"
        if private:
            result["libraries"] = {"mode": "existing-board-directory", "path": private, "compatibility": "UNVERIFIED"}
        else:
            discovery = discover_libraries(runner, native_build)
            libraries = [_file_record(path.resolve(), "lib/" + str(_relative(path.name))) for path in discovery["libraries"]]
            dependencies = [{"name": PurePosixPath(entry["remote"]).name, "kind": "runner" if entry["remote"] == "runner" else "library",
                             "sha256": entry["sha256"], **dynamic_info(entry["local"])} for entry in [uploads[-1], *libraries]]
            blockers = candidate_blockers(dependencies, require_pair=False)
            if blockers:
                raise ValueError("cannot select board HCL deployment: " + "; ".join(blockers))
            uploads.extend(entry for entry in libraries if PurePosixPath(entry["remote"]).name not in DDK_HCL_EXCLUDED)
            paths.insert(0, str(board / "lib"))
            result["libraries"] = {"mode": "candidate_without_ddk_hcl", "discovery": discovery["metadata"],
                                   "excluded_names": sorted(DDK_HCL_EXCLUDED), "dependencies": dependencies,
                                   "backend_selection": "UNVERIFIED"}
        result["stage"] = "upload"
        save_json(out / "upload-plan.json", {"board_root": str(board), "files": uploads})
        hdc.shell("mkdir " + shlex.quote(str(board)), "create-root")
        directories = sorted({str((board / entry["remote"]).parent) for entry in uploads})
        hdc.shell("mkdir -p " + " ".join(map(shlex.quote, directories)), "create-subdirs")
        _upload(hdc, board, uploads, out / "payloads.sha256", checksum_connection_retries=2)
        hdc.shell("chmod 755 " + shlex.quote(str(board / "runner")), "runner-permissions")
        environment = "LD_LIBRARY_PATH=" + shlex.quote(":".join(paths)) + " "
        result["stage"] = "runtime-probe"
        try:
            hdc.shell("cd " + shlex.quote(str(board)) + " && " + environment + "./runner --probe > runtime.json 2> runtime.stderr; "
                      "rc=$?; printf '%s\\n' \"$rc\" > " + shlex.quote(str(board / "runtime.exitcode")) + "; exit \"$rc\"", "runtime-probe")
        finally:
            collect(("runtime.json", "runtime.stderr", "runtime.exitcode"))
        if any(name not in result["retrieved_sha256"] for name in ("runtime.json", "runtime.stderr", "runtime.exitcode")):
            raise RuntimeError("runtime probe files were not all retrieved and hash-verified")
        runtime_probe = _stateful_json(out / "runtime.json")
        if (out / "runtime.exitcode").read_text().strip() != "0" or runtime_probe.get("npu_supported") is not True:
            raise RuntimeError("runtime capability probe failed; see runtime.stderr")
        runtime = _text(runtime_probe.get("runtime_version"), "probed runtime version")
        if os.environ.get("HIAI_RUNTIME_VERSION") and os.environ["HIAI_RUNTIME_VERSION"] != runtime:
            raise ValueError("HIAI_RUNTIME_VERSION differs from the actual runtime probe")
        firmware = os.environ.get("FIRMWARE_ID")
        if not firmware:
            _, firmware = hdc.shell("param get const.ohos.fullname", "firmware")
            firmware = firmware.strip()
            if re.search(r"errNum|Get parameter.*fail|error:", firmware, re.I):
                raise ValueError("firmware query failed; supply the actual FIRMWARE_ID")
        firmware = _text(firmware, "firmware identity")
        result.update(runtime_version=runtime, firmware_id=firmware, stage="execute")
        print("fused-board: executing " + ", ".join(result["selected_cases"]), flush=True)
        execution_attempted = True
        command = environment + shlex.join(["sh", str(board / "bundle/run-board.sh"), str(board / "runner"),
                                            str(board / "results"), runtime, ddk, firmware])
        try:
            status, _ = hdc.shell(command + " > " + shlex.quote(str(board / "batch.stdout")) + " 2> "
                                 + shlex.quote(str(board / "batch.stderr")) + "; rc=$?; printf '%s\\n' \"$rc\" > "
                                 + shlex.quote(str(board / "batch.exitcode")) + "; exit \"$rc\"", "execute", allow_failure=True)
            result["remote_exitcode"] = status
        except Exception as error:
            result["execution_error"] = f"{type(error).__name__}: {error}"
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        if execution_attempted:
            collect(("batch.stdout", "batch.stderr", "batch.exitcode"))
            for case in cases:
                name = case["name"]
                collect([f"results/{name}.{suffix}" for suffix in ("stdout", "stderr", "exitcode")]
                        + [f"results/{name}/result.json"]
                        + [f"results/{name}/output_{item['index']:04d}.bin" for item in case["outputs"]])
            try:
                if sha256_file(root / "manifest.json") != result["probe_manifest_sha256"]:
                    raise ValueError("probe manifest changed during deployment")
                comparison = compare_fused_probes(root, out / "results", out / "comparison",
                                                 **({"case_names": selected} if selected is not None else {}))
                result["comparison"] = str(out / "comparison/report.txt")
                rows = {case["name"]: case for case in comparison["cases"]}
                result["identity_errors"] = []
                for case in cases:
                    name = f"results/{case['name']}/result.json"
                    if name not in result["retrieved_sha256"]:
                        result["identity_errors"].append(case["name"] + ": result.json not hash-verified")
                        continue
                    native = _stateful_json(out / name)
                    runtime_key = "runtime_version" if native.get("status") == "EXECUTED" else "expected_runtime_version"
                    if native.get(runtime_key) != result["runtime_version"] or native.get("firmware_id") != result["firmware_id"]:
                        result["identity_errors"].append(case["name"] + ": runtime/firmware differs from this invocation")
                logs = [f"results/{case['name']}.{suffix}" for case in cases for suffix in ("stdout", "stderr", "exitcode")]
                all_logs = all(name in result["retrieved_sha256"] for name in ["batch.stdout", "batch.stderr", "batch.exitcode", *logs])
                exits = all((out / name).read_text().strip() == "0" for name in ["batch.exitcode", *[p for p in logs if p.endswith(".exitcode")]]) if all_logs else False
                if (result.get("remote_exitcode") == 0 and not result.get("execution_error") and not result.get("error")
                        and not result["retrieval_errors"] and not result["identity_errors"] and all_logs and exits
                        and all(rows[case["name"]].get("process") == "PASSED" for case in cases)):
                    result.update(status="EXECUTION_COMPLETE", stage="complete")
                else:
                    detail = (result.get("execution_error")
                              or next((item["error"] for item in result["retrieval_errors"]), None)
                              or "; ".join(result["identity_errors"]))
                    result.setdefault("error", detail or "operator execution or result verification failed; see comparison and retained logs")
            except Exception as error:
                result["comparison_error"] = f"{type(error).__name__}: {error}"
        result["hdc_invocations"] = hdc.count if hdc else 0
        save_json(out / "result.json", result)
        if result["status"] != "EXECUTION_COMPLETE":
            print(f"fused-board: FAILED stage={result['stage']}: "
                  f"{result.get('error') or result.get('comparison_error') or 'see result.json'}; "
                  f"evidence={out / 'result.json'}", file=sys.stderr, flush=True)
            for path in [out / "runner-build.log", out / "native/build.log", out / "runtime.stderr",
                         out / "batch.stderr", out / "batch.stdout", *sorted((out / "results").glob("*.stderr")),
                         *sorted((out / "results").glob("*.stdout"))]:
                _tail(path)
    return result


def _compile_missing(root, output, missing, *, tbe_diagnosis=None):
    manifest = _stateful_json(root / "manifest.json")
    diagnosis = os.environ.get("TBE_DIAG_JOB")
    return build_fused_probes(
        output, _step_file(root, manifest["bindings"]),
        ge_root=_text(os.environ.get("CANN_ROOT"), "CANN_ROOT"),
        ddk_root=_text(os.environ.get("CPRO_DDK_ROOT"), "CPRO_DDK_ROOT"),
        ddk_id=manifest["ddk_id"], cases=missing, conv_history_width=4,
        tbe_diagnosis=str(Path(diagnosis) / "diagnosis") if diagnosis else tbe_diagnosis,
    )


def _probe_groups(root, out):
    """Read one prior batch's explicit manifests, never recursively discover runs."""
    if (root / "manifest.json").is_file():
        paths, recorded = [root], {}
    else:
        batch = _stateful_json(root / "result.json")
        paths = batch.get("probe_runs")
        if (batch.get("model_compilation") not in {"MISSING_CASES_ONLY", "FRESH_ALL_CASES"} or not isinstance(paths, list)
                or not 1 <= len(paths) <= len(_CASES) or any(not isinstance(path, str) for path in paths)):
            raise ValueError("expected a probe manifest or all-ops result with at most nine explicit probe_runs")
        recorded = batch.get("probe_manifests", {})
        if not isinstance(recorded, dict):
            raise ValueError("invalid recorded probe manifest hashes")
        if any(not Path(path).expanduser().is_absolute() for path in paths):
            raise ValueError("recorded probe paths must be absolute")
        paths = [Path(_text(path, "recorded probe path")).expanduser().resolve() for path in paths]
        if len(set(paths)) != len(paths):
            raise ValueError("duplicate recorded probe directory")
        if recorded and set(recorded) != {str(path) for path in paths}:
            raise ValueError("recorded manifest hashes do not cover the probe directories")
    groups, compiled_names, identity = [], set(), None
    for path in paths:
        if path == out or path in out.parents or out in path.parents:
            raise ValueError("batch output must be separate from every recorded probe directory")
        digest = recorded.get(str(path))
        if digest is not None and digest != sha256_file(path / "manifest.json"):
            raise ValueError("recorded probe manifest SHA256 mismatch: " + str(path))
        manifest, compiled, _ = _prepare(path, allow_empty=True)
        current = (manifest["ddk_id"], manifest["binding_contract"])
        if identity is not None and identity != current:
            raise ValueError("recorded probe DDK or binding identities differ")
        identity = current
        names = {case["name"] for case in compiled}
        if compiled_names & names:
            raise ValueError("compiled cases occur in multiple recorded manifests: " + ", ".join(sorted(compiled_names & names)))
        compiled_names.update(names)
        groups.append((path, manifest))
    return groups, compiled_names


def _error_detail(error):
    text = str(error)
    return text if len(text) <= 16000 else text[:8000] + "\n[excerpt omitted; see full log]\n" + text[-8000:]


def inspect_existing(args):
    """Inspect explicit recorded groups and print a bounded failure summary."""
    root = Path(_text(args.probe_run, "FUSED_PROBE_RUN / --probe-run")).expanduser().resolve()
    ddk = Path(_text(os.environ.get("CPRO_DDK_ROOT"), "CPRO_DDK_ROOT")).expanduser().resolve()
    storage = Path(_text(os.environ.get("AI_RUN_DIR"), "AI_RUN_DIR"))
    out = output_path(args.output or storage / ("fused-inspect-" + uuid.uuid4().hex[:12]))
    if root == out or root in out.parents or out in root.parents:
        raise ValueError("inspection output must be separate from the existing batch")
    groups, _ = _probe_groups(root, out)
    out.mkdir(parents=True, exist_ok=False)
    result = {"status": "INSPECTED", "output": str(out), "source": str(root), "inspections": [],
              "model_compilation": "NOT_RUN", "device_execution": "NOT_RUN"}
    lines = ["Existing fused batch: read-only inspection", "No compilation or board execution.", ""]
    faults = re.compile(r"Get format configs failed|find op format config failed|infershape func failed|"
                        r"GE recipe failed|Not supported OP|Input .*must be|error:", re.I)
    for index, (probe_run, _) in enumerate(groups, 1):
        destination = out / f"inspection-{index}"
        report = inspect_fused_probes(probe_run, ddk, destination)
        result["inspections"].append({"probe_run": str(probe_run), "output": str(destination),
                                      "probe_manifest_sha256": report["probe_manifest_sha256"]})
        lines.append("Probe source: " + str(probe_run))
        for case in report["cases"]:
            lines.append(f"  {case['name']}: export={case['export']} compile={case['compile']}")
            if case["export"] != "FAILED" and case["compile"] != "FAILED":
                continue
            for log in case["logs"]:
                snippets = [entry for entry in log.get("excerpts", []) if faults.search(entry["text"])]
                if snippets:
                    lines.append("    " + log["source"])
                    lines.extend(f"    L{entry['line']}: {entry['text'][:700]}" for entry in snippets[:6])
                elif log["status"] != "READ":
                    lines.append(f"    {log['source']} [{log['status']}]")
        registration = report.get("installed_registration", {})
        lines.append(f"  DDK registration scan: files={registration.get('files_read', 0)}, "
                     f"limited={registration.get('limited', False)}")
        for item in registration.get("roots", []):
            lines.append(f"    {item['path']} exists={item['exists']}")
        sources = [item for item in report.get("installed_sources", [])
                   if re.search(r"conv|rms", Path(item["source"]).name, re.I)]
        for item in [*registration.get("records", []), *sources]:
            lines.append(f"  {item['source']} [{item['status']}]")
            lines.extend(f"    L{entry['line']}: {entry['text'][:700]}" for entry in item.get("excerpts", [])[:12])
        if not registration.get("records"):
            lines.append("  No matching registration text in scanned roots; implementation remains unresolved.")
        lines.append("  Full copied evidence: " + str(destination / "report.txt"))
    lines.extend(["", "Inspection does not change previous failures or establish numerical accuracy/placement.",
                  "Missing text is not proof of no kernel; scan limits are recorded in each report.json."])
    result["comparison"] = str(out / "report.txt")
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    save_json(out / "result.json", result)
    return result


def _fresh_probes(out, result, args):
    """Rebuild deleted run artifacts once, preserving every stage's evidence."""
    ddk = Path(_text(os.environ.get("CPRO_DDK_ROOT"), "CPRO_DDK_ROOT")).expanduser().resolve()
    ddk_id = _text(os.environ.get("CPRO_DDK_ID"), "CPRO_DDK_ID")
    ge = _text(os.environ.get("CANN_ROOT"), "CANN_ROOT")
    bindings, diagnosis, generated = out / "bindings.json", out / "diagnosis", out / "probes"
    selected = [name for name, _, _ in (_MODEL_CASES if getattr(args, "model_ops", False) else _CASES)]
    result.update(fresh_stage="bindings", bindings=str(bindings), tbe_diagnosis=str(diagnosis),
                  generated_probes=str(generated), reused_cases=[],
                  compiled_cases_requested=selected)
    print("fused-all: fresh bindings from selected DDK FP16 GDN state source", flush=True)
    write_document_bindings(bindings, target="kirinx90", ddk_id=ddk_id,
                            gdn_initial_state_dtype="float16",
                            gdn_state_source=ddk / "ddk/ascendc/ops/impl/custom/chunk_gated_delta_rule.cpp")
    result["fresh_stage"] = "tbe-diagnosis"
    print("fused-all: fresh TBE diagnosis (12 cases, 4 workers)", flush=True)
    report = check_tbe_platform(ddk, diagnosis, python=sys.executable, diagnose=True,
                                jobs=4, timeout_seconds=30)
    requested = next((case for case in report.get("cases", [])
                      if case.get("name") == "current_omg_first_requested"), {})
    native = requested.get("native") or {}
    # A fresh environment may already accept the requested platform. In that
    # case retain the diagnosis but let real OMG compile without adaptation.
    direct = (requested.get("status") == "RAW_UPDATE_SUCCEEDED" and requested.get("returncode") == 0
              and native.get("stage") == "COMPLETE" and native.get("native_success") is True
              and native.get("initialized_soc_preserved") is True
              and native.get("native_arguments") == ["kirinx90", "AiCore", "", "false"])
    result["tbe_compatibility_policy"] = "DIRECT_REQUEST_SUCCEEDED" if direct else "VALIDATE_DIAGNOSED_ADAPTER"
    result["fresh_stage"] = "compile"
    print(f"fused-probes: compiling {len(selected)} selected fresh cases once", flush=True)
    manifest = build_fused_probes(generated, bindings, ge_root=ge, ddk_root=ddk, ddk_id=ddk_id,
                                  cases=selected, conv_history_width=4, timeout_seconds=args.timeout_seconds,
                                  tbe_diagnosis=None if direct else diagnosis)
    result["fresh_stage"] = "probes-built"
    return [(generated, manifest)], set()


def run_all(args):
    """Keep old and new manifests separate; reuse only their recorded binaries."""
    storage = Path(_text(os.environ.get("AI_RUN_DIR"), "AI_RUN_DIR"))
    model_scope = getattr(args, "model_ops", False)
    selected = _MODEL_CASES if model_scope else _CASES
    out = output_path(args.output or storage / (("fused-model-" if model_scope else "fused-all-") + uuid.uuid4().hex[:12]))
    out.mkdir(parents=True, exist_ok=False)
    result = {"status": "FAILED", "output": str(out), "executions": [], "probe_runs": [],
              "model_compilation": "FRESH_ALL_CASES" if getattr(args, "fresh", False) else "MISSING_CASES_ONLY", "numerical_acceptance": "NOT_DEFINED",
              "runtime_placement": "UNVERIFIED", "cases": []}
    result.update(requested_scope="MODEL_CUSTOM_OPS" if model_scope else "ALL_DOCUMENTED_OPS",
                  selected_cases=[name for name, _, _ in selected],
                  excluded_cases=[name for name, _, _ in _CASES if name not in {item[0] for item in selected}])
    rows = {name: {"name": name, "role": role, "construction": "NOT_RUN", "export": "NOT_RUN",
                   "compile": "NOT_RUN", "load": "NOT_RUN", "process": "NOT_RUN", "numerical": "NOT_RUN"}
            for name, role, _ in selected}
    try:
        requested = getattr(args, "retry_cases", None)
        if getattr(args, "fresh", False):
            groups, reusable = _fresh_probes(out, result, args)
        else:
            root = Path(_text(args.probe_run, "FUSED_PROBE_RUN / --probe-run")).expanduser().resolve()
            if root == out or root in out.parents:
                raise ValueError("batch output must be separate from the existing probe directory")
            groups, reusable = _probe_groups(root, out)
            reusable.intersection_update(rows)
            # A recreated batch owns its diagnosis. Keep that explicit path
            # across retries so a new shell need not export a transient job ID.
            # An explicitly configured TBE_DIAG_JOB still takes precedence.
            diagnosis_job = os.environ.get("TBE_DIAG_JOB")
            if diagnosis_job:
                result["tbe_diagnosis"] = str(Path(diagnosis_job).expanduser().resolve() / "diagnosis")
            elif not (root / "manifest.json").is_file():
                previous = _stateful_json(root / "result.json")
                if previous.get("tbe_diagnosis"):
                    diagnosis = Path(_text(previous["tbe_diagnosis"], "recorded TBE diagnosis")).expanduser()
                    if not diagnosis.is_absolute():
                        raise ValueError("recorded TBE diagnosis must be an absolute path")
                    result["tbe_diagnosis"] = str(diagnosis)
                    result["tbe_compatibility_policy"] = previous.get("tbe_compatibility_policy", "VALIDATE_DIAGNOSED_ADAPTER")
            source, original = groups[0]
            if os.environ.get("CPRO_DDK_ID") and os.environ["CPRO_DDK_ID"] != original["ddk_id"]:
                raise ValueError("CPRO_DDK_ID differs from the compiled probe manifest")
            missing = [name for name in rows if name not in reusable]
            if requested is not None:
                selected = requested.split(",")
                if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(rows):
                    raise ValueError("--retry-cases must be unique known comma-separated case names")
                missing = [name for name in missing if name in selected]
            result.update(reused_cases=sorted(reusable), compiled_cases_requested=missing)
            if missing:
                print("fused-all: compiling missing cases: " + ", ".join(missing), flush=True)
                generated = out / "probes"
                try:
                    inherited_diagnosis = result.get("tbe_diagnosis")
                    if (not diagnosis_job and inherited_diagnosis
                            and result.get("tbe_compatibility_policy") != "DIRECT_REQUEST_SUCCEEDED"):
                        manifest = _compile_missing(source, generated, missing, tbe_diagnosis=inherited_diagnosis)
                    else:
                        manifest = _compile_missing(source, generated, missing)
                    groups.append((generated, manifest))
                except Exception as error:
                    result["compile_error"] = f"{type(error).__name__}: {error}"
        # Fresh and ordinary runs need the same evidence as explicit retries.
        # Ignore a historical failure once a recorded/new group compiled it.
        successful = {case["name"] for _, manifest in groups for case in manifest["cases"]
                      if case.get("status") == "COMPILED"}
        failed = next((path for path, manifest in reversed(groups)
                       if any(case["name"] in rows and case["role"] in {"conv_state", "rms"} and case.get("compile") == "FAILED"
                              and case["name"] not in successful for case in manifest["cases"])), None)
        if failed is not None:
            try:
                inspection = inspect_fused_probes(failed, _text(os.environ.get("CPRO_DDK_ROOT"), "CPRO_DDK_ROOT"), out / "inspection")
                result["inspection"] = str(out / "inspection")
                result["installed_registration"] = inspection.get("installed_registration", {})
            except Exception as error:
                result["inspection_error"] = f"{type(error).__name__}: {error}"
        runner_cache = {}
        result["probe_manifests"] = {}
        for index, (probe_run, manifest) in enumerate(groups):
            result["probe_runs"].append(str(probe_run))
            result["probe_manifests"][str(probe_run)] = sha256_file(probe_run / "manifest.json")
            for case in manifest["cases"]:
                if case["name"] not in rows:
                    continue
                if case["name"] in reusable and case.get("status") != "COMPILED":
                    continue
                rows[case["name"]].pop("error", None)
                rows[case["name"]].update(load="NOT_RUN", process="NOT_RUN", numerical="NOT_RUN")
                rows[case["name"]].update({key: case[key] for key in
                    ("construction", "export", "compile", "error") if key in case}, probe_run=str(probe_run))
            if not any(case["name"] in rows and case.get("status") == "COMPILED" for case in manifest["cases"]):
                continue
            board_args = argparse.Namespace(**vars(args))
            board_args.probe_run, board_args.output = str(probe_run), str(out / f"board-{index + 1}")
            if model_scope:
                board_args.selected_cases = [case["name"] for case in manifest["cases"] if case["name"] in rows]
            try:
                execution = execute(board_args, runner_cache=runner_cache)
                result["executions"].append(execution)
                comparison = Path(board_args.output) / "comparison/report.json"
                if comparison.is_file():
                    for case in _stateful_json(comparison)["cases"]:
                        if case["name"] not in rows:
                            continue
                        if rows[case["name"]].get("probe_run") != str(probe_run):
                            continue
                        # A failed legacy case can be retried by the new group.
                        rows[case["name"]].update({key: case[key] for key in
                            ("load", "process", "numerical", "diagnostics", "error") if key in case})
            except Exception as error:
                result["executions"].append({"status": "FAILED", "output": board_args.output,
                                             "error": f"{type(error).__name__}: {error}"})
        if (not result.get("compile_error") and result["executions"]
                and all(item["status"] == "EXECUTION_COMPLETE" for item in result["executions"])
                and all(row["compile"] == "PASSED" and row["process"] == "PASSED" for row in rows.values())):
            result["status"] = "EXECUTION_COMPLETE"
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    result["cases"] = list(rows.values())
    lines = ["Three selected custom operator invocation coverage" if model_scope else "Five custom operator invocation coverage", "",
             "case | construction | export | compile | load | process | numerical diagnostic"]
    for row in rows.values():
        lines.append(" | ".join(str(row[key]) for key in
            ("name", "construction", "export", "compile", "load", "process", "numerical")))
        if row.get("error"):
            lines.append("  " + _error_detail(row["error"]))
        for diagnostic in row.get("diagnostics", []):
            summary = {key: diagnostic[key] for key in (
                "name", "reference", "raw_nonfinite", "actual_nonzero", "max_abs_error",
                "relative_l2", "all_zero_with_nonzero_reference") if key in diagnostic}
            lines.append("  " + json.dumps(summary))
    for item in result["executions"]:
        lines.append(f"Board result: {item['output']} ({item['status']})")
        if item.get("error"):
            lines.append("  " + _error_detail(item["error"]))
    for probe_run in result["probe_runs"]:
        lines.append("Compilation details: " + str(Path(probe_run) / "report.txt"))
    if "fresh_stage" in result:
        lines.extend(["Fresh stage: " + result["fresh_stage"], "Bindings: " + result["bindings"],
                      "TBE diagnosis: " + str(Path(result["tbe_diagnosis"]) / "report.txt"),
                      "Fresh probes: " + result["generated_probes"]])
    elif result.get("tbe_diagnosis"):
        lines.append("TBE diagnosis: " + str(Path(result["tbe_diagnosis"]) / "report.txt"))
    if result.get("inspection"):
        lines.append("Read-only DDK inspection: " + str(Path(result["inspection"]) / "report.txt"))
        registration = result.get("installed_registration", {})
        for record in registration.get("records", [])[:12]:
            lines.append("  " + str(record.get("path", "registration metadata")))
            for snippet in record.get("excerpts", [])[:6]:
                lines.append(f"    L{snippet.get('line', '?')}: {str(snippet.get('text', ''))[:500]}")
        if not registration.get("records"):
            lines.append("  No matching registration text in the scanned roots; installed implementation remains unresolved.")
    if result.get("inspection_error"):
        lines.append("Read-only inspection: " + _error_detail(result["inspection_error"]))
    for key in ("error", "compile_error"):
        if key in result:
            lines.append(result[key])
    if model_scope:
        lines += ["", "Conv/RMS custom probes excluded by model policy; their historical results are unchanged.",
                  "Use --fused-ops gdn for the GE target's existing base Conv/RMS path.",
                  "Cache/Attention standalone probes do not establish integration into the full GE target."]
    else:
        lines += ["", "Conv width 4 is an explicit invocation candidate; its state ordering is not validated."]
    lines.append("Numerical acceptance is NOT_DEFINED; execution placement remains UNVERIFIED.")
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    result["comparison"] = str(out / "report.txt")
    save_json(out / "result.json", result)
    return result


def main(argv=None):
    try:
        args = parser().parse_args(argv)
        if args.all_ops and args.model_ops:
            raise ValueError("choose --model-ops or --all-ops")
        if args.inspect_only and (args.fresh or args.all_ops or args.model_ops or args.retry_cases is not None):
            raise ValueError("--inspect-only cannot be combined with execution modes")
        if args.fresh and (not (args.all_ops or args.model_ops) or args.retry_cases is not None):
            raise ValueError("--fresh requires --model-ops or --all-ops and cannot be combined with --retry-cases")
        if args.retry_cases is not None and not (args.all_ops or args.model_ops):
            raise ValueError("--retry-cases requires --model-ops or --all-ops")
        result = inspect_existing(args) if args.inspect_only else run_all(args) if args.all_ops or args.model_ops else execute(args)
    except Exception as error:
        print(f"fused-board: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps({key: result[key] for key in ("status", "output", "comparison", "error", "execution_error", "comparison_error") if key in result}, ensure_ascii=False))
    if result.get("comparison"):
        print(Path(result["comparison"]).read_text(), end="")
    return 0 if result["status"] in {"EXECUTION_COMPLETE", "INSPECTED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
