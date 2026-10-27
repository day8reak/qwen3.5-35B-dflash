#!/usr/bin/env python3
"""Build a runner and execute existing compiled fused-op bundles through HDC."""
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
from qwen35_moe_dflash.compile import _stateful_json
from qwen35_moe_dflash.elf_dynamic import dynamic_info
from qwen35_moe_dflash.ge_fused.bindings import OperatorBindings
from qwen35_moe_dflash.ge_fused.export import _step_file
from qwen35_moe_dflash.ge_fused.probes import _validate_probe_manifest, _wire, compare_fused_probes
from qwen35_moe_dflash.hashing import sha256_file
from qwen35_moe_dflash.hdc_run import Hdc, _file_record, _positive, _relative, _text, _upload
from qwen35_moe_dflash.hiai_deployment import (
    BOARD_LIBRARY_SEARCH_PATHS, DDK_HCL_EXCLUDED, candidate_blockers, discover_libraries,
)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--probe-run", default=os.environ.get("FUSED_PROBE_RUN"), help="existing compiled probe directory; defaults to FUSED_PROBE_RUN")
    result.add_argument("--output", help="new directory below AI_RUN_DIR; otherwise allocated automatically")
    result.add_argument("--timeout-seconds", type=_positive, default=3600)
    return result


def _prepare(root):
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
    compiled = []
    for case in manifest["cases"]:
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
    if not compiled:
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


def execute(args):
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
        manifest, cases, uploads = _prepare(root)
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
                      skipped_cases=[case["name"] for case in manifest["cases"] if case["status"] != "COMPILED"],
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
        print("fused-board: building current runner; reusing compiled OMC files", flush=True)
        runner = _build_runner(out, args.timeout_seconds)
        uploads.append(_file_record(runner, "runner"))
        if private:
            result["libraries"] = {"mode": "existing-board-directory", "path": private, "compatibility": "UNVERIFIED"}
        else:
            discovery = discover_libraries(runner, out / "native")
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
        _upload(hdc, board, uploads, out / "payloads.sha256")
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
                comparison = compare_fused_probes(root, out / "results", out / "comparison")
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
            for path in [out / "runner-build.log", out / "native/build.log", out / "runtime.stderr",
                         out / "batch.stderr", out / "batch.stdout", *sorted((out / "results").glob("*.stderr")),
                         *sorted((out / "results").glob("*.stdout"))]:
                _tail(path)
    return result


def main(argv=None):
    try:
        result = execute(parser().parse_args(argv))
    except Exception as error:
        print(f"fused-board: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps({key: result[key] for key in ("status", "output", "comparison", "error", "execution_error", "comparison_error") if key in result}, ensure_ascii=False))
    if result.get("comparison"):
        print(Path(result["comparison"]).read_text(), end="")
    return 0 if result["status"] == "EXECUTION_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
