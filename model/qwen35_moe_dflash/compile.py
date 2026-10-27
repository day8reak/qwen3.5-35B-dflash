"""Hash-checked AIR -> OMG --framework=1 -> OMC. Never invokes ATC or ONNX."""
import json
import math
import os
from pathlib import Path
import re
import subprocess
import shutil
import sys
import tempfile
from contextlib import contextmanager
from .artifacts import checked_file, files_under, output_path, save_json
from .hashing import sha256_file
from .tool_logs import log_excerpt
from .compiler_evidence import compile_evidence


def _prepend_paths(paths, inherited):
    # Keep vendor environment extensions while preferring this conversion's DDK.
    entries = [str(path) for path in paths] + inherited.split(os.pathsep)
    return os.pathsep.join(dict.fromkeys(entry for entry in entries if entry))


def _foreign_cann_paths(env, ddk_root):
    """Remove only explicitly identified server CANN paths from the OMG child.

    AIR's server GE libraries and Python packages must not supply an older/newer
    half of the DDK's TE runtime. Unknown vendor extensions remain available.
    Canonical paths also catch CANN's ``latest`` symlink aliases.
    """
    roots = [Path(env[key]).resolve() for key in
             ("CANN_ROOT", "ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME") if env.get(key)]

    def foreign(entry):
        path = Path(entry).resolve()
        return (not path.is_relative_to(ddk_root)
                and any(path.is_relative_to(root) for root in roots))

    for key in ("PYTHONPATH", "LD_LIBRARY_PATH", "PATH"):
        if key in env:
            env[key] = os.pathsep.join(entry for entry in env[key].split(os.pathsep)
                                      if entry and not foreign(entry))
    # These selectors can bypass path ordering. Preserve selected-DDK values,
    # including internal layouts; remove only declared foreign CANN values.
    for key in ("CANN_ROOT", "ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_OPP_PATH", "ASCEND_AICPU_PATH"):
        if env.get(key) and foreign(env[key]):
            env.pop(key)


def omg_environment(ddk_root, inherited=None):
    """Use the actual binary, avoiding the vendor wrapper's shared-tree writes."""
    ddk_root = Path(ddk_root).resolve()
    master = ddk_root / "tools/tools_omg/master"
    plugin = ddk_root / "tools/platform/kirinx90"
    if not (master / "omg").is_file():
        raise FileNotFoundError("DDK needs tools/tools_omg/master/omg")
    env = dict(os.environ if inherited is None else inherited)
    _foreign_cann_paths(env, ddk_root)
    libs = [master / "lib64"]
    # Public packages use a separate platform plugin. Internal DDKs may bundle
    # platform support elsewhere and supply its paths through their env script.
    # The real OMG invocation determines whether kirinx90 is supported.
    for candidate in [plugin / "lib64", ddk_root / "ddk/tbe/lib64", ddk_root / "ddk/ascendcpp/lib64"]:
        if candidate.is_dir(): libs.append(candidate)
    env["LD_LIBRARY_PATH"] = _prepend_paths(libs, env.get("LD_LIBRARY_PATH", ""))
    env["CCE_LIB_DIR"] = str(master / "lib64")
    env["SOC_VERSION"] = "kirinx90"
    env["DDK_PATH"] = str(ddk_root)
    env["TOOLCHAIN_HOME"] = str(ddk_root)
    env["HIAI_VERSION"] = str(master)
    python_paths = [path for path in (ddk_root / "tools/tools_ascendc/package/python",
                                     ddk_root / "ddk/tbe/ops", plugin / "ops/impl") if path.is_dir()]
    if python_paths:
        env["PYTHONPATH"] = _prepend_paths(python_paths, env.get("PYTHONPATH", ""))
    # Do not populate shared DDK trees with interpreter caches.
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    config = ddk_root / "ddk/ascendcpp/conf"
    if config.is_dir(): env["KIRIN_CONF_JSON_PATH"] = str(config)
    for candidate in [ddk_root / "ddk/ccec_compiler/bin", ddk_root / "tools/tools_ascendc/bisheng/bin"]:
        if candidate.is_dir(): env["PATH"] = str(candidate) + os.pathsep + env.get("PATH", "")
    return master / "omg", env


def _environment_record(ddk_root, env):
    """Report candidate package locations without importing vendor Python."""
    candidates = []
    for entry in env.get("PYTHONPATH", "").split(os.pathsep):
        if entry:
            path = Path(entry) / "te_fusion/log_util.py"
            if path.is_file():
                candidates.append(str(path.resolve()))
    keys = ("PYTHONPATH", "LD_LIBRARY_PATH", "PATH", "DDK_PATH", "TOOLCHAIN_HOME", "HIAI_VERSION",
            "CCE_LIB_DIR", "SOC_VERSION", "KIRIN_CONF_JSON_PATH", "ASCEND_OPP_PATH",
            "TMPDIR", "TMP", "TEMP")
    return {"ddk_root": str(Path(ddk_root).resolve()),
            "environment": {key: env[key] for key in keys if key in env},
            "te_fusion_log_util_candidates": candidates,
            "candidate_note": "PYTHONPATH order only; embedded Python and site-packages may differ."}


_PLATFORM_CHECK = r'''
import hashlib, importlib, inspect, json, pathlib, sys, traceback
output = pathlib.Path(sys.argv[1])
result = {"initialization": "IMPORT_ONLY", "worker_reproduction": "NOT_ESTABLISHED",
          "python": sys.executable, "python_version": sys.version,
          "native_arguments": ["kirinx90", "AiCore", "", "false"], "stage": "IMPORT"}
def save():
    output.write_text(json.dumps(result, indent=2) + "\n")
def specs(platform):
    values = {}
    for key in ("SOC_VERSION", "SHORT_SOC_VERSION", "CORE_TYPE", "CORE_NUM"):
        try:
            values[key] = str(platform.get_soc_spec(key))
        except Exception as error:
            values[key] = {"error": repr(error)}
    return values
def libraries():
    try:
        return sorted({line.split()[-1] for line in pathlib.Path("/proc/self/maps").read_text().splitlines()
                       if any(name in line for name in ("libtvm", "libplatform", "libpython", "libcce"))})
    except OSError as error:
        return {"error": str(error)}
save()
try:
    import tbe.common.platform as platform
    from tbe.common.platform import platform_info
    result["modules"] = {}
    for name in ("tbe", "tbe.tvm", "tbe.common.platform", "tbe.common.platform.platform_info"):
        path = getattr(sys.modules.get(name), "__file__", None)
        item = {"path": path}
        if path and pathlib.Path(path).is_file():
            item["sha256"] = hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
        result["modules"][name] = item
    result["before"] = specs(platform)
    result["libraries_before"] = libraries()
    if len(sys.argv) > 2 and sys.argv[2] == "initialize":
        result.update(initialization="INSTALLED_API_REQUESTED", stage="INIT_API_UNRESOLVED")
        save()
        try:
            module = importlib.import_module("te.platform.cce_conf")
            initialize = getattr(module, "te_set_version", None)
        except ModuleNotFoundError as error:
            if error.name not in {"te", "te.platform", "te.platform.cce_conf"}:
                raise
            initialize = None
        if initialize is None:
            initialize = getattr(platform, "te_set_version", None)
        if not callable(initialize):
            raise RuntimeError("installed te_set_version API is unavailable; no initializer guessed")
        signature = inspect.signature(initialize)
        path = inspect.getsourcefile(initialize)
        result["init_api"] = {"module": initialize.__module__, "name": initialize.__name__,
                              "signature": str(signature), "path": path, "arguments": ["kirinx90"]}
        if path and pathlib.Path(path).is_file():
            result["init_api"]["sha256"] = hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
        try:
            result["init_api"]["source"] = inspect.getsource(initialize)
        except (OSError, TypeError) as error:
            result["init_api"]["source_error"] = str(error)
        signature.bind("kirinx90")
        result["stage"] = "INITIALIZE"
        save()
        try:
            value = initialize("kirinx90")
            result.update(init_return_type=type(value).__name__, init_return_repr=repr(value))
        finally:
            result["after_init"] = specs(platform)
            result["libraries_after_init"] = libraries()
            save()
    result["stage"] = "LOOKUP"
    save()
    native = platform_info.get_global_func("cce.te_update_version")
    result["stage"] = "NATIVE_UPDATE"
    save()
    value = native(*result["native_arguments"])
    result.update(native_return_type=type(value).__name__, native_return_repr=repr(value))
    result.update(native_success=bool(value == "success"), stage="COMPLETE")
    result["after"] = specs(platform)
except Exception as error:
    result.update(error=repr(error), traceback=traceback.format_exc(), native_success=False)
finally:
    result["mapped_libraries"] = libraries()
    save()
'''


def check_tbe_platform(ddk_root, output_dir, *, python, timeout_seconds=30,
                       compare_libraries=False, initialize=False):
    """Expose TBE's native return in a fresh process; no compiler/kernel call.

    Import-only initialization is intentionally not declared equivalent to the
    embedded OMG worker. Neither a pass nor failure establishes target support.
    """
    interpreter = Path(python).absolute()
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise ValueError("--python must name the existing executable used by the failed worker")
    if not 0 < timeout_seconds <= 60:
        raise ValueError("platform check timeout must be between 1 and 60 seconds")
    if type(compare_libraries) is not bool or type(initialize) is not bool:
        raise ValueError("platform comparison/initialization selectors must be boolean")
    _, env = omg_environment(ddk_root)
    root = output_path(output_dir)
    if root.is_relative_to(Path(ddk_root).resolve()):
        raise ValueError("platform check output must be outside DDK")
    if not compare_libraries:
        return _check_tbe_platform_arm(ddk_root, root, interpreter, env, initialize, timeout_seconds)
    root.mkdir(parents=True, exist_ok=False)
    tbe = Path(ddk_root).resolve() / "ddk/tbe/lib64"
    result = {"schema_version": 1, "artifact_kind": "qwen35-tbe-platform-comparison",
              "status": "COMPARISON_COMPLETE", "compiler_execution": "NOT_RUN", "device_execution": "NOT_RUN",
              "scope": "Up to four fresh Python children: selected-DDK library order and same-target initialization. "
                       "No production OMG environment change, SoC substitution, or worker reproduction claim.", "cases": []}
    for order in ("omg_first", "tbe_first"):
        for init in (False, True):
            name = order + ("_initialized" if init else "_import_only")
            selected_env = dict(env)
            if order == "tbe_first":
                if not tbe.is_dir():
                    result["cases"].append({"name": name, "status": "UNAVAILABLE", "reason": "selected DDK TBE lib64 missing"})
                    continue
                selected_env["LD_LIBRARY_PATH"] = _prepend_paths([tbe], env.get("LD_LIBRARY_PATH", ""))
            case = _check_tbe_platform_arm(ddk_root, root / name, interpreter, selected_env, init, timeout_seconds)
            result["cases"].append({"name": name, "status": case["status"], "native": case.get("native"),
                                    "returncode": case["returncode"], "report": name + "/report.json"})
    result["successful_arms"] = [case["name"] for case in result["cases"] if case["status"] == "RAW_UPDATE_SUCCEEDED"]
    result["files"] = files_under(root, root)
    save_json(root / "report.json", result)
    lines = ["TBE library order / same-target initialization comparison", result["scope"], "",
             "case | status | initial SOC | after init SOC | native return"]
    shown_sources, shown_maps = set(), set()
    for case in result["cases"]:
        native = case.get("native") or {}
        lines.append(" | ".join(str(value) for value in (case["name"], case["status"],
                     native.get("before", {}).get("SOC_VERSION"), native.get("after_init", {}).get("SOC_VERSION"),
                     native.get("native_return_repr", "NOT_CALLED"))))
        if native.get("error") or case.get("reason"):
            lines.append("  " + str(native.get("error") or case["reason"]))
        api = native.get("init_api", {})
        if api and api.get("sha256", api.get("path")) not in shown_sources:
            shown_sources.add(api.get("sha256", api.get("path")))
            lines.append("  Init API: " + json.dumps(api))
        maps = json.dumps(native.get("mapped_libraries", []))
        if maps not in shown_maps:
            shown_maps.add(maps)
            lines.append("  Loaded libraries: " + maps)
        lines.append("  Full evidence: " + case.get("report", case["name"]))
    (root / "report.txt").write_text("\n".join(lines) + "\n")
    return result


def _check_tbe_platform_arm(ddk_root, root, interpreter, env, initialize, timeout_seconds):
    root.mkdir(parents=True, exist_ok=False)
    script = root / "platform-check.py"
    script.write_text(_PLATFORM_CHECK)
    child_result = root / "native-result.json"
    argv = [str(interpreter), str(script), str(child_result)] + (["initialize"] if initialize else [])
    save_json(root / "command.json", {"argv": argv, "cwd": str(root), "timeout_seconds": timeout_seconds})
    result = {"schema_version": 1, "artifact_kind": "qwen35-tbe-platform-check", "status": "FAILED",
              "compiler_execution": "NOT_RUN", "device_execution": "NOT_RUN", "returncode": None,
              "scope": "Fresh Python process; " + ("installed same-target initializer" if initialize else "import-only initialization") + " and one native platform update. "
                       "Not an OMG worker reproduction or a compiler/target support gate."}
    with _omg_ipc_environment(root, env) as (child_env, _), (root / "platform.log").open("x") as log:
        save_json(root / "omg-environment.json", _environment_record(ddk_root, child_env))
        try:
            proc = subprocess.run(argv, cwd=root, env=child_env, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=timeout_seconds, check=False)
            result["returncode"] = proc.returncode
        except (subprocess.TimeoutExpired, OSError) as error:
            result["error"] = str(error)
    if child_result.is_file():
        try:
            result["native"] = json.loads(child_result.read_text())
        except (OSError, ValueError) as error:
            result["error"] = str(error)
    if (result["returncode"] == 0 and result.get("native", {}).get("stage") == "COMPLETE"
            and result["native"].get("native_success") is True):
        result["status"] = "RAW_UPDATE_SUCCEEDED"
    elif result.get("native", {}).get("stage") == "INIT_API_UNRESOLVED":
        result["status"] = "INIT_API_UNRESOLVED"
    result["files"] = files_under(root, root)
    save_json(root / "report.json", result)
    lines = ["TBE raw platform update (no model compilation or device execution)",
             result["scope"], "status=" + result["status"],
             json.dumps({key: value for key, value in result.items() if key in {"returncode", "error"}}),
             json.dumps(result.get("native", {}), indent=2), "Full child output: platform.log"]
    # Include the child's output even on crashes/import errors which preceded JSON completion.
    lines.append((root / "platform.log").read_text(errors="replace")[-12000:])
    (root / "report.txt").write_text("\n".join(lines) + "\n")
    return result


@contextmanager
def _omg_ipc_environment(scratch, env):
    """Keep multiprocessing socket names short, with all data inside this run.

    CPython adds /pymp-xxxxxxxx/listener-xxxxxxxx to TMPDIR. A deeply nested
    model run exceeds Linux's AF_UNIX pathname limit. A procfs directory-FD
    alias keeps the spelling short without creating a shared /tmp directory.
    Use the *parent* PID: OMG and its workers need not inherit the directory FD.
    Hold that FD until the synchronous compiler invocation has returned.
    """
    with tempfile.TemporaryDirectory(prefix=".ipc-", dir=scratch) as temp:
        physical = Path(temp).resolve()
        fd = os.open(physical.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            # Include the unique directory name so an orphaned compiler worker
            # cannot access a later invocation if this descriptor is reused.
            alias = Path(f"/proc/{os.getpid()}/fd/{fd}") / physical.name
            try:
                usable = alias.samefile(physical)
            except OSError:
                usable = False
            # Leave room for the two CPython-generated path components.
            if usable and len(os.fsencode(alias)) <= 60:
                selected, mode = str(alias), "parent-proc-fd"
            elif len(os.fsencode(physical)) <= 60:
                selected, mode = str(physical), "short-physical-path"
            else:
                raise RuntimeError("OMG IPC temporary path is too long and the parent /proc/PID/fd "
                                   "alias is unavailable. Enable procfs access for the compiler process "
                                   "or use a shorter run/output directory; no shared /tmp fallback was used.")
            child = dict(env, TMPDIR=selected, TMP=selected, TEMP=selected)
            yield child, {"mode": mode, "path": selected, "physical_path": str(physical),
                          "lifetime": "Only while this OMG invocation is running; not a reusable path."}
        finally:
            os.close(fd)


def _retain_compiler_diagnostics(scratch, folder, *, failure=False, max_bytes=16 << 20):
    """Retain bounded compiler evidence on success or failure, never input links.

    Reports and graph dumps are host diagnostics, not board model payloads.
    A pre-check report describes candidate support, not final device placement;
    retained dumps still need graph-aware interpretation, excluding const folding.
    """
    if max_bytes < 0:
        raise ValueError("diagnostic byte budget must be nonnegative")
    candidates = []
    report = scratch / "check_result.json"
    if not report.is_symlink() and report.is_file():
        candidates.append(report)
    source = scratch / "dump_graph"
    if not source.is_symlink() and source.is_dir():
        candidates.extend(path for path in source.glob("*.irpb")
                          if not path.is_symlink() and path.is_file())
    scope = ("Host compiler diagnostics only. Pre-check support is not final placement; "
             "constant-folding graphs are not board execution evidence.")
    if not candidates:
        return {"status": "NOT_EMITTED", "scope": scope}

    def priority(path):
        name = path.name.lower()
        if failure and "fail" in name:
            rank = 0
        elif path == report:
            rank = 1
        elif "after_partition" in name:
            rank = 2
        elif "after_connectionpointoptimize" in name:
            rank = 3
        elif "subgraph" in name:
            rank = 4
        else:
            rank = 5
        return rank, path.name

    destination = folder / "diagnostics"
    if destination.is_symlink():
        raise OSError("compiler diagnostics destination must not be a symlink")
    destination.mkdir(parents=True, exist_ok=True)
    records, used = [], 0
    for path in sorted(candidates, key=priority):
        size = path.stat().st_size
        record = {"name": path.name, "source": str(path.relative_to(scratch)), "bytes": size,
                  "role": "pre-check-report" if path == report else "graph-dump"}
        if used + size > max_bytes:
            record["status"] = "SKIPPED_SIZE_BUDGET"
        else:
            target = destination / path.name
            if target.exists() or target.is_symlink():
                raise FileExistsError(target)
            shutil.copyfile(path, target)
            if target.stat().st_size != size:
                raise OSError(f"compiler diagnostic changed size during retention: {path.name}")
            used += size
            record.update(status="RETAINED", sha256=sha256_file(target))
        records.append(record)
    status = "PARTIAL" if any(row["status"] != "RETAINED" for row in records) else "RETAINED"
    save_json(destination / "manifest.json", {"status": status, "scope": scope,
        "compile_outcome": "failure" if failure else "success", "max_bytes": max_bytes,
        "retained_bytes": used, "files": records})
    return {"status": status, "manifest": "diagnostics/manifest.json", "scope": scope}


def _retain_failure_diagnostics(scratch, folder, *, max_bytes=16 << 20):
    """Compatibility entry point retaining the failure-first ordering."""
    return _retain_compiler_diagnostics(scratch, folder, failure=True, max_bytes=max_bytes)


def _compiled_payloads(folder, root):
    """Preserve existing payload collection while excluding host diagnostics."""
    records = []
    for path in sorted(folder.iterdir()):
        if path.name == "diagnostics":
            continue
        if path.is_symlink():
            raise ValueError(f"artifact payload must not be a symlink: {path}")
        if path.is_dir():
            records.extend(files_under(path, root))
        elif path.is_file():
            records.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                            "sha256": sha256_file(path)})
    return records


def _checked_lowering_report(root, manifest, graph, *, policy, summary, filename, label):
    """Bind new lowering summaries to the export's recorded construction file."""
    if policy not in manifest and summary not in graph:
        return None  # Older AIR bundles predate this optional emission policy.
    expected_path = f"{graph['name']}/{filename}"
    entries = [item for item in graph.get("construction_files", []) if item.get("path") == expected_path]
    if len(entries) != 1 or policy not in manifest:
        raise ValueError(f"{label} lowering requires a hash-locked report and explicit mode")
    path = checked_file(root, entries[0])
    report = json.loads(path.read_text())
    mode = manifest[policy]
    if (not isinstance(report, dict) or report.get("mode") != mode
            or report.get("graph") != graph["name"] or not isinstance(report.get("counts"), dict)
            or graph.get(summary) != {"mode": mode, "counts": report["counts"]}):
        raise ValueError(f"{label} lowering mode/counts differ from the hash-locked report")
    return entries[0]


def compile_air(bundle, output_dir, ddk_root, *, ddk_id, timeout_seconds=3600,
                reject_cpu_fallback=False):
    root = Path(bundle).resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("artifact_kind") != "qwen35-cpro-air-pair" or manifest.get("status") != "EXPORTED"
            or manifest.get("runtime") != "hiai"):
        raise ValueError("only a successful HIAI AIR bundle is accepted")
    if manifest.get("abi_version") != 2 or manifest.get("feature_dtype") != "torch.float16":
        raise ValueError("ABI v2 with FP16 features is required")
    if manifest.get("export_root") != str(root):
        raise ValueError("compile AIR at its export_root: FileConstant weights may contain absolute paths")
    if manifest.get("fp16_reductions", "native") not in {"native", "matmul"}:
        raise ValueError("AIR fp16_reductions must be native or matmul")
    if manifest.get("fp16_row_mul", "native") not in {"native", "compact"}:
        raise ValueError("AIR fp16_row_mul must be native or compact")
    if manifest.get("gdn_indexed_copy", "native") not in {"native", "gather"}:
        raise ValueError("AIR gdn_indexed_copy must be native or gather")
    graphs = manifest["graphs"]
    if [g["name"] for g in graphs] != ["target", "draft"]:
        raise ValueError("AIR bundle must contain target then draft")
    if not ddk_id or timeout_seconds <= 0:
        raise ValueError("DDK identity and positive timeout required")
    lowering_reports = {}
    for graph in graphs:
        paths = {r["path"]: checked_file(root, r) for r in graph["files"]}
        if graph["air"] not in paths or paths[graph["air"]].suffix != ".air":
            raise ValueError("AIR graph is missing from hash-locked payloads")
        lowering_reports[graph["name"]] = [
            _checked_lowering_report(root, manifest, graph, policy="fp16_reductions",
                                     summary="reduction_lowering", filename="reduction-lowering.json",
                                     label="reduction"),
            _checked_lowering_report(root, manifest, graph, policy="fp16_row_mul",
                                     summary="row_mul_lowering", filename="row-mul-lowering.json",
                                     label="row-Mul"),
            _checked_lowering_report(root, manifest, graph, policy="gdn_indexed_copy",
                                     summary="gdn_copy_lowering", filename="gdn-copy-lowering.json",
                                     label="GDN copy"),
        ]
    omg, env = omg_environment(ddk_root)
    if not os.access(omg, os.X_OK):
        raise PermissionError(f"OMG is not executable: {omg}; fix the run-local extraction permissions")
    output = output_path(output_dir); output.mkdir(parents=True, exist_ok=False)
    records = []
    result = {k: manifest[k] for k in ["sequence_length", "draft_width", "feature_width", "vocab_size",
                                      "precision", "state_policy", "expert_dispatch", "source", "pad_token_id", "eos_token_ids",
                                      "abi_version", "feature_dtype", "precision_audit", "quantization"]}
    if "expert_execution" in manifest:
        result["expert_execution"] = manifest["expert_execution"]
    if "fp16_reductions" in manifest:
        result["fp16_reductions"] = manifest["fp16_reductions"]
    if "fp16_row_mul" in manifest:
        result["fp16_row_mul"] = manifest["fp16_row_mul"]
    if "gdn_indexed_copy" in manifest:
        result["gdn_indexed_copy"] = manifest["gdn_indexed_copy"]
    result.update(schema_version=2, artifact_kind="qwen35-cpro-omc-pair", status="COMPILING",
                  platform="kirinx90", runtime="hiai", device_validation="PENDING", graphs=records,
                  ddk_id=ddk_id, omg_sha256=sha256_file(omg), air_manifest_sha256=sha256_file(manifest_path),
                  reject_cpu_fallback=reject_cpu_fallback)
    stage = "prepare"
    name = None
    log_path = None
    try:
        for graph in graphs:
            name = graph["name"]; folder = output / name
            stage = f"{name}-setup"
            log_path = folder / "omg.log"
            folder.mkdir()
            prefix = folder / name
            air = root / graph["air"]
            # Short relative names bypass OMG's restrictive output-path check.
            # The actual run directory may contain dots or long model/run names.
            command = [str(omg), "--framework=1", "--model=input.air", f"--output={name}",
                       "--target=omc", "--platform=kirinx90", "--use_origin_format=true",
                       "--weight_data_type=FP16"]
            # Quantized matmuls must already be native packed operators in AIR.
            save_json(folder / "command.json", {"argv": command, "ddk_id": ddk_id,
                                               "air_source": str(air), "cwd_policy": "isolated-run-scratch"})
            with tempfile.TemporaryDirectory(prefix=f"omg-{name}-", dir=output) as temp:
                scratch = Path(temp)
                for item in air.parent.iterdir():
                    (scratch / item.name).symlink_to(item, target_is_directory=item.is_dir())
                if not (scratch / "input.air").exists(): (scratch / "input.air").symlink_to(air)
                stage = f"{name}-omg"
                try:
                    with _omg_ipc_environment(scratch, env) as (local_env, ipc), log_path.open("x") as log:
                        environment_record = _environment_record(ddk_root, local_env)
                        environment_record["ipc_temporary_directory"] = ipc
                        save_json(folder / "omg-environment.json", environment_record)
                        subprocess.run(command, cwd=scratch, env=local_env, stdout=log, stderr=subprocess.STDOUT,
                                       check=True, timeout=timeout_seconds)
                    stage = f"{name}-output-check"
                    generated = scratch / f"{name}.omc"
                    if not generated.is_file() or not generated.stat().st_size:
                        raise RuntimeError(f"OMG succeeded without a nonempty {name}.omc")
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError, RuntimeError) as error:
                    diagnostic_note = ""
                    try:
                        _retain_failure_diagnostics(scratch, folder)
                    except OSError as diagnostic_error:
                        # Retaining diagnostics must never hide the OMG failure.
                        diagnostic_note = f"\nCould not retain graph diagnostics: {diagnostic_error}"
                    # Read after closing/flushing the log, while the retained
                    # path remains outside the automatically cleaned scratch.
                    raise RuntimeError(f"OMG graph={name} stage={stage}: {error}\n"
                                       f"Full log: {log_path}\n{log_excerpt(log_path)}{diagnostic_note}") from error
                try:
                    compiler_diagnostics = _retain_compiler_diagnostics(scratch, folder)
                except OSError as diagnostic_error:
                    # A diagnostics storage error does not invalidate an OMC
                    # already produced successfully; never claim full evidence.
                    compiler_diagnostics = {"status": "FAILED", "error": str(diagnostic_error)}
                stage = f"{name}-collect"
                # Preserve compiled outputs/sidecars, but never archive input
                # symlinks or compiler scratch as model payloads.
                for item in scratch.iterdir():
                    if item.name.startswith(name) and not item.is_symlink():
                        shutil.move(str(item), str(folder / item.name))
            omc = folder / f"{name}.omc"
            # Detect accidental compiler mutation of AIR or external weights.
            for payload in graph["files"]: checked_file(root, payload)
            for report in lowering_reports[name]:
                if report is not None:
                    checked_file(root, report)
            evidence = compile_evidence(log_path.read_text(encoding="utf-8", errors="replace"))
            evidence["log_sha256"] = sha256_file(log_path)
            records.append({"name": name, "omc": str(omc.relative_to(output)),
                            "sha256": sha256_file(omc), "bytes": omc.stat().st_size,
                            "inputs": graph["inputs"], "outputs": graph["outputs"],
                            "input_shapes": graph["input_shapes"], "input_dtypes": graph["input_dtypes"],
                            "output_shapes": graph["output_shapes"], "output_dtypes": graph["output_dtypes"],
                            "compiler_diagnostics": compiler_diagnostics,
                            "compiler_evidence": evidence,
                            "payloads": _compiled_payloads(folder, output)})
            if "reduction_lowering" in graph:
                records[-1]["reduction_lowering"] = graph["reduction_lowering"]
            if "row_mul_lowering" in graph:
                records[-1]["row_mul_lowering"] = graph["row_mul_lowering"]
            if "gdn_copy_lowering" in graph:
                records[-1]["gdn_copy_lowering"] = graph["gdn_copy_lowering"]
            # OMG may return success after abandoning all NPU partitions.
            # Constant-folding CPU graphs are expected and are not deployment
            # evidence. Unknown scope remains unknown, but an explicit rollback
            # must not silently pass a requested NPU bring-up check.
            rollbacks = [event for event in evidence.get("rollback_events", [])
                         if event["scope"] in {"main", "unknown"} and event["device"] == "NPUCL"]
            if rollbacks:
                event = rollbacks[-1]
                message = (f"graph={name}: OMG reported NPU partition limit {event['limit']} exceeded "
                           f"and rollback to CPU (scope={event['scope']}, log line {event['line']}). "
                           f"OMC and diagnostics retained: {folder}. "
                           "This is compiler fallback evidence, not successful NPU execution.")
                print(f"compile-air: {message}", file=sys.stderr)
                if reject_cpu_fallback:
                    stage = f"{name}-placement-check"
                    result["cpu_fallback_rejections"] = rollbacks
                    raise RuntimeError(message)
        result["status"] = "COMPILED"
        save_json(output / "manifest.json", result)
    except Exception as error:
        result["status"] = "FAILED"; result["error"] = f"{type(error).__name__}: {error}"
        cause = error.__cause__ or error
        result.update(failed_graph=name, stage=stage, log_path=str(log_path) if log_path else None,
                      returncode=cause.returncode if isinstance(cause, subprocess.CalledProcessError) else None)
        if isinstance(cause, subprocess.TimeoutExpired):
            result["timeout_seconds"] = cause.timeout
        save_json(output / "failure.json", result)
        raise
    return result


def _stateful_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate artifact JSON key: {key}")
            result[key] = value
        return result
    return json.loads(path.read_text(), object_pairs_hook=unique)


def _same_ordered_json(left, right):
    # Named custom ports retain registration order, including output ordinals.
    return json.dumps(left, separators=(",", ":")) == json.dumps(right, separators=(",", ":"))


def _stateful_file(root, record):
    if not isinstance(record, dict):
        raise ValueError("stateful artifact requires file records")
    path = record.get("path")
    if (not isinstance(path, str) or not path or Path(path).is_absolute()
            or ".." in Path(path).parts or Path(path).as_posix() != path
            or type(record.get("bytes")) is not int or record["bytes"] < 0
            or not re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256", "")))):
        raise ValueError("invalid stateful artifact file record")
    relative = Path(path)
    if any((root / part).is_symlink() for part in [relative, *relative.parents] if part != Path(".")):
        raise ValueError(f"stateful artifact must not use symlinks: {path}")
    return checked_file(root, record)


def _checked_stateful_air(root, *, ddk_id):
    """Check retained construction, recipe boundaries and declared provenance.

    AIR remains opaque here: its export-time hash binds it to these records;
    host verification cannot establish installed kernels or device behavior.
    """
    from .ge_export import weight_identity
    from .ge_fused.export import selected_fusion
    from .ge_fused.recipe import read_recipe, _attribute

    if (root / "manifest.json").is_symlink():
        raise ValueError("stateful AIR manifest must not be a symlink")
    manifest = _stateful_json(root / "manifest.json")
    expected = {"schema_version": 1, "artifact_kind": "qwen35-stateful-ge-air",
                "status": "EXPORTED", "export_backend": "direct-ge", "model_role": "target",
                "state_policy": "explicit", "legacy_hiai_abi_compatible": False,
                "precision": "float16", "feature_dtype": "float16", "quantization": None}
    if (any(key not in manifest or manifest[key] != value for key, value in expected.items())
            or type(manifest["schema_version"]) is not int or manifest["legacy_hiai_abi_compatible"] is not False):
        raise ValueError("compile-stateful requires an EXPORTED qwen35-stateful-ge-air target; SOURCE_READY is not AIR")
    if manifest.get("export_root") != str(root):
        raise ValueError("compile stateful AIR at its export_root: FileConstant and recipe paths are absolute")
    if type(manifest.get("external_weights")) is not bool:
        raise ValueError("stateful AIR external_weights must be boolean")
    if (not isinstance(manifest.get("ge_root"), str) or not Path(manifest["ge_root"]).is_absolute()
            or type(manifest.get("cxx11_abi")) is not int or manifest["cxx11_abi"] not in (0, 1)
            or manifest.get("expert_dispatch") not in {"static", "selected"}):
        raise ValueError("stateful AIR requires GE toolchain and expert dispatch provenance")
    for field in ("prepared_manifest_sha256", "target_fp16_payload_sha256", "libgraph_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(manifest.get(field, ""))):
            raise ValueError(f"stateful AIR requires {field}")
    source = manifest.get("source")
    if (not isinstance(source, dict) or not source.get("kind")
            or not re.fullmatch(r"[0-9a-f]{64}", str(source.get("fp16_payload_sha256", "")))):
        raise ValueError("stateful AIR requires the prepared source FP16 identity")
    roles = list(selected_fusion(manifest.get("fused_ops")))
    if roles != manifest["fused_ops"]:
        raise ValueError("stateful AIR fused_ops order differs from its export")
    construction = manifest.get("construction_files")
    if not isinstance(construction, list) or not construction:
        raise ValueError("stateful AIR requires hash-locked construction files")
    paths = {}
    for record in construction:
        path = _stateful_file(root, record)
        if record["path"] in paths or not record["path"].startswith("target/"):
            raise ValueError("duplicate or misplaced stateful construction file")
        paths[record["path"]] = path
    required = {"target/target.cpp", "target/graph.json", "target/state-abi.json",
                "target/recipe/graph.recipe", "target/recipe/constants.bin"}
    if not required <= paths.keys():
        raise ValueError("stateful AIR lacks retained graph/state ABI/recipe construction")
    graph, abi = manifest.get("graph"), manifest.get("state_abi")
    if (not isinstance(graph, dict) or graph.get("name") != "stateful_target"
            or not _same_ordered_json(graph, _stateful_json(paths["target/graph.json"]))
            or not isinstance(abi, dict) or not _same_ordered_json(abi, _stateful_json(paths["target/state-abi.json"]))
            or not _same_ordered_json(abi, graph.get("state_abi"))):
        raise ValueError("stateful graph/state ABI differs from retained construction")
    for policy, choices, filename in (
            ("fp16_reductions", {"native", "matmul"}, "reduction-lowering.json"),
            ("fp16_row_mul", {"native", "compact"}, "row-mul-lowering.json"),
            ("gdn_indexed_copy", {"native", "gather"}, "gdn-copy-lowering.json")):
        report_path = paths.get("target/" + filename)
        report = _stateful_json(report_path) if report_path else {}
        if (manifest.get(policy) not in choices or report.get("mode") != manifest[policy]
                or report.get("graph") != graph["name"] or not isinstance(report.get("counts"), dict)):
            raise ValueError(f"stateful {policy} differs from retained lowering report")
    weights = manifest.get("weights")
    if not isinstance(weights, dict) or not weights or set(weights) != set(graph.get("weights", {})):
        raise ValueError("stateful graph weight set differs from retained weights")
    weight_paths = {}
    for key, item in weights.items():
        path = _stateful_file(root, item)
        shape = item.get("shape")
        desc = graph["weights"][key]
        if (not key.startswith("target.") or not isinstance(shape, list) or not shape
                or any(type(dim) is not int or dim <= 0 for dim in shape)
                or item.get("dtype") != "float16" or item["bytes"] != math.prod(shape) * 2
                or any(desc.get(field) != item.get(field) for field in ("shape", "dtype", "bytes"))
                or Path(item["path"]).parts[0] != "weights" or str(path) in weight_paths):
            raise ValueError(f"invalid stateful weight descriptor: {key}")
        weight_paths[str(path)] = item
    if weight_identity(weights) != manifest["target_fp16_payload_sha256"]:
        raise ValueError("stateful target weight identity mismatch")
    binding_record = manifest.get("bindings")
    if roles:
        from .ge_fused.bindings import OperatorBindings
        bindings = OperatorBindings.from_file(_stateful_file(root, binding_record))
        bindings.require(roles)
        if not _same_ordered_json(bindings.manifest(), manifest.get("binding_contract")):
            raise ValueError("stateful binding contract differs from retained bindings")
        if bindings.manifest()["ddk_id"] != ddk_id:
            raise ValueError("stateful binding ddk_id differs from selected DDK identity")
        if binding_record["path"] != "bindings.json":
            raise ValueError("stateful bindings must use the retained bindings.json")
        specs = [bindings.manifest()["operators"][role] for role in roles]
        for node in graph.get("custom_ops", []):
            candidates = [spec for spec in specs if spec["op_type"] == node["type"]]
            def matches(spec):
                for direction in ("inputs", "outputs"):
                    ports = list(spec[direction].values())
                    if [port["port"] for port in ports] != list(node[direction]):
                        return False
                    for port in ports:
                        desc = node[direction][port["port"]]
                        try:
                            bindings._tensor_contract(port, desc["shape"], desc["dtype"], node["name"])
                        except ValueError:
                            return False
                attrs = node["attrs"]
                if set(attrs) != {attr["name"] for attr in spec["attrs"].values()} | set(spec.get("static_attrs", {})):
                    return False
                return (all(attrs[attr["name"]]["type"] == attr["type"] for attr in spec["attrs"].values())
                        and all(attrs[key] == value for key, value in spec.get("static_attrs", {}).items()))
            if not any(matches(spec) for spec in candidates):
                raise ValueError("stateful custom operator differs from selected binding contract")
    elif binding_record is not None or manifest.get("binding_contract") is not None or graph.get("custom_ops"):
        raise ValueError("stateful unfused graph must not carry custom bindings or operators")
    _check_stateful_abi(manifest)
    recipe = read_recipe(paths["target/recipe/graph.recipe"])
    records = recipe["records"]
    def boundary(index, port=0):
        record = records[index]
        desc = record["outputs"][port]
        return {"name": record["name"], "shape": desc["shape"], "dtype": desc["dtype"]}
    if (recipe["graph_name"] != graph["name"]
            or [boundary(index) for index in recipe["inputs"]] != graph["inputs"]
            or [boundary(index, port) for index, port in recipe["outputs"]] != graph["outputs"]):
        raise ValueError("stateful recipe boundary order/descriptors differ from graph")
    recipe_by_name = {record["name"]: record for record in records}
    for node in graph.get("custom_ops", []):
        record = recipe_by_name.get(node["name"])
        if (record is None or record["type"] != node["type"] or set(record["attrs"]) != set(node["attrs"])
                or any(_attribute(record["attrs"][key]) != _attribute(value) for key, value in node["attrs"].items())):
            raise ValueError("stateful custom operator differs from retained recipe")
        for direction in ("inputs", "outputs"):
            expected_ports = [{"port": name, "shape": desc["shape"], "dtype": desc["dtype"]}
                              for name, desc in node[direction].items()]
            actual_ports = [{key: desc[key] for key in ("port", "shape", "dtype")}
                            for desc in record[direction]]
            if expected_ports != actual_ports:
                raise ValueError("stateful custom port order/descriptors differ from recipe")
    for record in records:
        constant = record.get("constant")
        if record["type"] == "FileConstant":
            path = record["attrs"]["file_path"]["value"]
            desc = record["outputs"][0]
            item = weight_paths.get(path)
            if not manifest["external_weights"] or item is None or any(item[k] != desc[k] for k in ("shape", "dtype")):
                raise ValueError("stateful FileConstant is not bound to exported weights")
        elif constant:
            path = constant["path"]
            if constant["kind"] == 1:
                if (path != str(paths["target/recipe/constants.bin"])
                        or constant["offset"] + constant["size"] > Path(path).stat().st_size):
                    raise ValueError("stateful recipe constant escapes retained constants")
            else:
                item = weight_paths.get(path)
                if item is None or constant["size"] != item["bytes"]:
                    raise ValueError("stateful recipe weight is not bound to exported weights")
    air = _stateful_file(root, manifest.get("air"))
    if manifest["air"]["path"] != "target/target.air" or not air.stat().st_size:
        raise ValueError("stateful AIR requires a nonempty target/target.air")
    return manifest, air


def _check_stateful_abi(manifest):
    graph, abi = manifest["graph"], manifest["state_abi"]
    sequence, past = manifest.get("sequence_length"), manifest.get("past_length")
    if (type(sequence) is not int or sequence <= 0 or type(past) is not int or past < 0
            or abi.get("schema_version") != 1 or abi.get("batch_size") != 1
            or abi.get("sequence_length") != sequence or abi.get("past_length") != past
            or abi.get("next_past_length") != sequence + past
            or abi.get("valid_tokens") != "all_sequence_positions"
            or abi.get("cache_layout") != "ND" or abi.get("cache_update") != "functional"
            or abi.get("requires_matching_past_length_graph") is not True
            or manifest.get("recurrent_dtype") not in {"float16", "float32"}
            or manifest.get("gdn_state_casts") not in {"reject", "allow"}):
        raise ValueError("invalid explicit state ABI")
    for field in ("fused_ops", "recurrent_dtype", "gdn_state_casts", "gdn_indexed_copy"):
        if abi.get(field) != manifest.get(field):
            raise ValueError(f"stateful {field} differs from state ABI")
    boundaries = {}
    for direction in ("inputs", "outputs"):
        values = graph.get(direction)
        if not isinstance(values, list) or not values:
            raise ValueError("stateful graph requires ordered input/output descriptors")
        names = [value.get("name") for value in values]
        if len(set(names)) != len(names) or names != abi.get(direction[:-1] + "_order"):
            raise ValueError("stateful graph order differs from state ABI")
        for value in values:
            if (not isinstance(value.get("name"), str) or not value["name"]
                    or value.get("dtype") not in {"float16", "float32", "int32"}
                    or not isinstance(value.get("shape"), list) or not value["shape"]
                    or any(type(dim) is not int or dim <= 0 for dim in value["shape"])):
                raise ValueError("invalid stateful boundary descriptor")
        boundaries[direction] = dict(zip(names, values))
    if (graph["inputs"][0] != {"name": "input_ids", "shape": [1, sequence], "dtype": "int32"}
            or graph["outputs"][0] != {"name": "target_top1", "shape": [1, sequence], "dtype": "int32"}
            or len(graph["outputs"]) < 3 or graph["outputs"][1]["name"] != "target_features"
            or graph["outputs"][1]["dtype"] != "float16" or graph["outputs"][1]["shape"][:2] != [1, sequence]
            or len(graph["outputs"][1]["shape"]) != 3):
        raise ValueError("stateful target token/features boundaries are invalid")
    states = abi.get("states")
    if not isinstance(states, list) or not states:
        raise ValueError("stateful target requires explicit cache records")
    if ([state.get("output") for state in states] != abi["output_order"][2:]
            or [state.get("input") for state in states if state.get("input") is not None] != abi["input_order"][1:]):
        raise ValueError("stateful cache order differs from graph boundaries")
    seen_states = set()
    for state in states:
        output = boundaries["outputs"][state["output"]]
        role, layer = state.get("role"), state.get("layer")
        if (role not in {"key", "value", "conv_history", "recurrent"}
                or type(layer) is not int or layer < 0 or (layer, role) in seen_states
                or state.get("layer_type") != ("linear_attention" if role in {"conv_history", "recurrent"} else "full_attention")
                or state.get("output_shape") != output["shape"] or state.get("dtype") != output["dtype"]
                or state["dtype"] != (manifest["recurrent_dtype"] if role == "recurrent" else "float16")
                or state.get("initialization") != ("input" if past else "zeros")
                or (past == 0 and state.get("input") is not None)):
            raise ValueError("stateful cache descriptor differs from graph")
        seen_states.add((layer, role))
        shape = state["output_shape"]
        if (shape[0] != 1 or len(shape) != (3 if role == "conv_history" else 4)
                or (role in {"key", "value"} and shape[2] != sequence + past)
                or (role in {"conv_history", "recurrent"} and state.get("input_shape") != shape)):
            raise ValueError("stateful cache shape disagrees with its role or past length")
        if past:
            incoming = boundaries["inputs"].get(state.get("input"), {})
            if incoming.get("shape") != state.get("input_shape") or incoming.get("dtype") != state.get("dtype"):
                raise ValueError("stateful input cache descriptor differs from graph")
            if role in {"key", "value"} and incoming["shape"] != [*shape[:2], past, shape[3]]:
                raise ValueError("stateful KV cache growth differs from past length")


def compile_stateful_air(bundle, output_dir, ddk_root, *, ddk_id, timeout_seconds=3600,
                         reject_cpu_fallback=False):
    """Compile one checked explicit-state target; never adapts it to legacy ABI v2."""
    if not isinstance(ddk_id, str) or not ddk_id.strip() or timeout_seconds <= 0:
        raise ValueError("declare a nonempty ddk_id and positive timeout_seconds")
    root = Path(bundle).resolve()
    manifest_path = root / "manifest.json"
    manifest_sha = sha256_file(manifest_path)
    manifest, air = _checked_stateful_air(root, ddk_id=ddk_id)
    omg, env = omg_environment(ddk_root)
    if not os.access(omg, os.X_OK):
        raise PermissionError(f"OMG is not executable: {omg}; fix the run-local extraction permissions")
    output = output_path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    # Keep the complete source contract in the OMC manifest, with source records
    # explicitly rooted in air_source_root (not presented as OMC payload paths).
    result = {key: manifest[key] for key in (
        "model_role", "state_policy", "state_abi", "legacy_hiai_abi_compatible", "fused_ops",
        "binding_contract", "sequence_length", "past_length", "recurrent_dtype", "feature_dtype",
        "precision", "quantization", "expert_dispatch", "fp16_reductions", "fp16_row_mul",
        "gdn_indexed_copy", "gdn_state_casts", "source", "prepared_manifest_sha256",
        "target_fp16_payload_sha256", "external_weights", "graph")}
    result.update(schema_version=1, artifact_kind="qwen35-stateful-omc", status="COMPILING",
                  runtime="hiai", platform="kirinx90", ddk_id=ddk_id,
                  ddk_root=str(Path(ddk_root).resolve()), omg_sha256=sha256_file(omg),
                  air_source_root=str(root), air_manifest_sha256=manifest_sha,
                  air_provenance={key: manifest[key] for key in (
                      "air", "construction_files", "weights", "bindings", "export_root",
                      "ge_root", "libgraph_sha256", "cxx11_abi")},
                  reject_cpu_fallback=reject_cpu_fallback, graphs=[], device_validation="PENDING",
                  kernel_registration="UNVERIFIED", runtime_placement="UNVERIFIED",
                  validation_scope="Host OMG compilation only; explicit-state runner and device execution remain unvalidated.")
    folder = output / "target"
    log_path = folder / "omg.log"
    stage = "target-setup"
    try:
        folder.mkdir()
        command = [str(omg), "--framework=1", "--model=input.air", "--output=target", "--target=omc",
                   "--platform=kirinx90", "--use_origin_format=true", "--weight_data_type=FP16"]
        save_json(folder / "command.json", {"argv": command, "ddk_id": ddk_id, "air_source": str(air),
                                           "cwd_policy": "isolated-run-scratch"})
        with tempfile.TemporaryDirectory(prefix="omg-target-", dir=output) as temp:
            scratch = Path(temp)
            (scratch / "input.air").symlink_to(air)
            stage = "target-omg"
            try:
                with _omg_ipc_environment(scratch, env) as (local_env, ipc), log_path.open("x") as log:
                    environment_record = _environment_record(ddk_root, local_env)
                    environment_record["ipc_temporary_directory"] = ipc
                    save_json(folder / "omg-environment.json", environment_record)
                    subprocess.run(command, cwd=scratch, env=local_env, stdout=log, stderr=subprocess.STDOUT,
                                   check=True, timeout=timeout_seconds)
                stage = "target-output-check"
                generated = scratch / "target.omc"
                if generated.is_symlink() or not generated.is_file() or not generated.stat().st_size:
                    raise RuntimeError("OMG succeeded without a nonempty regular target.omc")
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError, RuntimeError) as error:
                diagnostic_note = ""
                try:
                    _retain_failure_diagnostics(scratch, folder)
                except OSError as diagnostic_error:
                    diagnostic_note = f"\nCould not retain graph diagnostics: {diagnostic_error}"
                raise RuntimeError(f"OMG graph=target stage={stage}: {error}\nFull log: {log_path}\n"
                                   f"{log_excerpt(log_path)}{diagnostic_note}") from error
            try:
                compiler_diagnostics = _retain_compiler_diagnostics(scratch, folder)
            except OSError as diagnostic_error:
                compiler_diagnostics = {"status": "FAILED", "error": str(diagnostic_error)}
            stage = "target-collect"
            for item in scratch.iterdir():
                if item.name.startswith("target") and not item.is_symlink():
                    shutil.move(str(item), str(folder / item.name))
        stage = "target-source-check"
        if sha256_file(manifest_path) != manifest_sha:
            raise ValueError("stateful AIR manifest changed during compilation")
        _checked_stateful_air(root, ddk_id=ddk_id)
        omc = folder / "target.omc"
        evidence = compile_evidence(log_path.read_text(encoding="utf-8", errors="replace"))
        evidence["log_sha256"] = sha256_file(log_path)
        graph = manifest["graph"]
        result["graphs"].append({"name": "target", "graph_name": graph["name"], "omc": "target/target.omc",
                                 "bytes": omc.stat().st_size, "sha256": sha256_file(omc),
                                 "inputs": graph["inputs"], "outputs": graph["outputs"],
                                 "state_abi": manifest["state_abi"],
                                 "compiler_diagnostics": compiler_diagnostics, "compiler_evidence": evidence,
                                 "payloads": _compiled_payloads(folder, output)})
        rollbacks = [event for event in evidence.get("rollback_events", [])
                     if event["scope"] in {"main", "unknown"} and event["device"] == "NPUCL"]
        if rollbacks:
            message = "stateful target: OMG reported NPU rollback to CPU; compiler output and diagnostics retained"
            print(f"compile-stateful: {message}", file=sys.stderr)
            if reject_cpu_fallback:
                stage = "target-placement-check"
                result["cpu_fallback_rejections"] = rollbacks
                raise RuntimeError(message)
        result["status"] = "COMPILED"
        save_json(output / "manifest.json", result)
        return result
    except Exception as error:
        cause = error.__cause__ or error
        result.update(status="FAILED", stage=stage, failed_graph="target", log_path=str(log_path),
                      error=f"{type(error).__name__}: {error}",
                      returncode=cause.returncode if isinstance(cause, subprocess.CalledProcessError) else None)
        if isinstance(cause, subprocess.TimeoutExpired):
            result["timeout_seconds"] = cause.timeout
        save_json(output / "failure.json", result)
        raise
