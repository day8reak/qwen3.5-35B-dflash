"""Hash-checked AIR -> OMG --framework=1 -> OMC. Never invokes ATC or ONNX."""
import json
import os
from pathlib import Path
import subprocess
import shutil
import tempfile
from contextlib import contextmanager
from .artifacts import checked_file, files_under, output_path, save_json
from .hashing import sha256_file
from .tool_logs import log_excerpt


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


def _retain_failure_diagnostics(scratch, folder, *, max_bytes=16 << 20):
    """Keep bounded OMG graph dumps before deleting scratch; never input links."""
    source = scratch / "dump_graph"
    if source.is_symlink() or not source.is_dir():
        return
    records = []
    used = 0
    # Prefer the failure graphs if successful intermediate dumps fill the budget.
    files = sorted(source.glob("*.irpb"), key=lambda path: ("fail" not in path.name, path.name))
    destination = folder / "diagnostics"
    destination.mkdir(exist_ok=True)
    for path in files:
        if path.is_symlink() or not path.is_file():
            continue
        size = path.stat().st_size
        record = {"name": path.name, "bytes": size}
        if used + size > max_bytes:
            record["status"] = "SKIPPED_SIZE_BUDGET"
        else:
            target = destination / path.name
            shutil.copyfile(path, target)
            used += size
            record.update(status="RETAINED", sha256=sha256_file(target))
        records.append(record)
    save_json(destination / "manifest.json", {"max_bytes": max_bytes, "retained_bytes": used,
                                              "files": records})


def compile_air(bundle, output_dir, ddk_root, *, ddk_id, timeout_seconds=3600):
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
    graphs = manifest["graphs"]
    if [g["name"] for g in graphs] != ["target", "draft"]:
        raise ValueError("AIR bundle must contain target then draft")
    if not ddk_id or timeout_seconds <= 0:
        raise ValueError("DDK identity and positive timeout required")
    for graph in graphs:
        paths = {r["path"]: checked_file(root, r) for r in graph["files"]}
        if graph["air"] not in paths or paths[graph["air"]].suffix != ".air":
            raise ValueError("AIR graph is missing from hash-locked payloads")
    omg, env = omg_environment(ddk_root)
    if not os.access(omg, os.X_OK):
        raise PermissionError(f"OMG is not executable: {omg}; fix the run-local extraction permissions")
    output = output_path(output_dir); output.mkdir(parents=True, exist_ok=False)
    records = []
    result = {k: manifest[k] for k in ["sequence_length", "draft_width", "feature_width", "vocab_size",
                                      "precision", "state_policy", "expert_dispatch", "source", "pad_token_id", "eos_token_ids",
                                      "abi_version", "feature_dtype", "precision_audit", "quantization"]}
    result.update(schema_version=2, artifact_kind="qwen35-cpro-omc-pair", status="COMPILING",
                  platform="kirinx90", runtime="hiai", device_validation="PENDING", graphs=records,
                  ddk_id=ddk_id, omg_sha256=sha256_file(omg), air_manifest_sha256=sha256_file(manifest_path))
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
                stage = f"{name}-collect"
                # Preserve compiled outputs/sidecars, but never archive input
                # symlinks or compiler scratch as model payloads.
                for item in scratch.iterdir():
                    if item.name.startswith(name) and not item.is_symlink():
                        shutil.move(str(item), str(folder / item.name))
            omc = folder / f"{name}.omc"
            # Detect accidental compiler mutation of AIR or external weights.
            for payload in graph["files"]: checked_file(root, payload)
            records.append({"name": name, "omc": str(omc.relative_to(output)),
                            "sha256": sha256_file(omc), "bytes": omc.stat().st_size,
                            "inputs": graph["inputs"], "outputs": graph["outputs"],
                            "input_shapes": graph["input_shapes"], "input_dtypes": graph["input_dtypes"],
                            "output_shapes": graph["output_shapes"], "output_dtypes": graph["output_dtypes"],
                            "payloads": files_under(folder, output)})
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
