"""Hash-checked AIR -> OMG --framework=1 -> OMC. Never invokes ATC or ONNX."""
import json
import os
from pathlib import Path
import subprocess
import shutil
import tempfile
from .artifacts import checked_file, files_under, output_path, save_json
from .weights import sha256_file


def omg_environment(ddk_root, inherited=None):
    """Use the actual binary, avoiding the vendor wrapper's shared-tree writes."""
    ddk_root = Path(ddk_root).resolve()
    master = ddk_root / "tools/tools_omg/master"
    plugin = ddk_root / "tools/platform/kirinx90"
    if not (master / "omg").is_file() or not plugin.is_dir():
        raise FileNotFoundError("DDK needs tools/tools_omg/master/omg and tools/platform/kirinx90")
    env = dict(os.environ if inherited is None else inherited)
    libs = [master / "lib64", plugin / "lib64"]
    for candidate in [ddk_root / "ddk/tbe/lib64", ddk_root / "ddk/ascendcpp/lib64"]:
        if candidate.is_dir(): libs.append(candidate)
    env["LD_LIBRARY_PATH"] = ":".join(map(str, libs))
    env["CCE_LIB_DIR"] = str(master / "lib64")
    env["SOC_VERSION"] = "kirinx90"
    env["PYTHONPATH"] = str(plugin / "ops/impl")
    env["KIRIN_CONF_JSON_PATH"] = str(ddk_root / "ddk/ascendcpp/conf")
    for candidate in [ddk_root / "ddk/ccec_compiler/bin", ddk_root / "tools/tools_ascendc/bisheng/bin"]:
        if candidate.is_dir(): env["PATH"] = str(candidate) + os.pathsep + env.get("PATH", "")
    return master / "omg", env


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
    try:
        for graph in graphs:
            name = graph["name"]; folder = output / name; folder.mkdir()
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
                local_env = dict(env, TMPDIR=str(scratch))
                with (folder / "omg.log").open("w") as log:
                    subprocess.run(command, cwd=scratch, env=local_env, stdout=log, stderr=subprocess.STDOUT,
                                   check=True, timeout=timeout_seconds)
                generated = scratch / f"{name}.omc"
                if not generated.is_file() or not generated.stat().st_size:
                    raise RuntimeError(f"OMG succeeded without a nonempty {name}.omc")
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
        save_json(output / "failure.json", result)
        raise
    return result
