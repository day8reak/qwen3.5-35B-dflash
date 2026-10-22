#!/usr/bin/env python3
"""Independent GE C++ Add AIR smoke; Python uses only the standard library."""
import argparse
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "model"))
from qwen35_moe_dflash.artifacts import checked_file, files_under, output_path, save_json
from qwen35_moe_dflash.compile import omg_environment
from qwen35_moe_dflash.hashing import sha256_file
from qwen35_moe_dflash.resources import load_resources, resource_path


def tensor(name):
    return {"name": name, "shape": [1, 4], "dtype": "float16", "format": "ND"}


def _record(path):
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _logged(command, *, cwd, env, log, timeout):
    with log.open("x") as stream:
        subprocess.run(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT,
                       check=True, timeout=timeout)


def _failure(output, stage, error):
    save_json(output / "failure.json", {"status": "FAILED", "stage": stage,
              "device_validation": "PENDING", "error": f"{type(error).__name__}: {error}"})


def export_smoke(ge_root, output_dir, *, cxx="c++", cxx11_abi=0, timeout_seconds=120):
    """Serialize a real GE graph; this neither compiles OMC nor runs a device."""
    output = output_path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    stage = "ge-toolchain"
    try:
        if cxx11_abi not in (0, 1) or timeout_seconds <= 0:
            raise ValueError("C++ ABI must be 0 or 1 and timeout must be positive")
        ge = Path(ge_root).expanduser().resolve()
        include, lib = ge / "include", ge / "lib64"
        headers = [include / "graph" / name for name in ("graph.h", "operator_reg.h", "tensor.h")]
        library = lib / "libgraph.so"
        missing = [str(path) for path in [*headers, library] if not path.is_file()]
        if missing:
            raise FileNotFoundError("GE requires matching graph headers and libgraph.so; missing: "
                                    + ", ".join(missing))
        source = REPO / "targets/cpro/air/ge_add_air.cpp"
        if not source.is_file():
            raise FileNotFoundError(source)
        compiler = shutil.which(cxx)
        if compiler is None:
            raise FileNotFoundError(f"C++ compiler not found: {cxx}")
        build, payload = output / "build", output / "payload"
        build.mkdir(); payload.mkdir()
        binary, air = build / "ge_add_air", payload / "add.air"
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            value for value in (str(lib), env.get("LD_LIBRARY_PATH", "")) if value)
        command = [compiler, "-std=c++17", f"-D_GLIBCXX_USE_CXX11_ABI={cxx11_abi}",
                   "-I", str(include), str(source), "-L", str(lib),
                   f"-Wl,-rpath,{lib}", "-lgraph"]
        # Newer GE SDKs split Operator/Tensor symbols into graph_base.
        graph_base = lib / "libgraph_base.so"
        if graph_base.is_file():
            command.append("-lgraph_base")
        command.extend(["-o", str(binary)])
        save_json(output / "build-command.json", {"argv": command, "cwd": str(build)})
        stage = "ge-build"
        _logged(command, cwd=build, env=env, log=output / "build.log", timeout=timeout_seconds)
        if not binary.is_file() or not binary.stat().st_size:
            raise RuntimeError("compiler succeeded without a nonempty GE exporter")
        command = [str(binary), str(air)]
        save_json(output / "export-command.json", {"argv": command, "cwd": str(payload)})
        stage = "ge-serialize"
        _logged(command, cwd=payload, env=env, log=output / "export.log", timeout=timeout_seconds)
        if not air.is_file() or not air.stat().st_size:
            raise RuntimeError("GE exporter succeeded without a nonempty add.air")
        (payload / "input.bin").write_bytes(struct.pack("<4e", 0, 1, 2, 3))
        (payload / "expected.bin").write_bytes(struct.pack("<4e", 1, 3, 5, 7))
        manifest = {
            "schema_version": 1, "artifact_kind": "ge-air-smoke", "status": "AIR_EXPORTED",
            "device_validation": "PENDING", "scope": "single-fp16-add-graph",
            "graph": {"name": "add", "air": "payload/add.air", "operator": "Add",
                      "constant_values": [1, 2, 3, 4], "weights": "embedded",
                      "inputs": [tensor("x")], "outputs": [tensor("y")]},
            "fixture": {"input": "payload/input.bin", "expected": "payload/expected.bin",
                        "byte_order": "little", "comparison": "exact-fp16"},
            "source": _record(source), "ge_root": str(ge), "cxx11_abi": cxx11_abi,
            "headers": [_record(path) for path in headers], "library": _record(library),
            "additional_libraries": [_record(graph_base)] if graph_base.is_file() else [],
            "exporter": _record(binary), "files": files_under(payload, output),
        }
        save_json(output / "manifest.json", manifest)
        return manifest
    except Exception as error:
        _failure(output, stage, error)
        raise


def _air_bundle(bundle):
    root = Path(bundle).resolve()
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    if (manifest.get("schema_version") != 1 or manifest.get("artifact_kind") != "ge-air-smoke"
            or manifest.get("status") != "AIR_EXPORTED"):
        raise ValueError("only a successful independent ge-air-smoke bundle is accepted")
    graph, fixture = manifest["graph"], manifest["fixture"]
    if (graph.get("name") != "add" or graph.get("operator") != "Add"
            or graph.get("weights") != "embedded" or graph.get("constant_values") != [1, 2, 3, 4]
            or graph.get("inputs") != [tensor("x")] or graph.get("outputs") != [tensor("y")]
            or fixture.get("byte_order") != "little" or fixture.get("comparison") != "exact-fp16"):
        raise ValueError("smoke requires fixed ND FP16 Add ABI [1,4] with embedded constants")
    records = manifest["files"]
    paths = {record["path"]: checked_file(root, record) for record in records}
    expected_paths = {graph["air"], fixture["input"], fixture["expected"]}
    if len(paths) != len(records) or set(paths) != expected_paths or len(paths) != 3:
        raise ValueError("smoke bundle needs exactly AIR, input and expected hash-locked files")
    air = paths[graph["air"]]
    if air.suffix != ".air" or not air.stat().st_size:
        raise ValueError("smoke AIR must be a nonempty .air file")
    if (paths[fixture["input"]].read_bytes() != struct.pack("<4e", 0, 1, 2, 3)
            or paths[fixture["expected"]].read_bytes() != struct.pack("<4e", 1, 3, 5, 7)):
        raise ValueError("smoke fixture differs from the fixed FP16 Add case")
    return root, manifest, paths, sha256_file(path)


def compile_smoke(bundle, output_dir, ddk_root, *, ddk_id, timeout_seconds=3600):
    """Compile only this independent smoke ABI, using the selected mobile DDK."""
    output = output_path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    root, manifest, paths, manifest_hash = _air_bundle(bundle)
    if not ddk_id or "$" in ddk_id or timeout_seconds <= 0:
        raise ValueError("resolved DDK identity and positive timeout required")
    omg, env = omg_environment(ddk_root)
    if not os.access(omg, os.X_OK):
        raise PermissionError(f"OMG is not executable: {omg}")
    output.mkdir(parents=True, exist_ok=False)
    try:
        payload = output / "payload"; payload.mkdir()
        command = [str(omg), "--framework=1", "--model=input.air", "--output=add",
                   "--target=omc", "--platform=kirinx90", "--use_origin_format=true",
                   "--weight_data_type=FP16"]
        save_json(output / "command.json", {"argv": command, "ddk_id": ddk_id,
                  "air_manifest_sha256": manifest_hash, "cwd_policy": "isolated-run-scratch"})
        with tempfile.TemporaryDirectory(prefix="omg-add-", dir=output) as temp:
            scratch = Path(temp)
            shutil.copyfile(paths[manifest["graph"]["air"]], scratch / "input.air")
            _logged(command, cwd=scratch, env=dict(env, TMPDIR=str(scratch)),
                    log=output / "omg.log", timeout=timeout_seconds)
            omc = scratch / "add.omc"
            if not omc.is_file() or omc.is_symlink() or not omc.stat().st_size:
                raise RuntimeError("OMG succeeded without a nonempty regular add.omc")
            for item in scratch.iterdir():
                if item.name.startswith("add"):
                    if item.is_symlink():
                        raise ValueError("OMG output must not be a symlink")
                    shutil.move(str(item), payload / item.name)
        for record in manifest["files"]:
            checked_file(root, record)
        if sha256_file(root / "manifest.json") != manifest_hash:
            raise ValueError("AIR manifest changed during OMG conversion")
        for name in ("input", "expected"):
            shutil.copyfile(paths[manifest["fixture"][name]], payload / f"{name}.bin")
        graph = {key: value for key, value in manifest["graph"].items() if key != "air"}
        graph["omc"] = "payload/add.omc"
        result = {"schema_version": 1, "artifact_kind": "ge-omc-smoke", "status": "COMPILED",
                  "device_validation": "PENDING", "scope": "single-fp16-add-graph",
                  "platform": "kirinx90", "runtime": "hiai", "ddk_id": ddk_id,
                  "omg_sha256": sha256_file(omg), "air_manifest_sha256": manifest_hash,
                  "source": manifest["source"], "graph": graph,
                  "fixture": dict(manifest["fixture"], input="payload/input.bin", expected="payload/expected.bin"),
                  "files": files_under(payload, output)}
        save_json(output / "manifest.json", result)
        return result
    except Exception as error:
        _failure(output, "omg-compile", error)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--ge-root")
    export.add_argument("--resources")
    export.add_argument("--output", required=True)
    export.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    export.add_argument("--cxx11-abi", choices=(0, 1), type=int, default=0)
    export.add_argument("--timeout-seconds", type=int, default=120)
    compile_parser = commands.add_parser("compile")
    compile_parser.add_argument("--air-bundle", required=True)
    compile_parser.add_argument("--output", required=True)
    compile_parser.add_argument("--ddk-root")
    compile_parser.add_argument("--ddk-id", default=os.environ.get("CPRO_DDK_ID"))
    compile_parser.add_argument("--resources")
    compile_parser.add_argument("--timeout-seconds", type=int, default=3600)
    args = parser.parse_args()
    try:
        if args.command == "export":
            ge_root = args.ge_root
            if not ge_root and args.resources:
                resources, _ = load_resources(args.resources)
                ge_root = resource_path(resources, "ge_root", directory=True)
            if not ge_root:
                ge_root = os.environ.get("CANN_ROOT") or os.environ.get("ASCEND_HOME_PATH")
            if not ge_root:
                resources, _ = load_resources(args.resources)
                ge_root = resource_path(resources, "ge_root", directory=True)
            export_smoke(ge_root, args.output, cxx=args.cxx, cxx11_abi=args.cxx11_abi,
                         timeout_seconds=args.timeout_seconds)
        else:
            resources = {}
            if not args.ddk_root or not args.ddk_id:
                resources, _ = load_resources(args.resources)
            ddk = Path(args.ddk_root).expanduser().resolve() if args.ddk_root else resource_path(resources, "ddk_root", directory=True)
            ddk_id = args.ddk_id or os.path.expandvars(resources.get("ddk_id", ""))
            compile_smoke(args.air_bundle, args.output, ddk, ddk_id=ddk_id,
                          timeout_seconds=args.timeout_seconds)
    except Exception as error:
        print(f"direct-air-smoke: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
