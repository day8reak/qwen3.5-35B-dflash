"""Generate, compile and execute bounded operator contracts on the SN/BIP board.

Uses the production GE AIR serializer and OMG environment. A successful model
load or Process call alone never establishes NPU placement.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from .artifacts import checked_file, files_under, output_path, save_json
from .compile import (_environment_record, _omg_ipc_environment,
                      _retain_compiler_diagnostics, omg_environment)
from .ge_cpp import emit_cpp
from .ge_export import _logged, graph_summary
from .hashing import sha256_file
from .hiai_deployment import _elf, discover_libraries
from .op_probe_cases import PROBE_FAMILIES, PROBE_SUITES, cases
from .op_probe_report import compile_evidence, write_report
from .resources import repository_root


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", help="new directory below AI_RUN_DIR")
    selection = p.add_mutually_exclusive_group()
    selection.add_argument("--case", action="append", default=[], help="case name; repeat to select several")
    selection.add_argument("--suite", choices=PROBE_SUITES,
                           help="named batch of related controls/candidates; excludes unrelated historical probes")
    p.add_argument("--exclude-family", choices=PROBE_FAMILIES, action="append", default=[],
                   help="with --suite, skip an already tested family; repeat to exclude several")
    p.add_argument("--list-cases", action="store_true")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true", help="write recipes and references without vendor tools")
    mode.add_argument("--compile-only", action="store_true", help="compile AIR/OMC without building or contacting the board")
    p.add_argument("--keep-board", action="store_true", help="retain this probe's remote directory after retrieval")
    p.add_argument("--timeout-seconds", type=int, default=300, help="timeout per host/HDC command (default 300)")
    return p


def _required_path(key):
    value = os.environ.get(key)
    if not value:
        raise ValueError(f"{key} is required")
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"{key} is not a directory: {path}")
    return path


def _payload(path):
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def prepare_case(case, root):
    """Freeze inputs, oracle outputs and emitted construction before conversion."""
    folder = root / "cases" / case["name"]
    folder.mkdir(parents=True)
    graph = case["graph"]
    source = folder / "graph.cpp"
    source.write_text(emit_cpp(graph, recipe_dir=folder / "recipe"))
    rows, inputs, outputs = [], [], []
    for i, tensor in enumerate(graph.inputs):
        value = np.asarray(case["inputs"][tensor.name], dtype=tensor.dtype)
        if tuple(value.shape) != tuple(tensor.shape):
            raise ValueError(f"input shape differs from graph: {case['name']}/{tensor.name}")
        path = folder / f"input_{i}.bin"
        value.astype(value.dtype.newbyteorder('<')).tofile(path)
        rows.append("\t".join(("input", tensor.dtype, ",".join(map(str, tensor.shape)), path.name)))
        inputs.append({**_payload(path), "name": tensor.name, "shape": list(tensor.shape), "dtype": tensor.dtype})
    for i, (tensor, value, tolerance) in enumerate(zip(graph.outputs, case["expected"], case["tolerances"])):
        value = np.asarray(value, dtype=tensor.dtype)
        if tuple(value.shape) != tuple(tensor.shape):
            raise ValueError(f"reference shape differs from graph: {case['name']}/{tensor.name}")
        path = folder / f"expected_{i}.bin"
        value.astype(value.dtype.newbyteorder('<')).tofile(path)
        output = f"output_{i}.bin"
        rows.append("\t".join(("output", tensor.dtype, ",".join(map(str, tensor.shape)), output)))
        outputs.append({"path": output, "name": tensor.name, "shape": list(tensor.shape),
                        "dtype": tensor.dtype, "expected": path.name, "reference": _payload(path), **tolerance})
    io = folder / "io.tsv"
    io.write_text("\n".join(rows) + "\n")
    record = {"name": case["name"], "description": case["description"],
              "directory": str(folder), "io": str(io), "io_sha256": sha256_file(io),
              "model": str(folder / "model.omc"), "inputs": [str(folder / v["path"]) for v in inputs],
              "input_tensors": inputs, "outputs": outputs,
              "source_sha256": sha256_file(source), "graph": graph_summary(graph),
              "construction_files": files_under(folder, folder),
              "equivalence_group": case.get("equivalence_group"), "compile": {"status": "NOT_RUN"}}
    if case.get("probe_suite"):
        record.update(probe_suite=case["probe_suite"], probe_family=case["probe_family"])
    if case.get("input_domain"):
        record["input_domain"] = case["input_domain"]
    save_json(folder / "contract.json", record)
    return record


def compiler_resources():
    ge, ddk = _required_path("CANN_ROOT"), _required_path("CPRO_DDK_ROOT")
    for path in (ge / "include/graph/graph.h", ge / "lib64/libgraph.so"):
        if not path.is_file():
            raise FileNotFoundError(f"matching GE headers/libraries required: {path}")
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        raise FileNotFoundError("C++ compiler not found")
    omg, env = omg_environment(ddk)
    return ge, ddk, compiler, omg, env


def compile_case(record, resources, timeout):
    ge, ddk, compiler, omg, env = resources
    folder = Path(record["directory"])
    stage = "build-exporter"
    result = {"status": "FAILED", "stage": stage}
    try:
        for item in record["construction_files"]:
            checked_file(folder, item)
        lib = ge / "lib64"
        ge_env = dict(os.environ)
        ge_env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, (str(lib), ge_env.get("LD_LIBRARY_PATH"))))
        binary, air = folder / "exporter", folder / "input.air"
        command = [compiler, "-std=c++17", "-D_GLIBCXX_USE_CXX11_ABI=0", "-I", str(ge / "include"),
                   str(folder / "graph.cpp"), "-L", str(lib), f"-Wl,-rpath,{lib}", "-lgraph"]
        if (lib / "libgraph_base.so").is_file():
            command.append("-lgraph_base")
        command += ["-o", str(binary)]
        save_json(folder / "build-command.json", {"argv": command})
        _logged(command, folder=folder, env=ge_env, log=folder / "build.log", timeout=timeout)
        stage = "export-air"
        command = [str(binary), str(air)]
        save_json(folder / "export-command.json", {"argv": command})
        _logged(command, folder=folder, env=ge_env, log=folder / "export.log", timeout=timeout)
        if not air.is_file() or not air.stat().st_size or air.is_symlink():
            raise RuntimeError("exporter produced no nonempty regular AIR")
        stage = "omg"
        with tempfile.TemporaryDirectory(prefix=".omg-", dir=folder) as temp:
            scratch = Path(temp)
            shutil.copyfile(air, scratch / "input.air")
            command = [str(omg), "--framework=1", "--model=input.air", "--output=model",
                       "--target=omc", "--platform=kirinx90", "--use_origin_format=true", "--weight_data_type=FP16"]
            failed = True
            try:
                with _omg_ipc_environment(scratch, env) as (child, ipc):
                    save_json(folder / "omg-command.json", {"argv": command, "ipc": ipc,
                                                           **_environment_record(ddk, child)})
                    _logged(command, folder=scratch, env=child, log=folder / "omg.log", timeout=timeout)
                model = scratch / "model.omc"
                if not model.is_file() or model.is_symlink() or not model.stat().st_size:
                    raise RuntimeError("OMG exited zero but produced no nonempty regular OMC")
                if model.stat().st_size > 64 << 20:
                    raise RuntimeError("probe OMC exceeds the native runner's 64 MiB limit")
                shutil.copyfile(model, record["model"])
                failed = False
            finally:
                try:
                    result["diagnostics"] = _retain_compiler_diagnostics(scratch, folder, failure=failed)
                except Exception as exc:
                    result["diagnostics"] = {"status": "ERROR", "error": str(exc)}
        for item in record["construction_files"]:
            checked_file(folder, item)
        result.update(status="COMPILED", stage="complete", model=_payload(Path(record["model"])), air=_payload(air))
    except Exception as exc:
        result.update(status="FAILED", stage=stage, error=f"{type(exc).__name__}: {exc}")
    log = folder / "omg.log"
    result["evidence"] = compile_evidence(log.read_text(errors="replace") if log.is_file() else "")
    save_json(folder / "compile-result.json", result)
    record["compile"] = result
    return result


def native_build(root, timeout):
    """Reuse only recorded OHOS/HIAI settings, never arbitrary saved argv."""
    build = _required_path("HIAI_BUILD_DIR")
    old_runner = build / "build/qwen35_dflash_hiai"
    deployment = discover_libraries(old_runner, build)
    original = json.loads((build / "command.json").read_text())
    allowed = {"CMAKE_TOOLCHAIN_FILE", "OHOS_ARCH", "OHOS_PLATFORM", "OHOS_STL",
               "OHOS_COMPATIBLE_SDK_VERSION", "HIAI_INCLUDE_DIR"}
    settings = {}
    for arg in original["argv"]:
        if arg.startswith("-D") and "=" in arg:
            key, value = arg[2:].split("=", 1)
            if key in allowed:
                if key in settings and settings[key] != value:
                    raise ValueError(f"conflicting recorded CMake setting: {key}")
                settings[key] = value
    for key in ("CMAKE_TOOLCHAIN_FILE", "HIAI_INCLUDE_DIR"):
        if key not in settings or not Path(settings[key]).is_absolute():
            raise ValueError(f"build record needs an absolute {key}")
    if (settings.get("OHOS_ARCH") != "arm64-v8a" or settings.get("OHOS_PLATFORM") != "OHOS"
            or settings.get("OHOS_STL") != "c++_shared"):
        raise ValueError("probe requires the recorded arm64-v8a OHOS c++_shared toolchain")
    toolchain = Path(settings["CMAKE_TOOLCHAIN_FILE"])
    header = Path(settings["HIAI_INCLUDE_DIR"]) / "HiAiModelManagerService.h"
    for path, key in ((toolchain, "sdk_toolchain_sha256"), (header, "header_sha256")):
        if not path.is_file():
            raise FileNotFoundError(path)
        if original.get(key) and original[key] != sha256_file(path):
            raise ValueError(f"{key} differs from original build: {path}")
    folder = root / "native"
    folder.mkdir()
    source = repository_root() / "targets/cpro/runtime"
    sources = [{"path": str(source / name), "sha256": sha256_file(source / name)}
               for name in ("CMakeLists.txt", "src/hiai_op_probe.cpp")]
    command = ["cmake", "-S", str(source), "-B", str(folder / "build")]
    command += [f"-D{key}={value}" for key, value in settings.items()]
    command += ["-DBUILD_HOST_TESTS=OFF", "-DBUILD_HIAI_RUNNER=ON", "-DBUILD_HIAI_OBJECTS=OFF",
                "-DHIAI_LIBRARIES=" + ";".join(v["path"] for v in original["libraries"])]
    save_json(folder / "command.json", {"argv": command, "libraries": original["libraries"],
        "sdk_toolchain_sha256": sha256_file(toolchain), "header_sha256": sha256_file(header),
        "source_build": str(build), "sources": sources, "library_discovery": deployment["metadata"]})
    _logged(command, folder=folder, env=os.environ.copy(), log=folder / "configure.log", timeout=timeout)
    _logged(["cmake", "--build", str(folder / "build"), "--target", "qwen35_hiai_op_probe", "-j", "4"],
            folder=folder, env=os.environ.copy(), log=folder / "build.log", timeout=timeout)
    runner = folder / "build/qwen35_hiai_op_probe"
    _elf(runner, executable=True)
    for item in sources:
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"native source changed during build: {item['path']}")
    save_json(folder / "result.json", {"status": "LINKED", "binary_sha256": sha256_file(runner),
                                      "command_sha256": sha256_file(folder / "command.json"), "device_validation": "NOT_RUN"})
    return runner, deployment["libraries"]


def run(args):
    if args.exclude_family and not args.suite:
        raise ValueError("--exclude-family requires --suite")
    selected = list(cases())
    names = {case["name"] for case in selected}
    if set(args.case) - names:
        raise ValueError("unknown probe cases: " + ", ".join(sorted(set(args.case) - names)))
    if args.case:
        selected = [c for c in selected if c["name"] in args.case]
    elif args.suite:
        selected = [c for c in selected if c.get("probe_suite") == args.suite
                    and c.get("probe_family") not in args.exclude_family]
        if not selected:
            raise ValueError(f"probe suite has no cases: {args.suite}")
    if args.list_cases:
        for case in selected:
            print(f"{case['name']}: {case['description']}")
        return 0
    if not args.output or args.timeout_seconds <= 0:
        raise ValueError("--output and a positive timeout are required")
    resources = None if args.prepare_only else compiler_resources()
    board_requested = not (args.prepare_only or args.compile_only)
    if board_requested:
        from .hdc_run import _text
        for key in ("SN", "BIP"):
            if _text(os.environ.get(key), key).startswith("-"):
                raise ValueError(f"invalid {key}")
        _required_path("HIAI_BUILD_DIR")
    root = output_path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "artifact_kind": "qwen35-cpro-op-probe",
                "created_utc": datetime.now(timezone.utc).isoformat(), "cases": [],
                "numpy_version": np.__version__, "status": "PREPARING", "board_requested": board_requested,
                "scope": "These shapes/dtypes/attributes through current GE lowering only; not universal support or full-model validation."}
    manifest["selection"] = {"suite": args.suite, "requested_cases": list(args.case),
                             "excluded_families": list(args.exclude_family),
                             "selected_cases": [c["name"] for c in selected]}
    package = Path(__file__).resolve().parent
    implementation = set(package.glob("ge*.py")) | set(package.glob("op_probe*.py"))
    implementation.update(Path(module.__file__) for name, module in list(sys.modules.items())
                          if name.startswith(__package__ + ".") and getattr(module, "__file__", None)
                          and (name.rsplit(".", 1)[-1].startswith(("ge", "op_probe"))
                               or name.rsplit(".", 1)[-1] in {"compile", "compiler_evidence", "hdc_run", "hiai_deployment"}))
    manifest["implementation_sha256"] = {path.name: sha256_file(path) for path in sorted(implementation)}
    if resources:
        ge, ddk, _, omg, _ = resources
        manifest["tools"] = {"ge_root": str(ge), "ddk_root": str(ddk), "ddk_id": os.environ.get("CPRO_DDK_ID"),
                             "omg_sha256": sha256_file(omg), "libgraph_sha256": sha256_file(ge / "lib64/libgraph.so")}
    board = None
    try:
        for i, case in enumerate(selected, 1):
            print(f"[{i}/{len(selected)}] {case['name']}", flush=True)
            record = prepare_case(case, root)
            manifest["cases"].append(record)
            if resources:
                result = compile_case(record, resources, args.timeout_seconds)
                print(f"  {result['status']} ({result['stage']})", flush=True)
        compiled = [r for r in manifest["cases"] if r["compile"]["status"] == "COMPILED"]
        manifest["status"] = "PREPARED" if args.prepare_only else "COMPILE_COMPLETE"
        if board_requested and compiled:
            runner, libraries = native_build(root, args.timeout_seconds)
            manifest["native"] = {"build": json.loads((root / "native/command.json").read_text()),
                                  "result": json.loads((root / "native/result.json").read_text())}
            for record in compiled:
                for item in record["construction_files"]:
                    checked_file(Path(record["directory"]), item)
                checked_file(Path(record["directory"]), record["compile"]["model"])
            from .op_probe_hdc import run_board
            board = run_board(root, runner, libraries, compiled,
                              hdc_executable=os.environ.get("HDC", "hdc"),
                              timeout=args.timeout_seconds, keep_board=args.keep_board)
            manifest["status"] = "COLLECTED"
        elif resources and not compiled:
            manifest["status"] = "NO_COMPILED_CASES"
    except Exception as exc:
        manifest.update(status="INFRASTRUCTURE_FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save_json(root / "manifest.json", manifest)
        write_report(root, manifest, board)
        print(f"Probe report: {root / 'report.txt'}", flush=True)
        print(f"Share results: {root / 'results.tar.gz'}", flush=True)
    return 1 if manifest["status"] == "NO_COMPILED_CASES" else 0


def main(argv=None):
    try:
        return run(parser().parse_args(argv))
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"probe-hdc-ops: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
