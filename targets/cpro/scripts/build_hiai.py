#!/usr/bin/env python3
"""Build with a user-selected HarmonyOS SDK and real HIAI libraries."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"model"))
from qwen35_moe_dflash.artifacts import output_path, save_json
from qwen35_moe_dflash.resources import load_resources, resource_path
from qwen35_moe_dflash.hashing import sha256_file


def positive_api(value):
    try:
        api = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("compatible SDK version must be a positive integer") from None
    if api <= 0:
        raise argparse.ArgumentTypeError("compatible SDK version must be a positive integer")
    return api


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resources")
    parser.add_argument("--output", required=True)
    parser.add_argument("--headers-only", action="store_true")
    parser.add_argument("--compatible-sdk-version", type=positive_api,
        help="optional minimum API; omitted means use the selected SDK toolchain default")
    args = parser.parse_args()
    resources, _ = load_resources(args.resources)
    sdk = resource_path(resources, "ohos_native_sdk", directory=True)
    toolchain = sdk / "build/cmake/ohos.toolchain.cmake"
    compiler = sdk / "llvm/bin/clang++"
    if not toolchain.is_file() or not compiler.is_file() or not (sdk / "sysroot").is_dir():
        raise ValueError(f"incomplete OHOS Native SDK: {sdk}; OHOS_NATIVE_SDK must point to "
            "native/ containing build/cmake/ohos.toolchain.cmake, llvm/bin/clang++, and sysroot/ "
            "(not command-line-tools/ or libs/ohos/)")
    if not os.access(compiler, os.X_OK):
        raise ValueError(f"OHOS Native compiler is not executable: {compiler}; use the matching host SDK package")
    include = resource_path(resources, "hiai_include_dir", directory=True)
    libraries = resources.get("hiai_libraries", [])
    if not libraries and os.environ.get("HIAI_LIBRARIES"):
        libraries = os.environ["HIAI_LIBRARIES"].split(os.pathsep)
    libraries = [str(Path(os.path.expandvars(p)).resolve()) for p in libraries]
    if not args.headers_only and (not libraries or not all(Path(p).is_file() for p in libraries)):
        raise ValueError("supply internal HarmonyOS ARM64 HIAI libraries through hiai_libraries or HIAI_LIBRARIES")
    out = output_path(args.output); out.mkdir(parents=True, exist_ok=False)
    command = ["cmake", "-S", str(ROOT/"targets/cpro/runtime"), "-B", str(out/"build"),
        "-DCMAKE_TOOLCHAIN_FILE="+str(toolchain),
        "-DOHOS_ARCH=arm64-v8a", "-DOHOS_PLATFORM=OHOS", "-DOHOS_STL=c++_shared",
        "-DBUILD_HOST_TESTS=OFF", "-DHIAI_INCLUDE_DIR="+str(include),
        "-DBUILD_HIAI_OBJECTS=ON", "-DBUILD_HIAI_RUNNER="+("OFF" if args.headers_only else "ON"),
        "-DHIAI_LIBRARIES="+";".join(libraries)]
    if args.compatible_sdk_version is not None:
        command.append("-DOHOS_COMPATIBLE_SDK_VERSION="+str(args.compatible_sdk_version))
    save_json(out/"command.json", {"argv": command, "headers_only": args.headers_only,
        "compatible_sdk_version_requested": args.compatible_sdk_version,
        "sdk_api_policy": "explicit" if args.compatible_sdk_version is not None else "sdk_default",
        "sdk_toolchain_sha256": sha256_file(toolchain),
        "header_sha256": sha256_file(include/"HiAiModelManagerService.h"),
        "libraries": [{"path": p, "sha256": sha256_file(Path(p))} for p in libraries]})
    with (out/"build.log").open("w") as log:
        subprocess.run(command, cwd=out, stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(["cmake", "--build", str(out/"build"), "-j", "4"], cwd=out, stdout=log, stderr=subprocess.STDOUT, check=True)
    binary = out/"build/qwen35_dflash_hiai"
    save_json(out/"result.json", {"status": "HEADERS_COMPILED" if args.headers_only else "LINKED",
        "device_validation": "PENDING", "binary_sha256": sha256_file(binary) if binary.is_file() else None})
    print(out)


if __name__ == "__main__":
    main()
