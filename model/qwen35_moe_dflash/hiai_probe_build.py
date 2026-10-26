"""Explicitly refresh only the native diagnostic probe from verified build inputs."""
import os
from pathlib import Path
import re

from .artifacts import save_json
from .elf_dynamic import dynamic_info
from .ge_export import _logged
from .hashing import sha256_file
from .resources import repository_root


def _hash(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"refresh requires recorded {label} SHA256")
    return value


def build_probe(root, validated, libraries, frozen, timeout):
    """Build one new runner; never discover libraries or compile model graphs."""
    original, snapshot = validated["command"], validated["snapshot"]
    allowed = {"CMAKE_TOOLCHAIN_FILE", "OHOS_ARCH", "OHOS_PLATFORM", "OHOS_STL",
               "OHOS_COMPATIBLE_SDK_VERSION", "HIAI_INCLUDE_DIR"}
    settings = {}
    argv = original.get("argv")
    if not isinstance(argv, list) or not all(isinstance(arg, str) for arg in argv):
        raise ValueError("refresh requires recorded CMake argv")
    for arg in argv:
        if arg.startswith("-D") and "=" in arg:
            key, value = arg[2:].split("=", 1)
            if key in allowed:
                if key in settings and settings[key] != value:
                    raise ValueError(f"conflicting recorded CMake setting: {key}")
                settings[key] = value
    if (settings.get("OHOS_ARCH") != "arm64-v8a" or settings.get("OHOS_PLATFORM") != "OHOS"
            or settings.get("OHOS_STL") != "c++_shared"):
        raise ValueError("refresh requires recorded arm64-v8a OHOS c++_shared settings")
    if ("OHOS_COMPATIBLE_SDK_VERSION" in settings
            and not re.fullmatch(r"[1-9][0-9]*", settings["OHOS_COMPATIBLE_SDK_VERSION"])):
        raise ValueError("recorded OHOS_COMPATIBLE_SDK_VERSION must be a positive integer")
    for key in ("CMAKE_TOOLCHAIN_FILE", "HIAI_INCLUDE_DIR"):
        value = settings.get(key, "")
        if not Path(value).is_absolute() or any(ord(c) < 32 or ord(c) == 127 or c == ";" for c in value):
            raise ValueError(f"refresh requires an absolute recorded {key}")
    toolchain = Path(settings["CMAKE_TOOLCHAIN_FILE"])
    header = Path(settings["HIAI_INCLUDE_DIR"]) / "HiAiModelManagerService.h"
    snapshot.add(toolchain, expected=_hash(original.get("sdk_toolchain_sha256"), "toolchain"), role="refresh-toolchain")
    snapshot.add(header, expected=_hash(original.get("header_sha256"), "header"), role="refresh-header")
    source = repository_root() / "targets/cpro/runtime"
    sources = [snapshot.add(source / name, role="refresh-source")
               for name in ("CMakeLists.txt", "src/hiai_op_probe.cpp")]
    deployed = {path.name: path for path in libraries}
    linked = []
    for entry in original["libraries"]:
        old = snapshot.add(entry["path"], expected=_hash(entry["sha256"], "link input"), role="link-input")
        path = deployed.get(old.name)
        if path is None:
            raise ValueError(f"frozen runtime set lacks recorded link input: {old.name}")
        frozen.add(path, expected=entry["sha256"], role="refresh-link-input")
        linked.append({"path": str(path), "original_path": str(old), "sha256": entry["sha256"]})
    snapshot.verify()
    frozen.verify()
    folder = Path(root) / "refreshed-native"
    folder.mkdir(exist_ok=False)
    command = ["cmake", "-S", str(source), "-B", str(folder / "build")]
    command += [f"-D{key}={value}" for key, value in settings.items()]
    command += ["-DBUILD_HOST_TESTS=OFF", "-DBUILD_HIAI_RUNNER=ON", "-DBUILD_HIAI_OBJECTS=OFF",
                "-DHIAI_LIBRARIES=" + ";".join(item["path"] for item in linked)]
    build_command = ["cmake", "--build", str(folder / "build"), "--target", "qwen35_hiai_op_probe", "-j", "4"]
    provenance = {"argv": command, "libraries": linked,
        "sources": [snapshot.files[str(path)] for path in sources],
        "header_sha256": sha256_file(header), "sdk_toolchain_sha256": sha256_file(toolchain),
        "source_command": str(validated["source"] / "native/command.json"),
        "source_command_sha256": sha256_file(validated["source"] / "native/command.json"),
        "historical_binary": {"path": str(validated["runner"]), "sha256": sha256_file(validated["runner"])},
        "scope": "Explicit native diagnostic runner refresh only; existing model artifacts and runtime library inventory are reused."}
    save_json(folder / "command.json", provenance)
    save_json(folder / "build-command.json", {"argv": build_command})
    for name in ("command.json", "build-command.json"):
        frozen.add(folder / name, role="refresh-build-metadata")
    stage = "configure"
    try:
        _logged(command, folder=folder, env=os.environ.copy(), log=folder / "configure.log", timeout=timeout)
        snapshot.verify()
        frozen.verify()
        stage = "build"
        _logged(build_command, folder=folder, env=os.environ.copy(), log=folder / "build.log", timeout=timeout)
        snapshot.verify()
        frozen.verify()
        stage = "validate-binary"
        runner = folder / "build/qwen35_hiai_op_probe"
        if not runner.is_file() or runner.is_symlink() or not runner.stat().st_size:
            raise ValueError("native refresh produced no nonempty regular runner")
        digest = sha256_file(runner)
        dependencies = dynamic_info(runner)
        frozen.add(runner, expected=digest, role="refreshed-native-runner")
        snapshot.verify()
        frozen.verify()
        result = {"status": "LINKED", "binary_sha256": digest,
                  "command_sha256": sha256_file(folder / "command.json"),
                  "build_command_sha256": sha256_file(folder / "build-command.json"),
                  "historical_binary_sha256": provenance["historical_binary"]["sha256"],
                  "dependencies": dependencies, "device_validation": "NOT_RUN"}
        save_json(folder / "result.json", result)
        frozen.add(folder / "result.json", role="refresh-build-metadata")
        return runner, {"mode": "REFRESHED_NATIVE", "path": str(runner), "sha256": digest,
                        "historical_binary": provenance["historical_binary"],
                        "build_command": provenance, "build_result": result,
                        "dependencies": dependencies}
    except Exception as error:
        save_json(folder / "failure.json", {"status": "FAILED", "stage": stage, "board_started": False,
                  "error": f"{type(error).__name__}: {error}"})
        raise
