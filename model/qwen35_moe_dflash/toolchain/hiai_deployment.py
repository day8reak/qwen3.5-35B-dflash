"""Discover board libraries from the selected runner's own build records.

This is a bounded packaging helper, not an ELF dependency resolver. No board
system libraries, loader configuration, or shared DDK contents are modified.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import struct

from qwen35_moe_dflash.common.hashing import sha256_file


DDK_HCL_EXCLUDED = frozenset(("libhiai_hcl_model_runtime.so", "libhiai_hcl_model_runtime_impl.so"))
BOARD_LIBRARY_SEARCH_PATHS = ("/vendor/lib64/chipsetsdk",)


def candidate_blockers(dependencies, *, require_pair=True):
    """Guard the private-library exclusion shared by probe and model deployment.

    The A/B probe requires both original providers to establish its baseline.
    Production can start with an already reduced library set, but retained
    DT_NEEDED entries and aliases must still not refer to the excluded pair.
    This checks explicit ELF dependencies, not dlopen or board ABI compatibility.
    """
    libraries = [item for item in dependencies if item["kind"] == "library"]
    excluded = [item for item in libraries if item["name"] in DDK_HCL_EXCLUDED]
    missing = DDK_HCL_EXCLUDED - {item["name"] for item in excluded}
    if require_pair and missing:
        return ["original deployment lacks the exact HCL exclusion pair: " + ", ".join(sorted(missing))]
    aliases = set(DDK_HCL_EXCLUDED) | {item["soname"] for item in excluded if item.get("soname")}
    aliases.update(Path(name).name for name in tuple(aliases))
    excluded_hashes = {item["sha256"] for item in excluded}
    reasons = []
    for item in dependencies:
        if item["kind"] == "library" and item["name"] in DDK_HCL_EXCLUDED:
            continue
        required = [name for name in item["needed"] if name in aliases or Path(name).name in aliases]
        if required:
            reasons.append(f"{item['name']} has DT_NEEDED on excluded provider(s): {', '.join(required)}")
        soname = item.get("soname")
        if item["kind"] == "library" and soname and (soname in aliases or Path(soname).name in aliases):
            reasons.append(f"retained {item['name']} has colliding SONAME {item['soname']}")
        if item["kind"] == "library" and item["sha256"] in excluded_hashes:
            reasons.append(f"retained {item['name']} duplicates excluded provider bytes")
    return reasons


def _absolute_path(value, label):
    if (not isinstance(value, str) or not value or not Path(value).is_absolute()
            or any(ord(c) < 32 or ord(c) == 127 for c in value)):
        raise ValueError(f"{label} must be an absolute path without control characters")
    return Path(value)


def _elf(path, *, executable=False):
    with path.open("rb") as stream:
        header = stream.read(64)
    if (len(header) < 64 or header[:7] != b"\x7fELF\x02\x01\x01"
            or struct.unpack_from("<H", header, 18)[0] != 183
            or struct.unpack_from("<H", header, 16)[0] not in ((2, 3) if executable else (3,))):
        kind = "executable" if executable else "shared library"
        raise ValueError(f"expected ELF64 little-endian AArch64 {kind}: {path}")


def _build_directory(runner, explicit):
    if explicit is not None:
        folder = Path(explicit).expanduser().resolve()
        if not all((folder / name).is_file() for name in ("result.json", "command.json")):
            raise ValueError(f"HIAI build directory needs result.json and command.json: {folder}")
        return folder
    for folder in (runner.parent.parent, runner.parent):
        if all((folder / name).is_file() for name in ("result.json", "command.json")):
            return folder
    raise ValueError("cannot discover HIAI libraries: keep the runner at its build_hiai.py "
                     "output location with adjacent result.json and command.json")


def _sdk_cxx(command):
    """Search only SDK locations derived from this build's toolchain argument."""
    prefix = "-DCMAKE_TOOLCHAIN_FILE="
    argv = command.get("argv", [])
    if not isinstance(argv, list) or not all(isinstance(arg, str) for arg in argv):
        raise ValueError("HIAI build command argv must be a list of strings")
    toolchains = list(dict.fromkeys(arg[len(prefix):] for arg in argv if arg.startswith(prefix)))
    if len(toolchains) > 1:
        raise ValueError("HIAI build command contains conflicting SDK toolchain paths")
    info = {"status": "not_found", "searched": [], "rejected": []}
    if not toolchains:
        info["reason"] = "build command has no OHOS SDK toolchain path"
        return None, info
    toolchain = _absolute_path(toolchains[0], "SDK toolchain path")
    if toolchain.parts[-3:] != ("build", "cmake", "ohos.toolchain.cmake"):
        info["reason"] = "build toolchain path does not identify a Native SDK layout"
        return None, info
    sdk = toolchain.parents[2]
    info["sdk"] = str(sdk)
    standard = [sdk / "llvm/lib/aarch64-linux-ohos/libc++_shared.so",
                sdk / "sysroot/usr/lib/aarch64-linux-ohos/libc++_shared.so"]
    patterns = ("llvm/lib/*/libc++_shared.so", "llvm/lib/*/c++/libc++_shared.so",
                "llvm/lib/clang/*/lib/*/libc++_shared.so",
                "sysroot/usr/lib/*/libc++_shared.so")
    seen = set()
    for candidates in (standard, None):
        if candidates is None:
            candidates = sorted({path for pattern in patterns for path in sdk.glob(pattern)})
        valid = []
        for path in candidates:
            if path in seen:
                continue
            seen.add(path)
            info["searched"].append(str(path))
            if not path.is_file():
                continue
            try:
                _elf(path)
            except ValueError as exc:
                info["rejected"].append({"path": str(path), "reason": str(exc)})
                continue
            valid.append((path, sha256_file(path)))
        if valid:
            if len({digest for _, digest in valid}) != 1:
                raise ValueError("conflicting AArch64 libc++_shared.so files in recorded SDK: "
                                 + ", ".join(str(path) for path, _ in valid))
            info.update(status="sdk", path=str(valid[0][0]), sha256=valid[0][1])
            return valid[0][0], info
    info["reason"] = "no AArch64 libc++_shared.so found in bounded SDK locations; board availability is unverified"
    return None, info


def discover_libraries(runner: Path, build_dir: Path | None = None) -> dict:
    """Return library paths, their adjacent build directory, and audit metadata.

    Library paths retain symlink basenames for deployment. The recorded link
    inputs are checked before their immediate siblings are considered. Legacy
    records without command_sha256 are accepted and explicitly identified.
    """
    runner = Path(runner).expanduser().resolve()
    _elf(runner, executable=True)
    folder = _build_directory(runner, build_dir)
    result_bytes = (folder / "result.json").read_bytes()
    command_bytes = (folder / "command.json").read_bytes()
    result, command = json.loads(result_bytes), json.loads(command_bytes)
    if not isinstance(result, dict) or not isinstance(command, dict):
        raise ValueError("HIAI build result.json and command.json must be JSON objects")
    if result.get("status") != "LINKED":
        raise ValueError("automatic HIAI library discovery requires a LINKED build result")
    if result.get("binary_sha256") != sha256_file(runner):
        raise ValueError(f"HIAI build runner SHA256 mismatch: {runner}")
    command_hash = hashlib.sha256(command_bytes).hexdigest()
    if "command_sha256" in result and result["command_sha256"] != command_hash:
        raise ValueError(f"HIAI build command SHA256 mismatch: {folder / 'command.json'}")
    recorded = command.get("libraries")
    if not isinstance(recorded, list) or not recorded:
        raise ValueError("HIAI build command.json needs nonempty recorded link libraries")
    directories = []
    selected = {}

    def add(path, origin, expected=None):
        if not path.is_file():
            raise ValueError(f"HIAI runtime library is missing or has a broken symlink: {path}")
        _elf(path)
        digest = sha256_file(path)
        if expected is not None and digest != expected:
            raise ValueError(f"HIAI library SHA256 mismatch with recorded link input: {path}")
        old = selected.get(path.name)
        if old and old[1]["sha256"] != digest:
            raise ValueError(f"conflicting library name {path.name}: {old[0]} and {path}")
        if not old:
            selected[path.name] = (path, {"name": path.name, "path": str(path),
                                         "sha256": digest, "origin": origin})

    for entry in recorded:
        if not isinstance(entry, dict):
            raise ValueError("recorded HIAI link library must be a JSON object")
        path = _absolute_path(entry.get("path"), "recorded link-input path")
        digest = entry.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"invalid recorded link-input SHA256: {path}")
        add(path, "recorded-link-input", digest)
        if path.parent not in directories:
            directories.append(path.parent)
    for directory in directories:
        for path in sorted(directory.glob("*.so*")):
            add(path, "link-library-directory")
    if "libc++_shared.so" in selected:
        cxx = {"status": "ddk", "path": str(selected["libc++_shared.so"][0])}
    else:
        path, cxx = _sdk_cxx(command)
        if path is not None:
            add(path, "build-sdk")
    return {"build_dir": folder, "libraries": [path for path, _ in selected.values()],
            "metadata": {"mode": "build-records", "build_dir": str(folder),
                         "command_sha256": command_hash,
                         "command_binding": "sha256-verified" if "command_sha256" in result else "legacy-unbound",
                         "library_directories": [str(path) for path in directories],
                         "libraries": [item for _, item in selected.values()], "cxx_runtime": cxx,
                         "scope": "Recorded link inputs and their immediate shared-library siblings, with an optional matching SDK C++ runtime. ELF architecture is checked; DT_NEEDED closure and board ABI compatibility are not verified."}}
