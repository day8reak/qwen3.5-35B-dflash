"""Read selected HIAI headers and ELF symbols; never load a runtime or a model."""
from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess

from .draft_workspace import DraftWorkspace
from .hashing import sha256_file


API_PATTERN = re.compile(
    r"RestoreFromFile|RestoreFromBuffer|CreateBuiltModel|CreateModelManager|"
    r"\bI(?:BuiltModel|ModelManager)\b|ModelInitOptions|ModelBuildOptions|ExternalWeight|ModelWeight|"
    r"(?:Set|Get|Create|Load|Restore)\w*Weight|Weight\w*(?:Path|File|Buffer)|"
    r"HIAI_\w*(?:BuiltModel|ModelManager)\w*(?:Create|Restore|Init)", re.I)
MAX_TEXT = 2 << 20
API_HEADERS = {"built_model.h", "model_manager.h", "model_manager_types.h", "model_init_options.h",
               "hiai_built_model.h", "hiai_model_manager.h", "hiai_model_manager_types.h"}


def _json(path):
    if not path.is_file() or path.stat().st_size > MAX_TEXT:
        raise ValueError("missing or oversized metadata: " + str(path))
    return json.loads(path.read_text())


def _argument(argv, prefix):
    values = [value[len(prefix):] for value in argv if value.startswith(prefix)]
    if len(values) != 1 or not Path(values[0]).is_absolute():
        raise ValueError("expected one absolute recorded " + prefix)
    return Path(values[0])


def inspect_load_api(job, *, run_command=subprocess.run):
    job = Path(job).absolute()
    with DraftWorkspace(job):
        out = job / "load-diagnostic"
        if out.is_symlink() or _json(out / ".owner.json") != {"artifact_kind": "qwen35-hiai-descriptor-diagnostic"}:
            raise ValueError("expected the completed descriptor diagnostic directory")
        for name in ("api-report.json", "api-report.txt"):
            path = out / name
            if path.is_symlink() or (path.exists() and not path.is_file()):
                raise ValueError("API report must be a regular file: " + str(path))
        observed = _json(out / "result.json")
        if observed.get("status") != "INSPECTED":
            raise ValueError("inspect API after a completed descriptor observation")
        build = out / "native"
        command_path = build / "command.json"
        command, result = _json(command_path), _json(build / "result.json")
        digest = sha256_file(command_path)
        if result.get("status") != "LINKED" or result.get("command_sha256") != digest:
            raise ValueError("native build command hash mismatch")
        discovery = observed.get("libraries", {}).get("discovery", {})
        if discovery.get("command_sha256") != digest:
            raise ValueError("API inspection requires the diagnostic's recorded library discovery")
        include = _argument(command["argv"], "-DHIAI_INCLUDE_DIR=")
        toolchain = _argument(command["argv"], "-DCMAKE_TOOLCHAIN_FILE=")
        if toolchain.parts[-3:] != ("build", "cmake", "ohos.toolchain.cmake"):
            raise ValueError("recorded toolchain does not identify an OHOS Native SDK")
        nm = toolchain.parents[2] / "llvm/bin/llvm-nm"
        if sha256_file(include / "HiAiModelManagerService.h") != command.get("header_sha256"):
            raise ValueError("selected HIAI header changed since descriptor probe")
        if sha256_file(toolchain) != command.get("sdk_toolchain_sha256"):
            raise ValueError("selected SDK toolchain changed since descriptor probe")
        report = {"schema_version": 1, "artifact_kind": "qwen35-hiai-load-api-inspection",
                  "status": "INSPECTED", "build_command_sha256": digest,
                  "descriptor_report_sha256": sha256_file(out / "result.json"),
                  "runtime_version_observed": observed.get("runtime_version"),
                  "include": str(include), "headers": [], "symbols": [], "errors": [],
                  "scope": "Header declarations and defined ELF symbols only. No HDC, runtime loading, model compilation, Load or Process; file-restore external-weight support remains unverified."}
        scanned_bytes = 0
        headers = sorted(include.rglob("*.h"))
        if len(headers) > 4096:
            raise ValueError("selected include directory exceeds header scan bound")
        for path in headers:
            # Stay within the declared include root, including symlink targets.
            if include.resolve() not in path.resolve().parents or not path.is_file():
                continue
            if path.stat().st_size > MAX_TEXT:
                continue
            raw = path.read_bytes()
            scanned_bytes += len(raw)
            if scanned_bytes > 32 << 20:
                raise ValueError("selected headers exceed total scan byte bound")
            lines = raw.decode("utf-8", errors="replace").splitlines()
            hits = [i for i, line in enumerate(lines) if API_PATTERN.search(line)]
            if hits or path.name.lower() in API_HEADERS:
                selected = (list(range(len(lines))) if path.name.lower() in API_HEADERS else
                            sorted({j for i in hits for j in range(max(0, i - 2), min(len(lines), i + 7))}))
                report["headers"].append({"path": str(path), "sha256": sha256_file(path),
                    "matching_lines": len(hits), "truncated": len(selected) > 320,
                    "lines": [{"line": j + 1, "text": lines[j][:2000]} for j in selected[:320]]})
        libraries = discovery.get("libraries", [])
        if not libraries or len(libraries) > 64:
            raise ValueError("expected bounded recorded runtime libraries")
        for row in libraries:
            if not row["name"].startswith("libhiai"):
                continue
            path = Path(row["path"])
            item = {"path": str(path), "sha256": row["sha256"], "lines": []}
            report["symbols"].append(item)
            try:
                if sha256_file(path) != row["sha256"]:
                    raise ValueError("library changed since descriptor probe")
                argv = [str(nm), "--dynamic", "--defined-only", "--demangle", str(path)]
                execution = run_command(argv, capture_output=True, text=True, errors="replace", timeout=30, check=False)
                if execution.returncode != 0:
                    raise RuntimeError(execution.stderr[:2000])
                if len(execution.stdout) > 8 << 20:
                    raise ValueError("symbol output exceeds byte bound")
                selected = [line for line in execution.stdout.splitlines() if API_PATTERN.search(line)]
                item.update(status="READ", lines=selected[:160], matching_lines=len(selected), truncated=len(selected) > 160)
            except Exception as error:
                item.update(status="FAILED", error=f"{type(error).__name__}: {error}")
                report["errors"].append({"path": str(path), "error": item["error"]})
        if report["errors"]:
            report["status"] = "INCOMPLETE"
        rendered = ["Selected HIAI load API inspection: " + report["status"], report["scope"],
                    "Recorded runtime: " + str(report["runtime_version_observed"])]
        for row in report["headers"]:
            rendered.append("\nHEADER " + row["path"])
            rendered.extend(f"L{line['line']}: {line['text']}" for line in row["lines"])
            if row["truncated"]:
                rendered.append("[header context truncated]")
        if not report["headers"]:
            rendered.append("No matching API declarations in the selected include directory.")
        for row in report["symbols"]:
            rendered.append("\nLIBRARY " + row["path"])
            rendered.extend(row["lines"])
            if row.get("error"):
                rendered.append(row["error"])
            elif not row["lines"]:
                rendered.append("No matching defined dynamic symbols.")
            if row.get("truncated"):
                rendered.append("[symbol matches truncated]")
        # Replace only these two named API reports; preserve the device probe
        # and its manifests, runner build, model stages and existing logs.
        with (out / "api-report.json").open("w") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        text = "\n".join(rendered) + "\n"
        (out / "api-report.txt").write_text(text)
        return report, text
