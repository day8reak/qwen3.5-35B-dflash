"""Portable resource resolution from project.yaml -> specs/resources.json."""
import json
import os
from pathlib import Path
import re


def repository_root():
    return Path(__file__).resolve().parents[2]


def load_resources(path=None):
    project = json.loads((repository_root() / "project.yaml").read_text())
    declared = project["resource_manifest"]
    path = Path(path or os.environ.get("QWEN35_RESOURCES", repository_root() / declared)).resolve()
    data = json.loads(path.read_text())
    if data.get("schema_version") != 1:
        raise ValueError("unsupported resource schema")
    return data, path


def resource_path(resources, key, *, directory=False):
    value = resources.get(key)
    if not value:
        raise ValueError(f"missing resource {key}; set QWEN35_RESOURCES to a local JSON manifest")
    path_text = os.path.expandvars(os.path.expanduser(value))
    if re.search(r"\$[A-Za-z_{]", path_text):
        raise ValueError(f"unresolved resource {key}: {value}")
    path = Path(path_text).resolve()
    if not (path.is_dir() if directory else path.is_file()):
        raise FileNotFoundError(f"{key}: {path}")
    return path


def dopt_resource(resources):
    if resources.get("dopt_root"):
        return resource_path(resources, "dopt_root", directory=True)
    root = resource_path(resources, "ddk_root", directory=True) / "tools/tools_dopt/dopt_pytorch_py3"
    if not root.is_dir():
        raise FileNotFoundError("set dopt_root in the resource manifest or use DDK with tools_dopt")
    return root
