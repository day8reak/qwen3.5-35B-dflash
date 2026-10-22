"""Hash-locked artifact paths and per-run output ownership."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
from .hashing import sha256_file


def output_path(path):
    run = os.environ.get("AI_RUN_DIR")
    if not run:
        raise RuntimeError("AI_RUN_DIR is required; use ws session start (or a dedicated run directory)")
    root, path = Path(run).resolve(), Path(path).resolve()
    if path == root or root not in path.parents:
        raise ValueError("generated output must be below AI_RUN_DIR")
    return path


def save_json(path, value):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # No overwritten evidence; a new output path is needed after a failed run.
    with path.open("x") as out:
        json.dump(value, out, indent=2, ensure_ascii=False, allow_nan=False)
        out.write("\n")


def files_under(folder, root):
    records = []
    for path in sorted(folder.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"artifact payload must not be a symlink: {path}")
        if path.is_file():
            records.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                            "sha256": sha256_file(path)})
    return records


def checked_file(root, record):
    path = (root / record["path"]).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise ValueError(f"artifact missing or escapes bundle: {record['path']}")
    if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
        raise ValueError(f"artifact SHA256/size mismatch: {record['path']}")
    return path


@contextmanager
def working_directory(path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)
