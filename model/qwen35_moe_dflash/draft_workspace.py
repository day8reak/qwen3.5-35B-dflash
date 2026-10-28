"""One owned, resumable draft workspace; payload manifests remain immutable.

Only the standalone draft CLI uses this exception to per-run output retention.
Moving a complete stage preserves its relative payload paths and manifest hashes.
The journal records moves before they happen so a retry need not find the old
absolute component directory. Failed legacy compiler recovery must happen before
adoption: its original command paths are deliberately not rewritten here.
"""
from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
import errno
import fcntl
import json
import os
from pathlib import Path
import shutil
import stat
import uuid

from .artifacts import output_path
from .hashing import sha256_file


STAGES = {
    "prepared": ("prepared.json", "qwen35-draft-ge-prepared", "PREPARED"),
    "fixture": ("report.json", "qwen35-draft-reference", "REFERENCE_COMPLETE"),
    "air": ("manifest.json", "qwen35-cpro-draft-air", "EXPORTED"),
    "compilation": ("report.json", "qwen35-existing-air-compilation", "COMPILED"),
    "bundle": ("manifest.json", "qwen35-standalone-draft-bundle", "PREPARED"),
}
JOB_KIND = "qwen35-full-draft-job"
OWNER = ".draft-workspace.json"
LOCK = ".draft-workspace.lock"
JOURNAL = ".draft-migration.json"
_METADATA_LIMIT = 16 * 1024 * 1024


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _read_json(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > _METADATA_LIMIT:
        raise ValueError(f"invalid draft workspace metadata: {path}")
    return json.loads(path.read_text())


def _state(value):
    if (not isinstance(value, dict) or value.get("artifact_kind") != JOB_KIND
            or not isinstance(value.get("components"), dict)
            or set(value["components"]) - STAGES.keys()):
        raise ValueError("expected a standalone full-draft job with known component stages")
    return deepcopy(value)


class DraftWorkspace:
    """Hold the workspace lock while adopting, resetting and running its stages.

    The stable lock inode is never removed. CLOEXEC prevents a normal executed
    child from retaining it after the parent exits; this is not a claim that an
    unrecorded orphan compiler is stopped after a hard parent-process crash.
    """

    def __init__(self, root):
        if Path(root).is_symlink():
            raise ValueError("draft workspace must not be a symlink")
        self.root = output_path(root)
        self._fd = None
        self._owner = None

    def __enter__(self):
        if self._fd is not None:
            raise RuntimeError("draft workspace lock is already held")
        if self.root.exists():
            # Reject unrelated directories before even creating our lock file.
            self._claimable()
        self.root.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.root / LOCK, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError("invalid draft workspace lock file")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"draft workspace is locked by another process: {self.root}") from error
            self._fd = fd
            owner = self._claimable()
            if owner is None:
                owner = {"schema_version": 1, "artifact_kind": "qwen35-draft-workspace",
                         "root": str(self.root), "owner_id": uuid.uuid4().hex}
                self._write(OWNER, owner)
            self._owner = owner
            return self
        except BaseException:
            self._fd = None
            os.close(fd)
            raise

    def _claimable(self):
        if not self.root.is_dir() or self.root.is_symlink():
            raise ValueError("draft workspace must be a real directory")
        marker = self.root / OWNER
        if marker.exists() or marker.is_symlink():
            owner = _read_json(marker)
            if (owner.get("artifact_kind") != "qwen35-draft-workspace"
                    or owner.get("root") != str(self.root)
                    or not isinstance(owner.get("owner_id"), str) or not owner["owner_id"]):
                raise ValueError("draft workspace ownership marker is invalid")
            return owner
        entries = {path.name for path in self.root.iterdir()} - {LOCK}
        recognized = False
        for name in ("job.json", "result.json"):
            if (self.root / name).exists():
                _state(_read_json(self.root / name))
                recognized = True
        if entries and not recognized:
            raise ValueError("refusing to claim a nonempty directory without a full-draft job")
        return None

    def __exit__(self, *_):
        if self._fd is not None:
            fd, self._fd = self._fd, None
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _owned(self):
        if self._fd is None or self._owner is None:
            raise RuntimeError("draft workspace must be locked before mutation")
        if _read_json(self.root / OWNER) != self._owner:
            raise ValueError("draft workspace ownership changed")

    def _write(self, name, value):
        path = self.root / name
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError(f"refusing to replace nonregular workspace metadata: {name}")
        data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
        if len(data) > _METADATA_LIMIT:
            raise ValueError("draft workspace metadata is too large")
        temporary = self.root / (".draft-metadata-" + uuid.uuid4().hex + ".tmp")
        try:
            with temporary.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            _sync_directory(self.root)
        finally:
            temporary.unlink(missing_ok=True)

    def save(self, name, value):
        """Atomically replace only a known top-level job or stage receipt."""
        self._owned()
        allowed = {"job.json", "result.json", "comparison-review.json"} | {f"completed-{stage}.json" for stage in STAGES}
        if name not in allowed:
            raise ValueError("unknown draft workspace metadata name")
        self._write(name, value)

    def _unlink(self, name):
        path = self.root / name
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError(f"refusing to remove nonregular workspace metadata: {name}")
        path.unlink(missing_ok=True)
        _sync_directory(self.root)

    def _record_path(self, name, record, *, directory=None):
        if not isinstance(record, dict) or not isinstance(record.get("directory"), str):
            raise ValueError(f"invalid {name} component record")
        raw = Path(record["directory"] if directory is None else directory)
        if not raw.is_absolute() or raw.is_symlink():
            raise ValueError(f"component directory must be absolute and not a symlink: {name}")
        folder = output_path(raw)
        destination = self.root / name
        if folder != destination and (folder == self.root or self.root in folder.parents or folder in self.root.parents):
            raise ValueError(f"component directory overlaps the workspace: {name}")
        return folder

    def _verify_record(self, name, record, *, directory=None):
        folder = self._record_path(name, record, directory=directory)
        item = record.get("manifest", {})
        filename, kind, status = STAGES[name]
        if (not isinstance(item, dict) or item.get("path") != filename
                or type(item.get("bytes")) is not int or not 0 < item["bytes"] <= _METADATA_LIMIT
                or not isinstance(item.get("sha256"), str)):
            raise ValueError(f"invalid {name} manifest receipt")
        path = folder / filename
        if (not folder.is_dir() or path.is_symlink() or not path.is_file()
                or path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]):
            raise ValueError(f"{name} manifest SHA256/size mismatch")
        payload = _read_json(path)
        if payload.get("artifact_kind") != kind or payload.get("status") != status:
            raise ValueError(f"{name} is not a completed draft component")
        return folder

    def load_pending(self):
        """Return a pending job with current paths, without moving any files."""
        self._owned()
        path = self.root / JOURNAL
        if not path.exists() and not path.is_symlink():
            return None
        journal = _read_json(path)
        if (journal.get("artifact_kind") != "qwen35-draft-workspace-journal"
                or journal.get("owner_id") != self._owner["owner_id"]
                or journal.get("operation") not in {"adopt", "reset"}):
            raise ValueError("invalid draft workspace migration journal")
        state = _state(journal["state"])
        for name, record in state["components"].items():
            source = self._record_path(name, record)
            destination = self.root / name
            if source != destination and destination.exists():
                if source.exists():
                    raise ValueError(f"both source and destination exist during {name} migration")
                record["directory"] = str(destination)
            self._verify_record(name, record)
        return state

    def _journal(self, operation, state):
        self._write(JOURNAL, {"schema_version": 1,
            "artifact_kind": "qwen35-draft-workspace-journal", "owner_id": self._owner["owner_id"],
            "operation": operation, "state": state})

    def _checkpoint(self, state):
        # The journal stays present until all these updates are durable, so an
        # interrupted retry cannot revive a stale result or obsolete receipt.
        self._write("job.json", state)
        self._unlink("result.json")
        for name in STAGES:
            receipt = f"completed-{name}.json"
            if name in state["components"]:
                self._write(receipt, state["components"][name])
            else:
                self._unlink(receipt)

    @contextmanager
    def _source_locks(self, paths):
        """Keep other managed workspaces idle while their stage trees move.

        Historical immutable jobs have no lock file; do not create or rewrite
        metadata in them. A managed source must retain its existing lock inode.
        """
        descriptors = []
        try:
            for parent in sorted({path.parent for path in paths if path.parent != self.root}):
                lock = parent / LOCK
                if not lock.exists() and not lock.is_symlink():
                    if (parent / OWNER).exists():
                        raise ValueError(f"source draft workspace is missing its lock: {parent}")
                    continue
                fd = os.open(lock, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
                descriptors.append(fd)
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    raise ValueError("invalid source draft workspace lock file")
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise RuntimeError(f"source draft workspace is locked by another process: {parent}") from error
            yield
        finally:
            for fd in reversed(descriptors):
                os.close(fd)

    def adopt(self, state):
        """Rename selected validated stage trees into this workspace, never copy."""
        self._owned()
        value = _state(state)
        # Acquire every existing source lock before validating or moving any
        # stage. This also prevents a locked later source causing partial moves.
        sources = [self._record_path(name, record) for name, record in value["components"].items()]
        with self._source_locks(sources):
            return self._adopt_locked(value)

    def _adopt_locked(self, value):
        paths = {}
        for name, record in value["components"].items():
            source = self._verify_record(name, record)
            destination = self.root / name
            if source != destination and (destination.exists() or destination.is_symlink()):
                raise FileExistsError(f"draft stage destination already exists: {destination}")
            if source.stat().st_dev != self.root.stat().st_dev:
                raise OSError(errno.EXDEV, "draft adoption requires same-filesystem rename; no payload copy attempted")
            paths[name] = source
        for name, source in paths.items():
            if any(other != name and (source == path or source in path.parents or path in source.parents)
                   for other, path in paths.items()):
                raise ValueError("selected draft component directories overlap")
        value["output"] = str(self.root)
        self._journal("adopt", value)
        for name, source in paths.items():
            destination = self.root / name
            if source != destination:
                os.rename(source, destination)
                _sync_directory(source.parent)
                _sync_directory(self.root)
            value["components"][name]["directory"] = str(destination)
            self._journal("adopt", value)
        self._checkpoint(value)
        self._unlink(JOURNAL)
        return value

    def reset(self, state):
        """Discard only unselected owned stages and the previous board attempt."""
        self._owned()
        value = _state(state)
        for name, record in value["components"].items():
            if self._verify_record(name, record) != self.root / name:
                raise ValueError("adopt external draft components before resetting the workspace")
        remove = [name for name in STAGES if name not in value["components"]] + ["board"]
        for name in remove:
            path = self.root / name
            if path.is_symlink() or (path.exists() and not path.is_dir()):
                raise ValueError(f"refusing to clean non-directory draft stage: {name}")
        value["output"] = str(self.root)
        self._journal("reset", value)
        self._checkpoint(value)
        for name in remove:
            path = self.root / name
            if path.exists():
                shutil.rmtree(path)
                _sync_directory(self.root)
        self._unlink(JOURNAL)
        return value
