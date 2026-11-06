"""Explicitly discard all Draft tests while retaining verified source weights."""
from contextlib import ExitStack
import fcntl
import os
from pathlib import Path
import shutil

from qwen35_moe_dflash.common.hashing import sha256_file
from qwen35_moe_dflash.execution.runtime_config import read_json
from qwen35_moe_dflash.toolchain.compiler_storage import _plain, _tree_bytes, require_idle


def _source_candidates(root, tests, source):
    if source:
        yield Path(source).resolve(strict=True)
        return
    preferred = root/'draft/source/prepared'
    if (preferred/'prepared.json').is_file():
        yield preferred
    # Find preparation metadata only, never inspect arbitrary weight payloads.
    # Test exports are disposable copies, not independent rebuild sources.
    for base, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if Path(base)/d != tests and not (Path(base)/d).is_symlink()]
        folder = Path(base)
        if 'prepared.json' in files and folder != preferred:
            yield folder


def _verify_source(root, tests, source):
    from qwen35_moe_dflash.draft.reference import _validated_prepared
    failures = []
    for folder in _source_candidates(root, tests, source):
        try:
            _plain(folder, directory=True)
            if folder == tests or tests in folder.parents:
                raise ValueError('rebuild source is inside disposable draft/tests; move it outside first')
            manifest = folder/'prepared.json'
            _plain(manifest)
            data = read_json(manifest, 16 << 20)
            if data.get('tiny') is not False:
                raise ValueError('full-model prepared weights required; tiny fixtures cannot justify cleanup')
            print('draft-clean: verifying rebuild weights at '+str(folder), flush=True)
            _, data, _, payloads = _validated_prepared(folder)
            for payload in payloads.values():
                if tests in payload.parents:
                    raise ValueError('prepared weight still depends on disposable tests: '+str(payload))
            return dict(path=str(folder), prepared_sha256=sha256_file(manifest),
                        verified_weight_files=len(payloads), verification='FULL_TOPOLOGY_ALL_PREPARED_WEIGHT_SHAPES_SIZES_AND_SHA256')
        except (OSError, ValueError, KeyError, TypeError) as error:
            failures.append(str(folder)+': '+str(error))
    raise ValueError('no verified FP16 rebuild source outside draft/tests; '
        'pass --source /existing/directory/containing/prepared.json. '+ '; '.join(failures[:3]))


def clean_all_tests(root, *, source=None, apply=False):
    """All old test models/results may be removed by explicit user request.

    No historical result/accuracy gate is needed to retire an experiment. Live
    local users and held writer locks still block deletion. Sources, runtime,
    common, prefill and decode are outside the deletion scope.
    """
    root = Path(root).resolve(strict=True)
    tests = root/'draft/tests'
    if not tests.exists() and not tests.is_symlink():
        return dict(status='ALREADY_CLEAN', directory=str(tests), free_bytes_after=shutil.disk_usage(root).free)
    _plain(tests, directory=True)
    if tests in Path(__file__).resolve().parents:
        raise ValueError('cleanup source code is inside disposable draft/tests')
    if not any(tests.iterdir()):
        return dict(status='ALREADY_CLEAN', directory=str(tests), free_bytes_after=shutil.disk_usage(root).free)
    require_idle(tests)
    with ExitStack() as stack:
        paths = [root/'draft/.all-tests-cleanup.lock']
        for base, dirs, files in os.walk(tests, followlinks=False):
            for name in dirs + files:
                p = Path(base)/name
                if not p.is_symlink() and p.is_mount():
                    raise ValueError('mounted test entry retained: '+str(p))
            paths.extend(Path(base)/name for name in files if name.endswith('.lock'))
        for path in sorted(paths):
            fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            lock = stack.enter_context(os.fdopen(fd, 'a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original = _verify_source(root, tests, source)
        allocated = _tree_bytes(tests, allow_inactive_sockets=True)
        selected = []
        for path in sorted(tests.iterdir()):
            info = path.lstat()
            selected.append(dict(name=path.name, device=info.st_dev, inode=info.st_ino))
        before = shutil.disk_usage(root).free
        removed = []
        if apply:
            require_idle(tests)
            # Recheck socket registrations after source verification/inventory,
            # before deleting any entry. A closed pathname may be rebound.
            _tree_bytes(tests, allow_inactive_sockets=True)
            for row in selected:
                path = tests/row['name']
                info = path.lstat()
                if (info.st_dev, info.st_ino) != (row['device'], row['inode']):
                    raise ValueError('test entry changed before removal: '+str(path))
                if path.is_symlink() or not path.is_dir():
                    path.unlink()
                else:
                    shutil.rmtree(path)
                removed.append(row['name'])
        free = shutil.disk_usage(root).free
        return dict(status='CLEANED' if apply else 'PLANNED',
            scope='ALL_DRAFT_TEST_CONTENTS_INCLUDING_BASELINE_AND_RESULTS; USER_AUTHORIZED',
            directory=str(tests), retained_source=original,
            selected=[r['name'] for r in selected], removed=removed,
            allocated_bytes_selected=allocated, free_bytes_before=before, free_bytes_after=free,
            observed_free_bytes_change=free-before,
            device_query='NOT_RUN', device_execution='NOT_RUN',
            next='REBUILD_FROM_SOURCE_BEFORE_REPLAY_OR_HEAD_PROBE',
            limitation='Linked files may release less space. No historical results are archived; runtime configurations are not rewritten.')
