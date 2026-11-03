"""Bounded probe transfer and progress; never launch or unlock a model job."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import threading
import time

from .run_layout import _save
from .hdc_run import _check_local, _file_record


class Progress:
    def __init__(self, out, state):
        self.out, self.state = out, state
        self.started = time.monotonic()
        self.message = ''
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()

    def _print(self, message):
        print(f'vector-validation: {message}; elapsed={time.monotonic()-self.started:.0f}s',
              file=sys.stderr, flush=True)

    def _heartbeat(self):
        while not self.stopped.wait(30):
            self._print('Waiting: '+self.message)

    def update(self, stage, message, **values):
        self.message = message
        self.state.update(stage=stage, progress=message, updated_at_unix=time.time(), **values)
        _save(self.out/'result.json', self.state)
        self._print(message)

    def close(self):
        self.stopped.set()
        self.thread.join()


def pack_uploads(out, rows):
    """One transfer replaces per-file round trips; remote checks still cover every file."""
    names = [r['remote'] for r in rows]
    if len(set(names)) != len(names) or any(not re.fullmatch(r'[A-Za-z0-9_./-]+', n)
            or PurePosixPath(n).is_absolute() or any(p in ('', '.', '..') for p in n.split('/'))
            or n in ('payload.sha256', 'payload.tar') for n in names):
        raise ValueError('invalid or duplicate probe payload destination')
    required = sum(r['bytes']+1024 for r in rows)+(1 << 20)
    if shutil.disk_usage(out).free < required:
        raise OSError('insufficient local space for probe payload archive')
    checksums = ''.join(r['sha256']+'  '+r['remote']+'\n' for r in rows).encode()
    archive = out/'payload.tar'
    with tarfile.open(archive, 'x', format=tarfile.USTAR_FORMAT) as stream:
        for row in rows:
            _check_local(row)
            info = tarfile.TarInfo(row['remote']); info.size = row['bytes']; info.mode = 0o644
            with open(row['local'], 'rb') as source:
                stream.addfile(info, source)
        info = tarfile.TarInfo('payload.sha256'); info.size = len(checksums); info.mode = 0o644
        stream.addfile(info, io.BytesIO(checksums))
    return _file_record(archive, 'payload.tar')


def unpack_logs(archive, out, names):
    """Only known regular log files, bounded individually and in total."""
    allowed = set(names); seen = set(); total = 0
    with tarfile.open(archive) as stream:
        for member in stream:
            if (member.name not in allowed or member.name in seen or not member.isfile()
                    or not 0 <= member.size <= 4 << 20):
                raise ValueError('unexpected probe log archive member')
            total += member.size
            if total > 64 << 20:
                raise ValueError('probe log archive exceeds size limit')
            seen.add(member.name)
            with stream.extractfile(member) as source, (out/member.name).open('xb') as target:
                shutil.copyfileobj(source, target)


def _json(path, limit=1 << 20):
    if not path.is_file() or path.is_symlink():
        return None
    with path.open('rb') as stream:
        raw = stream.read(limit+1)
    if len(raw) > limit:
        raise ValueError('probe status record exceeds limit: '+str(path))
    return json.loads(raw)


def local_status(root):
    """Read-only, no HDC or output lock; works while the original launcher is stopped."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError('execution directory is missing')
    request, result = _json(root/'request.json'), _json(root/'result.json')
    jobs = []
    for entry in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = entry.read_bytes().decode(errors='replace').split('\0')
            if int(entry.parent.name) == os.getpid() or not any(
                    Path(a).name == 'validate_draft_vector.py' for a in args[:3]):
                continue
            output = next((a.split('=', 1)[1] for a in args if a.startswith('--output=')), None)
            if '--output' in args:
                output = args[args.index('--output')+1]
            if output is None:
                continue
            path = Path(output)
            if not path.is_absolute():
                path = (entry.parent/'cwd').resolve()/path
            if path.resolve() != root:
                continue
            state = next(line.split(':', 1)[1].strip() for line in
                         (entry.parent/'status').read_text().splitlines() if line.startswith('State:'))
            jobs.append(dict(pid=int(entry.parent.name), state=state,
                             suspended=state.split()[0] in ('T', 't')))
        except (OSError, ValueError, IndexError, StopIteration):
            continue
    calls = []
    for p in (root/'calls').glob('*.json'):
        if re.fullmatch(r'\d+-[A-Za-z0-9_-]+\.json', p.name) and not p.is_symlink():
            calls.append(p)
        if len(calls) > 16384:
            raise ValueError('too many probe transport records for bounded status')
    calls.sort(key=lambda p: (int(p.name.split('-', 1)[0]), p.name))
    last = None
    if calls:
        p = calls[-1]; r = _json(p, 4 << 20) or {}
        last = dict(path=str(p), label=r.get('label'), returncode=r.get('returncode'),
                    age_seconds=round(max(0, time.time()-p.stat().st_mtime), 1))
        if r.get('label') == 'send' and r.get('argv'):
            last['destination'] = r['argv'][-1]
        if r.get('error'):
            last['error'] = r['error'][:1000]
    return dict(artifact_kind='qwen35-draft-vector-status', status='INSPECTED',
        execution_source=str(root), model_execution='NOT_RUN', device_query='NOT_RUN',
        local_jobs=jobs, board_attempt=(request or {}).get('board_attempt'),
        requested_cases=len((request or {}).get('cases', [])),
        recorded_state={k: result[k] for k in ('status', 'stage', 'progress', 'launch_requested',
                       'lock_state', 'updated_at_unix', 'error') if result and k in result},
        completed_transport_calls=len(calls), last_completed_transport_call=last,
        limitation='Local snapshots only. The last completed transport call is not the current call; '
                   'missing completion does not prove device execution or a hang.')
