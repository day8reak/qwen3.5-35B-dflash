"""Shared board lock utilities used by model preparation and execution."""
from __future__ import annotations
import shlex
import time

def _space(text):
    rows = [line.split() for line in text.splitlines() if line.strip()]
    candidates = [int(row[3]) * 1024 for row in rows if len(row) >= 6 and all(v.isdigit() for v in row[1:4])]
    if len(candidates) != 1:
        raise ValueError("board df -Pk did not return one unambiguous available-byte observation")
    return candidates[0]

def _acquire_global(hdc, lock, token, wait_seconds):
    started, last_update = time.monotonic(), -30.0
    busy = "printf 'BUSY\\n'"
    if wait_seconds:
        owner = shlex.quote(str(lock / 'owner'))
        # Another runner can be between mkdir and writing owner, or already releasing it.
        busy = (f"{{ if test ! -L {owner} && test -f {owner} && IFS= read -r probe_lock_owner 2>/dev/null < {owner}; "
                "then printf 'BUSY %s\\n' \"$probe_lock_owner\"; else printf 'BUSY\\n'; fi; }")
    while True:
        # HDC remote shells merge stderr into their returned stream. An
        # existing lock is expected contention, so its mkdir diagnostic must
        # not precede the exact BUSY protocol response.
        command = ("if mkdir " + shlex.quote(str(lock)) + " 2>/dev/null; then printf '%s\\n' " + shlex.quote(token)
                   + " > " + shlex.quote(str(lock / "owner")) + " && printf 'ACQUIRED\\n'; else test -d "
                   + shlex.quote(str(lock)) + " && test ! -L " + shlex.quote(str(lock)) + " && " + busy + "; fi")
        # A busy shared HDC server can take longer than ten seconds even for
        # this small command. A timeout still remains uncertain and is never
        # replayed automatically; the overall queue budget is separate.
        _, text = hdc.shell(command, "acquire-global-model-io-lock", timeout=min(60, hdc.timeout))
        if text == "ACQUIRED":
            return
        state, _, observed_owner = text.partition(" ")
        if text != "BUSY" and not (wait_seconds and state == "BUSY" and "\n" not in text):
            detail = ""
            if callable(getattr(hdc, "_diagnostics", None)) and getattr(hdc, "last_call", None) is not None:
                detail = hdc._diagnostics()
            excerpt = repr(text[:2000]) + (" [truncated]" if len(text) > 2000 else "")
            raise ValueError("unrecognized global target model-I/O lock response: " + excerpt + detail)
        owner_note = f" owner={observed_owner[:128]!r}" if observed_owner else ""
        elapsed = time.monotonic() - started
        if elapsed >= wait_seconds:
            raise ValueError("BOARD_MODEL_IO_BUSY: another target probe owns the board; no model upload or loading attempted" + owner_note)
        if elapsed - last_update >= 30:
            print(f"target-model-io: waiting for the other board model probe; elapsed={elapsed:.0f}s/{wait_seconds}s{owner_note}", flush=True)
            last_update = elapsed
        time.sleep(min(5, wait_seconds - elapsed))

def _release_global(hdc, lock, token, *, completed_work=None):
    path = shlex.quote(str(lock))
    owner = shlex.quote(str(lock / 'owner'))
    expected = shlex.quote(token)
    absent = "echo LOCK_ABSENT; exit 1"
    if completed_work is not None:
        work = shlex.quote(str(completed_work))
        marker = shlex.quote(str(completed_work / '.qwen35-hdc-owner'))
        absent = f'''test -d {work} && test ! -L {work} && test -f {marker} && test ! -L {marker} &&
    IFS= read -r work_owner < {marker} && test "$work_owner" = {expected} ||
    {{ echo LOCK_ABSENT_WORK_OWNER_UNCONFIRMED; exit 1; }}
    echo ALREADY_ABSENT; exit 0'''
    command = f'''test -x {shlex.quote(str(lock.parent))} || {{ echo LOCK_PARENT_INACCESSIBLE; exit 1; }}
if test -L {path}; then echo LOCK_SYMLINK; exit 1; fi
if test ! -e {path}; then {absent}; fi
test -d {path} || {{ echo LOCK_NOT_DIRECTORY; exit 1; }}
test -f {owner} && test ! -L {owner} && IFS= read -r lock_owner < {owner} ||
    {{ echo LOCK_OWNER_UNREADABLE; exit 1; }}
test "$lock_owner" = {expected} || {{ echo LOCK_OWNER_MISMATCH; exit 1; }}
rm {owner} && rmdir {path} && echo RELEASED'''
    status, response = hdc.shell(command, 'release-global-model-io-lock', allow_failure=True,
                                 timeout=min(60, hdc.timeout))
    if status or response not in ('RELEASED', 'ALREADY_ABSENT'):
        raise RuntimeError(f'Cannot release {lock}: remote exit {status}, {response!r}')
    return response
