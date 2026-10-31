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
    while True:
        # HDC remote shells merge stderr into their returned stream. An
        # existing lock is expected contention, so its mkdir diagnostic must
        # not precede the exact BUSY protocol response.
        command = ("if mkdir " + shlex.quote(str(lock)) + " 2>/dev/null; then printf '%s\\n' " + shlex.quote(token)
                   + " > " + shlex.quote(str(lock / "owner")) + " && printf 'ACQUIRED\\n'; else test -d "
                   + shlex.quote(str(lock)) + " && test ! -L " + shlex.quote(str(lock)) + " && printf 'BUSY\\n'; fi")
        # A busy shared HDC server can take longer than ten seconds even for
        # this small command. A timeout still remains uncertain and is never
        # replayed automatically; the overall queue budget is separate.
        _, text = hdc.shell(command, "acquire-global-model-io-lock", timeout=min(60, hdc.timeout))
        if text == "ACQUIRED":
            return
        if text != "BUSY":
            detail = ""
            if callable(getattr(hdc, "_diagnostics", None)) and getattr(hdc, "last_call", None) is not None:
                detail = hdc._diagnostics()
            excerpt = repr(text[:2000]) + (" [truncated]" if len(text) > 2000 else "")
            raise ValueError("unrecognized global target model-I/O lock response: " + excerpt + detail)
        elapsed = time.monotonic() - started
        if elapsed >= wait_seconds:
            raise ValueError("BOARD_MODEL_IO_BUSY: another target probe owns the board; no model upload or loading attempted")
        if elapsed - last_update >= 30:
            print(f"target-model-io: waiting for the other board model probe; elapsed={elapsed:.0f}s", flush=True)
            last_update = elapsed
        time.sleep(min(5, wait_seconds - elapsed))

def _release_global(hdc, lock, token):
    hdc.shell("test -d " + shlex.quote(str(lock)) + " && test ! -L " + shlex.quote(str(lock))
              + " && read -r probe_lock_owner < " + shlex.quote(str(lock / "owner"))
              + " && test \"$probe_lock_owner\" = " + shlex.quote(token)
              + " && rm " + shlex.quote(str(lock / "owner")) + " && rmdir " + shlex.quote(str(lock)),
              "release-global-model-io-lock", timeout=min(60, hdc.timeout))
