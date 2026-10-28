"""Bounded observations while waiting for a compiler; never process control."""
from contextlib import contextmanager
import math
import re
import threading
import time


def _log_status(log):
    try:
        stat = log.stat()
        with log.open("rb") as stream:
            stream.seek(max(0, stat.st_size - 4096))
            tail = stream.read(4096).decode("utf-8", errors="replace")
        lines = [line.strip() for line in tail.splitlines() if line.strip()]
        last = lines[-1] if lines else ""
        last = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", last)
        last = "".join(char for char in last if char.isprintable())
        if len(last) > 240:
            last = last[:240] + " [truncated]"
        return {"log_bytes": stat.st_size,
                "log_idle_seconds": int(max(0, time.time() - stat.st_mtime)), "last_line": last}
    except OSError as error:
        return {"log_error": f"{type(error).__name__}: {error}", "last_line": ""}


@contextmanager
def compiler_heartbeat(log, progress, *, interval_seconds=30):
    """Report elapsed time and bounded log tail without changing subprocess.run.

    Silence does not establish a hang: tool output can be buffered. Diagnostic
    failures never retry/terminate the compiler or mask its original result.
    """
    if not math.isfinite(interval_seconds) or interval_seconds <= 0:
        raise ValueError("heartbeat interval must be finite and positive")
    report = {"enabled": progress is not None, "interval_seconds": interval_seconds,
              "samples": 0, "error": None, "last": None,
              "scope": "Waiting observations only; log activity does not prove forward progress."}
    if progress is None:
        yield report
        return
    stop, lock = threading.Event(), threading.Lock()
    observed = {"samples": 0, "error": None, "last": None}
    started = time.monotonic()

    def monitor():
        while not stop.wait(interval_seconds):
            elapsed = int(time.monotonic() - started)
            status = _log_status(log)
            if stop.is_set():
                return
            message = f"Waiting for OMG: elapsed={elapsed}s"
            if "log_bytes" in status:
                message += f"; log={status['log_bytes']} bytes; last log write {status['log_idle_seconds']}s ago"
            else:
                message += "; log unavailable"
            if status["last_line"]:
                message += "; last: " + status["last_line"]
            row = {"stage": "omg-wait", "message": message, "elapsed_seconds": elapsed, **status}
            with lock:
                observed["samples"] += 1
                observed["last"] = row
            if stop.is_set():
                return
            try:
                progress(row)
            except Exception as error:
                with lock:
                    observed["error"] = f"{type(error).__name__}: {error}"
                return

    worker = threading.Thread(target=monitor, name="omg-wait-observer", daemon=True)
    try:
        worker.start()
    except RuntimeError as error:
        report["error"] = f"{type(error).__name__}: {error}"
        yield report
        return
    try:
        yield report
    finally:
        stop.set()
        # A blocked terminal/filesystem must not turn a bounded compiler timeout
        # into an unbounded diagnostic wait. The worker owns no compiler handle.
        worker.join(timeout=1)
        with lock:
            report.update(observed)
        report["monitor_stop_timeout"] = worker.is_alive()
        report["elapsed_seconds"] = int(time.monotonic() - started)
