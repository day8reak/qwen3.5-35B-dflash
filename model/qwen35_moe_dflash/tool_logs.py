"""Standard-library-only diagnostics for external GE/OMG tools."""

import re


def _is_error(line):
    # Internal OMG uses E/AI_FMK and untagged CPUCL diagnostics, whereas GE
    # uses [ERROR]. Optional plugin messages at W/I level also contain 'failed'.
    low = line.lower()
    if re.search(rb"(?:^|\s)[diwv]/[a-z_][a-z0-9_.-]*(?=[\s:(])", low) or any(
        tag in low for tag in (b"[debug]", b"[info]", b"[warning]", b"[warn]", b"[trace]")
    ):
        return False
    return bool(re.search(rb"(?:^|\s)[ef]/[a-z_][a-z0-9_.-]*(?=[\s:(])", low)) or any(
        word in low for word in (b"[error]", b"[fatal]", b"error:", b"failed", b"unsupported",
                                 b"not support", b"init error", b"not equals to")
    )


def log_excerpt(log):
    try:
        return _read_excerpt(log)
    except OSError as error:
        # Diagnostic failure must not mask the subprocess exception.
        return f"Log excerpt unavailable: {error}"


def _read_excerpt(log):
    """Show bounded early diagnostics and tail without loading the entire tool log."""
    if not log.is_file():
        return ""
    first = []
    budget = 4096
    with log.open("rb") as stream:
        line_number = 0
        while budget > 0 and len(first) < 6:
            # Bound memory even for a malformed tool log with a huge line.
            chunk = stream.readline(8192)
            if not chunk:
                break
            line_number += 1
            matched = _is_error(chunk)
            if matched:
                excerpt = chunk[:min(2048, budget)]
                suffix = " [truncated]" if len(chunk) > len(excerpt) or not chunk.endswith(b"\n") else ""
                first.append(f"L{line_number}: " + excerpt.decode("utf-8", errors="replace").rstrip() + suffix)
                budget -= len(excerpt)
            while chunk and not chunk.endswith(b"\n"):
                chunk = stream.readline(8192)
        stream.seek(max(0, log.stat().st_size - 8192))
        tail = stream.read(8192).decode("utf-8", errors="replace")
    initial = "First error messages:\n" + "\n".join(first) + "\n" if first else ""
    return initial + "Log tail:\n" + tail
