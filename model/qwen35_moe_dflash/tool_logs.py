"""Standard-library-only diagnostics for external GE/OMG tools."""

import re


_OMG_TRACE_PATTERNS = tuple(re.compile(pattern) for pattern in (
    # Some internal OMG builds emit these ordinary graph traces at E severity.
    # Match complete, observed messages rather than dropping all [lsy] lines.
    rb'air_op_convert_funcs\.cpp ConvertReshape\(\d+\)::"\[lsy\] node-[^"\r\n]+ input has Const Node:[^"\r\n]+"\s*$',
    rb'air_op_convert_funcs\.cpp ConvertReshape\(\d+\)::"\+{4} \[lsy\] change const, origin = [-\d\s]+, newDim = [-\d\s]+"\s*$',
    rb'shape\.cpp GetTotalDimNum\(\d+\)::"\[lsy\] get shape from scrShape:[-\d\s]*"\s*$',
    rb'ir_infer_shape_optimizer\.cpp operator\(\)\(\d+\)::"\[lsy\] node\[[^"\r\n]+\] input shape:[-\d\s]*"\s*$',
    # An absent optional config is explicitly successful in these messages.
    rb'memory_config_parser_host\.cpp LoadConfigFromFile\(\d+\)::"!path\.empty\(\)" "false, return hiai::SUCCESS\."\s*$',
    rb'llm_parser_host\.cpp GetAllLLMInfosFromFile\(\d+\)::"!llmConfigPath\.empty\(\)" "false, return hiai::SUCCESS\."\s*$',
))


def _is_error(line):
    # Internal OMG uses E/AI_FMK and untagged CPUCL diagnostics, whereas GE
    # uses [ERROR]. Optional plugin messages at W/I level also contain 'failed'.
    low = line.lower()
    if re.search(rb"(?:^|\s)[diwv]/[a-z_][a-z0-9_.-]*(?=[\s:(])", low) or any(
        tag in low for tag in (b"[debug]", b"[info]", b"[warning]", b"[warn]", b"[trace]")
    ):
        return False
    if any(pattern.search(line) for pattern in _OMG_TRACE_PATTERNS):
        return False
    return bool(re.search(rb"(?:^|\s)[ef]/[a-z_][a-z0-9_.-]*(?=[\s:(])", low)) or any(
        word in low for word in (b"[error]", b"[fatal]", b"error:", b"failed", b"unsupported",
                                 b"not support", b"init error", b"not equals to",
                                 b"set input basebuffer error")
    ) or bool(re.search(rb"\[cpucl\].*input ix\[\d+\] size\[\d+\] must >= tensor size\[\d+\]", low))


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
