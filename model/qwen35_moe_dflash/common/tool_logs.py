"""Standard-library-only diagnostics for external GE/OMG tools."""

import re


_MAX_SCAN_BYTES = 64 << 20
_NAMED_NODE_REJECTION = re.compile(
    rb"(?:\bnode\s+[^\s\"']+\s+type\s+[^\s\"']+|\[(?:op|node):[^\]\r\n]+\btype:[^\]\r\n]+\])"
    rb".*(?:don't support|does not support|not support|unsupported)", re.I)
_MATRIX_SIZE_LIMIT = re.compile(
    rb"\bMatrix\s+[mnk]\s+(?:greater|less)\s+than\s+\d+\s+is\s+not\s+supported\b", re.I)
_CONCRETE_FAILURE = re.compile(
    rb"(?:\bnode\b|\bop\b|\btype\b).*(?:don't support|does not support|not support|unsupported)"
    rb"|(?:infershape|infer.?shape).*(?:failed|missing|not found)"
    rb"|(?:get|find).*format config.*failed", re.I)
_SUPPORT_DTYPE_FAILURE = re.compile(
    rb"(?:CheckIfTensorTypeSupport|(?:input|output)(?:data)?Type).*(?:not support|unsupported)", re.I)
_KERNEL_FAILURE = re.compile(rb"kernel.*(?:failed|missing|not found|not valid|invalid)", re.I)
_PYTHON_FAILURE = re.compile(rb"^(?:Traceback \(most recent call last\):|[\w.]+(?:Error|Exception):)")


_OMG_TRACE_PATTERNS = tuple(re.compile(pattern) for pattern in (
    # Some internal OMG builds emit these ordinary graph traces at E severity.
    # Match complete, observed messages rather than dropping all [lsy] lines.
    rb'air_op_convert_funcs\.cpp ConvertReshape\(\d+\)::"\[lsy\] node-[^"\r\n]+ input has (?:Const|Reshape) Node:[^"\r\n]+"\s*$',
    rb'air_op_convert_funcs\.cpp ConvertReshape\(\d+\)::"\+{4} \[lsy\] change (?:const|reshape), origin = [-\d\s]+, newDim = [-\d\s]+"\s*$',
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
                                 b"not support", b"don't support", b"does not support", b"init error", b"not equals to",
                                 b"set input basebuffer error")
    ) or bool(re.search(rb"\[cpucl\].*input ix\[\d+\] size\[\d+\] must >= tensor size\[\d+\]", low))


def log_excerpt(log):
    try:
        return _read_excerpt(log)
    except OSError as error:
        # Diagnostic failure must not mask the subprocess exception.
        return f"Log excerpt unavailable: {error}"


def _read_excerpt(log):
    """Prioritize concrete failures over E-level traces; retain the raw tail."""
    if not log.is_file():
        return ""
    candidates = {priority: [] for priority in range(6)}
    with log.open("rb") as stream:
        line_number = 0
        scanned = 0
        while scanned < _MAX_SCAN_BYTES:
            # Bound memory even for a malformed tool log with a huge line.
            chunk = stream.readline(min(8192, _MAX_SCAN_BYTES - scanned))
            if not chunk:
                break
            scanned += len(chunk)
            line_number += 1
            python_failure = bool(_PYTHON_FAILURE.search(chunk))
            if _is_error(chunk) or python_failure:
                # A backend can reject many candidate dtypes before declaring
                # the actual named node unsupported. Never let those attempts
                # consume every slot ahead of the terminal node/type message.
                priority = (0 if _NAMED_NODE_REJECTION.search(chunk) else
                            1 if _MATRIX_SIZE_LIMIT.search(chunk) else
                            3 if _SUPPORT_DTYPE_FAILURE.search(chunk) else
                            2 if _CONCRETE_FAILURE.search(chunk) else
                            4 if python_failure or _KERNEL_FAILURE.search(chunk) else 5)
                if len(candidates[priority]) < 6:
                    candidates[priority].append((line_number, chunk[:2048],
                                                 len(chunk) > 2048 or not chunk.endswith(b"\n")))
            while chunk and not chunk.endswith(b"\n") and scanned < _MAX_SCAN_BYTES:
                chunk = stream.readline(min(8192, _MAX_SCAN_BYTES - scanned))
                scanned += len(chunk)
        scan_limited = stream.tell() < log.stat().st_size
        stream.seek(max(0, log.stat().st_size - 8192))
        tail = stream.read(8192).decode("utf-8", errors="replace")
    first, budget = [], 4096
    selected = [row for priority in range(6) for row in candidates[priority]][:6]
    for number, chunk, truncated in selected:
        if budget <= 0:
            break
        excerpt = chunk[:budget]
        suffix = " [truncated]" if truncated or len(excerpt) < len(chunk) else ""
        first.append(f"L{number}: " + excerpt.decode("utf-8", errors="replace").rstrip() + suffix)
        budget -= len(excerpt)
    label = "Priority error messages:" if any(candidates[p] for p in range(5)) else "First error messages:"
    initial = label + "\n" + "\n".join(first) + "\n" if first else ""
    if scan_limited:
        initial += f"Diagnostic scan limited to first {_MAX_SCAN_BYTES} bytes; raw tail retained.\n"
    return initial + "Log tail:\n" + tail
