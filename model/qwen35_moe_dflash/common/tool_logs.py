"""Standard-library-only diagnostics for external GE/OMG tools."""

import json
from pathlib import Path
import re


_MAX_SCAN_BYTES = 64 << 20
_NAMED_NODE_REJECTION = re.compile(
    rb"(?:\bnode\s+[^\s\"']+\s+type\s+[^\s\"']+|\[(?:op|node):[^\]\r\n]+\btype:[^\]\r\n]+\])"
    rb".*(?:don't support|does not support|not support|unsupported|Verify failed.*DataType.*wrong)", re.I)
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
_TRACEBACK_START = b"Traceback (most recent call last):"
_WORKER_FAILURE = re.compile(rb"COMPILER PROCESS DIED OF NODE|compiler process died, caused by current task|Failed to (?:pre-)?compile node", re.I)
_PASS_FAILURE = re.compile(rb"ByPassNode.*PreCheck|Pass Run failed|AfterInferShape.*failed to run", re.I)
_BUILD_FAILURE = re.compile(rb"create OpGraphBuilder fail|opGraphBuilder is null|Op Build4NPU failed|OpPreBuild failed\. Node:", re.I)
_TILING_FAILURE = re.compile(rb"GetDataParam.*inputTensorPtr is NULL|\bProduceTiling failed\b|\bcompileInfo is null\b", re.I)


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


def compilation_diagnostics(directory):
    """Read retained reports and bounded log excerpts; never invoke a tool."""
    root = Path(directory)
    result = {'report': str(root / 'report.json'), 'compile_log': str(root / 'compile.log')}
    try:
        with (root / 'report.json').open('rb') as stream:
            raw = stream.read((8 << 20) + 1)
        if len(raw) > 8 << 20:
            raise ValueError('compiler report exceeds 8 MiB')
        report = json.loads(raw)
        if not isinstance(report, dict):
            raise ValueError('compiler report must be an object')
        result.update({key: report[key] for key in
                       ('status', 'stage', 'model_compilation', 'returncode', 'error') if key in report})
    except (OSError, ValueError) as error:
        result['report_error'] = str(error)
    log = root / 'compile.log'
    if not log.is_file():
        log = root / 'help.log'
    result['excerpt_source'] = str(log)
    result['log_excerpt'] = log_excerpt(log)
    return result


def _read_excerpt(log):
    """Prioritize concrete failures over E-level traces; retain the raw tail."""
    if not log.is_file():
        return ""
    candidates = {priority: [] for priority in range(6)}
    terminal_pass = []
    tracebacks, active_trace = [], None
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
            # Worker exceptions can be far from EOF, after thousands of
            # recoverable constant-folding errors. Keep their frames and
            # terminal exception independently of the six-message budget.
            # A tool/worker can interleave text on the traceback header line.
            if _TRACEBACK_START in chunk:
                active_trace = {'lines': [], 'bytes': 0, 'truncated': False}
                tracebacks.append(active_trace)
                tracebacks = tracebacks[-2:]
            if active_trace is not None:
                remaining = 8192 - active_trace['bytes']
                excerpt = chunk[:min(1024, remaining)]
                active_trace['lines'].append((line_number, excerpt))
                active_trace['bytes'] += len(excerpt)
                active_trace['truncated'] |= len(excerpt) < len(chunk)
                if python_failure and _TRACEBACK_START not in chunk:
                    active_trace = None
                elif len(active_trace['lines']) >= 48 or active_trace['bytes'] >= 8192:
                    active_trace['truncated'] = True
                    active_trace = None
            if _is_error(chunk) or python_failure:
                if any(pattern.search(chunk) for pattern in
                       (_PASS_FAILURE, _BUILD_FAILURE, _WORKER_FAILURE, _TILING_FAILURE)):
                    terminal_pass.append((line_number, chunk[:2048], len(chunk) > 2048))
                    terminal_pass = terminal_pass[-3:]
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
    # Late pass failures can terminate compilation after early constant-folding
    # diagnostics. Show both; neither is automatically a root-cause verdict.
    pass_lines = {row[0] for row in terminal_pass}
    selected = (terminal_pass + [row for priority in range(6) for row in candidates[priority]
                                 if row[0] not in pass_lines])[:6]
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
    contexts = []
    for trace in tracebacks:
        contexts.append('Python traceback context (bounded):\n' + '\n'.join(
            f'L{number}: ' + chunk.decode('utf-8', errors='replace').rstrip()
            for number, chunk in trace['lines']))
        if trace['truncated']:
            contexts.append('[Traceback context truncated]')
    if contexts:
        initial = '\n'.join(contexts) + '\n' + initial
    return initial + "Log tail:\n" + tail
