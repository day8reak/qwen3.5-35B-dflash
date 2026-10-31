"""Keep dependency root causes visible without importing model frameworks."""
import subprocess
import sys


_REFERENCE_IMPORT_TIMEOUT_SECONDS = 60
_MAX_IMPORT_ERROR_CHARS = 8192
_REFERENCE_IMPORT_CODE = (
    "from qwen35_moe_dflash.reference.backends import configure_reference_backend\n"
    "configure_reference_backend('cpu')\n"
    "from transformers import Qwen3_5MoeTextConfig, Qwen3_5MoeForCausalLM\n"
    "from qwen35_moe_dflash.model import DFlashDraftModel\n"
)


def format_error(error):
    messages, seen = [], set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        messages.append(f"{type(error).__name__}: {error}")
        error = error.__cause__ or (None if error.__suppress_context__ else error.__context__)
    return " <- ".join(messages)


def _bounded_import_error(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    value = (value or "").strip()
    if len(value) <= _MAX_IMPORT_ERROR_CHARS:
        return value
    marker = "\n... import output truncated ...\n"
    # Keep both the initial dependency failure and the final exception wrapper.
    head = _MAX_IMPORT_ERROR_CHARS // 2
    tail = _MAX_IMPORT_ERROR_CHARS - head - len(marker)
    return value[:head] + marker + value[-tail:]


def probe_reference_import():
    """Import model classes in an isolated process; do not allocate models/weights.

    The current interpreter, PYTHONPATH and environment are inherited, then the
    child selects the same CPU policy as the reference CLI. Partial dependency
    imports and CPU backend selection cannot affect preflight's own process.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", _REFERENCE_IMPORT_CODE],
            capture_output=True, text=True, timeout=_REFERENCE_IMPORT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        detail = error.stderr or error.stdout
        message = f"Reference model import timed out after {_REFERENCE_IMPORT_TIMEOUT_SECONDS}s"
        if detail:
            if isinstance(detail, bytes):
                detail = detail.decode("utf-8", errors="replace")
            message += f":\n{detail}"
    except OSError as error:
        message = f"Reference model import could not start: {format_error(error)}"
    else:
        if result.returncode == 0:
            return {"reference_backend": "cpu", "reference_import_ready": True, "reference_import_error": None}
        detail = result.stderr or result.stdout
        message = f"Reference model import exited with code {result.returncode}"
        if detail:
            message += f":\n{detail}"
    return {"reference_backend": "cpu", "reference_import_ready": False,
            "reference_import_error": _bounded_import_error(message)}
