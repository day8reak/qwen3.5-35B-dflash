"""Shared prompt inputs utilities used by model preparation and execution."""
from __future__ import annotations
import hashlib
import struct
from qwen35_moe_dflash.execution.original_embedding_lookup import OriginalEmbeddingLookup, _current

def _backends():
    from qwen35_moe_dflash.target.original_target_smoke_backend import OriginalTargetSmokeBackend
    from qwen35_moe_dflash.draft.existing_draft_smoke_backend import ExistingDraftSmokeBackend
    return OriginalTargetSmokeBackend, ExistingDraftSmokeBackend

def _prompt(path, length):
    if type(length) is not int or not 1 <= length <= 63:
        raise ValueError("prompt length must be1..63 explicit tokens")
    path, before = _current(path)
    if not 4 * length <= before["bytes"] <= 16384 or before["bytes"] % 4:
        raise ValueError("prompt source must contain enough little-endian INT32 token IDs")
    raw = path.read_bytes()
    if _current(path)[1] != before:
        raise ValueError("prompt source changed during read")
    values = list(struct.unpack("<" + str(length) + "i", raw[:4 * length]))
    if any(not 0 <= token < 248320 for token in values):
        raise ValueError("prompt contains an out-of-vocabulary token ID")
    return values, {"path": str(path), "bytes": before["bytes"], "sha256": hashlib.sha256(raw).hexdigest(),
        "identity": before, "selected_prefix_tokens": length,
        "policy": "NEW_SHORT_PROMPT_FROM_EXPLICIT_SOURCE_ID_PREFIX_NOT_CONTINUATION_OF_OLD64_RUN",
        "tokenizer_provenance": "USER_SUPPLIED_NOT_REVALIDATED"}
