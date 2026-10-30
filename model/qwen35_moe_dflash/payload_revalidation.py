"""Purpose-scoped comparison after a caller freshly hashes a stable payload.

No file is read here. Historical ctime drift can be reconciled only for the
listed decode artifacts; all other identity fields and content remain exact.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re


_INPUT_NAMES = frozenset({"inputs_embeds.bin", "attention_mask.bin", "position_ids.bin",
    "past_k.bin", "past_v.bin", "past_conv_state.bin", "past_recurrent_state.bin",
    "new_kv_cache_pos.bin", "output_pos.bin", "embed_scale.bin", "allQLen.bin",
    "conv_state_pos.bin", "effective_length.bin"})
_STAT_KEYS = ("device", "inode", "bytes", "mtime_ns", "ctime_ns")
_FULL_KEYS = {*_STAT_KEYS, "mode", "uid", "gid", "nlink"}
_ROLES = frozenset({"decode-input", "decode-trial-copied", "decode-trial-generated",
    "decode-trial-input", "decode-output", "decode-log", "source-air"})


def _collector(observations):
    if observations is not None and not isinstance(observations, list):
        raise ValueError("payload revalidation observations must be a list")


def _view(value, role):
    if not isinstance(value, dict) or not isinstance(role, str) or role not in _ROLES:
        return None
    path, size, digest = value.get("path"), value.get("bytes"), value.get("sha256")
    if (not isinstance(path, str) or not Path(path).is_absolute() or ".." in Path(path).parts
            or str(Path(path)) != path or type(size) is not int or size < 0
            or not re.fullmatch(r"[0-9a-f]{64}", str(digest))):
        return None
    name = Path(path).name
    if role.startswith("decode-trial-") or role == "decode-input":
        allowed = name in _INPUT_NAMES and (role == "decode-input" or Path(path).parent.name == "inputs")
    elif role == "decode-output":
        # The caller binds the exact pathname from the verified tool log. A
        # filename never supplies a runtime slot number or state owner here.
        allowed = bool(re.fullmatch(r"[A-Za-z0-9_.-]+", name)) and name not in {".", ".."} and Path(path).parent.name == "outputs"
    elif role == "decode-log": allowed = name in {"tool.stdout", "tool.stderr", "tool.exitcode", "tool.pid"}
    else: allowed = Path(path).suffix == ".air"
    maximum = (64 if role == "source-air" else 8) << 20
    if not allowed or size > maximum or (not size and role != "decode-log"):
        return None
    if role == "decode-input":
        full = value.get("identity")
        if (not isinstance(full, dict) or set(full) != _FULL_KEYS
                or any(type(item) is not int or item < 0 for item in full.values())):
            return None
        stats = [full[key] for key in _STAT_KEYS]
        extra = {key: full[key] for key in ("mode", "uid", "gid", "nlink")}
    else:
        stats, extra = value.get("stat"), None
        if (not isinstance(stats, (list, tuple)) or len(stats) != 5
                or any(type(item) is not int or item < 0 for item in stats)):
            return None
    if stats[2] != size:
        return None
    return path, size, digest, list(stats), extra


def payload_identity_matches(recorded, current, *, role, observations=None):
    """Compare role-bound identities supplied by a fresh stable SHA reader.

    Extra row annotations must be validated by callers. This compares only file
    identity fields, including every nonctime full9 field for decode-input.
    """
    _collector(observations)
    old, new = _view(recorded, role), _view(current, role)
    if old is None or new is None or old[:3] != new[:3] or old[3][:4] != new[3][:4] or old[4] != new[4]:
        return False
    if old[3][4] != new[3][4] and observations is not None:
        event = {"kind": "PAYLOAD_CTIME_ONLY_REVALIDATED", "status": "SHA256_VERIFIED",
            "role": role, "path": new[0], "bytes": new[1], "sha256": new[2],
            "recorded_stat": old[3], "current_stat": new[3], "receipt_mutation": "NONE",
            "scope": "FRESH_BOUNDED_STABLE_PAYLOAD_SHA256_ONLY"}
        if new[4] is not None:
            event["recorded_identity"] = deepcopy(recorded["identity"])
            event["current_identity"] = deepcopy(current["identity"])
        observations.append(event)
    return True


def payload_sequence_matches(recorded, current, *, role, observations=None):
    """Compare complete ordered file bindings, recording only successful sets."""
    _collector(observations)
    if not isinstance(recorded, list) or not isinstance(current, list) or len(recorded) != len(current):
        return False
    pending = []
    if not all(payload_identity_matches(old, new, role=role, observations=pending) for old, new in zip(recorded, current)):
        return False
    if observations is not None:
        observations.extend(pending)
    return True
