"""Compare historical small metadata with freshly hashed, stable observations.

Callers must obtain ``current`` from their bounded, stable metadata reader in
this invocation. This module does not read files or repair historical receipts.
Only explicitly assigned evidence roles admit a ctime-only difference; raw
payloads, executable source, model graphs and weights are outside this policy.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re


MAX_METADATA_BYTES = 32 << 20
_JOB_NAMES = frozenset({"result.json", "report.json", "compile-profile.json"})
_PROBE_NAMES = _JOB_NAMES | {"model-io.json", "runtime.json", "model-io-command.json",
    "runtime.exitcode", "model-io.exitcode"}
_EXECUTION_NAMES = frozenset({"result.json", "plan.json", "runtime-decode-proof.json", "command.json"})
_JSON_EVIDENCE_NAMES = (_PROBE_NAMES - {"runtime.exitcode", "model-io.exitcode"}) | _EXECUTION_NAMES | {"manifest.json"}
_ROLE_NAMES = {
    "runtime-probe-source-metadata": _JOB_NAMES,
    "decode-plan-metadata": _PROBE_NAMES,
    "decode-execution-metadata": _EXECUTION_NAMES,
    "decode-trial-manifest": frozenset({"manifest.json"}),
    "decode-output-inventory-metadata": _JSON_EVIDENCE_NAMES,
    "embedding-audit-metadata": _JSON_EVIDENCE_NAMES,
    "row-transfer-request-metadata": frozenset({"request.json"}),
    "row-transfer-response-metadata": frozenset({"response.json"}),
    "row-transfer-prepared-manifest-metadata": frozenset({"manifest.json"}),
}


def _collector(observations):
    if observations is not None and not isinstance(observations, list):
        raise ValueError("metadata revalidation observations must be a list")


def _identity(value, role):
    if (not isinstance(role, str) or not isinstance(value, dict) or not {"path", "sha256", "stat"} <= value.keys()
            or not value.keys() <= {"path", "bytes", "sha256", "stat"}):
        return False
    path, stats = value["path"], value["stat"]
    if (not isinstance(path, str) or not Path(path).is_absolute() or ".." in Path(path).parts
            or str(Path(path)) != path or not re.fullmatch(r"[0-9a-f]{64}", str(value["sha256"]))
            or not isinstance(stats, (list, tuple)) or len(stats) != 5
            or any(type(item) is not int or item < 0 for item in stats)
            or not 0 < stats[2] <= MAX_METADATA_BYTES
            or ("bytes" in value and (type(value["bytes"]) is not int or value["bytes"] != stats[2]))):
        return False
    if role == "embedding-config-metadata":
        return Path(path).suffix in {".yaml", ".yml"}
    return role in _ROLE_NAMES and Path(path).name in _ROLE_NAMES[role]


def metadata_identity_matches(recorded, current, *, role, observations=None):
    """Match one role-bound identity; ``current`` must be freshly SHA-verified.

    Legacy recorded identities may omit ``bytes``; their recorded stat extent
    remains mandatory. Unknown roles/paths and any other difference return false.
    """
    _collector(observations)
    if not _identity(recorded, role) or not _identity(current, role):
        return False
    if (recorded["path"] != current["path"] or recorded["sha256"] != current["sha256"]
            or list(recorded["stat"][:4]) != list(current["stat"][:4])):
        return False
    if recorded["stat"][4] != current["stat"][4] and observations is not None:
        observations.append({"kind": "METADATA_CTIME_ONLY_REVALIDATED", "status": "SHA256_VERIFIED",
            "role": role, "path": current["path"], "bytes": current["stat"][2], "sha256": current["sha256"],
            "recorded_stat": list(recorded["stat"]), "current_stat": list(current["stat"]),
            "receipt_mutation": "NONE", "scope": "FRESH_BOUNDED_STABLE_METADATA_SHA256_ONLY"})
    return True


def metadata_sequence_matches(recorded, current, *, role, observations=None):
    """Match an ordered identity list without changing either list or its rows."""
    _collector(observations)
    if not isinstance(recorded, list) or not isinstance(current, list) or len(recorded) != len(current):
        return False
    pending = []
    if not all(metadata_identity_matches(old, new, role=role, observations=pending)
               for old, new in zip(recorded, current)):
        return False
    if observations is not None:
        observations.extend(deepcopy(pending))
    return True
