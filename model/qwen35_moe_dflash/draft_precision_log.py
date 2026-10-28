"""Bounded, hash-bound compiler messages for selected source operators."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re

from .draft_compile_inspect import _bound, _read


def inspect_precision_log(compile_root, compiled, target_names):
    """Return contextual messages, never inferred placement or kernel behavior."""
    result = {"status": "NOT_AVAILABLE", "lines": [], "source": None,
              "scope": "Exact source-node compiler messages only; per-store support checks do not prove final placement or runtime arithmetic."}
    try:
        digest = compiled.get("compile_log_sha256")
        relative = compiled.get("compile_log")
        if (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
                or not isinstance(relative, str)):
            result["reason"] = "No hash-bound compiler log receipt"
            return result
        names = sorted(set(target_names))
        if (len(names) > 32 or any(not isinstance(n, str) or not n or len(n) > 256 for n in names)):
            raise ValueError("final-normalization name selection exceeds its bound")
        if not names:
            result["reason"] = "No unambiguous final-normalization source names"
            return result
        sources = []
        raw = _read(_bound(Path(compile_root), {"path": relative}), 64 << 20,
                    "compiler log", sources)
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("compiler log SHA256 differs from the compilation receipt")
        result["source"] = sources[0]
        matcher = re.compile(r"(?<![A-Za-z0-9_./-])(?:" + "|".join(map(re.escape, names))
                             + r")(?![A-Za-z0-9_./-])")
        lines = raw.decode("utf-8", errors="replace").splitlines()
        hits = [i for i, line in enumerate(lines) if matcher.search(line)]
        selected = set()
        detail = re.compile(r"dtype|data.?type|support|cast|precision|failed|reject|kernel|cpucl|npucl", re.I)
        detailed = [i for i in hits if detail.search(lines[i])]
        anchors = list(dict.fromkeys([*detailed[-12:], *hits[-8:]]))
        # Preserve type/support context even when later GetDataParam messages
        # repeat a node name. A per-store rejection is not a compile failure.
        for i in reversed(anchors):
            selected.add(i)
        for i in reversed(anchors):
            for j in (i - 1, i + 1, i - 2, i + 2):
                if len(selected) >= 60:
                    break
                if 0 <= j < len(lines):
                    selected.add(j)
        result.update(status="INSPECTED", target_names=names, matched_lines=len(hits),
                      truncated=len(hits) > len(anchors),
                      lines=[{"line": i + 1, "text": lines[i][:512],
                              "exact_node_match": bool(matcher.search(lines[i]))}
                             for i in sorted(selected)])
    except (OSError, ValueError, TypeError) as error:
        result.update(reason=f"{type(error).__name__}: {error}", lines=[])
    return result
