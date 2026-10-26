"""Actual executor: only ND trailing-one padding preserves the locked ABI."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

runner = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2])
root.mkdir(parents=True, exist_ok=True)
root = Path(tempfile.mkdtemp(prefix="case-", dir=root))
target, draft = root / "target.omc", root / "draft.omc"
target.write_bytes(b"target-fixture")
draft.write_bytes(b"draft-fixture")
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
base = [str(runner), "--mode", "generate", "--target", str(target),
        "--target-sha256", sha(target), "--draft", str(draft), "--draft-sha256", sha(draft),
        "--expected-runtime-version", "FAKE-HIAI-X90", "--ddk-id", "fake", "--firmware-id", "fake",
        "--feature-width", "32", "--vocab-size", "64", "--prompt-token-ids", "1,2",
        "--pad-token-id", "0", "--eos-token-ids", "10", "--max-new-tokens", "12"]
for layout in ("", "padded", "padded-mixed", "leading-one", "middle-one", "rank-five",
               "rank-three-token", "non-one-tail", "zero-tail", "non-nd", "permuted",
               "draft-width-tail", "draft-feature-width"):
    report = root / (layout or "original")
    trace = root / (report.name + ".trace")
    result = subprocess.run(base + ["--output", str(report)], capture_output=True, text=True,
        env=dict(os.environ, FAKE_HIAI_FAILURE="", FAKE_HIAI_LAYOUT=layout, FAKE_HIAI_TRACE=str(trace)))
    assert "LEAK" not in result.stderr, result.stderr
    events = trace.read_text().splitlines()
    assert events.count("load") == events.count("unload") == 2
    if layout in ("", "padded", "padded-mixed"):
        assert result.returncode == 0, result.stderr
        data = json.loads(report.read_text())
        assert data["generation"]["generated_token_ids"] == list(range(3, 11))
        assert data["sequence_length"] == 32
        if layout:
            assert "accepted ND trailing-one padding" in result.stderr
            assert "using runtime descriptor" in result.stderr
    else:
        assert result.returncode == 1 and not report.exists(), (layout, result.stderr)
        assert "process" not in events, (layout, events)
        assert "shape" in result.stderr, result.stderr
print("HIAI ND padding: original/padded execution and incompatible layouts checked")
