"""Actual executor: bounded ND singleton padding preserves the locked ABI."""
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
for layout in ("", "padded", "padded-mixed", "leading-feature", "mixed-feature",
               "non-nd-leading-feature", "leading-one", "middle-one", "rank-five",
               "rank-three-token", "non-one-tail", "zero-tail", "non-nd", "permuted",
               "draft-width-tail", "draft-feature-width"):
    report = root / (layout or "original")
    trace = root / (report.name + ".trace")
    init_trace = root / (report.name + ".init-trace")
    result = subprocess.run(base + ["--output", str(report)], capture_output=True, text=True,
        env=dict(os.environ, FAKE_HIAI_FAILURE="", FAKE_HIAI_LAYOUT=layout, FAKE_HIAI_TRACE=str(trace),
                 FAKE_HIAI_INIT_TRACE=str(init_trace)))
    assert "LEAK" not in result.stderr, result.stderr
    events = trace.read_text().splitlines()
    assert events.count("load") == events.count("unload") == 2
    if layout in ("", "padded", "padded-mixed", "leading-feature", "mixed-feature"):
        assert result.returncode == 0, result.stderr
        data = json.loads(report.read_text())
        assert data["generation"]["generated_token_ids"] == list(range(3, 11))
        assert data["sequence_length"] == 32
        if layout in ("padded", "padded-mixed", "mixed-feature"):
            assert "accepted ND trailing-one padding" in result.stderr
        if layout in ("leading-feature", "mixed-feature"):
            assert result.stderr.count("accepted ND leading-one padding") == (2 if layout == "leading-feature" else 1)
            # Target feature bytes must survive transfer to a draft that may use
            # a different singleton convention. Fake Process checks both ends.
            shapes = [[1, 32], [1, 32], [1, 32], [1, 1, 32, 32],
                      [1, 1, 32, 32] if layout == "leading-feature" else [1, 32, 32, 1],
                      [1, 32], [1, 32], [1, 3]]
            if layout == "mixed-feature":
                shapes = [s + [1] * (4 - len(s)) for s in shapes]
            assert init_trace.read_text().splitlines() == [
                "format=2;shape=" + "".join(f"{d}," for d in s) for s in shapes]
        if layout:
            assert "using runtime descriptor" in result.stderr
    else:
        assert result.returncode == 1 and not report.exists(), (layout, result.stderr)
        assert "process" not in events, (layout, events)
        assert "shape" in result.stderr, result.stderr
print("HIAI ND padding: original/padded execution and incompatible layouts checked")
