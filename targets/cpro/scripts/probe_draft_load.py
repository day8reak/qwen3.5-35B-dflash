#!/usr/bin/env python3
"""Inspect HIAI descriptor path/buffer setters; no AIR/OMC compilation or model Load."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "model"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", help="completed external draft job; default AI_RUN_DIR/full-draft")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    args = parser.parse_args(argv)
    from qwen35_moe_dflash.descriptor_probe import run_probe
    try:
        run = os.environ.get("AI_RUN_DIR")
        if not run:
            raise ValueError("AI_RUN_DIR is required; source the project env.sh")
        result = run_probe(args.job or Path(run) / "full-draft", timeout_seconds=args.timeout_seconds)
    except Exception as error:
        print(f"draft-load-probe: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if result["status"] != "INSPECTED":
        for name in ("runner-build.log", "probe.stderr"):
            path = Path(result["output"]) / name
            if path.is_file():
                from qwen35_moe_dflash.tool_logs import log_excerpt
                print(f"{path}\n{log_excerpt(path)}", flush=True)
    return 0 if result["status"] == "INSPECTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
