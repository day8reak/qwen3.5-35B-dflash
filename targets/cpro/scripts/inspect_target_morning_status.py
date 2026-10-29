#!/usr/bin/env python3
"""Print a read-only morning summary of retained target evidence."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "model"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prefill-probe", "decode-probe", "prefill-plan", "prefill-smoke"):
        parser.add_argument("--" + name, help="existing stage directory; default uses the corresponding name under AI_RUN_DIR")
    args = parser.parse_args(argv)
    from qwen35_moe_dflash.target_morning_status import inspect_target_morning_status
    try:
        report = inspect_target_morning_status(**vars(args))
        print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
        return 1 if report["next"] == "INSPECT_RETAINED_FAILURE" else 0
    except (OSError, ValueError, TypeError) as error:
        print(json.dumps({"status": "INVALID_METADATA", "actions_executed": "NONE", "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
