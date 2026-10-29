#!/usr/bin/env python3
"""Probe an observed CCE directory predicate and short Python IPC paths."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "model"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, help="retained target compile job; read-only source metadata")
    parser.add_argument("--expected-ccec-sha256", required=True, help="SHA256 reported for this exact job's ccec.py")
    parser.add_argument("--output", required=True, help="new probe directory below AI_RUN_DIR, outside retained job")
    parser.add_argument("--python", help="existing Python interpreter; defaults to the interpreter executing this command")
    args = parser.parse_args(argv)
    from qwen35_moe_dflash.tbe_tmpdir_probe import probe_tbe_tmpdir
    report = probe_tbe_tmpdir(args.job, args.output, expected_ccec_sha256=args.expected_ccec_sha256, python=args.python)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if report["status"] == "PROBE_PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
