#!/usr/bin/env python3
"""Run one isolated residual repair with retained draft weights and goldens."""
import sys

from run_draft_trace import main


if __name__ == "__main__":
    raise SystemExit(main([*sys.argv[1:], "--direct", "--residual-policy", "cpu-fp32-add"]))
