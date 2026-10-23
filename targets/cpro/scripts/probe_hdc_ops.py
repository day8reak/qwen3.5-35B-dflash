#!/usr/bin/env python3
"""Probe bounded operator contracts using the configured CPro compiler and board."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "model"))

from qwen35_moe_dflash.op_probe import main

if __name__ == "__main__":
    raise SystemExit(main())
