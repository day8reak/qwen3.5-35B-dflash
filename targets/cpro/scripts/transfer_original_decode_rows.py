#!/usr/bin/env python3
"""Prepare, extract, import or diagnose bounded original decode rows."""
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "model"))

from qwen35_moe_dflash.original_decode_row_transfer import main


if __name__ == "__main__":
    raise SystemExit(main())
