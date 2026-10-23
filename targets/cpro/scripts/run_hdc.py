#!/usr/bin/env python3
"""Upload and validate an OMC pair using the selected SN/BIP HDC connection."""
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "model"))

from qwen35_moe_dflash.hdc_run import main

if __name__ == "__main__":
    raise SystemExit(main())
