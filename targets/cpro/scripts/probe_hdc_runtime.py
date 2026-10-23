#!/usr/bin/env python3
"""Compare HIAI runtime deployments without recompiling existing probe cases."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "model"))

from qwen35_moe_dflash.hiai_runtime_probe import main

if __name__ == "__main__":
    raise SystemExit(main())
