#!/usr/bin/env python3
"""Host-only GDN FP16/FP32 diagnostic using synthetic or captured NPZ inputs."""
from pathlib import Path
import sys

repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo))
sys.path.insert(0, str(repo / "model"))

from qwen35_moe_dflash.backends import configure_reference_backend

configure_reference_backend("cpu")

from reference.gdn_precision import main

if __name__ == "__main__":
    raise SystemExit(main())
