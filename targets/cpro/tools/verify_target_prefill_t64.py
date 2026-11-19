#!/usr/bin/env python3
"""Compare CPU, native 310P, and attention-corrected 310P on a four-layer T64 fixture."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))
from qwen35_moe_dflash.target.prefill_t64.reference import main

if __name__ == '__main__':
    main()
