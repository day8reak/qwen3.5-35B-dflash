#!/usr/bin/env python3
"""Export the L0 GDR power-of-two scale control on the original prepared module inputs."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))
from qwen35_moe_dflash.target.prefill_t64.gdn_scale_control import export_control

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True, help='prepared attention-layers-t64-valid19 job')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--readout-scale', type=int, choices=(1, 16),
                   help='apply this factor to normalized Q only, keeping V and state at their original scale')
    args = p.parse_args()
    print(json.dumps(export_control(args.baseline, args.output, readout_scale=args.readout_scale)), flush=True)
