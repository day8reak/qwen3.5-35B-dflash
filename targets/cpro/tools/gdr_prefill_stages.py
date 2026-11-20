#!/usr/bin/env python3
"""Read the GDR, RMSNorm and SiLU product boundaries for prefill T64/valid19."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cmds = parser.add_subparsers(dest='action', required=True)
    prepare = cmds.add_parser('prepare', help='expose the measured stage edges in the baseline AIRs')
    compare = cmds.add_parser('compare', help='write per-backend raw binary tensors, a file map and CPro-vs-310P CSV metrics')
    for cmd in (prepare, compare):
        cmd.add_argument('--baseline', required=True, type=Path, help='original unscaled whole-module board batch')
        cmd.add_argument('--reference', required=True, type=Path, help='310P prefill capture with measured GDR and norm stages')
        cmd.add_argument('--output', required=True, type=Path)
    compare.add_argument('--trace', required=True, type=Path, help='new compiled and recovered batch')
    args = parser.parse_args()
    if args.action == 'prepare':
        from qwen35_moe_dflash.target.prefill_t64.gdr_stage_air import prepare
        prepare(args.baseline, args.reference, args.output)
    else:
        from qwen35_moe_dflash.target.prefill_t64.gdr_stage_report import compare
        compare(args.baseline, args.reference, args.trace, args.output)


if __name__ == '__main__':
    main()
