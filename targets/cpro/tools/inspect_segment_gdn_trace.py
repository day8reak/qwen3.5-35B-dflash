#!/usr/bin/env python3
"""Inspect one recovered physical64/valid19 whole GDN trace against its original board run."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))
from qwen35_moe_dflash.target.prefill_t64.segment_trace_compare import compare, summarize_hidden

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path, help='original attention-layers group')
    parser.add_argument('--trace', type=Path, help='trace group with compiled batch, execution and recovery')
    args = parser.parse_args()
    summarize_hidden(args.baseline)
    if args.trace:
        compare(args.baseline, args.trace)
