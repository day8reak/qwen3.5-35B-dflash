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
    parser.add_argument('--core-precision', action='store_true',
                        help='test GDR core substitution at hidden and fixed-input numerical hypotheses offline')
    args = parser.parse_args()
    if args.core_precision and not args.trace:
        parser.error('--core-precision requires --trace')
    if not args.core_precision:
        summarize_hidden(args.baseline)
    if args.trace:
        compare(args.baseline, args.trace, core_precision=args.core_precision)
