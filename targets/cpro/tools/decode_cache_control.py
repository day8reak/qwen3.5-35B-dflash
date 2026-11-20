#!/usr/bin/env python3
"""Test the first L3 decode with CPU-updated K, V, or both caches."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))
from qwen35_moe_dflash.target.prefill_t64.decode_cache_control import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    cpu = commands.add_parser('cpu', help='compute post-update KV on the reference host')
    cpu.add_argument('--fixture', required=True, type=Path)
    cpu.add_argument('--stages', action='store_true', help='also save the CPU query, core and o_proj input')
    trace = commands.add_parser('trace', help='expose native cache writes and the CPU-KV attention boundaries')
    inspect = commands.add_parser('inspect', help='compare recovered trace tensors with their original board runs')
    for cmd in (trace, inspect):
        cmd.add_argument('--baseline', required=True, type=Path, help='original cpro batch directory')
        cmd.add_argument('--control', required=True, type=Path, help='prior cache-control batch directory')
    trace.add_argument('--cpu', required=True, type=Path)
    inspect.add_argument('--trace', required=True, type=Path)
    batch = commands.add_parser('prepare', help='derive three diagnostic AIRs from the original decode AIR')
    batch.add_argument('--batch', required=True, type=Path)
    batch.add_argument('--cpu', required=True, type=Path)
    for command in (cpu, batch, trace):
        command.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.action == 'cpu':
        build('cpu', args.fixture, args.output, stages=args.stages)
    elif args.action == 'trace':
        build('trace', args.baseline / 'air', args.output, args.cpu, control=args.control / 'air')
    elif args.action == 'inspect':
        from qwen35_moe_dflash.target.prefill_t64.decode_cache_trace import inspect_trace
        inspect_trace(args.baseline, args.control, args.trace)
    else:
        build('prepare', args.batch, args.output, args.cpu)


if __name__ == '__main__':
    main()
