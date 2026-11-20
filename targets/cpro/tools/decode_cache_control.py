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
    batch = commands.add_parser('prepare', help='derive three diagnostic AIRs from the original decode AIR')
    batch.add_argument('--batch', required=True, type=Path)
    batch.add_argument('--cpu', required=True, type=Path)
    for command in (cpu, batch):
        command.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.action == 'cpu':
        build('cpu', args.fixture, args.output)
    else:
        build('prepare', args.batch, args.output, args.cpu)


if __name__ == '__main__':
    main()
