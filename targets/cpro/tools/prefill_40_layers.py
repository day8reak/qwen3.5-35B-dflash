#!/usr/bin/env python3
"""Measure CPro's continuous T64 valid19 prefill and greedy first token versus segmented 310P."""
import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    for name in ('capture', 'export'):
        cmd = actions.add_parser(name)
        for flag in ('source-root', 'config', 'baseline' if name == 'capture' else 'reference', 'output'):
            cmd.add_argument('--' + flag, type=Path, required=True)
    board = actions.add_parser('run')
    for flag in ('compiled', 'config', 'ddk-root', 'output'):
        board.add_argument('--' + flag, type=Path, required=True)
    a = parser.parse_args()
    if a.action == 'capture':
        from qwen35_moe_dflash.target.prefill_t64.chain_reference import capture
        capture(a.source_root.resolve(), a.config, a.baseline.resolve(), a.output)
    elif a.action == 'export':
        from qwen35_moe_dflash.target.prefill_t64.chain_air import export
        export(a.source_root.resolve(), a.config, a.reference, a.output)
    elif a.action == 'run':
        from qwen35_moe_dflash.target.prefill_t64.chain_board import run
        run(a.compiled.resolve(), a.output, a.config.resolve(), a.ddk_root.resolve())


if __name__ == '__main__':
    main()
