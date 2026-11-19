#!/usr/bin/env python3
"""Prepare a source-bound four-layer T64 Prefill job and optionally export its AIR."""
import argparse
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'model'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True, type=Path, help='Installed private CPro source')
    parser.add_argument('--config', required=True, type=Path, help='Original model YAML')
    parser.add_argument('--fixture', required=True, type=Path, help='T64 Prefill fixture directory')
    parser.add_argument('--output', required=True, type=Path, help='Prepared job directory')
    parser.add_argument('--export', action='store_true', help='Run the prepared AIR export in this Python environment')
    parser.add_argument('--attention-layers', action='store_true',
                        help='export independent whole attention modules on their captured valid19 inputs')
    args = parser.parse_args()
    from qwen35_moe_dflash.target.prefill_t64.prepare import prepare
    job = prepare(args.source_root, args.config, args.fixture, args.output, REPO, attention_layers=args.attention_layers)
    print(f'Prepared four-layer T64 Prefill job at {job}', flush=True)
    if args.export:
        subprocess.run([sys.executable, str(job / 'export_t64.py')], check=True)


if __name__ == '__main__':
    main()
