#!/usr/bin/env python3
"""Compile an existing standalone GDR AIR or summarize its recovered CPro vs CPU results."""
import argparse
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))
from qwen35_moe_dflash.common.artifacts import output_path
from qwen35_moe_dflash.target.prefill_t64 import standalone_gdr as gdr


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(required=True, dest='action')
    build = commands.add_parser('compile', help='use gdr_t64.air and its existing input/CPU-golden manifest')
    build.add_argument('--case-dir', type=Path, required=True)
    build.add_argument('--output', type=Path, required=True, help='fresh OMC directory outside the exported case')
    build.add_argument('--gdr-source', type=Path, required=True, help='intended allops op_kernel/chunk_gated_delta_rule.cpp')
    build.add_argument('--ddk-root', type=Path, default=os.environ.get('CPRO_DDK_ROOT'))
    build.add_argument('--ddk-id', default=os.environ.get('CPRO_DDK_ID') or '100.520.020.010')
    build.add_argument('--tbe-diagnosis', type=Path, help='run-local TBE diagnosis used with the selected DDK')
    report = commands.add_parser('report', help='write accuracy.csv for all measured calls and valid tokens')
    report.add_argument('--recovery', type=Path, required=True)
    args = cli.parse_args()
    if args.action == 'report':
        return int(gdr.summarize(output_path(args.recovery)) == 'FAILED')
    if not args.ddk_root:
        cli.error('set CPRO_DDK_ROOT or pass --ddk-root')
    gdr.compile_case(args.case_dir.resolve(), output_path(args.output), args.ddk_root.resolve(),
                     args.ddk_id, args.gdr_source.resolve(), args.tbe_diagnosis)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        sys.exit(f'gdr-t64: {exc}')
