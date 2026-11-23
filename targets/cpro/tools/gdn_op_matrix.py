#!/usr/bin/env python3
"""Review, prepare and run independent L0 operators on the same native 310P inputs."""
import argparse
from pathlib import Path
import sys

from gdn_tail_accuracy import board, board_arguments
from qwen35_moe_dflash.common.artifacts import output_path
from qwen35_moe_dflash.target.prefill_t64 import gdn_op_matrix as ops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    review = actions.add_parser('review', help='list exact boundaries and all GE nodes, entirely offline')
    review.add_argument('--air', type=Path, required=True)
    prep = actions.add_parser('prepare', help='78: capture the three native operands and generate fixtures')
    prep.add_argument('--source-air', type=Path, required=True, help='original attention AIR batch or previous six-prefix L0 tail AIR batch')
    audit = actions.add_parser('audit-proj', help='78: audit live quantized operands against a previous matrix run')
    for cmd in (prep, audit):
        cmd.add_argument('--source-root', type=Path, required=True, help='original segmented 310P source')
        cmd.add_argument('--config', type=Path, default=Path('config/qwen3_5_moe.yaml'))
    execute = actions.add_parser('board', help='35: compile the three OMCs, run them and report the matrices')
    board_arguments(execute)
    report = actions.add_parser('report', help='recalculate the matrices from saved board results')
    for cmd in (prep, execute, report, audit):
        cmd.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'review':
        ops.review(args.air.resolve())
        return 0
    run = output_path(args.run)
    if args.action == 'prepare':
        ops.prepare(run, args.source_air.resolve(), args.source_root.resolve(), args.config)
        return 0
    if args.action == 'audit-proj':
        return 0 if ops.audit_projection(run, args.source_root.resolve(), args.config) else 1
    if args.action == 'board':
        return 0 if board(run, args, stages=tuple(ops.ROLES), summarize=ops.summarize, basis=ops.BASIS) else 1
    return 0 if ops.summarize(run) else 1


if __name__ == '__main__':
    sys.exit(main())
