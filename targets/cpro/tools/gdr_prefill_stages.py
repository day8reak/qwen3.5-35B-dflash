#!/usr/bin/env python3
"""Read the GDR, RMSNorm and SiLU product boundaries for prefill T64/valid19."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cmds = parser.add_subparsers(dest='action', required=True)
    bundle = cmds.add_parser('summarize', help='assemble all three layers from the recovered baseline, split L0 and paired SiLU runs')
    bundle.add_argument('--group', required=True, type=Path, help='original gdr-stages-t64-valid19 experiment directory')
    bundle.add_argument('--output', type=Path, help='new dump directory (default: GROUP/dump-summary)')
    audit = cmds.add_parser('audit-dump', help='check the original nine-case dump against its own recorded operands')
    audit.add_argument('--dump', required=True, type=Path)
    audit.add_argument('--trace', required=True, type=Path, help='recovered nine-case board batch')
    audit.add_argument('--layers', nargs='+', type=int, choices=range(3), default=[0, 1, 2])
    product = cmds.add_parser('audit-product', help='attribute the measured gated product gap from qualified split probes')
    product.add_argument('--dump', required=True, type=Path)
    product.add_argument('--layer', type=int, choices=range(3), default=0)
    inspect = cmds.add_parser('inspect-abi', help='print the retained AIR and compiled OMC output slots without running the model')
    inspect.add_argument('--compiled', required=True, type=Path, help='OMC batch directory')
    inspect.add_argument('--ddk-root', required=True, type=Path)
    inspect.add_argument('--case', default='l00_c00_norm_prefill_stages_o0')
    prepare = cmds.add_parser('prepare', help='expose the measured stage edges in the baseline AIRs')
    compare = cmds.add_parser('compare', help='write per-backend raw binary tensors, a file map and CPro-vs-310P CSV metrics')
    prepare.add_argument('--baseline', required=True, type=Path, help='export job containing the whole-module AIR batch')
    prepare.add_argument('--include-baseline', action='store_true', help='include the three fresh whole modules for a single nine-case board run')
    prepare.add_argument('--split-norm-layers', type=int, nargs='+', choices=range(3),
                         help='read one distinct RMSNorm/gate edge per graph alongside the original hidden')
    prepare.add_argument('--indexed-silu-layer', type=int, choices=range(3),
                         help='export just the indicated layer with a runtime identity gather on the SiLU output')
    prepare.add_argument('--paired-silu', action='store_true',
                         help='also return the actual SiLU input from the same indexed graph for a direct formula check')
    compare.add_argument('--silu-readout', type=Path, help='use a recovered indexed SiLU case in the split dump')
    compare.add_argument('--baseline', type=Path, help='whole-module board batch (default: same batch as --trace)')
    for cmd in (prepare, compare):
        cmd.add_argument('--reference', required=True, type=Path, help='310P prefill capture with measured GDR and norm stages')
        cmd.add_argument('--output', required=True, type=Path)
    compare.add_argument('--trace', required=True, type=Path, help='new compiled and recovered batch')
    args = parser.parse_args()
    if args.action == 'summarize':
        from qwen35_moe_dflash.target.prefill_t64.gdr_stage_report import compare
        g = args.group
        compare(g / 'gamma-readout', g / 'reference', g / 'norm-single-l0', args.output or g / 'dump-summary',
                silu_readout=g / 'l0-silu-pair', combined=True)
        return 0
    if args.action == 'audit-product':
        from qwen35_moe_dflash.target.prefill_t64.gdr_product_audit import audit_product
        audit_product(args.dump, args.layer)
        return 0
    if args.action == 'audit-dump':
        from qwen35_moe_dflash.target.prefill_t64.gdr_readback_audit import audit_dump
        audit_dump(args.dump, args.trace, args.layers)
        return 0
    if args.action == 'inspect-abi':
        from qwen35_moe_dflash.target.prefill_t64.gdr_stage_abi import inspect_abi
        return 0 if inspect_abi(args.compiled, args.ddk_root, args.case) else 1
    if args.action == 'prepare':
        from qwen35_moe_dflash.target.prefill_t64.gdr_stage_air import prepare
        prepare(args.baseline, args.reference, args.output, include_baseline=args.include_baseline,
                split_norm_layers=args.split_norm_layers, indexed_silu_layer=args.indexed_silu_layer, paired_silu=args.paired_silu)
    else:
        from qwen35_moe_dflash.target.prefill_t64.gdr_stage_report import compare
        result = compare(args.baseline or args.trace, args.reference, args.trace, args.output, silu_readout=args.silu_readout)
        if result is False:
            return 1


if __name__ == '__main__':
    raise SystemExit(main())
