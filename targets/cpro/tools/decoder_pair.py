#!/usr/bin/env python3
"""Export and compare the complete L0 linear and L3 full attention decoder blocks independently."""
import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    export = actions.add_parser('export', help='on the AIR host, export just the two complete decoders')
    check = actions.add_parser('check-air', help='verify the two exported decoder cases before transfer')
    check.add_argument('air', type=Path)
    diagnose = actions.add_parser('diagnose', help='read full tool logs and HiLog for existing board attempts')
    diagnose.add_argument('--run', required=True, type=Path)
    diagnose.add_argument('--hilog', type=Path, help='reuse a device HiLog snapshot for both layers')
    inspect_tail = actions.add_parser('tail-map', help='list source AIR quantized matmuls inside/outside each tail')
    inspect_tail.add_argument('parent', type=Path, help='retained decoder-pair board run')
    capture_tail = actions.add_parser('tail-capture', help='capture real post-attention residuals on 310P')
    capture_tail.add_argument('--source-root', type=Path, required=True)
    capture_tail.add_argument('--config', type=Path, default=Path('config/qwen3_5_moe.yaml'))
    capture_tail.add_argument('--run', type=Path, required=True)
    tail = actions.add_parser('tail-run', help='cut L0/L3 tails from the retained standalone decoder AIRs')
    tail.add_argument('--run', type=Path, required=True)
    tail.add_argument('--parent', type=Path, required=True)
    tail.add_argument('--reference', type=Path, required=True)
    probes = actions.add_parser('tail-projections', help='run individual QLinear probes from a retained decoder tail')
    probes.add_argument('--run', type=Path, required=True)
    probes.add_argument('--parent', type=Path, required=True, help='decoder-tail run')
    probes.add_argument('--layer', choices=(0, 3), type=int, default=0)
    remainder = actions.add_parser('tail-remainder', help='run the routed-expert tail with saved projection outputs')
    remainder.add_argument('--run', type=Path, required=True)
    remainder.add_argument('--projections', type=Path, required=True)
    remainder.add_argument('--runtime-config', type=Path, required=True)
    execute = actions.add_parser('run', help='on the DDK host, pull both cases, compile, run HIAI V2, and score',
                                 epilog='If the original board/attempt.json is missing, use a matching --trace '
                                        'or set SN (board serial) and BIP (HDC server host:port).')
    for action in (export, execute):
        action.add_argument('--run', required=True, type=Path)
        action.add_argument('--baseline', type=Path, help='original logits-only 40-layer run')
    export.add_argument('--source-root', required=True, type=Path)
    export.add_argument('--config', type=Path, default=Path('config/qwen3_5_moe.yaml'))
    export.add_argument('--reference', type=Path, help='measured T64/valid19 310P decoder capture')
    execute.add_argument('--remote-air', type=Path, help='two-case or 42-case AIR on the export host (default: RUN/air)')
    for action in (execute, tail, probes, remainder):
        action.add_argument('--ddk-root', required=True, type=Path)
        action.add_argument('--gdr-source', required=True, type=Path)
        action.add_argument('--tbe-diagnosis', type=Path)
        action.add_argument('--tool', help='override the board HIAI V2 model tool')
    execute.add_argument('--trace', type=Path, help='also compare to token18 from the existing sparse whole-model run')
    a = parser.parse_args()
    if a.action == 'check-air':
        from qwen35_moe_dflash.target.prefill_t64.decoder_pair import check_pair_air
        check_pair_air(a.air.resolve())
        return 0
    if a.action == 'tail-map':
        from qwen35_moe_dflash.target.prefill_t64.decoder_tail import inspect_pair
        inspect_pair(a.parent.resolve())
        return 0
    from qwen35_moe_dflash.common.artifacts import output_path
    run = output_path(a.run)
    if a.action == 'tail-remainder':
        from qwen35_moe_dflash.target.board_tool_inspect import DEFAULT_TOOL
        from qwen35_moe_dflash.target.prefill_t64.decoder_remainder import run_remainder
        ok = run_remainder(a.projections.resolve(), run, a.ddk_root.resolve(), a.gdr_source.resolve(),
                           a.tbe_diagnosis, a.tool or DEFAULT_TOOL, a.runtime_config.resolve(strict=True))
        return 0 if ok else 1
    if a.action == 'tail-capture':
        from qwen35_moe_dflash.target.prefill_t64.chain_reference import capture
        capture(a.source_root.resolve(), a.config, None, run, decoder_tails=True)
        return 0
    if a.action == 'tail-projections':
        from qwen35_moe_dflash.target.board_tool_inspect import DEFAULT_TOOL
        from qwen35_moe_dflash.target.prefill_t64.decoder_projection import probe_layer
        ok = probe_layer(a.parent.resolve(), a.layer, run, a.ddk_root.resolve(), a.gdr_source.resolve(),
                         a.tbe_diagnosis, a.tool or DEFAULT_TOOL)
        return 0 if ok else 1
    if a.action == 'tail-run':
        from qwen35_moe_dflash.target.board_tool_inspect import DEFAULT_TOOL
        from qwen35_moe_dflash.target.prefill_t64.decoder_tail import run_pair
        ok = run_pair(a.parent.resolve(), a.reference.resolve(), run, a.ddk_root.resolve(),
                      a.gdr_source.resolve(), a.tbe_diagnosis, a.tool or DEFAULT_TOOL)
        return 0 if ok else 1
    if a.action == 'diagnose':
        from qwen35_moe_dflash.target.prefill_t64.chain_hilog import diagnose_decoder_pair
        return 0 if diagnose_decoder_pair(run, a.hilog) else 1
    baseline = a.baseline.resolve() if a.baseline else None
    if a.action == 'export':
        from qwen35_moe_dflash.target.prefill_t64.chain_air import export
        from qwen35_moe_dflash.target.prefill_t64.whole_trace import source_reference
        if baseline is None and a.reference is None:
            parser.error('export requires --reference or --baseline')
        ref = source_reference(baseline, a.reference) if baseline else a.reference
        export(a.source_root.resolve(), a.config, ref, run / 'air', decoder_pair=True)
        return 0
    if a.trace and not baseline:
        parser.error('--trace requires --baseline')
    from qwen35_moe_dflash.target.board_tool_inspect import DEFAULT_TOOL
    from qwen35_moe_dflash.target.prefill_t64.decoder_pair import run_pair
    ok = run_pair(run, a.remote_air or run / 'air', baseline, a.ddk_root.resolve(), a.gdr_source.resolve(),
                  a.tbe_diagnosis, a.tool or DEFAULT_TOOL, a.trace)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
