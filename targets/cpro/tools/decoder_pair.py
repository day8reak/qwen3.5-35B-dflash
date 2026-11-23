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
    execute = actions.add_parser('run', help='on the DDK host, pull both cases, compile, run HIAI V2, and score')
    for action in (export, execute):
        action.add_argument('--run', required=True, type=Path)
        action.add_argument('--baseline', required=True, type=Path, help='original logits-only 40-layer run')
    export.add_argument('--source-root', required=True, type=Path)
    export.add_argument('--config', type=Path, default=Path('config/qwen3_5_moe.yaml'))
    export.add_argument('--reference', type=Path, help='310P capture used by the baseline')
    execute.add_argument('--remote-air', type=Path, help='two-case or 42-case AIR on the export host (default: RUN/air)')
    execute.add_argument('--ddk-root', required=True, type=Path)
    execute.add_argument('--gdr-source', required=True, type=Path)
    execute.add_argument('--tbe-diagnosis', type=Path)
    execute.add_argument('--trace', type=Path, help='also compare to token18 from the existing sparse whole-model run')
    execute.add_argument('--tool', help='override the board HIAI V2 model tool')
    a = parser.parse_args()
    from qwen35_moe_dflash.common.artifacts import output_path
    run = output_path(a.run)
    baseline = a.baseline.resolve()
    if a.action == 'export':
        from qwen35_moe_dflash.target.prefill_t64.chain_air import export
        from qwen35_moe_dflash.target.prefill_t64.whole_trace import source_reference
        ref = source_reference(baseline, a.reference)
        export(a.source_root.resolve(), a.config, ref, run / 'air', decoder_pair=True)
        return 0
    from qwen35_moe_dflash.target.board_tool_inspect import DEFAULT_TOOL
    from qwen35_moe_dflash.target.prefill_t64.decoder_pair import run_pair
    ok = run_pair(run, a.remote_air or run / 'air', baseline, a.ddk_root.resolve(), a.gdr_source.resolve(),
                  a.tbe_diagnosis, a.tool or DEFAULT_TOOL, a.trace)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
