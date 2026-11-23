#!/usr/bin/env python3
"""Export the original CPro 40-layer wrapper as one model and read its greedy first token."""
import argparse
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest='stage', required=True)
    for name in ('capture', 'air', 'compile', 'board'):
        p = stages.add_parser(name)
        p.add_argument('--run', type=Path, required=True)
        if name in ('capture', 'air'):
            p.add_argument('--source-root', type=Path, required=True)
            p.add_argument('--config', type=Path, default=Path('config/qwen3_5_moe.yaml'))
        if name == 'air':
            p.add_argument('--reference', type=Path, help='existing complete 40-layer 310P capture')
        if name == 'compile':
            p.add_argument('--ddk-root', type=Path, required=True)
            p.add_argument('--gdr-source', type=Path, required=True)
            p.add_argument('--tbe-diagnosis', type=Path)
        if name == 'board':
            from qwen35_moe_dflash.target.board_tool_inspect import DEFAULT_TOOL
            p.add_argument('--tool', default=DEFAULT_TOOL)
    slots = stages.add_parser('check-slots', help='Read HIAI V2 descriptors of the previously deployed OMC')
    slots.add_argument('--run', type=Path, required=True)
    slots.add_argument('--config', type=Path, help='runtime.json used for the board environment')
    args = parser.parse_args()
    from qwen35_moe_dflash.common.artifacts import output_path
    run = output_path(args.run)
    if args.stage == 'capture':
        from qwen35_moe_dflash.target.prefill_t64.chain_reference import capture
        capture(args.source_root.resolve(), args.config, None, run / 'reference')
    elif args.stage == 'air':
        from qwen35_moe_dflash.target.prefill_t64.whole_export import export
        export(args.source_root.resolve(), args.config, args.reference or run / 'reference', run / 'air')
    elif args.stage == 'compile':
        from qwen35_moe_dflash.target.prefill_t64.whole_run import compile_model
        compile_model(run, args.ddk_root.resolve(), args.gdr_source.resolve(), args.tbe_diagnosis)
    elif args.stage == 'check-slots':
        from qwen35_moe_dflash.target.prefill_t64.whole_slots import check_slots
        config = args.config or os.environ.get('RUNTIME_CONFIG') or Path(os.environ['AI_RUN_DIR']) / 'runtime/runtime.json'
        verdict = check_slots(run, Path(config).resolve())
        if verdict['status'] not in {'INPUT_ORDER_MATCH', 'EQUIVALENT_INPUT_BYTES'}:
            raise SystemExit(1)
    else:
        from qwen35_moe_dflash.target.prefill_t64.whole_run import board
        board(run, args.tool)


if __name__ == '__main__':
    main()
