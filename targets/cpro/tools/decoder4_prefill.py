#!/usr/bin/env python3
"""Review, export, compile and run the complete L0–L3 T64 decoder prefix."""
import argparse
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest='action', required=True)
    review = stages.add_parser('review')
    review.add_argument('--output', type=Path, required=True)
    air = stages.add_parser('air')
    for stage in (air, review):
        stage.add_argument('--source-root', type=Path, required=True)
    air.add_argument('--config', type=Path, required=True)
    air.add_argument('--reference', type=Path, required=True)
    air.add_argument('--prepared-source', type=Path, help='export these reviewed source files directly')
    build = stages.add_parser('compile')
    build.add_argument('--ddk-root', type=Path, required=True)
    build.add_argument('--gdr-source', type=Path, required=True)
    build.add_argument('--tbe-diagnosis', type=Path)
    board = stages.add_parser('board')
    board.add_argument('--runtime-config', type=Path, required=True)
    for stage in (air, build, board):
        stage.add_argument('--run', type=Path, required=True)
    a = parser.parse_args()
    from qwen35_moe_dflash.common.artifacts import output_path
    if a.action == 'review':
        from qwen35_moe_dflash.target.layer_probe.common import new_output
        from qwen35_moe_dflash.target.prefill_t64.decoder_four_source import stage
        out = new_output(a.output)
        stage(a.source_root.resolve(), out)
        print(out)
        return 0
    run = output_path(a.run)
    if a.action == 'air':
        from qwen35_moe_dflash.target.prefill_t64.whole_export import export
        export(a.source_root.resolve(), a.config, a.reference, run / 'air',
               four_decoders=True, prepared_source=a.prepared_source)
        return 0
    from qwen35_moe_dflash.target.prefill_t64 import whole_run
    from qwen35_moe_dflash.target.prefill_t64.decoder_pair import load
    meta = load(run / 'air/manifest.json')
    if meta['kind'] != whole_run._FOUR_DECODERS or meta['layers'] != [0, 1, 2, 3]:
        raise ValueError('Expected one complete L0–L3 decoder AIR')
    if a.action == 'compile':
        whole_run.compile_model(run, a.ddk_root.resolve(), a.gdr_source.resolve(), a.tbe_diagnosis)
        return 0
    from qwen35_moe_dflash.target.board_tool_inspect import DEFAULT_TOOL
    from qwen35_moe_dflash.target.prefill_t64.chain_hilog import diagnose_decoder_pair
    case = Path(tempfile.mkdtemp(prefix='replay-', dir=run))
    out = case / 'L0-L3/board'
    print(f'Board attempt: {out}', flush=True)
    try:
        result = whole_run.board(run, DEFAULT_TOOL, slot_config=a.runtime_config.resolve(strict=True), board_dir=out)
    except whole_run.HiaiRunFailure:
        diagnose_decoder_pair(case, names=('L0-L3',))
        raise
    for scope in ('valid19', 'token18'):
        m = result[scope]
        print(f"L0–L3 {scope}: relative_l2={m['relative_l2']:.4%} cosine={m['cosine']:.9f}")
    print(out / 'accuracy.json')
    return 0 if result['status'] == 'PASSED' else 1


if __name__ == '__main__':
    sys.exit(main())
