#!/usr/bin/env python3
"""Measure CPro's continuous T64 valid19 prefill and greedy first token versus segmented 310P."""
import argparse
from contextlib import closing
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
    for name in ('compile', 'compile-run'):
        cmd = actions.add_parser(name, help='compile each case and release its local AIR' +
                                 (' and run it before compiling the next case' if name == 'compile-run' else ''))
        for flag in ('run', 'ddk-root', 'tbe-diagnosis'):
            cmd.add_argument('--' + flag, type=Path, required=True)
        cmd.add_argument('--ddk-id', required=True)
        if name == 'compile-run':
            cmd.add_argument('--config', type=Path, required=True)
            cmd.add_argument('--remote-air', type=Path, help='AIR directory on the export server (defaults to RUN/air)')
            cmd.add_argument('--gdr-sha256', required=True, help='expected installed GDR C++ SHA-256')
    verify = actions.add_parser('check-air', help='check all 42 export records, fixtures and weight file extents')
    verify.add_argument('--air', type=Path, required=True)
    probe = actions.add_parser('diagnose', help='inspect the failed decoder request in device HiLog')
    probe.add_argument('--case-dir', type=Path, required=True)
    probe.add_argument('--hilog', type=Path, help='use an existing hilog -x snapshot')
    v2 = actions.add_parser('replay-v2', help='test the failed decoder OMC and its retained inputs once via HIAI V2')
    v2.add_argument('--case-dir', type=Path, required=True)
    v2.add_argument('--external-build', type=Path, help='use a matched L0 external-weight compilation')
    external = actions.add_parser('compile-l0-external', help='repeat the failed L0 V2 build with external weights')
    external.add_argument('--replay-dir', type=Path, required=True, help='failed embedded V2 replay directory')
    external.add_argument('--remote-air', type=Path, required=True, help='original split AIR root on the export host')
    external.add_argument('--output', type=Path, required=True)
    v2_logs = actions.add_parser('diagnose-v2', help='read the device log window for an existing V2 replay')
    v2_logs.add_argument('--replay-dir', type=Path, required=True)
    v2_logs.add_argument('--hilog', type=Path, help='use an existing hilog -x snapshot')
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
    elif a.action == 'check-air':
        from qwen35_moe_dflash.target.prefill_t64.chain_compile import check_air
        check_air(a.air.resolve())
    elif a.action == 'compile':
        from qwen35_moe_dflash.target.prefill_t64.chain_compile import compile_chain
        compile_chain(a.run, a.ddk_root, a.ddk_id, a.tbe_diagnosis)
    elif a.action == 'compile-run':
        from qwen35_moe_dflash.target.prefill_t64.chain_board import run
        from qwen35_moe_dflash.target.prefill_t64.chain_compile import compiled_cases
        root, ddk = a.run.resolve(), a.ddk_root.resolve()
        with closing(compiled_cases(root, ddk, a.ddk_id, a.tbe_diagnosis, gdr_sha256=a.gdr_sha256,
                                    remote_air=a.remote_air)) as cases:
            run(root / 'omc', root / 'board', a.config.resolve(), ddk, compiling=cases)
    elif a.action == 'diagnose':
        from qwen35_moe_dflash.target.prefill_t64.chain_hilog import diagnose
        diagnose(a.case_dir, a.hilog)
    elif a.action == 'replay-v2':
        from qwen35_moe_dflash.target.prefill_t64.chain_v2 import replay
        replay(a.case_dir, external_build=a.external_build)
    elif a.action == 'compile-l0-external':
        from qwen35_moe_dflash.target.prefill_t64.chain_external import compile_external
        compile_external(a.replay_dir, a.remote_air, a.output)
    elif a.action == 'diagnose-v2':
        from qwen35_moe_dflash.target.prefill_t64.chain_hilog import diagnose_v2
        diagnose_v2(a.replay_dir, a.hilog)
    elif a.action == 'run':
        from qwen35_moe_dflash.target.prefill_t64.chain_board import run
        run(a.compiled.resolve(), a.output, a.config.resolve(), a.ddk_root.resolve())


if __name__ == '__main__':
    main()
