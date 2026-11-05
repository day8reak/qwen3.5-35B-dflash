#!/usr/bin/env python3
"""Compile one exported incremental Draft gear using its original selected DDK."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--air-export')
    source.add_argument('--inspect-compile', metavar='PYTORCH_OMC_DIR',
                        help='read an existing PyTorch compilation report/log; no compile or device execution')
    parser.add_argument('--output', help='Shared GE OMC directory or PyTorch candidate build directory')
    parser.add_argument('--replace-failed', action='store_true', help='Reuse a finished failed compilation directory')
    parser.add_argument('--replace', action='store_true', help='Replace a finished PyTorch candidate in the same test slot')
    parser.add_argument('--timeout-seconds', type=int, default=7200)
    parser.add_argument('--tbe-diagnosis', help='optional existing selected-DDK TBE compatibility diagnosis')
    parser.add_argument('--ddk-root', help='selected DDK root; independent of old compilation jobs')
    parser.add_argument('--ddk-id', help='version of the selected DDK; required with --ddk-root')
    parser.add_argument('--rope-fp32-table', action='store_true',
                        help='PyTorch host-embedding candidate: precompute Sin/Cos from the original FP32 phase table in the compiler copy')
    parser.add_argument('--cpu-feature-pack', action='store_true',
                        help='C16 RoPE candidate: pack feature words on CPU; changes diagnostic input ABI only')
    parser.add_argument('--limit-partition-size', choices=('true', 'false'),
                        help='PyTorch only: forward selected OMG partition limit; true also requires observed GraphOp weights <= 2 GiB - 1 before packing')
    args = parser.parse_args(argv)
    if args.timeout_seconds <= 0:
        parser.error('--timeout-seconds must be positive')
    if args.air_export and not args.output:
        parser.error('--output is required with --air-export')
    if args.inspect_compile and (args.output or args.replace or args.replace_failed or args.rope_fp32_table or args.cpu_feature_pack or args.limit_partition_size is not None):
        parser.error('--inspect-compile is read-only; omit output and replacement options')
    try:
        if args.inspect_compile:
            from qwen35_moe_dflash.draft.incremental.torch_export import inspect_pytorch_compilation
            result = inspect_pytorch_compilation(args.inspect_compile)
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
            return 0
        source = json.loads((Path(args.air_export) / 'manifest.json').read_text())
        if source.get('artifact_kind') == 'qwen35-incremental-draft-pytorch-air':
            from qwen35_moe_dflash.draft.incremental.torch_export import compile_pytorch
            replace = args.replace
            if args.replace_failed and (Path(args.output) / 'manifest.json').is_file():
                old = json.loads((Path(args.output) / 'manifest.json').read_text())
                replace = replace or old.get('status') == 'FAILED'
            result = compile_pytorch(args.air_export, args.output, replace=replace,
                timeout_seconds=args.timeout_seconds, tbe_diagnosis=args.tbe_diagnosis,
                ddk_root=args.ddk_root or os.environ.get('CPRO_DDK_ROOT'),
                ddk_id=args.ddk_id or os.environ.get('CPRO_DDK_ID'),
                limit_partition_size=None if args.limit_partition_size is None else args.limit_partition_size == 'true',
                rope_fp32_table=args.rope_fp32_table,
                cpu_feature_pack=args.cpu_feature_pack,
                progress=lambda value: print('pytorch-draft: ' + value, flush=True))
        else:
            from qwen35_moe_dflash.draft.incremental.storage import compile_shared
            if args.replace:
                parser.error('--replace is only for a PyTorch candidate; GE uses --replace-failed')
            if args.rope_fp32_table or args.cpu_feature_pack:
                parser.error('RoPE/CPU feature packing requires the PyTorch host-embedding source')
            if args.limit_partition_size is not None:
                parser.error('--limit-partition-size is only for a PyTorch candidate')
            result = compile_shared(args.air_export, args.output, replace_failed=args.replace_failed,
                timeout_seconds=args.timeout_seconds, tbe_diagnosis=args.tbe_diagnosis,
                ddk_root=args.ddk_root, ddk_id=args.ddk_id,
                progress=lambda value: print('incremental-draft: ' + value, flush=True))
        keys = ('status', 'stage', 'context_rows', 'cache_capacity', 'max_draft_tokens', 'model_compilation',
            'compiled_precision', 'compiled_abi', 'device_execution', 'numerical_validation',
            'runtime_integration', 'full_loop_smoke', 'weight_files_copied', 'graphop_counts',
            'interface_inspection', 'interface_error', 'air_preparation', 'attention', 'composed_attention',
            'optimization', 'projection', 'host_embedding', 'top1', 'top1_placement', 'installation',
            'compiler_options', 'precision_repair', 'feature_pack', 'storage', 'partition_weights', 'omc', 'error', 'compilation_diagnostics', 'next')
        shown = {k: result[k] for k in keys if k in result}
        shown.update(input_count=len(result.get('inputs', [])), output_count=len(result.get('outputs', [])),
            manifest=result.get('manifest', str(Path(args.output).absolute() / 'manifest.json')))
        print(json.dumps(shown, ensure_ascii=False, allow_nan=False), flush=True)
        return 0 if result['status'] in {'DRAFT_INCREMENTAL_OMC_COMPILED_ABI_BOUND', 'PYTORCH_DRAFT_OMC_COMPILED'} else 1
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        print(json.dumps({'status': 'FAILED', 'model_compilation': 'NOT_RUN', 'device_execution': 'NOT_RUN',
            'error': f'{type(error).__name__}: {error}'}, ensure_ascii=False), flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
