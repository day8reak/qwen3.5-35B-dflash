#!/usr/bin/env python3
"""Generate same-invocation inputs and golden for original four-layer Prefill."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'model'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True, help='original 310P source root containing inference.py and models/')
    parser.add_argument('--config', required=True, help='original YAML; relative paths retain source-root semantics')
    parser.add_argument('--input-ids', help='optional little-endian INT32 tokens; default: tokenize YAML prompts/prompt')
    parser.add_argument('--sequence-length', type=int, help='optional exact token-count check; otherwise infer, never pad/truncate')
    parser.add_argument('--prompt-index', type=int, default=0, help='zero-based YAML prompt selection, default: 0')
    parser.add_argument('--prompt-repeat', type=int, default=1, help='explicit raw prompt repetition, default: 1; old source used 32')
    parser.add_argument('--output', required=True, help='new directory below AI_RUN_DIR')
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--intermediates', action='store_true', help='also snapshot layer, mixer, norm and MLP boundaries')
    parser.add_argument('--capture-boundaries', action='store_true', help='also capture same-call boundaries/weights for target_layer_probe export')
    probe = parser.add_mutually_exclusive_group()
    probe.add_argument('--probe-fused-value', metavar='FIXTURE', help='use tokens from an existing Prefill fixture and test the live fused attention with constant V after the original forward')
    probe.add_argument('--probe-fused-mask', metavar='T64_FIXTURE', help='record live argument descriptors and compare equivalent FP16 causal mask sentinels on the same fused call')
    args = parser.parse_args()
    from qwen35_moe_dflash.target.prefill_fixture import generate
    result = generate(args.source_root, args.config, args.input_ids, args.sequence_length,
                      args.output, device=args.device, intermediates=args.intermediates,
                      capture_boundaries=args.capture_boundaries,
                      prompt_index=args.prompt_index, prompt_repeat=args.prompt_repeat,
                      probe_fused_value=args.probe_fused_value, probe_fused_mask=args.probe_fused_mask)
    print(json.dumps(dict(status=result['status'], manifest=str(Path(args.output)/'manifest.json'),
                         inputs=len(result.get('inputs', [])), outputs=len(result.get('outputs', [])),
                         sequence_length=result.get('sequence_length'),
                         input_ids=str(Path(args.output)/result['tokens']['path']) if result.get('tokens') else None,
                         attention_audit=result.get('attention_audit', {}).get('status', 'NOT_RUN'),
                         reference_qualification=result.get('reference_qualification'),
                         boundary_capture=result.get('boundary_capture'),
                         attention_module_reference=result.get('attention_module_reference'),
                         **({'fused_value_control': result.get('fused_value_control')} if args.probe_fused_value else {}),
                         **({'fused_mask_control': result.get('fused_mask_control')} if args.probe_fused_mask else {}),
                         error=result.get('error')), indent=2, ensure_ascii=False))
    return 0 if result['status'] == 'CAPTURED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
