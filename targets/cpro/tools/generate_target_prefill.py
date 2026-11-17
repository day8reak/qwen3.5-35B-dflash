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
    parser.add_argument('--input-ids', required=True, help='exact prompt tokens, little-endian INT32, batch=1, no padding')
    parser.add_argument('--sequence-length', required=True, type=int)
    parser.add_argument('--output', required=True, help='new directory below AI_RUN_DIR')
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--intermediates', action='store_true', help='also snapshot layer, mixer, norm and MLP boundaries')
    parser.add_argument('--capture-boundaries', action='store_true', help='also capture same-call boundaries/weights for target_layer_probe export')
    args = parser.parse_args()
    from qwen35_moe_dflash.target.prefill_fixture import generate
    result = generate(args.source_root, args.config, args.input_ids, args.sequence_length,
                      args.output, device=args.device, intermediates=args.intermediates,
                      capture_boundaries=args.capture_boundaries)
    print(json.dumps(dict(status=result['status'], manifest=str(Path(args.output)/'manifest.json'),
                         inputs=len(result.get('inputs', [])), outputs=len(result.get('outputs', [])),
                         attention_audit=result.get('attention_audit', {}).get('status', 'NOT_RUN'),
                         reference_qualification=result.get('reference_qualification'),
                         boundary_capture=result.get('boundary_capture'),
                         attention_module_reference=result.get('attention_module_reference'),
                         error=result.get('error')), indent=2, ensure_ascii=False))
    return 0 if result['status'] == 'CAPTURED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
