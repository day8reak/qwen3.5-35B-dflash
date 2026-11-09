#!/usr/bin/env python3
"""Direct original load/quantize/prefill/decode, limited to four layers."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'model'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='npu:0')
    p.add_argument('--max_token', '--max-token', type=int, default=1)
    p.add_argument('--prompt-index', type=int, default=0)
    p.add_argument('--prompt-repeat', type=int, default=32,
        help='the supplied inference.py uses raw prompt*32, overriding its chat template')
    p.add_argument('--input-ids', help='reuse original golden input_ids.bin, INT32 batch=1, without padding')
    p.add_argument('--baseline-only', action='store_true', help='run original steps without observation hooks')
    a = p.parse_args()
    from qwen35_moe_dflash.target.layer_probe.steps import run_steps
    result = run_steps(a.source_root, a.config, a.output, device=a.device, max_token=a.max_token,
        prompt_index=a.prompt_index, prompt_repeat=a.prompt_repeat,
        input_ids_path=a.input_ids, baseline_only=a.baseline_only)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['status'] == 'EXECUTED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
