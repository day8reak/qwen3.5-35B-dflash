#!/usr/bin/env python3
"""Compare one captured T64 GDN layer with CPU recurrence, norm and output-projection math."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))


def print_scores(label, row):
    def number(name):
        value = row[name]
        return f'{value:.6g}' if value is not None else 'n/a'
    print(f"{label:43}  rel_l2={number('relative_l2'):>10}  cosine={number('cosine'):>10}"
          f"  {'PASS' if row['passed'] else 'FAIL'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', required=True, type=Path, help='original T64 fixture/capture directory')
    parser.add_argument('--layer', type=int, choices=(0,1,2), default=1, help='GDN layer, defaults to 1')
    parser.add_argument('--group', type=Path, help='optional completed single-token whole-GDN batch')
    parser.add_argument('--json', action='store_true', help='emit all metrics, links and per-token board errors')
    args = parser.parse_args()
    from qwen35_moe_dflash.target.prefill_t64.audit_gdn import audit
    report = audit(args.capture, args.layer, args.group)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
        return
    print(f"L{args.layer} T64, 310P capture {report['capture_sha256'][:16]}")
    gdr = report['gdr_comparisons']
    for key in ('cpu_core_vs_310p_core', 'cpu_state_vs_310p_raw_state',
                'gate_fp16_core_vs_cpu_core', 'gate_and_step_fp16_core_vs_gate_fp16_core'):
        print_scores(key, gdr[key])
    if args.group:
        print_scores('board_hidden_vs_310p_hidden', report['board']['hidden_vs_captured_out_proj']['whole'])
    tail = report['tail']
    if tail['status'] == 'CAPTURE_LINK_MISMATCH':
        print('Mismatched capture links:', ', '.join(k for k,v in tail['links'].items() if not v))
        return
    for key in ('norm_fixed_inputs_vs_310p_norm', 'out_proj_fixed_input_vs_310p_hidden',
                'math_core_cpu_tail_vs_310p_hidden', 'board_hidden_vs_math_core_cpu_tail',
                'board_hidden_vs_captured_core_cpu_tail'):
        if key in tail['comparisons']:
            print_scores(key, tail['comparisons'][key])


if __name__ == '__main__':
    main()
