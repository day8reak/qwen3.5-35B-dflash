#!/usr/bin/env python3
"""Isolate T64 GDN and MoE errors on captured inputs and replay the cumulative CPU prefix."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))


def relative_l2(section):
    value = section['relative_l2']
    return 'n/a' if value is None else f'{value:.6g}'


def print_drift(layers):
    def columns(row):
        return ' '.join('       n/a' if n is None else f'{n:10.5g}'
                        for group in ('all', 'last_four') for n in row[group])

    print('Each distance is divided by the norm of the captured 310P tensor for the same tokens.')
    print('fixed: CPU from captured branch input; prefix: CPU from propagated input; input_effect: CPU vs CPU')
    print('layer stage                         fixed     prefix input_effect    fixed60+  prefix60+  effect60+')
    for layer, detail in layers.items():
        for stage, row in detail['stages'].items():
            print(f'L{layer}    {stage:22} {columns(row)}')
        if 'routing' in detail:
            routing = detail['routing']
            tokens = routing['changed_expert_sets_tokens']
            print(f'L{layer} CPU router changed expert sets on {len(tokens)}/64 tokens: {tokens}')
            print('L1 decoder output, propagated MoE input. Columns are held-route vs 310P, '
                  'ordinary vs 310P, and route-only effect; then the same for tokens 60-63.')
            print('L1    decoder_routing_control ' + columns(routing['decoder_output']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('fixture', 'source-root', 'config'):
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--trace-drift', action='store_true',
                   help='trace GDN stages on fixed and propagated inputs and hold L1 CPU routing constant')
    args = p.parse_args()
    from qwen35_moe_dflash.target.prefill_t64.decoder_audit import audit
    result = audit(args.fixture, args.source_root, args.config, trace_drift=args.trace_drift)
    print('Historical 310P capture', result['capture_sha256'])
    if args.trace_drift:
        print_drift(result['layers'])
        return
    print('layer  comparison                       rel_L2_all   rel_L2_tokens_60_63   failed_metrics_all')
    for layer, rows in result['layers'].items():
        for label, row in rows.items():
            all_tokens = row['all']
            status = 'PASS' if all_tokens['passed'] else 'FAIL ' + ','.join(all_tokens['failed_metrics'])
            print(f"L{layer}     {label:29} {relative_l2(all_tokens):>10} "
                  f"{relative_l2(row['last_four']):>20}   {status}")


if __name__ == '__main__':
    main()
