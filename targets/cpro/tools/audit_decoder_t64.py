#!/usr/bin/env python3
"""Isolate T64 GDN and MoE errors on captured inputs and replay the cumulative CPU prefix."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))


def relative_l2(section):
    value = section['relative_l2']
    return 'n/a' if value is None else f'{value:.6g}'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('fixture', 'source-root', 'config'):
        p.add_argument('--' + name, required=True, type=Path)
    args = p.parse_args()
    from qwen35_moe_dflash.target.prefill_t64.decoder_audit import audit
    result = audit(args.fixture, args.source_root, args.config)
    print('Historical 310P capture', result['capture_sha256'])
    print('layer  comparison                       rel_L2_all   rel_L2_tokens_60_63   failed_metrics_all')
    for layer, rows in result['layers'].items():
        for label, row in rows.items():
            all_tokens = row['all']
            status = 'PASS' if all_tokens['passed'] else 'FAIL ' + ','.join(all_tokens['failed_metrics'])
            print(f"L{layer}     {label:29} {relative_l2(all_tokens):>10} "
                  f"{relative_l2(row['last_four']):>20}   {status}")


if __name__ == '__main__':
    main()
