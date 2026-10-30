#!/usr/bin/env python3
"""Plan experimental prefill-to-decode state mapping without device execution."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prefill-job', 'decode-job', 'prefill-execution', 'named-prefill-probe', 'decode-runtime-probe', 'output'):
        parser.add_argument('--' + name, required=True)
    for name in ('prefill-source-torchair-pb2', 'decode-source-torchair-pb2', 'prefill-source-export-receipt', 'prefill-output-inspection'):
        parser.add_argument('--' + name)
    args = parser.parse_args(argv)
    from qwen35_moe_dflash.target_prefill_decode_feedback import prepare_prefill_decode_feedback
    values = vars(args)
    result = prepare_prefill_decode_feedback(**values)
    shown = {key: result[key] for key in ('status', 'stage', 'error', 'mapped_state_count', 'feedback_ready',
        'decode_inputs_created', 'state_feedback', 'numerical_validation') if key in result}
    shown['next_seed'] = {key: result.get('next_seed', {}).get(key) for key in ('status', 'seed_accepted', 'executed_output_pos', 'trial_use')}
    shown['manifest'] = str(Path(args.output).absolute() / 'manifest.json')
    print(json.dumps(shown, ensure_ascii=False))
    return 1 if result['status'] == 'BLOCKED' else 0


if __name__ == '__main__':
    raise SystemExit(main())
