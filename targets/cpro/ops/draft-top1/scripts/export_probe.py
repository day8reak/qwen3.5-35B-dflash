#!/usr/bin/env python3
"""Export one StableArgMax node; no model compile, board run or Draft integration."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
import torch
from stable_arg_max import StableArgMax, register_ge, TORCH_NAME, GE_TYPE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--shape', type=int, nargs='+', default=[1, 15, 248320])
    parser.add_argument('--format', choices=('pt2', 'air'), default='pt2')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    model = StableArgMax().eval()
    # Validate without allocating a large input or reading tensor values.
    model(torch.empty(args.shape, dtype=torch.float16, device='meta'))
    output.mkdir(parents=True, exist_ok=False)
    example = torch.zeros(args.shape, dtype=torch.float16)
    if args.format == 'pt2':
        exported = torch.export.export(model, (example,))
        torch.export.save(exported, output / 'stable_arg_max.pt2')
        (output / 'graph.txt').write_text(str(exported.graph))
        artifact = output / 'stable_arg_max.pt2'
    else:
        # Same custom_op converter pattern as the original Target exporter.
        import torch_npu  # noqa: F401 -- initialize the selected export stack
        import torchair
        register_ge(torchair)
        torchair.dynamo_export(example, model=model, export_path=str(output),
                              export_name='stable_arg_max', dynamic=False)
        artifact = output / 'stable_arg_max.air'
        if not artifact.is_file():
            raise RuntimeError('TorchAir did not create the declared AIR output')
    report = {'artifact_kind': 'stable-arg-max-export', 'status': 'EXPORTED',
        'torch_op': TORCH_NAME, 'ge_op_type': GE_TYPE,
        'input': {'name': 'x', 'shape': args.shape, 'dtype': 'float16'},
        'output': {'name': 'indices', 'shape': args.shape[:-1], 'dtype': 'int32'},
        'artifact': artifact.name, 'sha256': hashlib.sha256(artifact.read_bytes()).hexdigest(),
        'format': args.format, 'model_compilation': 'NOT_RUN', 'device_execution': 'NOT_RUN',
        'scope': 'STANDALONE_EXPORT_ONLY; AIR_REQUIRES_SELECTED_DDK_REGISTRATION'}
    (output / 'export-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
