#!/usr/bin/env python3
"""Export a standalone, static L0 input RMSNorm using the CPro TorchAir environment."""
import argparse
import math
from pathlib import Path
import tempfile

import torch
import torch_npu
import torchair


class InputRMSNorm(torch.nn.Module):
    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x, gamma):
        return torch_npu.npu_rms_norm(x, gamma, self.epsilon)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('l0_input_rmsnorm.air'))
    parser.add_argument('--dtype', choices=('float16', 'float32'), default='float16',
                        help='activation dtype (default: float16, as captured on 310P)')
    parser.add_argument('--gamma-dtype', choices=('float16', 'float32'),
                        help='effective gamma dtype (defaults to --dtype)')
    parser.add_argument('--epsilon', type=float, default=1e-6)
    args = parser.parse_args()
    if not math.isfinite(args.epsilon) or args.epsilon <= 0:
        parser.error('--epsilon must be finite and positive')

    # Both tensors are runtime inputs. These examples only describe the static graph.
    x = torch.zeros((1, 64, 2048), dtype=getattr(torch, args.dtype))
    gamma = torch.ones((2048,), dtype=getattr(torch, args.gamma_dtype or args.dtype))
    output = args.output.expanduser().absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp, torch.no_grad():
        torchair.dynamo_export(x, gamma, model=InputRMSNorm(args.epsilon).eval(),
                              export_path=tmp, export_name='export', dynamic=False)
        air = Path(tmp)/'export.air'
        if not air.is_file() or not air.stat().st_size:
            raise RuntimeError('TorchAir did not produce export.air')
        air.replace(output)
    print(f'{output}\nx: {list(x.shape)} {x.dtype}\ngamma: {list(gamma.shape)} {gamma.dtype}'
          f'\nepsilon: {args.epsilon:g}\noutput: y')


if __name__ == '__main__':
    main()
