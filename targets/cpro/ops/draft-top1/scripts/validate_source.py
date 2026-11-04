#!/usr/bin/env python3
"""Actual kernel control-flow host check and real Torch export; never NPU proof."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import types

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
import stable_arg_max as binding
import reference
import prepare_project


def validate(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    checks = []
    binary = output / 'kernel-host-check'
    command = shlex.split(os.environ.get('CXX', 'c++')) + [
        '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror', '-I' + str(ROOT / 'tests'),
        str(ROOT / 'tests/kernel_host_driver.cpp'), '-o', str(binary)]
    subprocess.run(command, check=True, capture_output=True, text=True)

    def check_case(x, name):
        expected = np.argmax(x, axis=-1).astype('<i4')
        actual = binding.stable_arg_max(torch.from_numpy(x)).numpy()
        assert actual.shape == x.shape[:-1] and actual.dtype == np.int32
        assert np.array_equal(expected, actual), name
        src, dst = output / 'input.bin', output / 'output.bin'
        x.tofile(src)
        subprocess.run([str(binary), str(math.prod(x.shape[:-1])), str(x.shape[-1]), str(src), str(dst)], check=True)
        assert dst.read_bytes() == expected.tobytes(), name
        checks.append(name)

    rng = np.random.default_rng(350248320)
    for shape in [(1,), (2,), (15,), (17,), (1, 1), (17, 1), (2, 3, 5),
                  (2, 3, 4, 17), (1, 1, 1, 1, 1, 1, 1, 7),
                  (15, 485), (16, 512), (17, 513), (31, 127), (32, 128), (33, 129),
                  (2, 8191), (2, 8192), (2, 8193), (2, 16385), (2, 65537), (2, 248321)]:
        x = rng.normal(size=shape).astype('<f2')
        x[..., -1] = 65504
        check_case(x, 'shape_and_last/' + str(shape))
        x.fill(-65504)
        check_case(x, 'first_tie/' + str(shape))
    fixtures = reference.fixtures()
    for name, x in fixtures:
        check_case(x, 'qwen_fixture/' + name)
    # Small tails must not inherit the previous tile's much larger maximum.
    x = np.full((17, 8193), -3, dtype='<f2')
    x[:, 8191] = 5
    check_case(x, 'tail_does_not_read_stale_ub')

    for shape in [(1,), (17, 513), (2, 3, 8193), (1, 15, 248320)]:
        x = torch.empty(shape, dtype=torch.float16)
        exported = torch.export.export(binding.StableArgMax(), (x,))
        calls = [n for n in exported.graph.nodes if n.op == 'call_function']
        assert len(calls) == 1 and calls[0].target == torch.ops.custom_ops.stable_arg_max.default
        meta = calls[0].meta['val']
        assert meta.dtype == torch.int32 and tuple(meta.shape) == shape[:-1]
        checks.append('real_torch_export_single_custom_node/' + str(shape))
    for x in [torch.empty(3, dtype=torch.float32), torch.empty((), dtype=torch.float16),
              torch.empty(0, 2, dtype=torch.float16), torch.empty([1] * 9, dtype=torch.float16),
              torch.empty(2, 3, dtype=torch.float16).t(), torch.tensor([float('nan')], dtype=torch.float16),
              torch.tensor([float('inf')], dtype=torch.float16)]:
        try:
            binding.stable_arg_max(x)
        except (ValueError, RuntimeError):
            checks.append('reject/' + str((x.shape, x.dtype)))
        else:
            raise AssertionError('invalid input accepted')
    assert not torch._C._dispatch_has_kernel_for_dispatch_key(binding.TORCH_NAME, 'PrivateUse1')
    checks.append('no_unproven_device_dispatch_or_cpu_fallback')
    # Real converter invocation with a recording GE API; not a real AIR build.
    calls = {}
    def register(op):
        def decorator(fn):
            calls['op'], calls['fn'] = op, fn
            return fn
        return decorator
    def custom_op(name, **kwargs):
        return (name, kwargs)
    fake_air = types.SimpleNamespace(register_fx_node_ge_converter=register,
                                     ge=types.SimpleNamespace(custom_op=custom_op))
    binding.register_ge(fake_air)
    binding.register_ge(fake_air)
    token = object()
    assert calls['op'] == torch.ops.custom_ops.stable_arg_max.default
    assert calls['fn'](token) == ('StableArgMax', {'inputs': {'x': token}, 'attrs': {}, 'outputs': ['indices']})
    prototype = json.loads((ROOT / 'stable_arg_max.json').read_text())[0]
    assert prototype['op'] == binding.GE_TYPE
    assert prototype['input_desc'][0]['name'] == 'x' and prototype['output_desc'][0]['name'] == 'indices'
    for name in ['op_host/stable_arg_max.cpp', 'op_host/stable_arg_max_tiling.h']:
        assert 'StableArgMax' in (ROOT / name).read_text()
    assert 'void ' + binding.KERNEL_ENTRY + '(' in (ROOT / 'op_kernel/stable_arg_max.cpp').read_text()
    checks.append('torch_ge_prototype_host_kernel_names_and_ports_match')
    report = {'artifact_kind': 'stable-arg-max-source-validation', 'status': 'HOST_PASSED',
        'scope': 'ACTUAL_KERNEL_LOOPS_WITH_IDEAL_REDUCE_API_SHIM_AND_REAL_TORCH_EXPORT',
        'limitations': ['Not vendor compilation', 'Not hardware synchronization, DMA or reduction semantics',
                        'GE call recorded with stub; actual AIR/OMG/board pending'],
        'kernel_compilation': 'NOT_RUN', 'device_execution': 'NOT_RUN',
        'compiler_command': command, 'checks_passed': len(checks), 'checks': checks,
        'source_sha256': {str(p.relative_to(ROOT)): reference.sha(p)
                          for folder in ('op_host', 'op_kernel', 'python', 'tests')
                          for p in sorted((ROOT / folder).rglob('*')) if p.is_file() and p.suffix in ('.py', '.cpp', '.h')}}
    (output / 'source-validation.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = validate(args.output)
    print(json.dumps({k: v for k, v in report.items() if k not in ('checks', 'source_sha256')}))
