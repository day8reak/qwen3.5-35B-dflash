#!/usr/bin/env python3
"""Compute a CPU golden from a retained dump_310p_gdn_qkv.py capture."""
import argparse
import errno
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))

from qwen35_moe_dflash.target.layer_probe.common import identity, metrics, read_array, save_json, sha
from qwen35_moe_dflash.target.layer_probe.qlinear_math import GROUP_SIZE, group_matmul
from qwen35_moe_dflash.target.prefill_t64.gdn_tail import _unpack_i4zn


def is_qkv_dump(meta):
    return (meta.get('status') == 'CAPTURED' and meta.get('module') == 'layers.0.linear_attn.in_proj_qkv'
            and meta.get('operator') == 'torch_npu.npu_quant_matmul' and meta.get('call_index') == 0)


def find_capture():
    if not os.environ.get('AI_RUN_DIR'):
        raise ValueError('Specify --capture DIR or set AI_RUN_DIR to locate the retained dump')
    root = Path(os.environ['AI_RUN_DIR']) / 'target/tests'
    matches = []
    for golden in root.rglob('golden_310p.bin'):
        manifest = golden.with_name('manifest.json')
        if manifest.is_file() and is_qkv_dump(json.loads(manifest.read_bytes())):
            matches.append(golden.parent)
    if len(matches) != 1:
        raise ValueError(f'Found {len(matches)} completed L0 QKV dumps below {root}. '
                         'Select the original directory with --capture DIR.\n' +
                         '\n'.join(str(path) for path in sorted(matches)))
    return matches[0]


def generate(capture):
    root = capture.resolve(strict=True)
    manifest = root / 'manifest.json'
    meta = json.loads(manifest.read_bytes())
    if not is_qkv_dump(meta) or meta['byte_order'] != 'little' or (meta['physical_rows'], meta['valid_rows']) != (64, 19):
        raise ValueError(f'Expected a completed little-endian L0 T64/valid19 QKV capture at {root}')
    specs = meta['inputs']
    if set(specs) != {'x1', 'x2', 'scale', 'pertoken_scale'}:
        raise ValueError(f'Unexpected matmul operands at {root}: {list(specs)}')
    options = {key: value for key, value in meta['options'].items() if value is not None}
    if options and options != {'output_dtype': 'torch.float16'}:
        raise ValueError(f'Unsupported captured matmul options: {options}')

    x = read_array(root, specs['x1'])
    packed = read_array(root, specs['x2'])
    scale = read_array(root, specs['scale'])
    per_token = read_array(root, specs['pertoken_scale'])
    measured = read_array(root, meta['output'])
    m, k, n = 64, 2048, 8192
    if x.shape != (m, k) or x.dtype != np.int8 or measured.shape != (m, n) or measured.dtype != np.float16:
        raise ValueError(f'Unexpected x1/output: {x.shape} {x.dtype}, {measured.shape} {measured.dtype}')
    if scale.shape != (k // GROUP_SIZE, n) or scale.dtype not in (np.float16, np.float32):
        raise ValueError(f'Expected NZ group scales [K/256,N], got {scale.shape} {scale.dtype}')
    if per_token.shape not in ((m,), (m, 1)) or per_token.dtype not in (np.float16, np.float32):
        raise ValueError(f'Unexpected pertoken_scale: {per_token.shape} {per_token.dtype}')

    weight = _unpack_i4zn(packed, k, n)
    _, golden = group_matmul(x, per_token.reshape(m, 1).astype(np.float32),
                             weight, scale.T[:, :, None])
    score = metrics(golden[:meta['valid_rows']], measured[:meta['valid_rows']])
    dest = root / 'cpu-golden'
    # Publish the tensor and its report together. Repeated calls verify an existing pair.
    with tempfile.TemporaryDirectory(prefix='.cpu-golden-', dir=root) as tmp:
        stage = Path(tmp) / 'result'
        stage.mkdir()
        binary = stage / 'golden_cpu.bin'
        binary.write_bytes(golden.astype('<f2', copy=False).tobytes())
        result = dict(capture_manifest_sha256=sha(manifest), inputs=specs, measured_310p=meta['output'],
                      cpu=dict(identity(binary, stage), dtype='float16', shape=[m, n]),
                      arithmetic='Captured INT8 x1; signed I4ZN x2; group256 INT32 dot; '
                                 'FP32 group scale and sum in increasing K order; FP32 token scale; FP16 output',
                      actual='CPU', reference='310P', valid_rows=meta['valid_rows'], valid19=score)
        save_json(stage / 'comparison.json', result)
        try:
            stage.rename(dest)
        except OSError as error:
            if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            if (not (dest / binary.name).is_file() or sha(dest / binary.name) != result['cpu']['sha256']
                    or not (dest / 'comparison.json').is_file()
                    or json.loads((dest / 'comparison.json').read_bytes()) != result):
                raise ValueError(f'Existing CPU golden or report differs at {dest}') from error
    print(f"Captured matmul: {root}")
    print(f"CPU golden: {dest/'golden_cpu.bin'}  FP16[{m},{n}]  {golden.nbytes} bytes")
    print(f"CPU vs 310P valid19: cosine={score['cosine']} relative_l2={score['relative_l2']} "
          f"rmse={score['rmse']} max_abs={score['max_abs']} error_ratio={score['error_ratio']}")
    print(f"Comparison: {dest/'comparison.json'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, help='directory containing the original standalone manifest.json')
    args = parser.parse_args()
    generate(args.capture if args.capture is not None else find_capture())
