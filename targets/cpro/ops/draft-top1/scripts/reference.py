#!/usr/bin/env python3
"""Independent finite-FP16 Top1 fixtures and strict output comparison; no SDK."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SHAPE = (1, 15, 248320)
POSITIONS = (0, 1, 511, 512, 513, 1023, 1024, 65534, 65535, 65536,
             131071, 131072, 247807, 248318, 248319)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    return {str(p.relative_to(root)): sha(p) for p in (
        root / 'scripts/reference.py', root / 'src/reference/top1_cpu.cpp')}


def reference(x):
    if x.dtype != np.dtype('<f2') or x.shape != SHAPE or not x.flags.c_contiguous:
        raise ValueError('requires contiguous little-endian float16[1,15,248320]')
    if not np.isfinite(x).all():
        raise ValueError('finite logits required; no NaN/Inf acceptance waiver')
    return np.argmax(x, axis=-1).astype('<i4')


def fixtures():
    x = np.full(SHAPE, -2, dtype='<f2')
    for row, index in enumerate(POSITIONS):
        x[0, row, index] = 2
    yield 'index_boundaries', x.copy()
    pairs = ((0, 248319), (511, 512), (512, 513), (1023, 1024),
             (65534, 65535), (65535, 65536), (65536, 131072), (131071, 131072),
             (247807, 248319), (248318, 248319), (0, 1), (100, 200),
             (31, 32), (4095, 4096), (32767, 32768))
    x.fill(-2)
    for row, pair in enumerate(pairs):
        x[0, row, list(pair)] = 2
    yield 'ties_across_boundaries', x.copy()
    x.fill(-65504)
    for row in range(15):
        if row % 3 == 0:
            x[0, row, :] = -0.0
            x[0, row, 1::2] = 0.0
        elif row % 3 == 1:
            x[0, row, POSITIONS[row]] = 65504
        # Other rows remain all equal at minimum finite value.
    yield 'zeros_equal_extremes', x.copy()
    tiny = np.array([1], dtype='<u2').view('<f2')[0]
    x.fill(0)
    for row in range(15):
        if row % 2:
            x[0, row, :] = -2 * tiny
            x[0, row, POSITIONS[row]] = -tiny
        else:
            x[0, row, POSITIONS[row]] = tiny
    yield 'signed_subnormal', x.copy()
    x.fill(-65504)
    for row in range(15):
        a = np.float16([1, 1024, -1024, -1, 32752][row % 5])
        b = np.nextafter(a, np.float16(np.inf))
        x[0, row, 0] = a
        x[0, row, max(1, POSITIONS[row])] = b
    yield 'adjacent_representable', x.copy()
    rng = np.random.default_rng(350248320)
    yield 'random_seed350248320', rng.uniform(-32, 32, SHAPE).astype('<f2')
    # Every finite binary16 representation occurs, including both zero signs.
    words = np.arange(65536, dtype='<u2')
    finite = words[(words & 0x7c00) != 0x7c00].view('<f2')
    x.fill(-65504)
    for row in range(15):
        values = np.roll(finite, row * 4093)
        start = (row * 7919) % (SHAPE[-1] - len(values) + 1)
        x[0, row, start:start + len(values)] = values
    yield 'finite_binary16_alphabet', x


def generate(output):
    # Torch is an independent cross-check, never a silent optional dependency.
    import torch
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {'artifact_kind': 'qwen35-custom-top1-fixtures', 'status': 'HOST_GOLDEN',
              'device_execution': 'NOT_RUN', 'shape': list(SHAPE), 'dtype': 'float16',
              'output_shape': [1, 15], 'output_dtype': 'int32',
              'contract': 'FINITE_FP16_FIRST_MAXIMUM_EXACT_INT32',
              'source_sha256': source_hashes(), 'numpy': np.__version__,
              'torch': torch.__version__, 'cases': []}
    for name, x in fixtures():
        y = reference(x)
        torch_y = torch.argmax(torch.from_numpy(x).float(), dim=-1).numpy().astype('<i4')
        if y.tobytes() != torch_y.tobytes():
            raise AssertionError('independent reference mismatch: ' + name)
        folder = output / name
        folder.mkdir()
        x.tofile(folder / 'input.bin')
        y.tofile(folder / 'expected.bin')
        report['cases'].append({'case': name, 'expected': y.reshape(-1).tolist(),
            'input_bytes': x.nbytes, 'output_bytes': y.nbytes,
            'input_sha256': sha(folder / 'input.bin'),
            'expected_sha256': sha(folder / 'expected.bin')})
    (output / 'fixtures.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def checked_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('manifest path escapes fixture directory')
    return path


def compare(fixtures_dir, outputs_dir):
    root, actual_root = Path(fixtures_dir), Path(outputs_dir)
    manifest_path = root / 'fixtures.json'
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('artifact_kind') != 'qwen35-custom-top1-fixtures'
            or manifest.get('source_sha256') != source_hashes()
            or manifest.get('contract') != 'FINITE_FP16_FIRST_MAXIMUM_EXACT_INT32'
            or manifest.get('shape') != list(SHAPE)
            or manifest.get('output_dtype') != 'int32'
            or manifest.get('output_shape') != [1, 15]
            or manifest.get('dtype') != 'float16'):
        raise ValueError('fixture contract/source mismatch; regenerate from current sources')
    names = [name for name, _ in fixtures()]
    if [row['case'] for row in manifest['cases']] != names:
        raise ValueError('requires the complete seven-case suite, without duplicates')
    rows = []
    for case in manifest['cases']:
        name = case['case']
        inp = checked_path(root, name + '/input.bin')
        expected = checked_path(root, name + '/expected.bin')
        if inp.stat().st_size != 7449600 or expected.stat().st_size != 60:
            raise ValueError('wrong fixture length: ' + name)
        if sha(inp) != case['input_sha256'] or sha(expected) != case['expected_sha256']:
            raise ValueError('fixture hash mismatch: ' + name)
        # Never trust an edited expected file, even with a changed hash.
        golden = reference(np.fromfile(inp, dtype='<f2').reshape(SHAPE)).tobytes()
        if golden != expected.read_bytes():
            raise ValueError('fixture does not match direct argmax: ' + name)
        repeats = []
        for iteration in range(10):
            path = checked_path(actual_root, f'{name}/{iteration:03d}.bin')
            raw = path.read_bytes() if path.is_file() else None
            repeats.append({'iteration': iteration, 'passed': raw == golden,
                'bytes': None if raw is None else len(raw),
                'sha256': None if raw is None else hashlib.sha256(raw).hexdigest()})
        rows.append({'case': name, 'passed': all(r['passed'] for r in repeats),
                     'repeats': repeats})
    return {'artifact_kind': 'qwen35-custom-top1-output-comparison',
            'status': 'PASSED' if all(r['passed'] for r in rows) else 'FAILED',
            'scope': 'OUTPUT_FILES_ONLY_NOT_PROOF_OF_DEVICE_EXECUTION_OR_TIMING',
            'fixture_manifest_sha256': sha(manifest_path), 'repetitions': 10, 'cases': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    golden = sub.add_parser('golden')
    golden.add_argument('--output', required=True)
    comparison = sub.add_parser('compare')
    comparison.add_argument('--fixtures', required=True)
    comparison.add_argument('--outputs', required=True)
    comparison.add_argument('--report', required=True)
    args = parser.parse_args()
    if args.command == 'golden':
        result = generate(args.output)
    else:
        result = compare(args.fixtures, args.outputs)
        destination = Path(args.report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}))
    return 1 if result['status'] == 'FAILED' else 0


if __name__ == '__main__':
    raise SystemExit(main())
