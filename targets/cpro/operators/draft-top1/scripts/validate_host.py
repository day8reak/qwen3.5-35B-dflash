#!/usr/bin/env python3
"""Check executable specifications and collection safeguards, never NPU execution."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

import numpy as np
import inspect_ddk
import reference


def validate(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    checks = []

    def passed(name):
        checks.append(name)

    def rejects(name, action, exceptions=(ValueError, FileExistsError)):
        try:
            action()
        except exceptions:
            passed(name)
        else:
            raise AssertionError('expected rejection: ' + name)

    operator = Path(__file__).resolve().parents[1]
    binary = output / 'top1_cpu'
    blocked = subprocess.run(['bash', str(operator / 'scripts/build.sh')],
                             capture_output=True, text=True)
    assert blocked.returncode == 2
    assert json.loads(blocked.stdout)['model_compilation'] == 'NOT_RUN'
    passed('build_blocks_without_selected_adapter')
    command = shlex.split(os.environ.get('CXX', 'c++')) + [
        '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
        str(operator / 'src/reference/top1_cpu.cpp'), '-o', str(binary)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    generated = reference.generate(output / 'fixtures')
    snapshots = output / 'synthetic-output-files'
    for case in generated['cases']:
        name = case['case']
        src = output / 'fixtures' / name / 'input.bin'
        dst = snapshots / name
        dst.mkdir(parents=True)
        subprocess.run([str(binary), str(src), str(dst / '000.bin')], check=True)
        expected = (src.parent / 'expected.bin').read_bytes()
        actual = (dst / '000.bin').read_bytes()
        assert actual == expected and len(actual) == 60, name
        passed('cpp_integer_order_vs_numpy_and_torch/' + name)
        # Copies exercise the comparator's file contract, not repeat execution.
        for i in range(1, 10):
            (dst / f'{i:03d}.bin').write_bytes(actual)
    compared = reference.compare(output / 'fixtures', snapshots)
    assert compared['status'] == 'PASSED'
    passed('all_70_synthetic_files_exact')
    path = snapshots / generated['cases'][0]['case'] / '009.bin'
    original = path.read_bytes()
    for kind, raw in [('missing', None), ('short', original[:-1]),
                      ('padded', original + b'\0' * 4),
                      ('wrong_index', b'\xff' * 4 + original[4:])]:
        if raw is None:
            path.unlink()
        else:
            path.write_bytes(raw)
        assert reference.compare(output / 'fixtures', snapshots)['status'] == 'FAILED'
        path.write_bytes(original)
        passed('compare_rejects_' + kind)
    fixture = output / 'fixtures' / generated['cases'][0]['case'] / 'expected.bin'
    original_expected = fixture.read_bytes()
    fixture.write_bytes(b'\0' * 60)
    rejects('altered_expected_hash', lambda: reference.compare(output / 'fixtures', snapshots))
    manifest_file = output / 'fixtures/fixtures.json'
    original_manifest = manifest_file.read_text()
    modified = json.loads(original_manifest)
    modified['cases'][0]['expected_sha256'] = reference.sha(fixture)
    manifest_file.write_text(json.dumps(modified))
    rejects('forged_expected_hash_vs_argmax', lambda: reference.compare(output / 'fixtures', snapshots))
    fixture.write_bytes(original_expected)
    modified['cases'] = modified['cases'][:1]
    manifest_file.write_text(json.dumps(modified))
    rejects('partial_suite', lambda: reference.compare(output / 'fixtures', snapshots))
    manifest_file.write_text(original_manifest)
    rejects('path_escape', lambda: reference.checked_path(output, '../escape'))

    x = np.zeros(reference.SHAPE, dtype='<f2')
    rejects('wrong_shape', lambda: reference.reference(x.reshape(15, -1)))
    rejects('wrong_dtype', lambda: reference.reference(x.astype('<f4')))
    rejects('strided_input', lambda: reference.reference(x[..., ::-1]))
    for value in [np.nan, np.inf, -np.inf]:
        x.fill(0)
        x[0, -1, -1] = value
        rejects('numpy_rejects_' + str(value), lambda: reference.reference(x))
        bad_input, bad_output = output / 'invalid-input.bin', output / 'invalid-output.bin'
        x.tofile(bad_input)
        process = subprocess.run([str(binary), str(bad_input), str(bad_output)], capture_output=True)
        assert process.returncode == 2 and not bad_output.exists()
        passed('cpp_rejects_' + str(value) + '_before_writing')
    for length in [0, 7449599, 7449601]:
        bad_input.write_bytes(b'\0' * length)
        process = subprocess.run([str(binary), str(bad_input), str(bad_output)], capture_output=True)
        assert process.returncode == 2 and not bad_output.exists()
        passed('cpp_rejects_length_' + str(length))
    bad_input.unlink()

    sdk = output / 'fake-ddk'
    custom = sdk / 'ddk/ascendcpp/examples/add_custom'
    custom.mkdir(parents=True)
    sample = custom / 'add_custom.cpp'
    sample.write_text('// AddCustom\nREG_OP(AddCustom)\n// ReduceMax calIndex\n'
                      '// DataCopyPad DataCopyExtParams\n// GetCoreNum UB_SIZE\n'
                      '// op_kernel kernelName\n')
    (custom / 'CMakeLists.txt').write_text('ascendc_library(add_custom)\n')
    outside = output / 'outside'
    outside.mkdir()
    (outside / 'add_custom_secret.cpp').write_text('REG_OP(SECRET_OUTSIDE)\n')
    (custom / 'outside').symlink_to(outside, target_is_directory=True)
    (custom / 'cycle').symlink_to(custom, target_is_directory=True)
    env_file = output / 'saved-environment.json'
    env_file.write_text(json.dumps({'ddk_root': str(sdk), 'environment': {
        'KIRIN_CONF_JSON_PATH': str(custom), 'SOC_VERSION': 'kirinx90'}}))
    initial_hash = reference.sha(sample)
    summary = inspect_ddk.inspect(sdk, output / 'context', env_file)
    report = json.loads(Path(summary['report']).read_text())
    assert summary['unresolved'] == [] and reference.sha(sample) == initial_hash
    assert all('SECRET_OUTSIDE' not in p.read_text() for p in (output / 'context/sources').iterdir())
    assert any('outside declared roots' in row['reason'] for row in report['notes'])
    assert summary['model_compilation'] == summary['device_execution'] == 'NOT_RUN'
    passed('sdk_read_only_sources_topics_symlink_cycle_and_escape')
    rejects('sdk_output_inside_sdk', lambda: inspect_ddk.inspect(sdk, sdk / 'output'))
    rejects('sdk_output_parent_of_sdk', lambda: inspect_ddk.inspect(sdk, output))
    rejects('sdk_no_overwrite', lambda: inspect_ddk.inspect(sdk, output / 'context'))
    env_file.write_text(json.dumps({'ddk_root': str(outside)}))
    rejects('sdk_mismatched_environment', lambda: inspect_ddk.inspect(sdk, output / 'wrong-env', env_file))
    old_limit = inspect_ddk.LIMITS['file_bytes']
    try:
        inspect_ddk.LIMITS['file_bytes'] = 4
        limited = inspect_ddk.inspect(sdk, output / 'bounded-context')
        assert limited['status'] == 'PARTIAL' and limited['unresolved']
        passed('sdk_budget_is_partial_not_unsupported')
    finally:
        inspect_ddk.LIMITS['file_bytes'] = old_limit
    report = {'artifact_kind': 'qwen35-custom-top1-host-validation', 'status': 'HOST_PASSED',
        'scope': 'CPU_EXECUTABLE_SPEC_GOLDEN_COMPARE_AND_SYNTHETIC_SDK_ONLY',
        'kernel_build': 'NOT_RUN', 'device_execution': 'NOT_RUN',
        'source_sha256': {str(p.relative_to(operator)): reference.sha(p) for p in [
            operator / 'src/reference/top1_cpu.cpp', *sorted((operator / 'scripts').glob('*.py'))]},
        'compiler_command': command, 'cpu_binary_sha256': reference.sha(binary),
        'fixture_manifest_sha256': reference.sha(manifest_file),
        'checks_passed': len(checks), 'checks': checks,
        'npu_only_top1': 'PENDING_SELECTED_DDK_KERNEL_AND_REGISTRATION'}
    (output / 'host-validation.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = validate(args.output)
    print(json.dumps({k: v for k, v in result.items() if k != 'checks'}))
