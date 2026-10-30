"""Actual board supervisor and draft CLI with host-only native doubles.

This keeps production tensor extents and runs the real C++ process/TSV/hash
boundaries. The target tool and HIAI implementation are test doubles; no
hardware execution, numerical quality or performance is established here.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

import pytest


REPO = Path(__file__).parents[1]
RUNTIME = REPO / 'targets/cpro/runtime'


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            value.update(chunk)
    return value.hexdigest()


@pytest.fixture(scope='module')
def native_programs(tmp_path_factory):
    compiler = shutil.which('g++')
    if not compiler:
        pytest.skip('host C++ compiler is unavailable')
    root = tmp_path_factory.mktemp('board-serial-native-integration-build')
    supervisor = root / 'board-serial-host-fake'
    supervisor_source = root / 'board_serial_main.cpp'
    supervisor_source.write_bytes((RUNTIME / 'src/board_serial_main.cpp').read_bytes())
    (root / 'board_serial_source.sha256').write_text(digest(supervisor_source) + '\n')
    command = [compiler, '-std=c++17', '-O2', '-Wall', '-Wextra',
        '-DQWEN35_BOARD_SERIAL_HOST_FAKE=1', '-I' + str(RUNTIME / 'include'),
        str(supervisor_source), str(RUNTIME / 'src/sha256.cpp'),
        '-o', str(supervisor)]
    compiled = subprocess.run(command, capture_output=True, text=True, timeout=180)
    assert compiled.returncode == 0, compiled.stderr

    # The existing host HIAI fixture declares S8. Keep its implementation
    # unchanged and adjust only these descriptor extents in a run-owned copy.
    source = (RUNTIME / 'tests/fake_hiai/fake_hiai.cpp').read_text()
    needle = 'in={TensorDimension({1,8,16384}),TensorDimension({1,8}),TensorDimension({1,8})};'
    assert source.count(needle) == 1
    fake_source = root / 'fake_hiai_s64.cpp'
    fake_source.write_text(source.replace(needle,
        'in={TensorDimension({1,64,16384}),TensorDimension({1,64}),TensorDimension({1,64})};'))
    draft = root / 'actual-draft-cli-host-fake'
    command = [compiler, '-std=c++17', '-O2', '-DQWEN35_FAKE_HIAI=1',
        '-I' + str(RUNTIME / 'include'), '-I' + str(RUNTIME / 'tests/fake_hiai'),
        *[str(RUNTIME / 'src' / name) for name in ('main.cpp', 'hiai_executor.cpp', 'sha256.cpp', 'generation.cpp')],
        str(fake_source), '-o', str(draft)]
    compiled = subprocess.run(command, capture_output=True, text=True, timeout=180)
    assert compiled.returncode == 0, compiled.stderr
    return supervisor, draft


TARGET_SCRIPT = r'''
import json, os, struct, sys
from pathlib import Path

fixture = json.loads(Path(__file__).with_suffix('.json').read_text())
args = dict(item[2:].split('=', 1) for item in sys.argv[1:])
assert args['times'] == '1' and args['modelmanager_itf'] == 'V2'
assert args['direct_from_file'] == 'true'
assert args['input_type'].split(',') == [row['type'] for row in fixture['inputs']]
assert args['output_type'].split(',') == [row['type'] for row in fixture['outputs']]
paths = [Path(value) for value in args['input'].split(',')]
assert len(paths) == 89 and len(set(paths)) == 89
inputs = {row['name']: path for row, path in zip(fixture['inputs'], paths)}
for row, path in zip(fixture['inputs'], paths):
    assert path.stat().st_size == row['bytes'], (row, path)
token = inputs['arg1_1'].read_bytes()[0]
assert inputs['arg1_1'].read_bytes() == bytes([token]) * 2048
assert inputs['arg0_1'].read_bytes() == struct.pack('<f', (token + 1) / 1000)
position, = struct.unpack('<i', inputs['arg107_1'].read_bytes())
assert inputs['arg2_1'].read_bytes() == struct.pack('<i', position)
assert inputs['arg1238_1'].read_bytes() == struct.pack('<i', 0)
assert inputs['arg114_1'].read_bytes() == struct.pack('<q', 1)
assert inputs['arg14_1'].read_bytes() == struct.pack('<128i', 1, *([0] * 127))
assert inputs['arg18_1'].read_bytes() == struct.pack('<h', 1)
assert inputs['arg113_1'].read_bytes() == bytes(4 * (position + 1)) + struct.pack('<f', float('-inf')) * (2047 - position)
for state in fixture['states']:
    raw = paths[state['input_index']].read_bytes()
    expected = bytes(state['bytes']) if position == 0 else struct.pack('<ii', state['ref'], position) + bytes(state['bytes'] - 8)
    assert raw == expected, ('wrong parent state', state['ref'], position)

output = Path(args['output_dir']); output.mkdir(parents=True, exist_ok=True)
state_by_output = {row['output_index']: row for row in fixture['states']}
lines = []
fault = os.environ.get('QWEN_BOARD_TEST_FAULT', '')
for row in fixture['outputs']:
    index = row['index']
    # Filenames deliberately have no output ordinal semantics.
    path = output / ('tensor_' + str((index * 17 + 9) % 83).zfill(3) + '.data')
    with path.open('wb') as stream:
        stream.truncate(row['bytes'])
        if index == 0:
            stream.seek((token + 1) * 4); stream.write(struct.pack('<f', 12.0))
            if fault == 'nan-logit':
                stream.seek(0); stream.write(struct.pack('<f', float('nan')))
        elif index == 81:
            stream.write(struct.pack('<e', token) * 16384)
        else:
            stream.write(struct.pack('<ii', state_by_output[index]['ref'], position + 1))
    lines.append('output[%d] :%s' % (index, path))
if fault == 'duplicate-output-slot':
    lines.append(lines[0])
print('\n'.join(reversed(lines)))
print('Execute result :Success')
'''


def make_case(root, native_programs):
    supervisor, draft_runner = native_programs
    root.mkdir()
    target_model = root / 'target.omc'; target_model.write_bytes(b'original-target-tool-host-fixture')
    draft_model = root / 'draft.omc'; draft_model.write_bytes(b'draft-standalone-fixture')
    abi = root / 'draft-abi.json'
    abi.write_text(json.dumps({'artifact_kind': 'qwen35-standalone-draft-direct-abi',
        'retained_nested_binding': {'artifact_kind': 'not-the-top-level-kind'}}))
    target_tool = root / 'fake-target-tool.py'
    target_tool.write_text('#!' + sys.executable + '\n' + TARGET_SCRIPT)
    target_tool.chmod(0o700)
    weight = root / 'embedding-weight.bin'; scale = root / 'embedding-scale.bin'
    with weight.open('wb') as stream:
        stream.truncate(248320 * 2048)
        for token in range(40):
            stream.seek(token * 2048); stream.write(bytes([token]) * 2048)
    with scale.open('wb') as stream:
        stream.truncate(248320 * 4)
        for token in range(40):
            stream.seek(token * 4); stream.write(struct.pack('<f', (token + 1) / 1000))
    inputs = [dict(index=i, name=name, type=dtype, bytes=size) for i, (name, dtype, size) in enumerate([
        ('arg1_1', 'INT8', 2048), ('arg113_1', 'FP32', 8192), ('arg107_1', 'INT32', 4)])]
    states = []
    for ref in range(80):
        size = 2097152 if ref // 2 % 4 == 3 else (65536 if ref % 2 == 0 else 1048576)
        inputs.append(dict(index=len(inputs), name='RefData_' + str(ref), type='FP16', bytes=size))
        states.append(dict(ref=ref, input_index=3 + ref, output_index=1 + (ref * 37 % 80), bytes=size))
    for name, dtype, size in [('arg2_1', 'INT32', 4), ('arg1238_1', 'INT32', 4),
            ('arg0_1', 'FP32', 4), ('arg114_1', 'INT64', 8), ('arg14_1', 'INT32', 512), ('arg18_1', 'INT16', 2)]:
        inputs.append(dict(index=len(inputs), name=name, type=dtype, bytes=size))
    state_by_output = {row['output_index']: row for row in states}
    outputs = [dict(index=i, name='native-output-' + str(i), type='FP32' if i == 0 else 'FP16',
        bytes=993280 if i == 0 else 32768 if i == 81 else state_by_output[i]['bytes']) for i in range(82)]
    target_tool.with_suffix('.json').write_text(json.dumps(dict(inputs=inputs, outputs=outputs, states=states)))
    files = dict(target_model=target_model, draft_model=draft_model, draft_abi=abi,
        target_tool=target_tool, draft_runner=draft_runner, weight_table=weight, scale_table=scale)
    rows = ['QWEN35_BOARD_SERIAL\t1']
    rows += ['\t'.join((role, str(path), digest(path), str(path.stat().st_size))) for role, path in files.items()]
    strings = dict(target_library_path=str(root), draft_library_path=str(root), runtime_version='FAKE-HIAI-X90',
        ddk_id='host-fixture', firmware_id='host-fixture', prompt=','.join(map(str, range(1, 9))),
        max_new_tokens='8', max_proposals='2', min_draft_rounds='2', timeout_ms='60000')
    rows += ['\t'.join((key, value)) for key, value in strings.items()]
    rows += ['\t'.join(('target_input', str(row['index']), row['name'], row['type'], str(row['bytes']))) for row in inputs]
    rows += ['\t'.join(('target_output', str(row['index']), row['name'], row['type'], str(row['bytes']))) for row in outputs]
    rows += ['\t'.join(('state', 'RefData_' + str(row['ref']), str(row['input_index']), str(row['output_index']), str(row['bytes']))) for row in states]
    config = root / 'board-serial.tsv'; config.write_text('\n'.join(rows) + '\n')
    case = dict(root=root, config=config, files=files, inputs=inputs, outputs=outputs, states=states,
        supervisor=supervisor, output=root / 'result')
    case['manifest'] = verifier_manifest(case)
    (root / 'expected-prepared-manifest.json').write_text(json.dumps(case['manifest'], indent=2))
    return case


def verifier_manifest(case):
    def record(path):
        from qwen35_moe_dflash.board_serial_transport import _full
        return dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path), identity=_full(path))
    files = case['files']
    return dict(prompt_token_ids=list(range(1, 9)), max_new_tokens=8,
        max_proposals_per_round=2, minimum_draft_rounds=2,
        target_model=record(files['target_model']),
        draft=dict(model_source=record(files['draft_model']), abi=record(files['draft_abi'])),
        tables=dict(weight=record(files['weight_table']), scale=record(files['scale_table'])),
        target_binding=dict(base_plan=dict(runtime_version='FAKE-HIAI-X90', ddk_id='host-fixture',
            firmware_id='host-fixture',
            inputs=[dict(index=row['index'], name=row['name'], bytes=row['bytes'], tool_input_type=row['type'])
                for row in case['inputs']],
            outputs=[dict(index=row['index'], runtime_tensor_name=row['name'], bytes=row['bytes'], tool_output_type=row['type'])
                for row in case['outputs']]),
            state_mapping=[dict(source_refdata='RefData_' + str(row['ref']),
                runtime_input_index=row['input_index'], runtime_output_index=row['output_index'])
                for row in case['states']]))


def execute(case, fault=None):
    env = dict(os.environ)
    if fault:
        env['QWEN_BOARD_TEST_FAULT'] = fault
    stdout, stderr = case['root'] / 'supervisor.stdout', case['root'] / 'supervisor.stderr'
    # No live stdin/stdout pipe or callback drives a token after this spawn.
    # This local host test only waits for process exit; all children, control
    # decisions, state feedback and embedding reads occur under the supervisor.
    with stdout.open('wb') as out, stderr.open('wb') as err:
        process = subprocess.Popen([str(case['supervisor']), '--config', str(case['config']),
            '--output', str(case['output'])], stdin=subprocess.DEVNULL, stdout=out, stderr=err,
            env=env, start_new_session=True)
        status = process.wait(timeout=600)
    return subprocess.CompletedProcess(process.args, status, stdout.read_text(), stderr.read_text())


def test_actual_cpp_supervisor_and_original_draft_cli_run_all16_target_steps(tmp_path, native_programs):
    case = make_case(tmp_path / 'complete', native_programs)
    process = execute(case)
    assert process.returncode == 0, (process.stdout, process.stderr)
    report = json.loads((case['output'] / 'loop.json').read_text())
    assert report['status'] == 'BOARD_SERIAL_DFLASH_EXECUTED_UNVALIDATED'
    assert report['evidence_kind'] == 'host-fake-hiai'
    assert report['generated_token_ids'] == list(range(9, 17))
    counts = report['counts']
    assert counts['target_native_calls'] == 16
    assert counts['draft_native_calls'] >= 2
    assert counts['accepted'] == 2 and counts['bonuses'] == 1 and counts['corrections'] == 4
    expected_features = b''.join(struct.pack('<e', token) * 16384 for token in range(1, 17))
    assert (case['output'] / report['final_features']['path']).read_bytes() == expected_features
    for state in report['final_state_files']:
        ref = int(state['source_refdata'].split('_')[1])
        assert state['runtime_output_index'] == 1 + (ref * 37 % 80)
        assert (case['output'] / state['path']).read_bytes() == struct.pack('<ii', ref, 16) + bytes(state['bytes'] - 8)
    from qwen35_moe_dflash.board_serial_transport import validate_retrieved_loop
    with pytest.raises(ValueError, match='complete real request'):
        validate_retrieved_loop(case['manifest'], case['output'])
    checked = validate_retrieved_loop(case['manifest'], case['output'],
        _expected_evidence_kind='host-fake-hiai', _expected_config_sha256=digest(case['config']))
    assert checked == report


@pytest.mark.parametrize('fault,reason', [('nan-logit', 'nonfinite target logits'),
    ('duplicate-output-slot', 'duplicate logged target output')])
def test_actual_cpp_refuses_bad_target_outputs_without_next_step(tmp_path, native_programs, fault, reason):
    case = make_case(tmp_path / fault, native_programs)
    process = execute(case, fault)
    assert process.returncode != 0
    report = json.loads((case['output'] / 'loop.json').read_text())
    assert report['status'] == 'FAILED' and reason in report['error']
    assert report['counts']['target_native_calls'] == 1
    assert report['counts']['draft_native_calls'] == 0
    assert report['target_steps'] == []
    assert not (case['output'] / 'target/step_0001').exists()

