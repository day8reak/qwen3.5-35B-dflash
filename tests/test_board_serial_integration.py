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


@pytest.fixture(scope='module')
def resident_native_programs(tmp_path_factory):
    """Actual file-v2 executors/session, with an explicit fake HIAI provider."""
    existing = os.environ.get('QWEN35_RESIDENT_TEST_BINARY')
    if existing:
        binary = Path(existing)
        assert binary.is_absolute() and binary.is_file() and os.access(binary, os.X_OK)
        assert digest(binary) == os.environ.get('QWEN35_RESIDENT_TEST_BINARY_SHA256')
        return binary, binary
    compiler = shutil.which('g++')
    if not compiler:
        pytest.skip('host C++ compiler is unavailable')
    root = tmp_path_factory.mktemp('board-resident-native-integration-build')
    binary = root / 'board-resident-actual-session-host-fake'
    command = [compiler, '-std=c++17', '-O2', '-Wall', '-Wextra',
        '-DQWEN35_RESIDENT_V2=1', '-DQWEN35_BOARD_SERIAL_HOST_FAKE=1', '-DQWEN35_FAKE_HIAI=1',
        '-I' + str(RUNTIME / 'include'), '-I' + str(RUNTIME / 'tests/fake_hiai'),
        *[str(RUNTIME / 'src' / name) for name in ('board_serial_main.cpp', 'resident_session.cpp',
            'hiai_executor.cpp', 'sha256.cpp', 'generation.cpp')],
        str(RUNTIME / 'tests/fake_hiai/fake_hiai.cpp'), '-o', str(binary)]
    compiled = subprocess.run(command, capture_output=True, text=True, timeout=180)
    assert compiled.returncode == 0, compiled.stderr
    return binary, binary


TARGET_SCRIPT = r'''
import json, os, struct, sys
from pathlib import Path

fixture = json.loads(Path(__file__).with_suffix('.json').read_text())
args = dict(item[2:].split('=', 1) for item in sys.argv[1:])
assert args['times'] == '1' and args['modelmanager_itf'] == 'V2'
assert args['direct_from_file'] == 'true'

paths = [Path(value) for value in args['input'].split(',')]
assert len(paths) == 89 and len(set(paths)) == 89
prefill = Path(args['model']).name == 'prefill.omc'
if prefill:
    fixture = fixture['prefill']
inputs = {row['name']: path for row, path in zip(fixture['inputs'], paths)}
for row, path in zip(fixture['inputs'], paths):
    assert path.stat().st_size == row['bytes'], (row, path)
if prefill:
    length, = struct.unpack('<q', inputs['arg114_1'].read_bytes())
    rows = inputs['arg1_1'].read_bytes(); token = rows[(length - 1) * 2048]
    assert rows[:length * 2048] == b''.join(bytes([i]) * 2048 for i in range(1, length + 1))
    assert rows[length * 2048:] == bytes((64 - length) * 2048)
    assert inputs['arg0_1'].read_bytes() == b''.join(struct.pack('<f', (i + 1) / 1000) for i in range(1, length + 1)) + bytes((64 - length) * 4)
    assert inputs['arg107_1'].read_bytes() == inputs['arg2_1'].read_bytes() == struct.pack('<64i', *range(64))
    assert inputs['arg1238_1'].read_bytes() == struct.pack('<i', length - 1)
    assert inputs['arg14_1'].read_bytes() == struct.pack('<128i', length, *([0] * 127))
    assert inputs['arg18_1'].read_bytes() == struct.pack('<h', length)
    mask = b''.join(bytes((q + 1 if q < length else 1) * 4) + struct.pack('<f', float('-inf')) * (2048 - (q + 1 if q < length else 1)) for q in range(64))
    assert inputs['arg113_1'].read_bytes() == mask
    position = 0; history = length
else:
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
    history = position + 1
for state in fixture['states']:
    raw = paths[state['input_index']].read_bytes()
    expected = bytes(state['bytes']) if prefill or position == 0 else struct.pack('<ii', state['ref'], position) + bytes(state['bytes'] - 8)
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
            stream.write(b''.join(struct.pack('<e', i) * 16384 for i in range(1, length + 1)) + bytes((64 - length) * 32768) if prefill else struct.pack('<e', token) * 16384)
        else:
            stream.write(struct.pack('<ii', state_by_output[index]['ref'], history))
    lines.append('output[%d] :%s' % (index, path))
if fault == 'duplicate-output-slot':
    lines.append(lines[0])
print('\n'.join(reversed(lines)))
print('Execute result :Success')
'''


def make_case(root, native_programs, dump=None):
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
    if dump is not None:
        strings['dump'] = '1' if dump else '0'
    rows += ['\t'.join((key, value)) for key, value in strings.items()]
    rows += ['\t'.join(('target_input', str(row['index']), row['name'], row['type'], str(row['bytes']))) for row in inputs]
    rows += ['\t'.join(('target_output', str(row['index']), row['name'], row['type'], str(row['bytes']))) for row in outputs]
    rows += ['\t'.join(('state', 'RefData_' + str(row['ref']), str(row['input_index']), str(row['output_index']), str(row['bytes']))) for row in states]
    config = root / 'board-serial.tsv'; config.write_text('\n'.join(rows) + '\n')
    case = dict(root=root, config=config, files=files, inputs=inputs, outputs=outputs, states=states,
        supervisor=supervisor, output=root / 'result')
    case['manifest'] = verifier_manifest(case)
    if dump is not None:
        case['manifest']['dump'] = dump
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


def enable_prefill(case):
    """Distinct model, state references and output ports; never ordinal join."""
    from qwen35_moe_dflash.board_serial_transport import _full
    root = case['root']; model = root / 'prefill.omc'; model.write_bytes(b'original-prefill-host-fixture')
    sizes = {'arg1_1': 64 * 2048, 'arg113_1': 64 * 8192, 'arg107_1': 256,
        'arg2_1': 256, 'arg0_1': 256}
    inputs = [dict(row, bytes=sizes.get(row['name'], row['bytes'])) for row in case['inputs']]
    states, mapping = [], []
    for row in case['states']:
        ref = row['ref']; pref_ref = ref * 7 % 80; inp = 3 + pref_ref; out = 1 + pref_ref * 13 % 80
        inputs[inp] = dict(index=inp, name='RefData_' + str(pref_ref), type='FP16', bytes=row['bytes'])
        states.append(dict(row, input_index=inp, output_index=out))
        mapping.append({'owner_key': 'fixture-layer-owner-' + str(ref), 'bytes': row['bytes'],
            'decode': {'source_refdata': 'RefData_' + str(ref), 'runtime_input_index': row['input_index'], 'runtime_output_index': row['output_index']},
            'prefill': {'source_refdata': 'RefData_' + str(pref_ref), 'runtime_input_index': inp, 'runtime_output_index': out}})
    by_output = {row['output_index']: row for row in states}
    outputs = [dict(index=i, name='prefill-native-' + str(i), type='FP32' if i == 0 else 'FP16',
        bytes=993280 if i == 0 else 64 * 32768 if i == 81 else by_output[i]['bytes']) for i in range(82)]
    fixture_path = case['files']['target_tool'].with_suffix('.json'); fixture = json.loads(fixture_path.read_text())
    fixture['prefill'] = dict(inputs=inputs, outputs=outputs, states=states); fixture_path.write_text(json.dumps(fixture))
    base = {'inputs': [dict(index=row['index'], name=row['name'], bytes=row['bytes'], tool_input_type=row['type']) for row in inputs],
        'outputs': [dict(index=row['index'], runtime_tensor_name=row['name'], bytes=row['bytes'], tool_output_type=row['type']) for row in outputs]}
    case['manifest']['prefill'] = {'control_recipe': 'PREFILL64_PADDED_SHORT_PROMPT_EXPERIMENTAL',
        'model': dict(path=str(model), bytes=model.stat().st_size, sha256=digest(model), identity=_full(model)),
        'base_plan': base, 'state_mapping': mapping}
    rows = ['\t'.join(('prefill_model', str(model), digest(model), str(model.stat().st_size)))]
    rows += ['\t'.join(('prefill_input', str(row['index']), row['name'], row['type'], str(row['bytes']))) for row in inputs]
    rows += ['\t'.join(('prefill_output', str(row['index']), row['name'], row['type'], str(row['bytes']))) for row in outputs]
    rows += ['\t'.join(('prefill_state', row['decode']['source_refdata'], row['prefill']['source_refdata'],
        str(row['prefill']['runtime_input_index']), str(row['prefill']['runtime_output_index']), str(row['bytes']))) for row in mapping]
    with case['config'].open('a') as stream: stream.write('\n'.join(rows) + '\n')
    (root / 'expected-prepared-manifest.json').write_text(json.dumps(case['manifest'], indent=2))


def enable_resident(case):
    """The provider's explicit marker selects original89/82 plus actual draft3/2."""
    from qwen35_moe_dflash.board_serial_transport import _full
    enable_prefill(case)
    root = case['root']; manifest = case['manifest']
    case['files']['target_model'].write_bytes(b'original-resident-target-fixture')
    case['files']['draft_model'].write_bytes(b'draft-resident-loop64-fixture')
    tool = case['files']['target_tool']
    tool.write_text(tool.read_text().replace("struct.pack('<e', i) * 16384", "struct.pack('<e', 0.5) * 16384"))
    def record(path):
        return dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path), identity=_full(path))
    manifest['target_model'] = record(case['files']['target_model'])
    manifest['draft']['model_source'] = record(case['files']['draft_model'])
    enum = {'FP32': ('FLOAT32', 1, 4), 'FP16': ('FLOAT16', 2, 2), 'INT32': ('INT32', 3, 4),
        'INT8': ('INT8', 4, 1), 'INT16': ('INT16', 5, 2), 'INT64': ('INT64', 7, 8)}
    def state_shape(ref):
        return [1, 8192, 4] if ref % 8 in {0, 2, 4} else [1, 32, 128, 128] if ref % 8 in {1, 3, 5} else [64, 32, 32, 16]
    input_shapes = [[1, 1, 1, 2048], [1, 1, 1, 2048], [1, 1]] + [state_shape(ref) for ref in range(80)] + [[1], [1], [1, 1, 1], [1], [128], [1]]
    by_output = {row['output_index']: row['ref'] for row in case['states']}
    output_shapes = [[1, 1, 1, 248320]] + [state_shape(by_output[index]) for index in range(1, 81)] + [[1, 1, 1, 16384]]
    rows = ['QWEN35_RESIDENT_ABI\t1']
    for role, values, shapes in [('target_input', case['inputs'], input_shapes), ('target_output', case['outputs'], output_shapes)]:
        for row, shape in zip(values, shapes):
            dtype, raw, width = enum[row['type']]
            size = width
            for dimension in shape: size *= dimension
            assert size == row['bytes']
            rows.append('\t'.join(map(str, (role, row['index'], row['name'], dtype, raw, 2, ','.join(map(str, shape)), size))))
    path = root / 'resident-abi.tsv'; path.write_text('\n'.join(rows) + '\n')
    manifest.update(execution_mode='resident-v2', resident_abi=record(path),
        target_lifecycle_policy='init-before-weights-close-weights-first-v1')
    config = []
    for line in case['config'].read_text().splitlines():
        fields = line.split('\t')
        if fields[0] in {'target_model', 'draft_model', 'target_tool'}:
            actual = case['files'][fields[0]]; fields[2:] = [digest(actual), str(actual.stat().st_size)]; line = '\t'.join(fields)
        config.append(line)
    config += ['execution_mode\tresident-v2', 'target_lifecycle_policy\tinit-before-weights-close-weights-first-v1',
        '\t'.join(('resident_abi', str(path), digest(path), str(path.stat().st_size)))]
    case['config'].write_text('\n'.join(config) + '\n')
    (root / 'expected-prepared-manifest.json').write_text(json.dumps(manifest, indent=2))


@pytest.mark.parametrize('dump', [True, False])
def test_actual_resident_file_v2_session_prefill_and_multiround_loop(tmp_path, resident_native_programs, dump):
    from qwen35_moe_dflash.board_serial_transport import validate_retrieved_loop
    case = make_case(tmp_path / ('resident-debug' if dump else 'resident-compact'), resident_native_programs, dump=dump)
    enable_resident(case); trace = case['root'] / 'resident-sdk.trace'
    process = execute(case, native_trace=trace)
    assert process.returncode == 0, (process.stdout, process.stderr)
    loop = validate_retrieved_loop(case['manifest'], case['output'], _expected_evidence_kind='host-fake-hiai',
        _expected_config_sha256=digest(case['config']))
    assert loop['schema_version'] == 3 and loop['counts']['prefill_native_calls'] == 1
    assert loop['target_lifecycle_policy'] == 'init-before-weights-close-weights-first-v1'
    life = loop['resident_lifecycle']
    assert life['target_loads'] == life['draft_loads'] == life['target_close_calls'] == life['draft_close_calls'] == 1
    assert life['closed'] and not life['poisoned']
    assert life['target_lifecycle']['weight_init_policy'] == 'INIT_BEFORE_WEIGHTS_NO_FALLBACK'
    assert life['target_lifecycle']['weight_cleanup_policy'] == 'DEINIT_WEIGHTS_BEFORE_DEINIT'
    assert life['target_runs'] == loop['counts']['target_native_calls'] == 8
    assert life['draft_runs'] == loop['counts']['draft_native_calls'] >= 2
    assert loop['generated_tokens'] == list(range(9, 17))
    assert len(loop['child_lifecycle']['children']) == 1
    assert loop['child_lifecycle']['children'][0]['pid'] != loop['process_id']
    assert process.stdout.index('child-end') < process.stdout.index('resident-load-start')
    events = trace.read_text().splitlines()
    assert [event for event in events if event in {'v2-init', 'v2-init-weights'}] == ['v2-init', 'v2-init-weights', 'v2-init']
    assert [event for event in events if event in {'v2-deinit', 'v2-deinit-weights'}] == ['v2-deinit', 'v2-deinit-weights', 'v2-deinit']
    assert events.count('v2-create-tensor') == 176
    assert events.count('v2-run') == 8 + life['draft_runs']
    assert all(event not in events for event in ('load', 'process', 'unload'))
    if dump:
        calls = [event['call_index'] for kind in ('target_steps', 'draft_steps') for event in loop[kind]]
        assert sorted(calls) == list(range(8 + life['draft_runs']))
        assert not any('pid' in event or 'exitcode' in event for kind in ('target_steps', 'draft_steps') for event in loop[kind])
        _reject_rebound_prefill_claims(case, loop)
    else:
        assert not list(case['output'].rglob('*.bin'))
        failed = make_case(tmp_path / 'resident-failed-run', resident_native_programs, dump=False)
        enable_resident(failed); failure = execute(failed, native_fault='original-run')
        assert failure.returncode == 1, (failure.stdout, failure.stderr)
        from qwen35_moe_dflash.board_serial_transport import validate_retrieved_tokens
        retained = validate_retrieved_tokens(failed['manifest'], failed['output'], _expected_evidence_kind='host-fake-hiai',
            _expected_config_sha256=digest(failed['config']), _remote_exitcode=1)
        assert retained['status'] == 'FAILED' and retained['resident_lifecycle']['closed']
        assert retained['resident_lifecycle']['poisoned'] and retained['resident_lifecycle']['target_run_attempts'] == 1
        assert retained['counts']['target_native_calls'] == 0 and retained['counts']['draft_native_calls'] == 0
        assert retained['child_lifecycle']['children'] == [dict(pid=retained['child_lifecycle']['last_child_pid'], exitcode=0)]


def test_actual_resident_failed_constructor_retains_lifecycle_snapshot(tmp_path, resident_native_programs):
    from qwen35_moe_dflash.board_serial_transport import validate_retrieved_tokens
    case = make_case(tmp_path / 'resident-failed-weight-init', resident_native_programs, dump=False)
    enable_resident(case)
    process = execute(case, native_fault='original-init-weights')
    assert process.returncode == 1, (process.stdout, process.stderr)
    loop = validate_retrieved_tokens(case['manifest'], case['output'], _expected_evidence_kind='host-fake-hiai',
        _expected_config_sha256=digest(case['config']), _remote_exitcode=1)
    life = loop['resident_lifecycle']; target = life['target_lifecycle']
    assert life['stage'] == 'TARGET_LOAD/INIT_WEIGHTS_AFTER_INIT'
    assert life['target_load_attempts'] == 1 and life['target_loads'] == life['draft_load_attempts'] == 0
    assert life['closed'] and life['poisoned'] and life['cleanup_error'] == ''
    assert target['weights_init_attempted'] and not target['weights_initialized']
    assert target['weights_init_status'] == 72 and target['manager_init_status'] == 0
    assert target['manager_init_attempted'] and target['manager_initialized']
    assert target['manager_deinit_called'] and not target['weights_deinit_called']
    assert target['weights_deinit_status'] == -1 and target['closed']
    assert target['failure_stage'] == 'INIT_WEIGHTS_AFTER_INIT'
    assert target['cleanup_error'] == '' and 'InitWeights failed: 72' in loop['error']
    assert loop['counts']['prefill_native_calls'] == 1
    assert loop['counts']['target_native_calls'] == loop['counts']['draft_native_calls'] == 0
    assert len(loop['child_lifecycle']['children']) == 1  # Only the completed prefill is a child process.


@pytest.mark.parametrize('dump', [True, False])
def test_actual_prefill64_handoff_then_real_draft_and_multiround_decode(tmp_path, native_programs, dump):
    from qwen35_moe_dflash.board_serial_transport import validate_retrieved_loop
    case = make_case(tmp_path / ('prefill-debug' if dump else 'prefill-compact'), native_programs, dump=dump)
    enable_prefill(case)
    process = execute(case)
    assert process.returncode == 0, (process.stdout, process.stderr)
    report = validate_retrieved_loop(case['manifest'], case['output'], _expected_evidence_kind='host-fake-hiai',
        _expected_config_sha256=digest(case['config']))
    assert report['counts']['prefill_native_calls'] == 1 and report['counts']['target_native_calls'] == 8
    assert report['counts']['draft_native_calls'] >= 2
    assert report['generated_tokens'] == list(range(9, 17))
    assert report['phase_latency']['prefill']['calls'] == 1 and report['phase_latency']['decode']['calls'] == 8
    if dump:
        receipt = json.loads((case['output'] / report['prefill_steps'][0]['receipt']['path']).read_text())
        first = json.loads((case['output'] / report['target_steps'][0]['receipt']['path']).read_text())
        assert first['absolute_position'] == 8 and first['token_id'] == receipt['next_token_id']
        assert len(first['state_feedback']) == 80 and len(receipt['state_mapping']) == 80
        _reject_rebound_prefill_claims(case, report)
    else:
        assert {p.relative_to(case['output']).as_posix() for p in case['output'].rglob('*') if p.is_file()} == {'loop.json', 'progress.json', 'retrieval-manifest.tsv'}


def _reject_rebound_prefill_claims(case, original):
    """Fresh inventory hashes cannot bless wrong controls, owners or byte parents."""
    from copy import deepcopy
    from qwen35_moe_dflash.board_serial_transport import validate_retrieved_loop
    root = case['output']; loop_path = root / 'loop.json'; inventory = root / 'retrieval-manifest.tsv'
    saved_loop, saved_inventory = loop_path.read_bytes(), inventory.read_bytes()
    failures = [('prefill_steps', 'control', 'query controls'), ('prefill_steps', 'owner', 'cross-model map'),
        ('prefill_steps', 'feature', 'feature rows'), ('target_steps', 'parent', 'preceding80 outputs')]
    for kind, damage, reason in failures:
        loop = deepcopy(original); reference = loop[kind][0]['receipt']; path = root / reference['path']; saved = path.read_bytes()
        receipt = json.loads(saved)
        if damage == 'control':
            next(row for row in receipt['inputs'] if row['name'] == 'arg1238_1')['sha256'] = '0' * 64
        elif damage == 'owner':
            receipt['state_mapping'][0]['prefill_runtime_output_index'] = receipt['state_mapping'][1]['prefill_runtime_output_index']
        elif damage == 'feature':
            receipt['feature_prefix']['sha256'] = '0' * 64
        else:
            receipt['state_feedback'][0]['parent_output']['sha256'] = '0' * 64
        try:
            path.write_text(json.dumps(receipt)); reference.update(bytes=path.stat().st_size, sha256=digest(path))
            loop_path.write_text(json.dumps(loop))
            lines = []
            for line in saved_inventory.decode().splitlines():
                fields = line.split('\t')
                if len(fields) == 4 and fields[1] in {reference['path'], 'loop.json'}:
                    changed = root / fields[1]; fields[2:] = [digest(changed), str(changed.stat().st_size)]
                    line = '\t'.join(fields)
                lines.append(line)
            inventory.write_text('\n'.join(lines) + '\n')
            with pytest.raises(ValueError, match=reason):
                validate_retrieved_loop(case['manifest'], root, _expected_evidence_kind='host-fake-hiai',
                    _expected_config_sha256=digest(case['config']))
        finally:
            path.write_bytes(saved); loop_path.write_bytes(saved_loop); inventory.write_bytes(saved_inventory)


def execute(case, fault=None, native_fault=None, native_trace=None):
    env = dict(os.environ)
    if fault:
        env['QWEN_BOARD_TEST_FAULT'] = fault
    if native_fault:
        env['FAKE_HIAI_FAILURE'] = native_fault
    if native_trace is not None:
        env['FAKE_HIAI_TRACE'] = str(native_trace)
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


def test_compact_actual_draft_cli_matches_debug_tokens_and_removes_consumed_scratch(tmp_path, native_programs):
    reports = []
    from qwen35_moe_dflash.board_serial_transport import validate_retrieved_loop
    for dump in (True, False):
        case = make_case(tmp_path / ('debug' if dump else 'compact'), native_programs, dump=dump)
        model = case['files']['draft_model']
        model_sha = digest(model)
        nlink = model.stat().st_nlink
        process = execute(case)
        assert process.returncode == 0, (process.stdout, process.stderr)
        report = validate_retrieved_loop(case['manifest'], case['output'],
            _expected_evidence_kind='host-fake-hiai', _expected_config_sha256=digest(case['config']))
        reports.append(report)
        assert digest(model) == model_sha
        if not dump:
            assert report['schema_version'] == 2
            assert report['dump'] is False and report['retrieval_mode'] == 'FINAL_TOKENS_ONLY'
            assert model.stat().st_nlink == nlink
            assert {p.relative_to(case['output']).as_posix() for p in case['output'].rglob('*') if p.is_file()} == {
                'loop.json', 'progress.json', 'retrieval-manifest.tsv'}
            assert (case['output'] / 'loop.json').stat().st_size < 65536
            assert not any(key in report for key in ('target_steps', 'draft_steps', 'rounds', 'commits',
                'final_features', 'final_state_files', 'final_target_outputs'))
    assert reports[0]['generated_tokens'] == reports[1]['generated_tokens'] == list(range(9, 17))
    assert reports[0]['counts'] == reports[1]['counts']


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

