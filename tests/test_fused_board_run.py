"""Host shell/HDC transport checks only; never evidence of NPU execution.

The existing single-op artifact builder supplies a hash-bound fixture with fake
export/compile results. During the workflow only the native runner build and
device transport/execution are replaced. Shell scripts, copying, checksums,
retrieval and the production comparator execute for real inside tmp_path.
"""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from qwen35_moe_dflash.ge_fused.bindings import document_bindings
from qwen35_moe_dflash.ge_fused import probes
from qwen35_moe_dflash.hashing import sha256_file


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "targets/cpro/scripts/run_fused_probes.py"


FAKE_HDC = r'''
import json, os, pathlib, shlex, shutil, subprocess, sys
args = sys.argv[1:]
with open(os.environ['FUSED_TEST_HDC_CALLS'], 'a') as log:
    log.write(json.dumps(args) + '\n')
assert args[:4] == ['-t', 'SIMULATED-SERIAL', '-s', '127.0.0.1:8710']
args = args[4:]
remote, local = '/data/local/tmp', os.environ['FUSED_TEST_BOARD']
def translated(path):
    if not path.startswith(remote + '/'):
        raise RuntimeError('fake HDC path escaped isolated board root: ' + path)
    return pathlib.Path(local + path[len(remote):])
if args[:2] == ['file', 'send']:
    shutil.copyfile(args[2], translated(args[3]))
elif args[:2] == ['file', 'recv']:
    shutil.copyfile(translated(args[2]), args[3])
    if os.environ.get('FUSED_TEST_FAILURE') == 'corruption' and args[2].endswith('/output_0000.bin'):
        with open(args[3], 'ab') as out:
            out.write(b'corrupt transfer')
elif args[:1] == ['shell']:
    if os.environ.get('FUSED_TEST_FAILURE') == 'missing-marker':
        print('[Fail]Device not founded or connected')
        sys.exit(0)
    command = shlex.split(args[1])
    assert command[:2] == ['sh', '-c'] and len(command) == 3
    payload = command[2].replace(remote, local).replace('/vendor/lib64/chipsetsdk', local + '/vendor-libs')
    payload = payload.replace('param get const.ohos.fullname', "printf '%s\\n' SIMULATED-FIRMWARE")
    proc = subprocess.run(['sh', '-c', payload], capture_output=True)
    sys.stdout.buffer.write(proc.stdout)
    sys.stderr.buffer.write(proc.stderr)
    # HDC may mask remote failure: production must require its shell marker.
    sys.exit(0)
else:
    raise RuntimeError('unsupported fake HDC operation: ' + repr(args))
'''


FAKE_RUNNER = r'''
import hashlib, json, math, os, pathlib, struct, sys
args = sys.argv[1:]
with open(os.environ['FUSED_TEST_RUNNER_CALLS'], 'a') as log:
    log.write(json.dumps(args) + '\n')
if args == ['--probe']:
    print(json.dumps({'runtime_version': 'SIMULATED-HIAI', 'npu_supported': True}))
    sys.exit(0)
flags = dict(zip(args[::2], args[1::2]))
plan = pathlib.Path(flags['--fused-op-step'])
step = json.loads((plan.parent / 'manifest.json').read_text())
output = pathlib.Path(flags['--output'])
output.mkdir()
base = {'schema_version': 1, 'scope': 'fused-operator-probe',
        'operator_type': step['operator_type'], 'operator_calls': 1,
        'target_calls': 0, 'draft_calls': 0,
        'operator_abi_sha256': step['operator_abi']['sha256'],
        'step_plan_sha256': step['step_plan']['sha256'],
        'model_sha256': step['model']['sha256'], 'ddk_id': flags['--ddk-id'],
        'runtime_version': flags['--expected-runtime-version'],
        'firmware_id': flags['--firmware-id'],
        # Synthetic label tests the production comparison gate, not a device.
        'evidence_kind': 'device-hiai', 'simulation_boundary': 'HOST PROTOCOL FIXTURE ONLY',
        'inputs': step['inputs'], 'outputs': []}
if plan.parent.name == 'gdn_prefill':
    for key in ('runtime_version', 'firmware_id'):
        if os.environ.get('FUSED_TEST_FAILURE') == key + '-mismatch':
            base[key] = 'SIMULATED unexpected identity'
if os.environ.get('FUSED_TEST_FAILURE') == 'execute' and plan.parent.name == 'gdn_prefill':
    base.update(status='RUNTIME_FAILED', failure_stage='PROCESS', operator_calls=0,
                expected_runtime_version=flags['--expected-runtime-version'],
                reason='SIMULATED operator Process failure')
    (output / 'result.json').write_text(json.dumps(base))
    print('SIMULATED partial stdout')
    print('SIMULATED failure retained on stderr', file=sys.stderr)
    sys.exit(17)
for desc in step['outputs']:
    fmt = '<e' if desc['dtype'] == 'float16' else '<f'
    payload = struct.pack(fmt, 0.25) * math.prod(desc['shape'])
    path = output / ('output_%04d.bin' % desc['index'])
    path.write_bytes(payload)
    base['outputs'].append(dict(desc, path=path.name, bytes=len(payload),
                                sha256=hashlib.sha256(payload).hexdigest()))
base['status'] = 'EXECUTED'
(output / 'result.json').write_text(json.dumps(base))
print('SIMULATED fused operator transport complete')
'''


def _executable(path, source):
    path.write_text('#!' + sys.executable + '\n' + source)
    path.chmod(0o700)
    return path


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setenv('AI_RUN_DIR', str(tmp_path))
    monkeypatch.setenv('PYTHONDONTWRITEBYTECODE', '1')
    binding = tmp_path / 'bindings.json'
    contract = document_bindings(target='kirinx90', ddk_id='SIMULATED-DDK',
                                  gdn_initial_state_dtype='float16')
    contract['operators']['gdn']['outputs']['final_state']['dtype'] = 'float16'
    binding.write_text(json.dumps(contract))

    def export(folder, *unused):
        (folder / 'probe.air').write_bytes(b'SIMULATED AIR; no GE execution')
        return {'simulation': True}

    def compile_fixture(folder, *unused):
        destination = folder / 'omc'
        destination.mkdir()
        (destination / 'probe.omc').write_bytes(b'SIMULATED OMC; no model compilation')
        return {'simulation': True}

    probe = tmp_path / 'existing-probes'
    with monkeypatch.context() as build:
        build.setattr(probes, '_export_case', export)
        build.setattr(probes, '_compile_case', compile_fixture)
        manifest = probes.build_fused_probes(probe, binding, ge_root=tmp_path, ddk_root=tmp_path,
                                             ddk_id='SIMULATED-DDK', cases='gdn_prefill,gdn_decode')
    assert manifest['status'] == 'COMPILED'
    original = {str(path.relative_to(probe)): sha256_file(path) for path in probe.rglob('*') if path.is_file()}
    for name in ('_export_case', '_compile_case'):
        monkeypatch.setattr(probes, name, lambda *a, **k: pytest.fail('board workflow must reuse compiled OMC'))

    spec = importlib.util.spec_from_file_location('qwen35_fused_board_workflow_test', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    board = tmp_path / 'fake-board'
    libs = board / 'shared-hiai-libs'
    libs.mkdir(parents=True)
    (libs / 'libhiai.so').write_bytes(b'SIMULATED library; never loaded')
    (board / 'vendor-libs').mkdir()
    hdc = _executable(tmp_path / 'fake-hdc', FAKE_HDC)
    runner = _executable(tmp_path / 'fake-runner', FAKE_RUNNER)
    built = []

    def build_runner(output, timeout):
        built.append((output, timeout))
        return runner

    monkeypatch.setattr(module, '_build_runner', build_runner)
    env = {'HDC': str(hdc), 'SN': 'SIMULATED-SERIAL', 'BIP': '127.0.0.1:8710',
           'BOARD_HIAI_LIB_DIR': '/data/local/tmp/shared-hiai-libs',
           'FUSED_TEST_BOARD': str(board), 'FUSED_TEST_HDC_CALLS': str(tmp_path / 'hdc.jsonl'),
           'FUSED_TEST_RUNNER_CALLS': str(tmp_path / 'runner.jsonl'), 'FUSED_TEST_FAILURE': ''}
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    for name in ('HIAI_RUNTIME_VERSION', 'FIRMWARE_ID', 'CPRO_DDK_ID'):
        monkeypatch.delenv(name, raising=False)
    return {'module': module, 'probe': probe, 'output': tmp_path / 'board-run', 'board': board,
            'original': original, 'built': built, 'env': env, 'root': tmp_path}


def _run(case):
    code = case['module'].main(['--probe-run', str(case['probe']), '--output', str(case['output']),
                               '--timeout-seconds', '20'])
    return code, json.loads((case['output'] / 'result.json').read_text())


def _calls(case, runner=False):
    path = Path(case['env']['FUSED_TEST_RUNNER_CALLS' if runner else 'FUSED_TEST_HDC_CALLS'])
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_fused_board_workflow_reuses_omc_and_runs_selected_cases(case):
    code, result = _run(case)
    assert code == 0 and result['status'] == 'EXECUTION_COMPLETE', result
    assert len(case['built']) == 1
    assert case['original'] == {str(path.relative_to(case['probe'])): sha256_file(path)
                                for path in case['probe'].rglob('*') if path.is_file()}
    calls = _calls(case, runner=True)
    assert calls[0] == ['--probe'] and len(calls) == 3
    invocations = [dict(zip(row[::2], row[1::2])) for row in calls[1:]]
    assert [Path(row['--fused-op-step']).parent.name for row in invocations] == ['gdn_prefill', 'gdn_decode']
    assert all(row['--ddk-id'] == 'SIMULATED-DDK' and row['--expected-runtime-version'] == 'SIMULATED-HIAI'
               and row['--firmware-id'] == 'SIMULATED-FIRMWARE' for row in invocations)
    comparison = json.loads((case['output'] / 'comparison/report.json').read_text())
    assert all(row['process'] == 'PASSED' and row['result_validation'] == 'PASSED' for row in comparison['cases'])
    assert comparison['numerical_acceptance'] == 'NOT_DEFINED' and comparison['placement'] == 'UNVERIFIED'
    assert any('sha256sum' in ' '.join(row) for row in _calls(case))


def test_fused_board_zero_argument_entry_allocates_output(case, monkeypatch):
    monkeypatch.setenv('FUSED_PROBE_RUN', str(case['probe']))
    code = case['module'].main([])
    outputs = list(case['root'].glob('fused-board-*'))
    assert len(outputs) == 1 and outputs[0].is_dir()
    result = json.loads((outputs[0] / 'result.json').read_text())
    assert code == 0 and result['status'] == 'EXECUTION_COMPLETE', result
    assert result['output'] == str(outputs[0]) and result['probe_run'] == str(case['probe'])
    assert result['model_compilation'] == 'NOT_RUN'
    assert result['selected_cases'] == ['gdn_prefill', 'gdn_decode']
    assert (outputs[0] / 'comparison/report.txt').is_file()
    assert _calls(case, runner=True)[0] == ['--probe'] and len(case['built']) == 1


def test_fused_board_default_library_upload_excludes_ddk_hcl(case, monkeypatch):
    monkeypatch.delenv('BOARD_HIAI_LIB_DIR')
    folder = case['root'] / 'mock-build-libraries'
    folder.mkdir()
    names = ['libhiai.so', 'libhiai_hcl_model_runtime.so', 'libhiai_hcl_model_runtime_impl.so']
    libraries = []
    for name in names:
        path = folder / name
        path.write_bytes(('SIMULATED library ' + name).encode())
        libraries.append(path)
    calls = []

    def discover(runner, build_dir):
        calls.append((runner, build_dir))
        return {'libraries': libraries, 'metadata': {'simulation': 'HOST PROTOCOL ONLY'}}

    monkeypatch.setattr(case['module'], 'discover_libraries', discover)
    monkeypatch.setattr(case['module'], 'dynamic_info', lambda path: {'needed': [], 'soname': Path(path).name})
    code, result = _run(case)
    assert code == 0 and result['status'] == 'EXECUTION_COMPLETE', result
    assert len(calls) == 1 and calls[0][1] == case['output'] / 'native'
    uploads = [row[-1] for row in _calls(case) if row[4:6] == ['file', 'send']]
    assert any(path.endswith('/lib/libhiai.so') for path in uploads)
    assert not any(Path(path).name in names[1:] for path in uploads)
    assert _calls(case, runner=True)[0] == ['--probe']
    assert result['libraries']['mode'] == 'candidate_without_ddk_hcl'


def test_fused_board_execution_failure_retains_logs_and_compares(case, monkeypatch):
    monkeypatch.setenv('FUSED_TEST_FAILURE', 'execute')
    code, result = _run(case)
    assert code != 0 and result['status'] == 'FAILED'
    stderr = list(case['output'].rglob('gdn_prefill.stderr'))
    exits = list(case['output'].rglob('gdn_prefill.exitcode'))
    assert len(stderr) == len(exits) == 1
    assert 'SIMULATED failure retained on stderr' in stderr[0].read_text()
    assert exits[0].read_text().strip() == '17'
    comparison = json.loads((case['output'] / 'comparison/report.json').read_text())
    prefill, decode = comparison['cases']
    assert prefill['process'] == 'FAILED' and prefill['result_validation'] == 'PASSED'
    assert decode['process'] == 'PASSED'


def test_fused_board_missing_remote_marker_cannot_report_success(case, monkeypatch):
    monkeypatch.setenv('FUSED_TEST_FAILURE', 'missing-marker')
    code, result = _run(case)
    assert code != 0 and result['status'] == 'FAILED'
    assert 'remote exit marker' in json.dumps(result)
    assert not _calls(case, runner=True) and not case['built']


def test_fused_board_receive_corruption_is_rejected(case, monkeypatch):
    monkeypatch.setenv('FUSED_TEST_FAILURE', 'corruption')
    code, result = _run(case)
    assert code != 0 and result['status'] == 'FAILED'
    assert any(word in json.dumps(result).lower() for word in ('sha256', 'checksum', 'hash'))
    assert (case['output'] / 'comparison/report.json').is_file()


@pytest.mark.parametrize('identity', ['runtime_version', 'firmware_id'])
def test_fused_board_native_identity_must_match_this_invocation(case, monkeypatch, identity):
    monkeypatch.setenv('FUSED_TEST_FAILURE', identity + '-mismatch')
    code, result = _run(case)
    assert code != 0 and result['status'] == 'FAILED'
    assert result['remote_exitcode'] == 0 and not result['retrieval_errors']
    assert result['identity_errors'] == ['gdn_prefill: runtime/firmware differs from this invocation']
    comparison = json.loads((case['output'] / 'comparison/report.json').read_text())
    assert all(row['process'] == 'PASSED' for row in comparison['cases'])


def test_fused_board_rejects_unsafe_bundle_path_before_transport(case):
    path = case['probe'] / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['cases'][0]['bundle_manifest']['path'] = '../outside/manifest.json'
    path.write_text(json.dumps(manifest))
    code, result = _run(case)
    assert code != 0 and result['status'] == 'FAILED'
    assert not _calls(case) and not case['built']


@pytest.mark.parametrize('rms_failure', [False, True], ids=['all-invoked', 'rms-fails-others-continue'])
def test_all_ops_compiles_only_missing_cases_and_keeps_each_bundle(case, monkeypatch, rms_failure):
    """Compilation/execution are host substitutes; artifact and batch logic are real."""
    missing = ['conv_prefill', 'conv_decode', 'rms', 'cache_prefill', 'cache_decode',
               'attention_prefill', 'attention_decode']
    compilation_requests, compilation_calls, executions = [], [], []
    cache_objects = []

    def compile_missing(root, output, selected):
        compilation_requests.append((root, output, selected))

        def export(folder, *unused):
            (folder / 'probe.air').write_bytes(b'SIMULATED AIR; no GE execution')
            return {'simulation': True}

        def compile_one(folder, *unused):
            compilation_calls.append(folder.name)
            if rms_failure and folder.name == 'rms':
                raise RuntimeError('SIMULATED AdnRmsNorm infershape func failed')
            (folder / 'omc').mkdir()
            (folder / 'omc/probe.omc').write_bytes(b'SIMULATED OMC; no model compilation')
            return {'simulation': True}

        with monkeypatch.context() as build:
            build.setattr(probes, '_export_case', export)
            build.setattr(probes, '_compile_case', compile_one)
            return probes.build_fused_probes(output, root / 'bindings.json', ge_root=case['root'],
                                            ddk_root=case['root'], ddk_id='SIMULATED-DDK',
                                            cases=selected, conv_history_width=4)

    def execute(args, *, runner_cache):
        root, output = Path(args.probe_run), Path(args.output)
        manifest, compiled, _ = case['module']._prepare(root)
        cache_objects.append(runner_cache)
        if len(cache_objects) == 1:
            runner_cache['fixture_runner'] = 'SHARED HOST SUBSTITUTE'
        else:
            assert runner_cache is cache_objects[0]
            assert runner_cache['fixture_runner'] == 'SHARED HOST SUBSTITUTE'
        executions.append((root, [row['name'] for row in compiled]))
        (output / 'comparison').mkdir(parents=True)
        rows = [{'name': row['name'], 'load': 'PASSED' if row['status'] == 'COMPILED' else 'NOT_RUN',
                 'process': 'PASSED' if row['status'] == 'COMPILED' else 'NOT_RUN',
                 'numerical': 'DIAGNOSTIC_ONLY' if row['status'] == 'COMPILED' else 'NOT_RUN'}
                for row in manifest['cases']]
        (output / 'comparison/report.json').write_text(json.dumps({'cases': rows}))
        return {'status': 'EXECUTION_COMPLETE', 'output': str(output), 'simulation': 'HOST SUBSTITUTE ONLY'}

    monkeypatch.setattr(case['module'], '_compile_missing', compile_missing)
    monkeypatch.setattr(case['module'], 'execute', execute)
    monkeypatch.setenv('FUSED_PROBE_RUN', str(case['probe']))
    code = case['module'].main(['--all-ops', '--output', str(case['output'])])
    result = json.loads((case['output'] / 'result.json').read_text())
    assert code == (1 if rms_failure else 0), result
    assert result['status'] == ('FAILED' if rms_failure else 'EXECUTION_COMPLETE'), result
    assert compilation_requests == [(case['probe'], case['output'] / 'probes', missing)]
    assert compilation_calls == missing
    assert executions == [(case['probe'], ['gdn_prefill', 'gdn_decode']),
                          (case['output'] / 'probes', [name for name in missing
                                                       if not (rms_failure and name == 'rms')])]
    assert result['reused_cases'] == ['gdn_decode', 'gdn_prefill']
    assert result['compiled_cases_requested'] == missing
    assert result['model_compilation'] == 'MISSING_CASES_ONLY'
    assert result['numerical_acceptance'] == 'NOT_DEFINED' and result['runtime_placement'] == 'UNVERIFIED'
    rows = {row['name']: row for row in result['cases']}
    assert len(rows) == 9
    assert all(row['process'] == 'PASSED' for name, row in rows.items() if name != 'rms')
    assert rows['rms']['compile'] == ('FAILED' if rms_failure else 'PASSED')
    assert rows['rms']['process'] == ('NOT_RUN' if rms_failure else 'PASSED')
    if rms_failure:
        assert 'AdnRmsNorm infershape func failed' in (case['output'] / 'report.txt').read_text()
    assert case['original'] == {str(path.relative_to(case['probe'])): sha256_file(path)
                                for path in case['probe'].rglob('*') if path.is_file()}


def _partial_batch(case, monkeypatch):
    """Recorded first batch: GDN and Cache compile; all remaining calls fail."""
    batch = case['root'] / 'prior-all'
    batch.mkdir()

    def export(folder, *unused):
        if folder.name.startswith('attention'):
            raise RuntimeError('SIMULATED GE recipe failed: input port order mismatch')
        (folder / 'probe.air').write_bytes(b'SIMULATED AIR')
        return {'simulation': True}

    def compile_one(folder, *unused):
        if not folder.name.startswith('cache'):
            raise RuntimeError('SIMULATED long command ' + 'x' * 600 + '\nGet format configs failed')
        (folder / 'omc').mkdir()
        (folder / 'omc/probe.omc').write_bytes(b'SIMULATED OMC')
        return {'simulation': True}

    with monkeypatch.context() as build:
        build.setattr(probes, '_export_case', export)
        build.setattr(probes, '_compile_case', compile_one)
        probes.build_fused_probes(batch / 'probes', case['probe'] / 'bindings.json',
                                  ge_root=case['root'], ddk_root=case['root'], ddk_id='SIMULATED-DDK',
                                  cases=[name for name, _, _ in probes._CASES if not name.startswith('gdn')],
                                  conv_history_width=4)
    (batch / 'result.json').write_text(json.dumps({'model_compilation': 'MISSING_CASES_ONLY',
                                                  'probe_runs': [str(case['probe']), str(batch / 'probes')]}))
    return batch


def test_all_ops_resumes_batch_and_only_retries_attention(case, monkeypatch):
    batch = _partial_batch(case, monkeypatch)
    calls, executed, inspected = [], [], []

    def compile_missing(root, output, selected):
        calls.append(selected)

        def export(folder, *unused):
            (folder / 'probe.air').write_bytes(b'SIMULATED corrected AIR')
            return {'simulation': True}

        def compile_one(folder, *unused):
            (folder / 'omc').mkdir()
            (folder / 'omc/probe.omc').write_bytes(b'SIMULATED corrected OMC')
            return {'simulation': True}

        with monkeypatch.context() as build:
            build.setattr(probes, '_export_case', export)
            build.setattr(probes, '_compile_case', compile_one)
            return probes.build_fused_probes(output, root / 'bindings.json', ge_root=case['root'],
                                            ddk_root=case['root'], ddk_id='SIMULATED-DDK', cases=selected)

    def execute(args, *, runner_cache):
        manifest, compiled, _ = case['module']._prepare(Path(args.probe_run))
        executed.extend(row['name'] for row in compiled)
        output = Path(args.output)
        (output / 'comparison').mkdir(parents=True)
        rows = [{'name': row['name'], 'load': 'PASSED' if row['status'] == 'COMPILED' else 'NOT_RUN',
                 'process': 'PASSED' if row['status'] == 'COMPILED' else 'NOT_RUN', 'numerical': 'DIAGNOSTIC_COMPLETE'}
                for row in manifest['cases']]
        (output / 'comparison/report.json').write_text(json.dumps({'cases': rows}))
        return {'status': 'EXECUTION_COMPLETE', 'output': str(output)}

    def inspect(root, ddk, output):
        inspected.append(root)
        return probes.inspect_fused_probes(root, ddk, output)

    selected_ddk = case['root'] / 'selected-ddk'
    registration = selected_ddk / 'ddk/ascendc/ops/config/fixture.ini'
    registration.parent.mkdir(parents=True)
    registration.write_text('[LinearAttentionConvstateUpdate]\nfixture=SIMULATED Conv registration\n')

    monkeypatch.setattr(case['module'], '_compile_missing', compile_missing)
    monkeypatch.setattr(case['module'], 'execute', execute)
    monkeypatch.setattr(case['module'], 'inspect_fused_probes', inspect)
    monkeypatch.setenv('CPRO_DDK_ROOT', str(selected_ddk))
    code = case['module'].main(['--all-ops', '--probe-run', str(batch), '--output', str(case['output']),
                               '--retry-cases', 'attention_prefill,attention_decode'])
    result = json.loads((case['output'] / 'result.json').read_text())
    assert code == 1 and result['status'] == 'FAILED'  # Conv/RMS remain unresolved.
    assert calls == [['attention_prefill', 'attention_decode']]
    assert result['reused_cases'] == ['cache_decode', 'cache_prefill', 'gdn_decode', 'gdn_prefill']
    assert set(executed) == set(result['reused_cases']) | set(calls[0])
    assert inspected == [batch / 'probes']
    rows = {row['name']: row for row in result['cases']}
    assert all(rows[name]['process'] == 'PASSED' for name in executed)
    assert all(rows[name]['compile'] == 'FAILED' for name in ['conv_prefill', 'conv_decode', 'rms'])
    assert len(result['probe_runs']) == len(result['probe_manifests']) == 3
    report = (case['output'] / 'report.txt').read_text()
    assert 'Get format configs failed' in report  # Was hidden by [:400].
    assert 'SIMULATED Conv registration' in report
    assert case['original'] == {str(path.relative_to(case['probe'])): sha256_file(path)
                                for path in case['probe'].rglob('*') if path.is_file()}


@pytest.mark.parametrize('damage', ['model', 'manifest-hash', 'identity', 'nested-batch', 'duplicate-compiled'])
def test_all_ops_resumed_artifacts_are_validated_before_retry(case, monkeypatch, damage):
    batch = _partial_batch(case, monkeypatch)
    path = batch / 'result.json'
    record = json.loads(path.read_text())
    if damage == 'model':
        (batch / 'probes/cases/cache_prefill/omc/probe.omc').write_bytes(b'tampered')
    elif damage == 'manifest-hash':
        record['probe_manifests'] = {str(case['probe']): '0' * 64}
    elif damage == 'identity':
        monkeypatch.setenv('CPRO_DDK_ID', 'DIFFERENT-DDK')
    elif damage == 'nested-batch':
        record['probe_runs'] = [str(batch)]
    else:
        import shutil
        clone = case['root'] / 'duplicate-probes'
        shutil.copytree(case['probe'], clone)
        record['probe_runs'].append(str(clone))
    path.write_text(json.dumps(record))
    monkeypatch.setattr(case['module'], '_compile_missing', lambda *a: pytest.fail('invalid old artifacts must not compile'))
    code = case['module'].main(['--all-ops', '--probe-run', str(batch), '--output', str(case['output']),
                               '--retry-cases', 'attention_prefill,attention_decode'])
    result = json.loads((case['output'] / 'result.json').read_text())
    assert code == 1 and result['status'] == 'FAILED' and result.get('error')
    assert not result['executions'] and not _calls(case)


def _fresh_ddk(case, monkeypatch):
    ddk = case['root'] / 'fresh-ddk'
    source = ddk / 'ddk/ascendc/ops/impl/custom/chunk_gated_delta_rule.cpp'
    source.parent.mkdir(parents=True)
    source.write_text('''GlobalTensor<half> initialGlobal;
GlobalTensor<half> finalGlobal;
initialGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(initial_state), sizeState);
finalGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(last_recurrent_state), sizeState);
''')
    for name, value in {'CPRO_DDK_ROOT': str(ddk), 'CPRO_DDK_ID': 'SIMULATED-DDK',
                        'CANN_ROOT': str(case['root'] / 'fake-ge'),
                        'FUSED_PROBE_RUN': str(case['root'] / 'deleted-old-probes'),
                        'TBE_DIAG_JOB': str(case['root'] / 'deleted-old-diagnosis'),
                        'GE_CUSTOM_BINDINGS': str(case['root'] / 'deleted-bindings.json')}.items():
        monkeypatch.setenv(name, value)
    return ddk, source


@pytest.mark.parametrize('direct_update', [False, True], ids=['diagnosed-adapter', 'native-request-succeeds'])
def test_fresh_all_ops_rebuilds_deleted_artifacts_once_and_can_resume(case, monkeypatch, direct_update):
    """Host substitutes for GE/OMG/device only; source binding/artifacts are real."""
    ddk, source = _fresh_ddk(case, monkeypatch)
    diagnostics, compiled, executions, policies = [], [], [], []
    fail_rms = True

    def diagnose(root, output, **kwargs):
        diagnostics.append((root, output, kwargs))
        output.mkdir()
        report = {'status': 'DIAGNOSIS_COMPLETE', 'cases': [{
            'name': 'current_omg_first_requested',
            'status': 'RAW_UPDATE_SUCCEEDED' if direct_update else 'FAILED', 'returncode': 0,
            'native': {'stage': 'COMPLETE', 'native_success': direct_update,
                       'initialized_soc_preserved': True,
                       'native_arguments': ['kirinx90', 'AiCore', '', 'false']}}]}
        (output / 'report.json').write_text(json.dumps(report))
        (output / 'report.txt').write_text('SIMULATED TBE diagnosis; no native calls')
        return report

    def export(folder, *unused):
        (folder / 'probe.air').write_bytes(b'SIMULATED AIR; no GE execution')
        return {'simulation': True}

    def compile_one(folder, ddk_root, timeout, diagnosis):
        compiled.append(folder.name)
        policies.append((timeout, diagnosis))
        if fail_rms and folder.name == 'rms':
            raise RuntimeError('SIMULATED AdnRmsNorm infershape func failed')
        (folder / 'omc').mkdir()
        (folder / 'omc/probe.omc').write_bytes(b'SIMULATED OMC; no compilation')
        return {'simulation': True}

    def execute(args, *, runner_cache):
        manifest, selected, _ = case['module']._prepare(Path(args.probe_run))
        executions.append([row['name'] for row in selected])
        output = Path(args.output)
        (output / 'comparison').mkdir(parents=True)
        rows = [{'name': row['name'], 'load': 'PASSED', 'process': 'PASSED',
                 'numerical': 'DIAGNOSTIC_COMPLETE'} for row in selected]
        (output / 'comparison/report.json').write_text(json.dumps({'cases': rows}))
        return {'status': 'EXECUTION_COMPLETE', 'output': str(output), 'simulation': 'HOST ONLY'}

    monkeypatch.setattr(case['module'], 'check_tbe_platform', diagnose)
    monkeypatch.setattr(probes, '_export_case', export)
    monkeypatch.setattr(probes, '_compile_case', compile_one)
    monkeypatch.setattr(case['module'], 'execute', execute)
    code = case['module'].main(['--all-ops', '--fresh', '--output', str(case['output']),
                               '--timeout-seconds', '47'])
    result = json.loads((case['output'] / 'result.json').read_text())
    assert code == 1 and result['status'] == 'FAILED'  # RMS remains visible.
    names = [name for name, _, _ in probes._CASES]
    assert compiled == names  # Failed RMS was not immediately compiled again.
    assert executions == [[name for name in names if name != 'rms']]
    assert diagnostics == [(ddk, case['output'] / 'diagnosis',
                            {'python': sys.executable, 'diagnose': True, 'jobs': 4, 'timeout_seconds': 30})]
    assert policies == [(47, None if direct_update else case['output'] / 'diagnosis')] * 9
    assert result['reused_cases'] == [] and result['compiled_cases_requested'] == names
    assert result['model_compilation'] == 'FRESH_ALL_CASES'
    assert result['bindings'] == str(case['output'] / 'bindings.json')
    assert result['tbe_diagnosis'] == str(case['output'] / 'diagnosis')
    assert result['fresh_stage'] == 'probes-built'
    binding = json.loads((case['output'] / 'bindings.json').read_text())
    assert binding['operators']['gdn']['outputs']['final_state']['dtype'] == 'float16'
    evidence = json.loads((case['output'] / 'bindings.sources.json').read_text())
    assert evidence['installed_gdn_state']['sha256'] == sha256_file(source)
    assert all(row['process'] == ('NOT_RUN' if row['name'] == 'rms' else 'PASSED') for row in result['cases'])

    # A fresh batch is also a valid future --probe-run input; only failed RMS
    # gets another compile attempt, with the original artifacts left intact.
    fail_rms = False
    monkeypatch.delenv('TBE_DIAG_JOB')
    retry = case['root'] / 'fresh-retry'
    code = case['module'].main(['--all-ops', '--probe-run', str(case['output']),
                               '--retry-cases', 'rms', '--output', str(retry)])
    resumed = json.loads((retry / 'result.json').read_text())
    assert code == 0 and resumed['status'] == 'EXECUTION_COMPLETE', resumed
    assert compiled == names + ['rms']
    assert policies[-1][1] == (None if direct_update else str(case['output'] / 'diagnosis'))
    assert resumed['tbe_diagnosis'] == str(case['output'] / 'diagnosis')
    assert resumed['reused_cases'] == sorted(name for name in names if name != 'rms')
    assert diagnostics == [(ddk, case['output'] / 'diagnosis',
                            {'python': sys.executable, 'diagnose': True, 'jobs': 4, 'timeout_seconds': 30})]


@pytest.mark.parametrize('failure', ['bindings', 'tbe-diagnosis'])
def test_fresh_failure_keeps_stage_paths_without_using_deleted_runs(case, monkeypatch, failure):
    _, source = _fresh_ddk(case, monkeypatch)
    if failure == 'bindings':
        source.unlink()

    def diagnose(*args, **kwargs):
        assert failure == 'tbe-diagnosis'
        raise RuntimeError('SIMULATED diagnosis failure')

    monkeypatch.setattr(case['module'], 'check_tbe_platform', diagnose)
    monkeypatch.setattr(case['module'], 'build_fused_probes', lambda *a, **k: pytest.fail('failed bootstrap must not compile'))
    code = case['module'].main(['--all-ops', '--fresh', '--output', str(case['output'])])
    result = json.loads((case['output'] / 'result.json').read_text())
    assert code == 1 and result['status'] == 'FAILED' and result['fresh_stage'] == failure
    assert result['bindings'] == str(case['output'] / 'bindings.json')
    assert result['tbe_diagnosis'] == str(case['output'] / 'diagnosis')
    assert not result['executions'] and not _calls(case)
    assert 'Fresh stage: ' + failure in (case['output'] / 'report.txt').read_text()


def test_fresh_requires_all_ops_and_rejects_retry_or_explicit_probe(case):
    assert case['module'].main(['--fresh']) == 1
    assert case['module'].main(['--all-ops', '--fresh', '--retry-cases', 'rms']) == 1
    with pytest.raises(SystemExit) as error:
        case['module'].main(['--all-ops', '--fresh', '--probe-run', str(case['probe'])])
    assert error.value.code == 2
    assert not case['output'].exists() and not _calls(case)
