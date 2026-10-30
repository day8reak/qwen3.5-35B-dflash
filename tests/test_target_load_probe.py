"""Host-only load transport and actual fake-SDK receipt integration."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess

import pytest

from qwen35_moe_dflash import target_load_probe as probe
from qwen35_moe_dflash import board_serial_dflash as prepared_api


def _record(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), identity=probe.transport._full(path))


def _receipt(manifest, state, failure=False):
    base = manifest['target_binding']['base_plan']
    life = dict(weight_init_policy='EXPERIMENTAL_INIT_BEFORE_WEIGHTS_LOAD_PROBE_ONLY_NO_FALLBACK',
        ordering_validation='NOT_VALIDATED_ON_DEVICE', manager_init_attempted=True, manager_initialized=True,
        manager_init_status=0, weights_init_attempted=True, weights_initialized=not failure, weights_init_status=3 if failure else 0,
        manager_deinit_called=True, weights_deinit_called=not failure, weights_deinit_status=-1 if failure else 0,
        closed=True, failure_stage='EXPERIMENTAL_INIT_WEIGHTS_AFTER_INIT' if failure else '', cleanup_error='', run_attempts=0, successful_runs=0)
    return dict(schema_version=1, artifact_kind='qwen35-hiai-target-load-probe', evidence_kind='device-hiai',
        status='TARGET_LOAD_PROBE_FAILED' if failure else 'TARGET_LOAD_PROBE_PASSED',
        model_path=state['model_path'], model_sha256=manifest['target_model']['sha256'], model_bytes=manifest['target_model']['bytes'],
        weight_directory=str(Path(state['model_path']).parent), load_order='init-before-weights', model_load_api='file-v2',
        model_name='qwen35_resident_target', model_content_unchanged=True,
        expected_runtime_version=base['runtime_version'], runtime_version=base['runtime_version'], ddk_id=base['ddk_id'], firmware_id=base['firmware_id'],
        process_id=12346, elapsed_ms=139, run_called=False, process_called=False, tensor_allocations_requested=0,
        numerical_validation='NOT_RUN',
        error='InitWeights failed: 3' if failure else '', runtime_stage='EXPERIMENTAL_INIT_WEIGHTS_AFTER_INIT' if failure else 'CLOSED', lifecycle=life)


@pytest.fixture
def load_request(tmp_path, monkeypatch):
    monkeypatch.setenv('AI_RUN_DIR', str(tmp_path)); monkeypatch.setenv('SN', 'fake0'); monkeypatch.setenv('BIP', '127.0.0.1:8790')
    root = tmp_path / 'prepared'; root.mkdir()
    model = _record(tmp_path / 'model.omc', b'metadata-only-main')
    row = dict(path='compile-work/generated/model/model.omc', bytes=model['bytes'], sha256=model['sha256'])
    weight = dict(path='compile-work/generated/model/SubGraph_0.weight', bytes=18817991168, sha256='e' * 64)
    base = dict(compilation_report_sha256='a' * 64, runtime_version='runtime', ddk_id='ddk', firmware_id='firmware')
    manifest = dict(target_binding=dict(base_plan=base), target_model=model, board_model_root='/data/local/tmp/qwen3_5_moe',
        target_compiled=dict(omc=row, generated_artifacts=[row, weight]), target_payloads=[
            dict(remote='model/model.omc', bytes=row['bytes'], sha256=row['sha256'], record=row),
            dict(remote='model/SubGraph_0.weight', bytes=weight['bytes'], sha256=weight['sha256'], record=weight)])
    (root / 'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(prepared_api, 'validate_prepared', lambda path: deepcopy(manifest))
    def build(out, timeout):
        runner = _record(out / 'native/runner', b'new-native-runner')
        return [dict(local=runner['path'], remote='draft-runner', bytes=runner['bytes'], sha256=runner['sha256'])], ['/vendor/lib64'], dict(mode='host-fixture')
    monkeypatch.setattr(probe.transport, '_build_payloads', build)
    class Fake:
        instances = []; remote = {}; failure = False; cache_miss = False; foreign = False; fail_stage = None; disconnect = False
        def __init__(self, executable, serial, server, output, timeout):
            self.output = output; self.timeout = timeout; self.count = 0; self.calls = []; self.sends = []; self.receives = []
            self.__class__.instances.append(self)
        def shell(self, command, label, **kwargs):
            self.count += 1; self.calls.append((label, command))
            if label == self.fail_stage: raise RuntimeError('deliberate disconnect at ' + label)
            if label == 'load-probe-device-preflight': return 0, 'firmware'
            if label == 'acquire-global-model-io-lock': return 0, 'ACQUIRED'
            if label == 'require-existing-target-cache' and self.foreign: raise RuntimeError('foreign cache owner')
            if label in {'release-unlaunched-load-probe-locks', 'release-target-load-locks'}: return 0, 'RELEASED'
            if label == 'inspect-target-load-locks':
                if self.foreign: raise RuntimeError('foreign global token')
                return 0, 'OWNED'
            if label == 'launch-target-load-probe':
                state = json.loads((self.output / 'checkpoint.json').read_text()); board = state['board_attempt']
                data = _receipt(manifest, state, self.failure)
                self.remote.update({board + '/probe.json': json.dumps(data).encode(), board + '/tool.stdout': b'native JSON',
                    board + '/tool.stderr': b'', board + '/tool.pid': b'12345\n', board + '/tool.exitcode': b'1\n' if self.failure else b'0\n',
                    board + '/run.sh': state['launcher'].encode()})
                if self.disconnect: raise RuntimeError('launch response lost')
                return 0, 'LAUNCHED'
            if label == 'poll-whole-loop-completion': return 0, f'EXIT={1 if self.failure else 0}\nPID=12345\nPROCESS_ABSENT'
            if label == 'hash-result':
                path = re.search(r'if test -f ([^;]+); then', command)[1]
                return (0, hashlib.sha256(self.remote[path]).hexdigest() + '  ' + path) if path in self.remote else (0, '__MISSING__')
            return 0, ''
        def send(self, source, remote, *args, **kwargs):
            self.count += 1; self.sends.append((str(source), str(remote)))
        def receive(self, remote, local, **kwargs):
            self.count += 1; self.receives.append(str(remote)); Path(local).parent.mkdir(parents=True, exist_ok=True)
            Path(local).write_bytes(self.remote[str(remote)])
    monkeypatch.setattr(probe, 'Hdc', Fake)
    checked = []
    def inventory(hdc, rows):
        checked.append(rows)
        return {row['remote']: 'SEND' if Fake.cache_miss else 'OK' for row in rows}
    monkeypatch.setattr(probe.transport, '_cache_inventory', inventory)
    return root, manifest, Fake, checked


def test_plan_never_builds_or_uses_hdc(load_request, tmp_path, monkeypatch):
    root, _, Fake, _ = load_request
    monkeypatch.setattr(probe.transport, '_build_payloads', lambda *a: pytest.fail('plan built native code'))
    result = probe.run_target_load_probe(root, tmp_path / 'plan')
    assert result['status'] == 'TARGET_LOAD_PROBE_PLANNED' and not Fake.instances
    assert result['model_inference'] == result['model_compilation'] == 'NOT_RUN'


@pytest.mark.parametrize('failure', [False, True])
def test_cache_only_single_load_and_closed_failure_are_recovered_without_models(load_request, tmp_path, failure):
    root, _, Fake, checks = load_request; Fake.failure = failure
    value = probe.run_target_load_probe(root, tmp_path / 'run', apply=True)
    assert value['status'] == ('FAILED' if failure else 'TARGET_LOAD_PROBE_PASSED'), value.get('error')
    assert value['stage'] == 'complete' and value['lock_state'] == 'NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT'
    assert value['lifecycle']['closed'] and value['counts'] == dict(target_run_calls=0, draft_calls=0, prefill_calls=0)
    assert len(checks) == 1 and sum(row['bytes'] for row in checks[0]) > 18_000_000_000
    hdc = Fake.instances[-1]
    assert sum(label == 'launch-target-load-probe' for label, _ in hdc.calls) == 1
    assert not any('.omc' in local or '.weight' in local for local, _ in hdc.sends)
    if failure: assert value['error'] == 'InitWeights failed: 3' and value['execution_failure_stage'] == 'EXPERIMENTAL_INIT_WEIGHTS_AFTER_INIT'


@pytest.mark.parametrize('foreign', [False, True])
def test_known_unlaunched_miss_or_foreign_cache_releases_only_our_locks(load_request, tmp_path, foreign):
    root, _, Fake, _ = load_request; Fake.foreign = foreign; Fake.cache_miss = not foreign
    value = probe.run_target_load_probe(root, tmp_path / 'miss', apply=True)
    assert value['status'] == 'FAILED' and not value['launch_requested']
    assert value['lock_release'] == 'RELEASED_BEFORE_MODEL_LAUNCH'
    assert value['lock_state'] == 'NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT'
    hdc = Fake.instances[-1]; assert not hdc.sends and not any(label == 'launch-target-load-probe' for label, _ in hdc.calls)
    command = next(command for label, command in hdc.calls if label == 'release-unlaunched-load-probe-locks')
    assert ('target-lock.owner' in command) is (not foreign)


def test_ambiguous_cache_mkdir_is_retained_and_never_relaunched(load_request, tmp_path):
    root, _, Fake, _ = load_request; Fake.fail_stage = 'acquire-target-cache-lock'
    value = probe.run_target_load_probe(root, tmp_path / 'uncertain', apply=True)
    assert value['status'] == 'FAILED' and value['uncertain'] and value['lock_state'] == 'RETAINED_OR_UNCERTAIN'
    assert not any('release' in label or 'launch' in label for label, _ in Fake.instances[-1].calls)


def test_lost_launch_response_recovery_reads_same_attempt_and_never_restarts(load_request, tmp_path):
    root, _, Fake, _ = load_request; Fake.disconnect = True
    value = probe.run_target_load_probe(root, tmp_path / 'disconnected', apply=True, detach=True)
    assert value['status'] == 'FAILED' and value['launch_requested'] and value['uncertain']
    before = (tmp_path / 'disconnected/checkpoint.json').read_bytes()
    Fake.disconnect = False
    recovered = probe.recover_target_load_probe(tmp_path / 'disconnected', tmp_path / 'recovered')
    assert recovered['status'] == 'TARGET_LOAD_PROBE_PASSED', recovered.get('error')
    assert (tmp_path / 'disconnected/checkpoint.json').read_bytes() == before
    assert not Fake.instances[-1].sends and not any('launch' in label or 'acquire' in label for label, _ in Fake.instances[-1].calls)


@pytest.mark.parametrize('damage', ['foreign', 'missing', 'run-claim', 'bool-zero'])
def test_uncertain_or_malformed_completion_cannot_release(load_request, tmp_path, damage):
    root, _, Fake, _ = load_request
    value = probe.run_target_load_probe(root, tmp_path / 'started', apply=True, detach=True)
    assert value['status'] == 'TARGET_LOAD_PROBE_STARTED'
    key = value['board_attempt'] + '/probe.json'
    if damage == 'foreign': Fake.foreign = True
    elif damage == 'missing': del Fake.remote[key]
    else:
        record = json.loads(Fake.remote[key]); record['run_called' if damage == 'run-claim' else 'tensor_allocations_requested'] = True if damage == 'run-claim' else False
        Fake.remote[key] = json.dumps(record).encode()
    failed = probe.recover_target_load_probe(tmp_path / 'started', tmp_path / 'failed')
    assert failed['status'] == 'FAILED' and failed['lock_state'] == 'RETAINED_OR_UNCERTAIN'
    assert not any(label == 'release-target-load-locks' for label, _ in Fake.instances[-1].calls)


@pytest.mark.parametrize('fault', ['', 'original-init-weights', 'original-deinit-weights'])
def test_actual_native_fake_sdk_json_matches_host_receipt_contract(tmp_path, fault):
    selected = os.environ.get('QWEN35_TARGET_LOAD_TEST_RUNNER')
    if not selected: pytest.skip('provide the already built host fake-HIAI runner')
    model = _record(tmp_path / 'model.omc', b'original-target-load-probe-fixture')
    manifest = dict(target_model=model, target_binding=dict(base_plan=dict(runtime_version='FAKE-HIAI-X90', ddk_id='ddk', firmware_id='firmware')))
    output = tmp_path / 'probe.json'; trace = tmp_path / 'trace'
    argv = [selected, '--target-load-probe', model['path'], '--model-sha256', model['sha256'], '--weight-directory', str(tmp_path),
        '--load-order', probe.POLICY, '--expected-runtime-version', 'FAKE-HIAI-X90', '--ddk-id', 'ddk', '--firmware-id', 'firmware', '--output', str(output)]
    completed = subprocess.run(argv, capture_output=True, text=True, timeout=20,
        env=dict(os.environ, FAKE_HIAI_FAILURE=fault, FAKE_HIAI_TRACE=str(trace)))
    assert completed.returncode == (1 if fault else 0), completed.stderr
    value = json.loads(output.read_text()); state = dict(model_path=model['path'], completion=dict(remote_exitcode=completed.returncode))
    assert probe.validate_load_receipt(value, manifest, state, _expected_evidence_kind='host-fake-hiai') == value
    events = trace.read_text().splitlines()
    assert events.index('v2-init') < events.index('v2-init-weights')
    assert not any(event in events for event in ('v2-run', 'process', 'v2-create-tensor'))
    with pytest.raises(ValueError): probe.validate_load_receipt(value, manifest, state)


def test_cli_apply_and_recovery_are_explicit_and_print_native_failure(monkeypatch, tmp_path, capsys):
    path = Path(__file__).parents[1] / 'targets/cpro/scripts/run_target_load_probe.py'
    spec = importlib.util.spec_from_file_location('load_probe_cli', path); cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    calls = []
    def run(*args, **kwargs):
        calls.append(kwargs); return dict(status='FAILED', error='Init failed: 3', runtime_stage='INIT', lifecycle=dict(closed=True))
    monkeypatch.setattr(probe, 'run_target_load_probe', run)
    assert cli.main(['--prepared', '/bound/prepared', '--output', str(tmp_path / 'out'), '--apply']) == 1
    assert calls == [dict(apply=True, detach=False, timeout_seconds=600, lock_wait_seconds=1800)]
    assert json.loads(capsys.readouterr().out)['error'] == 'Init failed: 3'
    with pytest.raises(SystemExit): cli.main(['--recover', '/old', '--output', str(tmp_path / 'recovery'), '--apply'])
