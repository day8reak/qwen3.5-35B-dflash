"""The real board adapter through simulated HDC; never device evidence."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from qwen35_moe_dflash import existing_draft_smoke_backend as adapter
from qwen35_moe_dflash import real_feature_draft_board as board
from qwen35_moe_dflash.draft_board import _io, _wire
from qwen35_moe_dflash.hashing import sha256_file
from qwen35_moe_dflash.resources import repository_root
from test_real_feature_draft_board import FakeHdc


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value))


def record(path, root):
    return {'path': str(path.relative_to(root)), 'bytes': path.stat().st_size, 'sha256': sha256_file(path)}


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setenv('AI_RUN_DIR', str(tmp_path)); monkeypatch.setenv('SN', 'fixture-device')
    monkeypatch.setenv('BIP', 'fixture:8790'); monkeypatch.setenv('BOARD_HIAI_LIB_DIR', '/data/local/tmp/fixture_hiai')
    for key in ('CPRO_DDK_ID', 'HIAI_RUNTIME_VERSION', 'FIRMWARE_ID'): monkeypatch.delenv(key, raising=False)
    root = tmp_path / 'old-bundle'; (root / 'model').mkdir(parents=True)
    model = root / 'model/model.omc'; model.write_bytes(b'HOST_FAKE_FULL_MODEL_BINARY_NOT_DEVICE')
    cfg = json.loads((repository_root() / 'specs/draft-config.json').read_text())
    inputs, outputs = _io(64, direct_spec=[])
    abi = {'artifact_kind': 'qwen35-standalone-draft-direct-abi', 'draft_config': cfg, 'inputs': inputs, 'outputs': outputs,
        'sequence_length': 64, 'state_policy': 'prefix-recompute', 'direct_spec': [], 'source_output_precision': 'DECLARED_SOURCE_DTYPES'}
    save(root / 'draft-abi.json', abi)
    manifest = {'artifact_kind': 'qwen35-standalone-draft-direct-bundle', 'status': 'PREPARED', 'sequence_length': 64,
        'platform': 'kirinx90', 'weight_storage': 'embedded-stream', 'source_identity': 'MATCHED_EXPORT_AND_REFERENCE',
        'ddk_id': 'fixture-ddk', 'draft_config': cfg, 'direct_spec': [], 'source_output_precision': 'DECLARED_SOURCE_DTYPES',
        'model': record(model, root), 'draft_abi': record(root / 'draft-abi.json', root), 'outputs': outputs,
        'inputs': [{**row, 'path': f'old-unused-{row["index"]}.bin', 'sha256': 'a' * 64} for row in inputs]}
    manifest['model_payloads'] = [manifest['model']]
    save(root / 'weight-packing.json', {'status': 'EMBEDDED', 'output': manifest['model']})
    manifest['weight_packing'] = record(root / 'weight-packing.json', root)
    (root / 'step.tsv').write_bytes(_wire(manifest)); manifest['step_plan'] = record(root / 'step.tsv', root)
    save(root / 'manifest.json', manifest)
    class Hdc(FakeHdc):
        def __init__(self, executable, sn, bip, output, timeout):
            super().__init__(executable, sn, bip, output, timeout)
            self.manifest = json.loads((Path(output).parent / 'prepared/manifest.json').read_bytes())
    for key, value in {'cache_decision': 'MATCHED', 'fail_label': None, 'mutate_native': None, 'exitcode': 0,
                       'hidden_nonfinite': False, 'invalid_top1': False, 'sent': [], 'instances': []}.items():
        monkeypatch.setattr(Hdc, key, value)
    monkeypatch.setattr(board, 'Hdc', Hdc)
    builds = []
    def build(out, timeout):
        builds.append(out)
        binary = out / 'native/build/qwen35_dflash_hiai'; binary.parent.mkdir(parents=True)
        binary.write_bytes(b'HOST_FAKE_LINKED_RUNNER')
        save(out / 'native/command.json', {'argv': ['host-test-build'], 'libraries': []})
        save(out / 'native/result.json', {'status': 'LINKED', 'binary_sha256': sha256_file(binary),
            'command_sha256': sha256_file(out / 'native/command.json')})
        return binary
    monkeypatch.setattr(board, '_build_runner', build)
    session = adapter.ExistingDraftSmokeBackend(root, tmp_path / 'draft-session', runtime_version='fixture-runtime',
        firmware_id='fixture-firmware', _host_test=True)
    return {'root': root, 'model': model, 'session': session, 'hdc': Hdc, 'builds': builds}


def features(length):
    return ((np.arange(length * 16384, dtype=np.float32) % 101) / 100).reshape(1, length, 16384).astype('<f2')


def _automatic_discovery(case, tmp_path, monkeypatch):
    """Use real build-record/discovery logic; only ELF/device are host doubles."""
    from qwen35_moe_dflash import hiai_deployment
    monkeypatch.delenv('BOARD_HIAI_LIB_DIR')
    monkeypatch.setattr(hiai_deployment, '_elf', lambda *args, **kwargs: None)
    monkeypatch.setattr(board, 'dynamic_info', lambda path: {'needed': [], 'soname': None})
    directory = tmp_path / 'original-linked-libraries'; directory.mkdir()
    libraries = [directory / name for name in ('libhiai-fixture.so', 'libc++_shared.so')]
    for index, path in enumerate(libraries):
        path.write_bytes(b'HOST_ONLY_LIBRARY_' + str(index).encode())
    original_builder = board._build_runner
    def build(out, timeout):
        binary = original_builder(out, timeout)
        save(out / 'native/command.json', {'argv': ['host-test-build'], 'libraries': [
            {'path': str(path), 'sha256': sha256_file(path)} for path in libraries]})
        save(out / 'native/result.json', {'status': 'LINKED', 'binary_sha256': sha256_file(binary),
            'command_sha256': sha256_file(out / 'native/command.json')})
        return binary
    monkeypatch.setattr(board, '_build_runner', build)
    original_discovery = board.discover_libraries
    observed = []
    def discover(runner, build_dir):
        # The real regression: the second attempt owns no native build at all.
        assert Path(build_dir) == Path(runner).parent.parent
        assert (Path(build_dir) / 'result.json').is_file()
        assert (Path(build_dir) / 'command.json').is_file()
        observed.append(Path(build_dir))
        return original_discovery(runner, build_dir)
    monkeypatch.setattr(board, 'discover_libraries', discover)
    return libraries, observed


def test_automatic_library_discovery_reuses_exact_original_build_records_across_calls(case, tmp_path, monkeypatch):
    libraries, observed = _automatic_discovery(case, tmp_path, monkeypatch)
    session = case['session']
    session.propose((4, 5, 6), features(3))
    original = session.out / 'call-0001/execution/native'
    records = {name: ((original / name).read_bytes(), adapter._current(original / name)[1])
        for name in ('result.json', 'command.json')}
    library_pins = {str(path): adapter._current(path)[1] for path in libraries}
    session.propose((4, 5, 6, 7), features(4))
    assert observed == [original, original] and len(case['builds']) == 1
    assert session.runner_build_directory == original
    assert not (session.out / 'call-0002/execution/native').exists()
    for name, (raw, identity) in records.items():
        assert (original / name).read_bytes() == raw and adapter._current(original / name)[1] == identity
    for path in libraries:
        assert adapter._current(path)[1] == library_pins[str(path)]
    for instance in case['hdc'].instances:
        assert {Path(local).name for local, remote in instance.sends if '/lib/' in remote} == {path.name for path in libraries}
    reports = [json.loads((session.out / name / 'execution/result.json').read_text())
        for name in ('call-0001', 'call-0002')]
    assert all(row['libraries']['discovery']['build_dir'] == str(original) for row in reports)
    assert all(row['libraries']['discovery']['command_sha256'] == sha256_file(original / 'command.json') for row in reports)
    assert session.execution_summary()['native_calls'] == 2


@pytest.mark.parametrize('damage', ['command', 'result', 'library'])
def test_reused_runner_rejects_changed_original_build_or_library_before_second_device_attempt(case, tmp_path, monkeypatch, damage):
    libraries, _ = _automatic_discovery(case, tmp_path, monkeypatch)
    session = case['session']; session.propose((4, 5), features(2))
    original = session.runner_build_directory
    path = libraries[0] if damage == 'library' else original / (damage + '.json')
    # Even a same-byte rewrite changes the first build's full identity.
    path.write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match='pinned linked runner or library changed'):
        session.propose((4, 5, 6), features(3))
    assert len(case['hdc'].instances) == 1 and len(case['builds']) == 1
    assert session.execution_summary()['validated_native_calls'] is False


def test_first_runner_build_must_bind_actual_command_sha(case, monkeypatch):
    original = board._build_runner
    def unbound(out, timeout):
        binary = original(out, timeout)
        record_path = out / 'native/result.json'
        result = json.loads(record_path.read_text()); result['command_sha256'] = '0' * 64
        save(record_path, result)
        return binary
    monkeypatch.setattr(board, '_build_runner', unbound)
    with pytest.raises(ValueError, match='not completely validated'):
        case['session'].propose((4, 5), features(2))
    assert case['session'].execution_summary()['native_calls'] == 0
    assert not any(label == 'execute-real-feature-draft' for instance in case['hdc'].instances
        for label, _ in instance.commands)


def test_partial_prefix_native_roundtrip_has_exact_right_padding_no_modelcopy_and_one_runner_build(case, capsys):
    session = case['session']; before = case['model'].stat()
    assert session.preflight()['status'] == 'HOST_SOURCE_PINNED' and not case['builds'] and not case['hdc'].instances
    first = session.propose((4, 5, 6), features(3))
    second = session.propose((4, 5, 6, 7), features(4))
    assert len(first.token_ids) == len(second.token_ids) == 15
    assert first.evidence.evidence_kind == second.evidence.evidence_kind == 'host-fake-hiai'
    assert first.feature_input_sha256 == adapter.feature_sha256(features(3))
    prepared = session.out / 'call-0001/prepared'
    manifest = session.validate_step(prepared)
    ids = np.fromfile(prepared / manifest['inputs'][1]['path'], dtype='<i4')
    mask = np.fromfile(prepared / manifest['inputs'][2]['path'], dtype='<i4')
    wire = np.fromfile(prepared / manifest['inputs'][0]['path'], dtype='<f2').reshape(1, 64, 16384)
    assert ids[:3].tolist() == [4, 5, 6] and not np.any(ids[3:])
    assert mask.tolist() == [1] * 3 + [0] * 61
    assert wire[:, :3].tobytes() == features(3).tobytes() and not np.any(wire[:, 3:])
    assert manifest['token_provenance']['anchor_position'] == 2 and manifest['token_provenance']['anchor_token_id'] == 6
    assert not list(session.out.rglob('*.omc')) and len(case['builds']) == 1
    assert not any(local.endswith('.omc') for instance in case['hdc'].instances for local, _ in instance.sends)
    after = case['model'].stat()
    assert (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_nlink) == (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_nlink)
    summary = session.execution_summary()
    assert summary['native_calls'] == 2 and summary['runner_build_count'] == 1 and summary['validated_native_calls'] is True
    assert len(summary['receipt_paths']) == 2 and 'reusing pinned runner' in capsys.readouterr().out


def test_model_sha_is_bound_once_and_not_rehashed_per_proposal(case, monkeypatch):
    from qwen35_moe_dflash import real_feature_draft as source
    original = source.sha256_file
    def no_model_hash(path):
        assert str(path) != str(case['model']), 'already pinned OMC must not be rehashed each round'
        return original(path)
    monkeypatch.setattr(source, 'sha256_file', no_model_hash)
    case['session'].propose((1, 2), features(2)); case['session'].propose((1, 2, 3), features(3))
    assert len(case['builds']) == 1


def test_capacity_overflow_never_truncates_and_failed_session_stops(case):
    session = case['session']
    with pytest.raises(ValueError, match='complete prefix'):
        session.propose(tuple(range(65)), features(65))
    with pytest.raises(RuntimeError, match='failed draft session'):
        session.propose((1,), features(1))
    assert not case['hdc'].instances and not session.execution_summary()['validated_native_calls']


def test_mutated_prepared_input_fails_before_any_model_call_and_session_cannot_retry(case, monkeypatch):
    session = case['session']; actual = session._prepare
    def changed(*args):
        manifest = actual(*args)
        path = args[2] / manifest['inputs'][0]['path']; path.write_bytes(b'x' + path.read_bytes()[1:])
        return manifest
    monkeypatch.setattr(session, '_prepare', changed)
    with pytest.raises(ValueError, match='not completely validated'):
        session.propose((1, 2), features(2))
    assert not case['hdc'].instances and not session.execution_summary()['validated_native_calls']
    with pytest.raises(RuntimeError): session.propose((1, 2), features(2))


def test_native_failure_keeps_real_receipts_and_cannot_be_returned_as_success(case, monkeypatch):
    monkeypatch.setattr(case['hdc'], 'exitcode', 9)
    with pytest.raises(ValueError, match='not completely validated'):
        case['session'].propose((1, 2), features(2))
    assert (case['session'].out / 'call-0001/execution/result.json').exists()
    assert not case['session'].execution_summary()['validated_native_calls']


def test_post_native_output_change_blocks_return_even_after_proposal_metadata_write(case, monkeypatch):
    original = adapter.save_json
    def changed(path, value):
        original(path, value)
        if Path(path).name == 'proposal.json':
            output = Path(path).parent / 'execution/results/output_0000.bin'
            output.write_bytes(b'x' + output.read_bytes()[1:])
    monkeypatch.setattr(adapter, 'save_json', changed)
    with pytest.raises(ValueError, match='receipt/output changed'):
        case['session'].propose((1, 2), features(2))
    assert not case['session'].execution_summary()['validated_native_calls']


def test_final_prepared_revalidation_cannot_hide_native_output_change(case, monkeypatch):
    session = case['session']; original = session.validate_step
    def changed(root):
        manifest = original(root)
        if (Path(root).parent / 'proposal.json').exists():
            path = Path(root).parent / 'execution/results/output_0000.bin'
            path.write_bytes(b'x' + path.read_bytes()[1:])
        return manifest
    monkeypatch.setattr(session, 'validate_step', changed)
    with pytest.raises(ValueError, match='receipt/output changed'):
        session.propose((1, 2), features(2))
    assert not session.execution_summary()['validated_native_calls']
