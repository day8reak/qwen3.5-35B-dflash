"""Runtime-proved prefill cache reuse; synthetic payloads and real local sh.

No NPU is called. The existing host planner fixture is retained rather than
substituting a successful proof, and captured production cache shell runs only
inside pytest's owned temporary directories.
"""
import hashlib
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess

import pytest

from qwen35_moe_dflash import original_target_board as board
from qwen35_moe_dflash.hashing import sha256_file
from test_original_target_primary_revalidation import ctime_only
from test_target_prefill_smoke_cache import _add_small_external_weight
from test_target_prefill_smoke_execution import FakeHdc, case as prefill_case, execute


@pytest.fixture
def case(tmp_path, monkeypatch):
    value = prefill_case.__wrapped__(tmp_path, monkeypatch)
    weight = _add_small_external_weight(value)
    return dict(value, weight=weight)


def labels():
    return [label for instance in FakeHdc.instances for _, label, _ in instance.calls]


def cached_payloads(case, hdc=FakeHdc):
    cache = PurePosixPath('/data/local/tmp/qwen3_5_moe') / ('codex-target-' + sha256_file(case['compilation'] / 'report.json')[:20])
    for row in case['compiled']['generated_artifacts']:
        remote = cache / 'payload' / PurePosixPath(row['path']).relative_to('compile-work/generated')
        hdc.remote[str(remote)] = (case['compilation'] / row['path']).read_bytes()
    return cache


@pytest.mark.parametrize('changed', ['ctime', 'missing', 'different_bytes'])
def test_proved_prefill_cache_hit_never_validates_or_reads_local_nonprimary(case, monkeypatch, changed):
    cached_payloads(case)
    if changed == 'ctime': ctime_only(case['weight'])
    elif changed == 'missing': case['weight'].rename(case['weight'].with_suffix('.held'))
    else: case['weight'].write_bytes(b'changed local fixture bytes')
    original_record, original_hash = board._recorded_board_payload, board.sha256_file
    checked = []
    def primary_only(root, row, compiled, **options):
        checked.append(row['path'])
        assert not row['path'].endswith('.weight'), 'cache hit reached local artifact identity validation'
        return original_record(root, row, compiled, **options)
    def no_weight_hash(path):
        assert not str(path).endswith('.weight'), 'cache hit must not freshly hash a local weight'
        return original_hash(path)
    monkeypatch.setattr(board, '_recorded_board_payload', primary_only)
    monkeypatch.setattr(board, 'sha256_file', no_weight_hash)
    result = execute(case)
    assert result['status'] == 'EXECUTED_UNVALIDATED', result
    assert result['board_disk_budget']['missing_model_payload_bytes'] == 0
    assert checked == [case['compiled']['omc']['path']]
    assert all(row['upload'] == 'OK' for row in result['model_payloads'])
    assert not any(local.endswith(('.weight', '.omc')) for local, _ in FakeHdc.sends)
    assert result['local_cached_payload_access'] == 'NONPRIMARY_CACHE_HITS_NOT_OPENED; REMOTE_SHA_MATCHED'
    assert result['source_preservation'] == 'UNCHANGED_PREFILL_PLAN_METADATA_AND_ORIGINAL_INPUT_SHA256'
    assert result['numerical_validation'] == 'NOT_RUN_NO_REFERENCE'


@pytest.mark.parametrize('remote', ['missing', 'corrupt'])
def test_prefill_cache_miss_keeps_original_local_stat_gate_before_any_payload_send(case, remote):
    cache = cached_payloads(case)
    target = str(cache / 'payload/model/SubGraph_0.weight')
    if remote == 'missing': FakeHdc.remote.pop(target)
    else: FakeHdc.remote[target] = b'corrupt remote fixture'
    ctime_only(case['weight'])
    result = execute(case)
    assert result['status'] == 'FAILED' and 'changed since compilation' in result['error'], result
    assert 'execute-original-target' not in labels()
    assert not any(local.endswith(('.weight', '.bin')) for local, _ in FakeHdc.sends)
    assert result['lock_state'] == 'NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT'


def test_valid_prefill_cache_miss_sends_only_strictly_unchanged_local_weight(case):
    cache = cached_payloads(case)
    target = str(cache / 'payload/model/SubGraph_0.weight')
    FakeHdc.remote.pop(target)
    expected = case['weight'].read_bytes()
    result = execute(case)
    assert result['status'] == 'EXECUTED_UNVALIDATED', result
    assert result['board_disk_budget']['missing_model_payload_bytes'] == len(expected)
    assert FakeHdc.remote[target] == expected
    assert [(local, remote) for local, remote in FakeHdc.sends if local.endswith(('.omc', '.weight'))] == [(str(case['weight']), target)]


@pytest.mark.parametrize('stage', ['own-model-cache', 'model-directories', 'inspect-model-cache'])
def test_remote_cache_owner_or_sha_timeout_retains_locks_without_retry(case, monkeypatch, stage):
    cached_payloads(case)
    monkeypatch.setattr(FakeHdc, 'timeout_label', stage)
    result = execute(case)
    assert result['status'] == 'FAILED' and result['lock_state'] == 'RETAINED_OR_UNCERTAIN', result
    assert 'execute-original-target' not in labels() and 'release-global-model-io-lock' not in labels()
    assert not any(local.endswith(('.weight', '.bin', '.omc')) for local, _ in FakeHdc.sends)


@pytest.mark.parametrize('response', ['partial', 'duplicate', 'malformed'])
def test_incomplete_cache_inventory_remains_uncertain_and_cannot_send_or_execute(case, monkeypatch, response):
    cached_payloads(case)
    original = FakeHdc.shell
    def changed(self, command, label, **options):
        result = original(self, command, label, **options)
        if label != 'inspect-model-cache': return result
        rows = result[1].splitlines()
        if response == 'partial': rows = rows[:1]
        elif response == 'duplicate': rows.append(rows[0])
        else: rows = ['[Fail] Device not connected']
        return 0, '\n'.join(rows)
    monkeypatch.setattr(FakeHdc, 'shell', changed)
    result = execute(case)
    assert result['status'] == 'FAILED' and result['lock_state'] == 'RETAINED_OR_UNCERTAIN', result
    assert 'execute-original-target' not in labels() and 'release-global-model-io-lock' not in labels()
    assert not any(local.endswith(('.weight', '.bin', '.omc')) for local, _ in FakeHdc.sends)


def test_unproved_prefill_never_enters_cache_first_lane(case, monkeypatch):
    case['native']['inputs'][0]['raw_format'] = -1
    from test_target_prefill_smoke_plan import rewrite_native
    rewrite_native(case)
    monkeypatch.setattr(board, '_recorded_board_payload', lambda *a, **k: pytest.fail('invalid proof reached payload checks'))
    result = execute(case)
    assert result['status'] == 'FAILED' and not FakeHdc.instances and not FakeHdc.sends


def test_ordinary_decode_cache_hit_still_uses_strict_local_payload_first(tmp_path, monkeypatch):
    from test_target_decode_smoke_execution import DecodeFakeHdc, case as decode_case, execute as execute_decode
    value = decode_case.__wrapped__(tmp_path, monkeypatch)
    weight = _add_small_external_weight(value)
    cached_payloads(value, DecodeFakeHdc)
    ctime_only(weight)
    result = execute_decode(value)
    assert result['status'] == 'FAILED' and 'changed since compilation' in result['error'], result
    calls = [label for instance in DecodeFakeHdc.instances for _, label, _ in instance.calls]
    assert 'inspect-model-cache' not in calls and 'execute-original-target' not in calls


@pytest.fixture
def shell_case(case, tmp_path):
    result = execute(case)
    assert result['status'] == 'EXECUTED_UNVALIDATED', result
    remote_cache = result['board_model_cache']
    remote_root = str(PurePosixPath(remote_cache).parent)
    local_root = tmp_path / 'local-shell-cache-root'
    local_cache = local_root / PurePosixPath(remote_cache).name
    commands = {}
    for command, label, _ in FakeHdc.instances[0].calls:
        if label in {'model-directories', 'inspect-model-cache', 'own-model-cache'}:
            relocated = command.replace(remote_cache, shlex.quote(str(local_cache))).replace(remote_root, shlex.quote(str(local_root)))
            assert remote_cache not in relocated and remote_root not in relocated
            commands[label] = relocated
    assert len(commands) == 3
    local_root.mkdir()
    return {'case': case, 'cache': local_cache, 'commands': commands,
        'manifest': (case['out'] / 'model-files.tsv').read_text().replace(remote_cache, str(local_cache))}


def shell(value, label, prefix=''):
    return subprocess.run(['sh', '-c', prefix + value['commands'][label]], capture_output=True, text=True, timeout=10)


def populate(value):
    assert shell(value, 'own-model-cache').returncode == 0
    assert shell(value, 'model-directories').returncode == 0
    for row in value['case']['compiled']['generated_artifacts']:
        target = value['cache'] / 'payload' / PurePosixPath(row['path']).relative_to('compile-work/generated')
        shutil.copyfile(value['case']['compilation'] / row['path'], target)
    (value['cache'] / 'model-files.tsv').write_text(value['manifest'])


def test_production_prefill_cache_shell_hashes_all_plain_payloads_and_distinguishes_misses(shell_case):
    value = shell_case
    populate(value)
    observed = shell(value, 'inspect-model-cache')
    assert observed.returncode == 0, observed.stderr
    assert len(observed.stdout.splitlines()) == 2 and all(row.startswith('OK ') for row in observed.stdout.splitlines())
    weight = value['cache'] / 'payload/model/SubGraph_0.weight'
    weight.write_bytes(b'corrupt synthetic cache')
    corrupt = shell(value, 'inspect-model-cache')
    assert corrupt.returncode == 0 and 'SEND ' + str(weight) in corrupt.stdout.splitlines()
    weight.unlink()
    missing = shell(value, 'inspect-model-cache')
    assert missing.returncode == 0 and 'SEND ' + str(weight) in missing.stdout.splitlines()


@pytest.mark.parametrize('level', ['payload', 'payload/model', 'model-files.tsv', '.compilation-sha256'])
def test_production_prefill_directory_and_owner_shell_refuse_symlinks_without_target_mutation(shell_case, tmp_path, level):
    value = shell_case
    value['cache'].mkdir()
    target = tmp_path / 'unrelated-target'
    is_directory = level.startswith('payload')
    if is_directory:
        target.mkdir(); marker = target / 'sentinel'; marker.write_bytes(b'keep')
    else:
        target.write_text(sha256_file(value['case']['compilation'] / 'report.json') + '\n'); marker = target
    link = value['cache'] / level
    link.parent.mkdir(exist_ok=True)
    link.symlink_to(target, target_is_directory=is_directory)
    before = (marker.read_bytes(), marker.stat().st_mtime_ns)
    result = shell(value, 'own-model-cache' if level == '.compilation-sha256' else 'model-directories')
    assert result.returncode != 0 and link.is_symlink()
    assert (marker.read_bytes(), marker.stat().st_mtime_ns) == before


@pytest.mark.parametrize('kind', ['matching_symlink', 'dangling_symlink', 'fifo', 'directory'])
def test_production_prefill_hash_shell_refuses_unsafe_payload_types(shell_case, tmp_path, kind):
    value = shell_case
    populate(value)
    target = tmp_path / 'unrelated-synthetic-weight'
    link = value['cache'] / 'payload/model/SubGraph_0.weight'
    raw = link.read_bytes(); link.unlink()
    if kind == 'matching_symlink': target.write_bytes(raw); link.symlink_to(target)
    elif kind == 'dangling_symlink': link.symlink_to(target)
    elif kind == 'fifo': os.mkfifo(link)
    else: link.mkdir()
    result = shell(value, 'inspect-model-cache')
    assert result.returncode != 0
    assert not any(row.endswith(' ' + str(link)) for row in result.stdout.splitlines())
    if target.exists(): assert target.read_bytes() == raw


@pytest.mark.parametrize('replacement', ['return 1', "printf 'invalid hash\\n'"])
def test_production_prefill_hash_failure_is_not_classified_as_send(shell_case, replacement):
    populate(shell_case)
    result = shell(shell_case, 'inspect-model-cache', 'sha256sum() { ' + replacement + '; }; ')
    assert result.returncode != 0 and not result.stdout
