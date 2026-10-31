"""Movable runtime configuration and fixed test slots; no device execution."""
import json
from pathlib import Path
import shutil
import struct

import pytest

from qwen35_moe_dflash.runtime_config import export_config, prepare_config, validate_assets, read_json
from qwen35_moe_dflash.run_layout import organize, reuse_test_output, relocated_path
from test_board_serial_integration import (make_case, enable_resident_mixed, execute,
                                          mixed_native_programs)


@pytest.fixture
def portable(tmp_path, monkeypatch):
    monkeypatch.setenv('AI_RUN_DIR', str(tmp_path))
    case = make_case(tmp_path / 'fixture', (Path('/bin/true'), Path('/bin/true')), dump=False)
    enable_resident_mixed(case)
    manifest = case['manifest']
    manifest['board_model_root'] = '/data/local/tmp/models'
    for role, key in [('prefill', 'prefill_abi'), ('decode', 'resident_abi')]:
        base = manifest['prefill']['base_plan'] if role == 'prefill' else manifest['target_binding']['base_plan']
        desc = manifest[key]
        for direction in ('inputs', 'outputs'):
            desc[direction] = []
        for line in Path(desc['path']).read_text().splitlines()[1:]:
            kind, i, name, dtype, raw, fmt, shape, size = line.split('\t')
            direction = 'inputs' if kind.endswith('_input') else 'outputs'
            row = dict(index=int(i), name=name, dtype=dtype, raw_dtype=int(raw), raw_format=int(fmt),
                       shape=list(map(int, shape.split(','))), bytes=int(size))
            desc[direction].append(row)
            base[direction][int(i)].update(runtime_tensor_name=name, dtype=dtype, shape=row['shape'])
        base.update(compilation_report_sha256=('a' if role == 'prefill' else 'b')*64,
                    runtime_version='FAKE-HIAI-X90', ddk_id='host-fixture', firmware_id='host-fixture')
        model = manifest['prefill']['model'] if role == 'prefill' else manifest['target_model']
        folder = tmp_path / ('target-pytorch-' + role + '-compile')
        folder.mkdir(); dest = folder / 'model.omc'; Path(model['path']).rename(dest)
        model['path'] = str(dest)
        rows = [dict(local=str(dest), remote='model.omc', bytes=model['bytes'], primary=True)]
        if role == 'prefill': manifest['prefill']['payloads'] = rows
        else: manifest['target_payloads'] = rows
    base = manifest['target_binding']['base_plan']
    for row in manifest['target_binding']['state_mapping']:
        row.update(runtime_input_name=base['inputs'][row['runtime_input_index']]['runtime_tensor_name'],
                   runtime_output_name=base['outputs'][row['runtime_output_index']]['runtime_tensor_name'])
    for record, folder in [(manifest['draft']['model_source'], 'full-draft'),
            (manifest['tables']['weight'], 'original-embedding-tables'),
            (manifest['tables']['scale'], 'original-embedding-tables')]:
        directory = tmp_path / folder; directory.mkdir(exist_ok=True)
        source = Path(record['path']); destination = directory / source.name
        source.rename(destination); record['path'] = str(destination)
    manifest['draft']['outputs'] = []
    old = tmp_path / 'dflash-old-run'; prepared = old / 'prepared'; prepared.mkdir(parents=True)
    (prepared / 'manifest.json').write_text(json.dumps(manifest))
    config = tmp_path / 'runtime.json'; export_config(prepared, config)
    prompt = tmp_path / 'prompt.bin'; prompt.write_bytes(struct.pack('<8i', *range(1, 9)))
    return config, prompt, old


def completed(path, attempt='attempt-1', status='FAILED'):
    path.mkdir(exist_ok=True)
    (path / 'result.json').write_text(json.dumps(dict(artifact_kind='qwen35-board-serial-dflash-execution',
        status=status, stage='complete', board_attempt=attempt,
        lock_state='NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT', counts={'accepted': 0})))


def test_move_cleanup_then_real_native_loop(portable, tmp_path, mixed_native_programs):
    config, prompt, old = portable
    completed(old)
    (old / 'large-intermediate.bin').write_bytes(b'temporary')
    (old / 'op.csv').write_text('op,time\nx,1\n')
    preview = organize(tmp_path, config, clean_completed=True)
    assert old.name in preview['delete'] and old.exists()
    result = organize(tmp_path, config, apply=True, clean_completed=True)
    assert result['status'] == 'COMPLETE' and not old.exists()
    assert (tmp_path / 'results/dflash-old-run/profiling/op.csv').is_file()
    assert (tmp_path / 'prefill/target-pytorch-prefill-compile/model.omc').is_file()
    assert (tmp_path / 'decode/target-pytorch-decode-compile/model.omc').is_file()
    # Whole run root can also move. Runtime ignores stale hashes and timestamps.
    moved = tmp_path / 'relocated'
    moved.mkdir()
    for path in list(tmp_path.iterdir()):
        if path != moved: path.rename(moved / path.name)
    config, prompt = moved / config.name, moved / prompt.name
    draft = moved / read_json(config)['models']['draft']['path']
    draft.touch()
    prepared = prepare_config(config, moved / 'work/current/prepared', prompt_input_ids=prompt)
    from qwen35_moe_dflash.board_serial_dflash import make_config, validate_prepared
    from qwen35_moe_dflash.board_serial_transport import validate_retrieved_tokens
    assert validate_prepared(prepared['output'])['file_validation'] == 'size-and-abi-only'
    paths = dict(target_model=prepared['target_model']['path'], prefill_model=prepared['prefill']['model']['path'],
        draft_model=prepared['draft']['model_source']['path'], draft_abi=prepared['draft']['abi']['path'],
        weight_table=prepared['tables']['weight']['path'], scale_table=prepared['tables']['scale']['path'],
        resident_abi=prepared['resident_abi']['path'], prefill_abi=prepared['prefill_abi']['path'],
        target_library_path=str(moved), draft_library_path=str(moved))
    tsv = moved / 'loop.tsv'; tsv.write_bytes(make_config(prepared, paths))
    case = dict(root=moved, config=tsv, supervisor=mixed_native_programs[0], output=moved / 'loop')
    process = execute(case)
    assert process.returncode == 0, process.stderr
    loop = validate_retrieved_tokens(prepared, case['output'], _expected_evidence_kind='host-fake-hiai')
    assert loop['generated_tokens'] == list(range(9, 17))
    assert loop['counts']['prefill_native_calls'] == 1
    assert loop['startup_payload_validation'] == 'SIZE_AND_ABI_ONLY'


def test_layout_idempotence_and_unknown_active_retained(portable, tmp_path):
    config, _, old = portable
    pending = tmp_path / 'dflash-pending'; pending.mkdir()
    (pending / 'result.json').write_text(json.dumps(dict(board_attempt='pending', status='RUNNING')))
    unknown = tmp_path / 'unknown-data'; unknown.mkdir()
    result = organize(tmp_path, config, apply=True, clean_completed=True)
    assert pending.exists() and unknown.exists() and old.exists()
    assert result['moves'] == [] and result['unresolved_board_attempts'] == ['pending']
    completed(pending, 'pending')
    organize(tmp_path, config, apply=True, clean_completed=True)
    before = config.read_text()
    organize(tmp_path, config, apply=True, clean_completed=True)
    assert before == config.read_text() and not pending.exists()


def test_interrupted_moves_resume(portable, tmp_path, monkeypatch):
    config, _, _ = portable
    rename = Path.rename
    def fail_once(path, destination):
        if path.name.startswith('target-pytorch-prefill'):
            raise OSError('simulated interruption')
        return rename(path, destination)
    monkeypatch.setattr(Path, 'rename', fail_once)
    with pytest.raises(OSError): organize(tmp_path, config, apply=True)
    monkeypatch.setattr(Path, 'rename', rename)
    result = organize(tmp_path, config, apply=True)
    assert result['status'] == 'COMPLETE'
    assert relocated_path(tmp_path / 'full-draft/draft.omc', tmp_path).is_file()


def test_reuse_closed_failed_slot_not_live_slot(tmp_path, monkeypatch):
    monkeypatch.setenv('AI_RUN_DIR', str(tmp_path))
    slot = tmp_path / 'dflash-current'; slot.mkdir()
    (slot / 'result.json').write_text(json.dumps(dict(artifact_kind='qwen35-board-serial-dflash-job',
        status='STARTED', board_attempt='live', device_execution='RUNNING_OR_COMPLETION_PENDING')))
    with pytest.raises(ValueError, match='exit unknown'): reuse_test_output(slot)
    recovery = tmp_path / 'dflash-recovery'; completed(recovery, 'live')
    reuse_test_output(slot)
    assert not slot.exists() and recovery.exists()
    completed(slot, 'new')
    reuse_test_output(slot)
    assert len(list((tmp_path / 'results').iterdir())) == 1


def test_reject_corrupt_abi_and_truncated_model(portable, tmp_path):
    config, prompt, _ = portable
    assets = read_json(tmp_path / read_json(config)['assets'])
    assets['prefill']['state_mapping'][0]['prefill']['runtime_input_index'] = -1
    with pytest.raises(ValueError, match='outside runtime ABI'): validate_assets(assets)
    model = tmp_path / 'target-pytorch-decode-compile/model.omc'; model.write_bytes(b'x')
    with pytest.raises(ValueError, match='extent'): prepare_config(config, tmp_path / 'work/test', prompt_input_ids=prompt)


def test_config_cli_reuses_same_finished_output(portable, tmp_path):
    import importlib.util
    config, prompt, _ = portable
    script = Path(__file__).parents[1] / 'targets/cpro/scripts/run_board_serial_dflash.py'
    spec = importlib.util.spec_from_file_location('run_portable', script)
    cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    output = tmp_path / 'work/fixed-test'
    args = ['--config', str(config), '--prompt-input-ids', str(prompt), '--output', str(output)]
    assert cli.main(args) == 0
    (output / 'obsolete.bin').write_bytes(b'failed intermediate')
    assert cli.main(args) == 0
    assert not (output / 'obsolete.bin').exists()
    assert read_json(output / 'result.json')['status'] == 'BOARD_SERIAL_DFLASH_PLANNED'


def test_migration_conflict_changes_nothing(portable, tmp_path):
    config, _, _ = portable
    collision = tmp_path / 'draft/full-draft'; collision.mkdir(parents=True)
    (collision / 'unrelated').write_bytes(b'preserve')
    with pytest.raises(FileExistsError): organize(tmp_path, config, apply=True)
    assert (tmp_path / 'full-draft/draft.omc').is_file()
    assert (collision / 'unrelated').read_bytes() == b'preserve'


def test_new_policy_upload_has_no_sha_gate(tmp_path, monkeypatch):
    from qwen35_moe_dflash import board_serial_transport as t
    source = tmp_path / 'omc'; source.write_bytes(b'changed')
    assert t._local_upload(dict(local=str(source), bytes=7, sha256='f'*64),
                           {'file_validation': 'size-and-abi-only'}) == source
    with pytest.raises(ValueError, match='extent'):
        t._local_upload(dict(local=str(source), bytes=8, sha256='f'*64),
                        {'file_validation': 'size-and-abi-only'})


def test_interrupted_cleanup_preserves_csv_on_resume(portable, tmp_path, monkeypatch):
    config, _, old = portable
    completed(old)
    (old / 'op.csv').write_text('op,time\nx,7\n')
    remove = shutil.rmtree
    def interrupt(path, *args, **kwargs):
        if Path(path) == old:
            (old / 'op.csv').unlink()
            raise OSError('cleanup interrupted after archive')
        return remove(path, *args, **kwargs)
    monkeypatch.setattr(shutil, 'rmtree', interrupt)
    with pytest.raises(OSError): organize(tmp_path, config, apply=True, clean_completed=True)
    monkeypatch.setattr(shutil, 'rmtree', remove)
    organize(tmp_path, config, apply=True, clean_completed=True)
    assert not old.exists()
    assert (tmp_path / 'results/dflash-old-run/profiling/op.csv').read_text() == 'op,time\nx,7\n'


def test_prelaunch_unknown_lock_prevents_output_reuse(tmp_path, monkeypatch):
    monkeypatch.setenv('AI_RUN_DIR', str(tmp_path))
    path = tmp_path / 'dflash-failed'; path.mkdir()
    (path / 'result.json').write_text(json.dumps(dict(status='FAILED', device_execution='NOT_RUN',
        lock_state='RETAINED_OR_UNCERTAIN')))
    with pytest.raises(ValueError, match='exit unknown'): reuse_test_output(path)
    assert path.is_dir()
