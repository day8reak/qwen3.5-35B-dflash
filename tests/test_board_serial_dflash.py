"""Prepare the autonomous request without any model or transport execution."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import struct

import pytest

from qwen35_moe_dflash import board_serial_dflash as job


def write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return job._hash_source(path)


@pytest.fixture
def sources(tmp_path, monkeypatch):
    monkeypatch.setenv('AI_RUN_DIR', str(tmp_path))
    prompt = tmp_path / 'prompt.bin'
    write(prompt, struct.pack('<64i', *range(1, 65)))
    compiled = tmp_path / 'target/compilation'
    model = write(compiled / 'compile-work/generated/model/model.omc', b'target-fixture')
    artifact = dict(path='compile-work/generated/model/model.omc', bytes=model['bytes'], sha256=model['sha256'])
    report = write(compiled / 'report.json', json.dumps(dict(generated_artifacts=[artifact], omc=artifact)).encode())
    draft_root = tmp_path / 'draft'
    draft_model = write(draft_root / 'model/model.omc', b'draft-fixture')
    abi = write(draft_root / 'draft-abi.json', b'{"fixture":true}')
    base = dict(compilation=str(compiled), compilation_report_sha256=report['sha256'],
        primary_model_binding=model, model_sha256=model['sha256'], ddk_id='fixture-ddk',
        runtime_version='fixture-runtime', firmware_id='fixture-firmware')
    # The real native integration independently exercises the complete 89/82 ABI.
    # This preparation fixture only needs the already-bound source inventory.
    binding = dict(base_plan=base, state_mapping=[], active_source_identities=[report])
    source = dict(ddk_id='fixture-ddk', draft_abi=dict(path='draft-abi.json', sha256=abi['sha256']),
        outputs=[dict(index=0, name='draft_top1', dtype='int32', shape=[1, 15]),
                 dict(index=1, name='draft_hidden', dtype='float16', shape=[1, 15, 2048])])

    class Target:
        def __init__(self, *args, **kwargs):
            self.binding = deepcopy(binding)
            self.runtime_version = base['runtime_version']
            self.firmware_id = base['firmware_id']
        def preflight(self, prompt):
            assert prompt == list(range(1, 9))
            return dict(scope='host-fixture-only')

    class Draft:
        def __init__(self, *args, **kwargs):
            self._source = deepcopy(source)
            self._source_identities = {row['path']: row['identity'] for row in (draft_model, abi)}
        def preflight(self):
            return dict(model_source=deepcopy(draft_model))

    monkeypatch.setattr(job, '_backends', lambda: (Target, Draft))
    tables = {}
    for name, size in [('weight', 508559360), ('scale', 993280)]:
        path = tmp_path / (name + '.bin')
        with path.open('wb') as stream:
            stream.truncate(size)
        tables[name] = path
    options = dict(decode_job=tmp_path / 'target', runtime_probe=tmp_path / 'probe',
        state_lineage=tmp_path / 'lineage', source_torchair_pb2=tmp_path / 'schema.py',
        draft_bundle=draft_root, weight_table=tables['weight'], scale_table=tables['scale'],
        prompt_input_ids=prompt, output=tmp_path / 'new-run')
    return options, source, model


def test_preparation_binds_tables_and_sources_without_transport(sources, monkeypatch):
    options, _, model = sources
    from qwen35_moe_dflash import board_serial_transport as transport
    monkeypatch.setattr(transport, 'execute_board_serial', lambda *a, **k: pytest.fail('plan must not use HDC'))
    result = job.run_board_serial_dflash(**options)
    assert result['status'] == 'BOARD_SERIAL_DFLASH_PLANNED', result
    assert result['device_execution'] == 'NOT_RUN'
    manifest = job.validate_prepared(result['prepared'])
    assert manifest['prompt_token_ids'] == list(range(1, 9))
    assert manifest['target_payloads'][0]['primary']
    assert manifest['target_payloads'][0]['local'] == model['path']
    assert manifest['tables']['weight']['bytes'] == 508559360
    assert manifest['model_residency'].startswith('NOT_IMPLEMENTED')
    assert not list(options['output'].rglob('*.omc'))
    # Permissions/metadata are not silently reset or ignored by preparation.
    options['prompt_input_ids'].chmod(0o600)
    with pytest.raises(ValueError, match='source changed'):
        job.validate_prepared(result['prepared'])


def test_manifest_pin_and_complete_draft_contract_are_required(sources):
    options, source, _ = sources
    source['outputs'][1]['shape'] = [1, 15, 1024]
    rejected = job.run_board_serial_dflash(**options)
    assert rejected['status'] == 'FAILED'
    assert 'two-output draft ABI' in rejected['error']
    source['outputs'][1]['shape'] = [1, 15, 2048]
    manifest = job.prepare_board_serial(**(options | dict(output=options['output'].with_name('prepared'))))
    path = Path(manifest['output']) / 'manifest.json'
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='manifest SHA256 changed'):
        job.validate_prepared(path.parent)


@pytest.mark.parametrize('changes', [dict(prompt_length=60, max_new_tokens=8),
    dict(max_proposals_per_round=16), dict(board_model_root='/data/../escape')])
def test_invalid_request_stops_before_loading_models(sources, monkeypatch, changes):
    options, _, _ = sources
    monkeypatch.setattr(job, '_backends', lambda: pytest.fail('invalid request reached models'))
    result = job.run_board_serial_dflash(**(options | changes))
    assert result['status'] == 'FAILED' and result['device_execution'] == 'NOT_RUN'


def test_apply_forwards_one_detached_request_and_does_not_claim_completion(sources, monkeypatch):
    options, _, _ = sources
    from qwen35_moe_dflash import board_serial_transport as transport
    calls = []
    def start(prepared, output, **kwargs):
        calls.append((prepared, output, kwargs))
        return dict(status='BOARD_SERIAL_DFLASH_STARTED', device_execution='RUNNING_OR_COMPLETION_PENDING')
    monkeypatch.setattr(transport, 'execute_board_serial', start)
    result = job.run_board_serial_dflash(**options, apply=True, detach=True)
    assert result['status'] == 'BOARD_SERIAL_DFLASH_STARTED'
    assert len(calls) == 1 and calls[0][2]['detach'] is True
    assert result['device_execution'] == 'RUNNING_OR_COMPLETION_PENDING'


def test_wire_config_preserves_explicit_nonordinal_state_mapping(tmp_path):
    record = dict(bytes=1, sha256='a' * 64)
    manifest = dict(target_model=record, draft=dict(model_source=record, abi=record),
        tables=dict(weight=record, scale=record), prompt_token_ids=[1, 2],
        max_new_tokens=8, max_proposals_per_round=2, minimum_draft_rounds=2, timeout_seconds=120,
        target_binding=dict(base_plan=dict(runtime_version='v', ddk_id='d', firmware_id='f',
            inputs=[dict(index=0, name='RefData_9', tool_input_type='FP16', bytes=65536)],
            outputs=[dict(index=47, runtime_tensor_name='native:output.47', tool_output_type='FP16', bytes=65536)]),
            state_mapping=[dict(source_refdata='RefData_9', runtime_input_index=0, runtime_output_index=47)]))
    paths = {key: '/board/' + key for key in ('target_model', 'draft_model', 'draft_abi',
        'target_tool', 'draft_runner', 'weight_table', 'scale_table')}
    paths.update(target_tool_record=record, draft_runner_record=record,
        target_library_path='/system/lib64', draft_library_path='/board/lib')
    wire = job.make_config(manifest, paths)
    assert b'state\tRefData_9\t0\t47\t65536\n' in wire
    assert b'target_output\t47\tnative:output.47\tFP16\t65536\n' in wire
    paths['target_model'] = '/board/evil\nstate\tRefData_0'
    with pytest.raises(ValueError, match='TSV'):
        job.make_config(manifest, paths)


def test_cli_recovery_selects_existing_attempt_and_never_applies(tmp_path, monkeypatch, capsys):
    script = Path(__file__).parents[1] / 'targets/cpro/scripts/run_board_serial_dflash.py'
    spec = importlib.util.spec_from_file_location('board_serial_cli_test', script)
    cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    old = tmp_path / 'old'; (old / 'board').mkdir(parents=True)
    from qwen35_moe_dflash import board_serial_transport as transport
    calls = []
    def recover(execution, output, **kwargs):
        calls.append((execution, output))
        return dict(status='BOARD_SERIAL_DFLASH_RECOVERED_UNVALIDATED')
    monkeypatch.setattr(transport, 'recover_board_serial', recover)
    assert cli.main(['--recover', str(old), '--output', str(tmp_path / 'recovery')]) == 0
    assert calls[0][0] == old / 'board'
    with pytest.raises(SystemExit):
        cli.main(['--recover', str(old), '--output', str(tmp_path / 'bad'), '--apply'])
