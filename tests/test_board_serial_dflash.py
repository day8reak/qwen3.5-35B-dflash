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
    assert manifest['dump'] is False
    assert result['retrieval_mode'] == 'FINAL_TOKENS_ONLY'
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
    dict(max_proposals_per_round=16), dict(board_model_root='/data/../escape'), dict(dump=1), dict(resident=True)])
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
    assert b'dump\t1\n' in wire  # old prepared manifests preserve debug behavior
    assert b'dump\t0\n' in job.make_config(manifest | {'dump': False}, paths)
    assert b'dump\t1\n' in job.make_config(manifest | {'dump': True}, paths)
    with pytest.raises(ValueError, match='dump mode'):
        job.make_config(manifest | {'dump': 'false'}, paths)
    assert b'state\tRefData_9\t0\t47\t65536\n' in wire
    assert b'target_output\t47\tnative:output.47\tFP16\t65536\n' in wire
    resident = manifest | {'execution_mode': 'resident-v2', 'resident_abi': record}
    assert b'execution_mode\tresident-v2\n' in job.make_config(resident, paths | {'resident_abi': '/board/raw-abi.tsv'})
    assert b'resident_abi\t/board/raw-abi.tsv\t' in job.make_config(resident, paths | {'resident_abi': '/board/raw-abi.tsv'})
    assert b'target_lifecycle_policy\t' not in job.make_config(resident, paths | {'resident_abi': '/board/raw-abi.tsv'})
    assert b'target_lifecycle_policy\tinit-before-weights-close-weights-first-v1\n' in job.make_config(
        resident | {'target_lifecycle_policy': job.V2_TARGET_LIFECYCLE_POLICY}, paths | {'resident_abi': '/board/raw-abi.tsv'})
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
    with pytest.raises(SystemExit):
        cli.main(['--recover', str(old), '--output', str(tmp_path / 'bad'), '--dump'])
    with pytest.raises(SystemExit):
        cli.main(['--recover', str(old), '--output', str(tmp_path / 'bad'), '--resident'])


def _native_descriptor_binding(tmp_path):
    enums = dict(FLOAT32=7, FLOAT16=8, INT8=9, INT16=10, INT32=11, INT64=12)
    native = dict(artifact_kind='qwen35-hiai-model-io-probe', evidence_kind='device-hiai', status='IO_DESCRIPTORS_OBSERVED',
        model_sha256='a' * 64, model_bytes=17, runtime_version='v', ddk_id='d', firmware_id='f',
        dtype_enum='hiai::DataType', format_enum='hiai::Format', dtype_enum_values=enums, format_enum_values=dict(ND=13),
        input_count=89, output_count=82)
    base = dict(model_sha256='a' * 64, primary_model_binding=dict(bytes=17), runtime_version='v', ddk_id='d', firmware_id='f')
    for direction, count, type_key in [('inputs', 89, 'tool_input_type'), ('outputs', 82, 'tool_output_type')]:
        native[direction] = [dict(index=i, tensor_name_status='OBSERVED', tensor_name=direction + str(i),
            shape=[1, i + 1], raw_dtype=8, raw_format=13) for i in range(count)]
        base[direction] = [dict(index=i, runtime_tensor_name=direction + str(i), bytes=2 * (i + 1), **{type_key: 'FP16'}) for i in range(count)]
    probe = tmp_path / 'probe'; path = probe / 'model-io.json'; record = write(path, json.dumps(native).encode())
    return dict(base_plan=base, active_source_identities=[record]), probe, native


def test_resident_descriptor_abi_uses_actual_raw_enums_names_and_shapes(tmp_path):
    binding, probe, native = _native_descriptor_binding(tmp_path)
    result = job._resident_abi(binding, probe, tmp_path / 'resident-abi.tsv')
    wire = Path(result['path']).read_text()
    assert wire.startswith('QWEN35_RESIDENT_ABI\t1\n')
    assert 'target_input\t42\tinputs42\tFLOAT16\t8\t13\t1,43\t86\n' in wire
    assert len(result['inputs']) == 89 and len(result['outputs']) == 82
    assert result['descriptor_source']['sha256'] == job.sha256_file(probe / 'model-io.json')
    assert not list(tmp_path.glob('*.omc'))


def test_prefill_v1_abi_retains_v2_descriptor_provenance_without_invented_sdk_names(tmp_path):
    binding, probe, _ = _native_descriptor_binding(tmp_path)
    result = job._resident_abi(binding, probe, tmp_path / 'prefill-abi.tsv', prefix='prefill')
    wire = Path(result['path']).read_text()
    assert 'prefill_input\t42\tinputs42\tFLOAT16\t8\t13\t1,43\t86\n' in wire
    assert 'target_input\t' not in wire
    assert result['descriptor_provenance'] == 'HISTORICAL_FILE_V2_NAMED_MODEL_IO_PROBE'
    assert result['runtime_validation'] == 'V1_SLOT_SHAPE_AND_BYTE_EXTENT_REQUIRED_NO_SDK_TENSOR_NAME_CLAIM'


def test_new_resident_preparation_pins_both_v1_abis_and_keeps_source_models_read_only(sources, monkeypatch):
    options, _, model = sources
    from qwen35_moe_dflash import board_prefill_binding
    before = job._current(model['path'])[1]
    monkeypatch.setattr(board_prefill_binding, 'prepare_prefill_binding', lambda *args: dict(model=model,
        base_plan=dict(runtime_probe=str(options['runtime_probe'])), source_bindings=[],
        control_recipe=board_prefill_binding.RECIPE))
    selected = []
    def abi(binding, probe, output, *, prefix='target'):
        selected.append(prefix)
        record = write(Path(output), ('QWEN35_RESIDENT_ABI\t1\n' + prefix + '_input\tfixture\n').encode())
        return dict(record, inputs=[{}] * 89, outputs=[{}] * 82,
            descriptor_provenance='HISTORICAL_FILE_V2_NAMED_MODEL_IO_PROBE')
    monkeypatch.setattr(job, '_resident_abi', abi)
    manifest = job.prepare_board_serial(**options, resident=True, prefill_feedback=options['state_lineage'])
    assert selected == ['target', 'prefill']
    assert manifest['execution_mode'] == 'resident-v1'
    assert manifest['target_lifecycle_policy'] == 'legacy-v1-file-target-buffer-draft-v1'
    assert manifest['model_load_api'] == 'legacy-v1-mixed'
    assert manifest['model_load_apis'] == {'prefill': 'legacy-file', 'target': 'legacy-file', 'draft': 'legacy-buffer'}
    assert manifest['prefill_abi']['path'].endswith('/prefill-abi.tsv')
    assert job._current(model['path'])[1] == before
    assert not list(options['output'].rglob('*.omc'))


def test_v1_wire_uses_native_abi_and_no_v2_tool_or_legacy_child_runner(tmp_path):
    record = dict(bytes=1, sha256='a' * 64)
    manifest = dict(target_model=record, draft=dict(model_source=record, abi=record),
        tables=dict(weight=record, scale=record), resident_abi=record, prefill_abi=record,
        execution_mode='resident-v1', target_lifecycle_policy=job.TARGET_LIFECYCLE_POLICY,
        model_load_api=job.V1_MODEL_LOAD_API, model_load_apis=dict(job.V1_MODEL_LOAD_APIS),
        prompt_token_ids=[1, 2], max_new_tokens=8, max_proposals_per_round=2, minimum_draft_rounds=2,
        timeout_seconds=120, target_binding=dict(base_plan=dict(runtime_version='v', ddk_id='d', firmware_id='f',
            inputs=[], outputs=[]), state_mapping=[]))
    paths = {key: '/board/' + key for key in ('target_model', 'draft_model', 'draft_abi', 'weight_table',
        'scale_table', 'resident_abi', 'prefill_abi')}
    paths.update(target_library_path='/board/lib', draft_library_path='/board/lib')
    wire = job.make_config(manifest, paths)
    assert b'execution_mode\tresident-v1\n' in wire
    assert b'target_lifecycle_policy\tlegacy-v1-file-target-buffer-draft-v1\n' in wire
    assert b'prefill_abi\t/board/prefill_abi\t' in wire
    assert b'target_tool\t' not in wire and b'draft_runner\t' not in wire
    old = dict(manifest, target_lifecycle_policy=job.HISTORICAL_V1_LIFECYCLE_POLICY)
    for key in ('model_load_api', 'model_load_apis'): old.pop(key)
    historical = job.make_config(old, paths)
    assert b'target_lifecycle_policy\tlegacy-v1-load-process-unload-v1\n' in historical
    assert b'legacy-v1-file-target-buffer-draft-v1' not in historical


@pytest.mark.parametrize('damage', ['missing-map', 'missing-api', 'prefill-buffer', 'target-buffer',
    'draft-file', 'unknown-policy', 'old-policy-new-map'])
def test_v1_prepared_source_contract_rejects_mixed_or_implicit_modes(damage):
    manifest = dict(execution_mode='resident-v1', target_lifecycle_policy=job.TARGET_LIFECYCLE_POLICY,
        model_load_api=job.V1_MODEL_LOAD_API, model_load_apis=dict(job.V1_MODEL_LOAD_APIS))
    if damage == 'missing-map': manifest.pop('model_load_apis')
    elif damage == 'missing-api': manifest.pop('model_load_api')
    elif damage == 'prefill-buffer': manifest['model_load_apis']['prefill'] = 'legacy-buffer'
    elif damage == 'target-buffer': manifest['model_load_apis']['target'] = 'legacy-buffer'
    elif damage == 'draft-file': manifest['model_load_apis']['draft'] = 'legacy-file'
    elif damage == 'unknown-policy': manifest['target_lifecycle_policy'] = 'unbound-v1-policy'
    elif damage == 'old-policy-new-map': manifest['target_lifecycle_policy'] = job.HISTORICAL_V1_LIFECYCLE_POLICY
    with pytest.raises(ValueError, match='source|representation'):
        job.v1_load_contract(manifest)


@pytest.mark.parametrize('damage', ['enum', 'shape', 'name', 'changed-source'])
def test_resident_descriptor_abi_refuses_unbound_or_guessed_descriptor_fields(tmp_path, damage):
    binding, probe, native = _native_descriptor_binding(tmp_path)
    if damage == 'enum': native['inputs'][0].pop('raw_format')
    elif damage == 'shape': native['inputs'][0]['shape'] = [100]
    elif damage == 'name': native['outputs'][0]['tensor_name'] = 'guessed'
    path = probe / 'model-io.json'; path.write_text(json.dumps(native) + ('\n' if damage == 'changed-source' else ''))
    if damage != 'changed-source': binding['active_source_identities'] = [job._hash_source(path)]
    with pytest.raises(ValueError):
        job._resident_abi(binding, probe, tmp_path / 'resident-abi.tsv')
    assert not (tmp_path / 'resident-abi.tsv').exists()


def test_debug_mode_is_pinned_and_legacy_manifest_keeps_its_mode(sources):
    options, _, _ = sources
    result = job.run_board_serial_dflash(**options, dump=True)
    manifest = job.validate_prepared(result['prepared'])
    assert manifest['dump'] is True and result['retrieval_mode'] == 'FULL_DEBUG'
    path = Path(result['prepared']) / 'manifest.json'
    manifest.pop('dump')
    path.write_text(json.dumps(manifest))
    path.with_suffix('.sha256').write_text(job.sha256_file(path) + '\n')
    assert 'dump' not in job.validate_prepared(path.parent)
    manifest['dump'] = 0
    path.write_text(json.dumps(manifest))
    path.with_suffix('.sha256').write_text(job.sha256_file(path) + '\n')
    with pytest.raises(ValueError, match='dump mode'):
        job.validate_prepared(path.parent)


@pytest.mark.parametrize('flag,expected', [([], False), (['--dump'], True)])
def test_cli_pins_dump_mode_for_new_attempts(tmp_path, monkeypatch, capsys, flag, expected):
    script = Path(__file__).parents[1] / 'targets/cpro/scripts/run_board_serial_dflash.py'
    spec = importlib.util.spec_from_file_location('board_serial_dump_cli_test', script)
    cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    calls = []
    def run(**options):
        calls.append(options)
        return dict(status='BOARD_SERIAL_DFLASH_PLANNED', dump=options['dump'],
            generated_token_ids=[6, 7], acceptance={'accepted': 1, 'considered': 2, 'rate': 0.5},
            phase_latency={'prompt_ingest': {'calls': 8, 'total_ms': 80, 'mean_ms': 10}},
            supervisor_timing={'total_wall_ms': 100}, internal_payload={'not_a_summary': True})
    monkeypatch.setattr(job, 'run_board_serial_dflash', run)
    argv = ['--output', str(tmp_path / 'new'), '--resident']
    for name in ('decode-job', 'runtime-probe', 'state-lineage', 'source-torchair-pb2',
                 'draft-bundle', 'weight-table', 'scale-table', 'prompt-input-ids'):
        argv += ['--' + name, str(tmp_path / name)]
    with pytest.raises(SystemExit):
        cli.main([value for value in argv if value != '--resident'] + flag)
    assert not calls
    assert 'all-V1' in capsys.readouterr().err
    assert cli.main(argv + flag) == 0
    assert len(calls) == 1 and calls[0]['dump'] is expected
    summary = json.loads(capsys.readouterr().out)
    assert summary['generated_token_ids'] == [6, 7] and summary['acceptance']['rate'] == 0.5
    assert summary['phase_latency']['prompt_ingest']['mean_ms'] == 10
    assert summary['supervisor_timing']['total_wall_ms'] == 100
    assert 'internal_payload' not in summary
