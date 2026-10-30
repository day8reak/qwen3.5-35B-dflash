"""Real graph/recipe generation with a mocked host tool; never compile a model."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from qwen35_moe_dflash import draft_capacity_export as cap
from qwen35_moe_dflash import draft_standalone as ds
from qwen35_moe_dflash.ge_export import configurations
from qwen35_moe_dflash.ge_recipe_read import read_recipe
from test_draft_standalone import fake_ge


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setenv('AI_RUN_DIR', str(tmp_path))
    target, draft = deepcopy(configurations(tiny=True))
    draft.update(num_hidden_layers=6, layer_types=['sliding_attention'] * 5 + ['full_attention'], max_position_embeddings=512)
    configs = (target, draft)
    monkeypatch.setattr(ds, 'configurations', lambda **kwargs: configs)
    monkeypatch.setattr(cap, 'configurations', lambda **kwargs: configs)
    root = tmp_path / 'prepared'
    plan = ds.prepare_draft(root, sequence_length=64, tiny=True)
    assert plan['draft_layers'] == 6
    return root, plan


def snapshot(root):
    return {str(path.relative_to(root)): (path.read_bytes(), cap._current(path)[1]) for path in root.rglob('*') if path.is_file()}


def tools(monkeypatch, *, during_export=None):
    calls = []
    def logged(command, **options):
        calls.append((list(command), options))
        if '-o' in command:
            Path(command[command.index('-o') + 1]).write_bytes(b'HOST_TEST_EXPORTER_NOT_MODEL')
        else:
            if during_export is not None: during_export()
            Path(command[1]).write_bytes(b'HOST_TEST_EXTERNAL_AIR_NOT_DEVICE_EVIDENCE')
    monkeypatch.setattr(cap, '_logged', logged)
    return calls


def test_locked_full128_graph_has_all_six_layers_full_vocabulary_and_same71_weight_shapes():
    _, config, old = ds._graph(64)
    _, _, new = ds._graph(128)
    assert (config.num_hidden_layers, config.vocab_size, config.feature_size) == (6, 248320, 16384)
    assert len(new.weights) == 71 and {key: value.shape for key, value in old.weights.items()} == {key: value.shape for key, value in new.weights.items()}
    assert [value.shape for value in new.inputs] == [(1, 128, 16384), (1, 128), (1, 128)]
    assert [value.shape for value in new.outputs] == [(1, 15), (1, 15, 2048)]


def test_actual128_graph_recipe_preserves_six_layers_and_original_source_files_without_links_or_copies(prepared, tmp_path, monkeypatch):
    root, plan = prepared
    before = snapshot(root)
    calls = tools(monkeypatch)
    monkeypatch.setattr(cap.os, 'link', lambda *a, **k: pytest.fail('capacity export must not hardlink original weights'))
    monkeypatch.setattr(cap.os, 'symlink', lambda *a, **k: pytest.fail('capacity export must not create weight views'))
    monkeypatch.setattr(cap.shutil, 'copyfile', lambda *a, **k: pytest.fail('capacity export must not copy weights'))
    output = tmp_path / 'capacity'
    result = cap._export_source(root, plan, output, fake_ge(tmp_path), sequence_length=128, residual_policy='cpu-fp32-add')
    assert result['status'] == 'HOST_FIXTURE_CAPACITY_AIR_EXPORTED'
    assert result['sequence_length'] == 128 and result['original_sequence_length'] == 64
    assert result['draft_layers'] == 6 and result['weight_files_copied'] == result['weight_links_created'] == 0
    assert calls[1][1]['folder'] == root and calls[1][0][1] == str(output / 'draft.air')
    assert calls[0][1]['folder'] == output / 'draft'
    assert not (output / 'weights').exists() and not result['compilation_ready']
    assert before == snapshot(root), 'even nlink/ctime must remain unchanged'
    recipe = read_recipe(output / 'draft/recipe/graph.recipe')
    inputs = [recipe['records'][index]['outputs'][0] for index in recipe['inputs']]
    assert [row['shape'][1] for row in inputs] == [128, 128, 128]
    assert [row['name'] for row in result['outputs']] == ['draft_top1', 'draft_hidden']
    assert all(row['name'].startswith(('draft_top1', 'draft_hidden')) for row in result['outputs'])
    residual = json.loads((output / 'draft/residual-lowering.json').read_text())
    assert len(residual['replaced']) == 12
    constants = [row for row in recipe['records'] if row['type'] == 'FileConstant']
    assert len(constants) == len(plan['weights'])
    assert all(row['attrs']['file_path']['value'].startswith('weights/') for row in constants)
    assert result['model_compilation'] == result['device_execution'] == 'NOT_RUN'
    assert result['observed_new_file_bytes'] < 16 << 20


@pytest.mark.parametrize('fault', ['same_capacity', 'bool_capacity', 'wrong_residual', 'weight_content', 'weight_shape', 'incomplete_weight_set'])
def test_capacity_export_refuses_changed_or_incomplete_sources_before_any_tool_runs(prepared, tmp_path, monkeypatch, fault):
    root, plan = prepared
    capacity, policy = 128, 'cpu-fp32-add'
    if fault == 'same_capacity': capacity = 64
    elif fault == 'bool_capacity': capacity = True
    elif fault == 'wrong_residual': policy = 'native'
    elif fault == 'weight_content':
        row = next(iter(plan['weights'].values())); path = root / row['path']
        path.write_bytes(bytes([path.read_bytes()[0] ^ 1]) + path.read_bytes()[1:])
    elif fault == 'weight_shape': next(iter(plan['weights'].values()))['shape'] = [1]
    else: plan['weights'].pop(next(iter(plan['weights'])))
    monkeypatch.setattr(cap, '_logged', lambda *a, **k: pytest.fail('tool must not run with invalid preparation'))
    with pytest.raises(ValueError):
        cap._export_source(root, plan, tmp_path / 'capacity', fake_ge(tmp_path), sequence_length=capacity, residual_policy=policy)
    assert not (tmp_path / 'capacity').exists()


def test_original_weight_change_during_real_exporter_invocation_is_rejected(prepared, tmp_path, monkeypatch):
    root, plan = prepared
    path = root / next(iter(plan['weights'].values()))['path']
    tools(monkeypatch, during_export=lambda: path.chmod(0o600))
    with pytest.raises(ValueError, match='original prepared weight identity changed'):
        cap._export_source(root, plan, tmp_path / 'capacity', fake_ge(tmp_path), sequence_length=128, residual_policy='cpu-fp32-add')
    assert (tmp_path / 'capacity/draft/export-command.json').exists()


def test_small_output_budget_refuses_low_disk_before_building_exporter(prepared, tmp_path, monkeypatch):
    from types import SimpleNamespace
    root, plan = prepared
    monkeypatch.setattr(cap.shutil, 'disk_usage', lambda *a: SimpleNamespace(free=128 << 20))
    monkeypatch.setattr(cap, '_logged', lambda *a, **k: pytest.fail('tool cannot run when output budget is unavailable'))
    with pytest.raises(OSError, match='384 MiB'):
        cap._export_source(root, plan, tmp_path / 'capacity', fake_ge(tmp_path), sequence_length=128, residual_policy='cpu-fp32-add')


def test_source_root_and_existing_output_are_preserved_before_any_weight_rehash(prepared, tmp_path, monkeypatch):
    root, plan = prepared
    monkeypatch.setattr(cap, 'sha256_file', lambda *a: pytest.fail('invalid output should fail before source payload hashing'))
    with pytest.raises(ValueError, match='outside the original prepared'):
        cap._export_source(root, plan, root / 'new-capacity', tmp_path / 'ge', sequence_length=128, residual_policy='cpu-fp32-add')
    output = tmp_path / 'existing'; output.mkdir(); (output / 'retained.txt').write_text('do not change')
    with pytest.raises(FileExistsError):
        cap._export_source(root, plan, output, tmp_path / 'ge', sequence_length=128, residual_policy='cpu-fp32-add')
    assert (output / 'retained.txt').read_text() == 'do not change' and not (root / 'new-capacity').exists()


@pytest.mark.parametrize('values', [(128, 64, 65), (128, True, 32), (0, 64, 32), (128, 64, 0), (16385, 64, 32)])
def test_complete_prefix_capacity_budget_is_checked_before_reading_job(values, tmp_path):
    with pytest.raises(ValueError, match='complete prompt'):
        cap.export_draft_capacity(tmp_path / 'missing', tmp_path / 'out', sequence_length=values[0], prompt_length=values[1], max_new_tokens=values[2])


def test_public_command_rejects_tiny_or_unbound_lane_instead_of_masquerading_as_full_model(prepared, tmp_path, monkeypatch):
    root, plan = prepared
    job = tmp_path / 'job'; job.mkdir()
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True); raw = json.dumps(value).encode(); path.write_bytes(raw)
        return {'path': path.name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    root.rename(job / 'prepared')
    manifest = {'path': 'prepared.json', 'bytes': (job / 'prepared/prepared.json').stat().st_size,
        'sha256': hashlib.sha256((job / 'prepared/prepared.json').read_bytes()).hexdigest()}
    save(job / 'result.json', {'artifact_kind': 'qwen35-full-draft-job',
        'components': {'prepared': {'directory': str(job / 'prepared'), 'manifest': manifest}}})
    save(job / 'residual-fix/air/manifest.json', {'artifact_kind': 'qwen35-cpro-draft-air', 'status': 'EXPORTED'})
    save(job / 'residual-fix/bundle/manifest.json', {'artifact_kind': 'qwen35-standalone-draft-direct-bundle', 'status': 'PREPARED'})
    monkeypatch.setattr(cap, '_export_source', lambda *a, **k: pytest.fail('public full-model lane must reject host fixtures'))
    with pytest.raises(ValueError, match='complete preparation'):
        cap.export_draft_capacity(job, tmp_path / 'out')


@pytest.fixture
def bound_job(prepared, tmp_path):
    root, plan = prepared
    job = tmp_path / 'job'; job.mkdir(); root.rename(job / 'prepared')
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True); raw = json.dumps(value).encode(); path.write_bytes(raw)
        return {'path': path.name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    # This fixture tests only wrapper pins. The actual six-layer serializer
    # core above labels tiny payloads as host fixtures, and no model runs here.
    plan = dict(plan, tiny=False)
    prepared_id = save(job / 'prepared/prepared.json', plan)
    source = {'artifact_kind': 'qwen35-full-draft-job',
        'components': {'prepared': {'directory': str(job / 'prepared'), 'manifest': prepared_id}}}
    save(job / 'result.json', source)
    ge = fake_ge(tmp_path)
    from qwen35_moe_dflash.ge_residual import EXPORT_SUFFIX
    air = {'artifact_kind': 'qwen35-cpro-draft-air', 'status': 'EXPORTED', 'residual_policy': 'cpu-fp32-add',
        'sequence_length': 64, 'prepared_manifest_sha256': prepared_id['sha256'], 'ge_root': str(ge), 'cxx11_abi': 0,
        'libgraph_sha256': cap.sha256_file(ge / 'lib64/libgraph.so'),
        'export_contract': ds.CURRENT_DRAFT_EXPORT_CONTRACT + EXPORT_SUFFIX + '-direct-outputs-v1',
        'fp16_reductions': 'matmul', 'fp16_row_mul': 'compact'}
    air_id = save(job / 'residual-fix/air/manifest.json', air)
    bundle = {'artifact_kind': 'qwen35-standalone-draft-direct-bundle', 'status': 'PREPARED',
        'draft_config': plan['draft_config'], 'sequence_length': 64, 'export_manifest_sha256': air_id['sha256'],
        'weight_storage': 'embedded-stream'}
    save(job / 'residual-fix/bundle/manifest.json', bundle)
    return job, ge, save


def test_public_export_binds_old_sdk_and_manifest_and_returns_new_distinct_capacity_without_reading_old_omc(bound_job, tmp_path, monkeypatch):
    job, ge, _ = bound_job
    before = snapshot(job)
    calls = []
    def export(prepared_root, plan, output, selected_ge, **options):
        calls.append((prepared_root, selected_ge, options))
        output.mkdir()
        return {'status': 'DRAFT_CAPACITY_AIR_EXPORTED', 'source_weight_identities': [], 'sequence_length': 128,
            'original_sequence_length': 64, 'model_compilation': 'NOT_RUN', 'device_execution': 'NOT_RUN'}
    monkeypatch.setattr(cap, '_export_source', export)
    result = cap.export_draft_capacity(job, tmp_path / 'new-capacity')
    assert calls[0][0] == job / 'prepared' and calls[0][1] == ge
    assert calls[0][2]['sequence_length'] == 128 and calls[0][2]['residual_policy'] == 'cpu-fp32-add'
    assert result['sequence_length'] == 128 and result['original_sequence_length'] == 64
    assert result['source_prepared']['sha256'] == hashlib.sha256((job / 'prepared/prepared.json').read_bytes()).hexdigest()
    assert before == snapshot(job) and (tmp_path / 'new-capacity/manifest.json').is_file()
    assert not list(job.rglob('*.omc')), 'the old OMC is not a dependency of AIR capacity export'


@pytest.mark.parametrize('fault', ['sdk_bytes', 'prepared_pin', 'air_prepared_pin', 'bundle_export_pin', 'residual_policy', 'reduction_policy', 'export_contract'])
def test_public_export_refuses_unbound_or_changed_source_policy_before_calling_tools(bound_job, tmp_path, monkeypatch, fault):
    job, ge, save = bound_job
    if fault == 'sdk_bytes': (ge / 'lib64/libgraph.so').write_bytes(b'changed GE')
    elif fault == 'prepared_pin':
        source = json.loads((job / 'result.json').read_bytes()); source['components']['prepared']['manifest']['sha256'] = 'f' * 64
        save(job / 'result.json', source)
    elif fault == 'bundle_export_pin':
        bundle = json.loads((job / 'residual-fix/bundle/manifest.json').read_bytes()); bundle['export_manifest_sha256'] = 'f' * 64
        save(job / 'residual-fix/bundle/manifest.json', bundle)
    else:
        air = json.loads((job / 'residual-fix/air/manifest.json').read_bytes())
        key, value = {'air_prepared_pin': ('prepared_manifest_sha256', 'f' * 64), 'residual_policy': ('residual_policy', 'native'),
            'reduction_policy': ('fp16_reductions', 'native'), 'export_contract': ('export_contract', 'obsolete-builder')}[fault]
        air[key] = value
        air_id = save(job / 'residual-fix/air/manifest.json', air)
        bundle = json.loads((job / 'residual-fix/bundle/manifest.json').read_bytes()); bundle['export_manifest_sha256'] = air_id['sha256']
        save(job / 'residual-fix/bundle/manifest.json', bundle)
    monkeypatch.setattr(cap, '_export_source', lambda *a, **k: pytest.fail('invalid source identity must fail before tools'))
    with pytest.raises(ValueError):
        cap.export_draft_capacity(job, tmp_path / 'unused')
    assert not (tmp_path / 'unused').exists()


def test_public_final_guard_cannot_rebase_source_metadata_drift_after_export(bound_job, tmp_path, monkeypatch):
    job, _, _ = bound_job
    def export(*args, **kwargs):
        (job / 'prepared/prepared.json').chmod(0o600)
        return {'source_weight_identities': []}
    monkeypatch.setattr(cap, '_export_source', export)
    with pytest.raises(ValueError, match='metadata changed within'):
        cap.export_draft_capacity(job, tmp_path / 'unused')


def test_owned_failed_export_keeps_failed_manifest_and_never_claims_capacity_success(bound_job, tmp_path, monkeypatch):
    job, _, _ = bound_job
    output = tmp_path / 'failed'
    def failed(prepared, plan, destination, ge, **options):
        destination.mkdir()
        cap.save_json(destination / 'export-request.json', {'invocation_token': options['_invocation_token']})
        (destination / 'draft.air').write_bytes(b'partial AIR')
        raise RuntimeError('serializer failed before acceptance')
    monkeypatch.setattr(cap, '_export_source', failed)
    with pytest.raises(RuntimeError, match='serializer failed'):
        cap.export_draft_capacity(job, output)
    result = json.loads((output / 'manifest.json').read_bytes())
    assert result['status'] == 'FAILED' and result['compilation_ready'] is False
    assert result['source_preservation'] == 'NOT_CERTIFIED_AFTER_FAILURE' and (output / 'draft.air').exists()


def test_foreign_failed_output_is_not_mutated_even_if_its_source_capacity_matches(bound_job, tmp_path, monkeypatch):
    job, _, _ = bound_job
    output = tmp_path / 'other-attempt'; output.mkdir()
    cap.save_json(output / 'export-request.json', {'invocation_token': 'different-attempt'})
    before = snapshot(output)
    monkeypatch.setattr(cap, '_export_source', lambda *a, **k: (_ for _ in ()).throw(FileExistsError('already owned')))
    with pytest.raises(FileExistsError):
        cap.export_draft_capacity(job, output)
    assert before == snapshot(output) and not (output / 'manifest.json').exists()
