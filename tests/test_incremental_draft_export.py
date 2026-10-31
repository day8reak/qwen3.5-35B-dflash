"""Cached AIR recipe and readonly source tests; subprocesses are host fixtures."""
from copy import deepcopy
import json

import pytest

from qwen35_moe_dflash import draft_capacity_export as cap
from qwen35_moe_dflash import draft_standalone as ds
from qwen35_moe_dflash import incremental_draft_export as inc
from qwen35_moe_dflash.ge_export import configurations
from qwen35_moe_dflash.ge_recipe_read import read_recipe
from test_draft_capacity_export import snapshot, tools, bound_job
from test_draft_standalone import fake_ge


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setenv('AI_RUN_DIR', str(tmp_path))
    target, draft = deepcopy(configurations(tiny=True))
    draft.update(num_hidden_layers=6, layer_types=['sliding_attention'] * 5 + ['full_attention'],
        max_position_embeddings=512, num_target_layers=8,
        dflash_config={'target_layer_ids': list(range(8)), 'block_size': 16, 'mask_token_id': 63})
    configs = (target, draft)
    monkeypatch.setattr(ds, 'configurations', lambda **kwargs: configs)
    monkeypatch.setattr(cap, 'configurations', lambda **kwargs: configs)
    root = tmp_path / 'prepared'
    return root, ds.prepare_draft(root, sequence_length=64, tiny=True)


@pytest.mark.parametrize('width', [15, 7, 3, 1])
def test_real_recipe_retains_all12_kv_outputs_static_block_gear_and_original_weights(prepared, tmp_path, monkeypatch, width):
    root, plan = prepared
    before = snapshot(root)
    calls = tools(monkeypatch)
    out = tmp_path / 'incremental'
    result = cap._export_source(root, plan, out, fake_ge(tmp_path), sequence_length=16,
        residual_policy='cpu-fp32-add', _incremental_options={'cache_capacity': 192, 'max_draft_tokens': width})
    assert result['status'] == 'HOST_FIXTURE_INCREMENTAL_AIR_EXPORTED'
    assert result['context_rows'] == 16 and result['max_draft_tokens'] == width
    assert 'sequence_length' not in result and 'direct_spec' not in result
    assert len(result['inputs']) == 17 and len(result['outputs']) == 14
    assert result['outputs'][0]['shape'] == [1, width]
    assert result['outputs'][1]['shape'] == [1, width, plan['draft_config']['hidden_size']]
    assert [r['name'] for r in result['outputs'][2:]] == [
        f'present_{kind}_{layer}' for layer in range(6) for kind in ('k', 'v')]
    assert result['incremental_abi']['block_rows'] == width + 1
    assert result['weight_files_copied'] == result['weight_links_created'] == 0
    assert before == snapshot(root)
    assert calls[1][1]['folder'] == root
    recipe = read_recipe(out / 'draft/recipe/graph.recipe')
    assert len(recipe['inputs']) == 17 and len(recipe['outputs']) == 14
    weights = [r for r in recipe['records'] if r['type'] == 'FileConstant']
    assert len(weights) == 71
    assert all(r['attrs']['file_path']['value'].startswith('weights/') for r in weights)
    assert len(json.loads((out / 'draft/residual-lowering.json').read_text())['replaced']) == 12
    assert result['model_compilation'] == result['device_execution'] == 'NOT_RUN'


def test_public_incremental_export_pins_successful_source_and_selected_gear(bound_job, tmp_path, monkeypatch):
    job, ge, _ = bound_job
    before = snapshot(job)
    captured = []
    def export(prepared, plan, output, sdk, **options):
        captured.append(options)
        output.mkdir()
        return {'source_weight_identities': [], 'status': 'DRAFT_INCREMENTAL_AIR_EXPORTED'}
    monkeypatch.setattr(inc, '_export_source', export)
    result = inc.export_incremental_draft(job, tmp_path / 'out', max_draft_tokens=7)
    assert captured[0]['sequence_length'] == 16
    assert captured[0]['_incremental_options'] == {'cache_capacity': 192, 'max_draft_tokens': 7}
    assert result['artifact_kind'] == 'qwen35-incremental-draft-air'
    assert before == snapshot(job) and result['ge_root'] == str(ge)


@pytest.mark.parametrize('width', [0, 2, 8, 16, True, '7'])
def test_bad_candidate_width_is_rejected_before_reading_or_creating_files(tmp_path, width):
    with pytest.raises(ValueError, match='max_draft_tokens'):
        inc.export_incremental_draft(tmp_path / 'missing', tmp_path / 'out', max_draft_tokens=width)
    assert not (tmp_path / 'out').exists()


def test_failure_receipt_is_owned_and_partial_air_is_never_promoted(bound_job, tmp_path, monkeypatch):
    job, _, _ = bound_job
    out = tmp_path / 'failed'
    def fail(prepared, plan, output, ge, **options):
        output.mkdir()
        cap.save_json(output / 'export-request.json', {'invocation_token': options['_invocation_token']})
        raise RuntimeError('selected GE rejected Where')
    monkeypatch.setattr(inc, '_export_source', fail)
    with pytest.raises(RuntimeError):
        inc.export_incremental_draft(job, out)
    assert json.loads((out / 'manifest.json').read_text())['status'] == 'FAILED'
    before = snapshot(out)
    with pytest.raises(FileExistsError):
        inc.export_incremental_draft(job, out)
    assert snapshot(out) == before


@pytest.mark.parametrize('width', [15, 7, 3, 1])
def test_full35b_geometry_keeps_fc_compatibility_and_six_residual_pairs(tmp_path, width):
    from qwen35_moe_dflash.config import Qwen35DFlashConfig
    from qwen35_moe_dflash.ge_incremental_draft import build_incremental_draft
    from qwen35_moe_dflash.ge_cpp import emit_cpp
    graph = build_incremental_draft(Qwen35DFlashConfig.from_dict(configurations()[1]),
        context_rows=16, cache_capacity=192, max_draft_tokens=width)
    paths = {name: str(tmp_path / 'weights' / f'w{index}.bin') for index, name in enumerate(graph.weights)}
    projection, residual = {}, {}
    # Emit construction recipes only. No real weights, SDK or compiler runs.
    emit_cpp(graph, paths, external_weights=True, recipe_dir=tmp_path / 'recipe',
        fp16_reductions='matmul', fp16_row_mul='compact',
        file_constant_policy='cpro-relative-int', file_constant_root=tmp_path,
        draft_fc_policy='conv2d', draft_fc_report=projection,
        draft_residual_policy='cpu-fp32-add', draft_residual_report=residual)
    assert len(projection['replaced']) == 1 and len(residual['replaced']) == 12
    assert all(row['shape'] == [1, width + 1, 2048] for row in residual['replaced'])
    assert graph.outputs[0].shape == (1, width)


def linked_sdk(ge):
    """Typical toolkit layout: include/lib64 aliases plus versioned .so."""
    arch = ge / 'arch'; arch.mkdir()
    for name in ('include', 'lib64'):
        (ge / name).rename(arch / name)
        (ge / name).symlink_to('arch/' + name, target_is_directory=True)
    library = arch / 'lib64/libgraph.so'
    library.rename(library.with_name('libgraph.so.1'))
    library.symlink_to('libgraph.so.1')
    base = arch / 'lib64/libgraph_base.so.1'; base.write_bytes(b'fixture graph base')
    base.with_name('libgraph_base.so').symlink_to(base.name)
    header = arch / 'include/graph/graph.h'
    header.rename(header.with_name('graph-v1.h'))
    header.symlink_to('graph-v1.h')
    return arch


def test_incremental_recipe_build_uses_resolved_sdk_directories_and_accepts_versioned_library(prepared, tmp_path, monkeypatch):
    root, plan = prepared
    before = snapshot(root)
    ge = fake_ge(tmp_path); arch = linked_sdk(ge)
    calls = tools(monkeypatch)
    result = cap._export_source(root, plan, tmp_path / 'out', ge, sequence_length=16,
        residual_policy='cpu-fp32-add', _incremental_options={'cache_capacity': 192, 'max_draft_tokens': 3})
    assert result['status'] == 'HOST_FIXTURE_INCREMENTAL_AIR_EXPORTED'
    argv = calls[0][0]
    assert argv[argv.index('-I') + 1] == str(arch / 'include')
    assert argv[argv.index('-L') + 1] == str(arch / 'lib64')
    assert '-lgraph_base' in argv
    assert calls[1][1]['env']['LD_LIBRARY_PATH'].split(':')[0] == str(arch / 'lib64')
    assert result['sdk_path_bindings'][str(ge / 'lib64/libgraph.so')]['resolved'] == str(arch / 'lib64/libgraph.so.1')
    assert len(result['outputs']) == 14 and before == snapshot(root)


def test_public_export_still_matches_original_library_hash_through_sdk_links(bound_job, tmp_path, monkeypatch):
    job, ge, _ = bound_job
    linked_sdk(ge)
    def export(prepared, plan, output, sdk, **options):
        output.mkdir()
        return {'source_weight_identities': [], 'status': 'HOST_FIXTURE_INCREMENTAL_AIR_EXPORTED'}
    monkeypatch.setattr(inc, '_export_source', export)
    result = inc.export_incremental_draft(job, tmp_path / 'out')
    assert result['libgraph_sha256'] == cap.sha256_file(ge / 'lib64/libgraph.so')
    assert (ge / 'lib64').is_symlink()


@pytest.mark.parametrize('fault', ['changed_library', 'dangling_library'])
def test_sdk_link_support_does_not_bypass_original_library_content_pin(bound_job, tmp_path, monkeypatch, fault):
    job, ge, _ = bound_job
    arch = linked_sdk(ge)
    payload = arch / 'lib64/libgraph.so.1'
    if fault == 'changed_library':
        payload.write_bytes(b'other SDK')
    else:
        payload.unlink()
    monkeypatch.setattr(inc, '_export_source', lambda *a, **kw: pytest.fail('invalid SDK must fail before build'))
    with pytest.raises((ValueError, FileNotFoundError)):
        inc.export_incremental_draft(job, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_sdk_link_retarget_during_export_is_rejected_even_with_identical_bytes(prepared, tmp_path, monkeypatch):
    root, plan = prepared
    ge = fake_ge(tmp_path); arch = linked_sdk(ge)
    link = arch / 'lib64/libgraph.so'
    other = arch / 'lib64/libgraph.so.2'
    other.write_bytes(link.read_bytes())
    def retarget():
        link.unlink()
        link.symlink_to(other.name)
    tools(monkeypatch, during_export=retarget)
    with pytest.raises(ValueError, match='SDK path changed'):
        cap._export_source(root, plan, tmp_path / 'out', ge, sequence_length=16,
            residual_policy='cpu-fp32-add', _incremental_options={'cache_capacity': 192, 'max_draft_tokens': 3})


def test_sdk_link_support_does_not_relax_prepared_weight_path_checks(prepared, tmp_path, monkeypatch):
    root, plan = prepared
    weight = root / next(iter(plan['weights'].values()))['path']
    payload = weight.with_suffix('.payload'); weight.rename(payload)
    weight.symlink_to(payload.name)
    monkeypatch.setattr(cap, '_logged', lambda *a, **kw: pytest.fail('must reject weight link before build'))
    with pytest.raises(ValueError, match='symlink is unsupported'):
        cap._export_source(root, plan, tmp_path / 'out', fake_ge(tmp_path), sequence_length=16,
            residual_policy='cpu-fp32-add', _incremental_options={'cache_capacity': 192, 'max_draft_tokens': 3})
