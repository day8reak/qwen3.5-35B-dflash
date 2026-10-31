"""Real tiny six-layer recipes and input views, with the OMG process mocked.

No test result constitutes a model compilation or device execution observation.
"""
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest

from qwen35_moe_dflash import draft_capacity_export as cap
from qwen35_moe_dflash import incremental_draft_inputs as inputs
from qwen35_moe_dflash import incremental_draft_compile as job
from qwen35_moe_dflash import existing_air as compiler
from qwen35_moe_dflash.hashing import sha256_file
from test_incremental_draft_export import prepared
from test_draft_capacity_export import snapshot, tools
from test_draft_standalone import fake_ge


@pytest.fixture
def exported(prepared, tmp_path, monkeypatch):
    source, plan = prepared
    tools(monkeypatch)
    root = tmp_path / 'incremental-air'
    result = cap._export_source(source, plan, root, fake_ge(tmp_path), sequence_length=16,
        residual_policy='cpu-fp32-add', _incremental_options={'cache_capacity': 192, 'max_draft_tokens': 7})
    monkeypatch.setattr(inputs, 'configurations', lambda: (None, plan['draft_config']))
    guards = []
    _, identity = cap._metadata(source / 'prepared.json', guards)
    result.update(status='DRAFT_INCREMENTAL_AIR_EXPORTED', artifact_kind='qwen35-incremental-draft-air',
        source_prepared=identity, source_bindings=guards, draft_config=plan['draft_config'])
    (root / 'manifest.json').write_text(json.dumps(result))
    return source, root, result


def _ddk(tmp_path):
    ddk = tmp_path / 'selected-ddk'
    omg = ddk / 'tools/tools_omg/master/omg'
    omg.parent.mkdir(parents=True)
    omg.write_bytes(b'\x7fELF HOST FIXTURE NOT EXECUTED')
    omg.chmod(0o755)
    return ddk, omg


def _mock_omg(monkeypatch, *, tamper=None):
    calls = []
    def invoke(argv, **kwargs):
        if '--help' in argv:
            kwargs['stdout'].write('OMG usage: --use_origin_format --weight_data_type --save_weights_as_external_data --limit_partition_size\n')
            return subprocess.CompletedProcess(argv, 0)
        calls.append(argv)
        cwd = kwargs['cwd']
        aliases = sorted((cwd / 'weights').iterdir())
        assert len(aliases) == 71 and all(p.is_symlink() for p in aliases)
        assert (cwd / 'draft.air').is_file() and not (cwd / 'draft.air').is_symlink()
        assert '--weight_data_type=FP16' in argv and '--save_weights_as_external_data=true' in argv
        assert kwargs['env']['SOC_VERSION'] == 'kirinx90'
        if tamper == 'source':
            p = aliases[0].resolve(); data = p.read_bytes(); p.write_bytes(bytes([data[0] ^ 1]) + data[1:])
        elif tamper == 'alias':
            p = aliases[0]; p.unlink(); p.symlink_to(aliases[1].resolve())
        elif tamper == 'air':
            (cwd / 'draft.air').chmod(0o644)
            (cwd / 'draft.air').write_bytes(b'damaged air')
        elif tamper == 'fail':
            raise subprocess.CalledProcessError(7, argv)
        (cwd / 'model.omc').write_bytes(b'HOST MOCK OMC NOT DEPLOYABLE')
        (cwd / 'SubGraph_0.weight').write_bytes(b'HOST MOCK COMPILED WEIGHT')
        kwargs['stdout'].write('mock compiler complete\n')
        return subprocess.CompletedProcess(argv, 0)
    monkeypatch.setattr(subprocess, 'run', invoke)
    return calls


def _compile(exported, tmp_path, monkeypatch, tamper=None):
    _, export, _ = exported
    ddk, _ = _ddk(tmp_path)
    calls = _mock_omg(monkeypatch, tamper=tamper)
    out = tmp_path / 'compilation'
    result = compiler.compile_existing_air(export / 'draft.air', out, ddk, ddk_id='HOST_FIXTURE',
        incremental_draft_export=export, use_origin_format=True, weight_data_type='FP16',
        limit_partition_size=False, save_weights_as_external_data=True)
    return out, result, calls


def test_exact_original_weight_view_no_copy_hardlink_or_source_mutation(exported, tmp_path, monkeypatch):
    source, export, _ = exported
    before = snapshot(source), snapshot(export)
    copied = []
    original = compiler._snapshot
    def snapshot_air(src, dst):
        assert src == export / 'draft.air'
        copied.append(src)
        return original(src, dst)
    monkeypatch.setattr(compiler, '_snapshot', snapshot_air)
    out, result, calls = _compile(exported, tmp_path, monkeypatch)
    assert result['status'] == 'COMPILED', result
    assert result['incremental_input_view']['status'] == 'PASSED'
    assert result['incremental_input_view']['weight_file_aliases_created'] == 71
    assert result['incremental_input_view']['weight_files_copied'] == 0
    assert len(copied) == len(calls) == 1
    assert result['skipped_nonregular_artifacts'] == []
    assert [r['path'] for r in result['external_data_candidates']] == ['compile-work/generated/SubGraph_0.weight']
    assert (snapshot(source), snapshot(export)) == before
    assert result['device_execution'] == 'NOT_RUN'


@pytest.mark.parametrize('tamper', ['source', 'alias', 'air', 'fail'])
def test_compiler_failure_or_input_mutation_never_passes(exported, tmp_path, monkeypatch, tamper):
    out, result, calls = _compile(exported, tmp_path, monkeypatch, tamper)
    assert result['status'] == 'FAILED' and len(calls) == 1
    assert (out / 'failure.json').is_file()
    if tamper != 'fail':
        assert result['incremental_input_view']['status'] == 'FAILED'


@pytest.mark.parametrize('damage', ['weight', 'metadata', 'symlink', 'old-s64', 'missing-weight', 'gear'])
def test_rejects_stale_export_before_compiler(exported, tmp_path, monkeypatch, damage):
    source, export, manifest = exported
    weight = next((source / 'weights').iterdir())
    if damage == 'weight':
        data = weight.read_bytes(); weight.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    elif damage == 'metadata':
        p = source / 'prepared.json'; p.write_text(p.read_text() + ' ')
    elif damage == 'symlink':
        moved = weight.with_suffix('.saved'); weight.rename(moved); weight.symlink_to(moved.name)
    elif damage == 'missing-weight':
        weight.unlink()
    else:
        if damage == 'old-s64': manifest['artifact_kind'] = 'qwen35-cpro-draft-air'
        else: manifest['max_draft_tokens'] = 15
        (export / 'manifest.json').write_text(json.dumps(manifest))
    out, result, calls = _compile(exported, tmp_path, monkeypatch)
    assert result['status'] == 'FAILED' and calls == []
    assert result['model_compilation'] == 'NOT_RUN'


@pytest.fixture
def whole(exported, tmp_path, monkeypatch):
    source, export, manifest = exported
    ddk, omg = _ddk(tmp_path)
    schema = omg.parent / 'lib64/libai_fmk_graph_proto_host.so'
    schema.parent.mkdir(); schema.write_bytes(b'HOST SCHEMA MOCK')
    previous = tmp_path / 'old/residual-fix/compilation/report.json'
    previous.parent.mkdir(parents=True)
    previous.write_text(json.dumps({'status': 'COMPILED', 'model_compilation': 'PASSED',
        'ddk_root': str(ddk), 'ddk_id': 'HOST_FIXTURE', 'omg_path': str(omg), 'omg_sha256': sha256_file(omg)}))
    bundle = previous.parent.parent / 'bundle/manifest.json'
    bundle.parent.mkdir(); bundle.write_text(json.dumps({'compiler_report_sha256': sha256_file(previous)}))
    _, bundle_id = cap._metadata(bundle, [])
    manifest['source_bundle_manifest'] = bundle_id
    (export / 'manifest.json').write_text(json.dumps(manifest))
    _mock_omg(monkeypatch)
    monkeypatch.setattr(job, 'verify_compiled_precision', lambda *a: {'status': 'PASSED'})
    monkeypatch.setattr(job, 'inspect_incremental_abi', lambda *a: dict(status='PASSED', inputs=manifest['inputs'], outputs=manifest['outputs']))
    packs = []
    def pack(src, dst, schema, **kwargs):
        packs.append(kwargs)
        assert len(kwargs['sidecars']) == 1
        dst.write_bytes(b'HOST MOCK PACKED MODEL NOT DEPLOYABLE')
        return {'status': 'EMBEDDED', 'output': {'bytes': dst.stat().st_size, 'sha256': sha256_file(dst)}}
    monkeypatch.setattr(job, 'embed_external_weights', pack)
    return export, packs


def test_whole_host_orchestration_pins_old_compiler_and_new_model(whole, tmp_path):
    export, packs = whole
    out = tmp_path / 'new-compiled'
    result = job.compile_incremental_draft(export, out)
    assert result['status'] == 'DRAFT_INCREMENTAL_OMC_COMPILED_ABI_BOUND', result
    assert len(packs) == 1 and len(result['inputs']) == 17 and len(result['outputs']) == 14
    assert result['device_execution'] == 'NOT_RUN' and result['full_loop_smoke'] == 'NOT_RUN_NEW_INCREMENTAL_MODEL'
    assert result['model']['path'] == 'model/model.omc'
    with pytest.raises(FileExistsError):
        job.compile_incremental_draft(export, out)


@pytest.mark.parametrize('failed_gate', ['precision', 'abi'])
def test_failed_precision_or_abi_retains_both_checks_but_never_packs(whole, tmp_path, monkeypatch, failed_gate):
    export, packs = whole
    name = 'verify_compiled_precision' if failed_gate == 'precision' else 'inspect_incremental_abi'
    monkeypatch.setattr(job, name, lambda *a: {'status': 'FAILED', 'error': 'fixture rejection'})
    out = tmp_path / 'failed-compiled'
    result = job.compile_incremental_draft(export, out)
    assert result['status'] == 'FAILED' and result['model_compilation'] == 'PASSED'
    assert packs == []
    assert (out / 'compiled-abi.json').is_file() and (out / 'compiled-precision.json').is_file()


def test_changed_selected_compiler_rejected_before_help_or_compile(whole, tmp_path, monkeypatch):
    export, packs = whole
    omg = tmp_path / 'selected-ddk/tools/tools_omg/master/omg'
    omg.write_bytes(b'\x7fELF DIFFERENT COMPILER')
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: pytest.fail('must not run changed compiler'))
    result = job.compile_incremental_draft(export, tmp_path / 'rejected')
    assert result['status'] == 'FAILED' and result['model_compilation'] == 'NOT_RUN'
    assert 'OMG changed' in result['error'] and packs == []


def test_insufficient_output_budget_stops_before_compiler(whole, tmp_path, monkeypatch):
    from types import SimpleNamespace
    export, packs = whole
    monkeypatch.setattr(job.shutil, 'disk_usage', lambda p: SimpleNamespace(free=1))
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: pytest.fail('must not run without budget'))
    result = job.compile_incremental_draft(export, tmp_path / 'no-space')
    assert result['status'] == 'FAILED' and result['model_compilation'] == 'NOT_RUN'
    assert 'free bytes' in result['error'] and packs == []


def test_no_output_created_under_original_preparation(exported):
    source, export, _ = exported
    before = snapshot(source)
    with pytest.raises(ValueError, match='original source'):
        job.compile_incremental_draft(export, source / 'new-compile')
    assert not (source / 'new-compile').exists() and snapshot(source) == before
