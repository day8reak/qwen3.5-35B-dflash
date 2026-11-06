"""Compact shared PyTorch Draft artifacts, independent of disposable tests.

Binding is imported once while compiler snapshots exist. Publication does not
promote this diagnostic ABI into the older token-ID resident runtime.
"""
from pathlib import Path
import os
import shutil

from qwen35_moe_dflash.common.artifacts import checked_file
from qwen35_moe_dflash.common.hashing import sha256_file
from qwen35_moe_dflash.execution.runtime_config import read_json
from qwen35_moe_dflash.toolchain.compiler_storage import _plain, require_idle, _tree_bytes
from .runtime import validate_abi
from .torch_export import AIR_KIND, OMC_KIND, _save, _air_locks, _completed_air

POLICY = 'pytorch-draft-shared-v1'


def load_shared(root, air, prepared, output):
    """Diagnostic binding from compact metadata; no DDK or compiler directory."""
    root, air, prepared, output = map(Path, (root, air, prepared, output))
    meta = read_json(root/'manifest.json')
    proof = read_json(root/'pytorch-abi.json')
    export, plan = read_json(air/'manifest.json'), read_json(prepared/'prepared.json')
    publication = meta.get('publication', {})
    if (publication.get('policy') != POLICY or meta.get('artifact_kind') != OMC_KIND
            or meta.get('status') != 'PYTORCH_DRAFT_OMC_COMPILED'
            or proof.get('status') != 'PASSED'
            or proof.get('artifact_kind') != 'qwen35-pytorch-draft-diagnostic-abi'
            or meta['inputs'] != proof['inputs'] or meta['outputs'] != proof['outputs']
            or export.get('artifact_kind') != AIR_KIND
            or export.get('status') != 'PYTORCH_DRAFT_AIR_EXPORTED'):
        raise ValueError('invalid shared Draft ABI/export binding')
    for path, expected in ((air/'export.air', proof['source_air_sha256']),
                           (prepared/'prepared.json', proof['prepared_sha256']),
                           (root/'compiler.air', proof['compiler_air_sha256'])):
        if sha256_file(path) != expected:
            raise ValueError('shared Draft diagnostic source changed: '+str(path))
    if meta['model']['sha256'] != proof['packed_model_sha256']:
        raise ValueError('shared Draft model/ABI identities differ')
    checked_file(root, meta['model'])
    if plan.get('draft_config') != meta['draft_config']:
        raise ValueError('prepared Draft configuration differs')
    validate_abi(meta)
    output.mkdir(parents=True, exist_ok=True)
    if output.resolve() != root.resolve():
        _save(output/'pytorch-abi.json', proof)
    return meta


def _package(compiled, air, prepared, ddk_root, output):
    from .torch_abi import bind_candidate
    if output.exists() or output.is_symlink():
        _plain(output, directory=True)
        old = read_json(output/'manifest.json')
        if old.get('publication', {}).get('policy') != POLICY:
            raise ValueError('unknown package retained: '+str(output))
        _tree_bytes(output)
        shutil.rmtree(output)
    output.mkdir()
    _save(output/'manifest.json', dict(artifact_kind=OMC_KIND,
        status='PREPARING_PUBLIC_DRAFT', publication=dict(policy=POLICY)))
    meta = bind_candidate(compiled, air, prepared, ddk_root, output)
    receipt = read_json(compiled/'compilation/report.json', 8 << 20)
    snapshots = {r['source_relative']: r for r in receipt['input_snapshots']}
    # Also retain a rebuildable AIR, not merely same-sized external files.
    # This one-time migration check does not run in ordinary inference.
    for relative, record in snapshots.items():
        if relative.startswith('weight/'):
            checked_file(air, dict(record, path=relative))
    compiler = checked_file(compiled/'compilation', receipt.get('air_preparation', {}).get(
        'compiler_air', snapshots['export.air']))
    # Same filesystem links keep the 1.8 GB model single-copy until old tests
    # are deleted. No weight sidecars or compilation tree enter the bundle.
    os.link(compiled/'model.omc', output/'model.omc')
    shutil.copyfile(compiler, output/'compiler.air')
    keep = ('artifact_kind', 'status', 'stage', 'ddk_id', 'context_rows', 'cache_capacity',
        'max_draft_tokens', 'inputs', 'outputs', 'rms', 'gamma', 'attention', 'optimization',
        'projection', 'host_embedding', 'host_embedding_binding', 'top1', 'top1_placement',
        'feature_pack', 'precision_repair', 'model', 'draft_config', 'diagnostic_frontend',
        'weight_storage', 'compiler_options', 'graphop_counts')
    compact = {k: meta[k] for k in keep if k in meta}
    compact.update(omc='model.omc', model_compilation='PASSED', compiled_abi='BOUND_DIAGNOSTIC',
        compiled_precision='NOT_VALIDATED', numerical_validation='NOT_RUN',
        device_execution='NOT_RUN', runtime_integration='PACKED_DIAGNOSTIC_ONLY',
        production_promotion='NOT_PERFORMED', publication=dict(policy=POLICY,
            abi='pytorch-abi.json', compiler_air='compiler.air',
            air_weight_verification='COMPILER_INPUT_SNAPSHOT_HASHES_CHECKED_AT_PUBLICATION',
            recipe=dict(frontend='pytorch', optimization='packed-spatial-host-embedding',
                top1='stable-arg-max', rope_fp32_table=True, cpu_feature_pack=True,
                offline_weights=None)))
    if meta.get('offline_weights'):
        raise ValueError('shared baseline publication excludes offline-weight experiments')
    if (not meta.get('feature_pack') or not meta.get('precision_repair')
            or meta.get('optimization', {}).get('policy') != 'packed-spatial-host-embedding'
            or meta.get('top1', {}).get('policy') != 'stable-arg-max'):
        raise ValueError('shared publication requires the packed/RoPE/StableArgMax baseline')
    _save(output/'manifest.json', compact)
    load_shared(output, air, prepared, output)
    return compact


def _destination(path, kinds):
    if not path.exists() and not path.is_symlink():
        return
    _plain(path, directory=True)
    meta = read_json(path/'manifest.json')
    if meta.get('artifact_kind') not in kinds or meta.get('status') not in {
            'PYTORCH_DRAFT_AIR_EXPORTED', 'DRAFT_INCREMENTAL_AIR_EXPORTED',
            'PYTORCH_DRAFT_OMC_COMPILED', 'DRAFT_INCREMENTAL_OMC_COMPILED_ABI_BOUND', 'FAILED'}:
        raise ValueError('active or unknown public artifact retained: '+str(path))


def _install(source, destination, previous):
    """Idempotent rename step; caller's journal owns the previous slot."""
    if not source.exists():
        if not destination.is_dir():
            raise ValueError('publication source and destination are both missing')
        return
    if destination.exists():
        if previous.exists():
            raise ValueError('both destination and previous exist before install')
        destination.rename(previous)
    source.rename(destination)


def _retain_runtime_previous(role, config):
    """Keep an old configured resident model and rebase its explicit references."""
    if not config.is_file():
        return []
    from qwen35_moe_dflash.execution.run_layout import _refs
    data = read_json(config)
    retained = set()
    for keys, value in list(_refs(data)):
        target = (config.parent/value).resolve()
        for name in ('air', 'omc'):
            old, previous = role/name, role/('.'+name+'-previous')
            if target == previous or previous in target.parents:
                retained.add(previous)
            elif previous.exists() and (target == old or old in target.parents):
                new = previous/target.relative_to(old)
                owner = data
                for key in keys[:-1]:
                    owner = owner[key]
                owner[keys[-1]] = os.path.relpath(new, config.parent)
                retained.add(previous)
    _save(config, data)
    return sorted(str(p) for p in retained)


def archive_baseline(root, compiled, air, prepared, comparison, ddk_root, *, apply=False):
    """Archive a completed measured baseline without compilation or execution."""
    root, compiled, air, prepared = map(lambda p: Path(p).absolute(), (root, compiled, air, prepared))
    role = root/'draft'
    _plain(root, directory=True)
    _plain(role, directory=True)
    if prepared == role/'tests' or role/'tests' in prepared.parents:
        raise ValueError('original prepared weights must remain outside disposable tests')
    journal, work = role/'.draft-archive.json', role/'.archive-build'
    archive = role/'archive.json'
    # Rerunning after cleanup must not need deleted experiment inputs.
    if not journal.exists() and archive.is_file():
        saved = read_json(archive)
        model = load_shared(role/'omc', role/'air', prepared, role/'omc')
        if model['model']['sha256'] != saved['model']['sha256']:
            raise ValueError('archive and current model differ')
        return dict(saved, status='ALREADY_ARCHIVED', output=str(role))
    if not apply:
        _plain(compiled, directory=True)
        _plain(air, directory=True)
        return dict(status='PLANNED', retained_model=str(compiled/'model.omc'),
            air_destination=str(role/'air'), omc_destination=str(role/'omc'),
            summary=str(archive), model_compilation='NOT_RUN', device_execution='NOT_RUN')
    require_idle(role)
    with _air_locks(role/'air', role/'omc', work, compiled, air):
        if not journal.exists():
            from .feature_pack_compare import compare_result
            from .test_cleanup import _verify_source
            _verify_source(root, role/'tests', prepared)
            report = compare_result(comparison, offline_weights=True)
            model = report['baseline_model']
            checked_file(compiled, dict(model, path='model.omc'))
            _completed_air(air)
            for name, kinds in (('air', {AIR_KIND, 'qwen35-incremental-draft-air'}),
                                ('omc', {OMC_KIND, 'qwen35-incremental-draft-compiled'})):
                _destination(role/name, kinds)
                if (role/('.'+name+'-previous')).exists():
                    raise ValueError('previous public artifact retained; resolve it before a new archive')
            if work.exists():
                partial = read_json(work/'manifest.json')
                if (partial.get('publication', {}).get('policy') != POLICY
                        or partial.get('status') not in {'PREPARING_PUBLIC_DRAFT', 'PYTORCH_DRAFT_OMC_COMPILED'}):
                    raise ValueError('unknown archive preparation retained: '+str(work))
                _tree_bytes(work)
                shutil.rmtree(work)
            _package(compiled, air, prepared, ddk_root, work)
            # Resolve and compare before any rename. Never select the slower
            # candidate just because it was the most recent compilation.
            counts = report['baseline_case_counts']
            baseline_numerical = 'PASSED' if all(v['passed'] == v['total'] for v in counts.values()) else 'FAILED'
            saved = dict(artifact_kind='qwen35-draft-baseline-archive', status='ARCHIVED',
                model=dict(model, path='omc/model.omc'), comparison=report,
                numerical_validation=baseline_numerical,
                runtime_integration='PACKED_DIAGNOSTIC_ONLY',
                next='TARGET_ACCURACY', model_compilation='NOT_RUN', device_execution='NOT_RUN')
            _save(journal, dict(policy=POLICY, air_source=str(air), saved=saved))
        state = read_json(journal)
        if state.get('policy') != POLICY:
            raise ValueError('unknown publication journal')
        air = Path(state['air_source'])
        if air != role/'air':
            _install(air, role/'air', role/'.air-previous')
        _install(work, role/'omc', role/'.omc-previous')
        model = load_shared(role/'omc', role/'air', prepared, role/'omc')
        if model['model']['sha256'] != state['saved']['model']['sha256']:
            raise ValueError('installed model differs from measured baseline')
        model['retained_validation'] = dict(numerical_validation=state['saved']['numerical_validation'],
            source='../archive.json', scope='USER_MEASURED_BASELINE; NOT_PRODUCTION_PROMOTION')
        _save(role/'omc/manifest.json', model)
        retained = _retain_runtime_previous(role, root/'runtime/runtime.json')
        for name in ('air', 'omc'):
            previous = role/('.'+name+'-previous')
            if previous.exists() and str(previous) not in retained:
                _tree_bytes(previous, allow_inactive_sockets=True)
                shutil.rmtree(previous)
        saved = dict(state['saved'], retained_runtime_dependencies=retained)
        _save(archive, saved)
        journal.unlink()
    return dict(saved, output=str(role), next='CLEAN_COMPLETED_DRAFT_TESTS_THEN_TARGET_ACCURACY')


def compile_shared(air, output, prepared, *, replace=False, **options):
    """Build outside tests; failed compilation never replaces the public model."""
    from .torch_export import compile_pytorch
    air, output, prepared = map(lambda p: Path(p).absolute(), (air, output, prepared))
    if not prepared.is_dir():
        raise ValueError('shared compilation requires --source original prepared weights')
    if not options.get('rope_fp32_table') or not options.get('cpu_feature_pack') or options.get('offline_weights'):
        raise ValueError('shared baseline requires --rope-fp32-table --cpu-feature-pack without offline weights')
    scratch = output.parent/'.omc-build'
    package = scratch/'package'
    with _air_locks(output, air):
        _destination(output, {OMC_KIND, 'qwen35-incremental-draft-compiled'})
        previous = output.parent/'.omc-previous'
        if previous.exists() and not output.exists():
            _destination(previous, {OMC_KIND, 'qwen35-incremental-draft-compiled'})
            previous.rename(output)
        if previous.exists():
            raise ValueError('previous shared model retained; validate current model before retiring it')
        if output.exists() and not replace:
            raise ValueError('shared model exists; use --replace for a new build')
        old = read_json(scratch/'manifest.json') if (scratch/'manifest.json').is_file() else {}
        if old.get('artifact_kind') == OMC_KIND and old.get('status') == 'PYTORCH_DRAFT_OMC_COMPILED':
            if (old.get('ddk_id') != options.get('ddk_id') or
                    old.get('compiler_options', {}).get('limit_partition_size') != options.get('limit_partition_size')):
                raise ValueError('completed staging build uses different compiler options; retained')
            # Resume packaging after an interrupted host operation. Binding
            # below independently checks the supplied AIR/prepared identity.
            result = old
        else:
            result = compile_pytorch(air, scratch, replace=replace, **options)
        if result['status'] != 'PYTORCH_DRAFT_OMC_COMPILED':
            return dict(result, manifest=str(scratch/'manifest.json'))
        _package(scratch, air, prepared, options.get('ddk_root'), package)
        try:
            _install(package, output, previous)
            result = load_shared(output, air, prepared, output)
        except BaseException:
            if not package.exists() and output.exists():
                output.rename(package)
            if previous.exists():
                previous.rename(output)
            raise
        _tree_bytes(scratch, allow_inactive_sockets=True)
        shutil.rmtree(scratch)
        _retain_runtime_previous(output.parent, output.parent.parent/'runtime/runtime.json')
        return dict(result, manifest=str(output/'manifest.json'))
