"""Retire completed probes and superseded compiler copies, retaining dependencies."""
from contextlib import ExitStack
import fcntl
import os
from pathlib import Path
import shutil

from qwen35_moe_dflash.common.hashing import sha256_file
from qwen35_moe_dflash.execution.runtime_config import read_json
from qwen35_moe_dflash.execution.run_layout import _runtime_paths, _save
from qwen35_moe_dflash.toolchain.compiler_storage import _plain, _tree_bytes, require_idle

KIND = 'qwen35-layer0-probe-retention'
VECTOR = 'qwen35-draft-vector-validation'
SUITE = 'qwen35-draft-layer0-prefix-suite'
DIRECTORIES = ('compiled', 'run', 'result')
CASES = [f'hidden_layer0_c{c:02d}_p{p:02d}' for p in range(55) for c in (0, 2, 17)]


def clean_retired_rope_scratch(root, config, *, apply=False):
    """Keep the accepted packed baseline; release only its predecessor's scratch.

    Both embedded OMCs, source exports, fixtures and recovered snapshots stay in
    place. The old RoPE compiler copies are no longer needed for replay/profile;
    fresh ABI rebinding of that retired build would require rebuilding them.
    """
    from .accuracy import KIND as ACCURACY
    from .feature_pack_compare import compare_result
    from ...toolchain.compiler_storage import release_packed_scratch
    root = Path(root).resolve(strict=True)
    top = root/'draft/tests/top1-integration'
    current = top/'feature-pack'
    previous = top/'rope-fp32/compiled'
    for folder in (top, current, current/'compiled', previous):
        _plain(folder, directory=True)
    protected = _runtime_paths(Path(config).resolve(strict=True))
    require_idle(top)
    with ExitStack() as stack:
        for folder in (current/'compiled', previous, current/'run', current/'baseline-run'):
            fd = os.open(folder.parent/('.'+folder.name+'.lock'), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            lock = stack.enter_context(os.fdopen(fd, 'a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for run_name, result_name in (('run', 'result'), ('baseline-run', 'baseline-result')):
            execution = current/run_name
            _plain(execution/'request.json')
            _plain(current/result_name/'result.json')
            request = read_json(execution/'request.json')
            result = read_json(current/result_name/'result.json')
            fixture = (current/result_name/Path(result.get('fixture', ''))).resolve()
            if (result.get('artifact_kind') != ACCURACY or result.get('stage') != 'complete'
                    or result.get('status') not in ('PASSED', 'NUMERICAL_FAILED')
                    or result.get('device_execution') != 'EXECUTED_OUTPUTS_RETRIEVED'
                    or result.get('lock_state') != 'NO_OWNED_LOCK_AFTER_KNOWN_COMPLETION'
                    or Path(result.get('recovery_source', '')).resolve() != execution
                    or request.get('artifact_kind') != ACCURACY
                    or not result.get('board_attempt') or request.get('board_attempt') != result['board_attempt']
                    or (execution/Path(request.get('fixture', 'fixture'))).resolve() != fixture
                    or request.get('fixture_sha256') != result.get('fixture_sha256')
                    or result.get('fixture_sha256') != sha256_file(fixture/'manifest.json')):
                raise ValueError('packed baseline and predecessor need matching completed recoveries and released locks')
            protected.append(fixture)
        comparison = compare_result(current/'result')
        if (comparison.get('baseline_equivalence') != 'PASSED'
                or comparison.get('differences') != [] or comparison.get('compared_regions') != 488
                or Path(comparison['candidate_model']['path']).resolve() != current/'compiled/model.omc'
                or Path(comparison['baseline_model']['path']).resolve() != previous/'model.omc'):
            raise ValueError('full packed baseline equivalence or fixed model binding differs; scratch retained')
        # Preserve the current compiler's complete independent inputs. Do not
        # retire the old tree if current inputs/export links still depend on it.
        for folder in (current/'compiled/compilation/compile-work', top/'export'):
            _plain(folder, directory=True)
            for path in folder.rglob('*'):
                resolved = path.resolve()
                if previous == resolved or previous in resolved.parents:
                    raise ValueError('current build/export still depends on old RoPE compiler files; retained')
        model = current/'compiled/model.omc'
        _plain(model)
        if (model.stat().st_size != comparison['candidate_model']['bytes']
                or sha256_file(model) != comparison['candidate_model']['sha256']):
            raise ValueError('accepted packed model changed; predecessor scratch retained')
        _plain(previous/'weight-packing.json')
        packing = read_json(previous/'weight-packing.json')
        if any(packing.get('output', {}).get(k) != comparison['baseline_model'][k] for k in ('bytes', 'sha256')):
            raise ValueError('retired RoPE OMC differs from the compared predecessor')
        require_idle(top)
        result = release_packed_scratch(previous, apply=apply, protected=protected)
        return dict(result, scope='PREVIOUS_ROPE_COMPILER_SCRATCH_ONLY; CURRENT_PACKED_BUILD_AND_ALL_MODELS_FIXTURES_RESULTS_RETAINED',
            accepted_model=comparison['candidate_model'], baseline_equivalence='PASSED',
            free_bytes_after=shutil.disk_usage(root).free,
            limitation='Old RoPE fresh compiler-ABI binding needs regenerated scratch; retained fixture replay, offline comparison and profiling remain available.')


def _completed(base):
    paths = {name: base/name for name in (
        'compiled/result.json', 'run/request.json', 'run/fixture/manifest.json', 'result/result.json')}
    for path in paths.values():
        _plain(path)
    build, request, fixture, result = [read_json(p, 32 << 20) for p in paths.values()]
    diagnostics = result.get('execution_diagnostics', {})
    if (result.get('artifact_kind') != VECTOR or result.get('stage') != 'complete'
            or result.get('status') not in ('PASSED', 'NUMERICAL_FAILED')
            or result.get('device_execution') != 'EXECUTED_OUTPUTS_RETRIEVED'
            or result.get('lock_state') != 'NO_OWNED_LOCK_AFTER_KNOWN_COMPLETION'
            or Path(result.get('recovery_source', '')).resolve() != base/'run'
            or result.get('requested_cases') != CASES or result.get('validated_cases') != CASES
            or result.get('coverage_complete') is not True
            or diagnostics.get('native_exitcode') != 0 or diagnostics.get('archive_exitcode') != 0
            or diagnostics.get('case_exitcodes') != dict.fromkeys(CASES, 0) or diagnostics.get('errors')):
        raise ValueError('layer0 batch lacks complete recovered outputs and released-lock evidence; retained')
    if (request.get('artifact_kind') != VECTOR or request.get('cases') != CASES
            or not result.get('board_attempt') or request.get('board_attempt') != result['board_attempt']
            or request.get('fixture_sha256') != sha256_file(paths['run/fixture/manifest.json'])
            or fixture.get('source_evidence') != build
            or [c.get('case') for c in fixture.get('cases', [])] != CASES
            or [c.get('case') for c in result.get('cases', [])] != CASES
            or build.get('artifact_kind') != SUITE or build.get('stage') != 'complete'
            or build.get('status') != 'COMPILED' or build.get('model_compilation') != 'PASSED'
            or len(build.get('cases', [])) != 55 or any(c.get('status') != 'COMPILED' for c in build['cases'])
            or build.get('coverage', {}).get('compiled_checkpoints') != 55
            or build.get('coverage', {}).get('missing_checkpoints') != []):
        raise ValueError('layer0 compile/request/fixture/result binding differs; retained')
    source_result = Path(build['source_result']).resolve()
    if source_result == base or base in source_result.parents:
        raise ValueError('layer0 source accuracy result is inside the cleanup directory; retained')
    keys = ('status', 'stage', 'numerical_validation', 'device_execution', 'board_attempt', 'lock_state',
            'execution_diagnostics', 'warmup', 'repetitions', 'atol', 'rtol', 'scope',
            'repeat_bitwise_stable', 'coverage_complete', 'probe_coverage')
    compact = {k: result[k] for k in keys if k in result}
    points = {c['case']: c['checkpoint'] for c in fixture['cases']}
    compact['cases'] = [dict({k: c[k] for k in ('case', 'passed', 'repeat_bitwise_stable', 'numerics') if k in c},
                            checkpoint=points[c['case']]) for c in result['cases']]
    return dict(artifact_kind=KIND, status='PLANNED', base=str(base),
        original_model_sha256=build['original_model_sha256'], source_air_sha256=build['source_air_sha256'],
        source_result=str(source_result), ddk_id=build['ddk_id'], validation=compact,
        report_hashes={name: sha256_file(path) for name, path in paths.items()},
        scope='HOST_LAYER0_COMPILED_RUN_RESULT_ONLY; NO_DEVICE_ACCESS',
        retained='THIS_SUMMARY_AND_EXISTING_TOP_LEVEL_ANALYSIS_FILES',
        limitation='RAW_PREFIX_MODELS_AND_TENSORS_REMOVED_ON_APPLY; PREFIX_ANALYSIS_CANNOT_BE_REPLAYED',
        directories=[], removed=[])


def clean_layer0(root, config, *, apply=False):
    root = Path(root).resolve(strict=True)
    base = root/'draft/tests/top1-integration/hidden-layer0'
    if not base.exists() and not base.is_symlink():
        return dict(status='NOT_FOUND', directory=str(base), free_bytes_after=shutil.disk_usage(root).free)
    _plain(base, directory=True)
    protected = _runtime_paths(Path(config).resolve(strict=True))
    require_idle(base)
    summary = base/'retained-summary.json'
    if summary.is_symlink():
        raise ValueError('retention summary is linked; retained')
    temporary = summary.with_name(summary.name+'.tmp')
    if temporary.exists() or temporary.is_symlink():
        _plain(temporary)
    with ExitStack() as stack:
        for name in DIRECTORIES:
            fd = os.open(base/('.'+name+'.lock'), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            lock = stack.enter_context(os.fdopen(fd, 'a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            path = base/name
            if path.exists() or path.is_symlink():
                _plain(path, directory=True)
            if any(asset == path or path in asset.parents or asset in path.parents for asset in protected):
                raise ValueError('runtime references the layer0 probe directory; retained')
        old = read_json(summary, 32 << 20) if summary.exists() else {}
        if old and (old.get('artifact_kind') != KIND or old.get('base') != str(base)):
            raise ValueError('foreign retention summary; retained')
        if old.get('status') == 'CLEANED' and not any((base/n).exists() for n in DIRECTORIES):
            return dict(status='ALREADY_CLEAN', summary=str(summary), free_bytes_after=shutil.disk_usage(root).free)
        resuming = old.get('status') == 'REMOVING'
        result = old if resuming else _completed(base)
        if not resuming:
            for name in DIRECTORIES:
                info = (base/name).stat()
                result['directories'].append(dict(path=name, device=info.st_dev, inode=info.st_ino,
                                                  allocated_bytes=_tree_bytes(base/name)))
        if [r.get('path') for r in result['directories']] != list(DIRECTORIES):
            raise ValueError('unexpected retention directory set; retained')
        for row in result['directories']:
            path = base/row['path']
            if path.exists():
                info = path.stat()
                if (info.st_dev, info.st_ino) != (row['device'], row['inode']):
                    raise ValueError('probe directory was replaced after cleanup preparation; retained')
                _tree_bytes(path)  # Reject nested mounts or special files on resume as well.
        allocated = sum(r['allocated_bytes'] for r in result['directories'] if (base/r['path']).exists())
        before = shutil.disk_usage(root).free
        if apply:
            require_idle(base)
            result['status'] = 'REMOVING'
            _save(summary, result)
            for name in DIRECTORIES:
                if (base/name).exists():
                    shutil.rmtree(base/name)
                if name not in result['removed']:
                    result['removed'].append(name)
                _save(summary, result)
            result.update(status='CLEANED', free_bytes_after=shutil.disk_usage(root).free,
                          observed_free_bytes_change=shutil.disk_usage(root).free-before)
            _save(summary, result)
        return dict(status=result['status'] if apply else 'PLANNED', scope=result['scope'],
            directories=[str(base/n) for n in DIRECTORIES], allocated_bytes_selected=allocated,
            summary=str(summary), retained_analysis_files=[str(p) for p in base.glob('*.json') if p != summary],
            free_bytes_after=shutil.disk_usage(root).free,
            observed_free_bytes_change=result.get('observed_free_bytes_change') if apply else None,
            limitation='Allocated bytes can overestimate space freed from linked files; use free_bytes_after.')
