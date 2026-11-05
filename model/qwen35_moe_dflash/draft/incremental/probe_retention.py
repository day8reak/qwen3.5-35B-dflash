"""Retire the completed 55-checkpoint layer0 batch, retaining its conclusions."""
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
