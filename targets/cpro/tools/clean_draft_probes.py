#!/usr/bin/env python3
"""Clean completed Draft probe families or obsolete compiler copies on the host."""
import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))
from qwen35_moe_dflash.common.hashing import sha256_file
from qwen35_moe_dflash.execution.runtime_config import read_json
from qwen35_moe_dflash.execution.run_layout import _active_processes, _runtime_paths

FAMILIES = {'pytorch-embedding': 'pytorch_embedding', 'pytorch-input-rms': 'pytorch_input_rms',
            'pytorch-input-rms-taps': 'pytorch_input_rms_taps',
            'pytorch-input-rms-no-scale': 'pytorch_input_rms_no_scale'}
KIND = 'qwen35-draft-vector-validation'


def _idle(root):
    if _active_processes(root):
        raise ValueError('local model job is active; probes retained')
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        if int(p.parent.name) == os.getpid():
            continue
        try:
            args = p.read_bytes().split(b'\0')
        except OSError:
            continue
        if (any(b'probe_draft_embedding.py' in a for a in args[:3])
                and any(str(root).encode() in a for a in args)):
            raise ValueError('embedding probe compilation is active; probes retained')


def _completed(base, case):
    run, result = base.with_name(base.name+'-run'), base.with_name(base.name+'-result')
    recovered = read_json(result/'result.json')
    if (recovered.get('artifact_kind') != KIND or recovered.get('stage') != 'complete'
            or recovered.get('status') not in {'PASSED', 'NUMERICAL_FAILED'}
            or recovered.get('device_execution') != 'EXECUTED_OUTPUTS_RETRIEVED'
            or recovered.get('lock_state') != 'NO_OWNED_LOCK_AFTER_KNOWN_COMPLETION'
            or not recovered.get('board_attempt') or recovered.get('requested_cases') != [case]
            or Path(recovered.get('recovery_source', '')).resolve() != run.resolve()):
        raise ValueError('known completion is missing; recover the same execution first')
    # Resume an interrupted deletion without keeping an archive or a journal.
    if not base.exists() and not run.exists():
        return recovered['board_attempt']
    request = read_json(run/'request.json')
    fixture = run/'fixture/manifest.json'
    meta = read_json(fixture)
    compiled = read_json(base/'result.json') if base.exists() else meta.get('source_evidence', {})
    if (compiled.get('artifact_kind') != 'qwen35-draft-pytorch-embedding-probe'
            or compiled.get('status') != 'COMPILED' or compiled.get('model_compilation') != 'PASSED'
            or compiled.get('probe_case', 'pytorch_embedding') != case
            or request.get('artifact_kind') != KIND or request.get('cases') != [case]
            or request.get('fixture_sha256') != sha256_file(fixture)
            or not request.get('board_attempt') or recovered.get('board_attempt') != request['board_attempt']
            or len(meta.get('cases', [])) != 1 or meta['cases'][0].get('case') != case
            or not compiled.get('model', {}).get('sha256')
            or meta['cases'][0].get('model', {}).get('sha256') != compiled['model']['sha256']):
        raise ValueError('completion/model binding is missing or differs; recover the same execution first')
    return request['board_attempt']


def clean(root, config, *, apply=False):
    root = Path(root).resolve(strict=True)
    tests = root/'draft/tests'
    if not tests.is_dir() or tests.resolve() != tests:
        raise ValueError('expected owned draft/tests directory without symlink parents')
    protected = _runtime_paths(Path(config).resolve(strict=True))
    _idle(root)
    removed, planned, retained = [], [], []
    with ExitStack() as stack:
        for name, case in FAMILIES.items():
            base = tests/name
            paths = [base, tests/(name+'-run'), tests/(name+'-result')]
            if not any(p.exists() or p.is_symlink() for p in paths):
                continue
            try:
                if any(p.is_symlink() or (p.exists() and not p.is_dir()) for p in paths):
                    raise ValueError('unexpected file or symlink; retained')
                if any(asset == p or p in asset.parents for p in paths for asset in protected):
                    raise ValueError('runtime configuration references this probe; retained')
                lock = stack.enter_context((tests/('.'+name+'.lock')).open('a'))
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                attempt = _completed(base, case)
                size = sum(p.lstat().st_size for folder in paths for p in folder.rglob('*') if p.is_file() and not p.is_symlink())
                planned.append(dict(family=name, directories=[str(p) for p in paths if p.exists()],
                                    logical_bytes=size, board_attempt=attempt))
            except (OSError, ValueError, KeyError, TypeError) as error:
                retained.append(dict(family=name, reason=str(error)))
        _idle(root)
        if apply:
            for row in planned:
                base = tests/row['family']
                if _completed(base, FAMILIES[row['family']]) != row['board_attempt']:
                    raise ValueError('probe attempt changed during cleanup')
                # Remove compile data first; preserve completion evidence until
                # its family's large payloads have been removed successfully.
                for path in row['directories']:
                    shutil.rmtree(path)
                    removed.append(path)
    return dict(status=('PARTIAL' if retained else 'CLEANED') if apply else 'PLANNED',
                scope='HOST_ONLY_FOUR_COMPLETED_PROBE_FAMILIES; NO_BOARD_ACCESS_OR_ARCHIVE',
                candidates=planned, removed=removed, retained=retained,
                logical_bytes_selected=sum(r['logical_bytes'] for r in planned))


def clean_compiler_scratch(root, config, *, apply=False, integration_sidecars=False):
    from qwen35_moe_dflash.toolchain.compiler_storage import release_packed_scratch
    from qwen35_moe_dflash.draft.incremental.torch_export import OMC_KIND
    root = Path(root).resolve(strict=True)
    tests = root/'draft/tests'
    if not tests.is_dir() or tests.resolve() != tests:
        raise ValueError('expected owned draft/tests directory without symlink parents')
    protected = _runtime_paths(Path(config).resolve(strict=True))
    results, retained = [], []
    # Current candidates keep compiler AIR, snapshots and the small external
    # OMC descriptor: standalone ABI binding still reads them.
    names = (('top1-integration/compiled', 'top1-integration/rope-fp32/compiled',
              'top1-integration/feature-pack/compiled')
             if integration_sidecars else ('pytorch-omc', 'pytorch-optimized-omc'))
    for name in names:
        folder = tests/name
        if not folder.exists() and not folder.is_symlink():
            continue
        try:
            if folder.resolve() != folder or not folder.is_dir():
                raise ValueError('candidate directory is linked or foreign')
            fd = os.open(folder.parent/('.'+folder.name+'.lock'), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, 'a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                manifest = read_json(folder/'manifest.json')
                if (manifest.get('artifact_kind') != OMC_KIND or manifest.get('stage') != 'complete'
                        or manifest.get('status') != 'PYTORCH_DRAFT_OMC_COMPILED'
                        or manifest.get('omc') != 'model.omc'
                        or manifest.get('weight_storage') != 'embedded-stream'):
                    raise ValueError('not a completed packed PyTorch candidate')
                print('draft-clean: checking '+name+' packed model and compiler completion', file=sys.stderr, flush=True)
                results.append(dict(candidate=name, **release_packed_scratch(folder, apply=apply,
                                    protected=protected, sidecars_only=integration_sidecars)))
        except (OSError, ValueError, KeyError, TypeError) as error:
            retained.append(dict(candidate=name, reason=str(error)))
    return dict(status=('PARTIAL' if retained else 'CLEANED') if apply else 'PLANNED',
        scope=('TOP1_ROPE_FEATURE_PACK_WEIGHT_DUPLICATES_ONLY; NO_DEVICE_ACCESS' if integration_sidecars
               else 'TWO_OLD_CANDIDATES_COMPILER_SCRATCH_ONLY; NO_DEVICE_ACCESS'),
        candidates=results, retained=retained,
        allocated_bytes_selected=sum(r['allocated_bytes_selected'] for r in results),
        free_bytes_after=shutil.disk_usage(root).free)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--config')
    p.add_argument('--apply', action='store_true')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--compiler-scratch', action='store_true',
                   help='retain old packed models/reports; remove only their completed compiler copies')
    mode.add_argument('--integration-sidecars', action='store_true',
                      help='remove packed weight duplicates from top1/RoPE/full feature-pack builds; retain ABI/source snapshots')
    mode.add_argument('--layer0-probes', action='store_true',
                      help='retire completed hidden-layer0 compiled/run/result; retain summary and top-level analysis JSON')
    a = p.parse_args()
    try:
        if a.integration_sidecars:
            result = clean_compiler_scratch(a.root, a.config or Path(a.root)/'runtime/runtime.json',
                                            apply=a.apply, integration_sidecars=True)
        elif a.layer0_probes:
            from qwen35_moe_dflash.draft.incremental.probe_retention import clean_layer0
            action = clean_layer0
        else:
            action = clean_compiler_scratch if a.compiler_scratch else clean
        if not a.integration_sidecars:
            result = action(a.root, a.config or Path(a.root)/'runtime/runtime.json', apply=a.apply)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result = dict(status='FAILED', error=f'{type(error).__name__}: {error}')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(result['status'] in {'FAILED', 'PARTIAL'})


if __name__ == '__main__':
    raise SystemExit(main())
