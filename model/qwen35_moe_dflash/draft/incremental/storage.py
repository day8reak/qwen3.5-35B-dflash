"""Build in a reusable test slot; publish one flat, self-contained Draft bundle."""
from copy import deepcopy
import fcntl
import os
from pathlib import Path
import shutil

from qwen35_moe_dflash.common.artifacts import output_path
from qwen35_moe_dflash.execution.runtime_config import read_json
from qwen35_moe_dflash.execution.run_layout import _active_processes, _save
from .runtime import bind_compiled

KIND = 'qwen35-incremental-draft-compiled'
SUCCESS = 'DRAFT_INCREMENTAL_OMC_COMPILED_ABI_BOUND'
FILES = {'manifest.json', 'model.omc', 'compiled-abi.json', 'weight-packing.json', 'compiled-precision.json'}


def _known_bundle(folder):
    if not folder.exists():
        return
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError('Draft output must be a real directory')
    children = list(folder.iterdir())
    if any(p.name not in FILES or p.is_symlink() or not p.is_file() for p in children):
        raise ValueError('Draft output contains unselected files; organize it before replacing')
    if (folder / 'manifest.json').exists():
        if read_json(folder / 'manifest.json').get('artifact_kind') != KIND:
            raise ValueError('output is not an incremental Draft bundle')
    elif any(p.name != 'model.omc' for p in children):
        raise ValueError('unrecognized Draft output')


def _validate(folder, work):
    meta = read_json(folder / 'manifest.json')
    scratch = work / '.binding-check'
    scratch.mkdir(parents=True, exist_ok=True)
    generated = scratch / 'incremental-draft-abi.json'
    generated.unlink(missing_ok=True)
    try:
        bind_compiled(folder, scratch, ddk_id=meta['ddk_id'], max_draft_tokens=meta['max_draft_tokens'],
                      max_proposals=1, total_tokens=1, verify_sources=False)
    finally:
        generated.unlink(missing_ok=True)
        scratch.rmdir()
    return meta


def _finish_publication(output, work, journal):
    receipt = read_json(journal)
    if receipt['output'] != output.name:
        raise ValueError('pending Draft publication belongs to another output')
    package, previous = work / 'package', output.with_name('.' + output.name + '-previous')
    if package.exists():
        _validate(package, work)
        if output.exists():
            if previous.exists():
                raise ValueError('both old and current Draft directories exist before publication')
            _known_bundle(output)
            output.rename(previous)
        package.rename(output)
    meta = _validate(output, work)
    if meta['model'] != receipt['model']:
        raise ValueError('published Draft does not match pending publication')
    # Keep at most one rollback bundle until the new model completes a board
    # loop. Compilation/ABI checks alone do not demonstrate device execution.
    if previous.exists():
        _known_bundle(previous)
    shutil.rmtree(work)
    if previous.exists():
        _save(journal, dict(receipt, stage='WAITING_BOARD_SMOKE'))
    else:
        journal.unlink()
    return meta


def compile_shared(export_dir, output, *, replace_failed=False, **options):
    if Path(output).is_symlink():
        raise ValueError('shared Draft output must not be a symlink')
    output = output_path(output)
    if (output.parent / 'tests').is_symlink() or (output.parent / 'tests/compile').is_symlink():
        raise ValueError('Draft compile slot must not be a symlink')
    work = output_path(output.parent / 'tests/compile')
    if output == work or output in work.parents or work in output.parents:
        raise ValueError('shared OMC directory must be separate from tests/compile')
    if Path(export_dir).resolve().is_relative_to(output):
        raise ValueError('Draft AIR must be separate from its OMC output')
    output.parent.mkdir(parents=True, exist_ok=True)
    journal = output.parent / '.draft-publication.json'
    # One compilation owns this role's fixed slot; do not race two retries.
    fd = os.open(output.parent / '.draft-compile.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if _active_processes(Path(os.environ['AI_RUN_DIR']).resolve()):
            raise ValueError('finish active local model jobs before replacing shared Draft artifacts')
        if journal.exists():
            if read_json(journal).get('stage') == 'WAITING_BOARD_SMOKE':
                raise ValueError('current Draft awaits its board loop; run/recover it before compiling again')
            return _finish_publication(output, work, journal)
        if output.with_name('.' + output.name + '-previous').exists():
            raise ValueError('unowned Draft previous directory; not replaced')
        _known_bundle(output)
        if work.exists():
            old = read_json(work / 'manifest.json')
            if old.get('artifact_kind') != KIND or old.get('device_execution') != 'NOT_RUN':
                raise ValueError('compile slot is not an inactive Draft build')
            if old.get('status') == SUCCESS:
                result = old  # Retry publication without rerunning OMG.
            elif replace_failed and old.get('status') == 'FAILED':
                shutil.rmtree(work)
                result = None
            else:
                raise ValueError('reuse failed compilation with --replace-failed')
        else:
            result = None
        if result is None:
            from .compile import compile_incremental_draft
            result = compile_incremental_draft(export_dir, work, **options)
        if result['status'] != SUCCESS:
            return dict(result, manifest=str(work / 'manifest.json'), shared_output=str(output))
        _validate(work, work)
        package = work / 'package'
        package.mkdir(exist_ok=True)
        meta = deepcopy(result)
        meta.update(output='.', next='Run the board-local incremental Draft smoke using this shared directory')
        meta['model']['path'] = 'model.omc'
        # Metadata and OMC are linked within the same filesystem, never copied.
        # The job is removed only after the standalone bundle validates.
        for record in (result['model'], result['compiled_abi_file'], result['weight_packing']):
            source = work / record['path']
            destination = package / ('model.omc' if record is result['model'] else record['path'])
            if destination.exists():
                if not os.path.samefile(source, destination):
                    raise FileExistsError(str(destination))
            else:
                os.link(source, destination)
        precision = work / 'compiled-precision.json'
        if precision.exists() and not (package / precision.name).exists():
            os.link(precision, package / precision.name)
        meta.pop('compilation_report', None)
        _save(package / 'manifest.json', meta)
        _validate(package, work)
        _save(journal, dict(output=output.name, model=meta['model']))
        return _finish_publication(output, work, journal)
    finally:
        os.close(fd)


def retire_previous_after_smoke(result):
    """Discard the one rollback bundle only after this exact new Draft ran."""
    if (result.get('status') != 'BOARD_SERIAL_DFLASH_EXECUTED_UNVALIDATED'
            or result.get('device_execution') != 'EXECUTED_OUTPUTS_RETRIEVED'
            or result.get('counts', {}).get('draft_native_calls', 0) < 1):
        return
    prepared = read_json(Path(result['prepared']) / 'manifest.json')
    draft = prepared['draft']
    if not draft.get('compiled_manifest'):
        return
    output = output_path(Path(draft['compiled_manifest']['path']).parent)
    journal = output.parent / '.draft-publication.json'
    if not journal.exists():
        return
    fd = os.open(output.parent / '.draft-compile.lock', os.O_RDWR | os.O_NOFOLLOW)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = read_json(journal)
        current = read_json(output / 'manifest.json')
        if (receipt.get('stage') != 'WAITING_BOARD_SMOKE' or receipt['output'] != output.name
                or receipt['model'] != current['model']
                or any(current['model'][k] != draft['model_source'][k] for k in ('bytes', 'sha256'))):
            return
        previous = output.with_name('.' + output.name + '-previous')
        if previous.exists():
            _known_bundle(previous)
            shutil.rmtree(previous)
        journal.unlink()
        result['draft_artifact_cleanup'] = 'PREVIOUS_MODEL_REMOVED_AFTER_MATCHING_BOARD_LOOP'
    finally:
        os.close(fd)
