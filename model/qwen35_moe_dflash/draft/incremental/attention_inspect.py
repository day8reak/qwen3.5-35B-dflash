"""Read retained Attention compilation evidence and selected-DDK source text.

This does not import vendor modules, invoke OMG, decode opaque NPU tasks or
change a length. A source candidate's presence does not prove it was loaded.
"""
import hashlib
import json
import os
from pathlib import Path
import re

from qwen35_moe_dflash.draft.ir_inspect import _read, _select, inspect_draft_ir
from .attention_probe import _compiled_probe


_SOURCE_ROOTS = (
    'ddk/tbe/ops', 'tools/platform/kirinx90/ops', 'ddk/ascendcpp',
    'tools/tools_ascendc/include', 'tools/tools_ascendc/package', 'include',
)
_TEXT_SUFFIXES = {'.py', '.h', '.hpp', '.c', '.cc', '.cpp', '.json', '.ini', '.cfg', '.txt', '.info', '.log'}
_LIMITS = dict(directory_entries=24000, source_files=32, file_bytes=2 << 20,
               source_bytes=16 << 20, compiler_files=256, compiler_bytes=16 << 20,
               matching_compiler_files=16, log_bytes=32 << 20, excerpt_lines=96,
               source_excerpt_lines=360, compiler_excerpt_lines=160,
               line_characters=400, output_bytes=384 << 10)
_LENGTH = re.compile(r'(?:seq.*len|len.*seq|kv.*len|len.*kv|seq.*size|s[12].*size|tiling.*key)', re.I)
_OP = re.compile(r'(?:adn[_\W]*fused[_\W]*infer[_\W]*attention|probe_attention_(?:key0|key15|unmasked|kv32))', re.I)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _within(root, relative):
    """Do not follow inventory paths or SDK links outside their declared tree."""
    rel = Path(relative)
    if rel.is_absolute() or not rel.parts or any(p in ('.', '..') for p in rel.parts):
        raise ValueError('inspection path must be relative without parent traversal')
    path = root
    for part in rel.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError('inspection does not follow symlinks: '+str(rel))
    return path


def _excerpts(raw, *, source=False, max_lines=None):
    if b'\0' in raw:
        return dict(status='BINARY_NOT_DECODED')
    try:
        content = raw.decode('utf-8')
    except UnicodeDecodeError:
        return dict(status='NON_UTF8_NOT_DECODED')
    # Preserve original line numbers. Long/minified records are explicitly
    # truncated; do not pretend reserialized JSON lines came from the file.
    lines = content.splitlines()
    hits = [i for i, line in enumerate(lines) if _LENGTH.search(line)]
    op_hits = [i for i, line in enumerate(lines) if _OP.search(line)]
    wanted = set()
    limit = min(_LIMITS['excerpt_lines'], _LIMITS['excerpt_lines'] if max_lines is None else max_lines)
    for i in hits + op_hits + (list(range(min(16, len(lines)))) if source else []):
        for n in range(max(0, i-3), min(len(lines), i+7)):
            if len(wanted) < limit:
                wanted.add(n)
    selected = sorted(wanted)
    return dict(status='TEXT', total_lines=len(lines), length_match_lines=len(hits),
                operator_match_lines=len(op_hits), excerpts=[dict(line=i+1,
                    text=lines[i][:_LIMITS['line_characters']],
                    truncated=len(lines[i])>_LIMITS['line_characters']) for i in selected],
                omitted_matching_lines=sum(i not in wanted for i in set(hits+op_hits)))


def _ddk_sources(ddk):
    result = dict(scope='FILENAME_SELECTED_TEXT_ONLY; CANDIDATES_NOT_PROVEN_LOADED',
                  roots=[], files=[], entries_seen=0, read_bytes=0, limited=False)
    candidates = []
    for relative in _SOURCE_ROOTS:
        record = dict(path=relative, status='NOT_PRESENT')
        result['roots'].append(record)
        try:
            folder = _within(ddk, relative)
            if not folder.is_dir():
                continue
            record['status'] = 'SCANNED'
            pending = [folder]
            while pending:
                current = pending.pop()
                with os.scandir(current) as entries:
                    children = []
                    for entry in entries:
                        result['entries_seen'] += 1
                        if result['entries_seen'] > _LIMITS['directory_entries']:
                            result['limited'] = True
                            record['status'] = 'ENTRY_LIMIT'
                            break
                        if entry.is_symlink():
                            continue
                        path = Path(entry.path)
                        if entry.is_dir(follow_symlinks=False):
                            if len(path.relative_to(folder).parts) < 16:
                                children.append(path)
                            else:
                                result['limited'] = True
                        elif entry.is_file(follow_symlinks=False):
                            normalized = re.sub('[^a-z0-9]', '', str(path.relative_to(folder)).lower())
                            if 'fusedinferattention' in normalized and path.suffix.lower() in _TEXT_SUFFIXES:
                                candidates.append(path)
                    if result['limited'] and result['entries_seen'] > _LIMITS['directory_entries']:
                        break
                    pending.extend(sorted(children, reverse=True))
        except (OSError, ValueError) as error:
            record.update(status='UNREADABLE', error=str(error)[:512])
        if result['entries_seen'] > _LIMITS['directory_entries']:
            break
    candidates = sorted(set(candidates), key=lambda p: (
        'adnfusedinferattention' not in re.sub('[^a-z0-9]', '', p.name.lower()), str(p)))
    result['candidate_count'] = len(candidates)
    result['omitted_candidates'] = max(0, len(candidates)-_LIMITS['source_files'])
    line_budget = _LIMITS['source_excerpt_lines']
    for path in candidates[:_LIMITS['source_files']]:
        record = dict(path=str(path.relative_to(ddk)))
        result['files'].append(record)
        try:
            size = path.stat().st_size
            if result['read_bytes']+size > _LIMITS['source_bytes']:
                record['status'] = 'TOTAL_BYTE_LIMIT'
                result['limited'] = True
                break
            raw = _read(path, _LIMITS['file_bytes'])
            result['read_bytes'] += len(raw)
            excerpt = _excerpts(raw, source=True, max_lines=line_budget)
            line_budget -= len(excerpt.get('excerpts', []))
            record.update(bytes=len(raw), sha256=_sha(raw), **excerpt)
        except (OSError, ValueError) as error:
            record.update(status='UNREADABLE', error=str(error)[:512])
    result['status'] = ('CANDIDATES_FOUND' if any(r['status']=='TEXT' for r in result['files'])
                        else 'NO_READABLE_SOURCE_CANDIDATES')
    return result


def _compiler_records(root, ddk, ddk_id):
    folder = _within(root, 'compilation')
    raw = _read(_within(folder, 'report.json'), 16 << 20)
    report = json.loads(raw)
    if (report.get('ddk_id') != ddk_id or report.get('platform') != 'kirinx90'
            or report.get('model_compilation') != 'PASSED'):
        raise ValueError('retained compilation does not match selected X90 DDK or success state')
    omg = ddk/'tools/tools_omg/master/omg'
    omg_raw = _read(omg, 64 << 20)
    if _sha(omg_raw) != report.get('omg_sha256'):
        raise ValueError('selected OMG hash differs from retained compilation')
    result = dict(report_sha256=_sha(raw), selected_omg_sha256=_sha(omg_raw),
                  recorded_ddk_root=report.get('ddk_root'), files=[], read_bytes=0,
                  inspected_files=0, limited=False,
                  scope='HASH_CHECKED_RETAINED_TEXT; OPAQUE_NPU_TASKS_NOT_DECODED')
    log = folder/'compile.log'
    if log.exists():
        log_raw = _read(log, _LIMITS['log_bytes'])
        if _sha(log_raw) != report.get('compile_log_sha256'):
            raise ValueError('retained compile.log hash mismatch')
        result['log'] = dict(path='compile.log', bytes=len(log_raw), sha256=_sha(log_raw), **_excerpts(log_raw))
    else:
        result['log'] = dict(status='NOT_PRESENT')
    inventory = {}
    for key in ('generated_artifacts', 'compiler_work_artifacts'):
        for row in report.get(key, []):
            name = row['path']
            if name in inventory:
                raise ValueError('duplicate compiler inventory path')
            inventory[name] = row
    candidates = [name for name in sorted(inventory) if Path(name).suffix.lower() in _TEXT_SUFFIXES]
    # Check relevant filenames first; hash-named kernel metadata is also read
    # within the budget and selected by its contents.
    candidates.sort(key=lambda n: (not bool(_OP.search(n)), n))
    result['text_candidates'] = len(candidates)
    line_budget = _LIMITS['compiler_excerpt_lines']
    for name in candidates:
        if (result['inspected_files'] >= _LIMITS['compiler_files'] or
                len(result['files']) >= _LIMITS['matching_compiler_files']):
            result['limited'] = True
            break
        row = inventory[name]
        if row['bytes'] > _LIMITS['file_bytes']:
            result.setdefault('oversize_files', []).append(name)
            continue
        if result['read_bytes']+row['bytes'] > _LIMITS['compiler_bytes']:
            result['limited'] = True
            break
        path = _within(folder, name)
        data = _read(path, _LIMITS['file_bytes'])
        if len(data) != row['bytes'] or _sha(data) != row['sha256']:
            raise ValueError('compiler metadata inventory mismatch: '+name)
        result['inspected_files'] += 1
        result['read_bytes'] += len(data)
        if _OP.search(name) or _OP.search(data.decode('utf-8', errors='replace')):
            excerpt = _excerpts(data, max_lines=line_budget)
            line_budget -= len(excerpt.get('excerpts', []))
            result['files'].append(dict(path=name, bytes=len(data), sha256=_sha(data), **excerpt))
    result['status'] = 'INSPECTED'
    return result


def inspect_attention_lengths(root, ddk, ddk_id):
    """One read-only report; all values retain their evidence scope."""
    root, ddk = Path(root).resolve(), Path(ddk).resolve()
    report, model, schema, document, enums = _compiled_probe(root, ddk, ddk_id)
    compiled = _compiler_records(root, ddk, ddk_id)
    extraction = report['extraction']
    branches = extraction.get('control_experiment', {}).get('branches', [])
    names = [extraction['attention']] + [r['node'] for r in branches]
    if len(names)>5 or len(set(names))!=len(names):
        raise ValueError('unexpected Attention probe node list')
    result = dict(artifact_kind='qwen35-draft-attention-length-inspection', status='INSPECTED',
        scope='READ_ONLY_SELECTED_DDK_SOURCE_AND_RETAINED_COMPILATION',
        model_compilation='NOT_RUN', device_execution='NOT_RUN', numerical_validation='NOT_RUN',
        model=dict(report['model'], path=str(model)), ddk_id=ddk_id, ddk_root=str(ddk),
        schema=schema.evidence, limits=_LIMITS,
        source_control_experiment=extraction.get('control_experiment'),
        source_port_contract=dict(scope='AUDITED_SOURCE_CONVERTER_NOT_COMPILED_KERNEL_ABI',
            slots={'4':'atten_mask', '5':'actual_seq_lengths_q', '6':'actual_seq_lengths_kv',
                   '14':'block_table', '16':'all_seq_lengths_q'},
            original_values={'actual_seq_lengths_q':16, 'actual_seq_lengths_kv':224, 'all_seq_lengths_q':16}),
        compiled_main_graph_matches=[row for name in names for row in _select(document, name, enums)],
        compiler_records=compiled, ddk_source_candidates=_ddk_sources(ddk),
        limitations=[
            'No vendor module or compiler is executed. No fixture, AIR, OMC, DDK or runtime file is written.',
            'A source candidate is not proof of the implementation loaded during compilation.',
            'Serialized ports and logical readbacks are not physical kernel input readback.',
            'Absent IR/source text leaves length handling unresolved; no alternate DDK or schema is guessed.'])
    result['compiler_ir'] = inspect_draft_ir(root/'compilation',
        ddk/'tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so', node_names=names)
    if len(json.dumps(result, ensure_ascii=True).encode()) > _LIMITS['output_bytes']:
        # Keep source/compile excerpts useful even if optional IR is enormous.
        result['compiler_ir'] = dict(status='OUTPUT_LIMIT', instruction='use retained IR directly',
                                     previous_status=result['compiler_ir']['status'])
    if len(json.dumps(result, ensure_ascii=True).encode()) > _LIMITS['output_bytes']:
        raise ValueError('Attention inspection exceeds report budget')
    return result
