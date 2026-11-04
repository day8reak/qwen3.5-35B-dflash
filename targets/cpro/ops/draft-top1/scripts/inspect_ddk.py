#!/usr/bin/env python3
"""Collect selected X90 custom-op sources read-only; no imports/build/device calls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tarfile

ROOTS = ('ddk/ascendcpp', 'tools/platform/kirinx90', 'tools/tools_ascendc',
         'ddk/ai_ddk_lib/include', 'ddk/ccec_compiler', 'tools/tools_omg/master',
         'ddk/tbe/ops', 'samples', 'examples')
TOPICS = {
    'custom_registration': r'AddCustom|add_custom|OpDefRegistry|OP_ADD\(|REGISTER_CUSTOM_OP|REGISTER_OP|REG_OP\(|KIRIN_CONF_JSON_PATH|ascendc_op_kernel_info',
    'reduction_index': r'WholeReduceMax|ReduceMax|calIndex|ReinterpretCast|uint16_t.*index',
    'output_tail': r'DataCopyPad|DataCopyExtParams|SetValue\(|DataCacheCleanAndInvalid',
    'platform_capacity': r'GetCoreNum|GetCoreMemSize|UB_SIZE|Unified_Buffer|kirinx90|CORE_NUM',
    'build_registration': r'ascendc_library|add_custom_command|op_kernel|op_host|op_info_cfg|opFile|kernelName|bisheng|te_update_version',
}
PATTERNS = {key: re.compile(pattern, re.I) for key, pattern in TOPICS.items()}
SUFFIXES = {'.h', '.hpp', '.c', '.cc', '.cpp', '.py', '.json', '.ini', '.cfg', '.cmake', '.sh', '.md', '.txt'}
LIMITS = {'entries': 50000, 'file_bytes': 2 * 1024 * 1024,
          'read_bytes': 64 * 1024 * 1024, 'copied_bytes': 12 * 1024 * 1024,
          'copied_files': 160, 'excerpt_lines': 360, 'identity_bytes': 128 * 1024 * 1024}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def priority(path):
    value = str(path).lower()
    # Real complete examples and registration config precede generic headers.
    if (re.search(r'add_?custom|custom.*(?:sample|example)|(?:sample|example).*custom', value)
            or any(part in {'samples', 'examples', 'op_host', 'op_kernel'} for part in path.parts)):
        return 0
    if 'kirinx90' in value or '/ascendcpp/conf/' in value:
        return 1
    if re.search(r'kernel_operator_(?:reduce|data_copy|tensor|common)|platform_ascendc|op_def_registry|custom.*cmake', value):
        return 2
    if path.name in ('CMakeLists.txt', 'CMakePresets.json') or path.suffix in ('.ini', '.cfg'):
        return 3
    if re.search(r'bitcast|arg_max|argmax|reduce_max|register|build|compile|kernel_operator', value):
        return 4
    return 5


def inspect(root, output, compile_environment=None, source_roots=(), *, make_archive=True):
    root, output = Path(root).resolve(strict=True), Path(output).resolve()
    if not root.is_dir():
        raise ValueError('DDK root must be a directory')
    extra = [Path(p).resolve(strict=True) for p in source_roots]
    approved = [root, *extra]
    if any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in approved):
        raise ValueError('output must be separate from all inspected roots')
    if output.exists():
        raise FileExistsError('retain the existing bundle; choose a fresh output only for a changed SDK')
    output.mkdir(parents=True)
    (output / 'sources').mkdir()
    report = {'artifact_kind': 'qwen35-custom-top1-ddk-context', 'status': 'INSPECTED',
        'platform': 'kirinx90', 'ddk_root': str(root), 'declared_ddk_id': os.environ.get('CPRO_DDK_ID'),
        'scope': 'READ_ONLY_SELECTED_SDK_FILES_NOT_BUILD_OR_SUPPORT_PROOF',
        'model_compilation': 'NOT_RUN', 'device_execution': 'NOT_RUN',
        'collector_sha256': digest(Path(__file__).read_bytes()), 'limits': LIMITS,
        'roots': [], 'identity': [], 'files': [], 'notes': [], 'symlinks': [],
        'topic_evidence': {key: [] for key in TOPICS}}
    used = {'entries': 0, 'read_bytes': 0, 'copied_bytes': 0, 'copied_files': 0, 'identity_bytes': 0}

    def note(path, reason):
        report['status'] = 'PARTIAL'
        if len(report['notes']) < 100:
            report['notes'].append({'path': str(path), 'reason': str(reason)})

    def permitted(path):
        return any(path == p or path.is_relative_to(p) for p in approved)

    def read(path):
        size = path.stat().st_size
        if size > LIMITS['file_bytes'] or used['read_bytes'] + size > LIMITS['read_bytes']:
            note(path, 'read budget exceeded; existence is not implementation evidence')
            return None
        with path.open('rb') as stream:
            raw = stream.read(size + 1)
        if len(raw) != size:
            note(path, 'file changed while reading')
            return None
        used['read_bytes'] += size
        return raw

    starts = [root / path for path in ROOTS] + extra
    if compile_environment:
        env_path = Path(compile_environment).resolve(strict=True)
        raw = read(env_path)
        if raw is None:
            raise ValueError('compile environment unreadable within budget')
        env = json.loads(raw)
        if Path(env.get('ddk_root', '')).resolve() != root:
            raise ValueError('saved compile environment belongs to a different DDK')
        values = env.get('environment', {})
        conf = values.get('KIRIN_CONF_JSON_PATH')
        report['compile_environment'] = {'path': str(env_path), 'sha256': digest(raw),
            'KIRIN_CONF_JSON_PATH': conf, 'SOC_VERSION': values.get('SOC_VERSION')}
        if conf:
            config = Path(conf)
            if config.is_absolute() and permitted(config.resolve()):
                starts.insert(0, config)
            else:
                note(config, 'config outside selected roots or relative; declare its containing --source-root')

    # Traverse only declared roots. Internal symlinks may expose vendor includes;
    # external links are recorded, never used to silently change the SDK.
    pending, seen, files = list(dict.fromkeys(starts)), set(), set()
    while pending:
        path = pending.pop(0)
        used['entries'] += 1
        if used['entries'] > LIMITS['entries']:
            note(root, 'entry limit reached; inventory incomplete')
            break
        try:
            resolved = path.resolve(strict=True)
            if path.is_symlink() and len(report['symlinks']) < 100:
                report['symlinks'].append({'path': str(path), 'target': str(resolved), 'allowed': permitted(resolved)})
            if not permitted(resolved):
                note(path, 'symlink target outside declared roots; skipped')
                continue
            if path in starts:
                report['roots'].append({'path': str(path), 'resolved': str(resolved)})
            if resolved in seen:
                continue
            seen.add(resolved)
            if resolved.is_dir():
                if resolved.name in {'__pycache__', '.git', 'node_modules'}:
                    continue
                pending.extend(sorted(resolved.iterdir(), key=lambda p: (priority(p), str(p))))
            elif resolved.is_file():
                files.add(resolved)
        except (OSError, RuntimeError) as error:
            note(path, error)

    # ELF/config identity is hashed, not executed, and not copied into the bundle.
    identity_names = {'omg', 'bisheng', 'ccec', 'libai_fmk_graph_proto_host.so',
                      'libai_npucore_ascendc_kernel.so', 'version.info'}
    for path in sorted(files):
        if path.name not in identity_names:
            continue
        try:
            size = path.stat().st_size
            row = {'path': str(path), 'bytes': size}
            if used['identity_bytes'] + size <= LIMITS['identity_bytes']:
                hasher = hashlib.sha256()
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        hasher.update(block)
                row['sha256'] = hasher.hexdigest()
                used['identity_bytes'] += size
            else:
                row['hash_status'] = 'BUDGET_SKIPPED'
                note(path, 'identity hash budget exceeded')
            report['identity'].append(row)
        except OSError as error:
            note(path, error)

    for path in sorted(files, key=lambda p: (priority(p), str(p))):
        if path.suffix.lower() not in SUFFIXES and path.name not in ('CMakeLists.txt', 'Makefile'):
            continue
        # Unrelated generic files consume no read budget. Prioritized headers,
        # config, custom samples and build scripts cover the bounded ABI questions.
        if priority(path) == 5:
            continue
        try:
            raw = read(path)
            if raw is None:
                continue
            content = raw.decode('utf-8')
            if '\0' in content:
                continue
            topics = [key for key, pattern in PATTERNS.items() if pattern.search(content)]
            if not topics and priority(path) != 0:
                continue
            row = {'path': str(path), 'bytes': len(raw), 'sha256': digest(raw), 'topics': topics}
            if (used['copied_files'] < LIMITS['copied_files']
                    and used['copied_bytes'] + len(raw) <= LIMITS['copied_bytes']):
                relative = 'sources/' + row['sha256'][:16] + '-' + path.name
                (output / relative).write_bytes(raw)
                row['bundle_path'] = relative
                used['copied_files'] += 1
                used['copied_bytes'] += len(raw)
            else:
                note(path, 'source copy budget exceeded')
            lines = content.splitlines()
            for topic in topics:
                evidence = report['topic_evidence'][topic]
                budget = LIMITS['excerpt_lines'] - sum(len(e['lines']) for e in evidence)
                if budget <= 0:
                    continue
                hits = [i for i, line in enumerate(lines) if PATTERNS[topic].search(line)]
                selected = set()
                for hit in hits:
                    if len(selected) >= min(100, budget):
                        break
                    selected.update(range(max(0, hit - 3), min(len(lines), hit + 9)))
                excerpt = [{'line': i + 1, 'text': lines[i][:400]} for i in sorted(selected)[:min(100, budget)]]
                evidence.append({'path': str(path), 'sha256': row['sha256'], 'lines': excerpt})
            report['files'].append(row)
        except (OSError, UnicodeError) as error:
            note(path, error)

    report['used'] = used
    report['unresolved'] = [key for key, value in report['topic_evidence'].items() if not value]
    report['next'] = 'RECONCILE_SELECTED_REGISTRATION_BUILD_REDUCTION_AND_60_BYTE_OUTPUT_ABI_BEFORE_KERNEL_BUILD'
    report['implementation_status'] = 'CUSTOM_KERNEL_AND_REGISTRATION_PENDING_SELECTED_DDK_EVIDENCE'
    if report['unresolved']:
        report['status'] = 'PARTIAL'
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    summary = {key: report[key] for key in ('artifact_kind', 'status', 'scope', 'ddk_root',
        'model_compilation', 'device_execution', 'unresolved', 'used', 'next')}
    summary.update(report=str(output / 'report.json'), bundle=None, bundle_sha256=None)
    if make_archive:
        archive = output / 'ddk-context.tar.gz'
        with tarfile.open(archive, 'w:gz') as tar:
            tar.add(output / 'report.json', arcname='report.json')
            tar.add(output / 'sources', arcname='sources')
        summary.update(bundle=str(archive), bundle_sha256=digest(archive.read_bytes()))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def paste_report(report_path, ddk_root=None, max_chars=30000):
    """Render retained excerpts only: no SDK reads, archive or output mutation.

    Give each ABI question its own budget so a large registration file cannot
    crowd out reduction or output-tail declarations. Report every omission.
    """
    if not 6000 <= max_chars <= 60000:
        raise ValueError('paste size must be between 6000 and 60000 characters')
    report_path = Path(report_path)
    with report_path.open('rb') as stream:
        raw = stream.read(8 * 1024 * 1024 + 1)
    if len(raw) > 8 * 1024 * 1024:
        raise ValueError('saved report exceeds 8 MiB')
    report = json.loads(raw)
    if (report.get('artifact_kind') != 'qwen35-custom-top1-ddk-context'
            or report.get('platform') != 'kirinx90'
            or not isinstance(report.get('ddk_root'), str)
            or not isinstance(report.get('topic_evidence'), dict)):
        raise ValueError('not a selected X90 custom Top1 context report')
    if ddk_root and Path(report['ddk_root']).resolve() != Path(ddk_root).resolve():
        raise ValueError('saved report belongs to a different DDK; do not reuse it')
    # Paths are displayed from the retained report. Never follow them here.
    header = {key: report.get(key) for key in ('artifact_kind', 'status', 'platform',
        'ddk_root', 'declared_ddk_id', 'collector_sha256', 'compile_environment', 'unresolved')}
    header.update(presentation='BOUNDED_RETAINED_EXCERPTS_NOT_COMPLETE_SOURCES',
                  saved_report_sha256=digest(raw), sdk_rescan='NOT_RUN',
                  model_compilation='NOT_RUN', device_execution='NOT_RUN')
    # A character limit, not a byte limit; JSON is printed without cutting tokens.
    text = 'BEGIN CUSTOM TOP1 DDK TEXT\n' + json.dumps(header, ensure_ascii=False) + '\n'
    if len(text) > max_chars // 4:
        raise ValueError('saved report metadata too large for the paste budget')
    omitted = {}
    section_budget = (max_chars - len(text) - 2000) // len(TOPICS)
    for topic in TOPICS:
        entries = report['topic_evidence'].get(topic, [])
        if not isinstance(entries, list):
            raise ValueError('invalid retained topic: ' + topic)
        section = '\n[' + topic + ']\n'
        if not entries:
            section += 'NO_RETAINED_EVIDENCE; not a hardware-support conclusion\n'
        shown_files, shown_lines = 0, 0
        # Split the topic budget across up to three sources; at small limits,
        # show one useful excerpt instead of three headers with no code.
        picked = entries[:max(1, min(3, section_budget // 1200))]
        per_file = max(0, (section_budget - len(section) - 120) // max(1, len(picked)))
        for entry in picked:
            chunk = 'FILE ' + json.dumps(entry['path'], ensure_ascii=False) + '\nSHA256 ' + entry['sha256'] + '\n'
            if len(chunk) + 100 > per_file:
                continue
            lines = entry['lines']
            selected = []
            # Keep context around actual topic matches ahead of unrelated text.
            hit_positions = [i for i, line in enumerate(lines) if PATTERNS[topic].search(line['text'])]
            indices = []
            for i in hit_positions:
                indices.extend(range(max(0, i - 2), min(len(lines), i + 7)))
            indices.extend(range(len(lines)))
            count = len(chunk)
            for i in dict.fromkeys(indices):
                line = lines[i]
                rendered = f"{line['line']}: {line['text']}\n"
                if count + len(rendered) + 100 > per_file:
                    continue
                selected.append((i, rendered))
                count += len(rendered)
            chunk += ''.join(line for _, line in sorted(selected))
            chunk += f"[retained lines shown {len(selected)}/{len(lines)}; full source not printed]\n"
            section += chunk
            shown_files += 1
            shown_lines += len(selected)
        total_lines = sum(len(e['lines']) for e in entries)
        omitted[topic] = {'files': len(entries) - shown_files, 'lines': total_lines - shown_lines}
        section += f'[retained files shown {shown_files}/{len(entries)}]\n'
        text += section
    footer = '\nOMITTED_FROM_TEXT ' + json.dumps(omitted) + '\n'
    notes = report.get('notes', [])
    footer += 'COLLECTION_NOTES ' + json.dumps(notes[:4], ensure_ascii=False) + '\n'
    footer += f'[collection notes shown {min(4, len(notes))}/{len(notes)}]\nEND CUSTOM TOP1 DDK TEXT\n'
    if len(text) + len(footer) > max_chars:
        # Metadata/diagnostics can contain arbitrary long filenames; keep the
        # evidence and omission counts, instead of silently truncating JSON.
        footer = '\nOMITTED_FROM_TEXT ' + json.dumps(omitted) + '\nCOLLECTION_NOTES omitted for size\nEND CUSTOM TOP1 DDK TEXT\n'
    if len(text) + len(footer) > max_chars:
        raise ValueError('saved report exceeds bounded rendering budget')
    return text + footer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ddk-root', default=os.environ.get('CPRO_DDK_ROOT'))
    parser.add_argument('--compile-environment')
    parser.add_argument('--source-root', action='append', default=[], help='explicit additional vendor sample/config root')
    parser.add_argument('--output', required=True)
    parser.add_argument('--paste', action='store_true', help='print bounded text; reuse report.json if present, otherwise collect without archive')
    parser.add_argument('--max-chars', type=int, default=30000, help='paste limit, 6000..60000 (default 30000)')
    args = parser.parse_args()
    if not 6000 <= args.max_chars <= 60000:
        parser.error('--max-chars must be between 6000 and 60000')
    saved = Path(args.output) / 'report.json'
    if args.paste and saved.is_file():
        print(paste_report(saved, args.ddk_root, args.max_chars), end='')
        return
    if not args.ddk_root:
        parser.error('set CPRO_DDK_ROOT or pass --ddk-root')
    summary = inspect(args.ddk_root, args.output, args.compile_environment, args.source_root,
                      make_archive=not args.paste)
    if args.paste:
        print(paste_report(saved, args.ddk_root, args.max_chars), end='')
    else:
        print(json.dumps(summary))


if __name__ == '__main__':
    main()
