"""Read an existing aligned-format-host report and locate the two OMG diagnostics."""
import argparse
import hashlib
import json
import mmap
from pathlib import Path
import re


LOOKUP = 'getformatconfigs failed'
EMPTY = 'outputTensorFormats size of node'
SOURCE = 'format_converter_pass.cc'
METHODS = ('FindOpFormatConfig', 'ConnectedNetouputFormatProc')


def digest(path):
    with path.open('rb') as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def source_lines(path):
    lines = path.read_text(errors='replace').splitlines()
    # Print the code leading to each diagnostic, including intervening lookup calls.
    anchors = [i for i, line in enumerate(lines) if LOOKUP in line or EMPTY in line]
    if not anchors:
        anchors = [i for i, line in enumerate(lines) if any('::' + m in line for m in METHODS)]
    selected = set()
    for i in anchors:
        start = next((j for j in range(i, max(-1, i-100), -1)
                      if any('::' + m in lines[j] for m in METHODS)), max(0, i-12))
        selected.update(range(start, min(len(lines), i+14)))
    for i in sorted(selected):
        print(f'{path}:{i+1} {lines[i]}')
    return bool(selected)


def inspect_report(report_path, source_root):
    report = json.loads(report_path.read_bytes())
    probes = report['probes']
    gate = probes.get('gate-mul', {})
    if gate.get('status') != 'EMPTY_OUTPUT_FORMAT':
        raise ValueError('Select a host-check result in which gate-mul reproduced EMPTY_OUTPUT_FORMAT')
    print(f'Host report: {report_path}')
    for name in ('gate-standalone', 'gate-mul'):
        row = probes[name]
        log = Path(row['log'])
        air = log.with_name('export.air')
        if digest(air) != row['air_sha256']:
            raise ValueError(f'AIR changed since compilation: {air}')
        print(f'\n{name} {row["status"]}  AIR={air}  log={log}')
        with log.open(errors='replace') as stream:
            for i, line in enumerate(stream, 1):
                if ('silu_patch' in line and (LOOKUP in line or EMPTY in line)
                        or 'format_converter_pass.cc' in line and 'SingleOpConverter fail' in line):
                    print(f'L{i} {line.rstrip()}')

    for filename, entry in report.get('registrations', {}).items():
        path = Path(filename)
        if not path.is_file() or digest(path) != entry.get('sha256'):
            print(f'AddCustom: {path} unavailable or changed since compilation')
            continue
        peer = json.loads(path.read_bytes()).get('AddCustom')
        ports = {k: v for k, v in (peer or {}).items() if re.fullmatch(r'(input|output)\d+', k)}
        print(f'AddCustom in {path.name}: {ports if peer is not None else "entry absent"}')

    # LD_DEBUG recorded initialization paths for the failing compile's process tree.
    paths = {Path(p) for p in gate['loader']['initialized_libraries']}
    paths.add(Path(report['ddk']) / 'tools/tools_omg/master/omg')
    seen = set()
    matched = 0
    for path in sorted(paths):
        try:
            path = path.resolve(strict=True)
            if path in seen:
                continue
            seen.add(path)
            with path.open('rb') as stream:
                if stream.read(4) != b'\x7fELF':
                    continue
                with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
                    hits = [s for s in (LOOKUP, EMPTY) if data.find(s.encode()) >= 0]
            if hits:
                matched += 1
                print(f'ELF containing log text: {path}  sha256={digest(path)}  strings={hits}')
        except OSError as error:
            print(f'Could not inspect {path}: {error}')
    if not matched:
        print('No matching log strings in the recorded ELF paths and OMG executable.')
    if source_root:
        path = source_root.resolve(strict=True)
        candidates = [path] if path.is_file() else sorted(path.rglob(SOURCE))
        found = False
        for src in candidates:
            if src.name == SOURCE and source_lines(src):
                found = True
        if not found:
            print(f'No matching converter source found under {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path, help='host-check-*/result.json')
    parser.add_argument('--source', type=Path, help='matching DDK source tree or format_converter_pass.cc')
    args = parser.parse_args()
    try:
        inspect_report(args.report.resolve(strict=True), args.source)
    except (KeyError, ValueError, OSError) as error:
        parser.exit(1, f'Cannot inspect this host-check report: {error}\n')
