#!/usr/bin/env python3
"""Copy only StableArgMax host/kernel sources into an existing allops project."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
FILES = ('op_host/stable_arg_max.cpp', 'op_host/stable_arg_max_tiling.h',
         'op_kernel/stable_arg_max.cpp')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(allops, output, replace=False):
    allops, output = Path(allops).resolve(), Path(output).absolute()
    if output.is_symlink():
        raise ValueError('output must not be a symlink')
    output = output.resolve()
    for root in (allops, ROOT):
        if output == root or output in root.parents or root in output.parents:
            raise ValueError('report/backup output must be separate from source and allops')
    for folder in ('op_host', 'op_kernel'):
        path = allops / folder
        if path.is_symlink() or not path.is_dir():
            raise ValueError('existing allops needs real op_host and op_kernel directories')
    rows = []
    for name in FILES:
        dest = allops / name
        if dest.is_symlink() or (dest.exists() and not dest.is_file()):
            raise ValueError('refusing linked or non-file destination: ' + name)
        before = sha(dest) if dest.exists() else None
        after = sha(ROOT / name)
        if before and before != after and not replace:
            raise FileExistsError('different source exists: ' + name + '; --replace retains a backup')
        rows.append(dict(path=name, before_sha256=before, source_sha256=after))
    output.mkdir(parents=True, exist_ok=False)
    report = dict(artifact_kind='stable-arg-max-allops-sources', status='PREPARING',
        allops=str(allops), files=rows, build='NOT_RUN', installation='NOT_RUN')
    try:
        # Back up all changed files before the first copy. No CMake/other op edits.
        for row in rows:
            if row['before_sha256'] and row['before_sha256'] != row['source_sha256']:
                backup = output / 'previous' / row['path']
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(allops / row['path'], backup)
        for row in rows:
            if row['before_sha256'] != row['source_sha256']:
                shutil.copyfile(ROOT / row['path'], allops / row['path'])
            row['copied_sha256'] = sha(allops / row['path'])
            if row['copied_sha256'] != row['source_sha256']:
                raise ValueError('copied source hash differs: ' + row['path'])
        report.update(status='SOURCES_COPIED', next='Run the existing allops build command for kirinx90.')
    except Exception as error:
        report.update(status='FAILED', error=str(error))
        raise
    finally:
        (output / 'prepare-report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allops-root', required=True)
    parser.add_argument('--output', required=True, help='new report/backup directory outside allops')
    parser.add_argument('--replace', action='store_true', help='back up differing own files before copying')
    args = parser.parse_args()
    print(json.dumps(prepare(args.allops_root, args.output, args.replace)))


if __name__ == '__main__':
    main()
