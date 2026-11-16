#!/usr/bin/env python3
"""Read selected DDK ConvState host functions without loading its libraries."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess


HOST_FUNCTIONS = {
    'desc_check': 'LinearAttentionConvstateUpdateDesc5Check',
    'desc_init': 'LinearAttentionConvstateUpdateDesc8InitAttr',
    'check_support': 'LinearAttentionConvstateUpdateCheckSupport9IsSupport',
    'produce_tiling': 'LinearAttentionConvstateUpdateStrategy13ProduceTiling',
    'build_params': 'BuildLinearAttentionConvstateUpdateParam',
    'offchip_data': 'LinearAttentionConvstateUpdateGenerator20BuildOffChipDataInfo',
    'set_format': 'LinearAttentionConvstateUpdateGraphBuilder20SetFormatAndDataType',
    'output_desc': 'LinearAttentionConvstateUpdateGraphBuilder13GetOutputDesc',
}


def run(*args):
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f'{args[0]} exited {result.returncode}: {result.stderr.strip()}')
    return result.stdout


def inspect_library(path, functions, dynamic, output):
    table = run('nm', *(['-D'] if dynamic else []), '-S', '--defined-only', str(path))
    symbols = []
    for line in table.splitlines():
        fields = line.split()
        if len(fields) == 4 and fields[2] in ('t', 'T'):
            symbols.append((int(fields[0], 16), int(fields[1], 16), fields[3]))
    rows = []
    for label, pattern in functions.items():
        row = dict(function=label, symbol_pattern=pattern)
        try:
            matches = {(start, size, name) for start, size, name in symbols if pattern in name}
            if len(matches) != 1:
                raise ValueError(f'expected one defined function, found {len(matches)}')
            start, size, name = matches.pop()
            if not 0 < size <= 1 << 20:
                raise ValueError(f'unexpected function byte extent: {size}')
            assembly = run('objdump', '-d', '-C', f'--start-address={start:#x}',
                           f'--stop-address={start + size:#x}', str(path))
            # A tool accepting the command without decoding instructions is insufficient.
            if not any(line.lstrip().startswith(f'{start:x}:') for line in assembly.splitlines()):
                raise ValueError('objdump did not decode the requested function start')
            dest = output/f'{label}.txt'
            dest.write_text(f'LIB={path}\nSYMBOL={name}\nSTART={start:#x} SIZE={size:#x}\n\n{assembly}')
            row.update(status='EXTRACTED', symbol=name, start=hex(start), bytes=size, file=dest.name)
        except Exception as exc:
            row.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ddk-root', required=True)
    parser.add_argument('--output', required=True, help='new directory for bounded host disassembly')
    args = parser.parse_args()
    missing = [name for name in ('nm', 'objdump') if shutil.which(name) is None]
    if missing:
        parser.error('required binutils tools not found: ' + ', '.join(missing))
    root = Path(args.ddk_root).resolve()/'tools/tools_omg/master/lib64'
    jobs = [
        (root/'libai_npucore_elementary.so', HOST_FUNCTIONS, False),
        (root/'libai_npucore_elementary_kernel.so',
         {'executor': 'run_executor_linear_attention_convstate_update'}, True),
    ]
    for path, _, _ in jobs:
        if not path.is_file():
            parser.error(f'library not found: {path}')
    output = Path(args.output).resolve()
    if output.exists():
        parser.error(f'output already exists; inspect retained report before reusing: {output}')
    output.mkdir(parents=True)
    report = dict(status='INSPECTED', scope='HOST_ELF_FUNCTIONS_ONLY; NOT_DECODED_DEVICE_KERNEL',
                  model_compilation='NOT_RUN', device_execution='NOT_RUN',
                  numerical_validation='NOT_RUN', libraries=[])
    for path, functions, dynamic in jobs:
        entry = dict(path=str(path), symbol_table='DYNAMIC' if dynamic else 'ORDINARY')
        try:
            entry['elf_header'] = run('objdump', '-f', str(path))
            entry['functions'] = inspect_library(path, functions, dynamic, output)
            entry['status'] = 'EXTRACTED' if all(r['status'] == 'EXTRACTED' for r in entry['functions']) else 'PARTIAL'
        except Exception as exc:
            entry.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        report['libraries'].append(entry)
    if any(r['status'] != 'EXTRACTED' for r in report['libraries']):
        report['status'] = 'PARTIAL'
    (output/'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status=report['status'], report=str(output/'report.json'),
                         scope=report['scope']), indent=2))
    return 0 if report['status'] == 'INSPECTED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
