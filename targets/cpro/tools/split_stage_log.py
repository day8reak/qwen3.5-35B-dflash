#!/usr/bin/env python3
"""Collect device logs for retained split-prefill failures without another model run."""
import argparse
from pathlib import Path
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))
from qwen35_moe_dflash.common.artifacts import output_path
from qwen35_moe_dflash.execution.hdc_run import Hdc
from qwen35_moe_dflash.execution.serial_transport import _environment
from qwen35_moe_dflash.target.layer_probe.common import save_json
from qwen35_moe_dflash.target.prefill_t64.chain_hilog import (
    _client_records, _snapshot, _window, load, retained_tool_window,
)


def payload(stage, receipt):
    # run_v2 completes the board's sha256sum -c before starting the V2 tool.
    files = {}
    for line in (stage / 'files.sha256').read_text().splitlines():
        digest, name = line.split('  ', 1)
        if not re.fullmatch('[0-9a-f]{64}', digest) or name in files:
            raise ValueError(f'Invalid files.sha256 entry: {line}')
        files[name] = digest
    inputs = [files.pop(f'input_{i}.bin') for i in range(len(receipt['input_hashes']))]
    model = files.pop('model.omc')
    files.pop('run.sh')
    if inputs != receipt['input_hashes'] or model != receipt['omc_sha256']:
        raise ValueError('Upload manifest differs from V2 receipt')
    return dict(model_sha256=model, input_hashes=inputs, external_weights=files)


def compile_manifest(stage, uploaded):
    parent = stage.parent
    if re.fullmatch(r'L[0-9]{2}', parent.name):
        root = parent.parent.parent
        build = (root / 'silu-omc' if stage.name.endswith('_silu') else root / parent.name / 'omc') / stage.name
    else:
        build = parent.parent / 'omc' / stage.name
    if stage.name == 'decoder_hidden' and load(parent/'chain.json').get('embed_decoder_hidden'):
        build = build.with_name('decoder_hidden-embedded')
    path = build / 'manifest.json'
    if not path.is_file():
        return dict(path=str(path), available=False)
    m = load(path)
    model = Path(m['model']['path'])
    expected = {str(Path(w['path']).relative_to(model.parent)): w['sha256'] for w in m['weights']}
    return dict(path=str(path), available=True, air_sha256=m['source']['air_sha256'], omc_bytes=m['model']['bytes'],
                external_weight_bytes=sum(w['bytes'] for w in m['weights']),
                matches_upload=(m['model']['sha256'] == uploaded['model_sha256']
                                and expected == uploaded['external_weights']))


def service_context(window, tids):
    """Use the client PID on the service's failed Load RPC to select its worker."""
    lines = [line.split(maxsplit=5) for line in window.read_text(errors='replace').splitlines()]
    clients = {str(tid) for tid in tids}
    matches = []
    for i, row in enumerate(lines):
        if len(row) != 6 or '/hiaiserver/' not in row[5]:
            continue
        rpc = re.search(r'GraphExecutorInit.*model loaded fail, Pid:\s*(\d+)\b', row[5])
        if not rpc or rpc[1] not in clients:
            continue
        worker = [r for r in lines[:i + 1] if len(r) == 6 and r[2:4] == row[2:4] and '/hiaiserver/' in r[5]]
        errors = [j for j, r in enumerate(worker) if 'OpUpdateKernelArgs failed' in r[5]]
        # Do not attach an older caller's error on a reused server thread.
        prior_rpc = max((j for j, r in enumerate(worker[:-1]) if 'model loaded fail, Pid:' in r[5]), default=-1)
        relevant = [j for j in errors if j > prior_rpc]
        first = relevant[-1] if relevant else len(worker)-1
        context = [' '.join(r) for r in worker[max(prior_rpc+1, first-16):]]
        matches.append(dict(client=rpc[1], pid=row[2], tid=row[3],
                            op_update_kernel_args_seen=bool(relevant), lines=context))
        print(f'Failure context from hiaiserver {row[2]}/{row[3]} for client {rpc[1]}:', *context, sep='\n')
    return matches


def diagnose(stages, hilog=None):
    env = _environment()
    stages = [output_path(stage) for stage in stages]
    out = Path(tempfile.mkdtemp(dir=stages[0].parent, prefix='load-diagnostics-'))
    print(f'Diagnostics: {out}', flush=True)
    hdc = Hdc(*env, out, 30)
    report, windows = [], []
    for stage in stages:
        folder = out / f'attempt-{len(report) + 1}'
        folder.mkdir()
        row = dict(stage=str(stage), directory=folder.name)
        report.append(row)
        try:
            r = load(stage / 'result.json')
            remote = r['remote']
            if (r['case'] != stage.name or r['status'] != 'FAILED' or not r.get('exitcode')
                    or (r['device'], r['server']) != env[1:3]
                    or not re.fullmatch(r'/data/local/tmp/qwen35_step_[a-f0-9]{32}', remote)):
                raise ValueError('Expected a failed split stage from the selected SN and BIP')
            row.update(remote=remote, exitcode=r['exitcode'], payload=payload(stage, r))
            print(f'{stage}: OMC {r["omc_sha256"]}, {len(r["input_hashes"])} inputs, '
                  f'{len(row["payload"]["external_weights"])} external weight files', flush=True)
            try:
                row['compile'] = compile_manifest(stage, row['payload'])
                print('Compiled model and weight inventory:', row['compile'])
                if (stage / 'model-io.json').is_file():
                    native = load(stage / 'model-io.json')
                    if native.get('model_sha256') == r['omc_sha256']:
                        row['descriptor_probe'] = {k: native.get(k) for k in ('status', 'manager_init')}
                        print('Descriptor probe:', row['descriptor_probe'])
            except (OSError, ValueError, KeyError) as exc:
                row['metadata_error'] = str(exc)
                print(f'{stage}: local metadata unavailable: {exc}', file=sys.stderr)
            start, end, tids = retained_tool_window(hdc, folder.name, remote, stage, folder, expected_exit=r['exitcode'])
            row.update(start=start.isoformat(), end=end.isoformat(), client_tids=tids)
            windows.append((row, folder, start, end, tids))
        except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
            row['error'] = str(exc)
            print(f'{stage}: {exc}', file=sys.stderr)
    if len(report) > 1 and all('payload' in row for row in report):
        same = all(row['payload'] == report[0]['payload'] for row in report[1:])
        print('Identical recorded OMC, ordered inputs and external weights across attempts:', same)
        print('Identical ordered input hashes:',
              all(r['payload']['input_hashes'] == report[0]['payload']['input_hashes'] for r in report))
        if all(r.get('compile', {}).get('matches_upload') for r in report):
            print('Identical compiler input AIR:', len({r['compile']['air_sha256'] for r in report}) == 1)
    try:
        if windows:
            snapshot = _snapshot(hdc, out, hilog)
            for row, folder, start, end, tids in windows:
                window = folder / 'window.log'
                print(f'\nDevice-window candidates (shared across processes): {row["stage"]}', flush=True)
                row['window_records'] = _window(snapshot, window, start, end)
                _client_records(window, folder, tids)
                row['service_failures'] = service_context(window, tids)
    finally:
        save_json(out / 'result.json', report)
        print('Report:', out / 'result.json', flush=True)
    return all(row.get('window_records', 0) > 0 for row in report)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', nargs='+', type=Path, help='retained replay stage directory containing result.json')
    p.add_argument('--hilog', type=Path, help='existing hilog -x snapshot')
    args = p.parse_args()
    sys.exit(0 if diagnose(args.stage, args.hilog) else 1)
