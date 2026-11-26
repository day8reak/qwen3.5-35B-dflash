#!/usr/bin/env python3
"""Collect device logs for retained split-prefill failures without another model run."""
import argparse
import hashlib
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
from qwen35_moe_dflash.target.prefill_t64.whole_run import hiai_failure_stage


def payload(stage, receipt):
    # The host records which compiled files it submitted to HDC.
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
    from decoder4_original import L0_DIR

    layer = stage.parent
    replay = layer.parent
    template = None
    if re.fullmatch(r'L[0-9]{2}', layer.name):
        if replay.name in ('cpu', '310p') and replay.parent.name.startswith('replay-'):
            replay = replay.parent
        run = replay.parent
        family = 'full' if int(layer.name[1:]) % 4 == 3 else 'linear'
        name = re.sub(r'^(key_cache|value_cache)_\d+$', r'\1', stage.name)
        template = run/'templates'/family/name
        base = run/'silu-omc' if stage.name.endswith('_silu') else run/layer.name/'omc'
    else:
        base = layer.parent/'omc'
    names = [stage.name]
    if layer.name == 'L00':
        names.append(stage.name + '-after-embedding')

    variants = [(base/n, 'external') for n in names] + [(base/(n+'-embedded'), 'embedded') for n in names]
    if template is not None:
        variants.append((template, 'template'))
    seen = []
    for folder, kind in variants:
        path = folder/'manifest.json'
        if not path.is_file():
            continue
        seen.append(str(path))
        m = load(path)
        model = Path(m['model']['path'])
        weights = [] if kind == 'template' else m['weights']
        expected = {str(Path(w['path']).relative_to(model.parent)): w['sha256'] for w in weights}
        if m['model']['sha256'] == uploaded['model_sha256'] and expected == uploaded['external_weights']:
            air = folder/'export.air' if kind == 'template' else folder/L0_DIR/'export.air'
            return dict(path=str(path), available=True, matches_upload=True, build_kind=kind, air=str(air),
                        air_sha256=m['source']['air_sha256'], omc_bytes=m['model']['bytes'],
                        external_weight_bytes=sum(w['bytes'] for w in weights))
    return dict(path=str(base/stage.name/'manifest.json'), available=bool(seen), matches_upload=False,
                checked_manifests=seen)


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


def execution_context(folder):
    """Client messages are PID-filtered by _client_records. Service records share the device window."""
    client_log = folder / 'client.log'
    client = client_log.read_text(errors='replace').splitlines() if client_log.is_file() else []
    client = [line for line in client if re.search(r'rtModelExecute failed|graph executor execute return error', line, re.I)]
    service = []
    for line in (folder / 'window.log').read_text(errors='replace').splitlines():
        if '/hiaiserver/' in line and re.search(
                r'load model succ:.*modelId=|ExecuteInfoDump\(|run failed stubName|hwts exception info', line, re.I):
            service.append(line)
    print('Execution errors from matched client:', *client, sep='\n')
    print('Service records in the device window:', *service, sep='\n')
    return dict(client=client, service_window=service)


def stage_operators(build, service_lines):
    from qwen35_moe_dflash.target.prefill_t64.qkv_silu import CONVS
    from qwen35_moe_dflash.target.prefill_t64.decode_cache_control import graph_message
    from qwen35_moe_dflash.target.prefill_t64.decoder_projection import QUANT, weight_sources

    air = Path(build['air'])
    raw = air.read_bytes()
    if hashlib.sha256(raw).hexdigest() != build['air_sha256']:
        raise ValueError(f'{air} differs from the compiler input')
    _, graph = graph_message(raw, 'NetOutput')
    nodes = {op.name: op for op in graph.op}
    stubs = sorted({m[1] for line in service_lines if (m := re.search(r'run failed stubName\s*=\s*([\w.]+)', line))})
    def normalize(kind):
        return re.sub(r'[^a-z0-9]', '', kind.lower())

    candidates = {stub: [op.name for op in graph.op if normalize(op.type) == normalize(stub.split('__kernel')[0])]
                  for stub in stubs}
    for stub, names in candidates.items():
        print(f'Device-window stub {stub}, AIR nodes matching the operator type: {names}')
    rows, convolutions = [], []
    for op in graph.op:
        if op.type in CONVS and len(op.input) > 1:
            convolutions.append(dict(name=op.name, type=op.type, filter_edge=op.input[1],
                                     filter_sources=weight_sources(nodes, op.input[1])))
        if 'matmul' in op.type.lower():
            row = dict(name=op.name, type=op.type)
            if op.type == QUANT:
                row['weight_edge'] = op.input[1]
                row['weight_sources'] = weight_sources(nodes, op.input[1])
                row['weight_files'] = sorted({s['file'] for s in row['weight_sources'] if 'file' in s})
            rows.append(row)
    print(f'Matmul nodes in {air}:', *rows, sep='\n')
    if convolutions:
        print('Convolution filter operands in the compiler AIR:', *convolutions, sep='\n')
    return dict(stub_type_candidates=candidates, matmuls=rows, convolutions=convolutions)


def summarize_execution(row, folder):
    row['failure_stage'] = hiai_failure_stage((folder / 'tool.stdout').read_text(errors='replace'))
    print(f"HIAI failure stage: {row['failure_stage']}")
    if row['failure_stage'] == 'execute':
        row['execution'] = execution_context(folder)
    conv_load = row['failure_stage'] == 'load_model' and Path(row['stage']).name == 'conv_input'
    if conv_load or row['failure_stage'] == 'execute':
        build = compile_manifest(Path(row['stage']), row['payload'])
        row['compile'] = build
        if build.get('matches_upload'):
            print(f"Matched {build['build_kind']} OMC: {build['path']}")
            row['air_operators'] = stage_operators(build, row.get('execution', {}).get('service_window', []))
        else:
            print('No matching compiler manifest for the uploaded model and weights.')


def load_report(replay, expected_kinds=None):
    """Inspect one aligned V2 replay using only its saved receipt, upload list and HiLog."""
    from decoder4_original import ALIGNED_KIND, FULL_KINDS
    from qwen35_moe_dflash.target.layer_probe.common import sha

    replay = replay.resolve(strict=True)
    root = replay.parent
    receipt = load(replay/'result.json')
    manifest = load(root/'omc/manifest.json')
    stdout = (replay/'tool.stdout').read_text(errors='replace')
    if expected_kinds is None:
        expected_kinds = (ALIGNED_KIND, *FULL_KINDS)
    if (receipt.get('case') not in expected_kinds or receipt.get('status') != 'FAILED'
            or not receipt.get('exitcode') or hiai_failure_stage(stdout) != 'load_model'):
        raise ValueError(f'{replay}: expected a failed native decoder model load')
    uploaded = payload(replay, receipt)
    model = manifest['primary']
    folder = Path(model['path']).parent
    weights = {str(Path(w['path']).relative_to(folder)): w['sha256'] for w in manifest['weights']}
    io = load(replay/'io.json')
    if (uploaded['model_sha256'] != model['sha256'] or uploaded['external_weights'] != weights
            or uploaded['input_hashes'] != [r['sha256'] for r in io['inputs']]
            or manifest['air_manifest_sha256'] != sha(root/'air/manifest.json')
            or load(root/'air/manifest.json')['kind'] != receipt['case']):
        raise ValueError(f'{replay}: failed receipt or upload list differs from the compiled model, AIR or inputs')
    print(f'{replay}: {receipt["case"]} load_model, V2 exit={receipt["exitcode"]}; OMC {model["sha256"]}')
    print(f'Manifests and replay receipt agree: {len(io["inputs"])} ordered inputs, {len(weights)} external weight files, '
          f'{sum(row["bytes"] for row in manifest["weights"]):,} external weight bytes.')
    tids = re.findall(r'\bclient_tid=(\d+)', stdout)
    window = replay/'window.log'
    if window.is_file():
        matches = service_context(window, tids)
        if not matches:
            print('No failed service Load RPC matched this client in the saved window.')
    else:
        print(f'No saved HiLog window at {window}')
    return io, receipt, manifest


def four_layer_load_report(replay):
    from decoder4_original import ALIGNED_KIND
    return load_report(replay, (ALIGNED_KIND,))


def saved_report(path):
    """Analyze a retained diagnose result without accessing HDC or changing the evidence."""
    ok = True
    for row in load(path / 'result.json'):
        folder = path / row['directory']
        print(f"{row['stage']} ({folder})")
        try:
            for failure in row.get('service_failures', []):
                print(f"Saved hiaiserver context for client {failure['client']}:", *failure['lines'], sep='\n')
            summarize_execution(row, folder)
        except (OSError, ValueError, KeyError) as exc:
            print(f'{folder}: {exc}', file=sys.stderr)
            ok = False
    return ok


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
                    or not re.fullmatch(r'/data/local/tmp/(?:qwen35_step_[a-f0-9]{32}|qwen35_case_[A-Za-z0-9_-]+/work)', remote)):
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
            if r.get('attempt'):
                _, owner = hdc.shell(f'cat {remote}/.qwen35-hdc-owner', folder.name + '-owner')
                if owner != r['attempt']:
                    raise ValueError(f'{remote}: another attempt has replaced the device logs')
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
                try:
                    summarize_execution(row, folder)
                except (OSError, ValueError, KeyError) as exc:
                    row['summary_error'] = str(exc)
                    print(f'{folder}: {exc}', file=sys.stderr)
    finally:
        save_json(out / 'result.json', report)
        print('Report:', out / 'result.json', flush=True)
    return all(row.get('window_records', 0) > 0 for row in report)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', nargs='*', type=Path, help='retained replay stage directory containing result.json')
    p.add_argument('--hilog', type=Path, help='existing hilog -x snapshot')
    p.add_argument('--saved', type=Path, help='summarize an existing diagnostic directory offline')
    p.add_argument('--load-replay', '--four-layer-replay', type=Path, help='read a saved native decoder load failure offline')
    args = p.parse_args()
    if args.load_replay:
        if args.stage or args.saved or args.hilog:
            p.error('--load-replay takes only the saved replay directory')
        load_report(args.load_replay)
        sys.exit(0)
    if (args.saved and (args.stage or args.hilog)) or not (args.stage or args.saved):
        p.error('select stage directories or --saved DIRECTORY')
    ok = saved_report(args.saved) if args.saved else diagnose(args.stage, args.hilog)
    sys.exit(0 if ok else 1)
