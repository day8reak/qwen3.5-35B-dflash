#!/usr/bin/env python3
"""Stage and compare the four independent T64/valid19 attention modules."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))

from qwen35_moe_dflash.common.artifacts import output_path, share_for_transfer
from qwen35_moe_dflash.target.layer_probe.common import KIND, checked, metrics, save_json, sha

CASES = [f'l{i:02d}_c00_gdn_layer_hidden_o0' for i in range(3)] + ['l03_c00_full_attention_layer_hidden_o0']

def load(path):
    return json.loads(path.read_text())


def exports(root):
    doc = load(root / 'manifest.json')
    if not isinstance(doc, dict) or doc.get('artifact_kind') != KIND or doc.get('scope') != 'ISOLATED_WHOLE_ATTENTION_MODULES_PHYSICAL64_VALID19':
        return None
    cases = {c.get('id'): c for c in doc['cases']}
    if doc.get('status') != 'EXPORTED' or len(doc['cases']) != 4 or set(cases) != set(CASES):
        return None
    records = []
    for i, name in enumerate(CASES):
        c = cases[name]
        if (c['kind'], c['phase'], c['call'], c['layer'], c['status']) != ('segment_module_hidden', 'prefill', 0, i, 'EXPORTED'):
            return None
        if c['export']['path'] != f'models/{name}/export.json':
            raise ValueError(f'Unexpected export path for {name}')
        spec = load(checked(root, c['export']))
        if (spec['physical_rows'], spec['valid_rows'], spec['reference_position']) != (64, 19, 18):
            return None
        golden = spec['golden']
        if (golden['name'], golden['dtype'], golden['shape']) != ('golden_hidden', 'float16', [1, 64, 2048]):
            return None
        records.append(spec)
    return records


def verify_air(root):
    specs = exports(root)
    if not specs:
        raise ValueError(f'Expected four independent T64/valid19 golden_hidden cases in {root}')
    for name, spec in zip(CASES, specs):
        folder = root/'models'/name
        for binding in [spec['air'], spec['golden'], *spec['source_inputs']]:
            path = checked(folder, binding)
            if binding.get('semantic_name') in ('effective_length', 'all_length'):
                if int.from_bytes(path.read_bytes(), byteorder='little', signed=True) != 19:
                    raise ValueError(f'{name}: expected a valid length of 19')


def stage(run, source):
    dest = run/'air'
    if dest.exists():
        verify_air(dest)
        if source and (dest/'manifest.json').read_bytes() != (source/'manifest.json').read_bytes():
            raise ValueError('The existing destination differs from --source-air. Choose a new run directory.')
    else:
        if source is None:
            tests = Path(os.environ['AI_RUN_DIR'])/'target/tests'
            candidates = []
            for parent, dirs, files in os.walk(tests):
                root = Path(parent)
                if 'manifest.json' in files and 'models' in dirs and 'compile.json' not in files:
                    try:
                        if exports(root):
                            candidates.append(root)
                    except (FileNotFoundError, ValueError, KeyError) as exc:
                        print(f'Skipping incomplete AIR at {root}: {exc}', file=sys.stderr)
                dirs[:] = [d for d in dirs if d not in ('models', 'omc', 'board', 'execution', 'fixture', 'native',
                                                        'reference', 'recovery', 'weight') and not d.startswith('.')]
            original = tests/'attention-layers-t64-valid19'
            choices = [p for p in candidates if original in p.parents or p == original] or candidates
            if not choices:
                raise ValueError('No four-case AIR on the export host. Re-export using --attention-layers and supply --source-air.')
            if len({(p/'manifest.json').read_bytes() for p in choices}) != 1:
                raise ValueError('Select the original batch with --source-air from: ' + ', '.join(map(str, choices)))
            source = sorted(choices)[0]
        verify_air(source)
        run.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=run) as tmp:
            staged = Path(tmp)/'air'
            for name in CASES:
                shutil.copytree(source/'models'/name, staged/'models'/name)
            shutil.copyfile(source/'manifest.json', staged/'manifest.json')
            verify_air(staged)
            staged.rename(dest)
        print(f'Copied four AIR cases from {source}')
    share_for_transfer(dest)
    print(f'Ready for transfer: {dest}', flush=True)


def signature(case):
    return {direction: {s['semantic_name']: (s.get('golden_dtype', s['dtype']), tuple(s['shape']),
                                              s['fixture_bytes'], s['fixture_sha256']) for s in case[direction]}
            for direction in ('inputs', 'outputs')}


def results(run):
    report = load(run/'recovery/result.json')
    if report.get('device_execution') != 'EXECUTED_OUTPUTS_RETRIEVED':
        raise ValueError(f"{run}: {report.get('error', report.get('status'))}")
    execution = run/'execution'
    request = load(execution/'request.json')
    if request['board_attempt'] != report['board_attempt']:
        raise ValueError(f'{run}: execution differs from recovery')
    path = execution/'fixture/manifest.json'
    if sha(path) != request['fixture_sha256']:
        raise ValueError(f'{run}: the fixture changed after launch')
    manifest = load(path)
    cases = {c['case']: c for c in manifest['cases']}
    return {r['case']: (r, cases[r['case']]) for r in report['cases'] if r.get('output_name') == 'golden_hidden'}


def summary(run):
    from qwen35_moe_dflash.target.prefill_t64.segment_trace_compare import board_case
    new = results(run)
    if set(new) != set(CASES):
        raise ValueError('Expected retrieved golden_hidden for all four cases')
    tests = Path(os.environ['AI_RUN_DIR'])/'target/tests'
    for i, name in enumerate(CASES):
        # The historical v1024 batch contained only L0-L2. Label the independent L3 control separately.
        base = tests/('linear-l0-l2-chunk64-v1024-01' if i < 3 else 'attention-layers-t64-valid19')
        row, case = new[name]
        print(f'L{i}     source             valid19 L2 / cos       token18 L2 / cos      stable  verdict')
        versions = []
        if all((base/file).is_file() for file in ('execution/request.json', 'execution/fixture/manifest.json',
                                                   'recovery/result.json')):
            try:
                old = results(base).get(name)
            except (ValueError, OSError, KeyError) as exc:
                print(f'        Could not verify historical recovery: {exc}')
                old = None
            if old and signature(old[1]) == signature(case):
                versions.append(('v1024' if i < 3 else 'prior full-attn', old[0]))
            else:
                print('        Historical comparison unavailable: missing case or differing inputs/golden.')
        else:
            print(f'        Historical fixture unavailable at {base}')
        for label, record in [*versions, ('Q-scale moved' if i < 3 else 'full-attn control', row)]:
            if not record.get('invocations'):
                raise ValueError(f"{name}: {record.get('error', 'missing board invocation metrics')}")
            m = record['invocations'][0]
            t = m['token_18']
            print(f"        {label:16} {m['relative_l2']:7.3%} / {m['cosine']:.7f}   "
                  f"{t['relative_l2']:7.3%} / {t['cosine']:.7f}   {record['stable']}   {record['status']}")
        if not row['stable']:
            print('        Per-token comparison skipped: the repeated outputs differ.')
            continue
        _, refs, actual, _ = board_case(run, name)
        token_metrics = [metrics(actual['golden_hidden'][0, j], refs['golden_hidden'][0, j]) for j in range(19)]
        print('        New per-token five-metric failures:', [j for j, m in enumerate(token_metrics) if not m['passed']])
    return all(row['stable'] and row['status'] == 'PASSED' for row, _ in new.values())


def board(run, source=None):
    """Compile the staged AIR, then launch or recover the same board request."""
    ddk = Path(os.environ['CPRO_DDK_ROOT']).resolve(strict=True)
    config = Path(os.environ.get('RUNTIME_CONFIG') or Path(os.environ['AI_RUN_DIR'])/'runtime/runtime.json')
    diagnosis = Path(os.environ.get('TBE_DIAGNOSIS') or Path(os.environ['AI_RUN_DIR'])/'common/tbe-diagnosis')
    tools = Path(__file__).resolve().parent
    probe = [sys.executable, str(tools/'target_layer_probe.py'), 'compile']
    validate = [sys.executable, str(tools/'validate_draft_vector.py')]
    registrations = [ddk/'ddk/ascendc/ops/impl/custom/chunk_gated_delta_rule.cpp',
                     ddk/'tools/tools_ascendc/custom_op/ascend310b/src/chunk_gated_delta_rule.cpp']
    verify_air(run/'air')
    digest = sha(registrations[0])
    if sha(registrations[1]) != digest or (source is not None and sha(source) != digest):
        raise ValueError('Installed DDK GDR sources differ from each other or from --gdr-source.')
    print(f'Installed GDR source SHA-256: {digest}', flush=True)
    ddk_id = os.environ.get('CPRO_DDK_ID') or '100.520.020.010'
    provenance = dict(ddk_root=str(ddk), ddk_id=ddk_id, gdr_sha256=digest,
                      manifest_sha256=sha(run/'air/manifest.json'))
    receipt = run/'compiled-gdr.json'
    if not receipt.exists():
        subprocess.run([*probe, '--batch', run/'air', '--output', run/'omc', '--ddk-root', ddk,
                        '--ddk-id', ddk_id, '--compiler-profile', 'isolated-kirinx90',
                        '--tbe-diagnosis', diagnosis, '--use-origin-format'], check=True)
        compiled = load(run/'omc/compile.json')
        if (compiled['manifest_sha256'] != provenance['manifest_sha256'] or set(compiled['cases']) != set(CASES)
                or any(c['status'] != 'COMPILED' for c in compiled['cases'].values())):
            raise ValueError('Incomplete four-case compilation; see omc/compile.json.')
        if any(sha(path) != digest for path in registrations):
            raise ValueError('The shared GDR sources changed during compilation. Use a fresh run.')
        save_json(receipt, dict(provenance, compile_sha256=sha(run/'omc/compile.json')))
    elif load(receipt) != dict(provenance, compile_sha256=sha(run/'omc/compile.json')):
        raise ValueError('Compiled models, AIR or installed GDR identity changed. Select a fresh run.')
    if not (run/'execution/request.json').exists():
        subprocess.run([*validate, '--config', config, '--compiled', run/'omc', '--ddk-root', ddk,
                        '--output', run/'execution', '--apply'], check=True)
    recovery = run/'recovery/result.json'
    if not recovery.exists() or load(recovery).get('stage') != 'complete':
        done = subprocess.run([*validate, '--recover', run/'execution', '--wait', '--output', run/'recovery'])
        if done.returncode and not recovery.is_file():
            done.check_returncode()
    return summary(run)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True, dest='action')
    air = commands.add_parser('stage-air')
    air.add_argument('--source-air', type=Path)
    device = commands.add_parser('board')
    device.add_argument('--gdr-source', type=Path, help='compare both DDK copies with the intended local kernel')
    for cmd in (air, device, commands.add_parser('report')):
        cmd.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'stage-air':
        stage(output_path(args.run), args.source_air)
    elif args.action == 'board':
        sys.exit(0 if board(output_path(args.run), args.gdr_source) else 1)
    else:
        sys.exit(0 if summary(args.run.resolve()) else 1)
