#!/usr/bin/env python3
"""A/B of both SiLU sites inside one complete L0 linear attention, against measured 310P out_proj."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'model'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))

from attention4_qscale import hidden_scores
from qwen35_moe_dflash.common.accuracy import ERROR_RATIO_FIELDS, THRESHOLDS
from qwen35_moe_dflash.common.artifacts import output_path, share_for_transfer
from qwen35_moe_dflash.common.command_trace import subprocess_event
from qwen35_moe_dflash.target.layer_probe.common import checked, identity, read_array, save_json, sha
from qwen35_moe_dflash.target.prefill_t64.attention_cpu_silu import CASE, silu_sites
from qwen35_moe_dflash.target.prefill_t64.attention_slice import source_graph
from qwen35_moe_dflash.target.prefill_t64.segment_export import MODULE_BASIS

VARIANTS = ('original', 'cpu_silu')


def load(path):
    return json.loads(path.read_text())


def fixture(air):
    manifest = load(air/'manifest.json')
    case, = [c for c in manifest['cases'] if c['id'] == CASE]
    folder, spec, _, graph = source_graph(air, 0)
    if (case['kind'] != 'segment_module_hidden' or case['phase'] != 'prefill' or
            manifest['golden_source'] != MODULE_BASIS or spec['reference_basis'] != MODULE_BASIS or
            spec['capture_sha256'] != manifest['capture_sha256'] or spec['quantized_projections'] != 5):
        raise ValueError(f'{air}: expected original 310P-checked whole L0 attention fixture')
    golden = spec['golden']
    if (golden['name'], golden['shape'], golden['dtype']) != ('golden_hidden', [1, 64, 2048], 'float16'):
        raise ValueError('Expected the complete L0 out_proj output')
    read_array(folder, golden)
    inputs = {s['semantic_name']: s for s in spec['source_inputs']}
    for s in inputs.values():
        checked(folder, s)
    if read_array(folder, inputs['effective_length']).tolist() != [19]:
        raise ValueError('Expected effective_length=[19]')
    weights = {}
    for op in graph.op:
        if op.type == 'FileConstant':
            file = (folder / op.attr['file_path'].s.decode()).resolve(strict=True)
            rel = file.relative_to(folder.resolve())
            if str(rel) not in weights:
                weights[str(rel)] = identity(file, folder.resolve())
    if not weights:
        raise ValueError('Original L0 attention AIR is missing its FileConstant weights')
    return dict(manifest, cases=[case], status='EXPORTED', weight_files=weights,
                silu_sites={k: list(v) for k, v in silu_sites(graph).items()})


def stage(source, dest):
    manifest = fixture(source)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
            staged = Path(tmp)/'air'
            shutil.copytree(source/'models'/CASE, staged/'models'/CASE)
            save_json(staged/'manifest.json', manifest)
            if fixture(staged) != manifest:
                raise ValueError('Source files changed during staging')
            staged.rename(dest)
    if load(dest/'manifest.json') != manifest or fixture(dest) != manifest:
        raise ValueError(f'Different or incomplete staged fixture: {dest}. Select a fresh output.')
    share_for_transfer(dest)
    print('78 AIR and measured 310P inputs/golden:', dest, flush=True)
    print('SiLU input/output edges:', manifest['silu_sites'], flush=True)


def board(run, remote, ddk, source, config, diagnosis):
    from qwen35_moe_dflash.target.layer_probe.batch import compile_batch
    from qwen35_moe_dflash.target.prefill_t64.attention_step_v2 import execute
    from pull_artifacts import DEFAULT_CONFIG, connection, pull
    from qwen35_moe_dflash.target.prefill_t64.whole_run import kernel_identity
    run.mkdir(parents=True, exist_ok=True)
    air = run/'air'
    if not air.exists():
        if not remote.is_absolute():
            raise ValueError('AIR_ON_78 must be an absolute directory on the export host')
        with connection(DEFAULT_CONFIG, 'air') as client, client.open_sftp() as sftp:
            sftp.get_channel().settimeout(120)
            pull(sftp, air, remote)
    manifest = load(air/'manifest.json')
    if fixture(air) != manifest:
        raise ValueError(f'Input, AIR or sidecar differs from the staged fixture: {air}')
    gdr = kernel_identity(ddk, source)
    print('Installed GDR SHA-256:', gdr, flush=True)
    stamp = dict(ddk_root=str(ddk), ddk_id=os.environ.get('CPRO_DDK_ID') or '100.520.020.010',
                 source_manifest=sha(air/'manifest.json'), gdr_source_sha256=gdr, source_78=str(remote))
    path = run/'experiment.json'
    if path.exists() and load(path) != stamp:
        raise ValueError(f'Experiment inputs or compiler changed. Select a fresh run: {path}')
    save_json(path, stamp)
    for name in VARIANTS:
        variant = run/name
        record = variant/'omc/compile.json'
        cpu = name == 'cpu_silu'
        if not record.exists():
            print(f'Compiling complete L0 ({name})', flush=True)
            compile_batch(air, variant/'omc', ddk, stamp['ddk_id'], tbe_diagnosis=diagnosis,
                          use_origin_format=True, cases=[CASE], cpu_silu=cpu)
        build = load(record)
        row = build['cases'][CASE]
        if row['status'] != 'COMPILED':
            raise RuntimeError(f"{name}: {row.get('error', row['status'])}; see {record}")
        if (bool(build.get('cpu_silu')) != cpu or
                build['manifest_sha256'] != stamp['source_manifest'] or build['ddk_id'] != stamp['ddk_id'] or
                kernel_identity(ddk, source) != gdr or fixture(air) != manifest):
            raise ValueError(f'{name}: the build differs from the experiment')
        if cpu:
            print('CPUCL placement:', row['cpu_silu_placement']['sites'], flush=True)
        execute(variant, CASE, ddk, config)
    report(run)


def report(run):
    from qwen35_moe_dflash.target.prefill_t64.attention_step_v2 import read_case
    baseline = None
    rows = []
    print('A/B against 310P out_proj; Δ = CPU SiLU minus original, in percentage points', flush=True)
    fields = (*THRESHOLDS, 'mean_abs', 'max_abs', *ERROR_RATIO_FIELDS, 'reference_metrics_passed',
              'failed_metrics', 'nonfinite_actual', 'nonfinite_golden')
    for name in VARIANTS:
        variant = run/name
        v2 = variant/'v2'
        receipt = load(v2/'result.json')
        meta = load(v2/'fixture/manifest.json')
        inputs, gold, actual = read_case(variant, CASE, meta)
        ref, value = gold['golden_hidden'], actual['golden_hidden']
        same = (inputs, ref.tobytes(), receipt['device'], receipt['server'],
                load(variant/'omc/compile.json')['manifest_sha256'])
        if baseline is not None and same != baseline:
            raise ValueError('Cannot compare A/B: input tensors, golden, source batch or device differ')
        if ref.shape != (1, 64, 2048) or value.shape != ref.shape or value.dtype != ref.dtype:
            raise ValueError('Unexpected out_proj ABI')
        scores = hidden_scores(value[0, :19], ref[0, :19])
        if baseline is None:
            baseline, original = same, scores
        for scope, m in scores.items():
            row = dict(variant=name, scope=scope, **{k: m[k] for k in fields})
            row['failed_metrics'] = ','.join(m['failed_metrics'])
            for metric in ('relative_l2', 'error_ratio'):
                before, after = original[scope][metric], m[metric]
                row['delta_' + metric + '_pp'] = (100 * (after - before)
                                                  if before is not None and after is not None else None)
            rows.append(row)
            if scope in ('valid19', 'token18'):
                l2 = 'n/a' if m['relative_l2'] is None else f"{m['relative_l2']:.4%}"
                delta = row['delta_relative_l2_pp']
                change = 'n/a' if delta is None else f'{delta:+.4f}'
                print(f"{name:9} {scope:7} cosine={m['cosine']}  rel_L2={l2}  ΔL2(pp)={change}  "
                      f"RMSE={m['rmse']}  error_ratio={m['error_ratio']:.4%} "
                      f"({m['mismatched_elements']}/{m['total_elements']})  "
                      f"five_metrics_passed={m['reference_metrics_passed']}  "
                      f"error_ratio_passed={m['error_ratio_passed']}", flush=True)
    path = run/'comparison.csv'
    with tempfile.NamedTemporaryFile(mode='w', newline='', dir=run, delete=False) as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    Path(f.name).replace(path)
    print('Per-token and valid19 comparison:', path)


def main():
    if os.getenv('QWEN35_TRACE_COMMANDS') == '1':
        sys.addaudithook(subprocess_event)
    commands = argparse.ArgumentParser(description=__doc__)
    sub = commands.add_subparsers(required=True, dest='action')
    prepare = sub.add_parser('prepare')
    prepare.add_argument('source', type=Path)
    prepare.add_argument('dest', type=Path)
    execute = sub.add_parser('board')
    for name in ('run', 'remote_air', 'ddk', 'gdr_source', 'runtime_config', 'tbe_diagnosis'):
        execute.add_argument(name, type=Path)
    sub.add_parser('report').add_argument('run', type=Path)
    a = commands.parse_args()
    if a.action == 'prepare':
        stage(a.source.resolve(), output_path(a.dest))
    elif a.action == 'board':
        board(output_path(a.run), a.remote_air, a.ddk.resolve(), a.gdr_source.resolve(),
              a.runtime_config.resolve(), a.tbe_diagnosis.resolve())
    else:
        report(output_path(a.run))


if __name__ == '__main__':
    main()
