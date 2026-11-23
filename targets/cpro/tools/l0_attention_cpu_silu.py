#!/usr/bin/env python3
"""A/B of both SiLU sites inside one complete L0 linear attention, against measured 310P out_proj."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import uuid

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'model'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))

from attention4_qscale import hidden_scores
from qwen35_moe_dflash.common.accuracy import ERROR_RATIO_FIELDS, THRESHOLDS
from qwen35_moe_dflash.common.artifacts import output_path, share_for_transfer
from qwen35_moe_dflash.common.command_trace import subprocess_event
from qwen35_moe_dflash.common.tool_logs import log_excerpt
from qwen35_moe_dflash.target.layer_probe.common import checked, identity, read_array, save_json, sha
from qwen35_moe_dflash.target.prefill_t64.attention_cpu_silu import CASE, audit_graphs, silu_sites
from qwen35_moe_dflash.target.prefill_t64.decode_cache_control import graph_message
from qwen35_moe_dflash.target.prefill_t64.attention_slice import source_graph
from qwen35_moe_dflash.target.prefill_t64.segment_export import MODULE_BASIS

VARIANTS = ('original', 'cpu_silu')


def load(path):
    return json.loads(path.read_text())


def fixture(air, *, show_weights=False):
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
    # The module exporter may put parameters directly in Const.value.t.data.
    # checked() in source_graph binds those bytes to export.json via the AIR hash.
    weights = {}
    # Shape literals can also have inline data. Check direct packed projection
    # weights independently so they cannot be mistaken for those literals.
    packed_inputs = {op.input[1] for op in graph.op if op.type == 'QuantBatchMatmulV4444'}
    inline_bytes = 0
    inline_count = 0
    for op in graph.op:
        if op.type in ('Const', 'Constant'):
            node_bytes = 0
            for attr in op.attr.values():
                if attr.HasField('t'):
                    size = len(attr.t.data)
                    if size:
                        node_bytes += size
                        inline_count += 1
            if not node_bytes and op.name + ':0' in packed_inputs:
                raise ValueError(f'{op.name}: missing inline packed projection weight in {folder}')
            inline_bytes += node_bytes
        elif op.type == 'FileConstant':
            rel = Path(op.attr['file_path'].s.decode())
            if rel.is_absolute() or '..' in rel.parts or not rel.parts:
                raise ValueError(f'{op.name}: invalid external weight path {rel}')
            file = (folder / rel).resolve(strict=True)
            rel = file.relative_to(folder.resolve())
            if str(rel) not in weights:
                weights[str(rel)] = identity(file, folder.resolve())
            if not weights[str(rel)]['bytes'] and op.name + ':0' in packed_inputs:
                raise ValueError(f'{op.name}: empty packed projection weight file {file}')
    if not weights and not inline_count:
        raise ValueError(f'{folder}: found neither inline tensor data nor FileConstant weights')
    if show_weights:
        print(f'L0 AIR constants: {inline_count} inline tensors ({inline_bytes:,} bytes), '
              f'{len(weights)} external files; AIR SHA-256={spec["air"]["sha256"]}', flush=True)
    return dict(manifest, cases=[case], status='EXPORTED', weight_files=weights,
                silu_sites={k: list(v) for k, v in silu_sites(graph).items()})


def stage(source, dest):
    manifest = fixture(source, show_weights=True)
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


def audit(run):
    folder, spec, _, original = source_graph(run/'air', 0)
    build = run/'cpu_silu/omc/models'/CASE/'omc'
    receipt = load(build/'air-preparation.json')
    if receipt['source_sha256'] != sha(checked(folder, spec['air'])):
        raise ValueError('Prepared AIR and original fixture have different source hashes')
    prepared = checked(build, receipt['compiler_air'])
    _, candidate = graph_message(prepared.read_bytes(), 'ChunkGatedDeltaRule')
    print(json.dumps(dict(source_sha256=receipt['source_sha256'], prepared_air=str(prepared),
                          prepared_sha256=receipt['compiler_air']['sha256'],
                          **audit_graphs(original, candidate)), indent=2), flush=True)


def board(run, remote, ddk, source, config, diagnosis, *, retry_failed_compile=False, split_silu=False):
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
    if fixture(air, show_weights=True) != manifest:
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
        if cpu and split_silu:
            from qwen35_moe_dflash.target.prefill_t64.silu_split_run import split_run
            return split_run(run, ddk, source, stamp, config, diagnosis)
        if cpu and retry_failed_compile and record.is_file():
            previous = load(record)
            if (previous.get('status') == 'PARTIAL' and previous['cases'][CASE]['status'] == 'FAILED'
                    and not (variant/'v2').exists()):
                archived = variant/('omc-failed-' + uuid.uuid4().hex[:12])
                record.parent.rename(archived)
                print('Previous CPU SiLU compilation retained at', archived, flush=True)
        if not record.exists():
            print(f'Compiling complete L0 ({name})', flush=True)
            compile_batch(air, variant/'omc', ddk, stamp['ddk_id'], tbe_diagnosis=diagnosis,
                          use_origin_format=True, cases=[CASE], cpu_silu=cpu)
        build = load(record)
        row = build['cases'][CASE]
        if row['status'] != 'COMPILED':
            log = record.parent/'models'/CASE/'omc/compile.log'
            raise RuntimeError(f"{name}: {row.get('error', row['status'])}\n{log}\n{log_excerpt(log)}")
        if (bool(build.get('cpu_silu')) != cpu or
                build['manifest_sha256'] != stamp['source_manifest'] or build['ddk_id'] != stamp['ddk_id'] or
                kernel_identity(ddk, source) != gdr or fixture(air) != manifest):
            raise ValueError(f'{name}: the build differs from the experiment')
        if cpu:
            placement = row['cpu_silu_placement']
            if not placement.get('cpu_arithmetic_only_silu'):
                from qwen35_moe_dflash.target.prefill_t64.attention_cpu_silu import inspect
                placement = inspect(checked(record.parent, row['omc']), ddk)
            print('CPUCL SiLU only:', placement['sites'], 'partitions:', placement['graphop_counts'],
                  'boundary casts:', placement['boundary_casts'], flush=True)
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
    mode = execute.add_mutually_exclusive_group()
    mode.add_argument('--split-silu', action='store_true', help='run independent board SiLU OMCs with shared NPU prefix and suffix')
    mode.add_argument('--retry-failed-compile', action='store_true',
                         help='keep a failed CPU SiLU build and retry it in the same experiment')
    sub.add_parser('audit', help='compare the saved original and candidate AIR').add_argument('run', type=Path)
    sub.add_parser('report').add_argument('run', type=Path)
    a = commands.parse_args()
    if a.action == 'prepare':
        stage(a.source.resolve(), output_path(a.dest))
    elif a.action == 'audit':
        audit(a.run.resolve())
    elif a.action == 'board':
        run = output_path(a.run)
        run.mkdir(parents=True, exist_ok=True)
        with (run/'.board.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            board(run, a.remote_air, a.ddk.resolve(), a.gdr_source.resolve(),
                  a.runtime_config.resolve(), a.tbe_diagnosis.resolve(),
                  retry_failed_compile=a.retry_failed_compile, split_silu=a.split_silu)
    else:
        report(output_path(a.run))


if __name__ == '__main__':
    main()
