#!/usr/bin/env python3
"""L0 GatedDeltaNet operator captures, board replay and directed three-backend matrices."""
import argparse
import hashlib
import csv
from functools import partial
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))
from qwen35_moe_dflash.common.artifacts import output_path, share_for_transfer
from qwen35_moe_dflash.target.prefill_t64.gdn_layer_stages import (
    DESCRIPTION, ENTRY, SHAPES, STAGES, STATE, WHOLE_OUTPUTS, canonical, compare_view,
)

KIND = 'l0-gdn-operator-reference-t64-valid19-v3'
BACKENDS = ('cpu', '310p', 'cpro')
FLOAT_METRICS = ('cosine', 'relative_l2', 'kl_divergence', 'rmse', 'mean_relative_error',
                 'max_abs', 'mean_abs', 'error_ratio')
OUTPUTS = tuple(k for ins, outs in STAGES.values() for k in outs) + ('recurrent_cache',)


def load(path):
    return json.loads(path.read_bytes())


def original(bundle, *, fetch=False, whole=False):
    from qwen35_moe_dflash.target.layer_probe.common import checked
    from decoder4_original import L0_DIR, L0_KIND, ssh_bundle
    from decoder4_prefill import pull_air
    from qwen35_moe_dflash.target.prefill_t64.gdn_layer_air import stages

    if fetch and not bundle.exists():
        pull_air(bundle.parent, bundle, ssh_download=partial(ssh_bundle, kind=L0_KIND))
    meta = load(bundle/'manifest.json')
    if meta['kind'] != L0_KIND or meta.get('layer', 0) != 0:
        raise ValueError(f'Expected the corrected L0 split AIR at {bundle}')
    print('L0 CPro decay:', 'SoftplusPatch' if 'softplus_patch' in meta else 'native Softplus (retained AIR)')
    for row in meta['files']:
        checked(bundle, row)
    raw = (bundle/L0_DIR/'export.air').read_bytes()
    return meta, raw, stages(raw, meta, whole=whole)


def write_arrays(root, backend, arrays, roles):
    from qwen35_moe_dflash.target.layer_probe.common import identity

    folder = root/backend
    folder.mkdir()
    records = {}
    for role in roles:
        value = canonical(role, arrays[role])
        file = folder/(role + '.bin')
        value.tofile(file)
        records[role] = dict(identity(file, root), shape=list(value.shape), dtype=value.dtype.name)
    return records


def print_fixed(fixed):
    for name, item in fixed.items():
        for role in ('packed_w4', 'group_scale'):
            pairs = item[role]['pairs']
            print(name, role, ' '.join(f"{k}={v['status']}" for k, v in pairs.items()), flush=True)


def prepare(args, run, bundle):
    from qwen35_moe_dflash.target.layer_probe.common import checked, metrics, read_array, save_json, sha
    from trace_segment_attention import observe_l1_gdr_length
    from qwen35_moe_dflash.target.prefill_t64.gdn_layer_reference import capture, evaluate

    from decoder4_original import L0_DIR
    from qwen35_moe_dflash.target.prefill_t64.gdn_linear_operands import fixed_operands

    meta, raw, steps = original(bundle)
    source_hash = sha(bundle/'manifest.json')
    dest = run/'reference'
    if dest.exists():
        existing = load(dest/'manifest.json')
        cfg = (args.source/args.config).resolve(strict=True)
        if (existing['kind'] != KIND or existing['bundle_sha256'] != source_hash
                or existing['capture']['driver_sha256'] != sha(args.source/'inference_segment.py')
                or existing['capture']['config_sha256'] != sha(cfg)):
            raise ValueError(f'{dest} belongs to different source inputs; choose a new --run')
        for backend in existing['arrays'].values():
            for rec in backend.values():
                checked(dest, rec)
        print_fixed(existing['linear_parameters'])
        print('Verified retained reference:', dest)
        return
    native, params, info = capture(args.source.resolve(), args.config, observe_l1_gdr_length)
    fixed = fixed_operands(raw, bundle/L0_DIR, steps, params)
    print_fixed(fixed)
    # Bind the new capture to the independently retained full-layer golden.
    comparisons = {}
    for role, old in [('out', 'out_proj'), ('new_history', 'conv_state_0'), ('recurrent_cache', 'recurrent_state_0')]:
        reference = read_array(bundle, meta['reference']['tensors'][old])
        comparisons[role] = metrics(compare_view(role, native[role].astype(reference.dtype)),
                                    compare_view(role, reference))
    temp = Path(tempfile.mkdtemp(prefix='capture-', dir=run))
    arrays = {'310p': write_arrays(temp, '310p', native, (*ENTRY, *OUTPUTS))}
    arrays['cpu'] = write_arrays(temp, 'cpu', evaluate(native, params), OUTPUTS)
    save_json(temp/'manifest.json', dict(kind=KIND, capture=info, linear_parameters=fixed, bundle_sha256=source_hash,
              cpro_softplus='SoftplusPatch' if 'softplus_patch' in meta else 'native',
              physical_rows=64, valid_rows=19, retained_310p_checks=comparisons, arrays=arrays, stages=steps))
    temp.rename(dest)
    share_for_transfer(dest)
    for role, cmp in comparisons.items():
        print(f"310P vs retained 310P {role}: relative_l2={cmp['relative_l2']}")
    print('Captured reference:', dest)


def ssh_reference(client, remote, staged):
    from decoder4_prefill import ssh_file
    remote = Path(remote)
    staged.mkdir()
    manifest = staged/'manifest.json'
    ssh_file(client, remote/'manifest.json', manifest)
    meta = load(manifest)
    if meta['kind'] != KIND:
        raise ValueError(f'Unexpected reference at {remote}')
    for backend, roles in meta['arrays'].items():
        if backend not in ('310p', 'cpu'):
            raise ValueError(f'Unknown reference backend {backend}')
        (staged/backend).mkdir()
        for k, row in roles.items():
            rel = Path(backend)/(k+'.bin')
            if k not in SHAPES or row['path'] != str(rel):
                raise ValueError(f'Unexpected tensor path {row}')
            ssh_file(client, remote/rel, staged/rel, row)


def run_board(args, run, bundle):
    import numpy as np
    from qwen35_moe_dflash.target.layer_probe.common import read_array, save_json, sha
    from decoder4_prefill import pull_air
    from l0_silu_omcs import DEFAULT_TOOL, compile_stage, run_v2

    root = run/'reference'
    if not root.exists():
        pull_air(run, root, destination_name='reference', ssh_download=ssh_reference)
    ref = load(root/'manifest.json')
    _, raw, steps = original(bundle, fetch=True, whole=args.whole)
    digest = sha(bundle/'manifest.json')
    if ref['kind'] != KIND or ref['bundle_sha256'] != digest or (not args.whole and steps != ref['stages']):
        raise ValueError('Reference and selected L0 AIR or stage plan differ')
    feeds = {k: read_array(root, ref['arrays']['310p'][k]) for k in ENTRY}
    config = args.runtime_config.resolve(strict=True)
    runtime = load(config.parent/load(config)['assets'])['environment']
    os.environ.setdefault('BOARD_HIAI_LIB_DIR', str(Path(DEFAULT_TOOL).parent))
    fixture = sha(root/'manifest.json')
    lane = run/('board-whole' if args.whole else 'board')
    lane.mkdir(exist_ok=True)
    if args.whole:
        print('1 OMC: complete L0 GDN', flush=True)
    for step in steps:
        name, wires = step['name'], step['wires']
        build, model, weights = compile_stage(run, bundle, step, step, raw, args.ddk.resolve(strict=True),
                                               args.diagnosis, digest)
        dest = lane/name
        arrays, records = [], []
        for spec in build['inputs']:
            role = wires[spec['edge']]['name']
            source = feeds[role]
            data = np.ascontiguousarray(source.reshape(spec['shape']), dtype=spec['dtype'])
            if not np.array_equal(source.reshape(-1), data.reshape(-1), equal_nan=True):
                raise ValueError(f'{name}: moving {role} into the OMC would change its values')
            arrays.append(data)
            records.append(dict(role=role, edge=spec['edge'], source_dtype=source.dtype.name,
                                dtype=data.dtype.name, source_sha256=hashlib.sha256(source.tobytes()).hexdigest(),
                                sha256=hashlib.sha256(memoryview(data).cast('B')).hexdigest()))
        provenance = dict(fixture=fixture, model=build['model']['sha256'], inputs=records,
                          runtime=runtime, device=os.environ['SN'], server=os.environ['BIP'])
        if dest.exists():
            state = load(dest/'stage.json')
            if state['source'] != provenance:
                raise ValueError(f'{dest}: different inputs or model; select a fresh run')
            print(f'{name}: verified checkpoint', flush=True)
        else:
            work = Path(tempfile.mkdtemp(prefix=name+'-', dir=lane))
            files, operands = [], []
            for i, (data, spec, record) in enumerate(zip(arrays, build['inputs'], records, strict=True)):
                path = work/f'operand_{i}.bin'
                data.tofile(path)
                files.append(path)
                operands.append(dict(spec, sha256=record['sha256']))
            receipt = run_v2(work, name, model, operands, files, build['outputs'], runtime,
                             external_weights=weights, build_records=(build['model'], build['weights']),
                             fixture_sha256=fixture, release_after_pull=True,
                             slot_config=config if len(operands)>1 or len(build['outputs'])>1 else None)
            outputs = {}
            for s, r in zip(build['outputs'], receipt['outputs'], strict=True):
                role = wires[s['edge']]['name']
                outputs[role] = dict(r, dtype=s['dtype'], shape=s['shape'], edge=s['edge'])
            state = dict(source=provenance, outputs=outputs)
            save_json(work/'stage.json', state)
            work.rename(dest)
        if set(state['outputs']) != {wires[edge]['name'] for edge in step['outputs']}:
            raise ValueError(f'{dest}: incomplete board output inventory')
        for role, record in state['outputs'].items():
            feeds[role] = canonical(role, read_array(dest, record))


def write_report(run, tensor=None, metric='relative_l2', *, whole=False):
    import numpy as np
    from qwen35_moe_dflash.target.layer_probe.common import metrics, read_array, save_json, sha

    root = run/'reference'
    ref = load(root/'manifest.json')
    digest = sha(root/'manifest.json')
    from qwen35_moe_dflash.target.prefill_t64.gdn_linear_operands import dynamic_operands

    fixed = ref.get('linear_parameters', {})
    linear = {k: dict(v) for k, v in fixed.items()}
    if fixed:
        print_fixed(fixed)
    report, missing = {}, []
    lane = run/('board-whole' if whole else 'board')
    lane.mkdir(parents=True, exist_ok=True)
    with (lane/'matrices.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('stage', 'tensor', 'source_variable', 'scope', 'metric', 'actual/reference', *BACKENDS))
        for name in (('whole_gdn',) if whole else STAGES):
            stage = lane/name
            if not (stage/'stage.json').exists():
                missing.append(name)
                continue
            checkpoint = load(stage/'stage.json')
            if checkpoint['source']['fixture'] != digest:
                raise ValueError(f'Checkpoint belongs to another reference or mode: {stage}')
            outputs = {role: record['edge'] for role, record in checkpoint['outputs'].items()}
            if set(outputs) != set(WHOLE_OUTPUTS if whole else STAGES[name][1]):
                raise ValueError(f'{stage}: output roles differ from the selected mode')
            if name in fixed:
                p, = outputs
                linear[name]['valid19'] = dynamic_operands(p, root, ref, stage, checkpoint, fixed[name])
            if 'final_state' in outputs:
                # The split plan proved that the returned FP16 cache is a lossless writeback.
                outputs['recurrent_cache'] = outputs['final_state']
            for k, edge in outputs.items():
                values = {key: compare_view(k, read_array(root, ref['arrays'][key][k]))
                          for key in ('cpu','310p')}
                device_role = k if k in checkpoint['outputs'] else 'final_state'
                values['cpro'] = compare_view(k, read_array(stage, checkpoint['outputs'][device_role]))
                if k == 'recurrent_cache' and values['cpro'].dtype != np.float16:
                    raise ValueError('Expected FP16 CPro recurrent cache')
                scope = 'full_state' if k in STATE else 'valid19'
                table = {}
                integer = values['310p'].dtype.kind in 'iu'
                for b in BACKENDS:
                    table[b] = {}
                    for other in BACKENDS:
                        result = metrics(values[b], values[other])
                        if integer:
                            result['error_ratio'] = 1 - result['exact_elements']/result['total_elements']
                            result['max_abs'] = int(np.abs(values[b].astype('int32') - values[other]).max())
                        table[b][other] = result
                report[k] = dict(stage=name, air_edge=edge, variable=DESCRIPTION[k], scope=scope, scores=table)
                if k == 'recurrent_cache':
                    report[k]['cpro_source'] = ('returned_cache_output' if whole else
                                                'measured_gdr_final_state_after_verified_lossless_writeback')
                for field in (('error_ratio','max_abs') if integer else FLOAT_METRICS):
                    for b in BACKENDS:
                        writer.writerow((name, k, DESCRIPTION[k], scope, field, b,
                                         *(table[b][other].get(field) for other in BACKENDS)))
                cmp = table['cpro']['310p']
                print(f"{k:14} {scope:10} CPro/310P rel_L2={cmp.get('relative_l2')} "
                      f"error_ratio={cmp['error_ratio']} passed={cmp['passed']}")
    failed = [k for k, row in report.items() if not row['scores']['cpro']['310p']['passed']]
    parameter_mismatches = [f'{name}.{role}.{pair}' for name, item in fixed.items()
                            for role in ('packed_w4', 'group_scale') for pair, row in item[role]['pairs'].items()
                            if row['status'] not in ('EXACT', 'FP16_ROUNDING_ONLY')]
    if fixed:
        save_json(lane/'linear_operands.json', dict(reference_sha256=digest,
                  cpro_fixed_source='exported AIR constants used by OMG',
                  cpro_dynamic_source='board stage input files with verified AIR pertoken_scale Cast applied',
                  projections=linear, parameter_mismatches=parameter_mismatches))
        print('Matmul operands:', lane/'linear_operands.json', 'unexplained fixed differences:', parameter_mismatches)
    status = 'FAILED_ACCURACY' if failed else 'PASSED'
    if parameter_mismatches:
        status = 'PARAMETER_MISMATCH'
    if missing:
        status = 'INCOMPLETE'
    save_json(lane/'comparison.json', dict(mode='shared_310p_layer_entry', whole_omc=whole, status=status,
              cpro_softplus=ref.get('cpro_softplus', 'native'),
              reference_sha256=digest, parameter_mismatches=parameter_mismatches,
              failed_cpro_vs_310p=failed, missing=missing, tensors=report))
    print(f'{status}  CPro vs 310P failed at {failed}')
    if tensor in report:
        table = report[tensor]['scores']
        if metric not in table['cpro']['310p']:
            raise ValueError(f'{tensor}: select error_ratio or max_abs for INT8 codes')
        print(f'\n{tensor}  {report[tensor]["scope"]}  {metric}')
        print('actual/reference', *[f'{b:>14}' for b in BACKENDS])
        for b in BACKENDS:
            print(f'{b:16}', *[f'{str(table[b][other][metric]):>14}' for other in BACKENDS])
    print('Matrices:', lane/'matrices.csv', 'pending stages:', missing)
    return not missing  # Execution/report completeness; per-tensor accuracy is recorded above.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'prepare', 'board', 'compare'))
    parser.add_argument('--tensor', choices=OUTPUTS, help='also print this 3×3 matrix after compare')
    parser.add_argument('--whole', action='store_true', help='one complete GDN OMC, with out_proj and both returned states')
    parser.add_argument('--metric', choices=FLOAT_METRICS, default='relative_l2')
    parser.add_argument('--run', type=Path)
    parser.add_argument('--l0-run', type=Path)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--config', type=Path, default=Path('config/qwen3_5_moe.yaml'))
    parser.add_argument('--ddk', type=Path)
    parser.add_argument('--runtime-config', type=Path)
    parser.add_argument('--diagnosis', type=Path)
    args = parser.parse_args()
    if args.whole and args.tensor and args.tensor not in WHOLE_OUTPUTS:
        parser.error('--whole exposes ' + ', '.join(WHOLE_OUTPUTS))
    if args.action == 'plan':
        if args.whole:
            print(f"1 OMC: whole_gdn: {', '.join(ENTRY)} -> {', '.join(WHOLE_OUTPUTS)}")
            for k in WHOLE_OUTPUTS:
                print('  ', k, SHAPES[k], DESCRIPTION[k])
            return True
        for stage, (ins, outs) in STAGES.items():
            print(f"{stage:19} {', '.join(ins)} -> {', '.join(outs)}")
            for k in outs:
                print('  ', k, SHAPES[k], DESCRIPTION[k])
        print('   recurrent_cache', SHAPES['recurrent_cache'], DESCRIPTION['recurrent_cache'])
        print(f'{len(STAGES)} OMCs, {len(OUTPUTS)} comparison tensors')
        return True
    if not args.run:
        parser.error('--run is required')
    run = output_path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    if args.action == 'compare':
        return write_report(run, args.tensor, args.metric, whole=args.whole)
    if not args.l0_run:
        parser.error('--l0-run is required')
    bundle = args.l0_run.resolve()/'air'
    if args.action == 'prepare':
        if args.source is None:
            parser.error('prepare needs --source')
        prepare(args, run, bundle)
        return True
    if not args.ddk or not args.runtime_config:
        parser.error('board needs --ddk and --runtime-config')
    run_board(args, run, bundle)
    if not write_report(run, args.tensor, args.metric, whole=args.whole):
        return False
    from gdn_matrices_txt import render
    render(run / ('board-whole' if args.whole else 'board') / 'matrices.csv')
    return True


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
