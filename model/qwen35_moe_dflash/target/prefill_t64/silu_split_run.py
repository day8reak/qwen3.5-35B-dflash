"""Run L0 as prefix -> (native or CPU SiLU) -> shared suffix using compiled wire types."""
from collections import Counter
import csv
from pathlib import Path
import tempfile

import numpy as np

from ...common.accuracy import ERROR_RATIO_FIELDS, THRESHOLDS
from ...common.tool_logs import log_excerpt
from ...draft.ir_inspect import _enum_fields
from ...execution.runtime_config import read_json
from ...execution.serial_transport import _environment
from ...toolchain.existing_air import compile_existing_air
from ...toolchain.omc_container import read_container, read_partition
from ...toolchain.omc_cpu_inspect import _task
from ...toolchain.omc_wire import load_schema
from ..layer_probe.batch import bind_compiled
from ..layer_probe.common import checked, metrics, read_array, save_json, sha
from .attention_cpu_silu import CASE, inspect
from .attention_slice import source_graph
from .attention_step_v2 import read_case, run_v2
from .silu_split import STAGES, VIEWS, cut
from .whole_run import kernel_identity


def compile_stage(work, stage, air, spec, ddk, ddk_id, diagnosis, shapes, *, reuse=None):
    def transform(snapshot):
        if sha(snapshot) != spec['air']['sha256']:
            raise ValueError('Original L0 AIR changed before compilation')
        return cut(snapshot.read_bytes(), stage, spec['source_inputs'], spec['golden'], prefix_shapes=shapes)

    folder = (reuse or work)/stage
    if reuse is None:
        build = compile_existing_air(air, folder, ddk, ddk_id=ddk_id, air_transform=transform,
                                     tbe_diagnosis=diagnosis, use_origin_format=True,
                                     save_weights_as_external_data=False)
    else:
        build = read_json(folder/'report.json')
        if (build['air']['sha256'] != spec['air']['sha256'] or build['ddk_id'] != ddk_id
                or build['ddk_root'] != str(ddk) or build['air_preparation']['stage'] != stage):
            raise ValueError(f'{stage}: saved compile report differs from this experiment: {folder}')
    if build['status'] != 'COMPILED':
        log = folder/'compile.log'
        raise RuntimeError(f'{stage}: {build.get("error")}\n{log}\n{log_excerpt(log)}')
    model = checked(folder, build['omc'])
    meta = build['air_preparation']
    schema = load_schema(ddk/'tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so')
    part, = [p for p in read_container(model)['partitions'] if p['type'] == 0]
    compiled = schema.decode(read_partition(model, part), 'ModelDef')
    graph, = compiled['graph']
    owners = Counter(_task(op)['cl_name'] for op in graph['op'] if op['type'] == 'GraphOp')
    if stage == 'cpu_silu':
        placement = inspect(model, ddk, standalone=True)
    else:
        extra = [(op['name'], op['type']) for op in graph['op']
                 if op['type'] not in ('Data', 'Const', 'Constant', 'GraphOp', 'NetOutput', 'Cast', 'CastT')]
        if not owners or set(owners) != {'NPUCL'} or extra:
            raise ValueError(f'{stage}: expected NPUCL arithmetic; owners={owners}, other ops={extra}')
        placement = dict(graphop_counts=dict(owners))
    outputs = meta['outputs']
    contract = dict(order=[s['name'] for s in outputs], source_edges=meta['output_edges'],
                    required_dtypes={str(i): s['dtype'] for i, s in enumerate(outputs)})
    if stage == 'prefix':
        contract['stage_views'] = {'split_' + k + '_in': [list(shape) for shape in views]
                                   for k, views in VIEWS.items()}
    elif stage in ('suffix', 'suffix_proj_input'):
        contract['stage_views'] = {'golden_hidden': [[64, 2048], [1, 64, 2048], [1, 1, 64, 2048]]}
        if stage == 'suffix_proj_input':
            from .gdr_stage_data import shapes as stage_shapes
            contract['stage_views']['out_proj_input'] = stage_shapes('core') + [[1, 1, 64, 4096]]
    io = bind_compiled(compiled, meta['inputs'], outputs, _enum_fields(schema), contract)
    if {s['name'] for s in io['inputs']} != {s['name'] for s in meta['inputs']}:
        raise ValueError(f'{stage}: compiled input set differs from AIR')
    print(f"{stage}: {dict(owners)}, GDR nodes={meta['gdr_nodes']}, "
          f"outputs={[(s['name'], s['shape'], s['dtype']) for s in io['outputs']]}", flush=True)
    save_json(work/stage/'boundaries.json', dict(io=io, placement=placement, model=str(model),
                                                omc_sha256=build['omc']['sha256']))
    return model, io


def execute_stage(out, name, build, feeds, runtime, config):
    model, io = build
    inputs, files = [], []
    for spec in io['inputs']:
        path, source = feeds[spec['name']]
        if (spec['shape'], spec['dtype']) != (source['shape'], source['dtype']):
            raise ValueError(f"{name}: mismatched wire for {spec['name']}")
        checked(path.parent, dict(source, path=path.name))
        files.append(path)
        inputs.append(dict(spec, sha256=source['sha256']))
    out.mkdir()
    receipt = run_v2(out, name, model, inputs, files, io['outputs'], runtime, slot_config=config)
    return stage_outputs(out, io, receipt)


def stage_outputs(out, io, receipt):
    values = {}
    for spec, saved in zip(io['outputs'], receipt['outputs'], strict=True):
        if saved['bytes'] != spec['bytes']:
            raise ValueError(f"{out}: wrong byte count for {spec['name']}")
        values[spec['name']] = (checked(out, saved), dict(spec, sha256=saved['sha256']))
    return values


def saved_prefix(previous, build, feeds, device):
    model, io = build
    digest = sha(model)
    for folder in previous:
        out = folder/'run-prefix'
        if not (out/'result.json').is_file() or not (out/'slots.json').is_file():
            continue
        receipt = read_json(out/'result.json')
        if (receipt['status'] != 'EXECUTED' or receipt['case'] != 'prefix'
                or (receipt['device'], receipt['server']) != tuple(device[1:3])
                or receipt['omc_sha256'] != digest):
            continue
        slots = read_json(out/'slots.json')['inputs']
        if sorted(s['name'] for s in slots) != sorted(s['name'] for s in io['inputs']):
            continue
        if receipt['input_hashes'] != [feeds[s['name']][1]['sha256'] for s in slots]:
            continue
        try:
            values = stage_outputs(out, io, receipt)
        except (OSError, ValueError) as exc:
            print(f'Skipping incomplete saved prefix at {out}: {exc}', flush=True)
            continue
        print('Reusing board prefix outputs:', out, flush=True)
        return values
    return None


def compare(work, outputs, golden, *, cpu_only=False):
    rows = []
    native = {}
    pairs = [('split_cpu_vs_310p', outputs['cpu_silu'], golden)]
    if not cpu_only:
        pairs = [('whole_vs_310p', outputs['whole'], golden),
                 ('split_native_vs_310p', outputs['native_silu'], golden), *pairs,
                 ('split_native_vs_whole', outputs['native_silu'], outputs['whole']),
                 ('split_cpu_vs_split_native', outputs['cpu_silu'], outputs['native_silu'])]
    for label, actual, reference in pairs:
        if any(x.dtype != np.float16 or x.shape != (1, 64, 2048) for x in (reference, actual)):
            raise ValueError(f'{label}: unexpected whole L0 output')
        a, b = actual[0, :19], reference[0, :19]
        scopes = [('valid19', a, b)] + [(f'token{i}', a[i], b[i]) for i in range(19)]
        for scope, x, g in scopes:
            m = metrics(x, g)
            bits = int(np.count_nonzero(x.view(np.uint16) != g.view(np.uint16)))
            fields = (*THRESHOLDS, 'mean_abs', 'max_abs', *ERROR_RATIO_FIELDS, 'reference_metrics_passed',
                      'failed_metrics', 'nonfinite_actual', 'nonfinite_golden')
            row = dict(comparison=label, scope=scope, changed_fp16_elements=bits, **{k: m[k] for k in fields})
            if label == 'split_native_vs_310p':
                native[scope] = m
            for key in ('relative_l2', 'error_ratio'):
                delta = None
                if label == 'split_cpu_vs_310p' and scope in native and m[key] is not None and native[scope][key] is not None:
                    delta = 100 * (m[key] - native[scope][key])
                row['delta_' + key + '_pp'] = delta
            rows.append(row)
            if scope in ('valid19', 'token18'):
                print(f"{label:26} {scope:7} cosine={m['cosine']} rel_L2={m['relative_l2']} "
                      f"RMSE={m['rmse']} error_ratio={m['error_ratio']} changed_fp16_elements={bits}", flush=True)
                if label == 'split_cpu_vs_310p' and scope in native:
                    print(f"  CPU minus split native (pp): rel_L2={row['delta_relative_l2_pp']} "
                          f"error_ratio={row['delta_error_ratio_pp']}", flush=True)
    with (work/'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print('L0 out_proj results:', work/'comparison.csv', flush=True)


def split_run(run, ddk, source, stamp, config, diagnosis, *, cpu_only=False):
    device = _environment()
    runtime = read_json(config.parent/read_json(config)['assets'])['environment']
    if runtime['ddk_id'] != stamp['ddk_id']:
        raise ValueError('Runtime configuration and whole L0 DDK differ')
    folder, spec, _, _ = source_graph(run/'air', 0)
    feeds = {s['name']: (checked(folder, s), s) for s in spec['source_inputs']}
    golden = read_array(folder, spec['golden'])
    output = {}
    provenance = dict(stamp)
    stages = ('prefix', 'cpu_silu', 'suffix') if cpu_only else STAGES
    if not cpu_only:
        original = run/'original'
        base_inputs, refs, baseline = read_case(original, CASE, read_json(original/'v2/fixture/manifest.json'))
        base_receipt = read_json(original/'v2/result.json')
        if (base_receipt['device'], base_receipt['server']) != tuple(device[1:3]):
            raise ValueError('Split and whole L0 require the same board and HDC server')
        for s in spec['source_inputs']:
            if base_inputs[s['semantic_name']] != (s['dtype'], s['shape'], s['sha256']):
                raise ValueError(f"Whole L0 input differs: {s['semantic_name']}")
        if golden.tobytes() != refs['golden_hidden'].tobytes():
            raise ValueError('Golden does not match the original whole L0 execution fixture')
        provenance['original_v2_sha256'] = sha(original/'v2/result.json')
        output['whole'] = baseline['golden_hidden']
    previous = []
    for path in sorted(run.glob('split-silu-*/source.json'), key=lambda p: p.stat().st_mtime_ns, reverse=True):
        saved = read_json(path)
        if all(saved.get(k) == v for k, v in provenance.items()):
            previous.append(path.parent)
    reuse = next((p for p in previous if all((p/s/'report.json').is_file()
                  and read_json(p/s/'report.json').get('status') == 'COMPILED' for s in stages)), None)
    if reuse:
        print('Reusing compiled split OMCs:', reuse, flush=True)
    work = Path(tempfile.mkdtemp(dir=run, prefix='split-silu-'))
    print('L0 split experiment:', work, flush=True)
    save_json(work/'source.json', provenance)
    builds, shapes = {}, None
    for stage in stages:
        builds[stage] = compile_stage(work, stage, checked(folder, spec['air']), spec, ddk,
                                      stamp['ddk_id'], diagnosis, shapes, reuse=reuse)
        if stage == 'prefix':
            shapes = {site: row['shape'] for site, row in zip(('conv', 'gate'), builds[stage][1]['outputs'], strict=True)}
    if kernel_identity(ddk, source) != stamp['gdr_source_sha256']:
        raise ValueError('Installed GDR changed during compilation')
    cut_inputs = saved_prefix(previous, builds['prefix'], feeds, device) if cpu_only else None
    if cut_inputs is None:
        cut_inputs = execute_stage(work/'run-prefix', 'prefix', builds['prefix'], feeds, runtime, config)
    for middle in stages[1:-1]:
        middle_outputs = execute_stage(work/('run-' + middle), middle, builds[middle], cut_inputs, runtime, config)
        tail = work/('run-suffix-' + middle)
        path, record = execute_stage(tail, 'suffix', builds['suffix'], {**feeds, **middle_outputs}, runtime, config)['golden_hidden']
        output[middle] = read_array(tail, dict(record, path=path.name)).reshape(golden.shape)
    compare(work, output, golden, cpu_only=cpu_only)
