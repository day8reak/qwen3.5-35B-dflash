"""Cut independent diagnostic prefixes from the original L0 CPro attention AIR."""
import csv
import json
from pathlib import Path
import tempfile

import numpy as np

from ...common.artifacts import output_path, share_for_transfer
from ..layer_probe.common import KIND, checked, identity, metrics, read_array, save_json, sha
from .attention_slice import DTYPES, extract
from .decode_cache_control import graph_message
from .decode_cache_trace import _record
from .gdr_stage_air import stage_edges
from .gdr_stage_data import shapes
from .standalone_gdr import FIELDS

STAGES = ('rms_output', 'silu_output', 'mul_output', 'gated_return', 'out_proj', 'out_proj_only')
SOURCE_CASE = 'l00_c00_gdn_layer_hidden_o0'
BASIS = 'MEASURED_L0_310P_TAIL_ON_SUPPLIED_CORES'


def case_id(stage):
    return f'l00_c00_tail_{stage}_o0'


def tensor(folder, name, array):
    path = folder / (name + '.bin')
    np.ascontiguousarray(array).tofile(path)
    return dict(identity(path, folder), dtype=array.dtype.name, shape=list(array.shape), name=name)


def core_file(path):
    data = np.fromfile(path, dtype='<f2')
    if path.stat().st_size != data.nbytes or data.size not in (19*4096, 64*4096):
        raise ValueError(f'{path}: expected FP16 [1,19 or 64,32,128]')
    rows = data.reshape(-1, 32, 128)
    if not np.isfinite(rows[:19]).all():
        raise ValueError(f'{path}: nonfinite valid input')
    return rows


def physical(rows):
    return np.pad(rows, ((0, 64-len(rows)), (0, 0), (0, 0)))[None]


def _view(op, port, default, views, nodes):
    dims = list(op.output_desc[port].shape.dim)
    if not dims and op.type == 'Reshape':
        producer = nodes[op.input[1].rsplit(':', 1)[0]]
        if producer.type == 'Const':
            value = producer.attr['value'].t
            enums = op.DESCRIPTOR.file.enum_types_by_name['DataType'].values_by_name
            for dtype, code in [('<i4', 'DT_INT32'), ('<i8', 'DT_INT64')]:
                if value.desc.dtype == enums[code].number and value.data:
                    dims = np.frombuffer(value.data, dtype=dtype).tolist()
    if dims.count(-1) == 1 and all(n == -1 or n > 0 for n in dims):
        dims[dims.index(-1)] = int(np.prod(default) // -np.prod(dims))
    if dims and dims not in views:
        raise ValueError(f'{op.name}: unexpected layout {dims}, expected one of {views}')
    return dims or list(default)


def cut(raw, stage, reference, values, dest, source):
    msg, graph = graph_message(raw, 'ChunkGatedDeltaRule')
    edges = stage_edges(graph, 'norm')
    nodes = {op.name: op for op in graph.op}
    net, = [op for op in graph.op if op.type == 'NetOutput']
    gdr, = [op for op in graph.op if op.type == 'ChunkGatedDeltaRule']
    hidden, = [edge for edge in net.input if not edge.endswith(':-1')]
    project = stage.startswith('out_proj')
    target = hidden if project else edges[stage]
    boundaries = ({edges['gated_return']: 'proj_input'} if stage == 'out_proj_only'
                  else {gdr.name + ':0': 'core', edges['gate']: 'z'})
    required = ({'proj_input'} if stage == 'out_proj_only' else {'z'} if stage == 'silu_output' else
                {'core'} if stage == 'rms_output' else {'core', 'z'})
    ports, input_specs = {}, []
    for edge, role in boundaries.items():
        if role not in required:
            continue
        owner, slot = edge.rsplit(':', 1)
        shape = _view(nodes[owner], int(slot), values[role].shape, shapes('core'), nodes)
        x = values[role].reshape(shape)
        name = 'tail_' + role
        ports[edge] = (name, x.dtype.name, list(x.shape))
        input_specs.append(dict(tensor(dest, name, x), semantic_name=role, fixture_index=len(input_specs)))
    sliced = extract(msg, graph, source, dest, ports, target)
    if sum(n['type'] == 'QuantBatchMatmulV4444' for n in sliced['nodes']) != int(project):
        raise ValueError(f'{stage}: unexpected projection count')
    output = tensor(dest, stage, reference)
    views = [[64, 2048], [1, 64, 2048]] if project else shapes('core')
    dtype = {'0': k for k, v in DTYPES.items() if v == sliced['output_desc']['dtype']}
    return dict(air=sliced['air'], source_inputs=input_specs, golden=output,
                output_contract=dict(stage_views={stage: views}, required_dtypes=dtype),
                source_edge=target, source_data_edges={role: e for e, role in boundaries.items() if role in required},
                weight_evidence=sliced['weight_evidence'], reference_basis=BASIS)


def prepare(run, module_air, core_path, ref_path, gate_path, source, config):
    from .gdn_tail_reference import measured
    run = output_path(run)
    run.mkdir(parents=True, exist_ok=True)
    target = run / 'air'
    if target.exists():
        raise FileExistsError(f'{target} already exists; reuse it for board execution or choose a fresh run')
    _, case, folder, record = _record(module_air, SOURCE_CASE)
    if (case['layer'] != 0 or case['point'] != 'gdn_layer' or
            (record['physical_rows'], record['valid_rows'], record['reference_position']) != (64, 19, 18)):
        raise ValueError('Expected the original independent L0 linear attention T64/valid19 AIR')
    raw_file = checked(folder, record['air'])
    raw = raw_file.read_bytes()
    core_rows, gold_rows, z_rows = (core_file(p) for p in (core_path, ref_path, gate_path))
    core, gold, z = map(physical, (core_rows, gold_rows, z_rows))
    gold[:, 19:] = core[:, 19:]
    with tempfile.TemporaryDirectory(dir=run) as tmp:
        root = Path(tmp) / 'air'
        ref_dir = root / 'reference'
        ref_dir.mkdir(parents=True)
        datasets, info = measured(source, config, core, gold, z, read_array(folder, record['golden']))
        provenance = dict(original_air=identity(raw_file, folder), original_export=case['export'],
                          supplied={role: dict(path=str(p), sha256=sha(p), rows=len(x)) for role, p, x in
                          [('core', core_path, core_rows), ('gold_core', ref_path, gold_rows), ('z', gate_path, z_rows)]},
                          padding='reference core uses same tail as board core; absent rows padded with zero',
                          native_310p=info)
        controls = {}
        for label, tensors in datasets.items():
            controls[label] = {k: tensor(ref_dir, label+'_'+k, v) for k, v in tensors.items()}
        controls['core'] = {label: tensor(ref_dir, label+'_raw', v) for label, v in [('same', core), ('gold', gold)]}
        save_json(ref_dir/'reference.json', controls)
        cases = []
        for i, stage in enumerate(STAGES):
            name = case_id(stage)
            dest = root/'models'/name
            dest.mkdir(parents=True)
            # The projection-only control uses the exact 310P tensor at out_proj entry.
            key = 'out_proj' if stage == 'out_proj_only' else stage
            label = 'same' if stage == 'out_proj_only' else 'gold'
            values = dict(core=core, z=z.reshape(info['gate_shape']),
                          proj_input=datasets['same']['gated_return'].reshape(64, 4096))
            exported = cut(raw, stage, datasets[label][key], values, dest, folder)
            save_json(dest/'export.json', exported)
            cases.append(dict(id=name, layer=0, call=0, phase='prefill', ordinal=i, kind='gdn_tail_prefix',
                              point=stage, output_name=stage, status='EXPORTED', export=identity(dest/'export.json', root)))
        save_json(root/'manifest.json', dict(artifact_kind=KIND, status='EXPORTED', cases=cases,
                  golden_source=BASIS, reference_model_layers=1, reference=identity(ref_dir/'reference.json', root),
                  scope='L0 GDR tail; independent CPro prefixes and measured 310P; T64/valid19', provenance=provenance))
        root.rename(target)
    share_for_transfer(target)
    print(target, flush=True)


def summarize(run):
    from .segment_trace_compare import board_case
    air = run/'air'
    manifest = json.loads((air/'manifest.json').read_text())
    fixture = json.loads((run/'execution/fixture/manifest.json').read_text())
    if fixture['source_evidence']['manifest_sha256'] != sha(air/'manifest.json'):
        raise ValueError('Board results and tail AIR refer to different batches')
    control_path = checked(air, manifest['reference'])
    controls = json.loads(control_path.read_text())
    def ref(group, key):
        return read_array(control_path.parent, controls[group][key])

    rows, previous = [], {}
    def compare(stage, comparison, actual, golden):
        a, b = (x.reshape(64, -1)[:19] for x in (actual, golden))
        for token in [None, *range(19)]:
            x, y = (a, b) if token is None else (a[token], b[token])
            m = metrics(x, y)
            scope = 'valid19' if token is None else f'token{token}'
            delta = None
            if comparison in ('own_core', 'same_core', '310p_core_change') and stage != 'silu_output':
                old = previous.get((comparison, scope))
                if old is not None and m['relative_l2'] is not None:
                    delta = 100 * (m['relative_l2'] - old)
                previous[comparison, scope] = m['relative_l2']
            rows.append(dict(stage=stage, comparison=comparison, scope=scope, delta_L2_pp=delta,
                             failed_metrics=','.join(m['failed_metrics']),
                             reference_rms=(m['golden_l2']/np.sqrt(y.size) if m['golden_l2'] is not None else None),
                             **{k: m[k] for k in FIELDS}))
            if token in (None, 18):
                l2 = 'undefined' if m['relative_l2'] is None else format(m['relative_l2'], '.4%')
                cos = 'undefined' if m['cosine'] is None else format(m['cosine'], '.9f')
                change = '' if delta is None else f'  delta_L2={delta:+.4f}pp'
                print(f"{stage:14} {comparison:18} {scope:8}  rel_L2={l2}  cos={cos}  "
                      f"error_ratio={m['error_ratio']:.4%}{change}")

    compare('raw_core', 'own_core', ref('core', 'same'), ref('core', 'gold'))
    for scope in ['valid19', *(f'token{i}' for i in range(19))]:
        previous['same_core', scope] = 0.0
        previous['310p_core_change', scope] = previous['own_core', scope]
    for stage in STAGES:
        _, goldens, outputs, _ = board_case(run, case_id(stage))
        key = 'out_proj' if stage == 'out_proj_only' else stage
        same = ref('same', key)
        is_fixed = stage == 'out_proj_only'
        golden = same if is_fixed else ref('gold', key)
        if goldens[stage].tobytes() != golden.tobytes():
            raise ValueError(f'{stage}: native golden differs from the board fixture')
        if is_fixed:
            compare(stage, 'same_proj_input', outputs[stage], same)
        else:
            compare(stage, 'own_core', outputs[stage], golden)
            compare(stage, 'same_core', outputs[stage], same)
            compare(stage, '310p_core_change', same, golden)
    path = run/'accuracy.csv'
    with tempfile.NamedTemporaryFile(dir=run, mode='w', newline='', delete=False) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    Path(stream.name).replace(path)
    for scope in ('valid19', 'token18'):
        increases = [r for r in rows if r['scope'] == scope and r['comparison'] == 'own_core'
                     and r['delta_L2_pp'] is not None and r['delta_L2_pp'] > 0]
        if increases:
            largest = max(increases, key=lambda r: r['delta_L2_pp'])
            print(f"{scope}: largest increase in relative L2 at {largest['stage']} "
                  f"({largest['delta_L2_pp']:+.4f} pp from preceding stage)")
        else:
            print(f'{scope}: no measured positive step in relative L2')
    print(path)
    return all(r['reference_metrics_passed'] for r in rows if r['scope'] in ('valid19','token18')
               and r['comparison'] in ('own_core', 'same_proj_input'))
