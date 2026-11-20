"""Read the native first-decode cache writes and the CPU-KV graph's attention boundaries."""
import json
import shutil

import numpy as np

from ..layer_probe.attention_reference import reference_math, unpack
from ..layer_probe.common import checked, identity, metrics, read_array, save_json, sha
from .decode_cache_control import CACHE_NAMES, CASE, graph_message, replace_updates
from .segment_export import DECODE_MODULE_BASIS

CPU_KV = 'l03_c01_full_attention_cpu_kv_o0'
CACHE_TRACE = 'l03_c01_decode_cache_trace_o0'
CORE_TRACE = 'l03_c01_decode_core_trace_o0'
STAGE_BASIS = 'FIXED_INPUT_CPU_FIRST_DECODE_STAGE'
VIEWS = {
    'key_update': [[1, 32, 16], [1, 1, 32, 16], [1, 32, 1, 16], [32, 1, 16]],
    'value_update': [[1, 32, 16], [1, 1, 32, 16], [1, 32, 1, 16], [32, 1, 16]],
    'query': [[1, 256, 1, 16], [1, 16, 1, 256], [1, 1, 16, 256]],
    'core': [[1, 256, 1, 16], [1, 16, 1, 256], [1, 1, 16, 256]],
    'proj_input': [[1, 4096], [1, 1, 4096]],
}


def _record(batch, name):
    manifest = json.loads((batch / 'manifest.json').read_text())
    case, = [c for c in manifest['cases'] if c['id'] == name and c['status'] == 'EXPORTED']
    path = checked(batch, case['export'])
    return manifest, case, path.parent, json.loads(path.read_text())


def _add_ports(graph, taps, *, allowed_dtypes=(0, 2)):
    """Extend GE's existing NetOutput without changing the hidden producer or its control edges."""
    nodes = {n.name: n for n in graph.op}
    net, = [n for n in graph.op if n.type == 'NetOutput']
    hidden, = [s for s in net.input if not s.endswith(':-1')]
    controls = [s for s in net.input if s.endswith(':-1')]
    if len(net.input_desc) != 1 or list(graph.output) not in ([], [hidden], [net.name + ':0']):
        raise ValueError('Unexpected single-hidden AIR NetOutput descriptors or GraphDef outputs')
    if net.input_i or net.output_i or net.output_desc:
        raise ValueError('Unexpected NetOutput metadata requiring port remapping')
    for slot, (name, edge) in enumerate(taps.items(), start=1):
        owner, idx = edge.rsplit(':', 1)
        index = int(idx)
        if index < 0 or len(nodes[owner].output_desc) <= index:
            raise ValueError(f'{name}: missing source AIR output descriptor at {edge}')
        desc = net.input_desc.add()
        desc.CopyFrom(nodes[owner].output_desc[index])
        desc.name = f'input{slot}'
        if desc.dtype not in allowed_dtypes:
            raise ValueError(f'{name}: unexpected stage type at {edge}, GE dtype {desc.dtype}')
        if net.input_name:
            net.input_name.append(desc.name)
        if net.is_input_const:
            net.is_input_const.append(False)
        if net.src_name:
            net.src_name.append(owner)
        if net.src_index:
            net.src_index.append(index)
        for attr, field, value in [('_input_name_key', 's', desc.name.encode()),
                                   ('_input_name_value', 'i', slot), ('is_input_const', 'b', False)]:
            if attr in net.attr:
                vector = getattr(net.attr[attr].list, field)
                if len(vector) != slot:
                    raise ValueError(f'Unexpected port count in NetOutput.{attr}')
                vector.append(value)
    if graph.output:
        graph.output.extend(taps.values() if graph.output[0] == hidden else
                            [f'{net.name}:{i}' for i in range(1, 1 + len(taps))])
    del net.input[:]
    net.input.extend([hidden, *taps.values(), *controls])
    return [hidden, *taps.values()]


def _project_input(nodes, hidden):
    def upstream(edge):
        while True:
            owner, port = edge.rsplit(':', 1)
            op = nodes[owner]
            if int(port) != 0 or op.type not in ('Identity', 'TensorMove', 'Reshape', 'Squeeze', 'Unsqueeze',
                                                'Cast', 'TransData', 'Transpose', 'Permute', 'ExpandDims'):
                return op, port
            edge = op.input[0]

    projection, port = upstream(hidden)
    if projection.type != 'QuantBatchMatmulV4444' or port != '0':
        raise ValueError(f'Could not bind hidden to o_proj: {projection.name}:{port}')
    quant, port = upstream(projection.input[0])
    if quant.type != 'DynamicQuant' or port != '0':
        raise ValueError(f'Could not bind o_proj activation to DynamicQuant: {quant.name}:{port}')
    return quant.input[0]


def prepare_trace(baseline, control, cpu, root):
    facts = json.loads((cpu / 'report.json').read_text())
    page, offset = facts['page'], facts['offset']
    if not 0 <= page < 64 or offset != 19 or set(facts['stages']) != {'query', 'core', 'proj_input'}:
        raise ValueError('Expected CPU stage tensors for the first decode at position 19')
    manifest, _, _, original = _record(baseline, CASE)
    cases = []
    for batch, source, trace in [(baseline, CASE, CACHE_TRACE), (control, CPU_KV, CORE_TRACE)]:
        _, case, folder, record = _record(batch, source)
        raw = checked(folder, record['air']).read_bytes()
        checked(folder, record['golden'])
        if (record['reference_basis'] != DECODE_MODULE_BASIS
                or (record['physical_rows'], record['valid_rows'], record['reference_position']) != (1, 1, 19)
                or record['capture_sha256'] != facts['capture_sha256']
                or record['cpu_audit_sha256'] != facts['cpu_audit_sha256']
                or record['golden']['sha256'] != facts['golden_hidden_sha256']):
            raise ValueError('Trace and CPU stages must use the same qualified first decode')
        for s in record['source_inputs']:
            checked(folder, s)
            name = s['semantic_name']
            if name in CACHE_NAMES:
                spec = facts['caches'][name]
                expected = spec['before_sha256'] if source == CASE else spec['sha256']
                if s['sha256'] != expected:
                    raise ValueError(f'{source}: CPU replay uses a different {name} cache')
        msg, graph = graph_message(raw)
        nodes = {n.name: n for n in graph.op}
        fused, = [n for n in graph.op if n.type == 'AdnFusedInferAttention']
        net, = [n for n in graph.op if n.type == 'NetOutput']
        hidden, = [s for s in net.input if not s.endswith(':-1')]
        mask_spec = facts['decode_mask']
        mask = read_array(cpu, mask_spec)
        if mask.shape != (1, 1, 1, 2048) or np.any(mask[..., :20] != 0):
            raise ValueError('Expected visible positions 0 through 19 in the decode mask')
        taps = {name + '_after': fused.input[i] for i, name in enumerate(CACHE_NAMES, 1)}
        if source == CASE:
            _, paths = replace_updates(raw, record['source_inputs'], ())
            for name in CACHE_NAMES:
                taps[name + '_update'] = nodes[paths[name]['update']].input[1]
        else:
            if (record['cache_control']['baseline_air'] != original['air']['sha256']
                    or record['cache_control']['replaced'] != list(CACHE_NAMES)
                    or any(n.type == 'CacheUpdate' for n in graph.op)):
                raise ValueError('Expected the original CPU-KV control AIR')
            taps.update(query=fused.input[0], core=fused.name + ':0', proj_input=_project_input(nodes, hidden))
        edges = _add_ports(graph, taps)
        dest = root / 'models' / trace
        shutil.copytree(folder, dest)
        air = dest / record['air']['path']
        air.write_bytes(msg.SerializeToString())
        goldens = [dict(record['golden'], reference_basis=DECODE_MODULE_BASIS)]
        for name in taps:
            if name in facts['stages']:
                tensor = read_array(cpu, facts['stages'][name])
            else:
                role, where = name.split('_')
                tensor = read_array(cpu, facts['caches'][role])
                if where == 'update':
                    tensor = tensor[page, :, offset].reshape(1, 32, 16)
            path = dest / ('stage_' + name + '.bin')
            tensor.tofile(path)
            goldens.append(dict(identity(path, dest), shape=list(tensor.shape), name=name,
                                dtype=tensor.dtype.name, reference_basis=STAGE_BASIS))
        order = [s['name'] for s in goldens]
        contract = dict(order=order, source_edges=edges, required_dtypes={str(i): 'float16' for i in range(len(order))},
                        first_decode_views={k: VIEWS[k] for k in taps if k in VIEWS})
        mask_path = dest / 'cpu_decode_mask.bin'
        shutil.copyfile(checked(cpu, mask_spec), mask_path)
        meta = dict(source=source, source_air=record['air']['sha256'], page=page, offset=offset,
                    mask=dict(mask_spec, **identity(mask_path, dest)))
        export = dest / 'export.json'
        save_json(export, dict(record, air=identity(air, dest), goldens=goldens, output_contract=contract,
                               decode_trace=meta))
        cases.append(dict(case, id=trace, point=trace.split('_', 2)[2][:-3], kind='decode_attention_trace',
                          output_name='trace', export=identity(export, root)))
    save_json(root / 'manifest.json', dict(manifest, cases=cases, scope='L3_FIRST_DECODE_CACHE_AND_ATTENTION_TRACE'))


def _score(label, actual, reference):
    score = metrics(actual, reference)
    def fmt(x):
        return 'undefined' if x is None else f'{x:.7g}'
    print(f"{label:42s} L2={fmt(score['relative_l2']):>12}  cos={fmt(score['cosine']):>12}  "
          f"{'PASS' if score['passed'] else 'FAIL'}")


def inspect_trace(baseline, control, trace):
    from .segment_trace_compare import board_case

    fixture = trace / 'execution/fixture'
    build = json.loads((trace / 'omc/compile.json').read_text())
    air = trace / 'omc' / build['source_batch']
    evidence = json.loads((fixture / 'manifest.json').read_text())['source_evidence']
    if (evidence['compile_sha256'] != sha(trace / 'omc/compile.json')
            or build['manifest_sha256'] != sha(air / 'manifest.json')):
        raise ValueError('Recovered trace and AIR compilation differ')
    for group, source, name in [(baseline, CASE, CACHE_TRACE), (control, CPU_KV, CORE_TRACE)]:
        _, _, _, original = _record(group / 'air', source)
        _, _, folder, export = _record(air, name)
        meta = export['decode_trace']
        old_inputs, old_refs, old, _ = board_case(group, source)
        inputs, refs, board, _ = board_case(trace, name)
        compiled_source = json.loads((group / 'omc/compile.json').read_text())
        original_fixture = json.loads((group / 'execution/fixture/manifest.json').read_text())
        if (compiled_source['manifest_sha256'] != sha(group / 'air/manifest.json')
                or original_fixture['source_evidence']['compile_sha256'] != sha(group / 'omc/compile.json')):
            raise ValueError(f'{source}: original AIR and recovered execution differ')
        source_inputs = {s['semantic_name']: (s['dtype'], s['shape'], s['sha256']) for s in original['source_inputs']}
        if (meta['source_air'] != original['air']['sha256']
                or any(source_inputs.get(k) != v for table in (old_inputs, inputs) for k, v in table.items())
                or old_refs['golden_hidden'].tobytes() != refs['golden_hidden'].tobytes()):
            raise ValueError(f'{name}: source AIR, runtime inputs or hidden golden changed')
        reproduced = board['golden_hidden'].tobytes() == old['golden_hidden'].tobytes()
        print(f'{source}: hidden equal bits to previous run = {reproduced}')
        _score('hidden vs 310P', board['golden_hidden'], refs['golden_hidden'])
        if not reproduced:
            print('TRACE_CHANGED_HIDDEN: skipping attribution for this graph')
            continue
        page, offset = meta['page'], meta['offset']
        for role in CACHE_NAMES:
            key = role + '_after'
            actual, expected = board[key], refs[key]
            diff = actual.view(np.uint16) != expected.view(np.uint16)
            row = actual[page, :, offset]
            other = int(diff.sum() - diff[page, :, offset].sum())
            history = diff[page, :, :offset]
            print(f'{role}: history[0..18] changed FP16 elements={int(history.sum())}/{history.size}; '
                  f'all slots outside position 19 changed={other}/{diff.size - row.size}')
            if name == CACHE_TRACE:
                update = board[role + '_update'].reshape(row.shape)
                _score(role + ' write operand vs CPU', update, expected[page, :, offset])
                distance = np.count_nonzero(row.view(np.uint16) != update.view(np.uint16))
                print(f'{role}: position 19 vs board write operand changed={distance}/{row.size}')
            else:
                print(f'{role}: CPU-KV input vs attention operand, all changed={int(diff.sum())}/{diff.size}')
        if name == CORE_TRACE:
            for stage in ('query', 'core', 'proj_input'):
                _score(stage + ' vs fixed-input CPU', board[stage], refs[stage])
            mask = read_array(folder, meta['mask'])
            if 'mask' in inputs and inputs['mask'][2] != meta['mask']['sha256']:
                raise ValueError('Board attention mask differs from the retained decode mask')
            # The replay consumes only the first 20 additive mask values, all exact zeros.
            mask = mask[..., :offset + 1].astype(np.float16)
            packed = (1, 256, 1, 16)
            replay = reference_math(board['query'].reshape(packed), board['key_after'], board['value_after'],
                                    mask, np.array([[page]], np.int32), offset + 1, heads=16, kv_heads=2,
                                    block_size=32, scale=0.0625)
            if replay['output'] is None:
                raise ValueError('Same-operand CPU replay has no defined attention output')
            _score('core vs CPU on board Q/K/V', unpack(board['core'].reshape(packed), 16), replay['output'])
