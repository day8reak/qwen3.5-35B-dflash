"""Extract L0 ConvState ports from a four-layer replay and check the actual CPro operands."""
import json

import numpy as np

from decoder_gate_probe import air_bundle
from operator_dump_files import calibrate, decoded, index_files
from qwen35_moe_dflash.target.layer_probe.common import metrics, read_tensor, save_json, sized_file
from qwen35_moe_dflash.target.prefill_t64.attention_steps import only, operands
from qwen35_moe_dflash.target.prefill_t64.decode_cache_control import graph_message


SHAPES = dict(mixed_qkv=(1, 64, 8192), old_conv_state=(1, 8192, 4), linear_update_tensor=(128,),
              new_conv_state=(1, 8192, 4), padded=(1, 8192, 67))
# These steps preserve the logical flat order. Never bypass a transpose or dtype-changing cast.
VIEWS = {'Identity', 'TensorMove', 'Reshape', 'View', 'Squeeze', 'ExpandDims', 'Unsqueeze'}


def view_chain(nodes, edge):
    while True:
        yield edge
        name, port = edge.rsplit(':', 1)
        node = nodes[name]
        ins = operands(node)
        if node.type in ('Assign', 'AssignVariableOp') and len(ins) == 2:
            edge = ins[1]
        elif port == '0' and node.type == 'Cast' and ins and node.input_desc and node.output_desc and (
                node.input_desc[0].dtype == node.output_desc[0].dtype):
            edge = ins[0]
        elif port == '0' and node.type in VIEWS and ins:
            edge = ins[0]
        else:
            return


def shaped(name, value):
    shape = SHAPES[name]
    if tuple(d for d in value.shape if d != 1) != tuple(d for d in shape if d != 1):
        raise ValueError(f'{name}: unexpected axes {value.shape}, expected {shape}')
    return value.reshape(shape)


def report_ports(case, replay, fixture, ref, returned_cache):
    source, meta, air = air_bundle(case, case.parents[2]/'export/package'/case.name)
    _, graph = graph_message(sized_file(source, air).read_bytes(), 'LinearAttentionConvstateUpdate')
    nodes = {n.name: n for n in graph.op}
    outputs = {r['name']: r['edge'] for r in meta['outputs']}
    cache_path = set(view_chain(nodes, outputs['conv_state_0']))
    targets = [op for op in graph.op if op.type == 'LinearAttentionConvstateUpdate'
               and (op.name+':0' in cache_path or cache_path.intersection(view_chain(nodes, operands(op)[1])))]
    if len(targets) != 1:
        raise ValueError('Cannot identify a unique ConvState update from the public conv_state_0 edge')
    update = targets[0]
    ins = operands(update)
    if len(ins) != 3 or len(update.output_desc) != 2:
        raise ValueError(f'{update.name}: unexpected operator signature')
    edges = dict(zip(SHAPES, [*ins, update.name+':0', update.name+':1'], strict=True))
    inventory = json.loads((replay/'dump-inventory.json').read_bytes())
    receipt = json.loads((replay/'result.json').read_bytes())
    if (receipt['status'] != 'EXECUTED' or inventory['attempt'] != receipt['attempt']
            or inventory['omc_sha256'] != receipt['omc_sha256'] or inventory['model_exitcode'] or inventory['parser_exitcode']):
        raise ValueError('The parser inventory does not match a successful replay')
    index = index_files(inventory['files'])
    public_dir, public_row = returned_cache
    public_state = shaped('new_conv_state', read_tensor(sized_file(public_dir, public_row), public_row))
    anchors = {outputs['conv_state_0']: public_state}
    if edges['new_conv_state'] in cache_path:
        anchors[edges['new_conv_state']] = public_state
    enum = graph.DESCRIPTOR.file.enum_types_by_name['DataType'].values_by_number
    air_types = {op.name+f':{i}': enum[desc.dtype].name for op in graph.op for i, desc in enumerate(op.output_desc)}
    rules, calibration = calibrate(replay/'dump', index, anchors, air_types)
    valid = ref['valid_rows']
    report = dict(node=update.name, layer=0, attempt=receipt['attempt'], valid_rows=valid,
                  ports={}, calibration=calibration, same_input={}, reference={})
    values, archive = {}, {}
    for name, edge in edges.items():
        edge_read = edge
        readings, issues = decoded(replay/'dump', index, edge, air_types[edge], rules)
        # Some OMCs keep the matmul or the convolution's input reshape under a different edge.
        alternatives = []
        if name == 'mixed_qkv':
            alternatives = list(view_chain(nodes, edge))[1:]
        elif name == 'padded':
            alternatives = [n.input[0] for n in graph.op if n.type == 'Conv2D' and n.input
                            and edge in view_chain(nodes, n.input[0])]
        if not readings and not issues and alternatives:
            for src in alternatives:
                if air_types[src] != air_types[edge]:
                    continue
                readings, issues = decoded(replay/'dump', index, src, air_types[src], rules)
                if readings or issues:
                    edge_read = src
                    break
        point = report['ports'][name] = dict(edge=edge, candidates=[], issues=issues)
        for value, sources in readings:
            expected_kind = 'int32' if name == 'linear_update_tensor' else 'float16'
            try:
                value = shaped(name, value)
                if value.dtype.name != expected_kind:
                    raise ValueError(f'{name}: unresolved dump encoding {value.dtype.name}; expected {expected_kind}')
            except ValueError as exc:
                issues.append(dict(files=sources, issue=str(exc)))
                continue
            key = f'{name}_candidate_{len(point["candidates"])}'
            archive[key] = value
            point['candidates'].append(dict(key=key, files=sources, edge=edge_read))
        candidates = point['candidates']
        if len(candidates) == 1 and not issues:
            values[name] = archive[candidates[0]['key']]
            point['source'] = 'dump'
        if name in ('old_conv_state', 'linear_update_tensor'):
            role = 'past_conv_state' if name == 'old_conv_state' else 'conv_state_pos'
            path = view_chain(nodes, edge)
            origin = only((e for e in path if e.endswith(':0') and meta['inputs'].get(e[:-2]) == role),
                          f'{name} replay input')
            spec = meta['input_files'][role]
            sent = shaped(name, read_tensor(sized_file(case, spec), spec))
            point.update(feed_edge=origin, feed_file=spec['path'])
            archive['sent_'+name] = sent
            if name in values:
                point['matches_sent'] = bool(np.array_equal(values[name], sent))
            if not point.get('matches_sent'):
                # RefData dumps can show the cache after an in-place update. Use its bound initial buffer.
                values[name] = sent
                point['source'] = 'replay_input'
        if name == 'new_conv_state':
            archive['returned_conv_state_0'] = public_state
            if name in values:
                point['matches_returned_state'] = bool(np.array_equal(values[name], public_state))
            elif edge in cache_path and not readings and not issues:
                values[name] = public_state
                point['source'] = 'same_run_public_output'
        if name in values:
            archive[name] = values[name]
            point.update(shape=list(values[name].shape), dtype=values[name].dtype.name)
            if name == 'linear_update_tensor':
                point['values'] = values[name].tolist()

    if {'mixed_qkv', 'old_conv_state'} <= values.keys():
        joined = np.concatenate([values['old_conv_state'], values['mixed_qkv'].transpose(0, 2, 1)], axis=2)
        if 'new_conv_state' in values:
            report['same_input']['state_vs_physical_tail'] = metrics(values['new_conv_state'], joined[..., -4:])
            report['same_input']['state_vs_valid_tail'] = metrics(values['new_conv_state'], joined[..., valid:valid+4])
        if 'padded' in values:
            expected = joined[..., -67:]
            report['same_input']['window_full'] = metrics(values['padded'], expected)
            report['same_input']['window_for_valid_tokens'] = metrics(values['padded'][..., :valid+3], expected[..., :valid+3])
    upstream = {name: read_tensor(sized_file(fixture, row), row)
                for name, row in ref.get('l0_convstate_310p', {}).items()}
    if not upstream:
        for folder in (fixture, case.parents[2]):
            saved = folder/'l0-steps-reference/values.npz'
            if saved.is_file():
                from l0_accuracy_report import reference_matches
                with np.load(saved, allow_pickle=False) as previous:
                    if reference_matches(previous, fixture, ref):
                        upstream = {'mixed_qkv': previous['310p_qkv'], 'padded': previous['310p_window']}
                        report['reference_archive'] = str(saved)
                        break
    for name, golden in upstream.items():
        if name in values:
            observed, golden = values[name], shaped(name, golden)
            if name == 'mixed_qkv':
                observed, golden = observed[:, :valid], golden[:, :valid]
            elif name == 'padded':
                observed, golden = observed[..., :valid+3], golden[..., :valid+3]
            report['reference'][name+'_vs_310p_valid'] = metrics(observed, golden)
    for backend, group in [('cpu', 'cpu'), ('310p', 'outputs')]:
        row = ref[group]['conv_state_0']
        report['reference']['returned_state_vs_'+backend] = metrics(public_state, read_tensor(sized_file(fixture, row), row))
    report['status'] = 'COMPLETE' if values.keys() == SHAPES.keys() else 'INCOMPLETE'
    np.savez(replay/'l0-convstate.npz', **archive)
    save_json(replay/'l0-convstate.json', report)
    print(f'[L0 ConvState] {report["status"]} {replay}/l0-convstate.json', flush=True)
    for name, point in report['ports'].items():
        print(f'  {name} {point["edge"]} {point.get("source", "UNRESOLVED")}', flush=True)
    for label, score in report['same_input'].items():
        print(f'  {label}: 相同元素 {score["exact_elements"]}/{score["total_elements"]}，max_abs={score["max_abs"]}', flush=True)
    return report['status'] == 'COMPLETE'
