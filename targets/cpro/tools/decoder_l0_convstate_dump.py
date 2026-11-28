"""Extract L0 ConvState ports from a four-layer replay and check the actual CPro operands."""
import json

import numpy as np

from decoder_gate_probe import air_bundle
from operator_dump_files import calibrate, decoded, index_files
from qwen35_moe_dflash.target.layer_probe.common import metrics, read_tensor, save_json, sha, sized_file
from qwen35_moe_dflash.target.prefill_t64.attention_steps import only, operands
from qwen35_moe_dflash.target.prefill_t64.decode_cache_control import graph_message


SHAPES = dict(mixed_qkv=(1, 64, 8192), old_conv_state=(1, 8192, 4), linear_update_tensor=(128,),
              new_conv_state=(1, 8192, 4), padded=(1, 8192, 67))
# These steps preserve the logical flat order. Never bypass a transpose or dtype-changing cast.
VIEWS = {'Identity', 'TensorMove', 'Reshape', 'View', 'Squeeze', 'ExpandDims', 'Unsqueeze'}


def dump_encoding(value, air_type):
    """Record the parser's dtype; keep floating payloads in the precision actually read."""
    storage = value.dtype.name
    info = dict(air_dtype=air_type, dump_dtype=storage)
    if (storage, air_type) in {('int32', 'DT_INT32'), ('float16', 'DT_FLOAT16'), ('float32', 'DT_FLOAT')}:
        return dict(info, interpretation='declared_dtype')
    if storage == 'float32' and air_type == 'DT_FLOAT16':
        with np.errstate(over='ignore', invalid='ignore'):
            exact = np.isfinite(value) & (value.astype(np.float16).astype(np.float32) == value)
        info.update(fp16_roundtrip_exact=int(exact.sum()), total_elements=value.size)
        if exact.all():
            return dict(info, interpretation='fp32_dump_losslessly_representable_as_fp16')
    return dict(info, interpretation='unresolved')


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


def inspect_attempt(board, attempt):
    from decoder_cases_board import output_order, read
    from qwen35_moe_dflash.target.prefill_t64.decoder_four_boundary import output_shape

    base = board/'L00-L03'
    replay = base/'operator-dump'/attempt/base.name/'replay'
    receipt = read(replay/'result.json')
    dump = receipt.get('operator_dump', {})
    print(f"[L0 ConvState] {replay} | attempt={receipt.get('attempt')} | "
          f"V2={receipt.get('exitcode')} parser={dump.get('parser_exitcode')} "
          f"files={len(dump.get('files', {}))}", flush=True)
    if (receipt.get('status') != 'EXECUTED' or receipt.get('case') != base.name or receipt.get('exitcode') != 0
            or dump.get('status') != 'FILES_RECEIVED_REVIEW_REQUIRED' or dump.get('parser_exitcode') != 0):
        raise ValueError(f"{replay}: incomplete model or dump receipt: {receipt.get('error') or dump.get('status')}")
    bundle = board/'package'
    case = bundle/base.name
    fixture = bundle/'fixture'
    build = read(base/'omc/build.json')
    if (receipt['fixture_sha256'] != sha(bundle/'manifest.json') or build['stamp']['bundle'] != receipt['fixture_sha256']
            or receipt['omc_sha256'] != build['model']['sha256']):
        raise ValueError(f'{replay}: the retained model or reference package differs from this attempt')
    meta, ref = read(case/'case.json'), read(fixture/'reference.json')
    if not meta.get('native_convstate'):
        raise ValueError(f'{case}: the retained AIR must use the native ConvState update')
    golden = dict(ref['outputs'])
    if meta.get('hidden_trace'):
        golden['hidden_layers'] = dict(shape=list(output_shape('hidden_layers', len(meta['layers']))))
    names = output_order(build['outputs'], meta['outputs'], golden)
    if len(receipt['outputs']) != len(names):
        raise ValueError(f'{replay}: incomplete public model outputs')
    slot = names.index('conv_state_0')
    output = dict(receipt['outputs'][slot], dtype=build['outputs'][slot]['dtype'], shape=golden['conv_state_0']['shape'])
    return report_ports(case, replay, fixture, ref, (replay, output))


def report_ports(case, replay, fixture, ref, returned_cache, *, input_overrides=None):
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
        # Search only aliases linked by value-preserving AIR operations. Some
        # compiled graphs emit an intermediate view and omit its neighbours.
        candidate_edges = [edge]
        if name == 'mixed_qkv':
            candidate_edges.extend(view_chain(nodes, edge))
        elif name == 'padded':
            for node in graph.op:
                if node.type == 'Conv2D' and node.input:
                    chain = list(view_chain(nodes, node.input[0]))
                    if edge in chain:
                        candidate_edges.extend(reversed(chain[:chain.index(edge)]))
        candidate_edges = dict.fromkeys(src for src in candidate_edges if air_types[src] == air_types[edge])
        point = report['ports'][name] = dict(edge=edge, candidates=[], issues=[],
                                             indexed_files={src: len(index.get(src, ())) for src in candidate_edges})
        for src in candidate_edges:
            readings, issues = decoded(replay/'dump', index, src, air_types[src], rules)
            for value, sources in readings:
                try:
                    value = shaped(name, value)
                except ValueError as exc:
                    issues.append(dict(files=sources, issue=str(exc)))
                    continue
                encoding = dump_encoding(value, air_types[src])
                if encoding['interpretation'] == 'unresolved':
                    key = f'{name}_unverified_{len(archive)}'
                    archive[key] = value
                    issues.append(dict(files=sources, key=key, encoding=encoding))
                    continue
                key = f'{name}_candidate_{len(point["candidates"])}'
                archive[key] = value
                point['candidates'].append(dict(key=key, files=sources, edge=src, encoding=encoding))
            if issues:
                point['issues'].append(dict(edge=src, details=issues))
            if point['candidates']:
                if len(point['candidates']) == 1 and not issues:
                    values[name] = archive[point['candidates'][0]['key']]
                    point['source'] = 'dump' if src == edge else 'equivalent_edge_dump'
                break
        if name in ('old_conv_state', 'linear_update_tensor'):
            role = 'past_conv_state' if name == 'old_conv_state' else 'conv_state_pos'
            path = view_chain(nodes, edge)
            origin = only((e for e in path if e.endswith(':0') and meta['inputs'].get(e[:-2]) == role),
                          f'{name} replay input')
            root, spec = (input_overrides or {}).get(role, (case, meta['input_files'][role]))
            sent = shaped(name, read_tensor(sized_file(root, spec), spec))
            point.update(feed_edge=origin, feed_file=str(root/spec['path']))
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
            elif edge in cache_path and not point['indexed_files'][edge]:
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
    if fixture is not None:
        compare_references(report, values, public_state, case, fixture, ref, valid)
    else:
        scores = report['same_input']
        if 'mixed_qkv' in values:
            tail = values['mixed_qkv'][:, 60:64].transpose(0, 2, 1)
            scores['returned_state_vs_qkv_60_63'] = metrics(public_state, tail)
            if 'old_conv_state' in values:
                report['tail_differs_from_old_state'] = not np.array_equal(tail, values['old_conv_state'])
        report['basis'] = 'CPro L0 same-run QKV, initial state and public conv_state_0; valid64'
        checks = ('window_full', 'state_vs_physical_tail', 'returned_state_vs_qkv_60_63')
        if (values.keys() == SHAPES.keys() and report.get('tail_differs_from_old_state') and
                report['ports']['linear_update_tensor'].get('matches_sent') is not False):
            exact = all(scores[k]['exact_elements'] == scores[k]['total_elements'] and
                        not (scores[k]['nonfinite_actual'] or scores[k]['nonfinite_golden']) for k in checks)
            report['valid64_accuracy'] = 'PASS' if exact else 'FAIL'
        else:
            report['valid64_accuracy'] = 'INCOMPLETE'
    report['status'] = 'COMPLETE' if values.keys() == SHAPES.keys() else 'INCOMPLETE'
    np.savez(replay/'l0-convstate.npz', **archive)
    save_json(replay/'l0-convstate.json', report)
    print(f'[L0 ConvState] {report["status"]} {replay}/l0-convstate.json', flush=True)
    for name, point in report['ports'].items():
        print(f'  {name} {point["edge"]} {point.get("source", "UNRESOLVED")}', flush=True)
        if point.get('source') in ('dump', 'equivalent_edge_dump'):
            hit, = point['candidates']
            print(f'    {hit["edge"]} {hit["encoding"]}', flush=True)
        if name not in values:
            print(f'    匹配文件数 {point["indexed_files"]}，可选张量 {len(point["candidates"])}，'
                  f'解析问题 {len(point["issues"])}（详见 JSON）', flush=True)
    for label, score in report['same_input'].items():
        print(f'  {label}: 相同元素 {score["exact_elements"]}/{score["total_elements"]}，'
              f'max_abs={score["max_abs"]}, cos={score["cosine"]}', flush=True)
    for label, score in report['reference'].items():
        print(f'  {label}: cosine={score["cosine"]}, max_abs={score["max_abs"]}, '
              f'exact={score["exact_elements"]}/{score["total_elements"]}', flush=True)
    if 'valid64_accuracy' in report:
        print(f'  valid64_accuracy={report["valid64_accuracy"]}', flush=True)
        return report['status'] == 'COMPLETE' and report['valid64_accuracy'] == 'PASS'
    return report['status'] == 'COMPLETE'


def compare_references(report, values, public_state, case, fixture, ref, valid):
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
    reference_states = {}
    for backend, group in [('cpu', 'cpu'), ('310p', 'outputs')]:
        row = ref[group]['conv_state_0']
        reference_states[backend] = shaped('new_conv_state', read_tensor(sized_file(fixture, row), row))
        report['reference']['returned_state_vs_'+backend] = metrics(public_state, reference_states[backend])
    report['reference']['310p_state_vs_cpu_state'] = metrics(reference_states['310p'], reference_states['cpu'])
    # The reference QKV can still localize a difference between the two time windows
    # when the optimized board dump omits the CPro projection. Keep it out of same_input.
    if 'mixed_qkv' in upstream and valid >= 4:
        qkv = shaped('mixed_qkv', upstream['mixed_qkv']).transpose(0, 2, 1)
        for label, tail in [('physical_tail', qkv[..., -4:]), ('valid_tail', qkv[..., valid-4:valid])]:
            report['reference']['310p_state_vs_310p_'+label] = metrics(reference_states['310p'], tail)
            report['reference']['returned_state_vs_310p_'+label] = metrics(public_state, tail)
    if 'mixed_qkv' not in upstream:
        print('  该用例未附可绑定的 310P L0 QKV 参考值。', flush=True)
