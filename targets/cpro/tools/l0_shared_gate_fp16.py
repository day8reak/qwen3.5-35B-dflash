"""Replay the L0 scalar shared-expert gate with an explicit FP16 MatMul result."""
import numpy as np

from decoder_cases_board import read
from decoder_gate_probe import air_bundle, compile_cases, execute_case
from l0_accuracy_report import (compiled_output_slots, compiled_readouts, get_reference,
                                reference_matches, same_entry_feeds)
from l0_shared_gate import FILENAME, project, rows, weighted
from qwen35_moe_dflash.common.accuracy import floating_metrics
from qwen35_moe_dflash.target.layer_probe.common import save_json
from qwen35_moe_dflash.target.prefill_t64.decode_cache_control import graph_message
from qwen35_moe_dflash.target.silu_patch_air import port_names


def narrow_matmul(raw, edge):
    msg, graph = graph_message(raw, 'NetOutput')
    nodes = {op.name: op for op in graph.op}
    name, slot = edge.rsplit(':', 1)
    old = nodes[name]
    enums = graph.DESCRIPTOR.file.enum_types_by_name['DataType'].values_by_name
    half, single = (enums[k].number for k in ('DT_FLOAT16', 'DT_FLOAT'))
    users = [op for op in graph.op if edge in op.input and op.type != 'NetOutput']
    if (old.type != 'MatMul' or slot != '0' or len(old.input) < 2 or not all(old.input[:2])
            or any(old.input[2:]) or len(old.input_desc) < 2
            or any(d.dtype != single for d in old.input_desc[:2]) or len(old.output_desc) != 1
            or old.output_desc[0].dtype != single or len(users) != 1 or users[0].type != 'Sigmoid'
            or users[0].input[0] != edge or users[0].input_desc[0].dtype != single):
        raise ValueError(f'{edge}: expected the FP32 two-operand scalar MatMul feeding the gate sigmoid')

    def set_half(desc):
        desc.dtype = half
        if '_meta' in desc.attr:
            meta = desc.attr['_meta']
            meta.s = meta.s.replace(b'torch.float32', b'torch.float16')
        if '_fx_tensor_name' in desc.attr:
            del desc.attr['_fx_tensor_name']

    def cast(label, ident, source, desc, to_half):
        op = type(old)(name=label, type='Cast', id=ident)
        op.input.append(source)
        port_names(op, 'x', 'y')
        x, y = op.input_desc.add(), op.output_desc.add()
        x.CopyFrom(desc)
        y.CopyFrom(desc)
        x.name, y.name = 'x', 'y'
        set_half(y if to_half else x)
        # Cast.dst_type uses the GE enum, distinct from the serialized descriptor enum.
        op.attr['dst_type'].i = 1 if to_half else 0
        return op

    start = max(n.id for n in graph.op) + 1
    mm = type(old)()
    mm.CopyFrom(old)  # Keep transpose flags, original shapes and optional bias descriptor.
    mm.name, mm.id = name + '_gate_fp16', start + 2
    seq = []
    for i, suffix in enumerate(('_gate_x_fp16', '_gate_w_fp16')):
        op = cast(name + suffix, start + i, old.input[i], old.input_desc[i], True)
        seq.append(op)
        mm.input[i] = op.name + ':0'
        if mm.src_name:
            mm.src_name[i] = op.name
        if mm.src_index:
            mm.src_index[i] = 0
        set_half(mm.input_desc[i])
    set_half(mm.output_desc[0])
    for key in ('dtype', 'dst_dtype'):
        if key in mm.attr:
            attr = mm.attr[key]
            if attr.HasField('dt') and attr.dt == single:
                attr.dt = half
            elif attr.HasField('i') and attr.i == 0:
                attr.i = 1
    seq.append(mm)
    if any(op.name in nodes for op in seq):
        raise ValueError('Gate FP16 node name already exists in AIR')
    # Reuse the public FP32 edge for the numerical cast, including the NetOutput tap.
    seq.append(cast(name, old.id, mm.name + ':0', old.output_desc[0], False))
    ordered = [item for op in graph.op for item in (seq if op.name == name else [op])]
    del graph.op[:]
    graph.op.extend(ordered)
    print(f'{mm.name} (FP16 × FP16 → FP16) → {name} (Cast → FP32) → {users[0].name}', flush=True)
    return msg.SerializeToString(deterministic=True)


def run(root, remote, action, ddk, config, args):
    parent = root/'board/L00-L03'
    control = parent/'l0-steps-same-input'
    folder = parent/args.gate_probe
    group = 'shared-router'
    baseline = read(control/'compile.json')
    comparison = read(control/'comparison.json')
    bundle = root/'board/package'
    case, meta, source = air_bundle(bundle/'L00-L03', root/'export/package/L00-L03' if action == 'compile' else None)
    if (source != baseline['air'] or source != comparison['source_air'] or not comparison['block_entry_exact']
            or any(g in comparison['missing'] for g in (group, 'attention-output'))):
        raise ValueError('Expected same-input L0 baseline readouts from the current four-layer AIR')
    if action == 'compile':
        compile_cases(folder, case, meta, source, ddk, args.gate_probe)
    build = read(folder/'compile.json')
    row = build['cases'][group]
    if row['status'] != 'COMPILED':
        print(f'{group}: {row["status"]}: {row.get("error", "")}', flush=True)
        return False
    original = baseline['cases'][group]
    if (source != build['air'] or not build['complete']
            or any(row[k] != original[k] for k in ('feeds', 'outputs', 'nodes'))):
        raise ValueError('Baseline and FP16 experiment have different AIR cuts or bindings')
    if action == 'compile':
        return True

    fixture = bundle/'fixture'
    ref = read(fixture/'reference.json')
    ref_path = control/FILENAME
    if not ref_path.is_file():
        ref_path = get_reference(root, remote, folder, FILENAME)
    with np.load(ref_path, allow_pickle=False) as saved:
        if not reference_matches(saved, fixture, ref) or str(saved['checkpoint']) != ref['model_file']:
            raise ValueError('Gate reference and four-layer fixture differ')
        golden = {k: saved[k] for k in ('shared_gate_logit', 'shared_weighted', 'weight')}
    valid = ref['valid_rows']
    feeds = same_entry_feeds(root, case, folder, meta['input_files'], build)
    runtime = read(config.parent/read(config)['assets'])['environment']
    result = execute_case(folder/group, row, runtime, config, real_inputs=feeds,
                          replace_inputs=args.replace_inputs, replace_weights=args.replace_weights,
                          board_case='4-prefill-t64-l0-shared-gate-fp16')
    dest = folder/'result.json'
    result.update(source_air=source, baseline_slice=original['source'], fp16_slice=row['source'], valid=valid)
    save_json(dest, result)
    if result['status'] != 'EXECUTED':
        print(f'{group}: {result["status"]}  {dest}', flush=True)
        return False
    slot, _, spec, path = next(s for s in compiled_output_slots(folder/group, row)
                               if 'shared_gate_logit' in s[1]['aliases'])
    result['gate_output'] = dict(slot=slot, file=str(path), **spec)
    save_json(dest, result)
    print(f'门控输出槽 {slot}: {spec}, {path}')
    if spec['dtype'] != 'float32' or spec['bytes'] != 64 * 4:
        raise ValueError(f'{path}: expected 64 FP32 logits at the output of the numerical cast')
    attention = compiled_readouts(control/'attention-output', baseline['cases']['attention-output'])
    dot = project(rows(attention['post_norm'], 'post_norm', valid), golden['weight'])
    metrics = result['comparisons'] = {}
    for label, base, info in [('原图', control/group, original), ('FP16 MatMul', folder/group, row)]:
        tensors = compiled_readouts(base, info)
        logit, down, contribution = (rows(tensors[k], k, valid) for k in
                                     ('shared_gate_logit', 'shared_down', 'shared_weighted'))
        pairs = [('logit vs 同入口 CPro post_norm 点积', logit, dot),
                 ('logit vs 310P', logit, golden['shared_gate_logit']),
                 ('shared_weighted vs 310P', contribution, golden['shared_weighted']),
                 ('down × sigmoid(logit) vs 实测 shared_weighted', weighted(down, logit), contribution)]
        if base == control/group:
            control_down = down
        else:
            pairs.append(('shared_down vs 原图 CPro', down, control_down))
        metrics[label] = {}
        for key, actual, reference in pairs:
            values = floating_metrics(actual, reference)
            metrics[label][key] = {k: values[k] for k in ('cosine', 'relative_l2', 'max_abs')}
            print(f'{label} | {key}: {metrics[label][key]}')
    save_json(dest, result)
    print(f'{valid} 个有效 token 的门控对照：{dest}')
    return True
