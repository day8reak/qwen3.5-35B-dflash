"""Replay L0's scalar shared-expert gate in host arithmetic using saved valid token rows."""
from pathlib import Path
import tempfile

import numpy as np

from qwen35_moe_dflash.common.artifacts import share_for_transfer
from qwen35_moe_dflash.common.accuracy import floating_metrics
from qwen35_moe_dflash.target.layer_probe.common import save_json, sized_file
from decoder_cases_board import read


FILENAME = 'shared-gate.npz'
WIDTHS = {'post_norm': 2048, 'shared_down': 2048, 'shared_gate_logit': 1, 'shared_weighted': 2048}


def rows(value, role, valid):
    width = WIDTHS[role]
    axes = tuple(n for n in value.shape if n != 1)
    if axes != ((64, width) if width > 1 else (64,)) or value.dtype.kind != 'f':
        raise ValueError(f'{role}: unexpected axes or dtype: {value.shape}, {value.dtype}')
    result = value.reshape(64, width)[:valid].astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f'{role}: nonfinite values in the valid token rows')
    return result


def prepare(root):
    from qwen35_moe_dflash.target.layer_probe.target_weights import read_tensors
    from l0_accuracy_report import reference_matches

    fixture = root/'export/fixture'
    ref = read(fixture/'reference.json')
    checkpoint = Path(ref['model_file'])
    suffix = 'layers.0.mlp.shared_expert_gate.weight'
    weights = read_tensors(checkpoint, {p + suffix for p in ('model.language_model.', 'model.', '')})
    if len(weights) != 1:
        raise ValueError(f'{checkpoint}: expected exactly one {suffix}, found {list(weights)}')
    key, tensor = next(iter(weights.items()))
    if tuple(tensor.shape) != (1, 2048) or not tensor.is_floating_point():
        raise ValueError(f'{key}: unexpected checkpoint tensor {tensor.shape} {tensor.dtype}')
    folder = root/'l0-steps-reference'
    with np.load(folder/'values.npz', allow_pickle=False) as refs:
        if not reference_matches(refs, fixture, ref):
            raise ValueError('L0 reference and four-layer fixture differ')
        valid = int(refs['valid'])
        values = {name: refs[name] for name in ('valid', 'ids', 'fixture_hidden', 'fixture_state')}
        for role in WIDTHS:
            values[role] = rows(refs['310p_' + role], role, valid)
    with tempfile.TemporaryDirectory(dir=folder) as tmp:
        dest = Path(tmp)/FILENAME
        np.savez(dest, **values, checkpoint=str(checkpoint), key=key, weight=tensor.half().numpy())
        dest.replace(folder/FILENAME)
    share_for_transfer(folder/FILENAME)
    print(f'{key} {list(tensor.shape)}，310P 有效行 {valid}，已保存到 {folder/FILENAME}')


def gate_constant(root, groups, source, trace):
    """Inspect only the dependency cone between post_norm and the input of the gate sigmoid."""
    from decoder_gate_probe import air_bundle
    from qwen35_moe_dflash.target.prefill_t64.attention_slice import descriptor, port_info
    from qwen35_moe_dflash.target.prefill_t64.decode_cache_control import graph_message
    from qwen35_moe_dflash.target.prefill_t64.l0_split import slice_dependencies
    from qwen35_moe_dflash.target.prefill_t64.layer_template import parameter

    case, _, record = air_bundle(root/'board/package/L00-L03', root/'export/package/L00-L03')
    if source != record:
        raise ValueError('Retained four-layer AIR differs from the shared gate readout source')
    path = sized_file(case, record)
    _, graph = graph_message(path.read_bytes(), 'NetOutput')
    nodes = {op.name: op for op in graph.op}
    def edge(group, role):
        item, = [o for o in groups[group]['outputs'] if role in o['aliases']]
        return item['edge']
    boundary = edge('attention-output', 'post_norm')
    logit = edge('shared-router', 'shared_gate_logit')
    trace.update(air=str(path), input=boundary, output=logit)
    producer = nodes[logit.rsplit(':', 1)[0]]
    trace['producer_ports'] = dict(node=producer.name, op=producer.type,
                                   inputs=[port_info(graph, d) for d in producer.input_desc],
                                   output=port_info(graph, descriptor(nodes, logit)))
    trace['consumers'] = [dict(node=op.name, op=op.type, slot=i, port=port_info(graph, op.input_desc[i]))
                          for op in graph.op for i, e in enumerate(op.input)
                          if e == logit and i < len(op.input_desc)]
    print(f'AIR 门控输出：{trace["producer_ports"]}，消费端：{trace["consumers"]}')
    kept, used = slice_dependencies(nodes, [logit], {boundary}, 'L0 scalar gate')
    if used != {boundary}:
        raise ValueError(f'{logit} does not consume {boundary}')
    branch = [op for op in graph.op if op.name in kept]
    trace['nodes'] = [dict(name=n.name, op=n.type, inputs=list(n.input)) for n in branch]
    candidates = []
    with tempfile.TemporaryDirectory() as tmp:
        for op in branch:
            if op.type not in ('Const', 'Constant', 'FileConstant'):
                continue
            if op.type == 'FileConstant':
                dims = op.attr['shape'].list.i
            elif 'value' in op.attr:
                dims = op.attr['value'].t.desc.shape.dim
                if not dims and op.output_desc:
                    dims = op.output_desc[0].shape.dim
            else:
                continue
            if [n for n in dims if n != 1] != [2048]:
                continue
            item = parameter(op, {1: 'float32', 2: 'float16'}, path.parent, Path(tmp))
            if item is None:
                continue
            dtype, shape, data = item
            if dtype in ('float16', 'float32'):
                trace.setdefault('vectors', []).append(dict(node=op.name, dtype=dtype, shape=shape))
                candidates.append(np.fromfile(data, dtype=dtype).reshape(2048, 1))
    if len(candidates) != 1:
        raise ValueError(f'Found {len(candidates)} floating vectors of length 2048 on the AIR scalar gate branch')
    return candidates[0]


def project(hidden, weight):
    # Use the reference model's FP16 operands and an FP32 dot product.
    return hidden.astype(np.float16).astype(np.float32) @ weight.astype(np.float32).reshape(2048, 1)


def weighted(down, logit):
    gate = np.exp(-np.logaddexp(0., -logit.astype(np.float64)))
    return down * gate


def report(root, remote):
    from l0_accuracy_report import compiled_output_slots, compiled_readouts, get_reference, reference_matches

    folder = root/'board/L00-L03/l0-steps-same-input'
    control = read(folder/'comparison.json')
    compiled = read(folder/'compile.json')
    if control.get('block_entry_exact') is not True or control['source_air'] != compiled['air']:
        raise ValueError('Expected same-input readouts with all 64 L0 entry rows matched')
    fixture = root/'board/package/fixture'
    ref = read(fixture/'reference.json')
    groups = compiled['cases']
    valid = ref['valid_rows']
    observations = {}
    for group in ('attention-output', 'shared-router'):
        if group in control['missing']:
            raise ValueError(f'{group}: missing readout: {control["missing"][group]}')
        observations.update(compiled_readouts(folder/group, groups[group]))
    cpro = {role: rows(observations[role], role, valid) for role in WIDTHS}
    ref_path = get_reference(root, remote, folder, FILENAME)
    with np.load(ref_path, allow_pickle=False) as saved:
        if not reference_matches(saved, fixture, ref) or str(saved['checkpoint']) != ref['model_file']:
            raise ValueError('Scalar gate weight and 310P rows belong to a different fixture')
        weight = saved['weight']
        nz = {role: saved[role] for role in WIDTHS}
        result = dict(source_air=compiled['air'], valid=valid, checkpoint=str(saved['checkpoint']), key=str(saved['key']))
    stats = result['comparisons'] = {}
    def compare(label, actual, reference):
        values = floating_metrics(actual, reference)
        stats[label] = {k: values.get(k) for k in ('cosine', 'relative_l2', 'max_abs')}
        print(f'{label}: {stats[label]}')
    measured_nz, measured_cpro = nz['shared_gate_logit'], cpro['shared_gate_logit']
    pred_nz = project(nz['post_norm'], weight)
    pred_cpro = project(cpro['post_norm'], weight)
    compare('310P 输入 × checkpoint 权重 vs 310P 实测 logit', pred_nz, measured_nz)
    compare('仅替换 post_norm 对预测 logit 的影响', pred_cpro, pred_nz)
    compare('CPro 输入 × checkpoint 权重 vs CPro 实测 logit', pred_cpro, measured_cpro)
    compare('CPro 实测 logit vs 310P 实测 logit', measured_cpro, measured_nz)
    trace = result['air_gate'] = {}
    try:
        vector = gate_constant(root, groups, compiled['air'], trace)
    except (ValueError, OSError, KeyError) as exc:
        trace['unavailable'] = str(exc)
        print(f'AIR 门控常量未能单独识别：{exc}')
    else:
        trace['different_after_fp16'] = int(np.count_nonzero(vector.ravel().astype(np.float16) != weight.ravel()))
        compare('AIR 门控支路原始常量 vs 参考 FP16 权重', vector.ravel(), weight.ravel())
        print(f'将两份权重统一为 FP16 后数值不同的元素：{trace["different_after_fp16"]}/2048')
        compare('CPro 输入 × AIR 原始常量 vs CPro 实测 logit',
                project(cpro['post_norm'], vector), measured_cpro)
    for backend, data in [('310P', nz), ('CPro', cpro)]:
        compare(f'{backend} down × sigmoid(实测 logit) vs 实测 shared_weighted',
                weighted(data['shared_down'], data['shared_gate_logit']), data['shared_weighted'])
    compare('CPro 实测 shared_weighted vs 310P 实测 shared_weighted', cpro['shared_weighted'], nz['shared_weighted'])
    compare('CPro down × sigmoid(310P logit) vs 310P 实测 shared_weighted',
            weighted(cpro['shared_down'], measured_nz), nz['shared_weighted'])
    result['tokens'] = [dict(token=t, observed_310p=float(measured_nz[t, 0]), predicted_310p=float(pred_nz[t, 0]),
                             observed_cpro=float(measured_cpro[t, 0]), predicted_cpro=float(pred_cpro[t, 0]))
                        for t in range(valid)]
    # Read the returned bytes before any dtype conversion. Only the first valid scalars
    # have a host comparison here; a padded public buffer may contain unrelated trailing bytes.
    (slot, wire, spec, path), = (item for item in compiled_output_slots(folder/'shared-router', groups['shared-router'])
                                 if 'shared_gate_logit' in item[1]['aliases'])
    result['gate_readout'] = dict(slot=slot, air_edge=wire['edge'], omc=spec, file=str(path),
                                  file_bytes=path.stat().st_size)
    print(f'OMC 门控输出槽与原始文件：{result["gate_readout"]}')
    if spec['dtype'] == 'float32':
        half = np.fromfile(path, dtype='<f2', count=valid).astype(np.float32).reshape(valid, 1)
        if np.isfinite(half).all():
            compare('原始输出前缀按连续 FP16 读取 vs CPro 输入点积', half, pred_cpro)
            compare('CPro down × sigmoid(FP16 解读) vs CPro 实测 shared_weighted',
                    weighted(cpro['shared_down'], half), cpro['shared_weighted'])
            compare('CPro down × sigmoid(FP16 解读) vs 310P 实测 shared_weighted',
                    weighted(cpro['shared_down'], half), nz['shared_weighted'])
            for token, value in zip(result['tokens'], half[:, 0], strict=True):
                token['raw_as_fp16'] = float(value)
        else:
            print('原始输出前缀按 FP16 解读含有非有限值')
    dest = folder/'shared-gate.json'
    save_json(dest, result)
    print(f'{valid} 个有效 token 的门控分数和 AIR 依赖：{dest}')
