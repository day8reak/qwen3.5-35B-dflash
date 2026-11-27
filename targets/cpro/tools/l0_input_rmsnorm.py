"""Extract and replay the first decoder input RMSNorm on the captured 310P embedding."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from decoder_cases_board import compiled, read
from decoder_gate_probe import air_bundle, execute_case
from l0_accuracy_report import compiled_output_slots, get_reference, reference_matches
from qwen35_moe_dflash.common.accuracy import floating_metrics
from qwen35_moe_dflash.target.layer_probe.common import identity, read_tensor, save_json, sized_file
from qwen35_moe_dflash.target.prefill_t64.attention_slice import DTYPES
from qwen35_moe_dflash.target.prefill_t64.attention_steps import NORMS, operands
from qwen35_moe_dflash.target.prefill_t64.decode_cache_control import graph_message
from qwen35_moe_dflash.target.prefill_t64.decoder_four_silu import activation_port_shape
from qwen35_moe_dflash.target.prefill_t64.l0_split import slice_air, wire
from qwen35_moe_dflash.target.prefill_t64.layer_template import parameter


KIND = 'l0-input-rmsnorm'
SHAPE = (1, 64, 2048)


def extract(air, meta, temp):
    raw = air.read_bytes()
    _, graph = graph_message(raw, 'NetOutput')
    nodes = {op.name: op for op in graph.op}
    enums = graph.DESCRIPTOR.file.enum_types_by_name['DataType'].values_by_name
    types = {enums[ge].number: dtype for dtype, ge in DTYPES.items()}
    public = {name + ':0' for name, role in meta['inputs'].items() if role in ('inputs_embeds', 'embed_scale')}

    def producer(edge):
        name, slot = edge.rsplit(':', 1)
        if slot != '0':
            raise ValueError(f'Unexpected RMSNorm operand {edge}')
        return nodes[name]

    def unview(edge):
        while edge.endswith(':0') and producer(edge).type in ('Cast', 'Reshape', 'View', 'Identity', 'TensorMove'):
            edge = producer(edge).input[0]
        return edge

    candidates = []
    for op in graph.op:
        if op.type in NORMS and op.input:
            edge = unview(op.input[0])
            if not edge.endswith(':0'):
                continue
            p = producer(edge)
            if p.type == 'Mul' and len(public) == 2 and {unview(e) for e in operands(p)} == public:
                candidates.append(op)
    if len(candidates) != 1:
        raise ValueError(f'Expected one input RMSNorm on the embedding × scale path, found {[n.name for n in candidates]}')
    norm = candidates[0]
    if len(norm.input) != 2 or len(norm.input_desc) != 2 or not norm.output_desc:
        raise ValueError(f'Unexpected RMSNorm ports on {norm.name}: {list(norm.input)}')
    x, gamma = norm.input
    y = norm.name + ':0'

    def constant(edge):
        op = producer(edge)
        if op.type == 'FileConstant':
            dtype, shape, path = parameter(op, types, air.parent, temp)
            return np.fromfile(path, dtype=dtype).reshape(shape)
        if op.type in ('Const', 'Constant') and 'value' in op.attr:
            t = op.attr['value'].t
            if t.desc.data_offset or t.desc.layout not in ('', 'ND'):
                raise ValueError(f'{op.name}: unsupported inline constant storage')
            shape = tuple(t.desc.shape.dim) or tuple(op.output_desc[0].shape.dim)
            return np.frombuffer(t.data, dtype=types[t.desc.dtype]).reshape(shape)
        if op.type in ('Identity', 'TensorMove'):
            return constant(op.input[0])
        if op.type == 'Cast':
            return constant(op.input[0]).astype({0: 'float32', 1: 'float16'}[op.attr['dst_type'].i])
        if op.type in ('Add', 'AddV2'):
            left, right = (constant(e) for e in operands(op))
            dtype = types.get(op.output_desc[0].dtype) or np.result_type(left, right)
            return np.add(left, right, dtype=dtype)
        if op.type in ('Reshape', 'View'):
            return constant(op.input[0]).reshape(tuple(constant(op.input[1]).ravel()))
        raise ValueError(f'{op.name} ({op.type}): cannot reconstruct the constant gamma at {edge}')

    def binding(role, port, default):
        info = activation_port_shape(port)
        shapes = [tuple(s) for s in (info['ge_dims'], info['fx_dims']) if s and min(s) > 0]
        dtype = types.get(port.dtype)
        if dtype not in ('float16', 'float32') or port.layout not in ('', 'ND'):
            raise ValueError(f'{norm.name}.{role}: unsupported {dtype} {port.layout}')
        shape = shapes[0] if shapes else default
        if (any(s != shape for s in shapes) or shape[-1] != 2048
                or tuple(n for n in shape if n != 1) != tuple(n for n in default if n != 1)):
            raise ValueError(f'{norm.name}.{role}: unexpected shape {shape} for {default}')
        return wire(role, shape, dtype)

    weight = constant(gamma)
    if weight.size != 2048 or weight.dtype not in (np.float32, np.float16) or not np.isfinite(weight).all():
        raise ValueError(f'{gamma}: expected a finite floating gamma of width 2048')
    wires = {x: binding('x', norm.input_desc[0], SHAPE),
             gamma: binding('gamma', norm.input_desc[1], tuple(weight.shape))}
    wires[y] = dict(name='y', edge=y, aliases=['y'], shape=None, dtype=None)
    step = dict(nodes=[norm.name], inputs=[x, gamma], outputs=[y])
    payload = slice_air(raw, step, wires, required_op=norm.type)
    epsilon = norm.attr['epsilon'].f if 'epsilon' in norm.attr else None
    info = dict(node=norm.name, op=norm.type, epsilon=epsilon, x_edge=x, gamma_edge=gamma, y_edge=y)
    return payload, info, {f'l0_feed_{i}': wires[e] for i, e in enumerate(step['inputs'])}, wires[y], weight


def export(root, remote, base, temp):
    package = root/'board/package'
    if not (package/'L00-L03/case.json').is_file():
        package = root/'export/package'
    case, meta, source = air_bundle(package/'L00-L03', root/'export/package/L00-L03')
    payload, op, feeds, output, gamma = extract(sized_file(case, source), meta, temp)
    fixture = package/'fixture'
    if not (fixture/'reference.json').is_file():
        fixture = root/'export/fixture'
    ref = read(fixture/'reference.json')
    original_x = sized_file(fixture, ref['inputs']['hidden'])
    x = read_tensor(original_x, ref['inputs']['hidden'])
    archives = [base.parent/k/'values.npz' for k in ('l0-steps-same-input', 'l0-steps')]
    archives.append(root/'l0-steps-reference/values.npz')
    archive = None
    for candidate in archives:
        if candidate.is_file():
            with np.load(candidate, allow_pickle=False) as refs:
                if reference_matches(refs, fixture, ref):
                    archive = candidate
                    break
    if archive is None:
        archive = get_reference(root, remote, temp)
    with np.load(archive, allow_pickle=False) as refs:
        if (not reference_matches(refs, fixture, ref) or x.shape != SHAPE or x.dtype != np.float16
                or not np.array_equal(refs['310p_block_entry'], x)):
            raise ValueError('310P RMSNorm capture differs from this four-layer fixture')
        golden = refs['310p_hidden']
        if golden.shape != SHAPE or golden.dtype not in (np.float32, np.float16):
            raise ValueError(f'Unexpected 310P normalized hidden: {golden.shape} {golden.dtype}')
    stage = temp/'case'
    air = stage/'weight/export.air'
    air.parent.mkdir(parents=True)
    air.write_bytes(payload)
    air_rec = identity(air, stage)

    def tensor(name, array):
        path = stage/(name + '.bin')
        array.tofile(path)
        return dict(identity(path, stage), dtype=array.dtype.name, shape=list(array.shape))

    captured = tensor('input_310p', x)
    expected = tensor('output_310p', golden)
    data = {}
    for feed in feeds.values():
        role = feed['name']
        values = x if role == 'x' else gamma
        arr = values.astype(feed['dtype'], copy=False).reshape(feed['shape'])
        if role == 'x' and x.dtype == arr.dtype and x.shape == arr.shape:
            data[role] = captured
        else:
            data[role] = tensor(role, arr)
    info = dict(kind=KIND, native_tool_args=True, air_directory='weight', files=[air_rec], source=air_rec['sha256'],
                outputs=[output], feeds=feeds, real_inputs=data, golden=expected, captured=captured,
                source_air=source, norm=op, input_source=str(original_x), valid=ref['valid_rows'])
    save_json(stage/'case.json', info)
    if base.exists():
        if base.is_symlink() or read(base/'case.json').get('kind') != KIND:
            raise ValueError(f'Unexpected RMSNorm diagnostic directory: {base}')
        shutil.rmtree(base)
    stage.rename(base)
    print(f'Single-op AIR: {base/air_rec["path"]}  {op}')
    for name, item in [('310P RMSNorm 输入', captured), ('310P 实测输出', expected), *data.items()]:
        print(f'{name}: {base/item["path"]}  {item["dtype"]}{item["shape"]}  {item["bytes"]} bytes')
    return info


def run(root, remote, action, ddk, config, args):
    base = root/'board/L00-L03'/KIND
    if action == 'compile':
        base.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base.parent) as tmp:
            info = export(root, remote, base, Path(tmp))
        if args.gate_probe == 'l0-input-rmsnorm-air':
            script = Path(__file__).resolve().parents[1]/'scripts/export_l0_rmsnorm_air.py'
            subprocess.run([sys.executable, str(script), '--case-dir', str(base)], check=True)
        if ddk is not None:
            compiled(base, base, ddk, None, info['source'])
        return True
    info = read(base/'case.json')
    runtime = read(config.parent/read(config)['assets'])['environment']
    result = execute_case(base, info, runtime, config, replace_inputs=args.replace_inputs,
                          replace_weights=args.replace_weights, board_case='4-prefill-t64-l0-input-rmsnorm',
                          real_inputs=(base, info['real_inputs']))
    dest = base/'comparison.json'
    save_json(dest, result)
    if result['status'] != 'EXECUTED':
        print(f'{result["status"]}  {dest}')
        return False
    (_, _, spec, path), = compiled_output_slots(base, info)
    if spec['dtype'] not in ('float16', 'float32') or [v for v in spec['shape'] if v != 1] != [64, 2048]:
        raise ValueError(f'Unexpected normalized output: {spec}')
    actual = read_tensor(path, spec).reshape(64, 2048)
    golden = read_tensor(sized_file(base, info['golden']), info['golden']).reshape(64, 2048)
    valid = info['valid']
    result.update(output=dict(file=str(path), **spec), valid_rows=valid)
    for label, selection in [(f'valid{valid}', slice(0, valid)), ('full64', slice(None))]:
        scores = floating_metrics(actual[selection], golden[selection])
        result[label] = scores
        print(f'{label} CPro vs 310P: ' + '  '.join(f'{k}={scores[k]}' for k in
              ('cosine', 'relative_l2', 'max_abs', 'error_ratio')))
    valid_scores = result[f'valid{valid}']
    result['accuracy'] = 'PASSED' if valid_scores['reference_metrics_passed'] else 'FAILED_ACCURACY'
    save_json(dest, result)
    print(f'CPro output: {path}\nComparison: {dest}')
    print(f'{result["accuracy"]} on {valid} valid tokens; failed metrics: {valid_scores["failed_metrics"]}')
    return valid_scores['reference_metrics_passed']
