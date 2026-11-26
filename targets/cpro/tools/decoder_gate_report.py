"""Compare a retained four-layer compile with the RMS gate extracted from that AIR."""
import json
import re

from parse_model_json import graphs, read_omc
from qwen35_moe_dflash.target.layer_probe.common import sized_file
from qwen35_moe_dflash.target.prefill_t64.attention_slice import descriptor, port_info
from qwen35_moe_dflash.target.prefill_t64.decode_cache_control import graph_message
from qwen35_moe_dflash.target.prefill_t64.decoder_four_boundary import data_edges
from qwen35_moe_dflash.target.prefill_t64.decoder_four_silu import activation_port_shape, gate_product
from qwen35_moe_dflash.target.prefill_t64.whole_run import hiai_failure_stage


GATE_SHAPE = re.compile(r"'(?:ori_)?shape'\s*:\s*\(2048,\s*128\)")
CONV_SHAPE = re.compile(r"'(?:ori_)?shape'\s*:\s*\(1,\s*(?:8192,\s*64|64,\s*8192)\)")
PORT_FIELDS = re.compile(
    r"'(shape|ori_shape|dtype|format|ori_format|addr_type)'\s*:\s*(\([^)]*\)|'[^']*'|-?\d+)"
)


def read(path):
    return json.loads(path.read_bytes())


def show_graph(air, names, trace_shapes):
    _, graph = graph_message(air.read_bytes(), 'SiluPatch')
    nodes = {op.name: op for op in graph.op}
    views = {'Cast', 'Reshape', 'View', 'Identity', 'TensorMove', 'Squeeze', 'Unsqueeze', 'ExpandDims'}

    def describe(edge):
        op = nodes[edge.rsplit(':', 1)[0]]
        port = port_info(graph, descriptor(nodes, edge))
        return f"{edge}({op.type}, {port['dtype']}, {port['shape']}, {port['layout']})"

    for name, expected in trace_shapes.items():
        op = nodes[name]
        print(f'  [AIR shape] {name} 导出器输入 {expected}，上游 {op.input[0]}')
        x, z = activation_port_shape(op.input_desc[0]), activation_port_shape(op.output_desc[0])
        print(f'    x={x}\n    z={z}')
        if x['fx_dims'] is not None and z['fx_dims'] is not None and x['fx_dims'] != z['fx_dims']:
            print('    输入与输出的 FX shape 注释不一致')
        for reader in graph.op:
            for slot, source in enumerate(reader.input):
                if source == name + ':0' and slot < len(reader.input_desc):
                    print(f'    → {reader.name} ({reader.type}) input[{slot}]={activation_port_shape(reader.input_desc[slot])} '
                          f'outputs={[activation_port_shape(d) for d in reader.output_desc]}')

    chains = {}
    for name in names:
        gate = nodes[name]
        edge, = data_edges(gate)
        path = [describe(edge)]
        seen = {edge}
        while nodes[edge.rsplit(':', 1)[0]].type in views:
            edge = data_edges(nodes[edge.rsplit(':', 1)[0]])[0]
            if edge in seen:
                raise ValueError(f'{name}: cycle at {edge}')
            seen.add(edge)
            path.append(describe(edge))
        gate_edge = name + ':0'
        mul, rms = gate_product(graph, gate_edge)
        cast, = [e for e in data_edges(mul) if e != rms]
        print('  门控分支', ' → '.join([*reversed(path), describe(gate_edge), describe(cast), mul.name]))
        print('  乘法另一侧', describe(rms), '→', mul.name)
        chains[name] = (cast.rsplit(':', 1)[0], mul.name)
    return chains


def show_omc_edges(model, chains):
    """Read explicit references in each serialized graph's own namespace."""
    document, _ = read_omc(model)
    targets = set(chains)
    expected_names = targets | {name for pair in chains.values() for name in pair}
    seen = set()
    graph_count = 0
    print(f'  OMC 连线 {model}')
    for path, graph in graphs(document, include_nested=True):
        graph_count += 1
        ops = [op for op in graph['op'] if isinstance(op, dict)]
        named = [op for op in ops if op.get('name') in expected_names]
        uses = {name: [] for name in targets}
        unknown_ports = []

        def record(edge, where):
            if isinstance(edge, str):
                name = edge.removesuffix(':0')
                if name in uses:
                    uses[name].append(where)

        for reader in ops:
            who = f'{reader.get("name")} ({reader.get("type")})'
            for slot, edge in enumerate(reader.get('input', [])):
                record(edge, f'{who} input[{slot}]={edge}')
            sources = reader.get('src_name', reader.get('srcName', []))
            indices = reader.get('src_index', reader.get('srcIndex', []))
            for slot, source in enumerate(sources):
                if slot < len(indices) and str(indices[slot]) == '0':
                    record(source, f'{who} src_name[{slot}]={source}, src_index=0')
                elif slot >= len(indices) and isinstance(source, str) and source in targets:
                    unknown_ports.append(f'{who} src_name[{slot}]={source}，未记录源槽号')
        for slot, edge in enumerate(graph.get('output', [])):
            record(edge, f'graph.output[{slot}]={edge}')
        if not named and not any(uses.values()) and not unknown_ports:
            continue
        print(f'    图 {graph.get("name", "")}  {path}')
        for entry in unknown_ports:
            print('      ', entry)
        for op in named:
            name = op['name']
            if name not in targets:
                print(f'      {name} ({op.get("type")}) input={op.get("input", [])} '
                      f'src_name={op.get("src_name", op.get("srcName", []))} '
                      f'src_index={op.get("src_index", op.get("srcIndex", []))}')
                continue
            seen.add(name)
            dest = op.get('dst_name', op.get('dstName', []))
            slots = op.get('dst_index', op.get('dstIndex', []))
            print(f'      {name}:0 ({op.get("type")}) →', ' | '.join(uses[name]) or '本图未见显式消费者输入或图输出')
            print(f'        生产者记录 dst_name={dest}  dst_index={slots}')
        local_names = {op['name'] for op in named}
        for name in sorted(targets - local_names):
            if uses[name]:
                print(f'      本图引用 {name}:0，但未见同名生产者：', ' | '.join(uses[name]))
    missing = sorted(targets - seen)
    if missing:
        print(f'    在 {graph_count} 个可见图中未找到同名门控节点：{", ".join(missing)}')


def show_compile(folder, names):
    log = folder/'compile.log'
    by_name = {name: set() for name in names}
    kernels = set()
    conv_kernels = set()
    observed = {}
    print(f'  OMG 日志 {log}')
    if not log.is_file():
        print('  该次编译日志未保留')
        return observed
    with log.open(errors='replace') as lines:
        for number, line in enumerate(lines, 1):
            if 'SiluPatch' in line and 'RuntimeConfig(' in line:
                node = re.search(r'OP\[([^,\]]+),\s*SiluPatch\]', line, re.I)
                if node and node[1] in by_name:
                    by_name[node[1]].add(line.split('::')[-1].strip().strip('"'))
            if 'BuildSingleOp(' not in line or 'te_silupatch_' not in line:
                continue
            kernel = re.search(r'kernelName\[([^\]]+)\]', line)
            _, sep, ports = line.partition('op inputs:')
            if not kernel or not sep:
                continue
            inputs = ports.partition('outputs:')[0]
            if CONV_SHAPE.search(inputs) and kernel[1] not in conv_kernels:
                conv_kernels.add(kernel[1])
                print(f'  L{number} 卷积 SiLU 形状对应的 kernel {kernel[1]}')
                for label, tensor in zip(('输入', '输出'), ports.split('outputs:', 1)):
                    print(f'    {label}', dict(PORT_FIELDS.findall(tensor.partition('}')[0])))
            if not GATE_SHAPE.search(inputs):
                continue
            first = kernel[1] not in kernels
            if first:
                kernels.add(kernel[1])
                print(f'  L{number} 门控形状对应的 kernel {kernel[1]}')
            for label, tensor in zip(('输入', '输出'), ports.split('outputs:', 1)):
                fields = PORT_FIELDS.findall(tensor.partition('}')[0])
                for key, value in fields:
                    observed.setdefault(f'{label}.{key}', set()).add(value.replace(' ', ''))
                if first:
                    print(f'    {label}', dict(fields))
    if not kernels:
        print('  编译日志未记录可按门控形状识别的 BuildSingleOp 行')
    for name, lines in by_name.items():
        print(' ', name, ' | '.join(sorted(lines)) or '无具名 RuntimeConfig 记录')
    fields = ('kernelName', 'binFileName', 'magic', 'coreType', 'blockDim', 'opParaSize', 'workspace', 'parameters')
    metadata = sorted(p for p in folder.rglob('*.json') if p.stem in kernels)
    for file in metadata:
        data = read(file)
        print(' ', file.relative_to(folder), {k: data[k] for k in fields if k in data})
    if kernels and not metadata:
        print('  此 OMC 目录没有保留与上述 kernel 同名的 JSON')
    return observed


def report(root):
    layer = root/'board/L00-L03'
    probes = layer/'gate-probes'
    case = root/'board/package/L00-L03'
    index = read(probes/'compile.json')
    meta = read(case/'case.json')
    record = index['air']
    if record not in meta['files'] or record['path'] != meta['air_directory'] + '/export.air':
        raise ValueError('当前四层 AIR 与本次 gate-probe 的来源记录不一致')
    if not (case/record['path']).is_file():
        case = root/'export/package/L00-L03'
        if record not in read(case/'case.json')['files']:
            raise ValueError('当前导出目录与本次 gate-probe 的来源记录不一致')
    control = probes/'silu-cast-mul'
    row = index['cases'][control.name]
    gate = row['nodes'][0]
    gates = [n['name'] for n in meta['silu_patch']['nodes'] if n['shape'] == [2048, 128]]
    shapes = {n['name']: n['shape'] for n in meta['silu_patch']['nodes'] if n['shape'] in ([1, 8192, 64], [2048, 128])}
    if gate not in gates:
        raise ValueError(f'四层 AIR 门控记录里没有 {gate}')
    port_reports = []
    for base, air, names in ((layer, sized_file(case, record), gates), (control, control/'weight/export.air', [gate])):
        build = read(base/'omc/build.json')
        replay = read(base/'replay/result.json')
        if replay['omc_sha256'] != build['model']['sha256']:
            raise ValueError(f'{base}: 当前编译与上板记录不匹配')
        if base == control and build['stamp']['bundle'] != row['source']:
            raise ValueError(f'{control}: 小图 OMC 与切图记录不匹配')
        status = replay['status']
        if status == 'FAILED':
            status += ' ' + hiai_failure_stage(replay.get('error', ''))
        print(f'[{base.name}] 上板记录 {status}，退出码 {replay.get("exitcode")}，AIR {air}')
        chains = show_graph(air, names, shapes if base == layer else {gate: shapes[gate]})
        try:
            show_omc_edges(sized_file(base/'omc', build['model']), chains)
        except (OSError, ValueError) as error:
            print(f'  {base.name} 的 OMC 连线未能读取：{error}')
        port_reports.append(show_compile(base/'omc', names))
    full, single = port_reports
    common = sorted(full.keys() & single.keys())
    differences = [k for k in common if full[k] != single[k]]
    print(f'两份 BuildSingleOp 日志可比较 {len(common)} 个端口字段，取值集合有差异的字段 {len(differences)} 个')
    for k in differences:
        print(f'  {k} 四层={sorted(full[k])}  小图={sorted(single[k])}')
    for k in sorted(full.keys() ^ single.keys()):
        print(f'  {k} 仅在{"四层" if k in full else "小图"}日志中读到')
