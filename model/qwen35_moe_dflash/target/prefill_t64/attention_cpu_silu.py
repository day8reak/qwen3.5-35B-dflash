"""Replace the two SiLU sites in the original L0 attention AIR, preserving its external ports."""
from collections import Counter
import hashlib
from pathlib import Path

from ..layer_probe.common import save_json, sha
from .attention_slice import descriptor
from .decode_cache_control import graph_message
from .gdr_stage_air import stage_edges, VIEWS
from .qkv_silu import activation_edges, CONVS

CASE = 'l00_c00_gdn_layer_hidden_o0'
SITES = ('conv', 'gate')


def silu_sites(graph):
    gate = stage_edges(graph, 'norm')
    return {'conv': activation_edges(graph), 'gate': (gate['silu_input'], gate['silu_output'])}


def replace_sites(graph, sites):
    nodes = {op.name: op for op in graph.op}
    enums = graph.DESCRIPTOR.file.enum_types_by_name['DataType'].values_by_name
    fp16, fp32 = (enums[k].number for k in ('DT_FLOAT16', 'DT_FLOAT'))
    next_id = max(op.id for op in graph.op) + 1
    replacements, policy_nodes, evidence = {}, [], {}

    def dtype(edge):
        op = nodes[edge.rsplit(':', 1)[0]]
        # GE integer attributes and protobuf TensorDescriptor dtypes use different enums.
        ge_float = {0: fp32, 1: fp16}
        if op.type == 'Cast' and 'dst_type' in op.attr:
            return ge_float.get(op.attr['dst_type'].i)
        if op.type == 'FileConstant' and 'dtype' in op.attr:
            attr = op.attr['dtype']
            if attr.HasField('i'):
                return ge_float.get(attr.i)
            if attr.HasField('dt'):
                return attr.dt
        code = descriptor(nodes, edge).dtype
        if code:
            return code
        if op.type in VIEWS | {'Transpose', 'Permute', 'Silu', 'SiLU', 'Swish', 'SwishV2', 'Sigmoid', 'Mul'}:
            return dtype(op.input[0])
        if op.type in CONVS:
            # Convolution preserves the floating activation type. Its filter is
            # another typed operand if the activation has no static descriptor.
            return dtype(op.input[0]) or dtype(op.input[1])
        if op.type == 'Const' and 'value' in op.attr:
            return op.attr['value'].t.desc.dtype
        return None

    for site in SITES:
        x, y = sites[site]
        old = nodes[y.rsplit(':', 1)[0]]
        if (not y.endswith(':0') or len(old.output_desc) != 1 or old.name in replacements or
                old.type not in ('SiLU', 'Silu', 'Swish', 'SwishV2', 'Mul')):
            raise ValueError(f'Unexpected {site} SiLU producer {old.name} ({old.type})')
        if old.type != 'Mul' and (len([e for e in old.input if e and not e.endswith(':-1')]) != 1 or
                                  any(old.attr[k].f != 1 for k in ('scale', 'beta') if k in old.attr)):
            raise ValueError(f'{old.name} is not a unit-scale SiLU')
        src, dest = dtype(x), dtype(y)
        if src not in (fp16, fp32) or dest not in (fp16, fp32):
            raise ValueError(f'Cannot establish FP16/FP32 at {site}: {x} ({src}) -> {y} ({dest})')
        original_desc = old.output_desc[0]
        prefix = f'l0_cpu_silu_{site}_'
        seq = []
        controls = [e for e in old.input if e.endswith(':-1')]

        def emit(label, kind, edges, dt, *, final=False):
            nonlocal next_id
            name = old.name if final else prefix + label
            if not final and name in nodes:
                raise ValueError(f'Inserted SiLU name collides with original AIR: {name}')
            op = type(old)(name=name, type=kind, id=old.id if final else next_id)
            next_id += 1
            for arg, code in edges:
                op.input.append(arg)
                inp = op.input_desc.add()
                inp.CopyFrom(descriptor(nodes, arg))
                inp.dtype = code
            if not seq:
                op.input.extend(controls)
            out = op.output_desc.add()
            out.CopyFrom(original_desc if final else op.input_desc[0])
            out.dtype = dt
            if kind == 'Cast':
                op.attr['dst_type'].i = 0 if dt == fp32 else 1
            seq.append(op)
            nodes[name] = op
            policy_nodes.append(name)
            return name + ':0'

        wide = x if src == fp32 else emit('wide', 'Cast', [(x, src)], fp32)
        sigmoid = emit('sigmoid', 'Sigmoid', [(wide, fp32)], fp32)
        product = emit('mul', 'Mul', [(wide, fp32), (sigmoid, fp32)], fp32, final=dest == fp32)
        if dest == fp16:
            emit('narrow', 'Cast', [(product, fp32)], fp16, final=True)
        replacements[old.name] = seq
        evidence[site] = dict(input=x, output=y, source_op=old.type,
                              input_dtype='float16' if src == fp16 else 'float32',
                              output_dtype='float16' if dest == fp16 else 'float32',
                              sigmoid=sigmoid, product=product)
    ordered = [item for op in graph.op for item in replacements.get(op.name, [op])]
    del graph.op[:]
    graph.op.extend(ordered)
    return policy_nodes, evidence


def compile_options(path, folder):
    from google.protobuf import message_factory
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    msg, graph = graph_message(raw, 'ChunkGatedDeltaRule')
    names, evidence = replace_sites(graph, silu_sites(graph))
    if not graph.name:
        graph.name = 'l0_attention_cpu_silu'
    policy = folder/'cpu-silu-policy.json'
    group = dict(groups_info=[dict(npucl_implementation=7)], nodes_info={n: dict(group_id=0) for n in names})
    save_json(policy, dict(npucl_strategy=dict(npucl_graphs={graph.name: group})))
    if msg is graph:
        desc = graph.DESCRIPTOR.file.pool.FindMessageTypeByName('ge.proto.ModelDef')
        cls = (message_factory.GetMessageClass(desc) if hasattr(message_factory, 'GetMessageClass')
               else message_factory.MessageFactory(desc.file.pool).GetPrototype(desc))
        msg = cls()
        msg.graph.add().CopyFrom(graph)

    def transform(snapshot):
        if sha(snapshot) != digest:
            raise ValueError('AIR changed while preparing CPU SiLU compilation')
        return msg.SerializeToString(), dict(source_sha256=digest, sites=evidence)
    return dict(air_transform=transform, cl_customization=policy)


def inspect(model, ddk):
    from ...toolchain.omc_container import read_container, read_partition
    from ...toolchain.omc_cpu_inspect import _attr, _task
    from ...toolchain.omc_wire import load_schema
    from ...draft.ir_inspect import _enum_fields
    from ...draft.compile_precision import _observed_desc

    schema = load_schema(Path(ddk)/'tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so')
    enums = _enum_fields(schema)
    parts = {p['type']: p for p in read_container(model)['partitions']}
    main, = schema.decode(read_partition(model, parts[0]), 'ModelDef')['graph']
    counts, proved = Counter(), {}
    for owner in main['op']:
        if owner['type'] != 'GraphOp':
            continue
        task = _task(owner)
        counts[task['cl_name']] += 1
        if task['cl_name'] != 'CPUCL':
            continue
        part = parts[2]
        if not 0 <= task['offset'] < part['bytes'] or not 0 < task['bytes'] <= part['bytes'] - task['offset']:
            raise ValueError('Invalid CPUCL task slice')
        raw = read_partition(model, dict(offset=part['offset'] + task['offset'], bytes=task['bytes']))
        ops = schema.decode(raw, 'GraphDef')['op']
        for sig in ops:
            if sig['type'] not in ('Sigmoid', 'Activation') or (
                    sig['type'] == 'Activation' and _attr(sig, 'mode').get('i') != 0):
                continue
            matching = [s for s in SITES if f'l0_cpu_silu_{s}_sigmoid' in sig['name']]
            if not matching:
                continue
            site, = matching
            x, = sig['input']
            mul, = [n for n in ops if n['type'] == 'Mul' and
                    sorted(n['input']) == sorted([x, sig['name'] + ':0'])]
            ports = [v for n in (sig, mul) for k in ('input_desc', 'output_desc') for v in n[k]]
            if site in proved or len(ports) != 5 or any(_observed_desc(p, enums)['dtype'] != 'float32' for p in ports):
                raise ValueError(f'Ambiguous or non-FP32 compiled {site} SiLU')
            proved[site] = dict(graphop=owner['name'], sigmoid=sig['name'], mul=mul['name'], dtype='float32')
    if set(proved) != set(SITES) or not counts['NPUCL']:
        raise ValueError(f'Expected two verified CPUCL SiLU sites and NPUCL attention work; got {proved}, {counts}')
    return dict(model_sha256=sha(model), sites=proved, graphop_counts=dict(counts))
