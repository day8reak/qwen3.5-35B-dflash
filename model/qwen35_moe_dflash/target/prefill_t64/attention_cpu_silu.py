"""Replace the two SiLU sites in the original L0 attention AIR, preserving its external ports."""
from collections import Counter
import hashlib
from pathlib import Path

from ..layer_probe.common import sha
from .attention_slice import descriptor, port_info
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
    replacements, evidence = {}, {}

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
        if op.type in ('Const', 'Constant') and 'value' in op.attr:
            stored_dtype = op.attr['value'].t.desc.dtype
            if stored_dtype:
                return stored_dtype
        code = descriptor(nodes, edge).dtype
        if code:
            return code
        if op.type in VIEWS | {'Transpose', 'Permute', 'Silu', 'SiLU', 'Swish', 'SwishV2', 'Sigmoid', 'Mul'}:
            return dtype(op.input[0])
        if op.type in CONVS:
            # Convolution preserves the floating activation type. Its filter is
            # another typed operand if the activation has no static descriptor.
            return dtype(op.input[0]) or dtype(op.input[1])
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
            elif kind in ('Sigmoid', 'Mul'):
                # Pin the implementation on these four new arithmetic nodes only.
                # The resulting device placement is verified in the compiled OMC.
                op.attr['npucl_implementation'].i = 7
            seq.append(op)
            nodes[name] = op
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
    return evidence


def audit_graphs(original, candidate):
    """Compare the two AIR graphs outside the selected SiLU producers, including inline constants."""
    sites = silu_sites(original)
    original_count = len(original.op)
    gdr, = [op for op in original.op if op.type == 'ChunkGatedDeltaRule']
    observed = {op.name: op for op in candidate.op}
    if len(observed) != len(candidate.op):
        raise ValueError('Duplicate node names in candidate AIR')
    # Reconstruct the expected edits from original bytes. Earlier compiles applied
    # the implementation selector in a separate JSON, so accept its absence there.
    site_info = replace_sites(original, sites)
    arithmetic = {s[key].rsplit(':', 1)[0] for s in site_info.values() for key in ('sigmoid', 'product')}
    expected_names = {op.name for op in original.op}
    changed = []
    for op in original.op:
        actual = observed.get(op.name)
        if actual is None:
            continue
        if op.name in arithmetic and 'npucl_implementation' not in actual.attr:
            del op.attr['npucl_implementation']
        if op.SerializeToString(deterministic=True) != actual.SerializeToString(deterministic=True):
            changed.append(op.name)
    missing = sorted(expected_names - observed.keys())
    extra = sorted(observed.keys() - expected_names)
    if (changed or missing or extra or list(original.input) != list(candidate.input)
            or list(original.output) != list(candidate.output)):
        raise ValueError(f'AIR comparison: changed={changed}, missing={missing}, extra={extra}, '
                         f'original ports={(list(original.input), list(original.output))}, '
                         f'candidate ports={(list(candidate.input), list(candidate.output))}')
    port_names = {}
    if '_input_name_key' in gdr.attr and '_input_name_value' in gdr.attr:
        port_names = dict(zip(gdr.attr['_input_name_value'].list.i,
                              (s.decode() for s in gdr.attr['_input_name_key'].list.s), strict=True))
    edges = [edge for edge in gdr.input if edge and not edge.endswith(':-1')]
    ports = [dict(slot=i, name=port_names.get(i), edge=edge,
                  declared=port_info(original, gdr.input_desc[i]) if i < len(gdr.input_desc) else None)
             for i, edge in enumerate(edges)]
    attrs = {k: gdr.attr[k].i for k in ('chunk_size', 'output_final_state', 'use_qk_l2norm_in_kernel')
             if k in gdr.attr}
    return dict(sites=site_info, gate_matches_pytorch_fp32=site_info['gate']['input_dtype'] == 'float32',
                unchanged_non_silu_nodes=original_count - len(sites), gdr_node_identical=True,
                gdr_inputs=ports, gdr_attrs=attrs)


def compile_options(path):
    digest = sha(path)

    def transform(snapshot):
        raw = snapshot.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError('AIR changed while preparing CPU SiLU compilation')
        msg, graph = graph_message(raw, 'ChunkGatedDeltaRule')
        replace_sites(graph, silu_sites(graph))
        prepared = msg.SerializeToString()
        _, original = graph_message(raw, 'ChunkGatedDeltaRule')
        audit = audit_graphs(original, graph)
        print('L0 AIR: two SiLU sites:', audit['sites'], 'unchanged GDR inputs:', audit['gdr_inputs'], flush=True)
        return prepared, dict(source_sha256=digest, **audit)
    return dict(air_transform=transform)


def inspect(model, ddk, *, standalone=False):
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
    cpu_owners = {op['name'] for op in main['op']
                  if op['type'] == 'GraphOp' and _task(op)['cl_name'] == 'CPUCL'}
    cpu_inputs = {edge.rsplit(':', 1)[0] for op in main['op'] if op['name'] in cpu_owners for edge in op['input']}
    boundary_casts = []
    for owner in main['op']:
        if owner['type'] in ('Data', 'NetOutput', 'Const', 'Constant'):
            continue
        if owner['type'] in ('Cast', 'CastT') and (owner['name'] in cpu_inputs or any(
                edge.rsplit(':', 1)[0] in cpu_owners for edge in owner['input'])):
            boundary_casts.append(owner['name'])
            continue
        if owner['type'] != 'GraphOp':
            raise ValueError(f"Unexpected main-graph computation: {owner['name']} ({owner['type']})")
        task = _task(owner)
        counts[task['cl_name']] += 1
        if task['cl_name'] == 'NPUCL':
            continue
        if task['cl_name'] != 'CPUCL':
            raise ValueError(f"Unexpected compute library {task['cl_name']} in {owner['name']}")
        part = parts[2]
        if not 0 <= task['offset'] < part['bytes'] or not 0 < task['bytes'] <= part['bytes'] - task['offset']:
            raise ValueError('Invalid CPUCL task slice')
        raw = read_partition(model, dict(offset=part['offset'] + task['offset'], bytes=task['bytes']))
        ops = schema.decode(raw, 'GraphDef')['op']
        silu_ops = set()
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
            silu_ops.update((sig['name'], mul['name']))
            proved[site] = dict(graphop=owner['name'], sigmoid=sig['name'], mul=mul['name'], dtype='float32')
        extra = [(op['name'], op['type']) for op in ops
                 if op['type'] not in ('Data', 'NetOutput') and op['name'] not in silu_ops]
        if extra or not silu_ops:
            raise ValueError(f"CPUCL task {owner['name']} has unexpected work beyond SiLU: {extra}")
    if set(proved) != set(SITES) or bool(counts['NPUCL']) == standalone:
        raise ValueError(f'Unexpected SiLU placement (standalone={standalone}): {proved}, {counts}')
    return dict(model_sha256=sha(model), sites=proved, graphop_counts=dict(counts),
                boundary_casts=boundary_casts, cpu_arithmetic_only_silu=True)
