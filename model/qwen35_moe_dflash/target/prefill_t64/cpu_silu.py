"""Build an FP32 CPUCL SiLU control with the measured operand's wire ABI."""
from functools import partial
from pathlib import Path

from ..layer_probe.common import save_json, sha
from .decode_cache_control import graph_message

TYPES = {'float16': 'DT_FLOAT16', 'float32': 'DT_FLOAT'}


def control_air(path, *, operand, output_dtype):
    from google.protobuf import message_factory

    _, original = graph_message(path.read_bytes(), 'Data')
    source, = [op for op in original.op if op.type == 'Data']
    if source.name != operand['name']:
        raise ValueError('SiLU control input name differs from the source AIR')
    name, shape = source.name, operand['shape']
    enum = original.DESCRIPTOR.file.enum_types_by_name['DataType'].values_by_name
    fp32 = enum['DT_FLOAT'].number
    wire = enum[TYPES[operand['dtype']]].number
    dest = enum[TYPES[output_dtype]].number
    if source.output_desc[0].dtype not in (0, wire):
        raise ValueError('SiLU control input dtype differs from the original AIR')
    # TorchAir slices can be raw GraphDefs. OMG's --framework=1 reads a ModelDef.
    desc = original.DESCRIPTOR.file.pool.FindMessageTypeByName('ge.proto.ModelDef')
    cls = (message_factory.GetMessageClass(desc) if hasattr(message_factory, 'GetMessageClass')
           else message_factory.MessageFactory(desc.file.pool).GetPrototype(desc))
    msg = cls()
    graph = msg.graph.add(name='silu_probe')

    def node(label, kind, inputs, dtype=None):
        op = graph.op.add(name=label, type=kind, id=len(graph.op))
        for edge, dt in inputs:
            op.input.append(edge)
            op.input_desc.add(dtype=dt, layout='ND').shape.dim.extend(shape)
        if dtype is not None:
            op.output_desc.add(dtype=dtype, layout='ND').shape.dim.extend(shape)
        return op

    node(name, 'Data', [('', wire)], wire).attr['index'].i = 0
    x = name + ':0'
    graph.input.append(x)
    if wire != fp32:
        node('wide', 'Cast', [(x, wire)], fp32).attr['dst_type'].i = 0
        x = 'wide:0'
    node('sigmoid', 'Sigmoid', [(x, fp32)], fp32)
    node('silu', 'Mul', [(x, fp32), ('sigmoid:0', fp32)], fp32)
    y = 'silu:0'
    if dest != fp32:
        node('narrow', 'Cast', [(y, fp32)], dest).attr['dst_type'].i = 1
        y = 'narrow:0'
    node('output', 'NetOutput', [(y, dest)])
    graph.output.append(y)
    return msg.SerializeToString(), dict(source_sha256=sha(path), formula='x * sigmoid(x)',
                                         input=operand, output_dtype=output_dtype)


def compile_options(stage, spec, folder):
    if stage not in ('qkv_conv_silu', 'silu_output'):
        raise ValueError(f'CPU SiLU requires a standalone SiLU fixture, got {stage}')
    operand, = spec['source_inputs']
    out_type = spec['output_contract']['required_dtypes'].get('0', spec['golden']['dtype'])
    names = ['sigmoid', 'silu']
    if operand['dtype'] == 'float16':
        names.append('wide')
    if out_type == 'float16':
        names.append('narrow')
    policy = folder/'cpu-silu-policy.json'
    group = dict(groups_info=[dict(npucl_implementation=7)], nodes_info={n: dict(group_id=0) for n in names})
    save_json(policy, dict(npucl_strategy=dict(npucl_graphs=dict(silu_probe=group))))
    return dict(cl_customization=policy, air_transform=partial(control_air, operand=operand, output_dtype=out_type))


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
    nodes = main['op']
    owners = [_task(n) for n in nodes if n['type'] == 'GraphOp']
    if ([t['cl_name'] for t in owners] != ['CPUCL'] or
            any(n['type'] not in ('Data', 'NetOutput', 'CastT', 'GraphOp') for n in nodes)):
        raise ValueError('OMG did not place the standalone SiLU arithmetic entirely in CPUCL')
    task, = owners
    part = parts[2]
    if not 0 < task['bytes'] <= part['bytes'] - task['offset'] or task['offset'] < 0:
        raise ValueError('Invalid CPUCL task slice')
    raw = read_partition(model, dict(offset=part['offset'] + task['offset'], bytes=task['bytes']))
    cpu = schema.decode(raw, 'GraphDef')['op']
    data, = [n for n in cpu if n['type'] == 'Data']
    net, = [n for n in cpu if n['type'] == 'NetOutput']
    sig, = [n for n in cpu if n['type'] in ('Sigmoid', 'Activation')]
    mul, = [n for n in cpu if n['type'] == 'Mul']
    x = data['name'] + ':0'
    if (len(cpu) != 4 or sig['input'] != [x] or
            (sig['type'] == 'Activation' and _attr(sig, 'mode').get('i') != 0) or
            sorted(mul['input']) != sorted([x, sig['name'] + ':0']) or net['input'] != [mul['name'] + ':0']):
        raise ValueError('Unexpected compiled CPUCL SiLU dataflow')
    for n in (sig, mul):
        if any(_observed_desc(v, enums)['dtype'] != 'float32' for v in n['input_desc'] + n['output_desc']):
            raise ValueError('Compiled SiLU arithmetic has a non-FP32 port')
    source, = [n for n in nodes if n['type'] == 'Data']
    output, = [n for n in nodes if n['type'] == 'NetOutput']
    return dict(model_sha256=sha(model), graphop_counts={'CPUCL': 1},
                input=_observed_desc(source['output_desc'][0], enums),
                output=_observed_desc(output['input_desc'][0], enums),
                cpu_ops=[sig['type'], 'Mul'], arithmetic_dtype='float32',
                main_ops=[n['type'] for n in nodes], evidence='compiled OMC')
