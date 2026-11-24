"""Build small static AIR controls around the original SiluPatch consumer."""
from .decode_cache_control import graph_message


def downstream(source, name):
    edge = name + ':0'
    return [(n, i) for n in source.op for i, arg in enumerate(n.input) if arg == edge]


def with_consumer(raw, source, consumer):
    """Append the source Cast, Reshape or GDN gate Mul to the small AIR's output."""
    nodes = {n.name: n for n in source.op}
    message, graph = graph_message(raw, 'SiluPatch')
    net = graph.op[-1]
    previous, = graph.output
    if net.type != 'NetOutput' or list(net.input) != [previous] or len(consumer.output_desc) != 1:
        raise ValueError('Expected a single output for the probe and consumer')
    if consumer.type == 'Cast':
        if len(consumer.input) != 1 or len(consumer.input_desc) != 1 or (
                consumer.input_desc[0].dtype, consumer.output_desc[0].dtype) != (1, 2):
            raise ValueError(f'{consumer.name}: expected the original FP32 to FP16 Cast')
    elif consumer.type == 'Mul':
        if (len(consumer.input) != 2 or [d.dtype for d in consumer.input_desc] != [1, 1]
                or consumer.output_desc[0].dtype != 1):
            raise ValueError(f'{consumer.name}: expected the original FP32 gate Mul')
    elif consumer.type != 'Reshape':
        raise ValueError(f'Unsupported probe consumer {consumer.type}')
    additions = []
    copied = {n.name for n in graph.op}
    for arg in (consumer.input[1:] if consumer.type == 'Reshape' else ()):
        if not arg:
            continue
        key, _, slot = arg.rpartition(':')
        const = nodes.get(key)
        if slot != '0' or const is None or const.type != 'Const' or any(const.input):
            raise ValueError(f'{consumer.name}: shape operand {arg} is not a literal Const in the AIR')
        if key not in copied:
            additions.append(const)
            copied.add(key)
    if consumer.name in copied:
        raise ValueError(f'Name collision in control: {consumer.name}')
    # Parse through bytes because each decoded AIR owns a separate protobuf schema pool.
    def append(src):
        op = graph.op.add()
        op.ParseFromString(src.SerializeToString())
        op.id = len(graph.op) - 1
        for field in ('src_name', 'src_index', 'dst_name', 'dst_index'):
            op.ClearField(field)
        return op
    saved_net = type(net)()
    saved_net.CopyFrom(net)
    del graph.op[-1]
    for const in additions:
        append(const)
    if consumer.type == 'Mul':
        # In RMSNormGated the other operand is the FP32 normalized tensor of the same shape.
        data = graph.op[0]
        dims = tuple(data.output_desc[0].shape.dim)
        declared = tuple(consumer.input_desc[0].shape.dim)
        if declared and all(d > 0 for d in declared) and declared != dims:
            raise ValueError(f'{consumer.name}: RMSNorm operand {declared} differs from gate {dims}')
        peer = append(data)
        peer.name = 'rms_norm_output'
        peer.attr['index'].i = 1
        for desc in (peer.input_desc[0], peer.output_desc[0]):
            desc.dtype = 1
        graph.input.append(peer.name + ':0')
    placed = append(consumer)
    if consumer.type == 'Mul':
        placed.input[:] = [peer.name + ':0', previous]
    else:
        placed.input[0] = previous
    output = append(saved_net)
    output.input[0] = placed.name + ':0'
    output.input_desc[0].CopyFrom(placed.output_desc[0])
    graph.output[:] = [output.input[0]]
    return message.SerializeToString(deterministic=True)


def without_silu(raw):
    """Replace only SiluPatch and its FP16 Data by an FP32 Data at z's shape."""
    message, graph = graph_message(raw, 'SiluPatch')
    data, silu = graph.op[:2]
    edges = downstream(graph, silu.name)
    if (data.type, silu.type) != ('Data', 'SiluPatch') or len(edges) != 1 or edges[0][0].type not in ('Cast', 'Mul'):
        raise ValueError('Expected a single Cast or Mul after SiluPatch')
    consumer, slot = edges[0]
    for desc in (data.input_desc[0], data.output_desc[0]):
        desc.dtype = silu.output_desc[0].dtype
    consumer.input[slot] = data.name + ':0'
    del graph.op[1]
    for i, op in enumerate(graph.op):
        op.id = i
    return message.SerializeToString(deterministic=True)


def with_source_squeeze(raw, source, silu_name):
    """Prefix the real unary Squeeze if its static input shape is recoverable."""
    nodes = {n.name: n for n in source.op}
    inputs = list(nodes[silu_name].input)
    if len(inputs) != 1 or not inputs[0].endswith(':0'):
        raise ValueError(f'{silu_name}: cannot isolate producer of {inputs}')
    src = nodes[inputs[0][:-2]]
    if src.type != 'Squeeze' or len(src.input) != 1 or len(src.input_desc) != 1 or len(src.output_desc) != 1:
        raise ValueError(f'{silu_name}: immediate producer is {src.name} ({src.type}), expected a unary Squeeze')
    message, graph = graph_message(raw, 'SiluPatch')
    data, silu, net = graph.op
    output_shape = list(data.output_desc[0].shape.dim)
    dims = list(src.input_desc[0].shape.dim)
    axes = list(src.attr['axis'].list.i) if 'axis' in src.attr else []
    rank = len(output_shape) + len(axes)
    positions = {i if i >= 0 else i + rank for i in axes}
    if len(positions) != len(axes) or any(i < 0 or i >= rank for i in positions):
        raise ValueError(f'{src.name}: invalid squeeze axes {axes} for output {output_shape}')
    if axes:
        inferred = output_shape.copy()
        for i in sorted(positions):
            inferred.insert(i, 1)
        if dims and dims != [-2] and (len(dims) != rank or any(a > 0 and a != b for a, b in zip(dims, inferred))):
            raise ValueError(f'{src.name}: input descriptor {dims} disagrees with axes {axes}')
        dims = inferred
    elif not dims or any(d <= 0 for d in dims) or [d for d in dims if d != 1] != output_shape:
        raise ValueError(f'{src.name}: cannot determine input shape from {dims} and axes {axes}')
    if any(d.dtype not in (0, 2) for d in (src.input_desc[0], src.output_desc[0])):
        raise ValueError(f'{src.name}: Squeeze ports disagree with the FP16 SiluPatch input')
    for desc in (data.input_desc[0], data.output_desc[0]):
        desc.shape.dim[:] = dims
    squeeze = type(silu)()
    squeeze.ParseFromString(src.SerializeToString())
    for field in ('src_name', 'src_index', 'dst_name', 'dst_index'):
        squeeze.ClearField(field)
    squeeze.input[0] = data.name + ':0'
    silu.input[0] = squeeze.name + ':0'
    graph.op.insert(1, squeeze)
    for i, op in enumerate(graph.op):
        op.id = i
    connected = message.SerializeToString(deterministic=True)
    del graph.op[2]
    net = graph.op[-1]
    net.id = 2
    net.input[0] = squeeze.name + ':0'
    net.input_desc[0].CopyFrom(squeeze.output_desc[0])
    graph.output[:] = list(net.input)
    return message.SerializeToString(deterministic=True), connected
