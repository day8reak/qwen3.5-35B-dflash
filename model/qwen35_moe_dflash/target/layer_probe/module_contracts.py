"""Explicit source contracts for whole GDN and the attention query boundary."""
from contextlib import contextmanager
import functools
import importlib
import inspect

import torch

from .source_regions import definition
from .attention_inspect import attention_length_constant


@contextmanager
def gdn_state_contract(enabled, state_dtype='float32'):
    if not enabled:
        yield None
        return
    if state_dtype not in ('float16', 'float32'):
        raise ValueError('GDN state dtype must be float16 or float32')
    module = importlib.import_module('torchair._ge_concrete_graph.fx2ge_converter')
    ge = importlib.import_module('torchair._ge_concrete_graph.ge_apis')
    from torchair.ge import DataType
    half = state_dtype == 'float16'
    dtype = DataType.DT_FLOAT16 if half else DataType.DT_FLOAT
    torch_dtype = getattr(torch, state_dtype)
    op = torch.ops.npu.npu_chunk_gated_delta_rule.default
    getter, registry = module._get_converter, module._CONVERTERS
    original = getter(op)
    signature = inspect.signature(original)
    if not {'query', 'value', 'g', 'initial_state'} <= set(signature.parameters):
        raise ValueError('review installed GDN converter signature')
    _, source = definition(inspect.unwrap(original))
    evidence = dict(calls=0, restored=False, original_converter=source,
        state_dtype=state_dtype,
        policy='EXPLICIT_OPERATOR_STATE_DTYPE; NO_POSTHOC_AIR_REWRITE',
        gate_policy='SOURCE_COMPUTATION_THEN_GE_FP16_CAST' if half else 'SOURCE_UNCHANGED',
        frontend_state_policy=('FP16_KERNEL_OUTPUT_CAST_TO_STOCK_FP32_META_THEN_SOURCE_FP16_WRITEBACK'
                               if half else 'FP32_META_AND_OPERATOR_OUTPUT'),
        internal_accumulation='NOT_ESTABLISHED_BY_OUTPUT_METADATA')

    @functools.wraps(original)
    def adapted(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        values = bound.arguments
        state = values['initial_state']
        if (state.dtype != dtype or not isinstance(state.meta, torch.Tensor)
                or state.meta.dtype != torch_dtype):
            raise ValueError(f'GDN candidate requires {state_dtype} initial_state and metadata')
        if half:
            gate = values['g']
            if (gate.dtype != DataType.DT_FLOAT or not isinstance(gate.meta, torch.Tensor)
                    or gate.meta.dtype != torch.float32):
                raise ValueError('expected original FP32 decay calculation before the FP16 GDN input')
            gate_half = ge.Cast(gate, dst_type=DataType.DT_FLOAT16)
            gate_half.set_meta(gate.meta.to(torch.float16))
            values['g'] = gate_half
        outputs = list(original(*bound.args, **bound.kwargs))
        if len(outputs) != 2:
            raise ValueError('GDN candidate requires two outputs')
        # Metadata for a newly emitted GE operation, not a conversion of existing
        # values. Target kernel support is checked again after compilation.
        outputs[1].set_meta(torch.empty_like(state.meta))
        if outputs[1].dtype != dtype:
            raise ValueError('GDN GE state output did not retain the selected contract')
        if half:
            # The private exporter's stock FakeTensor output is FP32. Return an
            # explicit numeric Cast to satisfy it, keeping the actual custom op
            # output FP16. The original forward then writes FP16 state. This
            # lossless widening bridge does not request FP32 kernel arithmetic.
            bridge = ge.Cast(outputs[1], dst_type=DataType.DT_FLOAT)
            bridge.set_meta(state.meta.to(torch.float32))
            outputs[1] = bridge
        evidence['calls'] += 1
        return outputs

    existed, previous = op in registry, registry.get(op)
    had_attr, previous_attr = hasattr(op, '_ge_converter'), getattr(op, '_ge_converter', None)
    try:
        registry[op], op._ge_converter = adapted, adapted
        if getter(op) is not adapted:
            raise ValueError('TorchAir did not select the GDN state adapter')
        yield evidence
        if evidence['calls'] != 1:
            raise ValueError('whole GDN must contain one ChunkGatedDeltaRule call')
    finally:
        if existed:
            registry[op] = previous
        else:
            registry.pop(op, None)
        if had_attr:
            op._ge_converter = previous_attr
        else:
            delattr(op, '_ge_converter')
        evidence['restored'] = getter(op) is original


def check_gdn_source(graph, schema, phase, state_dtype='float32'):
    from qwen35_moe_dflash.target.operator_air_review import descriptor, _dtype, port_names, _data_edges
    from qwen35_moe_dflash.target.original_air_roundtrip import _attributes
    from .candidates import check_graph
    enums = schema.enums[schema.package + '.DataType']
    nodes = graph['op']
    selected = [n for n in nodes if n['type'] == 'ChunkGatedDeltaRule']
    if len(selected) != 1:
        raise ValueError('whole GDN must retain one ChunkGatedDeltaRule')
    op = selected[0]
    names = port_names(_attributes(op), 'input')
    if state_dtype not in ('float16', 'float32'):
        raise ValueError('GDN state dtype must be float16 or float32')
    expected = {'initial_state': state_dtype}
    if state_dtype == 'float16':
        expected.update(query='float16', key='float16', value='float16', g='float16',
                        beta='float16', effective_length='int16')
    for name, dtype in expected.items():
        slots = [i for i, n in names.items() if n == name]
        if len(slots) != 1 or _dtype(descriptor(op['input_desc'][slots[0]], enums)) != dtype:
            raise ValueError(f'GDN source {name} must be {dtype}')
    if state_dtype == 'float16':
        gate_slot = next(i for i, name in names.items() if name == 'g')
        edge = _data_edges(op)[gate_slot]
        producer = next((n for n in nodes if n['name'] == edge.rsplit(':', 1)[0]), {})
        if (not edge.endswith(':0') or producer.get('type') != 'Cast'
                or _dtype(descriptor(producer['input_desc'][0], enums)) != 'float32'
                or _dtype(descriptor(producer['output_desc'][0], enums)) != 'float16'):
            raise ValueError('GDN g must use an explicit FP32-to-FP16 Cast after source computation')
    nets = [n for n in nodes if n['type'] == 'NetOutput']
    if (len(op.get('output_desc', [])) != 2
            or _dtype(descriptor(op['output_desc'][1], enums)) != state_dtype
            or len(nets) != 1 or len(nets[0].get('input_desc', [])) != 3
            or _dtype(descriptor(nets[0]['input_desc'][2], enums)) != state_dtype):
        raise ValueError(f'GDN source operator output and writeback must both be {state_dtype}')
    chunk = _attributes(op).get('chunk_size', {}).get('i')
    if chunk != (1 if phase == 'decode' else 64):
        raise ValueError('GDN chunk_size differs from original phase branch')
    projections = [n for n in nodes if n['type'] == 'QuantBatchMatmulV4444']
    if len(projections) != 5 or sum(n['type'] == 'LinearAttentionConvstateUpdate' for n in nodes) != 1:
        raise ValueError('whole GDN must contain all five projections and convstate update')
    for p in projections:
        check_graph(dict(op=[n for n in nodes if n['type'] != 'QuantBatchMatmulV4444' or n is p]), schema, True)
    return dict(status='PASSED', initial_state=state_dtype, last_recurrent_state=state_dtype,
                checked_inputs=expected, writeback=state_dtype,
                quantized_projections=5, chunk_size=chunk, scope='SOURCE_AIR_ONLY')


def attention_contract(record, inputs, graph, schema, bindings):
    from qwen35_moe_dflash.target.operator_air_review import _data_edges, port_names, trace_input
    from qwen35_moe_dflash.target.original_air_roundtrip import _attributes
    x = dict(zip((s['name'] for s in record['inputs']), inputs))
    q, a = x['query'], record['attrs']
    if q.ndim != 4 or q.shape[-1] != 16 or a['input_layout'] != 'BNSD':
        raise ValueError('expected original packed Q [B, heads*D/16, S, 16], BNSD')
    b, width, seq, _ = q.shape
    heads = a['num_heads']
    lengths = list(a['actual_seq_lengths_q'])
    if (b != 1 or width % heads or lengths != [seq]
            or any(type(n) is not int for n in lengths)
            or seq != record['sequence_length']
            or (record['phase'] == 'prefill' and (seq <= 1 or max(lengths) <= 1))
            or (record['phase'] == 'decode' and (seq != 1 or lengths != [1] * b))):
        raise ValueError(f'attention phase/Q length mismatch: {record["phase"]}, Q={list(q.shape)}, lengths={lengths}')
    nodes = {n['name']: n for n in graph['op']}
    ops = [n for n in nodes.values() if n['type'] == 'AdnFusedInferAttention']
    if len(ops) != 1:
        raise ValueError('attention probe must contain one AdnFusedInferAttention')
    ports = port_names(_attributes(ops[0]), 'input')
    edges = _data_edges(ops[0])
    query_slot = [i for i, n in ports.items() if n == 'query']
    all_slot = [i for i, n in ports.items() if n == 'all_seq_lengths_q']
    length_slot = [i for i, n in ports.items() if n == 'actual_seq_lengths_q']
    if len(query_slot) != 1 or len(all_slot) != 1 or len(length_slot) != 1:
        raise ValueError('attention query/actual_seq_lengths_q/all_seq_lengths_q ports unresolved')
    expected = {record['inputs'][s['fixture_index']]['name']: s['name'] + ':0' for s in bindings}
    if edges[query_slot[0]] != expected['query']:
        raise ValueError('attention query is no longer the unchanged captured Q input')
    trace = trace_input(edges[all_slot[0]], nodes, schema.enums[schema.package + '.DataType'])
    if not trace['chain'] or trace['chain'][-1]['name'] + ':0' != expected['all_seq_lengths_q']:
        raise ValueError('attention allQLen carrier was lost during export')
    length_const = attention_length_constant(edges[length_slot[0]], nodes,
        schema.enums[schema.package + '.DataType'], lengths)
    return dict(status='PASSED', phase=record['phase'], query_shape=list(q.shape),
        query_sequence_axis=2, physical_query_length=seq, head_dim=width * 16 // heads,
        actual_seq_lengths_q=lengths, actual_seq_lengths_kv=list(a['actual_seq_lengths_kv']),
        all_seq_lengths_q=x['all_seq_lengths_q'].tolist(),
        all_query_length_scope='SEPARATE_CONTROL; NOT_ASSUMED_EQUAL_TO_CURRENT_Q_LENGTH',
        source_query_edge=edges[query_slot[0]], all_length_trace=trace,
        query_length_constant=length_const,
        dispatch='Q_LENGTH_CONTRACT_CHECKED; KERNEL_DISPATCH_AND_TILING_NOT_OBSERVED')


def compiled_state_contract(observed, output_index=2, state_dtype='float32'):
    """Check the requested ABI; FP32 still rejects hidden FP16 producers."""
    if state_dtype not in ('float16', 'float32'):
        raise ValueError('compiled GDN requires an explicit state dtype contract')
    nodes = {n['name']: n for n in observed['nodes']}
    nets = [n for n in nodes.values() if n['type'] == 'NetOutput']
    if len(nets) != 1 or len(nets[0]['inputs']) <= output_index:
        raise ValueError('compiled whole GDN state output is missing')
    if nets[0]['inputs'][output_index]['dtype'] != state_dtype:
        raise ValueError(f'compiled GDN state ABI is not {state_dtype}')
    edge = nets[0]['input'][output_index]
    chain, seen = [], set()
    while edge not in seen:
        seen.add(edge)
        name, slot = edge.rsplit(':', 1)
        node = nodes.get(name)
        if node is None:
            break
        dtype = node['outputs'][int(slot)]['dtype']
        chain.append(dict(name=name, type=node['type'], slot=int(slot), dtype=dtype))
        if dtype != state_dtype:
            raise ValueError('compiled state producer differs from selected dtype: ' + str(chain))
        if state_dtype == 'float16':
            break  # Explicitly selected FP16 output; no internal precision claim.
        if node['type'] not in ('Cast', 'CastT', 'Identity'):
            break
        if node.get('inputs') and node['inputs'][0]['dtype'] != 'float32':
            raise ValueError('compiled state Cast consumes non-FP32 data')
        edge = node['input'][0]
    return dict(status=('FP16' if state_dtype == 'float16' else 'FP32') + '_VISIBLE_BOUNDARY_VERIFIED',
        state_dtype=state_dtype, chain=chain,
        internal_accumulation='UNVERIFIED_INSIDE_OPAQUE_KERNEL; NOT_PROVEN_BY_NETOUTPUT_OR_GRAPHOP')
