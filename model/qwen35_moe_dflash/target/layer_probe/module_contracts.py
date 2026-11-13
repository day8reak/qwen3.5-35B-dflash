"""Explicit source contracts for whole GDN and the attention query boundary."""
from contextlib import contextmanager
import functools
import importlib
import inspect

import torch

from .source_regions import definition
from .attention_inspect import attention_length_constant


@contextmanager
def gdn_fp32_state(enabled):
    if not enabled:
        yield None
        return
    module = importlib.import_module('torchair._ge_concrete_graph.fx2ge_converter')
    from torchair.ge import DataType
    op = torch.ops.npu.npu_chunk_gated_delta_rule.default
    getter, registry = module._get_converter, module._CONVERTERS
    original = getter(op)
    signature = inspect.signature(original)
    if not {'query', 'value', 'initial_state'} <= set(signature.parameters):
        raise ValueError('review installed GDN converter signature')
    _, source = definition(inspect.unwrap(original))
    evidence = dict(calls=0, restored=False, original_converter=source,
        policy='FP32_INITIAL_STATE_AND_NEW_GE_STATE_OUTPUT; NO_POSTHOC_AIR_REWRITE',
        internal_accumulation='NOT_ESTABLISHED_BY_OUTPUT_METADATA')

    @functools.wraps(original)
    def adapted(*args, **kwargs):
        values = signature.bind(*args, **kwargs).arguments
        state = values['initial_state']
        if state.dtype != DataType.DT_FLOAT or not isinstance(state.meta, torch.Tensor):
            raise ValueError('GDN FP32 candidate requires FP32 initial_state at the operator')
        outputs = original(*args, **kwargs)
        if len(outputs) != 2 or state.meta.dtype != torch.float32:
            raise ValueError('GDN FP32 candidate requires two outputs and FP32 state metadata')
        # Metadata for a newly emitted GE operation, not a conversion of existing
        # FP16 values. Target kernel support is checked again after compilation.
        outputs[1].set_meta(torch.empty_like(state.meta))
        if outputs[1].dtype != DataType.DT_FLOAT:
            raise ValueError('GDN GE state output did not retain the FP32 contract')
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


def check_gdn_source(graph, schema, phase):
    from qwen35_moe_dflash.target.operator_air_review import descriptor, _dtype, port_names
    from qwen35_moe_dflash.target.original_air_roundtrip import _attributes
    from .candidates import check_graph
    enums = schema.enums[schema.package + '.DataType']
    nodes = graph['op']
    selected = [n for n in nodes if n['type'] == 'ChunkGatedDeltaRule']
    if len(selected) != 1:
        raise ValueError('whole GDN must retain one ChunkGatedDeltaRule')
    op = selected[0]
    names = port_names(_attributes(op), 'input')
    slots = [i for i, n in names.items() if n == 'initial_state']
    if (len(slots) != 1 or _dtype(descriptor(op['input_desc'][slots[0]], enums)) != 'float32'
            or len(op.get('output_desc', [])) != 2
            or _dtype(descriptor(op['output_desc'][1], enums)) != 'float32'):
        raise ValueError('GDN source initial_state and last_recurrent_state must both be FP32')
    chunk = _attributes(op).get('chunk_size', {}).get('i')
    if chunk != (1 if phase == 'decode' else 64):
        raise ValueError('GDN chunk_size differs from original phase branch')
    projections = [n for n in nodes if n['type'] == 'QuantBatchMatmulV4444']
    if len(projections) != 5 or sum(n['type'] == 'LinearAttentionConvstateUpdate' for n in nodes) != 1:
        raise ValueError('whole GDN must contain all five projections and convstate update')
    for p in projections:
        check_graph(dict(op=[n for n in nodes if n['type'] != 'QuantBatchMatmulV4444' or n is p]), schema, True)
    return dict(status='PASSED', initial_state='float32', last_recurrent_state='float32',
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


def compiled_state_contract(observed, output_index=2):
    """Reject a visible FP16 state producer hidden by an FP32 output Cast."""
    nodes = {n['name']: n for n in observed['nodes']}
    nets = [n for n in nodes.values() if n['type'] == 'NetOutput']
    if len(nets) != 1 or len(nets[0]['inputs']) <= output_index:
        raise ValueError('compiled whole GDN state output is missing')
    if nets[0]['inputs'][output_index]['dtype'] != 'float32':
        raise ValueError('compiled GDN state ABI is not FP32; update selected operator support')
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
        if dtype != 'float32':
            raise ValueError('compiled state producer is not FP32: ' + str(chain))
        if node['type'] not in ('Cast', 'CastT', 'Identity'):
            break
        if node.get('inputs') and node['inputs'][0]['dtype'] != 'float32':
            raise ValueError('compiled state Cast consumes non-FP32 data')
        edge = node['input'][0]
    return dict(status='FP32_VISIBLE_BOUNDARY_VERIFIED', chain=chain,
        internal_accumulation='UNVERIFIED_INSIDE_OPAQUE_KERNEL; NOT_PROVEN_BY_NETOUTPUT_OR_GRAPHOP')
