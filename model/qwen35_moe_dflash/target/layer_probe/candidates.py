"""Explicit, opt-in QLinear dtype and CacheUpdate indexing experiments.

These change only exported test modules. Captured reference outputs, original
model source, installed converters and the shared operator/DDK are untouched.
"""
from __future__ import annotations

import ast
import copy
from contextlib import contextmanager
import functools
import importlib
import inspect
import types

import numpy as np
import torch

from .common import checked, identity, read_array
from .source_regions import assignment, call, definition


def cache_indices(position, table, block_size, mode):
    if mode == 'integer':
        # Replay construction validates [0, INT32_MAX]. Cast BEFORE selecting
        # row zero: its Gather data input cannot be INT64 in the selected DDK.
        position = position.to(torch.int32)
        block = (position[0] // block_size).to(torch.int32)
        # Selected DDK rejects INT32 FloorMod. For nonnegative p and positive
        # block_size, p - floor(p/block_size)*block_size is the exact remainder;
        # the product is <= p, so the validated INT32 range cannot overflow.
        # export_one also scopes its GE Mul metadata adapter to this candidate.
        offset = position[0] - block * block_size
        target = table[:, block].reshape(-1)
    elif mode == 'float':
        position = position.to(torch.float32)
        block = (position[0].float() * (1.0 / block_size)).to(torch.int32)
        offset = (position[0] - block.float() * float(block_size)).to(torch.int32)
        target = table.to(torch.float32)[:, block.to(torch.int32)][0]
    else:
        raise ValueError('cache indexing must be float or integer')
    return target.to(torch.int32), offset


def reconstruct_cache_position(cache, updates, blocks, offsets, table):
    """Invert captured index arguments without changing their cardinality.

    A multirow update may have one captured index. Keep that one index; this
    probe does not infer a broadcast or expand it to per-row destinations.
    """
    if (cache.ndim != 4 or cache.shape[-1] != 16 or updates.ndim != 3
            or cache.dtype != np.dtype('float16') or updates.dtype != np.dtype('float16')
            or updates.shape[1:] != (cache.shape[1], 16)
            or table.ndim != 2 or table.shape[0] != 1 or table.dtype != np.dtype('int32')):
        raise ValueError('requires captured B=1 NZ cache, updates and INT32 [1,blocks] table')
    count, size = updates.shape[0], cache.shape[2]
    observed = (f'cache={list(cache.shape)}, updates={list(updates.shape)}, '
                f'target_block={list(blocks.shape)}/{blocks.dtype}, '
                f'offset_in_block={list(offsets.shape)}/{offsets.dtype}')
    if size != 32:
        raise ValueError('requires original block_size=32; observed ' + observed)
    if blocks.dtype != np.dtype('int32') or offsets.dtype != np.dtype('int32'):
        raise ValueError('requires captured INT32 cache indices; observed ' + observed)
    if (count < 1 or blocks.ndim != 1 or offsets.shape != blocks.shape
            or blocks.size not in (1, count)):
        raise ValueError('requires matching captured [1] or [T] cache indices; observed ' + observed)
    if np.any((blocks < 0) | (blocks >= cache.shape[0]) | (offsets < 0) | (offsets >= size)):
        raise ValueError('captured cache write is out of bounds')
    positions = []
    for block, offset in zip(blocks.tolist(), offsets.tolist()):
        matches = np.flatnonzero(table[0] == block)
        if len(matches) != 1:
            raise ValueError(f'cannot uniquely recover position for physical block {block}; capture high-level position directly')
        positions.append(int(matches[0]) * size + offset)
    position = np.asarray([positions], dtype='<i8')
    if position.size == 0 or int(position.max()) > np.iinfo(np.int32).max:
        raise ValueError('position exceeds the selected INT32 block-index contract')
    target, offset = cache_indices(torch.from_numpy(position), torch.from_numpy(table.copy()), size, 'integer')
    if not np.array_equal(target.numpy(), blocks) or not np.array_equal(offset.numpy(), offsets):
        raise ValueError('reconstructed position does not round-trip captured indices')
    return position, size


def prepare_cache_inputs(capture, manifest, record, root, mode):
    """Keep original cache/updates/golden; obtain page table from the same call."""
    by_name = {s['name']: s for s in record['inputs']}
    if set(by_name) != {'cache', 'updates', 'target_block', 'offset_in_block'}:
        raise ValueError('unexpected captured CacheUpdate boundary')
    matches = [r for r in manifest['regions'] if r['kind'] == 'attention'
               and all(r[k] == record[k] for k in ('layer', 'call', 'phase'))]
    if len(matches) != 1:
        raise ValueError('need one captured fused_attention from the same layer/call/phase for its real block_table')
    table_specs = [s for s in matches[0]['inputs'] if s['name'] == 'block_table']
    if len(table_specs) != 1:
        raise ValueError('captured attention has no unique block_table')
    table_spec = table_specs[0]
    arrays = {k: read_array(capture, v) for k, v in by_name.items()}
    table = read_array(capture, table_spec)
    position, size = reconstruct_cache_position(*(arrays[k] for k in ('cache', 'updates', 'target_block', 'offset_in_block')), table)
    float_target, float_offset = cache_indices(torch.from_numpy(position), torch.from_numpy(table.copy()), size, 'float')
    evidence = dict(mode=mode, scope='INDEX_COMPUTATION_PLUS_KERNEL; CLONED_CACHE_NO_ALIAS_FEEDBACK',
        position_origin='UNIQUE_INVERSE_OF_CAPTURED_KERNEL_INDICES_AND_SAME_CALL_PAGE_TABLE; NOT_DIRECT_HIGH_LEVEL_CAPTURE',
        index_contract=dict(update_rows=int(arrays['updates'].shape[0]),
            index_count=int(arrays['target_block'].size),
            captured_shape=list(arrays['target_block'].shape),
            replay_shape=list(position.shape[1:]),
            position_input_dtype='int64',
            position_compute_dtype='int32' if mode == 'integer' else 'float32',
            position_range=[int(position.min()), int(position.max())],
            integer_cast=('BEFORE_ROW_SELECTION; VALIDATED_NONNEGATIVE_INT32_RANGE'
                          if mode == 'integer' else 'NOT_APPLICABLE_FLOAT_CONTROL'),
            offset_formula=('POSITION_MINUS_BLOCK_TIMES_BLOCK_SIZE; NO_FLOORMOD'
                            if mode == 'integer' else 'ORIGINAL_FLOAT_CONTROL'),
            policy='PRESERVE_CAPTURED_INDEX_CARDINALITY; NO_BROADCAST_OR_PER_ROW_EXPANSION'),
        block_table_region=matches[0]['id'], block_table=table_spec,
        original_indices={k: by_name[k] for k in ('target_block', 'offset_in_block')},
        integer_indices_equal_capture=True,
        float_indices_equal_capture=bool(np.array_equal(float_target.numpy(), arrays['target_block'])
                                         and np.array_equal(float_offset.numpy(), arrays['offset_in_block'])),
        block_size=size, golden='UNCHANGED_CAPTURED_WHOLE_CACHE')
    folder = root/'candidate-inputs'
    folder.mkdir()
    path = folder/'cache_position.bin'
    position.tofile(path)
    position_spec = dict(identity(path, root), name='cache_position', dtype='int64', shape=list(position.shape))
    inputs = [by_name['cache'], by_name['updates'], position_spec, table_spec]
    paths = [checked(capture, by_name['cache']), checked(capture, by_name['updates']), path, checked(capture, table_spec)]
    revised = copy.deepcopy(record)
    revised['inputs'] = inputs
    revised['attrs'].update(block_size=size, cache_indexing=mode)
    tensors = [torch.from_numpy(a.copy()) for a in (arrays['cache'], arrays['updates'], position, table)]
    return revised, tensors, paths, evidence


def validate_cache_source(target):
    """The floating control must be the supplied CPro update's index formula."""
    tree, source = definition(target.Qwen3_5MoeAttention.update)
    expected = {
        'cache_position': 'cache_position.to(torch.float32)',
        'block_idx': '(cache_position[0].float() * self.inv_block_size).to(torch.int32)',
        'block_table': 'self.block_table.to(torch.float32)',
        'target_blocks': 'block_table[:, block_idx.to(torch.int32)][0]',
    }
    for name, expression in expected.items():
        if ast.dump(assignment(tree, name).value) != ast.dump(ast.parse(expression, mode='eval').body):
            raise ValueError('CPro CacheUpdate indexing changed; review control expression: ' + name)
    offsets = [n.value for n in tree.body if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'offset_in_block' for t in n.targets)]
    expected_offsets = ['cache_position[0] - block_idx.float() * self.block_size_float', 'offset_in_block.to(torch.int32)']
    if [ast.dump(x) for x in offsets] != [ast.dump(ast.parse(x, mode='eval').body) for x in expected_offsets]:
        raise ValueError('CPro cache offset expression changed')
    init, _ = definition(target.Qwen3_5MoeAttention.__init__)
    for name, expression in (('self.block_size', '32'), ('self.block_size_float', '32.0'), ('self.inv_block_size', '1.0 / self.block_size')):
        assignments = [n for n in init.body if isinstance(n, ast.Assign) and len(n.targets) == 1 and ast.unparse(n.targets[0]) == name]
        if len(assignments) != 1 or ast.dump(assignments[0].value) != ast.dump(ast.parse(expression, mode='eval').body):
            raise ValueError('CPro cache block constants changed: ' + name)
    return source


def adapt_qlinear(region, root):
    """Keep the selected original forward, replacing only the two scale paths."""
    from .replay import tensor_metadata
    tree, source = definition(region.op.forward)
    op = call(tree, 'torch_npu.npu_quant_matmul')
    if len(op.args) != 3 or ast.unparse(op.args[2]) != 'self.scale.to(torch.float32).to(x.device)':
        raise ValueError('QLinear scale expression changed; review the actual replacement factory')
    token = [k for k in op.keywords if k.arg == 'pertoken_scale']
    if len(token) != 1:
        raise ValueError('QLinear must have explicit pertoken_scale')
    before = tensor_metadata(region.op.scale)
    if region.op.scale.dtype != torch.float32 or region.op.W_q.dtype != torch.int32:
        raise ValueError('candidate expects source FP32 scale and unchanged INT32 packed W4')
    scale = region.op.scale.to(torch.float16).contiguous()
    if not torch.isfinite(region.op.scale).all() or not torch.isfinite(scale).all():
        raise ValueError('source scale is nonfinite or FP16 conversion overflows')
    if region.op.W_q.ndim != 2 or scale.ndim != 2 or region.op.W_q.shape[0] % 256:
        raise ValueError('candidate supports only unbatched K256 packed projections')
    if tuple(scale.shape) != (region.op.W_q.shape[0]//256, region.op.W_q.shape[1]*8):
        raise ValueError('weight/scale shape is not [K,N/8] / [K/256,N]')
    underflow = int(((scale == 0) & (region.op.scale != 0)).sum())
    # Convert the real buffer before export; never relabel existing FP32 bytes.
    region.op.scale = scale
    op.args[2] = ast.parse('self.scale.to(torch.float16).to(x.device)', mode='eval').body
    # The stock A8W4 FakeTensor contract requires FP32 per-token scale.
    # Keep that frontend expression; adapt it only inside the GE converter.
    tree.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[tree], type_ignores=[]))
    code = ast.unparse(module) + '\n'
    path = root/'candidate-qlinear-forward.py'
    path.write_text(code)
    namespace = dict(region.op.forward.__func__.__globals__)
    exec(compile(code, str(path), 'exec'), namespace)
    region.op.forward = types.MethodType(namespace[tree.name], region.op)
    record = dict(kind='QLINEAR_DIRECT_FP16_SCALES', source=source,
        effective_forward=identity(path, root), source_scale=before,
        exported_scale=tensor_metadata(scale), packed_weight=tensor_metadata(region.op.W_q),
        scale_nonzero_rounded_to_zero=underflow,
        pertoken='FRONTEND_FP32_UNCHANGED; GE_RESHAPE_M_THEN_FP16_CAST',
        reference='ORIGINAL_CAPTURE_UNCHANGED; FP16_ROUNDING_IS_PART_OF_THIS_CANDIDATE')
    return record


@contextmanager
def qlinear_ge_scales(enabled):
    """Adapt the selected converter's inputs after stock FakeTensor validation.

    Use the installed converter unchanged after the cast; do not replace its
    V4444 attributes, optional inputs, packed weight handling or output logic.
    The temporary registration is restored even if export raises.
    """
    if not enabled:
        yield None
        return
    module = importlib.import_module('torchair._ge_concrete_graph.fx2ge_converter')
    ge = importlib.import_module('torchair._ge_concrete_graph.ge_apis')
    from torchair.ge import DataType
    op = torch.ops.npu.npu_quant_matmul.default
    getter = getattr(module, '_get_converter', None)
    registry = getattr(module, '_CONVERTERS', None)
    if not callable(getter) or not isinstance(registry, dict):
        raise ValueError('installed TorchAir converter registry differs; inspect fx2ge_converter.py')
    original = getter(op)
    if not callable(original):
        raise ValueError('selected TorchAir has no npu_quant_matmul converter')
    signature = inspect.signature(original)
    if not {'x1', 'x2', 'scale', 'pertoken_scale'} <= set(signature.parameters):
        raise ValueError('selected quant_matmul converter signature differs: ' + str(signature))
    _, source = definition(inspect.unwrap(original))
    evidence = dict(calls=0, restored=False, original_converter=source,
        policy='STOCK_META_UNCHANGED; GE_PERTOKEN_RESHAPE_CAST; ORIGINAL_CONVERTER_DELEGATED')

    @functools.wraps(original)
    def adapted(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        values = bound.arguments
        expected = dict(x1=DataType.DT_INT8, x2=DataType.DT_INT32,
                        scale=DataType.DT_FLOAT16, pertoken_scale=DataType.DT_FLOAT)
        if any(getattr(values.get(k), 'dtype', None) != dtype for k, dtype in expected.items()):
            raise ValueError('GE candidate requires INT8 x, INT32 packed W4, FP16 scale and frontend FP32 pertoken')
        token = values['pertoken_scale']
        meta = getattr(token, 'meta', None)
        if not isinstance(meta, torch.Tensor) or meta.dtype != torch.float32:
            raise ValueError('GE pertoken input has no FP32 tensor metadata')
        flat = ge.Reshape(token, [-1])
        flat.set_meta(meta.reshape(-1))
        half = ge.Cast(flat, dst_type=DataType.DT_FLOAT16)
        half.set_meta(meta.reshape(-1).to(torch.float16))
        values['pertoken_scale'] = half
        evidence['calls'] += 1
        evidence['pertoken'] = dict(frontend=token.tensor, reshape=flat.tensor, cast=half.tensor,
                                   frontend_dtype='float32', consumer_dtype='float16')
        return original(*bound.args, **bound.kwargs)

    existed, previous = op in registry, registry.get(op)
    had_attr, previous_attr = hasattr(op, '_ge_converter'), getattr(op, '_ge_converter', None)
    try:
        registry[op] = adapted
        op._ge_converter = adapted
        if getter(op) is not adapted:
            raise ValueError('selected TorchAir did not select the process-local converter adapter')
        yield evidence
        if evidence['calls'] != 1:
            raise ValueError('candidate requires exactly one adapted quant_matmul conversion')
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


def check_graph(graph, schema, qlinear_fp16=False, cache_mode='kernel'):
    from qwen35_moe_dflash.target.operator_air_review import _data_edges, _dtype, descriptor, port_names
    from qwen35_moe_dflash.target.original_air_roundtrip import _attributes
    enums = schema.enums[schema.package + '.DataType']
    nodes = {n['name']: n for n in graph['op']}
    result = dict(status='PASSED', scope='EXPORTED_AIR_PORTS; NOT_RUNTIME_BYTES')
    if qlinear_fp16:
        selected = [n for n in nodes.values() if n['type'] == 'QuantBatchMatmulV4444']
        if len(selected) != 1:
            raise ValueError('candidate must retain exactly one QuantBatchMatmulV4444')
        node = selected[0]
        names = port_names(_attributes(node), 'input')
        observed = {}
        for port, dtype in (('x1', 'int8'), ('x2', 'int32'), ('scale', 'float16'), ('pertoken_scale', 'float16')):
            slots = [i for i, name in names.items() if name == port]
            if len(slots) != 1:
                raise ValueError('candidate AIR has no unique named port: ' + port)
            slot = slots[0]
            actual = _dtype(descriptor(node['input_desc'][slot], enums))
            if actual != dtype:
                raise ValueError(f'candidate AIR {port} dtype {actual} != {dtype}')
            observed[port] = dict(slot=slot, dtype=actual, edge=_data_edges(node)[slot])
        result['quant_matmul_inputs'] = observed
        # A dtype label alone is insufficient: this candidate must retain the
        # executable FP32 -> FP16 conversion at the per-token boundary.
        edge = observed['pertoken_scale']['edge']
        producer = nodes[edge.rsplit(':', 1)[0]]
        if (not edge.endswith(':0') or producer['type'] != 'Cast'
                or _dtype(descriptor(producer['input_desc'][0], enums)) != 'float32'
                or _dtype(descriptor(producer['output_desc'][0], enums)) != 'float16'):
            raise ValueError('candidate AIR pertoken must be fed by an explicit FP32-to-FP16 Cast')
    if cache_mode != 'kernel':
        selected = [n for n in nodes.values() if n['type'] == 'CacheUpdate']
        if len(selected) != 1:
            raise ValueError('candidate must retain exactly one CacheUpdate')
        node = selected[0]
        names = port_names(_attributes(node), 'input')
        pending, seen, floating, gathers, multiplies = [], set(), [], [], []
        for port in ('targetBlock', 'offsetInBlock'):
            slots = [i for i, name in names.items() if name == port]
            if len(slots) != 1:
                raise ValueError('CacheUpdate input name unresolved: ' + port)
            slot = slots[0]
            if _dtype(descriptor(node['input_desc'][slot], enums)) != 'int32':
                raise ValueError('CacheUpdate index consumer is not INT32')
            pending.append(_data_edges(node)[slot])
        while pending:
            edge = pending.pop()
            if not edge or edge.endswith(':-1') or edge in seen:
                continue
            seen.add(edge)
            name, slot = edge.rsplit(':', 1)
            producer = nodes[name]
            if cache_mode == 'integer' and producer['type'] == 'FloorMod':
                raise ValueError('integer cache index path must not use FloorMod; selected DDK rejects INT32 input')
            descriptors = producer.get('output_desc', [])
            dtype = _dtype(descriptor(descriptors[int(slot)] if int(slot) < len(descriptors) else {}, enums))
            if cache_mode == 'integer' and dtype is None:
                raise ValueError('integer index edge dtype is unresolved: ' + edge)
            if dtype in ('float16', 'float32', 'bfloat16'):
                floating.append(dict(edge=edge, dtype=dtype))
            if cache_mode == 'integer' and producer['type'] == 'Mul':
                inputs = _data_edges(producer)
                input_desc = producer.get('input_desc', [])
                if len(inputs) != 2 or len(input_desc) != 2:
                    raise ValueError('integer cache Mul requires two resolved input descriptors: ' + name)
                for input_edge, raw_desc in zip(inputs, input_desc):
                    input_name, input_slot = input_edge.rsplit(':', 1)
                    source_desc = nodes[input_name].get('output_desc', [])
                    source_dtype = _dtype(descriptor(source_desc[int(input_slot)] if int(input_slot) < len(source_desc) else {}, enums))
                    consumer_dtype = _dtype(descriptor(raw_desc, enums))
                    if source_dtype != 'int32' or consumer_dtype != 'int32':
                        raise ValueError(f'integer cache Mul input must be INT32: {name} input={input_edge}, '
                                         f'producer={source_dtype}, consumer={consumer_dtype}')
                multiplies.append(dict(node=name, inputs=inputs, output_dtype=dtype))
            if cache_mode == 'integer' and producer['type'] in ('Gather', 'GatherV2', 'GatherV2D'):
                data_edge = _data_edges(producer)[0]
                source_name, source_slot = data_edge.rsplit(':', 1)
                source_desc = nodes[source_name].get('output_desc', [])
                data_dtype = _dtype(descriptor(source_desc[int(source_slot)] if int(source_slot) < len(source_desc) else {}, enums))
                consumer_desc = producer.get('input_desc', [])
                consumer_dtype = _dtype(descriptor(consumer_desc[0], enums)) if consumer_desc else None
                if data_dtype != 'int32' or (consumer_dtype is not None and consumer_dtype != 'int32'):
                    raise ValueError(f'integer cache Gather data must be INT32 before selection: '
                                     f'{producer["name"]} input0={data_edge}, producer={data_dtype}, consumer={consumer_dtype}')
                gathers.append(dict(node=producer['name'], input0=data_edge, dtype=data_dtype))
            pending.extend(_data_edges(producer))
        if cache_mode == 'integer' and floating:
            raise ValueError('integer index path was lowered through floating tensors: ' + str(floating))
        source_names = {edge.rsplit(':', 1)[0] for edge in seen if nodes[edge.rsplit(':', 1)[0]]['type'] == 'Data'}
        if len(source_names) != 2:
            raise ValueError('cache index path must depend on position and block table runtime inputs')
        result['cache_indexing'] = dict(mode=cache_mode, data_inputs=sorted(source_names),
                                       floating_edges=floating, integer_gather_data=gathers,
                                       integer_multiply=multiplies)
    return result
