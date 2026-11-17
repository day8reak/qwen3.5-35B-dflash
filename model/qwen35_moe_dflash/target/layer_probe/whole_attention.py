"""Replay one source FullAttention module, keeping both reference provenances."""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import torch

from .common import checked, identity, read_array, sha
from .source_regions import call, definition

PROJECTIONS = ('q_proj', 'k_proj', 'v_proj', 'o_proj')
OUTPUTS = ('golden_hidden', 'golden_key_cache', 'golden_value_cache')
INPUTS = ('hidden_states', 'attention_mask', 'position_ids', 'old_key_cache',
          'old_value_cache', 'new_kv_cache_pos', 'allQLen')


def reference_scope(basis):
    if basis == 'ORIGINAL_FULL_ATTENTION_MODULE_WITH_INDEPENDENT_ATTENTION_MATH':
        return 'SOURCE_MODULE_WITH_MATH_ATTENTION_CORE; OTHER_SOURCE_OPS_RETAINED'
    if basis == 'ORIGINAL_FULL_ATTENTION_CAPTURE_WITH_AUDITED_CORE':
        return 'SAME_CALL_SOURCE_MODULE_OUTPUTS; CORE_MATH_AUDITED; WHOLE_MODULE_MATH_UNVERIFIED'
    raise ValueError('unknown whole FullAttention reference basis')


def assemble(manifest, *, capture=None, reference='math-replay'):
    if reference not in ('math-replay', 'captured-audited'):
        raise ValueError('unknown whole FullAttention reference selection')
    records = []
    for original in manifest['regions']:
        if original['point'] != 'full_attention_layer':
            continue
        if [s['name'] for s in original['inputs']] != list(INPUTS):
            raise ValueError('FullAttention needs direct module entry capture; recapture this Prefill')
        if [s['name'] for s in original['outputs']] != list(OUTPUTS):
            raise ValueError('FullAttention captured output tuple changed')
        siblings = [r for r in manifest['regions'] if all(
            r[k] == original[k] for k in ('layer', 'call', 'phase'))]
        components = []
        for point in [*('self_attn_' + n for n in PROJECTIONS),
                      'self_attn_q_norm', 'self_attn_k_norm', 'rope', 'cache_k', 'cache_v', 'fused_attention']:
            matches = [r for r in siblings if r['point'] == point]
            if len(matches) != 1:
                raise ValueError(f'{original["id"]}: needs one {point} from the same invocation')
            components.append(matches[0])
        by_point = {r['point']: r for r in components}
        def same_tensor(spec, other):
            return all(spec[k] == other[k] for k in ('shape', 'dtype', 'bytes', 'sha256'))
        for name in PROJECTIONS[:3]:
            if not same_tensor(original['inputs'][0], by_point['self_attn_' + name]['inputs'][0]):
                raise ValueError('FullAttention Q/K/V projections do not share the captured module input')
        if not same_tensor(original['outputs'][0], by_point['self_attn_o_proj']['outputs'][0]):
            raise ValueError('FullAttention module hidden differs from captured o_proj output')
        if any(r['sequence_length'] != original['sequence_length'] for r in components):
            raise ValueError('FullAttention boundary sequence lengths differ')
        if reference == 'captured-audited':
            if capture is None:
                raise ValueError('captured-audited requires the original capture directory')
            validate_capture(original, capture)
            from .captured_attention_reference import audited_capture_reference
            evidence = audited_capture_reference(Path(capture), manifest, original, components)
            records.append(dict(original, captured_outputs=original['outputs'], components=components,
                reference_basis='ORIGINAL_FULL_ATTENTION_CAPTURE_WITH_AUDITED_CORE',
                reference_evidence=evidence, provenance='SAME_CALL_ORIGINAL_MODULE; CORE_MATH_AUDITED'))
            continue
        refs = [r for r in manifest.get('attention_math_replays', []) if all(
            r[k] == original[k] for k in ('layer', 'call', 'phase'))]
        if len(refs) != 1 or refs[0].get('status') != 'COMPUTED':
            raise ValueError('whole FullAttention requires its separate math-core replay; generate Prefill with --capture-boundaries')
        ref = refs[0]
        if ref.get('policy') != 'ORIGINAL_FULL_ATTENTION_MODULE_WITH_INDEPENDENT_ATTENTION_MATH':
            raise ValueError('unreviewed whole FullAttention reference policy')
        arguments = ref.get('source_forward_arguments', {})
        for spec in original['inputs']:
            name = spec['name']
            if name in ('old_key_cache', 'old_value_cache'):
                source = arguments.get('past_key_values', [None, None])[name == 'old_value_cache']
            else:
                source = arguments.get(name)
            if name == 'allQLen' and isinstance(source, list):
                value = np.asarray(source, dtype='<i8')
                source = dict(shape=list(value.shape), dtype='int64', sha256=hashlib.sha256(value.tobytes()).hexdigest())
            if not isinstance(source, dict) or any(source.get(k) != spec[k] for k in ('shape', 'dtype', 'sha256')):
                raise ValueError('math-core replay did not use identical captured module input: ' + name)
        if [s['name'] for s in ref['outputs']] != list(OUTPUTS):
            raise ValueError('math-core replay output names differ from captured FullAttention tuple')
        for a, b in zip(original['outputs'], ref['outputs']):
            if any(a[k] != b[k] for k in ('shape', 'dtype', 'bytes')):
                raise ValueError('math-core replay changed the original FullAttention output ABI')
        if any(a['sha256'] != b['sha256'] for a, b in zip(original['outputs'][1:], ref['outputs'][1:])):
            raise ValueError('math-core replay changed upstream K/V; this is not an isolated core reference')
        records.append(dict(original, outputs=ref['outputs'], captured_outputs=original['outputs'],
            components=components, reference_basis=ref['policy'], reference_evidence=ref,
            provenance='SAME_MODULE_INPUTS; SEPARATE_ORIGINAL_AND_MATH_CORE_REFERENCE'))
    if not records:
        raise ValueError('capture has no full_attention_layer entry; recapture one Prefill with --capture-boundaries')
    return records


def adapt_forward(owner):
    """Use the actual CPro body; only apply the already selected FP32 rotation."""
    tree, evidence = definition(owner)
    tree = copy.deepcopy(tree)
    tree.decorator_list = []
    # The host FP32 carrier is valid only while both uses go directly to the
    # source update(), whose first operation casts cache_position to FP32.
    updates = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
               and ast.unparse(n.func) == 'self.update']
    position_uses = [n for n in ast.walk(tree) if isinstance(n, ast.Name)
                     and n.id == 'new_kv_cache_pos']
    expected_updates = [
        ['key_states', 'new_kv_cache_pos', 'past_key_values[0]'],
        ['value_states', 'new_kv_cache_pos', 'past_key_values[1]']]
    if (len(position_uses) != 2 or len(updates) != 2 or
            [[ast.unparse(a) for a in n.args] for n in updates] != expected_updates
            or any(n.keywords for n in updates)):
        raise ValueError('review changed FullAttention cache-position consumers')
    rotation = call(tree, 'apply_rotary_pos_emb')
    names = ('query_states', 'key_states', 'cos', 'sin')
    allowed = [(name, name + '.float()') for name in names]
    if (len(rotation.args) != 4 or rotation.keywords or any(
            ast.unparse(arg) not in options for arg, options in zip(rotation.args, allowed))):
        raise ValueError('review changed FullAttention RoPE arguments')
    rotation.args = [ast.parse(name + '.float()', mode='eval').body for name in names]
    statements = [i for i, node in enumerate(tree.body) if rotation in ast.walk(node)]
    if len(statements) != 1:
        raise ValueError('FullAttention rotation must be a direct forward statement')
    index = statements[0] + 1
    tree.body[index:index] = ast.parse(
        'query_states = query_states.to(hidden_states.dtype)\n'
        'key_states = key_states.to(hidden_states.dtype)').body
    returns = [n for n in tree.body if isinstance(n, ast.Return)]
    if len(returns) != 1 or ast.unparse(returns[0].value) != '(attn_output, None, present_key_value)':
        raise ValueError('review changed FullAttention return contract')
    namespace = dict(owner.__globals__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[tree], type_ignores=[])), evidence['file'], 'exec'), namespace)
    evidence['adaptations'] = ['FP32_PARTIAL_ROPE_RESTORE_HIDDEN_DTYPE']
    return namespace[tree.name], evidence


class WholeAttention(torch.nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op
        self.export_adjustments = []

    def forward(self, probe_input_000, probe_input_001, probe_input_002, probe_input_003,
                probe_input_004, probe_input_005, probe_input_006):
        result = self.op(probe_input_000, attention_mask=probe_input_001,
            position_ids=probe_input_002.to(torch.int32),
            past_key_values=(probe_input_003.clone(), probe_input_004.clone()),
            new_kv_cache_pos=probe_input_005, allQLen=probe_input_006,
            use_cache=False, output_attentions=False, export_flag=True)
        return result[0], result[2][0], result[2][1]


def validate_capture(record, capture):
    x = {s['name']: read_array(capture, s) for s in record['inputs']}
    a = record['attrs']
    hidden, mask, positions = (x[n] for n in INPUTS[:3])
    if (record['phase'] != 'prefill' or hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[1] <= 1
            or hidden.dtype != np.float16 or positions.shape != hidden.shape[:2]
            or positions.dtype.kind not in 'iu' or mask.shape != (1, 1, hidden.shape[1], a['kv_max_len'])):
        raise ValueError('whole FullAttention currently requires captured B1 Prefill hidden/mask/positions')
    length = hidden.shape[1]
    if (length != record['sequence_length'] or length > a['kv_max_len']
            or a['kv_max_len'] % a['block_size'] or a['head_dim'] % 16
            or a['num_heads'] % a['num_key_value_heads'] or not np.isfinite(hidden).all()):
        raise ValueError('FullAttention source dimensions or hidden values are invalid')
    if not np.array_equal(positions, np.arange(length)[None]) or not np.array_equal(x['allQLen'], [length]):
        raise ValueError('whole FullAttention probe requires fresh full-length Prefill at positions 0..T-1')
    pos = x['new_kv_cache_pos']
    if pos.dtype.kind not in 'iu' or pos.shape != (length, 1) or not np.array_equal(pos[:, 0], positions[0]):
        raise ValueError('whole FullAttention preserves the source [T,1] cache-position input')
    expected = (a['kv_max_len']//a['block_size'], a['num_key_value_heads']*a['head_dim']//16, a['block_size'], 16)
    for name in ('old_key_cache', 'old_value_cache'):
        if x[name].shape != expected or x[name].dtype != np.float16 or np.count_nonzero(x[name]):
            raise ValueError('whole FullAttention requires the captured zero FP16 paged initial cache')
    if x['allQLen'].dtype != np.int64:
        raise ValueError('allQLen must be captured INT64[1]')
    return x


def prepare_cache_position_input(record, capture, root):
    """Move the source's first FP32 cast to an exact, recorded host carrier.

    Keep the captured record and file immutable. This is input serialization,
    not a replacement of the source's floating block/offset computations.
    """
    spec = record['inputs'][INPUTS.index('new_kv_cache_pos')]
    if spec['name'] != 'new_kv_cache_pos':
        raise ValueError('FullAttention cache-position input order changed')
    position = read_array(capture, spec)
    length = record['sequence_length']
    if (record['phase'] != 'prefill' or position.dtype.kind not in 'iu'
            or position.shape != (length, 1) or length < 2
            or not np.array_equal(position[:, 0], np.arange(length))):
        raise ValueError('FP32 cache-position carrier requires fresh Prefill positions 0..T-1')
    carrier = position.astype('<f4')
    if not np.isfinite(carrier).all() or not np.array_equal(carrier.astype(position.dtype), position):
        raise ValueError('cache positions are not exactly representable in FP32')
    folder = Path(root)/'candidate-inputs'
    folder.mkdir(exist_ok=True)
    path = folder/'new_kv_cache_pos_fp32.bin'
    with path.open('xb') as stream:
        stream.write(carrier.tobytes())
    serialized = dict(spec, dtype='float32', **identity(path, root))
    evidence = dict(status='EXACT_HOST_CAST', original=dict(spec), serialized=serialized,
        values_preserved=True, shape_preserved=True,
        source_operation='cache_position.to(torch.float32); BEFORE_ROW_SELECTION',
        scope='FRESH_PREFILL_ONLY; SOURCE_INDEX_FORMULA_AND_GOLDENS_UNCHANGED')
    return serialized, torch.from_numpy(carrier), path, evidence


def build(record, capture, target, converters, factory, config_path, source_root, root):
    import yaml
    from .candidates import adapt_qlinear, validate_cache_source
    from .target_weights import load_weights
    values = validate_capture(record, capture)
    cache_source = validate_cache_source(target)
    update_tree, _ = definition(target.Qwen3_5MoeAttention.update)
    # validate_cache_source checks the formulas. Also require the cast to be
    # the first use, so moving it across the model input cannot change a use.
    uses = [n for n in update_tree.body if any(isinstance(x, ast.Name)
            and x.id == 'cache_position' for x in ast.walk(n))]
    if not uses or ast.unparse(uses[0]) != 'cache_position = cache_position.to(torch.float32)':
        raise ValueError('source update uses cache_position before its FP32 cast')
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text())
    model_path = Path(config['model_file']).expanduser()
    if not model_path.is_absolute():
        model_path = Path(source_root)/model_path
    wrapper = importlib.import_module('models.export_model_wrapper_qwen3_5_moe')
    _, loader = definition(wrapper._load_text_config)
    if Path(source_root).resolve() not in Path(loader['file']).resolve().parents:
        raise ValueError('FullAttention config loader is outside source-root')
    model_config = wrapper._load_text_config(str(model_path))
    if model_config.kv_cache_max_len != record['attrs']['kv_max_len']:
        raise ValueError('capture and CPro model cache capacities differ')
    # The source Rotary constructor uses autocast(device_type=inv_freq.device).
    # It cannot run on meta. Construct its deterministic CPU buffers once, while
    # retaining meta allocation for the much larger projection parameters.
    constructor, constructor_evidence = definition(target.Qwen3_5MoeAttention.__init__)
    rotary_call = call(constructor, 'Qwen3_5MoeRotaryEmbedding')
    if rotary_call.args or [(k.arg, ast.unparse(k.value)) for k in rotary_call.keywords] != [('config', 'config')]:
        raise ValueError('review changed FullAttention Rotary constructor')
    namespace = target.Qwen3_5MoeAttention.__init__.__globals__
    rotary_class = namespace['Qwen3_5MoeRotaryEmbedding']
    if rotary_class is not target.Qwen3_5MoeRotaryEmbedding:
        raise ValueError('FullAttention constructor uses an unexpected Rotary class')
    rotary = rotary_class(model_config, device='cpu')
    rotary_calls = 0
    def source_rotary(*, config):
        nonlocal rotary_calls
        rotary_calls += 1
        if config is not model_config or rotary_calls != 1:
            raise ValueError('FullAttention changed its Rotary construction contract')
        return rotary
    namespace['Qwen3_5MoeRotaryEmbedding'] = source_rotary
    try:
        with torch.device('meta'):
            op = target.Qwen3_5MoeAttention(model_config, record['layer'])
    finally:
        namespace['Qwen3_5MoeRotaryEmbedding'] = rotary_class
    if rotary_calls != 1 or op.rotary_emb is not rotary:
        raise ValueError('FullAttention did not bind its original CPU Rotary buffers')
    op.block_table = torch.tensor(op.kv_block_table, dtype=torch.int32)
    components = {r['point']: r for r in record['components']}
    with torch.no_grad():
        coefficients = op.rotary_emb(torch.from_numpy(values['hidden_states'].copy()),
                                     torch.from_numpy(values['position_ids'].copy()).to(torch.int32))
    for name, value in zip(('cos', 'sin'), coefficients):
        captured = next(s for s in components['rope']['inputs'] if s['name'] == name)
        expected = read_array(capture, captured)
        if (list(value.shape) != captured['shape'] or str(value.dtype).removeprefix('torch.') != captured['dtype']
                or value.contiguous().numpy().tobytes() != expected.tobytes()):
            raise ValueError('source-generated RoPE coefficient differs from captured ' + name)
    attention = components['fused_attention']
    table = next(s for s in attention['inputs'] if s['name'] == 'block_table')
    if not np.array_equal(op.block_table.numpy(), read_array(capture, table)):
        raise ValueError('CPro block table differs from the same-invocation source table')
    for name in ('num_heads', 'num_key_value_heads', 'head_dim', 'block_size', 'kv_max_len'):
        if getattr(op, name) != record['attrs'][name]:
            raise ValueError('CPro FullAttention config differs on ' + name)
    weights, quant = {}, []
    for member in (*PROJECTIONS, 'q_norm', 'k_norm'):
        r = components['self_attn_' + member]
        captured = torch.load(checked(capture, r['weights']), map_location='cpu', weights_only=True)
        loaded, evidence = load_weights(r, captured, target, converters, config_path, source_root)
        weights[member] = evidence
        if member in PROJECTIONS:
            if r['kind'] != 'qlinear':
                raise ValueError('whole FullAttention requires original captured W4 QLinear projections')
            child = factory(W_q=loaded['W_q'], scale=loaded['scale'], idx=0)
            setattr(op, member, child)
            folder = root/member
            folder.mkdir()
            quant.append(adapt_qlinear(SimpleNamespace(op=child), folder))
        else:
            from .replay import Region
            import torch_npu
            # Reuse the separately verified zero-centered gamma folding.
            setattr(op, member, Region(r, loaded, target, torch_npu, 0))
    if any(t.is_meta for t in (*op.parameters(), *op.buffers())):
        raise ValueError('whole FullAttention has unbound meta weights/buffers')
    forward, source = adapt_forward(target.Qwen3_5MoeAttention.forward)
    op.forward = MethodType(forward, op)
    evidence = dict(status='BOUND_TO_SOURCE', kind='full_attention_layer', source=source,
        cache_index_source=cache_source,
        config_loader=loader, config_sha256=sha(model_path/'config.json'),
        constructor=dict(constructor_evidence, adaptation='META_WEIGHTS_WITH_ORIGINAL_CPU_ROTARY_BUFFERS',
                         temporary_rotary_factory_restored=namespace['Qwen3_5MoeRotaryEmbedding'] is rotary_class),
        input_scope='CAPTURED_MODULE_ENTRY_AND_PRE_CALL_STATES; EXACT_HOST_FP32_CACHE_POSITION; SAME_PREFILL',
        output_scope='FULL_ATTENTION_HIDDEN_KEY_VALUE; NO_RESIDUAL_MOE_OR_DECODER_LAYER',
        weight_scope='ORIGINAL_CHECKPOINT_BOUND', weights=weights, qlinear_candidates=quant,
        reference_basis=record['reference_basis'],
        reference_scope=reference_scope(record['reference_basis']),
        position_serialization='VALUE_PRESERVING_INT32_CAST; FRESH_PREFILL_BOUNDS_CHECKED',
        rope_coefficients='SOURCE_CONFIG_BUFFERS_BIT_EQUAL_TO_CAPTURED_COS_SIN',
        state_feedback='NOT_TESTED')
    return WholeAttention(op).eval(), evidence


def check_graph(graph, schema, record, bindings):
    from qwen35_moe_dflash.target.operator_air_review import _data_edges, port_names, trace_input
    from qwen35_moe_dflash.target.original_air_roundtrip import _attributes
    from .module_contracts import attention_length_constant
    from .candidates import check_graph as check_quant
    nodes = {n['name']: n for n in graph['op']}
    projections = [n for n in nodes.values() if n['type'] == 'QuantBatchMatmulV4444']
    attentions = [n for n in nodes.values() if n['type'] == 'AdnFusedInferAttention']
    caches = [n for n in nodes.values() if n['type'] == 'CacheUpdate']
    nets = [n for n in nodes.values() if n['type'] == 'NetOutput']
    if len(projections) != 4 or len(attentions) != 1 or len(caches) != 2 or len(nets) != 1 or len(_data_edges(nets[0])) != 3:
        raise ValueError('full module AIR requires 4 QLinear, 2 CacheUpdate, 1 attention and hidden/K/V outputs')
    for p in projections:
        check_quant(dict(op=[n for n in nodes.values() if n['type'] != 'QuantBatchMatmulV4444' or n is p]), schema, True)
    op = attentions[0]
    ports = port_names(_attributes(op), 'input')
    edges = _data_edges(op)
    def edge(name):
        slots = [i for i, value in ports.items() if value == name]
        if len(slots) != 1:
            raise ValueError('FullAttention source port unresolved: ' + name)
        return edges[slots[0]]
    enums = schema.enums[schema.package + '.DataType']
    cache_binding = [b for b in bindings if b['fixture_index'] == INPUTS.index('new_kv_cache_pos')]
    if (len(cache_binding) != 1 or cache_binding[0]['dtype'] != 'float32'
            or cache_binding[0]['shape'] != [record['sequence_length'], 1]):
        raise ValueError('FullAttention cache-position fixture must use the exact FP32 carrier')
    carrier_node = nodes[cache_binding[0]['name']]
    descs = carrier_node.get('output_desc', [])
    fp32 = {enums[n] for n in ('DT_FLOAT', 'DT_FLOAT32') if n in enums}
    if (carrier_node['type'] != 'Data' or len(descs) != 1 or descs[0].get('dtype') not in fp32):
        raise ValueError('FullAttention AIR cache-position Data is not FP32')
    dims = descs[0].get('shape', {}).get('dim', [])
    if dims and dims != cache_binding[0]['shape']:
        raise ValueError('FullAttention AIR cache-position shape differs from the fixture')
    length = attention_length_constant(edge('actual_seq_lengths_q'), nodes, enums, [record['sequence_length']])
    all_length = trace_input(edge('all_seq_lengths_q'), nodes, enums)
    binding = [b for b in bindings if b['fixture_index'] == INPUTS.index('allQLen')]
    if len(binding) != 1 or not all_length['chain'] or all_length['chain'][-1]['name'] != binding[0]['name']:
        raise ValueError('whole FullAttention lost its captured allQLen carrier')
    if binding[0]['dtype'] != 'int64':
        raise ValueError('FullAttention allQLen must retain its captured INT64 dtype')
    return dict(status='PASSED', scope='SOURCE_AIR_ONLY; COMPILED_PRECISION_UNVERIFIED',
        quantized_projections=4, cache_updates=2, attention=1, outputs=3,
        query_length_constant=length, allQLen_trace=all_length,
        cache_position_input=dict(name=carrier_node['name'], dtype='float32',
            shape=cache_binding[0]['shape'], fixture_index=INPUTS.index('new_kv_cache_pos')),
        rope='SOURCE_FORWARD_EXPLICIT_FP32_ROTATION; NOT_COMPILED_PRECISION_PROOF')
