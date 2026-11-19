"""One CPro region per process; never construct a complete model or decoder layer."""
from __future__ import annotations
from contextlib import contextmanager
import ast
import hashlib
import importlib
import inspect
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import textwrap

import torch
import torch.nn.functional as F
from .common import KIND, checked, identity, new_output, read_array, save_json, sha


def folded_norm_gamma(target, weight):
    """Fold only the inspected CPro zero-centered forward, at its original dtype."""
    expected = '''
def forward(self, hidden_states):
    output = torch_npu.npu_rms_norm(hidden_states, self.weight + 1.0, self.variance_epsilon)[0]
    return output.type_as(hidden_states)
'''
    actual = textwrap.dedent(inspect.getsource(target.Qwen3_5MoeRMSNorm.forward))
    if ast.dump(ast.parse(actual)) != ast.dump(ast.parse(textwrap.dedent(expected))):
        raise ValueError('CPro ordinary RMSNorm forward changed; inspect its gamma expression before folding')
    with torch.no_grad():
        return (weight + 1.0).detach().contiguous()


def tensor_metadata(value):
    raw = value.detach().contiguous().view(torch.uint8).numpy().tobytes()
    return dict(shape=list(value.shape), dtype=str(value.dtype).removeprefix('torch.'),
                bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def import_file(path):
    spec = importlib.util.spec_from_file_location('_qwen35_original_export_converters', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _data_input_positions(inputs, export_graph):
    """Locate each original GE input by its Data index, before weight conversion."""
    positions = {}
    for position, node in enumerate(export_graph.op):
        if node.type not in ('Data', 'RefData'):
            continue
        if 'index' not in node.attr or not node.attr['index'].HasField('i'):
            raise ValueError(f'{node.type} {node.name} has no integer GE input index')
        index = node.attr['index'].i
        if node.type == 'RefData':
            raise ValueError(f'GE input {index} ({node.name}) is mutable RefData; this probe requires explicit state outputs')
        if index < 0 or index >= len(inputs) or index in positions:
            raise ValueError(f'Invalid or duplicate GE input index {index} on Data {node.name}')
        positions[index] = position
    if len(positions) != len(inputs):
        missing = sorted(set(range(len(inputs))) - positions.keys())
        raise ValueError(f'Export graph has no Data node for input indices {missing}')
    return positions


@contextmanager
def export_input_identity(example_inputs):
    """Bind fixture tensors and weights to their original GE Data nodes."""
    module = importlib.import_module('torchair._utils.export_utils')
    original = getattr(module, '_convert_data_to_const', None)
    if not callable(original) or tuple(inspect.signature(original).parameters) != ('inputs', 'export_graph', 'file_path', 'weight_name'):
        raise ValueError('installed TorchAir input-binding hook differs; inspect its export_utils.py')
    by_id = {id(t): i for i, t in enumerate(example_inputs)}
    if len(by_id) != len(example_inputs):
        raise ValueError('replay input objects must have distinct identities')
    evidence = dict(calls=0, nodes={}, policy='ACTUAL_EXPORT_INPUT_OBJECT_IDENTITY', restored=False)
    def observed(inputs, export_graph, file_path, weight_name):
        evidence['calls'] += 1
        if evidence['calls'] != 1:
            raise ValueError('one region must export as one graph')
        positions = _data_input_positions(inputs, export_graph)
        for i, value in enumerate(inputs):
            if id(value) in weight_name:
                continue
            if id(value) not in by_id:
                raise ValueError(f'exporter supplied unbound runtime input {i}, shape={getattr(value, "shape", None)}, dtype={getattr(value, "dtype", None)}; no shape/value guessing permitted')
            name = export_graph.op[positions[i]].name
            if name in evidence['nodes']:
                raise ValueError('duplicate source Data name')
            evidence['nodes'][name] = by_id[id(value)]
        # TorchAir converts each named weight in place at op[i]. Keep graph order
        # intact even when GE interleaves Data and computation nodes. Empty
        # positions have no weight identity and the converter skips them.
        if any(positions[i] != i for i, value in enumerate(inputs) if id(value) in weight_name):
            aligned = [None] * (max(positions.values()) + 1)
            for i, position in positions.items():
                aligned[position] = inputs[i]
            inputs = aligned
        return original(inputs, export_graph, file_path, weight_name)
    module._convert_data_to_const = observed
    try:
        yield evidence
        if evidence['calls'] != 1 or not evidence['nodes']:
            raise ValueError('TorchAir did not expose one identity-bound region graph')
    finally:
        module._convert_data_to_const = original
        evidence['restored'] = True


class Region(torch.nn.Module):
    def __init__(self, record, weights, target, npu, output_index, qlinear_class=None):
        super().__init__()
        self.kind, self.attrs = record['kind'], record['attrs']
        self.names = tuple(s['name'] for s in record['inputs'])
        self.output_index, self.target, self.npu = output_index, target, npu
        self.export_adjustments = []
        if self.kind == 'qlinear':
            self.op = (qlinear_class or target.QLinear)(W_q=weights['W_q'], scale=weights['scale'], idx=0)
        elif self.kind == 'linear':
            w = weights['weight']
            self.op = torch.nn.Linear(w.shape[1], w.shape[0], bias='bias' in weights)
            self.op.weight = torch.nn.Parameter(w, requires_grad=False)
            if 'bias' in weights:
                self.op.bias = torch.nn.Parameter(weights['bias'], requires_grad=False)
        elif self.kind == 'norm':
            self.register_buffer('gamma', folded_norm_gamma(target, weights['weight']))
            self.export_adjustments.append(dict(kind='FOLD_ZERO_CENTERED_GAMMA',
                expression='weight + 1.0', arithmetic='BOUND_WEIGHT_DTYPE_BEFORE_TRACING',
                source_forward='AST_MATCHED_CPRO_NPU_RMS_NORM',
                weight=tensor_metadata(weights['weight']), gamma=tensor_metadata(self.gamma)))
        elif self.kind == 'norm_gated':
            self.op = target.Qwen3_5MoeRMSNormGated(weights['weight'].numel(), eps=self.attrs['epsilon'])
            self.op.weight = torch.nn.Parameter(weights['weight'], requires_grad=False)
        else:
            for name, value in weights.items():
                self.register_buffer(name, value)
            if self.kind == 'conv':
                self.export_adjustments.append(dict(kind='EXPLICIT_CONV2D_SINGLETON_HEIGHT',
                    input_axis=2, weight_axis=2, output_squeeze_axis=2,
                    groups=self.attrs['groups'], arithmetic='ORIGINAL_INPUT_AND_WEIGHT_DTYPES'))
        # Explicit forward arguments make tracing inspectable. The export hook
        # records identity before AOT input renaming and Data renumbering.
        names = [f'probe_input_{i:03d}' for i in range(len(self.names))]
        namespace = {}
        exec('def forward(self, ' + ', '.join(names) + '):\n    return self.compute(' + ', '.join(names) + ')\n', namespace)
        self.__class__ = type('NamedRegion', (Region,), {'forward': namespace['forward']})
        self.eval()

    def compute(self, *values):
        x = dict(zip(self.names, values))
        k, a = self.kind, self.attrs
        if hasattr(self, 'source_forward'):
            if k == 'gates':
                args = (x['a'], x['b'])
            elif k == 'gdn':
                if hasattr(self, 'gdr_valid_rows'):
                    q, key, v, g, beta = (x[name] for name in ('query', 'key', 'value', 'g', 'beta'))
                    state = x['initial_state'].to(torch.float16)
                    if self.single_step:
                        from .whole_gdn import single_token_gdr
                        valid_inputs = (t[:, :self.gdr_valid_rows] for t in (q, key, v, g, beta))
                        core, state = single_token_gdr(*valid_inputs, a['chunk_size'], state, self.step_length,
                            a['output_final_state'], a['use_qk_l2norm_in_kernel'])
                    else:
                        core, state = self.source_forward(self, q, key, v, g, beta, state, x['effective_length'], q.shape[1])
                    return core[:, :self.gdr_valid_rows] if self.output_index == 0 else state.to(torch.float16)
                args = tuple(x[n] for n in ('query', 'key', 'value', 'g', 'beta', 'initial_state', 'effective_length')) + (x['query'].shape[1],)
            elif k == 'convstate':
                args = (x['mixed_qkv'], x['old_conv_state'], x['linear_update_tensor'])
            elif k == 'silu':
                args = (x['x'],)
            elif k == 'state_cast':
                args = (x['state'],)
            elif k == 'add':
                args = (x['residual'], x['branch'])
            elif k == 'attention':
                args = tuple(x[n] for n in ('query', 'key', 'value', 'atten_mask', 'block_table')) + (
                    a['actual_seq_lengths_q'], a['actual_seq_lengths_kv'], x['all_seq_lengths_q'], True)
            elif k == 'attention_post':
                b, t, _ = x['gate'].shape
                args = (x['attention_nz'], (b, a['num_heads'], t, a['head_dim']), (b, t), x['gate'])
            else:
                raise ValueError('unsupported source fragment: ' + k)
            value = self.source_forward(self, *args)
            out = value if k in ('gates', 'gdn', 'convstate') else (value,)
        elif k in ('qlinear', 'linear'):
            out = (self.op(x['x']),)
        elif k == 'norm':
            out = (self.npu.npu_rms_norm(x['x'], self.gamma, a['epsilon'])[0].type_as(x['x']),)
        elif k == 'norm_gated':
            out = (self.op(x['x'], x['gate']),)
        elif k == 'convstate':
            out = torch.ops.my_ops.linear_attention_convstate_update(
                x['mixed_qkv'], x['old_conv_state'], x['linear_update_tensor'])
        elif k == 'conv':
            out = (F.conv2d(x['padded'].unsqueeze(2), self.weight.unsqueeze(2),
                           getattr(self, 'bias', None), groups=a['groups']).squeeze(2),)
        elif k == 'silu':
            out = (F.silu(x['x']),)
        elif k == 'gates':
            out = (-self.A_log.float().exp() * F.softplus(x['a'].float() + self.dt_bias), x['b'].sigmoid())
        elif k == 'repeat_qk':
            out = (self.target.repeat_qk(x['q'], a['ratio']), self.target.repeat_qk(x['k'], a['ratio']))
        elif k == 'gdn':
            out = self.npu.npu_chunk_gated_delta_rule(x['query'], x['key'], x['value'].contiguous(),
                g=x['g'], beta=x['beta'], initial_state=x['initial_state'],
                effective_length=x['effective_length'], chunk_size=a['chunk_size'],
                output_final_state=int(a['output_final_state']),
                use_qk_l2norm_in_kernel=int(a['use_qk_l2norm_in_kernel']))
        elif k == 'state_cast':
            out = (x['state'].to(getattr(torch, a['dtype'])),)
        elif k == 'rope':
            if getattr(self, 'rope_precision', 'source') == 'fp32':
                from .rope import apply_fp32
                out = apply_fp32(self.target, x['q'], x['k'], x['cos'], x['sin'], self.rope_output_dtypes)
            else:
                out = self.target.apply_rotary_pos_emb(x['q'], x['k'], x['cos'], x['sin'])
        elif k == 'cache':
            cache = x['cache'].clone()
            if a.get('cache_indexing', 'kernel') == 'kernel':
                blocks, offsets = x['target_block'], x['offset_in_block']
            else:
                from .candidates import cache_indices
                blocks, offsets = cache_indices(x['cache_position'], x['block_table'], a['block_size'], a['cache_indexing'])
            self.npu.npu_cache_update_(cache, x['updates'], blocks, offsets)
            out = (cache,)
        elif k == 'attention':
            out = (self.npu.adn_fused_infer_attention(query=x['query'], key=[x['key']], value=[x['value']],
                atten_mask=x['atten_mask'], block_table=x['block_table'], pse_shift=x['all_seq_lengths_q'], **a),)
        elif k == 'attention_post':
            b, t, width = x['gate'].shape
            h, d = a['num_heads'], a['head_dim']
            nz = x['attention_nz'].reshape(b, h, d//16, t, 16)
            nd = nz.transpose(2, 3).contiguous().reshape(b, h, t, d).transpose(1, 2)
            out = (nd.reshape(b, t, width) * x['gate'].sigmoid(),)
        elif k == 'add':
            out = (x['residual'] + x['branch'],)
        else:
            raise ValueError('unsupported CPro region: ' + k)
        return out[self.output_index]

    def transform_nz_2_nd(self, value):
        return self.target.Qwen3_5MoeAttention.transform_nz_2_nd(self, value)


def source_bindings(graph, specs, actual_identities=None, output_count=1):
    result = []
    for node in graph['op']:
        if node['type'] != 'Data':
            continue
        indices = ({actual_identities[node['name']]} if actual_identities is not None and node['name'] in actual_identities
                   else set() if actual_identities is not None
                   else set(int(v) for v in re.findall(r'probe_input_(\d{3})(?!\d)', node['name'])))
        if len(indices) != 1:
            raise ValueError('source Data name does not identify one explicit argument: ' + node['name'])
        index = indices.pop()
        if index >= len(specs) or any(r['fixture_index'] == index for r in result):
            raise ValueError('source Data argument is out of range or duplicated')
        result.append(dict(specs[index], name=node['name'], fixture_index=index))
    if not result:
        raise ValueError('region has no runtime input')
    nets = [n for n in graph['op'] if n['type'] == 'NetOutput']
    if len(nets) != 1 or len([v for v in nets[0].get('input', []) if not v.endswith(':-1')]) != output_count:
        raise ValueError(f'diagnostic graph must expose exactly {output_count} ordered tensor outputs')
    return result


def check_adjusted_graph(kind, graph):
    """Verify the intended source graph change before accepting a new AIR."""
    from qwen35_moe_dflash.target.original_air_roundtrip import _attributes
    nodes = {n['name']: n for n in graph['op']}

    def producer(edge):
        name, slot = edge.rsplit(':', 1)
        if slot != '0':
            raise ValueError('adjusted graph requires output slot 0: ' + edge)
        return nodes[name]

    def shape(node, depth=0):
        if depth > 8:
            return []
        descs = node.get('output_desc', [])
        dims = descs[0].get('shape', {}).get('dim', []) if descs else []
        if dims:
            return dims
        if node['type'] == 'Unsqueeze' and _attributes(node).get('axes', {}).get('list', {}).get('i') == [2]:
            source = shape(producer(node['input'][0]), depth + 1)
            if len(source) == 3:
                return source[:2] + [1] + source[2:]
        return []

    if kind == 'norm':
        norms = [n for n in nodes.values() if n['type'] == 'RmsNorm']
        if (len(norms) != 1 or any(n['type'] == 'Add' for n in nodes.values())
                or producer(norms[0]['input'][1])['type'] not in ('Const', 'FileConstant')):
            raise ValueError('ordinary Norm AIR must use one RmsNorm with constant gamma and no Add')
        return dict(status='PASSED', scope='SOURCE_GRAPH_ONLY', gamma='CONSTANT_NO_ADD')
    if kind == 'conv':
        convs = [n for n in nodes.values() if n['type'] == 'Conv2D']
        if len(convs) != 1:
            raise ValueError('explicit Conv1d lowering must export one Conv2D')
        shapes = [shape(producer(e)) for e in convs[0]['input'][:2]]
        if len(shapes) != 2 or any(len(s) != 4 or s[2] != 1 or min(s) <= 0 for s in shapes):
            raise ValueError(f'Conv2D input and filter must have explicit singleton height: {shapes}')
        return dict(status='PASSED', scope='SOURCE_DESCRIPTORS_OR_EXPLICIT_UNSQUEEZE_EDGES',
                    input_shape=shapes[0], filter_shape=shapes[1])
    return dict(status='NOT_APPLICABLE')


def export_one(capture, region_id, output_index, output, source_root, converter_script, torchair_pb2=None, target_config=None,
               *, qlinear_scales='source', cache_indexing='kernel', rope_precision='source', gdn_state_dtype='source',
               gdn_convstate='kernel', full_attention_reference='math-replay', gdr_mode='chunk', gdn_trace_layer=None,
               gdr_inputs=False):
    root = new_output(output)
    result = dict(artifact_kind=KIND, status='EXPORTING', region=region_id, output_index=output_index,
        model_scope='ONE_CPRO_REGION_ONE_OUTPUT; NO_FULL_MODEL_OR_LAYER_CONSTRUCTION')
    try:
        if (qlinear_scales not in ('source', 'fp16') or cache_indexing not in ('kernel', 'float', 'integer')
                or rope_precision not in ('source', 'fp32') or gdn_state_dtype not in ('source', 'float16', 'float32')
                or gdn_convstate not in ('kernel', 'reference') or gdr_mode not in ('chunk', 'single-token')
                or full_attention_reference not in ('math-replay', 'captured-audited')):
            raise ValueError('unknown module candidate')
        capture = Path(capture).resolve()
        manifest = json.loads((capture/'manifest.json').read_text())
        if manifest.get('artifact_kind') != KIND or manifest.get('status') != 'CAPTURED':
            raise ValueError('requires a completed original-inference capture')
        regions = manifest['regions']
        if region_id.endswith('_gdn_layer'):
            from .whole_gdn import assemble
            regions = assemble(manifest, trace_layer=gdn_trace_layer, trace_inputs=gdr_inputs)
        if region_id.endswith('_full_attention_layer'):
            from .whole_attention import assemble
            regions = assemble(manifest, capture=capture, reference=full_attention_reference)
        matches = [r for r in regions if r['id'] == region_id]
        if len(matches) != 1:
            raise ValueError('region must identify one captured invocation')
        record = matches[0]
        whole_gdn = record['kind'] == 'gdn_layer'
        segment_gdr = record['kind'] == 'gdn' and 'gdr_valid_rows' in record
        whole_attention = record['kind'] == 'full_attention_layer'
        if full_attention_reference != 'math-replay' and not whole_attention:
            raise ValueError('--full-attention-reference requires full_attention_layer')
        whole = whole_gdn or whole_attention
        single_token = gdr_mode == 'single-token'
        trace_tail = gdn_trace_layer is not None
        if gdr_inputs and (not trace_tail or gdn_state_dtype != 'float16'):
            raise ValueError('--gdr-inputs requires --gdn-trace-layer and --gdn-state-dtype float16')
        if trace_tail and (not whole_gdn or record['layer'] != gdn_trace_layer or not single_token):
            raise ValueError('--gdn-trace-layer requires the selected single-token GDN Prefill layer')
        if single_token and (not (whole_gdn or segment_gdr) or record['phase'] != 'prefill'):
            raise ValueError('--gdr-mode single-token requires a whole Prefill GDN or qualified segmented GDR')
        gdr_calls = record.get('gdr_valid_rows', record['sequence_length']) if single_token else 1
        if gdn_convstate != 'kernel' and not whole_gdn:
            raise ValueError('--gdn-convstate reference requires a whole gdn_layer region')
        if whole_gdn:
            if output_index is not None or gdn_state_dtype == 'source' or qlinear_scales != 'fp16' or not target_config:
                raise ValueError('whole GDN requires all outputs, --gdn-state-dtype float16|float32, --qlinear-scales fp16 and --target-config')
            result['model_scope'] = 'ONE_COMPLETE_GATED_DELTANET_LAYER; THREE_OUTPUTS; NO_MOE_OR_DECODER_LAYER'
            if trace_tail:
                result['model_scope'] = 'ONE_COMPLETE_GATED_DELTANET_LAYER_WITH_THREE_ADDITIONAL_TAIL_OUTPUTS'
                if gdr_inputs:
                    result['model_scope'] = 'ONE_COMPLETE_GATED_DELTANET_LAYER_WITH_TAIL_AND_GDR_INPUTS'
        elif whole_attention:
            if output_index is not None or qlinear_scales != 'fp16' or rope_precision != 'fp32' or not target_config:
                raise ValueError('whole FullAttention requires all outputs, --qlinear-scales fp16 --rope-precision fp32 and --target-config')
            result['model_scope'] = 'ONE_COMPLETE_FULL_ATTENTION_MODULE; THREE_OUTPUTS; NO_RESIDUAL_MOE_OR_DECODER_LAYER'
            result['reference_basis'] = record['reference_basis']
            result['reference_evidence'] = record['reference_evidence']
            from .whole_attention import reference_scope
            result['reference_scope'] = reference_scope(record['reference_basis'])
        elif output_index is None or not 0 <= output_index < len(record['outputs']):
            raise ValueError('single boundary requires a valid --output-index')
        goldens = record['outputs'] if whole else [record['outputs'][output_index]]
        for golden in goldens:
            checked(capture, golden)
        weights = torch.load(checked(capture, record['weights']), map_location='cpu', weights_only=True) if 'weights' in record else {}
        inputs = [torch.from_numpy(read_array(capture, spec).copy()) for spec in record['inputs']]
        input_paths = [checked(capture, spec) for spec in record['inputs']]
        fixture_specs = record['inputs']
        if segment_gdr:
            from ..prefill_t64.segment_gdr import verify_capture
            if gdn_state_dtype not in ('source', 'float16'):
                raise ValueError('segmented GDR replay requires the FP16 CPro state path')
            verify_capture(capture, manifest, record, inputs)
            gdn_state_dtype = 'float16'
        elif single_token:
            lengths = inputs[4]
            if (inputs[0].shape[:2] != (1, gdr_calls) or gdr_calls < 2 or lengths.shape != (1,)
                    or lengths.dtype != torch.int16 or lengths.item() != gdr_calls):
                raise ValueError('single-token GDR requires an unpadded B=1 Prefill and effective_length=[T] (INT16)')
        if whole_gdn and gdn_convstate == 'reference':
            from .convstate_reference import validate_capture
            result['convstate_reference_check'] = validate_capture(record, capture)
        sys.path.insert(0, str(Path(source_root).resolve()))
        import torch_npu
        import torchair
        target = importlib.import_module('models.modeling_qwen3_5_moe_hiai_cpro')
        # This defines the original GE converters; __main__ is never executed.
        converters = import_file(Path(converter_script).resolve())
        torch._dynamo.config.suppress_errors = False
        from .source_regions import bind_region, qlinear_factory
        # Follow the function actually called by the exporter, including its
        # globals. A same-named class in modeling_*_cpro is not enough.
        factory, factory_evidence = qlinear_factory(converters)
        source_directory = Path(source_root).resolve()
        for source_file in (target.__file__, factory_evidence['replacement']['file'], inspect.getfile(factory)):
            if source_directory not in Path(source_file).resolve().parents:
                raise ValueError('loaded original module is outside --source-root: ' + str(source_file))
        result['captured_weights'] = {name: tensor_metadata(value) for name, value in weights.items()}
        from .target_weights import load_weights
        weights, result['target_weights'] = load_weights(record, weights, target, converters, target_config, source_root)
        from .candidates import adapt_qlinear, prepare_cache_inputs, validate_cache_source, check_graph, qlinear_ge_scales
        from .cache_index_export import cache_ge_integer_mul
        cache_mode = cache_indexing if record['kind'] == 'cache' else 'kernel'
        if cache_mode != 'kernel':
            source = validate_cache_source(target)
            record, inputs, input_paths, candidate = prepare_cache_inputs(capture, manifest, record, root, cache_mode)
            fixture_specs = record['inputs']
            candidate['source'] = source
            result['candidate'] = candidate
        if whole_gdn:
            from .whole_gdn import build
            model, result['source_contract'] = build(record, capture, target, converters, factory,
                target_config, source_root, root, gdn_state_dtype, gdn_convstate, gdr_mode,
                trace_tail=trace_tail, trace_inputs=gdr_inputs)
            result['target_weights'] = dict(status='ORIGINAL_CHECKPOINT_BOUND')
            result['candidate'] = dict(kind='WHOLE_GDN_' + ('FP16' if gdn_state_dtype == 'float16' else 'FP32') + '_STATE',
                state_dtype=gdn_state_dtype, golden_state_dtype='float32', convstate_mode=gdn_convstate,
                qlinear_scales='fp16', gdr_mode=gdr_mode, gdr_calls=gdr_calls, reference='ORIGINAL_CAPTURE_UNCHANGED')
        elif whole_attention:
            from .whole_attention import build, prepare_cache_position_input, INPUTS
            model, result['source_contract'] = build(record, capture, target, converters, factory,
                target_config, source_root, root)
            spec, tensor, path, carrier = prepare_cache_position_input(record, capture, root)
            slot = INPUTS.index('new_kv_cache_pos')
            fixture_specs = list(record['inputs'])
            fixture_specs[slot], inputs[slot], input_paths[slot] = spec, tensor, path
            result['source_contract']['cache_position_carrier'] = carrier
            result['target_weights'] = dict(status='ORIGINAL_CHECKPOINT_BOUND')
            result['candidate'] = dict(kind='WHOLE_FULL_ATTENTION', qlinear_scales='fp16',
                rope_precision='fp32', reference=record['reference_basis'],
                cache_position_input='EXACT_FP32_HOST_CARRIER')
        else:
            model = Region(record, weights, target, torch_npu, output_index, factory)
            result['source_contract'] = bind_region(model, target, record)
            if segment_gdr:
                model.gdr_valid_rows, model.single_step = record['gdr_valid_rows'], single_token
                if single_token:
                    model.register_buffer('step_length', torch.tensor([1], dtype=torch.int16))
                result['candidate'] = dict(kind='SEGMENT_GDR_FP16_STATE', physical_rows=64, valid_rows=model.gdr_valid_rows,
                    gdr_mode=gdr_mode, gdr_calls=gdr_calls, state_dtype=gdn_state_dtype)
                result['reference_basis'] = 'CPU_CHECKED_310P_SEGMENT_GDR_VALID19'
        result['source_contract']['weight_scope'] = result['target_weights']['status']
        result['qlinear_resolution'] = factory_evidence
        result['exported_weights'] = {name: tensor_metadata(value) for name, value in weights.items()}
        qlinear_fp16 = record['kind'] == 'qlinear' and qlinear_scales == 'fp16'
        if qlinear_fp16:
            result['candidate'] = adapt_qlinear(model, root)
            result['exported_weights']['scale'] = tensor_metadata(model.op.scale)
        rope_fp32 = record['kind'] == 'rope' and rope_precision == 'fp32'
        if record['kind'] == 'rope':
            from .rope import prepare
            result['rope_contract'] = prepare(model, record, inputs, capture, rope_precision)
            if rope_fp32:
                result['candidate'] = dict(kind='ROPE_ROTATION_FP32',
                    arithmetic='CAST_INPUTS_TO_FP32; SOURCE_HELPER; RESTORE_CAPTURED_OUTPUT_DTYPE',
                    reference='ORIGINAL_CAPTURE_UNCHANGED',
                    coefficient_generation='OUTSIDE_CAPTURE; NO_SINCOS_RECOMPUTATION')
        if 'candidate' in result:
            result['source_contract']['candidate'] = result['candidate']
            result['source_contract']['equivalence'] = 'EXPLICIT_EXPERIMENT; ORIGINAL_GOLDEN_UNCHANGED'
            if cache_mode != 'kernel':
                result['source_contract']['input_scope'] = 'CAPTURED_CACHE_UPDATES_TABLE; UNIQUELY_RECONSTRUCTED_POSITION'
                result['source_contract']['source'].update(mode='INDEX_COMPUTATION_PLUS_KERNEL',
                    excluded='RefData alias, original high-level position capture and inter-call feedback')
        result['export_adjustments'] = model.export_adjustments
        from qwen35_moe_dflash.draft.incremental.torch_export import _producer_schema, _schema
        from qwen35_moe_dflash.target.original_air_roundtrip import _decode
        descriptor, metadata = _producer_schema(torchair_pb2)
        (root/'air-schema.pb').write_bytes(descriptor)
        old = Path.cwd()
        from .module_contracts import gdn_state_contract
        try:
            os.chdir(root)
            with (torch.no_grad(), export_input_identity(inputs) as input_identity,
                  qlinear_ge_scales(qlinear_fp16 or whole, expected_calls=5 if whole_gdn else 4 if whole_attention else 1) as ge_adapter,
                  gdn_state_contract(whole_gdn or segment_gdr, gdn_state_dtype, gdr_calls) as gdn_adapter,
                  cache_ge_integer_mul(cache_mode == 'integer', record['attrs'].get('block_size')) as cache_adapter):
                if ge_adapter is not None:
                    result['candidate']['ge_adapter'] = ge_adapter
                if cache_adapter is not None:
                    result['candidate']['ge_adapter'] = cache_adapter
                if gdn_adapter is not None:
                    result['candidate']['state_ge_adapter'] = gdn_adapter
                torchair.dynamo_export(*inputs, model=model, export_path='weight', export_name='export', dynamic=False)
        finally:
            os.chdir(old)
        (root/'weight/export.air').replace(root/'export.air')
        schema = _schema(descriptor, metadata)
        _, _, graph = _decode(schema, (root/'export.air').read_bytes())
        result['export_graph_check'] = check_adjusted_graph(record['kind'], graph)
        if 'candidate' in result and not whole_attention:
            if whole_gdn:
                from .module_contracts import check_gdn_source
                result['candidate_graph_check'] = check_gdn_source(graph, schema, record['phase'], gdn_state_dtype,
                    gdn_convstate, expected_calls=gdr_calls, export_root=root,
                    readouts=record['outputs'][3:] if trace_tail else ())
            elif segment_gdr:
                from .module_contracts import check_gdr_nodes
                nodes = check_gdr_nodes(graph, schema, record['phase'], gdn_state_dtype, gdr_calls, root)
                result['candidate_graph_check'] = dict(status='PASSED', **nodes)
                result['output_contract'] = dict(required_dtypes={0: 'float16'})
            elif rope_fp32:
                from .rope import check_graph as check_rope_graph
                result['candidate_graph_check'] = check_rope_graph(graph, schema, golden['dtype'])
            else:
                result['candidate_graph_check'] = check_graph(graph, schema, qlinear_fp16, cache_mode)
        from .source_regions import internal_boundaries
        result['internal_boundaries'] = internal_boundaries(graph, schema)
        fixture = root/'fixture'
        fixture.mkdir()
        specs = []
        for i, spec in enumerate(fixture_specs):
            path = fixture/f'input_{i}.bin'
            shutil.copyfile(input_paths[i], path)
            specs.append(dict(spec, **identity(path, root)))
        copied_goldens = []
        for i, golden in enumerate(goldens):
            path = fixture/(f'golden_{i}.bin' if whole else 'golden.bin')
            shutil.copyfile(checked(capture, golden), path)
            copied_goldens.append(dict(golden, **identity(path, root)))
        bindings = source_bindings(graph, specs, input_identity['nodes'], len(copied_goldens))
        if segment_gdr and not single_token:
            gdr, = [n for n in result['internal_boundaries']['nodes'] if n['type'] == 'ChunkGatedDeltaRule']
            length_port, = [p for p in gdr['inputs'] if p['port'] == 'effective_length']
            data, = [b for b in bindings if fixture_specs[b['fixture_index']]['name'] == 'effective_length']
            source = length_port['producer']['chain'][-1]
            if source['type'] != 'Data' or source['name'] != data['name'] or source['output_slot'] != 0:
                raise ValueError('chunk GDR must consume the captured effective_length input')
            result['candidate_graph_check']['effective_length_data'] = data['name']
        if whole_attention:
            from .whole_attention import check_graph as check_whole_attention
            result['candidate_graph_check'] = check_whole_attention(graph, schema, record, bindings)
            core = next(r for r in record['components'] if r['point'] == 'fused_attention')
            table = next(s for s in core['inputs'] if s['name'] == 'block_table')
            result['full_attention_contract'] = dict(sequence_length=record['sequence_length'],
                block_size=record['attrs']['block_size'], block_table=read_array(capture, table).tolist(),
                cache_input_fixture_indices={'1': 3, '2': 4},
                scope='FRESH_PREFILL; CAPTURED_BLOCK_TABLE; ACTIVE_ROWS_AND_UNTOUCHED_CACHE',
                block_table_source=table)
            result['captured_goldens'] = []
            for i, spec in enumerate(record['captured_outputs']):
                path = fixture/f'captured_golden_{i}.bin'
                shutil.copyfile(checked(capture, spec), path)
                result['captured_goldens'].append(dict(spec, **identity(path, root)))
        if record['kind'] == 'attention':
            from .module_contracts import attention_contract
            contract = attention_contract(record, inputs, graph, schema, bindings)
            result['attention_contract'] = contract
            if 'audited_valid_query_rows' in record:
                from .attention_reference import unpack
                from .common import metrics
                valid = record['audited_valid_query_rows']
                if (not 0 < valid <= contract['physical_query_length'] or contract['all_seq_lengths_q'] != [valid]
                        or golden['shape'] != contract['query_shape'] or golden['dtype'] != 'float16'):
                    raise ValueError('valid-row golden does not match the exported attention ABI')
                heads = record['attrs']['num_heads']
                actual = unpack(read_array(capture, golden), heads)[:, :, :valid]
                cpu_reference = read_array(capture, manifest['cpu_valid_bnsd'])
                if not (metrics(actual, cpu_reference)['passed']
                        and metrics(actual[:, :, -1], cpu_reference[:, :, -1])['passed']):
                    raise ValueError('saved 310P attention output failed its same-call CPU check')
                contract.update(audited_valid_query_rows=valid, num_heads=heads)
                result['output_contract'] = dict(required_dtypes={0: golden['dtype']})
        if whole:
            result['goldens'] = copied_goldens
            result['output_contract'] = dict(order=[s['name'] for s in copied_goldens],
                source_edges=next(n for n in graph['op'] if n['type'] == 'NetOutput')['input'],
                binding='EXPLICIT_PYTHON_TUPLE_AND_NETOUTPUT_SLOT_ORDER',
                required_dtypes={2: gdn_state_dtype} if whole_gdn else {i: s['dtype'] for i, s in enumerate(copied_goldens)})
            if trace_tail:
                result['output_contract']['required_dtypes'].update(
                    {i: s.get('board_dtype', s['dtype']) for i, s in enumerate(copied_goldens) if i != 2})
        result.update(status='EXPORTED', air=identity(root/'export.air', root),
            schema={k: metadata[k] for k in ('schema_file', 'schema_package')},
            source_inputs=bindings, golden=copied_goldens[0], export_input_identity=input_identity,
            capture_manifest_sha256=sha(capture/'manifest.json'),
            source_files=[dict(path=str(Path(p).resolve()), sha256=sha(p))
                          for p in dict.fromkeys((target.__file__, converter_script,
                              factory_evidence['replacement']['file'], inspect.getfile(factory)))],
            qlinear_factory=factory_evidence['factory'],
            weights=record.get('weights'), input_scope=result['source_contract']['input_scope'],
            graph_types=sorted({n['type'] for n in graph['op']}),
            torch_version=torch.__version__, torchair_version=getattr(torchair, '__version__', 'unknown'))
    except Exception as exc:
        result.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save_json(root/'export.json', result)
    return result
