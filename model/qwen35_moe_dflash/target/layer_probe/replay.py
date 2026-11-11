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


@contextmanager
def export_input_identity(example_inputs):
    """Observe the exporter's real input identities before Data renumbering.

    Hook verified against Ascend/torchair export_utils._convert_data_to_const.
    The weight-externalization decision and graph remain unchanged.
    """
    module = importlib.import_module('torchair._utils.export_utils')
    original = getattr(module, '_is_weight_externalized', None)
    if not callable(original) or tuple(inspect.signature(original).parameters) != ('inputs', 'weight_name', 'export_graph'):
        raise ValueError('installed TorchAir input-binding hook differs; inspect its export_utils.py')
    by_id = {id(t): i for i, t in enumerate(example_inputs)}
    if len(by_id) != len(example_inputs):
        raise ValueError('replay input objects must have distinct identities')
    evidence = dict(calls=0, nodes={}, policy='ACTUAL_EXPORT_INPUT_OBJECT_IDENTITY', restored=False)
    def observed(inputs, weight_name, export_graph):
        evidence['calls'] += 1
        if evidence['calls'] != 1:
            raise ValueError('one region must export as one graph')
        for i, value in enumerate(inputs):
            if id(value) in weight_name:
                continue
            if id(value) not in by_id:
                raise ValueError(f'exporter supplied unbound runtime input {i}, shape={getattr(value, "shape", None)}, dtype={getattr(value, "dtype", None)}; no shape/value guessing permitted')
            if i >= len(export_graph.op) or export_graph.op[i].type != 'Data':
                raise ValueError('export input is not its original Data node')
            name = export_graph.op[i].name
            if name in evidence['nodes']:
                raise ValueError('duplicate source Data name')
            evidence['nodes'][name] = by_id[id(value)]
        return original(inputs, weight_name, export_graph)
    module._is_weight_externalized = observed
    try:
        yield evidence
        if evidence['calls'] != 1 or not evidence['nodes']:
            raise ValueError('TorchAir did not expose one identity-bound region graph')
    finally:
        module._is_weight_externalized = original
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
            out = self.target.apply_rotary_pos_emb(x['q'], x['k'], x['cos'], x['sin'])
        elif k == 'cache':
            cache = x['cache'].clone()
            self.npu.npu_cache_update_(cache, x['updates'], x['target_block'], x['offset_in_block'])
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


def source_bindings(graph, specs, actual_identities=None):
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
    if len(nets) != 1 or len([v for v in nets[0].get('input', []) if not v.endswith(':-1')]) != 1:
        raise ValueError('each diagnostic graph must expose exactly one tensor output')
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


def export_one(capture, region_id, output_index, output, source_root, converter_script, torchair_pb2=None, target_config=None):
    root = new_output(output)
    result = dict(artifact_kind=KIND, status='EXPORTING', region=region_id, output_index=output_index,
        model_scope='ONE_CPRO_REGION_ONE_OUTPUT; NO_FULL_MODEL_OR_LAYER_CONSTRUCTION')
    try:
        capture = Path(capture).resolve()
        manifest = json.loads((capture/'manifest.json').read_text())
        if manifest.get('artifact_kind') != KIND or manifest.get('status') != 'CAPTURED':
            raise ValueError('requires a completed original-inference capture')
        matches = [r for r in manifest['regions'] if r['id'] == region_id]
        if len(matches) != 1:
            raise ValueError('region must identify one captured invocation')
        record = matches[0]
        golden = record['outputs'][output_index]
        checked(capture, golden)
        weights = torch.load(checked(capture, record['weights']), map_location='cpu', weights_only=True) if 'weights' in record else {}
        inputs = [torch.from_numpy(read_array(capture, spec).copy()) for spec in record['inputs']]
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
        model = Region(record, weights, target, torch_npu, output_index, factory)
        result['source_contract'] = bind_region(model, target, record)
        result['source_contract']['weight_scope'] = result['target_weights']['status']
        result['qlinear_resolution'] = factory_evidence
        result['exported_weights'] = {name: tensor_metadata(value) for name, value in weights.items()}
        result['export_adjustments'] = model.export_adjustments
        from qwen35_moe_dflash.draft.incremental.torch_export import _producer_schema, _schema
        from qwen35_moe_dflash.target.original_air_roundtrip import _decode
        descriptor, metadata = _producer_schema(torchair_pb2)
        (root/'air-schema.pb').write_bytes(descriptor)
        old = Path.cwd()
        try:
            os.chdir(root)
            with torch.no_grad(), export_input_identity(inputs) as input_identity:
                torchair.dynamo_export(*inputs, model=model, export_path='weight', export_name='export', dynamic=False)
        finally:
            os.chdir(old)
        (root/'weight/export.air').replace(root/'export.air')
        schema = _schema(descriptor, metadata)
        _, _, graph = _decode(schema, (root/'export.air').read_bytes())
        result['export_graph_check'] = check_adjusted_graph(record['kind'], graph)
        from .source_regions import internal_boundaries
        result['internal_boundaries'] = internal_boundaries(graph, schema)
        fixture = root/'fixture'
        fixture.mkdir()
        specs = []
        for i, spec in enumerate(record['inputs']):
            path = fixture/f'input_{i}.bin'
            shutil.copyfile(checked(capture, spec), path)
            specs.append(dict(spec, **identity(path, root)))
        path = fixture/'golden.bin'
        shutil.copyfile(checked(capture, golden), path)
        golden = dict(golden, **identity(path, root))
        bindings = source_bindings(graph, specs, input_identity['nodes'])
        result.update(status='EXPORTED', air=identity(root/'export.air', root),
            schema={k: metadata[k] for k in ('schema_file', 'schema_package')},
            source_inputs=bindings, golden=golden, export_input_identity=input_identity,
            capture_manifest_sha256=sha(capture/'manifest.json'),
            source_files=[dict(path=str(Path(p).resolve()), sha256=sha(p))
                          for p in dict.fromkeys((target.__file__, converter_script,
                              factory_evidence['replacement']['file'], inspect.getfile(factory)))],
            qlinear_factory=factory_evidence['factory'],
            weights=record.get('weights'), input_scope='GOLDEN_BOUNDARY_INPUTS',
            graph_types=sorted({n['type'] for n in graph['op']}),
            torch_version=torch.__version__, torchair_version=getattr(torchair, '__version__', 'unknown'))
    except Exception as exc:
        result.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save_json(root/'export.json', result)
    return result
