"""One CPro region per process; never construct a complete model or decoder layer."""
from __future__ import annotations
from contextlib import contextmanager
import importlib
import inspect
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys

import torch
import torch.nn.functional as F
from .common import KIND, checked, identity, new_output, read_array, save_json, sha


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
        if self.kind == 'qlinear':
            self.op = (qlinear_class or target.QLinear)(W_q=weights['W_q'], scale=weights['scale'], idx=0)
        elif self.kind == 'linear':
            w = weights['weight']
            self.op = torch.nn.Linear(w.shape[1], w.shape[0], bias='bias' in weights)
            self.op.weight = torch.nn.Parameter(w, requires_grad=False)
            if 'bias' in weights:
                self.op.bias = torch.nn.Parameter(weights['bias'], requires_grad=False)
        elif self.kind in ('norm', 'norm_gated'):
            cls = target.Qwen3_5MoeRMSNorm if self.kind == 'norm' else target.Qwen3_5MoeRMSNormGated
            self.op = cls(weights['weight'].numel(), eps=self.attrs['epsilon'])
            self.op.weight = torch.nn.Parameter(weights['weight'], requires_grad=False)
        else:
            for name, value in weights.items():
                self.register_buffer(name, value)
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
        if k in ('qlinear', 'linear', 'norm'):
            out = (self.op(x['x']),)
        elif k == 'norm_gated':
            out = (self.op(x['x'], x['gate']),)
        elif k == 'convstate':
            out = torch.ops.my_ops.linear_attention_convstate_update(
                x['mixed_qkv'], x['old_conv_state'], x['linear_update_tensor'])
        elif k == 'conv':
            out = (F.conv1d(x['padded'], self.weight, getattr(self, 'bias', None), groups=a['groups']),)
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


def export_one(capture, region_id, output_index, output, source_root, converter_script, torchair_pb2=None):
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
        # The original export's replacement utility chooses the QLinear class;
        # it need not be the similarly named class in the CPro model module.
        utilities = importlib.import_module('utils')
        model = Region(record, weights, target, torch_npu, output_index, utilities.QLinear)
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
        _, _, graph = _decode(_schema(descriptor, metadata), (root/'export.air').read_bytes())
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
                          for p in (target.__file__, converter_script, utilities.__file__, inspect.getfile(utilities.QLinear))],
            qlinear_factory=utilities.QLinear.__module__ + '.' + utilities.QLinear.__qualname__,
            weights=record.get('weights'), input_scope='GOLDEN_BOUNDARY_INPUTS',
            graph_types=sorted({n['type'] for n in graph['op']}),
            torch_version=torch.__version__, torchair_version=getattr(torchair, '__version__', 'unknown'))
    except Exception as exc:
        result.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save_json(root/'export.json', result)
    return result
