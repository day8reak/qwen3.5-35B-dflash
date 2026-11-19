"""Export each original CPro attention module on its own observed NZ module input."""
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch

from ..layer_probe.common import KIND, checked, identity, read_array, save_json, sha
from ..layer_probe.replay import export_input_identity, source_bindings
from ..layer_probe.steps import working_directory

from .segment_export import MODULE_BASIS


def prepare_modules(fixture, capture, job):
    from ..layer_probe.common import metrics
    cpu_root = fixture / 'module-cpu'
    audit = json.loads((cpu_root / 'report.json').read_text())
    capture_hash = sha(fixture / 'manifest.json')
    if (audit['scope'] != 'SEGMENT_VALID19_FIXED_INPUT_ATTENTION_MODULE_HIDDEN'
            or audit['capture_sha256'] != capture_hash):
        raise ValueError('Expected the fixed-input module CPU audit for this same capture')
    cases = []
    for i in range(4):
        record = capture['attention_modules'][str(i)]
        expected_member = 'linear_attn' if i < 3 else 'self_attn'
        if record['module'] != expected_member:
            raise ValueError(f'Unexpected attention module at L{i}')
        point = 'gdn_layer' if i < 3 else 'full_attention_layer'
        name = f'l{i:02}_c00_{point}_hidden_o0'
        case = dict(id=name, layer=i, call=0, point=point, kind='segment_module_hidden',
                    ordinal=i, phase='prefill', output_name='golden_hidden')
        native = read_array(fixture, record['golden_hidden'])
        cpu = read_array(cpu_root, audit['layers'][str(i)]['cpu_hidden'])
        if native.shape != (1, 64, 2048) or native.dtype != np.float16 or cpu.shape != (1, 19, 2048):
            raise ValueError(f'L{i}: unexpected module hidden geometry')
        if not (metrics(native[:, :19], cpu)['passed'] and metrics(native[:, 18], cpu[:, 18])['passed']):
            cases.append(dict(case, status='CPU_REFERENCE_MISMATCH', error='See module-cpu/report.json'))
            continue
        folder = job / 'module-inputs' / name
        folder.mkdir(parents=True)
        specs = {}
        for key, spec in record['inputs'].items():
            x = read_array(fixture, spec)
            if key in ('conv', 'state', 'key', 'value'):
                if np.any(x):
                    raise ValueError(f'L{i}: expected a fresh zero {key}')
                x = x.astype('<f2')
            if key in ('positions', 'cache_positions'):
                if x.dtype.kind not in 'iu' or not np.array_equal(x.reshape(-1), np.arange(64)):
                    raise ValueError(f'L{i}: unexpected {key}')
                x = x.astype('<i4') if key == 'positions' else x.reshape(64, 1).astype('<f4')
            path = folder / (key + '.bin')
            x.tofile(path)
            specs[key] = dict(identity(path, job), dtype=x.dtype.name, shape=list(x.shape))
        path = folder / 'golden_hidden.bin'
        native.tofile(path)
        cases.append(dict(case, inputs=specs, golden=dict(identity(path, job), name='golden_hidden',
                                                          dtype=native.dtype.name, shape=list(native.shape))))
    save_json(job / 'segment.json', dict(attention_modules=cases, capture_sha256=capture_hash,
                                         cpu_audit_sha256=sha(cpu_root / 'report.json')))


class _LinearHidden(torch.nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def forward(self, hidden, conv, state, effective_length, linear_update_tensor=None):
        return self.op(hidden, past_key_values=(conv.clone(), state.clone()), export_flag=True,
                       linear_update_tensor=linear_update_tensor, effective_length=effective_length)[0]


class _FullHidden(torch.nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def forward(self, hidden, key, value, mask, positions, cache_positions, all_length):
        return self.op(hidden, attention_mask=mask, position_ids=positions, past_key_values=(key.clone(), value.clone()),
                       new_kv_cache_pos=cache_positions, allQLen=all_length, use_cache=True,
                       output_attentions=False, export_flag=True)[0]


def export_modules(job, segment, layers, exporter):
    from qwen35_moe_dflash.draft.incremental.torch_export import _producer_schema, _schema
    from ..original_air_roundtrip import _attributes, _decode
    from ..operator_air_review import _data_edges, port_names

    desc, metadata = _producer_schema()
    schema = _schema(desc, metadata)
    cases = []
    with tempfile.TemporaryDirectory(dir=job, prefix='.export-modules-') as tmp:
        root = Path(tmp) / 'air'
        root.mkdir()
        for entry in segment['attention_modules']:
            case = {k: v for k, v in entry.items() if k not in ('inputs', 'golden')}
            cases.append(case)
            if case.get('status') == 'CPU_REFERENCE_MISMATCH':
                continue
            i = entry['layer']
            linear = i < 3
            model = _LinearHidden(layers[i].linear_attn) if linear else _FullHidden(layers[i].self_attn)
            folder = root / 'models' / case['id']
            folder.mkdir(parents=True)
            args, specs = {}, []
            for key, spec in entry['inputs'].items():
                args[key] = torch.from_numpy(read_array(job, spec))
                path = folder / (key + '.bin')
                shutil.copyfile(checked(job, spec), path)
                specs.append(dict(spec, **identity(path, folder), semantic_name=key))
            torch._dynamo.reset()
            print('export', case['id'], flush=True)
            with working_directory(folder), export_input_identity(list(args.values())) as evidence:
                with exporter.cpro_export_contracts(True, chunk_size=64, gdn_layers=int(linear)):
                    exporter.torchair.dynamo_export(model=model.eval(), export_path='weight', dynamic=False, **args)
            air = folder / 'export.air'
            (folder / 'weight/export.air').replace(air)
            _, _, graph = _decode(schema, air.read_bytes())
            bindings = source_bindings(graph, specs, evidence['nodes'], 1)
            edges = {s['semantic_name']: s['name'] + ':0' for s in bindings}
            projections = sum(n['type'] == 'QuantBatchMatmulV4444' for n in graph['op'])
            if projections != (5 if linear else 4):
                raise ValueError(f'L{i}: unexpected number of quantized attention projections: {projections}')
            op_type, port, arg = (('ChunkGatedDeltaRule', 'effective_length', 'effective_length') if linear
                                  else ('AdnFusedInferAttention', 'all_seq_lengths_q', 'all_length'))
            op, = [n for n in graph['op'] if n['type'] == op_type]
            attrs = _attributes(op)
            slot, = [j for j, name in port_names(attrs, 'input').items() if name == port]
            if (_data_edges(op)[slot] != edges[arg] or 'hidden' not in edges
                    or (linear and attrs['chunk_size']['i'] != 64)):
                raise ValueError(f'L{i}: invalid hidden or valid-length binding')
            saved = folder / 'golden_hidden.bin'
            shutil.copyfile(checked(job, entry['golden']), saved)
            golden = dict(entry['golden'], **identity(saved, folder))
            record = folder / 'export.json'
            save_json(record, dict(air=identity(air, folder), source_inputs=bindings, golden=golden,
                      reference_basis=MODULE_BASIS, physical_rows=64, valid_rows=19, reference_position=18,
                      quantized_projections=projections, length_edge=edges[arg],
                      capture_sha256=segment['capture_sha256'], cpu_audit_sha256=segment['cpu_audit_sha256'],
                      output_contract=dict(required_dtypes={'0': 'float16'})))
            case.update(status='EXPORTED', export=identity(record, root))
        complete = all(c['status'] == 'EXPORTED' for c in cases)
        save_json(root / 'manifest.json', dict(artifact_kind=KIND, status='EXPORTED' if complete else 'PARTIAL',
                  cases=cases, scope='ISOLATED_WHOLE_ATTENTION_MODULES_PHYSICAL64_VALID19', golden_source=MODULE_BASIS,
                  reference_model_layers=4, capture_sha256=segment['capture_sha256']))
        root.rename(job / 'air')
    print('Standalone module AIR batch:', job / 'air/manifest.json', flush=True)
