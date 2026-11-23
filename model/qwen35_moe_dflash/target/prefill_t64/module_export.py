"""Export each original CPro attention module on its own observed NZ module input."""
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch

from ...common.command_trace import note
from ..layer_probe.common import KIND, checked, identity, read_array, save_json, sha
from ..layer_probe.replay import export_input_identity, source_bindings
from ..layer_probe.steps import working_directory

from .segment_export import DECODE_MODULE_BASIS, MODULE_BASIS, NATIVE_GDR_MODULE_BASIS
from .segment_gdn_trace import BASIS as TRACE_BASIS, TAPS


def prepare_modules(fixture, capture, job):
    from ..layer_probe.common import metrics
    cpu_root = fixture / 'module-cpu'
    capture_hash = sha(fixture / 'manifest.json')
    decode = capture.get('phase') == 'decode'
    rows, valid = (1, 1) if decode else (64, 19)
    trace = capture.get('linear_trace')
    native_stages = 'gdr_stages' in capture
    if native_stages:
        if decode or trace is not None or set(capture['gdr_stages']) != {'0', '1', '2'}:
            raise ValueError('Expected native prefill GDR stage measurements for L0–L2')
        layers = range(3)
    else:
        scope = 'SEGMENT_DECODE_AFTER_VALID19_FIXED_INPUT_MODULE_HIDDEN' if decode else 'SEGMENT_VALID19_FIXED_INPUT_ATTENTION_MODULE_HIDDEN'
        audit = json.loads((cpu_root / 'report.json').read_text())
        if audit['scope'] != scope or audit['capture_sha256'] != capture_hash:
            raise ValueError('Expected the fixed-input module CPU audit for this same capture')
        if trace is not None and trace['layer'] not in (0, 1, 2):
            raise ValueError('Expected a linear attention layer')
        layers = [trace['layer']] if trace is not None else range(4)
    cases = []
    for i in layers:
        record = capture['attention_modules'][str(i)]
        expected_member = 'linear_attn' if i < 3 else 'self_attn'
        if record['module'] != expected_member:
            raise ValueError(f'Unexpected attention module at L{i}')
        point = 'gdn_layer' if i < 3 else 'full_attention_layer'
        call = int(decode)
        name = f'l{i:02}_c{call:02}_{point}_hidden_o0'
        case = dict(id=name, layer=i, call=call, point=point, kind='segment_module_hidden',
                    ordinal=i, phase='decode' if decode else 'prefill', output_name='golden_hidden')
        if trace is not None:
            case.update(kind='segment_gdn_trace', id=f'l{i:02}_c00_gdn_trace_o0')
        native = read_array(fixture, record['golden_hidden'])
        if native.shape != (1, rows, 2048) or native.dtype != np.float16:
            raise ValueError(f'L{i}: unexpected native module hidden geometry')
        if not native_stages:
            cpu = read_array(cpu_root, audit['layers'][str(i)]['cpu_hidden'])
            if cpu.shape != (1, valid, 2048):
                raise ValueError(f'L{i}: unexpected CPU module hidden geometry')
            if not (metrics(native[:, :valid], cpu)['passed'] and metrics(native[:, valid-1], cpu[:, -1])['passed']):
                cases.append(dict(case, status='CPU_REFERENCE_MISMATCH', error='See module-cpu/report.json'))
                continue
        folder = job / 'module-inputs' / case['id']
        folder.mkdir(parents=True)
        specs = {}
        for key, spec in record['inputs'].items():
            x = read_array(fixture, spec)
            if key in ('conv', 'state', 'key', 'value'):
                if not decode and np.any(x):
                    raise ValueError(f'L{i}: expected a fresh zero {key}')
                half = x.astype('<f2')
                if decode and not np.array_equal(half.astype(x.dtype), x):
                    raise ValueError(f'L{i}: {key} cannot be carried losslessly in the CPro FP16 input')
                x = half
            if key in ('positions', 'cache_positions'):
                expected = [19] if decode else np.arange(64)
                if x.dtype.kind not in 'iu' or not np.array_equal(x.reshape(-1), expected):
                    raise ValueError(f'L{i}: unexpected {key}')
                x = x.astype('<i4') if key == 'positions' else x.reshape(rows, 1).astype('<f4')
            path = folder / (key + '.bin')
            x.tofile(path)
            specs[key] = dict(identity(path, job), dtype=x.dtype.name, shape=list(x.shape))
        path = folder / 'golden_hidden.bin'
        native.tofile(path)
        case.update(inputs=specs, golden=dict(identity(path, job), name='golden_hidden',
                                              dtype=native.dtype.name, shape=list(native.shape)))
        if trace is not None:
            case['taps'] = []
            for key in TAPS:
                ref = trace['tensors'][key]
                x = read_array(fixture, ref)
                width = 32 if key in ('g', 'beta') else 4096
                if x.shape != (1, 19, width) or x.dtype.name != ('float32' if key == 'g' else 'float16'):
                    raise ValueError(f'Unexpected {key} capture: {x.shape}, {x.dtype}')
                dest = folder / (key + '.bin')
                x.tofile(dest)
                case['taps'].append(dict(ref, **identity(dest, job), name=key))
            control = case['replay'] = dict(epsilon=trace['epsilon'])
            projection = audit['trace_projection']
            for key, origin, spec in (('gamma', fixture, trace['gamma']), ('weight', cpu_root, projection['weight']),
                                      ('scale', cpu_root, projection['scale'])):
                dest = folder / ('cpu_replay_' + key + '.bin')
                shutil.copyfile(checked(origin, spec), dest)
                control[key] = dict(spec, **identity(dest, job))
        cases.append(case)
    segment = dict(attention_modules=cases, capture_sha256=capture_hash)
    if native_stages:
        segment['reference_basis'] = NATIVE_GDR_MODULE_BASIS
    else:
        segment['cpu_audit_sha256'] = sha(cpu_root / 'report.json')
    if decode:
        segment['phase'] = 'decode'
    save_json(job / 'segment.json', segment)


class _LinearHidden(torch.nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def forward(self, hidden, conv, state, effective_length, linear_update_tensor=None):
        return self.op(hidden, past_key_values=(conv.clone(), state.clone()), export_flag=True,
                       linear_update_tensor=linear_update_tensor, effective_length=effective_length)[0]


class _LinearTrace(_LinearHidden):
    def forward(self, hidden, conv, state, effective_length, linear_update_tensor=None):
        result = self.op(hidden, past_key_values=(conv.clone(), state.clone()), export_flag=True,
                         linear_update_tensor=linear_update_tensor, effective_length=effective_length)
        return (result[0], *result[2:])


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
        basis = TRACE_BASIS if any(c['kind'] == 'segment_gdn_trace' for c in segment['attention_modules']) else MODULE_BASIS
        decode = segment.get('phase') == 'decode'
        rows = 1 if decode else 64
        if decode:
            basis = DECODE_MODULE_BASIS
        elif segment.get('reference_basis') == NATIVE_GDR_MODULE_BASIS:
            basis = NATIVE_GDR_MODULE_BASIS
        for entry in segment['attention_modules']:
            case = {k: v for k, v in entry.items() if k not in ('inputs', 'golden', 'taps', 'replay')}
            cases.append(case)
            if case.get('status') == 'CPU_REFERENCE_MISMATCH':
                continue
            i = entry['layer']
            linear = i < 3
            traced = case['kind'] == 'segment_gdn_trace'
            linear_wrapper = _LinearTrace if traced else _LinearHidden
            model = linear_wrapper(layers[i].linear_attn) if linear else _FullHidden(layers[i].self_attn)
            refs = [entry['golden'], *entry.get('taps', [])]
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
                with exporter.cpro_export_contracts(True, chunk_size=rows, gdn_layers=int(linear), sequence_rows=rows):
                    note(f'[AIR export in Python] cwd={folder} case={case["id"]} '
                         f'torchair.dynamo_export(model=model.eval(), export_path="weight", dynamic=False, **args); '
                         f'inputs={list(args)}')
                    exporter.torchair.dynamo_export(model=model.eval(), export_path='weight', dynamic=False, **args)
            air = folder / 'export.air'
            (folder / 'weight/export.air').replace(air)
            _, _, graph = _decode(schema, air.read_bytes())
            bindings = source_bindings(graph, specs, evidence['nodes'], len(refs))
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
                    or (linear and attrs['chunk_size']['i'] != rows)):
                raise ValueError(f'L{i}: invalid hidden or valid-length binding')
            goldens = []
            for ref in refs:
                saved = folder / (ref['name'] + '.bin')
                shutil.copyfile(checked(job, ref), saved)
                goldens.append(dict(ref, **identity(saved, folder)))
            net, = [n for n in graph['op'] if n['type'] == 'NetOutput']
            output_contract = dict(order=[r['name'] for r in refs], source_edges=net['input'],
                                   required_dtypes={str(j): 'float16' for j in range(len(refs))})
            replay = {}
            if traced:
                control = entry['replay']
                replay['epsilon'] = control['epsilon']
                for key in ('gamma', 'weight', 'scale'):
                    spec = control[key]
                    dest = folder / ('cpu_replay_' + key + '.bin')
                    shutil.copyfile(checked(job, spec), dest)
                    replay[key] = dict(spec, **identity(dest, folder))
            record = folder / 'export.json'
            save_json(record, dict(air=identity(air, folder), source_inputs=bindings, golden=goldens[0], goldens=goldens,
                      reference_basis=basis, physical_rows=rows, valid_rows=1 if decode else 19, reference_position=19 if decode else 18,
                      quantized_projections=projections, length_edge=edges[arg],
                      capture_sha256=segment['capture_sha256'], cpu_audit_sha256=segment.get('cpu_audit_sha256'),
                      output_contract=output_contract, **({'trace_replay': replay} if traced else {})))
            case.update(status='EXPORTED', export=identity(record, root))
        complete = all(c['status'] == 'EXPORTED' for c in cases)
        save_json(root / 'manifest.json', dict(artifact_kind=KIND, status='EXPORTED' if complete else 'PARTIAL',
                  cases=cases, scope=('ISOLATED_DECODE_MODULES_AFTER_VALID19' if decode else 'ISOLATED_WHOLE_ATTENTION_MODULES_PHYSICAL64_VALID19'), golden_source=basis,
                  reference_model_layers=4, capture_sha256=segment['capture_sha256']))
        root.rename(job / 'air')
    print('Standalone module AIR batch:', job / 'air/manifest.json', flush=True)
