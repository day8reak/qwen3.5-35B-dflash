"""Bind the fresh segmented four-layer prefill to the original CPro wrapper."""
import json
import shutil

import numpy as np

from ..layer_probe.common import KIND, checked, identity, read_array, save_json, sha
from .prepare import STATES

CASE = 'l03_c00_prefill_four_layers_all'
OUTPUTS = ('logits', *STATES)
BASIS = 'SEGMENTED_310P_FOUR_LAYER_CAPTURE; WHOLE_PREFIX_CPU_AGREEMENT_UNVERIFIED'
MODULE_BASIS = 'CPU_CHECKED_310P_SEGMENT_WHOLE_ATTENTION_MODULE_HIDDEN_VALID19'
NATIVE_GDR_MODULE_BASIS = 'MEASURED_310P_SEGMENT_GDN_MODULE_HIDDEN_VALID19'
DECODE_MODULE_BASIS = 'CPU_CHECKED_310P_SEGMENT_WHOLE_ATTENTION_MODULE_HIDDEN_DECODE_AFTER_VALID19'


def score_module_hidden(actual, golden, atol=None, rtol=None):
    from ..layer_probe.common import metrics
    if actual.shape != (1, 64, 2048) or golden.shape != actual.shape:
        raise ValueError('Standalone prefill module must return physical [1,64,2048] hidden')
    def score(rows):
        return metrics(actual[:, rows], golden[:, rows], atol=atol, rtol=rtol)
    row = score(slice(0, 19))
    last = row['token_18'] = score(18)
    row['padding'] = score(slice(19, 64))
    if not last['passed']:
        row['passed'] = False
        row['failed_metrics'] = [*row['failed_metrics'], *('token_18_' + k for k in last['failed_metrics'])]
    return row


def prepare_segment(fixture, manifest, job, *, attention_layers=False):
    gdr = dict(physical_q_rows=64, effective_length=[19], length_dtype='torch.int16', chunk_size=64)
    if (manifest['status'] != 'CAPTURED' or manifest['layers'] != [0, 1, 2, 3]
            or (manifest['physical_rows'], manifest['valid_rows'], manifest['output_pos']) != (64, 19, 18)
            or manifest['l1_gdr']['calls'] != [gdr] or set(manifest['outputs']) != set(OUTPUTS)
            or not all(manifest['conv_history'][str(i)]['equals_physical_tail'] for i in range(3))):
        raise ValueError('Expected the fresh four-layer physical64/valid19 segmented capture')

    specs = manifest['inputs']
    prepared = {}
    (job / 'inputs').mkdir()

    def store(name, value):
        path = job / 'inputs' / (name + '.bin')
        value.tofile(path)
        prepared[name] = dict(identity(path, job), shape=list(value.shape), dtype=value.dtype.name)

    def read(name, shape, dtypes):
        value = read_array(fixture, specs[name])
        if value.shape != shape or value.dtype.name not in dtypes:
            raise ValueError(f'Unexpected segmented input {name}: {value.shape} {value.dtype}')
        return value

    ids = read('input_ids', (1, 64), ('int32', 'int64'))
    if ids.min() < 0 or ids.max() >= 248320 or not (ids[:, 19:] == 1).all():
        raise ValueError('Unexpected prompt or padding IDs')
    store('input_ids', ids.astype('<i4'))
    store('embedding', read('embedding', (1, 64, 2048), ('float16',)))
    pos = np.arange(64)
    for name, expected, dtype in (
        ('new_kv_cache_pos', pos, '<f4'), ('position_ids', pos[None], '<i4'),
        ('effective_length', np.array([19]), '<i2'), ('allQLen', np.array([19]), '<i8'),
        ('block_table', pos[None], '<i4'),
    ):
        value = read_array(fixture, specs[name])
        if value.dtype.kind not in 'iu' or not np.array_equal(value, expected):
            raise ValueError(f'Unexpected segmented prefill control: {name}')
        store(name, value.astype(dtype))
    store('output_pos', np.array([manifest['output_pos']], dtype='<i4'))
    mask = read('attention_mask', (1, 1, 64, 2048), ('float16', 'float32'))
    allowed = (pos[:, None] < 19) & (np.arange(2048)[None, :] <= pos[:, None])
    blocked = np.isneginf(mask) | (mask == -65504.) | (mask == np.finfo(mask.dtype).min)
    if not np.array_equal(mask[0, 0] == 0, allowed) or not np.array_equal(blocked[0, 0], ~allowed):
        raise ValueError('Captured mask differs from the valid19 causal mask')
    store('attention_mask', mask.astype('<f4'))
    for name in STATES:
        if name.startswith('conv_state'):
            shape = (1, 8192, 4)
        elif name.startswith('recurrent_state'):
            shape = (1, 32, 128, 128)
        else:
            shape = (64, 32, 32, 16)
        value = read('initial_' + name, shape, ('float32', 'float16') if name.startswith('recurrent') else ('float16',))
        if np.any(value):
            raise ValueError(f'Expected a fresh zero state for {name}')
        store(name, value.astype('<f2'))

    save_json(job / 'inputs.json', prepared)
    if attention_layers:
        from .module_export import prepare_modules
        prepare_modules(fixture, manifest, job)
        return
    goldens = []
    (job / 'reference').mkdir()
    for name in OUTPUTS:
        row = manifest['outputs'][name]
        expected = [1, 1, 248320] if name == 'logits' else prepared[name]['shape']
        if row['name'] != name or row['shape'] != expected or row['dtype'] not in ('float16', 'float32'):
            raise ValueError(f'Unexpected reference output: {name}')
        path = job / 'reference' / (name + '.bin')
        shutil.copyfile(checked(fixture, row), path)
        goldens.append(dict(row, **identity(path, job)))
    cpu_outputs = []
    cpu_report = fixture / 'cpu/report.json'
    capture_sha = sha(fixture / 'manifest.json')
    if cpu_report.is_file():
        cpu = json.loads(cpu_report.read_text())
        if cpu['capture_sha256'] != capture_sha or cpu['scope'] != 'FOUR_LAYER_CPU_VS_NATIVE_310P_VALID19':
            raise ValueError('CPU report belongs to a different prefill capture')
        (job / 'cpu').mkdir()
        for name in OUTPUTS:
            row = cpu['cpu_outputs'][name]
            path = job / 'cpu' / (name + '.bin')
            shutil.copyfile(checked(cpu_report.parent, row), path)
            cpu_outputs.append(dict(row, **identity(path, job)))
    save_json(job / 'segment.json', dict(physical_rows=64, valid_rows=19, output_pos=18,
              capture_sha256=capture_sha, gdr_mode='chunk', goldens=goldens, cpu_outputs=cpu_outputs,
              reference_basis=BASIS, selected_golden=None))


def _position_gathers(graph, runtime_edges, enums):
    from ..operator_air_review import _data_edges, trace_input

    controls = {runtime_edges[name]: name for name in ('position_ids', 'output_pos')}
    gathers = {name: [] for name in controls.values()}
    nodes = {node['name']: node for node in graph['op']}
    for node in nodes.values():
        if node['type'] not in ('Gather', 'GatherV2', 'GatherV2D'):
            continue
        edges = _data_edges(node)
        if len(edges) < 2:
            continue
        chain = trace_input(edges[1], nodes, enums)['chain']
        if chain and chain[-1]['type'] == 'Data':
            control = controls.get(chain[-1]['name'] + ':0')
            if control:
                gathers[control].append(node['name'])
    if len(gathers['position_ids']) != 2 or len(gathers['output_pos']) != 1:
        raise ValueError(f'Expected two RoPE table gathers and one readout gather bound to the position inputs: {gathers}')
    return gathers


def finish_segment_export(job, root, segment, manifest, runtime_names, evidence):
    from qwen35_moe_dflash.draft.incremental.torch_export import _producer_schema, _schema
    from ..original_air_roundtrip import _decode, _attributes
    from ..operator_air_review import port_names, _data_edges
    from ..layer_probe.replay import source_bindings

    folder = root / 'models' / CASE
    air = folder / 'export.air'
    descriptor, metadata = _producer_schema()
    schema = _schema(descriptor, metadata)
    _, _, graph = _decode(schema, air.read_bytes())
    specs = []
    for name in runtime_names:
        spec = manifest[name]
        destination = folder / spec['path']
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(checked(job, spec), destination)
        specs.append(dict(spec, semantic_name=name))
    bindings = source_bindings(graph, specs, evidence['nodes'], len(OUTPUTS))
    if {s['semantic_name'] for s in bindings} != set(runtime_names):
        raise ValueError('Exported graph does not retain the declared segment runtime inputs')
    runtime_edges = {s['semantic_name']: s['name'] + ':0' for s in bindings}
    position_gathers = _position_gathers(graph, runtime_edges, schema.enums[metadata['schema_package'] + '.DataType'])
    gdr_nodes = [n for n in graph['op'] if n['type'] == 'ChunkGatedDeltaRule']
    attention = [n for n in graph['op'] if n['type'] == 'AdnFusedInferAttention']
    if len(gdr_nodes) != 3 or len(attention) != 1:
        raise ValueError('Expected three GDR chunks and one full attention core')
    for node, port, control in ([(n, 'effective_length', 'effective_length') for n in gdr_nodes]
                                + [(attention[0], 'all_seq_lengths_q', 'allQLen')]):
        attrs = _attributes(node)
        edges = dict(enumerate(_data_edges(node)))
        slots = [i for i, name in port_names(attrs, 'input').items() if name == port]
        if len(slots) != 1 or edges.get(slots[0]) != runtime_edges[control]:
            raise ValueError(f'{node["name"]}.{port} is not bound to {control}')
        if port == 'effective_length' and attrs.get('chunk_size', {}).get('i') != 64:
            raise ValueError('Expected a full-width GDR chunk')
    net, = [n for n in graph['op'] if n['type'] == 'NetOutput']
    goldens = segment['goldens']
    for spec in [*goldens, *segment['cpu_outputs']]:
        destination = folder / spec['path']
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(checked(job, spec), destination)
    required = {str(i): 'float16' if name != 'logits' else 'float32' for i, name in enumerate(OUTPUTS)}
    record = folder / 'export.json'
    save_json(record, dict(status='EXPORTED', air=identity(air, folder), source_inputs=bindings,
              golden=goldens[0], goldens=goldens, cpu_outputs=segment['cpu_outputs'], output_contract=dict(order=list(OUTPUTS),
              required_dtypes=required, source_edges=net['input']), capture_sha256=segment['capture_sha256'],
              reference_basis=BASIS, physical_rows=64, valid_rows=19, output_pos=18, gdr_calls=len(gdr_nodes),
              position_gathers=position_gathers))
    case = dict(id=CASE, status='EXPORTED', layer=3, call=0, point='prefill_four_layers',
                phase='prefill', kind='segment_prefill', ordinal=0, output_name='logits_and_states',
                export=identity(record, root))
    save_json(root / 'manifest.json', dict(artifact_kind=KIND, status='EXPORTED', cases=[case],
              golden_source=BASIS, capture_sha256=segment['capture_sha256'], reference_model_layers=4,
              scope='CPRO_L0_THROUGH_L3_AND_PREFIX_READOUT_PHYSICAL64_VALID19', selected_golden=None))


def compare_segment_output(name, actual, native, cpu, atol=None, rtol=None, *, reference_is_310p=True, valid_rows=19):
    """Score the native trajectory. Also report the independent propagated CPU trajectory."""
    from ..layer_probe.common import metrics

    def score(a, b):
        return metrics(a, b, atol=atol, rtol=rtol)

    if name.startswith(('key_cache_', 'value_cache_')):
        from .decoder_cases import cache_rows
        if actual.shape != (64, 32, 32, 16) or actual.dtype != native.dtype:
            raise ValueError('Unexpected segmented K/V output descriptor')
        a, b = (cache_rows(x, 0, valid_rows) for x in (actual, native))
        row = score(a, b)
        if reference_is_310p and np.any(native[2:].view(np.uint16)):
            raise ValueError('Native cache changed outside the two physical write pages')
        preserved = actual[2:].tobytes() == native[2:].tobytes()
        row.update(valid_rows=valid_rows, untouched_cache_equal_bits=preserved)
        row['passed'] &= preserved
        if not preserved:
            row.setdefault('failed_metrics', []).append('untouched_cache_bits')
        if cpu is not None:
            row['vs_propagated_cpu'] = score(a, cache_rows(cpu, 0, valid_rows))
        return row
    row = score(actual, native)
    if cpu is not None:
        row['vs_propagated_cpu'] = score(actual, cpu)
        if name.startswith('conv_state_'):
            row['cpu_conv_history'] = 'LAST_FOUR_VALID_ROWS; NATIVE_RETURNS_LAST_FOUR_PHYSICAL_ROWS'
    if name == 'logits':
        row.update(actual_top1=int(actual.argmax()), native_top1=int(native.argmax()))
        if cpu is not None:
            row['cpu_top1'] = int(cpu.argmax())
        if row['actual_top1'] != row['native_top1']:
            row['passed'] = False
            row.setdefault('failed_metrics', []).append('top1')
    return row
