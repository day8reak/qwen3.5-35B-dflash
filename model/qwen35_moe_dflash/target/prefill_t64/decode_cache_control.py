"""Replace the first decode's L3 cache update with independently computed CPU KV."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from ...common.artifacts import output_path, share_for_transfer
from ..layer_probe.common import checked, identity, metrics, read_array, save_json, sha
from .segment_export import DECODE_MODULE_BASIS

CASE = 'l03_c01_full_attention_layer_hidden_o0'
CACHE_NAMES = ('key', 'value')


def make_cpu_cache(fixture, root):
    import torch
    from .segment_prefix import tensor_record
    from .segment_prefix_cpu import _load

    manifest = json.loads((fixture / 'manifest.json').read_text())
    parent = Path(manifest['prefill_fixture'])
    capture_sha = sha(fixture / 'manifest.json')
    geometry = (manifest['physical_rows'], manifest['valid_rows'], manifest['position'],
                manifest['allQLen'], manifest['effective_length'])
    if (manifest['status'] != 'CAPTURED' or manifest['phase'] != 'decode' or geometry != (1, 1, 19, 20, 1)
            or manifest['prefill_sha256'] != sha(parent / 'manifest.json')):
        raise ValueError('Expected the first decode of the retained 64/19 prompt')
    item = manifest['attention_modules']['3']
    old = {k: torch.from_numpy(read_array(fixture, s)) for k, s in item['inputs'].items()}
    audit_root = fixture / 'module-cpu'
    audit = json.loads((audit_root / 'report.json').read_text())
    if audit['capture_sha256'] != capture_sha or not audit['layers']['3']['passed']:
        raise ValueError('Missing qualified same-capture L3 hidden')
    with torch.inference_mode():
        _, cpu_model, _, _ = _load(parent, {})
        layer = cpu_model.model.model.layers[3].self_attn
        table = read_array(fixture, manifest['block_table'])
        if (not np.array_equal(table, layer.block_table.numpy()) or layer.block_size != 32
                or old['cache_positions'].reshape(-1).tolist() != [19] or old['positions'].tolist() != [[19]]
                or old['all_length'].tolist() != [20]):
            raise ValueError('Unexpected page table or decode position')
        hidden, after = cpu_model.attention(old['hidden'], layer, 'layers.3.self_attn',
            (old['key'], old['value']), dict(position_ids=old['positions'], attention_mask=old['mask'],
            new_kv_cache_pos=old['cache_positions'].reshape(-1), kv_length=20))
    hidden_score = metrics(read_array(fixture, item['golden_hidden']), hidden.numpy())
    previous = read_array(audit_root, audit['layers']['3']['cpu_hidden'])
    if not hidden_score['passed'] or not metrics(hidden.numpy(), previous)['passed']:
        raise ValueError('CPU replay did not reproduce the qualified hidden')
    page = int(table[0, 0])
    report = dict(capture_sha256=capture_sha, cpu_audit_sha256=sha(audit_root / 'report.json'),
                  golden_hidden_sha256=item['golden_hidden']['sha256'], page=page, offset=19,
                  cpu_hidden_check=hidden_score, caches={})
    for name, cache in zip(CACHE_NAMES, after, strict=True):
        before = old[name].half().contiguous()
        cache = cache.half().contiguous()
        if (not torch.equal(before.to(old[name].dtype), old[name]) or cache.shape != (64, 32, 32, 16)
                or not 0 <= page < len(cache) or not torch.isfinite(cache).all()):
            raise ValueError(f'Unexpected {name} CPU cache')
        different = before.view(torch.int16) != cache.view(torch.int16)
        changed = int(different[page, :, 19].sum())
        if changed != int(different.sum()):
            raise ValueError(f'CPU {name} update did not change only the new token slot')
        report['caches'][name] = dict(tensor_record(root, name, cache),
            before_sha256=hashlib.sha256(before.numpy().tobytes()).hexdigest(), changed_elements=changed)
    save_json(root / 'report.json', report)
    return report


def graph_message(raw):
    from google.protobuf import descriptor_pool, message_factory
    from google.protobuf.message import DecodeError
    descriptor = (Path(__file__).resolve().parents[4] / 'specs/torchair-ge-ir.pb').read_bytes()
    pool = descriptor_pool.DescriptorPool()
    pool.AddSerializedFile(descriptor)
    for typename in ('ModelDef', 'GraphDef'):
        desc = pool.FindMessageTypeByName('ge.proto.' + typename)
        cls = (message_factory.GetMessageClass(desc) if hasattr(message_factory, 'GetMessageClass')
               else message_factory.MessageFactory(pool).GetPrototype(desc))
        try:
            message = cls.FromString(raw)
        except DecodeError:
            continue
        graphs = message.graph if typename == 'ModelDef' else [message]
        if len(graphs) == 1 and any(op.type == 'AdnFusedInferAttention' for op in graphs[0].op):
            return message, graphs[0]
    raise ValueError('Expected an L3 AIR with one attention graph')


def replace_updates(raw, bindings, selected):
    message, graph = graph_message(raw)
    nodes = {op.name: op for op in graph.op}
    attention, = [op for op in graph.op if op.type == 'AdnFusedInferAttention']
    updates = [op for op in graph.op if op.type == 'CacheUpdate']
    if len(updates) != 2 or len(nodes) != len(graph.op):
        raise ValueError('Expected a single L3 graph containing two CacheUpdate nodes')

    def producer(edge):
        if not edge.endswith(':0') or edge[:-2] not in nodes:
            raise ValueError(f'Unresolved L3 cache edge {edge}')
        return nodes[edge[:-2]]

    def origin(edge):
        node = producer(edge)
        seen = set()
        while node.type in ('Identity', 'TensorMove', 'Reshape', 'Squeeze', 'Unsqueeze', 'Cast',
                            'Transpose', 'TransData', 'Permute', 'ExpandDims'):
            if node.name in seen:
                raise ValueError('Cycle in cache path: ' + node.name)
            seen.add(node.name)
            node = producer(node.input[0])
        return node.name

    # Use original exporter slots (query, dynamic K[0], dynamic V[0]).
    # Bind each update independently to the captured cache Data by object identity.
    replacements = {}
    paths = {}
    for port, name in enumerate(CACHE_NAMES, start=1):
        source = next(s['name'] for s in bindings if s['semantic_name'] == name)
        if nodes[source].type != 'Data':
            raise ValueError(f'Invalid {name} runtime input')
        op, = [op for op in updates if origin(op.input[0]) == source]
        upstream = origin(attention.input[port])
        if upstream not in (op.name, source):
            raise ValueError(f'Unexpected attention {name} input from {upstream}')
        paths[name] = dict(data=source, update=op.name, attention_input=attention.input[port],
                           attention_reads_update_edge=upstream == op.name)
        if name in selected:
            replacements[op.name] = op.input[0]

    if paths['key']['update'] == paths['value']['update']:
        raise ValueError('K and V cache updates alias')
    # Bypass the actual GE write, including its control dependencies. Preserve
    # the original graph inputs, attention attributes, weights and output edge.
    retained = []
    for node in graph.op:
        if node.name in replacements:
            continue
        for i, edge in enumerate(node.input):
            owner, _, slot = edge.rpartition(':')
            if owner in replacements:
                if slot not in ('0', '-1'):
                    raise ValueError(f'Unexpected CacheUpdate edge {edge}')
                feed = replacements[owner]
                node.input[i] = feed if slot == '0' else feed.rsplit(':', 1)[0] + ':-1'
        retained.append(node)
    del graph.op[:]
    graph.op.extend(retained)
    return message.SerializeToString(), paths


def prepare_batch(baseline, cpu, root):
    manifest = json.loads((baseline / 'manifest.json').read_text())
    case, = [c for c in manifest['cases'] if c['id'] == CASE and c['status'] == 'EXPORTED']
    record_path = checked(baseline, case['export'])
    folder = record_path.parent
    record = json.loads(record_path.read_text())
    facts = json.loads((cpu / 'report.json').read_text())
    if (record['reference_basis'] != DECODE_MODULE_BASIS or record['capture_sha256'] != facts['capture_sha256']
            or record['cpu_audit_sha256'] != facts['cpu_audit_sha256']
            or record['golden']['sha256'] != facts['golden_hidden_sha256']
            or (record['physical_rows'], record['valid_rows'], record['reference_position']) != (1, 1, 19)):
        raise ValueError('CPU intervention and baseline must use the same first decode, CPU audit and hidden')
    raw = checked(folder, record['air']).read_bytes()
    checked(folder, record['golden'])
    for spec in record['source_inputs']:
        checked(folder, spec)
        name = spec['semantic_name']
        if name in CACHE_NAMES:
            cache = facts['caches'][name]
            checked(cpu, cache)
            if (cache['before_sha256'], cache['shape'], cache['dtype']) != (spec['sha256'], spec['shape'], spec['dtype']):
                raise ValueError(f'Original cache or ABI differs for {name}')
    cases = []
    for label, names in [('k', ('key',)), ('v', ('value',)), ('kv', CACHE_NAMES)]:
        data, paths = replace_updates(raw, record['source_inputs'], names)
        if label == 'kv':
            print('Original L3 cache paths:', json.dumps(paths), flush=True)
        name = f'l03_c01_full_attention_cpu_{label}_o0'
        dest = root / 'models' / name
        shutil.copytree(folder, dest)
        air = dest / record['air']['path']
        air.write_bytes(data)
        sources = []
        for spec in record['source_inputs']:
            source = dict(spec)
            role = spec['semantic_name']
            if role in names:
                path = dest / spec['path']
                shutil.copyfile(checked(cpu, facts['caches'][role]), path)
                source.update(identity(path, dest))
            sources.append(source)
        control = dict(replaced=list(names), baseline_air=record['air']['sha256'], cache_paths=paths,
                       page=facts['page'], offset=facts['offset'], cpu_report_sha256=sha(cpu / 'report.json'))
        export = dest / 'export.json'
        save_json(export, dict(record, air=identity(air, dest), source_inputs=sources, cache_control=control))
        cases.append(dict(case, id=name, point='full_attention_cpu_' + label, export=identity(export, root)))
    save_json(root / 'manifest.json', dict(manifest, cases=cases, scope='L3_FIRST_DECODE_CPU_CACHE_INTERVENTION'))


def build(action, source, output, cpu=None):
    source = source.resolve(strict=True)
    output = output_path(output)
    inputs = (source,) if cpu is None else (source, cpu)
    for existing in inputs:
        existing = existing.resolve(strict=True)
        if output == existing or existing in output.parents or output in existing.parents:
            raise ValueError('Keep the control separate from its inputs')
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as staging:
        root = Path(staging) / 'result'
        root.mkdir()
        if action == 'cpu':
            report = make_cpu_cache(source, root)
            print('310P hidden vs CPU: relative_l2', report['cpu_hidden_check']['relative_l2'],
                  'cosine', report['cpu_hidden_check']['cosine'])
        else:
            prepare_batch(source, cpu, root)
        root.rename(output)
    share_for_transfer(output)
    print(output, flush=True)
