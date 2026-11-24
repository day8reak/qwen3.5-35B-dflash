"""Replay one original tail QLinear at a time with its actual activation from CPro."""
import shutil

from ...draft.compile_precision import _observed_desc
from ...execution.serial_transport import _environment
from ..layer_probe.batch import bind_compiled
from ..layer_probe.common import checked, save_json, sha
from . import whole_run
from .attention_slice import descriptor, extract
from .chain_hilog import diagnose_decoder_pair
from .decode_cache_control import graph_message
from .decode_cache_trace import _project_input
from .decoder_pair import load


QUANT = 'QuantBatchMatmulV4444'


def probe_abi(document, enums, meta):
    graph, = document['graph']
    net, = [n for n in graph['op'] if n['type'] == 'NetOutput']
    desc, = net['input_desc']
    observed = _observed_desc(desc, enums)
    dtype, shape = observed['dtype'], observed['shape']
    if dtype not in ('float16', 'float32') or not shape or any(n <= 0 for n in shape):
        raise ValueError(f"Expected a static floating probe output, got {observed}")
    if meta['output_dtype'] and dtype != meta['output_dtype']:
        raise ValueError(f"{meta['output_edge']}: source {meta['output_dtype']} compiled to {dtype}")
    output = dict(name='readout', dtype=dtype, shape=shape)
    io = bind_compiled(document, meta['inputs'], output, enums)
    if len(io['inputs']) != len(meta['inputs']) or (meta['kind'] == whole_run._PROBE and len(io['inputs']) != 1):
        raise ValueError('Compiled readout does not consume the selected inputs')
    # The descriptor above came from NetOutput. There is no numerical reference at this edge.
    del io['source_dtype_preserved'], io['outputs'][0]['golden_dtype']
    io['output_binding'] = 'SOLE_COMPILED_NETOUTPUT'
    return io


def weight_sources(nodes, edge):
    """Describe the leaves of the selected weight operand without materializing tensor data."""
    found, seen, pending = [], set(), [edge]
    while pending:
        op = nodes[pending.pop().rsplit(':', 1)[0]]
        if op.name in seen:
            continue
        seen.add(op.name)
        inputs = [e for e in op.input if e and not e.endswith(':-1')]
        if op.type in ('FileConstant', 'Const', 'Constant') or not inputs:
            leaf = dict(node=op.name, type=op.type)
            if op.type == 'FileConstant':
                leaf['file'] = op.attr['file_path'].s.decode()
            if 'value' in op.attr and op.attr['value'].HasField('t'):
                leaf['shape'] = list(op.attr['value'].t.desc.shape.dim)
            elif op.output_desc:
                leaf['shape'] = list(op.output_desc[0].shape.dim)
            found.append(leaf)
        else:
            pending.extend(inputs)
    return sorted(found, key=lambda leaf: leaf['node'])


def weight_files(nodes, edge):
    return sorted({leaf['file'] for leaf in weight_sources(nodes, edge) if 'file' in leaf})


def wire_shape(fixture, desc):
    """Use the AIR rank if static. Only accept physically identical singleton views."""
    shape, source = fixture['shape'], list(desc.shape.dim)
    if source and all(n > 0 for n in source):
        if [n for n in shape if n != 1] != [n for n in source if n != 1]:
            raise ValueError(f'Board output shape {shape} differs from source edge {source}')
        return source
    return shape


def execute_slice(work, parent, raw, feed, input_edge, output_edge, *, isolated=False):
    folder = work / 'air'
    folder.mkdir(parents=True)
    msg, graph = graph_message(raw, 'NetOutput')
    nodes = {op.name: op for op in graph.op}
    path, fixture = feed
    shape = wire_shape(fixture, descriptor(nodes, input_edge))
    name = 'probe_activation'
    cut = extract(msg, graph, parent, folder, {input_edge: (name, fixture['dtype'], shape)}, output_edge)
    retained = [op['name'] for op in cut['nodes'] if op['type'] == QUANT]
    if isolated and retained != [output_edge.rsplit(':', 1)[0]]:
        raise ValueError(f'Projection probe unexpectedly includes quantized projections {retained}')
    print(f"{work.name}: {input_edge} -> {output_edge}; retained {QUANT}: {retained}", flush=True)
    weights = list(cut['weight_evidence'].values())
    (folder / 'weight').mkdir(exist_ok=True)
    dst = folder / 'activation.bin'
    shutil.copyfile(path, dst)
    operand = dict(fixture, path=dst.name, name=name, semantic_name=name, shape=shape)
    checked(folder, operand)
    meta = load(parent / 'manifest.json')
    inventory = {w['path']: w for w in meta['weights']}
    if any(row != inventory.get(row['path']) for row in weights):
        raise ValueError('Projection weights differ from the failed tail')
    dtype = cut['output_desc']['dtype']
    explicit = {'DT_FLOAT16': 'float16', 'DT_FLOAT': 'float32'}
    if dtype not in (*explicit, 'DT_UNDEFINED', 0):
        raise ValueError(f'{output_edge}: unexpected source output dtype {dtype}')
    save_json(folder / 'manifest.json', dict(kind=whole_run._PROBE, layer=meta['layer'], air=cut['air'],
              parent_air_sha256=meta['air']['sha256'], input_edge=input_edge, output_edge=output_edge,
              output_dtype=explicit.get(dtype), inputs=[operand], weights=weights, quant_nodes=retained))


def probe_layer(parent, layer, root, ddk, source, diagnosis, tool):
    case = parent / f'L{layer}'
    folder = case / 'air'
    meta = load(folder / 'manifest.json')
    built = load(case / 'omc/manifest.json')
    attempt = load(case / 'board/attempt.json')
    _, device, server = _environment()
    if (meta['kind'] != 'single-cpro-decoder-tail' or meta['layer'] != layer
            or (device, server) != (attempt['device'], attempt['server'])
            or attempt['omc_sha256'] != built['primary']['sha256']
            or whole_run.kernel_identity(ddk, source) != built['gdr_source_sha256']
            or built['air_manifest_sha256'] != sha(folder / 'manifest.json')):
        raise ValueError(f'L{layer}: expected the failed T64 decoder tail on the same board and build')
    raw = checked(folder, meta['air']).read_bytes()
    _, graph = graph_message(raw, 'NetOutput')
    nodes = {op.name: op for op in graph.op}
    spec, = meta['inputs']
    residual = spec['name'] + ':0'
    original = checked(folder, spec), spec
    candidates = [op for op in graph.op if op.type == QUANT]
    if not candidates:
        raise ValueError(f'L{layer}: no {QUANT} in source AIR. Inspect compiled lowering before attributing the stub.')
    report = dict(layer=layer, tail_air_sha256=meta['air']['sha256'], tail_omc_sha256=attempt['omc_sha256'],
                  residual_sha256=spec['sha256'], numerical_validation='NOT_RUN', probes={}, candidates=[])
    prefixes = {residual: (original, 'retained-input')}
    failed_boards = []

    def probe(name, start, end, feed, isolated):
        work = root / name
        row = report['probes'][name] = dict(input_edge=start, output_edge=end, input_sha256=feed[1]['sha256'],
                                            input_shape=feed[1]['shape'], input_dtype=feed[1]['dtype'], stage='prepare')
        try:
            execute_slice(work, folder, raw, feed, start, end, isolated=isolated)
            row['stage'] = 'compile'
            whole_run.compile_model(work, ddk, source, diagnosis)
            row['stage'] = 'board'
            receipt = whole_run.board(work, tool)
            output = receipt['output']
            output_path = checked(work / 'board', output)
            row.update(status=receipt['status'], output=output)
            return output_path, output
        except Exception as exc:
            row.update(status='FAILED', error=str(exc))
            if row['stage'] == 'board' and (work / 'board/attempt.json').is_file():
                failed_boards.append(name)
                if isinstance(exc, whole_run.HiaiRunFailure):
                    row['stage'] = exc.stage
            print(f"{name} {row['stage']} FAILED: {exc}", flush=True)
            return None
        finally:
            save_json(root / 'projections.json', report)

    for i, op in enumerate(candidates):
        name = f'projection-{i:02d}'
        target = op.name + ':0'
        row = dict(node=op.name, probe=name, weight_files=weight_files(nodes, op.input[1]))
        report['candidates'].append(row)
        print(f"L{layer} {op.name}  {row['weight_files']}", flush=True)
        try:
            start = _project_input(nodes, target)
        except ValueError as exc:
            row.update(status='UNBOUND', error=str(exc))
            continue
        if start not in prefixes:
            prefix = f'input-{i:02d}'
            prefixes[start] = probe(prefix, residual, start, original, False), prefix
        feed, prefix = prefixes[start]
        row.update(activation_edge=start, input_probe=prefix)
        if feed is None:
            row['status'] = 'INPUT_UNAVAILABLE'
        else:
            probe(name, start, target, feed, True)
            row['status'] = report['probes'][name]['status']
        save_json(root / 'projections.json', report)
    save_json(root / 'projections.json', report)
    for row in report['candidates']:
        phase = report['probes'].get(row['probe'], {}).get('stage')
        status = f"FAILED at {phase}" if row['status'] == 'FAILED' else row['status']
        print(f"{row['node']}  {row['weight_files']}  input={row.get('input_probe')}  {status}", flush=True)
    print(root / 'projections.json', flush=True)
    if failed_boards:
        diagnose_decoder_pair(root, names=failed_boards)
    return all(row['status'] == 'EXECUTED' for row in report['candidates'])
