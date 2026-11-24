"""Inspect the retained four-layer AIR and isolate the failing SiluPatch format lookup on 53."""
import argparse
from collections import Counter
from difflib import unified_diff
import json
from pathlib import Path
from math import prod
import re
import tempfile

from decoder4_original import ALIGNED_KIND, EXPORT_DIR, VECTOR_ALIGNED_KIND, checked, compile_omg, output_path
from export_silu_patch_air import build_air
from qwen35_moe_dflash.target.gdn_probe import REGISTRATION
from qwen35_moe_dflash.target.prefill_t64.decode_cache_control import graph_message
from qwen35_moe_dflash.target.prefill_t64.decoder_four_silu import vector_length
from qwen35_moe_dflash.target.prefill_t64.silu_probe_graphs import downstream, with_consumer, with_source_squeeze, without_silu
from qwen35_moe_dflash.toolchain.existing_air import select_generated_omc


def format_failure(log, names):
    pattern = re.compile(r'outputTensorFormats size of node:\s*(\S+) is 0, too small', re.I)
    with log.open(errors='replace') as lines:
        for line in lines:
            match = pattern.search(line)
            if match and match[1] in names:
                return match[1]
    return None


def format_trace(log, name):
    methods = re.compile(r'OpCheckSupport|CheckSupported|opDeviceSupport|GetOutputMatchedFormatConfig|'
                         r'GetOpSupportedFormat|FindOpFormatConfig|ConnectedNetouputFormatProc', re.I)
    node = re.compile(r'(?:\bname\s*[:=]\s*|\bop\s*=\s*|\bnode\s*:\s*|\bnode\s+|\bop\[)'
                      + re.escape(name) + r'(?!\w)', re.I)
    rows = {}
    with log.open(errors='replace') as lines:
        for i, line in enumerate(lines, 1):
            if methods.search(line) and node.search(line):
                msg = line.partition('::')[-1].strip()
                if msg not in rows:
                    rows[msg] = f'L{i} {line.strip()[:700]}'
    return list(rows.values())


def outcome(folder, rc, failed):
    result = dict(returncode=rc, empty_format_node=failed, log=str(folder / 'compile.log'))
    if rc != 0:
        return dict(result, status='EMPTY_OUTPUT_FORMAT' if failed == 'silu_patch' else 'OTHER_FAILURE')
    try:
        model = select_generated_omc(folder, required=False)
    except ValueError as error:
        return dict(result, status='INVALID_OMC', detail=str(error))
    if model is None:
        return dict(result, status='OMC_NOT_FOUND', detail=f'Check {folder}/compile.log for the save path')
    return dict(result, status='COMPILED', model=str(model), bytes=model.stat().st_size)


def compile_probe(folder, raw, ddk, diagnosis):
    folder.mkdir()
    (folder / 'export.air').write_bytes(raw)
    rc, _, _ = compile_omg(folder, 'export.air', ddk, diagnosis, platform=None)
    failed = format_failure(folder / 'compile.log', {'silu_patch'})
    result = outcome(folder, rc, failed)
    print(f'{folder.name}: {json.dumps(result)}', flush=True)
    if rc:
        print(f'Failing probe log: {folder / "compile.log"}', flush=True)
    return result


def isolated_cases(op, shape, out=None):
    # Change only the op attributes and port descriptors. Producers, consumers,
    # node names and graph-level metadata are identical in the two controls.
    raw = build_air(shape=shape)
    message, graph = graph_message(raw, 'SiluPatch')
    probe, = [n for n in graph.op if n.type == 'SiluPatch']
    canonical = probe.SerializeToString(deterministic=True)
    before = str(probe).splitlines(keepends=True)
    for field in ('input_desc', 'output_desc'):
        probe.ClearField(field)
        for desc in getattr(op, field):
            getattr(probe, field).add().ParseFromString(desc.SerializeToString())
    probe.ClearField('attr')
    for key, value in op.attr.items():
        probe.attr[key].ParseFromString(value.SerializeToString())
    yield 'v3-static', raw
    if probe.SerializeToString(deterministic=True) != canonical:
        if out is not None:
            diff = unified_diff(before, str(probe).splitlines(keepends=True),
                                fromfile='v3-static', tofile='four-layer-ports')
            (out / 'ports.diff').write_text(''.join(diff))
        yield 'four-layer-ports', message.SerializeToString(deterministic=True)


def diagnose(base, ddk, diagnosis):
    air = base / 'air'
    manifest = json.loads((air / 'manifest.json').read_bytes())
    if manifest['kind'] not in (ALIGNED_KIND, VECTOR_ALIGNED_KIND):
        raise ValueError(f'Expected the aligned four-layer export in {air}')
    entry, = [r for r in manifest['files'] if r['path'] == EXPORT_DIR + '/export.air']
    path = checked(air, entry)
    _, graph = graph_message(path.read_bytes(), 'SiluPatch')
    nodes = {op.name: op for op in graph.op if op.type == 'SiluPatch'}
    print(f'AIR {path} sha256={entry["sha256"]}')
    producers = {op.name: op for op in graph.op}
    if manifest['kind'] == ALIGNED_KIND:
        shapes = {n['name']: tuple(n['shape']) for n in manifest['silu_patch']['nodes']}
        if set(shapes) != set(nodes):
            raise ValueError('SiluPatch manifest and AIR nodes differ')
    else:
        shapes = {name: (vector_length(op, producers),) for name, op in nodes.items()}
    counts = Counter(prod(shape) for shape in shapes.values())
    print('SiluPatch FP16 -> FP32, static shape counts:', dict(Counter(shapes.values())))
    run = base / 'no-platform'
    logs = list(run.glob('omg-*/compile.log'))
    if not logs:
        raise ValueError(f'No retained failed OMG attempt in {run}')
    log = max(logs, key=lambda p: (p.stat().st_mtime_ns, p))
    name = format_failure(log, nodes)
    if name not in nodes:
        raise ValueError(f'{log}: no SiluPatch found at the reported empty format list ({name})')
    op = nodes[name]
    consumers = [f'{n.name}({n.type}):input{i}' for n in graph.op
                 for i, edge in enumerate(n.input) if edge == name + ':0']
    print(f'OMG {log}\n{name}: x={list(op.input)} z -> {consumers}')
    for label, desc in (('x', op.input_desc[0]), ('z', op.output_desc[0])):
        print(f'{label}: dtype={desc.dtype}, shape={list(desc.shape.dim)}, layout={desc.layout!r}')
    selection_lines = format_trace(log, name)
    for line in selection_lines:
        print(line)
    # This is the same registration file the project checks for a working GDR.
    # Entries are evidence on disk; the compiler log establishes runtime selection.
    reg = ddk / REGISTRATION
    registration = {}
    if reg.is_file():
        entries = json.loads(reg.read_bytes())
        for key in ('SiluPatch', 'ChunkGatedDeltaRule'):
            row = entries.get(key)
            fields = ({k: row.get(k) for k in ('input0', 'output0', 'opFile')} if isinstance(row, dict)
                      else 'entry missing')
            registration[key] = fields
            print(f'{reg} [{key}] {json.dumps(fields, ensure_ascii=False)}')
    else:
        print(f'Registration file absent at {reg}')

    out = Path(tempfile.mkdtemp(prefix='silu-diagnosis-', dir=run))
    print(f'Diagnosis directory: {out}', flush=True)
    results = {}
    for label, raw in isolated_cases(op, shapes[name], out):
        results[label] = compile_probe(out / label, raw, ddk, diagnosis)
    if 'four-layer-ports' not in results:
        print('four-layer-ports: identical to v3-static; one compile covers both.')
    (out / 'result.json').write_text(json.dumps(dict(four_layer_log=str(log), failing_node=name,
        air_sha256=entry['sha256'], static_lengths=counts, failing_shape=shapes[name], inputs=list(op.input), consumers=consumers,
        selection_lines=selection_lines, registration_file=str(reg), registration=registration,
        results=results), indent=2) + '\n')
    print(out / 'result.json')
    return out


def consumer_probe(directory, label, raw, details, ddk, diagnosis):
    """Reuse a previous consumer result only for the identical small AIR in this diagnosis."""
    for path in sorted(directory.glob('consumer-*/result.json'), key=lambda p: p.stat().st_mtime_ns,
                       reverse=True):
        folder = path.parent / label
        air = folder / 'export.air'
        if not air.is_file() or air.read_bytes() != raw or not (folder / 'compile.log').is_file():
            continue
        record = json.loads(path.read_bytes())
        if (record.get('source_node'), record.get('consumer')) != (details['source_node'], details['consumer']):
            continue
        row = record.get('result') or record['output_reshape']
        result = outcome(folder, row['returncode'], format_failure(folder / 'compile.log', {'silu_patch'}))
        if result['status'] in ('COMPILED', 'EMPTY_OUTPUT_FORMAT'):
            print(f'Reusing {path}\n{label}: {json.dumps(result)}', flush=True)
            return result

    out = Path(tempfile.mkdtemp(prefix='consumer-', dir=directory))
    result = compile_probe(out / label, raw, ddk, diagnosis)
    path = out / 'result.json'
    path.write_text(json.dumps(dict(details, probe=label, result=result), indent=2) + '\n')
    print(path)
    return result


def print_cases(directory):
    cases = {}
    for path in sorted(directory.glob('consumer-*/result.json'), key=lambda p: p.stat().st_mtime_ns):
        record = json.loads(path.read_bytes())
        if 'probe' in record:
            cases[record['probe']] = (path, record)
    for label, (path, record) in cases.items():
        folder = path.parent / label
        log = folder / 'compile.log'
        if log.is_file():
            row = record.get('result') or record['output_reshape']
            status = outcome(folder, row['returncode'], format_failure(log, {'silu_patch'}))['status']
            print(f'{label}: {status}  {log}')


def native_probes(directory, source, manifest, report, results, ddk, diagnosis):
    """Compile independent controls for the native convolution edge and the L0 gate."""
    from qwen35_moe_dflash.target.silu_patch_air import nd_desc

    source_node = report['failing_node']
    ports = 'four-layer-ports' if 'four-layer-ports' in results else 'v3-static'
    standalone = (directory / ports / 'export.air').read_bytes()
    std = (directory / 'v3-static/export.air').read_bytes()
    rows = dict(results)

    def probe(label, raw, consumer, node=source_node):
        details = dict(source_node=node, consumer=consumer)
        try:
            rows[label] = consumer_probe(directory, label, raw, details, ddk, diagnosis)
        except ValueError as error:
            rows[label] = dict(status='PROBE_ERROR', detail=str(error))
            print(f'{label}: {error}', flush=True)

    cast, slot = downstream(source, source_node)[0]
    if slot != 0 or cast.type != 'Cast':
        raise ValueError(f'Unexpected native convolution consumer {cast.name}:{slot}')

    raw = with_consumer(standalone, source, cast)
    probe('output-cast', raw, cast.name)
    probe('output-cast-without-silu', without_silu(raw), cast.name)
    if ports != 'v3-static':
        probe('v3-output-cast', with_consumer(std, source, cast), cast.name)

    try:
        squeeze_only, upstream = with_source_squeeze(standalone, source, source_node)
    except ValueError as error:
        rows['squeeze-silu'] = dict(status='SKIPPED', detail=str(error))
    else:
        probe('squeeze-only', squeeze_only, None)
        probe('squeeze-silu', upstream, None)
        probe('squeeze-silu-cast', with_consumer(upstream, source, cast), cast.name)

    following = downstream(source, cast.name)
    if len(following) == 1 and following[0][1] == 0 and following[0][0].type == 'Reshape':
        reshape = following[0][0]
        try:
            chain = with_consumer(raw, source, reshape)
        except ValueError as error:
            rows['output-cast-reshape'] = dict(status='SKIPPED', detail=str(error))
        else:
            names = [cast.name, reshape.name]
            probe('output-cast-reshape', chain, names)
            probe('output-cast-reshape-without-silu', without_silu(chain), names)
    else:
        rows['output-cast-reshape'] = dict(status='SKIPPED', detail=f'{cast.name} lacks a single downstream Reshape')

    # Both ranks use the same Cast attributes and explicit ND port metadata.
    # Only the tensor dimensions change. The source-descriptor cases above are independent.
    for shape in ((8192, 64), (1, 8192, 64)):
        rank = len(shape)
        base = build_air(shape=shape)
        if rank == 2:
            probe('nd-2d-standalone', base, None)
        static_cast = type(cast)()
        static_cast.CopyFrom(cast)
        for desc in (static_cast.input_desc[0], static_cast.output_desc[0]):
            dtype = desc.dtype
            desc.Clear()
            desc.dtype = dtype
            nd_desc(desc, shape)
        probe(f'nd-{rank}d-cast', with_consumer(base, source, static_cast), cast.name)

    gate = next((r for r in manifest['silu_patch']['nodes'] if r['shape'] == [2048, 128]), None)
    if gate is not None:
        edges = downstream(source, gate['name'])
        if len(edges) == 1 and edges[0][0].type == 'Mul' and edges[0][1] == 1:
            mul = edges[0][0]
            gate_op = next(op for op in source.op if op.name == gate['name'])
            # Use the original gate ports and attributes, in the same standalone harness.
            base = dict(isolated_cases(gate_op, gate['shape']))
            gate_raw = base.get('four-layer-ports', base['v3-static'])
            probe('gate-standalone', gate_raw, None, gate['name'])
            try:
                gate_mul = with_consumer(gate_raw, source, mul)
            except ValueError as error:
                rows['gate-mul'] = dict(status='SKIPPED', detail=str(error))
            else:
                probe('gate-mul', gate_mul, mul.name, gate['name'])
                probe('gate-mul-without-silu', without_silu(gate_mul), mul.name, gate['name'])
    if 'gate-mul' not in rows:
        rows['gate-mul'] = dict(status='SKIPPED', detail='No matching [2048,128] gate at Mul input 1')

    path = directory / 'probes.json'
    pairs = [
        ('nd-2d-standalone', 'v3-static', 'standalone ND rank 2 vs 3'),
        ('nd-2d-cast', 'nd-3d-cast', 'ND rank 2 vs 3 with the same Cast'),
        ('v3-static', ports, 'canonical vs exported SiluPatch ports and attributes'),
        ('v3-output-cast', 'output-cast', 'canonical vs exported SiluPatch ports and attributes with Cast'),
        (ports, 'output-cast', 'adding the original Cast'),
        ('output-cast-without-silu', 'output-cast', 'feeding Cast from FP32 Data vs SiluPatch'),
        ('output-cast', 'output-cast-reshape', 'adding the original Reshape after Cast'),
        ('output-cast-reshape-without-silu', 'output-cast-reshape', 'feeding Cast and Reshape from FP32 Data vs SiluPatch'),
        (ports, 'squeeze-silu', 'adding the upstream Squeeze'),
        ('output-cast', 'squeeze-silu-cast', 'adding the upstream Squeeze with Cast'),
        ('gate-standalone', 'gate-mul', 'adding the original FP32 gate Mul and RMSNorm input'),
        ('gate-mul-without-silu', 'gate-mul', 'feeding gate Mul from FP32 Data vs SiluPatch'),
    ]
    contrasts = []
    for a, b, change in pairs:
        if a in rows and b in rows and {rows[a]['status'], rows[b]['status']} == {'COMPILED', 'EMPTY_OUTPUT_FORMAT'}:
            contrasts.append(f'{change}: {a}={rows[a]["status"]}, {b}={rows[b]["status"]}')
    for row in rows.values():
        if 'log' in row:
            row['format_trace'] = format_trace(Path(row['log']), 'silu_patch')
    full_log = Path(report['four_layer_log'])
    full_trace = {op: format_trace(full_log, op) for op in (source_node, 'ChunkGatedDeltaRule')}
    with tempfile.NamedTemporaryFile(mode='w', dir=directory, delete=False) as staging:
        json.dump(dict(air_sha256=report['air_sha256'], four_layer_log=report['four_layer_log'],
                       failing_node=source_node, four_layer_format_trace=full_trace,
                       cases=rows, format_failure_contrasts=contrasts), staging, indent=2)
    Path(staging.name).replace(path)
    print(f'Probe results: {path}')
    for label, row in rows.items():
        print(f'{label:32} {row["status"]}  {row.get("detail", "")}')
    for line in contrasts:
        print(f'CONTRAST  {line}')
    if not any(r['status'] == 'EMPTY_OUTPUT_FORMAT' for r in rows.values()):
        print('The tested small graphs did not reproduce the four-layer empty output-format list. '
              'Inspect the four-layer partition; see individual statuses for incomplete controls.')


def review(directory, base, *, ddk=None, diagnosis=None):
    report = json.loads((directory / 'result.json').read_bytes())
    rows = report['results']
    if set(rows) not in ({'v3-static'}, {'v3-static', 'four-layer-ports'}):
        raise ValueError(f'Unexpected standalone cases in {directory}')
    results = {}
    for label, row in rows.items():
        result = outcome(directory / label, row['returncode'], row['empty_format_node'])
        print(f'{label}: {json.dumps(result)}', flush=True)
        results[label] = result
    if not ddk:
        print_cases(directory)
        return
    bundle = base / 'air'
    manifest = json.loads((bundle / 'manifest.json').read_bytes())
    entry, = [r for r in manifest['files'] if r['path'] == EXPORT_DIR + '/export.air']
    if manifest['kind'] not in (ALIGNED_KIND, VECTOR_ALIGNED_KIND) or report['air_sha256'] != entry['sha256']:
        raise ValueError('Retained standalone report and four-layer AIR differ')
    _, source = graph_message(checked(bundle, entry).read_bytes(), 'SiluPatch')
    label = 'four-layer-ports' if 'four-layer-ports' in rows else 'v3-static'
    source_node = report['failing_node']
    edges = downstream(source, source_node)
    if (report.get('failing_shape') == [1, 8192, 64] and manifest['kind'] == ALIGNED_KIND
            and len(edges) == 1 and edges[0][0].type == 'Cast' and edges[0][1] == 0):
        return native_probes(directory, source, manifest, report, results, ddk, diagnosis)
    if any(r['status'] != 'COMPILED' for r in results.values()):
        print('Standalone control failed. Diagnose its compile.log before adding a consumer.')
        return
    if len(edges) != 1 or edges[0][1] != 0 or edges[0][0].type not in ('Reshape', 'Cast'):
        print(f"Cannot isolate a single Cast or Reshape on input 0: {report['consumers']}")
        return
    consumer = edges[0][0]
    case = 'output-' + consumer.type.lower()
    raw = with_consumer((directory / label / 'export.air').read_bytes(), source, consumer)
    details = dict(controls=results, source_node=source_node, consumer=consumer.name)
    print(f'Checking {label} -> {consumer.name} ({consumer.type})', flush=True)
    result = consumer_probe(directory, case, raw, details, ddk, diagnosis)
    if result['status'] == 'EMPTY_OUTPUT_FORMAT' and label == 'four-layer-ports':
        other = with_consumer((directory / 'v3-static/export.air').read_bytes(), source, consumer)
        consumer_probe(directory, 'v3-' + case, other, details, ddk, diagnosis)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--ddk', type=Path)
    parser.add_argument('--tbe-diagnosis', type=Path)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--report', action='store_true', help='recheck saved OMG products without compiling')
    action.add_argument('--consumer', action='store_true', help='compile or reuse standalone controls and test the actual downstream edge')
    parser.add_argument('--diagnostic-dir', type=Path, help='select a previous silu-diagnosis directory')
    args = parser.parse_args()
    base = output_path(args.run)
    if not args.report and not args.ddk:
        parser.error('compilation requires --ddk')
    if args.consumer or args.report:
        directory = args.diagnostic_dir
        if directory is None:
            prior = list((base / 'no-platform').glob('silu-diagnosis-*/result.json'))
            if prior:
                directory = max(prior, key=lambda p: (p.stat().st_mtime_ns, p)).parent
            elif args.consumer:
                directory = diagnose(base, args.ddk.resolve(strict=True), args.tbe_diagnosis)
            else:
                parser.error('no saved diagnosis; run aligned-diagnose first')
        review(output_path(directory), base, ddk=args.ddk.resolve(strict=True) if args.consumer else None,
               diagnosis=args.tbe_diagnosis)
    else:
        diagnose(base, args.ddk.resolve(strict=True), args.tbe_diagnosis)
