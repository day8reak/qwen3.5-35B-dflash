"""Inspect the retained four-layer AIR and isolate the failing SiluPatch format lookup on 53."""
import argparse
from collections import Counter
from difflib import unified_diff
import json
from pathlib import Path
import re
import tempfile

from decoder4_original import ALIGNED_KIND, EXPORT_DIR, checked, compile_omg, output_path
from export_silu_patch_air import build_air
from qwen35_moe_dflash.target.gdn_probe import REGISTRATION
from qwen35_moe_dflash.target.prefill_t64.decode_cache_control import graph_message
from qwen35_moe_dflash.toolchain.existing_air import select_generated_omc


def format_failure(log, names):
    pattern = re.compile(r'outputTensorFormats size of node:\s*(\S+) is 0, too small', re.I)
    with log.open(errors='replace') as lines:
        for line in lines:
            match = pattern.search(line)
            if match and match[1] in names:
                return match[1]
    return None


def ports(op):
    return [(desc.dtype, tuple(desc.shape.dim), desc.layout)
            for desc in (*op.input_desc, *op.output_desc)]


def outcome(folder, rc, failed):
    result = dict(returncode=rc, empty_format_node=failed)
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
    return result


def isolated_cases(op, length, out):
    # Change only the op attributes and port descriptors. Producers, consumers,
    # node names and graph-level metadata are identical in the two controls.
    raw = build_air(length)
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
        diff = unified_diff(before, str(probe).splitlines(keepends=True),
                            fromfile='v3-static', tofile='four-layer-ports')
        (out / 'ports.diff').write_text(''.join(diff))
        yield 'four-layer-ports', message.SerializeToString(deterministic=True)


def diagnose(base, ddk, diagnosis):
    air = base / 'air'
    manifest = json.loads((air / 'manifest.json').read_bytes())
    if manifest['kind'] != ALIGNED_KIND:
        raise ValueError(f'Expected the aligned four-layer export in {air}')
    entry, = [r for r in manifest['files'] if r['path'] == EXPORT_DIR + '/export.air']
    path = checked(air, entry)
    _, graph = graph_message(path.read_bytes(), 'SiluPatch')
    nodes = {op.name: op for op in graph.op if op.type == 'SiluPatch'}
    print(f'AIR {path} sha256={entry["sha256"]}')
    counts = Counter()
    for op in nodes.values():
        signature = ports(op)
        if len(op.input_desc) != 1 or len(op.output_desc) != 1:
            raise ValueError(f'{op.name}: expected one input and one output, got {signature}')
        shape = signature[0][1]
        if len(shape) != 1 or shape[0] <= 0 or signature != [(2, shape, 'ND'), (1, shape, 'ND')]:
            raise ValueError(f'{op.name}: incompatible serialized x/z (dtype, dims, layout): {signature}')
        counts[shape[0]] += 1
    print('Static SiluPatch FP16[N] ND -> FP32[N] ND (N: count):', dict(sorted(counts.items())))
    run = base / 'no-platform'
    logs = list(run.glob('omg-*/compile.log'))
    if not logs:
        raise ValueError(f'No retained failed OMG attempt in {run}')
    log = max(logs, key=lambda p: (p.stat().st_mtime_ns, p))
    name = format_failure(log, nodes)
    if name not in nodes:
        raise ValueError(f'{log}: no SiluPatch found at the reported empty format list ({name})')
    op = nodes[name]
    length = op.input_desc[0].shape.dim[0]
    consumers = [f'{n.name}({n.type}):input{i}' for n in graph.op
                 for i, edge in enumerate(n.input) if edge == name + ':0']
    print(f'OMG {log}\n{name}: x={list(op.input)} z -> {consumers}')
    selection = re.compile(r'OpCheckSupport|CheckSupported|opDeviceSupport|GetOutputMatchedFormatConfig|GetOpSupportedFormat|FindOpFormatConfig', re.I)
    token = re.compile(r'(?<!\w)' + re.escape(name) + r'(?!\w)')
    selection_lines = []
    with log.open(errors='replace') as lines:
        seen = set()
        for number, line in enumerate(lines, 1):
            if token.search(line) and selection.search(line):
                msg = line.partition('::')[-1].strip()
                if msg not in seen:
                    selection_lines.append(f'L{number} {line.strip()[:700]}')
                    print(selection_lines[-1])
                    seen.add(msg)
                    if len(seen) == 20:
                        break
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
    for label, raw in isolated_cases(op, length, out):
        results[label] = compile_probe(out / label, raw, ddk, diagnosis)
    if 'four-layer-ports' not in results:
        print('four-layer-ports: identical to v3-static; one compile covers both.')
    (out / 'result.json').write_text(json.dumps(dict(four_layer_log=str(log), failing_node=name,
        air_sha256=entry['sha256'], static_lengths=counts, inputs=list(op.input), consumers=consumers,
        selection_lines=selection_lines, registration_file=str(reg), registration=registration,
        results=results), indent=2) + '\n')
    print(out / 'result.json')


def add_output_reshape(raw, source, name):
    """Append the exact source Reshape and its literal shape operands to the retained control."""
    edge = name + ':0'
    consumers = [n for n in source.op if edge in n.input]
    if len(consumers) != 1 or consumers[0].type != 'Reshape' or consumers[0].input[0] != edge:
        raise ValueError(f'{name}: expected one Reshape at output slot 0')
    reshape = consumers[0]
    nodes = {n.name: n for n in source.op}
    message, graph = graph_message(raw, 'SiluPatch')
    data, silu, net = graph.op
    if (data.type, silu.name, net.type) != ('Data', 'silu_patch', 'NetOutput') or len(reshape.output_desc) != 1:
        raise ValueError('Expected the retained single-op control and a single-output source Reshape')
    additions = []
    copied = {n.name for n in graph.op}
    for arg in reshape.input[1:]:
        if not arg:
            continue
        key, _, slot = arg.rpartition(':')
        const = nodes.get(key)
        if slot != '0' or const is None or const.type != 'Const' or any(const.input):
            raise ValueError(f'{reshape.name}: shape operand {arg} is not a literal Const in the AIR')
        if key not in copied:
            additions.append(const)
            copied.add(key)
    if reshape.name in copied:
        raise ValueError(f'Name collision in control: {reshape.name}')
    # Parse through bytes because each decoded AIR owns a separate protobuf schema pool.
    def append(src):
        op = graph.op.add()
        op.ParseFromString(src.SerializeToString())
        op.id = len(graph.op) - 1
        for field in ('src_name', 'src_index', 'dst_name', 'dst_index'):
            op.ClearField(field)
        return op
    saved_net = type(net)()
    saved_net.CopyFrom(net)
    del graph.op[-1]
    for const in additions:
        append(const)
    placed = append(reshape)
    placed.input[0] = silu.name + ':0'
    output = append(saved_net)
    output.input[0] = placed.name + ':0'
    output.input_desc[0].CopyFrom(placed.output_desc[0])
    graph.output[:] = [output.input[0]]
    return message.SerializeToString(deterministic=True), reshape.name


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
    if not ddk or any(r['status'] != 'COMPILED' for r in results.values()):
        return
    bundle = base / 'air'
    manifest = json.loads((bundle / 'manifest.json').read_bytes())
    entry, = [r for r in manifest['files'] if r['path'] == EXPORT_DIR + '/export.air']
    if manifest['kind'] != ALIGNED_KIND or report['air_sha256'] != entry['sha256']:
        raise ValueError('Retained standalone report and four-layer AIR differ')
    _, source = graph_message(checked(bundle, entry).read_bytes(), 'SiluPatch')
    label = 'four-layer-ports' if 'four-layer-ports' in rows else 'v3-static'
    raw, reshape = add_output_reshape((directory / label / 'export.air').read_bytes(), source, report['failing_node'])
    out = Path(tempfile.mkdtemp(prefix='consumer-', dir=directory))
    print(f'Reusing {label}; adding {reshape} after {report["failing_node"]}', flush=True)
    result = compile_probe(out / 'output-reshape', raw, ddk, diagnosis)
    path = out / 'result.json'
    path.write_text(json.dumps(dict(controls=results, source_node=report['failing_node'], consumer=reshape,
                                    output_reshape=result), indent=2) + '\n')
    print(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--ddk', type=Path)
    parser.add_argument('--tbe-diagnosis', type=Path)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--report', action='store_true', help='recheck saved OMG products without compiling')
    action.add_argument('--consumer', action='store_true', help='recheck saved products, then probe the output Reshape')
    parser.add_argument('--diagnostic-dir', type=Path, help='select a previous silu-diagnosis directory')
    args = parser.parse_args()
    base = output_path(args.run)
    if not args.report and not args.ddk:
        parser.error('compilation requires --ddk')
    if args.consumer or args.report:
        directory = args.diagnostic_dir
        if directory is None:
            prior = list((base / 'no-platform').glob('silu-diagnosis-*/result.json'))
            if not prior:
                parser.error('no saved diagnosis; run aligned-diagnose first')
            directory = max(prior, key=lambda p: (p.stat().st_mtime_ns, p)).parent
        review(output_path(directory), base, ddk=args.ddk.resolve(strict=True) if args.consumer else None,
               diagnosis=args.tbe_diagnosis)
    else:
        diagnose(base, args.ddk.resolve(strict=True), args.tbe_diagnosis)
