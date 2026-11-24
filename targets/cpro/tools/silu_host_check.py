"""Compare installed registrations, actual loader observations and compiled SiluPatch output."""
import json
from pathlib import Path
import re
import tempfile

from diagnose_silu_format import REGISTRATION, compile_probe, format_trace, graph_message
from parse_model_json import GE_PROTO, dtype, inspect, read_omc
from qwen35_moe_dflash.common.hashing import sha256_file


def registrations(ddk):
    base = (ddk / REGISTRATION).parents[1]
    result = {}
    for p in (ddk / REGISTRATION, base / 'info/aic-ascend310b-ops-info.json'):
        row = result[str(p)] = {}
        try:
            data = json.loads(p.read_bytes())
            row['sha256'] = sha256_file(p)
            row['entries'] = {name: data.get(name) for name in ('SiluPatch', 'ChunkGatedDeltaRule')}
            for name, entry in row['entries'].items():
                port = entry.get('output0') if isinstance(entry, dict) else None
                print(f'{p.name} {name}.output0 = {port}')
            peers = []
            for name, entry in data.items():
                if name == 'SiluPatch' or not isinstance(entry, dict):
                    continue
                for key, port in entry.items():
                    if not re.fullmatch(r'output\d+', key) or not isinstance(port, dict):
                        continue
                    labels = str(port.get('dtype', '')).split(',')
                    if any(dtype(label.strip(), {}) == 'FP32' for label in labels):
                        peers.append(dict(op=name, slot=key, port=port))
            row['other_registered_fp32_outputs'] = peers
            examples = ', '.join(f'{v["op"]}.{v["slot"]}' for v in peers[:6])
            print(f'{p.name}: {len(peers)} other registered FP32 outputs. {examples}')
        except (OSError, ValueError, AttributeError) as error:
            row['error'] = str(error)
            print(f'{p}: {error}')
    return result


def loader_observation(folder, expected):
    files = sorted(folder.glob('loader.*'))
    loaded = set()
    for path in files:
        with path.open(errors='replace') as lines:
            for line in lines:
                match = re.search(r'calling init:\s+(/[^\n]+)', line)
                if match:
                    loaded.add(Path(match[1].strip()))
    result = dict(trace_files=len(files), initialized_libraries=sorted(map(str, loaded)))
    if not loaded:
        print(f'{folder.name}: no library initialization paths captured')
    if expected:
        matches = []
        issues = []
        for p in sorted(loaded):
            try:
                if p.stat().st_size == expected['bytes'] and sha256_file(p) == expected['sha256']:
                    matches.append(str(p))
            except OSError as error:
                issues.append(str(error))
        result.update(expected_hash_matches=matches, unreadable=issues)
        print(f'{folder.name}: {len(loaded)} library init paths, expected host SHA256 matches={matches}')
    return result


def run_host_check(directory, ddk, diagnosis, host_library):
    saved = json.loads((directory / 'probes.json').read_bytes())
    folder = Path(tempfile.mkdtemp(prefix='host-check-', dir=directory))
    report = dict(ddk=str(ddk), registrations=registrations(ddk), probes={})
    expected = None
    if host_library:
        path = host_library.resolve(strict=True)
        expected = dict(path=str(path), sha256=sha256_file(path), bytes=path.stat().st_size)
        report['expected_library'] = expected

    def compile_case(name, raw):
        dest = folder / name
        launcher = ['env', 'LD_DEBUG=files', f'LD_DEBUG_OUTPUT={dest}/loader']
        row = compile_probe(dest, raw, ddk, diagnosis, launcher=launcher)
        report['probes'][name] = row
        row['air_sha256'] = sha256_file(dest / 'export.air')
        row['loader'] = loader_observation(dest, expected)
        row['format_lookup'] = format_trace(dest / 'compile.log', 'silu_patch')
        if row['status'] == 'COMPILED':
            try:
                doc, _ = read_omc(Path(row['model']), ddk_root=ddk)
                row['compiled_outputs'] = inspect(doc, enum=GE_PROTO)['outputs']
                print(f'{name} OMC outputs: {[(v["dtype"], v["shape"]) for v in row["compiled_outputs"]]}')
            except (OSError, ValueError) as error:
                row['descriptor_error'] = str(error)
                print(f'{name} OMC inspection: {error}')
        return row

    def retained(name):
        p = Path(saved['cases'][name]['log']).with_name('export.air').resolve(strict=True)
        if directory not in p.parents:
            raise ValueError(f'{p} is outside the requested diagnosis')
        raw = p.read_bytes()
        _, graph = graph_message(raw, 'SiluPatch')
        node, = [op for op in graph.op if op.type == 'SiluPatch']
        port = (GE_PROTO.get(node.input_desc[0].dtype), GE_PROTO.get(node.output_desc[0].dtype))
        if port != ('FP16', 'FP32'):
            raise ValueError(f'{name} retained AIR port is {port}, expected FP16 -> FP32')
        print(f'{name} retained AIR: {port[0]} -> {port[1]}')
        return raw

    gate = retained('gate-mul')
    standalone = retained('gate-standalone')
    ports = []
    for path, item in report['registrations'].items():
        entry = item.get('entries', {}).get('SiluPatch')
        port = entry.get('output0') if isinstance(entry, dict) else None
        if isinstance(port, dict) and isinstance(port.get('dtype'), str):
            ports.append((dtype(port['dtype'].strip(), {}), port.get('format')))
        else:
            print(f'{path}: SiluPatch output0.dtype unavailable')
    # Replay the saved singleton even when the files disagree: its public port
    # can be read from a fresh OMC independently of the installed descriptions.
    single = compile_case('gate-standalone', standalone)
    types = {kind for kind, _ in ports}
    if 'INT32' in types:
        msg, graph = graph_message(standalone, 'SiluPatch')
        node, = [op for op in graph.op if op.type == 'SiluPatch']
        net, = [op for op in graph.op if op.type == 'NetOutput']
        int32 = node.output_desc[0].DESCRIPTOR.fields_by_name['dtype'].enum_type.values_by_name['DT_INT32'].number
        node.output_desc[0].dtype = net.input_desc[0].dtype = int32
        compile_case('standalone-int32', msg.SerializeToString(deterministic=True))
    if len(ports) != 2 or ports[0] != ports[1] or ports[0][0] is None or not ports[0][1]:
        decision = 'The two installed SiluPatch output entries are incomplete or disagree. Verify the host package and the two output0 entries against the intended ABI.'
    elif types == {'INT32'}:
        decision = 'Both installed entries declare INT32. The original gate Mul takes FP32. Inspect both standalone OMC descriptors before designing any bit-preserving consumer.'
    elif types == {'FP32'}:
        mul = compile_case('gate-mul', gate)
        actual = [port['dtype'] for port in single.get('compiled_outputs', [])]
        if actual != ['FP32']:
            decision = f'The singleton FP32 ABI was not confirmed (status={single["status"]}, outputs={actual}); inspect its compile log and descriptor dump result.'
        elif mul['status'] == 'EMPTY_OUTPUT_FORMAT':
            decision = 'Singleton OMC output is FP32. The same gate producer with Mul still fails at the subgraph output format lookup. Use the gate-mul AIR and log to inspect FindOpFormatConfig in this DDK.'
        elif mul['status'] == 'COMPILED':
            decision = 'The FP32 gate pair now compiles. Test the retained output-cast-reshape case before recompiling four layers.'
        else:
            decision = f'Gate compile status is {mul["status"]}. Inspect its compile.log for the current failure.'
    else:
        decision = f'Unsupported installed output0: {ports}. Verify the intended ABI.'
    report['decision'] = decision
    path = folder / 'result.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'{decision}\nReport: {path}')
