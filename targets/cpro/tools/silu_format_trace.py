"""Compare preserved SiluPatch AIRs across OMG format modes on the compile host."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from diagnose_silu_format import REGISTRATION, compile_probe, downstream, graph_message, standalone_for_ports


def handwritten_gate(raw):
    """Change only the gate's SiluPatch node to the static, non-TorchAir definition."""
    msg, graph = graph_message(raw, 'SiluPatch')
    silu, = [op for op in graph.op if op.type == 'SiluPatch']
    data, = [op for op in graph.op if [op.name + ':0'] == list(silu.input)]
    shape = tuple(data.output_desc[0].shape.dim)
    edges = downstream(graph, silu.name)
    if len(edges) == 1 and edges[0][0].type == 'Cast':
        edges = downstream(graph, edges[0][0].name)
    if shape != (2048, 128) or len(edges) != 1 or edges[0][0].type != 'Mul':
        raise ValueError('Expected the retained native [2048,128] gate feeding Mul')
    _, manual = graph_message(standalone_for_ports(shape, silu.output_desc[0].dtype), 'SiluPatch')
    new, = [op for op in manual.op if op.type == 'SiluPatch']
    new.name, new.id = silu.name, silu.id
    new.input[:] = silu.input
    silu.ParseFromString(new.SerializeToString())
    return msg.SerializeToString(deterministic=True)


def opened_resources(folder, ddk):
    # -ff gives each process a separate file. -y resolves relative opens on successful return.
    opened = set()
    captured = 0
    for path in folder.glob('opens.*'):
        with path.open(errors='replace') as lines:
            for line in lines:
                match = re.search(r'=\s*\d+<(/[^>]*)>\s*$', line)
                if not match or ('O_RDONLY' not in line and 'O_RDWR' not in line):
                    continue
                captured += 1
                file = Path(match[1])
                if file.suffix.lower() in {'.json', '.ini', '.cfg', '.conf', '.xml', '.yaml', '.yml'} or (
                        ddk in file.parents and ('.so' in file.suffixes or file.name == 'silu_patch.py')):
                    opened.add(str(file))
    return dict(decoded_opens=captured, resources=sorted(opened),
                registration_open_observed=str((ddk / REGISTRATION).resolve()) in opened if captured else None)


def run_trace(directory, ddk, diagnosis):
    saved = json.loads((directory / 'probes.json').read_bytes())
    out = Path(tempfile.mkdtemp(prefix='format-trace-', dir=directory))
    report = dict(source=str(directory / 'probes.json'), ddk=str(ddk), cases={})
    inputs = {}
    for name in ('gate-standalone', 'gate-mul', 'output-reshape', 'output-cast-reshape'):
        if name.startswith('output-') and 'log' not in saved['cases'].get(name, {}):
            continue
        path = Path(saved['cases'][name]['log']).with_name('export.air').resolve(strict=True)
        if directory not in path.parents:
            raise ValueError(f'{name}: probe belongs to another diagnosis: {path}')
        inputs[name] = path.read_bytes()
        report['cases'][name] = dict(source=str(path))
    inputs['handwritten-gate-mul'] = handwritten_gate(inputs['gate-mul'])
    report['cases']['handwritten-gate-mul'] = dict(
        source='gate-mul with standalone-generator SiluPatch node',
        identical_air_to_gate_mul=inputs['handwritten-gate-mul'] == inputs['gate-mul'])
    for name, raw in inputs.items():
        case = report['cases'][name]
        case['sha256'] = hashlib.sha256(raw).hexdigest()
        folder = out / name
        folder.mkdir()
        case['modes'] = {}
        for mode in (True, False):
            label = str(mode).lower()
            case['modes'][label] = compile_probe(folder / label, raw, ddk, diagnosis, use_origin_format=mode)

    # Keep the A/B above independent of tracer availability and instrumentation.
    tracer = shutil.which('strace')
    trace = report['trace'] = {}
    if tracer:
        prefix = [tracer, '-ff', '-y', '-s', '1024', '-e', 'trace=open,openat']
        check = subprocess.run([*prefix, '-o', str(out / 'preflight'), '/bin/true'], capture_output=True, text=True)
        if check.returncode:
            trace['unavailable'] = check.stderr.strip() or f'strace preflight exit {check.returncode}'
    else:
        trace['unavailable'] = 'strace not installed'
    if 'unavailable' in trace:
        print(f'File tracing unavailable: {trace["unavailable"]}')
    else:
        for name in ('gate-standalone', 'gate-mul'):
            folder = out / name / 'traced-true'
            result = compile_probe(folder, inputs[name], ddk, diagnosis, launcher=[*prefix, '-o', str(folder / 'opens')])
            observed = opened_resources(folder, ddk)
            trace[name] = dict(**observed, status=result['status'], log=result['log'],
                               matches_untraced=result['status'] == report['cases'][name]['modes']['true']['status'])
            print(f'{name}: registration open observed={observed["registration_open_observed"]}, '
                  f'decoded file opens={observed["decoded_opens"]}')
    for name, case in report['cases'].items():
        modes = case['modes']
        print(f'{name:24} true={modes["true"]["status"]:20} false={modes["false"]["status"]}')
    path = out / 'result.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Report: {path}')
