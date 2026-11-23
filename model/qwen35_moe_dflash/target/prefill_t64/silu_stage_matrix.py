"""Read the six boundaries of the split board CPU-SiLU route and compare all three backends."""
import csv
from pathlib import Path
import tempfile

import numpy as np

from ...execution.runtime_config import read_json
from ...execution.serial_transport import _environment
from ..layer_probe.common import checked, read_array, save_json
from .attention_cpu_silu import CASE
from .decode_cache_trace import _record
from .gdn_op_matrix import BACKENDS, valid19_matrix_rows
from .silu_split_run import compile_stage, execute_stage, stage_outputs
from .silu_stage_reference import WIDTH, load_bundle, valid_rows, same_bits
from .whole_run import kernel_identity


def cached(previous, stage, dirname, feeds, board):
    for split in previous:
        out = split/dirname
        if not all(path.is_file() for path in (out/'result.json', out/'slots.json', split/stage/'boundaries.json')):
            continue
        receipt = read_json(out/'result.json')
        bounds = read_json(split/stage/'boundaries.json')
        if (receipt.get('status') != 'EXECUTED' or receipt.get('case') != stage
                or (receipt['device'], receipt['server']) != board or receipt['omc_sha256'] != bounds['omc_sha256']):
            continue
        slots = read_json(out/'slots.json')['inputs']
        names = [s['name'] for s in slots]
        if set(names) != {s['name'] for s in bounds['io']['inputs']} or len(set(names)) != len(names):
            raise ValueError(f'Inconsistent cached input binding at {out}')
        if (any(k not in feeds for k in names) or receipt['input_hashes'] != [feeds[k][1]['sha256'] for k in names]
                or any((s['shape'], s['dtype']) != (feeds[s['name']][1]['shape'], feeds[s['name']][1]['dtype'])
                       for s in slots)):
            continue
        try:
            values = stage_outputs(out, bounds['io'], receipt)
        except (OSError, ValueError) as exc:
            print(f'Skipping incomplete {out}: {exc}', flush=True)
            continue
        print('Reusing', out, flush=True)
        return values
    return None


def matrix_report(path, values):
    stages = [k for k in WIDTH if k in values['cpro']]
    with tempfile.NamedTemporaryFile(mode='w', newline='', dir=path.parent, delete=False) as f:
        writer = csv.DictWriter(f, fieldnames=['operator', 'scope', 'metric', 'actual', *BACKENDS])
        writer.writeheader()
        for name in stages:
            for row in valid19_matrix_rows(name, {k: values[k][name] for k in BACKENDS}):
                writer.writerow(row)
                if row['scope'] in ('valid19', 'token18') and row['metric'] in ('cosine', 'relative_l2', 'error_ratio'):
                    if row['actual'] == BACKENDS[0]:
                        print(f"\n{name}  {row['scope']}  {row['metric']}\nactual\\reference              CPU           310P           CPro")
                    print(row['actual'].ljust(16), *(format(row[k], '14.9g') if row[k] is not None
                                                      else '     undefined' for k in BACKENDS))
    Path(f.name).replace(path)
    print('Three-backend stage matrices:', path, flush=True)


def run_matrices(run, refs, ddk, source, stamp, config, diagnosis):
    _, case, folder, spec = _record(run/'air', CASE)
    digest = case['export']['sha256']
    values = {b: load_bundle(refs, b, digest) for b in ('cpu', '310p')}
    evidence = {b: read_json(refs/b/'manifest.json') for b in values}
    if any(evidence['cpu'][key] != evidence['310p'][key] for key in ('config_sha256', 'driver_sha256')):
        raise ValueError('CPU and 310P reference bundles use different source configurations or drivers')
    golden = valid_rows('out_proj_output', read_array(folder, spec['golden']))
    if not same_bits(values['310p']['out_proj_output'], golden):
        raise ValueError('310P references do not reproduce the staged valid19 golden')
    feeds = {s['name']: (checked(folder, s), s) for s in spec['source_inputs']}
    previous = []
    for path in sorted(run.glob('split-silu-*/source.json'), key=lambda p: p.stat().st_mtime_ns, reverse=True):
        saved = read_json(path)
        if all(saved.get(k) == v for k, v in stamp.items()):
            previous.append(path.parent)
    work = Path(tempfile.mkdtemp(dir=run, prefix='split-silu-matrices-'))
    print('L0 stage comparison:', work, flush=True)
    save_json(work/'source.json', stamp)
    device = tuple(_environment()[1:3])
    runtime = read_json(config.parent/read_json(config)['assets'])['environment']
    if runtime['ddk_id'] != stamp['ddk_id']:
        raise ValueError('Runtime DDK differs from the experiment')

    def obtain(stage, dirname, inputs, shapes=None):
        found = cached(previous, stage, dirname, inputs, device)
        if found is not None:
            return found
        saved = None
        for split in previous:
            record = split/stage/'report.json'
            if record.is_file():
                build = read_json(record)
                if build.get('status') == 'COMPILED' and (record.parent/build['omc']['path']).is_file():
                    saved = split
                    break
        build = compile_stage(work, stage, checked(folder, spec['air']), spec, ddk,
                              stamp['ddk_id'], diagnosis, shapes, reuse=saved)
        if kernel_identity(ddk, source) != stamp['gdr_source_sha256']:
            raise ValueError('Installed GDR changed during compilation')
        return execute_stage(work/dirname, stage, build, inputs, runtime, config)

    before = obtain('prefix', 'run-prefix', feeds)
    shapes = {k: before['split_' + k + '_in'][1]['shape'] for k in ('conv', 'gate')}
    after = obtain('cpu_silu', 'run-cpu_silu', before, shapes)
    suffix_inputs = {**feeds, **after}
    old = cached(previous, 'suffix', 'run-suffix-cpu_silu', suffix_inputs, device)

    def array(wire):
        path, row = wire
        return read_array(path.parent, dict(row, path=path.name))

    wires = dict(conv_silu_in=before['split_conv_in'], conv_silu_out=after['split_conv_out'],
                 gate_silu_in=before['split_gate_in'], gate_silu_out=after['split_gate_out'])
    if old is not None:
        wires['out_proj_output'] = old['golden_hidden']

    def report_available():
        values['cpro'] = {k: valid_rows(k, array(w)) for k, w in wires.items()}
        matrix_report(work/'matrices-available.csv', values)

    try:
        traced = obtain('suffix_proj_input', 'run-readout', suffix_inputs, shapes)
    except (OSError, ValueError, RuntimeError):
        report_available()
        raise
    changed = None
    if old is not None:
        a, b = (array(w['golden_hidden']).reshape(64, 2048) for w in (old, traced))
        changed = int(np.count_nonzero(a.view(np.uint16) != b.view(np.uint16)))
        print(f'Diagnostic vs previous CPU-group out_proj: {changed}/{a.size} T64 FP16 elements changed', flush=True)
        if changed:
            report_available()
            raise ValueError(f'Tapped suffix changed out_proj; inspect {traced["golden_hidden"][0]} and {old["golden_hidden"][0]}')
    else:
        print('No matching archived suffix output; reporting both outputs of the diagnostic suffix.', flush=True)
        wires['out_proj_output'] = traced['golden_hidden']
    wires['out_proj_input'] = traced['out_proj_input']
    values['cpro'] = {name: valid_rows(name, array(w)) for name, w in wires.items()}
    save_json(work/'comparison-inputs.json', dict(case_sha256=digest, reference_dir=str(refs),
              changed_t64_fp16_elements=changed, previous_suffix_found=old is not None,
              diagnostic_output=str(traced['golden_hidden'][0]),
              cpro={k: dict(v[1], path=str(v[0])) for k, v in wires.items()}))
    matrix_report(work/'matrices.csv', values)
