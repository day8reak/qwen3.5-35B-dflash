"""Check the DDK format lookup and replay the two failing SiluPatch edges on 53."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tempfile

from diagnose_silu_format import REGISTRATION, compile_probe, downstream, format_failure, graph_message, output_path
from qwen35_moe_dflash.toolchain.compile import omg_environment


CASES = {'gate-mul': ('Mul', 'Cast'), 'output-cast-reshape': ('Cast',), 'output-reshape': ('Reshape',)}
PORTS = {'producer': (0,), 'consumer': (1,), 'both': (0, 1)}


def inspect_ddk(ddk):
    omg, env = omg_environment(ddk, soc_version=None)
    registration = ddk / REGISTRATION
    issues = []
    try:
        opinfo = json.loads(registration.read_bytes())
    except (OSError, ValueError) as error:
        issues.append(str(error))
        opinfo = {}
    configured = env.get('KIRIN_CONF_JSON_PATH')
    hits = {}
    for root in (configured or '').split(':'):
        if not root or not Path(root).is_dir():
            continue
        for file in sorted(Path(root).rglob('*')):
            if file.suffix.lower() not in {'.json', '.ini', '.cfg', '.xml'} or not file.is_file():
                continue
            matches = set()
            try:
                with file.open('rb') as stream:
                    for line in stream:
                        matches.update(m.decode() for m in re.findall(
                            rb'\b(?:silu_?patch|chunk_?gated_?delta_?rule)\b', line, re.I))
            except OSError as error:
                issues.append(str(error))
            if matches:
                hits[str(file)] = sorted(matches)
    sources = {}
    for file in sorted(ddk.rglob('format_converter_pass.cc')):
        try:
            lines = file.read_text(errors='replace').splitlines()
        except OSError as error:
            issues.append(str(error))
            continue
        for i, line in enumerate(lines):
            if re.search(r'::FindOpFormatConfig\s*\(', line):
                sources[f'{file}:{i+1}'] = lines[i:i+36]
    ports = {op: {k: opinfo.get(op, {}).get(k) for k in ('input0', 'output0')}
             for op in ('SiluPatch', 'ChunkGatedDeltaRule')}
    print(f'OMG {omg}\nRegistration {registration}: {json.dumps(ports)}\n'
          f'KIRIN_CONF_JSON_PATH={configured}; named config hits={hits}\n'
          f'FindOpFormatConfig source locations in this DDK: {list(sources)}', flush=True)
    return dict(omg=str(omg), registration_file=str(registration), registered_ports=ports,
                config_path=configured, config_hits=hits, lookup_source=sources, read_issues=issues)


def edge_variant(raw, case, sides):
    msg, graph = graph_message(raw, 'SiluPatch')
    silu, = [op for op in graph.op if op.type == 'SiluPatch']
    edges = downstream(graph, silu.name)
    if len(edges) != 1:
        raise ValueError(f'{case}: expected one SiluPatch consumer, found {len(edges)}')
    consumer, slot = edges[0]
    first_ops = CASES[case]
    x = next(op for op in graph.op if op.name + ':0' == silu.input[0])
    shape = tuple(x.output_desc[0].shape.dim)
    if (consumer.type not in first_ops or (consumer.type != 'Mul' and slot != 0)
            or x.type != 'Data' or not shape or min(shape) <= 0
            or (x.output_desc[0].dtype, silu.input_desc[0].dtype) != (2, 2)):
        raise ValueError(f'{case}: unexpected SiLU input or consumer in retained AIR')
    descs = [silu.output_desc[0], consumer.input_desc[slot]]
    for i in sides:
        desc = descs[i]
        dims = tuple(desc.shape.dim)
        if desc.dtype not in (1, 2) or desc.dtype != silu.output_desc[0].dtype or (dims and dims != (-2,) and (
                len(dims) != len(shape) or any(d >= 0 and d != n for d, n in zip(dims, shape)))):
            raise ValueError(f'{case}: SiluPatch edge {shape} conflicts with {desc.dtype} {dims}')
        if desc.layout not in ('', 'ND'):
            raise ValueError(f'{case}: explicit {desc.layout} on edge; cannot fill ND')
        desc.shape.dim[:] = shape
        desc.layout = 'ND'
        # The standalone probe's ND hints, without changing dtype, device or other op metadata.
        for key, field, value in (('origin_format', 's', b'ND'), ('format_for_int', 'i', 2),
                                  ('origin_format_for_int', 'i', 2), ('origin_shape_initialized', 'b', False)):
            if key in desc.attr:
                if key != 'origin_shape_initialized' and getattr(desc.attr[key], field) != value:
                    raise ValueError(f'{case}: explicit {key} conflicts with ND')
            else:
                setattr(desc.attr[key], field, value)
    return msg.SerializeToString(deterministic=True)


def run_check(directory, ddk, diagnosis):
    saved = json.loads((directory / 'probes.json').read_bytes())
    out = Path(tempfile.mkdtemp(prefix='format-lookup-', dir=directory))
    result = dict(input_report=str(directory / 'probes.json'), air_sha256=saved['air_sha256'],
                  ddk=inspect_ddk(ddk), cases={})
    for case in CASES:
        old = saved['cases'].get(case, {})
        if old.get('status') != 'EMPTY_OUTPUT_FORMAT':
            continue
        log = Path(old['log']).resolve(strict=True)
        if (directory not in log.parents or old['status'] != 'EMPTY_OUTPUT_FORMAT'
                or format_failure(log, {'silu_patch'}) != 'silu_patch'):
            raise ValueError(f'{case}: expected the retained empty-output-format reproduction')
        air = log.with_name('export.air')
        raw = air.read_bytes()
        folder = out / case
        folder.mkdir()
        rows = {}
        result['cases'][case] = dict(source=str(air), sha256=hashlib.sha256(raw).hexdigest(), results=rows)
        rows['original'] = compile_probe(folder / 'original', raw, ddk, diagnosis)
        if rows['original']['status'] != 'EMPTY_OUTPUT_FORMAT':
            continue
        # These probe AIRs are deterministic. Compare canonical encodings to skip no-op edits.
        canonical = edge_variant(raw, case, ())
        seen = {canonical: 'original'}
        for name, sides in PORTS.items():
            try:
                variant = edge_variant(raw, case, sides)
            except ValueError as error:
                rows[name] = dict(status='SKIPPED', detail=str(error))
                continue
            if variant in seen:
                prior = seen[variant]
                rows[name] = dict(rows[prior], identical_to=prior)
            else:
                rows[name] = compile_probe(folder / name, variant, ddk, diagnosis)
                seen[variant] = name
    passed = {}
    for case, evidence in result['cases'].items():
        rows = evidence['results']
        passed[case] = [n for n, row in rows.items() if row['status'] == 'COMPILED']
        print(f'{case}: ' + ', '.join(f'{n}={v["status"]}' for n, v in rows.items()))
    if not passed:
        decision = 'No retained case reproduced the empty output-format error.'
    elif all('original' in names for names in passed.values()):
        decision = 'The retained reproducers compile with the current DDK. Retry the unmodified four-layer compile.'
    elif all(passed.values()):
        decision = 'Each retained site has a compiling candidate. Validate the passing port choices in a four-layer AIR.'
    elif any(all(row['status'] == 'EMPTY_OUTPUT_FORMAT' for row in c['results'].values())
             for c in result['cases'].values()):
        decision = 'At least one reproducer still fails with all tested ND edge variants. Inspect the DDK format lookup with that AIR and log.'
    else:
        decision = 'Inconclusive. Inspect the skipped variants or the other compile failures before changing the model.'
    result.update(compiled_variants=passed, decision=decision)
    path = out / 'result.json'
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(f'{decision}\nReport: {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--host-check', action='store_true', help='Compare installed and compiled output types')
    parser.add_argument('--host-library', type=Path, help='Optional host .so from the new build for content identity')
    mode.add_argument('--trace', action='store_true', help='Compare OMG format modes and trace file opens')
    parser.add_argument('--diagnostic-dir', type=Path)
    parser.add_argument('--ddk', type=Path, required=True)
    parser.add_argument('--tbe-diagnosis', type=Path)
    args = parser.parse_args()
    parent = output_path(args.run) / 'no-platform'
    if args.diagnostic_dir:
        directory = output_path(args.diagnostic_dir)
    else:
        reports = sorted(parent.glob('silu-diagnosis-*/probes.json'), key=lambda p: p.stat().st_mtime_ns)
        if not reports:
            parser.error(f'No previous probes.json in {parent}; run aligned-diagnose-next first')
        directory = reports[-1].parent
    if directory.parent != parent:
        parser.error('The diagnosis directory must belong to this four-layer run')
    if args.host_check:
        from silu_host_check import run_host_check
        run_host_check(directory, args.ddk.resolve(strict=True), args.tbe_diagnosis, args.host_library)
    elif args.trace:
        from silu_format_trace import run_trace
        run_trace(directory, args.ddk.resolve(strict=True), args.tbe_diagnosis)
    else:
        run_check(directory, args.ddk.resolve(strict=True), args.tbe_diagnosis)
