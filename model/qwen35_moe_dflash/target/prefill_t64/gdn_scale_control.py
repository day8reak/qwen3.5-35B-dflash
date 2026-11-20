"""One scaled L0 hidden A/B from the previously exported valid19 module job."""
import ast
from collections import Counter
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from ..layer_probe.common import checked, save_json, sha
from .prepare import EXPORTER, MODELING, NZ, WRAPPER
from .segment_export import MODULE_BASIS
from .source_model import _call, _definition, _one

SCALE = 16
CASE = 'l00_c00_gdn_layer_hidden_o0'


def _scale_readout_query(tree, op, scale):
    if type(scale) is not int or scale not in (1, 16):
        raise ValueError('Readout pair supports factors 1 and 16')
    _definition(tree, 'l2norm')
    flag = _one((kw for kw in op.keywords if kw.arg == 'use_qk_l2norm_in_kernel'), 'GDR Q/K normalization')
    if not isinstance(flag.value, ast.Constant) or flag.value.value != 1:
        raise ValueError('Expected the baseline GDR to normalize Q/K')
    flag.value = ast.Constant(False)
    # Q enters both readout terms after normalization. The kernel keeps its
    # 1/sqrt(Dk) factor. K is normalized identically for the paired graphs.
    for slot, name in enumerate(('query', 'key')):
        keyword = None if slot < len(op.args) else _one(
            (kw for kw in op.keywords if kw.arg == name), f'GDR {name} input')
        operand = op.args[slot] if keyword is None else keyword.value
        if not isinstance(operand, ast.Name):
            raise ValueError(f'Expected a local {name} tensor in the L0 GDR call')
        # ReduceSumCpro tiles a matrix. Each (batch, token, head) gets its own row.
        rows = f'{operand.id}.reshape(-1, {operand.id}.shape[-1]).float()'
        expr = f'l2norm({rows}, dim=1, eps=1e-6)'
        if name == 'query' and scale != 1:
            expr = f'({expr} * {float(scale)})'
        normalized = ast.parse(f'{expr}.to({operand.id}.dtype).reshape({operand.id}.shape).contiguous()',
                               mode='eval').body
        if keyword is None:
            op.args[slot] = normalized
        else:
            keyword.value = normalized


def scaled_source(text, *, readout_scale=None):
    tree = ast.parse(text)
    cls = _definition(tree, 'Qwen3_5MoeGatedDeltaNet')
    forward = _definition(cls, 'forward')
    op = _call(forward, 'torch_npu.npu_chunk_gated_delta_rule')
    body, statement = _one(((suite, s) for node in ast.walk(forward)
                            for _, suite in ast.iter_fields(node) if isinstance(suite, list)
                            for s in suite if isinstance(s, ast.Assign) and s.value is op),
                           'original two-output GDR assignment')
    core, state = statement.targets[0].elts
    if not isinstance(state, ast.Name) or not isinstance(core, ast.Name):
        raise ValueError('Expected local core and state results')
    scale = SCALE if readout_scale is None else readout_scale
    if readout_scale is None:
        # For fixed Q/K/g/beta the recurrence is linear in V and initial state.
        factor = ast.Constant(float(scale))
        if len(op.args) >= 3:
            op.args[2] = ast.BinOp(op.args[2], ast.Mult(), factor)
        else:
            value = _one((kw for kw in op.keywords if kw.arg == 'value'), 'GDR value input')
            value.value = ast.BinOp(value.value, ast.Mult(), factor)
        initial = _one((kw for kw in op.keywords if kw.arg == 'initial_state'), 'GDR initial state')
        initial.value = ast.BinOp(initial.value, ast.Mult(), factor)
        pos = body.index(statement) + 1
        body[pos:pos] = ast.parse(f'{state.id} = {state.id} * {1. / scale}').body
    else:
        _scale_readout_query(tree, op, scale)
    # Keep core scaled through the norm. Unscaling it into FP16 before the norm
    # would recreate the same small values under investigation.
    _call(forward, 'self.norm')
    gated_classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Qwen3_5MoeRMSNormGated']
    rms_calls = [c for owner in gated_classes for c in ast.walk(owner)
                 if isinstance(c, ast.Call) and ast.unparse(c.func) == 'torch_npu.npu_rms_norm']
    if rms_calls:
        rms, = rms_calls
        multiple = ast.Constant(float(scale**2))
        if len(rms.args) >= 3:
            rms.args[2] = ast.BinOp(rms.args[2], ast.Mult(), multiple)
        else:
            epsilon = next((kw for kw in rms.keywords if kw.arg == 'epsilon'), None)
            if epsilon is None:
                rms.keywords.append(ast.keyword(arg='epsilon', value=ast.Constant(1e-6 * scale**2)))
            else:
                epsilon.value = ast.BinOp(epsilon.value, ast.Mult(), multiple)
    else:
        init = _definition(cls, '__init__')
        init.body.extend(ast.parse(f'self.norm.variance_epsilon *= {scale**2}').body)
    return ast.unparse(ast.fix_missing_locations(tree)) + '\n'


def _air_facts(root, case, schema):
    from ..original_air_roundtrip import _attributes, _decode
    from ..operator_air_review import _data_edges, port_names

    record_path = checked(root, case['export'])
    record = json.loads(record_path.read_text())
    folder = record_path.parent
    _, _, graph = _decode(schema, checked(folder, record['air']).read_bytes())
    nodes = {n['name']: n for n in graph['op']}
    gdr, = [n for n in nodes.values() if n['type'] == 'ChunkGatedDeltaRule']
    ports = port_names(_attributes(gdr), 'input')
    inputs = {ports[i]: edge for i, edge in enumerate(_data_edges(gdr)) if i in ports}

    def producer_type(edge):
        node = nodes[edge.rsplit(':', 1)[0]]
        while node['type'] in ('Cast', 'Identity', 'Reshape', 'TensorMove', 'Squeeze', 'Unsqueeze'):
            node = nodes[_data_edges(node)[0].rsplit(':', 1)[0]]
        return node['type']

    norm, = [n for n in nodes.values() if n['type'] == 'RmsNorm']
    eps = _attributes(norm).get('epsilon', {}).get('f', 1e-6)
    weights = Counter()
    for node in nodes.values():
        if node['type'] == 'FileConstant':
            attrs = _attributes(node)
            file = (folder / attrs['file_path']['s']).resolve()
            if folder not in file.parents:
                raise ValueError('External weight path escapes the case')
            weights[(sha(file), attrs.get('offset', {}).get('i'), attrs.get('length', {}).get('i'))] += 1
        elif node['type'] == 'Const':
            data = _attributes(node).get('value', {}).get('t', {}).get('data', {})
            if data.get('bytes', 0) >= 128:
                weights[(data['sha256'], None, None)] += 1
    return dict(epsilon=eps, weights=weights,
                norm_flag=_attributes(gdr).get('use_qk_l2norm_in_kernel', {}).get('i'),
                producers={n: producer_type(inputs[n]) for n in ('query', 'key', 'value', 'initial_state')})


def _hidden_reference(folder, record):
    hidden = record['golden']
    # Early single-output batches store only golden. Later batches also list every output.
    if hidden['name'] != 'golden_hidden' or record.get('goldens', [hidden]) != [hidden]:
        raise ValueError('Expected a single golden_hidden export reference')
    checked(folder, hidden)
    return {key: hidden[key] for key in ('name', 'shape', 'dtype', 'bytes', 'sha256')}


def export_control(baseline, output, *, readout_scale=None):
    baseline = Path(baseline).resolve(strict=True)
    output = Path(output).resolve()
    group = baseline
    if not (baseline / 'segment.json').is_file():
        jobs = [p.parent for p in group.glob('*/segment.json') if (p.parent / 'export_t64.py').is_file()
                and any(c['id'] == CASE for c in json.loads(p.read_text()).get('attention_modules', []))]
        if len(jobs) != 1:
            raise ValueError('Set --baseline to the original attention module export job containing segment.json and config.yaml')
        baseline = jobs[0]
    air = baseline / 'air' if (baseline / 'air/manifest.json').is_file() else group / 'air'
    manifest = json.loads((air / 'manifest.json').read_text())
    segment = json.loads((baseline / 'segment.json').read_text())
    case, = [c for c in segment['attention_modules'] if c['id'] == CASE]
    original, = [c for c in manifest['cases'] if c['id'] == CASE and c['status'] == 'EXPORTED']
    record_path = checked(air, original['export'])
    record = json.loads(record_path.read_text())
    baseline_hidden = _hidden_reference(record_path.parent, record)
    if (record['reference_basis'] != MODULE_BASIS or (record['physical_rows'], record['valid_rows']) != (64, 19)
            or record['capture_sha256'] != segment['capture_sha256']
            or baseline_hidden['sha256'] != case['golden']['sha256'] or 'gdr_scale_control' in original):
        raise ValueError('Expected the original exported L0 physical64/valid19 hidden and its prepared job')
    for spec in record['source_inputs']:
        if spec['sha256'] != case['inputs'][spec['semantic_name']]['sha256']:
            raise ValueError('Prepared input differs from the baseline export: ' + spec['semantic_name'])
    if output in (baseline, air) or air in output.parents or output in baseline.parents or (output / 'air').exists():
        raise FileExistsError('Choose a separate output directory with no existing AIR')
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output, prefix='.export-l0-') as tmp:
        job = Path(tmp)
        for name in ('models/__init__.py', WRAPPER, NZ, EXPORTER, 'utils.py', 'config.yaml', 'export_t64.py'):
            dest = job / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(baseline / name, dest)
        (job / MODELING).write_text(scaled_source((baseline / MODELING).read_text(), readout_scale=readout_scale))
        inputs = json.loads((baseline / 'inputs.json').read_text())
        for spec in [*inputs.values(), *case['inputs'].values(), case['golden']]:
            src = checked(baseline, spec)
            dest = job / src.relative_to(baseline)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
        scale = SCALE if readout_scale is None else readout_scale
        control = dict(gated_rmsnorm_epsilon_multiplier=scale**2, baseline_manifest_sha256=sha(air / 'manifest.json'),
                       baseline_export=original['export'], staged_source_sha256=sha(job / MODELING))
        if readout_scale is None:
            control.update(value_scale=scale, initial_state_scale=scale, state_writeback_scale=1. / scale)
        else:
            control.update(readout_query_scale=scale, qk_normalization='FP32_SOURCE_L2NORM_THEN_FP16_INPUTS',
                           kernel_qk_normalization=False, value_scale=1, initial_state_scale=1, state_writeback_scale=1)
        save_json(job / 'inputs.json', inputs)
        save_json(job / 'segment.json', dict(segment, attention_modules=[dict(case, gdr_scale_control=control)]))
        env = dict(os.environ, REPO_DIR=str(Path(__file__).resolve().parents[4]), AI_RUN_DIR=str(job))
        subprocess.run([sys.executable, str(job / 'export_t64.py')], cwd=job, env=env, check=True)
        new_manifest = json.loads((job / 'air/manifest.json').read_text())
        new_case, = new_manifest['cases']
        if new_case['status'] != 'EXPORTED' or new_case['id'] != CASE or new_case['gdr_scale_control'] != control:
            raise ValueError('Scaled L0 hidden export did not complete')
        new_record_path = checked(job / 'air', new_case['export'])
        new_record = json.loads(new_record_path.read_text())
        scaled_hidden = _hidden_reference(new_record_path.parent, new_record)

        def bindings(r):
            return {s['semantic_name']: (s['sha256'], s['dtype'], s['shape']) for s in r['source_inputs']}

        if bindings(new_record) != bindings(record) or scaled_hidden != baseline_hidden:
            raise ValueError('Scaled and baseline exported module fixtures differ')
        if new_record['air']['sha256'] == record['air']['sha256']:
            raise ValueError('Scaled export produced the original AIR bytes')
        from qwen35_moe_dflash.draft.incremental.torch_export import _producer_schema, _schema
        schema = _schema(*_producer_schema())
        before = _air_facts(air, original, schema)
        after = _air_facts(job / 'air', new_case, schema)
        # The source RmsNorm may omit its default epsilon.
        if (not math.isclose(before['epsilon'], 1e-6, rel_tol=1e-6)
                or not math.isclose(after['epsilon'], 1e-6 * scale**2, rel_tol=1e-6)):
            raise ValueError(f"Unexpected gated RMSNorm epsilon: {before['epsilon']} -> {after['epsilon']}")
        if readout_scale is None:
            if any(after['producers'][n] not in ('Mul', 'Muls') for n in ('value', 'initial_state')):
                raise ValueError(f"Unexpected scaled GDR inputs: {after['producers']}")
        elif (before['norm_flag'] != 1 or after['norm_flag'] != 0
              or any(after['producers'][n] not in ('Mul', 'Muls') for n in ('query', 'key'))
              or any(before['producers'][n] != after['producers'][n] for n in ('value', 'initial_state'))):
            raise ValueError(f"Unexpected GDR readout graph: norm flag {before['norm_flag']} -> {after['norm_flag']}, "
                             f"inputs {after['producers']}")
        if not before['weights'] or after['weights'] != before['weights']:
            raise ValueError('Weights or non-scalar constants changed between baseline and scaled AIR')
        (job / 'air').rename(output / 'air')
    from qwen35_moe_dflash.common.artifacts import share_for_transfer
    share_for_transfer(output / 'air')
    return dict(status='EXPORTED', case=CASE, scale=scale, scaled_operand=(
        'normalized_query' if readout_scale is not None else 'value_and_initial_state'), manifest=str(output / 'air/manifest.json'))
