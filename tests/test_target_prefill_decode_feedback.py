"""Experimental cross-model mappings require literal source/FX/native/file joins."""
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import struct

import pytest

from qwen35_moe_dflash import target_prefill_decode_feedback as feedback
from qwen35_moe_dflash import target_state_lineage as lineage
from qwen35_moe_dflash.target_compile_failure import _read_small, _stamp
from qwen35_moe_dflash.target_prefill_smoke_plan import _identity, _source, _runtime_keys
from qwen35_moe_dflash.target_state_correspondence import analyze_state_correspondence
from test_target_parameter_fqn_owner import source
from test_target_state_lineage import compiled_graph, runtime_probe, ENUMS


def pair(tmp_path):
    documents, inventories, jobs, probes = {}, {}, {}, {}
    for label, sequence in (('prefill', 64), ('decode', 1)):
        document, inventory = source(sequence, label + '_params', permute=sequence == 1, deep_other_roles=False)
        compiled = compiled_graph(document)
        native = runtime_probe(compiled)
        native.update(runtime_version='same-runtime', firmware_id='same-firmware', ddk_id='same-ddk')
        src = lineage.analyze_state_lineage(document, ENUMS)
        joined = lineage._join_compiled(src, compiled, ENUMS)
        jobs[label] = {'source_lineage': src, 'compiled_lineage': joined,
            'runtime_bindings': lineage._runtime_bindings(joined, native, ENUMS)}
        documents[label], inventories[label], probes[label] = document, inventory, native
    contract = analyze_state_correspondence(documents['prefill'], documents['decode'], inventories['prefill'], inventories['decode'], ENUMS, ENUMS)['parameter_fqn_owner_contract']
    outputs = []
    for row, (dtype, shape) in zip(probes['prefill']['outputs'], _runtime_keys(probes['prefill'], 'outputs')):
        size = math.prod(shape) * (2 if dtype == 'FLOAT16' else 4)
        outputs.append({'runtime_index': row['index'], 'tensor_name': row['tensor_name'], 'dtype': dtype, 'shape': row['shape'],
            'actual_file': {'path': str(tmp_path / 'execution/board/outputs' / f'arbitrary_{row["index"] * 19 % 83}'),
                'bytes': size, 'sha256': hashlib.sha256(str(row['index']).encode()).hexdigest(), 'stat': [1, row['index'] + 2, size, 3, 4]}})
    return {'documents': documents, 'inventories': inventories, 'jobs': jobs, 'probes': probes, 'contract': contract, 'outputs': outputs}


def bind(case):
    return feedback.bind_prefill_decode_feedback(case['contract'], case['jobs']['prefill'], case['jobs']['decode'],
        case['outputs'], case['probes']['prefill'], case['probes']['decode'])


def test_all80_parameter_owners_join_reordered_source_compiled_native_ports_and_arbitrary_files(tmp_path):
    case = pair(tmp_path)
    result = bind(case)
    assert len(result) == len({row['owner_key'] for row in result}) == 80
    first = next(row for row in result if row['owner_key'] == 'layer.0.conv')
    assert first['prefill']['source_refdata'] == 'RefData_0'
    assert first['decode']['source_refdata'] == 'RefData_72'
    assert first['decode_runtime_input_name'] == 'RefData_72'
    assert first['prefill']['runtime_output_index'] != first['prefill']['source_output_index']
    assert Path(first['output_file']['path']).name.startswith('arbitrary_')
    assert first['bytes'] == 65536 and first['dtype'] == 'FLOAT16'
    assert case['contract']['tensor_numeric_dtype_and_extent'].startswith('UNVERIFIED')


def test_actual_empty_source_producer_shape_does_not_override_explicit_boundary_view_or_owner(tmp_path):
    case = pair(tmp_path)
    for label in ('prefill', 'decode'):
        document = case['documents'][label]
        compiled = compiled_graph(document)
        _, nodes, net = lineage._graph(document)
        for edge in net['input'][1:81]:
            node, port = lineage._edge(edge, nodes); node['output_desc'][port]['shape'] = {}
        src = lineage.analyze_state_lineage(document, ENUMS)
        assert src['status'] == 'STATE_LINEAGE_PARTIAL' and src['unique_state_owner_count'] == 80
        joined = lineage._join_compiled(src, compiled, ENUMS)
        case['jobs'][label] = {'source_lineage': src, 'compiled_lineage': joined,
            'runtime_bindings': lineage._runtime_bindings(joined, case['probes'][label], ENUMS)}
    case['contract'] = analyze_state_correspondence(case['documents']['prefill'], case['documents']['decode'],
        case['inventories']['prefill'], case['inventories']['decode'], ENUMS, ENUMS)['parameter_fqn_owner_contract']
    assert len(bind(case)) == 80


@pytest.mark.parametrize('fault', ['contract_gap', 'duplicate_owner', 'context_only', 'compiled_gap', 'runtime_gap',
    'duplicate_output_slot', 'wrong_file_extent', 'wrong_file_name', 'wrong_view', 'duplicate_file', 'duplicate_input',
    'nonstate_name_missing', 'nonstate_name_duplicate'])
def test_partial_or_conflicting_provenance_never_uses_refdata_order_or_equal_shapes(tmp_path, fault):
    case = pair(tmp_path)
    if fault == 'contract_gap': case['contract']['matched_owner_count'] = 79
    elif fault == 'duplicate_owner': case['contract']['mapping'][0]['owner_key'] = case['contract']['mapping'][1]['owner_key']
    elif fault == 'context_only': case['contract']['status'] = 'ALL80_UNIQUE_WEIGHT_CONTEXT_CANDIDATES_OBSERVED'
    elif fault == 'compiled_gap': case['jobs']['decode']['compiled_lineage']['states'][0]['status'] = 'UNVERIFIED'
    elif fault == 'runtime_gap': case['jobs']['decode']['runtime_bindings']['matched_count'] = 79
    elif fault == 'duplicate_output_slot': case['outputs'][0]['runtime_index'] = case['outputs'][1]['runtime_index']
    elif fault == 'wrong_file_extent':
        for row in case['outputs']: row['actual_file']['bytes'] += 2
    elif fault == 'wrong_file_name':
        for row in case['outputs']: row['tensor_name'] += '-wrong'
    elif fault == 'wrong_view':
        index = case['jobs']['decode']['runtime_bindings']['states'][0]['runtime_input_index']
        case['probes']['decode']['inputs'][index]['shape'] = [8192, 5]
    elif fault == 'duplicate_file':
        for row in case['outputs']: row['actual_file']['path'] = str(tmp_path / 'outputs/same')
    elif fault == 'duplicate_input':
        rows = case['jobs']['decode']['runtime_bindings']['states']
        rows[0]['runtime_input_index'] = rows[1]['runtime_input_index']
    elif fault == 'nonstate_name_missing':
        row = next(row for row in case['probes']['decode']['outputs'] if row['raw_dtype'] == 1)
        row.pop('tensor_name_status')
    else:
        row = next(row for row in case['probes']['decode']['outputs'] if row['raw_dtype'] == 1)
        row['tensor_name'] = case['probes']['decode']['outputs'][0]['tensor_name']
    with pytest.raises((ValueError, KeyError)):
        bind(case)


def test_real_encoded_producer_schema_preserves_parameter_roles_and_separate_dtype_domains(tmp_path):
    from test_target_output_order import _schema
    from qwen35_moe_dflash.target_source_schema import tensor_enum_fields
    case = pair(tmp_path)
    schema, model_class = _schema(tmp_path)
    decoded = {}
    def attrs(target, values):
        for raw in values:
            value = target.attr.add(key=raw['key']).value
            for key, item in raw['value'].items():
                if key == 'list':
                    for member, items in item.items():
                        getattr(value.list, member).extend([word.encode() for word in items] if member == 's' else items)
                else:
                    setattr(value, key, item.encode() if key == 's' else item)
    for label in ('prefill', 'decode'):
        model = model_class(name='encoded source fixture')
        graph = model.graph.add(name='source')
        for raw in case['documents'][label]['graph'][0]['op']:
            node = graph.op.add(name=raw['name'], type=raw['type']); node.input.extend(raw.get('input', []))
            attrs(node, raw.get('attr', []))
            for key in ('input_desc', 'output_desc'):
                for desc in raw.get(key, []):
                    target = getattr(node, key).add(); target.dtype = {77: 7, 78: 1}[desc['dtype']]
                    target.layout = desc.get('layout', 'ND'); target.shape.SetInParent()
                    target.shape.dim.extend(desc.get('shape', {}).get('dim', [])); attrs(target, desc.get('attr', []))
        decoded[label] = schema.decode(model.SerializeToString(), 'ModelDef')
    domain = tensor_enum_fields(schema)
    owner = analyze_state_correspondence(decoded['prefill'], decoded['decode'], case['inventories']['prefill'], case['inventories']['decode'], domain, domain)['parameter_fqn_owner_contract']
    jobs = {}
    for label in ('prefill', 'decode'):
        src = lineage.analyze_state_lineage(decoded[label], domain)
        # The compiled graph retains its independently decoded selected DDK enum77.
        comp = compiled_graph(case['documents'][label])
        joined = lineage._join_compiled(src, comp, ENUMS)
        jobs[label] = {'source_lineage': src, 'compiled_lineage': joined,
            'runtime_bindings': lineage._runtime_bindings(joined, case['probes'][label], ENUMS)}
    assert len(feedback.bind_prefill_decode_feedback(owner, jobs['prefill'], jobs['decode'], case['outputs'],
        case['probes']['prefill'], case['probes']['decode'])) == 80


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return _read_small(path, 4 << 20)[1]


@pytest.fixture
def prepared_case(tmp_path, monkeypatch):
    monkeypatch.setenv('AI_RUN_DIR', str(tmp_path))
    case = pair(tmp_path)
    execution = tmp_path / 'execution'; inputs = tmp_path / 'inputs'; inputs.mkdir()
    (inputs / 'input_ids.bin').write_bytes(struct.pack('<64i', *range(64)))
    (inputs / 'output_pos.bin').write_bytes(struct.pack('<i', 63))
    current_sources = [_source(inputs / 'output_pos.bin', 4)]
    for row in case['outputs']:
        path = Path(row['actual_file']['path']); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as stream: stream.truncate(row['actual_file']['bytes'])
        row['actual_file'].update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), stat=_stamp(path.stat()))
    output_desc = []
    for row in case['outputs']:
        output_desc.append({'index': row['runtime_index'], 'shape': row['shape'], 'dtype': row['dtype'], 'bytes': row['actual_file']['bytes']})
    logits = next(row for row in case['outputs'] if row['dtype'] == 'FLOAT32')
    path = Path(logits['actual_file']['path'])
    with path.open('r+b') as stream: stream.seek(17 * 4); stream.write(struct.pack('<f', 2.5))
    logits['actual_file'].update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), stat=_stamp(path.stat()))
    plan = {'compilation': str(tmp_path / 'prefill-job/compilation'), 'sequence_length': 64,
        'input_dir': str(inputs), 'compilation_report_sha256': 'a' * 64, 'model_sha256': 'b' * 64,
        'outputs': output_desc, 'logits_runtime_output_index': logits['runtime_index']}
    save(execution / 'plan.json', plan)
    records = {}
    for label, seq in (('prefill', 64), ('decode', 1)):
        job = tmp_path / (label + '-job'); probe = tmp_path / (label + '-probe')
        identity = save(job / 'record.json', {'literal': label})
        probe_id = save(probe / 'result.json', case['probes'][label])
        meta = {'job': str(job), 'profile': 'cpro-40block-prefill-64' if seq == 64 else 'cpro-40block-decode-1',
            'air': identity, 'model': {'sha256': 'b' * 64}, 'source_schema_binding': {'fixture': True},
            'compilation_report_sha256': 'a' * 64, 'model_sha256': 'b' * 64, 'source_enum_fields': ENUMS}
        records[str(job)] = (meta, identity)
    def load(job, profile, bindings, **options):
        meta, _ = records[str(job)]; bindings.append(_read_small(Path(job) / 'record.json', 4096)[1])
        label = 'prefill' if 'prefill' in Path(job).name else 'decode'
        return deepcopy(meta), deepcopy(case['documents'][label]), compiled_graph(case['documents'][label]), ENUMS
    def inventory(meta, bindings, seq):
        label = 'prefill' if seq == 64 else 'decode'
        return deepcopy(case['inventories'][label]), deepcopy(records[meta['job']][1])
    def probe(path, meta, bindings):
        value = json.loads((Path(path) / 'result.json').read_text())
        bindings.append(_read_small(Path(path) / 'result.json', 4 << 20)[1]); return value
    def bridge(actual, named, out):
        snapshots = [{'path': str(path), 'identity': _identity(path.stat())} for path in
            [execution / 'plan.json', inputs / 'output_pos.bin', *[Path(row['actual_file']['path']) for row in case['outputs']]]]
        return {'status': feedback._BRIDGE, 'input_count': 89, 'output_count': 82, 'execution': str(execution),
            'active_source_identities': snapshots, 'original_source_inputs': deepcopy(current_sources), 'retained_outputs': deepcopy(case['outputs'])}
    monkeypatch.setattr(feedback.lineage, 'load_target_graph', load)
    monkeypatch.setattr(feedback, '_bound_inventory', inventory)
    monkeypatch.setattr(feedback.lineage, '_probe', probe)
    monkeypatch.setattr(feedback, 'inspect_prefill_decode_bridge', bridge)
    return dict(case, root=tmp_path, execution=execution, inputs=inputs, out=tmp_path / 'prepared', plan=plan,
        kwargs={'prefill_job': tmp_path / 'prefill-job', 'decode_job': tmp_path / 'decode-job',
            'prefill_execution': execution, 'named_prefill_probe': tmp_path / 'prefill-probe',
            'decode_runtime_probe': tmp_path / 'decode-probe', 'output': tmp_path / 'prepared'})


def test_host_preparation_describes_actual_prefix_seed_and80_files_without_copy_execution_or_accuracy_claim(prepared_case, monkeypatch):
    case = prepared_case
    before = {path: path.read_bytes() for path in case['root'].rglob('*.json')}
    original = Path.open
    def no_weights(path, *args, **kwargs):
        assert path.suffix not in {'.weight', '.omc'}, 'host fixture must not request any weights or model execution'
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', no_weights)
    result = feedback.prepare_prefill_decode_feedback(**case['kwargs'])
    assert result['status'] == feedback._STATUS, result
    assert result['mapped_state_count'] == 80 and result['feedback_ready'] is False
    assert result['decode_inputs_created'] is False and not (case['out'] / 'inputs').exists()
    assert result['prefix']['input_ids'] == list(range(64)) and result['prefix']['length'] == 64
    assert result['next_seed']['statistics']['argmax']['token_id'] == 17
    assert result['next_seed']['seed_accepted'] is False
    assert result['next_seed']['trial_use'] == 'SOURCE_PYTHON_LAST_ROW63_EXPERIMENT_ONLY'
    assert result['parameter_owner_contract']['tensor_numeric_dtype_and_extent'].startswith('UNVERIFIED')
    assert len(result['required_execution_assumptions']) == 5
    assert json.loads((case['out'] / 'manifest.json').read_text())['mapped_state_count'] == 80
    assert all(path.read_bytes() == raw for path, raw in before.items())


def test_old_output_pos64_is_recorded_but_never_accepted_as_last_row_seed(prepared_case):
    case = prepared_case
    (case['inputs'] / 'output_pos.bin').write_bytes(struct.pack('<i', 64))
    # Rebuild the retained-source stub to model an actual independently executed64 receipt.
    original = feedback.inspect_prefill_decode_bridge
    def bridge(*args):
        value = original(*args); value['original_source_inputs'] = [_source(case['inputs'] / 'output_pos.bin', 4)]; return value
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(feedback, 'inspect_prefill_decode_bridge', bridge)
        result = feedback.prepare_prefill_decode_feedback(**case['kwargs'])
    assert result['status'] == feedback._STATUS, result
    assert result['next_seed']['executed_output_pos'] == 64 and not result['next_seed']['seed_accepted']
    assert result['next_seed']['trial_use'] == 'NOT_AN_ACCEPTED_LAST_ROW_SEED'


@pytest.mark.parametrize('fault', ['missing', 'truncated', 'invalid_token'])
def test_prefix_ids_are_required_exactly64_words_without_claiming_embedding_equivalence(prepared_case, fault):
    case = prepared_case; path = case['inputs'] / 'input_ids.bin'
    if fault == 'missing': path.unlink()
    elif fault == 'truncated': path.write_bytes(struct.pack('<63i', *range(63)))
    else: path.write_bytes(struct.pack('<64i', -1, *range(63)))
    result = feedback.prepare_prefill_decode_feedback(**case['kwargs'])
    assert result['status'] == 'BLOCKED', result
    assert not (case['out'] / 'inputs').exists()


def test_output_must_be_separate_from_all_retained_source_directories(prepared_case):
    case = prepared_case
    values = dict(case['kwargs'], output=case['inputs'] / 'new-plan')
    with pytest.raises(ValueError, match='separate'):
        feedback.prepare_prefill_decode_feedback(**values)
    assert not (case['inputs'] / 'new-plan').exists()


@pytest.mark.parametrize('fault', ['other_execution', 'wrong_sha'])
def test_optional_inspection_cannot_substitute_another_run_or_logits(prepared_case, fault):
    case = prepared_case
    output = next(row for row in case['outputs'] if row['dtype'] == 'FLOAT32')
    report = {'artifact_kind': 'qwen35-target-prefill-output-inventory', 'status': 'OUTPUT_FILES_BOUND_TO_LOGGED_SLOTS',
        'execution': str(case['execution']) + ('-different' if fault == 'other_execution' else ''),
        'slot_binding': 'EXPLICIT_COMPLETE_TOOL_LOG_AND_VERIFIED_BYTES', 'bound_tensors': {'logits': {
            'runtime_index': output['runtime_index'], 'path': Path(output['actual_file']['path']).relative_to(case['execution'] / 'board').as_posix(),
            'bytes': output['actual_file']['bytes'], 'sha256': 'f' * 64}}}
    directory = case['root'] / 'inspection'; save(directory / 'report.json', report)
    result = feedback.prepare_prefill_decode_feedback(**case['kwargs'], prefill_output_inspection=directory)
    assert result['status'] == 'BLOCKED' and 'observation' in result['error'], result


@pytest.mark.parametrize('phase', ['graph', 'numerical', 'manifest'])
def test_current_output_ctime_change_during_any_phase_blocks_plan(prepared_case, monkeypatch, phase):
    case = prepared_case; target = Path(case['outputs'][1]['actual_file']['path'])
    if phase == 'graph':
        original = feedback.lineage.load_target_graph
        def changed(*args, **kwargs):
            result = original(*args, **kwargs); os.chmod(target, target.stat().st_mode & 0o7777); return result
        monkeypatch.setattr(feedback.lineage, 'load_target_graph', changed)
    elif phase == 'numerical':
        original = feedback._numerical
        def changed(*args, **kwargs):
            result = original(*args, **kwargs); os.chmod(target, target.stat().st_mode & 0o7777); return result
        monkeypatch.setattr(feedback, '_numerical', changed)
    else:
        original = feedback.save_json
        def changed(path, value):
            original(path, value)
            if path.name == '.manifest.pending.json': os.chmod(target, target.stat().st_mode & 0o7777)
        monkeypatch.setattr(feedback, 'save_json', changed)
    result = feedback.prepare_prefill_decode_feedback(**case['kwargs'])
    assert result['status'] == 'BLOCKED' and any(word in result['error'] for word in ('changed', 'replaced')), result


def test_cli_prints_compact_plan_count_and_unaccepted_seed(prepared_case, capsys):
    import runpy
    case = prepared_case
    cli = runpy.run_path(str(Path(__file__).parents[1] / 'targets/cpro/scripts/prepare_target_prefill_decode_feedback.py'))
    argv = [part for key, value in case['kwargs'].items() for part in ('--' + key.replace('_', '-'), str(value))]
    assert cli['main'](argv) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary['mapped_state_count'] == 80 and not summary['next_seed']['seed_accepted']
    assert len(json.dumps(summary)) < 2048
