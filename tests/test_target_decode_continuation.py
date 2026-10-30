"""Complete host evidence chain for one 80-state same-model continuation.

Only the synthetic graph decoder is substituted. Runtime inventory, trial
receipts, original-row request/extract/import and structural lineage validators
run normally; no board, vendor model or real model weights are used.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import runpy
import struct
from types import SimpleNamespace

import pytest

from qwen35_moe_dflash import original_decode_row_transfer as transfer
from qwen35_moe_dflash import target_decode_continuation as continuation
from qwen35_moe_dflash import target_decode_state_lineage as state_audit
from qwen35_moe_dflash import target_state_lineage as graph_audit
from qwen35_moe_dflash.target_compile_failure import _stamp
from test_target_decode_smoke_plan import case as source_case
import test_target_decode_output_inventory as retained_fixture
from test_original_decode_embedding_probe import embedding_case, stamp
from test_original_decode_row_transfer import query_handoff
from test_target_prefill_smoke_plan import binding, rewrite_native, sha
from test_target_state_lineage import ENUMS, graph, save


def attach_graphs(value, tmp_path, monkeypatch):
    """Bind tiny original/compiled payloads to a real named-plan fixture."""
    source = graph(1)
    by_name = {row['name']: row for row in value['contract']['boundaries']['inputs_in_serialized_op_order']}
    kind = {'DT_FLOAT16': 77, 'DT_FLOAT': 78, 'DT_INT8': 79, 'DT_INT16': 80, 'DT_INT32': 81, 'DT_INT64': 82}
    for node in source['graph'][0]['op']:
        if node['name'] in by_name:
            desc = node['output_desc'][0]
            desc['shape'] = {'dim': by_name[node['name']]['shape']}
            desc['dtype'] = kind[by_name[node['name']]['schema_dtype_names'][0]]
    net = source['graph'][0]['op'][-1]
    net['input'][1:81] = reversed(net['input'][1:81])
    net['input_desc'][1:81] = reversed(net['input_desc'][1:81])
    roots = [deepcopy(node) for node in source['graph'][0]['op'] if node['type'] in {'Data', 'RefData'}]
    for node in roots: node['type'] = 'Data'
    raw = deepcopy(net['input_desc'])
    for index, row in enumerate(raw): row['name'] = f'output_{index}_fixture'
    compiled = {'graph': [{'name': 'compiled', 'op': roots + [
        {'name': 'opaque-task', 'type': 'GraphOp', 'output_desc': deepcopy(raw)},
        {'name': 'NetOutput', 'type': 'NetOutput', 'input': [f'opaque-task:{i}' for i in range(82)], 'input_desc': raw}]}]}
    library = tmp_path / 'ddk/tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so'
    library.parent.mkdir(parents=True); library.write_bytes(b'HOST TEST SCHEMA ONLY')
    documents = {b'fixture source graph': source, b'fixture compiled graph': compiled}
    class Schema:
        evidence = {'library': str(library), 'library_sha256': sha(library), 'descriptor_sha256': 'f' * 64}
        enums = {'fixture.DataType': ENUMS['dtype']['values']}
        _messages = {'hiai.proto.TensorDescriptor': {1: SimpleNamespace(kind=14, name='dtype', type_name='fixture.DataType')}}
        def decode(self, raw, message):
            if message != 'ModelDef' or bytes(raw) not in documents: raise ValueError('unsupported synthetic payload')
            return deepcopy(documents[bytes(raw)])
    monkeypatch.setattr(graph_audit, 'load_schema', lambda path: Schema())
    air = value['compilation'] / 'compile-work/generated/export.air'
    air.write_bytes(b'fixture source graph')
    payload = b'fixture compiled graph'
    task = b'opaque fixture task'
    content = struct.pack('<I', 2) + struct.pack('<III', 0, 0, len(payload)) + struct.pack('<III', 2, 0, len(task)) + payload + task
    header = bytearray(256); header[:4] = b'IMOD'; struct.pack_into('<I', header, 0x4c, len(content))
    value['model'].write_bytes(header + content)
    primary = dict(value['compiled']['omc'], bytes=value['model'].stat().st_size,
        sha256=sha(value['model']), stat=_stamp(value['model'].stat()))
    snapshots = [dict(path=air.relative_to(value['compilation']).as_posix(), bytes=air.stat().st_size,
        sha256=sha(air), snapshot_stat=_stamp(air.stat()), role='air')]
    value['compiled'].update(omc=primary, generated_artifacts=[primary], input_snapshots=snapshots, ddk_root=str(tmp_path / 'ddk'))
    save(value['compilation'] / 'report.json', value['compiled'])
    save(value['compilation'] / 'input-inventory.json', {'files': snapshots})
    report_sha = sha(value['compilation'] / 'report.json')
    save(value['job'] / 'result.json', {'artifact_kind': 'qwen35-original-target-job', 'status': 'COMPILED',
        'model_compilation': 'PASSED', 'output': str(value['job']), 'compilation_report_sha256': report_sha})
    value['contract'].update(compilation_report_sha256=report_sha, omc=primary)
    save(value['job'] / 'target-contract/report.json', value['contract'])
    for data in (value['outer'], value['native']):
        data.update(model_sha256=primary['sha256'], model_bytes=primary['bytes'])
    value['outer']['compilation_report_sha256'] = report_sha
    command = json.loads((value['probe'] / 'model-io-command.json').read_text())
    command['argv'][command['argv'].index('--model-sha256') + 1] = primary['sha256']
    save(value['probe'] / 'model-io-command.json', command)
    value['outer']['source_bindings'] = [binding(Path(row['path'])) for row in value['outer']['source_bindings']]
    rewrite_native(value)
    return documents


@pytest.fixture
def case(tmp_path, monkeypatch):
    value = source_case.__wrapped__(tmp_path)
    documents = attach_graphs(value, tmp_path, monkeypatch)
    (value['source'] / 'position_ids.bin').write_bytes(struct.pack('<i', 256))
    (value['source'] / 'new_kv_cache_pos.bin').write_bytes(struct.pack('<i', 256))
    (value['source'] / 'attention_mask.bin').write_bytes(bytes(257 * 4) + struct.pack('<f', float('-inf')) * (2048 - 257))
    monkeypatch.setattr(retained_fixture, 'source_case', SimpleNamespace(__wrapped__=lambda tmp: value))
    retained = retained_fixture.case.__wrapped__(tmp_path, monkeypatch)
    for row, relative in zip(retained['proof']['outputs'], retained['paths']):
        if row['index'] not in {retained['proof']['feature_runtime_output_index'], retained['proof']['logits_runtime_output_index']}:
            path = retained['board'] / relative
            with path.open('r+b') as stream: stream.write(struct.pack('<e', row['index']))
            retained['report']['retrieved'][relative] = retained_fixture.record(path)
    retained_fixture.refresh(retained)
    embed = embedding_case.__wrapped__(retained, tmp_path)
    handoff = query_handoff.__wrapped__(embed, tmp_path, SimpleNamespace(param=269))
    response = transfer.extract_rows(handoff['request'], *handoff['tables'], handoff['extracted'])
    assert response['status'] == transfer._EXTRACTED, response
    # Match the actual row-v4 explicit metadata recovery receipt.
    script = handoff['prepared'] / 'extract_rows.py'
    os.chmod(script, 0o777)
    imported = transfer.import_rows(handoff['prepared'], handoff['extracted'] / 'response.json', handoff['imported'],
        revalidate_extractor_metadata=True)
    assert imported['status'] == 'ORIGINAL_DECODE_TRANSFERRED_ROWS_BOUND_NEXT_TOKEN_PREPARED', imported
    lineage_dir = tmp_path / 'state-lineage'
    lineage = state_audit.inspect_target_decode_state_lineage(value['job'], value['probe'], retained['out'], lineage_dir)
    assert lineage['status'] == 'ALL80_SAME_MODEL_STATE_BINDINGS_OBSERVED', lineage
    return {**handoff, 'retained': retained, 'lineage': lineage, 'lineage_dir': lineage_dir,
        'out': tmp_path / 'continuation-prepared', 'documents': documents}


def prepare(case):
    return continuation.prepare_decode_continuation(case['lineage_dir'], case['imported'], case['out'], assumption=continuation.ASSUMPTION)


def test_real_host_row_inventory_graph_lineage_and_all89_independent_inputs_validate_without_board_or_weights(case, monkeypatch):
    watched = [*case['case']['source'].iterdir(), *case['case']['trial'].joinpath('inputs').iterdir(),
        *case['retained']['board'].joinpath('outputs').iterdir(), case['lineage_dir'] / 'report.json',
        case['imported'] / 'report.json', case['prepared'] / 'request.json', case['prepared'] / 'extract_rows.py']
    before = {path: stamp(path) for path in watched}
    original_open = os.open
    def guarded(path, *args, **kwargs):
        assert Path(path) not in case['tables'] and not str(path).endswith('.weight')
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(os, 'open', guarded)
    result = prepare(case)
    assert result['status'] == continuation._READY, result
    plan = continuation.validate_decode_continuation(case['out'])
    assert len(plan['inputs']) == 89 and len(plan['outputs']) == 82
    assert len(plan['state_feedback_bindings']) == 80
    assert result['continuation_token_id'] == 0 and result['next_absolute_position'] == 257
    assert plan['source_query_token_id'] == 269
    assert len({row['source_filename'] for row in plan['inputs']}) == 89
    assert len({(row['identity']['device'], row['identity']['inode']) for row in plan['source_inputs']}) == 89
    assert all(row['identity']['nlink'] == 1 for row in plan['source_inputs'])
    for state in plan['state_feedback_bindings']:
        path = Path(state['prepared_input_identity']['path'])
        parent = Path(state['output_file']['path'])
        assert path.read_bytes() == parent.read_bytes() and path.stat().st_ino != parent.stat().st_ino
    inputs = {row['name']: Path(row['source']).read_bytes() for row in plan['inputs']}
    assert inputs['arg1_1'] == case['expected_next_row'] and inputs['arg0_1'] == case['expected_next_scale']
    assert inputs['arg107_1'] == inputs['arg2_1'] == struct.pack('<i', 257)
    old_mask = (case['case']['trial'] / 'inputs/attention_mask.bin').read_bytes()
    assert inputs['arg113_1'] == old_mask[:257 * 4] + bytes(4) + old_mask[258 * 4:]
    assert plan['experimental_input_assumption']['contract_status'].startswith('EXPERIMENTAL')
    assert plan['state_continuity'] == 'NOT_VALIDATED' and plan['numerical_validation'] == 'NOT_RUN_NO_REFERENCE'
    assert all(stamp(path) == value for path, value in before.items())


def controls():
    return {'arg107_1': struct.pack('<i', 256), 'arg2_1': struct.pack('<i', 256),
        'arg1238_1': struct.pack('<i', 0), 'arg114_1': struct.pack('<q', 1),
        'arg18_1': struct.pack('<h', 1), 'arg14_1': struct.pack('<128i', 1, *([0] * 127)),
        'arg113_1': bytes(257 * 4) + struct.pack('<f', float('-inf')) * (2048 - 257)}


@pytest.mark.parametrize('fault', ['different_positions', 'negative_position', 'full_capacity', 'nonlocal_output',
    'nonlocal_allq', 'conv_guess', 'effective_guess', 'scattered_mask', 'negative_zero_mask'])
def test_control_assumption_refuses_unproven_or_inconsistent_source_values(fault):
    raw = controls()
    if fault == 'different_positions': raw['arg2_1'] = struct.pack('<i', 255)
    elif fault == 'negative_position': raw['arg107_1'] = raw['arg2_1'] = struct.pack('<i', -1)
    elif fault == 'full_capacity': raw['arg107_1'] = raw['arg2_1'] = struct.pack('<i', 2047)
    elif fault == 'nonlocal_output': raw['arg1238_1'] = struct.pack('<i', 255)
    elif fault == 'nonlocal_allq': raw['arg114_1'] = struct.pack('<q', 1224)
    elif fault == 'conv_guess': raw['arg14_1'] = struct.pack('<128i', 257, *([0] * 127))
    elif fault == 'effective_guess': raw['arg18_1'] = struct.pack('<h', 256)
    elif fault == 'negative_zero_mask': raw['arg113_1'] = struct.pack('<f', -0.0) + raw['arg113_1'][4:]
    else:
        mask = bytearray(raw['arg113_1']); mask[:4], mask[257*4:258*4] = mask[257*4:258*4], mask[:4]; raw['arg113_1'] = bytes(mask)
    with pytest.raises(ValueError): continuation._advance(raw)


def test_assumption_is_explicit_and_existing_output_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.setenv('AI_RUN_DIR', str(tmp_path))
    result = continuation.prepare_decode_continuation(tmp_path / 'absent', tmp_path / 'absent', tmp_path / 'output', assumption='guess')
    assert result['status'] == 'FAILED' and 'explicit' in result['error']
    with pytest.raises(FileExistsError):
        continuation.prepare_decode_continuation(tmp_path / 'absent', tmp_path / 'absent', tmp_path / 'output', assumption=continuation.ASSUMPTION)


@pytest.mark.parametrize('fault', ['tool_path', 'model_sha', 'metadata_sha', 'state_extra_annotation', 'owner', 'input_order', 'bool_index'])
def test_fresh_validation_does_not_hide_resigned_plan_metadata_owner_or_order_changes(case, fault):
    assert prepare(case)['status'] == continuation._READY
    plan_path, manifest_path = case['out'] / 'plan.json', case['out'] / 'manifest.json'
    plan, manifest = json.loads(plan_path.read_text()), json.loads(manifest_path.read_text())
    if fault == 'tool_path': plan['tool_binding']['path'] = '/different/tool'
    elif fault == 'model_sha': plan['primary_model_binding']['sha256'] = 'f' * 64
    elif fault == 'metadata_sha': plan['metadata_bindings'][0]['sha256'] = 'f' * 64
    elif fault == 'state_extra_annotation': plan['state_feedback_bindings'][0]['output_file']['relative_path'] += '-invented'
    elif fault == 'bool_index': plan['inputs'][0]['index'] = False
    elif fault == 'owner': plan['state_feedback_bindings'][0]['source_refdata'] = plan['state_feedback_bindings'][1]['source_refdata']
    else: plan['inputs'][0], plan['inputs'][1] = plan['inputs'][1], plan['inputs'][0]
    save(plan_path, plan)
    manifest['plan'] = transfer._read(plan_path, 32 << 20)[1]
    manifest['state_feedback_bindings'] = plan['state_feedback_bindings']
    save(manifest_path, manifest)
    with pytest.raises((ValueError, KeyError)):
        continuation.validate_decode_continuation(case['out'])


@pytest.mark.parametrize('fault', ['state_bytes', 'input_alias', 'imported_next', 'copied_code'])
def test_raw_payload_or_code_change_is_never_a_metadata_recovery(case, fault):
    if fault in {'state_bytes', 'input_alias'}:
        assert prepare(case)['status'] == continuation._READY
        plan = json.loads((case['out'] / 'plan.json').read_text())
        path = Path(plan['state_feedback_bindings'][0]['prepared_input_identity']['path'])
        if fault == 'state_bytes':
            with path.open('r+b') as stream: stream.write(b'x')
        else: os.link(path, case['out'] / 'extra-state-alias')
        with pytest.raises(ValueError): continuation.validate_decode_continuation(case['out'])
    else:
        path = case['imported'] / 'next-token/inputs_embeds.bin' if fault == 'imported_next' else case['prepared'] / 'extract_rows.py'
        if fault == 'copied_code': os.chmod(path, path.stat().st_mode & 0o7777)
        else:
            with path.open('r+b') as stream: stream.write(b'x')
        result = prepare(case)
        assert result['status'] == 'FAILED' and result['error']
        assert not (case['out'] / 'inputs').exists()


@pytest.mark.parametrize('stage', ['after_plan', 'after_manifest', 'during_copy'])
def test_active_payload_change_during_preparation_never_leaves_ready(case, monkeypatch, stage):
    if stage == 'during_copy':
        original = continuation._copy_state
        def changed(state, destination):
            result = original(state, destination)
            path = Path(state['output_file']['path'])
            os.chmod(path, path.stat().st_mode & 0o7777)
            return result
        monkeypatch.setattr(continuation, '_copy_state', changed)
    elif stage == 'after_manifest':
        original = continuation._stable
        def changed(metadata, payloads):
            if (case['out'] / 'manifest.json').exists():
                target = next((case['out'] / 'inputs').iterdir())
                os.chmod(target, target.stat().st_mode & 0o7777)
            return original(metadata, payloads)
        monkeypatch.setattr(continuation, '_stable', changed)
    else:
        original = continuation.save_json
        def changed(path, value):
            original(path, value)
            if path.name == 'plan.json':
                target = next((case['out'] / 'inputs').iterdir())
                os.chmod(target, target.stat().st_mode & 0o7777)
        monkeypatch.setattr(continuation, 'save_json', changed)
    result = prepare(case)
    assert result['status'] == 'FAILED' and 'changed' in result['error']
    assert json.loads((case['out'] / 'manifest.json').read_text())['status'] == 'FAILED'


def test_active_raw_read_to_snapshot_gap_is_closed(case, monkeypatch):
    read = transfer._read
    path = case['imported'] / 'next-token/inputs_embeds.bin'
    def changed(target, maximum):
        result = read(target, maximum)
        if Path(target) == path: os.chmod(path, path.stat().st_mode & 0o7777)
        return result
    monkeypatch.setattr(transfer, '_read', changed)
    result = prepare(case)
    assert result['status'] == 'FAILED' and ('identity' in result['error'] or 'changed' in result['error'])
    assert not (case['out'] / 'inputs').exists()


def test_cli_requires_experimental_assumption_and_reports_ready_counts(case, capsys):
    module = runpy.run_path(str(Path(__file__).parents[1] / 'targets/cpro/scripts/prepare_target_decode_continuation.py'))
    main = module['main']
    with pytest.raises(SystemExit):
        main(['--state-lineage', str(case['lineage_dir']), '--original-rows', str(case['imported']), '--output', str(case['out'])])
    capsys.readouterr()
    assert main(['--state-lineage', str(case['lineage_dir']), '--original-rows', str(case['imported']),
        '--assumption', continuation.ASSUMPTION, '--output', str(case['out'])]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown['status'] == continuation._READY and shown['input_count'] == 89 and shown['independent_state_file_count'] == 80
    assert shown['continuation_token_id'] == 0 and shown['next_absolute_position'] == 257


def test_parent_historical_ctimes_are_freshly_revalidated_without_rewriting_old_receipts(case):
    assert prepare(case)['status'] == continuation._READY
    watched = [case['lineage_dir'] / 'report.json', case['imported'] / 'report.json', case['prepared'] / 'request.json',
        case['prepared'] / 'manifest.json', case['prepared'] / 'source-audit/report.json', case['extracted'] / 'response.json',
        case['out'] / 'manifest.json', case['out'] / 'plan.json']
    saved = {path: path.read_bytes() for path in watched}
    parent = case['lineage']['decode']
    changed = [Path(row['output_file']['path']) for row in parent['state_file_mapping']['states']]
    changed += [Path(parent['model']['path']), Path(parent['air']['path']), case['lineage_dir'] / 'report.json']
    for path in changed: os.chmod(path, path.stat().st_mode & 0o7777)
    result = continuation.validate_decode_continuation(case['out'])
    observations = result['continuation_validation_observations']
    assert observations['revalidation_observations'] and observations['payload_revalidation_observations']
    assert saved == {path: path.read_bytes() for path in watched}


@pytest.mark.parametrize('changed', ['logits', 'stdout', 'original_ids', 'primary', 'compile_metadata'])
def test_outer_current_snapshot_rejects_late_parent_identity_drift_after_nested_row_validation(case, monkeypatch, changed):
    original = continuation._bound_rows
    parent = case['lineage']['decode']
    path = {'logits': case['retained']['board'] / case['retained']['paths'][case['retained']['proof']['logits_runtime_output_index']],
        'stdout': case['retained']['board'] / 'tool.stdout', 'original_ids': case['case']['source'] / 'input_ids.bin',
        'primary': Path(parent['model']['path']), 'compile_metadata': Path(parent['job']) / 'compilation/report.json'}[changed]
    def mutated(*args, **kwargs):
        result = original(*args, **kwargs)
        os.chmod(path, path.stat().st_mode & 0o7777)
        return result
    monkeypatch.setattr(continuation, '_bound_rows', mutated)
    result = prepare(case)
    assert result['status'] == 'FAILED' and ('changed' in result['error'] or 'identity' in result['error']), result
    assert not (case['out'] / 'inputs').exists()


def test_untrusted_prepared_extent_is_checked_against_abi_before_payload_open(case, monkeypatch):
    assert prepare(case)['status'] == continuation._READY
    path, receipt = case['out'] / 'plan.json', case['out'] / 'manifest.json'
    plan, manifest = json.loads(path.read_text()), json.loads(receipt.read_text())
    plan['inputs'][0]['bytes'] = 64 << 20
    save(path, plan); manifest['plan'] = transfer._read(path, 32 << 20)[1]; save(receipt, manifest)
    source = continuation._source
    def guarded(target, size):
        assert not Path(target).is_relative_to(case['out'] / 'inputs'), 'untrusted extent reached prepared input reader'
        return source(target, size)
    monkeypatch.setattr(continuation, '_source', guarded)
    with pytest.raises(ValueError, match='ABI/order/read extents'):
        continuation.validate_decode_continuation(case['out'])


def test_prepared_directory_symlink_swap_during_write_never_reports_ready(case, monkeypatch):
    original = continuation.save_json
    def swap(path, value):
        original(path, value)
        if path.name == 'plan.json':
            directory = case['out'] / 'inputs'
            directory.rename(case['out'] / 'moved-inputs')
            directory.symlink_to(case['out'] / 'moved-inputs', target_is_directory=True)
    monkeypatch.setattr(continuation, 'save_json', swap)
    result = prepare(case)
    assert result['status'] == 'FAILED' and 'symlink' in result['error']
