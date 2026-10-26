import ast
import json
from pathlib import Path

import pytest

from qwen35_moe_dflash.compiler_support import support_snapshot


ROLLBACK = ('W/AI_FMK (158555): graph_partition.cpp Partition(320)::'
            '"Partition of type: NPUCL is more than 9," " will rollback to CPU"')
BEGIN = 'I/AI_FMK omg.cpp BuildOfflineCompiledModel(122)::"TCH iCompiler->Compile begin ..."'
END_BUILD = 'I/AI_FMK omg.cpp GenerateOMCModel(191)::"TCH BuildOrigin2IRGraph end ..."'


def support(name, devices):
    return f'I/AI_FMK model_compatibility_check.cpp CheckOpSupported(30)::"opDeviceSupport: op={name}, devices=[{devices}]"'


def rejection(name, op_type, store='elementary_lib'):
    return (f'W/AI_NPUCL fe_sub_stores_manager.cc CheckSupported(313)::"CheckSupported: '
            f'the op name [{name}] type [{op_type}] is not supported in npucl store [{store}]"')


def log(*lines, main=True):
    prefix = ['start InferShape for graph[qwen35_target] ....'] if main else []
    return '\n'.join(prefix + [BEGIN, *lines, ROLLBACK])


def test_real_strings_bound_main_round_exclude_constant_graph_and_preserve_ids():
    text = '\n'.join([
        'start InferShape for graph[qwen35_target] ....',
        'IR InferShape for Graph constNodeGraph Success',
        'opDeviceSupport size: 1', support('same', '1,'),
        rejection('same', 'FakeConstType'), ROLLBACK,
        END_BUILD, BEGIN,
        rejection('same', 'MatMul'),
        'opDeviceSupport size: 2', support('same', '0,1,'), support('input_ids', '1,'),
        ROLLBACK,
    ])
    result = support_snapshot(text)
    assert result['status'] == 'complete'
    assert result['scope'] == 'main' and result['graph'] == 'qwen35_target'
    assert result['start_line'] == 8 and result['end_line'] == 13
    assert result['rollback']['line'] == 13
    assert result['window']['status'] == 'bounded'
    assert result['counts']['expected_count'] == result['counts']['distinct_count'] == 2
    same, data = result['nodes']
    assert same['device_ids'] == [0, 1] and not same['device_conflict']
    assert same['type'] == 'MatMul' and same['type_evidence_lines'] == [9]
    assert same['support_line'] == 11
    assert data['type'] == 'unknown' and data['types'] == []
    assert result['runtime_placement'] == 'UNVERIFIED'
    assert all(item['line'] >= 8 for item in result['diagnostics'])
    json.dumps(result)


@pytest.mark.parametrize('phase', [BEGIN, END_BUILD])
def test_explicit_phase_allows_bounded_window_but_does_not_invent_graph_name(phase):
    result = support_snapshot('\n'.join([
        'IR InferShape for Graph constNodeGraph Success', phase,
        'opDeviceSupport size: 1', support('x', '1,'), ROLLBACK]))
    assert result['status'] == 'complete'
    assert result['scope'] == 'unknown' and result['graph'] is None
    assert result['nodes'][0]['device_ids'] == [1]


@pytest.mark.parametrize('text,reason', [
    ('opDeviceSupport size: 1\n' + support('x', '0,') + '\n' + ROLLBACK,
     'no_explicit_main_compile_boundary'),
    ('start InferShape for graph[target]\nopDeviceSupport size: 1\n' + support('x', '0,') + '\n' + ROLLBACK,
     'no_explicit_main_compile_boundary'),
    ('IR InferShape for Graph constNodeGraph Success\n' + ROLLBACK, 'no_main_or_unknown_rollback'),
    (BEGIN + '\nopDeviceSupport size: 1\n' + support('x', '0,') + '\npartition type NPU:1, CPU:0, GPU:0, ISP:0.',
     'no_main_or_unknown_rollback'),
])
def test_missing_explicit_boundary_or_main_rollback_returns_unknown_without_nodes(text, reason):
    result = support_snapshot(text)
    assert result['status'] == 'unknown'
    assert result['reason'] == reason
    assert result['nodes'] == []
    assert result['counts']['expected_count'] is None


def test_latest_main_rollback_chosen_even_when_last_event_is_constant():
    text = log('opDeviceSupport size: 1', support('old', '0,')) + '\n' + '\n'.join([
        BEGIN, 'opDeviceSupport size: 1', support('new', '1,'), ROLLBACK,
        'IR InferShape for Graph constNodeGraph Success', ROLLBACK])
    result = support_snapshot(text)
    assert result['rollback']['line'] == 9
    assert [node['name'] for node in result['nodes']] == ['new']


def test_previous_graph_compile_boundary_is_not_reused_for_a_later_graph():
    text = '\n'.join([
        'start InferShape for graph[old_main]', BEGIN,
        'opDeviceSupport size: 1', support('same_name', '0,'), ROLLBACK,
        'start InferShape for graph[new_main]',
        'opDeviceSupport size: 1', support('same_name', '1,'), ROLLBACK])
    result = support_snapshot(text)
    assert result['status'] == 'unknown' and result['nodes'] == []
    assert result['reason'] == 'main_compile_boundary_graph_mismatch'
    assert result['rollback']['graph'] == 'new_main'


def test_later_graph_with_its_own_boundary_does_not_reuse_old_types():
    text = '\n'.join([
        'start InferShape for graph[old_main]', BEGIN, rejection('x', 'OldType'),
        'opDeviceSupport size: 1', support('x', '0,'), ROLLBACK,
        'start InferShape for graph[new_main]', BEGIN,
        'opDeviceSupport size: 1', support('x', '1,'), ROLLBACK])
    result = support_snapshot(text)
    assert result['status'] == 'complete' and result['graph'] == 'new_main'
    node, = result['nodes']
    assert node['device_ids'] == [1] and node['type'] == 'unknown'


def test_last_partial_round_does_not_fall_back_to_earlier_complete_round():
    result = support_snapshot(log(
        'opDeviceSupport size: 1', support('old', '0,'),
        'opDeviceSupport size: 3', support('known', '1,')))
    assert result['status'] == 'partial'
    assert result['counts']['expected_count'] == 3
    assert result['counts']['distinct_count'] == 1
    assert [node['name'] for node in result['nodes']] == ['known']


def test_empty_last_round_remains_partial_not_previous_success():
    result = support_snapshot(log('opDeviceSupport size: 1', support('old', '0,'), 'opDeviceSupport size: 2'))
    assert result['status'] == 'partial' and result['nodes'] == []
    assert result['counts']['expected_count'] == 2


def test_duplicate_conflict_has_no_union_candidate_and_records_variants():
    result = support_snapshot(log('opDeviceSupport size: 1', support('x', '1,'), support('x', '0,')))
    node, = result['nodes']
    assert result['status'] == 'partial' and result['counts']['conflicting_count'] == 1
    assert result['counts']['duplicate_count'] == 1
    assert node['device_conflict'] and node['device_ids'] == []
    assert node['device_variants'] == [[1], [0]]
    assert node['support_lines'] == [4, 5]


def test_same_candidate_set_different_order_is_not_conflict_or_union():
    result = support_snapshot(log('opDeviceSupport size: 1', support('x', '1,0,'), support('x', '0,1,')))
    node, = result['nodes']
    assert result['status'] == 'complete'
    assert not node['device_conflict'] and node['device_ids'] == [1, 0]
    assert node['device_variants'] == [[1, 0], [0, 1]]


def test_duplicate_same_id_rows_are_bounded_and_do_not_replace_first_support_line():
    result = support_snapshot(log('opDeviceSupport size: 1', *[support('x', '1,')] * 100))
    node, = result['nodes']
    assert result['status'] == 'complete'
    assert node['support_line'] == 4 and len(node['support_lines']) == 8
    assert node['support_occurrences'] == 100


@pytest.mark.parametrize('broken', [
    'opDeviceSupport: op=x, devices=[1,whoops,]',
    'opDeviceSupport: op=x, devices=[1,,0,]',
    'opDeviceSupport: op=x, devices=[1,',
    'opDeviceSupport: op=, devices=[1,]',
])
def test_malformed_or_truncated_entries_are_partial(broken):
    result = support_snapshot(log('opDeviceSupport size: 1', broken))
    assert result['status'] == 'partial' and result['nodes'] == []
    assert result['counts']['malformed_count'] == 1


def test_malformed_final_size_cannot_reuse_previous_round():
    result = support_snapshot(log('opDeviceSupport size: 1', support('old', '0,'),
                                  'opDeviceSupport size: not-a-number', support('new', '1,')))
    assert result['status'] == 'partial'
    assert result['counts']['expected_count'] is None
    assert [node['name'] for node in result['nodes']] == ['new']


def test_unknown_device_ids_and_empty_candidate_list_preserved_without_guessing():
    result = support_snapshot(log('opDeviceSupport size: 2', support('unknown', '7,-1,'), support('empty', '')))
    assert result['status'] == 'complete'
    assert [node['device_ids'] for node in result['nodes']] == [[7, -1], []]
    assert result['device_id_legend'] == {'0': 'NPU candidate', '1': 'CPU candidate'}


def test_exact_name_type_resolution_and_conflicts_remain_unknown():
    result = support_snapshot(log(
        rejection('v1', 'MatMul'), rejection('v10', 'BatchMatMul'),
        'I shape.cpp RunInferShape(77)::"[op:v1 type:Cast] success"',
        'I shape.cpp RunInferShape(77)::"[op:__ge/v2/axes type:Const] success"',
        'I unrelated.cpp OtherMethod(77)::"[op:v3 type:Add]"',
        'opDeviceSupport size: 4', support('v1', '1,'), support('v10', '0,'),
        support('__ge/v2/axes', '1,'), support('v3', '1,')))
    first, second, third, fourth = result['nodes']
    assert first['type'] == 'unknown' and first['type_conflict']
    assert first['types'] == ['MatMul', 'Cast']
    assert first['type_evidence_lines'] == [3, 5]
    assert second['type'] == 'BatchMatMul' and third['type'] == 'Const'
    assert fourth['type'] == 'unknown'
    assert result['status'] == 'complete'  # Support list is complete; type conflict is explicit.


def test_constant_or_subgraph_type_and_support_rows_inside_window_are_excluded():
    result = support_snapshot(log(
        'IR InferShape for Graph constNodeGraph Success', rejection('x', 'ConstantFake'),
        'opDeviceSupport size: 1', support('const', '1,'),
        'start InferShape for graph[SubGraph_0]', rejection('x', 'SubgraphFake'),
        'start InferShape for graph[qwen35_target]', rejection('x', 'MatMul'),
        'opDeviceSupport size: 1', support('x', '0,')))
    assert result['status'] == 'complete'
    node, = result['nodes']
    assert node['name'] == 'x' and node['type'] == 'MatMul'
    assert all('Fake' not in item['text'] for item in result['diagnostics'])


def test_rows_without_size_marker_are_unknown_not_unsupported():
    result = support_snapshot(log(support('x', '1,')))
    assert result['status'] == 'unknown'
    assert result['reason'] == 'no_main_support_size_block'
    assert result['nodes'] == []


def test_zero_size_block_can_be_complete_but_does_not_prove_npu():
    result = support_snapshot(log('opDeviceSupport size: 0'))
    assert result['status'] == 'complete' and result['nodes'] == []
    assert result['runtime_placement'] == 'UNVERIFIED'


def test_split_rollback_end_line_is_retained():
    text = '\n'.join([BEGIN, 'opDeviceSupport size: 1', support('x', '1,'),
                      'Partition of type: NPUCL is more than 9,', ' will rollback to CPU'])
    result = support_snapshot(text)
    assert result['status'] == 'complete'
    assert result['end_line'] == 5 and result['rollback']['line'] == 4


def test_diagnostic_context_is_globally_bounded_and_samples_stores_and_type_changes():
    lines = []
    for store in ['custom_ascendc_lib', 'elementary_lib', 'ascendc_lib', 'fe_lib']:
        for op_type in ['MatMul', 'Cast', 'Neg']:
            for i in range(40):
                lines += [f'precheck dtype detail {store} {op_type} {i}', rejection('x', op_type, store)]
    result = support_snapshot(log(*lines, 'opDeviceSupport size: 1', support('x', '0,1,')))
    assert len(result['diagnostics']) <= 32
    assert result['diagnostic_representatives'] == 12
    text = '\n'.join(item['text'] for item in result['diagnostics'])
    assert all(store in text for store in ['custom_ascendc_lib', 'elementary_lib', 'ascendc_lib', 'fe_lib'])
    node, = result['nodes']
    assert node['types'] == ['MatMul', 'Cast', 'Neg'] and node['type'] == 'unknown'
    assert len(node['type_evidence_lines']) == 3
    assert node['device_ids'] == [0, 1]  # Store rejection does not erase support IDs.


def test_context_prioritizes_current_cpu_only_reductions_over_data_and_multidevice_flood():
    stores = ['custom_ascendc_lib', 'elementary_lib', 'ascendc_lib', 'fe_lib']
    lines = []
    entries = []
    for op_type in ['Data', 'Gather', 'ReduceSum', 'ReduceMean']:
        name = f'earlier_{op_type}'
        entries.append(support(name, '1,' if op_type == 'Data' else '0,1,'))
        for store in stores:
            lines += [f'earlier context {i}' for i in range(5)]
            lines += [rejection(name, op_type, store), 'earlier context end']
    # The rare type arrives first: priority must follow final-table frequency,
    # not the first logged support/rejection line.
    for op_type, count, first_name in [('Neg', 1, 'other_compute'), ('ReduceMean', 8, 'v18'),
                                       ('ReduceSum', 12, '__ge/v18/sum_1')]:
        for index in range(count):
            name = first_name if index == 0 else f'{first_name}_other_{index}'
            entries.append(support(name, '1,'))
            lines.append(f'I shape.cpp RunInferShape(77)::"[op:{name} type:{op_type}] success"')
        for store in stores:
            lines += [f'{first_name} {store} context {i}' for i in range(4)]
            if store == 'elementary_lib':
                lines.append(f'E/AI_NPUCL reduce_check.cc CheckReduceSupport(185)::check elem support fail [{first_name}]')
            if store == 'ascendc_lib':
                lines.append(f'W/GENE OpInfoCheckSupport: Op[name=clone_{first_name},type={op_type}] doesn\'t exist in tbe ops store.')
            lines += [rejection(first_name, op_type, store), f'{first_name} next line']
    result = support_snapshot(log(*lines, f'opDeviceSupport size: {len(entries)}', *entries))
    assert result['status'] == 'complete' and len(result['diagnostics']) <= 32
    assert result['diagnostic_cpu_only_representatives'] == [
        {'type': 'ReduceSum', 'count': 12, 'name': '__ge/v18/sum_1'},
        {'type': 'ReduceMean', 'count': 8, 'name': 'v18'},
        {'type': 'Neg', 'count': 1, 'name': 'other_compute'},
    ]
    text = '\n'.join(item['text'] for item in result['diagnostics'])
    assert 'op name [__ge/v18/sum_1] type [ReduceSum]' in text
    assert 'op name [v18] type [ReduceMean]' in text
    assert 'CheckReduceSupport(185)::check elem support fail' in text
    assert 'earlier context' not in text
    assert all(store in text for store in stores)
    assert result['runtime_placement'] == 'UNVERIFIED'


def test_context_cpu_priority_uses_last_partial_table_and_excludes_conflicts_unknowns():
    lines = [rejection('old_cpu', 'MatMul'),
             'opDeviceSupport size: 1', support('old_cpu', '1,')]
    for name, op_type in [('device_conflict', 'ReduceSum'), ('type_conflict', 'ReduceMean'),
                          ('type_conflict', 'Cast'), ('multiple_devices', 'Neg'),
                          ('unknown_device', 'Slice'), ('valid', 'Add'), ('input', 'Data')]:
        lines += ['context separator'] * 5 + [rejection(name, op_type), 'context end']
    lines += ['opDeviceSupport size: 10', support('device_conflict', '1,'), support('device_conflict', '0,'),
              support('type_conflict', '1,'), support('multiple_devices', '0,1,'),
              support('unknown_device', '7,'), support('valid', '1,'), support('input', '1,')]
    result = support_snapshot(log(*lines))
    assert result['status'] == 'partial'
    assert result['diagnostic_cpu_only_representatives'] == [{'type': 'Add', 'count': 1, 'name': 'valid'}]


def test_context_selects_first_current_cpu_node_with_rejection_for_each_type():
    result = support_snapshot(log(
        'I shape.cpp RunInferShape(77)::"[op:first type:ReduceSum] success"',
        rejection('second', 'ReduceSum'), rejection('third', 'ReduceSum'),
        'opDeviceSupport size: 3', support('first', '1,'), support('second', '1,'), support('third', '1,')))
    assert result['diagnostic_cpu_only_representatives'] == [{'type': 'ReduceSum', 'count': 3, 'name': 'second'}]


def test_module_imports_only_standard_library_and_existing_lightweight_evidence_module():
    import qwen35_moe_dflash.compiler_support as module

    tree = ast.parse(Path(module.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert [name.name for name in node.names] == ['re']
        elif isinstance(node, ast.ImportFrom):
            assert node.module in {'__future__', 'compiler_evidence'}
