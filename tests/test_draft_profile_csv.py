"""Vendor CSV attribution tests; synthetic durations are not board results."""
import csv
import json
from pathlib import Path

import pytest

from qwen35_moe_dflash import draft_profile_csv as profile


def write(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def test_multi_partition_attribution_keeps_raw_devices_and_fractional_microseconds(tmp_path, monkeypatch):
    monkeypatch.setenv('AI_RUN_DIR', str(tmp_path))
    raw = tmp_path / 'raw'
    first = write(raw / 'part0_op.csv', profile.OP_HEADER, [
        ['dense(MatMul)', 'opaque-clock-A', '110.5', 'NPU'],
        ['sum\nsecond(Add)', 'opaque-clock-B', '50', 'CPU']])
    write(raw / 'part1_op.csv', profile.OP_HEADER, [['dense(MatMul)', 'different-clock', '210', 'NPU']])
    write(raw / 'model.csv', profile.MODEL_HEADER,
          [[name, 'opaque', '300' if name == 'inference end' else '1'] for name in profile.EVENTS])
    result = profile.summarize_profile(raw, tmp_path / 'summary')
    assert result['operator_count'] == 3
    assert result['operator_duration_sum_ms'] == .3705
    assert result['by_operator_type'][0]['operator_type'] == 'MatMul'
    assert result['by_operator_type'][0]['total_us'] == 320.5
    assert result['by_device_type'][1]['device_type'] == 'CPU'
    assert result['model_profiles'][0]['inference_ms'] == .3
    assert 'overlapping' in result['sum_scope']
    assert result['profiling_activation'] == 'NOT_ESTABLISHED_BY_CSV_PARSER_ALONE'
    assert len(list(csv.DictReader((tmp_path / 'summary/operators.csv').open()))) == 3
    assert json.loads((tmp_path / 'summary/report.json').read_text())['sources'][1]['sha256']
    assert first.is_file()


@pytest.mark.parametrize('value', ['NaN', 'Infinity', '-1', 'invalid', '1e50'])
def test_invalid_duration_is_not_promoted(tmp_path, value):
    path = write(tmp_path / 'op.csv', profile.OP_HEADER, [['a(Add)', '0', value, 'CPU']])
    with pytest.raises(ValueError, match='duration'):
        profile.parse_profile([path])


def test_unknown_units_and_wrong_width_are_not_guessed(tmp_path):
    path = write(tmp_path / 'op.csv', ('name', 'time(ms)'), [['a', '1']])
    with pytest.raises(ValueError, match='unsupported'):
        profile.parse_profile([path])
    write(path, profile.OP_HEADER, [['a', '0', '2']])
    with pytest.raises(ValueError, match='width'):
        profile.parse_profile([path])


def test_missing_operator_data_and_duplicate_models_fail(tmp_path):
    path = write(tmp_path / 'model.csv', profile.MODEL_HEADER, [[name, '0', '0'] for name in profile.EVENTS])
    with pytest.raises(ValueError, match='no operator'):
        profile.parse_profile([path])
    write(path, profile.MODEL_HEADER, [['load begin', '0', '0']] * 2)
    with pytest.raises(ValueError, match='duplicate'):
        profile.parse_profile([path])


def test_zero_duration_and_unknown_types_remain_literal(tmp_path):
    path = write(tmp_path / 'op.csv', profile.OP_HEADER, [['unmapped', '0', '0', '7']])
    result = profile.parse_profile([path])
    assert result['operators'][0]['operator_type'] == 'UNKNOWN'
    assert result['operators'][0]['operator_sum_percent'] is None
    assert result['by_device_type'][0]['device_type'] == '7'
    assert result['model_timing_status'] == 'UNAVAILABLE'


def test_duplicate_inode_or_symlink_and_nested_summary_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv('AI_RUN_DIR', str(tmp_path))
    raw = tmp_path / 'raw'
    path = write(raw / 'op.csv', profile.OP_HEADER, [['a(Add)', '0', '1', 'CPU']])
    with pytest.raises(ValueError, match='duplicate'):
        profile.parse_profile([path, path])
    with pytest.raises(ValueError, match='separate'):
        profile.summarize_profile(raw, raw / 'summary')
    (raw / 'link.csv').symlink_to(path)
    with pytest.raises(ValueError, match='symlinks'):
        profile.summarize_profile(raw, tmp_path / 'summary')
