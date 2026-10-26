"""Bounded terminal HiLog windows, with unmodified full candidate evidence."""

import pytest

from qwen35_moe_dflash.hdc_run import _hilog_summary_details


def _record(message, *, level="I", pid=7542):
    return f"03-09 08:00:00.123 {pid} {pid} {level} A0FFFF/qwen35_dflash_hiai/HIAI: {message}\n"


def test_latest_draft_prepare_failure_keeps_mul_type_error_before_long_teardown(tmp_path):
    failure_messages = [
        'binary_common.cpp ReshapeUpdate(490)::param["input0Type_"] is not equals to["input1Type_"]',
        'binary_common.cpp ReshapeUpdate(90)::"reshapeupdate failed"',
        'cpu_common.cpp GenerateOp(32)::"Op[name:v29,type:Mul] Init error"',
        'cpu_subgraph.cpp Prepare(69)::"error"',
        'graph_executor.cpp Init(103)::"graph executor init return error 1 SubGraph_2"',
        'pipeline.cpp Prepare(55)::"false, return FAIL"',
        'model_manager.cpp PrepareModelManager(45)::"failed"',
        'hiai_model_manager.cpp Load(230)::"manager Init return 1 load failed"',
    ]
    records = [
        _record('HIAI_Foundation_Init(110)::init hiai foundation success'),
        _record('dlopen HCL library failed; try alternate backend', level="E"),
        _record('target Load success'),
    ]
    records += [_record(f"setup detail {index}") for index in range(60)]
    records += [_record('SubGraph_0 CPU Init success')]
    records += [_record(message, level="E") for message in failure_messages]
    failure_line = len(records)
    records += [_record(f"Deinit cleanup detail {index}") for index in range(60)]
    records += [_record('Load unrelated newer process failed', level="E", pid=99999)]
    original = "".join(records)
    path, candidate = tmp_path / "snapshot.txt", tmp_path / "candidate.txt"
    path.write_text(original)

    selected, _, metadata = _hilog_summary_details(path, recorded_pid=7542, candidate_output=candidate)

    display = "\n".join(selected)
    assert all(message in display for message in failure_messages)
    assert "dlopen HCL library failed" in display
    assert "unrelated newer process" not in display
    assert len(selected) <= 60
    assert metadata["runtime_failure_line"] == failure_line
    # The literal type mismatch has no failure keyword; preceding context must
    # preserve it alongside the later Init/Prepare/Load failure anchors.
    assert metadata["runtime_failure_candidates"] == len(failure_messages) - 1
    assert metadata["first_error_line"] == 2
    assert metadata["candidate_lines"] == len(records) - 1
    assert len(candidate.read_text().splitlines()) == metadata["candidate_lines"]
    assert path.read_text() == original


@pytest.mark.parametrize("message", [
    'RestoreFromBuffer(42)::"false, return FAIL"',
    'restore model failed',
    'Init(42)::"false, return FAIL"',
    'PrepareModelManager(42)::"failed"',
    'Failed to Load model',
    'ParseModel rejected malformed descriptor',
])
def test_terminal_failure_window_survives_long_setup_and_teardown(tmp_path, message):
    records = [_record('HIAI_Foundation_Init(110)::init hiai foundation success')]
    records += [_record(f"setup {index}") for index in range(60)]
    records += [_record("causal predecessor descriptor"), _record(message, level="E")]
    failure_line = len(records)
    records += [_record(f"cleanup {index}") for index in range(60)]
    path = tmp_path / "snapshot.txt"
    path.write_text("".join(records))

    selected, _, metadata = _hilog_summary_details(path)

    assert message in "\n".join(selected)
    assert "causal predecessor descriptor" in "\n".join(selected)
    assert metadata["runtime_failure_line"] == failure_line
    assert metadata["runtime_failure_candidates"] == 1
    assert len(selected) <= 60
    assert metadata["method"] == "latest-observed-app-session-candidate"
    assert "not confirmed as this run" in metadata["scope"]


def test_success_and_deinit_errors_do_not_displace_runtime_failure_window(tmp_path):
    records = [_record('HIAI_Foundation_Init(110)::init hiai foundation success')]
    records += [_record(f"setup {index}") for index in range(60)]
    records += [_record('GenerateOp(32)::"Op[name:v29,type:Mul] Init error"', level="E")]
    failure_line = len(records)
    records += [_record(f"cleanup {index}") for index in range(60)]
    records += [_record('Deinit(42)::"error freeing library"', level="E")]
    records += [_record('Prepare(42)::"success"'), _record('Load(42)::"success"')]
    path = tmp_path / "snapshot.txt"
    path.write_text("".join(records))

    selected, _, metadata = _hilog_summary_details(path)

    assert 'Op[name:v29,type:Mul] Init error' in "\n".join(selected)
    assert metadata["runtime_failure_line"] == failure_line
    assert metadata["runtime_failure_candidates"] == 1
    assert len(selected) <= 60


def test_latest_load_failure_replaces_earlier_restore_fallback_window(tmp_path):
    records = [_record('HIAI_Foundation_Init(110)::init hiai foundation success')]
    records += [_record(f"setup {index}") for index in range(30)]
    records += [_record('RestoreFromBuffer failed; alternate backend will be tried', level="E")]
    restore_line = len(records)
    records += [_record(f"alternate backend setup {index}") for index in range(60)]
    records += [_record('GenerateOp(32)::"Op[name:v29,type:Mul] Init error"', level="E")]
    records += [_record('Load(230)::"PrepareModelManager failed"', level="E")]
    failure_line = len(records)
    records += [_record(f"cleanup {index}") for index in range(60)]
    path = tmp_path / "snapshot.txt"
    path.write_text("".join(records))

    selected, _, metadata = _hilog_summary_details(path)

    assert 'Op[name:v29,type:Mul] Init error' in "\n".join(selected)
    assert 'Load(230)::"PrepareModelManager failed"' in "\n".join(selected)
    assert metadata["restore_failure_line"] == restore_line
    assert metadata["runtime_failure_line"] == failure_line
    assert metadata["runtime_failure_candidates"] == 3
    assert len(selected) <= 60
