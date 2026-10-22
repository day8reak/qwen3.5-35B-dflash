"""Vendor severity formats and bounded, failure-safe error reporting."""
from pathlib import Path

import pytest

from qwen35_moe_dflash.tool_logs import log_excerpt


def test_internal_omg_errors_survive_optional_plugin_messages_and_long_tail(tmp_path):
    log = tmp_path / "omg.log"
    optional = ('W/AI_FMK (1): dlopen so failed: libgpucl_host.so\n'
                'I/AI_NPUCL (1): load librl_search.so failed\n'
                '[DEBUG] GE: error: loading error catalog\n')
    log.write_text(optional * 30
                   + '[CPUCL]binary_common.cpp:ReshapeUpdate(490):param["input0Type_"] is not equals to["input1Type_"]\n'
                   + 'E/AI_INFRA (1): SelectVerify: "x1Shape.GetDimNum() != x2Shape.GetDimNum()" "false, return FAIL."\n'
                   + '[DEBUG] GE: catalog filler\n' * 600)
    excerpt = log_excerpt(log)
    early = excerpt.split("Log tail:")[0]
    assert 'L91: [CPUCL]' in early
    assert 'L92: E/AI_INFRA' in early
    assert 'libgpucl_host' not in early and 'librl_search' not in early
    assert 'x1Shape.GetDimNum()' in early
    assert len(excerpt) < 14000


def test_log_read_error_cannot_hide_tool_failure(tmp_path, monkeypatch):
    log = tmp_path / "omg.log"
    log.write_text("tool error\n")
    def denied(self, *args, **kwargs):
        raise PermissionError("diagnostic file unavailable")
    monkeypatch.setattr(Path, "open", denied)
    assert 'Log excerpt unavailable: diagnostic file unavailable' in log_excerpt(log)


def test_internal_omg_trace_noise_does_not_hide_missing_infer_function(tmp_path):
    log = tmp_path / "omg.log"
    traces = (
        'E/AI_FMK (1): air_op_convert_funcs.cpp ConvertReshape(1686)::"[lsy] node-v3 input has Const Node:__ge/v3/shape"\n'
        'E/AI_FMK (1): air_op_convert_funcs.cpp ConvertReshape(1707)::"++++ [lsy] change const, origin = 1 0 8 0 , newDim = 1 0 8 0 "\n'
        'E/AI_FMK (1): shape.cpp GetTotalDimNum(95)::"[lsy] get shape from scrShape:16 "\n'
        'E/AI_FMK (1): ir_infer_shape_optimizer.cpp operator()(184)::"[lsy] node[v51] input shape:1 8 2 "\n'
        'E/AI_INFRA (1): memory_config_parser_host.cpp LoadConfigFromFile(52)::"!path.empty()" "false, return hiai::SUCCESS."\n'
        'E/AI_INFRA (1): llm_parser_host.cpp GetAllLLMInfosFromFile(218)::"!llmConfigPath.empty()" "false, return hiai::SUCCESS."\n'
    )
    log.write_text(traces * 20
                   + '[CPUCL]operator_execute/op_execute.cpp:UpdateInput(52):"input ix[0] size[32] must >= tensor size[64]"\n'
                   + '[CPUCL]cpu_graph_executor.cpp:Execute(610):"Set input basebuffer error!"\n'
                   + 'E/AI_FMK (1): op_ir_func_factory.cpp GetInferFunc(31)::"get [op:v52 type:SoftplusV2] infershape func failed."\n'
                   + '[DEBUG] GE: catalog filler\n' * 600)
    excerpt = log_excerpt(log)
    early, tail = excerpt.split("Log tail:")
    assert 'L121: [CPUCL]' in early
    assert 'size[32] must >= tensor size[64]' in early
    assert 'Set input basebuffer error' in early
    assert 'SoftplusV2' in early and 'SoftplusV2' not in tail
    assert '[lsy]' not in early and 'return hiai::SUCCESS' not in early


@pytest.mark.parametrize("message", [
    'E/AI_FMK (1): unknown.cpp Unexpected(1)::"new diagnostic"',
    'E/AI_FMK (1): shape.cpp GetTotalDimNum(95)::"[lsy] invalid dimension"',
    'E/AI_FMK (1): air_op_convert_funcs.cpp ConvertReshape(1707)::"[lsy] failed to change const"',
    'E/AI_INFRA (1): memory_config_parser_host.cpp LoadConfigFromFile(52)::"!path.empty()" "false, return FAIL."',
])
def test_unknown_or_failing_e_severity_is_preserved(tmp_path, message):
    log = tmp_path / "omg.log"
    log.write_text(message + '\n' + '[DEBUG] filler\n' * 800)
    early = log_excerpt(log).split("Log tail:")[0]
    assert 'L1: ' + message in early


def test_omg_trace_filters_do_not_remove_lines_from_tail(tmp_path):
    log = tmp_path / "omg.log"
    trace = 'E/AI_FMK (1): shape.cpp GetTotalDimNum(95)::"[lsy] get shape from scrShape:16 "\n'
    log.write_text(trace)
    excerpt = log_excerpt(log)
    assert 'First error messages:' not in excerpt
    assert excerpt == 'Log tail:\n' + trace
