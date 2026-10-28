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


def test_reshape_chain_trace_does_not_hide_cumsum_or_cpucl_error(tmp_path):
    log = tmp_path / "omg.log"
    trace = (
        'E/AI_FMK (1): air_op_convert_funcs.cpp ConvertReshape(1713)::"[lsy] node-__ge/v1156/flat_input input has Reshape Node:v1155"\n'
        'E/AI_FMK (1): air_op_convert_funcs.cpp ConvertReshape(1733)::"++++ [lsy] change reshape, origin = 8 0 16 0 , newDim = -1 0 16 0 "\n'
    )
    log.write_text(trace * 20
                   + '[CPUCL]op_execute.cpp:UpdateInput(52):"input ix[0] size[2] must >= tensor size[4]"\n'
                   + 'E/AI_FMK (1): op_ir_func_factory.cpp GetInferFunc(31)::"get [op:v92 type:Cumsum] infershape func failed."\n'
                   + '[DEBUG] filler\n' * 800)
    early = log_excerpt(log).split("Log tail:")[0]
    assert 'L41: [CPUCL]' in early and 'size[2] must >= tensor size[4]' in early
    assert 'Cumsum' in early and 'change reshape' not in early


@pytest.mark.parametrize("failure", [
    'E/AI_NPUCL (1): Node __ge/v20/flat_output type MatMul don\'t support!',
    'Node __ge/v20/flat_output type MatMul don\'t support!',
    'E/AI_FMK (1): get [op:rms type:AdnRmsNorm] infershape func failed.',
    'E/AI_NPUCL (1): Get format configs failed, node : conv_fused, substore: !',
])
def test_concrete_failure_survives_first_six_severity_only_lines(tmp_path, failure):
    log = tmp_path / "compile.log"
    log.write_text('E/AI_FMK (1): pass_manager.cpp Run(52)::"PassManager::Run begin"\n' * 20
                   + failure + '\n' + '[DEBUG] catalog filler\n' * 800)
    early, tail = log_excerpt(log).split("Log tail:")
    assert early.startswith('Priority error messages:\nL21: ' + failure)
    assert failure not in tail
    assert 'L1: E/AI_FMK' in early  # unknown E-level evidence is still retained
    assert len(log_excerpt(log)) < 14000


def test_kernel_and_python_failure_take_priority_without_optional_plugin_noise(tmp_path):
    log = tmp_path / "compile.log"
    log.write_text('E/AI_FMK (1): PassManager::Run begin\n' * 20
                   + 'W/ASC (1): optional kernel plugin not found\n'
                   + 'E/ASC (1): required kernel file path is not valid\n'
                   + 'Traceback (most recent call last):\n'
                   + '  File "export.py", line 3, in main\n'
                   + 'RuntimeError: exporter failed to serialize graph\n'
                   + '[DEBUG] filler\n' * 800)
    early = log_excerpt(log).split("Log tail:")[0]
    assert early.startswith('Priority error messages:\nL22: ')
    assert 'required kernel file path is not valid' in early
    assert 'Traceback (most recent call last)' in early and 'RuntimeError:' in early
    assert 'optional kernel plugin' not in early


@pytest.mark.parametrize("dtype", [3, 12])
def test_terminal_named_node_rejection_precedes_repeated_candidate_dtype_rejections(tmp_path, dtype):
    log = tmp_path / "compile.log"
    generic = ('E/AI_NPUCL (1): op_check_support.cc CheckIfTensorTypeSupport(100)::'
               f'"inputdataType is not support {dtype}"\n')
    terminal = 'E/AI_NPUCL (1): "Node __ge/v20/flat_output type MatMul don\'t support!"'
    log.write_text(generic * 20 + terminal + '\n' + '[DEBUG] filler\n' * 800)
    early = log_excerpt(log).split("Log tail:")[0]
    assert early.startswith('Priority error messages:\nL21: ' + terminal)
    assert 'CheckIfTensorTypeSupport' in early
    assert len([line for line in early.splitlines() if line.startswith('L')]) <= 6


def test_named_node_has_priority_over_other_concrete_rejections(tmp_path):
    log = tmp_path / "compile.log"
    prior = 'E/AI_NPUCL (1): type INT32 not support in candidate store\n'
    terminal = 'E/AI_NPUCL (1): [op:__ge/v20/flat_output type:MatMul] unsupported'
    log.write_text(prior * 20 + terminal + '\n' + '[DEBUG] filler\n' * 800)
    assert log_excerpt(log).split("Log tail:")[0].startswith('Priority error messages:\nL21: ' + terminal)


def test_causal_matrix_limit_is_kept_alongside_terminal_node_after_dtype_attempts(tmp_path):
    log = tmp_path / "compile.log"
    generic = ('E/AI_NPUCL (1): op_check_support.cc CheckIfTensorTypeSupport(100)::'
               '"inputdataType is not support {dtype}"\n')
    attempts = ''.join(generic.format(dtype=dtype) for dtype in [3, 12] * 20)
    limit = ('E/AI_NPUCL (1): elementary_matmul.cc CheckSupport(100)::'
             '"Matrix k greater than 8000 is not supported"')
    terminal = 'E/AI_NPUCL (1): "Node __ge/v20/flat_output type MatMul don\'t support!"'
    log.write_text(attempts + limit + '\n' + terminal + '\n' + '[DEBUG] filler\n' * 800)
    early, tail = log_excerpt(log).split("Log tail:")
    rows = early.splitlines()
    assert rows[1] == 'L42: ' + terminal
    assert rows[2] == 'L41: ' + limit
    assert 'CheckIfTensorTypeSupport' in early
    assert limit not in tail and terminal not in tail
    assert len([line for line in rows if line.startswith('L')]) <= 6


def test_diagnostic_scan_and_long_line_reads_are_bounded(tmp_path, monkeypatch):
    from qwen35_moe_dflash import tool_logs
    monkeypatch.setattr(tool_logs, '_MAX_SCAN_BYTES', 1024)
    log = tmp_path / "compile.log"
    payload = b'E/AI_FMK (1): long diagnostic ' + b'X' * 20000 + b'\nlast raw tail\n'
    log.write_bytes(payload)
    original_open = Path.open
    total_read = []
    class Reader:
        def __init__(self, stream):
            self.stream = stream
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return self.stream.__exit__(*args)
        def __getattr__(self, key):
            return getattr(self.stream, key)
        def read(self, size=-1):
            assert 0 < size <= 8192
            data = self.stream.read(size)
            total_read.append(len(data))
            return data
        def readline(self, size=-1):
            assert 0 < size <= 8192
            data = self.stream.readline(size)
            total_read.append(len(data))
            return data
    monkeypatch.setattr(Path, 'open', lambda path, *args, **kwargs: Reader(original_open(path, *args, **kwargs)))
    excerpt = log_excerpt(log)
    assert sum(total_read) <= 1024 + 8192
    assert 'Diagnostic scan limited to first 1024 bytes' in excerpt
    assert '[truncated]' in excerpt
    assert excerpt.endswith('last raw tail\n')
