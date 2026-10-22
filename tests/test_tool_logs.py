"""Vendor severity formats and bounded, failure-safe error reporting."""
from pathlib import Path

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
