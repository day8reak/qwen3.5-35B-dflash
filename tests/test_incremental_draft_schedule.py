import pytest

from qwen35_moe_dflash.config import Qwen35DFlashConfig
from qwen35_moe_dflash.ge_export import configurations
from qwen35_moe_dflash.incremental_draft_schedule import plan_incremental_round
from qwen35_moe_dflash.ge_incremental_draft import build_incremental_draft, validate_incremental_draft_controls


@pytest.mark.parametrize('width', [15, 7, 3, 1])
def test_complete_prefix_chunking_latest_anchor_exclusion_and_no_delta_round(width):
    config = Qwen35DFlashConfig.from_dict(configurations()[1])
    calls = plan_incremental_round(config, cached_rows=0, committed_tokens=66, anchor=8, max_draft_tokens=width)
    assert [(c.start_position, c.valid_rows, c.proposal_count) for c in calls] == [
        (0, 16, 0), (16, 16, 0), (32, 16, 0), (48, 16, 0), (64, 1, width)]
    assert calls[-1].context_end == 65
    again = plan_incremental_round(config, cached_rows=65, committed_tokens=66, anchor=9, max_draft_tokens=width)
    assert len(again) == 1 and again[0].valid_rows == 0 and again[0].proposal_count == width
    # Initial prompt excluded its anchor; a full accepted block plus bonus now
    # admits the old anchor + accepted tokens, excluding the new bonus anchor.
    later = plan_incremental_round(config, cached_rows=65, committed_tokens=66 + width + 1,
        anchor=10, max_draft_tokens=width)
    assert sum(c.valid_rows for c in later) == width + 1
    graph = build_incremental_draft(config, max_draft_tokens=width)
    assert graph.outputs[0].shape == (1, width)
    with pytest.raises(ValueError):
        validate_incremental_draft_controls(config, start_position=0, valid_rows=1,
            anchor=1, proposal_count=width + 1, max_draft_tokens=width)


@pytest.mark.parametrize('cursor,tokens', [(1, 1), (-1, 4), (True, 4), (0, 0), (0, 194)])
def test_no_rewind_no_silent_truncation_or_invalid_cursor(cursor, tokens):
    with pytest.raises(ValueError):
        plan_incremental_round(Qwen35DFlashConfig.from_dict(configurations()[1]),
            cached_rows=cursor, committed_tokens=tokens, anchor=0)
