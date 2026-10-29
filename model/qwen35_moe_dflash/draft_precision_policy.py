"""Generate the observed OMG selector used with the strict CPU precision gate.

Implementation 7 means ``custom_ascendc_lib``, not CPU. The same-family OMG
smoke selected this store for the FP32 RMS operations, found no implementation,
and compiled their complete arithmetic as CPUCL FP32. A selected DDK can have
different registrations, so this request alone never establishes placement:
``verify_compiled_precision`` must pass before the model reaches the board.

The groups_info/nodes_info format is shared with the previously exercised
qwen3.5-0.8b generate_npucl_partition_config.py. A single group is required for
partial node selection: the observed parser rejects omitted nodes when multiple
groups are requested. The two-RMS and complete tiny-draft compiler smokes accept
one group even with intervening FP16 work. Source Cast, constant, weight, and
FP16 arithmetic records are deliberately not assigned.
"""
from __future__ import annotations

import json

from .draft_compile_precision import _check_plan


NPUCL_IMPLEMENTATION = 7


def customization_bytes(plan):
    """Return deterministic public --cl_customization JSON for a checked plan.

    All requested nodes share the one observed partial-selection group. No plan
    receipts or diagnostic metadata enter the vendor configuration, and no input
    object or model file is changed. The compiled gate remains mandatory.
    """
    required = _check_plan(plan)
    groups = [{"npucl_implementation": NPUCL_IMPLEMENTATION}]
    nodes = {node["name"]: {"group_id": 0} for node in required}
    config = {"npucl_strategy": {"npucl_graphs": {
        plan["graph"]: {"groups_info": groups, "nodes_info": nodes}}}}
    return (json.dumps(config, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode("utf-8")
