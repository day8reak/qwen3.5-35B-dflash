"""Compile-time ABI parsing against synthetic selected-schema OMCs, no SDK run."""
from copy import deepcopy

import pytest

from qwen35_moe_dflash.config import Qwen35DFlashConfig
from qwen35_moe_dflash.ge_export import configurations
from qwen35_moe_dflash.ge_incremental_draft import build_incremental_draft
from qwen35_moe_dflash.incremental_draft_abi import inspect_incremental_abi
from test_draft_direct_abi import case
from test_draft_ir_inspect import _blob, _int, _desc, _attr
from test_draft_precision_omc import _container


def _write(case, *, damage=None):
    path, schema, _ = case
    graph = build_incremental_draft(Qwen35DFlashConfig.from_dict(configurations()[1]))
    exported = {kind: [{'name': r.name, 'dtype': r.dtype, 'shape': list(r.shape)} for r in getattr(graph, kind)]
                for kind in ('inputs', 'outputs')}
    exported.update(context_rows=16, cache_capacity=192, max_draft_tokens=15)
    data = []
    # Deliberately reverse physical indices relative to the source declaration:
    # the ABI must bind names to saved compiled indices, never list order.
    for i, row in enumerate(exported['inputs']):
        index = 16 - i
        if damage == 'duplicate-index' and i == 1: index = 16
        name = row['name'] if damage != 'unknown-name' or i != 0 else 'renamed_unbound'
        desc = _desc(3 if row['dtype'] == 'int32' else 7, row['shape'])
        if damage == 'input-dtype' and i == 5: desc = _desc(1, row['shape'])
        node = _blob(1, name) + _blob(2, 'Data') + _blob(34, desc)
        if damage != 'missing-index' or i != 0: node += _attr('index', _int(3, index))
        data.append(_blob(6, node))
    outputs = [_desc(3 if r['dtype'] == 'int32' else 7, r['shape']) for r in exported['outputs']]
    if damage == 'fp32-kv': outputs[2] = _desc(1, exported['outputs'][2]['shape'])
    if damage == 'cache-shape': outputs[2] = _desc(7, [1, 8, 64, 128])
    producer = _blob(1, 'partition') + _blob(2, 'GraphOp') + b''.join(_blob(34, d) for d in outputs)
    net = _blob(1, 'NetOutput') + _blob(2, 'NetOutput') + b''.join(_blob(5, f'partition:{i}') for i in range(14))
    net += b''.join(_blob(33, d) for d in outputs)
    compiled = _blob(1, 'draft') + b''.join(reversed(data)) + _blob(6, producer) + _blob(6, net)
    path.write_bytes(_container(_blob(7, compiled), b'opaque tasks'))
    return path, schema, exported


def test_actual_encoded_selected_schema_binds_all17_inputs_and14_outputs(case):
    path, schema, exported = _write(case)
    before = deepcopy(exported), path.read_bytes()
    result = inspect_incremental_abi(path, schema, exported)
    assert result['status'] == 'PASSED', result
    assert len(result['inputs']) == 17 and len(result['outputs']) == 14
    assert result['inputs'][0]['name'] == exported['inputs'][-1]['name']
    assert result['inputs'][-1]['name'] == 'target_features'
    assert result['runtime_descriptor_validation'] == 'NOT_RUN'
    assert (exported, path.read_bytes()) == before


@pytest.mark.parametrize('damage', ['duplicate-index', 'unknown-name', 'missing-index', 'input-dtype', 'fp32-kv', 'cache-shape'])
def test_no_dtype_promotion_missing_identity_or_cache_shape_fallback(case, damage):
    result = inspect_incremental_abi(*_write(case, damage=damage))
    assert result['status'] == 'FAILED' and 'inputs' not in result
