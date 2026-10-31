"""Strict compiled 17-input/14-output ABI for the new cached Draft graph."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

from qwen35_moe_dflash.draft.compile_precision import _observed_desc, _read, _identity
from qwen35_moe_dflash.draft.direct_abi import inspect_direct_output_abi
from qwen35_moe_dflash.draft.ir_inspect import _enum_fields
from qwen35_moe_dflash.toolchain.omc_container import read_container
from qwen35_moe_dflash.toolchain.omc_wire import load_schema


def bind_input_nodes(graph, declared, enums):
    """Use exact Data names and explicit index attributes, never op-list order."""
    nodes = [n for n in graph.get('op', []) if n.get('type') in ('Data', 'RefData')]
    if len(nodes) != 17 or len(declared) != 17 or any(n.get('type') != 'Data' for n in nodes):
        raise ValueError('incremental compiled graph must have exactly 17 Data inputs')
    expected = {row['name']: row for row in declared}
    result, seen_names, seen_indices = [], set(), set()
    for node in nodes:
        name = node.get('name')
        if name not in expected or name in seen_names:
            raise ValueError('compiled Data name is missing, duplicated or unknown')
        attrs = node.get('attr', [])
        values = [a.get('value', {}).get('i') for a in attrs if a.get('key') == 'index']
        if len(values) != 1 or type(values[0]) is not int or not 0 <= values[0] < 17 or values[0] in seen_indices:
            raise ValueError('compiled Data index is missing, duplicated or invalid')
        descs = node.get('output_desc')
        if not isinstance(descs, list) or len(descs) != 1:
            raise ValueError('compiled Data must have one descriptor')
        actual = _observed_desc(descs[0], enums)
        row = expected[name]
        if any(actual[k] != row[k] for k in ('shape', 'dtype')):
            raise ValueError('compiled incremental input dtype/shape changed: ' + name)
        result.append(dict(row, index=values[0], bytes=math.prod(row['shape']) * (2 if row['dtype'] == 'float16' else 4)))
        seen_names.add(name); seen_indices.add(values[0])
    return sorted(result, key=lambda row: row['index'])


def inspect_incremental_abi(omc, schema, exported):
    report = {'schema_version': 1, 'artifact_kind': 'qwen35-incremental-draft-compiled-abi',
        'status': 'FAILED', 'runtime_descriptor_validation': 'NOT_RUN',
        'scope': 'Selected-schema compiled Data indices and ordered NetOutput edges; runtime descriptors still required'}
    try:
        declared = exported['outputs']
        if len(declared) != 14 or len(exported['inputs']) != 17:
            raise ValueError('incremental ABI requires all 17 inputs and 14 outputs')
        spec = [dict(row, source=row['name'], output_index=i) for i, row in enumerate(declared)]
        output = inspect_direct_output_abi(omc, schema, spec)
        report['output_inspection'] = output
        if output['status'] != 'PASSED':
            raise ValueError('incremental KV/output ABI must preserve every dtype: ' + output.get('reason', output['status']))
        raw, identity = _read(omc)
        digest = hashlib.sha256(raw).hexdigest()
        if digest != output['omc']['sha256']:
            raise ValueError('OMC changed between output and input inspection')
        selected = load_schema(schema)
        if selected.evidence != output['schema']:
            raise ValueError('selected schema changed between inspections')
        part = read_container(Path(omc))['partitions'][0]
        model = selected.decode(memoryview(raw)[part['offset']:part['offset'] + part['bytes']], 'ModelDef')
        inputs = bind_input_nodes(model['graph'][0], exported['inputs'], _enum_fields(selected))
        if _identity(Path(omc).stat()) != identity or hashlib.sha256(Path(schema).read_bytes()).hexdigest() != selected.evidence['library_sha256']:
            raise ValueError('compiled model or selected schema changed during ABI inspection')
        outputs = [dict(row, index=i, bytes=math.prod(row['shape']) * (2 if row['dtype'] == 'float16' else 4),
            connection=output['output_connections'][i]) for i, row in enumerate(declared)]
        report.update(status='PASSED', inputs=inputs, outputs=outputs, omc=output['omc'], schema=selected.evidence,
            **{key: exported[key] for key in ('context_rows', 'cache_capacity', 'max_draft_tokens')})
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        report['error'] = f'{type(error).__name__}: {error}'
    return report
