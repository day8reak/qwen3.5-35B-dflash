"""Optional StableArgMax integration; compiler helpers do not import Torch."""
import importlib.util
import os
from pathlib import Path
import sys

POLICIES = ('native', 'stable-arg-max')


def contract(policy, shape):
    if policy not in POLICIES:
        raise ValueError('unknown Draft Top1 policy: ' + str(policy))
    return dict(policy=policy, input_shape=list(shape), output_shape=list(shape[:-1]),
                input_dtype='float16', output_dtype='int32', axis=-1,
                tie_rule='SMALLEST_INDEX',
                input_contract='FINITE_FP16' if policy == 'stable-arg-max' else 'PYTORCH_ARGMAX',
                op_type='StableArgMax' if policy == 'stable-arg-max' else 'ArgMax')


def register(torchair):
    """Reuse the operator's portable schema/Meta/converter in this process."""
    path = Path(__file__).resolve().parents[4] / 'targets/cpro/ops/draft-top1/python/stable_arg_max.py'
    module = sys.modules.get('stable_arg_max')
    if module is None:
        spec = importlib.util.spec_from_file_location('stable_arg_max', path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(spec.name, None)
            raise
    elif Path(module.__file__).resolve() != path:
        raise ValueError('another stable_arg_max module is already loaded')
    return module.register_ge(torchair)


def inspect_installation(ddk_root):
    """Resolve the selected DDK wrapper without importing vendor Python."""
    from importlib.machinery import PathFinder
    from qwen35_moe_dflash.toolchain.compile import omg_environment
    root = Path(ddk_root).resolve()
    _, env = omg_environment(root)
    search = [p for p in env.get('PYTHONPATH', '').split(os.pathsep) if p]
    package = PathFinder.find_spec('custom', search)
    paths = list(package.submodule_search_locations or []) if package else []
    module = PathFinder.find_spec('custom.stable_arg_max', paths) if paths else None
    origin = getattr(module, 'origin', None)
    if not origin or not Path(origin).is_file() or not Path(origin).resolve().is_relative_to(root):
        raise ValueError('selected DDK cannot resolve custom.stable_arg_max; install the existing allops operator before OMG')
    return dict(status='STATIC_PATH_RESOLVED', module='custom.stable_arg_max', origin=origin,
                scope='STATIC_IMPORT_PATH_ONLY_NOT_WORKER_EXECUTION')


def inspect_source(nodes, expected, desc, mismatch, deferred):
    """Check the custom reduction's ABI and connection to public output zero."""
    if expected is None:
        expected = dict(policy='native')
    if expected['policy'] not in POLICIES:
        raise ValueError('unknown Top1 policy in AIR manifest')
    custom = [n for n in nodes if n['type'] == 'StableArgMax']
    if expected['policy'] == 'native':
        if custom:
            mismatch('graph', 'top1.policy', 'native', 'StableArgMax')
        return dict(policy='native', custom_nodes=len(custom))
    if expected != contract('stable-arg-max', expected['input_shape']):
        raise ValueError('unsupported StableArgMax contract')
    reductions = [n['name'] for n in nodes if n['type'].lower().startswith('argmax')]
    if len(custom) != 1 or reductions:
        mismatch('graph', 'top1.nodes', {'StableArgMax': 1, 'ArgMax': []},
                 {'StableArgMax': len(custom), 'ArgMax': reductions})
    if len(custom) != 1:
        return dict(policy=expected['policy'], custom_nodes=len(custom))
    node = custom[0]
    if len(node.get('input', [])) != 1 or not node['input'][0]:
        mismatch(node['name'], 'input_count', 1, node.get('input', []))
    for direction, dtype in (('input', 'DT_FLOAT16'), ('output', 'DT_INT32')):
        rows = node.get(direction + '_desc', [])
        if len(rows) != 1:
            mismatch(node['name'], direction + '_count', 1, len(rows))
            continue
        actual = desc(rows[0])
        if dtype not in actual['dtype_names']:
            mismatch(node['name'], direction + '.dtype', dtype, actual['dtype_names'])
        shape = expected[direction + '_shape']
        if actual['shape'] is None:
            deferred.append(dict(node=node['name'], role='top1.' + direction, expected=shape,
                actual=None, status='DEFERRED_TO_OMG_SHAPE_INFERENCE'))
        elif actual['shape'] != shape:
            mismatch(node['name'], direction + '.shape', shape, actual['shape'])
    outputs = [n for n in nodes if n['type'] == 'NetOutput']
    edge = outputs[0].get('input', [None])[0] if len(outputs) == 1 and outputs[0].get('input') else None
    if edge != node['name'] + ':0':
        mismatch(node['name'], 'public_output_0', node['name'] + ':0', edge)
    return dict(policy=expected['policy'], node=node['name'], input=node.get('input'), output=edge,
                scope='SOURCE_ABI_AND_OUTPUT_EDGE; NOT_DEVICE_FINITE_LOGIT_PROOF')


def inspect_compiled(observed, expected):
    """Require the INT32 public Top1 output to come directly from an NPU task."""
    outputs = [n for n in observed['nodes'] if n['type'] == 'NetOutput']
    if len(outputs) != 1 or not outputs[0].get('input'):
        raise ValueError('cannot identify compiled Top1 output')
    edge = outputs[0]['input'][0]
    name, sep, port = edge.rpartition(':')
    producers = [n for n in observed['nodes'] if n['name'] == name]
    if not sep or not port.isdecimal() or len(producers) != 1:
        raise ValueError('cannot resolve compiled Top1 producer: ' + edge)
    node = producers[0]
    if node['type'] != 'GraphOp' or node.get('owner') != 'NPUCL':
        raise ValueError('StableArgMax output is not directly owned by NPUCL: ' + edge)
    index = int(port)
    if index >= len(node['outputs']):
        raise ValueError('compiled Top1 output port is missing')
    actual = node['outputs'][index]
    if actual['dtype'] != 'int32' or actual['shape'] != expected['output_shape']:
        raise ValueError('compiled Top1 descriptor differs: ' + str(actual))
    return dict(status='NPU_OUTPUT_BOUNDARY_PASSED', producer=edge, descriptor=actual,
                scope='COMPILED_TERMINAL_BOUNDARY_ONLY; NPU_KERNEL_INTERNALS_NOT_DECODED')
