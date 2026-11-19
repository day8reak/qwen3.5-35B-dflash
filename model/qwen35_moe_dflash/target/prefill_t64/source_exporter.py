"""Adapt an owner's local exporter and weight loader for the T64 CPro prefix."""
import ast
from contextlib import contextmanager
from contextvars import ContextVar


_GDR_WIDTH = ContextVar('cpro_t64_gdr_width', default=None)


def validate_gdr_inputs(chunk_size, *operands):
    width = _GDR_WIDTH.get()
    if width is not None:
        from torchair.ge import DataType
        if chunk_size != width or any(x.dtype != DataType.DT_FLOAT16 for x in operands):
            raise ValueError(f'T64 GDR requires chunk_size={width} and FP16 query, key, value and beta')


@contextmanager
def cpro_export_contracts(enabled, chunk_size=1):
    if not enabled:
        yield
        return
    from ..layer_probe.candidates import qlinear_ge_scales
    from ..layer_probe.module_contracts import gdn_state_contract

    if chunk_size not in (1, 64):
        raise ValueError('Unsupported GDR chunk width')
    active = _GDR_WIDTH.set(chunk_size)
    try:
        with qlinear_ge_scales(True, expected_calls=None) as projections:
            with gdn_state_contract(True, state_dtype='float16', expected_calls=3 * (64 // chunk_size),
                                    sequence_rows=chunk_size) as gdr:
                yield
        print('CPro GE adapter calls:', {'qlinear': projections['calls'], 'gdn': gdr['calls']})
    finally:
        _GDR_WIDTH.reset(active)


def _only(items, label):
    items = list(items)
    if len(items) != 1:
        raise ValueError(f'expected one {label}, found {len(items)}')
    return items[0]


def _function(tree, name):
    return _only((node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == name), name)


def _factory_parameter(function):
    name = 'qlinear_cls'
    if name not in [arg.arg for arg in function.args.args]:
        function.args.args.append(ast.arg(arg=name))
        function.args.defaults.append(ast.Constant(None))


def _calls(function, names):
    return [n for n in ast.walk(function) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name) and n.func.id in names]


def _replace_functions(text, edits):
    lines = text.splitlines(keepends=True)
    for function, replacement in sorted(edits, key=lambda item: item[0].lineno, reverse=True):
        start = min([function.lineno, *(d.lineno for d in function.decorator_list)]) - 1
        lines[start:function.end_lineno] = [replacement.rstrip() + '\n']
    return ''.join(lines)


def stage_utils(text):
    loader = _function(ast.parse(text), 'replace_linear_to_QLinear')
    _factory_parameter(loader)
    constructor = _only(_calls(loader, {'QLinear', 'factory'}), 'linear layer constructor')
    if constructor.func.id == 'factory':
        return text
    constructor.func = ast.parse('QLinear if qlinear_cls is None else qlinear_cls', mode='eval').body
    return _replace_functions(text, [(loader, ast.unparse(ast.fix_missing_locations(loader)))])


def stage_exporter(text):
    tree = ast.parse(text)
    quantize = _function(tree, 'quant_model_inplace')
    _factory_parameter(quantize)
    replace = _only(_calls(quantize, {'replace_linear_to_QLinear'}), 'weight loader call')
    if not any(kw.arg == 'qlinear_cls' for kw in replace.keywords):
        replace.keywords.append(ast.keyword(arg='qlinear_cls', value=ast.Name(id='qlinear_cls', ctx=ast.Load())))

    converter = _function(tree, 'convert_npu_chunk_gated_delta_rule')
    guard = ast.parse('validate_gdr_inputs(chunk_size, query, key, value, beta)').body[0]
    converter.body.insert(0, guard)
    imports = ('from qwen35_moe_dflash.target.prefill_t64.source_exporter '
               'import cpro_export_contracts, validate_gdr_inputs\n'
               'from models.modeling_qwen3_5_moe_hiai_cpro import QLinear as CproQLinear\n')
    edits = [(quantize, ast.unparse(ast.fix_missing_locations(quantize))),
             (converter, imports + '\n' + ast.unparse(ast.fix_missing_locations(converter)))]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'cpro_export_contracts':
            edits.append((node, ''))
    return _replace_functions(text, edits)
