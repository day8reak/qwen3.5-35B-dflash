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
def cpro_export_contracts(enabled, chunk_size=1, *, gdn_layers=3, sequence_rows=64):
    if not enabled:
        yield
        return
    from ..layer_probe.module_contracts import gdn_state_contract

    if chunk_size not in (1, 64) or sequence_rows not in (1, 64) or chunk_size > sequence_rows:
        raise ValueError('Unsupported GDR chunk width')
    active = _GDR_WIDTH.set(chunk_size)
    try:
        with gdn_state_contract(gdn_layers > 0, state_dtype='float16', expected_calls=gdn_layers * (sequence_rows // chunk_size),
                                sequence_rows=chunk_size) as gdr:
            yield
        print('CPro GDR adapter calls:', gdr['calls'] if gdr else 0)
    finally:
        _GDR_WIDTH.reset(active)


def _only(items, label):
    items = list(items)
    if len(items) != 1:
        raise ValueError(f'expected one {label}, found {len(items)}')
    return items[0]


def half_model_preserving_qlinear_scales(model, qlinear_cls):
    """Cast the floating model while retaining the loader's original QLinear buffers."""
    scales = [(module, module.scale) for module in model.modules() if isinstance(module, qlinear_cls)]
    model.half()
    for module, scale in scales:
        module.scale = scale


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
               'import cpro_export_contracts, validate_gdr_inputs, half_model_preserving_qlinear_scales\n'
               'from models.modeling_qwen3_5_moe_hiai_cpro import QLinear as CproQLinear\n')
    edits = [(quantize, ast.unparse(ast.fix_missing_locations(quantize))),
             (converter, imports + '\n' + ast.unparse(ast.fix_missing_locations(converter)))]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'cpro_export_contracts':
            edits.append((node, ''))
    return _replace_functions(text, edits)


def _prepare_prefill_weights(fn, quant):
    load = _only((stmt for stmt in fn.body if quant in ast.walk(stmt)), 'quantization branch')
    init = ast.parse('''
from qwen35_moe_dflash.target.prefill_t64.source_exporter import half_model_preserving_qlinear_scales
from models.modeling_qwen3_5_moe_hiai_cpro import Qwen3_5MoeRMSNorm
half_model_preserving_qlinear_scales(model_wrapper.model.eval(), CproQLinear)
for module in model_wrapper.model.modules():
    if isinstance(module, Qwen3_5MoeRMSNorm):
        gamma = (module.weight + 1.0).detach().contiguous()
        module.register_buffer('export_gamma', gamma, persistent=False)
''').body
    where = fn.body.index(load) + 1
    if [ast.dump(s) for s in fn.body[where:where + len(init)]] != [ast.dump(s) for s in init]:
        fn.body[where:where] = init


def initialize_native_prefill_weights(text):
    from .decoder_four_source import preserve_source
    tree = ast.parse(text)
    fn = _function(tree, 'export_air')
    quant = _only(_calls(fn, {'quant_model_inplace'}), 'W4 load')
    _prepare_prefill_weights(fn, quant)
    return preserve_source(text, ast.unparse(tree))


def stage_prefill_exporter(text, *, gdn_layers, split):
    """Bind the CPro W4 loader and checked converters around the native export call."""
    text = stage_exporter(text)
    fn = _function(ast.parse(text), 'export_air')
    quant = _only(_calls(fn, {'quant_model_inplace'}), 'split W4 load')
    quant.keywords.append(ast.keyword(arg='qlinear_cls', value=ast.Name(id='CproQLinear', ctx=ast.Load())))
    if split:
        _prepare_prefill_weights(fn, quant)
    call, = [n for n in ast.walk(fn) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
             and ast.unparse(n.value.func) == 'torchair.dynamo_export']
    branch, = [n for n in ast.walk(fn) if isinstance(n, ast.If) and call in n.body]
    count = str(gdn_layers) if split else 'num_hidden_layers - num_hidden_layers // 4'
    context = ast.parse(f'cpro_export_contracts(True, chunk_size=64, gdn_layers={count})',
                        mode='eval').body
    branch.body[branch.body.index(call)] = ast.With(items=[ast.withitem(context_expr=context)], body=[call])
    return _replace_functions(text, [(fn, ast.unparse(ast.fix_missing_locations(fn)))])


def stage_split_exporter(text, layer):
    return stage_prefill_exporter(text, gdn_layers=int(layer % 4 != 3), split=True)
