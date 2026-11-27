"""Use the FP16 x/gamma -> z RmsnormPatch in the generated CPro Python sources."""
import ast

from .decoder_four_source import preserve_source
from .source_model import _call, _definition, _expr


REGISTRATION = '''if not hasattr(torch.ops.npu, "rmsnorm_patch"):
    m_fragment.define("rmsnorm_patch(Tensor x, Tensor gamma) -> Tensor")


@torch.library.impl(m, "rmsnorm_patch")
def rmsnorm_patch_meta(x, gamma):
    if x.dtype != torch.float16 or gamma.dtype != torch.float16:
        raise ValueError(f"RmsnormPatch requires FP16 x and gamma, got {x.dtype} and {gamma.dtype}")
    if x.ndim < 1 or gamma.ndim != 1 or x.shape[-1] != gamma.shape[0]:
        raise ValueError("RmsnormPatch requires gamma to match the last dimension of x")
    return torch.empty_like(x)


@torchair.register_fx_node_ge_converter(torch.ops.npu.rmsnorm_patch.default)
def convert_rmsnorm_patch(x: Tensor, gamma: Tensor):
    return torchair.ge.custom_op("RmsnormPatch", inputs={"x": x, "gamma": gamma}, outputs=["z"])


'''


def before_float(expr):
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
        if ast.unparse(expr).endswith(('.float()', '.to(torch.float32)', '.to(dtype=torch.float32)')):
            return expr.func.value
    return expr


def patch_sources(model, exporter):
    tree = ast.parse(model)
    for name in ('Qwen3_5MoeRMSNorm', 'Qwen3_5MoeRMSNormGated'):
        forward = _definition(_definition(tree, name), 'forward')
        call = _call(forward, 'torch_npu.npu_rms_norm')
        parents = {child: node for node in ast.walk(forward) for child in ast.iter_child_nodes(node)}
        use = parents[call]
        x, gamma = call.args[:2]
        statement = call
        while not isinstance(statement, ast.stmt):
            statement = parents[statement]
        suite, = [v for _, v in ast.iter_fields(parents[statement]) if isinstance(v, list) and statement in v]
        at = suite.index(statement)
        # Native gated RMS may widen x just before its call. Read the original
        # operand while it is still in scope and preserve the native result dtype.
        operand = before_float(x)
        original_dtype = f'.type_as({ast.unparse(x)})' if operand is x else '.float()'
        capture = at
        if isinstance(x, ast.Name) and at:
            prev = suite[at - 1]
            if (isinstance(prev, ast.Assign) and len(prev.targets) == 1 and ast.unparse(prev.targets[0]) == x.id
                    and before_float(prev.value) is not prev.value):
                capture = at - 1
                operand = before_float(prev.value)
                original_dtype = '.float()'
        prefix = ast.parse(f'''rms_gamma = {ast.unparse(before_float(gamma))}
if rms_x.dtype == torch.float32:
    rms_x = rms_x.half()
if rms_gamma.dtype == torch.float32:
    rms_gamma = rms_gamma.half()
''').body
        suite[at:at] = prefix
        suite[capture:capture] = ast.parse(f'rms_x = {ast.unparse(operand)}').body
        z = _expr('torch.ops.npu.rmsnorm_patch(rms_x.contiguous(), rms_gamma.contiguous())' + original_dtype)
        if isinstance(use, ast.Subscript) and ast.unparse(use.slice) == '0':
            parent = parents[use]
            for field, value in ast.iter_fields(parent):
                if value is use:
                    setattr(parent, field, z)
                    break
                if isinstance(value, list) and use in value:
                    value[value.index(use)] = z
                    break
        elif (isinstance(use, ast.Assign) and len(use.targets) == 1 and isinstance(use.targets[0], ast.Tuple)
              and len(use.targets[0].elts) == 2 and isinstance(use.targets[0].elts[1], ast.Name)
              and not any(isinstance(n, ast.Name) and n.id == use.targets[0].elts[1].id and isinstance(n.ctx, ast.Load)
                          for n in ast.walk(forward))):
            use.targets = [use.targets[0].elts[0]]
            use.value = z
        else:
            raise ValueError(f'{name}: review the consumers of the native RMSNorm tuple before replacing it')
        if (capture < at and suite is forward.body
                and not any(isinstance(n, ast.Name) and n.id == x.id and isinstance(n.ctx, ast.Load)
                            for s in suite[suite.index(prev) + 1:] for n in ast.walk(s))):
            # The native norm was the last reader of the explicit FP32 copy.
            suite.remove(prev)
    # Install beside the other npu fragment registrations, ahead of the exporter entry point.
    anchor = 'if not hasattr(torch.ops.npu, "softplus_patch"):'
    if exporter.count(anchor) != 1:
        raise ValueError('Expected the prepared CPro SoftplusPatch registration in the exporter')
    return preserve_source(model, ast.unparse(tree)), exporter.replace(anchor, REGISTRATION + anchor)


def inspect_ports(graph):
    rows = []
    for op in graph.op:
        if op.type == 'RmsnormPatch':
            ports = [*op.input_desc, *op.output_desc]
            if (len(op.input_desc) != 2 or len(op.output_desc) != 1
                    or any(p.name and p.name != name for p, name in zip(ports, ('x', 'gamma', 'z')))
                    or any(p.dtype != 2 or p.layout != 'ND' for p in ports)):
                raise ValueError(f'{op.name}: expected RmsnormPatch FP16/ND (x, gamma) -> z')
            rows.append(dict(node=op.name, inputs=list(op.input), output=op.name + ':0'))
    if not rows:
        raise ValueError('The requested RmsnormPatch did not reach the decoder AIR')
    print(f'[RMSNorm AIR] {len(rows)} RmsnormPatch (FP16 x, gamma → FP16 z)', flush=True)
    return rows
