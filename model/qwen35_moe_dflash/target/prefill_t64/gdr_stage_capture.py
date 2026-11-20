"""Observe the actual gated norm expression and the adjacent unscaled chunk-64 GDR."""
import ast
from functools import wraps
import inspect
import textwrap

from .segment_gdr import audit_operands
from .segment_prefix import tensor_record
from ..layer_probe.capture import cpu

from .gdr_stage_data import NORM


def _retain(name, expr):
    return ast.NamedExpr(ast.Name('_read_' + name, ast.Store()), expr)


def norm_reader(method):
    """Return the original result and references to the tensors evaluated by this forward."""
    source = inspect.unwrap(method)
    tree = ast.parse(textwrap.dedent(inspect.getsource(source)))
    fn, = [s for s in tree.body if isinstance(s, ast.FunctionDef)]
    fn.decorator_list = []
    names = [x.arg for x in fn.args.args]
    _, x, z = names[:3]
    # Preserve the incoming values before source locals are reassigned.
    fn.body[:0] = ast.parse(f'_read_rms_input = {x}\n_read_gate = {z}\n_read_gamma = self.weight').body
    silu, = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and ast.unparse(n.func).endswith('.silu')]
    parents = {child: parent for parent in ast.walk(fn) for child in ast.iter_child_nodes(parent)}
    mul = parents[silu]
    if not (isinstance(mul, ast.BinOp) and isinstance(mul.op, ast.Mult) and mul.right is silu):
        raise ValueError('Expected the gated norm expression hidden_states * F.silu(gate.to(...))')
    silu.args[0] = _retain('silu_input', silu.args[0])
    mul.right = _retain('silu_output', silu)
    mul.left = _retain('mul_lhs', mul.left)
    parent = parents[mul]
    field, = [k for k, v in ast.iter_fields(parent) if v is mul]
    setattr(parent, field, _retain('mul_output', mul))
    calls = [c for c in ast.walk(fn) if isinstance(c, ast.Call) and ast.unparse(c.func).endswith('.npu_rms_norm')]
    rms_y = '_read_mul_lhs'
    if calls:
        rms, = calls
        rms.args[0] = _retain('rms_input', rms.args[0])
        rms.args[1] = _retain('gamma', rms.args[1])
        parent = parents[rms]
        field, = [k for k, v in ast.iter_fields(parent) if v is rms]
        setattr(parent, field, _retain('rms_result', rms))
        rms_y = '_read_rms_result[0]'
    ret, = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
    aliases = {'rms_output': rms_y, 'mul_rhs': '_read_silu_output'}
    args = [aliases.get(k, '_read_' + k) for k in NORM[:-1]]
    ret.value = ast.Tuple([ret.value, *(ast.parse(v, mode='eval').body for v in args)], ast.Load())
    ns = dict(source.__globals__)
    exec(compile(ast.fix_missing_locations(tree), inspect.getfile(source), 'exec'), ns)
    return ns[fn.name]


def observe_norm(module, snapshot, cleanup):
    norm = module.norm
    original = norm.forward
    observed = norm_reader(original)
    prior = vars(norm).get('forward')
    def restore():
        if prior is not None:
            norm.forward = prior
        elif 'forward' in vars(norm):
            del norm.forward
    @wraps(original)
    def forward(*args, **kwargs):
        restore()
        result, *tensors = observed(norm, *args, **kwargs)
        snapshot['norm'] = {k: cpu(t) for k, t in zip(NORM, [*tensors, result], strict=True)}
        snapshot['epsilon'] = float(norm.variance_epsilon)
        return result
    norm.forward = forward
    cleanup.callback(restore)


def capture_stages(root, manifest, snapshots):
    records = {}
    for layer, snapshot in snapshots.items():
        call, = manifest[f'l{layer}_gdr']['calls']
        data = snapshot['gdr']
        if (call.get('error') or call['chunk_size'] != 64 or call['effective_length'] != [19]
                or not data['attrs']['use_qk_l2norm_in_kernel'] or not data['attrs']['output_final_state']):
            raise ValueError(f'Unexpected L{layer} GDR call {call}')
        core, state = data['outputs']
        audit, _ = audit_operands(data['inputs'], core, state)
        tensors = dict(**data['inputs'], core=core, final_state=state, **snapshot['norm'])
        folder = root / 'stages' / f'l{layer}'
        folder.mkdir(parents=True)
        records[str(layer)] = dict(epsilon=snapshot['epsilon'], attrs=data['attrs'], cpu_on_310p_operands=audit,
            tensors={k: tensor_record(root, str((folder / k).relative_to(root)), v) for k, v in tensors.items()})
    manifest['gdr_stages'] = records
