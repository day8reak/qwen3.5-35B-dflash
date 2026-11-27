"""Prepare decoder export copies of the CPro PyTorch sources."""
import ast
from pathlib import Path
from shutil import copyfile

from ..original_target_source import EXPORTER, cpro_python_files
from .decoder_four_silu import MODELING
from .decoder_four_source import preserve_source
from .source_model import _call, _definition


def save_air_source(source, destination, *, decoder_files=()):
    """Keep the final CPro Python sources as regular files in the AIR package."""
    for path in cpro_python_files(source):
        saved = destination/'cpro'/path.relative_to(source)
        saved.parent.mkdir(parents=True, exist_ok=True)
        copyfile(path, saved)
    if decoder_files:
        driver = destination/'decoder'
        driver.mkdir(exist_ok=True)
        for path in decoder_files:
            copyfile(path, driver/path.name)


def fp16_shared_gate(source):
    tree = ast.parse(source)
    moe = _definition(tree, 'Qwen3_5MoeSparseMoeBlock')
    forward = _definition(moe, 'forward')
    sigmoid, = [call for call in ast.walk(forward) if isinstance(call, ast.Call)
                and ast.unparse(call.func) in ('F.sigmoid', 'torch.sigmoid')
                and len(call.args) == 1 and 'self.shared_expert_gate' in ast.unparse(call.args[0])]
    gate, = sigmoid.args
    expr = 'F.linear(hidden_states_flat.half(), self.shared_expert_gate.weight.half()).float()'
    if ast.unparse(gate) == expr:
        return source
    if ast.unparse(gate) not in ('self.shared_expert_gate(hidden_states_flat)',
                                 'self.shared_expert_gate(hidden_states_flat.to(self.shared_expert_gate.weight.dtype))'):
        raise ValueError(f'Unrecognized CPro scalar shared expert gate: {ast.unparse(gate)}')
    sigmoid.args = [ast.parse(expr, mode='eval').body]
    return preserve_source(source, ast.unparse(tree))


def lookup_exporter(source):
    """Use token IDs and the checkpoint table in the native 40-layer export."""
    tree = ast.parse(source)
    export = _definition(tree, 'export_air')
    # Keep W4 model preparation. The separate embedding tables are only used
    # for the external-embedding ABI and can be large even for dummy inputs.
    setup, = [s for s in export.body if isinstance(s, ast.If)
              and any(isinstance(n, ast.Call) and ast.unparse(n.func) == 'quant_model_inplace'
                      for n in ast.walk(s))]
    table_names = {'quant_embedding_weights', 'quant_embedding_scale'}
    setup.body = [s for s in setup.body if not (isinstance(s, ast.Assign)
                  and ast.unparse(s.targets[0]) in table_names)]
    dummy, = [s for s in export.body if isinstance(s, ast.If) and any(
        isinstance(n, ast.Name) and n.id == 'quant_embedding_weights' for n in ast.walk(s))]
    export.body.remove(dummy)
    call, = [n for n in ast.walk(export) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'torchair.dynamo_export']
    args = {kw.arg: kw for kw in call.keywords}
    for key in ('inputs_embeds', 'embed_scale'):
        args[key].value = ast.Constant(None)
    # Distinguish token IDs from position_ids even if OMG reorders identically shaped slots.
    args['input_ids'].value = ast.parse('input_ids.long()', mode='eval').body
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) in ('embedding_weight_path', 'embedding_scale_path'):
            key = ast.unparse(n.targets[0])
            if ast.unparse(n.value) == f"config_data['{key}']":
                n.value = ast.parse(f"config_data.get('{key}')", mode='eval').body
    return preserve_source(source, ast.unparse(tree))


def inspect_shared_gates(graph, layers):
    nodes = {n.name: n for n in graph.op}
    def source(edge):
        op = nodes[edge.rsplit(':', 1)[0]]
        while op.type in ('Identity', 'Reshape', 'View', 'Squeeze', 'Unsqueeze'):
            op = nodes[op.input[0].rsplit(':', 1)[0]]
        return op
    gates = []
    for sigmoid in graph.op:
        if sigmoid.type != 'Sigmoid' or not sigmoid.input:
            continue
        cast = source(sigmoid.input[0])
        if cast.type != 'Cast' or not cast.input:
            continue
        mm = source(cast.input[0])
        if (mm.type in ('MatMul', 'MatMulV2') and [p.dtype for p in mm.input_desc[:2]] == [2, 2]
                and [p.dtype for p in mm.output_desc] == [2] and [p.dtype for p in cast.output_desc] == [1]
                and sigmoid.input_desc[0].dtype == 1):
            gates.append(dict(matmul=mm.name, cast=cast.name, sigmoid=sigmoid.name))
    if len(gates) != layers:
        raise ValueError(f'Expected {layers} FP16 shared gate MatMuls with FP32 cast before sigmoid; found {gates}')
    print(f'[AIR] {len(gates)} shared gate: FP16 MatMul → FP32 Cast → Sigmoid', flush=True)
    return gates


def export_source(source, dest, *, lookup=False, token_gate=False, fold_norm_gamma=False):
    """Keep imported helpers beside the run-local model; preserve editable sources."""
    from .decoder_four_silu import inline_old_silu, token_view_rms_gate

    (dest/'models').mkdir(parents=True)
    for folder in (Path('.'), Path('models')):
        for path in (source/folder).iterdir():
            rel = folder/path.name
            if path.name == '__pycache__' or rel in (Path('models'), Path(EXPORTER), Path(MODELING)):
                continue
            (dest/rel).symlink_to(path.resolve(), target_is_directory=path.is_dir())
    model = fp16_shared_gate((source/MODELING).read_text())
    if token_gate:
        model = token_view_rms_gate(inline_old_silu(model))
    if fold_norm_gamma:
        tree = ast.parse(model)
        norm = _definition(_definition(tree, 'Qwen3_5MoeRMSNorm'), 'forward')
        call = _call(norm, 'torch_npu.npu_rms_norm')
        operand = ast.unparse(call.args[1])
        if operand == 'self.weight + 1.0':
            call.args[1] = ast.parse('self.export_gamma', mode='eval').body
            model = preserve_source(model, ast.unparse(tree))
        elif operand != 'self.export_gamma':
            raise ValueError(f'Unrecognized ordinary RMSNorm gamma expression: {operand}')
    (dest/MODELING).write_text(model)
    if lookup:
        (dest/EXPORTER).write_text(lookup_exporter((source/EXPORTER).read_text()))
    else:
        copyfile(source/EXPORTER, dest/EXPORTER)
    return dest
