"""Prepare decoder export copies of the CPro PyTorch sources."""
import ast
from pathlib import Path
from shutil import copyfile

from ..original_target_source import EXPORTER, cpro_python_files
from .decoder_four_silu import MODELING
from .decoder_four_source import preserve_source
from .source_model import _call, _definition, restore_qlinear_scales


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


def fp16_shared_gate(source, *, fp16_sigmoid=False):
    tree = ast.parse(source)
    moe = _definition(tree, 'Qwen3_5MoeSparseMoeBlock')
    forward = _definition(moe, 'forward')
    sigmoid, = [call for call in ast.walk(forward) if isinstance(call, ast.Call)
                and ast.unparse(call.func) in ('F.sigmoid', 'torch.sigmoid')
                and len(call.args) == 1 and 'self.shared_expert_gate' in ast.unparse(call.args[0])]
    gate, = sigmoid.args
    linear = 'F.linear(hidden_states_flat.half(), self.shared_expert_gate.weight.half())'
    expr = linear if fp16_sigmoid else linear + '.float()'
    if ast.unparse(gate) == expr:
        return source
    if ast.unparse(gate) not in (linear, linear + '.float()', 'self.shared_expert_gate(hidden_states_flat)',
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


def _shared_gate_output(graph, sigmoid):
    """Report where the sigmoid value goes, including any downstream dtype conversion."""
    def dtypes(ports):
        return [p.DESCRIPTOR.file.enum_types_by_name['DataType'].values_by_number[p.dtype].name
                for p in ports]

    sigmoid_dtypes = dtypes(sigmoid.output_desc)
    print(f'[AIR] {sigmoid.name} output={sigmoid_dtypes}', flush=True)
    pending = [sigmoid.name + ':0']
    seen = set()
    links = []
    views_or_casts = ('Cast', 'CastT', 'Identity', 'Reshape', 'View', 'Squeeze', 'Unsqueeze',
                      'ExpandDims', 'BroadcastTo')
    while pending:
        edge = pending.pop()
        if edge in seen:
            continue
        seen.add(edge)
        for op in graph.op:
            for slot, arg in enumerate(op.input):
                if arg != edge:
                    continue
                inputs, outputs = dtypes(op.input_desc), dtypes(op.output_desc)
                links.append(dict(source=edge, name=op.name, op=op.type, slot=slot,
                                  input_dtypes=inputs, output_dtypes=outputs))
                print(f'[AIR] {edge} → {op.name} ({op.type}) input[{slot}], '
                      f'inputs={inputs}, outputs={outputs}', flush=True)
                if slot == 0 and op.type in views_or_casts:
                    pending.append(op.name + ':0')
    muls = sorted({link['name'] for link in links if link['op'] in ('Mul', 'MulV2')})
    if len(muls) != 1:
        print(f'[AIR] {sigmoid.name}: expected one observable downstream Mul, found {muls}', flush=True)
    return dict(sigmoid_output_dtypes=sigmoid_dtypes, consumers=links, mul=muls[0] if len(muls) == 1 else None)


def inspect_shared_gates(graph, layers, *, fp16_sigmoid=False):
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
        producer = source(sigmoid.input[0])
        cast_name = None
        if fp16_sigmoid:
            mm = producer
        else:
            if producer.type != 'Cast' or not producer.input or [p.dtype for p in producer.output_desc] != [1]:
                continue
            cast_name = producer.name
            mm = source(producer.input[0])
        dtype = 2 if fp16_sigmoid else 1  # GE DT_FLOAT16 / DT_FLOAT
        if (mm.type not in ('MatMul', 'MatMulV2') or [p.dtype for p in mm.input_desc[:2]] != [2, 2]
                or [p.dtype for p in mm.output_desc] != [2] or not sigmoid.input_desc
                or sigmoid.input_desc[0].dtype != dtype):
            continue
        row = dict(matmul=mm.name, cast=cast_name, sigmoid=sigmoid.name)
        if fp16_sigmoid:
            print(f'[AIR] shared gate: {mm.name} (DT_FLOAT16) → {sigmoid.name} input[0] DT_FLOAT16', flush=True)
            row.update(_shared_gate_output(graph, sigmoid))
        gates.append(row)
    path = 'FP16 MatMul → Sigmoid (FP16 input)' if fp16_sigmoid else 'FP16 MatMul → FP32 Cast → Sigmoid'
    if len(gates) != layers:
        raise ValueError(f'Expected {layers} shared gates with {path}; found {gates}')
    print(f'[AIR] {len(gates)} shared gate: {path}', flush=True)
    return gates


def export_source(source, dest, *, lookup=False, fold_norm_gamma=False, rmsnorm_patch=False, shared_gate_fp16_sigmoid=False):
    """Keep imported helpers beside the run-local model; preserve editable sources."""
    from .decoder_four_silu import remove_activation_views, update_activation_calls

    (dest/'models').mkdir(parents=True)
    for folder in (Path('.'), Path('models')):
        for path in (source/folder).iterdir():
            rel = folder/path.name
            if path.name == '__pycache__' or rel in (Path('models'), Path(EXPORTER), Path(MODELING)):
                continue
            (dest/rel).symlink_to(path.resolve(), target_is_directory=path.is_dir())
    model = remove_activation_views((source/MODELING).read_text())
    model = update_activation_calls(fp16_shared_gate(model, fp16_sigmoid=shared_gate_fp16_sigmoid))
    tree = ast.parse(model)
    restore_qlinear_scales(tree)
    model = preserve_source(model, ast.unparse(tree))
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
    exporter = (source/EXPORTER).read_text()
    # A prepared tree can predate direct input forwarding. Update its Meta
    # registrations too, otherwise FX stops before the GE converter sees x.
    for name, op in (('silu_patch', 'SiluPatch'), ('softplus_patch', 'SoftplusPatch')):
        header = f'def {name}_meta(x):\n'
        guard = f'    if x.dtype != torch.float16:\n        raise ValueError("{op} expects FP16 input")\n'
        exporter = exporter.replace(header + guard, header)
    if rmsnorm_patch:
        from .decoder_rmsnorm import patch_sources
        model, exporter = patch_sources(model, exporter)
    (dest/MODELING).write_text(model)
    if lookup:
        exporter = lookup_exporter(exporter)
    (dest/EXPORTER).write_text(exporter)
    return dest
