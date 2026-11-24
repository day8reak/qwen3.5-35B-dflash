"""Materialize a reviewable four-layer variant of the installed CPro Python sources."""
import ast
from difflib import SequenceMatcher
from textwrap import indent

from .chain_air import stage_cpro_source
from .prepare import EXPORTER, MODELING, WRAPPER
from .source_wrapper import four_decoder_hidden_wrapper


def preserve_source(original, adapted):
    """Apply changed statements to the original text, retaining comments in unchanged code."""
    old, new = ast.parse(original), ast.parse(adapted)
    lines = original.splitlines(keepends=True)
    edits = []

    def start(node):
        return min([node.lineno, *(d.lineno for d in getattr(node, 'decorator_list', ()))]) - 1

    def key(node):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            header = type(node)(**{field: ([] if field == 'body' else getattr(node, field))
                                   for field in node._fields})
            return ast.dump(header)
        return ast.dump(node)

    def compare(before, after):
        diff = SequenceMatcher(None, list(map(key, before)), list(map(key, after)), autojunk=False)
        for tag, i, j, k, end in diff.get_opcodes():
            if tag == 'equal':
                for a, b in zip(before[i:j], after[k:end]):
                    if isinstance(a, (ast.ClassDef, ast.FunctionDef)):
                        compare(a.body, b.body)
                continue
            begin = start(before[i]) if i < len(before) else before[-1].end_lineno
            stop = before[j - 1].end_lineno if j > i else begin
            margin = ' ' * before[min(i, len(before) - 1)].col_offset
            text = ''.join(indent(ast.unparse(n), margin) + '\n' for n in after[k:end])
            edits.append((begin, stop, text))

    compare(old.body, new.body)
    for begin, stop, text in sorted(edits, reverse=True):
        lines[begin:stop] = [text]
    result = ''.join(lines)
    if ast.dump(ast.parse(result)) != ast.dump(new):
        raise ValueError('Readable source differs from the selected CPro adaptation')
    return result


def stage(source, dest, *, keep_weight_scale=False):
    stage_cpro_source(source, dest, four_decoder_hidden_wrapper, keep_weight_scale=keep_weight_scale)
    for name in (MODELING, WRAPPER, EXPORTER, 'utils.py'):
        path = dest / name
        path.write_text(preserve_source((source / name).read_text(), path.read_text()))
    path = dest / EXPORTER
    text = path.read_text()
    main_guard, = [n for n in ast.parse(text).body if isinstance(n, ast.If)
                   and ast.unparse(n.test) == "__name__ == '__main__'"]
    lines = text.splitlines(keepends=True)
    lines[main_guard.lineno - 1:main_guard.end_lineno] = [f'\nKEEP_WEIGHT_SCALE = {keep_weight_scale!r}\n' + """

def load_four_decoder_weights(model, weight_path):
    from models.modeling_qwen3_5_moe_hiai_cpro import Qwen3_5MoeRMSNorm

    quant_model_inplace(model, str(weight_path), model.model.config.num_experts,
                        start_layer_idx=0, device='cpu', quant_bits=4, qlinear_cls=CproQLinear)
    # Hold the loader's original tensors across the global cast to avoid FP16 rounding.
    scales = [(module, module.scale) for module in model.modules()
              if KEEP_WEIGHT_SCALE and isinstance(module, CproQLinear)]
    model.model.to(dtype=torch.float16)
    for module, scale in scales:
        module.scale = scale
    for module in model.modules():
        if isinstance(module, Qwen3_5MoeRMSNorm):
            gamma = (module.weight + 1.0).detach().contiguous()
            module.register_buffer('export_gamma', gamma, persistent=False)


def export_four_decoder_air(model, inputs):
    with cpro_export_contracts(True, chunk_size=64, gdn_layers=3, keep_weight_scale=KEEP_WEIGHT_SCALE):
        torchair.dynamo_export(model=model, export_path='weight', dynamic=False, **inputs)

if __name__ == '__main__':
    import argparse
    from pathlib import Path
    from qwen35_moe_dflash.target.prefill_t64.whole_export import export

    parser = argparse.ArgumentParser(description='Export one T64 L0–L3 CPro decoder AIR')
    parser.add_argument('--source-root', type=Path, required=True, help='original CPro source directory')
    parser.add_argument('--reference', type=Path, required=True, help='measured 310P capture directory')
    parser.add_argument('--config', type=Path, default=Path('config/qwen3_5_moe.yaml'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # The original module registers torch_npu converters at import time.
    sys.modules['export_glm_model_qwen3_5_moe'] = sys.modules[__name__]
    export(args.source_root.resolve(), args.config, args.reference, args.output,
           four_decoders=True, prepared_source=Path(__file__).resolve().parent, keep_weight_scale=KEEP_WEIGHT_SCALE)
"""]
    path.write_text(''.join(lines))
