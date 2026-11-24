"""Adapt the reviewed CPro wrapper for the four-layer prefill export."""
import ast
import hashlib


_SOURCE_DIGEST = "adffc184d14b606641ceb61db2628c886608996ead4f0abc21c9e9aa36903cf2"
_STAGED_DIGEST = "5d641ec2e050220f4c64fb934d24b57b939e99e25751ff2def141dea3ea44df7"


def four_decoder_hidden_wrapper(text):
    """Keep the original CPro decoder loop between measured embedding and L3 hidden."""
    tree = ast.parse(stage_wrapper(text))
    cls, = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Qwen3_5MoeForCausalLMCproWrapper']
    forward, = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward']
    embedding, = [n for n in forward.body if isinstance(n, ast.If) and
                  any(isinstance(c, ast.Call) and ast.unparse(c.func) == 'self.model.model.embed_tokens'
                      for c in ast.walk(n))]
    loop, = [n for n in forward.body if isinstance(n, ast.For) and
             ast.unparse(n.iter) == 'enumerate(self.model.model.layers)']
    # Keep the original cache-position transform, FP16 entry and complete layer calls.
    forward.body = [n for n in forward.body[:forward.body.index(loop) + 1] if n is not embedding]
    forward.body.extend(ast.parse('return (hidden_states,)').body)
    return ast.unparse(ast.fix_missing_locations(tree)) + '\n'


def stage_wrapper(text: str, *, segmented=False, layers=4) -> str:
    source = _stage_base_wrapper(text)
    if segmented or layers != 4:
        from .source_model import use_position_gathers
        tree = ast.parse(source)
        if layers != 4:
            if layers != 40:
                raise ValueError('Supported CPro wrapper sizes are 4 and 40')
            wrapper, = (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Qwen3_5MoeForCausalLMCproWrapper')
            init, = (n for n in wrapper.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
            loader, = (n for n in ast.walk(init) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'load_npu_model')
            count, = (kw for kw in loader.keywords if kw.arg == 'layer_num')
            count.value = ast.Constant(layers)
        if segmented:
            use_position_gathers(tree, 'Qwen3_5MoeForCausalLMCproWrapper', ('hidden_states',), 'output_pos')
        return ast.unparse(ast.fix_missing_locations(tree)) + '\n'
    return source


def _stage_base_wrapper(text: str) -> str:
    """Stage the layer count and hidden dtypes of the CPro wrapper only."""
    source = text.replace("\r\n", "\n").encode("utf-8")
    digest = hashlib.sha256(source).hexdigest()
    if digest == _STAGED_DIGEST:
        return source.decode("utf-8")
    if digest != _SOURCE_DIGEST:
        raise ValueError("Unrecognized source revision for the CPro wrapper")

    tree = ast.parse(source)
    wrapper, = (node for node in tree.body if isinstance(node, ast.ClassDef)
                and node.name == "Qwen3_5MoeForCausalLMCproWrapper")
    methods = {node.name: node for node in wrapper.body if isinstance(node, ast.FunctionDef)}
    loader, = (node for node in ast.walk(methods["__init__"]) if isinstance(node, ast.Call)
               and isinstance(node.func, ast.Name) and node.func.id == "load_npu_model")
    count, = (key.value for key in loader.keywords if key.arg == "layer_num")
    hidden = [node.value for node in ast.walk(methods["forward"]) if isinstance(node, ast.Assign)
              and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
              and node.targets[0].id == "hidden_states"]
    embedding, = (node for node in hidden if isinstance(node, ast.Name) and node.id == "inputs_embeds")
    head_dtype, = (arg for node in hidden if isinstance(node, ast.Call) for arg in node.args
                   if isinstance(arg, ast.Attribute) and arg.attr == "float32"
                   and isinstance(arg.value, ast.Name) and arg.value.id == "torch")

    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    edits = []
    for node, value in ((count, b"4"), (embedding, b"inputs_embeds.to(torch.float16)"),
                        (head_dtype, b"torch.float16")):
        start = offsets[node.lineno - 1] + node.col_offset
        stop = offsets[node.end_lineno - 1] + node.end_col_offset
        edits.append((start, stop, value))
    for start, stop, value in sorted(edits, reverse=True):
        source = source[:start] + value + source[stop:]
    if hashlib.sha256(source).hexdigest() != _STAGED_DIGEST:
        raise ValueError("Staged CPro wrapper differs from the reviewed T64 adapter")
    return source.decode("utf-8")
