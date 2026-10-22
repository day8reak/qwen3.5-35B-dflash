"""Reference-only execution of primitive direct-graph IR operations in PyTorch.

The production builder and GE serializer never import this module. Comparing
against the independently captured original model with the same arithmetic
backend isolates graph-definition errors from NumPy/GE kernel differences.
Passing this comparison does not validate AIR, OMG, HIAI or device arithmetic.
"""
import numpy as np
import torch
import torch.nn.functional as F


def evaluate_torch(graph, inputs, weights=None):
    graph.validate()
    weights = {} if weights is None else weights
    values = {}

    def put(value, array):
        tensor = array if isinstance(array, torch.Tensor) else torch.from_numpy(np.asarray(array))
        if tensor.dtype != getattr(torch, value.dtype) or tuple(tensor.shape) != value.shape:
            raise ValueError(f"{value.name}: expected {value.dtype} {value.shape}, got {tensor.dtype} {tuple(tensor.shape)}")
        values[value.name] = tensor

    for value in graph.inputs:
        put(value, inputs[value.name])
    for name, array in graph.constants.items():
        values[name] = torch.from_numpy(array)
    for name, value in graph.weights.items():
        put(value, weights[name])
    binary = {k: getattr(torch, k) for k in (
        "add", "sub", "mul", "div", "eq", "gt", "lt", "le", "ge",
        "logical_and", "logical_or", "maximum", "minimum")}
    unary = {k: getattr(torch, k) for k in (
        "neg", "exp", "log", "sqrt", "rsqrt", "cos", "sin", "sigmoid")}
    with torch.inference_mode():
        for node in graph.nodes:
            args = [values[v.name] for v in node.inputs]
            attrs, kind = node.attrs, node.kind
            if kind in binary:
                result = binary[kind](*args)
            elif kind in unary:
                result = unary[kind](args[0])
            elif kind in ("silu", "softplus"):
                result = getattr(F, kind)(args[0], **attrs)
            elif kind == "scalar_add":
                result = args[0] + attrs["value"]
            elif kind == "scalar_mul":
                result = args[0] * attrs["value"]
            elif kind == "reshape":
                result = args[0].reshape(attrs["shape"])
            elif kind == "transpose":
                result = args[0].permute(attrs["perm"])
            elif kind == "slice":
                result = args[0][tuple(slice(start, None if size == -1 else start + size)
                                      for start, size in zip(attrs["starts"], attrs["sizes"]))]
            elif kind == "concat":
                result = torch.cat(args, dim=attrs["axis"])
            elif kind in ("reduce_sum", "reduce_mean"):
                fn = torch.sum if kind == "reduce_sum" else torch.mean
                result = fn(args[0], dim=attrs["axes"], keepdim=attrs.get("keepdims", False))
            elif kind == "cast":
                result = args[0]
            elif kind == "gather":
                axis = attrs.get("axis", 0) % args[0].ndim
                indices = args[1].long()
                result = torch.index_select(args[0], axis, indices.flatten())
                result = result.reshape((*args[0].shape[:axis], *indices.shape, *args[0].shape[axis+1:]))
            elif kind == "gather_elements":
                result = torch.gather(args[0], attrs["axis"], args[1].long())
            elif kind == "softmax":
                result = torch.softmax(args[0], dim=attrs.get("axis", -1))
            elif kind == "argmax":
                result = torch.argmax(args[0], dim=attrs.get("axis", -1), keepdim=attrs.get("keepdims", False))
            elif kind == "where":
                result = torch.where(*args)
            elif kind == "cumsum":
                result = args[0].cumsum(attrs["axis"])
            elif kind in ("topk_values", "topk_indices"):
                result = torch.topk(args[0], attrs["k"], dim=attrs.get("axis", -1))[0 if kind == "topk_values" else 1]
            elif kind == "matmul":
                a, b = args
                if attrs.get("transpose_a", False):
                    a = a.transpose(-1, -2)
                if attrs.get("transpose_b", False):
                    b = b.transpose(-1, -2)
                result = a @ b
            elif kind == "broadcast_to":
                result = torch.broadcast_to(args[0], attrs["shape"])
            else:
                raise NotImplementedError(f"reference IR operation {kind!r}")
            put(node.output, result.to(getattr(torch, node.output.dtype)))
    return [values[value.name] for value in graph.outputs]
