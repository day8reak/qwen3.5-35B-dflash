"""Direct GE/AIR emitter for the independent stateful model fork.

Primitive lowering follows ge_cpp but uses a local emitter adapter. Custom
multi-output nodes are opaque and retain named port wiring and all live state
outputs. This module neither imports Torch nor registers device kernels.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .. import ge_cpp as _legacy
from ..ge_cpp import _q, _ints, _opindex, _validate_cast_selection, _FOOTER
from ..ge_ir import Node, Value
from ..ge_type_contracts import GE_DTYPE_IDS as _LEGACY_GE_DTYPE_IDS
from .ir import DTYPES, CustomNode, node_outputs, attention_registration_cpp

GE_DTYPE_IDS = {**_LEGACY_GE_DTYPE_IDS, "float32": 0, "int16": 6, "int8": 2}
_DT = {**_legacy._DT, "float32": "ge::DT_FLOAT", "int16": "ge::DT_INT16", "int8": "ge::DT_INT8"}
_HEADER = _legacy._HEADER.replace("DT_FLOAT16, DT_INT32, DT_INT64, DT_BOOL",
                                 "DT_FLOAT16, DT_INT32, DT_INT64, DT_BOOL, DT_FLOAT, DT_INT16, DT_INT8")
_HEADER = _HEADER.replace('    if (std::strcmp(type, "ConcatD") == 0)',
    '    if (std::strcmp(type, "AdnFusedInferAttention") == 0) {\n' + attention_registration_cpp()
    + '\n    } else if (std::strcmp(type, "ConcatD") == 0)')


def _desc(value):
    return f"Desc({_ints(value.shape)}, {_DT[value.dtype]})"


def _typed_attr(literal):
    """Decode this module's closed C++ literal vocabulary for recipe emission."""
    enum = {_DT[dtype]: value for dtype, value in GE_DTYPE_IDS.items()}
    if literal in enum:
        return {"type": "dtype", "value": {v: k for k, v in _DT.items()}[literal]}
    if literal in {"true", "false"}:
        return {"type": "bool", "value": literal == "true"}
    if literal.startswith("int64_t{"):
        value = literal[len("int64_t{"):-1]
        return {"type": "int", "value": enum[value] if value in enum else int(value)}
    if literal.startswith("float{"):
        return {"type": "float", "value": float(literal[len("float{"):-1])}
    if literal.startswith("std::vector<int64_t>{"):
        value = literal[len("std::vector<int64_t>{"):-1]
        return {"type": "ints", "value": [] if not value else [int(x) for x in value.split(",")]}
    if literal.startswith("ge::AscendString("):
        return {"type": "string", "value": json.loads(literal[len("ge::AscendString("):-1])}
    raise ValueError(f"unsupported internal GE attribute literal {literal!r}")


def _cpp_attr(attribute):
    """Render the same closed attribute vocabulary after a record rewrite."""
    kind, value = attribute["type"], attribute["value"]
    if kind == "dtype":
        return _DT[value]
    if kind == "bool":
        return "true" if value else "false"
    if kind == "int":
        return f"int64_t{{{value}}}"
    if kind == "float":
        return f"float{{{value!r}}}"
    if kind == "ints":
        return _ints(value)
    if kind == "string":
        return f"ge::AscendString({_q(value)})"
    raise ValueError(f"unsupported internal GE attribute kind {kind!r}")


def _inline_records(graph, records):
    """Rebuild inline C++ from the exact records also consumed by the recipe writer."""
    emitter = _Emitter(graph)
    bound = {}
    for index, record in enumerate(records):
        outputs = [(port["port"], Value(f"__render/{index}/{port['port']}",
                                       tuple(port["shape"]), port["dtype"])) for port in record["outputs"]]
        inputs = [(port["port"], bound[(port["src"], port["src_port"])]) for port in record["inputs"]]
        attrs = {key: _cpp_attr(value) for key, value in record["attrs"].items()}
        var = emitter.op(record["name"], record["type"], inputs, outputs, attrs)
        for port, value in outputs:
            bound[(index, port)] = value
            emitter.refs[value.name] = (var, port)
        if record["type"] == "Data":
            emitter.lines.append(f"  Check({var}.UpdateInputDesc(\"x\", {_desc(outputs[0][1])}), \"Data input descriptor\");")
        constant = record.get("constant")
        if constant is not None:
            if "data" in constant:
                payload = "std::vector<uint8_t>{" + ",".join(str(x) for x in constant["data"]) + "}"
            else:
                if constant.get("offset", 0) != 0:
                    raise ValueError("inline weight constants require zero offset")
                payload = f"Read({_q(str(Path(constant['path']).resolve()))}, {constant['size']}ULL)"
            emitter.lines.append(f"  SetConst({var}, {_desc(outputs[0][1])}, {payload});")
        if record["type"] == "FileConstant":
            path = str(Path(record["attrs"]["file_path"]["value"]).resolve())
            value = outputs[0][1]
            size = math.prod(value.shape) * DTYPES[value.dtype].itemsize
            emitter.lines.append(f"  if (!fs::is_regular_file({_q(path)}) || fs::file_size({_q(path)}) != {size}ULL) throw std::runtime_error(\"external weight size mismatch\");")
    return emitter


class _Emitter(_legacy._Emitter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.recipe:
            self.lines = [_HEADER]

    def op(self, name, typ, inputs, outputs, attrs=None):
        for _, value in inputs:
            self.ensure_constant(value)
        def desc(value):
            return {"shape": list(value.shape), "dtype": value.dtype}
        self.records.append({
            "name": name, "type": typ,
            "inputs": [{"port": port, "src": _opindex(self.refs[value.name][0]),
                        "src_port": self.refs[value.name][1], **desc(value)} for port, value in inputs],
            "outputs": [{"port": port, **desc(value)} for port, value in outputs],
            "attrs": {k: _typed_attr(v) for k, v in (attrs or {}).items()}, "constant": None,
        })
        var = f"ops[{self.counter}]"
        self.counter += 1
        in_ports = ",".join(_q(port) for port, _ in inputs)
        out_ports = ",".join(_q(port) for port, _ in outputs)
        self.lines.append(f"  ops.push_back(Op({_q(name)}, {_q(typ)}, {{{in_ports}}}, {{{out_ports}}}));")
        for port, value in inputs:
            src, srcport = self.refs[value.name]
            self.lines.append(f"  {var}.SetInput({_q(port)}, {src}, {_q(srcport)});")
            self.lines.append(f"  Check({var}.UpdateInputDesc({_q(port)}, {_desc(value)}), \"input descriptor\");")
        for port, value in outputs:
            self.lines.append(f"  Check({var}.UpdateOutputDesc({_q(port)}, {_desc(value)}), \"output descriptor\");")
        for attr, literal in (attrs or {}).items():
            self.lines.append(f"  {var}.SetAttr({_q(attr)}, {literal});")
        descs = ",".join(_desc(value) for _, value in outputs)
        self.lines.append(f"  expected.push_back({{{_q(name)},{_q(typ)},{{{descs}}}}});")
        return var

    def constant(self, data, dtype, name):
        array = np.asarray(data, dtype=DTYPES[dtype])
        value = Value(name, array.shape, dtype)
        self.pending_constants[name] = array
        if array.nbytes <= self.max_broadcast_tensor_bytes:
            self.inline_constants[name] = array
        return value

    def materialize_constant(self, array, value):
        var = self.op(value.name, "Const", [], [("y", value)])
        self.records[-1]["constant"] = {"data": array.tobytes()}
        if not self.recipe:
            literal = ",".join(str(x) for x in array.tobytes())
            self.lines.append(f"  SetConst({var}, {_desc(value)}, std::vector<uint8_t>{{{literal}}});")
        self.refs[value.name] = (var, "y")

    def static_broadcast(self, source, output):
        """Reuse small byte-exact Const expansions, avoiding CPUCL FP16 folding."""
        array = self.inline_constants.get(source.name)
        size = math.prod(output.shape) * DTYPES[source.dtype].itemsize
        if array is None or size > self.max_broadcast_tensor_bytes:
            return None
        key = (source.dtype, source.shape, array.tobytes(), output.shape)
        if key in self.broadcast_cache:
            return self.broadcast_cache[key]
        if self.broadcast_bytes + size > self.max_broadcast_total_bytes:
            return None
        # Copying elements preserves signed zeros and NaN payloads; no mask
        # arithmetic and no reading checkpoint/FileConstant tensors here.
        expanded = np.broadcast_to(array, output.shape).copy(order="C")
        value = self.constant(expanded, source.dtype, output.name)
        self.broadcast_cache[key] = value
        self.broadcast_bytes += size
        return value

    def emit_custom(self, node):
        attrs = {key: _cpp_attr(value) for key, value in node.attrs.items()}
        var = self.op(node.name, node.op_type,
                      list(zip(node.input_ports, node.inputs)),
                      list(zip(node.output_ports, node.outputs)), attrs)
        for port, value in zip(node.output_ports, node.outputs):
            self.refs[value.name] = (var, port)


def _liveness(graph, folded=None):
    folded = folded or {}
    live = {v.name for v in graph.outputs}
    for node in reversed(graph.nodes):
        outputs = node_outputs(node)
        if any(v.name in live and v.name not in folded for v in outputs):
            live.update(value.name for value in node.inputs)
    return live


def _folding_view(graph):
    # Legacy constant folding sees every custom output as an opaque producer.
    # This prevents custom execution/folding while correctly accounting for
    # consumers of secondary state ports and their dependencies.
    nodes = []
    for node in graph.nodes:
        if isinstance(node, CustomNode):
            nodes.extend(Node("custom", node.inputs, output, {}) for output in node.outputs)
        else:
            nodes.append(node)
    return SimpleNamespace(nodes=nodes, inputs=graph.inputs, outputs=graph.outputs,
                           constants=graph.constants, weights=graph.weights)


def emit_cpp(graph, weight_paths=None, *, external_weights=False, recipe_dir=None,
             fold_constant_arithmetic=True, fp16_reductions="native", reduction_report=None,
             fp16_row_mul="native", row_mul_report=None, fp16_native_reductions=()):
    """Return C++ that builds real GE nodes and serializes AIR via GE.

    Options match the legacy emitter. Recipe mode writes version Q35GF001,
    which additionally encodes FP32 recurrent state and INT16 valid lengths.
    Fused node schemas must already have been validated by explicit bindings.
    """
    if reduction_report is not None and not isinstance(reduction_report, dict):
        raise TypeError("reduction_report must be a mutable dictionary")
    if row_mul_report is not None and not isinstance(row_mul_report, dict):
        raise TypeError("row_mul_report must be a mutable dictionary")
    graph.validate()
    if recipe_dir is None and len(graph.nodes) > 10_000:
        raise ValueError("large graphs require recipe_dir to bound generated C++ compiler memory")
    weight_paths = {} if weight_paths is None else weight_paths
    missing = set(graph.weights) - set(weight_paths)
    if missing:
        raise ValueError(f"missing weight paths: {sorted(missing)[:4]}")
    total = sum(math.prod(v.shape) * DTYPES[v.dtype].itemsize for v in graph.weights.values())
    if not external_weights and total >= 1_800_000_000:
        raise ValueError("large graphs require external_weights=True (AIR protobuf size limit)")
    e = _Emitter(graph, recipe=recipe_dir is not None, fp16_reductions=fp16_reductions,
                 fp16_row_mul=fp16_row_mul, fp16_native_reductions=fp16_native_reductions)
    live = _liveness(graph)
    for node in graph.nodes:
        if any(v.name in live for v in node_outputs(node)):
            _validate_cast_selection(node)
            if node.kind in {"reduce_mean", "reduce_sum"}:
                if len(node.inputs) != 1:
                    raise ValueError(f"{node.kind}: unexpected input count {len(node.inputs)}")
                e.checked_reduction_contract(node.output, node.inputs[0], node.attrs.get("axes"),
                                             node.attrs.get("keepdims", False))
    folded = {}
    if fold_constant_arithmetic:
        from ..ge_constants import fold_constants
        folded = fold_constants(_folding_view(graph), weight_paths, live)
        # Recompute liveness: derived constants replace their entire static
        # producer chain, so do not keep unused raw parameter nodes in AIR.
        live = _liveness(graph, folded)
    for index, value in enumerate(graph.inputs):
        var = e.op(value.name, "Data", [], [("y", value)], {"index": f"int64_t{{{index}}}"})
        e.lines.append(f"  Check({var}.UpdateInputDesc(\"x\", {_desc(value)}), \"Data input descriptor\");")
        e.refs[value.name] = (var, "y")
    for name, array in graph.constants.items():
        if name in live:
            # A static broadcast may replace this source entirely. Materialize
            # only on first live use so Verify never expects disconnected nodes.
            e.pending_constants[name] = array
            if array.nbytes <= e.max_broadcast_tensor_bytes:
                e.inline_constants[name] = array
    for key, value in graph.weights.items():
        if value.name not in live:
            continue
        path = str(Path(weight_paths[key]).resolve())
        size = math.prod(value.shape) * DTYPES[value.dtype].itemsize
        if external_weights:
            var = e.op(value.name, "FileConstant", [], [("y", value)],
                       {"file_path": f"ge::AscendString({_q(path)})", "shape": _ints(value.shape),
                        "dtype": _DT[value.dtype], "file_id": 'ge::AscendString("")'})
            e.lines.append(f"  if (!fs::is_regular_file({_q(path)}) || fs::file_size({_q(path)}) != {size}ULL) throw std::runtime_error(\"external weight size mismatch\");")
        else:
            var = e.op(value.name, "Const", [], [("y", value)])
            e.records[-1]["constant"] = {"path": path, "offset": 0, "size": size}
            e.lines.append(f"  SetConst({var}, {_desc(value)}, Read({_q(path)}, {size}ULL));")
        e.refs[value.name] = (var, "y")

    binary = {"add": "Add", "sub": "Sub", "mul": "Mul", "div": "RealDiv",
              "maximum": "Maximum", "minimum": "Minimum", "eq": "Equal",
              "lt": "Less", "le": "LessEqual", "gt": "Greater", "ge": "GreaterEqual",
              "logical_and": "LogicalAnd", "logical_or": "LogicalOr"}
    unary = {"neg": "Neg", "sigmoid": "Sigmoid", "exp": "Exp", "log": "Log",
             "sqrt": "Sqrt", "rsqrt": "Rsqrt", "cos": "Cos", "sin": "Sin"}
    topk_cache = {}
    for node in graph.nodes:
        if isinstance(node, CustomNode):
            if any(v.name in live for v in node.outputs):
                e.emit_custom(node)
            continue
        kind, v, a, ins = node.kind, node.output, node.attrs, list(node.inputs)
        if v.name not in live:
            continue
        if v.name in folded:
            e.constant(folded[v.name], v.dtype, v.name)
            continue
        if kind == "where" and (len(ins) != 3 or ins[0].dtype != "bool" or
                                 any(x.dtype != v.dtype for x in ins[1:])):
            raise ValueError(f"{v.name}: Select needs BOOL condition and matching branch dtypes")
        # Public GE BasicType excludes BOOL. These operations preserve boolean
        # values exactly through an integer carrier, with no floating casts.
        boolean_output = None
        if v.dtype == "bool" and kind in {"concat", "slice", "transpose", "where"}:
            boolean_output = v
            for i, source in enumerate(ins):
                if source.dtype != "bool" or (kind == "where" and i == 0):
                    continue
                carrier = Value(f"__ge/{v.name}/integer_input_{i}", source.shape, "int32")
                if source.name in e.inline_constants:
                    carrier = e.constant(e.inline_constants[source.name].astype(DTYPES["int32"]),
                                         "int32", carrier.name)
                else:
                    var = e.op(carrier.name, "Cast", [("x", source)], [("y", carrier)],
                               {"dst_type": "int64_t{ge::DT_INT32}"})
                    e.refs[carrier.name] = (var, "y")
                ins[i] = carrier
            v = Value(f"__ge/{v.name}/integer_output", v.shape, "int32")
        attrs, outport = {}, "y"
        outputs = [("y", v)]
        if kind == "mul":
            if len(ins) != 2:
                raise ValueError(f"{v.name}: Mul requires exactly two inputs")
            e.emit_mul(v, ins[0], ins[1])
            continue
        elif kind in binary:
            typ, ports = binary[kind], ["x1", "x2"]
        elif kind in unary:
            typ, ports = unary[kind], ["x"]
            if kind in {"exp", "log"}:
                attrs = {"base": "float{-1.0}", "scale": "float{1.0}", "shift": "float{0.0}"}
        elif kind in {"scalar_mul", "scalar_add"}:
            typ, ports = "Power", ["x"]
            scalar = float(a["value"])
            if not math.isfinite(scalar):
                raise ValueError("Power scalar attributes must be finite")
            attrs = {"power": "float{1.0}",
                     "scale": f"float{{{scalar if kind == 'scalar_mul' else 1.0}}}",
                     "shift": f"float{{{scalar if kind == 'scalar_add' else 0.0}}}"}
        elif kind == "silu":
            typ, ports, attrs = "Swish", ["x"], {"scale": "float{1.0}"}
        elif kind == "softplus":
            # Internal OMG lacks SoftplusV2 shape inference. Native Softplus
            # expresses this model's beta=1 case without intermediate rounding.
            # Above threshold 20, log1p(exp(x)) and x round to the same FP16
            # value. Do not silently generalize this mapping to other settings.
            if (float(a.get("beta", 1)) != 1.0 or float(a.get("threshold", 20)) != 20.0
                    or v.dtype != "float16" or ins[0].dtype != "float16"):
                raise NotImplementedError("native Softplus lowering requires FP16 beta=1 threshold=20")
            typ, ports = "Softplus", ["x"]
        elif kind in {"reshape", "broadcast_to", "transpose"}:
            if kind == "broadcast_to":
                if ins[0].dtype != v.dtype or tuple(a["shape"]) != v.shape:
                    raise ValueError("BroadcastTo descriptor differs from its source/shape")
                if np.broadcast_shapes(ins[0].shape, v.shape) != v.shape:
                    raise ValueError("BroadcastTo input cannot expand to output shape")
                constant = e.static_broadcast(ins[0], v)
                if constant is not None:
                    e.alias_constant(v, constant)
                    continue
            typ = {"reshape": "Reshape", "broadcast_to": "BroadcastTo", "transpose": "Transpose"}[kind]
            key = "perm" if kind == "transpose" else "shape"
            ins.append(e.reshape_shape(v, a[key]) if kind == "reshape"
                       else e.helper(v, a[key], suffix=key))
            ports = ["x", key]
            if kind == "reshape":
                attrs = {"axis": "int64_t{0}", "num_axes": "int64_t{-1}"}
        elif kind in {"reduce_mean", "reduce_sum"}:
            if len(ins) != 1:
                raise ValueError(f"{kind}: unexpected input count {len(ins)}")
            e.emit_reduction(v, ins[0], kind, a.get("axes"), a.get("keepdims", False))
            continue
        elif kind == "slice":
            typ, ports = "Slice", ["x", "offsets", "size"]
            ins.extend([e.helper(v, a["starts"], suffix="starts"), e.helper(v, a["sizes"], suffix="sizes")])
        elif kind == "concat":
            if v.dtype == "float16" and len(v.shape) == 3:
                e.emit_concat(v, ins, a["axis"])
                continue
            typ, ports = "ConcatD", [f"x{i}" for i in range(len(ins))]
            attrs = {"concat_dim": f"int64_t{{{a['axis']}}}", "N": f"int64_t{{{len(ins)}}}"}
        elif kind == "gather":
            typ, ports = "GatherV2", ["x", "indices", "axis"]
            ins.append(e.helper(v, a.get("axis", 0), dtype="int32", suffix="axis"))
            attrs = {"batch_dims": "int64_t{0}", "negative_index_support": "false", "is_preprocessed": "false"}
        elif kind == "gather_elements":
            # The X90 verifier rejects INT32 data; token IDs must never be
            # rounded through a floating carrier to satisfy that restriction.
            if ins[0].dtype != "float16" or v.dtype != ins[0].dtype:
                raise NotImplementedError("GE GatherElements requires FP16 data; use an exact integer Gather path")
            typ, ports = "GatherElements", ["x", "index"]
            attrs = {"dim": f"int64_t{{{a['axis']}}}"}
        elif kind == "softmax":
            typ, ports = "SoftmaxV2", ["x"]
            attrs = {"axes": _ints([a.get("axis", -1)]), "half_to_float": "false"}
        elif kind == "argmax":
            typ, ports = "ArgMaxV2", ["x", "dimension"]
            ins.append(e.helper(v, a.get("axis", -1), dtype="int32", suffix="axis"))
            attrs = {"dtype": _DT[v.dtype]}
            if a.get("keepdims", False):
                axis = a.get("axis", -1) % len(ins[0].shape)
                squeezed = Value(f"__ge/{v.name}/squeezed",
                                 ins[0].shape[:axis] + ins[0].shape[axis + 1:], v.dtype)
                var = e.op(squeezed.name, typ, list(zip(ports, ins)), [("y", squeezed)], attrs)
                e.refs[squeezed.name] = (var, "y")
                shape = e.reshape_shape(v)
                typ, ports, ins = "Reshape", ["x", "shape"], [squeezed, shape]
                attrs = {"axis": "int64_t{0}", "num_axes": "int64_t{-1}"}
        elif kind in {"topk_values", "topk_indices"}:
            axis = a.get("axis", -1) % len(ins[0].shape)
            if axis != len(ins[0].shape) - 1:
                raise ValueError("GE TopKV2 lowering currently requires the last axis")
            outport = "values" if kind == "topk_values" else "indices"
            if kind == "topk_indices" and v.dtype != "int32":
                raise ValueError("TopKV2 indices must be int32")
            cache_key = (ins[0].name, int(a["k"]), axis)
            if cache_key in topk_cache:
                e.refs[v.name] = (topk_cache[cache_key], outport)
                continue
            typ, ports = "TopKV2", ["x", "k"]
            ins.append(e.helper(v, a["k"], dtype="int32", suffix="k"))
            values = Value(v.name, v.shape, node.inputs[0].dtype)
            indices = Value(v.name, v.shape, "int32")
            outputs = [("values", values), ("indices", indices)]
            attrs = {"sorted": "true", "dim": "int64_t{-1}", "largest": "true"}
        elif kind == "where":
            # X90 OMG lowers SelectV2 to Select, whose verifier does not accept
            # the implicit scalar/rank broadcasting used by NumPy/PyTorch.
            # Broadcast condition and both branches, including boolean carriers,
            # without arithmetic masking (0 * -inf/NaN would change semantics).
            ins = [e.broadcast(x, v, i) for i, x in enumerate(ins)]
            typ, ports = "SelectV2", ["condition", "then", "else"]
            outputs, outport = [("result", v)], "result"
        elif kind == "cast":
            typ, ports = "Cast", ["x"]
            attrs = {"dst_type": f"int64_t{{{_DT[v.dtype]}}}"}
        elif kind == "cumsum":
            from ..ge_scan import emit_cumsum
            emit_cumsum(e, v, ins[0], a["axis"], scope=a.get("scope"))
            continue
        elif kind == "matmul":
            if len(ins[0].shape) > 2 and len(ins[1].shape) == 2 and not a.get("transpose_a", False):
                # Dense projections need no mixed-rank broadcast matmul. Preserve
                # the exact FP16 matmul boundary by flattening only batch axes.
                flat = Value(f"__ge/{v.name}/flat_input",
                             (math.prod(ins[0].shape[:-1]), ins[0].shape[-1]), ins[0].dtype)
                shape = e.reshape_shape(flat)
                var = e.op(flat.name, "Reshape", [("x", ins[0]), ("shape", shape)], [("y", flat)],
                           {"axis": "int64_t{0}", "num_axes": "int64_t{-1}"})
                e.refs[flat.name] = (var, "y")
                projected = Value(f"__ge/{v.name}/flat_output", (flat.shape[0], v.shape[-1]), v.dtype)
                var = e.op(projected.name, "MatMul", [("x1", flat), ("x2", ins[1])], [("y", projected)],
                           {"transpose_x1": "false", "transpose_x2": str(bool(a.get("transpose_b", False))).lower()})
                e.refs[projected.name] = (var, "y")
                shape = e.reshape_shape(v)
                var = e.op(v.name, "Reshape", [("x", projected), ("shape", shape)], [("y", v)],
                           {"axis": "int64_t{0}", "num_axes": "int64_t{-1}"})
                e.refs[v.name] = (var, "y")
                continue
            if (max(len(x.shape) for x in ins) > 3
                    or (min(len(x.shape) for x in ins) > 2 and a.get("transpose_a", False))):
                from ..ge_matmul import emit_rank3_matmul
                emit_rank3_matmul(e, v, ins[0], ins[1], a)
                continue
            batched = max(len(x.shape) for x in ins) > 2
            typ, ports = ("BatchMatMulV2" if batched else "MatMul"), ["x1", "x2"]
            attrs = {"adj_x1" if batched else "transpose_x1": str(bool(a.get("transpose_a", False))).lower(),
                     "adj_x2" if batched else "transpose_x2": str(bool(a.get("transpose_b", False))).lower()}
            if batched:
                attrs["offset_x"] = "int64_t{0}"
        else:
            raise NotImplementedError(f"GE lowering for {kind!r}")
        if len(ports) != len(ins):
            raise ValueError(f"{kind}: unexpected input count {len(ins)}")
        var = e.op(v.name, typ, list(zip(ports, ins)), outputs, attrs)
        if kind in {"topk_values", "topk_indices"}:
            topk_cache[cache_key] = var
        e.refs[v.name] = (var, outport)
        if boolean_output is not None:
            var = e.op(boolean_output.name, "Cast", [("x", v)], [("y", boolean_output)],
                       {"dst_type": "int64_t{ge::DT_BOOL}"})
            e.refs[boolean_output.name] = (var, "y")
    for output in graph.outputs:
        e.ensure_constant(output)
    input_indices = [_opindex(e.refs[v.name][0]) for v in graph.inputs]
    output_pairs = []
    for value in graph.outputs:
        reference, port = e.refs[value.name]
        index = _opindex(reference)
        ports = [output["port"] for output in e.records[index]["outputs"]]
        output_pairs.append((index, ports.index(port)))
    records = e.records
    if fp16_row_mul == "compact":
        from ..ge_reshape_chains import compact_row_mul_reshape_chains
        records, input_indices, output_pairs, chains = compact_row_mul_reshape_chains(
            records, input_indices, output_pairs, [r["name"] for r in e.row_mul_report["replaced"]])
        e.row_mul_report["reshape_chains"] = chains
    # Inline and recipe serialization must enforce the same descriptor/edge
    # contracts, including when lowering creates or shares intermediate nodes.
    from .recipe import _validate
    _validate(records, graph.name, input_indices, output_pairs)
    if reduction_report is not None:
        reduction_report.clear()
        reduction_report.update(e.reduction_report)
    if row_mul_report is not None:
        row_mul_report.clear()
        row_mul_report.update(e.row_mul_report)
    # A compact recipe keeps full-model compiler memory independent of node count.
    if recipe_dir is not None:
        from .recipe import emit_recipe_cpp
        return emit_recipe_cpp(records, graph.name, input_indices, output_pairs, recipe_dir)
    if fp16_row_mul == "compact" and chains["rewrites"]:
        e = _inline_records(graph, records)
    # Explicit output indices preserve the ABI even for TopK's second output.
    e.lines.append(f"  ge::Graph graph({_q(graph.name)});")
    e.lines.append('  std::cout << "Constructing GE graph with SetInputs..." << std::endl;')
    e.lines.append("  graph.SetInputs(std::vector<ge::Operator>{" + ",".join(f"ops[{index}]" for index in input_indices) + "});")
    e.lines.append('  Require(graph.IsValid(), "GE Graph::SetInputs failed; inspect earlier GE errors in export.log");')
    output_ops = []
    for index, port in output_pairs:
        output_ops.append("{ops[" + str(index) + "],{" + str(port) + "}}")
    e.lines.append("  graph.SetOutputs(std::vector<std::pair<ge::Operator, std::vector<size_t>>>{" + ",".join(output_ops) + "});")
    e.lines.append(_FOOTER)
    return "\n".join(e.lines)
