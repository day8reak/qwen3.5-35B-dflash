"""Check this model's dtype relationships in its GE construction records.

This is not an OMG/CastT schema or a device support catalogue. In particular,
public GE Cast has an integer dst_type attribute and derives its source type
from its input descriptor; compiler-inserted CastT nodes need separate evidence.
"""
from __future__ import annotations


# Public GE enum values already used by the Cast recipe encoding. These are
# not Q35GE001's wire tags (1..5), nor a decoding table for private HIAI dumps.
GE_DTYPE_IDS = {"float16": 1, "float32": 0, "int32": 3, "int64": 9, "bool": 12}
_INTEGER = {"int32", "int64"}
_UNARY = {"Neg", "Sigmoid", "Exp", "Log", "Sqrt", "Rsqrt", "Cos", "Sin",
          "Power", "Swish", "Softplus", "SoftmaxV2"}
_BINARY = {"Add", "Sub", "Mul", "RealDiv", "Maximum", "Minimum", "MatMul", "BatchMatMulV2"}
_COMPARE = {"Equal", "Less", "LessEqual", "Greater", "GreaterEqual"}
_STRUCTURAL = {"Reshape": ("shape",), "BroadcastTo": ("shape",), "Transpose": ("perm",),
               "Slice": ("offsets", "size"), "ReduceMean": ("axes",), "ReduceSum": ("axes",),
               "GatherV2": ("indices", "axis"), "GatherElements": ("index",)}


def validate_record_types(record):
    """Reject internally inconsistent types before writing either export form.

    Check only the contracts emitted by this repository. Same-dtype arithmetic
    is a model-lowering invariant, not a claim about every dtype combination
    accepted by public GE (for example, Log and Mul have broader schemas).
    Other GE operations remain the responsibility of their own lowering.
    """
    typ = record["type"]
    inputs = {port["port"]: port for port in record["inputs"]}
    outputs = {port["port"]: port for port in record["outputs"]}
    attrs = record.get("attrs", {})

    def require(condition, detail):
        if not condition:
            raise ValueError(f"{record['name']} ({typ}): {detail}")

    def ports(ins, outs=("y",)):
        require(set(inputs) == set(ins) and set(outputs) == set(outs), "unexpected dtype contract ports")

    def same(*values):
        require(len({port["dtype"] for port in values}) == 1, "data input/output dtypes must match")

    def integer(name):
        require(inputs[name]["dtype"] in _INTEGER, f"{name} must have an integer dtype")

    if typ == "Cast":
        ports(("x",))
        require(list(inputs["x"]["shape"]) == list(outputs["y"]["shape"]), "Cast must preserve shape")
        require(attrs.get("dst_type") == {"type": "int", "value": GE_DTYPE_IDS[outputs["y"]["dtype"]]},
                "Cast dst_type must match its output descriptor")
    elif typ == "ArgMaxV2":
        ports(("x", "dimension"))
        integer("dimension")
        require(inputs["x"]["dtype"] != "bool", "ArgMax does not accept bool data")
        require(outputs["y"]["dtype"] in _INTEGER, "ArgMax indices must be int32 or int64")
        require(attrs.get("dtype") == {"type": "dtype", "value": outputs["y"]["dtype"]},
                "ArgMax dtype must match its output descriptor")
    elif typ == "ArgMaxExt2":
        # Selected HIAI math_defs.h frontend, not npu_generated_opinfo.json.
        ports(("x", "axis"))
        require(inputs["x"]["dtype"] in {"float32", "int32"}, "ArgMaxExt2 frontend data must be float32 or int32")
        require(inputs["axis"]["dtype"] == "int32", "ArgMaxExt2 axis must be int32")
        require(outputs["y"]["dtype"] in _INTEGER, "ArgMaxExt2 indices must be int32 or int64")
        require(attrs.get("output_type") ==
                {"type": "int", "value": GE_DTYPE_IDS[outputs["y"]["dtype"]]},
                "ArgMaxExt2 output_type must match its output descriptor")
        require(set(attrs) == {"output_type", "keep_dims", "outmaxval", "topk"},
                "ArgMaxExt2 probe requires explicit frontend attributes; axis is an input")
        require(attrs["keep_dims"] == {"type": "bool", "value": False}
                and attrs["outmaxval"] == {"type": "bool", "value": False}
                and attrs["topk"] == {"type": "int", "value": 1},
                "ArgMaxExt2 probe requires top1 indices with reduced rank")
    elif typ == "ReduceMax":
        ports(("x", "axes"))
        require(inputs["x"]["dtype"] in {"float32", "int32", "uint8"}, "ReduceMax frontend data dtype mismatch")
        require(inputs["axes"]["dtype"] == "int32", "ReduceMax axes must be int32")
        same(inputs["x"], outputs["y"])
        require(attrs == {"keep_dims": {"type": "bool", "value": False}},
                "ReduceMax probe uses an axes tensor and keep_dims=false")
    elif typ == "GatherV2D":
        ports(("x", "indices"))
        require(inputs["x"]["dtype"] in {"float32", "int32"}, "GatherV2D frontend data dtype mismatch")
        require(inputs["indices"]["dtype"] == "int32", "GatherV2D indices must be int32")
        same(inputs["x"], outputs["y"])
        require(set(attrs) == {"axis"} and attrs["axis"].get("type") == "int"
                and type(attrs["axis"].get("value")) is int, "GatherV2D requires an integer axis attribute")
    elif typ == "TopKV2":
        ports(("x", "k"), ("values", "indices"))
        require(inputs["x"]["dtype"] != "bool", "TopK does not accept bool data")
        require(inputs["k"]["dtype"] == "int32", "TopK k must be int32")
        same(inputs["x"], outputs["values"])
        require(outputs["indices"]["dtype"] == "int32", "TopK indices must be int32")
        require(list(outputs["values"]["shape"]) == list(outputs["indices"]["shape"]),
                "TopK values and indices shapes must match")
    elif typ in _COMPARE:
        ports(("x1", "x2"))
        same(inputs["x1"], inputs["x2"])
        require(outputs["y"]["dtype"] == "bool", "comparison output must be bool")
    elif typ in {"LogicalAnd", "LogicalOr"}:
        ports(("x1", "x2"))
        require(all(port["dtype"] == "bool" for port in (*inputs.values(), *outputs.values())),
                "logical inputs and output must be bool")
    elif typ == "SelectV2":
        ports(("condition", "then", "else"), ("result",))
        require(inputs["condition"]["dtype"] == "bool", "Select condition must be bool")
        same(inputs["then"], inputs["else"], outputs["result"])
    elif typ == "ConcatD":
        ports(tuple(f"x{i}" for i in range(len(inputs))))
        require(bool(inputs), "Concat needs data inputs")
        same(*inputs.values(), outputs["y"])
    elif typ in _STRUCTURAL:
        metadata = _STRUCTURAL[typ]
        ports(("x", *metadata))
        same(inputs["x"], outputs["y"])
        for name in metadata:
            integer(name)
    elif typ in _UNARY:
        ports(("x",))
        same(inputs["x"], outputs["y"])
        if typ == "SoftmaxV2":
            require(attrs.get("half_to_float", {"type": "bool", "value": False}) ==
                    {"type": "bool", "value": False}, "half_to_float conflicts with preserved output dtype")
    elif typ in _BINARY:
        ports(("x1", "x2"))
        same(inputs["x1"], inputs["x2"], outputs["y"])
